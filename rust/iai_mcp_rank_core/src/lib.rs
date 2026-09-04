//! iai_mcp_rank_core — resident rank-feature index.
//!
//! Holds the RAM-resident data a fused rank scan reads: vectors, graph
//! adjacency, lexical postings, resident surface text, and scalar rank
//! fields, keyed on `u128` record ids. `Inner` is struct-of-arrays: scalar
//! columns parallel to a slot index, one text arena for surface/aaak_index,
//! and CSR (compressed sparse row) tables for adjacency, tags, and lexical
//! postings — O(~20) heap allocations per build instead of one allocation
//! per record per field. Behind a generation-tagged double-buffer so a
//! rebuild never blocks a concurrent reader. Ships as a workspace rlib
//! re-exported by the `iai_mcp_native` wrapper as the `rank` sub-module.
//!
//! `fused_score` is the resident scorer: the corpus-side subset of the
//! fused rank formula, computed over a per-call candidate scope. Candidate
//! SELECTION (community gate, seeds, 2-hop spread, escalation widen) stays
//! caller-orchestrated; only its output index arrays cross into this call.

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::sync::Arc;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use parking_lot::Mutex;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// BM25 constants ported from `store/_lexical_index.py` — any drift here
/// silently changes BM25 fusion once a scorer consumes these postings.
pub const BM25_K1: f64 = 1.2;
pub const BM25_B: f64 = 0.75;
pub const MAX_QUERY_TOKENS: usize = 8;

/// `[A-Za-z_][A-Za-z0-9_]{1,}|[0-9]{3,}` — Rust's `regex` crate has no
/// lookaround, so both this and `camel_split` below are hand-rolled ASCII
/// scans, not a regex port. Non-ASCII bytes never start or continue a
/// token here, matching the Python pattern's explicit ranges.
fn find_raw_tokens(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let c = chars[i];
        if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            let mut j = i + 1;
            while j < n && (chars[j].is_ascii_alphanumeric() || chars[j] == '_') {
                j += 1;
            }
            if j - start >= 2 {
                out.push(chars[start..j].iter().collect());
                i = j;
                continue;
            }
            i += 1;
            continue;
        }
        if c.is_ascii_digit() {
            let start = i;
            let mut j = i + 1;
            while j < n && chars[j].is_ascii_digit() {
                j += 1;
            }
            if j - start >= 3 {
                out.push(chars[start..j].iter().collect());
                i = j;
                continue;
            }
            i += 1;
            continue;
        }
        i += 1;
    }
    out
}

/// `(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])` applied to the
/// original-cased raw token — split before an upper following a
/// lower/digit, and before the last upper of an acronym run.
fn camel_split(raw: &str) -> Vec<String> {
    let chars: Vec<char> = raw.chars().collect();
    let n = chars.len();
    if n == 0 {
        return Vec::new();
    }
    let mut parts = Vec::new();
    let mut start = 0usize;
    for i in 1..n {
        let prev = chars[i - 1];
        let cur = chars[i];
        let split_before_upper =
            (prev.is_ascii_lowercase() || prev.is_ascii_digit()) && cur.is_ascii_uppercase();
        let split_before_acronym_tail = prev.is_ascii_uppercase()
            && cur.is_ascii_uppercase()
            && i + 1 < n
            && chars[i + 1].is_ascii_lowercase();
        if split_before_upper || split_before_acronym_tail {
            parts.push(chars[start..i].iter().collect::<String>());
            start = i;
        }
    }
    parts.push(chars[start..n].iter().collect::<String>());
    parts
}

/// Whole-identifier plus snake/camel parts, lowercased — ported from
/// `store/_lexical_index.py:tokenize` byte-for-byte in emission order:
/// the lowercased whole token always; snake parts only when more than one
/// survives the length filter; camel parts only when more than one
/// survives, filtered by length before lowering.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in find_raw_tokens(text) {
        let low = raw.to_lowercase();
        out.push(low.clone());

        let parts: Vec<&str> = low.split('_').filter(|p| p.len() > 2).collect();
        if parts.len() > 1 {
            out.extend(parts.into_iter().map(str::to_string));
        }

        let camel_parts: Vec<String> = camel_split(&raw)
            .into_iter()
            .filter(|p| p.chars().count() > 2)
            .map(|p| p.to_lowercase())
            .collect();
        if camel_parts.len() > 1 {
            out.extend(camel_parts);
        }
    }
    out
}

/// Days since 1970-01-01 for a proleptic-Gregorian civil date (Howard
/// Hinnant's `days_from_civil`), used only by `parse_created_at`.
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let mp = (m as i64 + 9) % 12; // [0, 11]
    let doy = (153 * mp + 2) / 5 + d as i64 - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468
}

/// Parses one `datetime.isoformat()` shape the adapter emits — offset-aware
/// `+HH:MM`/`-HH:MM`/`Z`, with or without a fractional-second suffix — into
/// epoch seconds. `None` on any deviation from that shape.
fn parse_iso8601(s: &str) -> Option<i64> {
    let bytes = s.as_bytes();
    if bytes.len() < 19 {
        return None;
    }
    let year: i64 = s.get(0..4)?.parse().ok()?;
    if bytes.get(4) != Some(&b'-') {
        return None;
    }
    let month: u32 = s.get(5..7)?.parse().ok()?;
    if bytes.get(7) != Some(&b'-') {
        return None;
    }
    let day: u32 = s.get(8..10)?.parse().ok()?;
    match bytes.get(10) {
        Some(b'T') | Some(b' ') => {}
        _ => return None,
    }
    let hour: i64 = s.get(11..13)?.parse().ok()?;
    if bytes.get(13) != Some(&b':') {
        return None;
    }
    let minute: i64 = s.get(14..16)?.parse().ok()?;
    if bytes.get(16) != Some(&b':') {
        return None;
    }
    let second: i64 = s.get(17..19)?.parse().ok()?;

    let mut idx = 19usize;
    if bytes.get(idx) == Some(&b'.') {
        idx += 1;
        let start = idx;
        while idx < bytes.len() && bytes[idx].is_ascii_digit() {
            idx += 1;
        }
        if idx == start {
            return None;
        }
    }

    let mut offset_secs: i64 = 0;
    if idx < bytes.len() {
        match bytes[idx] {
            b'Z' | b'z' => idx += 1,
            b'+' | b'-' => {
                let sign: i64 = if bytes[idx] == b'+' { 1 } else { -1 };
                idx += 1;
                let oh: i64 = s.get(idx..idx + 2)?.parse().ok()?;
                idx += 2;
                if bytes.get(idx) == Some(&b':') {
                    idx += 1;
                }
                let om: i64 = s.get(idx..idx + 2)?.parse().ok()?;
                idx += 2;
                offset_secs = sign * (oh * 3600 + om * 60);
            }
            _ => return None,
        }
    }
    if idx != bytes.len() {
        return None;
    }
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) || hour > 23 || minute > 59 || second > 60 {
        return None;
    }

    let days = days_from_civil(year, month, day);
    Some(days * 86400 + hour * 3600 + minute * 60 + second - offset_secs)
}

/// The `created_at` ingest-boundary conversion: empty input is the
/// documented "unknown" default (epoch 0); a non-empty string that fails to
/// parse is `i64::MIN`, never silently collapsed into the empty-string 0 —
/// no scorer currently reads the age term from this column.
fn parse_created_at(s: &str) -> i64 {
    if s.is_empty() {
        return 0;
    }
    parse_iso8601(s).unwrap_or(i64::MIN)
}

fn intern16(pool: &mut Vec<String>, lookup: &mut HashMap<String, u16>, s: &str) -> u16 {
    if let Some(&id) = lookup.get(s) {
        return id;
    }
    // Closed vocabulary (tier names, edge types): id fits u16. A pool past
    // u16::MAX would wrap and collide two distinct strings onto one id.
    debug_assert!(pool.len() < u16::MAX as usize, "intern16 pool overflow: id would truncate");
    let id = pool.len() as u16;
    pool.push(s.to_string());
    lookup.insert(s.to_string(), id);
    id
}

fn intern32(pool: &mut Vec<String>, lookup: &mut HashMap<String, u32>, s: &str) -> u32 {
    if let Some(&id) = lookup.get(s) {
        return id;
    }
    let id = pool.len() as u32;
    pool.push(s.to_string());
    lookup.insert(s.to_string(), id);
    id
}

fn build_lookup16(pool: &[String]) -> HashMap<String, u16> {
    pool.iter().enumerate().map(|(i, s)| (s.clone(), i as u16)).collect()
}

#[derive(thiserror::Error, Debug)]
pub enum RankIndexError {
    #[error("vector length {got} does not match index dimension {expected}")]
    DimMismatch { got: usize, expected: usize },
    #[error("duplicate id {0} in bulk ingest")]
    DuplicateId(u128),
    #[error("feed target {0} not found for delete")]
    UnknownId(u128),
    #[error("generation {requested} is behind the currently published generation {current}")]
    GenerationRegression { requested: u64, current: u64 },
    #[error("text arena would grow to {attempted} bytes, past the u32 span-offset limit")]
    ArenaOverflow { attempted: u64 },
    #[error("fused-score input {field:?} has {got} entries/index, expected {expected}")]
    ScoreInputMismatch {
        field: &'static str,
        got: usize,
        expected: usize,
    },
}

impl From<RankIndexError> for PyErr {
    fn from(e: RankIndexError) -> PyErr {
        match e {
            RankIndexError::DimMismatch { .. }
            | RankIndexError::DuplicateId(_)
            | RankIndexError::ScoreInputMismatch { .. } => PyValueError::new_err(e.to_string()),
            RankIndexError::UnknownId(_)
            | RankIndexError::GenerationRegression { .. }
            | RankIndexError::ArenaOverflow { .. } => PyRuntimeError::new_err(e.to_string()),
        }
    }
}

/// Refuses an append that would push the arena past the `u32` span-offset
/// limit -- a pure length check, callable from a test with synthetic sizes
/// without allocating a multi-gigabyte `Vec`.
fn check_arena_capacity(current_len: usize, additional: usize) -> Result<(), RankIndexError> {
    let new_len = current_len + additional;
    if new_len > u32::MAX as usize {
        return Err(RankIndexError::ArenaOverflow {
            attempted: new_len as u64,
        });
    }
    Ok(())
}

/// Appends `bytes` to the arena and returns its `(offset, len)` span,
/// guarded by `check_arena_capacity` -- the single arena-growing entry
/// point, used by both bulk ingest and the incremental apply path so the
/// overflow fence can never be bypassed by a second, unchecked append site.
fn append_arena(arena: &mut Vec<u8>, bytes: &[u8]) -> Result<(u32, u32), RankIndexError> {
    check_arena_capacity(arena.len(), bytes.len())?;
    let off = arena.len() as u32;
    arena.extend_from_slice(bytes);
    Ok((off, bytes.len() as u32))
}

/// One-way 32-bit feature-id for a token -- no stored mapping from this id
/// back to token text exists anywhere in the resident index (irreversible
/// by construction, mirrors `content_hash128`'s fixed-key/in-process-only
/// discipline with a distinct key so the two hashes never coincide).
/// Deterministic within/across processes of the same build, so index-time
/// and query-time hashing of the same token always agree.
fn token_feature_id(token: &str) -> u32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    0x7f4a_7c15_9c4e_2b31u64.hash(&mut h);
    token.hash(&mut h);
    h.finish() as u32
}

/// Term-frequency map for one document's surface text, keyed by
/// `token_feature_id` -- never a literal token string. The per-touched-id
/// unit of work the incremental apply path pays, never re-tokenizing any
/// other resident document.
fn token_freq_map(surface: &str) -> HashMap<u32, u32> {
    let mut freq: HashMap<u32, u32> = HashMap::new();
    for t in tokenize(surface) {
        *freq.entry(token_feature_id(&t)).or_insert(0) += 1;
    }
    freq
}

/// Content-identity hash for the scalar-upsert same-surface skip: a
/// composite of two `DefaultHasher` runs over different prefixed inputs
/// (same fixed SipHash key, not independent seeds), no new crate
/// dependency. Not a search key and never persisted -- only ever
/// compared against a hash computed in the same process, so
/// `DefaultHasher`'s build-specific (not cross-Rust-version-stable) output
/// is safe here. A single 64-bit hasher is NOT an acceptable substitute:
/// a collision here silently staleifies a slot's `token_freqs`/postings, a
/// permanent correctness bug, not a self-correcting cost.
fn content_hash128(s: &str) -> u128 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h1 = DefaultHasher::new();
    0xa5a5_a5a5_a5a5_a5a5u64.hash(&mut h1);
    s.hash(&mut h1);
    let mut h2 = DefaultHasher::new();
    0x5a5a_5a5a_5a5a_5a5au64.hash(&mut h2);
    s.len().hash(&mut h2);
    s.hash(&mut h2);
    (u128::from(h1.finish()) << 64) | u128::from(h2.finish())
}

/// Packs three consecutive Unicode scalar values into one `u64` -- exact
/// and reversible (Unicode scalars are < 0x110000, well under 2^21 per
/// slot), never a hash. Call-scoped only: the >=128-bit irreversible-hash
/// bar (D-268-1) binds structures that outlive a single recall call; this
/// value is built and discarded within one `trigram_t11_flags` call and
/// never stored, so an exact/reversible encoding carries no residency risk
/// and gives byte-identical Jaccard with zero collision probability
/// (stronger than a hash bound).
fn pack_trigram(a: u32, b: u32, c: u32) -> u64 {
    ((a as u64) << 42) | ((b as u64) << 21) | (c as u64)
}

/// Sorted, deduplicated trigram-feature set for one already-lowercased
/// string -- mirrors Python's `{s[i:i+3] for i in range(len(s) - 2)}`
/// exactly (code-point windows, not byte windows). Empty for `len < 3`,
/// matching `_trigram_jaccard`'s own short-circuit.
fn trigram_features(s: &str) -> Vec<u64> {
    let cps: Vec<u32> = s.chars().map(|c| c as u32).collect();
    if cps.len() < 3 {
        return Vec::new();
    }
    let mut features: Vec<u64> = (0..cps.len() - 2)
        .map(|i| pack_trigram(cps[i], cps[i + 1], cps[i + 2]))
        .collect();
    features.sort_unstable();
    features.dedup();
    features
}

/// Jaccard over two sorted, deduplicated trigram-feature sets via a merge
/// scan -- mirrors `_trigram_jaccard`'s `len(set_a & set_b) / len(set_a |
/// set_b)`, `0.0` when either side is empty or the union is empty.
fn trigram_jaccard_exact(a: &[u64], b: &[u64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let mut i = 0usize;
    let mut j = 0usize;
    let mut intersection = 0usize;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                intersection += 1;
                i += 1;
                j += 1;
            }
        }
    }
    let union = a.len() + b.len() - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Batch T11 (trigram-jaccard > 0.3) flags for a candidate pool slice --
/// the reducible soft_gate cost `_t11_t12_flags` used to pay per-candidate
/// in pure Python (rebuilding two string-substring sets per call). Callers
/// must pass already-lowercased inputs (the same `.lower()` Python already
/// applied) so casing behavior stays exactly Python's, not Rust's Unicode
/// casing table. Released-GIL: string extraction happens before the
/// detach, the O(candidates) merge-scan work after.
#[pyfunction]
pub fn trigram_t11_flags(
    py: Python<'_>,
    cue_lower: String,
    surfaces_lower: Vec<String>,
) -> Vec<bool> {
    py.detach(|| {
        let cue_features = trigram_features(&cue_lower);
        surfaces_lower
            .iter()
            .map(|s| trigram_jaccard_exact(&cue_features, &trigram_features(s)) > 0.3)
            .collect()
    })
}

struct PostingsCsr {
    /// feature_id (`token_feature_id`) -> compact CSR index -- never a
    /// token string.
    token_lookup: HashMap<u32, u32>,
    post_offsets: Vec<u32>,
    post_doc_slots: Vec<u32>,
    post_tfs: Vec<u32>,
    doc_len: Vec<u32>,
    n_docs: u32,
    avg_len: f64,
    /// Exact sum of `doc_len` -- carried alongside the rounded/floored
    /// `avg_len` so the incremental apply path can adjust it exactly
    /// (integer arithmetic) rather than re-deriving it from `avg_len`.
    total_len: u64,
}

/// Rebuilds the whole postings CSR by aggregating the per-slot
/// `token_freqs` maps -- already tokenized at the transient-surface
/// touchpoints (`from_columns`, `upsert_scalar`), so this never re-reads
/// or re-tokenizes any resident raw surface bytes. The token pool and CSR
/// value/offset arrays are re-derived here, never from a stored token
/// list, so this is the single postings builder used both at bulk ingest
/// and at every write-triggered rebuild.
fn rebuild_postings_csr(n: usize, token_freqs: &[HashMap<u32, u32>]) -> PostingsCsr {
    let mut token_lookup: HashMap<u32, u32> = HashMap::new();
    let mut token_doc_count: Vec<u32> = Vec::new();
    let mut raw_postings: Vec<(u32, u32, u32)> = Vec::new(); // (compact_index, slot, tf)
    let mut doc_len = Vec::with_capacity(n);
    let mut total_len: u64 = 0;

    for slot in 0..n {
        let local_freq = &token_freqs[slot];
        let slot_len: u32 = local_freq.values().sum();
        doc_len.push(slot_len);
        total_len += slot_len as u64;

        for (&feature_id, &tf) in local_freq {
            let compact_id = match token_lookup.get(&feature_id) {
                Some(&id) => id,
                None => {
                    let id = token_lookup.len() as u32;
                    token_lookup.insert(feature_id, id);
                    id
                }
            };
            if compact_id as usize == token_doc_count.len() {
                token_doc_count.push(0);
            }
            token_doc_count[compact_id as usize] += 1;
            raw_postings.push((compact_id, slot as u32, tf));
        }
    }

    let ntokens = token_lookup.len();
    let mut post_offsets = Vec::with_capacity(ntokens + 1);
    post_offsets.push(0u32);
    for &c in &token_doc_count {
        post_offsets.push(post_offsets.last().unwrap() + c);
    }
    let total_postings = raw_postings.len();
    let mut post_doc_slots = vec![0u32; total_postings];
    let mut post_tfs = vec![0u32; total_postings];
    let mut cursor: Vec<u32> = post_offsets[..ntokens].to_vec();
    for (token_id, slot, tf) in raw_postings {
        let pos = cursor[token_id as usize] as usize;
        post_doc_slots[pos] = slot;
        post_tfs[pos] = tf;
        cursor[token_id as usize] += 1;
    }

    let n_docs = n as u32;
    let avg_len = if n_docs == 0 {
        1.0
    } else {
        (total_len as f64 / n_docs as f64).max(1.0)
    };

    PostingsCsr {
        token_lookup,
        post_offsets,
        post_doc_slots,
        post_tfs,
        doc_len,
        n_docs,
        avg_len,
        total_len,
    }
}

/// Rebuilds the adjacency CSR in `ids` slot order from an id-keyed working
/// map — the single adjacency builder used both at bulk ingest (the map is
/// the raw `edges` param) and at fold time (the map is every current id's
/// resolved adjacency, see `Inner::fold`).
fn rebuild_adjacency_csr(
    ids: &[u128],
    working: &HashMap<u128, Vec<(u128, f32, String)>>,
) -> (Vec<String>, Vec<u128>, Vec<f32>, Vec<u16>, Vec<u32>) {
    let n = ids.len();
    let total_edges: usize = ids
        .iter()
        .map(|id| working.get(id).map(|v| v.len()).unwrap_or(0))
        .sum();
    let mut edge_type_pool: Vec<String> = Vec::new();
    let mut edge_type_lookup: HashMap<String, u16> = HashMap::new();
    let mut adj_neighbors = Vec::with_capacity(total_edges);
    let mut adj_weights = Vec::with_capacity(total_edges);
    let mut adj_type_ids = Vec::with_capacity(total_edges);
    let mut adj_offsets = Vec::with_capacity(n + 1);
    adj_offsets.push(0u32);
    for id in ids {
        if let Some(edges) = working.get(id) {
            for (neighbor, weight, edge_type) in edges {
                adj_neighbors.push(*neighbor);
                adj_weights.push(*weight);
                adj_type_ids.push(intern16(&mut edge_type_pool, &mut edge_type_lookup, edge_type));
            }
        }
        adj_offsets.push(adj_neighbors.len() as u32);
    }
    (edge_type_pool, adj_neighbors, adj_weights, adj_type_ids, adj_offsets)
}

/// Rebuilds the tags CSR in `ids` slot order from an id-keyed working map —
/// mirrors `rebuild_adjacency_csr`.
fn rebuild_tags_csr(
    ids: &[u128],
    working: &HashMap<u128, Vec<String>>,
) -> (Vec<String>, Vec<u32>, Vec<u32>) {
    let n = ids.len();
    let total_tags: usize = ids
        .iter()
        .map(|id| working.get(id).map(|v| v.len()).unwrap_or(0))
        .sum();
    let mut tag_pool: Vec<String> = Vec::new();
    let mut tag_lookup: HashMap<String, u32> = HashMap::new();
    let mut tag_ids = Vec::with_capacity(total_tags);
    let mut tag_offsets = Vec::with_capacity(n + 1);
    tag_offsets.push(0u32);
    for id in ids {
        if let Some(tags) = working.get(id) {
            for t in tags {
                tag_ids.push(intern32(&mut tag_pool, &mut tag_lookup, t));
            }
        }
        tag_offsets.push(tag_ids.len() as u32);
    }
    (tag_pool, tag_ids, tag_offsets)
}

/// Per-id adjacency/tags/postings deltas for ids touched (upserted or
/// deleted) since the last full CSR rebuild (`Inner::fold`). Read
/// accessors resolve an id through this map FIRST, falling back to the
/// committed CSR (via `committed_id_to_slot`) only for an id the overlay
/// carries no verdict for. Bounded by touched-id count, never corpus
/// size; reset to empty by `fold`. Reuses the bounded id-keyed
/// replace-in-place overlay pattern already established in this codebase
/// (`store/_recency_buffer.py`), not a new data-structure shape.
#[derive(Clone, Debug, Default)]
struct Overlay {
    /// Present key = this id's CURRENT adjacency, replacing (never
    /// appending to) whatever the committed CSR holds for this id. The key
    /// set of this map IS the "touched, still-resident" set — every other
    /// overlay accessor treats membership here as the touched marker.
    adjacency: HashMap<u128, Vec<(u128, f32, String)>>,
    tags: HashMap<u128, Vec<String>>,
    /// `token_feature_id` -> term-frequency for a touched id's CURRENT
    /// surface text -- never a token string. A touched id's doc length is
    /// the sum of this map's values.
    postings: HashMap<u128, HashMap<u32, u32>>,
    /// Ids removed since the last fold — a committed-CSR row for one of
    /// these ids must be excluded everywhere a lookup could otherwise
    /// resolve it via `committed_id_to_slot`.
    tombstones: HashSet<u128>,
}

impl Overlay {
    fn is_empty(&self) -> bool {
        self.adjacency.is_empty() && self.tombstones.is_empty()
    }

    /// Distinct ids the overlay carries a verdict for (touched + deleted)
    /// — the read-cost bound this whole design exists to keep small.
    fn touched_len(&self) -> usize {
        let mut ids: HashSet<u128> = self.adjacency.keys().copied().collect();
        ids.extend(self.tombstones.iter().copied());
        ids.len()
    }

    fn is_touched(&self, id: u128) -> bool {
        self.adjacency.contains_key(&id)
    }

    fn is_tombstoned(&self, id: u128) -> bool {
        self.tombstones.contains(&id)
    }

    fn record_upsert(
        &mut self,
        id: u128,
        edges: Vec<(u128, f32, String)>,
        tags: Vec<String>,
        postings: HashMap<u32, u32>,
    ) {
        self.tombstones.remove(&id);
        self.adjacency.insert(id, edges);
        self.tags.insert(id, tags);
        self.postings.insert(id, postings);
        // `is_touched`/`touched_len`/`is_empty` all treat `adjacency`'s key
        // set as the authoritative "touched" set -- if a future edit ever
        // let these three maps' key sets diverge, is_empty() could report
        // empty (and fold() would silently skip real work) while tags or
        // postings still carried a stranded delta.
        debug_assert!(
            self.adjacency.contains_key(&id) && self.tags.contains_key(&id) && self.postings.contains_key(&id),
            "record_upsert must insert into adjacency/tags/postings in lockstep for id {id}"
        );
    }

    fn record_delete(&mut self, id: u128) {
        self.adjacency.remove(&id);
        self.tags.remove(&id);
        self.postings.remove(&id);
        self.tombstones.insert(id);
        debug_assert!(
            !self.adjacency.contains_key(&id) && !self.tags.contains_key(&id) && !self.postings.contains_key(&id),
            "record_delete must remove id {id} from adjacency/tags/postings in lockstep"
        );
    }

    fn clear(&mut self) {
        self.adjacency.clear();
        self.tags.clear();
        self.postings.clear();
        self.tombstones.clear();
    }
}

/// The resident rank-feature struct: struct-of-arrays, slot-indexed (slot =
/// position in `ids`). Immutable once published: mutation only happens by
/// building a new `Inner` and swapping the published pointer — see
/// `DoubleBuffer`. No `structure_hv` column exists: the naive layout's slot
/// had no producer anywhere in the type system and is dropped, not carried
/// forward as a stub.
///
/// `text_arena` holds ONLY `aaak_index` bytes -- the owner-locked
/// zero-new-plaintext invariant forbids a whole-corpus resident raw
/// literal-surface copy; `aaak_index` is the accepted keyword class (parsed
/// entity/doc-tag anchors, not a sentence). A candidate's literal surface
/// is scored via Python-computed T11/T12 flags (`FusedScoreParams`), never
/// read here.
///
/// Two column families live here under different freshness contracts. The
/// SCALAR/VECTOR/text columns (`ids`, `id_to_slot`, `vectors`, `stability`,
/// `centrality`, `salience_level`, `created_at`, `tier`, `pending`,
/// `text_arena`, `aaak_span`, `token_freqs`, `surface_hash`) are always
/// CURRENT — every `feed()` op is applied to them immediately on the next
/// `snapshot()`. The
/// CSR columns (`adj_*`, `tag_*`, `token_*`, `post_*`, `doc_len`, `n_docs`,
/// `avg_len`, `total_len`) are COMMITTED — valid only against
/// `committed_ids`/`committed_id_to_slot`, which can lag `ids`/`id_to_slot`
/// between folds. A committed array must never be indexed by a current
/// slot; resolve an id's live CSR-derived value through `overlay` first,
/// falling back to `committed_id_to_slot` — see `resolved_adjacency` et al.
#[derive(Clone, Debug)]
pub struct Inner {
    dim: usize,
    generation: u64,
    ids: Vec<u128>,
    id_to_slot: HashMap<u128, u32>,
    vectors: Vec<f32>, // n * dim, contiguous
    stability: Vec<f32>,
    centrality: Vec<f32>,
    salience_level: Vec<u8>,
    created_at: Vec<i64>, // epoch seconds
    tier_pool: Vec<String>,
    tier: Vec<u16>,
    /// `embedding_pending` membership. Fed by the Python adapter's bulk
    /// pending-row scan (`_build`) and by `feed`'s `pending` kwarg on an
    /// incremental upsert -- always current, never overlay-resolved (same
    /// freshness class as the other scalar columns). A `true` row must
    /// still participate in postings/BM25 (nothing in the postings build
    /// consults this flag); callers doing vector/cosine work over the
    /// resident matrix must filter it out themselves.
    pending: Vec<bool>,
    text_arena: Vec<u8>,
    aaak_span: Vec<(u32, u32)>,
    /// Per-slot token-frequency map for the CURRENT surface, keyed by
    /// `token_feature_id` -- never a literal token string (the
    /// zero-new-plaintext lock extended to tokens: no readable word is
    /// resident here). The postings rebuild source
    /// (`rebuild_postings_csr`), replacing a resident raw surface arena.
    /// Populated once at every scalar/text touchpoint (`from_columns`,
    /// `upsert_scalar`), never re-derived from stored raw bytes.
    token_freqs: Vec<HashMap<u32, u32>>,
    /// Per-slot content-identity hash of the last-seen surface -- the
    /// scalar-upsert same-surface skip compares against this instead of an
    /// arena byte-slice (`content_hash128`). Not a search key, not
    /// persisted -- only ever compared within the same process.
    surface_hash: Vec<u128>,
    tag_pool: Vec<String>,
    tag_ids: Vec<u32>,
    tag_offsets: Vec<u32>,
    edge_type_pool: Vec<String>,
    adj_neighbors: Vec<u128>, // u128: a neighbor may be non-resident
    adj_weights: Vec<f32>,
    adj_type_ids: Vec<u16>,
    adj_offsets: Vec<u32>, // slot -> range, len = n + 1
    /// feature_id (`token_feature_id`) -> compact CSR index -- never a
    /// token string.
    token_lookup: HashMap<u32, u32>,
    post_doc_slots: Vec<u32>, // doc SLOT per posting, not id
    post_tfs: Vec<u32>,
    post_offsets: Vec<u32>, // token id -> range, len = ntokens + 1
    doc_len: Vec<u32>,
    n_docs: u32,
    avg_len: f64,
    /// Exact sum of `doc_len` as of the last fold — see `PostingsCsr::total_len`.
    total_len: u64,
    /// Id order the COMMITTED CSR columns above are indexed by — distinct
    /// from `ids` once `feed()`/incremental apply has touched the buffer
    /// since the last fold.
    committed_ids: Vec<u128>,
    committed_id_to_slot: HashMap<u128, u32>,
    overlay: Overlay,
}

impl Inner {
    /// Builds a fresh `Inner` straight from the parallel columns handed in
    /// by the PyO3 boundary — no per-record wrapper struct is constructed;
    /// every resident array is pre-sized exactly from a counting pass before
    /// being filled once. `pub` so the crate's own benches can build
    /// production-shaped fixtures without a Python/GIL round trip; no
    /// PyO3 binding exposes this constructor.
    #[allow(clippy::too_many_arguments)]
    pub fn from_columns(
        dim: usize,
        generation: u64,
        ids: Vec<u128>,
        vectors_flat: Vec<f32>,
        edge_map: HashMap<u128, Vec<(u128, f32, String)>>,
        surfaces: Vec<String>,
        aaak_index: Vec<String>,
        created_at: Vec<i64>,
        stability: Vec<f32>,
        tier: Vec<String>,
        tags: Vec<Vec<String>>,
        salience_level: Vec<u8>,
        centrality: Vec<f32>,
        pending: Vec<bool>,
    ) -> Result<Inner, RankIndexError> {
        let n = ids.len();
        if vectors_flat.len() != n * dim {
            return Err(RankIndexError::DimMismatch {
                got: vectors_flat.len(),
                expected: n * dim,
            });
        }

        let mut id_to_slot: HashMap<u128, u32> = HashMap::with_capacity(n);
        for (slot, id) in ids.iter().enumerate() {
            if id_to_slot.insert(*id, slot as u32).is_some() {
                return Err(RankIndexError::DuplicateId(*id));
            }
        }

        // Text arena, pre-sized exactly from the aaak_index byte lengths —
        // one allocation for every aaak string, not one alloc per string.
        // No surface bytes go here (zero-new-plaintext lock): the token
        // representation below is the only surface-derived resident.
        let arena_bytes: usize = aaak_index.iter().map(|s| s.len()).sum::<usize>();
        let mut text_arena = Vec::with_capacity(arena_bytes);
        let mut aaak_span = Vec::with_capacity(n);
        let mut token_freqs: Vec<HashMap<u32, u32>> = Vec::with_capacity(n);
        let mut surface_hash: Vec<u128> = Vec::with_capacity(n);
        for (surface, aaak) in surfaces.into_iter().zip(aaak_index.into_iter()) {
            aaak_span.push(append_arena(&mut text_arena, aaak.as_bytes())?);
            surface_hash.push(content_hash128(&surface));
            token_freqs.push(token_freq_map(&surface));
        }

        let mut tier_pool: Vec<String> = Vec::new();
        let mut tier_lookup: HashMap<String, u16> = HashMap::new();
        let tier_col: Vec<u16> = tier
            .iter()
            .map(|t| intern16(&mut tier_pool, &mut tier_lookup, t))
            .collect();

        let tags_working: HashMap<u128, Vec<String>> =
            ids.iter().copied().zip(tags.into_iter()).collect();
        let (tag_pool, tag_ids, tag_offsets) = rebuild_tags_csr(&ids, &tags_working);

        let (edge_type_pool, adj_neighbors, adj_weights, adj_type_ids, adj_offsets) =
            rebuild_adjacency_csr(&ids, &edge_map);

        let postings = rebuild_postings_csr(n, &token_freqs);
        let committed_ids = ids.clone();
        let committed_id_to_slot = id_to_slot.clone();

        Ok(Inner {
            dim,
            generation,
            ids,
            id_to_slot,
            vectors: vectors_flat,
            stability,
            centrality,
            salience_level,
            created_at,
            tier_pool,
            tier: tier_col,
            pending,
            text_arena,
            aaak_span,
            token_freqs,
            surface_hash,
            tag_pool,
            tag_ids,
            tag_offsets,
            edge_type_pool,
            adj_neighbors,
            adj_weights,
            adj_type_ids,
            adj_offsets,
            token_lookup: postings.token_lookup,
            post_doc_slots: postings.post_doc_slots,
            post_tfs: postings.post_tfs,
            post_offsets: postings.post_offsets,
            doc_len: postings.doc_len,
            n_docs: postings.n_docs,
            avg_len: postings.avg_len,
            total_len: postings.total_len,
            committed_ids,
            committed_id_to_slot,
            overlay: Overlay::default(),
        })
    }

    /// Insert-or-overwrite the identity/vector/scalar/text columns for
    /// `id`. Adjacency and tags are untouched here — their CSR offset
    /// tables cover every slot at once, so a single record's edge/tag-count
    /// change can only be applied by a full CSR rebuild (see
    /// `rebuild_adjacency_csr`/`rebuild_tags_csr`), never in place.
    #[allow(clippy::too_many_arguments)]
    fn upsert_scalar(
        &mut self,
        id: u128,
        vector: &[f32],
        surface: &str,
        surface_token_freqs: &HashMap<u32, u32>,
        aaak_index: &str,
        created_at: i64,
        stability: f32,
        tier: &str,
        salience_level: u8,
        centrality: f32,
        pending: bool,
        tier_lookup: &mut HashMap<String, u16>,
    ) -> Result<(), RankIndexError> {
        if vector.len() != self.dim {
            return Err(RankIndexError::DimMismatch {
                got: vector.len(),
                expected: self.dim,
            });
        }
        let tier_id = intern16(&mut self.tier_pool, tier_lookup, tier);
        let surface_hash = content_hash128(surface);

        match self.id_to_slot.get(&id).copied() {
            Some(slot) => {
                let slot = slot as usize;
                // A scalar-only update (e.g. a salience raise) carries the
                // SAME surface/aaak content as what is already resident —
                // skip the token-representation recompute / arena append
                // entirely rather than redoing work on every touch.
                // Surface and aaak are compared and updated independently:
                // one can be unchanged while the other changed.
                let same_surface = self.surface_hash[slot] == surface_hash;
                if !same_surface {
                    self.token_freqs[slot] = surface_token_freqs.clone();
                    self.surface_hash[slot] = surface_hash;
                }
                let aaak_bytes = aaak_index.as_bytes();
                let same_aaak = {
                    let (off, len) = self.aaak_span[slot];
                    &self.text_arena[off as usize..off as usize + len as usize] == aaak_bytes
                };
                let a_span = if same_aaak {
                    self.aaak_span[slot]
                } else {
                    append_arena(&mut self.text_arena, aaak_bytes)?
                };

                let start = slot * self.dim;
                self.vectors[start..start + self.dim].copy_from_slice(vector);
                self.stability[slot] = stability;
                self.centrality[slot] = centrality;
                self.salience_level[slot] = salience_level;
                self.created_at[slot] = created_at;
                self.tier[slot] = tier_id;
                self.pending[slot] = pending;
                self.aaak_span[slot] = a_span;
            }
            None => {
                // A new id has no existing hash/span to compare against —
                // always applies.
                let a_span = append_arena(&mut self.text_arena, aaak_index.as_bytes())?;
                let slot = self.ids.len() as u32;
                self.vectors.extend_from_slice(vector);
                self.ids.push(id);
                self.id_to_slot.insert(id, slot);
                self.stability.push(stability);
                self.centrality.push(centrality);
                self.salience_level.push(salience_level);
                self.created_at.push(created_at);
                self.tier.push(tier_id);
                self.pending.push(pending);
                self.aaak_span.push(a_span);
                self.token_freqs.push(surface_token_freqs.clone());
                self.surface_hash.push(surface_hash);
            }
        }
        Ok(())
    }

    /// Removes `id` and compacts its slot with a swap-remove across every
    /// parallel column, keeping the columns dense. Errs on an unknown id —
    /// callers that need "delete if present" swallow that error explicitly
    /// (see `DoubleBuffer::snapshot`).
    fn delete_scalar(&mut self, id: u128) -> Result<(), RankIndexError> {
        let slot = self
            .id_to_slot
            .remove(&id)
            .ok_or(RankIndexError::UnknownId(id))?;
        let last = self.ids.len() as u32 - 1;
        if slot != last {
            let (slot, last) = (slot as usize, last as usize);
            let moved_id = self.ids[last];
            self.ids.swap(slot, last);
            let dim = self.dim;
            let (a_start, b_start) = (slot * dim, last * dim);
            let b_row: Vec<f32> = self.vectors[b_start..b_start + dim].to_vec();
            let a_row: Vec<f32> = self.vectors[a_start..a_start + dim].to_vec();
            self.vectors[a_start..a_start + dim].copy_from_slice(&b_row);
            self.vectors[b_start..b_start + dim].copy_from_slice(&a_row);
            self.stability.swap(slot, last);
            self.centrality.swap(slot, last);
            self.salience_level.swap(slot, last);
            self.created_at.swap(slot, last);
            self.tier.swap(slot, last);
            self.pending.swap(slot, last);
            self.aaak_span.swap(slot, last);
            self.token_freqs.swap(slot, last);
            self.surface_hash.swap(slot, last);
            self.id_to_slot.insert(moved_id, slot as u32);
        }
        let last = last as usize;
        self.ids.truncate(last);
        self.vectors.truncate(last * self.dim);
        self.stability.truncate(last);
        self.centrality.truncate(last);
        self.salience_level.truncate(last);
        self.created_at.truncate(last);
        self.tier.truncate(last);
        self.pending.truncate(last);
        self.aaak_span.truncate(last);
        self.token_freqs.truncate(last);
        self.surface_hash.truncate(last);
        Ok(())
    }

    /// Committed adjacency for `id`, resolved through
    /// `committed_id_to_slot` — NEVER a current slot index, which can
    /// disagree with committed slot order once a delete has swap-removed a
    /// different id into this id's old position.
    fn committed_adjacency(&self, id: u128) -> Vec<(u128, f32, String)> {
        match self.committed_id_to_slot.get(&id) {
            Some(&slot) => {
                let slot = slot as usize;
                let start = self.adj_offsets[slot] as usize;
                let end = self.adj_offsets[slot + 1] as usize;
                (start..end)
                    .map(|i| {
                        let et = self.edge_type_pool[self.adj_type_ids[i] as usize].clone();
                        (self.adj_neighbors[i], self.adj_weights[i], et)
                    })
                    .collect()
            }
            None => Vec::new(),
        }
    }

    fn committed_tags(&self, id: u128) -> Vec<String> {
        match self.committed_id_to_slot.get(&id) {
            Some(&slot) => {
                let slot = slot as usize;
                let start = self.tag_offsets[slot] as usize;
                let end = self.tag_offsets[slot + 1] as usize;
                (start..end)
                    .map(|i| self.tag_pool[self.tag_ids[i] as usize].clone())
                    .collect()
            }
            None => Vec::new(),
        }
    }

    fn committed_doc_len(&self, id: u128) -> u32 {
        match self.committed_id_to_slot.get(&id) {
            Some(&slot) => self.doc_len[slot as usize],
            None => 0,
        }
    }

    /// Current adjacency for `id`: the overlay's verdict if `id` was
    /// touched (upserted or deleted) since the last fold, else the
    /// committed CSR. O(1) map lookup plus the touched edge count — never
    /// a corpus-wide scan.
    fn resolved_adjacency(&self, id: u128) -> Vec<(u128, f32, String)> {
        if let Some(edges) = self.overlay.adjacency.get(&id) {
            return edges.clone();
        }
        if self.overlay.is_tombstoned(id) {
            return Vec::new();
        }
        self.committed_adjacency(id)
    }

    fn resolved_tags(&self, id: u128) -> Vec<String> {
        if let Some(tags) = self.overlay.tags.get(&id) {
            return tags.clone();
        }
        if self.overlay.is_tombstoned(id) {
            return Vec::new();
        }
        self.committed_tags(id)
    }

    fn resolved_degree(&self, id: u128) -> u32 {
        if let Some(edges) = self.overlay.adjacency.get(&id) {
            return edges.len() as u32;
        }
        if self.overlay.is_tombstoned(id) {
            return 0;
        }
        match self.committed_id_to_slot.get(&id) {
            Some(&slot) => {
                let slot = slot as usize;
                self.adj_offsets[slot + 1] - self.adj_offsets[slot]
            }
            None => 0,
        }
    }

    /// Current document length for `id` — the sum of the overlay's
    /// token-frequency map for a touched id, else the committed value. 0
    /// for a tombstoned or never-resident id.
    fn resolved_doc_len(&self, id: u128) -> u32 {
        if let Some(freq) = self.overlay.postings.get(&id) {
            return freq.values().sum();
        }
        if self.overlay.is_tombstoned(id) {
            return 0;
        }
        self.committed_doc_len(id)
    }

    /// Current resident document count — always `ids.len()`, never the
    /// committed `n_docs` field (which can lag between folds).
    pub fn current_n_docs(&self) -> u32 {
        self.ids.len() as u32
    }

    /// Current average document length across every CURRENTLY resident id.
    /// Starts from the committed `total_len` and adjusts it by every id the
    /// overlay carries a verdict for: a touched id's committed contribution
    /// is subtracted once and its live contribution added back (0 for a
    /// tombstoned id). O(overlay size), never O(corpus).
    pub fn current_avg_len(&self) -> f64 {
        let mut total: i64 = self.total_len as i64;
        let mut adjusted: HashSet<u128> = self.overlay.adjacency.keys().copied().collect();
        adjusted.extend(self.overlay.tombstones.iter().copied());
        for id in adjusted {
            // `resolved_doc_len` already resolves to 0 for a tombstoned
            // id, and `committed_doc_len` already resolves to 0 for an id
            // that never had a committed row (a new id) -- both no-op
            // correctly for either case without a separate branch.
            total -= self.committed_doc_len(id) as i64;
            total += self.resolved_doc_len(id) as i64;
        }
        let n = self.ids.len();
        if n == 0 {
            1.0
        } else {
            (total.max(0) as f64 / n as f64).max(1.0)
        }
    }

    /// Current `(id, tf)` postings for one token: committed postings for
    /// ids the overlay has NOT touched, merged with any overlay-touched id
    /// whose current token set contains this token. Owned (not a borrowed
    /// slice) because the merge cannot be a view into either source alone.
    /// O(committed postings for this token + overlay size).
    fn resolved_postings_for_token(&self, token: &str) -> Vec<(u128, u32)> {
        let feature_id = token_feature_id(token);
        let mut out = Vec::new();
        if let Some(&token_id) = self.token_lookup.get(&feature_id) {
            let start = self.post_offsets[token_id as usize] as usize;
            let end = self.post_offsets[token_id as usize + 1] as usize;
            for i in start..end {
                let id = self.committed_ids[self.post_doc_slots[i] as usize];
                if self.overlay.is_touched(id) || self.overlay.is_tombstoned(id) {
                    // This id's current state is resolved from the overlay
                    // branch below instead — the committed row is stale.
                    continue;
                }
                out.push((id, self.post_tfs[i]));
            }
        }
        for (&id, freq) in &self.overlay.postings {
            if let Some(&tf) = freq.get(&feature_id) {
                out.push((id, tf));
            }
        }
        out
    }

    /// Applies one queued write to the CURRENT scalar/vector/text columns
    /// (`upsert_scalar`/`delete_scalar`, already cheap — O(1) plus this
    /// one document's own byte length) and records the adjacency/tags/
    /// postings delta in `overlay`. Never touches the committed CSR — this
    /// is the entire read-path apply cost: O(1) columns plus O(this doc's
    /// token count), never O(corpus).
    fn apply_incremental(
        &mut self,
        op: PendingOp,
        tier_lookup: &mut HashMap<String, u16>,
    ) -> Result<(), RankIndexError> {
        match op {
            PendingOp::Upsert {
                id,
                vector,
                edges,
                surface,
                aaak_index,
                created_at,
                stability,
                tier,
                tags,
                salience_level,
                centrality,
                pending,
            } => {
                let postings = token_freq_map(&surface);
                self.upsert_scalar(
                    id,
                    &vector,
                    &surface,
                    &postings,
                    &aaak_index,
                    created_at,
                    stability,
                    &tier,
                    salience_level,
                    centrality,
                    pending,
                    tier_lookup,
                )?;
                self.overlay.record_upsert(id, edges, tags, postings);
            }
            PendingOp::Delete { id } => {
                // A replayed delete for an id already absent (deleted
                // twice before a fold, or never resident) is a benign
                // no-op, not a failure of the apply.
                let _ = self.delete_scalar(id);
                self.overlay.record_delete(id);
            }
        }
        Ok(())
    }

    /// Rebuilds `text_arena`/`aaak_span` from the CURRENT live slot set —
    /// reclaims every orphaned aaak byte an in-place update left behind
    /// (see `upsert_scalar`'s tail-append). `token_freqs`/`surface_hash`
    /// need no reclamation: they are per-slot scalars, already kept in
    /// current slot order by `upsert_scalar`/`delete_scalar`, never an
    /// accumulated arena log.
    fn compact_arena(&mut self) {
        let mut new_arena = Vec::with_capacity(self.text_arena.len());
        let mut new_aaak_span = Vec::with_capacity(self.ids.len());
        for slot in 0..self.ids.len() {
            let (off, len) = self.aaak_span[slot];
            let new_off = new_arena.len() as u32;
            new_arena.extend_from_slice(&self.text_arena[off as usize..off as usize + len as usize]);
            new_aaak_span.push((new_off, len));
        }
        self.text_arena = new_arena;
        self.aaak_span = new_aaak_span;
    }

    /// Off-critical-path full rebuild: folds every overlay-touched id's
    /// adjacency/tags/postings back into the committed CSR, reclaims
    /// orphaned arena bytes, and resets the overlay to empty. Must never be
    /// reachable from the recall read path — `DoubleBuffer::snapshot`'s
    /// stale branch never calls this; only `DoubleBuffer::fold` does.
    fn fold(&mut self) {
        self.compact_arena();

        let mut adj_map: HashMap<u128, Vec<(u128, f32, String)>> =
            HashMap::with_capacity(self.ids.len());
        let mut tag_map: HashMap<u128, Vec<String>> = HashMap::with_capacity(self.ids.len());
        for &id in &self.ids {
            let edges = self.resolved_adjacency(id);
            if !edges.is_empty() {
                adj_map.insert(id, edges);
            }
            let tags = self.resolved_tags(id);
            if !tags.is_empty() {
                tag_map.insert(id, tags);
            }
        }

        let (edge_type_pool, adj_neighbors, adj_weights, adj_type_ids, adj_offsets) =
            rebuild_adjacency_csr(&self.ids, &adj_map);
        self.edge_type_pool = edge_type_pool;
        self.adj_neighbors = adj_neighbors;
        self.adj_weights = adj_weights;
        self.adj_type_ids = adj_type_ids;
        self.adj_offsets = adj_offsets;

        let (tag_pool, tag_ids, tag_offsets) = rebuild_tags_csr(&self.ids, &tag_map);
        self.tag_pool = tag_pool;
        self.tag_ids = tag_ids;
        self.tag_offsets = tag_offsets;

        let postings = rebuild_postings_csr(self.ids.len(), &self.token_freqs);
        self.token_lookup = postings.token_lookup;
        self.post_offsets = postings.post_offsets;
        self.post_doc_slots = postings.post_doc_slots;
        self.post_tfs = postings.post_tfs;
        self.doc_len = postings.doc_len;
        self.n_docs = postings.n_docs;
        self.avg_len = postings.avg_len;
        self.total_len = postings.total_len;

        self.committed_ids = self.ids.clone();
        self.committed_id_to_slot = self.id_to_slot.clone();
        self.overlay.clear();
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn ids(&self) -> Vec<u128> {
        self.ids.clone()
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Read-only column accessors for the disposable scan-cost bench
    /// (`benches/scan_cost.rs`) only — Rust-visibility `pub`, no PyO3
    /// binding, no scoring formula. Mirrors `from_columns`'s widening
    /// rationale: a bench crate can only reach public items.
    pub fn vector_row(&self, slot: usize) -> &[f32] {
        let start = slot * self.dim;
        &self.vectors[start..start + self.dim]
    }

    pub fn created_at(&self) -> &[i64] {
        &self.created_at
    }

    pub fn stability(&self, slot: usize) -> f32 {
        self.stability[slot]
    }

    pub fn salience_level(&self, slot: usize) -> u8 {
        self.salience_level[slot]
    }

    pub fn degree(&self, slot: usize) -> u32 {
        self.adj_offsets[slot + 1] - self.adj_offsets[slot]
    }

    pub fn aaak_text(&self, slot: usize) -> &str {
        let (off, len) = self.aaak_span[slot];
        std::str::from_utf8(&self.text_arena[off as usize..off as usize + len as usize])
            .unwrap_or("")
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.token_lookup.get(&token_feature_id(token)).copied()
    }

    /// Total resident byte footprint of the per-slot token-frequency maps,
    /// fixed-width per entry regardless of the original token's length --
    /// the mechanical proxy for the zero-new-plaintext lock on TOKEN
    /// storage (mirrors `resident_text_arena_len`'s role for `aaak_index`).
    pub fn resident_token_footprint_bytes(&self) -> usize {
        self.token_freqs
            .iter()
            .map(|m| m.len() * std::mem::size_of::<(u32, u32)>())
            .sum()
    }

    pub fn posting_slots(&self, token_id: u32) -> &[u32] {
        let i = token_id as usize;
        let start = self.post_offsets[i] as usize;
        let end = self.post_offsets[i + 1] as usize;
        &self.post_doc_slots[start..end]
    }

    pub fn posting_tfs(&self, token_id: u32) -> &[u32] {
        let i = token_id as usize;
        let start = self.post_offsets[i] as usize;
        let end = self.post_offsets[i + 1] as usize;
        &self.post_tfs[start..end]
    }

    pub fn doc_len(&self, slot: usize) -> u32 {
        self.doc_len[slot]
    }

    pub fn n_docs(&self) -> u32 {
        self.n_docs
    }

    pub fn avg_len(&self) -> f64 {
        self.avg_len
    }
}

/// An incremental write queued between rebuilds. `feed` only ever appends
/// here — the published buffer is never mutated in place.
#[derive(Clone)]
pub enum PendingOp {
    Upsert {
        id: u128,
        vector: Vec<f32>,
        edges: Vec<(u128, f32, String)>,
        surface: String,
        aaak_index: String,
        created_at: i64,
        stability: f32,
        tier: String,
        tags: Vec<String>,
        salience_level: u8,
        centrality: f32,
        pending: bool,
    },
    Delete {
        id: u128,
    },
}

/// Generation-tagged active/standby buffer. `feed` enqueues into `pending`
/// lock-free relative to readers; `snapshot` on a matching generation is a
/// pure read (`Arc` clone, no lock beyond `active`'s). A stale target holds
/// `rebuild` for its whole drain -> clone -> rebuild -> commit body -- two
/// stale-path rebuilds racing on the same base must never both drain
/// `pending` against it, or the loser's drained op is gone for good.
/// Readers never take `rebuild`; they only clone the `Arc` under `active`'s
/// lock, held solely for that O(1) pointer read/store.
pub struct DoubleBuffer {
    dim: usize,
    active: Mutex<Arc<Inner>>,
    pending: Mutex<VecDeque<PendingOp>>,
    // Serializes the stale-path drain -> clone -> rebuild -> commit body.
    // Two rebuilds racing on the same stale base (different target
    // generations) must never both drain `pending` against it -- the
    // loser's drained op would be gone for good. Readers (the matching-
    // generation fast path) never take this lock.
    rebuild: Mutex<()>,
}

impl DoubleBuffer {
    pub fn new(inner: Inner) -> Self {
        DoubleBuffer {
            dim: inner.dim,
            active: Mutex::new(Arc::new(inner)),
            pending: Mutex::new(VecDeque::new()),
            rebuild: Mutex::new(()),
        }
    }

    /// Enqueues an incremental write; never touches the published buffer.
    /// The only lock taken is `pending`'s, held for the push.
    pub fn feed(&self, op: PendingOp) -> Result<(), RankIndexError> {
        if let PendingOp::Upsert { vector, .. } = &op {
            if vector.len() != self.dim {
                return Err(RankIndexError::DimMismatch {
                    got: vector.len(),
                    expected: self.dim,
                });
            }
        }
        self.pending.lock().push_back(op);
        Ok(())
    }

    /// A matching generation is a pure read: the published `Arc` is
    /// cloned (O(1)) and returned without consulting `pending`. A stale
    /// target clones the published `Inner` and applies the queued ops to
    /// the clone INCREMENTALLY (`Inner::apply_incremental` — scalar/vector
    /// columns updated in place, adjacency/tags/postings deltas recorded
    /// in the overlay), never a wholesale CSR rebuild; the clone is then
    /// committed with one O(1) pointer store under `active`'s lock. A
    /// regression (target below what's already published) is refused —
    /// the generation this struct reports must be monotonic, mirroring
    /// `graph._pool_content_version` itself never decreasing. The
    /// committed CSR only advances via `fold`, never here — this method
    /// must stay off the ~160-270ms wholesale-rebuild cost class entirely.
    pub fn snapshot(&self, generation: u64) -> Result<Arc<Inner>, RankIndexError> {
        let current = self.active.lock().clone();
        if current.generation == generation {
            return Ok(current);
        }
        if generation < current.generation {
            return Err(RankIndexError::GenerationRegression {
                requested: generation,
                current: current.generation,
            });
        }

        // Serializes the whole drain-apply-commit body against any other
        // stale-path apply; readers above never reach this line.
        let _rebuild_guard = self.rebuild.lock();

        // The published buffer may have advanced while this thread waited
        // for the rebuild lock -- re-read and re-check before draining.
        let current = self.active.lock().clone();
        if current.generation == generation {
            return Ok(current);
        }
        if generation < current.generation {
            return Err(RankIndexError::GenerationRegression {
                requested: generation,
                current: current.generation,
            });
        }

        let ops: Vec<PendingOp> = self.pending.lock().drain(..).collect();
        let mut next = (*current).clone();
        let mut tier_lookup = build_lookup16(&next.tier_pool);

        for op in ops {
            next.apply_incremental(op, &mut tier_lookup)?;
        }

        next.generation = generation;
        let next_arc = Arc::new(next);

        let mut guard = self.active.lock();
        if generation > guard.generation {
            *guard = Arc::clone(&next_arc);
        }
        Ok(Arc::clone(&guard))
    }

    pub fn current(&self) -> Arc<Inner> {
        self.active.lock().clone()
    }

    /// Distinct ids the currently published buffer's overlay carries a
    /// verdict for — a caller uses this to decide when `fold` is worth
    /// triggering (an idle tick, sleep-pipeline step, or size threshold),
    /// never to decide anything on the recall read path.
    pub fn overlay_len(&self) -> usize {
        self.active.lock().overlay.touched_len()
    }

    /// Off-critical-path full rebuild: folds every overlay-touched id's
    /// adjacency/tags/postings back into the committed CSR, reclaims
    /// orphaned arena bytes, and resets the overlay to empty. Takes the
    /// same `rebuild` mutex `snapshot`'s stale branch does, so the two
    /// never race each other. A caller MUST keep this off the recall
    /// critical path — it pays the same wholesale-rebuild cost class
    /// `snapshot` no longer does. A no-op (returns the published buffer
    /// unchanged) when the overlay is already empty.
    pub fn fold(&self) -> Arc<Inner> {
        let _rebuild_guard = self.rebuild.lock();
        let current = self.active.lock().clone();
        if current.overlay.is_empty() {
            return current;
        }
        let mut next = (*current).clone();
        next.fold();
        let next_arc = Arc::new(next);
        let mut guard = self.active.lock();
        *guard = Arc::clone(&next_arc);
        Arc::clone(&guard)
    }
}

// Fused rank scorer -- the corpus-resident subset of the fused rank
// formula, computed over a per-call candidate scope. Constants below are
// frozen against `pipeline.py`; any drift here silently changes ranking.
pub const W_COSINE: f64 = 1.0;
pub const W_AAAK: f64 = 0.3;
pub const W_AGE: f64 = 0.05;

/// Differential-gate non-vacuity hook: `IAI_MCP_RANK_PERTURB_W_COSINE`
/// overrides the cosine weight for one process's `fused_score` calls, proving
/// a real Rust scoring regression turns the differential RED. Precedence:
/// the env var (if set and parseable) wins over `fallback`, which callers
/// pass as `params.effective_w_cosine` -- the tuned weight, or `W_COSINE`
/// when untuned. Read once per `fused_score` call, never inside the
/// per-slot scan loop.
fn perturb_w_cosine(fallback: f64) -> f64 {
    std::env::var("IAI_MCP_RANK_PERTURB_W_COSINE")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(fallback)
}
const AGE_HALF_LIFE_DAYS: f64 = 30.0;
const STRUCTURE_HV_DIM: usize = 10000;
const CLEANUP_MAX_HAMMING_FRAC: f64 = 0.15;
const CLEANUP_SHORTLIST_CAP: usize = 200;
const LEX_FUSION_K: usize = 32;
const AAAK_CUE_STRIP: &str = " \t\r\n.,;:!?()[]{}\"'«»";

fn popcount_xor(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x ^ y).count_ones()).sum()
}

/// `1.0 - popcount(a XOR b) / STRUCTURE_HV_DIM` — ported from
/// `lilli/ops/hebbian.py::structural_similarity`; `0.0` on any length
/// mismatch or empty input, matching the Python guard.
fn structural_similarity(a: &[u8], b: &[u8]) -> f64 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    1.0 - (popcount_xor(a, b) as f64 / STRUCTURE_HV_DIM as f64)
}

/// Fractional Hamming distance — ported from `lilli/core/similarity.py::hamming`.
fn hamming_frac(a: &[u8], b: &[u8]) -> f64 {
    if a.len() != b.len() {
        return 1.0;
    }
    if a.is_empty() {
        return 0.0;
    }
    popcount_xor(a, b) as f64 / (a.len() * 8) as f64
}

/// Nearest codebook entry by Hamming distance, first-wins tie-break —
/// ported from `lilli/ops/cleanup.py::cleanup` (Python's `min()` keeps the
/// first minimal element under a strict `<` comparison).
fn cleanup_nearest<'a>(noisy: &[u8], codebook: &[&'a [u8]]) -> &'a [u8] {
    let mut best_dist = f64::INFINITY;
    let mut best = codebook[0];
    for entry in codebook {
        let d = hamming_frac(noisy, entry);
        if d < best_dist {
            best_dist = d;
            best = entry;
        }
    }
    best
}

/// Snap `noisy` to its nearest codebook entry only within the rejection
/// threshold — ported from `lilli/ops/cleanup.py::_cleanup_if_confident`.
fn cleanup_if_confident(noisy: &[u8], codebook: &[&[u8]]) -> Vec<u8> {
    if codebook.is_empty() {
        return noisy.to_vec();
    }
    let cleaned = cleanup_nearest(noisy, codebook);
    if hamming_frac(noisy, cleaned) > CLEANUP_MAX_HAMMING_FRAC {
        noisy.to_vec()
    } else {
        cleaned.to_vec()
    }
}

/// Cue-normalized token set — ported from `pipeline.py::_aaak_overlap`'s
/// cue-side extraction; computed once per call, not per candidate.
fn aaak_cue_tokens(cue_text: &str) -> HashSet<String> {
    let lowered = cue_text.to_lowercase().replace('/', " ");
    lowered
        .split_whitespace()
        .filter_map(|raw| {
            let t = raw.trim_matches(|c: char| AAAK_CUE_STRIP.contains(c));
            if t.is_empty() {
                None
            } else {
                Some(t.to_string())
            }
        })
        .collect()
}

/// Entity/doc-tag anchor set from an `aaak_index` string — ported from
/// `aaak.py::parse_aaak_index` + `pipeline.py::_aaak_overlap`'s
/// meaningful-token extraction (only the `E:`/`T:doc:` segments feed
/// anchors; wing/room segments never do).
fn aaak_meaningful_tokens(aaak_index: &str) -> HashSet<String> {
    let mut meaningful = HashSet::new();
    for seg in aaak_index.split('/') {
        let Some((key, value)) = seg.split_once(':') else {
            continue;
        };
        match key {
            "E" => {
                if value != "-" && !value.is_empty() {
                    for ent in value.split(',') {
                        for w in ent.to_lowercase().split_whitespace() {
                            meaningful.insert(w.to_string());
                        }
                    }
                }
            }
            "T" => {
                if value != "-" && !value.is_empty() {
                    for tag in value.split(',') {
                        if let Some((tk, tv)) = tag.split_once(':') {
                            if tk.eq_ignore_ascii_case("doc") && !tv.trim().is_empty() {
                                meaningful.insert(tv.to_lowercase().trim().to_string());
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    meaningful
}

/// Cue-normalized containment match — ported from `pipeline.py::_aaak_overlap`.
/// `cue_tokens` is precomputed once per call, not re-derived per candidate.
fn aaak_overlap(cue_tokens: &HashSet<String>, aaak_index: &str) -> f64 {
    if aaak_index.is_empty() || cue_tokens.is_empty() {
        return 0.0;
    }
    let meaningful = aaak_meaningful_tokens(aaak_index);
    if meaningful.is_empty() {
        return 0.0;
    }
    let mut matched = 0usize;
    for tok in cue_tokens {
        if meaningful.contains(tok) {
            matched += 1;
            continue;
        }
        let mut found = false;
        for anchor in &meaningful {
            let (short, long_) = if tok.chars().count() <= anchor.chars().count() {
                (tok.as_str(), anchor.as_str())
            } else {
                (anchor.as_str(), tok.as_str())
            };
            let short_len = short.chars().count();
            let long_len = long_.chars().count();
            if short_len >= 5 && long_len - short_len <= 3 && long_.starts_with(short) {
                found = true;
                break;
            }
        }
        if found {
            matched += 1;
        }
    }
    matched as f64 / cue_tokens.len() as f64
}

/// Ported from `pipeline.py::_age_penalty`; `now` is a required, frozen
/// caller input — never a wall-clock fallback, so two calls in a
/// differential comparison stay deterministic.
fn age_penalty(created_at: i64, now: i64) -> f64 {
    let days = now.saturating_sub(created_at) as f64 / 86400.0;
    if days < 0.0 {
        return 0.0;
    }
    (days / AGE_HALF_LIFE_DAYS).min(1.0)
}

/// Ported from `pipeline.py::_flat_cosine_damp`: proportional ramp, `1.0`
/// at or above `threshold`, `0.0` at a fully flat head; `1.0` unconditionally
/// when `threshold <= 0.0`.
fn flat_cosine_damp(head_spread: f64, threshold: f64) -> f64 {
    if threshold <= 0.0 {
        return 1.0;
    }
    (head_spread / threshold).clamp(0.0, 1.0)
}

/// Deduplicated (order-preserving), length-capped query tokens — ported
/// from `store/_lexical_index.py::query`'s `dict.fromkeys(tokenize(text))
/// [:MAX_QUERY_TOKENS]`, reusing this crate's already-parity-proven
/// `tokenize`.
fn cue_query_tokens(cue: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for t in tokenize(cue) {
        if seen.insert(t.clone()) {
            out.push(t);
            if out.len() >= MAX_QUERY_TOKENS {
                break;
            }
        }
    }
    out
}

/// Highest IDF among the cue's in-corpus tokens — ported from
/// `LexicalIndex.max_idf`, reading resident postings via the overlay-aware
/// `resolved_postings_for_token` so a post-write call never scores a
/// stale document frequency.
fn max_idf_for_cue(inner: &Inner, cue: &str) -> f64 {
    let tokens = cue_query_tokens(cue);
    let n_docs = inner.current_n_docs() as f64;
    let mut best = 0.0f64;
    for t in &tokens {
        let df = inner.resolved_postings_for_token(t).len() as f64;
        if df > 0.0 {
            let idf = (1.0 + (n_docs - df + 0.5) / (df + 0.5)).ln();
            if idf > best {
                best = idf;
            }
        }
    }
    best
}

fn bm25_score(tf: f64, df: f64, length: f64, n_docs: f64, avg_len: f64) -> f64 {
    let idf = (1.0 + (n_docs - df + 0.5) / (df + 0.5)).ln();
    let denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * length / avg_len);
    idf * tf * (BM25_K1 + 1.0) / denom
}

/// Cue rank map (`id -> 0-indexed BM25 rank`) computed INTERNALLY from the
/// resident postings — no `LexicalIndex`/precomputed side-input. Ported
/// from `LexicalIndex.query`: AND-of-tokens with a soft fallback (a token
/// intersection that would go empty is skipped, not applied), BM25-scored,
/// sorted by `(-score, id)`, truncated to `LEX_FUSION_K`.
fn compute_lex_rank(inner: &Inner, cue: &str) -> HashMap<u128, usize> {
    let tokens = cue_query_tokens(cue);
    if tokens.is_empty() {
        return HashMap::new();
    }
    let mut per_token: Vec<Vec<(u128, u32)>> =
        tokens.iter().map(|t| inner.resolved_postings_for_token(t)).collect();
    if per_token.iter().any(|p| p.is_empty()) {
        per_token.retain(|p| !p.is_empty());
        if per_token.is_empty() {
            return HashMap::new();
        }
    }
    let mut ids: HashSet<u128> = per_token[0].iter().map(|&(id, _)| id).collect();
    for p in &per_token[1..] {
        let pset: HashSet<u128> = p.iter().map(|&(id, _)| id).collect();
        let nxt: HashSet<u128> = ids.intersection(&pset).copied().collect();
        if !nxt.is_empty() {
            ids = nxt;
        }
    }
    let n_docs = inner.current_n_docs() as f64;
    let avg_len = inner.current_avg_len();
    let mut scored: Vec<(u128, f64)> = Vec::with_capacity(ids.len());
    for id in ids {
        let mut score = 0.0;
        for p in &per_token {
            if let Some(&(_, tf)) = p.iter().find(|&&(pid, _)| pid == id) {
                let df = p.len() as f64;
                let length = inner.resolved_doc_len(id) as f64;
                score += bm25_score(tf as f64, df, length, n_docs, avg_len);
            }
        }
        scored.push((id, score));
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
    scored.truncate(LEX_FUSION_K);
    scored.into_iter().enumerate().map(|(rank, (id, _))| (id, rank)).collect()
}

/// Current per-id degree, filtered to non-excluded edge types — the
/// byte-identity trap this exists to avoid: `degree_map()`/`resolved_degree`
/// counts ALL edges with no type filter, while today's degree term
/// excludes similarity/entity-anchor edges. Never call `resolved_degree`
/// for the fused score; this is the only degree accessor it uses.
fn degree_excluding(inner: &Inner, id: u128, excluded_edge_types: &HashSet<String>) -> u32 {
    inner
        .resolved_adjacency(id)
        .into_iter()
        .filter(|(_, _, et)| !excluded_edge_types.contains(et))
        .count() as u32
}

fn check_score_len(field: &'static str, got: usize, expected: usize) -> Result<(), RankIndexError> {
    if got != expected {
        return Err(RankIndexError::ScoreInputMismatch { field, got, expected });
    }
    Ok(())
}

/// One candidate's two-hop spread provenance: the seed it was reached
/// from, hop distance, and whether the path carries transfer (entity
/// anchors only — the similarity/hebbian mesh must not carry it).
#[derive(Clone, Copy, Debug)]
pub struct SpreadEntry {
    pub seed_id: u128,
    pub hop: u32,
    pub transfer_carrying: bool,
}

/// Per-call inputs to `fused_score`. Candidate SELECTION (community gate,
/// seeds, 2-hop spread, escalation widen) stays caller-orchestrated; the
/// four index arrays here are already that selection's OUTPUT, in pool
/// position space (`pool_ids[i]` is the id at pool position `i`).
pub struct FusedScoreParams<'a> {
    pub pool_ids: &'a [u128],
    /// Per-call ANN cosine, parallel to `pool_ids`.
    pub cosine: &'a [f64],
    /// Per-candidate structure HV (plaintext, already resident in the
    /// caller's rank-tier candidate view) — not an `Inner` column, so this
    /// crosses as a per-call side-input, parallel to `pool_ids`.
    pub structure_hv: &'a [Option<Vec<u8>>],
    pub cosine_top_indices: &'a [u32],
    pub spread_indices: &'a [u32],
    pub rich_indices: &'a [u32],
    pub lex_indices: &'a [u32],
    /// T11 (trigram-jaccard>0.3, x2.0) and T12 (whole-cue substring, x3.0),
    /// parallel to `pool_ids` -- computed Python-side from the candidate
    /// pool's own hydrated surfaces (never a resident surface read here).
    pub t11_flags: &'a [bool],
    pub t12_flags: &'a [bool],
    pub verbatim_filter: bool,
    pub cue: &'a str,
    pub now: i64,
    /// Pre-damp, lp-scaled, verbatim-zeroed value; the flat-cosine damp is
    /// applied internally.
    pub effective_w_degree: f64,
    /// The tuned cosine weight (or `W_COSINE` when untuned); `perturb_w_cosine`
    /// reads this as its fallback, so an env-unset process scores against the
    /// tuned value rather than always the module default.
    pub effective_w_cosine: f64,
    pub excluded_edge_types: &'a HashSet<String>,
    pub spread_provenance: &'a HashMap<u128, SpreadEntry>,
    pub w_spread_act: f64,
    pub spread_act_decay: f64,
    pub community_id_by_member: &'a HashMap<u128, u128>,
    pub community_scores: &'a HashMap<u128, f64>,
    pub max_community_score: f64,
    /// Pre-damp value; the flat-cosine damp is applied internally.
    pub mode_bias: f64,
    pub cos_spread_min: f64,
    pub structural_weight: f64,
    pub cue_structure_hv: Option<&'a [u8]>,
    pub lex_lane_enabled: bool,
    pub min_idf: f64,
    pub lex_fusion_w: f64,
    pub k: usize,
    pub k_margin: usize,
}

/// One over-fetched winner: `partial_score` sums only the corpus-resident
/// terms — the raw fields below are already-resident values the caller
/// applies per-call-state adjustments to (profile gain, tier/salience
/// gates, the temporal match boost, the historical-verbatim rewrite)
/// without a second fetch.
///
/// `pre_gain_base` and `term_multiplier` let a caller correctly reinsert a
/// per-call multiplicative gain at the SAME point today's Python formula
/// applies it (before the stability lift, inside the trigram/FTS
/// multiplier chain), without a second scoring pass:
/// `partial_score == (pre_gain_base + stability_lift) * term_multiplier + lex_add`.
/// Applying a gain `g` to `partial_score` directly would scale the
/// stability/lex terms too, which today's formula never does; the correct
/// reconstruction is
/// `partial_score + pre_gain_base * term_multiplier * (g - 1.0)`
/// (algebraically equivalent to recomputing with `pre_gain_base * g` in
/// place of `pre_gain_base`, without needing the stability/lex terms
/// separately).
#[derive(Clone, Debug)]
pub struct WinnerRow {
    pub id: u128,
    pub partial_score: f64,
    /// The cosine/aaak/degree/age/spread/community/structural sum, BEFORE
    /// the stability lift and the trigram/FTS/lex terms — the exact value
    /// a per-call multiplicative gain multiplies against today.
    pub pre_gain_base: f64,
    /// Accumulated trigram (`2.0`) * FTS (`3.0`) multiplier, `1.0` if
    /// neither fired.
    pub term_multiplier: f64,
    pub created_at: i64,
    pub salience_level: u8,
    pub tier: String,
    pub tags: Vec<String>,
    /// Raw Bucket-A per-term values, exposed so a caller can reconstruct
    /// the served-hit diagnostic breakdown for the bounded winner set
    /// without a second scoring pass over the candidate pool.
    pub cosine: f64,
    pub aaak: f64,
    pub deg_norm: f64,
    pub age: f64,
    pub spread_contrib: f64,
    pub community_contrib: f64,
    pub structural_score: f64,
}

/// Replaces the two per-call Python set builds this call boundary retires:
/// `expected` is the candidate scope size after the union and (if any)
/// verbatim filter; `resident` is how many of those pool positions
/// resolved to an actually-resident id in this `Inner`.
#[derive(Clone, Copy, Debug)]
pub struct CoverageInvariant {
    pub expected: u32,
    pub resident: u32,
    pub all_resident: bool,
}

#[derive(Clone, Debug)]
pub struct FusedScoreResult {
    pub winners: Vec<WinnerRow>,
    pub coverage: CoverageInvariant,
    /// `1.0` unless the candidate head's cosine spread fell below
    /// `cos_spread_min`, in which case this is the ramp factor already
    /// folded into the degree/community terms above — a caller applying
    /// the same damp to a later tier/salience boost step reads it from
    /// here rather than re-deriving it.
    pub flat_cosine_damp: f64,
}

/// Fused score over today's exact candidate-set construction: the
/// four-way ascending-pool-position union, the verbatim episodic filter,
/// the cleanup-attractor shortlist, the flat-cosine damp, and each frozen
/// corpus-resident term (cosine, aaak overlap, edge-type-excluded degree,
/// age, spread activation, community contribution, structural blend,
/// stability lift, trigram/FTS/BM25 boosts) — reading corpus-resident
/// features through the overlay-aware accessors so a post-write call
/// scores current, not stale, data. Returns the `k + k_margin` winners
/// ordered by `(-partial_score, id)` (ascending id breaks a tie; proven
/// equal to the `str(UUID(int=id))` string tie-break by
/// `tie_break_matches_uuid_string_order` in `tests/fused_score.rs`).
///
/// A valence multiplier is not computed: the live rank-tier candidate
/// view (`store/_store.py::RankCandidateView`) carries no `valence`
/// attribute, so `getattr(rec, "valence", None) or 0.0` is always `0.0` on
/// this data tier and the multiplier never fires — verified against the
/// dataclass definition, not assumed.
pub fn fused_score(inner: &Inner, params: &FusedScoreParams<'_>) -> Result<FusedScoreResult, RankIndexError> {
    let n_pool = params.pool_ids.len();
    check_score_len("cosine", params.cosine.len(), n_pool)?;
    check_score_len("structure_hv", params.structure_hv.len(), n_pool)?;
    check_score_len("t11_flags", params.t11_flags.len(), n_pool)?;
    check_score_len("t12_flags", params.t12_flags.len(), n_pool)?;
    for (name, idxs) in [
        ("cosine_top_indices", params.cosine_top_indices),
        ("spread_indices", params.spread_indices),
        ("rich_indices", params.rich_indices),
        ("lex_indices", params.lex_indices),
    ] {
        for &pos in idxs {
            if pos as usize >= n_pool {
                return Err(RankIndexError::ScoreInputMismatch {
                    field: name,
                    got: pos as usize,
                    expected: n_pool,
                });
            }
        }
    }

    let w_cosine = perturb_w_cosine(params.effective_w_cosine);

    let mut union_set: BTreeSet<u32> = BTreeSet::new();
    union_set.extend(params.cosine_top_indices.iter().copied());
    union_set.extend(params.spread_indices.iter().copied());
    union_set.extend(params.rich_indices.iter().copied());
    union_set.extend(params.lex_indices.iter().copied());
    let mut reachable: Vec<u32> = union_set.into_iter().collect();

    if params.verbatim_filter {
        reachable.retain(|&pos| {
            let id = params.pool_ids[pos as usize];
            match inner.id_to_slot.get(&id) {
                // `getattr(rec, "tier", "episodic")`'s default fires when
                // the attribute is genuinely absent; an empty resident tier
                // string (an unset upstream field) must be treated the
                // same way, not as a non-episodic tier.
                Some(&slot) => {
                    let t = inner.tier_pool[inner.tier[slot as usize] as usize].as_str();
                    t.is_empty() || t == "episodic"
                }
                None => false,
            }
        });
    }
    let expected = reachable.len() as u32;

    let cleanup_shortlist: Vec<&[u8]> = reachable
        .iter()
        .take(CLEANUP_SHORTLIST_CAP)
        .filter_map(|&pos| {
            let id = params.pool_ids[pos as usize];
            if !inner.id_to_slot.contains_key(&id) {
                return None;
            }
            params.structure_hv[pos as usize].as_deref().filter(|hv| !hv.is_empty())
        })
        .collect();

    let mut max_deg: u32 = 0;
    for &id in &inner.ids {
        let d = degree_excluding(inner, id, params.excluded_edge_types);
        if d > max_deg {
            max_deg = d;
        }
    }
    let log_max_deg = if max_deg > 0 { (1.0 + max_deg as f64).ln() } else { 0.0 };

    let (effective_w_degree, mode_bias, flat_cosine_damp_val) = {
        let mut w_degree = params.effective_w_degree;
        let mut bias = params.mode_bias;
        let mut damp = 1.0f64;
        if reachable.len() >= 3 {
            let mut pool_cos: Vec<f64> = reachable.iter().map(|&pos| params.cosine[pos as usize]).collect();
            pool_cos.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
            let head_len = pool_cos.len().min(10);
            let head = &pool_cos[..head_len];
            let cos_spread = head[0] - head[head_len - 1];
            damp = flat_cosine_damp(cos_spread, params.cos_spread_min);
            if damp < 1.0 {
                w_degree *= damp;
                bias *= damp;
            }
        }
        (w_degree, bias, damp)
    };

    let id_to_pool_pos: HashMap<u128, u32> =
        params.pool_ids.iter().enumerate().map(|(i, &id)| (id, i as u32)).collect();

    let cue_tokens_for_aaak = aaak_cue_tokens(params.cue);
    let cue_nonempty = !params.cue.is_empty();

    let lex_rank: HashMap<u128, usize> = if params.lex_lane_enabled && cue_nonempty {
        if max_idf_for_cue(inner, params.cue) >= params.min_idf {
            compute_lex_rank(inner, params.cue)
        } else {
            HashMap::new()
        }
    } else {
        HashMap::new()
    };

    let mut resident: u32 = 0;
    let mut winners: Vec<WinnerRow> = Vec::with_capacity(reachable.len());
    for &pos in &reachable {
        let id = params.pool_ids[pos as usize];
        let slot = match inner.id_to_slot.get(&id) {
            Some(&s) => s as usize,
            None => continue,
        };
        resident += 1;

        let cos = params.cosine[pos as usize];
        let aaak = aaak_overlap(&cue_tokens_for_aaak, inner.aaak_text(slot));
        let deg = degree_excluding(inner, id, params.excluded_edge_types) as f64;
        let deg_norm = if log_max_deg > 0.0 { (1.0 + deg).ln() / log_max_deg } else { 0.0 };
        let age = age_penalty(inner.created_at[slot], params.now);

        let mut base_s = w_cosine * cos + W_AAAK * aaak + effective_w_degree * deg_norm - W_AGE * age;

        let mut spread_contrib = 0.0f64;
        if params.w_spread_act > 0.0 {
            if let Some(prov) = params.spread_provenance.get(&id) {
                if prov.transfer_carrying {
                    if let Some(&seed_pos) = id_to_pool_pos.get(&prov.seed_id) {
                        let seed_cos = params.cosine[seed_pos as usize];
                        spread_contrib = params.w_spread_act * seed_cos * params.spread_act_decay.powi(prov.hop as i32);
                        base_s += spread_contrib;
                    }
                }
            }
        }

        let mut community_contrib = 0.0f64;
        if let Some(&community_id) = params.community_id_by_member.get(&id) {
            if params.max_community_score > 0.0 {
                let cs = params.community_scores.get(&community_id).copied().unwrap_or(0.0);
                let graded_weight = (cs / params.max_community_score).max(0.0);
                community_contrib = mode_bias * cos * graded_weight;
                base_s += community_contrib;
            }
        }

        let mut s = base_s;
        let mut structural_score = 0.0f64;
        if params.structural_weight > 0.0 {
            if let (Some(cue_hv), Some(rec_hv)) =
                (params.cue_structure_hv, params.structure_hv[pos as usize].as_deref())
            {
                if !rec_hv.is_empty() {
                    let cleaned = cleanup_if_confident(rec_hv, &cleanup_shortlist);
                    structural_score = structural_similarity(cue_hv, &cleaned);
                }
            }
            s = (1.0 - params.structural_weight) * base_s + params.structural_weight * structural_score;
        }
        // The exact value a later per-call multiplicative gain must
        // multiply against, matching today's insertion point.
        let pre_gain_base = s;

        let raw_stability = inner.stability[slot] as f64;
        let stability = if raw_stability == 0.0 { 0.5 } else { raw_stability };
        s += (1.0 - stability.min(1.0)) * 0.1;

        let mut term_multiplier = 1.0f64;
        if params.t11_flags[pos as usize] {
            s *= 2.0;
            term_multiplier *= 2.0;
        }
        if params.t12_flags[pos as usize] {
            s *= 3.0;
            term_multiplier *= 3.0;
        }
        if let Some(&rank) = lex_rank.get(&id) {
            s += params.lex_fusion_w / (1.0 + rank as f64);
        }

        winners.push(WinnerRow {
            id,
            partial_score: s,
            pre_gain_base,
            term_multiplier,
            created_at: inner.created_at[slot],
            salience_level: inner.salience_level[slot],
            tier: inner.tier_pool[inner.tier[slot] as usize].clone(),
            tags: inner.resolved_tags(id),
            cosine: cos,
            aaak,
            deg_norm,
            age,
            spread_contrib,
            community_contrib,
            structural_score,
        });
    }

    winners.sort_by(|a, b| {
        b.partial_score
            .partial_cmp(&a.partial_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    winners.truncate(params.k + params.k_margin);

    Ok(FusedScoreResult {
        winners,
        coverage: CoverageInvariant {
            expected,
            resident,
            all_resident: resident == expected,
        },
        flat_cosine_damp: flat_cosine_damp_val,
    })
}

/// Resident wrapper the PyO3 class holds.
#[pyclass(module = "iai_mcp_native.rank")]
pub struct RankIndex {
    buf: DoubleBuffer,
}

fn flatten_vectors(
    arr: &PyReadonlyArrayDyn<'_, f32>,
    dim: usize,
    expected_n: usize,
) -> PyResult<Vec<f32>> {
    let a = arr.as_array();
    if a.ndim() != 2 {
        return Err(PyValueError::new_err(format!(
            "expected a 2-D vector matrix, got {}-D",
            a.ndim()
        )));
    }
    let shape = a.shape();
    if shape[0] != expected_n || shape[1] != dim {
        return Err(PyValueError::new_err(format!(
            "vector matrix shape {shape:?} != expected ({expected_n}, {dim})"
        )));
    }
    Ok(a.iter().copied().collect())
}

fn check_parallel_len(name: &str, len: usize, expected: usize) -> PyResult<()> {
    if len != expected {
        return Err(PyValueError::new_err(format!(
            "{name} has {len} entries, expected {expected} to match ids"
        )));
    }
    Ok(())
}

#[pymethods]
impl RankIndex {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dim: usize,
        generation: u64,
        ids: Vec<u128>,
        vectors: PyReadonlyArrayDyn<'_, f32>,
        edges: Vec<(u128, Vec<(u128, f32, String)>)>,
        surfaces: Vec<String>,
        aaak_index: Vec<String>,
        created_at: Vec<String>,
        stability: Vec<f32>,
        tier: Vec<String>,
        tags: Vec<Vec<String>>,
        salience_level: Vec<u8>,
        centrality: Vec<f32>,
        pending: Vec<bool>,
    ) -> PyResult<Self> {
        let n = ids.len();
        check_parallel_len("surfaces", surfaces.len(), n)?;
        check_parallel_len("aaak_index", aaak_index.len(), n)?;
        check_parallel_len("created_at", created_at.len(), n)?;
        check_parallel_len("stability", stability.len(), n)?;
        check_parallel_len("tier", tier.len(), n)?;
        check_parallel_len("tags", tags.len(), n)?;
        check_parallel_len("salience_level", salience_level.len(), n)?;
        check_parallel_len("centrality", centrality.len(), n)?;
        check_parallel_len("pending", pending.len(), n)?;

        let vectors_flat = flatten_vectors(&vectors, dim, n)?;
        let edge_map: HashMap<u128, Vec<(u128, f32, String)>> = edges.into_iter().collect();
        // Ingest-boundary parse: the PyO3 string param stays stable,
        // only the resident column changes shape.
        let created_at_epoch: Vec<i64> = created_at.iter().map(|s| parse_created_at(s)).collect();

        let inner = Inner::from_columns(
            dim,
            generation,
            ids,
            vectors_flat,
            edge_map,
            surfaces,
            aaak_index,
            created_at_epoch,
            stability,
            tier,
            tags,
            salience_level,
            centrality,
            pending,
        )?;
        Ok(RankIndex {
            buf: DoubleBuffer::new(inner),
        })
    }

    fn __len__(&self) -> usize {
        self.buf.current().len()
    }

    fn ids(&self) -> Vec<u128> {
        self.buf.current().ids()
    }

    fn generation(&self) -> u64 {
        self.buf.current().generation()
    }

    /// Whole embedding matrix + id order for the currently held buffer.
    /// Bulk-only: no per-record accessor exists on this surface.
    fn vectors<'py>(&self, py: Python<'py>) -> PyResult<(Vec<u128>, Bound<'py, PyAny>)> {
        let inner = self.buf.current();
        let n = inner.len();
        let mat = Array2::from_shape_vec((n, inner.dim), inner.vectors.clone())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok((inner.ids.clone(), mat.into_pyarray(py).into_any()))
    }

    /// Whole degree map (neighbor count per id), overlay-resolved so a
    /// touched id's count reflects its post-write edge list, not the
    /// committed CSR's stale one — bulk, never per-candidate.
    fn degree_map<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = self.buf.current();
        let out = PyDict::new(py);
        for &id in &inner.ids {
            out.set_item(id, inner.resolved_degree(id))?;
        }
        Ok(out)
    }

    /// Whole per-edge-type neighbor-count map (`{id: {edge_type: count}}`),
    /// overlay-resolved — bulk, never per-candidate. The ranking-degree
    /// edge-type exclusion stays Python-side: the caller sums the counts
    /// for the types it wants, over this one call's result, never a second
    /// per-node accessor.
    fn adjacency_by_type<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = self.buf.current();
        let out = PyDict::new(py);
        for &id in &inner.ids {
            let by_type = PyDict::new(py);
            for (_, _, edge_type) in inner.resolved_adjacency(id) {
                let count: u32 = by_type
                    .get_item(&edge_type)?
                    .map(|v| v.extract::<u32>())
                    .transpose()?
                    .unwrap_or(0);
                by_type.set_item(&edge_type, count + 1)?;
            }
            out.set_item(id, by_type)?;
        }
        Ok(out)
    }

    /// Whole salience-level rank map (`SALIENCE_LEVEL_RANK` u8 per id) for
    /// the currently held buffer — bulk, never per-candidate. Lets a caller
    /// verify the resident mirror of this one field without a scoring pass.
    fn salience_levels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = self.buf.current();
        let out = PyDict::new(py);
        for slot in 0..inner.len() {
            out.set_item(inner.ids[slot], inner.salience_level[slot])?;
        }
        Ok(out)
    }

    /// Whole `embedding_pending` membership map (`id -> bool`) for the
    /// currently held buffer -- bulk, never per-candidate. A cosine
    /// consumer filters this map to exclude pending rows; a lexical/BM25
    /// consumer ignores it (postings already include pending rows by
    /// construction).
    fn pending<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = self.buf.current();
        let out = PyDict::new(py);
        for slot in 0..inner.len() {
            out.set_item(inner.ids[slot], inner.pending[slot])?;
        }
        Ok(out)
    }

    /// Incremental write queued between rebuilds. `op` is `"upsert"` or
    /// `"delete"`; upsert requires `vector` and `surface`, the remaining
    /// fields default when omitted. Released-GIL hot path.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (op, id, vector=None, edges=None, surface=None, aaak_index=None,
                         created_at=None, stability=None, tier=None, tags=None,
                         salience_level=None, centrality=None, pending=None))]
    fn feed(
        &self,
        py: Python<'_>,
        op: &str,
        id: u128,
        vector: Option<PyReadonlyArrayDyn<'_, f32>>,
        edges: Option<Vec<(u128, f32, String)>>,
        surface: Option<String>,
        aaak_index: Option<String>,
        created_at: Option<String>,
        stability: Option<f32>,
        tier: Option<String>,
        tags: Option<Vec<String>>,
        salience_level: Option<u8>,
        centrality: Option<f32>,
        pending: Option<bool>,
    ) -> PyResult<()> {
        let pending_op = match op {
            "delete" => PendingOp::Delete { id },
            "upsert" => {
                let vector = vector
                    .ok_or_else(|| PyValueError::new_err("feed(\"upsert\", ...) requires vector"))?;
                let arr = vector.as_array();
                if arr.ndim() != 1 {
                    return Err(PyValueError::new_err("feed vector must be 1-D"));
                }
                let vec_data: Vec<f32> = arr.iter().copied().collect();
                PendingOp::Upsert {
                    id,
                    vector: vec_data,
                    edges: edges.unwrap_or_default(),
                    surface: surface
                        .ok_or_else(|| PyValueError::new_err("feed(\"upsert\", ...) requires surface"))?,
                    aaak_index: aaak_index.unwrap_or_default(),
                    created_at: created_at.map(|s| parse_created_at(&s)).unwrap_or(0),
                    stability: stability.unwrap_or(0.5),
                    tier: tier.unwrap_or_default(),
                    tags: tags.unwrap_or_default(),
                    salience_level: salience_level.unwrap_or(0),
                    centrality: centrality.unwrap_or(0.0),
                    pending: pending.unwrap_or(false),
                }
            }
            other => return Err(PyValueError::new_err(format!("unknown feed op {other:?}"))),
        };
        py.detach(|| self.buf.feed(pending_op))?;
        Ok(())
    }

    /// Compares `generation` against the published buffer: a match returns
    /// bulk views with the GIL released for the read; a stale value
    /// rebuilds into a private standby off-lock, swaps in one O(1) pointer
    /// store, then serves. `tokens` selects which postings buckets are
    /// returned — never a per-record accessor.
    fn snapshot<'py>(
        &self,
        py: Python<'py>,
        generation: u64,
        tokens: Vec<String>,
    ) -> PyResult<(u64, Vec<u128>, Bound<'py, PyAny>, Bound<'py, PyDict>, Bound<'py, PyDict>)> {
        let inner = py.detach(|| self.buf.snapshot(generation))?;
        let n = inner.len();
        let mat = Array2::from_shape_vec((n, inner.dim), inner.vectors.clone())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let degree = PyDict::new(py);
        for &id in &inner.ids {
            degree.set_item(id, inner.resolved_degree(id))?;
        }

        let postings = PyDict::new(py);
        for tok in &tokens {
            let resolved = inner.resolved_postings_for_token(tok);
            if resolved.is_empty() {
                continue;
            }
            let d = PyDict::new(py);
            for (doc_id, tf) in resolved {
                d.set_item(doc_id, tf)?;
            }
            postings.set_item(tok, d)?;
        }

        Ok((
            inner.generation(),
            inner.ids.clone(),
            mat.into_pyarray(py).into_any(),
            degree,
            postings,
        ))
    }

    /// Off-critical-path full rebuild: folds every overlay-touched id's
    /// adjacency/tags/postings back into the committed CSR and reclaims
    /// orphaned arena bytes. The caller is responsible for keeping this off
    /// the recall critical path (an idle tick, sleep-pipeline step, or an
    /// `overlay_len()` threshold) — never a per-recall call.
    fn fold(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            self.buf.fold();
        });
        Ok(())
    }

    /// Count of ids the overlay currently carries a verdict for (touched
    /// since the last fold) — the signal a caller polls to decide when
    /// `fold()` is worth triggering.
    fn overlay_len(&self) -> usize {
        self.buf.overlay_len()
    }

    /// Total resident `text_arena` byte length -- `text_arena` holds ONLY
    /// `aaak_index` bytes (the zero-new-plaintext lock), so this is
    /// exactly the sum of every resident aaak string's UTF-8 byte length.
    /// No production code path reads this; it exists so the lock is
    /// mechanically verifiable from Python, since no surface-reading
    /// accessor exists to check against instead.
    fn resident_text_arena_len(&self) -> usize {
        self.buf.current().text_arena.len()
    }

    /// Total resident byte footprint of the per-slot token-frequency maps
    /// -- fixed-width per entry regardless of the original token's length,
    /// the mechanical proxy for the zero-new-plaintext lock extended to
    /// TOKEN storage.
    fn resident_token_footprint_bytes(&self) -> usize {
        self.buf.current().resident_token_footprint_bytes()
    }

    /// Runs `fused_score` over the currently held buffer for one recall
    /// call. `pool_ids` is the caller's already-selected per-call candidate
    /// pool (pool-position order); the four index arrays are positions into
    /// that pool and cross as zero-copy numpy buffers — width narrowing
    /// (`i64` -> `u32`, bounds-checked) and widening (`f32` -> `f64`)
    /// happen here, never on the Python side, so a caller allocates no
    /// per-candidate container to cross this boundary. `structure_hv` is
    /// not accepted from Python: the structural-blend term (T7) has no
    /// production knob writer (`structural_weight` is always `0.0` on the
    /// live path), so a length-matched all-`None` array is synthesized
    /// here — a caller that sets `structural_weight` above `0.0` gets the
    /// term computed against an always-absent vector, never a silently
    /// wrong nonzero one, since the multiply-by-zero default keeps the
    /// term's contribution at zero regardless.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    #[pyo3(signature = (
        pool_ids, cosine, cosine_top_indices, spread_indices, rich_indices,
        lex_indices, t11_flags, t12_flags, verbatim_filter, cue, now, effective_w_degree,
        effective_w_cosine,
        excluded_edge_types, spread_provenance, w_spread_act, spread_act_decay,
        community_id_by_member, community_scores, max_community_score, mode_bias,
        cos_spread_min, structural_weight, cue_structure_hv, lex_lane_enabled,
        min_idf, lex_fusion_w, k, k_margin,
    ))]
    fn score(
        &self,
        py: Python<'_>,
        pool_ids: Vec<u128>,
        cosine: PyReadonlyArrayDyn<'_, f32>,
        cosine_top_indices: PyReadonlyArrayDyn<'_, i64>,
        spread_indices: PyReadonlyArrayDyn<'_, i64>,
        rich_indices: PyReadonlyArrayDyn<'_, i64>,
        lex_indices: PyReadonlyArrayDyn<'_, i64>,
        t11_flags: PyReadonlyArrayDyn<'_, bool>,
        t12_flags: PyReadonlyArrayDyn<'_, bool>,
        verbatim_filter: bool,
        cue: &str,
        now: i64,
        effective_w_degree: f64,
        effective_w_cosine: f64,
        excluded_edge_types: HashSet<String>,
        spread_provenance: HashMap<u128, (u128, u32, bool)>,
        w_spread_act: f64,
        spread_act_decay: f64,
        community_id_by_member: HashMap<u128, u128>,
        community_scores: HashMap<u128, f64>,
        max_community_score: f64,
        mode_bias: f64,
        cos_spread_min: f64,
        structural_weight: f64,
        cue_structure_hv: Option<Vec<u8>>,
        lex_lane_enabled: bool,
        min_idf: f64,
        lex_fusion_w: f64,
        k: usize,
        k_margin: usize,
    ) -> PyResult<(
        Vec<(
            u128, f64, f64, f64, i64, u8, String, Vec<String>,
            (f64, f64, f64, f64, f64, f64, f64),
        )>,
        (u32, u32, bool),
        f64,
    )> {
        let n_pool = pool_ids.len();
        let cue_owned = cue.to_string();

        let cosine_f64: Vec<f64> = cosine.as_array().iter().map(|&v| v as f64).collect();
        check_score_len("cosine", cosine_f64.len(), n_pool)?;

        let narrow_indices = |name: &'static str,
                               arr: &PyReadonlyArrayDyn<'_, i64>|
         -> PyResult<Vec<u32>> {
            let view = arr.as_array();
            let mut out = Vec::with_capacity(view.len());
            for &v in view.iter() {
                if v < 0 || v > i64::from(u32::MAX) {
                    return Err(RankIndexError::ScoreInputMismatch {
                        field: name,
                        got: usize::try_from(v).unwrap_or(usize::MAX),
                        expected: usize::try_from(u32::MAX).unwrap_or(usize::MAX),
                    }
                    .into());
                }
                out.push(v as u32);
            }
            Ok(out)
        };
        let cosine_top_indices = narrow_indices("cosine_top_indices", &cosine_top_indices)?;
        let spread_indices = narrow_indices("spread_indices", &spread_indices)?;
        let rich_indices = narrow_indices("rich_indices", &rich_indices)?;
        let lex_indices = narrow_indices("lex_indices", &lex_indices)?;
        let t11_flags: Vec<bool> = t11_flags.as_array().iter().copied().collect();
        let t12_flags: Vec<bool> = t12_flags.as_array().iter().copied().collect();

        let structure_hv: Vec<Option<Vec<u8>>> = vec![None; n_pool];
        let spread_provenance_map: HashMap<u128, SpreadEntry> = spread_provenance
            .into_iter()
            .map(|(id, (seed_id, hop, transfer_carrying))| {
                (id, SpreadEntry { seed_id, hop, transfer_carrying })
            })
            .collect();

        let inner = self.buf.current();
        let result = py.detach(|| {
            let params = FusedScoreParams {
                pool_ids: &pool_ids,
                cosine: &cosine_f64,
                structure_hv: &structure_hv,
                cosine_top_indices: &cosine_top_indices,
                spread_indices: &spread_indices,
                rich_indices: &rich_indices,
                lex_indices: &lex_indices,
                t11_flags: &t11_flags,
                t12_flags: &t12_flags,
                verbatim_filter,
                cue: &cue_owned,
                now,
                effective_w_degree,
                effective_w_cosine,
                excluded_edge_types: &excluded_edge_types,
                spread_provenance: &spread_provenance_map,
                w_spread_act,
                spread_act_decay,
                community_id_by_member: &community_id_by_member,
                community_scores: &community_scores,
                max_community_score,
                mode_bias,
                cos_spread_min,
                structural_weight,
                cue_structure_hv: cue_structure_hv.as_deref(),
                lex_lane_enabled,
                min_idf,
                lex_fusion_w,
                k,
                k_margin,
            };
            fused_score(&inner, &params)
        })?;

        let winners = result
            .winners
            .into_iter()
            .map(|w| {
                (
                    w.id,
                    w.partial_score,
                    w.pre_gain_base,
                    w.term_multiplier,
                    w.created_at,
                    w.salience_level,
                    w.tier,
                    w.tags,
                    (
                        w.cosine,
                        w.aaak,
                        w.deg_norm,
                        w.age,
                        w.spread_contrib,
                        w.community_contrib,
                        w.structural_score,
                    ),
                )
            })
            .collect();
        let coverage = (
            result.coverage.expected,
            result.coverage.resident,
            result.coverage.all_resident,
        );
        Ok((winners, coverage, result.flat_cosine_damp))
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RankIndex>()?;
    m.add_function(wrap_pyfunction!(trigram_t11_flags, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only fixture row — NOT the production ingest path (which never
    /// builds a per-record wrapper struct); this exists solely to make
    /// small hand-written `Inner::from_columns` calls readable in tests.
    struct TestRow {
        id: u128,
        vector: Vec<f32>,
        surface: &'static str,
        edges: Vec<(u128, f32, &'static str)>,
        tags: Vec<&'static str>,
        aaak_index: &'static str,
        salience_level: u8,
        pending: bool,
    }

    fn row(id: u128, vector: Vec<f32>, surface: &'static str) -> TestRow {
        TestRow {
            id,
            vector,
            surface,
            edges: Vec::new(),
            tags: Vec::new(),
            aaak_index: "",
            salience_level: 0,
            pending: false,
        }
    }

    fn build_inner(dim: usize, generation: u64, rows: Vec<TestRow>) -> Result<Inner, RankIndexError> {
        let n = rows.len();
        let mut ids = Vec::with_capacity(n);
        let mut vectors_flat = Vec::with_capacity(n * dim);
        let mut edge_map: HashMap<u128, Vec<(u128, f32, String)>> = HashMap::new();
        let mut surfaces = Vec::with_capacity(n);
        let mut aaak_index = Vec::with_capacity(n);
        let created_at = vec![0i64; n];
        let stability = vec![0.5f32; n];
        let tier = vec!["episodic".to_string(); n];
        let mut tags = Vec::with_capacity(n);
        let mut salience_level = Vec::with_capacity(n);
        let centrality = vec![0.0f32; n];
        let mut pending = Vec::with_capacity(n);

        for r in rows {
            ids.push(r.id);
            vectors_flat.extend_from_slice(&r.vector);
            surfaces.push(r.surface.to_string());
            aaak_index.push(r.aaak_index.to_string());
            tags.push(r.tags.iter().map(|t| t.to_string()).collect());
            salience_level.push(r.salience_level);
            pending.push(r.pending);
            if !r.edges.is_empty() {
                edge_map.insert(
                    r.id,
                    r.edges
                        .into_iter()
                        .map(|(nbr, w, t)| (nbr, w, t.to_string()))
                        .collect(),
                );
            }
        }

        Inner::from_columns(
            dim,
            generation,
            ids,
            vectors_flat,
            edge_map,
            surfaces,
            aaak_index,
            created_at,
            stability,
            tier,
            tags,
            salience_level,
            centrality,
            pending,
        )
    }

    #[test]
    fn tokenize_matches_python_reference() {
        assert_eq!(
            tokenize("_hippo_cascade_loop"),
            vec!["_hippo_cascade_loop", "hippo", "cascade", "loop"]
        );
        assert_eq!(tokenize("HTTPServer"), vec!["httpserver", "http", "server"]);
        assert_eq!(
            tokenize("fooBar2Baz"),
            vec!["foobar2baz", "foo", "bar2", "baz"]
        );
        assert_eq!(
            tokenize("IAI_MCP_FORESIGHT_OFF"),
            vec!["iai_mcp_foresight_off", "iai", "mcp", "foresight", "off"]
        );
        assert_eq!(tokenize("ab"), vec!["ab"]);
        assert_eq!(tokenize("12345"), vec!["12345"]);
        assert_eq!(tokenize("x_y_zzz"), vec!["x_y_zzz"]);
        assert_eq!(
            tokenize("the quick brown fox jumps"),
            vec!["the", "quick", "brown", "fox", "jumps"]
        );
        assert_eq!(
            tokenize("record_id and salience_level fields"),
            vec![
                "record_id", "and", "salience_level", "salience", "level", "fields"
            ]
        );
    }

    #[test]
    fn created_at_parses_adapter_isoformat_shapes_to_epoch_i64() {
        // Cross-checked against Python's
        // datetime.fromisoformat(...).timestamp() for 2026-08-25T12:34:56 UTC.
        const EXPECTED: i64 = 1_787_661_296;
        assert_eq!(parse_created_at("2026-08-25T12:34:56.123456+00:00"), EXPECTED);
        assert_eq!(parse_created_at("2026-08-25T12:34:56+00:00"), EXPECTED);
        assert_eq!(
            parse_created_at("2026-08-25T12:34:56-07:00"),
            EXPECTED + 7 * 3600
        );

        assert_eq!(parse_created_at(""), 0, "empty created_at defaults to epoch 0");
        let unparseable = parse_created_at("not-a-timestamp");
        assert_ne!(
            unparseable, 0,
            "a genuinely unparseable string must not be silently indistinguishable from empty"
        );
        assert_eq!(unparseable, i64::MIN);
    }

    #[test]
    fn bulk_ingest_reports_length_and_ids() {
        let inner = build_inner(
            2,
            7,
            vec![
                row(1, vec![1.0, 0.0], "alpha token"),
                row(2, vec![0.0, 1.0], "beta token"),
            ],
        )
        .expect("bulk ingest should succeed");
        assert_eq!(inner.len(), 2);
        assert_eq!(inner.generation(), 7);
        let mut ids = inner.ids();
        ids.sort();
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn bulk_ingest_rejects_dim_mismatch() {
        let err = build_inner(2, 1, vec![row(1, vec![1.0, 0.0, 0.0], "x")]).unwrap_err();
        assert!(matches!(err, RankIndexError::DimMismatch { .. }));
    }

    #[test]
    fn bulk_ingest_rejects_duplicate_id() {
        let err = build_inner(
            2,
            1,
            vec![row(1, vec![1.0, 0.0], "x"), row(1, vec![0.0, 1.0], "y")],
        )
        .unwrap_err();
        assert!(matches!(err, RankIndexError::DuplicateId(1)));
    }

    // structure_hv_slot_present_but_unfed: deleted -- the columnar layout
    // carries no structure_hv column (dropped, no columnar equivalent; see
    // Inner's doc comment).

    #[test]
    fn salience_level_is_stored_as_u8_rank() {
        let mut r = row(1, vec![1.0, 0.0], "x");
        r.salience_level = 2;
        let inner = build_inner(2, 1, vec![r]).expect("bulk ingest should succeed");
        let slot = *inner.id_to_slot.get(&1).unwrap() as usize;
        assert_eq!(inner.salience_level[slot], 2u8);
    }

    #[test]
    fn adjacency_groups_by_edge_type_with_correct_counts() {
        let mut r1 = row(1, vec![1.0, 0.0], "x");
        r1.edges = vec![
            (2, 0.5, "hebbian"),
            (3, 0.5, "hebbian"),
            (4, 0.5, "entity_shared"),
        ];
        let inner = build_inner(
            2,
            1,
            vec![
                r1,
                row(2, vec![0.0, 1.0], "y"),
                row(3, vec![1.0, 1.0], "z"),
                row(4, vec![1.0, 1.0], "w"),
            ],
        )
        .expect("bulk ingest should succeed");

        let slot = *inner.id_to_slot.get(&1).unwrap() as usize;
        let start = inner.adj_offsets[slot] as usize;
        let end = inner.adj_offsets[slot + 1] as usize;
        let mut by_type: HashMap<String, u32> = HashMap::new();
        for i in start..end {
            let et = inner.edge_type_pool[inner.adj_type_ids[i] as usize].clone();
            *by_type.entry(et).or_insert(0) += 1;
        }
        assert_eq!(by_type.get("hebbian").copied(), Some(2));
        assert_eq!(by_type.get("entity_shared").copied(), Some(1));
        assert_eq!(by_type.len(), 2);
    }

    #[test]
    fn adjacency_and_postings_are_resident_after_bulk_ingest() {
        let mut r1 = row(1, vec![1.0, 0.0], "record_id salience_level");
        r1.edges = vec![(2, 0.5, "consolidated_from")];
        let inner = build_inner(
            2,
            1,
            vec![r1, row(2, vec![0.0, 1.0], "salience_level fields")],
        )
        .expect("bulk ingest should succeed");

        let slot1 = *inner.id_to_slot.get(&1).unwrap() as usize;
        let start = inner.adj_offsets[slot1] as usize;
        let end = inner.adj_offsets[slot1 + 1] as usize;
        assert_eq!(end - start, 1);
        assert_eq!(inner.adj_neighbors[start], 2);

        let token_id = *inner.token_lookup.get(&token_feature_id("salience_level")).unwrap();
        let pstart = inner.post_offsets[token_id as usize] as usize;
        let pend = inner.post_offsets[token_id as usize + 1] as usize;
        let slot2 = *inner.id_to_slot.get(&2).unwrap() as usize;
        let doc_slots: Vec<u32> = inner.post_doc_slots[pstart..pend].to_vec();
        assert!(doc_slots.contains(&(slot1 as u32)));
        assert!(doc_slots.contains(&(slot2 as u32)));
        for i in pstart..pend {
            assert_eq!(inner.post_tfs[i], 1);
        }
    }

    fn upsert_op(id: u128, surface: &str) -> PendingOp {
        PendingOp::Upsert {
            id,
            vector: vec![1.0, 0.0],
            edges: Vec::new(),
            surface: surface.to_string(),
            aaak_index: String::new(),
            created_at: 0,
            stability: 0.5,
            tier: String::new(),
            tags: Vec::new(),
            salience_level: 0,
            centrality: 0.0,
            pending: false,
        }
    }

    fn upsert_op_pending(id: u128, surface: &str) -> PendingOp {
        PendingOp::Upsert {
            id,
            vector: vec![0.0, 0.0],
            edges: Vec::new(),
            surface: surface.to_string(),
            aaak_index: String::new(),
            created_at: 0,
            stability: 0.5,
            tier: String::new(),
            tags: Vec::new(),
            salience_level: 0,
            centrality: 0.0,
            pending: true,
        }
    }

    fn two_record_buffer() -> DoubleBuffer {
        let inner = build_inner(
            2,
            1,
            vec![row(1, vec![1.0, 0.0], "alpha"), row(2, vec![0.0, 1.0], "beta")],
        )
        .expect("bulk ingest should succeed");
        DoubleBuffer::new(inner)
    }

    #[test]
    fn snapshot_matching_generation_is_pure_read_no_rebuild() {
        let buf = two_record_buffer();
        buf.feed(upsert_op(3, "gamma")).unwrap();
        let snap = buf.snapshot(1).expect("matching generation must not error");
        assert_eq!(
            snap.len(),
            2,
            "a matching-generation snapshot must not apply pending feed"
        );
    }

    #[test]
    fn snapshot_stale_generation_rebuilds_and_swaps_then_serves() {
        let buf = two_record_buffer();
        buf.feed(upsert_op(3, "gamma")).unwrap();
        let snap = buf.snapshot(2).expect("stale generation must rebuild");
        assert_eq!(snap.len(), 3);
        assert_eq!(snap.generation(), 2);
        assert!(snap.ids().contains(&3));

        // A second call at the same (now current) generation is a pure
        // read: no further pending was queued, so the length is stable.
        let snap2 = buf.snapshot(2).expect("re-read at current generation");
        assert_eq!(snap2.len(), 3);
    }

    #[test]
    fn feed_does_not_tear_concurrent_reader() {
        let buf = two_record_buffer();
        let held = buf.snapshot(1).expect("initial read");
        buf.feed(PendingOp::Delete { id: 1 }).unwrap();
        let _rebuilt = buf.snapshot(2).expect("stale generation must rebuild");
        assert_eq!(
            held.len(),
            2,
            "an Arc held before the rebuild must observe the old buffer, unmutated"
        );
        assert!(held.ids().contains(&1));
    }

    #[test]
    fn generation_reported_is_exactly_last_value_no_second_counter() {
        let buf = two_record_buffer();
        assert_eq!(buf.current().generation(), 1);
        buf.feed(upsert_op(3, "gamma")).unwrap();
        buf.feed(upsert_op(4, "delta")).unwrap();
        assert_eq!(
            buf.current().generation(),
            1,
            "feed alone must never advance the generation"
        );
        let snap = buf.snapshot(9).unwrap();
        assert_eq!(snap.generation(), 9);
        let snap_again = buf.snapshot(9).unwrap();
        assert_eq!(snap_again.generation(), 9);
    }

    #[test]
    fn snapshot_rejects_generation_regression() {
        let buf = two_record_buffer();
        let _ = buf.snapshot(5).expect("advance to generation 5");
        let err = buf.snapshot(1).unwrap_err();
        assert!(matches!(err, RankIndexError::GenerationRegression { .. }));
    }

    #[test]
    fn delete_then_rebuild_removes_id_and_reindexes_slot() {
        let buf = two_record_buffer();
        buf.feed(PendingOp::Delete { id: 1 }).unwrap();
        let snap = buf.snapshot(2).expect("stale generation must rebuild");
        assert_eq!(snap.len(), 1);
        assert!(!snap.ids().contains(&1));
        assert!(snap.ids().contains(&2));
    }

    #[test]
    fn bulk_ingest_threads_pending_column_from_input() {
        let mut pending_row = row(3, vec![0.0, 0.0], "quirkalpha pending doc");
        pending_row.pending = true;
        let inner = build_inner(
            2,
            1,
            vec![
                row(1, vec![1.0, 0.0], "alpha"),
                row(2, vec![0.0, 1.0], "beta"),
                pending_row,
            ],
        )
        .expect("bulk ingest should succeed");
        let slot1 = *inner.id_to_slot.get(&1).unwrap() as usize;
        let slot2 = *inner.id_to_slot.get(&2).unwrap() as usize;
        let slot3 = *inner.id_to_slot.get(&3).unwrap() as usize;
        assert!(!inner.pending[slot1]);
        assert!(!inner.pending[slot2]);
        assert!(inner.pending[slot3]);
    }

    #[test]
    fn incremental_upsert_sets_pending_flag_on_new_id() {
        let buf = two_record_buffer();
        buf.feed(upsert_op_pending(3, "quirkalpha pending doc")).unwrap();
        let snap = buf.snapshot(2).expect("stale generation must apply incrementally");
        let slot = *snap.id_to_slot.get(&3).unwrap() as usize;
        assert!(snap.pending[slot], "a fed pending upsert must land with pending=true");
        let start = slot * snap.dim;
        assert_eq!(
            &snap.vectors[start..start + snap.dim],
            &[0.0, 0.0],
            "the fed pending row's vector round-trips exactly as given"
        );
    }

    #[test]
    fn incremental_upsert_clears_pending_flag_on_existing_id_reupsert() {
        // The real production transition: a record starts pending (no
        // embedding yet), then its embedding lands and the write-time hook
        // re-upserts the SAME id with pending=false.
        let initial = build_inner(2, 1, vec![{
            let mut r = row(1, vec![0.0, 0.0], "quirkalpha pending doc");
            r.pending = true;
            r
        }])
        .expect("initial bulk ingest must succeed");
        let buf = DoubleBuffer::new(initial);
        buf.feed(upsert_op(1, "quirkalpha pending doc")).unwrap();
        let snap = buf.snapshot(2).expect("stale generation must apply incrementally");
        let slot = *snap.id_to_slot.get(&1).unwrap() as usize;
        assert!(
            !snap.pending[slot],
            "a re-upsert with pending omitted (defaults false) must clear a prior pending flag"
        );
    }

    #[test]
    fn pending_row_is_included_in_postings_csr() {
        let mut pending_row = row(3, vec![0.0, 0.0], "quirkalpha pending doc");
        pending_row.pending = true;
        let inner = build_inner(
            2,
            1,
            vec![row(1, vec![1.0, 0.0], "alpha"), pending_row],
        )
        .expect("bulk ingest should succeed");
        let resolved = inner.resolved_postings_for_token("quirkalpha");
        let ids: Vec<u128> = resolved.iter().map(|(id, _tf)| *id).collect();
        assert!(
            ids.contains(&3),
            "a pending row's surface tokens must still populate the postings CSR -- \
             lexical/BM25 membership is unaffected by the pending flag"
        );
    }

    #[test]
    fn delete_scalar_preserves_pending_flag_through_swap_remove() {
        let mut pending_row = row(1, vec![1.0, 0.0], "quirkalpha pending doc");
        pending_row.pending = true;
        let mut inner = build_inner(
            2,
            1,
            vec![pending_row, row(2, vec![0.0, 1.0], "beta")],
        )
        .expect("bulk ingest should succeed");
        // Deleting id 2 forces id 1's slot to move via swap-remove (id 1 is
        // not the last slot before the delete).
        inner.delete_scalar(2).expect("delete must succeed");
        let slot1 = *inner.id_to_slot.get(&1).unwrap() as usize;
        assert!(
            inner.pending[slot1],
            "a swap-remove must carry the pending flag along with the moved id"
        );
    }

    fn upsert_op_full(
        id: u128,
        surface: &str,
        edges: Vec<(u128, f32, &str)>,
        tags: Vec<&str>,
    ) -> PendingOp {
        PendingOp::Upsert {
            id,
            vector: vec![1.0, 0.0],
            edges: edges.into_iter().map(|(n, w, t)| (n, w, t.to_string())).collect(),
            surface: surface.to_string(),
            aaak_index: String::new(),
            created_at: 0,
            stability: 0.5,
            tier: String::new(),
            tags: tags.into_iter().map(str::to_string).collect(),
            salience_level: 0,
            centrality: 0.0,
            pending: false,
        }
    }

    fn upsert_op_with_aaak(id: u128, surface: &str, aaak_index: &str) -> PendingOp {
        let mut op = upsert_op_full(id, surface, vec![], vec![]);
        if let PendingOp::Upsert { aaak_index: ref mut a, .. } = op {
            *a = aaak_index.to_string();
        }
        op
    }

    fn sorted_edges(mut v: Vec<(u128, f32, String)>) -> Vec<(u128, String)> {
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v.into_iter().map(|(n, _w, t)| (n, t)).collect()
    }

    fn sorted_tags(mut v: Vec<String>) -> Vec<String> {
        v.sort();
        v
    }

    fn sorted_postings(mut v: Vec<(u128, u32)>) -> Vec<(u128, u32)> {
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }

    // -----------------------------------------------------------------
    // Incremental apply -- equivalence oracle + the critical
    // current-not-stale correctness proof.
    // -----------------------------------------------------------------

    #[test]
    fn incremental_apply_matches_full_rebuild_for_touched_and_untouched_ids() {
        let initial = build_inner(
            2,
            1,
            vec![
                {
                    let mut r = row(1, vec![1.0, 0.0], "alpha record_one");
                    r.edges = vec![(2, 0.5, "hebbian")];
                    r.tags = vec!["tag1"];
                    r
                },
                {
                    let mut r = row(2, vec![0.0, 1.0], "beta record_two");
                    r.edges = vec![(3, 0.5, "hebbian")];
                    r.tags = vec!["tag2"];
                    r
                },
                {
                    let mut r = row(3, vec![1.0, 1.0], "gamma record_three");
                    r.edges = vec![(1, 0.5, "hebbian")];
                    r.tags = vec!["tag3"];
                    r
                },
                {
                    let mut r = row(4, vec![0.5, 0.5], "delta record_four untouched");
                    r.edges = vec![(1, 0.2, "entity_shared")];
                    r.tags = vec!["tag4"];
                    r
                },
            ],
        )
        .expect("initial bulk ingest must succeed");
        let buf = DoubleBuffer::new(initial);

        // (a) upsert of an EXISTING id with a changed token count.
        buf.feed(upsert_op_full(
            2,
            "beta record_two now has many more distinct words present",
            vec![(5, 0.9, "entity_shared")],
            vec!["tag2b", "tag2c"],
        ))
        .unwrap();
        // (b) upsert of a brand-new id.
        buf.feed(upsert_op_full(6, "epsilon record_six new_token", vec![(1, 0.3, "hebbian")], vec!["tag6"]))
            .unwrap();
        // (c) delete -- exercises the swap-remove slot shift.
        buf.feed(PendingOp::Delete { id: 3 }).unwrap();
        // (d) re-add of the just-deleted id -- exercises tombstone clearing.
        buf.feed(upsert_op_full(
            3,
            "gamma record_three reborn distinct_content",
            vec![(1, 0.7, "hebbian")],
            vec!["tag3new"],
        ))
        .unwrap();

        let next = buf.snapshot(2).expect("stale generation must apply incrementally");

        // Structural proof this stayed incremental: the committed CSR's
        // own id ordering is untouched by an apply (only `fold` changes it).
        assert_eq!(
            next.committed_ids,
            vec![1u128, 2, 3, 4],
            "an incremental apply must never advance committed_ids -- that is fold's job alone"
        );

        let reference = build_inner(
            2,
            2,
            vec![
                {
                    let mut r = row(1, vec![1.0, 0.0], "alpha record_one");
                    r.edges = vec![(2, 0.5, "hebbian")];
                    r.tags = vec!["tag1"];
                    r
                },
                {
                    let mut r = row(2, vec![0.0, 1.0], "beta record_two now has many more distinct words present");
                    r.edges = vec![(5, 0.9, "entity_shared")];
                    r.tags = vec!["tag2b", "tag2c"];
                    r
                },
                {
                    let mut r = row(3, vec![1.0, 1.0], "gamma record_three reborn distinct_content");
                    r.edges = vec![(1, 0.7, "hebbian")];
                    r.tags = vec!["tag3new"];
                    r
                },
                {
                    let mut r = row(4, vec![0.5, 0.5], "delta record_four untouched");
                    r.edges = vec![(1, 0.2, "entity_shared")];
                    r.tags = vec!["tag4"];
                    r
                },
                {
                    let mut r = row(6, vec![1.0, 0.0], "epsilon record_six new_token");
                    r.edges = vec![(1, 0.3, "hebbian")];
                    r.tags = vec!["tag6"];
                    r
                },
            ],
        )
        .expect("reference bulk ingest must succeed");

        let mut next_ids = next.ids();
        next_ids.sort();
        let mut ref_ids = reference.ids();
        ref_ids.sort();
        assert_eq!(next_ids, ref_ids, "final id membership must match exactly");
        assert_eq!(next.current_n_docs(), reference.current_n_docs());

        for &id in &ref_ids {
            assert_eq!(
                sorted_edges(next.resolved_adjacency(id)),
                sorted_edges(reference.resolved_adjacency(id)),
                "adjacency mismatch for id {id}"
            );
            assert_eq!(
                sorted_tags(next.resolved_tags(id)),
                sorted_tags(reference.resolved_tags(id)),
                "tags mismatch for id {id}"
            );
            assert_eq!(
                next.resolved_degree(id),
                reference.resolved_degree(id),
                "degree mismatch for id {id}"
            );
            assert_eq!(
                next.resolved_doc_len(id),
                reference.resolved_doc_len(id),
                "doc_len mismatch for id {id}"
            );
        }

        let avg_delta = (next.current_avg_len() - reference.current_avg_len()).abs();
        assert!(
            avg_delta < 1e-9,
            "avg_len mismatch: incremental={} reference={}",
            next.current_avg_len(),
            reference.current_avg_len()
        );

        for tok in ["record_two", "distinct_content", "record_one", "tag2c_is_not_a_token"] {
            assert_eq!(
                sorted_postings(next.resolved_postings_for_token(tok)),
                sorted_postings(reference.resolved_postings_for_token(tok)),
                "postings mismatch for token {tok:?}"
            );
        }
    }

    #[test]
    fn incremental_apply_returns_current_not_stale_adjacency_after_write() {
        let initial = build_inner(
            2,
            1,
            vec![
                {
                    let mut r = row(1, vec![1.0, 0.0], "alpha");
                    r.edges = vec![(2, 0.5, "hebbian")];
                    r
                },
                row(2, vec![0.0, 1.0], "beta"),
                row(3, vec![1.0, 1.0], "gamma"),
            ],
        )
        .expect("initial bulk ingest must succeed");
        let buf = DoubleBuffer::new(initial);

        // A real post-write delta: id 1's edge list is REPLACED, not
        // merely re-asserted -- a fixture that fed back the same edges
        // would not distinguish a correct overlay-consulting read from a
        // stale committed-CSR read.
        buf.feed(upsert_op_full(1, "alpha", vec![(3, 0.9, "entity_shared")], vec![]))
            .unwrap();
        let snap = buf.snapshot(2).expect("stale generation must apply incrementally");

        let edges = snap.resolved_adjacency(1);
        assert_eq!(
            edges,
            vec![(3u128, 0.9f32, "entity_shared".to_string())],
            "post-write read must return the NEW edge, never the committed CSR's stale (2, hebbian) edge"
        );
        assert_eq!(snap.resolved_degree(1), 1);

        let by_type_dict = snap.resolved_adjacency(1);
        assert!(
            !by_type_dict.iter().any(|(n, _, _)| *n == 2),
            "the old edge to id 2 must not leak through"
        );
    }

    #[test]
    fn incremental_apply_resolves_brand_new_token_never_seen_by_committed_postings() {
        let initial = build_inner(2, 1, vec![row(1, vec![1.0, 0.0], "alpha")])
            .expect("initial bulk ingest must succeed");
        let buf = DoubleBuffer::new(initial);

        assert!(
            !buf.snapshot(1).unwrap().token_lookup.contains_key(&token_feature_id("zzzznew")),
            "sanity: this token must be absent from the committed pool before the write"
        );

        buf.feed(upsert_op_full(1, "alpha zzzznew", vec![], vec![])).unwrap();
        let snap = buf.snapshot(2).expect("stale generation must apply incrementally");

        let hits = snap.resolved_postings_for_token("zzzznew");
        assert_eq!(
            hits,
            vec![(1u128, 1u32)],
            "a token that only exists via an overlay-touched doc's post-write surface must still resolve"
        );
    }

    // -----------------------------------------------------------------
    // Arena bounding (byte-identical skip, independent surface/aaak
    // comparison, u32 overflow guard, fold reclamation).
    // -----------------------------------------------------------------

    #[test]
    fn arena_skips_append_on_byte_identical_surface_and_aaak_update() {
        let initial = build_inner(2, 1, vec![row(1, vec![1.0, 0.0], "stable surface text")])
            .expect("initial bulk ingest must succeed");
        let buf = DoubleBuffer::new(initial);
        let arena_len_before = buf.snapshot(1).unwrap().text_arena.len();

        // Scalar-only update: identical surface AND aaak, only salience
        // (a scalar field, not exercised here) would differ in production.
        buf.feed(upsert_op_full(1, "stable surface text", vec![], vec![])).unwrap();
        let snap = buf.snapshot(2).expect("stale generation must apply incrementally");
        assert_eq!(
            snap.text_arena.len(),
            arena_len_before,
            "a byte-identical surface+aaak update must not append to the arena at all"
        );
    }

    #[test]
    fn arena_appends_when_aaak_changes_but_surface_is_identical() {
        let initial = build_inner(2, 1, vec![row(1, vec![1.0, 0.0], "stable surface")])
            .expect("initial bulk ingest must succeed");
        let buf = DoubleBuffer::new(initial);
        let arena_len_before = buf.snapshot(1).unwrap().text_arena.len();

        let mut op = upsert_op_full(1, "stable surface", vec![], vec![]);
        if let PendingOp::Upsert { ref mut aaak_index, .. } = op {
            *aaak_index = "brand new aaak text".to_string();
        }
        buf.feed(op).unwrap();
        let snap = buf.snapshot(2).expect("stale generation must apply incrementally");
        assert!(
            snap.text_arena.len() > arena_len_before,
            "surface and aaak are compared and appended INDEPENDENTLY -- a changed aaak with an \
             identical surface must still grow the arena by the new aaak's bytes"
        );
    }

    #[test]
    fn arena_capacity_guard_refuses_past_u32_limit() {
        assert!(check_arena_capacity(0, 10).is_ok());
        assert!(check_arena_capacity(u32::MAX as usize - 3, 3).is_ok());
        let err = check_arena_capacity(u32::MAX as usize - 3, 10).unwrap_err();
        assert!(matches!(err, RankIndexError::ArenaOverflow { .. }));
    }

    #[test]
    fn arena_fold_reclaims_orphaned_bytes_after_touch() {
        // Surface bytes never enter the arena at all (the zero-new-
        // plaintext lock) -- aaak_index is the only remaining arena-
        // resident, orphan-on-touch class, so it is what this proof must
        // exercise now.
        let mut initial_rows = vec![row(1, vec![1.0, 0.0], "surface never touches the arena")];
        initial_rows[0].aaak_index = "a very long original aaak_index indeed";
        let initial = build_inner(2, 1, initial_rows).expect("initial bulk ingest must succeed");
        let buf = DoubleBuffer::new(initial);

        buf.feed(upsert_op_with_aaak(1, "surface never touches the arena", "short")).unwrap();
        let touched = buf.snapshot(2).expect("stale generation must apply incrementally");
        let arena_len_with_orphan = touched.text_arena.len();

        let folded = buf.fold();
        assert!(
            folded.text_arena.len() < arena_len_with_orphan,
            "fold must reclaim the orphaned original-aaak_index bytes left behind by the touch"
        );
        assert_eq!(folded.overlay.touched_len(), 0, "fold must reset the overlay");
        assert_eq!(folded.committed_ids, folded.ids, "fold must re-anchor committed_ids to the live id set");
    }

    // -----------------------------------------------------------------
    // Content-hash-gated token_freqs skip -- the resident-surface removal's
    // replacement for the old arena byte-compare (`upsert_scalar`, private,
    // called directly so the test can inject a caller-supplied token map
    // that DISAGREES with the surface string -- a wrong map landing on the
    // slot proves the skip did not fire; the skip firing is exactly what
    // must be observable here.)
    // -----------------------------------------------------------------

    #[test]
    fn same_surface_hash_skips_token_freqs_recompute() {
        let mut inner = build_inner(2, 1, vec![row(1, vec![1.0, 0.0], "original surface text")])
            .expect("initial bulk ingest must succeed");
        let slot = *inner.id_to_slot.get(&1).unwrap() as usize;
        assert_eq!(inner.token_freqs[slot], token_freq_map("original surface text"));

        // Same surface text (same content hash) but a deliberately WRONG
        // token map: if the hash-gated skip is broken and always
        // overwrites, this wrong map lands in token_freqs; if the skip
        // correctly fires, the stored map is untouched.
        let mut wrong_freqs: HashMap<u32, u32> = HashMap::new();
        wrong_freqs.insert(token_feature_id("bogus_token_that_must_never_appear"), 999);
        let mut tier_lookup: HashMap<String, u16> = HashMap::new();
        inner
            .upsert_scalar(
                1, &[1.0, 0.0], "original surface text", &wrong_freqs, "",
                0, 0.5, "", 0, 0.0, false, &mut tier_lookup,
            )
            .expect("upsert must succeed");

        assert_eq!(
            inner.token_freqs[slot],
            token_freq_map("original surface text"),
            "byte-identical surface must skip the token-representation recompute -- a wrong \
             caller-supplied map must never land on a slot whose content hash did not change"
        );
    }

    #[test]
    fn changed_surface_hash_applies_new_token_freqs() {
        let mut inner = build_inner(2, 1, vec![row(1, vec![1.0, 0.0], "original surface text")])
            .expect("initial bulk ingest must succeed");
        let slot = *inner.id_to_slot.get(&1).unwrap() as usize;

        let new_freqs = token_freq_map("a completely different surface");
        let mut tier_lookup: HashMap<String, u16> = HashMap::new();
        inner
            .upsert_scalar(
                1, &[1.0, 0.0], "a completely different surface", &new_freqs, "",
                0, 0.5, "", 0, 0.0, false, &mut tier_lookup,
            )
            .expect("upsert must succeed");

        assert_eq!(
            inner.token_freqs[slot], new_freqs,
            "a genuinely changed surface must apply the new token representation, not skip it"
        );
    }

    #[test]
    fn content_hash128_does_not_collide_on_two_engineered_distinct_surfaces() {
        let a = "the quick brown fox jumps over the lazy dog exactly once";
        let b = "the quick brown fox jumps over the lazy dog exactly twice";
        assert_ne!(content_hash128(a), content_hash128(b));
    }

    #[test]
    fn weak_truncated_hash_collides_but_the_production_hash_does_not_on_the_same_pair() {
        // Proves the collision-detection methodology is sound, not lucky: a
        // hash width this narrow WILL collide on some engineered pair
        // within a small search, so "no collision observed" at the
        // production >=128-bit width is a real property of the width.
        let weak = |s: &str| (content_hash128(s) & 0xFF) as u8;
        let mut seen: HashMap<u8, String> = HashMap::new();
        let mut pair: Option<(String, String)> = None;
        for i in 0..4000u32 {
            let s = format!("candidate surface number {i}");
            let h = weak(&s);
            if let Some(prior) = seen.get(&h) {
                pair = Some((prior.clone(), s));
                break;
            }
            seen.insert(h, s);
        }
        let (a, b) = pair.expect(
            "an 8-bit truncated hash must produce a collision within 4000 distinct short \
             strings (birthday bound ~2^4) -- if this never fires, the control below proves \
             nothing",
        );
        assert_eq!(weak(&a), weak(&b), "the engineered pair must actually collide under the weak hash");
        assert_ne!(
            content_hash128(&a), content_hash128(&b),
            "the SAME pair that collides under the weak/truncated hash must NOT collide under \
             the production >=128-bit hash -- proves the width, not luck, is what prevents the \
             dedup staleness bug"
        );
    }

    #[test]
    fn token_feature_id_does_not_collide_on_two_engineered_distinct_tokens() {
        assert_ne!(token_feature_id("marimba"), token_feature_id("xylophone"));
    }

    #[test]
    fn narrow_truncated_token_feature_collides_but_the_shipped_width_does_not_on_the_same_pair() {
        // Mirrors `weak_truncated_hash_collides_but_the_production_hash_does_not_
        // on_the_same_pair`'s methodology for the token feature-id: an 8-bit
        // bucket WILL collide on some engineered pair within a small search
        // (proving the search finds real collisions), while the shipped
        // 32-bit feature id does not collide on that same pair -- proves the
        // shipped width keeps the collision rate at the standard level for
        // feature hashing, not zero by luck.
        let narrow = |s: &str| (token_feature_id(s) & 0xFF) as u8;
        let mut seen: HashMap<u8, String> = HashMap::new();
        let mut pair: Option<(String, String)> = None;
        for i in 0..4000u32 {
            let s = format!("token_candidate_{i}");
            let h = narrow(&s);
            if let Some(prior) = seen.get(&h) {
                pair = Some((prior.clone(), s));
                break;
            }
            seen.insert(h, s);
        }
        let (a, b) = pair.expect(
            "an 8-bit truncated feature id must produce a collision within 4000 distinct \
             short tokens (birthday bound ~2^4) -- if this never fires, the control below \
             proves nothing",
        );
        assert_eq!(narrow(&a), narrow(&b), "the engineered pair must actually collide under the narrow width");
        assert_ne!(
            token_feature_id(&a), token_feature_id(&b),
            "the SAME pair that collides under the 8-bit width must NOT collide under the \
             shipped 32-bit feature id -- proves the width, not luck, keeps the shipped \
             collision rate at the standard level"
        );
    }

    #[test]
    fn trigram_features_matches_python_windowing_semantics() {
        // "abcd" -> {"abc", "bcd"}, packed and sorted -- checked against
        // pack_trigram directly rather than a string re-derivation, since
        // pack_trigram IS the encoding under test.
        let feats = trigram_features("abcd");
        let mut expected = vec![
            pack_trigram('a' as u32, 'b' as u32, 'c' as u32),
            pack_trigram('b' as u32, 'c' as u32, 'd' as u32),
        ];
        expected.sort_unstable();
        assert_eq!(feats, expected);
    }

    #[test]
    fn trigram_features_empty_below_three_chars() {
        assert!(trigram_features("").is_empty());
        assert!(trigram_features("a").is_empty());
        assert!(trigram_features("ab").is_empty());
    }

    #[test]
    fn pack_trigram_is_injective_over_the_full_unicode_scalar_range() {
        // Unicode scalar values are < 0x110000 (< 2^21), so three 21-bit
        // slots never overlap -- proven here at the boundary rather than
        // assumed, since an off-by-one shift width would silently merge
        // adjacent code points.
        let max_scalar = 0x10FFFFu32;
        assert!(max_scalar < (1u32 << 21));
        let a = pack_trigram(max_scalar, 0, 0);
        let b = pack_trigram(0, max_scalar, 0);
        let c = pack_trigram(0, 0, max_scalar);
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    #[test]
    fn trigram_jaccard_exact_matches_reference_set_arithmetic() {
        // "night" vs "nacht": {nig,igh,ght} vs {nac,ach,cht} -> no overlap.
        let a = trigram_features("night");
        let b = trigram_features("nacht");
        assert_eq!(trigram_jaccard_exact(&a, &b), 0.0);

        // Identical strings -> Jaccard 1.0.
        let c = trigram_features("the quick brown fox");
        assert_eq!(trigram_jaccard_exact(&c, &c), 1.0);

        // Partial overlap, checked against hand-counted set arithmetic.
        let d = trigram_features("abcdef"); // abc,bcd,cde,def
        let e = trigram_features("abcxyz"); // abc,bcx,cxy,xyz
        // intersection = {abc} = 1, union = 4 + 4 - 1 = 7
        assert!((trigram_jaccard_exact(&d, &e) - (1.0 / 7.0)).abs() < 1e-12);
    }

    #[test]
    fn trigram_jaccard_exact_zero_when_either_side_empty() {
        let a = trigram_features("ab"); // < 3 chars -> empty
        let b = trigram_features("abcdef");
        assert_eq!(trigram_jaccard_exact(&a, &b), 0.0);
        assert_eq!(trigram_jaccard_exact(&a, &a), 0.0);
    }

    #[test]
    fn delete_then_upsert_new_id_keeps_token_freqs_and_surface_hash_slot_aligned() {
        // token_freqs/surface_hash are swap-removed in lockstep with every
        // other scalar column on delete -- a missed pair here would desync
        // silently (wrong-slot postings), not panic.
        let mut inner = build_inner(
            2, 1,
            vec![
                row(1, vec![1.0, 0.0], "first record surface"),
                row(2, vec![0.0, 1.0], "second record surface"),
            ],
        )
        .expect("initial bulk ingest must succeed");
        inner.delete_scalar(1).expect("delete must succeed");
        let slot2 = *inner.id_to_slot.get(&2).unwrap() as usize;
        assert_eq!(
            inner.token_freqs[slot2],
            token_freq_map("second record surface"),
            "id 2's token_freqs must follow it to its swap-removed slot, not stay behind"
        );

        let mut tier_lookup: HashMap<String, u16> = HashMap::new();
        let new_freqs = token_freq_map("third record surface");
        inner
            .upsert_scalar(
                3, &[1.0, 0.0], "third record surface", &new_freqs, "",
                0, 0.5, "", 0, 0.0, false, &mut tier_lookup,
            )
            .expect("upsert must succeed");
        let slot3 = *inner.id_to_slot.get(&3).unwrap() as usize;
        assert_eq!(inner.token_freqs[slot3], new_freqs);
        assert_eq!(
            inner.token_freqs[slot2],
            token_freq_map("second record surface"),
            "a subsequent new-id append must not disturb an already-resident slot's token_freqs"
        );
    }

    // -----------------------------------------------------------------
    // Footprint gate: without this test, unbounded arena growth can
    // ship undetected.
    // -----------------------------------------------------------------

    #[test]
    fn footprint_repeated_identical_upsert_does_not_grow_arena() {
        // Surface bytes never enter the arena; aaak_index is the only
        // remaining arena-resident, append-on-touch class the byte-
        // identical skip still guards.
        let mut initial_rows = vec![row(1, vec![1.0, 0.0], "hot id stable content")];
        initial_rows[0].aaak_index = "stable aaak";
        let initial = build_inner(2, 1, initial_rows).expect("initial bulk ingest must succeed");
        let buf = DoubleBuffer::new(initial);
        let arena_len_before = buf.snapshot(1).unwrap().text_arena.len();

        let mut gen = 2u64;
        for _ in 0..200 {
            buf.feed(upsert_op_with_aaak(1, "hot id stable content", "stable aaak")).unwrap();
            let snap = buf.snapshot(gen).expect("stale generation must apply incrementally");
            assert_eq!(
                snap.text_arena.len(),
                arena_len_before,
                "repeated byte-identical upserts must never grow the arena, at any cycle count"
            );
            gen += 1;
        }
    }

    #[test]
    fn footprint_repeated_changing_upsert_bounded_by_live_content_after_fold() {
        let initial = build_inner(2, 1, vec![row(1, vec![1.0, 0.0], "hot id v0")])
            .expect("initial bulk ingest must succeed");
        let buf = DoubleBuffer::new(initial);

        // The real daemon pattern: reconsolidation rewrites the same hot
        // id's aaak_index on every touch -- each cycle's value differs from
        // the last, so the byte-identical skip never fires and the arena
        // grows touch-over-touch until a fold reclaims it. Surface changes
        // alongside it (the real touch shape) but never enters the arena.
        let mut gen = 2u64;
        let final_aaak = "hot id v199 final distinct aaak_index";
        for i in 0..200 {
            let surface = format!("hot id v{i} distinct content padding padding padding");
            let aaak = format!("hot id v{i} distinct aaak_index padding padding padding");
            buf.feed(upsert_op_with_aaak(1, &surface, &aaak)).unwrap();
            let _ = buf.snapshot(gen).expect("stale generation must apply incrementally");
            gen += 1;
        }
        buf.feed(upsert_op_with_aaak(1, "hot id final surface", final_aaak)).unwrap();
        let touched = buf.snapshot(gen).expect("stale generation must apply incrementally");
        let arena_len_after_touches = touched.text_arena.len();
        assert!(
            arena_len_after_touches > final_aaak.len() * 100,
            "sanity: 200 changing touches of one id must have actually grown the arena \
             proportionally to touch count before the fold -- a bound this loose would also \
             pass if the byte-identical skip silently broke and the arena grew far past what \
             201 touches of one id should produce"
        );

        let folded = buf.fold();
        // One resident id -- live content is exactly the final aaak_index's
        // bytes (surface never contributes arena bytes).
        let live_bytes: usize = final_aaak.len();
        assert!(
            (folded.text_arena.len() as usize) <= live_bytes * 4,
            "after fold, the arena must be bounded by a small multiple of LIVE content size \
             ({} bytes), not by the 200-touch history (was {} bytes before the fold)",
            live_bytes,
            arena_len_after_touches,
        );
        assert!(
            (folded.text_arena.len() as usize) < arena_len_after_touches,
            "fold must shrink the arena relative to the pre-fold touched state"
        );
    }
}
