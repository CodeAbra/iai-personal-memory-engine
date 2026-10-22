//! Symbol-lane orchestration: an on-disk manifest (own magic, own store
//! subdir) plus one immutable LSYMIDX1 segment per batch, published through
//! a `SegmentSetBuffer` instance -- the same generic double-buffer type the
//! whole-file lane uses, a second instance, never shared state. Triggers
//! `symbol_compaction` after publishing, the same way `index.rs` fires the
//! whole-file lane's compaction.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use thiserror::Error;

use crate::bm25::symbol_bm25_score_global;
use crate::segmentset::{SegmentHandle, SegmentSet, SegmentSetBuffer};
use crate::symbol_compaction;
use crate::symbol_scan::SymbolKind;
use crate::symbol_segment::{SymbolSegment, SymbolSegmentError};
use crate::termstats::TermStats;
use crate::tokenize::{feature_id, tokenize, tokenize_whole};

const MANIFEST_MAGIC_V1: &[u8; 8] = b"LSYMMNF1";
const MANIFEST_MAGIC_V2: &[u8; 8] = b"LSYMMNF2";
const MANIFEST_FILE: &str = "manifest.lsman";

/// Tier bases: chosen so the smallest exact-name tier score (Unknown, no
/// path boost) is orders of magnitude above the largest realistic b=0
/// sub-token BM25 sum -- a strict rank separation, never a multiplicative
/// nudge a high-tf sub-token hit could climb past.
const TIER1_DEF_BASE: f64 = 2_000_000.0;
const TIER1_UNKNOWN_BASE: f64 = 1_000_000.0;
/// Additive path/filename overlap boost, applied within and across tiers --
/// far smaller than the gap between tiers/kinds above, so it can only
/// break ties, never cross a tier or kind boundary.
const PATH_BOOST: f64 = 1.0;

/// True when any query token appears (as a `tokenize()` token) in `path`
/// -- covers both directory segments and the basename, since `tokenize`
/// runs over the whole path string.
fn path_matches_any_query_token(path: &str, query_fids: &[u32]) -> bool {
    tokenize(path)
        .iter()
        .any(|tok| query_fids.contains(&feature_id(tok)))
}

#[derive(Debug, Error)]
pub enum SymbolIndexError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("symbol manifest corrupt: {0}")]
    Corrupt(String),
    #[error(transparent)]
    Segment(#[from] SymbolSegmentError),
}

fn manifest_path(store_root: &Path) -> PathBuf {
    store_root.join(MANIFEST_FILE)
}

// Test-only per-thread call counter -- same thread-local single-load
// proof technique as `index::LOAD_MANIFEST_CALLS`.
#[cfg(test)]
thread_local! {
    pub(crate) static LOAD_MANIFEST_CALLS: std::cell::Cell<usize> = std::cell::Cell::new(0);
}

/// Loads the on-disk symbol manifest -- segment handles, generation, and
/// the resident term-stats table. Metadata I/O only; never opens a
/// `symbol-*.lsidx` file. Version-aware: an old-magic (v1) manifest parses
/// with an empty file-index and empty tombstones (safe degrade, no forced
/// rebuild); a new-magic (v2) manifest additionally carries both sections.
pub(crate) fn load_manifest(store_root: &Path) -> Result<SegmentSet, SymbolIndexError> {
    #[cfg(test)]
    LOAD_MANIFEST_CALLS.with(|c| c.set(c.get() + 1));
    let path = manifest_path(store_root);
    if !path.exists() {
        return Ok(SegmentSet::empty());
    }
    let raw = fs::read(&path)?;
    if raw.len() < 12 {
        return Err(SymbolIndexError::Corrupt("not a symbol manifest file".into()));
    }
    let magic: [u8; 8] = raw[..8].try_into().unwrap();
    let is_v2 = if magic == *MANIFEST_MAGIC_V2 {
        true
    } else if magic == *MANIFEST_MAGIC_V1 {
        false
    } else {
        return Err(SymbolIndexError::Corrupt("not a symbol manifest file".into()));
    };
    let stored_crc = u32::from_le_bytes(raw[8..12].try_into().unwrap());
    let payload = &raw[12..];
    if crc32fast::hash(payload) != stored_crc {
        return Err(SymbolIndexError::Corrupt("symbol manifest CRC mismatch".into()));
    }

    let need = |off: usize, len: usize| -> Result<(), SymbolIndexError> {
        if off + len > payload.len() {
            Err(SymbolIndexError::Corrupt("truncated symbol manifest".into()))
        } else {
            Ok(())
        }
    };

    let mut off = 0usize;
    need(off, 8)?;
    let generation = u64::from_le_bytes(payload[off..off + 8].try_into().unwrap());
    off += 8;

    need(off, 4)?;
    let segment_count = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()) as usize;
    off += 4;
    need(off, segment_count.saturating_mul(6))?;

    let mut segments = Vec::with_capacity(segment_count);
    for _ in 0..segment_count {
        need(off, 4)?;
        let id = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap());
        off += 4;
        need(off, 2)?;
        let name_len = u16::from_le_bytes(payload[off..off + 2].try_into().unwrap()) as usize;
        off += 2;
        need(off, name_len)?;
        let name = std::str::from_utf8(&payload[off..off + name_len])
            .map_err(|_| SymbolIndexError::Corrupt("non-utf8 segment name".into()))?
            .to_string();
        off += name_len;
        segments.push(SegmentHandle {
            id,
            file_name: name,
        });
    }

    need(off, 4)?;
    let df_count = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()) as usize;
    off += 4;
    need(off, df_count.saturating_mul(8))?;
    let mut df: HashMap<u32, u32> = HashMap::with_capacity(df_count);
    for _ in 0..df_count {
        need(off, 8)?;
        let fid = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap());
        let count = u32::from_le_bytes(payload[off + 4..off + 8].try_into().unwrap());
        df.insert(fid, count);
        off += 8;
    }

    need(off, 4)?;
    let n_docs = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap());
    off += 4;
    need(off, 8)?;
    let total_len = u64::from_le_bytes(payload[off..off + 8].try_into().unwrap());
    off += 8;

    let mut file_index: HashMap<String, (u32, u32, u64)> = HashMap::new();
    let mut tombstones: HashSet<(u32, u32)> = HashSet::new();

    if is_v2 {
        need(off, 4)?;
        let file_index_count =
            u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        // Lower bound only (path_len u16 + segment_id u32 + file_id u32 +
        // content_hash u64 = 18 fixed bytes; path bytes are variable).
        need(off, file_index_count.saturating_mul(18))?;
        for _ in 0..file_index_count {
            need(off, 2)?;
            let path_len = u16::from_le_bytes(payload[off..off + 2].try_into().unwrap()) as usize;
            off += 2;
            need(off, path_len)?;
            let path = std::str::from_utf8(&payload[off..off + path_len])
                .map_err(|_| SymbolIndexError::Corrupt("non-utf8 file-index path".into()))?
                .to_string();
            off += path_len;
            need(off, 16)?;
            let segment_id = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap());
            let file_id = u32::from_le_bytes(payload[off + 4..off + 8].try_into().unwrap());
            let content_hash = u64::from_le_bytes(payload[off + 8..off + 16].try_into().unwrap());
            off += 16;
            file_index.insert(path, (segment_id, file_id, content_hash));
        }

        need(off, 4)?;
        let tombstone_count =
            u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        need(off, tombstone_count.saturating_mul(8))?;
        for _ in 0..tombstone_count {
            need(off, 8)?;
            let segment_id = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap());
            let file_id = u32::from_le_bytes(payload[off + 4..off + 8].try_into().unwrap());
            tombstones.insert((segment_id, file_id));
            off += 8;
        }
    }

    if off != payload.len() {
        return Err(SymbolIndexError::Corrupt(
            "trailing bytes after symbol manifest payload".into(),
        ));
    }

    Ok(SegmentSet {
        segments,
        tombstones: Arc::new(tombstones),
        term_stats: Arc::new(TermStats {
            df,
            n_docs,
            total_len,
        }),
        file_index: Arc::new(file_index),
        generation,
    })
}

/// Returns `(file_index.len(), generation)` for the symbol manifest under
/// `store_root`, read off a SINGLE `load_manifest` binding -- same
/// torn-read hazard as the whole-file lane's own readback, avoided the
/// same way.
pub fn manifest_stats(store_root: &Path) -> Result<(u64, u64), SymbolIndexError> {
    let set = load_manifest(store_root)?;
    Ok((set.file_index.len() as u64, set.generation))
}

fn save_manifest(store_root: &Path, set: &SegmentSet) -> Result<(), SymbolIndexError> {
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(&set.generation.to_le_bytes());
    payload.extend_from_slice(&(set.segments.len() as u32).to_le_bytes());
    for h in &set.segments {
        payload.extend_from_slice(&h.id.to_le_bytes());
        let name_bytes = h.file_name.as_bytes();
        payload.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        payload.extend_from_slice(name_bytes);
    }

    let mut df_entries: Vec<(u32, u32)> =
        set.term_stats.df.iter().map(|(&k, &v)| (k, v)).collect();
    df_entries.sort_by_key(|&(k, _)| k);
    payload.extend_from_slice(&(df_entries.len() as u32).to_le_bytes());
    for (fid, count) in df_entries {
        payload.extend_from_slice(&fid.to_le_bytes());
        payload.extend_from_slice(&count.to_le_bytes());
    }
    payload.extend_from_slice(&set.term_stats.n_docs.to_le_bytes());
    payload.extend_from_slice(&set.term_stats.total_len.to_le_bytes());

    // v2 sections -- sorted so a manifest re-saved from an unchanged
    // logical state is byte-stable, same discipline as the df table above.
    let mut file_index_entries: Vec<(&String, &(u32, u32, u64))> = set.file_index.iter().collect();
    file_index_entries.sort_by(|a, b| a.0.cmp(b.0));
    payload.extend_from_slice(&(file_index_entries.len() as u32).to_le_bytes());
    for (path, &(segment_id, file_id, content_hash)) in file_index_entries {
        let path_bytes = path.as_bytes();
        payload.extend_from_slice(&(path_bytes.len() as u16).to_le_bytes());
        payload.extend_from_slice(path_bytes);
        payload.extend_from_slice(&segment_id.to_le_bytes());
        payload.extend_from_slice(&file_id.to_le_bytes());
        payload.extend_from_slice(&content_hash.to_le_bytes());
    }

    let mut tombstone_entries: Vec<&(u32, u32)> = set.tombstones.iter().collect();
    tombstone_entries.sort();
    payload.extend_from_slice(&(tombstone_entries.len() as u32).to_le_bytes());
    for &(segment_id, file_id) in tombstone_entries {
        payload.extend_from_slice(&segment_id.to_le_bytes());
        payload.extend_from_slice(&file_id.to_le_bytes());
    }

    let crc = crc32fast::hash(&payload);
    let path = manifest_path(store_root);
    let tmp = path.with_extension("tmp-symmanifest");
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(MANIFEST_MAGIC_V2)?;
        f.write_all(&crc.to_le_bytes())?;
        f.write_all(&payload)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, &path)?;
    Ok(())
}

fn segment_file_name(id: u32) -> String {
    format!("symbol-{id}.lsidx")
}

/// Indexes exactly the given paths as ONE new immutable symbol segment and
/// publishes it -- never opens, reads, or re-scans any file already
/// covered by a prior symbol segment.
pub fn append_batch(store_root: &Path, paths: &[PathBuf]) -> Result<(), SymbolIndexError> {
    if paths.is_empty() {
        return Ok(());
    }

    fs::create_dir_all(store_root)?;
    let current = load_manifest(store_root)?;

    let (segment, tombstones) = SymbolSegment::build_incremental(paths, &current.file_index)?;
    if segment.files.is_empty() && tombstones.is_empty() {
        // Every path in this batch is already current -- no new segment,
        // no tombstone, manifest stays byte-identical -- a genuine no-op.
        return Ok(());
    }

    let id = current.segments.iter().map(|h| h.id).max().unwrap_or(0) + 1;
    let file_name = segment_file_name(id);
    let file_hashes = segment.file_hashes();
    segment.save(&store_root.join(&file_name))?;

    let handle = SegmentHandle { id, file_name };
    let file_entries: Vec<(String, u32, u64)> = file_hashes
        .into_iter()
        .enumerate()
        .map(|(i, (path, hash))| (path, i as u32, hash))
        .collect();

    let buf = SegmentSetBuffer::from_set(current);
    buf.publish_incremental(handle, &segment.postings, &file_entries, &tombstones);

    let published = buf.snapshot();
    save_manifest(store_root, &published)?;

    if symbol_compaction::should_compact(published.segments.len()) {
        let prior_segments = published.segments.clone();
        // The append above is already durable in the manifest -- a
        // compaction failure (e.g. one unloadable source segment) must
        // degrade to "compaction skipped this round", never fail the
        // append that already succeeded. Otherwise a single bad segment
        // wedges every future append past the threshold permanently.
        match symbol_compaction::compact(store_root, &published) {
            Ok((merged_handle, source_ids, reconciled_stats, rebuilt_file_index)) => {
                buf.compact_reconciled(&source_ids, merged_handle, reconciled_stats, rebuilt_file_index);
                let final_set = buf.snapshot();
                save_manifest(store_root, &final_set)?;
                symbol_compaction::delete_sources(store_root, &source_ids, &prior_segments);
            }
            Err(err) => {
                eprintln!(
                    "iai_mcp_tools_core: symbol compaction skipped for this append ({err}); active segment count remains {}",
                    prior_segments.len()
                );
            }
        }
    }

    Ok(())
}

/// Tokenizes `query` and ranks candidate files in three strict tiers, per
/// query token (never a whole-query exact match -- prose-wrapped queries
/// resolve on whichever single token is an identifier):
///
/// 1. Exact-name short-circuit: a WHOLE, non-decomposed query token (via
///    `tokenize_whole` -- one raw identifier run as it appears, e.g.
///    "watch_scan_once" or "iai_mcp_store", never a snake/camel sub-part
///    split out of it) that equals a symbol's whole lowercased name
///    contributes its owning file, Def strictly above Unknown. A symbol
///    whose entire name happens to equal only a SUB-token of a multi-word
///    query (e.g. a def literally named "store" against the query
///    "IAI_MCP_STORE") must never reach this tier.
/// 2. b=0 BM25 over the symbol-name postings CSR (persisted since the
///    tracer), scored against the symbol lane's own `TermStats`, for
///    candidates the exact-name tier did not already claim -- reaches a
///    target whose only overlap with the query is a sub-token, using the
///    fully decomposed `tokenize()` token set (snake/camel parts included).
/// 3. A path/filename overlap boost, applied within and across both tiers
///    above as a tie-break, never large enough to cross a tier boundary --
///    also uses the fully decomposed token set, since a loose path-overlap
///    signal was never the precision-sensitive part of this ranking.
pub fn search(store_root: &Path, query: &str, k: usize) -> Result<Vec<(String, f64)>, SymbolIndexError> {
    let set = load_manifest(store_root)?;
    search_in(&set, store_root, query, k)
}

/// Same ranking as `search`, plus the `(n_files, generation)` pair read
/// off the SAME `load_manifest` binding the hits themselves came from --
/// same single-load rationale as `index::search_with_scope`.
pub fn search_with_scope(
    store_root: &Path,
    query: &str,
    k: usize,
) -> Result<(Vec<(String, f64)>, u64, u64), SymbolIndexError> {
    let set = load_manifest(store_root)?;
    let hits = search_in(&set, store_root, query, k)?;
    Ok((hits, set.file_index.len() as u64, set.generation))
}

fn search_in(
    set: &SegmentSet,
    store_root: &Path,
    query: &str,
    k: usize,
) -> Result<Vec<(String, f64)>, SymbolIndexError> {
    let mut query_fids: Vec<u32> = Vec::new();
    for tok in tokenize(query) {
        let fid = feature_id(&tok);
        if !query_fids.contains(&fid) {
            query_fids.push(fid);
        }
    }

    // Exact-name tier uses ONLY whole (non-decomposed) query tokens -- a
    // snake/camel sub-part of a multi-word query (folded into `query_fids`
    // above for the fuzzy BM25 tier) must never stand in for the full
    // identifier when deciding a Def-tier exact match.
    let mut exact_query_fids: Vec<u32> = Vec::new();
    for tok in tokenize_whole(query) {
        let fid = feature_id(&tok);
        if !exact_query_fids.contains(&fid) {
            exact_query_fids.push(fid);
        }
    }

    // Tier 1: path -> strongest kind observed across every exact-name match
    // in every loaded segment (Def wins over Unknown for the same path).
    let mut exact_best: HashMap<String, SymbolKind> = HashMap::new();
    // Tier 2: path -> summed b=0 BM25 score over sub-token postings hits.
    let mut bm25_scores: HashMap<String, f64> = HashMap::new();

    for handle in &set.segments {
        let seg = match SymbolSegment::load(&store_root.join(&handle.file_name)) {
            Ok(s) => s,
            Err(err) => {
                eprintln!(
                    "iai_mcp_tools_core: skipping unloadable symbol segment {} during search: {err}",
                    handle.file_name
                );
                continue;
            }
        };

        for entry in &seg.exact_entries {
            if !exact_query_fids.contains(&entry.feature_id) {
                continue;
            }
            if set.tombstones.contains(&(handle.id, entry.doc_slot)) {
                continue; // superseded by a newer posting -- must never surface
            }
            let path = seg.files[entry.doc_slot as usize].path.clone();
            let slot_kind = exact_best.entry(path).or_insert(SymbolKind::Unknown);
            if entry.kind == SymbolKind::Def {
                *slot_kind = SymbolKind::Def;
            }
        }

        for &fid in &query_fids {
            let compact_id = match seg.postings.token_lookup.get(&fid) {
                Some(&c) => c,
                None => continue,
            };
            let start = seg.postings.post_offsets[compact_id as usize] as usize;
            let end = seg.postings.post_offsets[compact_id as usize + 1] as usize;
            for i in start..end {
                let slot = seg.postings.post_doc_slots[i];
                if set.tombstones.contains(&(handle.id, slot)) {
                    continue; // superseded by a newer posting -- must never surface
                }
                let tf = seg.postings.post_tfs[i] as f64;
                let length = seg.postings.doc_len[slot as usize] as f64;
                let score = symbol_bm25_score_global(tf, fid, length, &set.term_stats);
                let path = seg.files[slot as usize].path.clone();
                *bm25_scores.entry(path).or_insert(0.0) += score;
            }
        }
    }

    let mut ranked: Vec<(String, f64)> = Vec::new();
    for (path, kind) in &exact_best {
        let base = match kind {
            SymbolKind::Def => TIER1_DEF_BASE,
            SymbolKind::Unknown => TIER1_UNKNOWN_BASE,
        };
        let boost = if path_matches_any_query_token(path, &query_fids) {
            PATH_BOOST
        } else {
            0.0
        };
        ranked.push((path.clone(), base + boost));
    }
    for (path, score) in &bm25_scores {
        // Strict separation: a path the exact-name tier already claimed
        // stays at its tier-1 score -- a sub-token BM25 sum never adds to
        // or replaces a genuine exact-name hit.
        if exact_best.contains_key(path) {
            continue;
        }
        let boost = if path_matches_any_query_token(path, &query_fids) {
            PATH_BOOST
        } else {
            0.0
        };
        ranked.push((path.clone(), score + boost));
    }

    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked.truncate(k);
    Ok(ranked)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_file(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn symbol_manifest_stats_matches_file_index_len_and_generation_after_append_and_reindex() {
        let dir = std::env::temp_dir().join(format!("lsym-manifest-stats-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("symbols");

        let (n0, g0) = manifest_stats(&store_root).unwrap();
        assert_eq!(
            (n0, g0),
            (0, 0),
            "an absent symbol manifest must report zero files and zero generation"
        );

        let a = write_file(&dir, "alpha.py", "def symbolmanifeststatsold():\n    pass\n");
        append_batch(&store_root, &[a.clone()]).unwrap();
        let manifest = load_manifest(&store_root).unwrap();
        let (n1, g1) = manifest_stats(&store_root).unwrap();
        assert_eq!(n1, manifest.file_index.len() as u64);
        assert_eq!(g1, manifest.generation);
        assert_eq!(n1, 1);

        write_file(&dir, "alpha.py", "def symbolmanifeststatsnew():\n    pass\n");
        append_batch(&store_root, &[a.clone()]).unwrap();
        let manifest2 = load_manifest(&store_root).unwrap();
        let (n2, g2) = manifest_stats(&store_root).unwrap();
        assert_eq!(n2, manifest2.file_index.len() as u64);
        assert_eq!(g2, manifest2.generation);
        assert_eq!(
            n2, 1,
            "a tombstoning re-index of an already-indexed symbol path must not inflate n_files"
        );
        assert!(g2 > g1, "a second publish must bump generation");

        fs::remove_dir_all(&dir).ok();
    }

    /// Non-vacuous single-load proof, mirroring
    /// `index::search_with_scope_reads_the_manifest_exactly_once`: value
    /// equality alone cannot distinguish a single combined read from two
    /// separate ones under single-writer quiescence, so this asserts the
    /// call count directly via the thread-local counter.
    #[test]
    fn search_with_scope_reads_the_manifest_exactly_once() {
        let dir = std::env::temp_dir().join(format!("lsym-single-load-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("symbols");

        let a = write_file(&dir, "alpha.py", "def singleloadsymbolmarker():\n    pass\n");
        append_batch(&store_root, &[a]).unwrap();

        LOAD_MANIFEST_CALLS.with(|c| c.set(0));
        let (hits, n_files, generation) =
            search_with_scope(&store_root, "singleloadsymbolmarker", 10).unwrap();
        assert_eq!(
            LOAD_MANIFEST_CALLS.with(|c| c.get()),
            1,
            "search_with_scope must read the manifest exactly once"
        );
        assert!(!hits.is_empty());
        assert_eq!(n_files, 1);
        assert!(generation >= 1);

        LOAD_MANIFEST_CALLS.with(|c| c.set(0));
        let _ = search(&store_root, "singleloadsymbolmarker", 10).unwrap();
        let _ = manifest_stats(&store_root).unwrap();
        assert_eq!(
            LOAD_MANIFEST_CALLS.with(|c| c.get()),
            2,
            "a separate search + manifest_stats pair reads the manifest twice -- the torn-read shape this fix replaces"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn search_ranks_def_owner_above_unknown_mention() {
        let dir = std::env::temp_dir().join(format!("lsym-rank-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("symbols");

        let def_file = write_file(
            &dir,
            "alpha.py",
            "def zebracorn_finder():\n    return 'zebracorn habitat lookup'\n",
        );
        let mention_file = write_file(
            &dir,
            "beta.py",
            "from alpha import zebracorn_finder\nresult = zebracorn_finder()\n",
        );

        append_batch(&store_root, &[def_file.clone(), mention_file.clone()]).unwrap();

        let hits = search(&store_root, "zebracorn_finder", 10).unwrap();
        assert!(!hits.is_empty(), "expected at least one hit");
        assert_eq!(
            hits[0].0,
            def_file.to_string_lossy(),
            "the defining file must rank first, got: {hits:?}"
        );
        assert!(
            hits.iter().any(|(p, _)| p == &mention_file.to_string_lossy()),
            "the mention-only file must still appear, got: {hits:?}"
        );

        fs::remove_dir_all(&dir).ok();
    }

    /// Tier separation gate: a Def-tier hit ranks strictly above an
    /// Unknown-tier hit, which ranks strictly above a b=0 sub-token BM25
    /// hit -- even when the BM25 hit carries a deliberately inflated raw
    /// term frequency across many files. Also proves the query survives
    /// prose-wrapping: the identifier resolves on its own token, not a
    /// whole-query exact match.
    #[test]
    fn exact_def_outranks_unknown_mention_outranks_inflated_subtoken_bm25() {
        let dir = std::env::temp_dir().join(format!("lsym-tier-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("symbols");

        let def_file = write_file(
            &dir,
            "alpha.py",
            "def rare_widget_locator():\n    return 'widget location'\n",
        );
        let mention_file = write_file(
            &dir,
            "beta.py",
            "from alpha import rare_widget_locator\nresult = rare_widget_locator()\n",
        );
        // Shares the "widget" sub-token with the query but never the exact
        // name -- many distinct widget_* mentions inflate the b=0 BM25 sum
        // for this file well past any single-occurrence score.
        let mut noisy_lines = String::new();
        for i in 0..40 {
            noisy_lines.push_str(&format!("widget_factory_variant_{i}()\n"));
        }
        let noisy_subtoken_file = write_file(&dir, "gamma.py", &noisy_lines);

        append_batch(
            &store_root,
            &[def_file.clone(), mention_file.clone(), noisy_subtoken_file.clone()],
        )
        .unwrap();

        let hits = search(&store_root, "where is rare_widget_locator defined", 10).unwrap();
        assert!(hits.len() >= 3, "expected hits from all three files, got: {hits:?}");
        assert_eq!(
            hits[0].0,
            def_file.to_string_lossy(),
            "the Def-tier file must rank first despite prose wrapping, got: {hits:?}"
        );
        assert_eq!(
            hits[1].0,
            mention_file.to_string_lossy(),
            "the Unknown exact-name mention must rank second, above the inflated sub-token BM25 hit, got: {hits:?}"
        );
        assert_eq!(
            hits[2].0,
            noisy_subtoken_file.to_string_lossy(),
            "the sub-token-only BM25 hit must rank last despite its inflated term frequency, got: {hits:?}"
        );
        assert!(
            hits[0].1 > hits[1].1 && hits[1].1 > hits[2].1,
            "scores must strictly separate the three tiers, got: {hits:?}"
        );

        fs::remove_dir_all(&dir).ok();
    }

    /// Precision regression gate for the Def-tier over-match bug: a
    /// multi-word query that is DEFINED NOWHERE as a whole identifier
    /// (e.g. "IAI_MCP_STORE", whose only occurrence anywhere is a string
    /// literal) must never reach the exact-name tier (Def or Unknown) just
    /// because one of its snake-decomposed sub-tokens ("store", "iai")
    /// happens to equal some OTHER, unrelated symbol's whole name. Those
    /// unrelated symbols may still surface through the b=0 BM25 sub-token
    /// tier; a file sharing no token at all must not appear.
    #[test]
    fn exact_tier_never_promotes_on_a_query_sub_token_matching_an_unrelated_whole_name_def() {
        let dir = std::env::temp_dir().join(format!("lsym-subtok-overmatch-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("symbols");

        // Each defines a DIFFERENT single-word symbol that is only ever a
        // SUB-token of the query "IAI_MCP_STORE" -- never its whole name.
        let store_def = write_file(&dir, "alpha.py", "def store():\n    pass\n");
        let iai_def = write_file(&dir, "beta.rs", "fn iai() {}\n");
        // Shares zero tokens with the query -- must not appear at all.
        let unrelated = write_file(&dir, "gamma.py", "def unrelated_marker():\n    pass\n");

        append_batch(&store_root, &[store_def.clone(), iai_def.clone(), unrelated.clone()]).unwrap();

        let hits = search(&store_root, "IAI_MCP_STORE", 10).unwrap();
        assert!(
            hits.iter().all(|(_, score)| *score < TIER1_UNKNOWN_BASE),
            "a query defined NOWHERE as a whole identifier must never reach the exact-name tier (Def or Unknown) via a decomposed sub-token, got: {hits:?}"
        );
        assert!(
            hits.iter().any(|(p, _)| p == &store_def.to_string_lossy()),
            "the sub-token match must still reach the target via the b=0 BM25 tier, got: {hits:?}"
        );
        assert!(
            hits.iter().any(|(p, _)| p == &iai_def.to_string_lossy()),
            "the sub-token match must still reach the target via the b=0 BM25 tier, got: {hits:?}"
        );
        assert!(
            !hits.iter().any(|(p, _)| p == &unrelated.to_string_lossy()),
            "a file sharing zero tokens with the query must never appear, got: {hits:?}"
        );

        fs::remove_dir_all(&dir).ok();
    }

    /// Positive counterpart to the over-match regression above: when the
    /// query's WHOLE token genuinely equals a symbol's whole name, that
    /// symbol must still rank first at the Def tier -- fixing the
    /// over-match must not also break the legitimate exact match. An
    /// unrelated file that only shares one decomposed sub-token
    /// ("store") must not be promoted to the exact tier alongside it.
    #[test]
    fn exact_tier_still_promotes_the_true_owner_of_a_multi_word_symbol_name() {
        let dir = std::env::temp_dir().join(format!("lsym-subtok-truematch-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("symbols");

        let def_file = write_file(&dir, "alpha.py", "def iai_mcp_store():\n    return 1\n");
        let sub_token_only = write_file(&dir, "beta.py", "def store():\n    pass\n");

        append_batch(&store_root, &[def_file.clone(), sub_token_only.clone()]).unwrap();

        let hits = search(&store_root, "IAI_MCP_STORE", 10).unwrap();
        assert!(!hits.is_empty(), "expected at least one hit, got: {hits:?}");
        assert_eq!(
            hits[0].0,
            def_file.to_string_lossy(),
            "the true whole-name owner must rank first at the Def tier, got: {hits:?}"
        );
        assert!(
            hits[0].1 >= TIER1_DEF_BASE,
            "the true whole-name owner must score at the Def tier, got: {hits:?}"
        );
        let sub_token_score = hits
            .iter()
            .find(|(p, _)| p == &sub_token_only.to_string_lossy())
            .map(|(_, s)| *s);
        assert!(
            sub_token_score.map_or(true, |s| s < TIER1_UNKNOWN_BASE),
            "a file matching only a decomposed sub-token must not reach the exact-name tier, got: {hits:?}"
        );

        fs::remove_dir_all(&dir).ok();
    }

    /// A query token that is only a SUB-token of the target symbol name
    /// (never an exact match anywhere) still reaches the target through
    /// the b=0 BM25 sub-token tier.
    #[test]
    fn subtoken_only_query_reaches_target_via_b_zero_bm25_tier() {
        let dir = std::env::temp_dir().join(format!("lsym-subtok-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("symbols");

        // Only ever mentioned, never defined, and never named exactly
        // "DAEMON_SOCKET_PATH" -- the query's identifier token is a
        // sub-token of this longer name.
        let target_file = write_file(
            &dir,
            "delta.py",
            "IAI_DAEMON_SOCKET_PATH = '/tmp/x.sock'\n",
        );
        let unrelated_file = write_file(&dir, "epsilon.py", "def unrelated_helper():\n    pass\n");

        append_batch(&store_root, &[target_file.clone(), unrelated_file.clone()]).unwrap();

        // Prose word "override" never appears in the corpus -- harmless.
        let hits = search(&store_root, "DAEMON_SOCKET_PATH override", 10).unwrap();
        assert!(
            hits.iter().any(|(p, _)| p == &target_file.to_string_lossy()),
            "the sub-token-only target must be reachable via the b=0 BM25 tier, got: {hits:?}"
        );
        assert!(
            !hits.iter().any(|(p, _)| p == &unrelated_file.to_string_lossy()),
            "an unrelated file sharing no token with the query must not appear, got: {hits:?}"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn append_batch_persists_across_calls_and_search_finds_both_batches() {
        let dir = std::env::temp_dir().join(format!("lsym-incr-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("symbols");

        let a = write_file(&dir, "one.py", "def alpha_marker():\n    pass\n");
        append_batch(&store_root, &[a.clone()]).unwrap();

        let b = write_file(&dir, "two.py", "def beta_marker():\n    pass\n");
        append_batch(&store_root, &[b.clone()]).unwrap();

        let manifest = load_manifest(&store_root).unwrap();
        assert_eq!(manifest.segments.len(), 2, "each batch writes one new segment");

        let hits = search(&store_root, "alpha_marker", 5).unwrap();
        assert!(hits.iter().any(|(p, _)| p == &a.to_string_lossy()));
        let hits = search(&store_root, "beta_marker", 5).unwrap();
        assert!(hits.iter().any(|(p, _)| p == &b.to_string_lossy()));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn append_batch_empty_paths_is_a_no_op() {
        let dir = std::env::temp_dir().join(format!("lsym-empty-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("symbols");

        let result = append_batch(&store_root, &[]);
        assert!(result.is_ok());
        assert!(!manifest_path(&store_root).exists());

        fs::remove_dir_all(&dir).ok();
    }

    /// Correctness GATE, not a note: indexes the SAME fixed corpus two
    /// ways -- arm A one batch, arm B more single-file batches than the
    /// locked compaction threshold (so >=1 merge fires by batch count
    /// alone) -- and asserts each file's own defining-name search ranks
    /// identically between arms.
    #[test]
    fn append_equals_rebuild_matches_single_shot_symbol_index_across_a_compaction() {
        let dir = std::env::temp_dir().join(format!("lsym-a-eq-r-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();

        let n = symbol_compaction::COMPACTION_THRESHOLD + 4;
        let mut paths = Vec::with_capacity(n);
        for i in 0..n {
            let content = format!("def rare_marker_{i}_def():\n    pass\n");
            paths.push(write_file(&dir, &format!("file{i}.py"), &content));
        }

        // Arm A: one batch, one store root.
        let root_a = dir.join("store-a");
        append_batch(&root_a, &paths).unwrap();

        // Arm B: one single-file batch per path -- more batches than the
        // locked threshold, so compaction fires by batch count alone,
        // never by lowering or overriding the constant.
        let root_b = dir.join("store-b");
        for p in &paths {
            append_batch(&root_b, std::slice::from_ref(p)).unwrap();
        }

        let set_b = load_manifest(&root_b).unwrap();
        assert!(
            set_b.segments.len() < n,
            "arm B must have compacted at least once, got {} segments for {n} batches",
            set_b.segments.len()
        );

        for i in 0..n {
            let q = format!("rare_marker_{i}_def");
            let hits_a = search(&root_a, &q, 10).unwrap();
            let hits_b = search(&root_b, &q, 10).unwrap();
            assert_eq!(
                hits_a, hits_b,
                "arm A (single-shot) and arm B (incremental+compacted) must rank identically for query {q:?}"
            );
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reindexing_unchanged_symbol_file_is_a_no_op() {
        let dir = std::env::temp_dir().join(format!("lsym-noop-reindex-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("symbols");

        let a = write_file(&dir, "alpha.py", "def uniquesymbolnoop():\n    pass\n");
        append_batch(&store_root, &[a.clone()]).unwrap();

        let manifest_bytes_before = fs::read(manifest_path(&store_root)).unwrap();
        let segments_before = load_manifest(&store_root).unwrap().segments.len();
        let hits_before = search(&store_root, "uniquesymbolnoop", 5).unwrap().len();
        assert_eq!(hits_before, 1);

        crate::symbol_segment::reset_segment_load_count();
        append_batch(&store_root, &[a.clone()]).unwrap();
        assert_eq!(
            crate::symbol_segment::segment_load_count(),
            0,
            "a no-op re-index of an unchanged symbol file must never open a prior segment file"
        );

        let manifest_bytes_after = fs::read(manifest_path(&store_root)).unwrap();
        let segments_after = load_manifest(&store_root).unwrap().segments.len();
        let hits_after = search(&store_root, "uniquesymbolnoop", 5).unwrap().len();

        assert_eq!(
            manifest_bytes_before, manifest_bytes_after,
            "re-indexing an unchanged symbol file must leave the symbol manifest byte-identical"
        );
        assert_eq!(
            segments_before, segments_after,
            "re-indexing an unchanged symbol file must add zero net new segments"
        );
        assert_eq!(
            hits_after, 1,
            "re-indexing an unchanged symbol file must not double the hit count"
        );

        fs::remove_dir_all(&dir).ok();
    }

    /// Exercises BOTH enumerated slot-to-path tombstone sites in one test:
    /// a whole-name marker only ever reachable through the exact-name tier
    /// (`seg.exact_entries`), and a snake-decomposed sub-token marker only
    /// ever reachable through the b=0 BM25 tier (`seg.postings`). Neither
    /// marker shares a token with the other's replacement, so a leak
    /// through either site is unambiguous.
    #[test]
    fn changed_symbol_file_replaces_prior_symbols() {
        let dir = std::env::temp_dir().join(format!("lsym-replace-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("symbols");

        let a = write_file(
            &dir,
            "alpha.py",
            "def markeroldonlywhole():\n    pass\ndef helper_oldsubtokenpart():\n    pass\n",
        );
        append_batch(&store_root, &[a.clone()]).unwrap();

        write_file(
            &dir,
            "alpha.py",
            "def markernewonlywhole():\n    pass\ndef helper_newsubtokenpart():\n    pass\n",
        );
        append_batch(&store_root, &[a.clone()]).unwrap();

        // Exact-name tier (whole identifier lookup).
        let old_hits_whole = search(&store_root, "markeroldonlywhole", 5).unwrap();
        assert!(
            old_hits_whole.is_empty(),
            "the old whole-name marker must not be findable after the file changed, got: {old_hits_whole:?}"
        );
        let new_hits_whole = search(&store_root, "markernewonlywhole", 5).unwrap();
        assert!(
            new_hits_whole.iter().any(|(p, _)| p == &a.to_string_lossy()),
            "the new whole-name marker must be findable after the file changed, got: {new_hits_whole:?}"
        );

        // b=0 BM25 sub-token tier (snake-decomposed part, never a whole
        // identifier -- unreachable through the exact-name tier).
        let old_hits_subtoken = search(&store_root, "oldsubtokenpart", 5).unwrap();
        assert!(
            old_hits_subtoken.is_empty(),
            "the old sub-token marker must not be findable after the file changed, got: {old_hits_subtoken:?}"
        );
        let new_hits_subtoken = search(&store_root, "newsubtokenpart", 5).unwrap();
        assert!(
            new_hits_subtoken.iter().any(|(p, _)| p == &a.to_string_lossy()),
            "the new sub-token marker must be findable after the file changed, got: {new_hits_subtoken:?}"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn search_skips_unloadable_symbol_segment_and_still_serves_healthy_ones() {
        let dir = std::env::temp_dir().join(format!("lsym-corrupt-seg-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("symbols");

        // No underscore/camel case in the name -- these tokenize to exactly
        // ONE token each, so the three files share zero sub-tokens, and the
        // b=0 sub-token BM25 tier cannot bridge a query for one file's name
        // to another file's postings.
        let a = write_file(&dir, "alpha.py", "def uniquetokenalphaonly():\n    pass\n");
        append_batch(&store_root, &[a.clone()]).unwrap();
        let b = write_file(&dir, "beta.py", "def uniquetokenbetaonly():\n    pass\n");
        append_batch(&store_root, &[b.clone()]).unwrap();

        // Corrupt the first-appended segment (id 1, alpha's) in place --
        // flips CRC mismatch, so `SymbolSegment::load` fails closed for it.
        let corrupt_path = store_root.join(segment_file_name(1));
        let mut raw = fs::read(&corrupt_path).unwrap();
        let last = raw.len() - 1;
        raw[last] ^= 0xFF;
        fs::write(&corrupt_path, &raw).unwrap();

        // The healthy segment's content must still be found -- a search
        // must not raise just because one OTHER segment is unloadable.
        let hits = search(&store_root, "uniquetokenbetaonly", 5).unwrap();
        assert!(
            hits.iter().any(|(p, _)| p == &b.to_string_lossy()),
            "the healthy segment must still serve results despite a sibling corrupt segment"
        );

        // The corrupted segment's own content is unreachable, but the
        // query must still return cleanly (empty), never error.
        let hits_alpha = search(&store_root, "uniquetokenalphaonly", 5).unwrap();
        assert!(hits_alpha.is_empty());

        // A store with one corrupt segment must still accept a new
        // append -- no permanent wedge.
        let c = write_file(&dir, "gamma.py", "def uniquetokengammaonly():\n    pass\n");
        let append_result = append_batch(&store_root, &[c.clone()]);
        assert!(
            append_result.is_ok(),
            "append must succeed even with an unloadable sibling symbol segment present"
        );
        let hits_gamma = search(&store_root, "uniquetokengammaonly", 5).unwrap();
        assert!(hits_gamma.iter().any(|(p, _)| p == &c.to_string_lossy()));

        fs::remove_dir_all(&dir).ok();
    }
}
