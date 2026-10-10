//! Xref-lane orchestration: an on-disk manifest (own magic, own store
//! subdir) plus one immutable LXRFIDX1 edge segment per batch. Unlike the
//! whole-file and symbol lanes, edges carry no BM25 term-stats table --
//! they are consulted by from-path lookup only, never ranked on their
//! own, so there is no resident stat to fold across a publish and no
//! `SegmentSetBuffer` instance is needed here -- the manifest is read
//! fresh from disk on every call instead. Triggers `xref_compaction`
//! after publishing, the same way `symbol_index.rs` fires the symbol
//! lane's compaction, but performs the (merged IN, sources OUT) manifest
//! swap directly against the plain handle list rather than through that
//! generic double-buffer.
//!
//! The bidirectional (from-symbol/to-symbol) edge query built in this
//! file is a light CSR keyed by `feature_id(name)`, copied by SHAPE from
//! `postings.rs`'s CSR builder (compact-id lookup + offsets + flat row
//! array) -- never its `PostingsCsr` type itself, whose tf/doc_len/avg_len
//! fields are BM25-specific and carry no meaning for an edge table. It is
//! rebuilt fresh from every currently-loaded segment on each call, strictly
//! 1-hop, no traversal/community/centrality machinery.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::index;
use crate::tokenize::feature_id;
use crate::xref_compaction;
use crate::xref_scan::XrefKind;
use crate::xref_segment::{XrefSegment, XrefSegmentError};

const MANIFEST_MAGIC_V1: &[u8; 8] = b"LXRFMNF1";
const MANIFEST_MAGIC_V2: &[u8; 8] = b"LXRFMNF2";
const MANIFEST_FILE: &str = "manifest.lxman";

#[derive(Debug, Error)]
pub enum XrefIndexError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("xref manifest corrupt: {0}")]
    Corrupt(String),
    #[error(transparent)]
    Segment(#[from] XrefSegmentError),
    #[error(transparent)]
    WholeFile(#[from] index::IndexError),
}

/// A lightweight reference to an on-disk xref segment -- an id (unique
/// among currently active segments) and its file name under the store
/// root. Never carries the loaded `XrefSegment`. `pub(crate)` so
/// `xref_compaction.rs` (a sibling module, never a descendant) can build
/// and swap handles without opening a segment it does not need to read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct XrefHandle {
    pub(crate) id: u32,
    pub(crate) file_name: String,
}

fn manifest_path(store_root: &Path) -> PathBuf {
    store_root.join(MANIFEST_FILE)
}

/// Full on-disk xref manifest state: segment handles plus the two
/// additive v2 sections -- a `path -> (segment_id, file_id, content_hash)`
/// index consulted by `append_batch` before scanning, and the
/// `(segment_id, file_id)` tombstone set consulted at every edge-emit
/// site. No resident term-stats table (unlike the whole-file/symbol
/// lanes) -- edges are never ranked on their own.
pub(crate) struct XrefManifestState {
    pub(crate) handles: Vec<XrefHandle>,
    pub(crate) file_index: HashMap<String, (u32, u32, u64)>,
    pub(crate) tombstones: HashSet<(u32, u32)>,
}

/// Loads just the segment handle list -- the pre-v2 accessor, kept as-is
/// so every existing caller/test that treats the manifest as a plain
/// `Vec<XrefHandle>` is unaffected by the v2 file-index/tombstone
/// sections.
pub(crate) fn load_manifest(store_root: &Path) -> Result<Vec<XrefHandle>, XrefIndexError> {
    Ok(load_manifest_full(store_root)?.handles)
}

/// Loads the on-disk xref manifest in full. Metadata I/O only; never
/// opens a `xref-*.lxidx` file. Version-aware: an old-magic (v1) manifest
/// parses with an empty file-index and empty tombstones (safe degrade, no
/// forced rebuild); a new-magic (v2) manifest additionally carries both
/// sections.
pub(crate) fn load_manifest_full(store_root: &Path) -> Result<XrefManifestState, XrefIndexError> {
    let path = manifest_path(store_root);
    if !path.exists() {
        return Ok(XrefManifestState {
            handles: Vec::new(),
            file_index: HashMap::new(),
            tombstones: HashSet::new(),
        });
    }
    let raw = fs::read(&path)?;
    if raw.len() < 12 {
        return Err(XrefIndexError::Corrupt("not an xref manifest file".into()));
    }
    let magic: [u8; 8] = raw[..8].try_into().unwrap();
    let is_v2 = if magic == *MANIFEST_MAGIC_V2 {
        true
    } else if magic == *MANIFEST_MAGIC_V1 {
        false
    } else {
        return Err(XrefIndexError::Corrupt("not an xref manifest file".into()));
    };
    let stored_crc = u32::from_le_bytes(raw[8..12].try_into().unwrap());
    let payload = &raw[12..];
    if crc32fast::hash(payload) != stored_crc {
        return Err(XrefIndexError::Corrupt(
            "xref manifest CRC mismatch".into(),
        ));
    }

    let need = |off: usize, len: usize| -> Result<(), XrefIndexError> {
        if off + len > payload.len() {
            Err(XrefIndexError::Corrupt("truncated xref manifest".into()))
        } else {
            Ok(())
        }
    };

    let mut off = 0usize;
    need(off, 4)?;
    let segment_count = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()) as usize;
    off += 4;
    need(off, segment_count.saturating_mul(6))?;

    let mut handles = Vec::with_capacity(segment_count);
    for _ in 0..segment_count {
        need(off, 4)?;
        let id = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap());
        off += 4;
        need(off, 2)?;
        let name_len = u16::from_le_bytes(payload[off..off + 2].try_into().unwrap()) as usize;
        off += 2;
        need(off, name_len)?;
        let name = std::str::from_utf8(&payload[off..off + name_len])
            .map_err(|_| XrefIndexError::Corrupt("non-utf8 segment name".into()))?
            .to_string();
        off += name_len;
        handles.push(XrefHandle {
            id,
            file_name: name,
        });
    }

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
                .map_err(|_| XrefIndexError::Corrupt("non-utf8 file-index path".into()))?
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
        return Err(XrefIndexError::Corrupt(
            "trailing bytes after xref manifest payload".into(),
        ));
    }

    Ok(XrefManifestState {
        handles,
        file_index,
        tombstones,
    })
}

fn save_manifest(store_root: &Path, state: &XrefManifestState) -> Result<(), XrefIndexError> {
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(&(state.handles.len() as u32).to_le_bytes());
    for h in &state.handles {
        payload.extend_from_slice(&h.id.to_le_bytes());
        let name_bytes = h.file_name.as_bytes();
        payload.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        payload.extend_from_slice(name_bytes);
    }

    // v2 sections -- sorted so a manifest re-saved from an unchanged
    // logical state is byte-stable.
    let mut file_index_entries: Vec<(&String, &(u32, u32, u64))> =
        state.file_index.iter().collect();
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

    let mut tombstone_entries: Vec<&(u32, u32)> = state.tombstones.iter().collect();
    tombstone_entries.sort();
    payload.extend_from_slice(&(tombstone_entries.len() as u32).to_le_bytes());
    for &(segment_id, file_id) in tombstone_entries {
        payload.extend_from_slice(&segment_id.to_le_bytes());
        payload.extend_from_slice(&file_id.to_le_bytes());
    }

    let crc = crc32fast::hash(&payload);
    let path = manifest_path(store_root);
    let tmp = path.with_extension("tmp-xrefmanifest");
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
    format!("xref-{id}.lxidx")
}

/// The corpus root a single batch resolves relative imports against: the
/// longest common ancestor directory of every path's own parent
/// directory. A single-file batch's corpus root is that file's own
/// parent -- the narrowest containment boundary the batch itself
/// supports, so a relative import climbing above it is rejected rather
/// than silently widened to some larger, unverified boundary.
fn batch_corpus_root(paths: &[PathBuf]) -> PathBuf {
    let mut root: Option<PathBuf> = None;
    for p in paths {
        let dir = p.parent().unwrap_or_else(|| Path::new(""));
        root = Some(match root {
            None => dir.to_path_buf(),
            Some(prev) => common_ancestor(&prev, dir),
        });
    }
    root.unwrap_or_else(|| PathBuf::from("."))
}

fn common_ancestor(a: &Path, b: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for (ca, cb) in a.components().zip(b.components()) {
        if ca != cb {
            break;
        }
        out.push(ca.as_os_str());
    }
    out
}

/// Scans exactly the given paths for the one recognized import shape and
/// publishes ONE new immutable edge segment under `store_root` (the xref
/// lane's own subdir) -- never opens, reads, or re-scans any file
/// already covered by a prior xref segment. A path whose content is
/// unchanged since the last append is skipped outright as a genuine
/// no-op; a path whose content changed has its prior (segment_id,
/// file_id) tombstoned so its superseded edges stop surfacing at query
/// time.
pub fn append_batch(store_root: &Path, paths: &[PathBuf]) -> Result<(), XrefIndexError> {
    if paths.is_empty() {
        return Ok(());
    }

    fs::create_dir_all(store_root)?;
    let mut state = load_manifest_full(store_root)?;

    let corpus_root = batch_corpus_root(paths);
    let (segment, new_tombstones) =
        XrefSegment::build_incremental(paths, &corpus_root, &state.file_index)?;
    if segment.files.is_empty() && new_tombstones.is_empty() {
        // Every path in this batch is already current -- no new segment,
        // no tombstone, manifest stays byte-identical -- a genuine no-op.
        return Ok(());
    }

    let id = state.handles.iter().map(|h| h.id).max().unwrap_or(0) + 1;
    let file_name = segment_file_name(id);
    let file_hashes = segment.file_hashes();
    segment.save(&store_root.join(&file_name))?;

    state.handles.push(XrefHandle {
        id,
        file_name: file_name.clone(),
    });
    for (slot, (path, hash)) in file_hashes.into_iter().enumerate() {
        state.file_index.insert(path, (id, slot as u32, hash));
    }
    for t in new_tombstones {
        state.tombstones.insert(t);
    }
    save_manifest(store_root, &state)?;

    if xref_compaction::should_compact(state.handles.len()) {
        let prior_handles = state.handles.clone();
        // The append above is already durable in the manifest -- a
        // compaction failure (e.g. one unloadable source segment) must
        // degrade to "compaction skipped this round", never fail the
        // append that already succeeded. Otherwise a single bad segment
        // wedges every future append past the threshold permanently.
        match xref_compaction::compact(store_root, &state.handles, &state.tombstones) {
            Ok((merged_handle, source_ids)) => {
                match XrefSegment::load(&store_root.join(&merged_handle.file_name)) {
                    Ok(merged_seg) => {
                        let mut next_handles: Vec<XrefHandle> = prior_handles
                            .iter()
                            .filter(|h| !source_ids.contains(&h.id))
                            .cloned()
                            .collect();
                        next_handles.push(merged_handle.clone());

                        // Rebuild the file-index wholesale from the merged
                        // segment -- compaction merges every active
                        // segment, so the merged file table is now the
                        // complete, authoritative set of every currently
                        // tracked path.
                        let mut rebuilt_file_index: HashMap<String, (u32, u32, u64)> =
                            HashMap::new();
                        for (slot, f) in merged_seg.files.iter().enumerate() {
                            rebuilt_file_index
                                .insert(f.path.clone(), (merged_handle.id, slot as u32, f.content_hash));
                        }

                        // Tombstones referencing a now-deleted source
                        // segment are permanently reclaimed here -- never
                        // carried forward as dead weight.
                        let remaining_tombstones: HashSet<(u32, u32)> = state
                            .tombstones
                            .into_iter()
                            .filter(|(seg_id, _)| !source_ids.contains(seg_id))
                            .collect();

                        let next_state = XrefManifestState {
                            handles: next_handles,
                            file_index: rebuilt_file_index,
                            tombstones: remaining_tombstones,
                        };
                        save_manifest(store_root, &next_state)?;
                        xref_compaction::delete_sources(store_root, &source_ids, &prior_handles);
                    }
                    Err(err) => {
                        eprintln!(
                            "iai_mcp_tools_core: xref compaction produced an unreadable merged segment ({err}); manifest swap skipped for this round"
                        );
                    }
                }
            }
            Err(err) => {
                eprintln!(
                    "iai_mcp_tools_core: xref compaction skipped for this append ({err}); active segment count remains {}",
                    prior_handles.len()
                );
            }
        }
    }

    Ok(())
}

/// Merges a base BM25 ranking with xref-injected adjacency files. A base
/// hit's presence and its position relative to every other base hit are
/// never altered by injection: `ranked` starts as `base` verbatim
/// (deduplicated, order preserved), and every injected file is appended
/// strictly after the LAST base hit -- never spliced between two base
/// hits, so a base hit can never be evicted or reordered. Injected files
/// are visited in referencing-hit order and deduplicated against both the
/// base and any file injected earlier in the same call. The caller
/// controls the output's upper bound by how large a `base` slice it
/// passes in -- this function itself never drops an element of `base`,
/// regardless of `base`'s length. Returns the merged rows plus the count
/// of rows that came from injection.
fn merge_base_with_injections(
    base: &[(String, f64)],
    edges_by_from: &HashMap<String, Vec<String>>,
) -> (Vec<(String, f64)>, usize) {
    let mut seen: HashSet<String> = HashSet::new();
    let mut ranked: Vec<(String, f64)> = Vec::with_capacity(base.len());
    for (path, score) in base {
        if seen.insert(path.clone()) {
            ranked.push((path.clone(), *score));
        }
    }

    let mut injected_count = 0usize;
    for (path, score) in base {
        if let Some(targets) = edges_by_from.get(path) {
            for target in targets {
                if seen.insert(target.clone()) {
                    ranked.push((target.clone(), *score));
                    injected_count += 1;
                }
            }
        }
    }

    (ranked, injected_count)
}

/// Runs the whole-file BM25 search over `store_root` (the existing search
/// path in `index.rs`) for its own top-`k` hits, then looks up every file
/// each of those hits references via `xref_store_root`'s edge table. A
/// base hit is never demoted, evicted, or reordered by injection -- see
/// `merge_base_with_injections`, which this function feeds exactly the
/// top-`k` base ranking (never more), so the base portion of the result
/// is always identical to the plain whole-file baseline's own top `k`.
/// With zero injected files the result is exactly that top `k`; with one
/// or more injected files the result holds those `k` base hits plus the
/// injected files appended after them, so it may exceed `k` rows. An
/// injected file's score is inherited from the referencing hit that
/// reached it -- callers rendering scores should not read two equal
/// scores as independently verified equal relevance.
pub fn search(
    store_root: &Path,
    xref_store_root: &Path,
    query: &str,
    k: usize,
) -> Result<Vec<(String, f64)>, XrefIndexError> {
    let base = index::search(store_root, query, k)?;

    let manifest = load_manifest_full(xref_store_root)?;
    let mut edges_by_from: HashMap<String, Vec<String>> = HashMap::new();
    for handle in &manifest.handles {
        let seg = match XrefSegment::load(&xref_store_root.join(&handle.file_name)) {
            Ok(s) => s,
            Err(err) => {
                eprintln!(
                    "iai_mcp_tools_core: skipping unloadable xref segment {} during search: {err}",
                    handle.file_name
                );
                continue;
            }
        };
        for edge in &seg.edges {
            let Some(to_slot) = edge.to_slot else {
                continue;
            };
            // Tombstone filter applied at BOTH the from-path and to-path
            // emit sites -- either end being superseded invalidates this
            // edge for injection.
            if manifest.tombstones.contains(&(handle.id, edge.from_slot))
                || manifest.tombstones.contains(&(handle.id, to_slot))
            {
                continue;
            }
            let from_path = seg.files[edge.from_slot as usize].path.clone();
            let to_path = seg.files[to_slot as usize].path.clone();
            edges_by_from.entry(from_path).or_default().push(to_path);
        }
    }

    let (ranked, _injected_count) = merge_base_with_injections(&base, &edges_by_from);
    Ok(ranked)
}

/// One edge, joined against its own segment's file table so a caller
/// gets ready-to-use paths without holding an `XrefSegment` handle.
struct EdgeRow {
    from_path: String,
    to_path: Option<String>,
    name: String,
    kind: XrefKind,
}

/// A light CSR keyed by `feature_id(name.to_lowercase())`: `token_lookup`
/// maps a feature id to a compact index, `offsets`/`row_ids` are the flat
/// CSR pair. Copied by shape from `postings.rs`'s two-pass counting
/// builder -- no tf/doc_len/BM25 fields, which mean nothing for an edge
/// table.
struct SymbolCsr {
    token_lookup: HashMap<u32, u32>,
    offsets: Vec<u32>,
    row_ids: Vec<u32>,
}

fn build_symbol_csr(rows: &[EdgeRow]) -> SymbolCsr {
    let mut token_lookup: HashMap<u32, u32> = HashMap::new();
    let mut token_row_count: Vec<u32> = Vec::new();
    let mut raw: Vec<(u32, u32)> = Vec::new(); // (compact_id, row_idx)

    for (idx, row) in rows.iter().enumerate() {
        let fid = feature_id(&row.name.to_lowercase());
        let compact_id = match token_lookup.get(&fid) {
            Some(&id) => id,
            None => {
                let id = token_lookup.len() as u32;
                token_lookup.insert(fid, id);
                id
            }
        };
        if compact_id as usize == token_row_count.len() {
            token_row_count.push(0);
        }
        token_row_count[compact_id as usize] += 1;
        raw.push((compact_id, idx as u32));
    }

    let ntokens = token_lookup.len();
    let mut offsets = Vec::with_capacity(ntokens + 1);
    offsets.push(0u32);
    for &c in &token_row_count {
        offsets.push(offsets.last().unwrap() + c);
    }
    let total = raw.len();
    let mut row_ids = vec![0u32; total];
    let mut cursor: Vec<u32> = offsets[..ntokens].to_vec();
    for (compact_id, row_idx) in raw {
        let pos = cursor[compact_id as usize] as usize;
        row_ids[pos] = row_idx;
        cursor[compact_id as usize] += 1;
    }

    SymbolCsr {
        token_lookup,
        offsets,
        row_ids,
    }
}

/// Loads every active segment under `xref_store_root` and flattens their
/// edges into join-ready rows. Skips (never fails on) an unloadable
/// sibling segment, matching `search`'s own fail-soft posture.
fn load_all_edge_rows(xref_store_root: &Path) -> Result<Vec<EdgeRow>, XrefIndexError> {
    let manifest = load_manifest_full(xref_store_root)?;
    let mut rows = Vec::new();
    for handle in &manifest.handles {
        let seg = match XrefSegment::load(&xref_store_root.join(&handle.file_name)) {
            Ok(s) => s,
            Err(err) => {
                eprintln!(
                    "iai_mcp_tools_core: skipping unloadable xref segment {} during symbol lookup: {err}",
                    handle.file_name
                );
                continue;
            }
        };
        for edge in &seg.edges {
            // Tombstone filter applied at BOTH the from-path and to-path
            // emit sites, feeding both discovery directions
            // (`definitions_for_reference`/`call_sites_for_definition`)
            // that consume this row set.
            if manifest.tombstones.contains(&(handle.id, edge.from_slot))
                || edge
                    .to_slot
                    .is_some_and(|t| manifest.tombstones.contains(&(handle.id, t)))
            {
                continue;
            }
            let from_path = seg.files[edge.from_slot as usize].path.clone();
            let to_path = edge.to_slot.map(|s| seg.files[s as usize].path.clone());
            rows.push(EdgeRow {
                from_path,
                to_path,
                name: edge.name.clone(),
                kind: edge.kind,
            });
        }
    }
    Ok(rows)
}

/// From-symbol direction: every edge whose reference target is `name`
/// (case-insensitive, whole-name match), regardless of resolution state
/// -- lets a caller starting from a REFERENCE reach its definition
/// candidate(s) (the `Some` half of each pair), while an unresolved
/// reference still surfaces as a `None` rather than vanishing. The
/// returned `XrefKind` distinguishes an `Import` row from a `Call` row
/// to the SAME target -- callers must never fold two edges of differing
/// kind into one, since that discards which reference shape produced
/// each row.
pub fn definitions_for_reference(
    xref_store_root: &Path,
    name: &str,
) -> Result<Vec<(String, Option<String>, XrefKind)>, XrefIndexError> {
    let rows = load_all_edge_rows(xref_store_root)?;
    let csr = build_symbol_csr(&rows);
    let fid = feature_id(&name.to_lowercase());
    let Some(&cid) = csr.token_lookup.get(&fid) else {
        return Ok(Vec::new());
    };
    let start = csr.offsets[cid as usize] as usize;
    let end = csr.offsets[cid as usize + 1] as usize;
    Ok(csr.row_ids[start..end]
        .iter()
        .map(|&i| {
            let row = &rows[i as usize];
            (row.from_path.clone(), row.to_path.clone(), row.kind)
        })
        .collect())
}

/// To-symbol direction: every RESOLVED edge whose definition target is
/// `name` AND whose resolved file is `def_path` -- lets a caller starting
/// from a DEFINITION (identified by its file, since a bare name alone
/// does not uniquely name a definition) surface its referencing call
/// sites. Consults the SAME CSR `definitions_for_reference` builds --
/// one exact-name index, two discovery directions.
#[allow(dead_code)] // called by this file's own tests; not yet wired to lib.rs/PyO3
pub fn call_sites_for_definition(
    xref_store_root: &Path,
    def_path: &str,
    name: &str,
) -> Result<Vec<String>, XrefIndexError> {
    let rows = load_all_edge_rows(xref_store_root)?;
    let csr = build_symbol_csr(&rows);
    let fid = feature_id(&name.to_lowercase());
    let Some(&cid) = csr.token_lookup.get(&fid) else {
        return Ok(Vec::new());
    };
    let start = csr.offsets[cid as usize] as usize;
    let end = csr.offsets[cid as usize + 1] as usize;
    Ok(csr.row_ids[start..end]
        .iter()
        .filter_map(|&i| {
            let row = &rows[i as usize];
            match &row.to_path {
                Some(p) if p == def_path => Some(row.from_path.clone()),
                _ => None,
            }
        })
        .collect())
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
    fn merge_base_with_injections_never_evicts_a_base_hit_below_k() {
        // Base ranking of 3 hits (A, B, C) with k=2 conceptually smaller
        // than the base itself: A references X, an injected file not
        // present in the base at all. Before the CR-01 fix, splicing X in
        // immediately after A and truncating to k=2 would evict C from
        // the returned window even though C outranks nothing that
        // displaced it -- exactly the measured regression.
        let base = vec![
            ("A".to_string(), 0.9),
            ("B".to_string(), 0.8),
            ("C".to_string(), 0.7),
        ];
        let mut edges_by_from: HashMap<String, Vec<String>> = HashMap::new();
        edges_by_from.insert("A".to_string(), vec!["X".to_string()]);

        let (ranked, injected_count) = merge_base_with_injections(&base, &edges_by_from);

        let base_positions: Vec<&str> = ranked
            .iter()
            .map(|(p, _)| p.as_str())
            .filter(|p| *p != "X")
            .collect();
        assert_eq!(
            base_positions,
            vec!["A", "B", "C"],
            "every base hit must still be present, in its original relative order, got: {ranked:?}"
        );

        let x_pos = ranked
            .iter()
            .position(|(p, _)| p == "X")
            .expect("expected X to be injected into the results");
        let c_pos = ranked.iter().position(|(p, _)| p == "C").unwrap();
        assert!(
            x_pos > c_pos,
            "an injected file must land after every base hit, not between them, got: {ranked:?}"
        );
        assert_eq!(injected_count, 1);
    }

    #[test]
    fn merge_base_with_injections_treats_every_base_hit_as_an_injection_source() {
        // search() now feeds merge_base_with_injections exactly the
        // top-k base slice (never the wider overfetch window) -- so
        // every hit in that slice, not just the first, must still be
        // able to contribute injected files. B (not A) references X.
        let base = vec![("A".to_string(), 0.9), ("B".to_string(), 0.8)];
        let mut edges_by_from: HashMap<String, Vec<String>> = HashMap::new();
        edges_by_from.insert("B".to_string(), vec!["X".to_string()]);

        let (ranked, injected_count) = merge_base_with_injections(&base, &edges_by_from);

        assert!(
            ranked.iter().any(|(p, _)| p == "X"),
            "a base hit other than the first must still be able to inject a file, got: {ranked:?}"
        );
        assert_eq!(injected_count, 1);
    }

    #[test]
    fn search_xref_appends_referenced_file_after_base_hits_without_eviction() {
        let dir = std::env::temp_dir().join(format!("lxrf-adjacency-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("tools");
        let xref_store_root = dir.join("tools-xref");

        let def_file = write_file(&dir, "def_file.py", "def zebracorn_helper():\n    return 1\n");
        let importer_file = write_file(
            &dir,
            "importer_file.py",
            "from def_file import zebracorn_helper\nresult = zebracorn_helper()\nprint('lighthouse beacon signal pattern seven')\n",
        );

        index::append_batch(&store_root, &[def_file.clone(), importer_file.clone()]).unwrap();
        append_batch(&xref_store_root, &[def_file.clone(), importer_file.clone()]).unwrap();

        let hits = search(
            &store_root,
            &xref_store_root,
            "lighthouse beacon signal pattern seven",
            10,
        )
        .unwrap();

        let importer_idx = hits
            .iter()
            .position(|(p, _)| p == &importer_file.to_string_lossy())
            .expect("expected the referencing file to be a base BM25 hit");
        let def_idx = hits
            .iter()
            .position(|(p, _)| p == &def_file.to_string_lossy())
            .expect("expected the referenced definition file to be injected into the results");
        assert_eq!(
            def_idx,
            importer_idx + 1,
            "the referenced file must be appended after the base hits without evicting any of \
             them; with a single base hit here that lands directly after it, got: {hits:?}"
        );
        assert!(def_idx < 10, "the injected file must land inside the top K, got: {hits:?}");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn search_xref_returns_base_hits_unchanged_with_no_xref_store() {
        let dir = std::env::temp_dir().join(format!("lxrf-empty-store-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("tools");
        let xref_store_root = dir.join("tools-xref");

        let a = write_file(&dir, "alpha.py", "def uniquealpha():\n    pass\n");
        index::append_batch(&store_root, &[a.clone()]).unwrap();

        let hits = search(&store_root, &xref_store_root, "uniquealpha", 10).unwrap();
        assert!(hits.iter().any(|(p, _)| p == &a.to_string_lossy()));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn append_batch_empty_paths_is_a_no_op() {
        let dir = std::env::temp_dir().join(format!("lxrf-empty-batch-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("xref");

        let result = append_batch(&store_root, &[]);
        assert!(result.is_ok());
        assert!(!manifest_path(&store_root).exists());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn append_batch_persists_across_calls() {
        let dir = std::env::temp_dir().join(format!("lxrf-incr-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("xref");

        let a = write_file(&dir, "one.py", "from two import marker\n");
        append_batch(&store_root, &[a.clone()]).unwrap();
        let b = write_file(&dir, "two.py", "def marker():\n    pass\n");
        append_batch(&store_root, &[b.clone()]).unwrap();

        let handles = load_manifest(&store_root).unwrap();
        assert_eq!(handles.len(), 2, "each batch writes one new segment");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn within_file_call_resolves_to_the_same_file_via_definitions_for_reference() {
        let dir = std::env::temp_dir().join(format!("lxrf-withinfile-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let xref_store_root = dir.join("xref");

        let only = write_file(
            &dir,
            "only.py",
            "def zebracorn_helper():\n    return 1\nresult = zebracorn_helper()\n",
        );
        append_batch(&xref_store_root, &[only.clone()]).unwrap();

        let pairs = definitions_for_reference(&xref_store_root, "zebracorn_helper").unwrap();
        assert_eq!(
            pairs,
            vec![(
                only.to_string_lossy().into_owned(),
                Some(only.to_string_lossy().into_owned()),
                XrefKind::Call,
            )],
            "a within-file call must resolve to its OWN file via an O(1) same-segment lookup, carrying Call kind"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn definitions_for_reference_carries_the_edge_kind_for_a_cross_file_import() {
        let dir = std::env::temp_dir().join(format!("lxrf-kind-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let xref_store_root = dir.join("xref");

        let def_file = write_file(&dir, "def_file.py", "def zebracornkindmarker():\n    return 1\n");
        let importer_file = write_file(
            &dir,
            "importer_file.py",
            "from def_file import zebracornkindmarker\n",
        );
        append_batch(&xref_store_root, &[def_file.clone(), importer_file.clone()]).unwrap();

        let rows = definitions_for_reference(&xref_store_root, "zebracornkindmarker").unwrap();
        assert_eq!(
            rows,
            vec![(
                importer_file.to_string_lossy().into_owned(),
                Some(def_file.to_string_lossy().into_owned()),
                XrefKind::Import,
            )],
            "the reverse lookup must carry the edge kind alongside from/to paths"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn definitions_for_reference_reaches_both_a_resolved_and_an_unresolved_candidate() {
        let dir = std::env::temp_dir().join(format!("lxrf-fromsym-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let xref_store_root = dir.join("xref");

        let def_file = write_file(&dir, "def_file.py", "def zebracorn_helper():\n    return 1\n");
        let importer_file = write_file(
            &dir,
            "importer_file.py",
            "from def_file import zebracorn_helper\nfrom ghost_module import ghost_name\n",
        );
        append_batch(&xref_store_root, &[def_file.clone(), importer_file.clone()]).unwrap();

        let resolved = definitions_for_reference(&xref_store_root, "zebracorn_helper").unwrap();
        assert_eq!(
            resolved,
            vec![(
                importer_file.to_string_lossy().into_owned(),
                Some(def_file.to_string_lossy().into_owned()),
                XrefKind::Import,
            )],
            "the from-symbol direction must reach the reference's resolved definition"
        );

        let unresolved = definitions_for_reference(&xref_store_root, "ghost_name").unwrap();
        assert_eq!(
            unresolved,
            vec![(importer_file.to_string_lossy().into_owned(), None, XrefKind::Unresolved)],
            "an unresolved reference must still surface, carrying None rather than vanishing"
        );

        let absent = definitions_for_reference(&xref_store_root, "nothing_ever_named_this").unwrap();
        assert!(absent.is_empty());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn call_sites_for_definition_surfaces_both_an_internal_and_an_external_reference() {
        let dir = std::env::temp_dir().join(format!("lxrf-tosym-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let xref_store_root = dir.join("xref");

        // def_file both defines AND calls the name (a within-file Call
        // edge back onto its own slot); importer_file reaches the same
        // name through a resolved cross-file Import edge.
        let def_file = write_file(
            &dir,
            "def_file.py",
            "def zebracorn_helper():\n    return 1\nresult = zebracorn_helper()\n",
        );
        let importer_file = write_file(
            &dir,
            "importer_file.py",
            "from def_file import zebracorn_helper\n",
        );
        append_batch(&xref_store_root, &[def_file.clone(), importer_file.clone()]).unwrap();

        let mut sites = call_sites_for_definition(
            &xref_store_root,
            &def_file.to_string_lossy(),
            "zebracorn_helper",
        )
        .unwrap();
        sites.sort();
        let mut expected = vec![
            def_file.to_string_lossy().into_owned(),
            importer_file.to_string_lossy().into_owned(),
        ];
        expected.sort();
        assert_eq!(
            sites, expected,
            "the to-symbol direction must surface every resolved edge whose target is this definition"
        );

        // A different (unrelated) file path must never be treated as the
        // definition's owner just because the name matches.
        let elsewhere = call_sites_for_definition(&xref_store_root, "/nonexistent/path.py", "zebracorn_helper")
            .unwrap();
        assert!(elsewhere.is_empty());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn append_does_not_rewrite_a_prior_segment_file() {
        let dir = std::env::temp_dir().join(format!("lxrf-notouch-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("xref");

        let a = write_file(&dir, "one.py", "from two import marker\n");
        append_batch(&store_root, &[a.clone()]).unwrap();

        let handles_after_first = load_manifest(&store_root).unwrap();
        assert_eq!(handles_after_first.len(), 1);
        let first_segment_path = store_root.join(&handles_after_first[0].file_name);
        let bytes_after_first = fs::read(&first_segment_path).unwrap();

        let b = write_file(&dir, "two.py", "def marker():\n    pass\n");
        append_batch(&store_root, &[b.clone()]).unwrap();

        let bytes_after_second = fs::read(&first_segment_path).unwrap();
        assert_eq!(
            bytes_after_first, bytes_after_second,
            "a later append must never re-read or rewrite an earlier segment's bytes"
        );

        let pairs = definitions_for_reference(&store_root, "marker").unwrap();
        assert!(
            pairs.iter().any(|(from, _, _)| from == &a.to_string_lossy()),
            "the bidirectional index must still reach a reference from the FIRST batch after a second append"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn common_ancestor_stops_at_the_first_diverging_component() {
        let a = Path::new("/corpus/pkg/sub");
        let b = Path::new("/corpus/pkg/other");
        assert_eq!(common_ancestor(a, b), PathBuf::from("/corpus/pkg"));
    }

    #[test]
    fn batch_corpus_root_is_the_common_parent_of_every_path_in_the_batch() {
        let paths = vec![
            PathBuf::from("/corpus/pkg/a.py"),
            PathBuf::from("/corpus/pkg/sub/b.py"),
        ];
        assert_eq!(batch_corpus_root(&paths), PathBuf::from("/corpus/pkg"));
    }

    #[test]
    fn search_xref_appends_a_relatively_resolved_import_after_base_hits_without_eviction() {
        let dir = std::env::temp_dir().join(format!("lxrf-relative-inject-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("tools");
        let xref_store_root = dir.join("tools-xref");
        let pkg = dir.join("pkg");
        fs::create_dir_all(&pkg).unwrap();

        let def_file = write_file(&pkg, "def_file.py", "def zebracorn_helper():\n    return 1\n");
        let importer_file = write_file(
            &pkg,
            "importer_file.py",
            "from .def_file import zebracorn_helper\nresult = zebracorn_helper()\nprint('nautilus copper harbor signal marker nine')\n",
        );

        index::append_batch(&store_root, &[def_file.clone(), importer_file.clone()]).unwrap();
        append_batch(&xref_store_root, &[def_file.clone(), importer_file.clone()]).unwrap();

        let hits = search(
            &store_root,
            &xref_store_root,
            "nautilus copper harbor signal marker nine",
            10,
        )
        .unwrap();

        let importer_idx = hits
            .iter()
            .position(|(p, _)| p == &importer_file.to_string_lossy())
            .expect("expected the referencing file to be a base BM25 hit");
        let def_idx = hits
            .iter()
            .position(|(p, _)| p == &def_file.to_string_lossy())
            .expect("expected the relatively-imported file to be injected into the results");
        assert_eq!(
            def_idx,
            importer_idx + 1,
            "the relatively-resolved file must be appended after the base hits without evicting \
             any of them; with a single base hit here that lands directly after it, got: {hits:?}"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reindexing_unchanged_xref_file_is_a_no_op() {
        let dir = std::env::temp_dir().join(format!("lxrf-noop-reindex-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("xref");

        let only = write_file(
            &dir,
            "only.py",
            "def zebracornuniquenoop():\n    return 1\nresult = zebracornuniquenoop()\n",
        );
        append_batch(&store_root, &[only.clone()]).unwrap();

        let manifest_bytes_before = fs::read(manifest_path(&store_root)).unwrap();
        let segments_before = load_manifest(&store_root).unwrap().len();
        let hits_before = definitions_for_reference(&store_root, "zebracornuniquenoop").unwrap();
        assert_eq!(hits_before.len(), 1);

        crate::xref_segment::reset_segment_load_count();
        append_batch(&store_root, &[only.clone()]).unwrap();
        assert_eq!(
            crate::xref_segment::segment_load_count(),
            0,
            "a no-op re-index of an unchanged xref file must never open a prior segment file"
        );

        let manifest_bytes_after = fs::read(manifest_path(&store_root)).unwrap();
        let segments_after = load_manifest(&store_root).unwrap().len();
        let hits_after = definitions_for_reference(&store_root, "zebracornuniquenoop").unwrap();

        assert_eq!(
            manifest_bytes_before, manifest_bytes_after,
            "re-indexing an unchanged xref file must leave the xref manifest byte-identical"
        );
        assert_eq!(
            segments_before, segments_after,
            "re-indexing an unchanged xref file must add zero net new segments"
        );
        assert_eq!(
            hits_after.len(),
            1,
            "re-indexing an unchanged xref file must not double the edge result count, got: {hits_after:?}"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn changed_xref_file_replaces_prior_edges() {
        let dir = std::env::temp_dir().join(format!("lxrf-replace-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("xref");

        let def_old = write_file(&dir, "def_old.py", "def zebracornoldtarget():\n    return 1\n");
        // Import-only, no invocation -- an invocation would also scan as a
        // Call candidate against importer's OWN (empty) def index, adding
        // an unrelated Unresolved edge under the same name and making
        // `definitions_for_reference` return two rows instead of one.
        let importer = write_file(
            &dir,
            "importer.py",
            "from def_old import zebracornoldtarget\n",
        );
        append_batch(&store_root, &[def_old.clone(), importer.clone()]).unwrap();

        let old_before = definitions_for_reference(&store_root, "zebracornoldtarget").unwrap();
        assert_eq!(
            old_before,
            vec![(
                importer.to_string_lossy().into_owned(),
                Some(def_old.to_string_lossy().into_owned()),
                XrefKind::Import,
            )],
            "sanity: the old target must resolve before the file changes"
        );

        // The importer's content changes to reference a brand-new target
        // file (never previously indexed) -- avoids the pre-existing
        // same-batch-only resolution limitation that would apply if the
        // new target were an UNCHANGED sibling re-included in this batch.
        let def_new = write_file(&dir, "def_new.py", "def zebracornnewtarget():\n    return 2\n");
        write_file(
            &dir,
            "importer.py",
            "from def_new import zebracornnewtarget\n",
        );
        append_batch(&store_root, &[def_old.clone(), def_new.clone(), importer.clone()]).unwrap();

        let old_after = definitions_for_reference(&store_root, "zebracornoldtarget").unwrap();
        assert!(
            old_after.is_empty(),
            "the old edge target must not be findable after the file changed, got: {old_after:?}"
        );
        let new_after = definitions_for_reference(&store_root, "zebracornnewtarget").unwrap();
        assert_eq!(
            new_after,
            vec![(
                importer.to_string_lossy().into_owned(),
                Some(def_new.to_string_lossy().into_owned()),
                XrefKind::Import,
            )],
            "the new edge target must resolve to a single current edge set for the file, got: {new_after:?}"
        );

        fs::remove_dir_all(&dir).ok();
    }
}
