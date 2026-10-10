//! Incremental index orchestration: an on-disk manifest (segment handles
//! + resident term-stats + generation) persisted alongside segment files
//! so a batch call resumes from disk state without opening any segment
//! file that is not part of THIS batch (or, during compaction, the
//! batch's own merge sources). The manifest is metadata, never a
//! "segment" itself -- reading it does not count as re-reading a prior
//! segment.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use thiserror::Error;

use crate::bm25::bm25_score_global;
use crate::compaction;
use crate::segment::{Segment, SegmentError};
use crate::segmentset::{SegmentHandle, SegmentSet, SegmentSetBuffer};
use crate::termstats::TermStats;
use crate::tokenize::{feature_id, tokenize};

const MANIFEST_MAGIC_V1: &[u8; 8] = b"LCMANFT1";
const MANIFEST_MAGIC_V2: &[u8; 8] = b"LCMANFT2";
const MANIFEST_FILE: &str = "manifest.lcman";

#[derive(Debug, Error)]
pub enum IndexError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("manifest corrupt: {0}")]
    Corrupt(String),
    #[error(transparent)]
    Segment(#[from] SegmentError),
}

fn manifest_path(store_root: &Path) -> PathBuf {
    store_root.join(MANIFEST_FILE)
}

// Test-only per-thread call counter -- thread-local (not a shared static)
// because libtest runs each `#[test]` fn to completion on its own thread,
// so a thread-local proves THIS test's own call count is exactly one
// without going flaky under unrelated tests calling `load_manifest`
// concurrently on other threads.
#[cfg(test)]
thread_local! {
    pub(crate) static LOAD_MANIFEST_CALLS: std::cell::Cell<usize> = std::cell::Cell::new(0);
}

/// Loads the on-disk manifest -- segment handles, generation, and the
/// resident term-stats table. Reading this is metadata I/O, never a
/// "prior segment" read: it never opens a `segment-*.lcidx` file.
/// Version-aware: an old-magic (v1) manifest parses with an empty
/// file-index and empty tombstones (safe degrade, no forced rebuild); a
/// new-magic (v2) manifest additionally carries both sections.
pub(crate) fn load_manifest(store_root: &Path) -> Result<SegmentSet, IndexError> {
    #[cfg(test)]
    LOAD_MANIFEST_CALLS.with(|c| c.set(c.get() + 1));
    let path = manifest_path(store_root);
    if !path.exists() {
        return Ok(SegmentSet::empty());
    }
    let raw = fs::read(&path)?;
    if raw.len() < 12 {
        return Err(IndexError::Corrupt("not a manifest file".into()));
    }
    let magic: [u8; 8] = raw[..8].try_into().unwrap();
    let is_v2 = if magic == *MANIFEST_MAGIC_V2 {
        true
    } else if magic == *MANIFEST_MAGIC_V1 {
        false
    } else {
        return Err(IndexError::Corrupt("not a manifest file".into()));
    };
    let stored_crc = u32::from_le_bytes(raw[8..12].try_into().unwrap());
    let payload = &raw[12..];
    if crc32fast::hash(payload) != stored_crc {
        return Err(IndexError::Corrupt("manifest CRC mismatch".into()));
    }

    let need = |off: usize, len: usize| -> Result<(), IndexError> {
        if off + len > payload.len() {
            Err(IndexError::Corrupt("truncated manifest".into()))
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
    // Reject before allocating -- entries are variable-length (id u32 +
    // name_len u16 + name bytes), so this is a cheap lower bound (6 bytes
    // minimum per entry) rather than an exact size, but it still catches
    // the pathological declared-count case before `Vec::with_capacity`.
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
            .map_err(|_| IndexError::Corrupt("non-utf8 segment name".into()))?
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
    // Fixed-width entries (fid u32 + count u32) -- exact bound.
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
                .map_err(|_| IndexError::Corrupt("non-utf8 file-index path".into()))?
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
        return Err(IndexError::Corrupt(
            "trailing bytes after manifest payload".into(),
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

/// Returns `(file_index.len(), generation)` for the manifest under
/// `store_root`, read off a SINGLE `load_manifest` binding -- never two
/// independent disk reads, which could pair an N from one generation
/// with a G from another under a concurrent append/compaction. An
/// absent manifest reports `(0, 0)` (`load_manifest` returns
/// `SegmentSet::empty()` in that case, never an error). No segment I/O,
/// no re-tokenize.
pub fn manifest_stats(store_root: &Path) -> Result<(u64, u64), IndexError> {
    let set = load_manifest(store_root)?;
    Ok((set.file_index.len() as u64, set.generation))
}

fn save_manifest(store_root: &Path, set: &SegmentSet) -> Result<(), IndexError> {
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(&set.generation.to_le_bytes());
    payload.extend_from_slice(&(set.segments.len() as u32).to_le_bytes());
    for h in &set.segments {
        payload.extend_from_slice(&h.id.to_le_bytes());
        let name_bytes = h.file_name.as_bytes();
        payload.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        payload.extend_from_slice(name_bytes);
    }

    // Deterministic order (sorted by feature_id) so a manifest re-saved
    // from an unchanged logical state is byte-stable.
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
    let tmp = path.with_extension("tmp-manifest");
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
    format!("segment-{id}.lcidx")
}

/// Indexes exactly the given paths as ONE new immutable segment and
/// publishes it -- never opens, reads, or re-tokenizes any file already
/// covered by a prior segment (the manifest carries the resident
/// term-stats table forward, so no prior segment content is ever
/// touched by an append). Triggers compaction afterward when the active
/// segment count has crossed the fixed threshold.
pub fn append_batch(store_root: &Path, paths: &[PathBuf]) -> Result<(), IndexError> {
    if paths.is_empty() {
        return Ok(());
    }

    fs::create_dir_all(store_root)?;
    let current = load_manifest(store_root)?;

    let (segment, tombstones) = Segment::build_incremental(paths, &current.file_index)?;
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

    if compaction::should_compact(published.segments.len()) {
        let prior_segments = published.segments.clone();
        // The append above is already durable in the manifest -- a
        // compaction failure (e.g. one unloadable source segment) must
        // degrade to "compaction skipped this round", never fail the
        // append that already succeeded. Otherwise a single bad segment
        // wedges every future append past the threshold permanently.
        match compaction::compact(store_root, &published) {
            Ok((merged_handle, source_ids, reconciled_stats, rebuilt_file_index)) => {
                buf.compact_reconciled(&source_ids, merged_handle, reconciled_stats, rebuilt_file_index);
                let final_set = buf.snapshot();
                save_manifest(store_root, &final_set)?;
                compaction::delete_sources(store_root, &source_ids, &prior_segments);
            }
            Err(err) => {
                eprintln!(
                    "iai_mcp_tools_core: compaction skipped for this append ({err}); active segment count remains {}",
                    prior_segments.len()
                );
            }
        }
    }

    Ok(())
}

/// Loads every active segment and returns up to `k` (path, score) hits,
/// ranked by summed BM25 read through the resident global term-stat
/// table, highest score first with a deterministic (score DESC, path
/// ASC) tie-break -- path is content-derived and segment-layout-
/// independent, so two differently-batched indexes of the same corpus
/// agree by construction.
pub fn search(store_root: &Path, query: &str, k: usize) -> Result<Vec<(String, f64)>, IndexError> {
    let set = load_manifest(store_root)?;
    search_in(&set, store_root, query, k)
}

/// Same ranking as `search`, plus the `(n_files, generation)` pair read
/// off the SAME `load_manifest` binding the hits themselves came from --
/// callers that must report a scope for these exact hits (not a second,
/// independently-timed read) use this instead of `search` + `manifest_stats`.
pub fn search_with_scope(
    store_root: &Path,
    query: &str,
    k: usize,
) -> Result<(Vec<(String, f64)>, u64, u64), IndexError> {
    let set = load_manifest(store_root)?;
    let hits = search_in(&set, store_root, query, k)?;
    Ok((hits, set.file_index.len() as u64, set.generation))
}

fn search_in(
    set: &SegmentSet,
    store_root: &Path,
    query: &str,
    k: usize,
) -> Result<Vec<(String, f64)>, IndexError> {
    let mut seen: Vec<u32> = Vec::new();
    for tok in tokenize(query) {
        let fid = feature_id(&tok);
        if !seen.contains(&fid) {
            seen.push(fid);
        }
    }

    let mut ranked: Vec<(String, f64)> = Vec::new();
    for handle in &set.segments {
        // A single corrupt/unreadable segment must degrade the store, not
        // wedge every query -- skip it (observable via stderr) and keep
        // serving the segments that do load.
        let seg = match Segment::load(&store_root.join(&handle.file_name)) {
            Ok(seg) => seg,
            Err(err) => {
                eprintln!(
                    "iai_mcp_tools_core: skipping unloadable segment {} during search: {err}",
                    handle.file_name
                );
                continue;
            }
        };
        let mut scores: HashMap<u32, f64> = HashMap::new();
        for &fid in &seen {
            let compact_id = match seg.postings.token_lookup.get(&fid) {
                Some(&c) => c,
                None => continue,
            };
            let start = seg.postings.post_offsets[compact_id as usize] as usize;
            let end = seg.postings.post_offsets[compact_id as usize + 1] as usize;
            for i in start..end {
                let slot = seg.postings.post_doc_slots[i];
                let tf = seg.postings.post_tfs[i] as f64;
                let length = seg.postings.doc_len[slot as usize] as f64;
                let score = bm25_score_global(tf, fid, length, &set.term_stats);
                *scores.entry(slot).or_insert(0.0) += score;
            }
        }
        for (slot, score) in scores {
            if set.tombstones.contains(&(handle.id, slot)) {
                continue; // superseded by a newer posting -- must never surface
            }
            ranked.push((seg.files[slot as usize].path.clone(), score));
        }
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
    use crate::segment;

    fn write_file(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn incremental_append_never_rereads_prior_segment() {
        let dir = std::env::temp_dir().join(format!("codeidx-incr-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("store");

        let a = write_file(&dir, "alpha.txt", "uniqueword_alpha shared_word");
        segment::reset_segment_load_count();
        append_batch(&store_root, &[a.clone()]).unwrap();
        assert_eq!(
            segment::segment_load_count(),
            0,
            "the FIRST batch has no prior segment to read"
        );

        let b = write_file(&dir, "beta.txt", "uniqueword_beta shared_word");
        segment::reset_segment_load_count();
        append_batch(&store_root, &[b.clone()]).unwrap();
        assert_eq!(
            segment::segment_load_count(),
            0,
            "a second incremental append must never open a prior segment file"
        );

        let manifest = load_manifest(&store_root).unwrap();
        assert_eq!(
            manifest.segments.len(),
            2,
            "each batch writes exactly one new immutable segment"
        );

        let hits = search(&store_root, "uniqueword_alpha", 5).unwrap();
        assert!(hits.iter().any(|(p, _)| p == &a.to_string_lossy()));
        let hits = search(&store_root, "uniqueword_beta", 5).unwrap();
        assert!(hits.iter().any(|(p, _)| p == &b.to_string_lossy()));

        fs::remove_dir_all(&dir).ok();
    }

    /// Correctness GATE, not a note: indexes the SAME fixed corpus two
    /// ways -- arm A one batch, arm B more single-file batches than the
    /// locked compaction threshold (so >=1 merge fires by batch count
    /// alone) -- and asserts the ranked result lists are IDENTICAL, and
    /// that arm B's incrementally-maintained term-stat table matches a
    /// from-scratch recomputation over its own active segments.
    #[test]
    fn append_equals_rebuild_matches_single_shot_index_across_a_compaction() {
        let dir = std::env::temp_dir().join(format!("codeidx-a-eq-r-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();

        let n = compaction::COMPACTION_THRESHOLD + 4;
        let mut paths = Vec::with_capacity(n);
        for i in 0..n {
            let content = format!("shared_token rare_marker_{i} common_word_{}", i % 3);
            paths.push(write_file(&dir, &format!("file{i}.txt"), &content));
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

        let queries = [
            "shared_token",
            "rare_marker_0",
            "common_word_1",
            "rare_marker_0 shared_token",
        ];
        for q in queries {
            let hits_a = search(&root_a, q, 100).unwrap();
            let hits_b = search(&root_b, q, 100).unwrap();
            assert_eq!(
                hits_a, hits_b,
                "arm A (single-shot) and arm B (incremental+compacted) must rank identically for query {q:?}"
            );
        }

        // The incrementally-maintained global term-stat table must equal
        // a from-scratch recomputation scanned over arm B's own active
        // segments -- any cross-segment drift fails this gate.
        let mut recomputed = TermStats::default();
        for h in &set_b.segments {
            let seg = Segment::load(&root_b.join(&h.file_name)).unwrap();
            recomputed.add_segment(&seg.postings);
        }
        assert_eq!(
            recomputed.df, set_b.term_stats.df,
            "no cross-segment df drift after a merge"
        );
        assert_eq!(recomputed.n_docs, set_b.term_stats.n_docs);
        assert_eq!(recomputed.total_len, set_b.term_stats.total_len);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn append_batch_empty_paths_is_a_no_op() {
        let dir = std::env::temp_dir().join(format!("codeidx-empty-batch-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("store");

        let result = append_batch(&store_root, &[]);
        assert!(result.is_ok(), "an empty batch must be a clean no-op");
        assert!(
            !manifest_path(&store_root).exists(),
            "an empty batch must not persist a wasted manifest/segment"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn manifest_stats_matches_file_index_len_and_generation_after_append_and_reindex() {
        let dir = std::env::temp_dir().join(format!("codeidx-manifest-stats-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("store");

        let (n0, g0) = manifest_stats(&store_root).unwrap();
        assert_eq!(
            (n0, g0),
            (0, 0),
            "an absent manifest must report zero files and zero generation"
        );

        let a = write_file(&dir, "alpha.txt", "manifeststatsalpha shared_word");
        append_batch(&store_root, &[a.clone()]).unwrap();
        let manifest = load_manifest(&store_root).unwrap();
        let (n1, g1) = manifest_stats(&store_root).unwrap();
        assert_eq!(n1, manifest.file_index.len() as u64);
        assert_eq!(g1, manifest.generation);
        assert_eq!(n1, 1);

        // Re-index the same path with changed content -- a tombstoning
        // re-index must never inflate n_files.
        write_file(&dir, "alpha.txt", "manifeststatschanged shared_word");
        append_batch(&store_root, &[a.clone()]).unwrap();
        let manifest2 = load_manifest(&store_root).unwrap();
        let (n2, g2) = manifest_stats(&store_root).unwrap();
        assert_eq!(n2, manifest2.file_index.len() as u64);
        assert_eq!(g2, manifest2.generation);
        assert_eq!(
            n2, 1,
            "a tombstoning re-index of an already-indexed path must not inflate n_files"
        );
        assert!(g2 > g1, "a second publish must bump generation");

        fs::remove_dir_all(&dir).ok();
    }

    /// Non-vacuous single-load proof: under single-writer quiescence, a
    /// separate `search` + `manifest_stats` pair and a combined
    /// `search_with_scope` call return the SAME n_files/generation, so a
    /// value-equality assertion alone cannot tell the two call shapes
    /// apart. The `LOAD_MANIFEST_CALLS` counter can: `search_with_scope`
    /// must read the manifest exactly once, while `search` followed by
    /// `manifest_stats` reads it twice -- this is the pre-fix shape the
    /// counter would also have caught.
    #[test]
    fn search_with_scope_reads_the_manifest_exactly_once() {
        let dir = std::env::temp_dir().join(format!("codeidx-single-load-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("store");

        let a = write_file(&dir, "alpha.txt", "singleloadmarker shared_word");
        append_batch(&store_root, &[a]).unwrap();

        LOAD_MANIFEST_CALLS.with(|c| c.set(0));
        let (hits, n_files, generation) = search_with_scope(&store_root, "singleloadmarker", 10).unwrap();
        assert_eq!(
            LOAD_MANIFEST_CALLS.with(|c| c.get()),
            1,
            "search_with_scope must read the manifest exactly once"
        );
        assert!(!hits.is_empty());
        assert_eq!(n_files, 1);
        assert!(generation >= 1);

        LOAD_MANIFEST_CALLS.with(|c| c.set(0));
        let _ = search(&store_root, "singleloadmarker", 10).unwrap();
        let _ = manifest_stats(&store_root).unwrap();
        assert_eq!(
            LOAD_MANIFEST_CALLS.with(|c| c.get()),
            2,
            "a separate search + manifest_stats pair reads the manifest twice -- the torn-read shape this fix replaces"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn manifest_v2_round_trips_file_index_and_tombstones() {
        let dir = std::env::temp_dir().join(format!("codeidx-manifest-v2-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("store");
        fs::create_dir_all(&store_root).unwrap();

        let mut file_index: HashMap<String, (u32, u32, u64)> = HashMap::new();
        file_index.insert("alpha.txt".to_string(), (1, 0, 111));
        file_index.insert("beta.txt".to_string(), (2, 0, 222));
        let mut tombstones: HashSet<(u32, u32)> = HashSet::new();
        tombstones.insert((1, 0));

        let set = SegmentSet {
            segments: vec![SegmentHandle {
                id: 2,
                file_name: "segment-2.lcidx".to_string(),
            }],
            tombstones: Arc::new(tombstones.clone()),
            term_stats: Arc::new(TermStats::default()),
            file_index: Arc::new(file_index.clone()),
            generation: 5,
        };

        save_manifest(&store_root, &set).unwrap();
        let loaded = load_manifest(&store_root).unwrap();

        assert_eq!(*loaded.file_index, file_index);
        assert_eq!(*loaded.tombstones, tombstones);
        assert_eq!(loaded.generation, 5);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn manifest_v1_magic_loads_with_empty_file_index_and_tombstones() {
        let dir = std::env::temp_dir().join(format!("codeidx-manifest-v1-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("store");
        fs::create_dir_all(&store_root).unwrap();

        // Hand-write a v1-shaped manifest -- generation, zero segments,
        // zero df entries, n_docs=0, total_len=0 -- with the OLD magic.
        let mut payload: Vec<u8> = Vec::new();
        payload.extend_from_slice(&7u64.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.extend_from_slice(&0u64.to_le_bytes());

        let crc = crc32fast::hash(&payload);
        let mut raw = Vec::new();
        raw.extend_from_slice(b"LCMANFT1");
        raw.extend_from_slice(&crc.to_le_bytes());
        raw.extend_from_slice(&payload);
        fs::write(manifest_path(&store_root), &raw).unwrap();

        let loaded = load_manifest(&store_root).unwrap();
        assert_eq!(loaded.generation, 7);
        assert!(loaded.file_index.is_empty());
        assert!(loaded.tombstones.is_empty());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reindexing_unchanged_file_is_a_no_op() {
        let dir = std::env::temp_dir().join(format!("codeidx-noop-reindex-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("store");

        let a = write_file(&dir, "alpha.txt", "uniquetokennoop sharedwordnoop");
        append_batch(&store_root, &[a.clone()]).unwrap();

        let manifest_bytes_before = fs::read(manifest_path(&store_root)).unwrap();
        let segments_before = load_manifest(&store_root).unwrap().segments.len();
        let hits_before = search(&store_root, "uniquetokennoop", 5).unwrap().len();
        assert_eq!(hits_before, 1);

        segment::reset_segment_load_count();
        append_batch(&store_root, &[a.clone()]).unwrap();
        assert_eq!(
            segment::segment_load_count(),
            0,
            "a no-op re-index of an unchanged file must never open a prior segment file"
        );

        let manifest_bytes_after = fs::read(manifest_path(&store_root)).unwrap();
        let segments_after = load_manifest(&store_root).unwrap().segments.len();
        let hits_after = search(&store_root, "uniquetokennoop", 5).unwrap().len();

        assert_eq!(
            manifest_bytes_before, manifest_bytes_after,
            "re-indexing an unchanged file must leave the manifest byte-identical"
        );
        assert_eq!(
            segments_before, segments_after,
            "re-indexing an unchanged file must add zero net new segments"
        );
        assert_eq!(
            hits_after, 1,
            "re-indexing an unchanged file must not double the hit count"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn changed_file_replaces_prior_postings() {
        let dir = std::env::temp_dir().join(format!("codeidx-replace-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("store");

        // No underscores in any marker -- tokenize()'s snake-case
        // decomposition must not cross-leak sub-token overlap between the
        // old-only and new-only markers.
        let a = write_file(&dir, "alpha.txt", "markeroldonly sharedtermcommon");
        append_batch(&store_root, &[a.clone()]).unwrap();

        write_file(&dir, "alpha.txt", "markernewonly sharedtermcommon");
        append_batch(&store_root, &[a.clone()]).unwrap();

        let old_hits = search(&store_root, "markeroldonly", 5).unwrap();
        assert!(
            old_hits.is_empty(),
            "the old-only marker must not be findable after the file changed"
        );

        let new_hits = search(&store_root, "markernewonly", 5).unwrap();
        assert!(
            new_hits.iter().any(|(p, _)| p == &a.to_string_lossy()),
            "the new-only marker must be findable after the file changed"
        );

        let shared_hits = search(&store_root, "sharedtermcommon", 5).unwrap();
        assert_eq!(
            shared_hits.iter().filter(|(p, _)| p == &a.to_string_lossy()).count(),
            1,
            "the shared term must return exactly one hit for the changed path, not one per version"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn search_skips_unloadable_segment_and_still_serves_healthy_ones() {
        let dir = std::env::temp_dir().join(format!("codeidx-corrupt-seg-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("store");

        // No shared substring/snake-case fragment between the two unique
        // tokens -- `tokenize` also emits snake_case sub-parts, so e.g.
        // "findme_alpha"/"findme_beta" would spuriously cross-match on
        // "findme" and defeat the emptiness assertion below.
        let a = write_file(&dir, "alpha.txt", "uniquetokenalpha shared_word");
        append_batch(&store_root, &[a.clone()]).unwrap();
        let b = write_file(&dir, "beta.txt", "uniquetokenbeta shared_word");
        append_batch(&store_root, &[b.clone()]).unwrap();

        // Corrupt the first-appended segment (id 1, alpha's) in place --
        // flips CRC mismatch, so `Segment::load` fails closed for it.
        let corrupt_path = store_root.join(segment_file_name(1));
        let mut raw = fs::read(&corrupt_path).unwrap();
        let last = raw.len() - 1;
        raw[last] ^= 0xFF;
        fs::write(&corrupt_path, &raw).unwrap();

        // The healthy segment's content must still be found -- a search
        // must not raise just because one OTHER segment is unloadable.
        let hits = search(&store_root, "uniquetokenbeta", 5).unwrap();
        assert!(
            hits.iter().any(|(p, _)| p == &b.to_string_lossy()),
            "the healthy segment must still serve results despite a sibling corrupt segment"
        );

        // The corrupted segment's own content is unreachable, but the
        // query must still return cleanly (empty), never error.
        let hits_alpha = search(&store_root, "uniquetokenalpha", 5).unwrap();
        assert!(hits_alpha.is_empty());

        // A store with one corrupt segment must still accept a new
        // append -- no permanent wedge.
        let c = write_file(&dir, "gamma.txt", "uniquetokengamma shared_word");
        let append_result = append_batch(&store_root, &[c.clone()]);
        assert!(
            append_result.is_ok(),
            "append must succeed even with an unloadable sibling segment present"
        );
        let hits_gamma = search(&store_root, "uniquetokengamma", 5).unwrap();
        assert!(hits_gamma.iter().any(|(p, _)| p == &c.to_string_lossy()));

        fs::remove_dir_all(&dir).ok();
    }
}
