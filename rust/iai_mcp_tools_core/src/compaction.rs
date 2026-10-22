//! Fixed-threshold compaction: merges the active segment set's sources
//! into ONE segment when the active count crosses a locked threshold.
//! `index.rs` publishes the merge and retires its sources in the same
//! `SegmentSetBuffer::compact` swap this module hands it a handle for --
//! this module only builds and durably writes the merged segment.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::postings::{rebuild_postings_csr, PostingsCsr};
use crate::segment::{FileMeta, Segment, SegmentError};
use crate::segmentset::{SegmentHandle, SegmentSet};
use crate::termstats::TermStats;

/// v1: a plain, unexposed constant -- not a config knob. Revisit only if
/// a real acceptance run shows compaction firing pathologically.
pub(crate) const COMPACTION_THRESHOLD: usize = 16;

pub(crate) fn should_compact(active_segment_count: usize) -> bool {
    active_segment_count > COMPACTION_THRESHOLD
}

fn segment_file_name(id: u32) -> String {
    format!("segment-{id}.lcidx")
}

/// Reconstructs each document's token -> tf map from an already-built
/// CSR -- O(total postings), needs no re-tokenize of any source file.
fn csr_to_doc_freqs(postings: &PostingsCsr) -> Vec<HashMap<u32, u32>> {
    let mut freqs: Vec<HashMap<u32, u32>> = vec![HashMap::new(); postings.n_docs as usize];
    for (&feature_id, &compact_id) in &postings.token_lookup {
        let start = postings.post_offsets[compact_id as usize] as usize;
        let end = postings.post_offsets[compact_id as usize + 1] as usize;
        for i in start..end {
            let slot = postings.post_doc_slots[i] as usize;
            freqs[slot].insert(feature_id, postings.post_tfs[i]);
        }
    }
    freqs
}

/// One candidate document pulled from an active segment, prior to the
/// tombstone-exclude and path-dedup passes below.
struct Candidate {
    segment_id: u32,
    meta: FileMeta,
    freq: HashMap<u32, u32>,
}

/// Reads every active segment fully (this is compaction's own, allowed
/// read -- distinct from an append, which must never do this), excludes
/// tombstoned (segment_id, file_id) docs, keeps at most ONE doc per path
/// (highest segment_id wins -- also heals any pre-existing duplicate-path
/// pollution), builds ONE merged segment over the survivors, and writes
/// it durably (tmp-write + sync_all + rename) BEFORE it is ever
/// referenced by the active set. Returns a reconciled `TermStats` and a
/// rebuilt `file_index` computed fresh over the merged postings -- a
/// merge that drops docs must never carry forward stale per-file
/// bookkeeping. All-or-nothing: if any active segment fails to load, the
/// whole merge aborts rather than reconciling over a subset, since a
/// wholesale file_index/term_stats replace over only the loaded segments
/// would silently orphan the bookkeeping for one that stays active. Does
/// not touch the manifest itself -- the caller performs the (merged IN,
/// sources OUT) swap as one step.
pub(crate) fn compact(
    store_root: &Path,
    set: &SegmentSet,
) -> Result<(SegmentHandle, HashSet<u32>, TermStats, HashMap<String, (u32, u32, u64)>), SegmentError> {
    let mut candidates: Vec<Candidate> = Vec::new();
    let mut source_ids: HashSet<u32> = HashSet::new();

    for handle in &set.segments {
        // Every active segment must load for this merge to proceed --
        // checked below. A partial merge would wholesale-replace
        // `file_index`/`term_stats` from only the segments that loaded,
        // silently dropping the bookkeeping for a segment that stays
        // active untouched -- reopening the dup-posting/stale-BM25 bug
        // this lane exists to close. The caller already treats a
        // compaction `Err` as "skip this round, retry next append".
        match Segment::load(&store_root.join(&handle.file_name)) {
            Ok(seg) => {
                let freqs = csr_to_doc_freqs(&seg.postings);
                for (slot, (meta, freq)) in seg.files.into_iter().zip(freqs).enumerate() {
                    if set.tombstones.contains(&(handle.id, slot as u32)) {
                        continue; // superseded -- excluded, permanently reclaimed here
                    }
                    candidates.push(Candidate {
                        segment_id: handle.id,
                        meta,
                        freq,
                    });
                }
                source_ids.insert(handle.id);
            }
            Err(err) => {
                eprintln!(
                    "iai_mcp_tools_core: skipping unloadable segment {} during compaction: {err}",
                    handle.file_name
                );
            }
        }
    }

    if source_ids.len() != set.segments.len() {
        return Err(SegmentError::Corrupt(
            "compaction requires every active segment to load -- a partial merge would drop \
             file_index/term_stats bookkeeping for a segment that stays active"
                .into(),
        ));
    }

    // At most one doc per path -- prefer the highest segment_id (most
    // recent).
    let mut best_by_path: HashMap<String, usize> = HashMap::new();
    for (i, c) in candidates.iter().enumerate() {
        match best_by_path.get(&c.meta.path) {
            Some(&existing) if candidates[existing].segment_id >= c.segment_id => {}
            _ => {
                best_by_path.insert(c.meta.path.clone(), i);
            }
        }
    }
    let kept: HashSet<usize> = best_by_path.into_values().collect();

    let mut all_files: Vec<FileMeta> = Vec::new();
    let mut all_freqs: Vec<HashMap<u32, u32>> = Vec::new();
    for (i, c) in candidates.into_iter().enumerate() {
        if kept.contains(&i) {
            all_files.push(c.meta);
            all_freqs.push(c.freq);
        }
    }

    let merged_postings = rebuild_postings_csr(all_files.len(), &all_freqs);

    let mut reconciled_stats = TermStats::default();
    reconciled_stats.add_segment(&merged_postings);

    let id = set.segments.iter().map(|h| h.id).max().unwrap_or(0) + 1;
    let mut file_index: HashMap<String, (u32, u32, u64)> = HashMap::new();
    for (slot, f) in all_files.iter().enumerate() {
        file_index.insert(f.path.clone(), (id, slot as u32, f.content_hash));
    }

    let merged = Segment {
        files: all_files,
        postings: merged_postings,
    };
    let file_name = segment_file_name(id);
    merged.save(&store_root.join(&file_name))?;

    Ok((
        SegmentHandle { id, file_name },
        source_ids,
        reconciled_stats,
        file_index,
    ))
}

/// Deletes the retired source segment files -- called ONLY after the
/// manifest swap that drops them is durable on disk. A crash before this
/// runs leaves harmless orphan files, never a correctness fault.
pub(crate) fn delete_sources(
    store_root: &Path,
    source_ids: &HashSet<u32>,
    prior_segments: &[SegmentHandle],
) {
    for handle in prior_segments.iter().filter(|h| source_ids.contains(&h.id)) {
        let _ = std::fs::remove_file(store_root.join(&handle.file_name));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index;
    use crate::tokenize::feature_id;

    fn write_file(dir: &Path, name: &str, content: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn compaction_fires_after_threshold_and_drops_active_segment_count() {
        let dir = std::env::temp_dir().join(format!("codeidx-compact-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("store");

        for i in 0..(COMPACTION_THRESHOLD + 1) {
            let f = write_file(&dir, &format!("f{i}.txt"), &format!("shared_token file_marker_{i}"));
            index::append_batch(&store_root, &[f]).unwrap();
        }

        let set = index::load_manifest(&store_root).unwrap();
        assert!(
            set.segments.len() < COMPACTION_THRESHOLD + 1,
            "active segment count must drop once the threshold is crossed, got {}",
            set.segments.len()
        );
        assert_eq!(
            set.term_stats.df(feature_id("shared_token")),
            (COMPACTION_THRESHOLD + 1) as u32,
            "a merge must not double-count the already-folded df"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compaction_search_after_merge_has_no_duplicated_or_missing_files() {
        let dir =
            std::env::temp_dir().join(format!("codeidx-compact-search-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("store");

        let mut expected = Vec::new();
        for i in 0..(COMPACTION_THRESHOLD + 1) {
            let f = write_file(&dir, &format!("f{i}.txt"), &format!("shared_token file_marker_{i}"));
            expected.push(f.to_string_lossy().into_owned());
            index::append_batch(&store_root, &[f]).unwrap();
        }

        let hits = index::search(&store_root, "shared_token", 100).unwrap();
        let mut hit_paths: Vec<String> = hits.into_iter().map(|(p, _)| p).collect();
        hit_paths.sort();
        let mut expected_sorted = expected.clone();
        expected_sorted.sort();
        assert_eq!(
            hit_paths, expected_sorted,
            "every merged file must appear exactly once -- no duplicates, none missing"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Regression gate for a partial-load compaction: when compaction merges
    /// only SOME active segments (one fails to load) it must never wholesale-
    /// replace `file_index`/`term_stats` -- doing so silently drops the
    /// unloaded-but-still-active segment's bookkeeping, so a later re-index
    /// of its own file treats that path as brand new (no tombstone) and, once
    /// the segment becomes readable again, produces a duplicate live posting.
    #[test]
    fn partial_compaction_failure_never_orphans_the_unloaded_segments_file_index_entry() {
        let dir = std::env::temp_dir().join(format!("codeidx-partial-compact-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("store");

        // f0 lands in segment id 1 -- the one this test corrupts.
        let mut paths = Vec::new();
        for i in 0..COMPACTION_THRESHOLD {
            let f = write_file(&dir, &format!("f{i}.txt"), &format!("shared_token file_marker_{i}"));
            index::append_batch(&store_root, &[f.clone()]).unwrap();
            paths.push(f);
        }
        let target = paths[0].clone();

        let corrupt_path = store_root.join(segment_file_name(1));
        let original_bytes = std::fs::read(&corrupt_path).unwrap();
        let mut raw = original_bytes.clone();
        let last = raw.len() - 1;
        raw[last] ^= 0xFF;
        std::fs::write(&corrupt_path, &raw).unwrap();

        // Crosses the threshold -- triggers a compaction attempt that hits
        // the corrupt segment 1 alongside every other (loadable) segment.
        let extra = write_file(&dir, "extra.txt", "shared_token extra_marker");
        index::append_batch(&store_root, &[extra]).unwrap();

        // Re-index target's OWN unchanged content. A partial-load
        // compaction that dropped target's file_index entry would treat
        // this as brand new and append a fresh, un-tombstoned posting.
        index::append_batch(&store_root, &[target.clone()]).unwrap();

        // Repair the corrupted segment in place -- simulates the transient
        // read failure healing, exactly the scenario the existing
        // `corrupt_segment_does_not_wedge_*` test already sets up as
        // recoverable.
        std::fs::write(&corrupt_path, &original_bytes).unwrap();

        let hits = index::search(&store_root, "file_marker_0", 100).unwrap();
        let target_str = target.to_string_lossy().into_owned();
        let matches: Vec<&(String, f64)> = hits.iter().filter(|(p, _)| p == &target_str).collect();
        assert_eq!(
            matches.len(),
            1,
            "target's own path must never appear more than once after a partial-load compaction \
             and a since-healed segment, got: {hits:?}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn corrupt_segment_does_not_wedge_compaction_or_future_appends() {
        let dir = std::env::temp_dir().join(format!("codeidx-compact-corrupt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("store");

        for i in 0..COMPACTION_THRESHOLD {
            let f = write_file(&dir, &format!("f{i}.txt"), &format!("shared_token file_marker_{i}"));
            index::append_batch(&store_root, &[f]).unwrap();
        }

        // Corrupt one active segment (id 1) so `Segment::load` fails on it.
        let corrupt_path = store_root.join(segment_file_name(1));
        let mut raw = std::fs::read(&corrupt_path).unwrap();
        let last = raw.len() - 1;
        raw[last] ^= 0xFF;
        std::fs::write(&corrupt_path, &raw).unwrap();

        // This append crosses the compaction threshold, triggering a
        // compaction attempt that hits the corrupt segment.
        let extra = write_file(&dir, "extra.txt", "shared_token extra_marker");
        let result = index::append_batch(&store_root, &[extra.clone()]);
        assert!(
            result.is_ok(),
            "append must succeed even though compaction encounters a corrupt segment"
        );

        // The store must still serve searches -- not wedged by the bad
        // segment reached during compaction.
        assert!(index::search(&store_root, "shared_token", 100).is_ok());

        // A further append that again crosses the threshold must also
        // succeed -- the corrupt segment never permanently blocks appends.
        let extra2 = write_file(&dir, "extra2.txt", "shared_token extra_marker_2");
        assert!(index::append_batch(&store_root, &[extra2]).is_ok());

        std::fs::remove_dir_all(&dir).ok();
    }
}
