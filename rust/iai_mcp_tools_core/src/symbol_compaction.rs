//! Fixed-threshold symbol-lane compaction: merges the active symbol
//! segment set's sources into ONE segment when the active count crosses a
//! locked threshold. Mirrors compaction.rs's shape by copy -- own segment
//! file naming, own threshold, never imports the whole-file lane's types.
//! `symbol_index.rs` publishes the merge and retires its sources in the
//! same `SegmentSetBuffer::compact` swap this module hands it a handle
//! for -- this module only builds and durably writes the merged segment.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::postings::{rebuild_postings_csr, PostingsCsr};
use crate::segmentset::{SegmentHandle, SegmentSet};
use crate::symbol_scan::SymbolKind;
use crate::symbol_segment::{ExactEntry, SymbolFileMeta, SymbolSegment, SymbolSegmentError};
use crate::termstats::TermStats;

/// v1: a plain, unexposed constant -- not a config knob. Same value as the
/// whole-file lane's own threshold, kept as an independent constant so the
/// two lanes never share a compile-time dependency on each other's tuning.
pub(crate) const COMPACTION_THRESHOLD: usize = 16;

pub(crate) fn should_compact(active_segment_count: usize) -> bool {
    active_segment_count > COMPACTION_THRESHOLD
}

fn segment_file_name(id: u32) -> String {
    format!("symbol-{id}.lsidx")
}

/// Reconstructs each document's token -> tf map from an already-built
/// CSR -- O(total postings), needs no re-tokenize/re-scan of any source
/// file.
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

/// One candidate document pulled from an active symbol segment, prior to
/// the tombstone-exclude and path-dedup passes below.
struct Candidate {
    segment_id: u32,
    meta: SymbolFileMeta,
    freq: HashMap<u32, u32>,
    exact: Vec<(u32, SymbolKind)>,
}

/// Reads every active symbol segment fully (this is compaction's own,
/// allowed read -- distinct from an append, which must never do this),
/// excludes tombstoned (segment_id, file_id) docs, keeps at most ONE doc
/// per path (highest segment_id wins -- also heals any pre-existing
/// duplicate-path pollution), builds ONE merged segment over the
/// survivors -- re-slotting each kept file's `exact_entries` by the
/// merged file offset so every entry still points at its own owning file
/// -- and writes it durably (tmp-write + sync_all + rename) BEFORE it is
/// ever referenced by the active set. Returns a reconciled `TermStats`
/// and a rebuilt `file_index` computed fresh over the merged postings --
/// a merge that drops docs must never carry forward stale per-file
/// bookkeeping. All-or-nothing: if any active segment fails to load, the
/// whole merge aborts rather than reconciling over a subset, since a
/// wholesale file_index/term_stats replace over only the loaded segments
/// would silently orphan the bookkeeping for one that stays active. Does
/// not touch the manifest itself -- the caller performs the (merged IN,
/// sources OUT) swap as one step.
pub(crate) fn compact(
    store_root: &Path,
    set: &SegmentSet,
) -> Result<(SegmentHandle, HashSet<u32>, TermStats, HashMap<String, (u32, u32, u64)>), SymbolSegmentError> {
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
        match SymbolSegment::load(&store_root.join(&handle.file_name)) {
            Ok(seg) => {
                let freqs = csr_to_doc_freqs(&seg.postings);
                let mut exact_by_slot: HashMap<u32, Vec<(u32, SymbolKind)>> = HashMap::new();
                for e in &seg.exact_entries {
                    exact_by_slot
                        .entry(e.doc_slot)
                        .or_default()
                        .push((e.feature_id, e.kind));
                }
                for (slot, (meta, freq)) in seg.files.into_iter().zip(freqs).enumerate() {
                    if set.tombstones.contains(&(handle.id, slot as u32)) {
                        continue; // superseded -- excluded, permanently reclaimed here
                    }
                    let exact = exact_by_slot.remove(&(slot as u32)).unwrap_or_default();
                    candidates.push(Candidate {
                        segment_id: handle.id,
                        meta,
                        freq,
                        exact,
                    });
                }
                source_ids.insert(handle.id);
            }
            Err(err) => {
                eprintln!(
                    "iai_mcp_tools_core: skipping unloadable symbol segment {} during compaction: {err}",
                    handle.file_name
                );
            }
        }
    }

    if source_ids.len() != set.segments.len() {
        return Err(SymbolSegmentError::Corrupt(
            "symbol compaction requires every active segment to load -- a partial merge would \
             drop file_index/term_stats bookkeeping for a segment that stays active"
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

    let mut all_files: Vec<SymbolFileMeta> = Vec::new();
    let mut all_freqs: Vec<HashMap<u32, u32>> = Vec::new();
    let mut all_exact: Vec<ExactEntry> = Vec::new();
    for (i, c) in candidates.into_iter().enumerate() {
        if kept.contains(&i) {
            let slot = all_files.len() as u32;
            for (feature_id, kind) in c.exact {
                all_exact.push(ExactEntry {
                    feature_id,
                    doc_slot: slot,
                    kind,
                });
            }
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

    // Deterministic order, matching `SymbolSegment::build`'s own sort, so
    // a merged segment's save()/load() round-trip is byte-stable.
    all_exact.sort_by_key(|e| (e.feature_id, e.doc_slot));
    let merged = SymbolSegment {
        files: all_files,
        postings: merged_postings,
        exact_entries: all_exact,
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
    use crate::symbol_index;

    fn write_file(dir: &Path, name: &str, content: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn compaction_fires_after_threshold_and_drops_active_segment_count() {
        let dir = std::env::temp_dir().join(format!("lsym-compact-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("symbols");

        for i in 0..(COMPACTION_THRESHOLD + 1) {
            let f = write_file(&dir, &format!("f{i}.py"), &format!("def file_marker_{i}_def():\n    pass\n"));
            symbol_index::append_batch(&store_root, &[f]).unwrap();
        }

        let set = symbol_index::load_manifest(&store_root).unwrap();
        assert!(
            set.segments.len() < COMPACTION_THRESHOLD + 1,
            "active segment count must drop once the threshold is crossed, got {}",
            set.segments.len()
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compaction_search_after_merge_has_no_duplicated_or_missing_files() {
        let dir = std::env::temp_dir().join(format!("lsym-compact-search-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("symbols");

        let mut expected = Vec::new();
        for i in 0..(COMPACTION_THRESHOLD + 1) {
            let f = write_file(&dir, &format!("f{i}.py"), &format!("def file_marker_{i}_def():\n    pass\n"));
            expected.push(f.to_string_lossy().into_owned());
            symbol_index::append_batch(&store_root, &[f]).unwrap();
        }

        for (i, path) in expected.iter().enumerate() {
            let hits = symbol_index::search(&store_root, &format!("file_marker_{i}_def"), 10).unwrap();
            let matches: Vec<&(String, f64)> = hits.iter().filter(|(p, _)| p == path).collect();
            assert_eq!(
                matches.len(),
                1,
                "each merged file must appear exactly once for its own defining name, got {matches:?}"
            );
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn corrupt_symbol_segment_does_not_wedge_compaction_or_future_appends() {
        let dir = std::env::temp_dir().join(format!("lsym-compact-corrupt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("symbols");

        for i in 0..COMPACTION_THRESHOLD {
            let f = write_file(&dir, &format!("f{i}.py"), &format!("def file_marker_{i}_def():\n    pass\n"));
            symbol_index::append_batch(&store_root, &[f]).unwrap();
        }

        // Corrupt one active segment (id 1) so `SymbolSegment::load` fails
        // on it.
        let corrupt_path = store_root.join(segment_file_name(1));
        let mut raw = std::fs::read(&corrupt_path).unwrap();
        let last = raw.len() - 1;
        raw[last] ^= 0xFF;
        std::fs::write(&corrupt_path, &raw).unwrap();

        // This append crosses the compaction threshold, triggering a
        // compaction attempt that hits the corrupt segment.
        let extra = write_file(&dir, "extra.py", "def extra_marker_def():\n    pass\n");
        let result = symbol_index::append_batch(&store_root, &[extra.clone()]);
        assert!(
            result.is_ok(),
            "append must succeed even though compaction encounters a corrupt symbol segment"
        );

        // The store must still serve searches -- not wedged by the bad
        // segment reached during compaction.
        assert!(symbol_index::search(&store_root, "extra_marker_def", 10).is_ok());

        // A further append that again crosses the threshold must also
        // succeed -- the corrupt segment never permanently blocks appends.
        let extra2 = write_file(&dir, "extra2.py", "def extra_marker_2_def():\n    pass\n");
        assert!(symbol_index::append_batch(&store_root, &[extra2]).is_ok());

        std::fs::remove_dir_all(&dir).ok();
    }
}
