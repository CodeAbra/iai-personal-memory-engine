//! Fixed-threshold xref-lane compaction: merges the active xref segment
//! set into ONE segment when the active count crosses a locked threshold.
//! Mirrors symbol_compaction.rs's shape by copy -- own segment file
//! naming, own threshold, never imports the symbol lane's types. Unlike
//! the symbol lane, the xref lane keeps no resident term-stats table and
//! takes its tombstone set as a parameter from the caller's own
//! already-loaded manifest snapshot (no `SegmentSetBuffer` instance), so
//! `xref_index.rs` performs the (merged IN, sources OUT) manifest swap
//! directly against the plain handle list this module hands it, rather
//! than through that generic double-buffer.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::xref_index::XrefHandle;
use crate::xref_scan::XrefKind;
use crate::xref_segment::{XrefEdge, XrefFileMeta, XrefSegment, XrefSegmentError};

/// v1: a plain, unexposed constant -- not a config knob. Same value as the
/// symbol lane's own threshold, kept as an independent constant so the
/// two lanes never share a compile-time dependency on each other's tuning.
pub(crate) const COMPACTION_THRESHOLD: usize = 16;

pub(crate) fn should_compact(active_segment_count: usize) -> bool {
    active_segment_count > COMPACTION_THRESHOLD
}

fn segment_file_name(id: u32) -> String {
    format!("xref-{id}.lxidx")
}

/// One kept file slot, prior to the merged-offset remap below.
struct Candidate {
    segment_id: u32,
    local_slot: u32,
    meta: XrefFileMeta,
}

/// Reads every active xref segment fully (this is compaction's own,
/// allowed read -- distinct from an append, which must never do this),
/// excludes tombstoned `(segment_id, file_id)` pairs, keeps at most ONE
/// file per path (highest segment_id wins -- also heals any pre-existing
/// duplicate-path pollution), and builds ONE merged segment over the
/// survivors. An edge's `from_slot` is dropped unless its own
/// `(segment_id, local_slot)` survived; a resolved `to_slot` is remapped
/// by the TARGET'S PATH (never a raw slot copy), since the specific file
/// instance it originally targeted may have been superseded by a
/// *different* segment's occurrence of the same path -- a resolution that
/// no longer finds its target degrades to `Unresolved` rather than
/// dangling. Writes the merged segment durably (tmp-write + sync_all +
/// rename) BEFORE it is ever referenced by the active set. Does not touch
/// the manifest -- the caller performs the (merged IN, sources OUT) swap
/// as one step, so a crash between this call returning and that swap
/// leaves the pre-merge manifest, and therefore the pre-merge store,
/// untouched and loadable. All-or-nothing: if any active segment fails to
/// load, the whole merge aborts rather than reconciling over a subset,
/// since a wholesale file_index replace over only the loaded segments
/// would silently orphan the bookkeeping for one that stays active.
/// `tombstones` is the caller's own already-loaded manifest snapshot --
/// never re-read from disk here, so this call's notion of "current
/// tombstones" is provably the same snapshot as `handles`.
pub(crate) fn compact(
    store_root: &Path,
    handles: &[XrefHandle],
    tombstones: &HashSet<(u32, u32)>,
) -> Result<(XrefHandle, HashSet<u32>), XrefSegmentError> {
    struct Loaded {
        segment_id: u32,
        seg: XrefSegment,
    }
    let mut loaded: Vec<Loaded> = Vec::new();
    let mut source_ids: HashSet<u32> = HashSet::new();

    for handle in handles {
        // Every active segment must load for this merge to proceed --
        // checked below. A partial merge would wholesale-replace
        // file_index from only the segments that loaded, silently
        // dropping the bookkeeping for a segment that stays active
        // untouched -- reopening the dup-posting bug this lane exists to
        // close. The caller already treats a compaction `Err` as "skip
        // this round, retry next append".
        match XrefSegment::load(&store_root.join(&handle.file_name)) {
            Ok(seg) => {
                loaded.push(Loaded {
                    segment_id: handle.id,
                    seg,
                });
                source_ids.insert(handle.id);
            }
            Err(err) => {
                eprintln!(
                    "iai_mcp_tools_core: skipping unloadable xref segment {} during compaction: {err}",
                    handle.file_name
                );
            }
        }
    }

    if source_ids.len() != handles.len() {
        return Err(XrefSegmentError::Corrupt(
            "xref compaction requires every active segment to load -- a partial merge would drop \
             file_index bookkeeping for a segment that stays active"
                .into(),
        ));
    }

    let mut candidates: Vec<Candidate> = Vec::new();
    for l in &loaded {
        for (local_slot, meta) in l.seg.files.iter().enumerate() {
            if tombstones.contains(&(l.segment_id, local_slot as u32)) {
                continue; // superseded -- excluded, permanently reclaimed here
            }
            candidates.push(Candidate {
                segment_id: l.segment_id,
                local_slot: local_slot as u32,
                meta: XrefFileMeta {
                    path: meta.path.clone(),
                    content_hash: meta.content_hash,
                    mtime: meta.mtime,
                    size: meta.size,
                },
            });
        }
    }

    // At most one file per path -- prefer the highest segment_id (most
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

    // Global (segment_id, local_slot) -> merged slot, and path -> merged
    // slot for the surviving files.
    let mut slot_map: HashMap<(u32, u32), u32> = HashMap::new();
    let mut path_to_slot: HashMap<String, u32> = HashMap::new();
    let mut all_files: Vec<XrefFileMeta> = Vec::new();
    for (i, c) in candidates.into_iter().enumerate() {
        if !kept.contains(&i) {
            continue;
        }
        let new_slot = all_files.len() as u32;
        slot_map.insert((c.segment_id, c.local_slot), new_slot);
        path_to_slot.insert(c.meta.path.clone(), new_slot);
        all_files.push(c.meta);
    }

    let mut all_edges: Vec<XrefEdge> = Vec::new();
    for l in &loaded {
        for e in &l.seg.edges {
            let Some(&new_from) = slot_map.get(&(l.segment_id, e.from_slot)) else {
                continue; // origin superseded -- the surviving instance's own scan carries this file's edges
            };
            let new_to = e.to_slot.and_then(|to_local| {
                let to_path = &l.seg.files[to_local as usize].path;
                path_to_slot.get(to_path).copied()
            });
            all_edges.push(XrefEdge {
                from_slot: new_from,
                to_slot: new_to,
                kind: if e.to_slot.is_some() && new_to.is_none() {
                    XrefKind::Unresolved
                } else {
                    e.kind
                },
                name: e.name.clone(),
            });
        }
    }

    let merged = XrefSegment {
        files: all_files,
        edges: all_edges,
    };

    let id = handles.iter().map(|h| h.id).max().unwrap_or(0) + 1;
    let file_name = segment_file_name(id);
    merged.save(&store_root.join(&file_name))?;

    Ok((XrefHandle { id, file_name }, source_ids))
}

/// Deletes the retired source segment files -- called ONLY after the
/// manifest swap that drops them is durable on disk. A crash before this
/// runs leaves harmless orphan files, never a correctness fault.
pub(crate) fn delete_sources(
    store_root: &Path,
    source_ids: &HashSet<u32>,
    prior_handles: &[XrefHandle],
) {
    for handle in prior_handles.iter().filter(|h| source_ids.contains(&h.id)) {
        let _ = std::fs::remove_file(store_root.join(&handle.file_name));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xref_index;

    fn write_file(dir: &Path, name: &str, content: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn compaction_fires_after_threshold_and_drops_active_segment_count() {
        let dir = std::env::temp_dir().join(format!("lxrf-compact-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("xref");

        for i in 0..(COMPACTION_THRESHOLD + 1) {
            let f = write_file(&dir, &format!("f{i}.py"), &format!("def marker_{i}():\n    pass\n"));
            xref_index::append_batch(&store_root, &[f]).unwrap();
        }

        let handles = xref_index::load_manifest(&store_root).unwrap();
        assert!(
            handles.len() < COMPACTION_THRESHOLD + 1,
            "active segment count must drop once the threshold is crossed, got {}",
            handles.len()
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compaction_merges_segments_and_preserves_correct_edge_resolution_across_offsets() {
        let dir = std::env::temp_dir().join(format!("lxrf-compact-reslot-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("xref");

        let mut def_files = Vec::new();
        let mut importer_files = Vec::new();
        for i in 0..(COMPACTION_THRESHOLD + 1) {
            let def_file = write_file(
                &dir,
                &format!("def_{i}.py"),
                &format!("def zebracorn_helper_{i}():\n    return {i}\n"),
            );
            let importer_file = write_file(
                &dir,
                &format!("importer_{i}.py"),
                &format!("from def_{i} import zebracorn_helper_{i}\n"),
            );
            xref_index::append_batch(&store_root, &[def_file.clone(), importer_file.clone()]).unwrap();
            def_files.push(def_file);
            importer_files.push(importer_file);
        }

        let handles = xref_index::load_manifest(&store_root).unwrap();
        assert!(
            handles.len() <= COMPACTION_THRESHOLD,
            "compaction must have merged the batches by now, got {} active segments",
            handles.len()
        );

        for i in 0..(COMPACTION_THRESHOLD + 1) {
            let pairs =
                xref_index::definitions_for_reference(&store_root, &format!("zebracorn_helper_{i}")).unwrap();
            assert_eq!(
                pairs,
                vec![(
                    importer_files[i].to_string_lossy().into_owned(),
                    Some(def_files[i].to_string_lossy().into_owned()),
                    XrefKind::Import,
                )],
                "after merge, edge {i} must still resolve to its OWN def file, not a neighbor's -- \
                 an off-by-one in the merged file-slot offset would mis-point this"
            );
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn interrupted_merge_leaves_pre_merge_store_intact_and_loadable() {
        let dir = std::env::temp_dir().join(format!("lxrf-compact-interrupted-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("xref");

        let def_file = write_file(&dir, "def_file.py", "def zebracorn_helper():\n    return 1\n");
        let importer_file = write_file(
            &dir,
            "importer_file.py",
            "from def_file import zebracorn_helper\n",
        );
        xref_index::append_batch(&store_root, &[def_file.clone(), importer_file.clone()]).unwrap();

        let handles_before = xref_index::load_manifest(&store_root).unwrap();
        assert_eq!(handles_before.len(), 1, "one segment before any compaction");

        // Call `compact` directly -- it durably writes the merged segment
        // file but never touches the manifest, simulating a crash between
        // that write and the manifest swap `append_batch` would otherwise
        // perform next.
        let (_merged_handle, _source_ids) =
            compact(&store_root, &handles_before, &HashSet::new()).unwrap();

        let handles_after = xref_index::load_manifest(&store_root).unwrap();
        assert_eq!(
            handles_after, handles_before,
            "an interrupted merge (crash before the manifest swap) must leave the pre-merge manifest untouched"
        );

        let pairs = xref_index::definitions_for_reference(&store_root, "zebracorn_helper").unwrap();
        assert_eq!(
            pairs,
            vec![(
                importer_file.to_string_lossy().into_owned(),
                Some(def_file.to_string_lossy().into_owned()),
                XrefKind::Import,
            )],
            "the pre-merge store must still answer queries correctly after an interrupted merge"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn corrupt_xref_segment_does_not_wedge_compaction_or_future_appends() {
        let dir = std::env::temp_dir().join(format!("lxrf-compact-corrupt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("xref");

        for i in 0..COMPACTION_THRESHOLD {
            let f = write_file(&dir, &format!("f{i}.py"), &format!("def marker_{i}():\n    pass\n"));
            xref_index::append_batch(&store_root, &[f]).unwrap();
        }

        // Corrupt one active segment (id 1) so `XrefSegment::load` fails
        // on it during the compaction this next append triggers.
        let corrupt_path = store_root.join(segment_file_name(1));
        let mut raw = std::fs::read(&corrupt_path).unwrap();
        let last = raw.len() - 1;
        raw[last] ^= 0xFF;
        std::fs::write(&corrupt_path, &raw).unwrap();

        let extra = write_file(&dir, "extra.py", "def extra_marker():\n    pass\n");
        let result = xref_index::append_batch(&store_root, &[extra]);
        assert!(
            result.is_ok(),
            "append must succeed even though compaction encounters a corrupt xref segment"
        );

        // The store must still serve queries -- not wedged by the bad
        // segment reached during compaction.
        assert!(xref_index::definitions_for_reference(&store_root, "marker_0").is_ok());

        // A further append that again crosses the threshold must also
        // succeed -- the corrupt segment never permanently blocks appends.
        let extra2 = write_file(&dir, "extra2.py", "def extra_marker_2():\n    pass\n");
        assert!(xref_index::append_batch(&store_root, &[extra2]).is_ok());

        std::fs::remove_dir_all(&dir).ok();
    }
}
