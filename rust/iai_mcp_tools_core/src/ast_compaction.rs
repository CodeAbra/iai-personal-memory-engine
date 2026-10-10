//! Fixed-threshold AST-lane compaction: merges the active ast segment set
//! into ONE segment when the active count crosses a locked threshold.
//! Mirrors xref_compaction.rs's shape by copy -- own segment file naming,
//! own threshold, never imports the xref lane's types. Takes its
//! tombstone set as a parameter from the caller's own already-loaded
//! manifest snapshot (no `SegmentSetBuffer` instance), so `ast_index.rs`
//! performs the (merged IN, sources OUT) manifest swap directly against
//! the plain handle list this module hands it.
//!
//! An AST ref's `to_def_slot` references a row in the SAME segment's own
//! def table. A def whose owning file was superseded is dropped here,
//! and every ref that pointed at it degrades to `to_def_slot = None`
//! rather than dangling.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::ast_index::AstHandle;
use crate::ast_segment::{AstDef, AstFileMeta, AstRef, AstSegment, AstSegmentError};

/// v1: a plain, unexposed constant -- not a config knob. Same value as
/// the other lanes' own threshold, kept as an independent constant so no
/// lane shares a compile-time dependency on another's tuning.
pub(crate) const COMPACTION_THRESHOLD: usize = 16;

pub(crate) fn should_compact(active_segment_count: usize) -> bool {
    active_segment_count > COMPACTION_THRESHOLD
}

fn segment_file_name(id: u32) -> String {
    format!("ast-{id}.lastidx")
}

struct FileCandidate {
    segment_id: u32,
    local_slot: u32,
    meta: AstFileMeta,
}

/// Reads every active ast segment fully, excludes tombstoned
/// `(segment_id, file_id)` pairs, keeps at most ONE file per path
/// (highest segment_id wins -- also heals any pre-existing duplicate-path
/// pollution), and builds ONE merged segment over the survivors. A def
/// whose own `(segment_id, file_slot)` did not survive is dropped, along
/// with every ref pointing at it (degraded to `to_def_slot = None`,
/// never left dangling). Writes the merged segment durably BEFORE it is
/// ever referenced by the active set. Does not touch the manifest -- the
/// caller performs the (merged IN, sources OUT) swap as one step, so a
/// crash between this call returning and that swap leaves the pre-merge
/// manifest, and therefore the pre-merge store, untouched and loadable.
/// All-or-nothing: if any active segment fails to load, the whole merge
/// aborts rather than reconciling over a subset.
pub(crate) fn compact(
    store_root: &Path,
    handles: &[AstHandle],
    tombstones: &HashSet<(u32, u32)>,
) -> Result<(AstHandle, HashSet<u32>), AstSegmentError> {
    struct Loaded {
        segment_id: u32,
        seg: AstSegment,
    }
    let mut loaded: Vec<Loaded> = Vec::new();
    let mut source_ids: HashSet<u32> = HashSet::new();

    for handle in handles {
        match AstSegment::load(&store_root.join(&handle.file_name)) {
            Ok(seg) => {
                loaded.push(Loaded {
                    segment_id: handle.id,
                    seg,
                });
                source_ids.insert(handle.id);
            }
            Err(err) => {
                eprintln!(
                    "iai_mcp_tools_core: skipping unloadable ast segment {} during compaction: {err}",
                    handle.file_name
                );
            }
        }
    }

    if source_ids.len() != handles.len() {
        return Err(AstSegmentError::Corrupt(
            "ast compaction requires every active segment to load -- a partial merge would drop \
             file_index bookkeeping for a segment that stays active"
                .into(),
        ));
    }

    let mut candidates: Vec<FileCandidate> = Vec::new();
    for l in &loaded {
        for (local_slot, meta) in l.seg.files.iter().enumerate() {
            if tombstones.contains(&(l.segment_id, local_slot as u32)) {
                continue; // superseded -- excluded, permanently reclaimed here
            }
            candidates.push(FileCandidate {
                segment_id: l.segment_id,
                local_slot: local_slot as u32,
                meta: AstFileMeta {
                    path: meta.path.clone(),
                    content_hash: meta.content_hash,
                    parse_ok: meta.parse_ok,
                },
            });
        }
    }

    // At most one file per path -- prefer the highest segment_id (most recent).
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

    // (segment_id, local file slot) -> merged file slot, for the surviving files.
    let mut file_slot_map: HashMap<(u32, u32), u32> = HashMap::new();
    let mut all_files: Vec<AstFileMeta> = Vec::new();
    for (i, c) in candidates.into_iter().enumerate() {
        if !kept.contains(&i) {
            continue;
        }
        let new_slot = all_files.len() as u32;
        file_slot_map.insert((c.segment_id, c.local_slot), new_slot);
        all_files.push(c.meta);
    }

    // (segment_id, old def index within that segment) -> merged def slot.
    let mut def_slot_map: HashMap<(u32, u32), u32> = HashMap::new();
    let mut all_defs: Vec<AstDef> = Vec::new();
    for l in &loaded {
        for (old_def_idx, d) in l.seg.defs.iter().enumerate() {
            let Some(&new_file_slot) = file_slot_map.get(&(l.segment_id, d.file_slot)) else {
                continue; // this def's file instance was superseded
            };
            let new_slot = all_defs.len() as u32;
            def_slot_map.insert((l.segment_id, old_def_idx as u32), new_slot);
            all_defs.push(AstDef {
                file_slot: new_file_slot,
                name: d.name.clone(),
                line: d.line,
                kind: d.kind,
            });
        }
    }

    let mut all_refs: Vec<AstRef> = Vec::new();
    for l in &loaded {
        for r in &l.seg.refs {
            let Some(&new_file_slot) = file_slot_map.get(&(l.segment_id, r.file_slot)) else {
                continue; // this ref's own file instance was superseded
            };
            let new_to_def_slot = r
                .to_def_slot
                .and_then(|old_def_idx| def_slot_map.get(&(l.segment_id, old_def_idx)).copied());
            all_refs.push(AstRef {
                file_slot: new_file_slot,
                name: r.name.clone(),
                line: r.line,
                kind: r.kind,
                to_def_slot: new_to_def_slot,
            });
        }
    }

    let merged = AstSegment {
        files: all_files,
        defs: all_defs,
        refs: all_refs,
    };

    let id = handles.iter().map(|h| h.id).max().unwrap_or(0) + 1;
    let file_name = segment_file_name(id);
    merged.save(&store_root.join(&file_name))?;

    Ok((AstHandle { id, file_name }, source_ids))
}

/// Deletes the retired source segment files -- called ONLY after the
/// manifest swap that drops them is durable on disk. A crash before this
/// runs leaves harmless orphan files, never a correctness fault.
pub(crate) fn delete_sources(
    store_root: &Path,
    source_ids: &HashSet<u32>,
    prior_handles: &[AstHandle],
) {
    for handle in prior_handles.iter().filter(|h| source_ids.contains(&h.id)) {
        let _ = std::fs::remove_file(store_root.join(&handle.file_name));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast_index;
    use crate::ast_segment::DefFact;

    fn write_file(dir: &Path, name: &str, content: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn compaction_fires_after_threshold_and_drops_active_segment_count() {
        let dir = std::env::temp_dir().join(format!("last-compact-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("ast");

        for i in 0..(COMPACTION_THRESHOLD + 1) {
            let f = write_file(&dir, &format!("f{i}.py"), &format!("def marker_{i}():\n    pass\n"));
            ast_index::append_batch(
                &store_root,
                vec![crate::ast_segment::FileFacts {
                    path: f.to_string_lossy().into_owned(),
                    parse_ok: true,
                    defs: vec![DefFact { name: format!("marker_{i}"), line: 1, kind: "function".into() }],
                    refs: vec![],
                    calls: vec![],
                    imports: vec![],
                }],
            )
            .unwrap();
        }

        let handles = ast_index::load_manifest(&store_root).unwrap();
        assert!(
            handles.len() < COMPACTION_THRESHOLD + 1,
            "active segment count must drop once the threshold is crossed, got {}",
            handles.len()
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compaction_preserves_correct_resolution_across_merged_offsets() {
        let dir = std::env::temp_dir().join(format!("last-compact-reslot-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("ast");

        for i in 0..(COMPACTION_THRESHOLD + 1) {
            let f = write_file(&dir, &format!("f{i}.py"), &format!("def marker_{i}():\n    return {i}\n"));
            ast_index::append_batch(
                &store_root,
                vec![crate::ast_segment::FileFacts {
                    path: f.to_string_lossy().into_owned(),
                    parse_ok: true,
                    defs: vec![DefFact { name: format!("marker_{i}"), line: 1, kind: "function".into() }],
                    refs: vec![],
                    calls: vec![],
                    imports: vec![],
                }],
            )
            .unwrap();
        }

        let handles = ast_index::load_manifest(&store_root).unwrap();
        assert!(
            handles.len() <= COMPACTION_THRESHOLD,
            "compaction must have merged the batches by now, got {} active segments",
            handles.len()
        );

        for i in 0..(COMPACTION_THRESHOLD + 1) {
            let f = dir.join(format!("f{i}.py"));
            match ast_index::goto_definition(&store_root, &format!("marker_{i}"), &f.to_string_lossy()).unwrap() {
                ast_index::GotoStatus::Resolved { line, .. } => assert_eq!(line, 1),
                _ => panic!("expected marker_{i} to still resolve after compaction"),
            }
        }

        std::fs::remove_dir_all(&dir).ok();
    }
}
