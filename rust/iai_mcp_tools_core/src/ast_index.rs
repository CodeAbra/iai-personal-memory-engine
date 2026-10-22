//! AST-lane orchestration: an on-disk manifest (own magic, own store
//! subdir) plus one immutable LASTIDX1 fact segment per batch. Reads
//! fresh from disk on every call -- no `SegmentSetBuffer` residency.
//!
//! v3 manifest: `handles` + a `path -> (segment_id, file_id, content_hash)`
//! `file_index` + a `(segment_id, file_id)` tombstone set + a
//! `generation` counter bumped on every real publish. A v1 or v2
//! manifest (the latter built under the case-insensitive resolver fixed
//! in v3) discards ALL of it -- handles included, not just file_index/
//! tombstones/generation -- so no old segment can resurrect a wrong
//! edge through compaction's own file_index rebuild, and every path
//! reads back stale instead.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::ast_compaction;
use crate::ast_segment::{AstSegment, AstSegmentError, FileFacts};

const MANIFEST_MAGIC_V1: &[u8; 8] = b"LASTMNF1";
/// Built under a case-insensitive resolver (fixed in v3) -- its
/// file_index/tombstones/generation/handles are never trusted, exactly
/// like v1. This forces `stale_paths` to report every previously-indexed
/// path as stale, so a v2 store converges to case-sensitive resolution
/// on its next index rather than silently keeping a wrong `to_def_slot`
/// edge forever.
const MANIFEST_MAGIC_V2: &[u8; 8] = b"LASTMNF2";
const MANIFEST_MAGIC_V3: &[u8; 8] = b"LASTMNF3";
const MANIFEST_FILE: &str = "manifest.laman";

#[derive(Debug, Error)]
pub enum AstIndexError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ast manifest corrupt: {0}")]
    Corrupt(String),
    #[error(transparent)]
    Segment(#[from] AstSegmentError),
}

/// A lightweight reference to an on-disk ast segment -- an id (unique
/// among currently active segments) and its file name under the store
/// root. Never carries the loaded `AstSegment`. `pub(crate)` so
/// `ast_compaction.rs` (a sibling module, never a descendant) can build
/// and swap handles without opening a segment it does not need to read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AstHandle {
    pub(crate) id: u32,
    pub(crate) file_name: String,
}

/// Full on-disk ast manifest state: segment handles, the `file_index`
/// consulted by `append_batch` before resolving and by `goto_definition`
/// for an O(1) locate, the `(segment_id, file_id)` tombstone set, and a
/// `generation` counter bumped on every real publish.
pub(crate) struct AstManifestState {
    pub(crate) handles: Vec<AstHandle>,
    pub(crate) file_index: HashMap<String, (u32, u32, u64)>,
    #[allow(dead_code)] // round-tripped for compaction's own tombstone-set parameter; consulted there, not re-read here
    pub(crate) tombstones: HashSet<(u32, u32)>,
    pub(crate) generation: u64,
}

fn manifest_path(store_root: &Path) -> PathBuf {
    store_root.join(MANIFEST_FILE)
}

/// Loads just the segment handle list -- the pre-v2 accessor, kept as-is
/// so a caller that only needs handles is unaffected by the v2
/// file-index/tombstone/generation sections.
pub(crate) fn load_manifest(store_root: &Path) -> Result<Vec<AstHandle>, AstIndexError> {
    Ok(load_manifest_full(store_root)?.handles)
}

pub(crate) fn load_manifest_full(store_root: &Path) -> Result<AstManifestState, AstIndexError> {
    let path = manifest_path(store_root);
    if !path.exists() {
        return Ok(AstManifestState {
            handles: Vec::new(),
            file_index: HashMap::new(),
            tombstones: HashSet::new(),
            generation: 0,
        });
    }
    let raw = fs::read(&path)?;
    if raw.len() < 12 {
        return Err(AstIndexError::Corrupt("not an ast manifest file".into()));
    }
    let magic: [u8; 8] = raw[..8].try_into().unwrap();
    // `has_v2_sections`: whether the payload's byte layout includes the
    // file_index/tombstones/generation sections at all (v1 does not).
    // `trust_v2_sections`: whether those sections' VALUES are trusted --
    // false for v2, whose sections were built by the case-insensitive
    // resolver. A v2 file is still walked byte-for-byte (so the trailing-
    // bytes check below stays correct) but its decoded file_index/
    // tombstones/generation are discarded in favor of empty/zero, forcing
    // every path to read back as stale on the next index rather than
    // trusting a persisted wrong `to_def_slot` edge.
    let (has_v2_sections, trust_v2_sections) = if magic == *MANIFEST_MAGIC_V3 {
        (true, true)
    } else if magic == *MANIFEST_MAGIC_V2 {
        (true, false)
    } else if magic == *MANIFEST_MAGIC_V1 {
        (false, false)
    } else {
        return Err(AstIndexError::Corrupt("not an ast manifest file".into()));
    };
    let stored_crc = u32::from_le_bytes(raw[8..12].try_into().unwrap());
    let payload = &raw[12..];
    if crc32fast::hash(payload) != stored_crc {
        return Err(AstIndexError::Corrupt("ast manifest CRC mismatch".into()));
    }

    let need = |off: usize, len: usize| -> Result<(), AstIndexError> {
        if off + len > payload.len() {
            Err(AstIndexError::Corrupt("truncated ast manifest".into()))
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
            .map_err(|_| AstIndexError::Corrupt("non-utf8 segment name".into()))?
            .to_string();
        off += name_len;
        handles.push(AstHandle {
            id,
            file_name: name,
        });
    }

    let mut file_index: HashMap<String, (u32, u32, u64)> = HashMap::new();
    let mut tombstones: HashSet<(u32, u32)> = HashSet::new();
    let mut generation: u64 = 0;

    if has_v2_sections {
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
                .map_err(|_| AstIndexError::Corrupt("non-utf8 file-index path".into()))?
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

        need(off, 8)?;
        generation = u64::from_le_bytes(payload[off..off + 8].try_into().unwrap());
        off += 8;
    }

    if off != payload.len() {
        return Err(AstIndexError::Corrupt(
            "trailing bytes after ast manifest payload".into(),
        ));
    }

    // Discard `handles` too, not just file_index/tombstones/generation:
    // `ast_compaction::compact` rebuilds file_index from EVERY active
    // handle's own segment payload, filtered only by `tombstones` (which
    // is also empty here) -- keeping the old handles would let a v1/v2
    // segment's wrong (case-insensitive) `to_def_slot` edges resurrect
    // into a freshly-compacted v3 file_index the very first time
    // compaction fires post-upgrade, silently undoing the discard above.
    let handles = if trust_v2_sections { handles } else { Vec::new() };

    if !trust_v2_sections {
        file_index = HashMap::new();
        tombstones = HashSet::new();
        generation = 0;
    }

    Ok(AstManifestState {
        handles,
        file_index,
        tombstones,
        generation,
    })
}

fn save_manifest(store_root: &Path, state: &AstManifestState) -> Result<(), AstIndexError> {
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(&(state.handles.len() as u32).to_le_bytes());
    for h in &state.handles {
        payload.extend_from_slice(&h.id.to_le_bytes());
        let name_bytes = h.file_name.as_bytes();
        payload.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        payload.extend_from_slice(name_bytes);
    }

    // Sorted so a manifest re-saved from an unchanged logical state is
    // byte-stable.
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

    payload.extend_from_slice(&state.generation.to_le_bytes());

    let crc = crc32fast::hash(&payload);
    let path = manifest_path(store_root);
    let tmp = path.with_extension("tmp-astmanifest");
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(MANIFEST_MAGIC_V3)?;
        f.write_all(&crc.to_le_bytes())?;
        f.write_all(&payload)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, &path)?;
    Ok(())
}

fn segment_file_name(id: u32) -> String {
    format!("ast-{id}.lastidx")
}

/// `(file_index.len(), generation)` for the AST manifest under
/// `store_root`, read fresh off disk on every call -- mirrors
/// `manifest_stats` (the whole-file lane's own accessor).
pub(crate) fn manifest_stats(store_root: &Path) -> Result<(u64, u64), AstIndexError> {
    let state = load_manifest_full(store_root)?;
    Ok((state.file_index.len() as u64, state.generation))
}

/// Publishes ONE new immutable AST fact segment under `store_root` (the
/// AST lane's own subdir) from already-extracted facts, incrementally: a
/// file whose content hash is unchanged since the last publish is
/// skipped (no new slot, no tombstone); a changed or new file's prior
/// `(segment_id, file_id)`, if any, is tombstoned. A no-op batch (every
/// file already current) leaves the manifest byte-identical and bumps no
/// generation.
pub(crate) fn append_batch(store_root: &Path, files: Vec<FileFacts>) -> Result<(), AstIndexError> {
    if files.is_empty() {
        return Ok(());
    }
    fs::create_dir_all(store_root)?;
    let mut state = load_manifest_full(store_root)?;

    let (segment, new_tombstones) = AstSegment::build_incremental(files, &state.file_index)?;
    if segment.files.is_empty() && new_tombstones.is_empty() {
        // Every file in this batch is already current -- no new segment,
        // no tombstone, manifest stays byte-identical -- a genuine no-op.
        return Ok(());
    }

    let id = state.handles.iter().map(|h| h.id).max().unwrap_or(0) + 1;
    let file_name = segment_file_name(id);
    let file_entries: Vec<(String, u64)> = segment
        .files
        .iter()
        .map(|f| (f.path.clone(), f.content_hash))
        .collect();
    segment.save(&store_root.join(&file_name))?;

    state.handles.push(AstHandle {
        id,
        file_name: file_name.clone(),
    });
    for (slot, (path, hash)) in file_entries.into_iter().enumerate() {
        state.file_index.insert(path, (id, slot as u32, hash));
    }
    for t in new_tombstones {
        state.tombstones.insert(t);
    }
    state.generation += 1;
    save_manifest(store_root, &state)?;

    if ast_compaction::should_compact(state.handles.len()) {
        let prior_handles = state.handles.clone();
        // The append above is already durable in the manifest -- a
        // compaction failure (e.g. one unloadable source segment) must
        // degrade to "compaction skipped this round", never fail the
        // append that already succeeded.
        match ast_compaction::compact(store_root, &state.handles, &state.tombstones) {
            Ok((merged_handle, source_ids)) => {
                match AstSegment::load(&store_root.join(&merged_handle.file_name)) {
                    Ok(merged_seg) => {
                        let mut next_handles: Vec<AstHandle> = prior_handles
                            .iter()
                            .filter(|h| !source_ids.contains(&h.id))
                            .cloned()
                            .collect();
                        next_handles.push(merged_handle.clone());

                        let mut rebuilt_file_index: HashMap<String, (u32, u32, u64)> =
                            HashMap::new();
                        for (slot, f) in merged_seg.files.iter().enumerate() {
                            rebuilt_file_index.insert(
                                f.path.clone(),
                                (merged_handle.id, slot as u32, f.content_hash),
                            );
                        }

                        let remaining_tombstones: HashSet<(u32, u32)> = state
                            .tombstones
                            .into_iter()
                            .filter(|(seg_id, _)| !source_ids.contains(seg_id))
                            .collect();

                        let next_state = AstManifestState {
                            handles: next_handles,
                            file_index: rebuilt_file_index,
                            tombstones: remaining_tombstones,
                            generation: state.generation,
                        };
                        save_manifest(store_root, &next_state)?;
                        ast_compaction::delete_sources(store_root, &source_ids, &prior_handles);
                    }
                    Err(err) => {
                        eprintln!(
                            "iai_mcp_tools_core: ast compaction produced an unreadable merged segment ({err}); manifest swap skipped for this round"
                        );
                    }
                }
            }
            Err(err) => {
                eprintln!(
                    "iai_mcp_tools_core: ast compaction skipped for this append ({err}); active segment count remains {}",
                    prior_handles.len()
                );
            }
        }
    }

    Ok(())
}

/// One go-to-definition query outcome. `Resolved` carries the definite
/// target; `Unresolved` means the file was indexed and parsed but no
/// definite target was found for `name` (absent entirely, ambiguous, or
/// reachable only through a wildcard import); `NotIndexed`/`ParseError`
/// distinguish the two honestly-scoped nothing-found cases that are NOT
/// about the queried name at all, but about `from_path` itself.
pub(crate) enum GotoStatus {
    Resolved { path: String, line: u32 },
    Unresolved,
    NotIndexed,
    ParseError,
}

/// Looks up `name` as referenced from `from_path`. A same-file
/// definition wins first; otherwise reads the ref's own `to_def_slot`,
/// resolved at build time -- never re-resolved here. `to_def_slot`
/// always indexes into the SAME segment as the ref (same-batch-only
/// resolution): it is never valid against any other segment.
pub(crate) fn goto_definition(
    store_root: &Path,
    name: &str,
    from_path: &str,
) -> Result<GotoStatus, AstIndexError> {
    let state = load_manifest_full(store_root)?;
    let Some(&(segment_id, file_id, _content_hash)) = state.file_index.get(from_path) else {
        return Ok(GotoStatus::NotIndexed);
    };
    let Some(handle) = state.handles.iter().find(|h| h.id == segment_id) else {
        return Err(AstIndexError::Corrupt(
            "ast file_index references a segment handle no longer active".into(),
        ));
    };
    let seg = AstSegment::load(&store_root.join(&handle.file_name))?;
    let file_slot = file_id;

    if !seg.files[file_slot as usize].parse_ok {
        return Ok(GotoStatus::ParseError);
    }

    if let Some(def) = seg
        .defs
        .iter()
        .find(|d| d.file_slot == file_slot && d.name == name)
    {
        return Ok(GotoStatus::Resolved {
            path: seg.files[file_slot as usize].path.clone(),
            line: def.line,
        });
    }

    if let Some(r) = seg
        .refs
        .iter()
        .find(|r| r.file_slot == file_slot && r.name == name)
    {
        if let Some(def_slot) = r.to_def_slot {
            let def = &seg.defs[def_slot as usize];
            return Ok(GotoStatus::Resolved {
                path: seg.files[def.file_slot as usize].path.clone(),
                line: def.line,
            });
        }
        return Ok(GotoStatus::Unresolved);
    }

    Ok(GotoStatus::Unresolved)
}

/// One find-references row's confirmation state: `Resolved` means the
/// reference's `to_def_slot` structurally confirms the queried
/// definition; `Unresolved` means the reference's name matches but no
/// build-time link confirmed it (a wildcard import's exports cannot be
/// enumerated, so a call reachable only through one stays `Unresolved`
/// rather than silently dropped or falsely marked resolved).
pub(crate) enum RefRowStatus {
    Resolved,
    Unresolved,
}

pub(crate) struct ReferenceRow {
    pub(crate) path: String,
    pub(crate) status: RefRowStatus,
}

/// One find-references query outcome, mirroring `GotoStatus`'s honestly-
/// scoped nothing-found axis.
pub(crate) enum FindReferencesStatus {
    Hits(Vec<ReferenceRow>),
    NoReferences,
    NotIndexed,
    ParseError,
}

/// Reverse of `goto_definition`: every reference in `def_path`'s own
/// segment whose name matches `name`, tagged `Resolved` when its
/// `to_def_slot` confirms the target def in `def_path`, `Unresolved`
/// otherwise. A reference confirmed to a DIFFERENT definition (same
/// name, different file) is excluded, not treated as ambiguous. Never
/// re-resolves at query time.
pub(crate) fn find_references(
    store_root: &Path,
    name: &str,
    def_path: &str,
) -> Result<FindReferencesStatus, AstIndexError> {
    let state = load_manifest_full(store_root)?;
    let Some(&(segment_id, file_id, _content_hash)) = state.file_index.get(def_path) else {
        return Ok(FindReferencesStatus::NotIndexed);
    };
    let Some(handle) = state.handles.iter().find(|h| h.id == segment_id) else {
        return Err(AstIndexError::Corrupt(
            "ast file_index references a segment handle no longer active".into(),
        ));
    };
    let seg = AstSegment::load(&store_root.join(&handle.file_name))?;
    let file_slot = file_id;

    if !seg.files[file_slot as usize].parse_ok {
        return Ok(FindReferencesStatus::ParseError);
    }

    let Some(target_slot) = seg
        .defs
        .iter()
        .position(|d| d.file_slot == file_slot && d.name == name)
    else {
        return Ok(FindReferencesStatus::NoReferences);
    };
    let target_slot = target_slot as u32;

    let rows: Vec<ReferenceRow> = seg
        .refs
        .iter()
        .filter(|r| r.name == name)
        .filter_map(|r| match r.to_def_slot {
            Some(d) if d == target_slot => Some(ReferenceRow {
                path: seg.files[r.file_slot as usize].path.clone(),
                status: RefRowStatus::Resolved,
            }),
            Some(_) => None,
            None => Some(ReferenceRow {
                path: seg.files[r.file_slot as usize].path.clone(),
                status: RefRowStatus::Unresolved,
            }),
        })
        .collect();

    if rows.is_empty() {
        Ok(FindReferencesStatus::NoReferences)
    } else {
        Ok(FindReferencesStatus::Hits(rows))
    }
}

/// Reads each of `paths`' current bytes and returns the subset whose
/// content hash is absent from or differs from the AST manifest's own
/// `file_index` -- the pre-`ast.parse` staleness gate. Reuses the SAME
/// hash function `AstSegment`'s own build path computes, never a second
/// hash. A path that read-fails is reported stale (fail toward re-parse,
/// never toward silently skipping a file that may have changed).
pub(crate) fn stale_paths(store_root: &Path, paths: &[String]) -> Result<Vec<String>, AstIndexError> {
    let state = load_manifest_full(store_root)?;
    let mut stale = Vec::new();
    for p in paths {
        let current_hash = match fs::read(p) {
            Ok(bytes) => crate::ast_segment::content_hash64(&bytes),
            Err(_) => {
                stale.push(p.clone());
                continue;
            }
        };
        match state.file_index.get(p) {
            Some(&(_, _, prior_hash)) if prior_hash == current_hash => {}
            _ => stale.push(p.clone()),
        }
    }
    Ok(stale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast_segment::{DefFact, ImportFact, RefFact};

    fn write_file(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn a_v2_manifest_on_disk_never_trusts_its_persisted_file_index() {
        let dir = std::env::temp_dir().join(format!("laman-v2discard-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("ast");
        let only = write_file(&dir, "only.py", "def target_symbol():\n    return 1\n");

        append_batch(
            &store_root,
            vec![FileFacts {
                path: only.to_string_lossy().into_owned(),
                parse_ok: true,
                defs: vec![DefFact { name: "target_symbol".into(), line: 1, kind: "function".into() }],
                refs: vec![],
                calls: vec![],
                imports: vec![],
            }],
        )
        .unwrap();

        // Rewrite the just-saved (v3) manifest's magic to v2, as if it had
        // been built by the pre-fix case-insensitive resolver, then patch
        // the CRC to match so the rewritten file still passes the fail-
        // closed integrity gate.
        let path = manifest_path(&store_root);
        let mut raw = fs::read(&path).unwrap();
        raw[..8].copy_from_slice(MANIFEST_MAGIC_V2);
        let new_crc = crc32fast::hash(&raw[12..]);
        raw[8..12].copy_from_slice(&new_crc.to_le_bytes());
        fs::write(&path, &raw).unwrap();

        let state = load_manifest_full(&store_root).unwrap();
        assert!(
            state.handles.is_empty(),
            "a v2 manifest's handles must also be discarded -- keeping them would let \
             compaction resurrect a wrong edge from an old segment's own payload"
        );
        assert!(
            state.file_index.is_empty(),
            "a v2 manifest's file_index must never be trusted -- it may carry wrong edges"
        );
        assert!(state.tombstones.is_empty());
        assert_eq!(state.generation, 0);

        // With file_index empty, the previously-indexed path must read
        // back as stale, forcing a real re-parse rather than silently
        // keeping the old (possibly wrong) resolution.
        let stale = stale_paths(&store_root, &[only.to_string_lossy().into_owned()]).unwrap();
        assert_eq!(stale, vec![only.to_string_lossy().into_owned()]);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_v2_manifest_never_resurrects_a_wrong_edge_through_compaction() {
        use crate::ast_segment::{AstDef, AstFileMeta, AstRef, DefKind, RefKind};

        let dir = std::env::temp_dir().join(format!("laman-v2compact-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("ast");
        fs::create_dir_all(&store_root).unwrap();
        let never_resubmitted =
            write_file(&dir, "never_resubmitted.py", "def Foo():\n    return 1\nfoo()\n");

        // The current (fixed) resolver can never PRODUCE a case-mismatched
        // edge, so this constructs one directly -- exactly the on-disk
        // shape a pre-fix build would have written -- and hand-assembles a
        // v2-tagged manifest pointing at it, mirroring what
        // `append_batch`/`save_manifest` would have done pre-fix.
        let content_hash =
            crate::ast_segment::content_hash64(&fs::read(&never_resubmitted).unwrap());
        let bad_segment = AstSegment {
            files: vec![AstFileMeta {
                path: never_resubmitted.to_string_lossy().into_owned(),
                content_hash,
                parse_ok: true,
            }],
            defs: vec![AstDef {
                file_slot: 0,
                name: "Foo".into(),
                line: 1,
                kind: DefKind::Function,
            }],
            refs: vec![AstRef {
                file_slot: 0,
                name: "foo".into(),
                line: 3,
                kind: RefKind::Call,
                to_def_slot: Some(0), // the wrong, case-insensitively-resolved edge a pre-fix build produced
            }],
        };
        bad_segment.save(&store_root.join("ast-1.lastidx")).unwrap();

        let mut file_index = HashMap::new();
        file_index.insert(
            never_resubmitted.to_string_lossy().into_owned(),
            (1u32, 0u32, content_hash),
        );
        let mut handles = Vec::new();
        handles.push(AstHandle { id: 1, file_name: "ast-1.lastidx".to_string() });
        save_manifest(
            &store_root,
            &AstManifestState {
                handles,
                file_index,
                tombstones: HashSet::new(),
                generation: 1,
            },
        )
        .unwrap();

        let path = manifest_path(&store_root);
        let mut raw = fs::read(&path).unwrap();
        raw[..8].copy_from_slice(MANIFEST_MAGIC_V2);
        let new_crc = crc32fast::hash(&raw[12..]);
        raw[8..12].copy_from_slice(&new_crc.to_le_bytes());
        fs::write(&path, &raw).unwrap();

        // Re-index only OTHER files past the compaction threshold --
        // `never_resubmitted.py` is never resubmitted, matching a caller
        // that only re-indexes files it knows changed after the upgrade.
        for i in 0..(crate::ast_compaction::COMPACTION_THRESHOLD + 1) {
            let f = write_file(&dir, &format!("other_{i}.py"), &format!("def marker_{i}():\n    pass\n"));
            append_batch(
                &store_root,
                vec![FileFacts {
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

        // Compaction has fired by now. The never-resubmitted file's wrong
        // edge must NOT have resurrected into the rebuilt file_index.
        match goto_definition(&store_root, "foo", &never_resubmitted.to_string_lossy()).unwrap() {
            GotoStatus::NotIndexed => {}
            other => panic!(
                "a v2-era file that is never resubmitted must stay NotIndexed after \
                 compaction, never resolve through a resurrected wrong edge: got {}",
                match other {
                    GotoStatus::Resolved { .. } => "Resolved",
                    GotoStatus::Unresolved => "Unresolved",
                    GotoStatus::ParseError => "ParseError",
                    GotoStatus::NotIndexed => unreachable!(),
                }
            ),
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn append_batch_empty_files_is_a_no_op() {
        let dir = std::env::temp_dir().join(format!("laman-empty-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("ast");

        let result = append_batch(&store_root, Vec::new());
        assert!(result.is_ok());
        assert!(!manifest_path(&store_root).exists());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn goto_definition_resolves_a_same_file_definition_after_persisting() {
        let dir = std::env::temp_dir().join(format!("laman-samefile-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("ast");
        let only = write_file(
            &dir,
            "only.py",
            "def target_symbol():\n    return 1\nresult = target_symbol()\n",
        );

        let facts = vec![FileFacts {
            path: only.to_string_lossy().into_owned(),
            parse_ok: true,
            defs: vec![DefFact {
                name: "target_symbol".into(),
                line: 1,
                kind: "function".into(),
            }],
            refs: vec![],
            calls: vec![RefFact {
                name: "target_symbol".into(),
                line: 3,
            }],
            imports: vec![],
        }];
        append_batch(&store_root, facts).unwrap();

        match goto_definition(&store_root, "target_symbol", &only.to_string_lossy()).unwrap() {
            GotoStatus::Resolved { path, line } => {
                assert_eq!(path, only.to_string_lossy());
                assert_eq!(line, 1);
            }
            _ => panic!("expected a resolved same-file definition"),
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn goto_definition_resolves_cross_file_through_an_import_in_the_same_batch() {
        let dir = std::env::temp_dir().join(format!("laman-crossfile-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("ast");
        let def_file = write_file(&dir, "def_file.py", "def target_symbol():\n    return 1\n");
        let importer_file = write_file(
            &dir,
            "importer_file.py",
            "from def_file import target_symbol\nresult = target_symbol()\n",
        );

        let facts = vec![
            FileFacts {
                path: def_file.to_string_lossy().into_owned(),
                parse_ok: true,
                defs: vec![DefFact { name: "target_symbol".into(), line: 1, kind: "function".into() }],
                refs: vec![],
                calls: vec![],
                imports: vec![],
            },
            FileFacts {
                path: importer_file.to_string_lossy().into_owned(),
                parse_ok: true,
                defs: vec![],
                refs: vec![],
                calls: vec![RefFact { name: "target_symbol".into(), line: 2 }],
                imports: vec![ImportFact {
                    name: Some("target_symbol".into()),
                    module: Some("def_file".into()),
                    line: 1,
                    wildcard: false,
                }],
            },
        ];
        append_batch(&store_root, facts).unwrap();

        match goto_definition(&store_root, "target_symbol", &importer_file.to_string_lossy()).unwrap() {
            GotoStatus::Resolved { path, line } => {
                assert_eq!(path, def_file.to_string_lossy());
                assert_eq!(line, 1);
            }
            _ => panic!("expected a cross-file resolved definition via the same-batch import"),
        }
        // Own-file def still wins first.
        match goto_definition(&store_root, "target_symbol", &def_file.to_string_lossy()).unwrap() {
            GotoStatus::Resolved { path, .. } => assert_eq!(path, def_file.to_string_lossy()),
            _ => panic!("expected the own-file definition to resolve"),
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn goto_definition_reports_not_indexed_for_an_unknown_path() {
        let dir = std::env::temp_dir().join(format!("laman-notindexed-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("ast");

        match goto_definition(&store_root, "anything", "/never/indexed.py").unwrap() {
            GotoStatus::NotIndexed => {}
            _ => panic!("expected NotIndexed for a store with no manifest at all"),
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn goto_definition_reports_unresolved_for_a_name_never_referenced() {
        let dir = std::env::temp_dir().join(format!("laman-unresolved-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("ast");
        let only = write_file(&dir, "only.py", "x = 1\n");

        let facts = vec![FileFacts {
            path: only.to_string_lossy().into_owned(),
            parse_ok: true,
            defs: vec![],
            refs: vec![],
            calls: vec![],
            imports: vec![],
        }];
        append_batch(&store_root, facts).unwrap();

        match goto_definition(&store_root, "never_seen_at_all", &only.to_string_lossy()).unwrap() {
            GotoStatus::Unresolved => {}
            _ => panic!("expected Unresolved for a name never referenced in an indexed, parsed file"),
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn goto_definition_never_resolves_a_case_mismatched_query_to_a_real_definition() {
        let dir = std::env::temp_dir().join(format!("laman-casemismatch-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("ast");
        let only = write_file(&dir, "only.py", "def Foo():\n    return 1\nfoo()\n");

        let facts = vec![FileFacts {
            path: only.to_string_lossy().into_owned(),
            parse_ok: true,
            defs: vec![DefFact { name: "Foo".into(), line: 1, kind: "function".into() }],
            refs: vec![],
            calls: vec![RefFact { name: "foo".into(), line: 3 }],
            imports: vec![],
        }];
        append_batch(&store_root, facts).unwrap();

        match goto_definition(&store_root, "foo", &only.to_string_lossy()).unwrap() {
            GotoStatus::Unresolved => {}
            other => panic!(
                "a query for \"foo\" must never resolve against a def named \"Foo\" -- Python \
                 identifiers are case-sensitive; got a status other than Unresolved: {}",
                match other {
                    GotoStatus::Resolved { .. } => "Resolved",
                    GotoStatus::NotIndexed => "NotIndexed",
                    GotoStatus::ParseError => "ParseError",
                    GotoStatus::Unresolved => unreachable!(),
                }
            ),
        }
        // The real, case-correct query must still resolve.
        match goto_definition(&store_root, "Foo", &only.to_string_lossy()).unwrap() {
            GotoStatus::Resolved { line, .. } => assert_eq!(line, 1),
            _ => panic!("expected the case-correct query \"Foo\" to resolve"),
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn goto_definition_reports_parse_error_for_a_parse_failed_file() {
        let dir = std::env::temp_dir().join(format!("laman-parseerr-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("ast");
        let only = write_file(&dir, "only.py", "def broken(:\n");

        let facts = vec![FileFacts {
            path: only.to_string_lossy().into_owned(),
            parse_ok: false,
            defs: vec![],
            refs: vec![],
            calls: vec![],
            imports: vec![],
        }];
        append_batch(&store_root, facts).unwrap();

        match goto_definition(&store_root, "anything", &only.to_string_lossy()).unwrap() {
            GotoStatus::ParseError => {}
            _ => panic!("expected ParseError for a file recorded with parse_ok=false"),
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn append_batch_persists_across_process_style_reload() {
        let dir = std::env::temp_dir().join(format!("laman-persist-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("ast");
        let only = write_file(
            &dir,
            "only.py",
            "def target_symbol():\n    return 1\nresult = target_symbol()\n",
        );

        let facts = vec![FileFacts {
            path: only.to_string_lossy().into_owned(),
            parse_ok: true,
            defs: vec![DefFact {
                name: "target_symbol".into(),
                line: 1,
                kind: "function".into(),
            }],
            refs: vec![],
            calls: vec![RefFact {
                name: "target_symbol".into(),
                line: 3,
            }],
            imports: vec![],
        }];
        append_batch(&store_root, facts).unwrap();

        // A second, independent load of the manifest -- proves the
        // segment is read fresh from disk, not held in any in-memory
        // buffer across calls.
        let handles = load_manifest(&store_root).unwrap();
        assert_eq!(handles.len(), 1);
        match goto_definition(&store_root, "target_symbol", &only.to_string_lossy()).unwrap() {
            GotoStatus::Resolved { line, .. } => assert_eq!(line, 1),
            _ => panic!("expected a resolved definition on the reloaded store"),
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reindexing_unchanged_ast_file_is_a_no_op() {
        let dir = std::env::temp_dir().join(format!("laman-noop-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("ast");
        let only = write_file(&dir, "only.py", "def stable_def():\n    return 1\n");

        let facts = vec![FileFacts {
            path: only.to_string_lossy().into_owned(),
            parse_ok: true,
            defs: vec![DefFact { name: "stable_def".into(), line: 1, kind: "function".into() }],
            refs: vec![],
            calls: vec![],
            imports: vec![],
        }];
        append_batch(&store_root, facts.clone()).unwrap();
        let (n_before, gen_before) = manifest_stats(&store_root).unwrap();

        append_batch(&store_root, facts).unwrap();
        let (n_after, gen_after) = manifest_stats(&store_root).unwrap();

        assert_eq!(n_before, n_after);
        assert_eq!(
            gen_before, gen_after,
            "re-indexing an unchanged ast file must never bump generation"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn changed_ast_file_tombstones_the_prior_entry_and_bumps_generation() {
        let dir = std::env::temp_dir().join(format!("laman-changed-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("ast");
        let only = write_file(&dir, "only.py", "def old_def():\n    return 1\n");

        append_batch(
            &store_root,
            vec![FileFacts {
                path: only.to_string_lossy().into_owned(),
                parse_ok: true,
                defs: vec![DefFact { name: "old_def".into(), line: 1, kind: "function".into() }],
                refs: vec![],
                calls: vec![],
                imports: vec![],
            }],
        )
        .unwrap();
        let (_n1, gen1) = manifest_stats(&store_root).unwrap();

        write_file(&dir, "only.py", "def new_def():\n    return 2\n");
        append_batch(
            &store_root,
            vec![FileFacts {
                path: only.to_string_lossy().into_owned(),
                parse_ok: true,
                defs: vec![DefFact { name: "new_def".into(), line: 1, kind: "function".into() }],
                refs: vec![],
                calls: vec![],
                imports: vec![],
            }],
        )
        .unwrap();
        let (_n2, gen2) = manifest_stats(&store_root).unwrap();

        assert!(gen2 > gen1, "a real re-index of a changed file must bump generation");
        match goto_definition(&store_root, "old_def", &only.to_string_lossy()).unwrap() {
            GotoStatus::Unresolved => {}
            _ => panic!("the superseded definition must no longer be reachable"),
        }
        match goto_definition(&store_root, "new_def", &only.to_string_lossy()).unwrap() {
            GotoStatus::Resolved { line, .. } => assert_eq!(line, 1),
            _ => panic!("the new definition must resolve"),
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn stale_paths_reports_only_the_changed_file() {
        let dir = std::env::temp_dir().join(format!("laman-stale-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("ast");
        let a = write_file(&dir, "a.py", "def a_def():\n    return 1\n");
        let b = write_file(&dir, "b.py", "def b_def():\n    return 2\n");

        append_batch(
            &store_root,
            vec![
                FileFacts {
                    path: a.to_string_lossy().into_owned(),
                    parse_ok: true,
                    defs: vec![DefFact { name: "a_def".into(), line: 1, kind: "function".into() }],
                    refs: vec![],
                    calls: vec![],
                    imports: vec![],
                },
                FileFacts {
                    path: b.to_string_lossy().into_owned(),
                    parse_ok: true,
                    defs: vec![DefFact { name: "b_def".into(), line: 1, kind: "function".into() }],
                    refs: vec![],
                    calls: vec![],
                    imports: vec![],
                },
            ],
        )
        .unwrap();

        write_file(&dir, "a.py", "def a_def():\n    return 99\n");
        let stale = stale_paths(
            &store_root,
            &[a.to_string_lossy().into_owned(), b.to_string_lossy().into_owned()],
        )
        .unwrap();
        assert_eq!(stale, vec![a.to_string_lossy().into_owned()]);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn stale_paths_reports_every_path_on_a_cold_store() {
        let dir = std::env::temp_dir().join(format!("laman-cold-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("ast");
        let a = write_file(&dir, "a.py", "x = 1\n");

        let stale = stale_paths(&store_root, &[a.to_string_lossy().into_owned()]).unwrap();
        assert_eq!(stale, vec![a.to_string_lossy().into_owned()]);

        fs::remove_dir_all(&dir).ok();
    }

    fn row_status<'a>(rows: &'a [ReferenceRow], path: &Path) -> Option<&'a RefRowStatus> {
        rows.iter()
            .find(|r| r.path == path.to_string_lossy())
            .map(|r| &r.status)
    }

    #[test]
    fn find_references_returns_resolved_and_unresolved_rows_for_a_matching_name() {
        let dir = std::env::temp_dir().join(format!("laman-findrefs-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("ast");
        let def_file = write_file(&dir, "def_file.py", "def target_symbol():\n    return 1\n");
        let importer_file = write_file(
            &dir,
            "importer_file.py",
            "from def_file import target_symbol\ntarget_symbol()\n",
        );
        let bare_file = write_file(&dir, "bare_file.py", "target_symbol()\n");

        let facts = vec![
            FileFacts {
                path: def_file.to_string_lossy().into_owned(),
                parse_ok: true,
                defs: vec![DefFact { name: "target_symbol".into(), line: 1, kind: "function".into() }],
                refs: vec![],
                calls: vec![],
                imports: vec![],
            },
            FileFacts {
                path: importer_file.to_string_lossy().into_owned(),
                parse_ok: true,
                defs: vec![],
                refs: vec![],
                calls: vec![RefFact { name: "target_symbol".into(), line: 2 }],
                imports: vec![ImportFact {
                    name: Some("target_symbol".into()),
                    module: Some("def_file".into()),
                    line: 1,
                    wildcard: false,
                }],
            },
            FileFacts {
                path: bare_file.to_string_lossy().into_owned(),
                parse_ok: true,
                defs: vec![],
                refs: vec![],
                calls: vec![RefFact { name: "target_symbol".into(), line: 1 }],
                imports: vec![],
            },
        ];
        append_batch(&store_root, facts).unwrap();

        match find_references(&store_root, "target_symbol", &def_file.to_string_lossy()).unwrap() {
            FindReferencesStatus::Hits(rows) => {
                assert_eq!(rows.len(), 2, "expected exactly the importer and bare-use rows, got {}", rows.len());
                assert!(matches!(row_status(&rows, &importer_file), Some(RefRowStatus::Resolved)));
                assert!(matches!(row_status(&rows, &bare_file), Some(RefRowStatus::Unresolved)));
            }
            _ => panic!("expected Hits with two reference rows"),
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn find_references_excludes_a_reference_confirmed_to_a_different_definition() {
        let dir = std::env::temp_dir().join(format!("laman-findrefs-distinct-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("ast");
        let def_a = write_file(&dir, "def_a.py", "def shared_name():\n    return 1\n");
        let def_b = write_file(&dir, "def_b.py", "def shared_name():\n    return 2\n");
        let importer = write_file(
            &dir,
            "importer.py",
            "from def_b import shared_name\nshared_name()\n",
        );

        let facts = vec![
            FileFacts {
                path: def_a.to_string_lossy().into_owned(),
                parse_ok: true,
                defs: vec![DefFact { name: "shared_name".into(), line: 1, kind: "function".into() }],
                refs: vec![],
                calls: vec![],
                imports: vec![],
            },
            FileFacts {
                path: def_b.to_string_lossy().into_owned(),
                parse_ok: true,
                defs: vec![DefFact { name: "shared_name".into(), line: 1, kind: "function".into() }],
                refs: vec![],
                calls: vec![],
                imports: vec![],
            },
            FileFacts {
                path: importer.to_string_lossy().into_owned(),
                parse_ok: true,
                defs: vec![],
                refs: vec![],
                calls: vec![RefFact { name: "shared_name".into(), line: 2 }],
                imports: vec![ImportFact {
                    name: Some("shared_name".into()),
                    module: Some("def_b".into()),
                    line: 1,
                    wildcard: false,
                }],
            },
        ];
        append_batch(&store_root, facts).unwrap();

        match find_references(&store_root, "shared_name", &def_a.to_string_lossy()).unwrap() {
            FindReferencesStatus::NoReferences => {}
            FindReferencesStatus::Hits(rows) => panic!(
                "a reference confirmed to a DIFFERENT definition must not appear as a row for this query, got {:?}",
                rows.iter().map(|r| &r.path).collect::<Vec<_>>()
            ),
            _ => panic!("expected NoReferences, since importer.py's reference resolved to def_b.py, not def_a.py"),
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn find_references_never_matches_a_case_mismatched_query() {
        let dir = std::env::temp_dir().join(format!("laman-findrefs-casemismatch-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("ast");
        let only = write_file(&dir, "only.py", "def Foo():\n    return 1\nfoo()\n");

        let facts = vec![FileFacts {
            path: only.to_string_lossy().into_owned(),
            parse_ok: true,
            defs: vec![DefFact { name: "Foo".into(), line: 1, kind: "function".into() }],
            refs: vec![],
            calls: vec![RefFact { name: "foo".into(), line: 3 }],
            imports: vec![],
        }];
        append_batch(&store_root, facts).unwrap();

        match find_references(&store_root, "foo", &only.to_string_lossy()).unwrap() {
            FindReferencesStatus::NoReferences => {}
            _ => panic!(
                "querying find_references for \"foo\" must not match a def named \"Foo\" -- \
                 Python identifiers are case-sensitive"
            ),
        }
        match find_references(&store_root, "Foo", &only.to_string_lossy()).unwrap() {
            FindReferencesStatus::NoReferences => {}
            _ => panic!(
                "the exact-case query \"Foo\" has no matching call site (the only call is \
                 lowercase \"foo\") -- must report NoReferences, not a false match"
            ),
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn find_references_reports_not_indexed_for_an_unknown_def_path() {
        let dir = std::env::temp_dir().join(format!("laman-findrefs-notindexed-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("ast");

        match find_references(&store_root, "anything", "/never/indexed.py").unwrap() {
            FindReferencesStatus::NotIndexed => {}
            _ => panic!("expected NotIndexed for a store with no manifest at all"),
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn find_references_reports_parse_error_for_a_parse_failed_def_path() {
        let dir = std::env::temp_dir().join(format!("laman-findrefs-parseerr-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("ast");
        let broken = write_file(&dir, "broken.py", "def broken(:\n");

        let facts = vec![FileFacts {
            path: broken.to_string_lossy().into_owned(),
            parse_ok: false,
            defs: vec![],
            refs: vec![],
            calls: vec![],
            imports: vec![],
        }];
        append_batch(&store_root, facts).unwrap();

        match find_references(&store_root, "anything", &broken.to_string_lossy()).unwrap() {
            FindReferencesStatus::ParseError => {}
            _ => panic!("expected ParseError for a def_path recorded with parse_ok=false"),
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn find_references_reports_no_references_when_indexed_and_parsed_but_never_called() {
        let dir = std::env::temp_dir().join(format!("laman-findrefs-noref-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("ast");
        let only = write_file(&dir, "only.py", "def never_called():\n    return 1\n");

        let facts = vec![FileFacts {
            path: only.to_string_lossy().into_owned(),
            parse_ok: true,
            defs: vec![DefFact { name: "never_called".into(), line: 1, kind: "function".into() }],
            refs: vec![],
            calls: vec![],
            imports: vec![],
        }];
        append_batch(&store_root, facts).unwrap();

        match find_references(&store_root, "never_called", &only.to_string_lossy()).unwrap() {
            FindReferencesStatus::NoReferences => {}
            _ => panic!("expected NoReferences for a defined but never-called symbol"),
        }

        fs::remove_dir_all(&dir).ok();
    }
}
