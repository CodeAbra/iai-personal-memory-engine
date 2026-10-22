//! LXRFIDX1 xref edge segment: sibling of symbol_segment.rs's LSYMIDX1
//! format, mirrored by copy for the magic+CRC32+atomic-rename discipline
//! only -- never imports symbol_segment.rs's format-specific types.
//!
//! Payload is frozen at v1, no further layout changes without a version
//! bump: version byte, file table (path + content hash + mtime + size) +
//! path arena (identical shape to symbol_segment.rs's own file table),
//! then an edge table (from_slot, to_slot-or-sentinel, kind byte,
//! name-arena offset+len) + a separate name arena for the raw imported
//! name. Edges are keyed by FROM-slot -- a search caller holding a
//! referencing file's path looks up its own slot's edges directly; the
//! bidirectional (from-symbol/to-symbol) lookup lives one layer up, in
//! xref_index.rs, built fresh from every loaded segment's edges. An edge
//! whose target never resolved to a confirmed definition is stored with
//! `to_slot = None` (the Unresolved kind) rather than dropped, so the
//! name and its referencing file stay round-trippable. `Call` edges
//! resolve against their own file's def index first (`to_slot =
//! from_slot`), then fall back to that same file's own already-resolved
//! `Import` targets -- reachable only through a same-file import, never
//! an arbitrary cross-file symbol lookup; an attribute-style call
//! (`module.fn()`) stays out of scope either way.
//!
//! Fail-closed gate mirrors symbol_segment.rs's shape: magic + CRC +
//! version, a bounds-before-allocate `need()` check ahead of every
//! `Vec::with_capacity`, and a range check on every decoded
//! cross-reference (edge from_slot/to_slot against the file table).

use std::collections::HashMap;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use thiserror::Error;

use crate::symbol_scan::{scan_file_for_language as scan_symbol_candidates, Language, SymbolKind};
use crate::tokenize::feature_id;
use crate::xref_scan::{
    candidate_extensions, resolve_relative_module, scan_file_for_language as scan_xref_candidates,
    RelativeResolution, XrefKind,
};

const MAGIC: &[u8; 8] = b"LXRFIDX1";
const VERSION: u8 = 1;
const TO_SLOT_SENTINEL: u32 = u32::MAX;

#[cfg(test)]
const XREF_SEGMENT_FILE: &str = "xref.lxidx";

// Test-only counter on the segment-read path: incremented once per
// `XrefSegment::load` call, so a test can assert an incremental append
// never opens a prior segment's file. Thread-local, not a shared global --
// mirrors segment.rs's own counter, since `cargo test` runs test functions
// concurrently on separate threads.
#[cfg(test)]
thread_local! {
    static XREF_SEGMENT_LOAD_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_segment_load_count() {
    XREF_SEGMENT_LOAD_COUNT.with(|c| c.set(0));
}

#[cfg(test)]
pub(crate) fn segment_load_count() -> usize {
    XREF_SEGMENT_LOAD_COUNT.with(|c| c.get())
}

#[derive(Debug, Error)]
pub enum XrefSegmentError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("xref segment corrupt: {0}")]
    Corrupt(String),
}

pub(crate) struct XrefFileMeta {
    pub(crate) path: String,
    #[allow(dead_code)] // round-tripped for parity with the symbol lane; no current consumer reads it
    pub(crate) content_hash: u64,
    #[allow(dead_code)]
    pub(crate) mtime: u64,
    #[allow(dead_code)]
    pub(crate) size: u64,
}

pub(crate) struct XrefEdge {
    pub(crate) from_slot: u32,
    pub(crate) to_slot: Option<u32>,
    pub(crate) kind: XrefKind,
    pub(crate) name: String,
}

pub(crate) struct XrefSegment {
    pub(crate) files: Vec<XrefFileMeta>,
    pub(crate) edges: Vec<XrefEdge>,
}

/// Reserved sentinel meaning "no real content hash was ever computed"
/// (the file was unreadable) -- `content_hash64` never returns this value
/// for actual content, so a stored `0` is unambiguous: it was never a
/// coincidental real hash. Consulted by `build_incremental`'s
/// unchanged/changed comparison against `prior_index`.
const NO_HASH_SENTINEL: u64 = 0;

/// FNV-1a 64-bit over raw file bytes -- copied from symbol_segment.rs's
/// own content-identity check (never a cryptographic digest). Guaranteed
/// non-zero: a genuine all-zero output is remapped to `1`, reserving `0`
/// exclusively as `NO_HASH_SENTINEL` so the two meanings never collide in
/// the same u64 domain.
fn content_hash64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    if hash == NO_HASH_SENTINEL {
        1
    } else {
        hash
    }
}

fn kind_to_byte(kind: XrefKind) -> u8 {
    match kind {
        XrefKind::Import => 0,
        XrefKind::Unresolved => 1,
        XrefKind::Call => 2,
    }
}

fn byte_to_kind(b: u8) -> Result<XrefKind, XrefSegmentError> {
    match b {
        0 => Ok(XrefKind::Import),
        1 => Ok(XrefKind::Unresolved),
        2 => Ok(XrefKind::Call),
        other => Err(XrefSegmentError::Corrupt(format!(
            "unknown xref kind byte {other}"
        ))),
    }
}

/// True when `path`'s file stem (the name with its extension stripped)
/// equals `module` once any leading `.` characters (relative-import
/// dots) are stripped from `module`. Legacy fallback for a module path
/// `resolve_relative_module` reports as `NotRelative` (an absolute
/// dotted Python module, a bare Rust path, or a bare TS/JS package
/// specifier) -- no directory marker to resolve against, so the batch is
/// searched by stem alone.
fn stem_matches_module(path: &Path, module: &str) -> bool {
    let module_tail = module.trim_start_matches('.');
    path.file_stem()
        .and_then(|s| s.to_str())
        .is_some_and(|stem| stem == module_tail)
}

/// True when `to_path`, minus its extension, equals `target_stem`
/// exactly -- directory-aware (unlike `stem_matches_module`, which
/// compares only the leaf file name), and only for an extension the
/// scanner recognizes for `lang`.
fn path_matches_resolved_stem(to_path: &Path, target_stem: &Path, lang: Language) -> bool {
    let Some(ext) = to_path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    if !candidate_extensions(lang).contains(&ext) {
        return false;
    }
    to_path.with_extension("") == *target_stem
}

/// Extension -> scan language, deliberately narrower than
/// symbol_segment.rs's own mapper (which defaults every unrecognized
/// extension to Python): an xref candidate carries cross-file resolution
/// weight, so scanning an unrelated text file under Python's import
/// grammar risks manufacturing edges from coincidental syntax. `None`
/// means "not a source file this lane scans" -- the path still gets a
/// file-table slot (a valid resolution TARGET), it is just never a
/// resolution SOURCE.
fn language_for_path(path: &Path) -> Option<Language> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("py") => Some(Language::Python),
        Some("rs") => Some(Language::Rust),
        Some("ts") | Some("tsx") | Some("js") | Some("jsx") | Some("mjs") | Some("cjs") => {
            Some(Language::TsJs)
        }
        _ => None,
    }
}

/// Builds the file-level exact-name Def tier ONCE (feature_id of the
/// lowercased name -> strongest kind seen): the SAME shape symbol lane's
/// own per-file map uses, reused here as the confirmation target for
/// BOTH within-file call resolution and cross-file import resolution --
/// never a second, separately-maintained exact-match index, and never a
/// rescan per candidate (an O(1) hash hit replaces the old
/// scan-per-candidate `confirms_def`).
fn build_def_index(text: &str, lang: Language) -> HashMap<u32, SymbolKind> {
    let mut index: HashMap<u32, SymbolKind> = HashMap::new();
    for cand in scan_symbol_candidates(text, lang) {
        let fid = feature_id(&cand.name.to_lowercase());
        let entry = index.entry(fid).or_insert(SymbolKind::Unknown);
        if cand.kind == SymbolKind::Def {
            *entry = SymbolKind::Def;
        }
    }
    index
}

/// Shared two-pass candidate resolution for one file at `from_slot`,
/// called by BOTH `XrefSegment::build` and `build_incremental` -- the
/// single point where a future resolution-logic change lands once,
/// never as two independently-maintained copies. Pass 1 resolves every
/// `Import` candidate ONCE (never re-run in pass 2) and folds a resolved
/// one into a local, THIS-FILE-ONLY `name -> to_slot` map; pass 2 emits
/// edges, letting a `Call` fall back to that map when the file's own
/// def-index misses -- own-file def still wins first, matching the
/// pre-existing within-file semantics. `paths`/`def_indexes` are the
/// caller's own full (or kept) batch, indexed by slot.
fn resolve_file_candidates(
    from_slot: usize,
    text: &str,
    lang: Language,
    paths: &[PathBuf],
    def_indexes: &[HashMap<u32, SymbolKind>],
    corpus_root: &Path,
) -> Vec<XrefEdge> {
    let candidates: Vec<_> = scan_xref_candidates(text, lang);
    let mut import_resolutions: Vec<Option<u32>> = Vec::with_capacity(candidates.len());
    let mut import_map: HashMap<u32, u32> = HashMap::new();
    for cand in &candidates {
        if cand.kind != XrefKind::Import {
            import_resolutions.push(None);
            continue;
        }
        let module = cand.module.as_deref().unwrap_or("");
        let target_fid = feature_id(&cand.name.to_lowercase());
        let confirm = |to_slot: usize| -> bool {
            if cand.whole_module {
                return true;
            }
            def_indexes[to_slot].get(&target_fid) == Some(&SymbolKind::Def)
        };
        let resolved = match resolve_relative_module(module, lang, &paths[from_slot], corpus_root) {
            // An escaping module path is kept unresolved outright --
            // never falls back to a different resolution mechanism for
            // the same candidate.
            RelativeResolution::Escaped => None,
            RelativeResolution::Resolved(target_stem) => {
                paths.iter().enumerate().find_map(|(to_slot, to_path)| {
                    if to_slot == from_slot || !path_matches_resolved_stem(to_path, &target_stem, lang) {
                        return None;
                    }
                    confirm(to_slot).then_some(to_slot as u32)
                })
            }
            RelativeResolution::NotRelative => {
                paths.iter().enumerate().find_map(|(to_slot, to_path)| {
                    if to_slot == from_slot || !stem_matches_module(to_path, module) {
                        return None;
                    }
                    confirm(to_slot).then_some(to_slot as u32)
                })
            }
        };
        if let Some(to_slot) = resolved {
            import_map.insert(target_fid, to_slot);
        }
        import_resolutions.push(resolved);
    }

    let mut edges: Vec<XrefEdge> = Vec::with_capacity(candidates.len());
    for (idx, cand) in candidates.into_iter().enumerate() {
        match cand.kind {
            XrefKind::Unresolved => edges.push(XrefEdge {
                from_slot: from_slot as u32,
                to_slot: None,
                kind: XrefKind::Unresolved,
                name: cand.name,
            }),
            XrefKind::Call => {
                let fid = feature_id(&cand.name.to_lowercase());
                let own_file_def = def_indexes[from_slot].get(&fid) == Some(&SymbolKind::Def);
                let to_slot = if own_file_def {
                    Some(from_slot as u32)
                } else {
                    // Cross-file fallback through an already-resolved
                    // same-file Import.
                    import_map.get(&fid).copied()
                };
                edges.push(XrefEdge {
                    from_slot: from_slot as u32,
                    to_slot,
                    kind: if to_slot.is_some() {
                        XrefKind::Call
                    } else {
                        XrefKind::Unresolved
                    },
                    name: cand.name,
                });
            }
            XrefKind::Import => {
                // Reuses pass 1's stored resolution -- never re-runs
                // resolve_relative_module here.
                let to_slot = import_resolutions[idx];
                edges.push(XrefEdge {
                    from_slot: from_slot as u32,
                    to_slot,
                    kind: if to_slot.is_some() {
                        XrefKind::Import
                    } else {
                        XrefKind::Unresolved
                    },
                    name: cand.name,
                });
            }
        }
    }
    edges
}

impl XrefSegment {
    /// Reads each given path's bytes ONCE, scans every recognized source
    /// extension (Python/Rust/TS-JS) for imports and call sites, and
    /// resolves each candidate: an `Import` first tries
    /// `resolve_relative_module`'s directory-aware, corpus-root-guarded
    /// resolution (Python relative dots, Rust `self`/`super`/`crate`,
    /// TS/JS `./`/`../`) against the OTHER paths in this same batch, then
    /// an O(1) Def-tier confirmation against the target file's own def
    /// index; a module path with no directory-relative marker (an
    /// absolute Python module, a bare Rust/TS-JS path) falls back to the
    /// legacy whole-batch stem match. A path whose relative markers would
    /// climb above `corpus_root` is kept `Unresolved` outright -- never
    /// falls back, never resolved outside the corpus. A `Call` resolves
    /// against its own file's def index first, falling back to that
    /// file's own already-resolved `Import` targets when the def index
    /// misses -- never an arbitrary cross-file lookup. Pure -- performs
    /// no disk writes, never opens a prior segment's file, never reads
    /// any store outside this batch.
    pub(crate) fn build(
        paths: &[PathBuf],
        corpus_root: &Path,
    ) -> Result<XrefSegment, XrefSegmentError> {
        let mut files: Vec<XrefFileMeta> = Vec::with_capacity(paths.len());
        let mut texts: Vec<Option<String>> = Vec::with_capacity(paths.len());
        let mut languages: Vec<Option<Language>> = Vec::with_capacity(paths.len());
        let mut def_indexes: Vec<HashMap<u32, SymbolKind>> = Vec::with_capacity(paths.len());

        for p in paths {
            let bytes = match fs::read(p) {
                Ok(b) => b,
                Err(err) => {
                    eprintln!(
                        "iai_mcp_tools_core: skipping unreadable file {} for xref scan: {err}",
                        p.display()
                    );
                    files.push(XrefFileMeta {
                        path: p.to_string_lossy().into_owned(),
                        content_hash: 0,
                        mtime: 0,
                        size: 0,
                    });
                    texts.push(None);
                    languages.push(None);
                    def_indexes.push(HashMap::new());
                    continue;
                }
            };
            let text = std::str::from_utf8(&bytes).ok().map(str::to_string);
            if text.is_none() {
                eprintln!(
                    "iai_mcp_tools_core: {} is not valid utf-8 -- kept in the xref file table, not scanned",
                    p.display()
                );
            }

            let meta = fs::metadata(p)?;
            let mtime = meta
                .modified()?
                .duration_since(UNIX_EPOCH)
                .unwrap_or(std::time::Duration::ZERO)
                .as_secs();
            files.push(XrefFileMeta {
                path: p.to_string_lossy().into_owned(),
                content_hash: content_hash64(&bytes),
                mtime,
                size: bytes.len() as u64,
            });

            let lang = language_for_path(p);
            let def_index = match (text.as_deref(), lang) {
                (Some(t), Some(l)) => build_def_index(t, l),
                _ => HashMap::new(),
            };
            languages.push(lang);
            def_indexes.push(def_index);
            texts.push(text);
        }

        let mut edges: Vec<XrefEdge> = Vec::new();
        for from_slot in 0..paths.len() {
            let Some(lang) = languages[from_slot] else {
                continue;
            };
            let Some(text) = texts[from_slot].as_deref() else {
                continue;
            };
            edges.extend(resolve_file_candidates(
                from_slot,
                text,
                lang,
                paths,
                &def_indexes,
                corpus_root,
            ));
        }

        Ok(XrefSegment { files, edges })
    }

    /// Each file's (path, content_hash), in file-slot order -- lets an
    /// append populate the manifest's path index from the segment it just
    /// built in memory, without ever reopening the segment from disk.
    pub(crate) fn file_hashes(&self) -> Vec<(String, u64)> {
        self.files
            .iter()
            .map(|f| (f.path.clone(), f.content_hash))
            .collect()
    }

    /// Builds a new xref segment from `paths`, skipping any path whose
    /// content hash matches `prior_index` (unchanged -- the append-time
    /// no-op) and returning the prior (segment_id, file_id) of any path
    /// whose hash differs, for the caller to tombstone. Reads each path's
    /// bytes ONCE; resolution (imports/calls) runs only against the
    /// surviving (kept) subset -- an unchanged sibling that is a
    /// resolution target for some other changed file in the same batch is
    /// not re-included here, matching the pre-existing same-batch-only
    /// resolution scope this lane has always had (a candidate never
    /// resolves against a file absent from the batch it was given). A
    /// read failure on a path with a prior entry is treated as
    /// transient, never as a content change -- the prior entry (and its
    /// edges) is left untouched rather than tombstoned, matching
    /// segment.rs/symbol_segment.rs's own read-failure handling.
    pub(crate) fn build_incremental(
        paths: &[PathBuf],
        corpus_root: &Path,
        prior_index: &HashMap<String, (u32, u32, u64)>,
    ) -> Result<(XrefSegment, Vec<(u32, u32)>), XrefSegmentError> {
        let mut kept_paths: Vec<PathBuf> = Vec::new();
        let mut files: Vec<XrefFileMeta> = Vec::new();
        let mut texts: Vec<Option<String>> = Vec::new();
        let mut languages: Vec<Option<Language>> = Vec::new();
        let mut def_indexes: Vec<HashMap<u32, SymbolKind>> = Vec::new();
        let mut tombstones: Vec<(u32, u32)> = Vec::new();

        for p in paths {
            let path_str = p.to_string_lossy().into_owned();
            let bytes = match fs::read(p) {
                Ok(b) => b,
                Err(err) => {
                    eprintln!(
                        "iai_mcp_tools_core: skipping unreadable file {} for xref scan: {err}",
                        p.display()
                    );
                    if prior_index.contains_key(&path_str) {
                        // A read failure says nothing about whether the
                        // file's content actually changed -- unlike
                        // segment.rs/symbol_segment.rs, this lane cannot
                        // just skip the whole file (a resolution target
                        // still needs a file-table slot), but it must
                        // never tombstone the prior entry on a failure
                        // that may be purely transient. Leave the prior
                        // (segment_id, file_id) and its edges untouched.
                        continue;
                    }
                    // First-ever sighting of this path and it is
                    // unreadable -- keep an edge-less placeholder row so
                    // the path still gets a file-table slot (a valid
                    // resolution TARGET), matching `build`'s own
                    // placeholder-row behavior.
                    kept_paths.push(p.clone());
                    files.push(XrefFileMeta {
                        path: path_str,
                        content_hash: NO_HASH_SENTINEL,
                        mtime: 0,
                        size: 0,
                    });
                    texts.push(None);
                    languages.push(None);
                    def_indexes.push(HashMap::new());
                    continue;
                }
            };
            let content_hash = content_hash64(&bytes);

            if let Some(&(prior_segment_id, prior_file_id, prior_hash)) =
                prior_index.get(&path_str)
            {
                if prior_hash == content_hash {
                    continue; // unchanged -- no slot, no tombstone
                }
                tombstones.push((prior_segment_id, prior_file_id));
            }

            kept_paths.push(p.clone());
            let text = std::str::from_utf8(&bytes).ok().map(str::to_string);
            if text.is_none() {
                eprintln!(
                    "iai_mcp_tools_core: {} is not valid utf-8 -- kept in the xref file table, not scanned",
                    p.display()
                );
            }

            let meta = fs::metadata(p)?;
            let mtime = meta
                .modified()?
                .duration_since(UNIX_EPOCH)
                .unwrap_or(std::time::Duration::ZERO)
                .as_secs();
            files.push(XrefFileMeta {
                path: path_str,
                content_hash,
                mtime,
                size: bytes.len() as u64,
            });

            let lang = language_for_path(p);
            let def_index = match (text.as_deref(), lang) {
                (Some(t), Some(l)) => build_def_index(t, l),
                _ => HashMap::new(),
            };
            languages.push(lang);
            def_indexes.push(def_index);
            texts.push(text);
        }

        let mut edges: Vec<XrefEdge> = Vec::new();
        for from_slot in 0..kept_paths.len() {
            let Some(lang) = languages[from_slot] else {
                continue;
            };
            let Some(text) = texts[from_slot].as_deref() else {
                continue;
            };
            edges.extend(resolve_file_candidates(
                from_slot,
                text,
                lang,
                &kept_paths,
                &def_indexes,
                corpus_root,
            ));
        }

        Ok((XrefSegment { files, edges }, tombstones))
    }

    pub(crate) fn save(&self, path: &Path) -> Result<(), XrefSegmentError> {
        let n = self.files.len() as u32;

        let mut path_arena: Vec<u8> = Vec::new();
        struct FileTableEntry {
            path_offset: u32,
            path_len: u16,
            content_hash: u64,
            mtime: u64,
            size: u64,
        }
        let mut file_table: Vec<FileTableEntry> = Vec::with_capacity(self.files.len());
        for f in &self.files {
            let bytes = f.path.as_bytes();
            file_table.push(FileTableEntry {
                path_offset: path_arena.len() as u32,
                path_len: bytes.len() as u16,
                content_hash: f.content_hash,
                mtime: f.mtime,
                size: f.size,
            });
            path_arena.extend_from_slice(bytes);
        }

        let mut payload: Vec<u8> = Vec::new();
        payload.push(VERSION);
        payload.extend_from_slice(&n.to_le_bytes());
        for e in &file_table {
            payload.extend_from_slice(&e.path_offset.to_le_bytes());
            payload.extend_from_slice(&e.path_len.to_le_bytes());
            payload.extend_from_slice(&e.content_hash.to_le_bytes());
            payload.extend_from_slice(&e.mtime.to_le_bytes());
            payload.extend_from_slice(&e.size.to_le_bytes());
        }
        payload.extend_from_slice(&path_arena);

        let mut name_arena: Vec<u8> = Vec::new();
        struct EdgeTableEntry {
            from_slot: u32,
            to_slot: u32,
            kind: u8,
            name_offset: u32,
            name_len: u16,
        }
        let mut edge_table: Vec<EdgeTableEntry> = Vec::with_capacity(self.edges.len());
        for e in &self.edges {
            let bytes = e.name.as_bytes();
            edge_table.push(EdgeTableEntry {
                from_slot: e.from_slot,
                to_slot: e.to_slot.unwrap_or(TO_SLOT_SENTINEL),
                kind: kind_to_byte(e.kind),
                name_offset: name_arena.len() as u32,
                name_len: bytes.len() as u16,
            });
            name_arena.extend_from_slice(bytes);
        }

        payload.extend_from_slice(&(edge_table.len() as u32).to_le_bytes());
        for e in &edge_table {
            payload.extend_from_slice(&e.from_slot.to_le_bytes());
            payload.extend_from_slice(&e.to_slot.to_le_bytes());
            payload.push(e.kind);
            payload.extend_from_slice(&e.name_offset.to_le_bytes());
            payload.extend_from_slice(&e.name_len.to_le_bytes());
        }
        payload.extend_from_slice(&name_arena);

        let crc = crc32fast::hash(&payload);

        let tmp = path.with_extension("tmp-xrefidx");
        {
            let mut f = fs::File::create(&tmp)?;
            f.write_all(MAGIC)?;
            f.write_all(&crc.to_le_bytes())?;
            f.write_all(&payload)?;
            f.sync_all()?;
        }
        fs::rename(&tmp, path)?;
        Ok(())
    }

    pub(crate) fn load(path: &Path) -> Result<Self, XrefSegmentError> {
        #[cfg(test)]
        XREF_SEGMENT_LOAD_COUNT.with(|c| c.set(c.get() + 1));

        let raw = fs::read(path)?;
        if raw.len() < 12 || &raw[..8] != MAGIC {
            return Err(XrefSegmentError::Corrupt("not an LXRFIDX1 segment".into()));
        }
        let stored_crc = u32::from_le_bytes(raw[8..12].try_into().unwrap());
        let payload = &raw[12..];
        if crc32fast::hash(payload) != stored_crc {
            return Err(XrefSegmentError::Corrupt("CRC mismatch".into()));
        }

        let need = |off: usize, len: usize| -> Result<(), XrefSegmentError> {
            if off + len > payload.len() {
                Err(XrefSegmentError::Corrupt("truncated payload".into()))
            } else {
                Ok(())
            }
        };

        let mut off = 0usize;
        need(off, 1)?;
        let version = payload[off];
        off += 1;
        if version != VERSION {
            return Err(XrefSegmentError::Corrupt(format!(
                "unsupported xref segment version {version}"
            )));
        }

        need(off, 4)?;
        let n = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        need(off, n.saturating_mul(30))?;

        struct RawFileEntry {
            path_offset: u32,
            path_len: u16,
            content_hash: u64,
            mtime: u64,
            size: u64,
        }
        let mut file_table: Vec<RawFileEntry> = Vec::with_capacity(n);
        for _ in 0..n {
            need(off, 30)?;
            let path_offset = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap());
            let path_len = u16::from_le_bytes(payload[off + 4..off + 6].try_into().unwrap());
            let content_hash = u64::from_le_bytes(payload[off + 6..off + 14].try_into().unwrap());
            let mtime = u64::from_le_bytes(payload[off + 14..off + 22].try_into().unwrap());
            let size = u64::from_le_bytes(payload[off + 22..off + 30].try_into().unwrap());
            file_table.push(RawFileEntry {
                path_offset,
                path_len,
                content_hash,
                mtime,
                size,
            });
            off += 30;
        }

        let path_arena_len = file_table
            .iter()
            .map(|e| e.path_offset as usize + e.path_len as usize)
            .max()
            .unwrap_or(0);
        need(off, path_arena_len)?;
        let path_arena = &payload[off..off + path_arena_len];
        off += path_arena_len;

        let mut files = Vec::with_capacity(n);
        for e in &file_table {
            let bytes =
                &path_arena[e.path_offset as usize..e.path_offset as usize + e.path_len as usize];
            let s = std::str::from_utf8(bytes)
                .map_err(|_| XrefSegmentError::Corrupt("non-utf8 path".into()))?;
            files.push(XrefFileMeta {
                path: s.to_string(),
                content_hash: e.content_hash,
                mtime: e.mtime,
                size: e.size,
            });
        }

        need(off, 4)?;
        let edge_count = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        need(off, edge_count.saturating_mul(15))?;

        struct RawEdgeEntry {
            from_slot: u32,
            to_slot: u32,
            kind: u8,
            name_offset: u32,
            name_len: u16,
        }
        let mut edge_table: Vec<RawEdgeEntry> = Vec::with_capacity(edge_count);
        for _ in 0..edge_count {
            need(off, 15)?;
            let from_slot = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap());
            let to_slot = u32::from_le_bytes(payload[off + 4..off + 8].try_into().unwrap());
            let kind = payload[off + 8];
            let name_offset = u32::from_le_bytes(payload[off + 9..off + 13].try_into().unwrap());
            let name_len = u16::from_le_bytes(payload[off + 13..off + 15].try_into().unwrap());
            edge_table.push(RawEdgeEntry {
                from_slot,
                to_slot,
                kind,
                name_offset,
                name_len,
            });
            off += 15;
        }

        let name_arena_len = edge_table
            .iter()
            .map(|e| e.name_offset as usize + e.name_len as usize)
            .max()
            .unwrap_or(0);
        need(off, name_arena_len)?;
        let name_arena = &payload[off..off + name_arena_len];
        off += name_arena_len;

        let mut edges = Vec::with_capacity(edge_count);
        for e in &edge_table {
            let bytes =
                &name_arena[e.name_offset as usize..e.name_offset as usize + e.name_len as usize];
            let name = std::str::from_utf8(bytes)
                .map_err(|_| XrefSegmentError::Corrupt("non-utf8 edge name".into()))?
                .to_string();
            let kind = byte_to_kind(e.kind)?;
            let to_slot = if e.to_slot == TO_SLOT_SENTINEL {
                None
            } else {
                Some(e.to_slot)
            };
            edges.push(XrefEdge {
                from_slot: e.from_slot,
                to_slot,
                kind,
                name,
            });
        }

        if off != payload.len() {
            return Err(XrefSegmentError::Corrupt(
                "trailing bytes after payload".into(),
            ));
        }

        // Referential integrity: every decoded cross-reference must stay
        // in-bounds relative to the file table it indexes into, checked
        // here so a corrupt-but-CRC-valid file fails closed on load
        // rather than panicking downstream in search.
        if edges.iter().any(|e| e.from_slot as usize >= n) {
            return Err(XrefSegmentError::Corrupt(
                "edge from_slot out of range".into(),
            ));
        }
        if edges
            .iter()
            .any(|e| e.to_slot.is_some_and(|t| t as usize >= n))
        {
            return Err(XrefSegmentError::Corrupt(
                "edge to_slot out of range".into(),
            ));
        }
        // Kind and slot must agree -- a corrupt-but-CRC-valid file
        // claiming a resolved kind (Import/Call) with no to_slot, or
        // Unresolved with one, is caught here rather than
        // misinterpreted downstream.
        if edges
            .iter()
            .any(|e| (e.kind == XrefKind::Unresolved) == e.to_slot.is_some())
        {
            return Err(XrefSegmentError::Corrupt(
                "edge kind disagrees with to_slot presence".into(),
            ));
        }

        Ok(XrefSegment { files, edges })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_resolves_import_to_the_confirmed_defining_file() {
        let dir = std::env::temp_dir().join(format!("lxrfidx-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let def_file = dir.join("def_file.py");
        std::fs::write(&def_file, "def zebracorn_helper():\n    return 1\n").unwrap();
        let importer_file = dir.join("importer_file.py");
        std::fs::write(
            &importer_file,
            "from def_file import zebracorn_helper\nresult = zebracorn_helper()\n",
        )
        .unwrap();

        let segment = XrefSegment::build(&[def_file.clone(), importer_file.clone()], &dir).unwrap();
        let edge = segment
            .edges
            .iter()
            .find(|e| e.kind == XrefKind::Import)
            .expect("expected one resolved Import edge");
        assert_eq!(segment.files[edge.from_slot as usize].path, importer_file.to_string_lossy());
        assert_eq!(
            segment.files[edge.to_slot.unwrap() as usize].path,
            def_file.to_string_lossy()
        );
        assert_eq!(edge.name, "zebracorn_helper");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_keeps_an_unresolvable_import_with_no_matching_module_stem() {
        let dir = std::env::temp_dir().join(format!("lxrfidx-unresolved-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let importer_file = dir.join("importer_file.py");
        std::fs::write(
            &importer_file,
            "from nonexistent_module import missing_helper\n",
        )
        .unwrap();

        let segment = XrefSegment::build(&[importer_file.clone()], &dir).unwrap();
        assert_eq!(segment.edges.len(), 1);
        let edge = &segment.edges[0];
        assert_eq!(edge.kind, XrefKind::Unresolved);
        assert!(edge.to_slot.is_none());
        assert_eq!(edge.name, "missing_helper");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_keeps_unresolved_when_module_stem_matches_but_name_is_not_defined_there() {
        let dir = std::env::temp_dir().join(format!("lxrfidx-nodef-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let other_file = dir.join("other_module.py");
        std::fs::write(&other_file, "def unrelated_thing():\n    pass\n").unwrap();
        let importer_file = dir.join("importer_file.py");
        std::fs::write(
            &importer_file,
            "from other_module import never_defined_name\n",
        )
        .unwrap();

        let segment = XrefSegment::build(&[other_file, importer_file], &dir).unwrap();
        let edge = segment
            .edges
            .iter()
            .find(|e| e.name == "never_defined_name")
            .unwrap();
        assert_eq!(edge.kind, XrefKind::Unresolved);
        assert!(edge.to_slot.is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_load_roundtrip_preserves_resolved_and_unresolved_edges() {
        let dir = std::env::temp_dir().join(format!("lxrfidx-roundtrip-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let def_file = dir.join("def_file.py");
        std::fs::write(&def_file, "def zebracorn_helper():\n    return 1\n").unwrap();
        let importer_file = dir.join("importer_file.py");
        std::fs::write(
            &importer_file,
            "from def_file import zebracorn_helper\nfrom ghost_module import ghost_name\n",
        )
        .unwrap();

        let segment = XrefSegment::build(&[def_file.clone(), importer_file.clone()], &dir).unwrap();
        let path = dir.join(XREF_SEGMENT_FILE);
        segment.save(&path).unwrap();
        let loaded = XrefSegment::load(&path).unwrap();

        assert_eq!(loaded.edges.len(), 2);
        let resolved = loaded
            .edges
            .iter()
            .find(|e| e.name == "zebracorn_helper")
            .unwrap();
        assert_eq!(resolved.kind, XrefKind::Import);
        assert_eq!(
            loaded.files[resolved.to_slot.unwrap() as usize].path,
            def_file.to_string_lossy()
        );

        let unresolved = loaded.edges.iter().find(|e| e.name == "ghost_name").unwrap();
        assert_eq!(unresolved.kind, XrefKind::Unresolved);
        assert!(unresolved.to_slot.is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_resolves_within_file_call_to_its_own_definition() {
        let dir = std::env::temp_dir().join(format!("lxrfidx-call-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let only = dir.join("only.py");
        std::fs::write(
            &only,
            "def zebracorn_helper():\n    return 1\nresult = zebracorn_helper()\n",
        )
        .unwrap();

        let segment = XrefSegment::build(&[only.clone()], &dir).unwrap();
        let edge = segment
            .edges
            .iter()
            .find(|e| e.kind == XrefKind::Call)
            .expect("expected one resolved Call edge");
        assert_eq!(edge.from_slot, 0);
        assert_eq!(
            edge.to_slot, Some(0),
            "a within-file call must resolve to its OWN file's slot"
        );
        assert_eq!(edge.name, "zebracorn_helper");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_keeps_unresolved_call_when_no_matching_def_in_file() {
        let dir = std::env::temp_dir().join(format!("lxrfidx-call-nodef-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let only = dir.join("only.py");
        std::fs::write(&only, "result = never_defined_call()\n").unwrap();

        let segment = XrefSegment::build(&[only], &dir).unwrap();
        let edge = segment
            .edges
            .iter()
            .find(|e| e.name == "never_defined_call")
            .expect("expected one candidate for the undefined call");
        assert_eq!(edge.kind, XrefKind::Unresolved);
        assert!(edge.to_slot.is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_resolves_a_bare_call_cross_file_through_an_already_resolved_import() {
        let dir = std::env::temp_dir().join(format!("lxrfidx-crosscall-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let def_file = dir.join("def_file.py");
        std::fs::write(&def_file, "def zebracornbarehelper():\n    return 1\n").unwrap();
        let importer_file = dir.join("importer_file.py");
        std::fs::write(
            &importer_file,
            "from def_file import zebracornbarehelper\nresult = zebracornbarehelper()\n",
        )
        .unwrap();

        let segment =
            XrefSegment::build(&[def_file.clone(), importer_file.clone()], &dir).unwrap();
        let call_edge = segment
            .edges
            .iter()
            .find(|e| e.kind == XrefKind::Call)
            .expect("expected the bare call to resolve, not stay Unresolved");
        assert_eq!(call_edge.from_slot, 1, "the call originates in the importer file");
        assert_eq!(
            segment.files[call_edge.to_slot.unwrap() as usize].path,
            def_file.to_string_lossy(),
            "a bare call to a name reached via an already-resolved same-file import must resolve cross-file"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_incremental_resolves_a_bare_call_cross_file_through_an_already_resolved_import() {
        let dir = std::env::temp_dir().join(format!("lxrfidx-crosscall-incr-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let def_file = dir.join("def_file.py");
        std::fs::write(&def_file, "def zebracornincrhelper():\n    return 1\n").unwrap();
        let importer_file = dir.join("importer_file.py");
        std::fs::write(
            &importer_file,
            "from def_file import zebracornincrhelper\nresult = zebracornincrhelper()\n",
        )
        .unwrap();

        let (segment, tombstones) = XrefSegment::build_incremental(
            &[def_file.clone(), importer_file.clone()],
            &dir,
            &HashMap::new(),
        )
        .unwrap();
        assert!(tombstones.is_empty(), "first-ever index has nothing to tombstone");
        let call_edge = segment
            .edges
            .iter()
            .find(|e| e.kind == XrefKind::Call)
            .expect("expected the bare call to resolve on the incremental path too");
        assert_eq!(
            segment.files[call_edge.to_slot.unwrap() as usize].path,
            def_file.to_string_lossy(),
            "the incremental build path (every real append_batch after the first) must apply \
             the identical cross-file Call fallback"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_keeps_own_file_definition_priority_over_a_same_named_import() {
        let dir = std::env::temp_dir().join(format!("lxrfidx-ownfilewins-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // importer_file both imports AND defines a name of the same
        // spelling as the imported one -- the bare call must resolve to
        // its OWN file's definition, never the cross-file import target.
        let other_file = dir.join("other_file.py");
        std::fs::write(&other_file, "def zebracornshadowed():\n    return 1\n").unwrap();
        let importer_file = dir.join("importer_file.py");
        std::fs::write(
            &importer_file,
            "from other_file import zebracornshadowed\ndef zebracornshadowed():\n    return 2\nresult = zebracornshadowed()\n",
        )
        .unwrap();

        let segment =
            XrefSegment::build(&[other_file.clone(), importer_file.clone()], &dir).unwrap();
        let call_edge = segment
            .edges
            .iter()
            .find(|e| e.kind == XrefKind::Call)
            .expect("expected the bare call to resolve");
        assert_eq!(
            segment.files[call_edge.to_slot.unwrap() as usize].path,
            importer_file.to_string_lossy(),
            "own-file def must win over an imported same-name binding"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_does_not_resolve_an_attribute_style_call_across_files() {
        let dir = std::env::temp_dir().join(format!("lxrfidx-attrcall-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // The scanner's own `!preceded_by_dot` guard means `mod.helper()`
        // never emits a matching Call candidate at all -- this shape is
        // out of scope for the cross-file fallback, not silently expected
        // to resolve.
        let def_file = dir.join("mod.py");
        std::fs::write(&def_file, "def helper():\n    return 1\n").unwrap();
        let importer_file = dir.join("importer_file.py");
        std::fs::write(
            &importer_file,
            "import mod\nresult = mod.helper()\n",
        )
        .unwrap();

        let segment =
            XrefSegment::build(&[def_file.clone(), importer_file.clone()], &dir).unwrap();
        assert!(
            !segment
                .edges
                .iter()
                .any(|e| e.kind == XrefKind::Call && e.name == "helper"),
            "an attribute-style call must never produce a resolved cross-file Call edge, got: {:?}",
            segment.edges.iter().map(|e| (&e.name, e.kind, e.to_slot)).collect::<Vec<_>>()
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_keeps_a_bare_call_unresolved_when_no_own_file_def_and_no_matching_import() {
        let dir = std::env::temp_dir().join(format!("lxrfidx-crosscall-nomatch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let def_file = dir.join("def_file.py");
        std::fs::write(&def_file, "def zebracornunrelated():\n    return 1\n").unwrap();
        let importer_file = dir.join("importer_file.py");
        std::fs::write(
            &importer_file,
            "from def_file import zebracornunrelated\nresult = mystery_call_never_matched()\n",
        )
        .unwrap();

        let segment =
            XrefSegment::build(&[def_file, importer_file], &dir).unwrap();
        let edge = segment
            .edges
            .iter()
            .find(|e| e.name == "mystery_call_never_matched")
            .expect("expected one candidate for the unmatched call");
        assert_eq!(
            edge.kind,
            XrefKind::Unresolved,
            "an unmatched call must stay Unresolved even when the file has an unrelated import"
        );
        assert!(edge.to_slot.is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_scans_rust_and_tsjs_sources_via_the_same_batch() {
        let dir = std::env::temp_dir().join(format!("lxrfidx-multilang-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let rust_file = dir.join("only.rs");
        std::fs::write(
            &rust_file,
            "fn zebracorn_open() {}\nfn zebracorn_run() {\n    zebracorn_open();\n}\n",
        )
        .unwrap();
        let ts_file = dir.join("only.ts");
        std::fs::write(
            &ts_file,
            "function zebracornOpen() {}\nfunction zebracornRun() {\n    zebracornOpen();\n}\n",
        )
        .unwrap();

        let rust_segment = XrefSegment::build(&[rust_file], &dir).unwrap();
        assert!(
            rust_segment
                .edges
                .iter()
                .any(|e| e.kind == XrefKind::Call && e.name == "zebracorn_open" && e.to_slot == Some(0)),
            "expected a resolved within-file Rust call edge, got: {:?}",
            rust_segment.edges.iter().map(|e| (&e.name, e.kind, e.to_slot)).collect::<Vec<_>>()
        );

        let ts_segment = XrefSegment::build(&[ts_file], &dir).unwrap();
        assert!(
            ts_segment
                .edges
                .iter()
                .any(|e| e.kind == XrefKind::Call && e.name == "zebracornOpen" && e.to_slot == Some(0)),
            "expected a resolved within-file TS/JS call edge, got: {:?}",
            ts_segment.edges.iter().map(|e| (&e.name, e.kind, e.to_slot)).collect::<Vec<_>>()
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_load_roundtrip_preserves_call_edges() {
        let dir = std::env::temp_dir().join(format!("lxrfidx-call-roundtrip-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let only = dir.join("only.py");
        std::fs::write(
            &only,
            "def zebracorn_helper():\n    return 1\nresult = zebracorn_helper()\n",
        )
        .unwrap();

        let segment = XrefSegment::build(&[only], &dir).unwrap();
        let path = dir.join(XREF_SEGMENT_FILE);
        segment.save(&path).unwrap();
        let loaded = XrefSegment::load(&path).unwrap();

        let edge = loaded
            .edges
            .iter()
            .find(|e| e.kind == XrefKind::Call)
            .expect("expected the Call edge to round-trip");
        assert_eq!(edge.to_slot, Some(0));
        assert_eq!(edge.name, "zebracorn_helper");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_foreign_magic() {
        let dir = std::env::temp_dir().join(format!("lxrfidx-foreign-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("foreign.lxidx");
        std::fs::write(&path, b"NOT-A-REAL-XREF-SEGMENT.....").unwrap();
        assert!(XrefSegment::load(&path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_corrupted_segment() {
        let dir = std::env::temp_dir().join(format!("lxrfidx-corrupt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let only = dir.join("only.py");
        std::fs::write(&only, "from mod import name\n").unwrap();
        let segment = XrefSegment::build(&[only], &dir).unwrap();
        let good = dir.join(XREF_SEGMENT_FILE);
        segment.save(&good).unwrap();

        let mut raw = std::fs::read(&good).unwrap();
        let last = raw.len() - 1;
        raw[last] ^= 0xFF;
        std::fs::write(&good, &raw).unwrap();
        assert!(XrefSegment::load(&good).is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_unsupported_version() {
        let dir = std::env::temp_dir().join(format!("lxrfidx-version-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let only = dir.join("only.py");
        std::fs::write(&only, "from mod import name\n").unwrap();
        let segment = XrefSegment::build(&[only], &dir).unwrap();
        let good = dir.join(XREF_SEGMENT_FILE);
        segment.save(&good).unwrap();

        let mut raw = std::fs::read(&good).unwrap();
        raw[12] = 2; // payload's first byte, right after magic(8)+crc(4)
        let payload = raw[12..].to_vec();
        let new_crc = crc32fast::hash(&payload);
        raw[8..12].copy_from_slice(&new_crc.to_le_bytes());
        std::fs::write(&good, &raw).unwrap();

        let result = XrefSegment::load(&good);
        match result {
            Err(XrefSegmentError::Corrupt(msg)) => assert!(msg.contains("version")),
            Err(other) => panic!("expected a version-corrupt error, got: {other}"),
            Ok(_) => panic!("expected load to reject an unsupported version"),
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_oversized_file_table_count_before_allocating() {
        let dir = std::env::temp_dir().join(format!("lxrfidx-oversized-n-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let only = dir.join("only.py");
        std::fs::write(&only, "from mod import name\n").unwrap();
        let segment = XrefSegment::build(&[only], &dir).unwrap();
        let good = dir.join(XREF_SEGMENT_FILE);
        segment.save(&good).unwrap();

        let mut raw = std::fs::read(&good).unwrap();
        let n_off = 12 + 1;
        raw[n_off..n_off + 4].copy_from_slice(&0xFFFF_FFF0u32.to_le_bytes());
        let payload = raw[12..].to_vec();
        let new_crc = crc32fast::hash(&payload);
        raw[8..12].copy_from_slice(&new_crc.to_le_bytes());
        std::fs::write(&good, &raw).unwrap();

        let result = XrefSegment::load(&good);
        assert!(
            result.is_err(),
            "an oversized declared file_table count must fail cleanly, not attempt a huge allocation"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_truncated_header() {
        let dir = std::env::temp_dir().join(format!("lxrfidx-truncated-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("truncated.lxidx");
        // Shorter than magic(8) + crc(4) -- must fail on the length check,
        // never index into a slice that does not exist.
        std::fs::write(&path, MAGIC).unwrap();
        assert!(
            XrefSegment::load(&path).is_err(),
            "a header shorter than magic+crc must fail cleanly, not panic on a short slice"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_oversized_edge_count_before_allocating() {
        let dir = std::env::temp_dir().join(format!("lxrfidx-oversized-edges-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // A hand-built minimal payload (empty file table, so no arena
        // bytes) isolates the edge-count bounds check from the file-table
        // one already covered above -- version(1) + n=0(4) + edge_count(4).
        let mut payload: Vec<u8> = Vec::new();
        payload.push(VERSION);
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.extend_from_slice(&0xFFFF_FFF0u32.to_le_bytes());
        let crc = crc32fast::hash(&payload);

        let path = dir.join("oversized-edges.lxidx");
        let mut raw = Vec::new();
        raw.extend_from_slice(MAGIC);
        raw.extend_from_slice(&crc.to_le_bytes());
        raw.extend_from_slice(&payload);
        std::fs::write(&path, &raw).unwrap();

        let result = XrefSegment::load(&path);
        assert!(
            result.is_err(),
            "an oversized declared edge_count must fail cleanly, not attempt a huge allocation"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_out_of_range_edge_to_slot_reference() {
        let dir = std::env::temp_dir().join(format!("lxrfidx-oob-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let def_file = dir.join("def_file.py");
        std::fs::write(&def_file, "def helper():\n    pass\n").unwrap();
        let importer_file = dir.join("importer_file.py");
        std::fs::write(&importer_file, "from def_file import helper\n").unwrap();

        let mut segment = XrefSegment::build(&[def_file, importer_file], &dir).unwrap();
        let edge = segment
            .edges
            .iter_mut()
            .find(|e| e.kind == XrefKind::Import)
            .unwrap();
        edge.to_slot = Some(9999);
        let good = dir.join(XREF_SEGMENT_FILE);
        segment.save(&good).unwrap();

        let result = XrefSegment::load(&good);
        assert!(
            result.is_err(),
            "an out-of-range edge to_slot reference must fail cleanly at load, not panic downstream"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_trailing_bytes_after_payload() {
        let dir = std::env::temp_dir().join(format!("lxrfidx-trailing-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let only = dir.join("only.py");
        std::fs::write(&only, "from mod import name\n").unwrap();
        let segment = XrefSegment::build(&[only], &dir).unwrap();
        let good = dir.join(XREF_SEGMENT_FILE);
        segment.save(&good).unwrap();

        let mut raw = std::fs::read(&good).unwrap();
        let payload_with_trailer: Vec<u8> =
            raw[12..].iter().copied().chain(std::iter::once(0u8)).collect();
        let new_crc = crc32fast::hash(&payload_with_trailer);
        raw.truncate(8);
        raw.extend_from_slice(&new_crc.to_le_bytes());
        raw.extend_from_slice(&payload_with_trailer);
        std::fs::write(&good, &raw).unwrap();

        let result = XrefSegment::load(&good);
        match result {
            Err(XrefSegmentError::Corrupt(msg)) => assert!(msg.contains("trailing")),
            Err(other) => panic!("expected a trailing-bytes-corrupt error, got: {other}"),
            Ok(_) => panic!("expected load to reject trailing bytes after the declared payload"),
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_resolves_a_relative_python_import_across_a_subdirectory() {
        let dir = std::env::temp_dir().join(format!("lxrfidx-relative-py-{}", std::process::id()));
        let pkg = dir.join("pkg");
        std::fs::create_dir_all(&pkg).unwrap();

        let def_file = pkg.join("def_file.py");
        std::fs::write(&def_file, "def zebracorn_helper():\n    return 1\n").unwrap();
        let importer_file = pkg.join("importer_file.py");
        std::fs::write(
            &importer_file,
            "from .def_file import zebracorn_helper\n",
        )
        .unwrap();

        let segment =
            XrefSegment::build(&[def_file.clone(), importer_file.clone()], &dir).unwrap();
        let edge = segment
            .edges
            .iter()
            .find(|e| e.kind == XrefKind::Import)
            .expect("expected the relative import to resolve");
        assert_eq!(
            segment.files[edge.to_slot.unwrap() as usize].path,
            def_file.to_string_lossy(),
            "a same-package relative import must resolve to its sibling file"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_keeps_an_escaping_relative_python_import_unresolved() {
        let dir = std::env::temp_dir().join(format!("lxrfidx-escape-py-{}", std::process::id()));
        let sub = dir.join("sub");
        std::fs::create_dir_all(&sub).unwrap();

        // A file sitting OUTSIDE the corpus root (`sub`, not `dir`) --
        // present on disk to prove the guard rejects the escaping
        // directory reference itself, before any candidate search.
        let outside_secret = dir.join("secret_module.py");
        std::fs::write(&outside_secret, "def leaked_helper():\n    return 1\n").unwrap();

        let importer_file = sub.join("importer_file.py");
        std::fs::write(
            &importer_file,
            "from ..secret_module import leaked_helper\n",
        )
        .unwrap();

        // corpus_root is `sub` itself -- the import's single climb level
        // (".." = one level up) lands at `dir`, outside `sub`.
        let segment = XrefSegment::build(&[importer_file.clone()], &sub).unwrap();
        assert_eq!(segment.edges.len(), 1);
        let edge = &segment.edges[0];
        assert_eq!(
            edge.kind,
            XrefKind::Unresolved,
            "an escaping relative import must be kept unresolved, never resolved outside the corpus root"
        );
        assert!(edge.to_slot.is_none());
        assert_eq!(edge.name, "leaked_helper");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_resolves_a_rust_super_path_across_a_subdirectory() {
        let dir = std::env::temp_dir().join(format!("lxrfidx-relative-rs-{}", std::process::id()));
        let sub = dir.join("sub");
        std::fs::create_dir_all(&sub).unwrap();

        let def_file = dir.join("helper.rs");
        std::fs::write(&def_file, "pub fn zebracorn_helper() {}\n").unwrap();
        let importer_file = sub.join("importer.rs");
        std::fs::write(
            &importer_file,
            "use super::helper::zebracorn_helper;\n",
        )
        .unwrap();

        let segment =
            XrefSegment::build(&[def_file.clone(), importer_file.clone()], &dir).unwrap();
        let edge = segment
            .edges
            .iter()
            .find(|e| e.kind == XrefKind::Import)
            .expect("expected the super:: path to resolve");
        assert_eq!(
            segment.files[edge.to_slot.unwrap() as usize].path,
            def_file.to_string_lossy(),
            "a super:: path must resolve to the parent directory's file"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Regression gate: a transient read failure (permission race, editor
    /// lock, concurrent rename) on a path that already has a prior entry
    /// must never be treated as a content change -- it must leave that
    /// prior entry (and its edges) untouched, never tombstoned.
    #[test]
    fn build_incremental_leaves_prior_entry_untouched_on_a_transient_read_failure() {
        use std::os::unix::fs::PermissionsExt;

        let dir = std::env::temp_dir().join(format!("lxrfidx-transient-read-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let def_file = dir.join("def_file.py");
        std::fs::write(&def_file, "def zebracorn_helper():\n    return 1\n").unwrap();
        let importer_file = dir.join("importer_file.py");
        std::fs::write(&importer_file, "from def_file import zebracorn_helper\n").unwrap();

        let (segment, tombstones) = XrefSegment::build_incremental(
            &[def_file.clone(), importer_file.clone()],
            &dir,
            &HashMap::new(),
        )
        .unwrap();
        assert!(tombstones.is_empty(), "first-ever index has nothing to tombstone");
        assert_eq!(segment.files.len(), 2);
        let import_edge_before = segment
            .edges
            .iter()
            .find(|e| e.kind == XrefKind::Import)
            .expect("expected the import to resolve on the first index");
        assert_eq!(
            segment.files[import_edge_before.to_slot.unwrap() as usize].path,
            def_file.to_string_lossy()
        );

        let mut prior_index: HashMap<String, (u32, u32, u64)> = HashMap::new();
        for (slot, (path, hash)) in segment.file_hashes().into_iter().enumerate() {
            prior_index.insert(path, (1, slot as u32, hash));
        }

        // Simulate a transient read failure on importer_file -- its
        // on-disk content is unchanged, only the read itself fails.
        let original_perms = std::fs::metadata(&importer_file).unwrap().permissions();
        let mut unreadable = original_perms.clone();
        unreadable.set_mode(0o000);
        std::fs::set_permissions(&importer_file, unreadable).unwrap();

        let result = XrefSegment::build_incremental(&[importer_file.clone()], &dir, &prior_index);

        // Restore permissions before any assertion below can panic and
        // skip cleanup.
        std::fs::set_permissions(&importer_file, original_perms).unwrap();

        let (segment2, tombstones2) = result.unwrap();
        assert!(
            tombstones2.is_empty(),
            "a transient read failure must never tombstone the prior entry, got: {tombstones2:?}"
        );
        assert!(
            segment2.files.is_empty(),
            "a transient read failure on an otherwise-unchanged path must add no new slot, got: {:?}",
            segment2.files.iter().map(|f| &f.path).collect::<Vec<_>>()
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
