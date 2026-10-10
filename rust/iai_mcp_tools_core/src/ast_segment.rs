//! LASTIDX1 AST-fact segment: own magic+CRC32+atomic-rename framing.
//! Rust never parses Python source -- this module only shapes and
//! persists facts `ast_facts.py` already extracted.
//!
//! A ref whose target never resolved is stored with `to_def_slot = None`
//! rather than dropped, so the name and its referencing file stay
//! round-trippable.
//!
//! Fail-closed: magic + CRC + version checked before decode, a
//! bounds-before-allocate `need()` check ahead of every
//! `Vec::with_capacity`, and a range check on every decoded
//! cross-reference against its own table.

use std::fs;
use std::io::Write as _;
use std::path::Path;

use serde::Deserialize;
use thiserror::Error;

const MAGIC: &[u8; 8] = b"LASTIDX1";
const VERSION: u8 = 1;
const TO_DEF_SLOT_SENTINEL: u32 = u32::MAX;

#[derive(Debug, Error)]
pub enum AstSegmentError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ast segment corrupt: {0}")]
    Corrupt(String),
}

/// Reserved sentinel meaning "no real content hash was ever computed"
/// (the file was unreadable) -- `content_hash64` never returns this value
/// for actual content, so a stored `0` is unambiguous. Copied by shape
/// from xref_segment.rs's own `NO_HASH_SENTINEL`; this lane keeps its own
/// copy rather than importing that one, matching the own-code-per-lane
/// discipline every existing lane already follows.
const NO_HASH_SENTINEL: u64 = 0;

/// FNV-1a 64-bit over raw file bytes -- own copy, never a cryptographic
/// digest, never imported from another lane's module. Guaranteed
/// non-zero: a genuine all-zero output is remapped to `1`.
pub(crate) fn content_hash64(bytes: &[u8]) -> u64 {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DefKind {
    Function,
    Class,
}

fn def_kind_to_byte(kind: DefKind) -> u8 {
    match kind {
        DefKind::Function => 0,
        DefKind::Class => 1,
    }
}

fn byte_to_def_kind(b: u8) -> Result<DefKind, AstSegmentError> {
    match b {
        0 => Ok(DefKind::Function),
        1 => Ok(DefKind::Class),
        other => Err(AstSegmentError::Corrupt(format!(
            "unknown ast def kind byte {other}"
        ))),
    }
}

/// Parses the def-kind label `ast_facts.py` emits ("function"/"class")
/// into the on-disk enum. Any other label is rejected rather than
/// silently coerced to a default -- a future new label needs an explicit
/// decision here, not a silent misclassification.
pub(crate) fn def_kind_from_label(label: &str) -> Result<DefKind, AstSegmentError> {
    match label {
        "function" => Ok(DefKind::Function),
        "class" => Ok(DefKind::Class),
        other => Err(AstSegmentError::Corrupt(format!(
            "unknown ast def kind label {other:?}"
        ))),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RefKind {
    Call,
    PlainRef,
}

fn ref_kind_to_byte(kind: RefKind) -> u8 {
    match kind {
        RefKind::Call => 0,
        RefKind::PlainRef => 1,
    }
}

fn byte_to_ref_kind(b: u8) -> Result<RefKind, AstSegmentError> {
    match b {
        0 => Ok(RefKind::Call),
        1 => Ok(RefKind::PlainRef),
        other => Err(AstSegmentError::Corrupt(format!(
            "unknown ast ref kind byte {other}"
        ))),
    }
}

pub(crate) struct AstFileMeta {
    pub(crate) path: String,
    pub(crate) content_hash: u64,
    pub(crate) parse_ok: bool,
}

pub(crate) struct AstDef {
    pub(crate) file_slot: u32,
    pub(crate) name: String,
    pub(crate) line: u32,
    pub(crate) kind: DefKind,
}

pub(crate) struct AstRef {
    pub(crate) file_slot: u32,
    pub(crate) name: String,
    pub(crate) line: u32,
    #[allow(dead_code)] // round-tripped for future find-refs kind filtering; no current consumer reads it
    pub(crate) kind: RefKind,
    pub(crate) to_def_slot: Option<u32>,
}

pub(crate) struct AstSegment {
    pub(crate) files: Vec<AstFileMeta>,
    pub(crate) defs: Vec<AstDef>,
    pub(crate) refs: Vec<AstRef>,
}

/// One file's already-extracted facts, deserialized from the JSON
/// `ast_facts.py` hands across the PyO3 seam. Rust never runs `ast.parse`
/// itself -- this struct only shapes what Python already walked.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct FileFacts {
    pub(crate) path: String,
    #[serde(default = "default_parse_ok")]
    pub(crate) parse_ok: bool,
    #[serde(default)]
    pub(crate) defs: Vec<DefFact>,
    #[serde(default)]
    pub(crate) refs: Vec<RefFact>,
    #[serde(default)]
    pub(crate) calls: Vec<RefFact>,
    #[serde(default)]
    pub(crate) imports: Vec<ImportFact>,
}

fn default_parse_ok() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct DefFact {
    pub(crate) name: String,
    pub(crate) line: u32,
    pub(crate) kind: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RefFact {
    pub(crate) name: String,
    pub(crate) line: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ImportFact {
    pub(crate) name: Option<String>,
    pub(crate) module: Option<String>,
    #[allow(dead_code)] // round-tripped for parity with the other fact kinds' line field; no current consumer reads it
    pub(crate) line: u32,
    pub(crate) wildcard: bool,
}

/// Reads each fact's own file bytes once and pairs it with its content
/// hash -- the shared first step `build` and `build_incremental` both
/// need before any resolution runs.
fn hash_each_file(files: Vec<FileFacts>) -> Vec<(FileFacts, u64)> {
    files
        .into_iter()
        .map(|f| {
            let content_hash = match fs::read(&f.path) {
                Ok(bytes) => content_hash64(&bytes),
                Err(err) => {
                    eprintln!(
                        "iai_mcp_tools_core: could not re-read {} for its AST content hash: {err}",
                        f.path
                    );
                    NO_HASH_SENTINEL
                }
            };
            (f, content_hash)
        })
        .collect()
}

/// The last dot-separated segment of a dotted module path, or the whole
/// string when it has no dots -- the same coarse "compare against a bare
/// module name" fallback xref_segment.rs's `stem_matches_module` uses for
/// its own `NotRelative` case. No directory-anchored relative-import
/// resolution (Python's `level`/leading-dot count) is attempted here.
fn module_last_segment(module: &str) -> &str {
    module.rsplit('.').next().unwrap_or(module)
}

/// Resolves every file's own imports to a target file + confirmed def
/// slot, SAME-BATCH only: an `ImportFact` with a concrete `module` and
/// `name` (never a wildcard) matches a candidate file elsewhere in
/// `entries` whose filename stem equals the module's last dotted
/// segment, confirmed by that candidate's OWN def index containing
/// `name`. A plain `import module` fact (no `module` field) is never
/// used for resolution here -- an attribute-style access (`module.fn()`)
/// stays out of scope, matching the locked decision. Case-sensitive
/// throughout, matching Python's own identifier semantics. Returns, per
/// file slot, a map of name -> the GLOBAL def slot it resolved to.
fn resolve_import_targets(
    entries: &[FileFacts],
    def_index_by_file: &[std::collections::HashMap<String, u32>],
) -> Vec<std::collections::HashMap<String, u32>> {
    use std::path::Path as StdPath;

    let stems: Vec<Option<&str>> = entries
        .iter()
        .map(|f| StdPath::new(&f.path).file_stem().and_then(|s| s.to_str()))
        .collect();

    let mut import_targets: Vec<std::collections::HashMap<String, u32>> =
        (0..entries.len()).map(|_| std::collections::HashMap::new()).collect();

    for (file_slot, f) in entries.iter().enumerate() {
        for imp in &f.imports {
            if imp.wildcard {
                continue;
            }
            let (Some(module), Some(name)) = (imp.module.as_deref(), imp.name.as_deref()) else {
                continue;
            };
            let want_stem = module_last_segment(module);
            for (target_slot, target_stem) in stems.iter().enumerate() {
                if target_slot == file_slot {
                    continue;
                }
                if *target_stem != Some(want_stem) {
                    continue;
                }
                if let Some(&def_slot) = def_index_by_file[target_slot].get(name) {
                    import_targets[file_slot].insert(name.to_string(), def_slot);
                    break;
                }
            }
        }
    }

    import_targets
}

/// The shared resolution body: builds the file table, the own-file def
/// index, the cross-file import-target map, then resolves every
/// ref/call candidate against own-file first, import-target second.
fn build_from_hashed(entries: Vec<(FileFacts, u64)>) -> Result<AstSegment, AstSegmentError> {
    use std::collections::HashMap;

    let (facts, hashes): (Vec<FileFacts>, Vec<u64>) = entries.into_iter().unzip();

    let mut file_metas: Vec<AstFileMeta> = Vec::with_capacity(facts.len());
    for (f, content_hash) in facts.iter().zip(hashes.iter()) {
        file_metas.push(AstFileMeta {
            path: f.path.clone(),
            content_hash: *content_hash,
            parse_ok: f.parse_ok,
        });
    }

    let mut defs: Vec<AstDef> = Vec::new();
    // file_slot -> (name -> GLOBAL def slot), the own-file O(1)
    // confirmation index every ref/call candidate consults first, and the
    // same index `resolve_import_targets` confirms an imported name
    // against. Case-sensitive: the first def for a given exact name wins,
    // matching Python's own identifier semantics.
    let mut def_index_by_file: Vec<HashMap<String, u32>> =
        (0..facts.len()).map(|_| HashMap::new()).collect();
    for (file_slot, f) in facts.iter().enumerate() {
        for d in &f.defs {
            let kind = def_kind_from_label(&d.kind)?;
            let slot = defs.len() as u32;
            defs.push(AstDef {
                file_slot: file_slot as u32,
                name: d.name.clone(),
                line: d.line,
                kind,
            });
            def_index_by_file[file_slot]
                .entry(d.name.clone())
                .or_insert(slot);
        }
    }

    let import_targets_by_file = resolve_import_targets(&facts, &def_index_by_file);

    let mut refs: Vec<AstRef> = Vec::new();
    for (file_slot, f) in facts.iter().enumerate() {
        let own_index = &def_index_by_file[file_slot];
        let import_targets = &import_targets_by_file[file_slot];
        let resolve = |name: &str| -> Option<u32> {
            own_index
                .get(name)
                .copied()
                .or_else(|| import_targets.get(name).copied())
        };
        for c in &f.calls {
            let to_def_slot = resolve(&c.name);
            refs.push(AstRef {
                file_slot: file_slot as u32,
                name: c.name.clone(),
                line: c.line,
                kind: RefKind::Call,
                to_def_slot,
            });
        }
        for r in &f.refs {
            let to_def_slot = resolve(&r.name);
            refs.push(AstRef {
                file_slot: file_slot as u32,
                name: r.name.clone(),
                line: r.line,
                kind: RefKind::PlainRef,
                to_def_slot,
            });
        }
    }

    Ok(AstSegment {
        files: file_metas,
        defs,
        refs,
    })
}

impl AstSegment {
    /// Resolution is two-pass and SAME-BATCH-only: own-file defs indexed
    /// first, then each ref/call resolved against its own file's defs,
    /// falling back to an import target within this same batch.
    /// Unresolved stays `to_def_slot = None`, recorded at build time --
    /// never re-resolved at query time.
    pub(crate) fn build(files: Vec<FileFacts>) -> Result<AstSegment, AstSegmentError> {
        let hashed = hash_each_file(files);
        build_from_hashed(hashed)
    }

    /// Same resolution as `build`, but skips a file whose content hash is
    /// unchanged since `prior_index`, and returns the prior
    /// `(segment_id, file_id)` of every changed file for the caller to
    /// tombstone. Resolution runs only against the surviving (kept)
    /// subset for this call -- same-batch-only, same as `build`. An
    /// unchanged file outside this call can never confirm another
    /// file's import, so cross-file resolution can regress from
    /// resolved to unresolved on every incremental re-index after the
    /// first (inherited from `XrefSegment::build_incremental`'s own
    /// same-batch scope, not specific to this lane).
    pub(crate) fn build_incremental(
        files: Vec<FileFacts>,
        prior_index: &std::collections::HashMap<String, (u32, u32, u64)>,
    ) -> Result<(AstSegment, Vec<(u32, u32)>), AstSegmentError> {
        let hashed = hash_each_file(files);
        let mut kept: Vec<(FileFacts, u64)> = Vec::new();
        let mut tombstones: Vec<(u32, u32)> = Vec::new();
        for (f, content_hash) in hashed {
            if let Some(&(prior_segment_id, prior_file_id, prior_hash)) =
                prior_index.get(&f.path)
            {
                if prior_hash == content_hash && content_hash != NO_HASH_SENTINEL {
                    continue; // unchanged -- no slot, no tombstone
                }
                tombstones.push((prior_segment_id, prior_file_id));
            }
            kept.push((f, content_hash));
        }
        let segment = build_from_hashed(kept)?;
        Ok((segment, tombstones))
    }

    pub(crate) fn save(&self, path: &Path) -> Result<(), AstSegmentError> {
        let mut path_arena: Vec<u8> = Vec::new();
        struct FileEntry {
            path_offset: u32,
            path_len: u16,
            content_hash: u64,
            parse_ok: u8,
        }
        let mut file_table: Vec<FileEntry> = Vec::with_capacity(self.files.len());
        for f in &self.files {
            let bytes = f.path.as_bytes();
            file_table.push(FileEntry {
                path_offset: path_arena.len() as u32,
                path_len: bytes.len() as u16,
                content_hash: f.content_hash,
                parse_ok: u8::from(f.parse_ok),
            });
            path_arena.extend_from_slice(bytes);
        }

        let mut def_name_arena: Vec<u8> = Vec::new();
        struct DefEntry {
            file_slot: u32,
            name_offset: u32,
            name_len: u16,
            line: u32,
            kind: u8,
        }
        let mut def_table: Vec<DefEntry> = Vec::with_capacity(self.defs.len());
        for d in &self.defs {
            let bytes = d.name.as_bytes();
            def_table.push(DefEntry {
                file_slot: d.file_slot,
                name_offset: def_name_arena.len() as u32,
                name_len: bytes.len() as u16,
                line: d.line,
                kind: def_kind_to_byte(d.kind),
            });
            def_name_arena.extend_from_slice(bytes);
        }

        let mut ref_name_arena: Vec<u8> = Vec::new();
        struct RefEntry {
            file_slot: u32,
            name_offset: u32,
            name_len: u16,
            line: u32,
            kind: u8,
            to_def_slot: u32,
        }
        let mut ref_table: Vec<RefEntry> = Vec::with_capacity(self.refs.len());
        for r in &self.refs {
            let bytes = r.name.as_bytes();
            ref_table.push(RefEntry {
                file_slot: r.file_slot,
                name_offset: ref_name_arena.len() as u32,
                name_len: bytes.len() as u16,
                line: r.line,
                kind: ref_kind_to_byte(r.kind),
                to_def_slot: r.to_def_slot.unwrap_or(TO_DEF_SLOT_SENTINEL),
            });
            ref_name_arena.extend_from_slice(bytes);
        }

        let mut payload: Vec<u8> = Vec::new();
        payload.push(VERSION);

        payload.extend_from_slice(&(file_table.len() as u32).to_le_bytes());
        for e in &file_table {
            payload.extend_from_slice(&e.path_offset.to_le_bytes());
            payload.extend_from_slice(&e.path_len.to_le_bytes());
            payload.extend_from_slice(&e.content_hash.to_le_bytes());
            payload.push(e.parse_ok);
        }
        payload.extend_from_slice(&path_arena);

        payload.extend_from_slice(&(def_table.len() as u32).to_le_bytes());
        for e in &def_table {
            payload.extend_from_slice(&e.file_slot.to_le_bytes());
            payload.extend_from_slice(&e.name_offset.to_le_bytes());
            payload.extend_from_slice(&e.name_len.to_le_bytes());
            payload.extend_from_slice(&e.line.to_le_bytes());
            payload.push(e.kind);
        }
        payload.extend_from_slice(&def_name_arena);

        payload.extend_from_slice(&(ref_table.len() as u32).to_le_bytes());
        for e in &ref_table {
            payload.extend_from_slice(&e.file_slot.to_le_bytes());
            payload.extend_from_slice(&e.name_offset.to_le_bytes());
            payload.extend_from_slice(&e.name_len.to_le_bytes());
            payload.extend_from_slice(&e.line.to_le_bytes());
            payload.push(e.kind);
            payload.extend_from_slice(&e.to_def_slot.to_le_bytes());
        }
        payload.extend_from_slice(&ref_name_arena);

        let crc = crc32fast::hash(&payload);

        let tmp = path.with_extension("tmp-astidx");
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

    pub(crate) fn load(path: &Path) -> Result<Self, AstSegmentError> {
        let raw = fs::read(path)?;
        if raw.len() < 12 || &raw[..8] != MAGIC {
            return Err(AstSegmentError::Corrupt("not an LASTIDX1 segment".into()));
        }
        let stored_crc = u32::from_le_bytes(raw[8..12].try_into().unwrap());
        let payload = &raw[12..];
        if crc32fast::hash(payload) != stored_crc {
            return Err(AstSegmentError::Corrupt("CRC mismatch".into()));
        }

        let need = |off: usize, len: usize| -> Result<(), AstSegmentError> {
            if off + len > payload.len() {
                Err(AstSegmentError::Corrupt("truncated payload".into()))
            } else {
                Ok(())
            }
        };

        let mut off = 0usize;
        need(off, 1)?;
        let version = payload[off];
        off += 1;
        if version != VERSION {
            return Err(AstSegmentError::Corrupt(format!(
                "unsupported ast segment version {version}"
            )));
        }

        need(off, 4)?;
        let n_files = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        // Fixed per-row size: path_offset(4) + path_len(2) + content_hash(8) + parse_ok(1) = 15.
        need(off, n_files.saturating_mul(15))?;

        struct RawFile {
            path_offset: u32,
            path_len: u16,
            content_hash: u64,
            parse_ok: u8,
        }
        let mut file_table: Vec<RawFile> = Vec::with_capacity(n_files);
        for _ in 0..n_files {
            need(off, 15)?;
            let path_offset = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap());
            let path_len = u16::from_le_bytes(payload[off + 4..off + 6].try_into().unwrap());
            let content_hash = u64::from_le_bytes(payload[off + 6..off + 14].try_into().unwrap());
            let parse_ok = payload[off + 14];
            file_table.push(RawFile {
                path_offset,
                path_len,
                content_hash,
                parse_ok,
            });
            off += 15;
        }

        let path_arena_len = file_table
            .iter()
            .map(|e| e.path_offset as usize + e.path_len as usize)
            .max()
            .unwrap_or(0);
        need(off, path_arena_len)?;
        let path_arena = &payload[off..off + path_arena_len];
        off += path_arena_len;

        let mut files: Vec<AstFileMeta> = Vec::with_capacity(n_files);
        for e in &file_table {
            let bytes =
                &path_arena[e.path_offset as usize..e.path_offset as usize + e.path_len as usize];
            let s = std::str::from_utf8(bytes)
                .map_err(|_| AstSegmentError::Corrupt("non-utf8 path".into()))?;
            files.push(AstFileMeta {
                path: s.to_string(),
                content_hash: e.content_hash,
                parse_ok: e.parse_ok != 0,
            });
        }

        need(off, 4)?;
        let n_defs = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        // file_slot(4) + name_offset(4) + name_len(2) + line(4) + kind(1) = 15.
        need(off, n_defs.saturating_mul(15))?;

        struct RawDef {
            file_slot: u32,
            name_offset: u32,
            name_len: u16,
            line: u32,
            kind: u8,
        }
        let mut def_table: Vec<RawDef> = Vec::with_capacity(n_defs);
        for _ in 0..n_defs {
            need(off, 15)?;
            let file_slot = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap());
            let name_offset = u32::from_le_bytes(payload[off + 4..off + 8].try_into().unwrap());
            let name_len = u16::from_le_bytes(payload[off + 8..off + 10].try_into().unwrap());
            let line = u32::from_le_bytes(payload[off + 10..off + 14].try_into().unwrap());
            let kind = payload[off + 14];
            def_table.push(RawDef {
                file_slot,
                name_offset,
                name_len,
                line,
                kind,
            });
            off += 15;
        }

        let def_name_arena_len = def_table
            .iter()
            .map(|e| e.name_offset as usize + e.name_len as usize)
            .max()
            .unwrap_or(0);
        need(off, def_name_arena_len)?;
        let def_name_arena = &payload[off..off + def_name_arena_len];
        off += def_name_arena_len;

        let mut defs: Vec<AstDef> = Vec::with_capacity(n_defs);
        for e in &def_table {
            let bytes = &def_name_arena
                [e.name_offset as usize..e.name_offset as usize + e.name_len as usize];
            let name = std::str::from_utf8(bytes)
                .map_err(|_| AstSegmentError::Corrupt("non-utf8 def name".into()))?
                .to_string();
            defs.push(AstDef {
                file_slot: e.file_slot,
                name,
                line: e.line,
                kind: byte_to_def_kind(e.kind)?,
            });
        }

        need(off, 4)?;
        let n_refs = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        // file_slot(4) + name_offset(4) + name_len(2) + line(4) + kind(1) + to_def_slot(4) = 19.
        need(off, n_refs.saturating_mul(19))?;

        struct RawRef {
            file_slot: u32,
            name_offset: u32,
            name_len: u16,
            line: u32,
            kind: u8,
            to_def_slot: u32,
        }
        let mut ref_table: Vec<RawRef> = Vec::with_capacity(n_refs);
        for _ in 0..n_refs {
            need(off, 19)?;
            let file_slot = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap());
            let name_offset = u32::from_le_bytes(payload[off + 4..off + 8].try_into().unwrap());
            let name_len = u16::from_le_bytes(payload[off + 8..off + 10].try_into().unwrap());
            let line = u32::from_le_bytes(payload[off + 10..off + 14].try_into().unwrap());
            let kind = payload[off + 14];
            let to_def_slot = u32::from_le_bytes(payload[off + 15..off + 19].try_into().unwrap());
            ref_table.push(RawRef {
                file_slot,
                name_offset,
                name_len,
                line,
                kind,
                to_def_slot,
            });
            off += 19;
        }

        let ref_name_arena_len = ref_table
            .iter()
            .map(|e| e.name_offset as usize + e.name_len as usize)
            .max()
            .unwrap_or(0);
        need(off, ref_name_arena_len)?;
        let ref_name_arena = &payload[off..off + ref_name_arena_len];
        off += ref_name_arena_len;

        let mut refs: Vec<AstRef> = Vec::with_capacity(n_refs);
        for e in &ref_table {
            let bytes = &ref_name_arena
                [e.name_offset as usize..e.name_offset as usize + e.name_len as usize];
            let name = std::str::from_utf8(bytes)
                .map_err(|_| AstSegmentError::Corrupt("non-utf8 ref name".into()))?
                .to_string();
            let to_def_slot = if e.to_def_slot == TO_DEF_SLOT_SENTINEL {
                None
            } else {
                Some(e.to_def_slot)
            };
            refs.push(AstRef {
                file_slot: e.file_slot,
                name,
                line: e.line,
                kind: byte_to_ref_kind(e.kind)?,
                to_def_slot,
            });
        }

        if off != payload.len() {
            return Err(AstSegmentError::Corrupt(
                "trailing bytes after payload".into(),
            ));
        }

        // Referential integrity: every decoded cross-reference must stay
        // in-bounds relative to the tables it indexes into, checked here
        // so a corrupt-but-CRC-valid file fails closed on load rather
        // than panicking downstream in query.
        if defs.iter().any(|d| d.file_slot as usize >= files.len()) {
            return Err(AstSegmentError::Corrupt(
                "def file_slot out of range".into(),
            ));
        }
        if refs.iter().any(|r| r.file_slot as usize >= files.len()) {
            return Err(AstSegmentError::Corrupt(
                "ref file_slot out of range".into(),
            ));
        }
        if refs
            .iter()
            .any(|r| r.to_def_slot.is_some_and(|d| d as usize >= defs.len()))
        {
            return Err(AstSegmentError::Corrupt(
                "ref to_def_slot out of range".into(),
            ));
        }

        Ok(AstSegment { files, defs, refs })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_resolves_a_cross_file_call_through_an_already_resolved_import() {
        let dir = std::env::temp_dir().join(format!("lastidx-crossfile-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let def_file = dir.join("def_file.py");
        std::fs::write(&def_file, "def target_symbol():\n    return 1\n").unwrap();
        let importer_file = dir.join("importer_file.py");
        std::fs::write(
            &importer_file,
            "from def_file import target_symbol\nresult = target_symbol()\n",
        )
        .unwrap();

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

        let segment = AstSegment::build(facts).unwrap();
        let importer_ref = segment
            .refs
            .iter()
            .find(|r| r.name == "target_symbol")
            .expect("expected the cross-file call candidate");
        let def_slot = importer_ref
            .to_def_slot
            .expect("expected the cross-file call to resolve via the already-resolved import");
        assert_eq!(segment.defs[def_slot as usize].line, 1);
        assert_eq!(
            segment.files[segment.defs[def_slot as usize].file_slot as usize].path,
            def_file.to_string_lossy()
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_never_crashes_on_a_file_recorded_parse_ok_false() {
        let dir = std::env::temp_dir().join(format!("lastidx-parsefail-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let broken = dir.join("broken.py");
        std::fs::write(&broken, "def broken(:\n").unwrap();
        let healthy = dir.join("healthy.py");
        std::fs::write(&healthy, "def healthy_def():\n    return 1\n").unwrap();

        let facts = vec![
            FileFacts {
                path: broken.to_string_lossy().into_owned(),
                parse_ok: false,
                defs: vec![],
                refs: vec![],
                calls: vec![],
                imports: vec![],
            },
            FileFacts {
                path: healthy.to_string_lossy().into_owned(),
                parse_ok: true,
                defs: vec![DefFact { name: "healthy_def".into(), line: 1, kind: "function".into() }],
                refs: vec![],
                calls: vec![],
                imports: vec![],
            },
        ];

        let segment = AstSegment::build(facts).unwrap();
        assert_eq!(segment.files.len(), 2);
        assert!(!segment.files[0].parse_ok);
        assert!(segment.files[1].parse_ok);
        assert_eq!(segment.defs.len(), 1, "the broken file must contribute zero facts, not crash the build");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_incremental_skips_an_unchanged_file_and_tombstones_a_changed_one() {
        let dir = std::env::temp_dir().join(format!("lastidx-incr-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let only = dir.join("only.py");
        std::fs::write(&only, "def target_symbol():\n    return 1\n").unwrap();

        let facts = vec![FileFacts {
            path: only.to_string_lossy().into_owned(),
            parse_ok: true,
            defs: vec![DefFact { name: "target_symbol".into(), line: 1, kind: "function".into() }],
            refs: vec![],
            calls: vec![],
            imports: vec![],
        }];
        let first = AstSegment::build(facts.clone()).unwrap();
        let mut prior_index = std::collections::HashMap::new();
        prior_index.insert(only.to_string_lossy().into_owned(), (1u32, 0u32, first.files[0].content_hash));

        // Unchanged: build_incremental must skip it entirely.
        let (unchanged_segment, tombstones) =
            AstSegment::build_incremental(facts, &prior_index).unwrap();
        assert!(unchanged_segment.files.is_empty());
        assert!(tombstones.is_empty());

        // Changed: build_incremental must tombstone the prior slot and re-include it.
        std::fs::write(&only, "def target_symbol():\n    return 2\n").unwrap();
        let changed_facts = vec![FileFacts {
            path: only.to_string_lossy().into_owned(),
            parse_ok: true,
            defs: vec![DefFact { name: "target_symbol".into(), line: 1, kind: "function".into() }],
            refs: vec![],
            calls: vec![],
            imports: vec![],
        }];
        let (changed_segment, tombstones2) =
            AstSegment::build_incremental(changed_facts, &prior_index).unwrap();
        assert_eq!(changed_segment.files.len(), 1);
        assert_eq!(tombstones2, vec![(1, 0)]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_resolves_a_same_file_call_to_its_own_definition() {
        let dir = std::env::temp_dir().join(format!("lastidx-samefile-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let only = dir.join("only.py");
        std::fs::write(
            &only,
            "def target_symbol():\n    return 1\nresult = target_symbol()\n",
        )
        .unwrap();

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

        let segment = AstSegment::build(facts).unwrap();
        assert_eq!(segment.defs.len(), 1);
        assert_eq!(segment.refs.len(), 1);
        let r = &segment.refs[0];
        assert_eq!(r.to_def_slot, Some(0));
        assert_eq!(segment.defs[0].line, 1);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_keeps_unresolved_when_no_matching_def_in_file() {
        let dir = std::env::temp_dir().join(format!("lastidx-nodef-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let only = dir.join("only.py");
        std::fs::write(&only, "result = never_defined_call()\n").unwrap();

        let facts = vec![FileFacts {
            path: only.to_string_lossy().into_owned(),
            parse_ok: true,
            defs: vec![],
            refs: vec![],
            calls: vec![RefFact {
                name: "never_defined_call".into(),
                line: 1,
            }],
            imports: vec![],
        }];

        let segment = AstSegment::build(facts).unwrap();
        assert_eq!(segment.refs.len(), 1);
        assert!(segment.refs[0].to_def_slot.is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_load_roundtrip_preserves_resolved_and_unresolved_refs() {
        let dir = std::env::temp_dir().join(format!("lastidx-roundtrip-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let only = dir.join("only.py");
        std::fs::write(
            &only,
            "def target_symbol():\n    return 1\nresult = target_symbol()\nother_call()\n",
        )
        .unwrap();

        let facts = vec![FileFacts {
            path: only.to_string_lossy().into_owned(),
            parse_ok: true,
            defs: vec![DefFact {
                name: "target_symbol".into(),
                line: 1,
                kind: "function".into(),
            }],
            refs: vec![],
            calls: vec![
                RefFact { name: "target_symbol".into(), line: 3 },
                RefFact { name: "other_call".into(), line: 4 },
            ],
            imports: vec![],
        }];

        let segment = AstSegment::build(facts).unwrap();
        let path = dir.join("segment.lastidx");
        segment.save(&path).unwrap();
        let loaded = AstSegment::load(&path).unwrap();

        assert_eq!(loaded.defs.len(), 1);
        assert_eq!(loaded.refs.len(), 2);
        let resolved = loaded.refs.iter().find(|r| r.name == "target_symbol").unwrap();
        assert_eq!(resolved.to_def_slot, Some(0));
        let unresolved = loaded.refs.iter().find(|r| r.name == "other_call").unwrap();
        assert!(unresolved.to_def_slot.is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_foreign_magic() {
        let dir = std::env::temp_dir().join(format!("lastidx-foreign-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("foreign.lastidx");
        std::fs::write(&path, b"NOT-A-REAL-AST-SEGMENT......").unwrap();
        assert!(AstSegment::load(&path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_corrupted_segment() {
        let dir = std::env::temp_dir().join(format!("lastidx-corrupt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let only = dir.join("only.py");
        std::fs::write(&only, "def target_symbol():\n    return 1\n").unwrap();
        let facts = vec![FileFacts {
            path: only.to_string_lossy().into_owned(),
            parse_ok: true,
            defs: vec![DefFact { name: "target_symbol".into(), line: 1, kind: "function".into() }],
            refs: vec![],
            calls: vec![],
            imports: vec![],
        }];
        let segment = AstSegment::build(facts).unwrap();
        let good = dir.join("segment.lastidx");
        segment.save(&good).unwrap();

        let mut raw = std::fs::read(&good).unwrap();
        let last = raw.len() - 1;
        raw[last] ^= 0xFF;
        std::fs::write(&good, &raw).unwrap();
        assert!(AstSegment::load(&good).is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_oversized_file_table_count_before_allocating() {
        let dir = std::env::temp_dir().join(format!("lastidx-oversized-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let only = dir.join("only.py");
        std::fs::write(&only, "x = 1\n").unwrap();
        let facts = vec![FileFacts {
            path: only.to_string_lossy().into_owned(),
            parse_ok: true,
            defs: vec![],
            refs: vec![],
            calls: vec![],
            imports: vec![],
        }];
        let segment = AstSegment::build(facts).unwrap();
        let good = dir.join("segment.lastidx");
        segment.save(&good).unwrap();

        let mut raw = std::fs::read(&good).unwrap();
        let n_off = 12 + 1;
        raw[n_off..n_off + 4].copy_from_slice(&0xFFFF_FFF0u32.to_le_bytes());
        let payload = raw[12..].to_vec();
        let new_crc = crc32fast::hash(&payload);
        raw[8..12].copy_from_slice(&new_crc.to_le_bytes());
        std::fs::write(&good, &raw).unwrap();

        let result = AstSegment::load(&good);
        assert!(
            result.is_err(),
            "an oversized declared file_table count must fail cleanly, not attempt a huge allocation"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
