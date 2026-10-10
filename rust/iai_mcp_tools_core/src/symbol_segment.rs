//! LSYMIDX1 symbol segment: sibling of segment.rs's LCODIDX1 whole-file
//! format, mirrored by copy for the magic+CRC32+atomic-rename discipline
//! only -- never imports segment.rs's format-specific types.
//!
//! v1 payload is frozen at first write: version byte, file table (path +
//! content hash + mtime + size), path arena, a symbol-name postings CSR
//! (feature_id keyed via `tokenize()` over each extracted symbol name,
//! sub-tokens included), then an exact-name table of
//! (feature_id-of-whole-lowercased-name, doc slot, kind) entries -- the
//! kind byte stored inline per entry doubles as the per-symbol kind array.
//! Every section ships now even though this tracer slice only queries the
//! exact-name/kind tier, so a later plan can light up the sub-token BM25
//! tier over the same bytes without a version bump.
//!
//! Fail-closed gate mirrors segment.rs's full shape: magic + CRC +
//! version, a bounds-before-allocate `need()` check ahead of every
//! `Vec`/`HashMap::with_capacity`, an offset-monotonicity check, and a
//! range check on every decoded cross-reference (postings doc-slot,
//! token_lookup compact_id, exact-entry doc-slot).

use std::collections::HashMap;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use thiserror::Error;

use crate::postings::{rebuild_postings_csr, PostingsCsr};
use crate::segmentset::Tombstone;
use crate::symbol_scan::{scan_file_for_language, Language, SymbolKind};
use crate::tokenize::{feature_id, tokenize};

const MAGIC: &[u8; 8] = b"LSYMIDX1";
const VERSION: u8 = 1;

#[cfg(test)]
const SYMBOL_SEGMENT_FILE: &str = "symbol.lsidx";

// Test-only counter on the segment-read path: incremented once per
// `SymbolSegment::load` call, so a test can assert an incremental append
// never opens a prior segment's file. Thread-local, not a shared global --
// mirrors segment.rs's own counter, since `cargo test` runs test functions
// concurrently on separate threads.
#[cfg(test)]
thread_local! {
    static SYMBOL_SEGMENT_LOAD_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_segment_load_count() {
    SYMBOL_SEGMENT_LOAD_COUNT.with(|c| c.set(0));
}

#[cfg(test)]
pub(crate) fn segment_load_count() -> usize {
    SYMBOL_SEGMENT_LOAD_COUNT.with(|c| c.get())
}

#[derive(Debug, Error)]
pub enum SymbolSegmentError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("symbol segment corrupt: {0}")]
    Corrupt(String),
}

pub(crate) struct SymbolFileMeta {
    pub(crate) path: String,
    #[allow(dead_code)] // round-tripped for parity with the whole-file lane; unread by this slice's search
    pub(crate) content_hash: u64,
    #[allow(dead_code)]
    pub(crate) mtime: u64,
    #[allow(dead_code)]
    pub(crate) size: u64,
}

pub(crate) struct ExactEntry {
    pub(crate) feature_id: u32,
    pub(crate) doc_slot: u32,
    pub(crate) kind: SymbolKind,
}

pub(crate) struct SymbolSegment {
    pub(crate) files: Vec<SymbolFileMeta>,
    pub(crate) postings: PostingsCsr,
    pub(crate) exact_entries: Vec<ExactEntry>,
}

/// FNV-1a 64-bit over raw file bytes -- copied from segment.rs's own
/// content-identity check (never a cryptographic digest).
fn content_hash64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

fn kind_to_byte(kind: SymbolKind) -> u8 {
    match kind {
        SymbolKind::Def => 0,
        SymbolKind::Unknown => 1,
    }
}

fn byte_to_kind(b: u8) -> Result<SymbolKind, SymbolSegmentError> {
    match b {
        0 => Ok(SymbolKind::Def),
        1 => Ok(SymbolKind::Unknown),
        other => Err(SymbolSegmentError::Corrupt(format!(
            "unknown symbol kind byte {other}"
        ))),
    }
}

/// Dispatches the per-language def matcher by the file's path suffix. Any
/// extension outside the three matched languages falls back to the Python
/// matcher, matching this build's pre-dispatch behavior for those files.
fn language_for_path(path: &Path) -> Language {
    match path.extension().and_then(|e| e.to_str()) {
        Some("rs") => Language::Rust,
        Some("ts") | Some("tsx") | Some("js") | Some("jsx") | Some("mjs") | Some("cjs") => {
            Language::TsJs
        }
        _ => Language::Python,
    }
}

impl SymbolSegment {
    /// Reads each given path's bytes ONCE, scans it for symbol candidates,
    /// and builds one in-memory segment. Pure -- performs no disk writes,
    /// never opens a prior segment's file.
    pub(crate) fn build(paths: &[PathBuf]) -> Result<SymbolSegment, SymbolSegmentError> {
        let mut token_freqs: Vec<HashMap<u32, u32>> = Vec::with_capacity(paths.len());
        let mut files: Vec<SymbolFileMeta> = Vec::with_capacity(paths.len());
        // Per-file (lowercased name -> strongest kind seen in that file);
        // Def wins over Unknown when a name occurs both as a definition
        // and a plain mention within the same file.
        let mut per_file_names: Vec<HashMap<String, SymbolKind>> = Vec::with_capacity(paths.len());

        for p in paths {
            let bytes = match fs::read(p) {
                Ok(b) => b,
                Err(err) => {
                    eprintln!(
                        "iai_mcp_tools_core: skipping unreadable file {} for symbol scan: {err}",
                        p.display()
                    );
                    continue;
                }
            };
            let text = match std::str::from_utf8(&bytes) {
                Ok(t) => t,
                Err(_) => {
                    eprintln!(
                        "iai_mcp_tools_core: skipping non-utf8 file {} for symbol scan",
                        p.display()
                    );
                    continue;
                }
            };

            let mut freq: HashMap<u32, u32> = HashMap::new();
            let mut names: HashMap<String, SymbolKind> = HashMap::new();
            for cand in scan_file_for_language(text, language_for_path(p)) {
                for tok in tokenize(&cand.name) {
                    *freq.entry(feature_id(&tok)).or_insert(0) += 1;
                }
                let lowered = cand.name.to_lowercase();
                let entry = names.entry(lowered).or_insert(SymbolKind::Unknown);
                if cand.kind == SymbolKind::Def {
                    *entry = SymbolKind::Def;
                }
            }

            let meta = fs::metadata(p)?;
            let mtime = meta
                .modified()?
                .duration_since(UNIX_EPOCH)
                .unwrap_or(std::time::Duration::ZERO)
                .as_secs();
            files.push(SymbolFileMeta {
                path: p.to_string_lossy().into_owned(),
                content_hash: content_hash64(&bytes),
                mtime,
                size: bytes.len() as u64,
            });
            token_freqs.push(freq);
            per_file_names.push(names);
        }

        let postings = rebuild_postings_csr(files.len(), &token_freqs);

        let mut exact_entries: Vec<ExactEntry> = Vec::new();
        for (slot, names) in per_file_names.into_iter().enumerate() {
            for (name, kind) in names {
                exact_entries.push(ExactEntry {
                    feature_id: feature_id(&name),
                    doc_slot: slot as u32,
                    kind,
                });
            }
        }
        // Deterministic order so save()/load() round-trip is byte-stable.
        exact_entries.sort_by_key(|e| (e.feature_id, e.doc_slot));

        Ok(SymbolSegment {
            files,
            postings,
            exact_entries,
        })
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

    /// Builds a new symbol segment from `paths`, skipping any path whose
    /// content hash matches `prior_index` (unchanged -- the append-time
    /// no-op) and returning the prior (segment_id, file_id) of any path
    /// whose hash differs, for the caller to tombstone. Reads and scans
    /// each path once, via the same read+scan `build` uses; never opens a
    /// prior segment file.
    pub(crate) fn build_incremental(
        paths: &[PathBuf],
        prior_index: &HashMap<String, (u32, u32, u64)>,
    ) -> Result<(SymbolSegment, Vec<Tombstone>), SymbolSegmentError> {
        let mut token_freqs: Vec<HashMap<u32, u32>> = Vec::new();
        let mut files: Vec<SymbolFileMeta> = Vec::new();
        let mut per_file_names: Vec<HashMap<String, SymbolKind>> = Vec::new();
        let mut tombstones: Vec<Tombstone> = Vec::new();

        for p in paths {
            let bytes = match fs::read(p) {
                Ok(b) => b,
                Err(err) => {
                    eprintln!(
                        "iai_mcp_tools_core: skipping unreadable file {} for symbol scan: {err}",
                        p.display()
                    );
                    continue;
                }
            };
            let text = match std::str::from_utf8(&bytes) {
                Ok(t) => t,
                Err(_) => {
                    eprintln!(
                        "iai_mcp_tools_core: skipping non-utf8 file {} for symbol scan",
                        p.display()
                    );
                    continue;
                }
            };

            let content_hash = content_hash64(&bytes);
            let path_str = p.to_string_lossy().into_owned();
            if let Some(&(prior_segment_id, prior_file_id, prior_hash)) =
                prior_index.get(&path_str)
            {
                if prior_hash == content_hash {
                    continue;
                }
                tombstones.push((prior_segment_id, prior_file_id));
            }

            let mut freq: HashMap<u32, u32> = HashMap::new();
            let mut names: HashMap<String, SymbolKind> = HashMap::new();
            for cand in scan_file_for_language(text, language_for_path(p)) {
                for tok in tokenize(&cand.name) {
                    *freq.entry(feature_id(&tok)).or_insert(0) += 1;
                }
                let lowered = cand.name.to_lowercase();
                let entry = names.entry(lowered).or_insert(SymbolKind::Unknown);
                if cand.kind == SymbolKind::Def {
                    *entry = SymbolKind::Def;
                }
            }

            let meta = fs::metadata(p)?;
            let mtime = meta
                .modified()?
                .duration_since(UNIX_EPOCH)
                .unwrap_or(std::time::Duration::ZERO)
                .as_secs();
            files.push(SymbolFileMeta {
                path: path_str,
                content_hash,
                mtime,
                size: bytes.len() as u64,
            });
            token_freqs.push(freq);
            per_file_names.push(names);
        }

        let postings = rebuild_postings_csr(files.len(), &token_freqs);

        let mut exact_entries: Vec<ExactEntry> = Vec::new();
        for (slot, names) in per_file_names.into_iter().enumerate() {
            for (name, kind) in names {
                exact_entries.push(ExactEntry {
                    feature_id: feature_id(&name),
                    doc_slot: slot as u32,
                    kind,
                });
            }
        }
        // Deterministic order so save()/load() round-trip is byte-stable.
        exact_entries.sort_by_key(|e| (e.feature_id, e.doc_slot));

        Ok((
            SymbolSegment {
                files,
                postings,
                exact_entries,
            },
            tombstones,
        ))
    }

    pub(crate) fn save(&self, path: &Path) -> Result<(), SymbolSegmentError> {
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

        let token_lookup_len = self.postings.token_lookup.len() as u32;
        payload.extend_from_slice(&token_lookup_len.to_le_bytes());
        let mut lookup_entries: Vec<(u32, u32)> = self
            .postings
            .token_lookup
            .iter()
            .map(|(&fid, &cid)| (fid, cid))
            .collect();
        lookup_entries.sort_by_key(|&(_, cid)| cid);
        for (fid, cid) in &lookup_entries {
            payload.extend_from_slice(&fid.to_le_bytes());
            payload.extend_from_slice(&cid.to_le_bytes());
        }
        for off in &self.postings.post_offsets {
            payload.extend_from_slice(&off.to_le_bytes());
        }
        for slot in &self.postings.post_doc_slots {
            payload.extend_from_slice(&slot.to_le_bytes());
        }
        for tf in &self.postings.post_tfs {
            payload.extend_from_slice(&tf.to_le_bytes());
        }
        for dl in &self.postings.doc_len {
            payload.extend_from_slice(&dl.to_le_bytes());
        }
        payload.extend_from_slice(&self.postings.n_docs.to_le_bytes());
        payload.extend_from_slice(&self.postings.avg_len.to_le_bytes());
        payload.extend_from_slice(&self.postings.total_len.to_le_bytes());

        payload.extend_from_slice(&(self.exact_entries.len() as u32).to_le_bytes());
        for e in &self.exact_entries {
            payload.extend_from_slice(&e.feature_id.to_le_bytes());
            payload.extend_from_slice(&e.doc_slot.to_le_bytes());
            payload.push(kind_to_byte(e.kind));
        }

        let crc = crc32fast::hash(&payload);

        let tmp = path.with_extension("tmp-symidx");
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

    pub(crate) fn load(path: &Path) -> Result<Self, SymbolSegmentError> {
        #[cfg(test)]
        SYMBOL_SEGMENT_LOAD_COUNT.with(|c| c.set(c.get() + 1));

        let raw = fs::read(path)?;
        if raw.len() < 12 || &raw[..8] != MAGIC {
            return Err(SymbolSegmentError::Corrupt("not an LSYMIDX1 segment".into()));
        }
        let stored_crc = u32::from_le_bytes(raw[8..12].try_into().unwrap());
        let payload = &raw[12..];
        if crc32fast::hash(payload) != stored_crc {
            return Err(SymbolSegmentError::Corrupt("CRC mismatch".into()));
        }

        let need = |off: usize, len: usize| -> Result<(), SymbolSegmentError> {
            if off + len > payload.len() {
                Err(SymbolSegmentError::Corrupt("truncated payload".into()))
            } else {
                Ok(())
            }
        };

        let mut off = 0usize;
        need(off, 1)?;
        let version = payload[off];
        off += 1;
        // Fail closed -- never forward-skip an unrecognized tail.
        if version != VERSION {
            return Err(SymbolSegmentError::Corrupt(format!(
                "unsupported symbol segment version {version}"
            )));
        }

        need(off, 4)?;
        let n = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        need(off, n.saturating_mul(30))?;

        struct RawEntry {
            path_offset: u32,
            path_len: u16,
            content_hash: u64,
            mtime: u64,
            size: u64,
        }
        let mut file_table: Vec<RawEntry> = Vec::with_capacity(n);
        for _ in 0..n {
            need(off, 30)?;
            let path_offset = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap());
            let path_len = u16::from_le_bytes(payload[off + 4..off + 6].try_into().unwrap());
            let content_hash = u64::from_le_bytes(payload[off + 6..off + 14].try_into().unwrap());
            let mtime = u64::from_le_bytes(payload[off + 14..off + 22].try_into().unwrap());
            let size = u64::from_le_bytes(payload[off + 22..off + 30].try_into().unwrap());
            file_table.push(RawEntry {
                path_offset,
                path_len,
                content_hash,
                mtime,
                size,
            });
            off += 30;
        }

        let arena_len = file_table
            .iter()
            .map(|e| e.path_offset as usize + e.path_len as usize)
            .max()
            .unwrap_or(0);
        need(off, arena_len)?;
        let arena = &payload[off..off + arena_len];
        off += arena_len;

        let mut files = Vec::with_capacity(n);
        for e in &file_table {
            let bytes = &arena[e.path_offset as usize..e.path_offset as usize + e.path_len as usize];
            let s = std::str::from_utf8(bytes)
                .map_err(|_| SymbolSegmentError::Corrupt("non-utf8 path".into()))?;
            files.push(SymbolFileMeta {
                path: s.to_string(),
                content_hash: e.content_hash,
                mtime: e.mtime,
                size: e.size,
            });
        }

        need(off, 4)?;
        let token_lookup_len =
            u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        need(off, token_lookup_len.saturating_mul(8))?;

        let mut token_lookup: HashMap<u32, u32> = HashMap::with_capacity(token_lookup_len);
        for _ in 0..token_lookup_len {
            need(off, 8)?;
            let fid = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap());
            let cid = u32::from_le_bytes(payload[off + 4..off + 8].try_into().unwrap());
            token_lookup.insert(fid, cid);
            off += 8;
        }

        need(off, (token_lookup_len + 1) * 4)?;
        let mut post_offsets = Vec::with_capacity(token_lookup_len + 1);
        for _ in 0..(token_lookup_len + 1) {
            post_offsets.push(u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()));
            off += 4;
        }

        let total_postings = *post_offsets.last().unwrap_or(&0) as usize;
        need(off, total_postings * 4)?;
        let mut post_doc_slots = Vec::with_capacity(total_postings);
        for _ in 0..total_postings {
            post_doc_slots.push(u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()));
            off += 4;
        }
        need(off, total_postings * 4)?;
        let mut post_tfs = Vec::with_capacity(total_postings);
        for _ in 0..total_postings {
            post_tfs.push(u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()));
            off += 4;
        }

        need(off, n * 4)?;
        let mut doc_len = Vec::with_capacity(n);
        for _ in 0..n {
            doc_len.push(u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()));
            off += 4;
        }

        need(off, 4)?;
        let n_docs = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap());
        off += 4;
        need(off, 8)?;
        let avg_len = f64::from_le_bytes(payload[off..off + 8].try_into().unwrap());
        off += 8;
        need(off, 8)?;
        let total_len = u64::from_le_bytes(payload[off..off + 8].try_into().unwrap());
        off += 8;

        need(off, 4)?;
        let exact_len = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        need(off, exact_len.saturating_mul(9))?;
        let mut exact_entries = Vec::with_capacity(exact_len);
        for _ in 0..exact_len {
            need(off, 9)?;
            let entry_fid = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap());
            let doc_slot = u32::from_le_bytes(payload[off + 4..off + 8].try_into().unwrap());
            let kind = byte_to_kind(payload[off + 8])?;
            exact_entries.push(ExactEntry {
                feature_id: entry_fid,
                doc_slot,
                kind,
            });
            off += 9;
        }

        if off != payload.len() {
            return Err(SymbolSegmentError::Corrupt(
                "trailing bytes after payload".into(),
            ));
        }

        // Referential integrity: every decoded cross-reference must stay
        // in-bounds relative to the arrays it indexes into, checked here
        // so a corrupt-but-CRC-valid file fails closed on load rather than
        // panicking downstream in `symbol_index::search`/`symbol_compaction::compact`.
        // Monotonicity alone bounds every `post_offsets` entry within
        // `post_doc_slots`/`post_tfs`, since those are sized exactly from
        // `post_offsets.last()` (`total_postings` above).
        if post_offsets.windows(2).any(|w| w[0] > w[1]) {
            return Err(SymbolSegmentError::Corrupt(
                "non-monotonic post_offsets".into(),
            ));
        }
        if token_lookup
            .values()
            .any(|&cid| cid as usize + 1 >= post_offsets.len())
        {
            return Err(SymbolSegmentError::Corrupt(
                "token_lookup compact_id out of range".into(),
            ));
        }
        if post_doc_slots.iter().any(|&slot| slot as usize >= n) {
            return Err(SymbolSegmentError::Corrupt(
                "post_doc_slots slot out of range".into(),
            ));
        }
        if exact_entries.iter().any(|e| e.doc_slot as usize >= n) {
            return Err(SymbolSegmentError::Corrupt(
                "exact_entries doc_slot out of range".into(),
            ));
        }
        // `n_docs` is an independently-stored count consumed by
        // `symbol_compaction::csr_to_doc_freqs` to size its working array;
        // every other cross-reference above is bounds-checked against `n`,
        // but a segment whose `n_docs` diverges from `n` would still load
        // successfully and panic downstream instead of failing closed here.
        if n_docs as usize != n {
            return Err(SymbolSegmentError::Corrupt(
                "n_docs does not match file table count".into(),
            ));
        }

        Ok(SymbolSegment {
            files,
            postings: PostingsCsr {
                token_lookup,
                post_offsets,
                post_doc_slots,
                post_tfs,
                doc_len,
                n_docs,
                avg_len,
                total_len,
            },
            exact_entries,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_save_load_roundtrip_carries_def_kind() {
        let dir = std::env::temp_dir().join(format!("lsymidx-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let f1 = dir.join("alpha.py");
        std::fs::write(&f1, "def zebracorn_finder():\n    return 1\n").unwrap();
        let f2 = dir.join("beta.py");
        std::fs::write(&f2, "result = zebracorn_finder()\n").unwrap();

        let segment = SymbolSegment::build(&[f1.clone(), f2.clone()]).unwrap();
        let path = dir.join(SYMBOL_SEGMENT_FILE);
        segment.save(&path).unwrap();
        let loaded = SymbolSegment::load(&path).unwrap();

        let fid = feature_id("zebracorn_finder");
        let mut matches: Vec<&ExactEntry> = loaded
            .exact_entries
            .iter()
            .filter(|e| e.feature_id == fid)
            .collect();
        matches.sort_by_key(|e| e.doc_slot);
        assert_eq!(matches.len(), 2, "the name occurs in both files");
        let def_entry = matches
            .iter()
            .find(|e| e.kind == SymbolKind::Def)
            .expect("one entry must be Def");
        assert_eq!(loaded.files[def_entry.doc_slot as usize].path, f1.to_string_lossy());
        let unknown_entry = matches
            .iter()
            .find(|e| e.kind == SymbolKind::Unknown)
            .expect("one entry must be Unknown");
        assert_eq!(
            loaded.files[unknown_entry.doc_slot as usize].path,
            f2.to_string_lossy()
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_n_docs_mismatched_with_file_table_count() {
        let dir = std::env::temp_dir().join(format!("lsymidx-ndocs-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let f1 = dir.join("alpha.py");
        std::fs::write(&f1, "def zebracorn_finder():\n    return 1\n").unwrap();
        let f2 = dir.join("beta.py");
        std::fs::write(&f2, "result = zebracorn_finder()\n").unwrap();

        let mut segment = SymbolSegment::build(&[f1.clone(), f2.clone()]).unwrap();
        // Forge a divergence between the independently-stored `n_docs` and
        // the file-table count `n` -- `save()` recomputes the CRC over this
        // already-forged payload, so the resulting file passes the CRC gate
        // exactly like the corruption shape the byte-flip test above cannot
        // reach (that one fails at the CRC check itself, before `load()`
        // ever decodes this far).
        segment.postings.n_docs = segment.files.len() as u32 - 1;

        let path = dir.join(SYMBOL_SEGMENT_FILE);
        segment.save(&path).unwrap();

        let result = SymbolSegment::load(&path);
        assert!(
            matches!(result, Err(SymbolSegmentError::Corrupt(_))),
            "load() must fail closed on n_docs != n, not return Ok and panic downstream in compaction"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_dispatches_language_by_path_suffix_and_tags_def_correctly() {
        let dir = std::env::temp_dir().join(format!("lsymidx-lang-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let rust_def = dir.join("alpha.rs");
        std::fs::write(&rust_def, "pub fn zebracorn_rust_def() {}\n").unwrap();
        let rust_mention = dir.join("alpha_caller.rs");
        std::fs::write(&rust_mention, "let _ = zebracorn_rust_def();\n").unwrap();

        let ts_def = dir.join("beta.ts");
        std::fs::write(&ts_def, "export function zebracorn_ts_def() {}\n").unwrap();
        let ts_mention = dir.join("beta_caller.ts");
        std::fs::write(&ts_mention, "zebracorn_ts_def();\n").unwrap();

        let segment = SymbolSegment::build(&[
            rust_def.clone(),
            rust_mention.clone(),
            ts_def.clone(),
            ts_mention.clone(),
        ])
        .unwrap();

        let rust_fid = feature_id("zebracorn_rust_def");
        let mut rust_matches: Vec<&ExactEntry> = segment
            .exact_entries
            .iter()
            .filter(|e| e.feature_id == rust_fid)
            .collect();
        rust_matches.sort_by_key(|e| e.doc_slot);
        assert_eq!(rust_matches.len(), 2, "the name occurs in both rust files");
        assert!(rust_matches.iter().any(|e| e.kind == SymbolKind::Def
            && segment.files[e.doc_slot as usize].path == rust_def.to_string_lossy()));
        assert!(rust_matches.iter().any(|e| e.kind == SymbolKind::Unknown
            && segment.files[e.doc_slot as usize].path == rust_mention.to_string_lossy()));

        let ts_fid = feature_id("zebracorn_ts_def");
        let mut ts_matches: Vec<&ExactEntry> = segment
            .exact_entries
            .iter()
            .filter(|e| e.feature_id == ts_fid)
            .collect();
        ts_matches.sort_by_key(|e| e.doc_slot);
        assert_eq!(ts_matches.len(), 2, "the name occurs in both ts files");
        assert!(ts_matches.iter().any(|e| e.kind == SymbolKind::Def
            && segment.files[e.doc_slot as usize].path == ts_def.to_string_lossy()));
        assert!(ts_matches.iter().any(|e| e.kind == SymbolKind::Unknown
            && segment.files[e.doc_slot as usize].path == ts_mention.to_string_lossy()));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_foreign_magic() {
        let dir = std::env::temp_dir().join(format!("lsymidx-foreign-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("foreign.lsidx");
        std::fs::write(&path, b"NOT-A-REAL-SYMBOL-SEGMENT..").unwrap();
        assert!(SymbolSegment::load(&path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_corrupted_segment() {
        let dir = std::env::temp_dir().join(format!("lsymidx-corrupt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let only = dir.join("only.py");
        std::fs::write(&only, "def helper():\n    pass\n").unwrap();
        let segment = SymbolSegment::build(&[only]).unwrap();
        let good = dir.join(SYMBOL_SEGMENT_FILE);
        segment.save(&good).unwrap();

        let mut raw = std::fs::read(&good).unwrap();
        let last = raw.len() - 1;
        raw[last] ^= 0xFF;
        std::fs::write(&good, &raw).unwrap();
        assert!(SymbolSegment::load(&good).is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_unsupported_version() {
        let dir = std::env::temp_dir().join(format!("lsymidx-version-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let only = dir.join("only.py");
        std::fs::write(&only, "def helper():\n    pass\n").unwrap();
        let segment = SymbolSegment::build(&[only]).unwrap();
        let good = dir.join(SYMBOL_SEGMENT_FILE);
        segment.save(&good).unwrap();

        let mut raw = std::fs::read(&good).unwrap();
        raw[12] = 2; // payload's first byte, right after magic(8)+crc(4)
        let payload = raw[12..].to_vec();
        let new_crc = crc32fast::hash(&payload);
        raw[8..12].copy_from_slice(&new_crc.to_le_bytes());
        std::fs::write(&good, &raw).unwrap();

        let result = SymbolSegment::load(&good);
        match result {
            Err(SymbolSegmentError::Corrupt(msg)) => assert!(msg.contains("version")),
            Err(other) => panic!("expected a version-corrupt error, got: {other}"),
            Ok(_) => panic!("expected load to reject an unsupported version"),
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_oversized_file_table_count_before_allocating() {
        let dir =
            std::env::temp_dir().join(format!("lsymidx-oversized-n-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let only = dir.join("only.py");
        std::fs::write(&only, "def helper():\n    pass\n").unwrap();
        let segment = SymbolSegment::build(&[only]).unwrap();
        let good = dir.join(SYMBOL_SEGMENT_FILE);
        segment.save(&good).unwrap();

        let mut raw = std::fs::read(&good).unwrap();
        // payload[0] is the version byte; payload[1..5] is the file_table
        // count `n` -- payload itself starts at raw[12] (magic+crc).
        let n_off = 12 + 1;
        raw[n_off..n_off + 4].copy_from_slice(&0xFFFF_FFF0u32.to_le_bytes());
        let payload = raw[12..].to_vec();
        let new_crc = crc32fast::hash(&payload);
        raw[8..12].copy_from_slice(&new_crc.to_le_bytes());
        std::fs::write(&good, &raw).unwrap();

        // A declared count this large would try to reserve ~128GB via
        // `Vec::with_capacity` pre-fix; the length check must reject it
        // before that allocation is ever attempted.
        let result = SymbolSegment::load(&good);
        assert!(
            result.is_err(),
            "an oversized declared file_table count must fail cleanly, not attempt a huge allocation"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_out_of_range_postings_doc_slot_reference() {
        let dir = std::env::temp_dir().join(format!("lsymidx-oob-slot-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f1 = dir.join("one.py");
        std::fs::write(&f1, "def alpha_marker():\n    pass\n").unwrap();
        let f2 = dir.join("two.py");
        std::fs::write(&f2, "alpha_marker()\n").unwrap();
        let mut segment = SymbolSegment::build(&[f1, f2]).unwrap();

        // Corrupt a valid postings entry to reference a doc slot that does
        // not exist -- `save` will still produce a CRC-valid file, since
        // the CRC covers whatever the in-memory payload actually is.
        assert!(!segment.postings.post_doc_slots.is_empty());
        segment.postings.post_doc_slots[0] = 9999;
        let good = dir.join(SYMBOL_SEGMENT_FILE);
        segment.save(&good).unwrap();

        let result = SymbolSegment::load(&good);
        assert!(
            result.is_err(),
            "an out-of-range post_doc_slots reference must fail cleanly at load, not panic downstream in search/compaction"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_out_of_range_exact_entry_doc_slot_reference() {
        let dir = std::env::temp_dir().join(format!("lsymidx-oob-exact-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f1 = dir.join("one.py");
        std::fs::write(&f1, "def alpha_marker():\n    pass\n").unwrap();
        let mut segment = SymbolSegment::build(&[f1]).unwrap();

        assert!(!segment.exact_entries.is_empty());
        segment.exact_entries[0].doc_slot = 9999;
        let good = dir.join(SYMBOL_SEGMENT_FILE);
        segment.save(&good).unwrap();

        let result = SymbolSegment::load(&good);
        assert!(
            result.is_err(),
            "an out-of-range exact_entries doc_slot reference must fail cleanly at load, not panic downstream"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_trailing_bytes_after_payload() {
        let dir = std::env::temp_dir().join(format!("lsymidx-trailing-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let only = dir.join("only.py");
        std::fs::write(&only, "def helper():\n    pass\n").unwrap();
        let segment = SymbolSegment::build(&[only]).unwrap();
        let good = dir.join(SYMBOL_SEGMENT_FILE);
        segment.save(&good).unwrap();

        let mut raw = std::fs::read(&good).unwrap();
        let payload_with_trailer: Vec<u8> =
            raw[12..].iter().copied().chain(std::iter::once(0u8)).collect();
        let new_crc = crc32fast::hash(&payload_with_trailer);
        raw.truncate(8);
        raw.extend_from_slice(&new_crc.to_le_bytes());
        raw.extend_from_slice(&payload_with_trailer);
        std::fs::write(&good, &raw).unwrap();

        let result = SymbolSegment::load(&good);
        match result {
            Err(SymbolSegmentError::Corrupt(msg)) => assert!(msg.contains("trailing")),
            Err(other) => panic!("expected a trailing-bytes-corrupt error, got: {other}"),
            Ok(_) => panic!("expected load to reject trailing bytes after the declared payload"),
        }

        std::fs::remove_dir_all(&dir).ok();
    }
}
