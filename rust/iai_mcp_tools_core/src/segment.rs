//! LCODEIDX1 segment: whole-file BM25 postings, persisted with the same
//! magic+CRC32+atomic-rename discipline as iai_mcp_vec_core's LVECIDX1.
//!
//! v1 payload is frozen: version byte, file table (path + content hash +
//! mtime + size), path arena, then the whole-file postings CSR. A future
//! version appends new sections behind a version bump; this reader hard-
//! rejects any version other than 1 rather than skip-parse an unknown tail.
//!
//! One segment == one immutable file. `Segment::build` never writes to
//! disk and `Segment::save`/`Segment::load` take an explicit path, so the
//! incremental engine (index.rs) can give each batch's segment its own
//! file name under a shared store root -- append writes exactly one new
//! file, it never touches a prior one.

use std::collections::HashMap;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use thiserror::Error;

use crate::postings::{rebuild_postings_csr, PostingsCsr};
use crate::segmentset::Tombstone;
use crate::tokenize::{feature_id, tokenize};

const MAGIC: &[u8; 8] = b"LCODIDX1";
const VERSION: u8 = 1;

#[cfg(test)]
const SEGMENT_FILE: &str = "segment.lcidx";

// Test-only counter on the segment-read path: incremented once per
// `Segment::load` call, so a test can assert an incremental append never
// opens a prior segment's file. Thread-local, not a shared global --
// `cargo test` runs test functions concurrently on separate threads, and
// a single crate-wide counter would pick up unrelated tests' loads.
#[cfg(test)]
thread_local! {
    static SEGMENT_LOAD_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_segment_load_count() {
    SEGMENT_LOAD_COUNT.with(|c| c.set(0));
}

#[cfg(test)]
pub(crate) fn segment_load_count() -> usize {
    SEGMENT_LOAD_COUNT.with(|c| c.get())
}

#[derive(Debug, Error)]
pub enum SegmentError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("segment corrupt: {0}")]
    Corrupt(String),
}

pub(crate) struct FileMeta {
    pub(crate) path: String,
    pub(crate) content_hash: u64,
    pub(crate) mtime: u64,
    pub(crate) size: u64,
}

pub(crate) struct Segment {
    pub(crate) files: Vec<FileMeta>,
    pub(crate) postings: PostingsCsr,
}

/// FNV-1a 64-bit over raw file bytes -- a segment-local content identity
/// check, never a cryptographic digest.
fn content_hash64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

impl Segment {
    /// Tokenizes every given path's content into ONE in-memory segment.
    /// Pure -- performs no disk writes.
    pub(crate) fn build(paths: &[PathBuf]) -> Result<Segment, SegmentError> {
        let mut token_freqs: Vec<HashMap<u32, u32>> = Vec::with_capacity(paths.len());
        let mut files: Vec<FileMeta> = Vec::with_capacity(paths.len());
        for p in paths {
            // A file that races non-UTF-8/unreadable between an upstream
            // pre-filter and this read must not abort the whole batch --
            // skip just that file, keep indexing the rest.
            let (freq, content_hash, mtime, size) = match read_file(p) {
                Ok(v) => v,
                Err(err) => {
                    eprintln!(
                        "iai_mcp_tools_core: skipping unreadable file {}: {err}",
                        p.display()
                    );
                    continue;
                }
            };
            token_freqs.push(freq);
            files.push(FileMeta {
                path: p.to_string_lossy().into_owned(),
                content_hash,
                mtime,
                size,
            });
        }
        let postings = rebuild_postings_csr(files.len(), &token_freqs);
        Ok(Segment { files, postings })
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

    /// Builds a new segment from `paths`, skipping any path whose content
    /// hash matches `prior_index` (unchanged -- the append-time no-op) and
    /// returning the prior (segment_id, file_id) of any path whose hash
    /// differs, for the caller to tombstone. Reads each path's bytes once,
    /// via the same `read_file` call `build` uses; never opens a prior
    /// segment file.
    pub(crate) fn build_incremental(
        paths: &[PathBuf],
        prior_index: &HashMap<String, (u32, u32, u64)>,
    ) -> Result<(Segment, Vec<Tombstone>), SegmentError> {
        let mut token_freqs: Vec<HashMap<u32, u32>> = Vec::new();
        let mut files: Vec<FileMeta> = Vec::new();
        let mut tombstones: Vec<Tombstone> = Vec::new();

        for p in paths {
            let (freq, content_hash, mtime, size) = match read_file(p) {
                Ok(v) => v,
                Err(err) => {
                    eprintln!(
                        "iai_mcp_tools_core: skipping unreadable file {}: {err}",
                        p.display()
                    );
                    continue;
                }
            };
            let path_str = p.to_string_lossy().into_owned();
            if let Some(&(prior_segment_id, prior_file_id, prior_hash)) =
                prior_index.get(&path_str)
            {
                if prior_hash == content_hash {
                    continue;
                }
                tombstones.push((prior_segment_id, prior_file_id));
            }
            token_freqs.push(freq);
            files.push(FileMeta {
                path: path_str,
                content_hash,
                mtime,
                size,
            });
        }
        let postings = rebuild_postings_csr(files.len(), &token_freqs);
        Ok((Segment { files, postings }, tombstones))
    }

    pub(crate) fn save(&self, path: &Path) -> Result<(), SegmentError> {
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

        // Deterministic order (sorted by compact_index) so load() rebuilds
        // the same HashMap contents without depending on iteration order.
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

        // CRC over the payload only -- magic is written and checked
        // separately, matching LVECIDX1's exact convention.
        let crc = crc32fast::hash(&payload);

        let tmp = path.with_extension("tmp-codeidx");
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

    pub(crate) fn load(path: &Path) -> Result<Self, SegmentError> {
        #[cfg(test)]
        SEGMENT_LOAD_COUNT.with(|c| c.set(c.get() + 1));

        let raw = fs::read(path)?;
        if raw.len() < 12 || &raw[..8] != MAGIC {
            return Err(SegmentError::Corrupt("not an LCODEIDX1 segment".into()));
        }
        let stored_crc = u32::from_le_bytes(raw[8..12].try_into().unwrap());
        let payload = &raw[12..];
        if crc32fast::hash(payload) != stored_crc {
            return Err(SegmentError::Corrupt("CRC mismatch".into()));
        }

        let need = |off: usize, len: usize| -> Result<(), SegmentError> {
            if off + len > payload.len() {
                Err(SegmentError::Corrupt("truncated payload".into()))
            } else {
                Ok(())
            }
        };

        let mut off = 0usize;
        need(off, 1)?;
        let version = payload[off];
        off += 1;
        // Fail closed on any version this reader does not understand --
        // never forward-skip an unrecognized tail.
        if version != VERSION {
            return Err(SegmentError::Corrupt(format!(
                "unsupported segment version {version}"
            )));
        }

        need(off, 4)?;
        let n = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        // Reject before allocating -- a corrupt/adversarial `n` must fail
        // closed here, never reach `Vec::with_capacity` (whose OOM path is
        // an uncatchable process abort, not a catchable panic).
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
                .map_err(|_| SegmentError::Corrupt("non-utf8 path".into()))?;
            files.push(FileMeta {
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
        // Same before-allocation bound as `file_table` above.
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

        if off != payload.len() {
            return Err(SegmentError::Corrupt("trailing bytes after payload".into()));
        }

        // Referential integrity: every decoded cross-reference must stay
        // in-bounds relative to the arrays it indexes into, checked here
        // so a corrupt-but-CRC-valid file fails closed on load rather than
        // panicking downstream in `index::search`/`compaction::compact`.
        // Monotonicity alone bounds every `post_offsets` entry within
        // `post_doc_slots`/`post_tfs`, since those are sized exactly from
        // `post_offsets.last()` (`total_postings` above).
        if post_offsets.windows(2).any(|w| w[0] > w[1]) {
            return Err(SegmentError::Corrupt("non-monotonic post_offsets".into()));
        }
        if token_lookup
            .values()
            .any(|&cid| cid as usize + 1 >= post_offsets.len())
        {
            return Err(SegmentError::Corrupt(
                "token_lookup compact_id out of range".into(),
            ));
        }
        if post_doc_slots.iter().any(|&slot| slot as usize >= n) {
            return Err(SegmentError::Corrupt(
                "post_doc_slots slot out of range".into(),
            ));
        }

        Ok(Segment {
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
        })
    }
}

fn read_file(path: &Path) -> Result<(HashMap<u32, u32>, u64, u64, u64), SegmentError> {
    let bytes = fs::read(path)?;
    let content_hash = content_hash64(&bytes);
    let size = bytes.len() as u64;
    let text = std::str::from_utf8(&bytes)
        .map_err(|_| SegmentError::Corrupt(format!("non-utf8 file content: {}", path.display())))?;
    let mut freq: HashMap<u32, u32> = HashMap::new();
    for tok in tokenize(text) {
        *freq.entry(feature_id(&tok)).or_insert(0) += 1;
    }
    let meta = fs::metadata(path)?;
    let mtime = meta
        .modified()?
        .duration_since(UNIX_EPOCH)
        .unwrap_or(std::time::Duration::ZERO)
        .as_secs();
    Ok((freq, content_hash, mtime, size))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_save_load_roundtrip_finds_owning_file() {
        let dir = std::env::temp_dir().join(format!("lcodeidx-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("store");
        std::fs::create_dir_all(&store_root).unwrap();

        let f1 = dir.join("alpha.txt");
        std::fs::write(&f1, "the quick brown fox jumps over the lazy dog").unwrap();
        let f2 = dir.join("beta.txt");
        std::fs::write(&f2, "completely unrelated content about spreadsheets").unwrap();
        let f3 = dir.join("gamma.txt");
        std::fs::write(&f3, "a zebra wanders across the savanna at dawn").unwrap();

        let segment = Segment::build(&[f1.clone(), f2.clone(), f3.clone()]).unwrap();
        segment.save(&store_root.join(SEGMENT_FILE)).unwrap();
        let loaded = Segment::load(&store_root.join(SEGMENT_FILE)).unwrap();

        let fid = feature_id("zebra");
        let compact_id = loaded.postings.token_lookup[&fid] as usize;
        let start = loaded.postings.post_offsets[compact_id] as usize;
        let end = loaded.postings.post_offsets[compact_id + 1] as usize;
        assert_eq!(end - start, 1, "zebra appears in exactly one file");
        let slot = loaded.postings.post_doc_slots[start];
        assert_eq!(loaded.files[slot as usize].path, f3.to_string_lossy());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn full_payload_roundtrips_byte_for_byte() {
        let dir = std::env::temp_dir().join(format!("lcodeidx-roundtrip-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("store");
        std::fs::create_dir_all(&store_root).unwrap();

        let f1 = dir.join("one.txt");
        std::fs::write(&f1, "cat dog").unwrap();
        let f2 = dir.join("two.txt");
        std::fs::write(&f2, "dog bird bird").unwrap();

        let segment = Segment::build(&[f1.clone(), f2.clone()]).unwrap();
        segment.save(&store_root.join(SEGMENT_FILE)).unwrap();
        let loaded = Segment::load(&store_root.join(SEGMENT_FILE)).unwrap();

        assert_eq!(loaded.files.len(), 2);
        assert_eq!(loaded.files[0].path, f1.to_string_lossy());
        assert_eq!(loaded.files[1].path, f2.to_string_lossy());

        // Independently recompute expected content_hash/mtime/size and
        // compare against what load() reconstructed from the payload.
        let (_, expected_hash_1, expected_mtime_1, expected_size_1) = read_file(&f1).unwrap();
        assert_eq!(loaded.files[0].content_hash, expected_hash_1);
        assert_eq!(loaded.files[0].mtime, expected_mtime_1);
        assert_eq!(loaded.files[0].size, expected_size_1);

        let (_, expected_hash_2, expected_mtime_2, expected_size_2) = read_file(&f2).unwrap();
        assert_eq!(loaded.files[1].content_hash, expected_hash_2);
        assert_eq!(loaded.files[1].mtime, expected_mtime_2);
        assert_eq!(loaded.files[1].size, expected_size_2);

        assert_eq!(loaded.postings.n_docs, 2);
        assert_eq!(loaded.postings.doc_len, vec![2, 3]);
        assert_eq!(loaded.postings.total_len, 5);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_corrupted_segment() {
        let dir = std::env::temp_dir().join(format!("lcodeidx-corrupt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let only = dir.join("only.txt");
        std::fs::write(&only, "hello world").unwrap();
        let segment = Segment::build(&[only]).unwrap();
        segment.save(&dir.join(SEGMENT_FILE)).unwrap();

        let good = dir.join(SEGMENT_FILE);
        let mut raw = std::fs::read(&good).unwrap();
        let last = raw.len() - 1;
        raw[last] ^= 0xFF;
        std::fs::write(&good, &raw).unwrap();
        assert!(Segment::load(&good).is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_foreign_magic() {
        let dir = std::env::temp_dir().join(format!("lcodeidx-foreign-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("foreign.lcidx");
        std::fs::write(&path, b"NOT-A-REAL-SEGMENT-FORMAT..").unwrap();
        assert!(Segment::load(&path).is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_unsupported_version() {
        let dir = std::env::temp_dir().join(format!("lcodeidx-version-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let only = dir.join("only.txt");
        std::fs::write(&only, "hello world").unwrap();
        let segment = Segment::build(&[only]).unwrap();
        segment.save(&dir.join(SEGMENT_FILE)).unwrap();

        let good = dir.join(SEGMENT_FILE);
        let mut raw = std::fs::read(&good).unwrap();
        // Payload starts right after magic(8) + crc(4); the version byte
        // is the payload's first byte.
        raw[12] = 2;
        let payload = raw[12..].to_vec();
        let new_crc = crc32fast::hash(&payload);
        raw[8..12].copy_from_slice(&new_crc.to_le_bytes());
        std::fs::write(&good, &raw).unwrap();

        let result = Segment::load(&good);
        match result {
            Err(SegmentError::Corrupt(msg)) => assert!(msg.contains("version")),
            Err(other) => panic!("expected a version-corrupt error, got: {other}"),
            Ok(_) => panic!("expected load to reject an unsupported version"),
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_oversized_file_table_count_before_allocating() {
        let dir =
            std::env::temp_dir().join(format!("lcodeidx-oversized-n-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let only = dir.join("only.txt");
        std::fs::write(&only, "hello world").unwrap();
        let segment = Segment::build(&[only]).unwrap();
        segment.save(&dir.join(SEGMENT_FILE)).unwrap();

        let good = dir.join(SEGMENT_FILE);
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
        let result = Segment::load(&good);
        assert!(
            result.is_err(),
            "an oversized declared file_table count must fail cleanly, not attempt a huge allocation"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_out_of_range_doc_slot_reference() {
        let dir = std::env::temp_dir().join(format!("lcodeidx-oob-slot-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f1 = dir.join("one.txt");
        std::fs::write(&f1, "cat dog").unwrap();
        let f2 = dir.join("two.txt");
        std::fs::write(&f2, "dog bird").unwrap();
        let mut segment = Segment::build(&[f1, f2]).unwrap();

        // Corrupt a valid postings entry to reference a doc slot that does
        // not exist -- `save` will still produce a CRC-valid file, since
        // the CRC covers whatever the in-memory payload actually is.
        assert!(!segment.postings.post_doc_slots.is_empty());
        segment.postings.post_doc_slots[0] = 9999;
        let good = dir.join(SEGMENT_FILE);
        segment.save(&good).unwrap();

        let result = Segment::load(&good);
        assert!(
            result.is_err(),
            "an out-of-range post_doc_slots reference must fail cleanly at load, not panic downstream in search/compaction"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_skips_individual_unreadable_or_non_utf8_file_without_aborting_batch() {
        let dir = std::env::temp_dir().join(format!("lcodeidx-toctou-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let f1 = dir.join("good1.txt");
        std::fs::write(&f1, "hello world").unwrap();
        // Never created -- simulates a file that raced away (deleted)
        // between an upstream pre-filter and this read.
        let missing = dir.join("does-not-exist.txt");
        let bad = dir.join("bad.bin");
        std::fs::write(&bad, [0xFFu8, 0xFE, 0x00, 0x01]).unwrap();
        let f2 = dir.join("good2.txt");
        std::fs::write(&f2, "second file content").unwrap();

        let segment = Segment::build(&[f1.clone(), missing, bad, f2.clone()]).unwrap();
        assert_eq!(
            segment.files.len(),
            2,
            "the missing and non-utf8 files must be skipped, not abort the whole batch"
        );
        assert_eq!(segment.files[0].path, f1.to_string_lossy());
        assert_eq!(segment.files[1].path, f2.to_string_lossy());

        std::fs::remove_dir_all(&dir).ok();
    }
}
