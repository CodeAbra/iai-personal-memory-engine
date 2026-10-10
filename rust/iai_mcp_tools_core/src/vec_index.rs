//! vec_index -- tools-owned exact-cosine vector index.
//!
//! Flat matrix of L2-normalized f32 vectors, own on-disk format (own magic,
//! distinct from the brain's iai_mcp_vec_core LVECIDX1), CRC32 + atomic
//! rename, fail-closed load. Copied by shape (never by dependency) from
//! iai_mcp_vec_core -- see this crate's top-level module doc.
//!
//! Unlike the whole-file/symbol/xref lanes' append-per-batch segment-set
//! model, this lane REBUILDS a single flat segment on every index pass --
//! a deliberate choice for a background-only, whole-corpus-reindex lane,
//! not an oversight; there is no incremental segment set to compact.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"LTVECIX1";
const FORMAT_VERSION: u32 = 1;
const SEGMENT_FILE_NAME: &str = "vectors.idx";

pub struct VecSegment {
    dim: usize,
    // Row-major, len == paths.len() * dim, each row L2-normalized.
    rows: Vec<f32>,
    paths: Vec<String>,
    // Set when the source text's length exceeded the embed window at
    // encode time. A whitespace-token-count proxy stands in for the
    // embedder's own tokenizer length -- an approximation, not an exact
    // count, documented here because callers must not treat it as exact.
    truncated: Vec<bool>,
}

fn normalize(src: &[f32]) -> Vec<f32> {
    let norm: f32 = src.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        src.iter().map(|x| x / norm).collect()
    } else {
        vec![0.0; src.len()]
    }
}

impl VecSegment {
    pub fn build(
        dim: usize,
        vectors: &[Vec<f32>],
        paths: &[String],
        truncated: &[bool],
    ) -> Result<Self, String> {
        if vectors.len() != paths.len() || vectors.len() != truncated.len() {
            return Err(format!(
                "mismatched row counts: {} vectors, {} paths, {} truncated flags",
                vectors.len(),
                paths.len(),
                truncated.len()
            ));
        }
        let mut rows = Vec::with_capacity(vectors.len() * dim);
        for v in vectors {
            if v.len() != dim {
                return Err(format!("vector length {} != dim {}", v.len(), dim));
            }
            rows.extend_from_slice(&normalize(v));
        }
        Ok(VecSegment {
            dim,
            rows,
            paths: paths.to_vec(),
            truncated: truncated.to_vec(),
        })
    }

    /// Cosine-ranked (path, score) pairs, highest similarity first. A
    /// serial dot-product sweep -- a rayon threshold is out of scope at
    /// this lane's tracer corpus size.
    pub fn top_k(&self, query: &[f32], k: usize) -> Vec<(String, f64)> {
        let n = self.paths.len();
        if n == 0 || query.len() != self.dim {
            return Vec::new();
        }
        let q = normalize(query);
        let mut scored: Vec<(f32, usize)> = (0..n)
            .map(|i| {
                let start = i * self.dim;
                let row = &self.rows[start..start + self.dim];
                let dot: f32 = row.iter().zip(q.iter()).map(|(a, b)| a * b).sum();
                (dot, i)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
            .into_iter()
            .map(|(score, i)| (self.paths[i].clone(), score as f64))
            .collect()
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let n = self.paths.len();
        let mut payload: Vec<u8> = Vec::new();
        payload.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        payload.extend_from_slice(&(self.dim as u64).to_le_bytes());
        payload.extend_from_slice(&(n as u64).to_le_bytes());
        for &v in &self.rows {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        for path_str in &self.paths {
            let bytes = path_str.as_bytes();
            payload.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            payload.extend_from_slice(bytes);
        }
        for &t in &self.truncated {
            payload.push(t as u8);
        }
        let crc = crc32fast::hash(&payload);

        let tmp = path.with_extension("tmp-vecidx");
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

    pub fn load(path: &Path) -> Result<Self, String> {
        let raw = fs::read(path).map_err(|e| format!("read {path:?}: {e}"))?;
        if raw.len() < 12 || &raw[..8] != MAGIC {
            return Err("not a tools vector index file".to_string());
        }
        let stored_crc = u32::from_le_bytes(raw[8..12].try_into().unwrap());
        let payload = &raw[12..];
        if crc32fast::hash(payload) != stored_crc {
            return Err("tools vector index CRC mismatch".to_string());
        }
        if payload.len() < 20 {
            return Err("tools vector index truncated header".to_string());
        }
        let version = u32::from_le_bytes(payload[0..4].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(format!(
                "unsupported tools vector index format version {version}"
            ));
        }
        let dim = u64::from_le_bytes(payload[4..12].try_into().unwrap()) as usize;
        let n = u64::from_le_bytes(payload[12..20].try_into().unwrap()) as usize;
        let mut off = 20;

        let row_count = n
            .checked_mul(dim)
            .ok_or_else(|| "tools vector index row count overflow".to_string())?;
        let need_rows = row_count
            .checked_mul(4)
            .ok_or_else(|| "tools vector index row byte length overflow".to_string())?;
        if payload.len() < off + need_rows {
            return Err("tools vector index truncated matrix".to_string());
        }
        let mut rows = Vec::with_capacity(row_count);
        for i in 0..row_count {
            let s = off + i * 4;
            rows.push(f32::from_le_bytes(payload[s..s + 4].try_into().unwrap()));
        }
        off += need_rows;

        let mut paths = Vec::with_capacity(n);
        for _ in 0..n {
            if off + 4 > payload.len() {
                return Err("tools vector index truncated path table".to_string());
            }
            let len = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()) as usize;
            off += 4;
            if off + len > payload.len() {
                return Err("tools vector index truncated path bytes".to_string());
            }
            let s = std::str::from_utf8(&payload[off..off + len])
                .map_err(|e| format!("tools vector index path is not valid UTF-8: {e}"))?
                .to_string();
            paths.push(s);
            off += len;
        }

        if payload.len() != off + n {
            return Err(format!(
                "tools vector index length {} != expected {}",
                payload.len(),
                off + n
            ));
        }
        let truncated: Vec<bool> = payload[off..off + n].iter().map(|&b| b != 0).collect();

        let expected_rows = paths
            .len()
            .checked_mul(dim)
            .ok_or_else(|| "tools vector index row/path/dim count overflow".to_string())?;
        if rows.len() != expected_rows {
            return Err("tools vector index row/path/dim count mismatch".to_string());
        }

        Ok(VecSegment {
            dim,
            rows,
            paths,
            truncated,
        })
    }
}

pub fn segment_path(store_root: &Path) -> PathBuf {
    store_root.join(SEGMENT_FILE_NAME)
}

/// Rebuilds the store's single vector segment from `vectors`/`paths`/
/// `truncated` (row-aligned, same length). Overwrites whatever segment
/// was there -- see the module doc for why this lane has no append model.
pub fn index_vectors(
    store_root: &Path,
    vectors: &[Vec<f32>],
    paths: &[String],
    truncated: &[bool],
) -> Result<(), String> {
    if vectors.is_empty() {
        return Err("index_vectors requires at least one vector".to_string());
    }
    let dim = vectors[0].len();
    let segment = VecSegment::build(dim, vectors, paths, truncated)?;
    fs::create_dir_all(store_root).map_err(|e| e.to_string())?;
    segment
        .save(&segment_path(store_root))
        .map_err(|e| e.to_string())
}

/// A missing store (never indexed) returns an empty result rather than an
/// error, matching the other lanes' "nothing indexed yet" posture. A store
/// that DOES exist but was built at a different embedding dimension (e.g.
/// after an embedder model change) is a distinct, louder failure -- it
/// returns Err rather than collapsing to the same silent empty result,
/// since that would be indistinguishable from "nothing indexed yet" and
/// mask a stale index that needs a rebuild.
pub fn search_vector(
    store_root: &Path,
    query: &[f32],
    k: usize,
) -> Result<Vec<(String, f64)>, String> {
    let path = segment_path(store_root);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let segment = VecSegment::load(&path)?;
    if query.len() != segment.dim {
        return Err(format!(
            "tools vector index dimension mismatch: index built at dim {}, query is dim {}",
            segment.dim,
            query.len()
        ));
    }
    Ok(segment.top_k(query, k))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "tools-vecidx-test-{name}-{}-{}",
            std::process::id(),
            name.len()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn save_load_roundtrip_preserves_vectors_paths_and_truncation_flags() {
        let dir = tmp_dir("roundtrip");
        let segment = VecSegment::build(
            3,
            &[vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
            &["a.py".to_string(), "b.py".to_string()],
            &[false, true],
        )
        .unwrap();
        let path = dir.join("vectors.idx");
        segment.save(&path).unwrap();

        let loaded = VecSegment::load(&path).unwrap();
        assert_eq!(loaded.paths, segment.paths);
        assert_eq!(loaded.truncated, segment.truncated);
        assert_eq!(loaded.rows, segment.rows);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_foreign_magic() {
        let dir = tmp_dir("foreign");
        let path = dir.join("foreign.idx");
        std::fs::write(&path, b"LVECIDX1....................").unwrap();
        assert!(VecSegment::load(&path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_overflowing_n_times_dim_header_instead_of_panicking() {
        let dir = tmp_dir("overflow");
        let path = dir.join("overflow.idx");

        // n * dim overflows usize on any realistic width; a naive
        // unchecked multiplication would wrap to a small truncation-check
        // value and let this payload through to the row-reading loop,
        // which would then panic on out-of-bounds slicing.
        let huge = 1u64 << 40;
        let mut payload: Vec<u8> = Vec::new();
        payload.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        payload.extend_from_slice(&huge.to_le_bytes()); // dim
        payload.extend_from_slice(&huge.to_le_bytes()); // n
        let crc = crc32fast::hash(&payload);

        let mut raw: Vec<u8> = Vec::new();
        raw.extend_from_slice(MAGIC);
        raw.extend_from_slice(&crc.to_le_bytes());
        raw.extend_from_slice(&payload);
        std::fs::write(&path, &raw).unwrap();

        assert!(VecSegment::load(&path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_crc_mismatch() {
        let dir = tmp_dir("crc");
        let segment =
            VecSegment::build(2, &[vec![1.0, 0.0]], &["a.py".to_string()], &[false]).unwrap();
        let path = dir.join("vectors.idx");
        segment.save(&path).unwrap();
        let mut raw = std::fs::read(&path).unwrap();
        let last = raw.len() - 1;
        raw[last] ^= 0xFF;
        std::fs::write(&path, &raw).unwrap();
        assert!(VecSegment::load(&path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn top_k_orders_two_known_vectors_by_cosine() {
        let segment = VecSegment::build(
            2,
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
            &["x-axis.py".to_string(), "y-axis.py".to_string()],
            &[false, false],
        )
        .unwrap();
        let hits = segment.top_k(&[1.0, 0.1], 2);
        assert_eq!(hits[0].0, "x-axis.py");
        assert!(hits[0].1 > hits[1].1);
    }

    #[test]
    fn index_vectors_and_search_vector_round_trip_through_store_root() {
        let dir = tmp_dir("store-root");
        index_vectors(
            &dir,
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
            &["found.py".to_string(), "other.py".to_string()],
            &[false, false],
        )
        .unwrap();
        let hits = search_vector(&dir, &[1.0, 0.0], 1).unwrap();
        assert_eq!(hits[0].0, "found.py");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn search_vector_over_missing_store_returns_empty() {
        let dir = tmp_dir("missing");
        std::fs::remove_dir_all(&dir).ok();
        let hits = search_vector(&dir, &[1.0, 0.0], 1).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn search_vector_over_dim_mismatch_errs_instead_of_silently_returning_empty() {
        let dir = tmp_dir("dim-mismatch");
        index_vectors(
            &dir,
            &[vec![1.0, 0.0]],
            &["found.py".to_string()],
            &[false],
        )
        .unwrap();
        // Index built at dim 2; query at dim 3 -- an embedder-model-change
        // scenario, distinct from "nothing indexed yet" and must not
        // collapse to the same silent empty result.
        let err = search_vector(&dir, &[1.0, 0.0, 0.0], 1).unwrap_err();
        assert!(err.contains("dimension mismatch"), "unexpected error: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
