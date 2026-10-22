//! iai_mcp_vec_core — exact cosine vector index.
//!
//! Flat matrix of L2-normalized vectors; a query is one dot-product sweep
//! with a bounded top-k heap. At the corpus scales this store serves
//! (tens of thousands of vectors) an exact sweep answers in single-digit
//! milliseconds AND returns perfect recall — an approximate graph index
//! buys nothing until the corpus is orders of magnitude larger.
//!
//! The Python surface mirrors the subset of the previous ANN library's API
//! the storage layer used, so the swap is an import change: labels are
//! caller-assigned i64s, re-adding an existing label updates its vector in
//! place, `mark_deleted` hides a label from queries without freeing its
//! slot, and `knn_query` returns `[1, k]`-shaped label/distance arrays with
//! cosine DISTANCE (1 - cosine similarity).
//!
//! Strict dependency isolation: no embedder deps, no candle. Ships as a
//! workspace rlib re-exported by the `iai_mcp_native` wrapper as the `vec`
//! sub-module.

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write as _};
use std::path::Path;
use std::sync::Mutex;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

const MAGIC: &[u8; 8] = b"LVECIDX1";
/// Streaming chunk size for save/load: bounds the transient read/write buffer
/// so peak allocation is ~1x the file, not the whole file held in one buffer.
const CHUNK_BYTES: usize = 1 << 20;

struct Inner {
    dim: usize,
    capacity: usize,
    // Slot-major flat matrix; slot i occupies rows[i*dim .. (i+1)*dim].
    rows: Vec<f32>,
    labels: Vec<i64>,
    deleted: Vec<bool>,
    label_to_slot: HashMap<i64, usize>,
}

impl Inner {
    fn new(dim: usize, capacity: usize) -> Self {
        Inner {
            dim,
            capacity,
            rows: Vec::with_capacity(capacity.saturating_mul(dim)),
            labels: Vec::with_capacity(capacity),
            deleted: Vec::with_capacity(capacity),
            label_to_slot: HashMap::with_capacity(capacity),
        }
    }

    fn normalize_into(&self, src: &[f32], dst: &mut [f32]) {
        let norm: f32 = src.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                *d = s / norm;
            }
        } else {
            dst.fill(0.0);
        }
    }

    fn add_one(&mut self, vec: &[f32], label: i64) -> Result<(), String> {
        if vec.len() != self.dim {
            return Err(format!(
                "vector length {} != index dim {}",
                vec.len(),
                self.dim
            ));
        }
        if let Some(&slot) = self.label_to_slot.get(&label) {
            let start = slot * self.dim;
            let mut buf = vec![0.0f32; self.dim];
            self.normalize_into(vec, &mut buf);
            self.rows[start..start + self.dim].copy_from_slice(&buf);
            self.deleted[slot] = false;
            return Ok(());
        }
        // Capacity is a sizing hint, never a wall: a flat matrix grows by
        // appending, so the fixed-allocation "index full" failure mode of
        // graph-ANN backends does not exist here.
        if self.labels.len() >= self.capacity {
            self.capacity = (self.labels.len() + 1).next_power_of_two();
        }
        let slot = self.labels.len();
        let start = slot * self.dim;
        self.rows.resize(start + self.dim, 0.0);
        let mut buf = vec![0.0f32; self.dim];
        self.normalize_into(vec, &mut buf);
        self.rows[start..start + self.dim].copy_from_slice(&buf);
        self.labels.push(label);
        self.deleted.push(false);
        self.label_to_slot.insert(label, slot);
        Ok(())
    }

    fn top_k(&self, query: &[f32], k: usize) -> (Vec<i64>, Vec<f32>) {
        let n = self.labels.len();
        let mut q = vec![0.0f32; self.dim];
        self.normalize_into(query, &mut q);

        // (distance, label) pairs; distance = 1 - dot on normalized vectors.
        // Serial sweep below ~32k live rows; rayon chunk-reduce above.
        let collect_chunk = |lo: usize, hi: usize| -> Vec<(f32, i64)> {
            let mut out: Vec<(f32, i64)> = Vec::with_capacity(k.min(hi - lo));
            for slot in lo..hi {
                if self.deleted[slot] {
                    continue;
                }
                let start = slot * self.dim;
                let row = &self.rows[start..start + self.dim];
                let dot: f32 = row.iter().zip(q.iter()).map(|(a, b)| a * b).sum();
                out.push((1.0 - dot, self.labels[slot]));
            }
            out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            out.truncate(k);
            out
        };

        let mut merged: Vec<(f32, i64)> = if n >= 32_768 {
            use rayon::prelude::*;
            let chunk = 8_192;
            let bounds: Vec<(usize, usize)> = (0..n)
                .step_by(chunk)
                .map(|lo| (lo, (lo + chunk).min(n)))
                .collect();
            let mut all: Vec<(f32, i64)> = bounds
                .into_par_iter()
                .map(|(lo, hi)| collect_chunk(lo, hi))
                .reduce(Vec::new, |mut a, b| {
                    a.extend(b);
                    a
                });
            all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            all
        } else {
            collect_chunk(0, n)
        };
        merged.truncate(k);
        (
            merged.iter().map(|(_, l)| *l).collect(),
            merged.iter().map(|(d, _)| *d).collect(),
        )
    }

    // crc offset (8) precedes payload (12+), so the real crc is patched in
    // after a single streamed pass; one sync_all covers both writes.
    fn save(&self, path: &Path) -> std::io::Result<()> {
        let n = self.labels.len();
        let tmp = path.with_extension("tmp-vecidx");
        {
            let mut f = fs::File::create(&tmp)?;
            f.write_all(MAGIC)?;
            f.write_all(&0u32.to_le_bytes())?; // crc placeholder, patched below

            let mut hasher = crc32fast::Hasher::new();
            {
                let mut w = std::io::BufWriter::new(&mut f);
                let mut buf: Vec<u8> = Vec::with_capacity(CHUNK_BYTES);

                let dim_bytes = (self.dim as u64).to_le_bytes();
                hasher.update(&dim_bytes);
                w.write_all(&dim_bytes)?;
                let n_bytes = (n as u64).to_le_bytes();
                hasher.update(&n_bytes);
                w.write_all(&n_bytes)?;

                for chunk in self.labels.chunks(CHUNK_BYTES / 8) {
                    buf.clear();
                    for &l in chunk {
                        buf.extend_from_slice(&l.to_le_bytes());
                    }
                    hasher.update(&buf);
                    w.write_all(&buf)?;
                }
                for chunk in self.deleted.chunks(CHUNK_BYTES) {
                    buf.clear();
                    buf.extend(chunk.iter().map(|&d| d as u8));
                    hasher.update(&buf);
                    w.write_all(&buf)?;
                }
                for chunk in self.rows[..n * self.dim].chunks(CHUNK_BYTES / 4) {
                    buf.clear();
                    for &v in chunk {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                    hasher.update(&buf);
                    w.write_all(&buf)?;
                }
                w.flush()?;
            }

            let crc = hasher.finalize();
            f.seek(SeekFrom::Start(8))?;
            f.write_all(&crc.to_le_bytes())?;
            f.sync_all()?;
        }
        fs::rename(&tmp, path)?;
        Ok(())
    }

    // Pass 1 verifies the crc before dim/n are trusted for any allocation;
    // pass 2 re-reads the payload now that n/dim are CRC-verified.
    fn load(path: &Path, capacity: usize) -> Result<Self, String> {
        let mut f = fs::File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
        let file_len = f
            .metadata()
            .map_err(|e| format!("stat {path:?}: {e}"))?
            .len();
        if file_len < 12 {
            return Err("not an exact-vec index file".to_string());
        }
        let mut header = [0u8; 12];
        f.read_exact(&mut header)
            .map_err(|e| format!("read {path:?}: {e}"))?;
        if &header[..8] != MAGIC {
            return Err("not an exact-vec index file".to_string());
        }
        let stored_crc = u32::from_le_bytes(header[8..12].try_into().unwrap());
        // Trusted: derived from the file's own actual size, never from a
        // header field the payload's own bytes could lie about.
        let payload_len = file_len - 12;

        let mut hasher = crc32fast::Hasher::new();
        let mut chunk_buf = vec![0u8; CHUNK_BYTES.min(payload_len.max(1) as usize)];
        let mut remaining = payload_len;
        while remaining > 0 {
            let want = (chunk_buf.len() as u64).min(remaining) as usize;
            f.read_exact(&mut chunk_buf[..want])
                .map_err(|e| format!("read {path:?}: {e}"))?;
            hasher.update(&chunk_buf[..want]);
            remaining -= want as u64;
        }
        if hasher.finalize() != stored_crc {
            return Err("exact-vec index CRC mismatch".to_string());
        }

        if payload_len < 16 {
            return Err("exact-vec index truncated header".to_string());
        }
        f.seek(SeekFrom::Start(12))
            .map_err(|e| format!("seek {path:?}: {e}"))?;
        let mut hdr2 = [0u8; 16];
        f.read_exact(&mut hdr2)
            .map_err(|e| format!("read {path:?}: {e}"))?;
        let dim = u64::from_le_bytes(hdr2[0..8].try_into().unwrap()) as usize;
        let n = u64::from_le_bytes(hdr2[8..16].try_into().unwrap()) as usize;
        let need = 16u64
            .checked_add(n as u64 * 8)
            .and_then(|v| v.checked_add(n as u64))
            .and_then(|v| v.checked_add(n as u64 * dim as u64 * 4))
            .ok_or_else(|| "exact-vec index size overflow".to_string())?;
        if payload_len != need {
            return Err(format!(
                "exact-vec index length {payload_len} != expected {need}"
            ));
        }

        let mut inner = Inner::new(dim, capacity.max(n));
        inner.labels.reserve_exact(n);
        read_i64_chunks(&mut f, n, &mut inner.labels, path)?;
        inner.deleted.reserve_exact(n);
        read_bool_chunks(&mut f, n, &mut inner.deleted, path)?;
        inner.rows.reserve_exact(n * dim);
        read_f32_chunks(&mut f, n * dim, &mut inner.rows, path)?;

        for (slot, &l) in inner.labels.iter().enumerate() {
            inner.label_to_slot.insert(l, slot);
        }
        Ok(inner)
    }
}

/// Stream `count` little-endian `i64`s from `f` into `out`, in ~1 MiB chunks.
fn read_i64_chunks(
    f: &mut fs::File,
    count: usize,
    out: &mut Vec<i64>,
    path: &Path,
) -> Result<(), String> {
    let per_chunk = (CHUNK_BYTES / 8).max(1);
    let mut buf = vec![0u8; per_chunk * 8];
    let mut remaining = count;
    while remaining > 0 {
        let take = remaining.min(per_chunk);
        let want = take * 8;
        f.read_exact(&mut buf[..want])
            .map_err(|e| format!("read {path:?}: {e}"))?;
        for chunk in buf[..want].chunks_exact(8) {
            out.push(i64::from_le_bytes(chunk.try_into().unwrap()));
        }
        remaining -= take;
    }
    Ok(())
}

/// Stream `count` bytes from `f` into `out` as bools (non-zero == true), in
/// ~1 MiB chunks.
fn read_bool_chunks(
    f: &mut fs::File,
    count: usize,
    out: &mut Vec<bool>,
    path: &Path,
) -> Result<(), String> {
    let per_chunk = CHUNK_BYTES.max(1);
    let mut buf = vec![0u8; per_chunk];
    let mut remaining = count;
    while remaining > 0 {
        let take = remaining.min(per_chunk);
        f.read_exact(&mut buf[..take])
            .map_err(|e| format!("read {path:?}: {e}"))?;
        out.extend(buf[..take].iter().map(|&b| b != 0));
        remaining -= take;
    }
    Ok(())
}

/// Stream `count` little-endian `f32`s from `f` into `out`, in ~1 MiB chunks.
fn read_f32_chunks(
    f: &mut fs::File,
    count: usize,
    out: &mut Vec<f32>,
    path: &Path,
) -> Result<(), String> {
    let per_chunk = (CHUNK_BYTES / 4).max(1);
    let mut buf = vec![0u8; per_chunk * 4];
    let mut remaining = count;
    while remaining > 0 {
        let take = remaining.min(per_chunk);
        let want = take * 4;
        f.read_exact(&mut buf[..want])
            .map_err(|e| format!("read {path:?}: {e}"))?;
        for chunk in buf[..want].chunks_exact(4) {
            out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        remaining -= take;
    }
    Ok(())
}

/// Exact cosine index with the ANN-shaped surface the storage layer calls.
#[pyclass]
pub struct ExactIndex {
    inner: Mutex<Inner>,
    dim: usize,
}

fn flatten_vecs(arr: &PyReadonlyArrayDyn<'_, f32>, dim: usize) -> PyResult<Vec<Vec<f32>>> {
    let a = arr.as_array();
    match a.ndim() {
        1 => {
            if a.len() != dim {
                return Err(PyValueError::new_err(format!(
                    "vector length {} != index dim {}",
                    a.len(),
                    dim
                )));
            }
            Ok(vec![a.iter().copied().collect()])
        }
        2 => {
            let shape = a.shape();
            if shape[1] != dim {
                return Err(PyValueError::new_err(format!(
                    "vector length {} != index dim {}",
                    shape[1], dim
                )));
            }
            let flat: Vec<f32> = a.iter().copied().collect();
            Ok(flat.chunks(dim).map(|c| c.to_vec()).collect())
        }
        d => Err(PyValueError::new_err(format!(
            "expected 1-D or 2-D vector array, got {d}-D"
        ))),
    }
}

#[pymethods]
impl ExactIndex {
    #[new]
    fn new(dim: usize, max_elements: usize) -> Self {
        ExactIndex {
            inner: Mutex::new(Inner::new(dim, max_elements)),
            dim,
        }
    }

    #[staticmethod]
    fn load(path: &str, max_elements: usize) -> PyResult<Self> {
        let inner = Inner::load(Path::new(path), max_elements).map_err(PyRuntimeError::new_err)?;
        let dim = inner.dim;
        Ok(ExactIndex {
            inner: Mutex::new(inner),
            dim,
        })
    }

    fn add_items(
        &self,
        py: Python<'_>,
        vecs: PyReadonlyArrayDyn<'_, f32>,
        labels: Vec<i64>,
    ) -> PyResult<()> {
        let rows = flatten_vecs(&vecs, self.dim)?;
        if rows.len() != labels.len() {
            return Err(PyValueError::new_err(format!(
                "{} vectors for {} labels",
                rows.len(),
                labels.len()
            )));
        }
        py.detach(|| {
            let mut inner = self.inner.lock().unwrap();
            for (row, &label) in rows.iter().zip(labels.iter()) {
                inner.add_one(row, label).map_err(PyRuntimeError::new_err)?;
            }
            Ok(())
        })
    }

    fn knn_query<'py>(
        &self,
        py: Python<'py>,
        vec: PyReadonlyArrayDyn<'_, f32>,
        k: usize,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let rows = flatten_vecs(&vec, self.dim)?;
        if rows.len() != 1 {
            return Err(PyValueError::new_err(
                "knn_query expects exactly one query vector",
            ));
        }
        let query = rows.into_iter().next().unwrap();
        let (labels, dists) = py.detach(|| {
            let inner = self.inner.lock().unwrap();
            inner.top_k(&query, k)
        });
        let n = labels.len();
        let labels_arr = Array2::from_shape_vec((1, n), labels)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let dists_arr = Array2::from_shape_vec((1, n), dists)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok((
            labels_arr.into_pyarray(py).into_any(),
            dists_arr.into_pyarray(py).into_any(),
        ))
    }

    /// Raises on a missing label AND on a label that is already deleted —
    /// callers distinguish "freshly hidden" from "was already hidden" by
    /// catching RuntimeError, so the double-delete raise is load-bearing.
    fn mark_deleted(&self, label: i64) -> PyResult<()> {
        let mut inner = self.inner.lock().unwrap();
        match inner.label_to_slot.get(&label).copied() {
            Some(slot) => {
                if inner.deleted[slot] {
                    return Err(PyRuntimeError::new_err(format!(
                        "label {label} is already deleted"
                    )));
                }
                inner.deleted[slot] = true;
                Ok(())
            }
            None => Err(PyRuntimeError::new_err(format!(
                "label {label} not in index"
            ))),
        }
    }

    /// Raises on a missing label AND on a label that is not deleted — the
    /// refill path treats that raise as "nothing to restore".
    fn unmark_deleted(&self, label: i64) -> PyResult<()> {
        let mut inner = self.inner.lock().unwrap();
        match inner.label_to_slot.get(&label).copied() {
            Some(slot) => {
                if !inner.deleted[slot] {
                    return Err(PyRuntimeError::new_err(format!(
                        "label {label} is not deleted"
                    )));
                }
                inner.deleted[slot] = false;
                Ok(())
            }
            None => Err(PyRuntimeError::new_err(format!(
                "label {label} not in index"
            ))),
        }
    }

    /// Count of occupied slots, INCLUDING mark_deleted ones (parity with the
    /// previous ANN library's element counting, which the swap accounting
    /// relies on).
    fn get_current_count(&self) -> usize {
        self.inner.lock().unwrap().labels.len()
    }

    fn get_ids_list(&self) -> Vec<i64> {
        self.inner.lock().unwrap().labels.clone()
    }

    /// Vectors for the given labels, in order. Raises on any missing label —
    /// callers use that raise as the "standby lacks labels" signal.
    fn get_items(&self, labels: Vec<i64>) -> PyResult<Vec<Vec<f32>>> {
        let inner = self.inner.lock().unwrap();
        let mut out = Vec::with_capacity(labels.len());
        for label in labels {
            match inner.label_to_slot.get(&label).copied() {
                Some(slot) => {
                    let start = slot * inner.dim;
                    out.push(inner.rows[start..start + inner.dim].to_vec());
                }
                None => {
                    return Err(PyRuntimeError::new_err(format!(
                        "label {label} not in index"
                    )))
                }
            }
        }
        Ok(out)
    }

    fn get_max_elements(&self) -> usize {
        self.inner.lock().unwrap().capacity
    }

    fn resize_index(&self, new_size: usize) -> PyResult<()> {
        let mut inner = self.inner.lock().unwrap();
        if new_size < inner.labels.len() {
            return Err(PyValueError::new_err(format!(
                "cannot shrink below element count {}",
                inner.labels.len()
            )));
        }
        inner.capacity = new_size;
        Ok(())
    }

    fn save_index(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        py.detach(|| {
            let inner = self.inner.lock().unwrap();
            inner
                .save(Path::new(path))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ExactIndex>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx3() -> Inner {
        let mut inner = Inner::new(3, 16);
        inner.add_one(&[1.0, 0.0, 0.0], 10).unwrap();
        inner.add_one(&[0.0, 1.0, 0.0], 20).unwrap();
        inner.add_one(&[0.7, 0.7, 0.0], 30).unwrap();
        inner
    }

    #[test]
    fn top_k_orders_by_cosine_distance() {
        let inner = idx3();
        let (labels, dists) = inner.top_k(&[1.0, 0.0, 0.0], 3);
        assert_eq!(labels[0], 10);
        assert!(dists[0] < 1e-6);
        assert_eq!(labels[1], 30);
        assert_eq!(labels[2], 20);
        assert!(dists[1] < dists[2]);
    }

    #[test]
    fn update_in_place_replaces_vector() {
        let mut inner = idx3();
        inner.add_one(&[0.0, 0.0, 1.0], 10).unwrap();
        let (labels, _) = inner.top_k(&[0.0, 0.0, 1.0], 1);
        assert_eq!(labels[0], 10);
        assert_eq!(inner.labels.len(), 3, "update must not add a slot");
    }

    #[test]
    fn deleted_labels_are_hidden_until_unmarked() {
        let mut inner = idx3();
        let slot = inner.label_to_slot[&10];
        inner.deleted[slot] = true;
        let (labels, _) = inner.top_k(&[1.0, 0.0, 0.0], 3);
        assert!(!labels.contains(&10));
        inner.deleted[slot] = false;
        let (labels, _) = inner.top_k(&[1.0, 0.0, 0.0], 3);
        assert_eq!(labels[0], 10);
    }

    #[test]
    fn save_load_roundtrip_preserves_everything() {
        let dir = std::env::temp_dir().join("lvecidx-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("roundtrip.idx");
        let mut inner = idx3();
        let slot = inner.label_to_slot[&20];
        inner.deleted[slot] = true;
        inner.save(&path).unwrap();

        let loaded = Inner::load(&path, 4).unwrap();
        assert_eq!(loaded.labels, inner.labels);
        assert_eq!(loaded.deleted, inner.deleted);
        assert_eq!(loaded.rows, inner.rows);
        assert!(loaded.capacity >= loaded.labels.len());
        let (labels, _) = loaded.top_k(&[0.0, 1.0, 0.0], 3);
        assert!(!labels.contains(&20));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_rejects_foreign_and_corrupt_files() {
        let dir = std::env::temp_dir().join("lvecidx-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("foreign.idx");
        std::fs::write(&path, b"HNSWLIB-OLD-FORMAT....").unwrap();
        assert!(Inner::load(&path, 4).is_err());

        let good = dir.join("corrupt.idx");
        idx3().save(&good).unwrap();
        let mut raw = std::fs::read(&good).unwrap();
        let last = raw.len() - 1;
        raw[last] ^= 0xFF;
        std::fs::write(&good, &raw).unwrap();
        assert!(Inner::load(&good, 4).is_err());
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(&good).ok();
    }

    #[test]
    fn truncated_file_errs_without_panicking() {
        let dir = std::env::temp_dir().join("lvecidx-test");
        std::fs::create_dir_all(&dir).unwrap();

        // Truncated below the 12-byte header: the length check must reject it
        // before any magic/crc read is attempted.
        let too_short = dir.join("too-short.idx");
        std::fs::write(&too_short, b"LVEC").unwrap();
        assert!(Inner::load(&too_short, 4).is_err());

        // Truncated mid-payload (past the header, short of the full file): the
        // streaming crc pass must hit EOF via a checked read and return Err,
        // never panic on a short read.
        let mid = dir.join("mid-truncated.idx");
        idx3().save(&mid).unwrap();
        let mut raw = std::fs::read(&mid).unwrap();
        raw.truncate(raw.len() - 4);
        std::fs::write(&mid, &raw).unwrap();
        assert!(Inner::load(&mid, 4).is_err());

        // Truncated to exactly the 12-byte header, zero payload bytes: the
        // crc pass sees an empty payload (never matches a real stored_crc)
        // and returns Err rather than reading a phantom dim/n.
        let header_only = dir.join("header-only.idx");
        idx3().save(&header_only).unwrap();
        let mut raw = std::fs::read(&header_only).unwrap();
        raw.truncate(12);
        std::fs::write(&header_only, &raw).unwrap();
        assert!(Inner::load(&header_only, 4).is_err());

        std::fs::remove_file(&too_short).ok();
        std::fs::remove_file(&mid).ok();
        std::fs::remove_file(&header_only).ok();
    }

    #[test]
    fn save_format_is_byte_identical_header_and_payload_order() {
        // A hand-decoded assertion of the on-disk layout — the round-trip
        // test alone is symmetric and blind to a field-order regression that
        // happens to still decode back correctly.
        let dir = std::env::temp_dir().join("lvecidx-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("layout.idx");
        let mut inner = Inner::new(2, 4);
        inner.add_one(&[1.0, 0.0], 7).unwrap();
        inner.add_one(&[0.0, 1.0], 8).unwrap();
        inner.deleted[0] = true;
        inner.save(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        assert_eq!(&raw[0..8], MAGIC);
        let stored_crc = u32::from_le_bytes(raw[8..12].try_into().unwrap());
        assert_eq!(crc32fast::hash(&raw[12..]), stored_crc);

        let dim = u64::from_le_bytes(raw[12..20].try_into().unwrap());
        let n = u64::from_le_bytes(raw[20..28].try_into().unwrap());
        assert_eq!(dim, 2);
        assert_eq!(n, 2);

        let labels_off = 28;
        let label0 = i64::from_le_bytes(raw[labels_off..labels_off + 8].try_into().unwrap());
        let label1 = i64::from_le_bytes(raw[labels_off + 8..labels_off + 16].try_into().unwrap());
        assert_eq!((label0, label1), (7, 8));

        let deleted_off = labels_off + 16;
        assert_eq!(raw[deleted_off], 1);
        assert_eq!(raw[deleted_off + 1], 0);

        let rows_off = deleted_off + 2;
        assert_eq!(raw.len(), rows_off + 2 * 2 * 4);
        let row0v0 = f32::from_le_bytes(raw[rows_off..rows_off + 4].try_into().unwrap());
        assert_eq!(row0v0, 1.0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn round_trip_spans_multiple_chunk_boundaries() {
        // dim=1 keeps per-row cost tiny; label count crosses the label
        // chunk's element boundary (CHUNK_BYTES/8 == 131072) so the streaming
        // loop's boundary handling is genuinely exercised, not just its
        // single-chunk common case.
        let dir = std::env::temp_dir().join("lvecidx-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("chunked.idx");

        const N: i64 = 131_072 + 5;
        let mut inner = Inner::new(1, N as usize);
        for i in 0..N {
            inner.add_one(&[(i % 7) as f32], i).unwrap();
        }
        inner.deleted[3] = true;
        inner.save(&path).unwrap();

        let loaded = Inner::load(&path, 4).unwrap();
        assert_eq!(loaded.labels, inner.labels);
        assert_eq!(loaded.deleted, inner.deleted);
        assert_eq!(loaded.rows, inner.rows);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn zero_vector_never_matches_anything_real() {
        let mut inner = idx3();
        inner.add_one(&[0.0, 0.0, 0.0], 99).unwrap();
        let (labels, dists) = inner.top_k(&[1.0, 0.0, 0.0], 4);
        // The zero vector's dot is 0 -> distance 1.0, ranked last among
        // positive-similarity rows, never first.
        assert_ne!(labels[0], 99);
        let pos_99 = labels.iter().position(|&l| l == 99).unwrap();
        assert!((dists[pos_99] - 1.0).abs() < 1e-6);
    }
}
