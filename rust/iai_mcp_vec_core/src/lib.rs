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
use std::io::Write as _;
use std::path::Path;
use std::sync::Mutex;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

const MAGIC: &[u8; 8] = b"LVECIDX1";

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

    fn save(&self, path: &Path) -> std::io::Result<()> {
        let n = self.labels.len();
        let mut payload: Vec<u8> = Vec::with_capacity(24 + n * 9 + n * self.dim * 4);
        payload.extend_from_slice(&(self.dim as u64).to_le_bytes());
        payload.extend_from_slice(&(n as u64).to_le_bytes());
        for &l in &self.labels {
            payload.extend_from_slice(&l.to_le_bytes());
        }
        for &d in &self.deleted {
            payload.push(d as u8);
        }
        for &v in &self.rows[..n * self.dim] {
            payload.extend_from_slice(&v.to_le_bytes());
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

    fn load(path: &Path, capacity: usize) -> Result<Self, String> {
        let raw = fs::read(path).map_err(|e| format!("read {path:?}: {e}"))?;
        if raw.len() < 12 || &raw[..8] != MAGIC {
            return Err("not an exact-vec index file".to_string());
        }
        let stored_crc = u32::from_le_bytes(raw[8..12].try_into().unwrap());
        let payload = &raw[12..];
        if crc32fast::hash(payload) != stored_crc {
            return Err("exact-vec index CRC mismatch".to_string());
        }
        if payload.len() < 16 {
            return Err("exact-vec index truncated header".to_string());
        }
        let dim = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
        let n = u64::from_le_bytes(payload[8..16].try_into().unwrap()) as usize;
        let need = 16 + n * 8 + n + n * dim * 4;
        if payload.len() != need {
            return Err(format!(
                "exact-vec index length {} != expected {}",
                payload.len(),
                need
            ));
        }
        let mut inner = Inner::new(dim, capacity.max(n));
        let mut off = 16;
        for _ in 0..n {
            inner.labels.push(i64::from_le_bytes(
                payload[off..off + 8].try_into().unwrap(),
            ));
            off += 8;
        }
        for i in 0..n {
            inner.deleted.push(payload[off + i] != 0);
        }
        off += n;
        inner.rows.reserve(n * dim);
        for _ in 0..(n * dim) {
            inner.rows.push(f32::from_le_bytes(
                payload[off..off + 4].try_into().unwrap(),
            ));
            off += 4;
        }
        for (slot, &l) in inner.labels.iter().enumerate() {
            inner.label_to_slot.insert(l, slot);
        }
        Ok(inner)
    }
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
