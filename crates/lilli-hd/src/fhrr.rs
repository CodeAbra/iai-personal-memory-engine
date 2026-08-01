//! Fourier holographic reduced representation kernel (semantic tier).
//!
//! Each byte is a phase in `[0, 256)` mapping to `[0, 2pi)`. Binding adds
//! phases mod 256; unbinding subtracts. Bundling is a per-component circular
//! mean over the phase angles. The quantization cast is truncate-toward-zero
//! (`as i32`), never round — matching the reference exactly so no bin-boundary
//! value flips.

use std::f64::consts::TAU;

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::errors::HvError;

/// FHRR dimension (bytes per hypervector).
pub const FHRR_DIM: usize = 10000;

/// Phase-add bind: `(a + b) mod 256` per byte.
pub fn bind_impl(a: &[u8], b: &[u8]) -> Result<Vec<u8>, HvError> {
    if a.len() != b.len() {
        return Err(HvError::LengthMismatch {
            left: a.len(),
            right: b.len(),
        });
    }
    Ok(a.iter().zip(b).map(|(x, y)| x.wrapping_add(*y)).collect())
}

/// Phase-subtract unbind: `(bound - key) mod 256` per byte.
pub fn unbind_impl(bound: &[u8], key: &[u8]) -> Result<Vec<u8>, HvError> {
    if bound.len() != key.len() {
        return Err(HvError::LengthMismatch {
            left: bound.len(),
            right: key.len(),
        });
    }
    Ok(bound
        .iter()
        .zip(key)
        .map(|(x, y)| x.wrapping_sub(*y))
        .collect())
}

/// Circular-mean bundle over phase hypervectors.
///
/// Empty input returns `FHRR_DIM` zero bytes. A single hypervector is returned
/// unchanged. Otherwise each component is the phase of the summed unit vectors,
/// quantized truncate-toward-zero into a byte.
pub fn bundle_impl(hvs: &[Vec<u8>]) -> Result<Vec<u8>, HvError> {
    if hvs.is_empty() {
        return Ok(vec![0u8; FHRR_DIM]);
    }
    // Every hypervector is indexed component-by-component below, so a short or
    // mismatched one would read out of bounds. Validate each against FHRR_DIM
    // before any indexing.
    for hv in hvs {
        if hv.len() != FHRR_DIM {
            return Err(HvError::Length {
                expected: FHRR_DIM,
                actual: hv.len(),
            });
        }
    }
    if hvs.len() == 1 {
        return Ok(hvs[0].clone());
    }
    let d = hvs[0].len();
    let mut out = vec![0u8; d];
    for j in 0..d {
        let mut c = 0f64;
        let mut s = 0f64;
        for hv in hvs {
            let radians = (hv[j] as f64 / 256.0) * TAU;
            c += radians.cos();
            s += radians.sin();
        }
        let mean = s.atan2(c).rem_euclid(TAU);
        let q = ((mean / TAU * 256.0) as i32) & 0xFF;
        out[j] = q as u8;
    }
    Ok(out)
}

/// Byte-roll permute. Positive `shift` moves bytes toward higher index, like
/// `np.roll` over the uint8 array.
pub fn permute_impl(hv: &[u8], shift: i64) -> Vec<u8> {
    let n = hv.len();
    if n == 0 {
        return Vec::new();
    }
    let n_i = n as i64;
    // Normalize the shift into [0, n) first so an extreme i64 shift cannot
    // overflow the per-element `i - shift`. The result is unchanged for in-range
    // shifts.
    let s = ((shift % n_i) + n_i) % n_i;
    let mut out = vec![0u8; n];
    for (i, dst) in out.iter_mut().enumerate() {
        let src = ((i as i64 - s) % n_i + n_i) % n_i;
        *dst = hv[src as usize];
    }
    out
}

/// Mean cosine of the phase-angle difference. Used for ranking, not persisted, so
/// last-ULP float noise is acceptable. Returns 0.0 on a length mismatch.
pub fn similarity_impl(a: &[u8], b: &[u8]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut acc = 0f64;
    for (x, y) in a.iter().zip(b) {
        let ar = (*x as f64 / 256.0) * TAU;
        let br = (*y as f64 / 256.0) * TAU;
        acc += (ar - br).cos();
    }
    acc / a.len() as f64
}

// --- PyO3 surface -----------------------------------------------------------

#[pyfunction]
#[pyo3(name = "fhrr_bind")]
pub fn bind<'py>(py: Python<'py>, a: &[u8], b: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    let out = bind_impl(a, b)?;
    Ok(PyBytes::new_bound(py, &out))
}

#[pyfunction]
#[pyo3(name = "fhrr_unbind")]
pub fn unbind<'py>(py: Python<'py>, bound: &[u8], key: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    let out = unbind_impl(bound, key)?;
    Ok(PyBytes::new_bound(py, &out))
}

#[pyfunction]
#[pyo3(name = "fhrr_bundle")]
pub fn bundle<'py>(
    py: Python<'py>,
    hvs: &Bound<'py, pyo3::types::PyList>,
) -> PyResult<Bound<'py, PyBytes>> {
    let mats: Vec<Vec<u8>> = hvs
        .iter()
        .map(|item| item.extract::<Vec<u8>>())
        .collect::<PyResult<_>>()?;
    let out = py.allow_threads(|| bundle_impl(&mats))?;
    Ok(PyBytes::new_bound(py, &out))
}

#[pyfunction]
#[pyo3(name = "fhrr_permute")]
pub fn permute<'py>(py: Python<'py>, hv: &[u8], shift: i64) -> PyResult<Bound<'py, PyBytes>> {
    let out = permute_impl(hv, shift);
    Ok(PyBytes::new_bound(py, &out))
}

#[pyfunction]
#[pyo3(name = "fhrr_similarity")]
pub fn similarity(a: &[u8], b: &[u8]) -> f64 {
    similarity_impl(a, b)
}
