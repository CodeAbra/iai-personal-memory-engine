//! Frozen SimHash projection-apply: a 384-d embedder vector lifted into the
//! 10000-bit hypervector space.
//!
//! The projection matrix is a frozen artifact. It is embedded in
//! the binary at compile time and self-checked against its locked sha256 and
//! length on first load, so it is always present and never depends on the
//! filesystem. It is never regenerated from an RNG. `project_impl` computes
//! `emb @ P`, takes the sign, and packs the resulting 10000 bits MSB-first into
//! 1250 bytes, bit-identical to the Python reference.

use std::sync::OnceLock;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use sha2::{Digest, Sha256};

use crate::errors::HvError;

/// The frozen projection matrix, embedded at compile time as the raw
/// little-endian `[384 * 10000]` f32 array. Embedding it removes any runtime
/// filesystem dependency: the matrix ships inside the binary and can never fail
/// to load. The bytes are self-checked against the locked sha256 and length on
/// first use so a wrong embed fails loud.
const FROZEN_P_BYTES: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/golden/hdc/projection.bin"));

/// Length in bytes of the frozen matrix: 384 * 10000 f32 values.
const FROZEN_P_LEN: usize = EMBED_DIM * HV_DIM * 4;

/// Embedder dimension (input).
pub const EMBED_DIM: usize = 384;
/// Hypervector dimension (output bits).
pub const HV_DIM: usize = 10000;
/// Packed output size in bytes (HV_DIM / 8, rounded up).
pub const HV_BYTES: usize = HV_DIM.div_ceil(8);

/// The locked projection-matrix hash. Constitutional invariant — never rotates.
pub const P_SHA256: &str =
    "df97cc72a960567da17edbba16107881340349bd47b69f9b58d3091d96eb4e4e";

static P: OnceLock<Vec<f32>> = OnceLock::new();

/// Read the raw matrix bytes the override env var points at, if it is set and a
/// `projection.bin` exists under it. This is a test-only opt-in; an unset or
/// non-existent override falls through to the embedded bytes.
fn override_bytes() -> Option<Vec<u8>> {
    let dir = std::env::var_os("IAI_MCP_HDC_FIXTURES")?;
    let bin = std::path::PathBuf::from(dir).join("projection.bin");
    std::fs::read(&bin).ok()
}

/// Validate the raw matrix bytes against the locked sha256 and length, then
/// decode them into a flat C-order `[384 * 10000]` f32 vector. A wrong length or
/// a drifted hash panics — a drifted P would silently invalidate every persisted
/// hypervector, so this must fail loudly.
fn decode_matrix(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(
        bytes.len(),
        FROZEN_P_LEN,
        "projection matrix length is not {FROZEN_P_LEN} bytes"
    );
    let digest = format!("{:x}", Sha256::digest(bytes));
    assert_eq!(
        digest, P_SHA256,
        "projection matrix hash drifted from the locked value"
    );
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// The raw embedded matrix bytes, for verifying the frozen artifact ships in the
/// binary with no filesystem dependency. The bytes are validated against the
/// locked sha256 and length on load via [`load_p`]; this exposes them so a test
/// can assert the embed is exactly the frozen 15,360,000-byte matrix.
pub fn embedded_matrix_bytes() -> &'static [u8] {
    FROZEN_P_BYTES
}

/// Load the frozen projection matrix once, self-checking its sha256 and length.
///
/// The matrix comes from the embedded bytes by default, so the load never
/// depends on the filesystem and can never panic on a missing fixture. An
/// explicit `IAI_MCP_HDC_FIXTURES` override is honored only when it is set and a
/// `projection.bin` exists under it; otherwise the embedded bytes are used.
/// Either way the bytes are validated against the locked sha256 and length.
pub fn load_p() -> &'static [f32] {
    P.get_or_init(|| match override_bytes() {
        Some(bytes) => decode_matrix(&bytes),
        None => decode_matrix(FROZEN_P_BYTES),
    })
}

/// Project a 384-d embedding into 1250 packed bytes.
///
/// Validates the length and finiteness of the input, computes `emb @ P` in
/// f32, packs `sign(projected) >= 0` MSB-first.
pub fn project_impl(emb: &[f32]) -> Result<Vec<u8>, HvError> {
    if emb.len() != EMBED_DIM {
        return Err(HvError::Length {
            expected: EMBED_DIM,
            actual: emb.len(),
        });
    }
    if emb.iter().any(|x| !x.is_finite()) {
        return Err(HvError::NonFinite);
    }
    let p = load_p();

    // projected[j] = sum_i emb[i] * P[i * HV_DIM + j]   (C-order P[384,10000])
    let mut acc = vec![0f32; HV_DIM];
    for i in 0..EMBED_DIM {
        let e = emb[i];
        let row = &p[i * HV_DIM..(i + 1) * HV_DIM];
        for (a, &pij) in acc.iter_mut().zip(row) {
            *a += e * pij;
        }
    }

    let mut out = vec![0u8; HV_BYTES];
    for (j, &v) in acc.iter().enumerate() {
        if v >= 0.0 {
            out[j / 8] |= 1 << (7 - (j % 8));
        }
    }
    Ok(out)
}

// --- PyO3 surface -----------------------------------------------------------

#[pyfunction]
#[pyo3(name = "project")]
pub fn project<'py>(
    py: Python<'py>,
    emb: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyBytes>> {
    let emb = emb.as_slice()?;
    let packed = py.allow_threads(|| project_impl(emb))?;
    Ok(PyBytes::new_bound(py, &packed))
}
