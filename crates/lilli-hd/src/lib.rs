//! Hyperdimensional computing kernels for the memory substrate.
//!
//! Provides the three hypervector tiers used across the brain: binary spatter
//! codes (BSC) for episodic memory, Fourier holographic reduced
//! representations (FHRR) for semantic memory, and sparse vector-symbolic
//! arithmetic (sparse VSA) for procedural memory. Also applies the frozen
//! 384-to-10000 SimHash projection that lifts embedder vectors into the
//! hypervector space.
//!
//! This crate carries the bit-level kernels only; higher-level memory policy
//! lives in the host. The crate API takes raw bytes and floats and returns raw
//! bytes and floats — no secret material and no persistence handle crosses this
//! boundary, and no telemetry is emitted here.

pub mod bsc;
pub mod codebook;
pub mod errors;
pub mod fhrr;
pub mod projection;
pub mod sparse;

use pyo3::prelude::*;

/// Register the hypervector kernels on a Python module owned by the wrapper
/// crate. Mirrors the embedder and graph crates so the wrapper mounts all
/// native sub-modules uniformly.
pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(projection::project, m)?)?;

    m.add_function(wrap_pyfunction!(bsc::bind, m)?)?;
    m.add_function(wrap_pyfunction!(bsc::bundle, m)?)?;
    m.add_function(wrap_pyfunction!(bsc::permute, m)?)?;
    m.add_function(wrap_pyfunction!(bsc::hamming, m)?)?;
    m.add_function(wrap_pyfunction!(bsc::cosine_packed, m)?)?;

    m.add_function(wrap_pyfunction!(fhrr::bind, m)?)?;
    m.add_function(wrap_pyfunction!(fhrr::unbind, m)?)?;
    m.add_function(wrap_pyfunction!(fhrr::bundle, m)?)?;
    m.add_function(wrap_pyfunction!(fhrr::permute, m)?)?;
    m.add_function(wrap_pyfunction!(fhrr::similarity, m)?)?;

    m.add_function(wrap_pyfunction!(sparse::bind, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::unbind, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::bundle, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::permute, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::similarity, m)?)?;
    Ok(())
}
