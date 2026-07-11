//! iai_mcp_embed_core — Rust BERT forward pass for bge-small-en-v1.5.
//!
//! Public API:
//! ```python
//! from iai_mcp_native import embed
//! e = embed.Embedder()           # weights load in __new__
//! v = e.encode("hello")          # 384-dim L2-normalized list[float]
//! ```
//!
//! The PyO3 `Embedder` is a thin facade over `bert::BertEmbedder`. The Rust
//! struct is the source of truth and is testable from `cargo nextest` without
//! a Python interpreter; the PyO3 wrapper only adapts errors and arg types.
//!
//! This crate is a pure `rlib` — it no longer ships its own wheel. The native
//! wrapper crate (`iai_mcp_native`) consumes the `register(py, m)` helper to
//! expose `Embedder` as a Python submodule.

// Pitfall 4 / Pattern 6: extern crate forces the linker to include Apple
// Accelerate BLAS symbols. Without this, `maturin build --features accelerate`
// succeeds but importing the wheel at runtime fails with "symbol not found".
#[cfg(feature = "accelerate")]
extern crate accelerate_src;

pub mod bert;
pub mod error;

use pyo3::prelude::*;
use pyo3_stub_gen::{define_stub_info_gatherer, derive::*};

use crate::bert::BertEmbedder;

/// Rust BERT embedder for BAAI/bge-small-en-v1.5 (384-dim L2-normalized output).
///
/// Constructed eagerly: weights load from the HF cache in `__new__`. If
/// `IAI_MCP_EMBED_OFFLINE=1` is set, construction fails loudly when any
/// required file is missing rather than triggering a silent network download.
///
/// The `module` attribute attributes this class to `iai_mcp_native.embed`
/// in the generated `.pyi` stub — pyo3-stub-gen derives parent/child module
/// relationships from these dotted names so a single stub file can describe
/// the sub-module tree consumed at runtime.
#[gen_stub_pyclass]
#[pyclass(module = "iai_mcp_native.embed")]
pub struct Embedder {
    inner: BertEmbedder,
}

#[gen_stub_pymethods]
#[pymethods]
impl Embedder {
    /// Load bge-small-en-v1.5 weights from the HF cache (lazy download on miss
    /// unless `IAI_MCP_EMBED_OFFLINE=1`).
    ///
    /// The GIL is released for the duration of the model load so that
    /// background warm threads cannot block the main thread.
    /// `BertEmbedder::load` is pure Rust and holds no Python objects,
    /// making this safe.
    #[new]
    fn py_new(py: Python<'_>) -> PyResult<Self> {
        let inner = py
            .allow_threads(|| BertEmbedder::load())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Encode a single text string to a 384-dim L2-normalized vector.
    ///
    /// The GIL is released for the duration of the BERT forward pass so that
    /// concurrent socket clients dispatched via `asyncio.to_thread` can run
    /// their inference in parallel. `BertEmbedder::encode` is pure Rust and
    /// holds no Python objects, making this safe.
    fn encode(&self, py: Python<'_>, text: &str) -> PyResult<Vec<f32>> {
        py.allow_threads(|| self.inner.encode(text))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

/// Register the embedder surface (currently the `Embedder` class) on a Python
/// module owned by an outer wrapper crate. Called by `iai_mcp_native::lib.rs`
/// during its `#[pymodule]` entry point so this crate can stay free of
/// `#[pymodule]` and the `extension-module` feature.
pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Embedder>()?;
    Ok(())
}

// Required by pyo3-stub-gen: collects stub metadata from all #[gen_stub_*]
// macros and exposes a `stub_info()` function for the wrapper crate's
// stub_gen binary to consume.
define_stub_info_gatherer!(stub_info);
