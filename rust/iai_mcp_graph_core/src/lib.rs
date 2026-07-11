//! iai_mcp_graph_core — pure-Rust graph algorithm layer.
//!
//! Strict dependency isolation: no embedder dependencies, no candle, no
//! accelerate, no tokenizers. The crate ships as a workspace `rlib` and is
//! re-exported as a Python submodule by the `iai_mcp_native` wrapper crate.
//!
//! Wave-1 stub surface: a single `answer() -> i64` returning the literal
//! `42`. The point of this skeleton is to validate the three-crate workspace
//! topology + the PyO3 sub-module wiring before any algorithm work begins.

pub mod centrality;
pub mod clustering;
pub mod connectivity;
pub mod error;
pub mod generators;
pub mod shortest;

use pyo3::prelude::*;
use pyo3_stub_gen::{define_stub_info_gatherer, derive::*};

/// Wave-1 wiring probe. Returning the literal `42` lets a downstream Python
/// smoke test prove that:
///   1. the wrapper crate exposes the `graph` sub-module successfully, and
///   2. the cross-crate `register(py, m)` indirection actually mounts a
///      live callable on the resulting PyModule.
///
/// Algorithm work begins in subsequent plans; this stub is the contract
/// the next wave will replace. The `module = "iai_mcp_native.graph"`
/// argument tells pyo3-stub-gen where to place this function in the
/// generated `.pyi`.
#[gen_stub_pyfunction(module = "iai_mcp_native.graph")]
#[pyfunction]
pub fn answer() -> i64 {
    42
}

/// Register the graph algorithm surface on a Python module owned by an
/// outer wrapper crate. Mirror of `iai_mcp_embed_core::register` — the two
/// crates share the same shape so the wrapper code that mounts them is
/// uniform.
pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(answer, m)?)?;
    m.add_function(wrap_pyfunction!(centrality::betweenness_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(clustering::average_clustering, m)?)?;
    m.add_function(wrap_pyfunction!(connectivity::connected_components, m)?)?;
    m.add_function(wrap_pyfunction!(connectivity::is_connected, m)?)?;
    m.add_function(wrap_pyfunction!(connectivity::selfloop_edges, m)?)?;
    m.add_function(wrap_pyfunction!(generators::gnm_random_graph, m)?)?;
    m.add_function(wrap_pyfunction!(shortest::average_shortest_path_length, m)?)?;
    Ok(())
}

// Stub-metadata gatherer used by the wrapper crate's stub_gen binary.
define_stub_info_gatherer!(stub_info);
