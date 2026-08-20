//! iai_mcp_native — single-cdylib wheel exposing five Python sub-modules
//! (`embed` + `graph` + `hd` + `store` + `engine`).
//!
//! The core crates (`iai_mcp_embed_core`, `iai_mcp_graph_core`, `lilli-hd`,
//! `lillibrain`, and `lilliengine`) are plain `rlib`s with no `#[pymodule]`
//! entry of their own —
//! instead they each expose a `register(py, m)` helper that this wrapper calls
//! from inside its `#[pymodule]` body. The result is one `.so` file with three
//! logical Python sub-modules:
//!
//! ```python
//! from iai_mcp_native import embed, graph, hd
//! e = embed.Embedder()
//! v = graph.answer()
//! p = hd.project(emb)
//! ```
//!
//! The wrapper also registers the dotted sub-module names into
//! `sys.modules` so `import iai_mcp_native.embed` works as a stand-alone
//! statement, not just `from iai_mcp_native import embed`. This is the
//! workaround documented in the Maturin Book for PyO3 sub-modules; without
//! it the dotted-import path raises `ModuleNotFoundError`.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::define_stub_info_gatherer;

#[pymodule]
fn iai_mcp_native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Each submodule is constructed with its fully-qualified dotted name so the
    // module __name__ matches its sys.modules key. Classes defined inside then
    // carry a resolvable __module__, which pickle and inspect rely on. The
    // submodule is attached to the parent under its SHORT attribute name so the
    // `from iai_mcp_native import embed` form keeps working — `add_submodule`
    // would otherwise derive the attribute from the dotted __name__.

    // Embedder sub-module — Bert / bge-small-en-v1.5 forward pass.
    let embed = PyModule::new(py, "iai_mcp_native.embed")?;
    iai_mcp_embed_core::register(py, &embed)?;
    m.add("embed", &embed)?;

    // Graph sub-module — pure-Rust algorithms layer (currently a wiring
    // probe; real algorithm work begins in later plans).
    let graph = PyModule::new(py, "iai_mcp_native.graph")?;
    iai_mcp_graph_core::register(py, &graph)?;
    m.add("graph", &graph)?;

    // Vector-index sub-module — exact cosine search over the record
    // embeddings, the recall index under Hippo.
    let vec = PyModule::new(py, "iai_mcp_native.vec")?;
    iai_mcp_vec_core::register(py, &vec)?;
    m.add("vec", &vec)?;

    // Hypervector sub-module — BSC / FHRR / sparse VSA bit-kernels plus the
    // frozen SimHash projection-apply.
    let hd = PyModule::new(py, "iai_mcp_native.hd")?;
    lilli_hd::register(py, &hd)?;
    m.add("hd", &hd)?;

    // Store sub-module — the paged record store (pager + B-tree + write-ahead
    // log) exposed for storage-level differential testing.
    let store = PyModule::new(py, "iai_mcp_native.store")?;
    lillibrain::register(py, &store)?;
    m.add("store", &store)?;

    // Engine sub-module — the sqlite3-shaped SQL engine (Connection / Cursor /
    // Row / RawConn) over the record store, the storage driver under Hippo.
    let engine = PyModule::new(py, "iai_mcp_native.engine")?;
    lilliengine::register(py, &engine)?;
    m.add("engine", &engine)?;

    // Register the dotted sub-module names in `sys.modules` so a separate
    // `import iai_mcp_native.embed` statement also resolves. Without this
    // step, only `from iai_mcp_native import embed` works.
    let sys_modules: Bound<'_, PyDict> = py.import("sys")?.getattr("modules")?.cast_into()?;
    sys_modules.set_item("iai_mcp_native.embed", &embed)?;
    sys_modules.set_item("iai_mcp_native.graph", &graph)?;
    sys_modules.set_item("iai_mcp_native.vec", &vec)?;
    sys_modules.set_item("iai_mcp_native.hd", &hd)?;
    sys_modules.set_item("iai_mcp_native.store", &store)?;
    sys_modules.set_item("iai_mcp_native.engine", &engine)?;

    Ok(())
}

// Stub-metadata gatherer for the `stub_gen` binary. The macro walks the
// `#[gen_stub_*]` attributes declared in the consumed core crates because
// each crate runs its own `define_stub_info_gatherer!(stub_info)` and the
// wrapper aggregates them at build time.
define_stub_info_gatherer!(stub_info);

// Must live in this lib crate, not the `stub_gen` binary — `from_project_root`
// walks the process-wide `inventory` registry, which only contains the
// `#[gen_stub_*]` items linked in via this crate's dependency graph.
pub fn stub_info_staged() -> pyo3_stub_gen::Result<pyo3_stub_gen::StubInfo> {
    pyo3_stub_gen::StubInfo::from_project_root(
        "iai_mcp_native".to_string(),
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("stubs"),
        true,
        pyo3_stub_gen::StubGenConfig::default(),
    )
}
