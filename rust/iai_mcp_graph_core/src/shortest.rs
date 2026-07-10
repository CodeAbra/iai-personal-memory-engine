//! Average shortest path length composer with largest-connected-component
//! guard. Composes from `rustworkx_core::shortest_path::distance_matrix`
//! (unweighted BFS for every source) and folds the result down to a scalar
//! by summing finite off-diagonal entries and dividing by N·(N−1).
//!
//! Disconnected-graph behaviour: the graph's largest connected component is
//! extracted before the distance-matrix call. NetworkX raises on
//! disconnected input to `average_shortest_path_length`; our PyO3 entry
//! mirrors the `_largest_cc`-guarded call chain used by the small-world
//! coefficient implementation.
//!
//! The PyO3 entry point releases the GIL during the Rust kernel via
//! `py.allow_threads(|| ...)`. `distance_matrix` is O(V·(V+E)) and at the
//! N≥2000 scale the daemon's status handler must stay responsive.
//!
//! Unreachable-cell handling: `distance_matrix` accepts a `null_value: f64`
//! sentinel for "no path"; we pass `f64::INFINITY` and filter via
//! `dist.is_finite()` before summing. The largest-CC guard makes that
//! filter redundant in practice (every off-diagonal cell is reachable on
//! one connected component), but the explicit `is_finite()` check is
//! deliberately retained so future refactors can drop the guard without
//! corrupting the sum with a sentinel leak.

use std::collections::HashSet;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
// `rustworkx-core` re-exports its own `petgraph` to keep its trait bounds
// consistent across the public surface. We use that re-export instead of
// the top-level `petgraph` crate so the `UnGraphMap` value we hand to
// `distance_matrix` and `connected_components` carries the matching
// `GraphProp` / `Visitable` / `IntoNeighbors*` impls — otherwise a
// "multiple different versions of crate `petgraph`" trait-resolution
// error fires at the call site.
use rustworkx_core::petgraph::graphmap::UnGraphMap;
use rustworkx_core::connectivity::connected_components;
use rustworkx_core::shortest_path::distance_matrix;

/// Parallel-threshold node count above which `distance_matrix` switches
/// from serial BFS to rayon-parallel BFS. The crate docs suggest 300; we
/// pin 50 so the four-decade σ-input range (karate N=34 through
/// live_n2000 N≥2000) crosses the boundary on every non-trivial fixture
/// and the rayon path receives consistent test coverage.
const PARALLEL_THRESHOLD: usize = 50;

/// CSR → `UnGraphMap<i64, ()>` constructor. Nodes are inserted up-front
/// (including isolates) so the produced graph's `node_count()` matches
/// `n_nodes` exactly; without this step a source with no outgoing edges
/// would silently disappear and the APSL denominator would be wrong.
///
/// The CSR layout matches the slicing idiom used elsewhere in this crate:
/// row `u` covers `indices[indptr[u]..indptr[u + 1]]`. Reverse edges are
/// expected to be present in the CSR (the source-side caller materializes
/// undirected adjacency before handing the buffers to PyO3); `add_edge`
/// is idempotent for `UnGraphMap`, so a double-listed edge is a no-op.
fn build_ungraph_from_csr(
    indptr: &[i64],
    indices: &[i64],
    n_nodes: usize,
) -> UnGraphMap<i64, ()> {
    let mut g: UnGraphMap<i64, ()> =
        UnGraphMap::with_capacity(n_nodes, indices.len() / 2);
    for u in 0..n_nodes {
        g.add_node(u as i64);
    }
    for u in 0..n_nodes {
        let start = indptr[u] as usize;
        let end = indptr[u + 1] as usize;
        for &v in &indices[start..end] {
            g.add_edge(u as i64, v, ());
        }
    }
    g
}

/// Return a fresh `UnGraphMap` containing only the nodes (and edges
/// between them) of the largest connected component of `graph`. Mirrors
/// the `nx.connected_components(g)` → `max(..., key=len)` → `g.subgraph(...)
/// .copy()` pattern from the small-world coefficient implementation.
///
/// On an empty input the empty graph is returned. On an already-connected
/// input a 1-component clone is returned — the post-call code path is
/// identical, no `Option` indirection needed.
fn largest_connected_component_subgraph(
    graph: &UnGraphMap<i64, ()>,
) -> UnGraphMap<i64, ()> {
    let components = connected_components(graph);
    let largest = match components.iter().max_by_key(|c| c.len()) {
        Some(c) => c,
        None => return UnGraphMap::new(),
    };
    let keep: HashSet<i64> = largest.iter().copied().collect();
    let mut sub: UnGraphMap<i64, ()> =
        UnGraphMap::with_capacity(keep.len(), keep.len());
    for &n in &keep {
        sub.add_node(n);
    }
    for (a, b, _) in graph.all_edges() {
        if keep.contains(&a) && keep.contains(&b) {
            sub.add_edge(a, b, ());
        }
    }
    sub
}

/// Compute APSL on an already-connected subgraph. Returns `0.0` for the
/// empty and singleton cases (matches `networkx.average_shortest_path_length`).
fn average_shortest_path_length_on_connected_subgraph(
    subgraph: &UnGraphMap<i64, ()>,
) -> f64 {
    let n = subgraph.node_count();
    if n <= 1 {
        return 0.0;
    }
    // `null_value = f64::INFINITY` so unreachable cells are easy to filter
    // out below via `is_finite()`. On the largest-CC input this matters
    // only for the diagonal (which is 0.0 and we skip it explicitly), but
    // the explicit finite filter keeps the kernel correct under any
    // future refactor that removes the largest-CC guard.
    let dm = distance_matrix(subgraph, PARALLEL_THRESHOLD, false, f64::INFINITY);
    let mut sum: f64 = 0.0;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = dm[(i, j)];
            if d.is_finite() {
                sum += d;
            }
        }
    }
    // N·(N−1) ordered pairs; sum already iterates over ordered (i, j).
    sum / ((n * (n - 1)) as f64)
}

/// `iai_mcp_native.graph.average_shortest_path_length(indptr, indices, n_nodes) -> float`.
///
/// Build an undirected graph from the CSR buffers, take its largest
/// connected component, and return the average shortest path length on
/// that component. The Rust kernel runs under `py.allow_threads(|| ...)`
/// so the daemon's other Python callers stay responsive on N≥2000
/// inputs.
///
/// **Stub-gen note:** `#[gen_stub_pyfunction]` is intentionally NOT
/// applied here. `pyo3-stub-gen 0.6` (the only line compatible with the
/// workspace's `pyo3 0.22` pin) does not implement `PyStubType` for
/// `numpy::PyReadonlyArray1<T>`; annotating the function emits a `the
/// trait bound … : PyStubType is not satisfied` compile error.
/// Downstream callers that need a typed `.pyi` entry should treat the
/// function as `def average_shortest_path_length(indptr: np.ndarray,
/// indices: np.ndarray, n_nodes: int) -> float: ...` and live with a
/// `--allow-untyped-call` exclusion until pyo3-stub-gen catches up.
#[pyfunction]
pub fn average_shortest_path_length(
    py: Python<'_>,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i64>,
    n_nodes: usize,
) -> PyResult<f64> {
    let indptr_slice = indptr.as_slice()?;
    let indices_slice = indices.as_slice()?;

    if indptr_slice.len() != n_nodes + 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "indptr length {} does not match n_nodes + 1 = {}",
            indptr_slice.len(),
            n_nodes + 1
        )));
    }

    // Snapshot the slices into owned buffers so the compute kernel can
    // run outside the GIL. The numpy borrows are GIL-bound; copying ~10⁴
    // i64 entries is cheap relative to the O(V²) BFS that follows.
    let indptr_owned: Vec<i64> = indptr_slice.to_vec();
    let indices_owned: Vec<i64> = indices_slice.to_vec();

    let result = py.allow_threads(move || {
        let graph =
            build_ungraph_from_csr(&indptr_owned, &indices_owned, n_nodes);
        let largest_cc = largest_connected_component_subgraph(&graph);
        average_shortest_path_length_on_connected_subgraph(&largest_cc)
    });

    Ok(result)
}
