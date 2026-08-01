//! Brandes 2001 betweenness centrality via
//! `rustworkx_core::centrality::betweenness_centrality`.
//!
//! UNWEIGHTED BFS-Brandes. The rustworkx-core 0.17 API signature is
//! `betweenness_centrality(graph, include_endpoints: bool, normalized: bool,
//! parallel_threshold: usize) -> Vec<Option<f64>>` — there is no weight-map
//! parameter, so the algorithm runs BFS hop counts only. The previous
//! networkx-weighted-Brandes semantic (`weight="weight"` on Hebbian edge
//! strengths) is intentionally dropped at the source-of-truth crate
//! boundary; the operator-visible behavioural change is documented in the
//! CHANGELOG entry that ships alongside the σ-assembly plan.
//!
//! The PyO3 entry point releases the GIL during the Rust kernel via
//! `py.allow_threads(move || ...)`. Brandes at N≥10k can take seconds,
//! and the daemon's status handler must stay responsive on multi-second
//! kernels — same discipline as `shortest::average_shortest_path_length`.
//!
//! Return contract: a 2-tuple of owned numpy arrays.
//!   1. `centrality: PyArray1<f64>` — one entry per CSR row, in CSR row
//!      order. `Option<f64>::None` from rustworkx-core is unwrapped to
//!      `0.0` (matches networkx's behaviour for isolated nodes).
//!   2. `node_indices: PyArray1<i64>` — `[0, 1, …, n_nodes - 1]`, the CSR
//!      row indices in the order the Python wrapper must consume to map
//!      each scalar back to a node identifier. The wrapper does NOT
//!      assume any other order — if a future kernel returns the array in
//!      a different order, the Python consumer simply iterates
//!      `zip(node_arr, centrality_arr)` and maps via its own CSR-row-to-
//!      UUID table.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rustworkx_core::centrality::betweenness_centrality as rwx_bc;
use rustworkx_core::petgraph::graph::UnGraph;

use crate::error::GraphError;

/// Parallel-threshold node count above which rustworkx-core's Brandes
/// kernel switches from serial BFS-from-each-source to a rayon-parallel
/// fan-out. The crate docs suggest 50; we honour that default so the
/// rayon path receives consistent coverage on every non-trivial fixture
/// (the smallest production graphs already exceed N=50 once daemon
/// uptime crosses ~1 day at typical capture cadence).
const PARALLEL_THRESHOLD: usize = 50;

/// Build an undirected petgraph from the CSR triple.
///
/// Mirrors the connectivity-side helper of the same name but uses the
/// top-level `petgraph` (no `rustworkx_core::petgraph` re-export) because
/// rustworkx-core 0.17's `betweenness_centrality` is generic over
/// `NodeIndexable + IntoNodeIdentifiers + ...` and the bound holds for
/// both `petgraph::graph::UnGraph` and the rustworkx-core re-export
/// equivalently. Using the local `petgraph` keeps the dependency surface
/// minimal — no extra trait-bound debugging needed.
///
/// Returns `Err(GraphError::InvalidNodeId)` on:
///   * indptr length mismatch (must equal `n_nodes + 1`),
///   * any negative `indices[i]`,
///   * any `indices[i] >= n_nodes` (out-of-range neighbour).
///
/// Edges are de-duplicated via the `u <= v` guard — the CSR layout
/// double-lists each undirected edge (once per endpoint); inserting only
/// the canonical orientation prevents duplicate-edge multigraph
/// behaviour that would inflate the BFS-derived centrality denominator.
fn build_graph_from_csr(
    indptr: &[i64],
    indices: &[i64],
    n_nodes: usize,
) -> Result<UnGraph<(), ()>, GraphError> {
    if indptr.len() != n_nodes + 1 {
        return Err(GraphError::InvalidNodeId(format!(
            "indptr length {} != n_nodes + 1 ({})",
            indptr.len(),
            n_nodes + 1
        )));
    }

    let mut g: UnGraph<(), ()> =
        UnGraph::<(), ()>::with_capacity(n_nodes, indices.len() / 2);
    // Pre-allocate node identifiers so isolated nodes (rows with no
    // outgoing edges in the CSR) still appear in `node_count()` and
    // receive a `Some(0.0)` slot in the rustworkx-core output. Without
    // this seed step a leaf row with no edges would silently disappear
    // and the Python wrapper would zip a shorter centrality_arr against
    // a full UUID list.
    let mut node_ids = Vec::with_capacity(n_nodes);
    for _ in 0..n_nodes {
        node_ids.push(g.add_node(()));
    }

    for u in 0..n_nodes {
        let start = indptr[u] as usize;
        let end = indptr[u + 1] as usize;
        for k in start..end {
            let v_raw = indices[k];
            if v_raw < 0 {
                return Err(GraphError::InvalidNodeId(format!(
                    "indices[{k}] = {v_raw} is negative"
                )));
            }
            let v = v_raw as usize;
            if v >= n_nodes {
                return Err(GraphError::InvalidNodeId(format!(
                    "indices[{k}] = {v} >= n_nodes ({n_nodes})"
                )));
            }
            // The CSR lists each undirected edge twice (once per
            // endpoint). Insert only the canonical (u, v) orientation
            // with `u <= v` so each undirected edge becomes exactly one
            // petgraph edge. Self-loops (u == v) pass the inclusive
            // guard and are added once — Brandes treats self-loops as
            // length-0 paths that don't contribute to centrality, so
            // the unweighted BFS-Brandes result is correct either way.
            if u <= v {
                g.add_edge(node_ids[u], node_ids[v], ());
            }
        }
    }

    Ok(g)
}

/// `iai_mcp_native.graph.betweenness_centrality(indptr, indices, n_nodes,
/// normalized=True) -> (centrality: numpy.ndarray[f64], node_indices:
/// numpy.ndarray[i64])`.
///
/// Build an undirected graph from the CSR buffers, run rustworkx-core's
/// Brandes 2001 kernel under `py.allow_threads(...)`, and return the
/// centrality scalars plus an explicit CSR-row-index array so the Python
/// consumer can map each scalar back to its node identifier without
/// assuming a coincidence between insertion order and CSR row order.
///
/// **UNWEIGHTED.** rustworkx-core 0.17's Brandes signature does not
/// accept a weight map; the `data` slice produced alongside `indptr` and
/// `indices` by `MemoryGraph.to_csr_arrays()` is discarded at the
/// PyO3-call boundary. Operator-visible behavioural change documented in
/// the σ-assembly plan's CHANGELOG entry.
/// `(scores, node_ids)` — per-node betweenness paired with the CSR row order
/// it was computed in, so callers never have to assume the two line up.
type BetweennessResult = (Py<PyArray1<f64>>, Py<PyArray1<i64>>);

#[pyfunction]
#[pyo3(signature = (indptr, indices, n_nodes, normalized=true))]
pub fn betweenness_centrality(
    py: Python<'_>,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i64>,
    n_nodes: usize,
    normalized: bool,
) -> PyResult<BetweennessResult> {
    let indptr_slice = indptr.as_slice()?;
    let indices_slice = indices.as_slice()?;

    if indptr_slice.len() != n_nodes + 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "indptr length {} does not match n_nodes + 1 = {}",
            indptr_slice.len(),
            n_nodes + 1
        )));
    }

    // Snapshot the numpy borrows into owned `Vec<i64>` so the compute
    // kernel can drop the GIL — `PyReadonlyArray1` borrows are GIL-bound
    // and cannot survive `py.allow_threads`. Copying ~10^4 i64 entries
    // is cheap relative to the O(V·(V+E)) Brandes BFS that follows.
    let indptr_owned: Vec<i64> = indptr_slice.to_vec();
    let indices_owned: Vec<i64> = indices_slice.to_vec();

    let centrality_vec = py.allow_threads(move || -> Result<Vec<f64>, GraphError> {
        let graph = build_graph_from_csr(&indptr_owned, &indices_owned, n_nodes)?;
        // include_endpoints=false matches networkx's
        // `betweenness_centrality` default; the differential parity gate
        // compares against networkx with no `endpoints=` override.
        let raw: Vec<Option<f64>> =
            rwx_bc(&graph, false, normalized, PARALLEL_THRESHOLD);
        // Unwrap `None` to `0.0`. networkx assigns `0.0` to isolated
        // nodes; rustworkx-core emits `None` for any node index that
        // exists in the bound but was never visited. The two semantics
        // agree because every CSR row 0..n_nodes is `add_node`'d above
        // and therefore has at least an empty neighbour list — the
        // unwrap path is exercised on isolated nodes only.
        Ok(raw.into_iter().map(|opt| opt.unwrap_or(0.0)).collect())
    })?;

    // Build the explicit CSR-row-index array. The Python wrapper
    // consumes this in lockstep with `centrality_vec` via
    // `zip(node_arr, centrality_arr)` — if a future kernel reorders the
    // centrality vector, the consumer's map-back to UUIDs remains
    // correct because each scalar carries its row index alongside.
    let node_indices_vec: Vec<i64> = (0..n_nodes as i64).collect();

    let centrality_py: Py<PyArray1<f64>> = centrality_vec.into_pyarray_bound(py).unbind();
    let node_indices_py: Py<PyArray1<i64>> =
        node_indices_vec.into_pyarray_bound(py).unbind();

    Ok((centrality_py, node_indices_py))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_graph_rejects_indptr_length_mismatch() {
        let indptr = vec![0_i64, 1, 2]; // length 3, claims n_nodes = 5
        let indices = vec![1_i64, 0];
        let err = build_graph_from_csr(&indptr, &indices, 5).unwrap_err();
        match err {
            GraphError::InvalidNodeId(msg) => assert!(msg.contains("indptr length")),
            other => panic!("expected InvalidNodeId, got {other:?}"),
        }
    }

    #[test]
    fn build_graph_rejects_out_of_range_neighbour() {
        let indptr = vec![0_i64, 1, 1];
        let indices = vec![99_i64]; // 99 >= n_nodes (2)
        let err = build_graph_from_csr(&indptr, &indices, 2).unwrap_err();
        match err {
            GraphError::InvalidNodeId(msg) => assert!(msg.contains("99")),
            other => panic!("expected InvalidNodeId, got {other:?}"),
        }
    }

    #[test]
    fn star_graph_hub_dominates_leaves() {
        // 5-node star: 0 is the hub; 1..4 are leaves. CSR lists each
        // hub-leaf edge twice (hub-leaf and leaf-hub).
        //   indptr: [0, 4, 5, 6, 7, 8]
        //   indices: [1,2,3,4, 0, 0, 0, 0]
        let indptr = vec![0_i64, 4, 5, 6, 7, 8];
        let indices = vec![1_i64, 2, 3, 4, 0, 0, 0, 0];
        let g = build_graph_from_csr(&indptr, &indices, 5).unwrap();
        let bc: Vec<Option<f64>> = rwx_bc(&g, false, true, PARALLEL_THRESHOLD);
        let hub = bc[0].unwrap();
        for leaf_idx in 1..5 {
            let leaf = bc[leaf_idx].unwrap();
            assert!(
                hub > leaf,
                "hub centrality ({hub}) should beat leaf {leaf_idx} ({leaf})"
            );
        }
    }

    #[test]
    fn empty_graph_returns_empty_vec() {
        let indptr = vec![0_i64];
        let indices: Vec<i64> = vec![];
        let g = build_graph_from_csr(&indptr, &indices, 0).unwrap();
        let bc: Vec<Option<f64>> = rwx_bc(&g, false, true, PARALLEL_THRESHOLD);
        assert!(bc.is_empty(), "empty graph should yield empty centrality");
    }

    #[test]
    fn isolated_nodes_unwrap_to_zero() {
        // 3 isolated nodes (no edges). Every row has Some(0.0).
        let indptr = vec![0_i64, 0, 0, 0];
        let indices: Vec<i64> = vec![];
        let g = build_graph_from_csr(&indptr, &indices, 3).unwrap();
        let bc: Vec<Option<f64>> = rwx_bc(&g, false, true, PARALLEL_THRESHOLD);
        assert_eq!(bc.len(), 3);
        for (i, c) in bc.iter().enumerate() {
            assert_eq!(c.unwrap_or(-1.0), 0.0, "node {i} should be 0.0");
        }
    }
}
