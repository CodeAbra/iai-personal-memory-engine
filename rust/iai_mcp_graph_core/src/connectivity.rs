//! Connectivity primitives — connected components, is_connected, self-loop edge filter.
//!
//! Three PyO3-exposed callables backed by the canonical
//! `rustworkx_core::connectivity` BFS-undirected implementation. The CSR
//! triple (`indptr`, `indices`, `n_nodes`) is the workspace-wide input shape
//! for every Rust algorithm in this crate — see crate-level docs for the
//! convention.
//!
//! ## Empty-graph contract
//!
//! `is_connected(n_nodes == 0)` raises `ValueError` so the native API mirrors
//! `networkx.is_connected(nx.Graph())`, which raises `NetworkXPointlessConcept`
//! ("Connectivity is undefined for the null graph"). This matches the dev pin
//! verified against networkx 3.6.1.
//!
//! ## Determinism
//!
//! `rustworkx_core::connectivity::connected_components` returns components as
//! `Vec<HashSet<NodeId>>` — iteration order over a `HashSet` is unstable
//! across runs. The wrappers sort each component's node IDs ascending before
//! returning so the Python-side output is deterministic.
//!
//! ## GIL release
//!
//! Pure-Rust kernels run inside `Python::detach` — Python threads can
//! make progress while the BFS runs.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
// The graph algorithm crate depends on `petgraph = "0.6"` for its own type
// surface, but `rustworkx-core` ships its own pinned `petgraph = "0.8"` and
// implements the trait bounds for that version. Re-export the matching
// types from rustworkx-core so the algorithms see a single petgraph copy.
use rustworkx_core::connectivity::connected_components as rwx_cc;
use rustworkx_core::petgraph::graph::{NodeIndex, UnGraph};

use crate::error::GraphError;

/// Build an undirected petgraph from the CSR triple.
///
/// Returns `Err(GraphError::InvalidNodeId)` on:
///   * indptr length mismatch (must equal `n_nodes + 1`),
///   * any `indices[i] >= n_nodes` (out-of-range neighbour).
///
/// Edges are de-duplicated implicitly — the CSR stores each undirected edge
/// in both endpoints' neighbour lists, so iterating with the `u < v` guard
/// adds each edge exactly once. Self-loops (`u == v`) are added once.
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

    let mut g = UnGraph::<(), ()>::with_capacity(n_nodes, indices.len() / 2);
    let mut node_ids = Vec::with_capacity(n_nodes);
    for _ in 0..n_nodes {
        node_ids.push(g.add_node(()));
    }

    for u in 0..n_nodes {
        let start = *indptr
            .get(u)
            .ok_or_else(|| GraphError::InvalidNodeId(format!("indptr[{u}] out of range")))?
            as usize;
        let end = *indptr
            .get(u + 1)
            .ok_or_else(|| GraphError::InvalidNodeId(format!("indptr[{}] out of range", u + 1)))?
            as usize;

        for k in start..end {
            let v_raw = *indices
                .get(k)
                .ok_or_else(|| GraphError::InvalidNodeId(format!("indices[{k}] out of range")))?;
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
            // Each undirected edge appears twice in the CSR (once per endpoint).
            // Add it once with the u <= v guard. Self-loops (u == v) are added
            // because the guard is inclusive.
            if u <= v {
                g.add_edge(node_ids[u], node_ids[v], ());
            }
        }
    }

    Ok(g)
}

/// Return the connected components of an undirected graph as a list of
/// sorted node-id lists.
///
/// Empty graph (`n_nodes == 0`) returns an empty list — matches
/// `nx.connected_components(nx.Graph())` which yields nothing.
#[pyfunction]
pub fn connected_components(
    py: Python<'_>,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i64>,
    n_nodes: usize,
) -> PyResult<Vec<Vec<i64>>> {
    let indptr_slice = indptr.as_slice()?;
    let indices_slice = indices.as_slice()?;

    let result = py.detach(|| -> Result<Vec<Vec<i64>>, GraphError> {
        let g = build_graph_from_csr(indptr_slice, indices_slice, n_nodes)?;
        let components = rwx_cc(&g);
        let mut out: Vec<Vec<i64>> = components
            .into_iter()
            .map(|set| {
                let mut ids: Vec<i64> = set
                    .into_iter()
                    .map(|nid: NodeIndex| nid.index() as i64)
                    .collect();
                ids.sort_unstable();
                ids
            })
            .collect();
        // Stable outer ordering: sort components by their smallest node id.
        out.sort_unstable_by_key(|comp| comp.first().copied().unwrap_or(i64::MAX));
        Ok(out)
    })?;

    Ok(result)
}

/// Return ``True`` iff the graph has exactly one connected component.
///
/// Raises ``ValueError`` for ``n_nodes == 0`` (matches
/// ``networkx.is_connected(nx.Graph())`` which raises
/// ``NetworkXPointlessConcept``).
#[pyfunction]
pub fn is_connected(
    py: Python<'_>,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i64>,
    n_nodes: usize,
) -> PyResult<bool> {
    if n_nodes == 0 {
        return Err(PyValueError::new_err(
            "Connectivity is undefined for the null graph.",
        ));
    }

    let indptr_slice = indptr.as_slice()?;
    let indices_slice = indices.as_slice()?;

    let result = py.detach(|| -> Result<bool, GraphError> {
        let g = build_graph_from_csr(indptr_slice, indices_slice, n_nodes)?;
        Ok(rwx_cc(&g).len() == 1)
    })?;

    Ok(result)
}

/// Return all ``(u, u)`` self-loop edges present in the CSR adjacency.
///
/// Each self-loop is yielded once (per ``u``-row, not per neighbour-list
/// occurrence). 5-LOC filter — does not need a graph object since the
/// information is direct in the CSR.
#[pyfunction]
pub fn selfloop_edges(
    py: Python<'_>,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i64>,
    n_nodes: usize,
) -> PyResult<Vec<(i64, i64)>> {
    let indptr_slice = indptr.as_slice()?;
    let indices_slice = indices.as_slice()?;

    let result = py.detach(|| -> Result<Vec<(i64, i64)>, GraphError> {
        if indptr_slice.len() != n_nodes + 1 {
            return Err(GraphError::InvalidNodeId(format!(
                "indptr length {} != n_nodes + 1 ({})",
                indptr_slice.len(),
                n_nodes + 1
            )));
        }
        let mut loops: Vec<(i64, i64)> = Vec::new();
        for u in 0..n_nodes {
            let start = indptr_slice[u] as usize;
            let end = indptr_slice[u + 1] as usize;
            let u_i64 = u as i64;
            // Yield once per (u, u) pair even if the CSR row lists it more than
            // once. networkx's selfloop_edges has the same contract: one entry
            // per actual self-loop in the underlying multigraph data.
            if (start..end).any(|k| indices_slice.get(k).copied() == Some(u_i64)) {
                loops.push((u_i64, u_i64));
            }
        }
        Ok(loops)
    })?;

    Ok(result)
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
    fn three_node_path_is_single_component() {
        // 0 -- 1 -- 2
        let indptr = vec![0_i64, 1, 3, 4];
        let indices = vec![1_i64, 0, 2, 1];
        let g = build_graph_from_csr(&indptr, &indices, 3).unwrap();
        let components = rwx_cc(&g);
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 3);
    }

    #[test]
    fn disjoint_pair_is_two_components() {
        // Edges: (0,1) and (2,3)
        let indptr = vec![0_i64, 1, 2, 3, 4];
        let indices = vec![1_i64, 0, 3, 2];
        let g = build_graph_from_csr(&indptr, &indices, 4).unwrap();
        assert_eq!(rwx_cc(&g).len(), 2);
    }
}
