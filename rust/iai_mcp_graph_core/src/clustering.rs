//! LOCAL Watts-Strogatz average clustering coefficient.
//!
//! Per-node formula:
//!
//! ```text
//!     c(v) = 2 * T(v) / (d(v) * (d(v) - 1))    for d(v) >= 2
//!     c(v) = 0                                  for d(v) <  2
//! ```
//!
//! Average over all nodes (arithmetic mean):
//!
//! ```text
//!     C_avg = (1 / n_nodes) * sum_v c(v)
//! ```
//!
//! Triangle enumeration draws on the per-node form of the Schank-Wagner
//! 2005 algorithm (Schank & Wagner, "Finding, Counting, and Listing
//! All Triangles in Large Graphs", WEA 2005). For each node `u`, iterate
//! ordered pairs `(v, w)` of its neighbors with `v < w` and check
//! whether the edge `(v, w)` exists via binary search in
//! `neighbors[v]`. Each triangle incident to `u` is counted exactly
//! once because the pair `(v, w)` is generated only with `v < w`.
//! Complexity per node: O(d(u)^2 * log d_max). Total: O(N * d^2 * log d).
//!
//! Does NOT delegate to the global transitivity coefficient from
//! `rustworkx-core` (`3 * triangles / connected_triples`). That is a
//! distinct quantity from the Watts-Strogatz LOCAL clustering coefficient.
//! Substituting one for the other would silently shift sigma values on
//! every non-regular graph (= every real graph). A CI grep guard
//! enforces this invariant in `test_global_transitivity_not_used_in_source`
//! by asserting that the dotted import path from rustworkx-core to the
//! transitivity submodule never appears in this crate's source tree.
//!
//! The PyO3 entry point copies the input CSR slices to owned `Vec`s
//! BEFORE entering `py.allow_threads(|| ...)` so the compute kernel
//! holds no Python-bound borrows -- releasing the GIL during the kernel
//! lets the daemon's status handler and other async tasks remain
//! responsive on multi-second computations over N=10^4 graphs.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::error::GraphError;

/// Slice into `indices` covering the neighbors of node `u`.
///
/// CSR layout: `neighbors(u) = indices[indptr[u]..indptr[u + 1]]`.
/// Caller guarantees `0 <= u < n_nodes` and `indptr.len() == n_nodes + 1`.
#[inline]
fn neighbors_of<'a>(indptr: &[i64], indices: &'a [i64], u: usize) -> &'a [i64] {
    let start = indptr[u] as usize;
    let end = indptr[u + 1] as usize;
    &indices[start..end]
}

/// Binary-search check: is `target` present in the sorted slice `slice`?
///
/// Preconditions: `slice` is sorted ascending. We rely on this in
/// `count_triangles_for_node` -- the test harness builds CSR with
/// `sorted(adj[u])` so the precondition holds at the boundary.
#[inline]
fn sorted_contains(slice: &[i64], target: i64) -> bool {
    slice.binary_search(&target).is_ok()
}

/// Count triangles incident to node `u` via per-node neighbor-pair
/// enumeration with strict `v < w` ordering to avoid double-counting.
///
/// For every ordered pair `(v, w)` with `v, w in neighbors(u)` and
/// `v < w`, increment the count if the edge `(v, w)` exists (binary
/// search in `neighbors[v]`). Each triangle `{u, v, w}` is reported
/// exactly once because the pair `(v, w)` is generated only in the
/// `v < w` direction.
///
/// Self-loops in `neighbors(u)` (rare but possible if the caller did
/// not strip them) would yield `v == u`, which is harmless because
/// `binary_search(&w)` is still well-defined. The test harness's
/// `_edges_to_csr` strips self-loops on the way in.
fn count_triangles_for_node(indptr: &[i64], indices: &[i64], u: usize) -> u64 {
    let nbrs_u = neighbors_of(indptr, indices, u);
    let d_u = nbrs_u.len();
    if d_u < 2 {
        return 0;
    }
    let mut count: u64 = 0;
    for i in 0..d_u {
        let v = nbrs_u[i];
        // Skip self-loops on u even if the CSR includes them: a (u, u)
        // entry would still produce a (v, w) pair with v == u, which is
        // structurally not a triangle and must not be counted. Guard
        // by skipping v == u outright.
        if v as usize == u {
            continue;
        }
        let nbrs_v = neighbors_of(indptr, indices, v as usize);
        for &w in nbrs_u.iter().skip(i + 1) {
            if w as usize == u {
                continue;
            }
            // (v < w) is implicit because nbrs_u is sorted ascending
            // and `w` comes from a strictly later index than `v`.
            if sorted_contains(nbrs_v, w) {
                count += 1;
            }
        }
    }
    count
}

/// Per-node LOCAL clustering coefficient.
///
/// `c(u) = 2 * T(u) / (d(u) * (d(u) - 1))` for `d(u) >= 2`, else `0.0`.
/// The factor of 2 in the numerator compensates for the fact that
/// `d * (d - 1)` counts ordered neighbor pairs while `T(u)` was
/// produced by enumerating unordered pairs `v < w`.
#[inline]
fn local_clustering_coefficient(t_u: u64, d_u: usize) -> f64 {
    if d_u < 2 {
        0.0
    } else {
        let denom = (d_u as u64) * ((d_u - 1) as u64);
        (2.0 * t_u as f64) / (denom as f64)
    }
}

/// Pure-Rust kernel: arithmetic mean of `c(u)` over `u in 0..n_nodes`.
///
/// Pre-condition: each `indices[indptr[u]..indptr[u + 1]]` slice is
/// sorted ascending (required for the binary-search edge check).
/// Caller (`_edges_to_csr` in the test harness, and `MemoryGraph.to_csr_arrays`
/// at the production boundary) is responsible for this.
fn compute_average_clustering(indptr: &[i64], indices: &[i64], n_nodes: usize) -> f64 {
    if n_nodes == 0 {
        // Match networkx semantics: average_clustering on an empty
        // graph returns 0.0 (not NaN -- networkx guards `if not G`).
        return 0.0;
    }
    let mut total: f64 = 0.0;
    for u in 0..n_nodes {
        let d_u = (indptr[u + 1] - indptr[u]) as usize;
        let t_u = count_triangles_for_node(indptr, indices, u);
        total += local_clustering_coefficient(t_u, d_u);
    }
    total / (n_nodes as f64)
}

/// LOCAL Watts-Strogatz average clustering coefficient -- PyO3 entry.
///
/// ```python
/// from iai_mcp_native import graph
/// c = graph.average_clustering(indptr, indices, n_nodes)
/// # c: float, equal to networkx.average_clustering(G) within 1e-9
/// ```
///
/// Returns `0.0` on empty graphs (matches networkx semantics).
/// Raises `ValueError` on `indptr.len() != n_nodes + 1`.
///
/// GIL release: the Python-bound `PyReadonlyArray1` borrows are
/// consumed into owned `Vec<i64>` BEFORE `py.allow_threads(...)` so
/// the compute kernel holds no Python references. This lets the
/// daemon's status handler stay responsive while the kernel runs on
/// large graphs.
// NOTE on stub generation: `pyo3_stub_gen` does not yet derive
// `PyStubType` for `PyReadonlyArray1`, so this function intentionally
// omits the `#[gen_stub_pyfunction]` attribute. The runtime binding
// still works -- only the `.pyi` lacks a typed signature for this
// callable. Callers documented in the module docstring.
#[pyfunction]
pub fn average_clustering(
    py: Python<'_>,
    indptr: PyReadonlyArray1<'_, i64>,
    indices: PyReadonlyArray1<'_, i64>,
    n_nodes: usize,
) -> PyResult<f64> {
    // Validate while we still hold the GIL -- error construction needs it.
    let indptr_slice = indptr.as_slice()?;
    let indices_slice = indices.as_slice()?;
    if indptr_slice.len() != n_nodes + 1 {
        return Err(GraphError::InvalidNodeId(format!(
            "indptr length {} != n_nodes + 1 = {}",
            indptr_slice.len(),
            n_nodes + 1
        ))
        .into());
    }

    // Copy to owned buffers BEFORE allow_threads -- the closure must
    // not hold Python-bound borrows because the GIL is released for
    // its duration.
    let indptr_owned: Vec<i64> = indptr_slice.to_vec();
    let indices_owned: Vec<i64> = indices_slice.to_vec();

    let result = py.allow_threads(move || {
        compute_average_clustering(&indptr_owned, &indices_owned, n_nodes)
    });

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a sorted-ascending CSR over an undirected simple graph
    /// (test helper -- mirrors `_edges_to_csr` in the Python parity
    /// test, just in Rust for unit-level coverage of the kernel).
    fn build_csr(n: usize, edges: &[(usize, usize)]) -> (Vec<i64>, Vec<i64>) {
        let mut adj: Vec<Vec<i64>> = vec![Vec::new(); n];
        for &(u, v) in edges {
            if u == v {
                continue;
            }
            if !adj[u].contains(&(v as i64)) {
                adj[u].push(v as i64);
            }
            if !adj[v].contains(&(u as i64)) {
                adj[v].push(u as i64);
            }
        }
        for a in adj.iter_mut() {
            a.sort();
        }
        let mut indptr: Vec<i64> = vec![0; n + 1];
        let mut indices: Vec<i64> = Vec::new();
        for u in 0..n {
            indices.extend_from_slice(&adj[u]);
            indptr[u + 1] = indptr[u] + adj[u].len() as i64;
        }
        (indptr, indices)
    }

    #[test]
    fn k5_complete_graph_clusters_to_unity() {
        let n = 5;
        let mut edges = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                edges.push((u, v));
            }
        }
        let (indptr, indices) = build_csr(n, &edges);
        let c = compute_average_clustering(&indptr, &indices, n);
        assert_eq!(c, 1.0);
    }

    #[test]
    fn cycle_5_has_zero_clustering() {
        // 5-cycle (triangle-free) -- every node has d=2 and T=0,
        // c=0/2=0 -> average=0.
        let n = 5;
        let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)];
        let (indptr, indices) = build_csr(n, &edges);
        let c = compute_average_clustering(&indptr, &indices, n);
        assert_eq!(c, 0.0);
    }

    #[test]
    fn empty_graph_returns_zero() {
        let indptr = vec![0i64];
        let indices: Vec<i64> = vec![];
        let c = compute_average_clustering(&indptr, &indices, 0);
        assert_eq!(c, 0.0);
    }

    #[test]
    fn disjoint_k3_pair_averages_to_unity() {
        let n = 6;
        let edges = [
            (0, 1), (0, 2), (1, 2),
            (3, 4), (3, 5), (4, 5),
        ];
        let (indptr, indices) = build_csr(n, &edges);
        let c = compute_average_clustering(&indptr, &indices, n);
        assert_eq!(c, 1.0);
    }

    #[test]
    fn star_with_isolate_clusters_to_zero() {
        // 4-leaf star centered on 0, plus isolated node 5.
        // Center has d=4 but no neighbor-of-neighbor edges -> c=0.
        // Leaves have d=1 -> c=0. Isolate has d=0 -> c=0.
        let n = 6;
        let edges = [(0, 1), (0, 2), (0, 3), (0, 4)];
        let (indptr, indices) = build_csr(n, &edges);
        let c = compute_average_clustering(&indptr, &indices, n);
        assert_eq!(c, 0.0);
    }
}
