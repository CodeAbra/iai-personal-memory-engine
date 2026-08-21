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
//! `py.detach(|| ...)`. `distance_matrix` is O(V·(V+E)) and at the
//! N≥2000 scale the daemon's status handler must stay responsive.
//!
//! Unreachable-cell handling: `distance_matrix` accepts a `null_value: f64`
//! sentinel for "no path"; we pass `f64::INFINITY` and filter via
//! `dist.is_finite()` before summing. The largest-CC guard makes that
//! filter redundant in practice (every off-diagonal cell is reachable on
//! one connected component), but the explicit `is_finite()` check is
//! deliberately retained so future refactors can drop the guard without
//! corrupting the sum with a sentinel leak.

use std::collections::{HashMap, HashSet, VecDeque};

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
// `rustworkx-core` re-exports its own `petgraph` to keep its trait bounds
// consistent across the public surface. We use that re-export instead of
// the top-level `petgraph` crate so the `UnGraphMap` value we hand to
// `distance_matrix` and `connected_components` carries the matching
// `GraphProp` / `Visitable` / `IntoNeighbors*` impls — otherwise a
// "multiple different versions of crate `petgraph`" trait-resolution
// error fires at the call site.
use rustworkx_core::connectivity::connected_components;
use rustworkx_core::petgraph::graphmap::UnGraphMap;
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
fn build_ungraph_from_csr(indptr: &[i64], indices: &[i64], n_nodes: usize) -> UnGraphMap<i64, ()> {
    let mut g: UnGraphMap<i64, ()> = UnGraphMap::with_capacity(n_nodes, indices.len() / 2);
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
fn largest_connected_component_subgraph(graph: &UnGraphMap<i64, ()>) -> UnGraphMap<i64, ()> {
    let components = connected_components(graph);
    // Deterministic tie-break: among equal-size components, pick the one with
    // the smallest minimum node id. connected_components returns HashSets in an
    // unstable order, so a bare max_by_key(len) would pick a run-dependent
    // component on a tie and make APSL — and therefore sigma — non-reproducible.
    let largest = match components.iter().max_by_key(|c| {
        (
            c.len(),
            core::cmp::Reverse(c.iter().copied().min().unwrap_or(i64::MAX)),
        )
    }) {
        Some(c) => c,
        None => return UnGraphMap::new(),
    };
    let keep: HashSet<i64> = largest.iter().copied().collect();
    let mut sub: UnGraphMap<i64, ()> = UnGraphMap::with_capacity(keep.len(), keep.len());
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
fn average_shortest_path_length_on_connected_subgraph(subgraph: &UnGraphMap<i64, ()>) -> f64 {
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
/// that component. The Rust kernel runs under `py.detach(|| ...)`
/// so the daemon's other Python callers stay responsive on N≥2000
/// inputs.
///
/// **Stub-gen note:** `#[gen_stub_pyfunction]` is intentionally NOT
/// applied here. `pyo3-stub-gen` does not implement `PyStubType` for
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
    // Full CSR validation (not just indptr length): build_ungraph_from_csr
    // indexes indptr[u+1] and treats every indices entry as a node id, so an
    // out-of-range neighbor would silently add a phantom node to the graph.
    crate::validate_csr(indptr_slice, indices_slice, n_nodes)?;

    // Snapshot the slices into owned buffers so the compute kernel can
    // run outside the GIL. The numpy borrows are GIL-bound; copying ~10⁴
    // i64 entries is cheap relative to the O(V²) BFS that follows.
    let indptr_owned: Vec<i64> = indptr_slice.to_vec();
    let indices_owned: Vec<i64> = indices_slice.to_vec();

    let result = py.detach(move || {
        let graph = build_ungraph_from_csr(&indptr_owned, &indices_owned, n_nodes);
        let largest_cc = largest_connected_component_subgraph(&graph);
        average_shortest_path_length_on_connected_subgraph(&largest_cc)
    });

    Ok(result)
}

/// Landmark-sampled APSL on an already-connected subgraph: BFS from an
/// even-stride deterministic subset of sources, averaging distances to every
/// other node. Memory is O(V) per BFS — never the O(V²) distance matrix —
/// which is what makes sigma viable as the corpus grows (the exact matrix is
/// ~3.2 GB at 20k nodes and quadratically worse beyond). Node ids derive from
/// UUIDs, so their sort order is topology-agnostic and an even stride over the
/// sorted ids approximates uniform source sampling. With `n_sources >= n` every
/// node is a source and the estimate EQUALS the exact APSL.
fn sampled_apsl_on_connected_subgraph(subgraph: &UnGraphMap<i64, ()>, n_sources: usize) -> f64 {
    let n = subgraph.node_count();
    if n <= 1 || n_sources == 0 {
        return 0.0;
    }
    let mut nodes: Vec<i64> = subgraph.nodes().collect();
    nodes.sort_unstable();
    let idx: HashMap<i64, usize> = nodes.iter().enumerate().map(|(i, &v)| (v, i)).collect();

    let k = n_sources.min(n);
    let mut total: f64 = 0.0;
    let mut pairs: u64 = 0;
    let mut dist: Vec<i32> = vec![-1; n];
    for s_i in 0..k {
        // Even stride over the sorted id list; multiplication before division
        // keeps the k sources distinct and spread across the full range.
        let source = nodes[s_i * n / k];
        dist.iter_mut().for_each(|d| *d = -1);
        let src_idx = idx[&source];
        dist[src_idx] = 0;
        let mut queue: VecDeque<i64> = VecDeque::new();
        queue.push_back(source);
        while let Some(u) = queue.pop_front() {
            let du = dist[idx[&u]];
            for v in subgraph.neighbors(u) {
                let vi = idx[&v];
                if dist[vi] < 0 {
                    dist[vi] = du + 1;
                    queue.push_back(v);
                }
            }
        }
        for (i, &d) in dist.iter().enumerate() {
            if i != src_idx && d > 0 {
                total += d as f64;
                pairs += 1;
            }
        }
    }
    if pairs == 0 {
        return 0.0;
    }
    total / pairs as f64
}

/// `iai_mcp_native.graph.average_shortest_path_length_sampled(indptr, indices,
/// n_nodes, n_sources) -> float`.
///
/// Bounded-memory APSL estimator over the largest connected component. Fully
/// deterministic (even-stride landmark selection, no RNG); `n_sources >= n`
/// degrades to the exact all-pairs value. See the stub-gen note on
/// `average_shortest_path_length` for why `#[gen_stub_pyfunction]` is absent.
#[pyfunction]
pub fn average_shortest_path_length_sampled(
    py: Python<'_>,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i64>,
    n_nodes: usize,
    n_sources: usize,
) -> PyResult<f64> {
    let indptr_slice = indptr.as_slice()?;
    let indices_slice = indices.as_slice()?;
    crate::validate_csr(indptr_slice, indices_slice, n_nodes)?;

    let indptr_owned: Vec<i64> = indptr_slice.to_vec();
    let indices_owned: Vec<i64> = indices_slice.to_vec();

    let result = py.detach(move || {
        let graph = build_ungraph_from_csr(&indptr_owned, &indices_owned, n_nodes);
        let largest_cc = largest_connected_component_subgraph(&graph);
        sampled_apsl_on_connected_subgraph(&largest_cc, n_sources)
    });

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    /// Two equal-size (3-node) components with different topology so the
    /// selection is observable in the output: a path {0,1,2} (APSL 4/3)
    /// and a triangle {3,4,5} (APSL 1). The deterministic tie-break must
    /// always pick the smaller-min-id component (the path), independent of
    /// the unstable order `connected_components` returns its HashSets in.
    /// Repeated so a regression to bare `max_by_key(len)` shows up as a
    /// flake rather than passing by luck.
    #[test]
    fn tie_break_picks_smallest_min_id_component() {
        let edges = [(0i64, 1i64), (1, 2), (3, 4), (4, 5), (3, 5)];
        for _ in 0..64 {
            let mut g: UnGraphMap<i64, ()> = UnGraphMap::new();
            for n in 0..6i64 {
                g.add_node(n);
            }
            for &(a, b) in &edges {
                g.add_edge(a, b, ());
            }
            let sub = largest_connected_component_subgraph(&g);
            assert!(sub.contains_node(0), "min-id component must win the tie");
            assert!(!sub.contains_node(3), "triangle must lose the tie");
            let apsl = average_shortest_path_length_on_connected_subgraph(&sub);
            assert!(
                approx(apsl, 4.0 / 3.0),
                "path-component APSL should be 4/3, got {apsl}"
            );
        }
    }

    #[test]
    fn triangle_apsl_is_unity() {
        let mut g: UnGraphMap<i64, ()> = UnGraphMap::new();
        for n in 0..3i64 {
            g.add_node(n);
        }
        g.add_edge(0, 1, ());
        g.add_edge(1, 2, ());
        g.add_edge(0, 2, ());
        let sub = largest_connected_component_subgraph(&g);
        let apsl = average_shortest_path_length_on_connected_subgraph(&sub);
        assert!(approx(apsl, 1.0));
    }

    /// With n_sources >= n every node becomes a BFS source, so the sampled
    /// estimator must EQUAL the exact all-pairs value on any graph.
    #[test]
    fn sampled_with_all_sources_equals_exact() {
        // 6-node path graph: heterogeneous distances exercise the estimator.
        let mut g: UnGraphMap<i64, ()> = UnGraphMap::new();
        for n in 0..6i64 {
            g.add_node(n);
        }
        for n in 0..5i64 {
            g.add_edge(n, n + 1, ());
        }
        let exact = average_shortest_path_length_on_connected_subgraph(&g);
        let sampled = sampled_apsl_on_connected_subgraph(&g, 6);
        assert!(approx(exact, sampled), "exact {exact} != sampled {sampled}");
        let oversampled = sampled_apsl_on_connected_subgraph(&g, 1000);
        assert!(approx(exact, oversampled));
    }

    #[test]
    fn sampled_is_deterministic_and_bounded() {
        let mut g: UnGraphMap<i64, ()> = UnGraphMap::new();
        for n in 0..40i64 {
            g.add_node(n);
        }
        for n in 0..39i64 {
            g.add_edge(n, n + 1, ());
        }
        // ring closure so eccentricities vary with the landmark choice
        g.add_edge(39, 0, ());
        let first = sampled_apsl_on_connected_subgraph(&g, 8);
        for _ in 0..16 {
            let again = sampled_apsl_on_connected_subgraph(&g, 8);
            assert!(approx(first, again), "sampled APSL not deterministic");
        }
        // 40-ring exact APSL is ~10.26; a stride sample must stay in range.
        assert!(first > 1.0 && first < 40.0);
    }

    #[test]
    fn sampled_zero_sources_and_singleton_return_zero() {
        let mut g: UnGraphMap<i64, ()> = UnGraphMap::new();
        g.add_node(1);
        assert!(approx(sampled_apsl_on_connected_subgraph(&g, 4), 0.0));
        let mut g2: UnGraphMap<i64, ()> = UnGraphMap::new();
        g2.add_node(1);
        g2.add_node(2);
        g2.add_edge(1, 2, ());
        assert!(approx(sampled_apsl_on_connected_subgraph(&g2, 0), 0.0));
    }

    #[test]
    fn empty_and_singleton_return_zero() {
        let empty: UnGraphMap<i64, ()> = UnGraphMap::new();
        assert!(approx(
            average_shortest_path_length_on_connected_subgraph(&empty),
            0.0
        ));
        let mut single: UnGraphMap<i64, ()> = UnGraphMap::new();
        single.add_node(7);
        assert!(approx(
            average_shortest_path_length_on_connected_subgraph(&single),
            0.0
        ));
    }
}
