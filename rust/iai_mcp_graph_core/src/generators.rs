//! G(n, m) random-graph generator.
//!
//! Local Pcg64-seeded sampler that draws an undirected Erdős-Rényi
//! graph with exactly ``m`` unique edges, no self-loops, and no
//! duplicates. Output is the edge list as a pair of node-index vectors
//! ``(u_list, v_list)`` each of length ``m``.
//!
//! ## Why not `rustworkx_core::generators::gnm_random_graph`?
//!
//! The upstream function in rustworkx-core 0.17.1 has a correctness
//! bug when called with ``UnGraph<(), ()>``: its inner ``find_edge``
//! helper checks only the ``(source, target)`` direction. When the
//! sampler picks ``(2, 1)`` after already adding ``(1, 2)``, the
//! duplicate-check misses it and the second edge is added — yielding
//! a multigraph instead of a simple graph. Verified empirically:
//! ``n=200, m=400, seed=42`` returns 398 unique canonical edges
//! instead of 400.
//!
//! Filing an upstream patch is out of scope for this plan; the local
//! sampler below uses the same ``Pcg64`` RNG plumbing rustworkx-core
//! uses internally, so determinism is preserved against our own
//! frozen baseline.
//!
//! ## Parity contract
//!
//! Per-seed determinism via ``rand_pcg::Pcg64::seed_from_u64``.
//! Graph-property invariants (``m`` edges, no self-loops, no duplicate
//! edges, node ids in ``[0, n)``) are the parity contract. This
//! generator **explicitly does NOT** chase ``networkx.gnm_random_graph``
//! bit-for-bit. Canonical edge sets per ``(n, m, seed)`` tuple are
//! frozen in ``tests/fixtures/sigma_baseline.json::gnm_baseline``;
//! any silent shift in our generator (e.g. a ``rand_pcg`` upgrade) is
//! caught by ``test_gnm_frozen_edge_set_matches_baseline``.
//!
//! ## Pre-validation
//!
//! For an undirected graph, the maximum number of unique edges is
//! ``n * (n - 1) / 2``. We raise ``ValueError`` when ``m`` exceeds
//! this maximum so callers see invalid requests loudly instead of a
//! silently capped or infinite-loop result.
//!
//! ## GIL release
//!
//! The compute kernel runs inside ``Python::allow_threads`` — Python
//! threads (e.g. the daemon's status handler) can make progress while
//! the sampler runs.

use std::collections::HashSet;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::distr::{Distribution, Uniform};
use rand::SeedableRng;
use rand_pcg::Pcg64;

/// Pure-Rust core, callable without a Python interpreter.
///
/// Returns ``Err(String)`` for invalid inputs so the PyO3 wrapper can
/// map the message to a ``ValueError``. Splitting the kernel out keeps
/// the ``#[cfg(test)]`` block testable via plain ``cargo test`` (no
/// ``auto-initialize`` feature, no ``prepare_freethreaded_python()``).
pub(crate) fn gnm_random_graph_core(
    n: usize,
    m: usize,
    seed: u64,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    if n == 0 {
        return Err("n must be >= 1 (G(n, m) is undefined for n=0)".to_string());
    }
    let max_edges = if n == 1 {
        0
    } else {
        n.checked_mul(n - 1)
            .ok_or_else(|| format!("n={n} overflows usize arithmetic"))?
            / 2
    };
    if m > max_edges {
        return Err(format!(
            "m={m} exceeds n*(n-1)/2={max_edges} for undirected G(n, m) with n={n}"
        ));
    }

    let mut u_list: Vec<i64> = Vec::with_capacity(m);
    let mut v_list: Vec<i64> = Vec::with_capacity(m);

    if m == 0 {
        return Ok((u_list, v_list));
    }

    // Complete-graph fast path. Otherwise the random-sampling loop
    // below would reject every new draw indefinitely when m equals the
    // maximum possible edge count.
    if m == max_edges {
        for u in 0..n {
            for v in (u + 1)..n {
                u_list.push(u as i64);
                v_list.push(v as i64);
            }
        }
        return Ok((u_list, v_list));
    }

    let mut rng = Pcg64::seed_from_u64(seed);
    // Uniform::new is fallible since rand 0.9; the bound is fixed and
    // known-valid (n >= 2 here because m > 0 requires n >= 2).
    let between =
        Uniform::new(0, n).map_err(|e| format!("rand::distr::Uniform setup failed: {e}"))?;

    let mut seen: HashSet<(usize, usize)> = HashSet::with_capacity(m);
    while u_list.len() < m {
        let u = between.sample(&mut rng);
        let v = between.sample(&mut rng);
        if u == v {
            continue;
        }
        let canon = if u < v { (u, v) } else { (v, u) };
        if seen.insert(canon) {
            // Preserve the SAMPLED (u, v) ordering — not the canonical
            // (min, max) form — so the output matches what a
            // bug-free rustworkx-core sampler would produce. Tests
            // canonicalize before deduplication.
            u_list.push(u as i64);
            v_list.push(v as i64);
        }
    }

    Ok((u_list, v_list))
}

/// Generate an undirected G(n, m) Erdős-Rényi random graph.
///
/// Returns the edge list as a pair of node-index vectors
/// ``(u_list, v_list)`` where each undirected edge ``(u, v)`` contributes
/// ``u`` to ``u_list`` and ``v`` to ``v_list``. Both lists have length
/// ``m``.
///
/// ## Errors
///
/// Raises ``ValueError`` when:
///   * ``n == 0``.
///   * ``m > n * (n - 1) / 2`` (undirected-max guard — see module doc).
#[pyfunction]
pub fn gnm_random_graph(
    py: Python<'_>,
    n: usize,
    m: usize,
    seed: u64,
) -> PyResult<(Vec<i64>, Vec<i64>)> {
    py.allow_threads(|| gnm_random_graph_core(n, m, seed))
        .map_err(PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_n_zero() {
        let err = gnm_random_graph_core(0, 0, 42).unwrap_err();
        assert!(err.contains("n must be >= 1"), "got: {err}");
    }

    #[test]
    fn rejects_m_over_undirected_max() {
        // n=5 max undirected edges = 5 * 4 / 2 = 10
        let err = gnm_random_graph_core(5, 20, 42).unwrap_err();
        assert!(err.contains("exceeds n*(n-1)/2"), "got: {err}");
    }

    #[test]
    fn zero_edges_returns_empty() {
        let (u, v) = gnm_random_graph_core(10, 0, 42).unwrap();
        assert!(u.is_empty());
        assert!(v.is_empty());
    }

    #[test]
    fn complete_graph_fast_path() {
        // n=5, max edges = 10. Complete-graph branch fires.
        let (u, v) = gnm_random_graph_core(5, 10, 42).unwrap();
        assert_eq!(u.len(), 10);
        assert_eq!(v.len(), 10);
        // Every edge is (u, v) with u < v.
        for (a, b) in u.iter().zip(v.iter()) {
            assert!(a < b, "complete-graph branch emitted ({a}, {b}) not u<v");
        }
    }

    #[test]
    fn edge_count_matches_m() {
        let (u, v) = gnm_random_graph_core(50, 60, 42).unwrap();
        assert_eq!(u.len(), 60);
        assert_eq!(v.len(), 60);
    }

    #[test]
    fn no_self_loops() {
        let (u, v) = gnm_random_graph_core(100, 200, 7).unwrap();
        for (a, b) in u.iter().zip(v.iter()) {
            assert_ne!(a, b, "self-loop ({a}, {b}) detected");
        }
    }

    #[test]
    fn no_duplicate_edges() {
        let (u, v) = gnm_random_graph_core(200, 400, 42).unwrap();
        let mut seen = HashSet::new();
        for (a, b) in u.iter().zip(v.iter()) {
            let canon = if a <= b { (*a, *b) } else { (*b, *a) };
            assert!(
                seen.insert(canon),
                "duplicate edge {canon:?} found in (u, v) output"
            );
        }
        assert_eq!(seen.len(), 400, "expected 400 unique edges");
    }

    #[test]
    fn deterministic_under_seed() {
        let a = gnm_random_graph_core(200, 400, 42).unwrap();
        let b = gnm_random_graph_core(200, 400, 42).unwrap();
        assert_eq!(
            a, b,
            "two calls with same seed must produce identical edge lists"
        );
    }

    #[test]
    fn node_range_valid() {
        let (u, v) = gnm_random_graph_core(200, 400, 42).unwrap();
        for (a, b) in u.iter().zip(v.iter()) {
            assert!((0..200).contains(&(*a as usize)), "u={a} outside [0, 200)");
            assert!((0..200).contains(&(*b as usize)), "v={b} outside [0, 200)");
        }
    }

    #[test]
    fn distinct_seeds_yield_distinct_outputs() {
        // Probabilistically the chance of two seeds producing identical
        // edge sets at (n=100, m=200) is negligible.
        let a = gnm_random_graph_core(100, 200, 42).unwrap();
        let b = gnm_random_graph_core(100, 200, 43).unwrap();
        assert_ne!(a, b);
    }
}
