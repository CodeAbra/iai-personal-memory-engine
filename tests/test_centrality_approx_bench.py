"""Fidelity bench that picks the warm-path approximate centrality method.

Exact Brandes betweenness is intractable on the warm envelope at scale, so the
warm graph child computes a bounded approximation instead. This bench is the
decision-maker: on synthetic scale-free / community graphs small enough that the
EXACT baseline is computable in-test, it measures each candidate against exact
betweenness and asserts the winner clears the fidelity + stability gate.

Gates (per the locked design):
  - Fidelity: Seed Jaccard@K >= 0.90 median, where K is the real seed count
    (top 10% by centrality, mirroring ``rich_club_nodes``). Kendall tau on the
    top-2K union is reported.
  - Stability: three runs with the same deterministic config produce an
    IDENTICAL top-K seed set (Jaccard@K == 1.0) -- proves no per-cycle flicker.
  - Ops: each candidate's wall-clock is a small fraction of exact at the test
    size; the ratio is reported with an extrapolation note for 30k / 100k.

The winning method (the one that clears fidelity + stability at the lowest cost)
must match ``centrality_approx.runtime_method()`` so the bench and the runtime
default cannot diverge.
"""
from __future__ import annotations

import math
import time
from uuid import UUID

import numpy as np
import pytest

pytest.importorskip("iai_mcp_native")

from iai_mcp import centrality_approx
from iai_mcp.centrality_approx import (
    METHOD_HARMONIC,
    METHOD_SAMPLED_BETWEENNESS,
    approximate_centrality,
)
from iai_mcp.graph import MemoryGraph


# Test sizes: large enough for genuine betweenness variation, small enough that
# the exact baseline runs in-test in a few seconds.
SIZES = [1000, 2500]
SEED_PERCENT = 0.10
JACCARD_MEDIAN_GATE = 0.90

# Sampled-betweenness pivot count for the bench. Deliberately a fraction of the
# smallest test size so the bench exercises a genuine approximation, not a
# disguised exact pass.
BENCH_K = 256


def _scale_free_community_graph(n: int, seed: int) -> MemoryGraph:
    """A community-structured, heavy-tailed-degree graph with real bridges.

    Built so betweenness has structure worth approximating: ``c`` dense
    communities (each an internal random graph) wired together by a sparse set
    of inter-community bridge edges. The bridges are the high-betweenness nodes
    the seed set must recover. Fully deterministic from ``seed``.
    """
    rng = np.random.default_rng(seed)
    # Derived from ``rng`` (not ``uuid4()``) so node identity is as reproducible
    # as the edge structure: with an unseeded uuid4, near-tied scores at the
    # top-K cutoff could swap which node wins the str(id) tie-break between
    # runs, making the fidelity gate flaky independent of the algorithm.
    ids = [UUID(bytes=rng.bytes(16)) for _ in range(n)]
    g = MemoryGraph()
    for uid in ids:
        g.add_node(uid, community_id=None, embedding=[0.0] * 8)

    n_comms = max(2, int(math.sqrt(n)) // 2)
    comm_size = n // n_comms
    # Intra-community edges: a backbone path + random chords for a heavy tail.
    for c in range(n_comms):
        lo = c * comm_size
        hi = n if c == n_comms - 1 else (c + 1) * comm_size
        members = list(range(lo, hi))
        for i in range(len(members) - 1):
            g.add_edge(ids[members[i]], ids[members[i + 1]], weight=1.0)
        n_chords = max(1, (hi - lo) // 3)
        for _ in range(n_chords):
            a = int(rng.integers(lo, hi))
            b = int(rng.integers(lo, hi))
            if a != b:
                g.add_edge(ids[a], ids[b], weight=1.0)

    # Inter-community bridges: connect a few nodes across communities so the
    # bridge nodes carry high betweenness.
    n_bridges = max(n_comms, n // 50)
    for _ in range(n_bridges):
        a = int(rng.integers(0, n))
        b = int(rng.integers(0, n))
        if a != b:
            g.add_edge(ids[a], ids[b], weight=1.0)
    return g


def _top_k_set(score_map: dict, k: int) -> set:
    """The ``k`` highest-scoring node ids, ties broken by node-id sort order.

    The tie-break matches ``rich_club_nodes`` (which sorts ``centrality.items()``
    by value descending) closely enough for the overlap metric: both rank by
    score, and the seed set is what feeds recall.
    """
    ranked = sorted(score_map.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return {nid for nid, _ in ranked[:k]}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _kendall_tau_on_union(exact: dict, approx: dict, union_nodes: list) -> float:
    """Kendall tau-b between exact and approx rankings over ``union_nodes``."""
    n = len(union_nodes)
    if n < 2:
        return 1.0
    ex = np.array([exact[u] for u in union_nodes], dtype=np.float64)
    ap = np.array([approx[u] for u in union_nodes], dtype=np.float64)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            de = ex[i] - ex[j]
            da = ap[i] - ap[j]
            prod = de * da
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
            # ties contribute to neither (tau-b denominator handles them)
    denom = concordant + discordant
    if denom == 0:
        return 1.0
    return (concordant - discordant) / denom


def _exact_centrality(graph: MemoryGraph) -> dict:
    """Exact betweenness as the bench baseline, percentile-normalized so it is
    directly comparable to the candidates' normalized output for tau."""
    return graph.centrality()


def _evaluate(method: str, k: int) -> dict:
    """Run ``method`` across all SIZES and aggregate fidelity / stability / ops."""
    jaccards: list[float] = []
    taus: list[float] = []
    exact_times: list[float] = []
    approx_times: list[float] = []
    stability_jaccards: list[float] = []

    for n in SIZES:
        graph = _scale_free_community_graph(n, seed=1234 + n)

        t0 = time.perf_counter()
        exact = _exact_centrality(graph)
        exact_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        approx = approximate_centrality(
            graph, method=method, k=k, normalize=False
        )
        approx_times.append(time.perf_counter() - t1)

        seed_k = max(1, int(round(len(exact) * SEED_PERCENT)))
        exact_top = _top_k_set(exact, seed_k)
        approx_top = _top_k_set(approx, seed_k)
        jaccards.append(_jaccard(exact_top, approx_top))

        union = list(exact_top | approx_top)
        taus.append(_kendall_tau_on_union(exact, approx, union))

        # Stability: re-run the candidate three times; the deterministic config
        # must yield an identical top-K set every time (Jaccard == 1.0).
        runs = [
            _top_k_set(
                approximate_centrality(graph, method=method, k=k, normalize=False),
                seed_k,
            )
            for _ in range(3)
        ]
        stability_jaccards.append(_jaccard(runs[0], runs[1]))
        stability_jaccards.append(_jaccard(runs[1], runs[2]))

    return {
        "method": method,
        "jaccard_median": float(np.median(jaccards)),
        "jaccard_min": float(np.min(jaccards)),
        "jaccards": jaccards,
        "tau_median": float(np.median(taus)),
        "taus": taus,
        "stability_min": float(np.min(stability_jaccards)),
        "cost_ratio_median": float(
            np.median(
                [a / e if e > 0 else 0.0 for a, e in zip(approx_times, exact_times)]
            )
        ),
        "approx_times": approx_times,
        "exact_times": exact_times,
    }


@pytest.fixture(scope="module")
def bench_results() -> dict:
    """Evaluate both candidates once for the whole module."""
    sampled = _evaluate(METHOD_SAMPLED_BETWEENNESS, k=BENCH_K)
    harmonic = _evaluate(METHOD_HARMONIC, k=BENCH_K)
    return {
        METHOD_SAMPLED_BETWEENNESS: sampled,
        METHOD_HARMONIC: harmonic,
    }


def _report(label: str, r: dict) -> str:
    return (
        f"[{label}] "
        f"Jaccard@K median={r['jaccard_median']:.3f} "
        f"min={r['jaccard_min']:.3f} per-size={[round(x, 3) for x in r['jaccards']]} | "
        f"Kendall tau median={r['tau_median']:.3f} per-size={[round(x, 3) for x in r['taus']]} | "
        f"stability min={r['stability_min']:.3f} | "
        f"cost ratio (approx/exact) median={r['cost_ratio_median']:.3f} "
        f"approx_s={[round(x, 3) for x in r['approx_times']]} "
        f"exact_s={[round(x, 3) for x in r['exact_times']]}"
    )


def test_report_both_candidates(bench_results: dict, capsys) -> None:
    """Always print both candidates' numbers so the decision is auditable."""
    sampled = bench_results[METHOD_SAMPLED_BETWEENNESS]
    harmonic = bench_results[METHOD_HARMONIC]
    with capsys.disabled():
        print()
        print(_report("sampled_betweenness", sampled))
        print(_report("harmonic", harmonic))


def test_sampled_betweenness_stability_no_flicker(bench_results: dict) -> None:
    """The deterministic config produces an identical top-K seed set every run."""
    r = bench_results[METHOD_SAMPLED_BETWEENNESS]
    assert r["stability_min"] == 1.0, (
        f"sampled_betweenness flickers across runs: {_report('sampled', r)}"
    )


def test_harmonic_stability_no_flicker(bench_results: dict) -> None:
    r = bench_results[METHOD_HARMONIC]
    assert r["stability_min"] == 1.0, (
        f"harmonic flickers across runs: {_report('harmonic', r)}"
    )


def test_winner_passes_fidelity_gate(bench_results: dict) -> None:
    """At least one candidate clears the fidelity gate, and the runtime default
    points at a candidate that does (so the bench and runtime cannot diverge)."""
    passing = {
        name: r
        for name, r in bench_results.items()
        if r["jaccard_median"] >= JACCARD_MEDIAN_GATE and r["stability_min"] == 1.0
    }
    assert passing, (
        "no candidate cleared the fidelity gate:\n  "
        + "\n  ".join(_report(n, r) for n, r in bench_results.items())
    )
    runtime_default = centrality_approx.runtime_method()
    assert runtime_default in passing, (
        f"runtime default method {runtime_default!r} does not clear the gate; "
        f"passing candidates: {sorted(passing)}"
    )


def test_centrality_for_runtime_exact_below_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Below the cutoff the dispatcher returns the exact map verbatim."""
    from iai_mcp.centrality_approx import centrality_for_runtime

    monkeypatch.setenv("IAI_MCP_RGC_CENTRALITY_EXACT_BELOW", "5000")
    graph = _scale_free_community_graph(800, seed=77)
    exact = graph.centrality()
    runtime = centrality_for_runtime(graph)
    assert set(runtime) == set(exact)
    for nid, val in exact.items():
        assert runtime[nid] == val, "below-cutoff dispatch must be exact"


def test_centrality_for_runtime_approx_above_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Above the cutoff the dispatcher approximates, and the approximation tracks
    the exact top-K bridge seeds at the fidelity gate."""
    from iai_mcp.centrality_approx import centrality_for_runtime

    monkeypatch.setenv("IAI_MCP_RGC_CENTRALITY_EXACT_BELOW", "100")
    monkeypatch.setenv("IAI_MCP_RGC_CENTRALITY_K", "512")
    monkeypatch.delenv("IAI_MCP_RGC_CENTRALITY_METHOD", raising=False)
    graph = _scale_free_community_graph(2000, seed=88)
    exact = graph.centrality()
    runtime = centrality_for_runtime(graph)
    assert set(runtime) == set(exact)
    seed_k = max(1, int(round(len(exact) * SEED_PERCENT)))
    jac = _jaccard(_top_k_set(exact, seed_k), _top_k_set(runtime, seed_k))
    assert jac >= JACCARD_MEDIAN_GATE, (
        f"above-cutoff dispatch Jaccard@K={jac:.3f} below gate {JACCARD_MEDIAN_GATE}"
    )


def test_centrality_for_runtime_deterministic_no_flicker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The approximate dispatch yields a byte-identical map on repeated calls."""
    from iai_mcp.centrality_approx import centrality_for_runtime

    monkeypatch.setenv("IAI_MCP_RGC_CENTRALITY_EXACT_BELOW", "100")
    monkeypatch.setenv("IAI_MCP_RGC_CENTRALITY_K", "512")
    graph = _scale_free_community_graph(2000, seed=99)
    a = centrality_for_runtime(graph)
    b = centrality_for_runtime(graph)
    assert a == b, "approximate dispatch must be deterministic across cycles"


def test_runtime_default_is_lowest_cost_passing(bench_results: dict) -> None:
    """The runtime default is the cheapest candidate that clears the gate --
    sampled betweenness preferred when it passes (closest to the exact signal)."""
    passing = {
        name: r
        for name, r in bench_results.items()
        if r["jaccard_median"] >= JACCARD_MEDIAN_GATE and r["stability_min"] == 1.0
    }
    assert passing
    # Prefer sampled betweenness when it passes; else the cheapest passing method.
    if METHOD_SAMPLED_BETWEENNESS in passing:
        expected = METHOD_SAMPLED_BETWEENNESS
    else:
        expected = min(passing, key=lambda n: passing[n]["cost_ratio_median"])
    assert centrality_approx.runtime_method() == expected, (
        f"runtime default {centrality_approx.runtime_method()!r} is not the "
        f"chosen winner {expected!r}"
    )
