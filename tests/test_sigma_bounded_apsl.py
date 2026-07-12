from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import numpy as np


def test_large_apsl_uses_deterministic_bounded_batches(monkeypatch) -> None:
    import iai_mcp.sigma as sigma
    import scipy.sparse.csgraph

    n = sigma.SIGMA_APSL_EXACT_MAX_N + 1
    indptr = np.zeros(n + 1, dtype=np.int64)
    indices = np.zeros(0, dtype=np.int64)
    calls: list[list[int]] = []

    def fake_dijkstra(_graph, *, indices, **_kwargs):
        sources = np.atleast_1d(indices).astype(int)
        calls.append(sources.tolist())
        distances = np.ones((len(sources), n), dtype=np.float64)
        distances[np.arange(len(sources)), sources] = 0.0
        return distances

    monkeypatch.setattr(scipy.sparse.csgraph, "dijkstra", fake_dijkstra)

    first = sigma._average_shortest_path_length(
        indptr, indices, n, seed=42
    )
    first_sources = [source for batch in calls for source in batch]
    calls.clear()
    second = sigma._average_shortest_path_length(
        indptr, indices, n, seed=42
    )
    second_sources = [source for batch in calls for source in batch]

    assert first == second == 1.0
    assert first_sources == second_sources
    assert len(first_sources) == sigma.SIGMA_APSL_SAMPLE_SIZE
    assert len(set(first_sources)) == sigma.SIGMA_APSL_SAMPLE_SIZE
    assert all(len(batch) <= sigma.SIGMA_APSL_BATCH_SIZE for batch in calls)


def test_exact_boundary_keeps_native_parity_path(monkeypatch) -> None:
    import iai_mcp.sigma as sigma

    n = sigma.SIGMA_APSL_EXACT_MAX_N
    indptr = np.zeros(n + 1, dtype=np.int64)
    indices = np.zeros(0, dtype=np.int64)
    calls: list[int] = []

    def fake_exact(_indptr, _indices, node_count):
        calls.append(node_count)
        return 3.25

    monkeypatch.setattr(
        sigma.lilli_graph, "average_shortest_path_length", fake_exact
    )
    assert sigma._average_shortest_path_length(
        indptr, indices, n, seed=42
    ) == 3.25
    assert calls == [n]


def test_production_scale_path_stays_within_two_percent() -> None:
    import iai_mcp.sigma as sigma

    n = 16_109
    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0
    flat: list[int] = []
    for node in range(n):
        if node:
            flat.append(node - 1)
        if node + 1 < n:
            flat.append(node + 1)
        indptr[node + 1] = len(flat)
    indices = np.asarray(flat, dtype=np.int64)

    sampled = sigma._average_shortest_path_length(
        indptr, indices, n, seed=42
    )
    exact = (n + 1) / 3.0

    assert abs(sampled - exact) / exact < 0.02


def test_topology_snapshot_reuses_single_sigma_pass(monkeypatch) -> None:
    import iai_mcp.sigma as sigma
    from iai_mcp.graph import MemoryGraph

    graph = MemoryGraph()
    for _ in range(sigma.SIGMA_N_FLOOR + 1):
        graph.add_node(uuid4(), community_id=None, embedding=[])

    monkeypatch.setattr(
        graph,
        "centrality",
        lambda: (_ for _ in ()).throw(AssertionError("centrality must stay cold")),
    )

    calls: list[MemoryGraph] = []

    def fake_fast(candidate, **_kwargs):
        calls.append(candidate)
        return (1.5, 0.2, 3.0, 0.1, 2.25)

    monkeypatch.setattr(sigma, "fast_sigma", fake_fast)
    monkeypatch.setattr(
        "iai_mcp.community.detect_communities",
        lambda *_args, **_kwargs: SimpleNamespace(community_centroids={}),
    )

    snapshot = sigma.compute_topology_snapshot(graph)

    assert calls == [graph]
    assert snapshot["sigma"] == 1.5
    assert snapshot["C"] == 0.2
    assert snapshot["L"] == 3.0
    assert snapshot["rich_club_ratio"] == 21 / 201
