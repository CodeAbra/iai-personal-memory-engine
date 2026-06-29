from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import networkx as nx
import pytest

from tests.conftest import _nx_graph_to_memory_graph


def test_sigma_module_exposes_constants_and_functions():
    from iai_mcp import sigma

    assert sigma.SIGMA_N_FLOOR == 200
    assert sigma.SIGMA_MID_LIFE_THRESHOLD == 500
    assert callable(sigma.fast_sigma)
    assert callable(sigma.compute_sigma)
    assert callable(sigma.classify_regime)
    assert callable(sigma.compute_topology_snapshot)
    assert callable(sigma.compute_and_emit)


def test_compute_sigma_returns_none_below_floor():
    from iai_mcp.sigma import compute_sigma

    g = nx.Graph()
    g.add_nodes_from(range(199))
    for i in range(10):
        g.add_edge(i, i + 1)
    mg = _nx_graph_to_memory_graph(g)
    assert compute_sigma(mg) is None


def test_fast_sigma_small_world_above_one_at_n_250():
    from iai_mcp.sigma import fast_sigma

    g = nx.connected_watts_strogatz_graph(250, k=6, p=0.1, seed=42)
    mg = _nx_graph_to_memory_graph(g)
    sigma_val, C, L, Cr, Lr = fast_sigma(mg, n_random=3, seed=42)
    assert sigma_val > 1.0, f"expected sigma > 1, got {sigma_val:.3f}"
    assert C > 0
    assert L > 0
    assert Cr > 0
    assert Lr > 0


def test_fast_sigma_random_graph_near_one_at_n_250():
    from iai_mcp.sigma import fast_sigma

    g = nx.gnm_random_graph(250, 750, seed=42)
    mg = _nx_graph_to_memory_graph(g)
    sigma_val, _C, _L, _Cr, _Lr = fast_sigma(mg, n_random=3, seed=43)
    assert 0.5 < sigma_val < 1.5, f"expected sigma ~ 1, got {sigma_val:.3f}"


def test_fast_sigma_handles_disconnected_input():
    from iai_mcp.sigma import fast_sigma

    g = nx.connected_watts_strogatz_graph(220, k=6, p=0.1, seed=7)
    for k in range(220, 230):
        g.add_node(k)
    mg = _nx_graph_to_memory_graph(g)
    sigma_val, _C, _L, _Cr, _Lr = fast_sigma(mg, n_random=2, seed=42)
    assert sigma_val > 0


def test_classify_regime_insufficient_data():
    from iai_mcp.sigma import classify_regime

    assert classify_regime(50, None) == "insufficient_data"
    assert classify_regime(0, None) == "insufficient_data"


def test_classify_regime_developmental_n_lt_500_sigma_lt_1():
    from iai_mcp.sigma import classify_regime

    assert classify_regime(300, 0.5) == "developmental"
    assert classify_regime(499, 0.99) == "developmental"


def test_classify_regime_mid_life_drift_n_ge_500_sigma_lt_1():
    from iai_mcp.sigma import classify_regime

    assert classify_regime(500, 0.5) == "mid_life_drift"
    assert classify_regime(1000, 0.99) == "mid_life_drift"


def test_classify_regime_healthy_sigma_ge_1():
    from iai_mcp.sigma import classify_regime

    assert classify_regime(300, 1.5) == "healthy"
    assert classify_regime(800, 5.0) == "healthy"
    assert classify_regime(200, 1.0) == "healthy"


def test_sigma_module_does_not_call_nx_sigma():
    src = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "sigma.py"
    text = src.read_text(encoding="utf-8")
    forbidden_calls = ["nx.sigma(", "networkx.sigma("]
    for needle in forbidden_calls:
        assert needle not in text, (
            f"sigma.py must NOT call {needle} -- use fast_sigma"
        )


def test_sigma_ceiling_constant_above_floor():
    # Defensive cap added in the 2026-06-21 remediation: bound the unbounded
    # small-worldness compute on a pathologically large graph.
    from iai_mcp import sigma

    assert isinstance(sigma.SIGMA_N_CEIL, int)
    assert sigma.SIGMA_N_CEIL > sigma.SIGMA_N_FLOOR


def test_compute_sigma_returns_none_above_ceiling(monkeypatch):
    # Above SIGMA_N_CEIL, compute_sigma skips the compute and returns None
    # (regime -> insufficient_data). WITHOUT the cap this 300-node small-world
    # graph yields a finite float, so the test fails -> the guard is load-bearing.
    from iai_mcp import sigma
    from iai_mcp.sigma import classify_regime, compute_sigma

    monkeypatch.setattr(sigma, "SIGMA_N_CEIL", 100)
    g = nx.connected_watts_strogatz_graph(300, 6, 0.1, seed=42)
    mg = _nx_graph_to_memory_graph(g)
    assert mg.node_count() == 300 > sigma.SIGMA_N_CEIL
    assert compute_sigma(mg) is None
    assert classify_regime(mg.node_count(), compute_sigma(mg)) == "insufficient_data"


def test_compute_sigma_ceiling_env_override(monkeypatch):
    from iai_mcp.sigma import compute_sigma

    monkeypatch.setenv("IAI_MCP_SIGMA_N_CEIL", "100")
    g = nx.connected_watts_strogatz_graph(300, 6, 0.1, seed=42)
    mg = _nx_graph_to_memory_graph(g)
    assert compute_sigma(mg) is None


def test_topology_snapshot_reuses_payload_centrality_for_rich_club(monkeypatch):
    from iai_mcp import sigma
    from iai_mcp.graph import MemoryGraph

    graph = MemoryGraph()
    nodes = [uuid4() for _ in range(12)]
    for idx, node_id in enumerate(nodes):
        graph.add_node(node_id, community_id=None, embedding=[0.0] * 8)
        graph.set_node_centrality(node_id, float(idx + 1))
    for left, right in zip(nodes, nodes[1:]):
        graph.add_edge(left, right, weight=1.0, edge_type="hebbian")

    def fail_exact_centrality(self):
        raise AssertionError("topology snapshot must not call exact centrality")

    monkeypatch.setattr(MemoryGraph, "centrality", fail_exact_centrality)

    snap = sigma.compute_topology_snapshot(
        graph,
        assignment=SimpleNamespace(community_centroids={0: [nodes[0]]}),
    )

    assert snap["rich_club_ratio"] > 0.0
