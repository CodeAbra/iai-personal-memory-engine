from __future__ import annotations

from pathlib import Path

import networkx as nx

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
            f"sigma.py must NOT call {needle} -- use fast_sigma "
            f"(RESEARCH §Pitfall 1)"
        )
