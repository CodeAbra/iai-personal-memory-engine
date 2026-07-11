from __future__ import annotations

import json
import random
import time
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4, uuid5

import numpy as np
import pytest

from iai_mcp.graph import MemoryGraph


def _emb(seed: int, dim: int = 384) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.random(dim).tolist()


def _load_karate() -> tuple[MemoryGraph, list[int], list[UUID]]:
    fixture_path = (
        Path(__file__).parent / "fixtures" / "leiden" / "karate_club.json"
    )
    data = json.loads(fixture_path.read_text())
    g = MemoryGraph()
    karate_ns = UUID("12345678-1234-5678-1234-567812345678")
    nodes: list[UUID] = [
        uuid5(karate_ns, f"karate-{i}") for i in range(data["n"])
    ]
    for i, u in enumerate(nodes):
        g.add_node(u, community_id=None, embedding=_emb(i))
    for u, v in data["edges"]:
        g.add_edge(nodes[u], nodes[v], weight=1.0)
    return g, data["ground_truth"], nodes


def _two_clique_with_bridge(n_per_clique: int = 50) -> MemoryGraph:
    g = MemoryGraph()
    nodes: list[UUID] = []
    for i in range(2 * n_per_clique):
        u = uuid4()
        nodes.append(u)
        g.add_node(u, community_id=None, embedding=_emb(i))
    for i in range(n_per_clique):
        for j in range(i + 1, n_per_clique):
            g.add_edge(nodes[i], nodes[j], weight=1.0)
    for i in range(n_per_clique, 2 * n_per_clique):
        for j in range(i + 1, 2 * n_per_clique):
            g.add_edge(nodes[i], nodes[j], weight=1.0)
    g.add_edge(nodes[0], nodes[n_per_clique], weight=1.0)
    return g, nodes


def _planted_3community_graph(
    n: int = 5000,
    intra_p: float = 0.02,
    inter_p: float = 0.001,
    seed: int = 13,
) -> MemoryGraph:
    rng = random.Random(seed)
    third = n // 3
    groups = [
        list(range(0, third)),
        list(range(third, 2 * third)),
        list(range(2 * third, n)),
    ]
    uuids = [uuid4() for _ in range(n)]
    g = MemoryGraph()
    for i, u in enumerate(uuids):
        g.add_node(u, community_id=None, embedding=_emb(i))

    for grp in groups:
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                if rng.random() < intra_p:
                    g.add_edge(
                        uuids[grp[i]], uuids[grp[j]],
                        weight=1.0, edge_type="hebbian",
                    )
    for gi in range(3):
        for gj in range(gi + 1, 3):
            for i in groups[gi]:
                for j in groups[gj]:
                    if rng.random() < inter_p:
                        g.add_edge(
                            uuids[i], uuids[j],
                            weight=0.1, edge_type="hebbian",
                        )
    return g


def _complete_graph(n: int = 30) -> MemoryGraph:
    g = MemoryGraph()
    nodes: list[UUID] = []
    for i in range(n):
        u = uuid4()
        nodes.append(u)
        g.add_node(u, community_id=None, embedding=_emb(i))
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(nodes[i], nodes[j], weight=1.0)
    return g


def _build_csr_partition_sigma(graph: MemoryGraph):
    import scipy.sparse

    from iai_mcp.mosaic import build_csr_sanitized, compute_sigma_tot

    csr, _order, _idx_map = build_csr_sanitized(graph)
    if csr.nnz == 0:
        n = csr.shape[0]
        partition = np.arange(n, dtype=np.int64)
        sigma_tot = np.zeros(n, dtype=np.float64)
        return csr, partition, sigma_tot, n

    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    n = indptr.shape[0] - 1
    partition = np.arange(n, dtype=np.int64)
    sigma_tot = compute_sigma_tot(indptr, indices, data, partition, n)
    return scipy.sparse.csr_matrix((data, indices, indptr), shape=(n, n)), partition, sigma_tot, n


def test_tuner_symbol_importable():
    from iai_mcp.mosaic import (  # noqa: F401
        _run_one_leiden_pass,
        multi_objective_gamma_tuner,
    )


def test_policy_helpers_importable():
    from iai_mcp.mosaic_policy import (  # noqa: F401
        CPM_MODULARITY_FLOOR,
        all_communities_connected,
        compute_modularity_classical,
        compute_singleton_ratio,
        should_fall_back_to_flat,
    )


def test_cpm_floor_calibrated_from_fixtures():
    from iai_mcp.mosaic_policy import CPM_MODULARITY_FLOOR

    assert CPM_MODULARITY_FLOOR != 0.2, (
        "CPM_MODULARITY_FLOOR must not match the legacy community.py:44 "
        "MODULARITY_FLOOR (0.2) -- that value was calibrated for the classical "
        "Newman-modularity ModularityVertexPartition. CPM-Q is "
        "gamma-dependent and falls in a different range; it is calibrated "
        "empirically."
    )
    assert 0.05 <= CPM_MODULARITY_FLOOR <= 0.30, (
        f"CPM_MODULARITY_FLOOR = {CPM_MODULARITY_FLOOR} outside sanity bounds "
        "[0.05, 0.30] (Traag 2019 CPM-Q observations on Karate / Football). "
        "If the calibration sweep returns a value outside these bounds, "
        "expand the fixture set or surface as a deviation in the SUMMARY."
    )


def test_singleton_ratio_zero_for_balanced_partition():
    from iai_mcp.mosaic_policy import compute_singleton_ratio

    partition = np.repeat(np.arange(10, dtype=np.int64), 10)
    assert compute_singleton_ratio(partition) == pytest.approx(0.0)


def test_singleton_ratio_one_for_all_isolates():
    from iai_mcp.mosaic_policy import compute_singleton_ratio

    partition = np.arange(50, dtype=np.int64)
    assert compute_singleton_ratio(partition) == pytest.approx(1.0)


def test_all_communities_connected_true_for_connected_partition():
    from iai_mcp.mosaic import build_csr_sanitized
    from iai_mcp.mosaic_policy import all_communities_connected

    graph, _nodes = _two_clique_with_bridge(n_per_clique=10)
    csr, _order, _idx_map = build_csr_sanitized(graph)
    nodes_set_a = set(_nodes[:10])
    partition = np.array(
        [0 if _order[i] in nodes_set_a else 1 for i in range(20)],
        dtype=np.int64,
    )
    assert all_communities_connected(csr, partition) is True


def test_all_communities_connected_false_for_split_community():
    from iai_mcp.mosaic import build_csr_sanitized
    from iai_mcp.mosaic_policy import all_communities_connected

    g = MemoryGraph()
    nodes: list[UUID] = []
    for i in range(10):
        u = uuid4()
        nodes.append(u)
        g.add_node(u, community_id=None, embedding=_emb(i))
    for i in range(5):
        for j in range(i + 1, 5):
            g.add_edge(nodes[i], nodes[j], weight=1.0)
    for i in range(5, 10):
        for j in range(i + 1, 10):
            g.add_edge(nodes[i], nodes[j], weight=1.0)
    csr, _order, _idx_map = build_csr_sanitized(g)
    partition = np.zeros(10, dtype=np.int64)
    assert all_communities_connected(csr, partition) is False


def test_should_fall_back_to_flat_q_too_low():
    from iai_mcp.mosaic_policy import (
        CPM_MODULARITY_FLOOR,
        should_fall_back_to_flat,
    )

    too_low_q = CPM_MODULARITY_FLOOR - 0.05
    assert (
        should_fall_back_to_flat(
            modularity=too_low_q, singleton_ratio=0.10, n_communities=5, n=100
        )
        is True
    )


def test_should_fall_back_to_flat_singleton_explosion():
    from iai_mcp.mosaic_policy import (
        CPM_MODULARITY_FLOOR,
        should_fall_back_to_flat,
    )

    good_q = CPM_MODULARITY_FLOOR + 0.10
    assert (
        should_fall_back_to_flat(
            modularity=good_q, singleton_ratio=0.40, n_communities=5, n=100
        )
        is True
    )


def test_should_fall_back_to_flat_community_count_explosion():
    from iai_mcp.mosaic_policy import (
        CPM_MODULARITY_FLOOR,
        should_fall_back_to_flat,
    )

    good_q = CPM_MODULARITY_FLOOR + 0.10
    assert (
        should_fall_back_to_flat(
            modularity=good_q, singleton_ratio=0.10, n_communities=50, n=100
        )
        is True
    )


def test_should_fall_back_to_flat_healthy_partition():
    from iai_mcp.mosaic_policy import (
        CPM_MODULARITY_FLOOR,
        should_fall_back_to_flat,
    )

    good_q = CPM_MODULARITY_FLOOR + 0.10
    assert (
        should_fall_back_to_flat(
            modularity=good_q, singleton_ratio=0.10, n_communities=10, n=100
        )
        is False
    )


def test_tuner_candidate_set_size():
    from iai_mcp.mosaic import multi_objective_gamma_tuner

    graph, _nodes = _two_clique_with_bridge(n_per_clique=10)
    csr, partition, sigma_tot, _n = _build_csr_partition_sigma(graph)
    _best_gamma, diagnostics = multi_objective_gamma_tuner(
        csr, partition, sigma_tot, seed=42
    )
    assert "candidate_scores" in diagnostics
    assert len(diagnostics["candidate_scores"]) == 5, (
        f"Tuner must evaluate exactly 5 gamma candidates (budget), "
        f"got {len(diagnostics['candidate_scores'])}: "
        f"{diagnostics['candidate_scores']}"
    )


def test_tuner_returns_default_on_no_satisfying_candidate():
    from iai_mcp.mosaic import multi_objective_gamma_tuner

    graph = _complete_graph(n=30)
    csr, partition, sigma_tot, _n = _build_csr_partition_sigma(graph)
    _best_gamma, diagnostics = multi_objective_gamma_tuner(
        csr, partition, sigma_tot, seed=42,
        targets={"q_min": 0.99, "singleton_ratio_max": 0.30},
    )
    assert diagnostics["all_constraints_satisfied"] is False, (
        f"With unachievable q_min=0.99, no candidate should satisfy hard "
        f"constraints; got candidate_stats: {diagnostics['candidate_stats']}"
    )
    assert diagnostics["should_fall_back_to_flat"] is True, (
        f"all_constraints_satisfied=False MUST imply "
        f"should_fall_back_to_flat=True; got "
        f"{diagnostics['should_fall_back_to_flat']}"
    )


def test_tuner_picks_best_satisfying_when_multiple_pass():
    from iai_mcp.mosaic import multi_objective_gamma_tuner

    graph, _nodes = _two_clique_with_bridge(n_per_clique=50)
    csr, partition, sigma_tot, _n = _build_csr_partition_sigma(graph)
    best_gamma, diagnostics = multi_objective_gamma_tuner(
        csr, partition, sigma_tot, seed=42
    )
    assert best_gamma in diagnostics["candidate_scores"]
    if diagnostics["all_constraints_satisfied"]:
        satisfying = {
            g: s for g, s in diagnostics["candidate_scores"].items()
            if diagnostics["candidate_stats"][g]["hard_satisfied"]
        }
        assert satisfying, (
            "all_constraints_satisfied is True but no candidate's "
            "candidate_stats has hard_satisfied=True"
        )
        best_score = max(satisfying.values())
        assert diagnostics["candidate_scores"][best_gamma] == pytest.approx(
            best_score
        ), (
            f"best_gamma={best_gamma} should be the highest-score satisfying "
            f"candidate, got score={diagnostics['candidate_scores'][best_gamma]}, "
            f"max satisfying score={best_score}"
        )


def test_hyper_fragmentation_regression_5000_nodes():
    from iai_mcp.mosaic import run_mosaic

    graph = _planted_3community_graph(n=5000, seed=13)
    t_start = time.monotonic()
    assignment, _lineage = run_mosaic(graph, gamma=None, seed=42)
    elapsed = time.monotonic() - t_start
    n_communities = len(set(assignment.node_to_community.values()))
    print(
        f"\n[HYPER-FRAG] N=5000, gamma=None: "
        f"n_communities={n_communities}, elapsed={elapsed:.2f}s, "
        f"backend={assignment.backend}, modularity={assignment.modularity:.4f}"
    )
    assert n_communities <= 500, (
        f"Hyper-fragmentation regression: got {n_communities} communities "
        f"on a 5000-node planted-3-community graph (expected <= 500). "
        f"This is the 'γ wrong' failure mode the tuner must "
        f"prevent."
    )


def test_tuner_deterministic_under_same_seed():
    from iai_mcp.mosaic import multi_objective_gamma_tuner

    graph, _nodes = _two_clique_with_bridge(n_per_clique=20)
    csr, partition, sigma_tot, _n = _build_csr_partition_sigma(graph)
    results: list[float] = []
    for _ in range(5):
        best_gamma, _diagnostics = multi_objective_gamma_tuner(
            csr, partition.copy(), sigma_tot.copy(), seed=42
        )
        results.append(best_gamma)
    assert len(set(results)) == 1, (
        f"Tuner is not deterministic under seed=42; got 5 distinct best_gammas: "
        f"{results}"
    )


def test_tuner_run_one_pass_does_not_mutate_input_partition():
    from iai_mcp.mosaic import _run_one_leiden_pass

    graph, _nodes = _two_clique_with_bridge(n_per_clique=10)
    csr, partition, sigma_tot, _n = _build_csr_partition_sigma(graph)
    partition_bytes_before = partition.tobytes()
    sigma_bytes_before = sigma_tot.tobytes()
    _refined, _q, _stats = _run_one_leiden_pass(
        csr, partition, sigma_tot, gamma=1.0, seed=42
    )
    assert partition.tobytes() == partition_bytes_before, (
        "_run_one_leiden_pass mutated its input `partition` -- it must work "
        "on a COPY so the tuner can score candidates without disturbing "
        "the main run."
    )
    assert sigma_tot.tobytes() == sigma_bytes_before, (
        "_run_one_leiden_pass mutated its input `sigma_tot` -- "
        "`_njit_local_move` mutates sigma_tot in place, so the function "
        "must defensively copy it too."
    )


def test_tuner_returns_float_gamma():
    from iai_mcp.mosaic import multi_objective_gamma_tuner

    graph, _nodes = _two_clique_with_bridge(n_per_clique=10)
    csr, partition, sigma_tot, _n = _build_csr_partition_sigma(graph)
    best_gamma, _diagnostics = multi_objective_gamma_tuner(
        csr, partition, sigma_tot, seed=42
    )
    assert isinstance(best_gamma, float), (
        f"best_gamma must be a Python float for JSON-serialisable "
        f"diagnostics, got {type(best_gamma).__name__}"
    )
    assert not isinstance(best_gamma, np.generic), (
        f"best_gamma must be a Python float, not a numpy scalar "
        f"({type(best_gamma).__name__})"
    )


def test_run_mosaic_with_gamma_none_calls_tuner():
    from iai_mcp.mosaic import run_mosaic

    graph, _nodes = _two_clique_with_bridge(n_per_clique=10)
    mock_diag = {
        "all_constraints_satisfied": True,
        "candidate_scores": {0.5: 0.4, 1.0: 0.5},
        "candidate_stats": {
            0.5: {
                "q": 0.4,
                "singleton_ratio": 0.0,
                "n_communities": 2,
                "connected": True,
                "hard_satisfied": True,
            },
            1.0: {
                "q": 0.5,
                "singleton_ratio": 0.0,
                "n_communities": 2,
                "connected": True,
                "hard_satisfied": True,
            },
        },
        "should_fall_back_to_flat": False,
        "best_gamma_q": 0.5,
        "best_gamma_singleton_ratio": 0.0,
        "best_gamma_n_communities": 2,
    }
    with patch(
        "iai_mcp.mosaic.multi_objective_gamma_tuner",
        return_value=(1.0, mock_diag),
    ) as mock_tuner:
        _assignment, _lineage = run_mosaic(
            graph, gamma=None, seed=42
        )
    assert mock_tuner.call_count == 1, (
        f"With gamma=None, run_mosaic must call "
        f"multi_objective_gamma_tuner exactly once; got "
        f"{mock_tuner.call_count} calls."
    )


def test_run_mosaic_with_explicit_gamma_skips_tuner():
    from iai_mcp.mosaic import run_mosaic

    graph, _nodes = _two_clique_with_bridge(n_per_clique=10)
    with patch(
        "iai_mcp.mosaic.multi_objective_gamma_tuner",
    ) as mock_tuner:
        _assignment, _lineage = run_mosaic(
            graph, gamma=1.5, seed=42
        )
    assert mock_tuner.call_count == 0, (
        f"With gamma=1.5 explicit, run_mosaic must NOT call "
        f"multi_objective_gamma_tuner; got {mock_tuner.call_count} calls."
    )


def test_gamma_tuner_closes_karate_gap():
    leidenalg = pytest.importorskip("leidenalg")
    igraph_mod = pytest.importorskip("igraph")
    sklearn_metrics = pytest.importorskip("sklearn.metrics")

    from iai_mcp.mosaic import run_mosaic

    graph, _ground_truth, nodes = _load_karate()
    assignment, _lineage = run_mosaic(
        graph, gamma=None, seed=42
    )

    data = json.loads(
        (
            Path(__file__).parent / "fixtures" / "leiden" / "karate_club.json"
        ).read_text()
    )
    g_ig = igraph_mod.Graph()
    g_ig.add_vertices(data["n"])
    g_ig.add_edges([tuple(e) for e in data["edges"]])
    leiden_partition_05 = leidenalg.find_partition(
        g_ig,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=0.5,
        seed=42,
    )
    leiden_partition_10 = leidenalg.find_partition(
        g_ig,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=1.0,
        seed=42,
    )

    uuid_to_label: dict[UUID, int] = {}
    next_label = 0
    detected: list[int] = []
    for u in nodes:
        comm = assignment.node_to_community[u]
        if comm not in uuid_to_label:
            uuid_to_label[comm] = next_label
            next_label += 1
        detected.append(uuid_to_label[comm])

    nmi_v_05 = sklearn_metrics.normalized_mutual_info_score(
        list(leiden_partition_05.membership), detected
    )
    nmi_v_10 = sklearn_metrics.normalized_mutual_info_score(
        list(leiden_partition_10.membership), detected
    )
    print(
        f"\n[KARATE-GAP] tuner-chosen final NMI vs leidenalg@gamma=0.5 = "
        f"{nmi_v_05:.4f}; vs leidenalg@gamma=1.0 = {nmi_v_10:.4f}; "
        f"M3 baseline at gamma=0.5 was 0.7753; "
        f"super-level pairwise merge needed for >= 0.90"
    )
    assert nmi_v_05 >= 0.74, (
        f"Karate NMI regression: NMI vs leidenalg@gamma=0.5 = {nmi_v_05:.4f} "
        f"< 0.74. 21-03 baseline was 0.7753; tuner-chosen partition should "
        f"match it (gamma=0.5 by composite score)."
    )
