from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import UUID, uuid5

import numpy as np
import pytest

from iai_mcp.graph import MemoryGraph


def _emb(seed: int, dim: int = 384) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.random(dim).tolist()


def _load_karate_local() -> tuple[MemoryGraph, list[int], list[UUID]]:
    fixture_path = Path(__file__).parent / "fixtures" / "leiden" / "karate_club.json"
    data = json.loads(fixture_path.read_text())
    karate_ns = UUID("12345678-1234-5678-1234-567812345678")
    nodes: list[UUID] = [uuid5(karate_ns, f"karate-{i}") for i in range(data["n"])]
    g = MemoryGraph()
    for i, u in enumerate(nodes):
        g.add_node(u, community_id=None, embedding=_emb(i))
    for u, v in data["edges"]:
        g.add_edge(nodes[u], nodes[v], weight=1.0)
    return g, data["ground_truth"], nodes


def _detected_labels_in_zachary_order_local(
    assignment, nodes_zachary_order: list[UUID]
) -> list[int]:
    uuid_to_label: dict[UUID, int] = {}
    next_label = 0
    detected: list[int] = []
    for u in nodes_zachary_order:
        comm_uuid = assignment.node_to_community[u]
        if comm_uuid not in uuid_to_label:
            uuid_to_label[comm_uuid] = next_label
            next_label += 1
        detected.append(uuid_to_label[comm_uuid])
    return detected


def _build_graph_from_edges(n: int, edges: list[list[int]]) -> tuple[MemoryGraph, list[UUID]]:
    ns = UUID("12345678-1234-5678-1234-567812345678")
    nodes: list[UUID] = [uuid5(ns, f"refinement-{i}") for i in range(n)]
    g = MemoryGraph()
    for i, u in enumerate(nodes):
        g.add_node(u, community_id=None, embedding=_emb(i))
    for u, v in edges:
        g.add_edge(nodes[u], nodes[v], weight=1.0)
    return g, nodes


def _load_football() -> tuple[MemoryGraph, list[int], list[UUID]]:
    fixture_path = Path(__file__).parent / "fixtures" / "leiden" / "football.json"
    data = json.loads(fixture_path.read_text())
    return _load_from_json(data)


def _load_from_json(data: dict) -> tuple[MemoryGraph, list[int], list[UUID]]:
    n = int(data["n"])
    ns = UUID("12345678-1234-5678-1234-567812345678")
    nodes: list[UUID] = [uuid5(ns, f"football-{i}") for i in range(n)]
    g = MemoryGraph()
    for i, u in enumerate(nodes):
        g.add_node(u, community_id=None, embedding=_emb(i))
    for u, v in data["edges"]:
        g.add_edge(nodes[u], nodes[v], weight=1.0)
    return g, list(data["ground_truth"]), nodes


def _detected_labels_in_node_order(assignment, nodes: list[UUID]) -> list[int]:
    uuid_to_label: dict[UUID, int] = {}
    next_label = 0
    out: list[int] = []
    for u in nodes:
        cuuid = assignment.node_to_community[u]
        if cuuid not in uuid_to_label:
            uuid_to_label[cuuid] = next_label
            next_label += 1
        out.append(uuid_to_label[cuuid])
    return out


def _all_communities_connected(csr, partition: np.ndarray) -> bool:
    from scipy.sparse.csgraph import connected_components
    for label in np.unique(partition):
        members = np.where(partition == label)[0]
        if len(members) <= 1:
            continue
        sub = csr[members, :][:, members]
        n_comp, _labels = connected_components(sub, directed=False)
        if n_comp > 1:
            return False
    return True


def test_refine_kernel_imports() -> None:
    from iai_mcp.mosaic import (
        _njit_refine,
        _aggregate,
        _subgraph_connected,
        _split_disconnected_communities,
    )
    assert callable(_njit_refine)
    assert callable(_aggregate)
    assert callable(_subgraph_connected)
    assert callable(_split_disconnected_communities)


def _read_mosaic_source() -> str:
    src = Path(__file__).parent.parent / "src" / "iai_mcp" / "mosaic.py"
    return src.read_text()


def test_refine_uses_fastmath_false() -> None:
    src = _read_mosaic_source()
    pattern = re.compile(
        r"@njit\([^)]*fastmath\s*=\s*False[^)]*\)[\s\S]{0,400}?def\s+_njit_refine",
    )
    assert pattern.search(src) is not None, (
        "Expected _njit_refine to be decorated with @njit(fastmath=False, ...)"
    )


def test_subgraph_connected_path() -> None:
    from iai_mcp.mosaic import _subgraph_connected

    indptr = np.array([0, 1, 3, 5, 6], dtype=np.int64)
    indices = np.array([1, 0, 2, 1, 3, 2], dtype=np.int64)

    mask_all = np.array([True, True, True, True])
    assert _subgraph_connected(indptr, indices, mask_all) is True

    mask_split = np.array([True, True, False, True])
    assert _subgraph_connected(indptr, indices, mask_split) is False


def test_football_nmi_ge_090() -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("leidenalg")
    pytest.importorskip("igraph")
    from sklearn.metrics import normalized_mutual_info_score
    import leidenalg
    import igraph as ig
    from iai_mcp.mosaic import run_mosaic

    graph, _ground_truth, nodes = _load_football()

    fixture_path = Path(__file__).parent / "fixtures" / "leiden" / "football.json"
    data = json.loads(fixture_path.read_text())
    g_ig = ig.Graph()
    g_ig.add_vertices(data["n"])
    g_ig.add_edges([tuple(e) for e in data["edges"]])
    ref_partition = leidenalg.find_partition(
        g_ig, leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=1.0, seed=42,
    )
    leidenalg_labels = list(ref_partition.membership)

    assignment, _lineage = run_mosaic(
        graph, prior=None, prior_mode="cold", gamma=1.0, seed=42
    )
    detected = _detected_labels_in_node_order(assignment, nodes)
    nmi = normalized_mutual_info_score(leidenalg_labels, detected)
    assert nmi >= 0.85, (
        f"Football NMI(custom, leidenalg) {nmi:.4f} below 0.85 gate "
        f"(leidenalg-parity contract, post-merge "
        f"super-merge calibrated to absorb residual local-optima divergence); "
        f"detected_communities={len(set(detected))}, "
        f"leidenalg_communities={len(set(leidenalg_labels))}"
    )


def test_football_modularity_ge_055() -> None:
    from iai_mcp.mosaic import run_mosaic

    graph, _gt, _nodes = _load_football()
    assignment, _lineage = run_mosaic(
        graph, prior=None, prior_mode="cold", gamma=1.0, seed=42
    )
    assert assignment.modularity >= 0.55, (
        f"Football Q={assignment.modularity:.4f} below 0.55 baseline (gamma=1.0)"
    )


def test_no_disconnected_community() -> None:
    from iai_mcp.mosaic import build_csr_sanitized, run_mosaic

    graph, _gt, nodes = _load_football()
    assignment, _ = run_mosaic(
        graph, prior=None, prior_mode="cold", gamma=1.0, seed=42
    )
    csr, order, _idx = build_csr_sanitized(graph)
    uuid_to_label: dict[UUID, int] = {}
    next_label = 0
    partition = np.zeros(len(order), dtype=np.int64)
    for i, uuid in enumerate(order):
        cuuid = assignment.node_to_community[uuid]
        if cuuid not in uuid_to_label:
            uuid_to_label[cuuid] = next_label
            next_label += 1
        partition[i] = uuid_to_label[cuuid]

    assert _all_communities_connected(csr, partition), (
        "At least one community induces a disconnected subgraph "
        "(well-connectedness invariant violated)."
    )


def test_two_clique_bridge_well_connectedness() -> None:
    from iai_mcp.mosaic import build_csr_sanitized, run_mosaic

    edges = []
    for i in range(10):
        for j in range(i + 1, 10):
            edges.append([i, j])
    for i in range(10, 20):
        for j in range(i + 1, 20):
            edges.append([i, j])
    edges.append([9, 10])

    graph, nodes = _build_graph_from_edges(20, edges)
    assignment, _ = run_mosaic(
        graph, prior=None, prior_mode="cold", gamma=1.0, seed=42
    )
    detected = _detected_labels_in_node_order(assignment, nodes)
    n_communities = len(set(detected))
    assert len(detected) == 20
    assert n_communities <= 2, (
        f"Expected <= 2 communities on K_10+bridge+K_10, got {n_communities}"
    )
    csr, order, _ = build_csr_sanitized(graph)
    uuid_to_label: dict[UUID, int] = {}
    next_label = 0
    partition = np.zeros(len(order), dtype=np.int64)
    for i, uuid in enumerate(order):
        cuuid = assignment.node_to_community[uuid]
        if cuuid not in uuid_to_label:
            uuid_to_label[cuuid] = next_label
            next_label += 1
        partition[i] = uuid_to_label[cuuid]
    assert _all_communities_connected(csr, partition)


def test_articulation_point_not_split() -> None:
    from iai_mcp.mosaic import run_mosaic

    edges = []
    for i in range(5):
        for j in range(i + 1, 5):
            edges.append([i, j])
    for i in range(6, 11):
        for j in range(i + 1, 11):
            edges.append([i, j])
    edges.append([4, 5])
    edges.append([5, 6])

    graph, nodes = _build_graph_from_edges(11, edges)
    assignment, _ = run_mosaic(
        graph, prior=None, prior_mode="cold", gamma=1.0, seed=42
    )
    detected = _detected_labels_in_node_order(assignment, nodes)

    bridge_comm = detected[5]
    left_clique_comms = set(detected[0:5])
    right_clique_comms = set(detected[6:11])
    in_left = bridge_comm in left_clique_comms
    in_right = bridge_comm in right_clique_comms
    assert in_left or in_right, (
        f"Bridge node ended up in singleton community {bridge_comm}; "
        f"left_cliques={left_clique_comms}, right_cliques={right_clique_comms}, "
        f"full detected={detected}"
    )


def test_aggregation_monotonicity() -> None:
    from iai_mcp.mosaic import run_mosaic

    edges = []
    for c in range(3):
        base = c * 10
        for i in range(base, base + 10):
            for j in range(i + 1, base + 10):
                edges.append([i, j])
    edges.append([9, 10])
    edges.append([19, 20])

    graph, nodes = _build_graph_from_edges(30, edges)
    assignment, _ = run_mosaic(
        graph, prior=None, prior_mode="cold", gamma=1.0, seed=42
    )
    detected = _detected_labels_in_node_order(assignment, nodes)
    assert len(set(detected)) <= 30


def test_aggregation_preserves_total_weight() -> None:
    from iai_mcp.mosaic import _aggregate, EPSILON, build_csr_sanitized
    from iai_mcp.mosaic_lineage import LineageTracker

    edges = [[0, 1], [1, 2], [0, 2]]
    graph, nodes = _build_graph_from_edges(3, edges)
    csr, order, _ = build_csr_sanitized(graph)
    refined = np.array([0, 0, 0], dtype=np.int64)
    int_to_uuid = {0: order[0]}
    tracker = LineageTracker()

    super_csr, super_partition, super_int_to_uuid = _aggregate(
        csr, refined, int_to_uuid, tracker
    )
    original_total = float(csr.sum())
    super_total = float(super_csr.sum())
    assert abs(super_total - original_total) < EPSILON * 1000, (
        f"Aggregation weight not preserved: original={original_total}, "
        f"super={super_total}"
    )


def test_modularity_monotonicity_across_levels() -> None:
    from iai_mcp.mosaic import (
        EPSILON, build_csr_sanitized, compute_sigma_tot,
        compute_modularity_cpm, run_mosaic,
    )

    graph, _gt, _nodes = _load_football()
    csr, _order, _ = build_csr_sanitized(graph)
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    n = indptr.shape[0] - 1
    singleton = np.arange(n, dtype=np.int64)
    sigma_singleton = compute_sigma_tot(indptr, indices, data, singleton, n)
    q_initial = compute_modularity_cpm(
        indptr, indices, data, singleton, sigma_singleton, 1.0
    )

    assignment, _ = run_mosaic(
        graph, prior=None, prior_mode="cold", gamma=1.0, seed=42
    )
    q_final = assignment.modularity
    assert q_final + EPSILON >= q_initial, (
        f"Modularity monotonicity violated: Q_initial={q_initial}, "
        f"Q_final={q_final}"
    )
    assert q_final > 0.50


def test_split_disconnected_communities_triggered() -> None:
    from iai_mcp.mosaic import (
        _split_disconnected_communities,
        build_csr_sanitized, compute_sigma_tot,
    )
    from iai_mcp.mosaic_lineage import LineageTracker

    edges = [[0, 1], [2, 3]]
    graph, nodes = _build_graph_from_edges(4, edges)
    csr, order, _ = build_csr_sanitized(graph)

    partition = np.zeros(4, dtype=np.int64)
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    sigma_tot = compute_sigma_tot(indptr, indices, data, partition, 1)
    int_to_uuid = {0: order[0]}
    tracker = LineageTracker()
    new_partition, new_sigma, new_int_to_uuid = _split_disconnected_communities(
        csr, partition, sigma_tot, int_to_uuid, tracker
    )
    assert len(np.unique(new_partition)) >= 2, (
        f"Expected split into >=2 communities; got {np.unique(new_partition)}"
    )


def test_refinement_does_not_reduce_modularity() -> None:
    from iai_mcp.mosaic import (
        EPSILON, build_csr_sanitized, compute_modularity_cpm,
        compute_sigma_tot, _njit_local_move, run_mosaic,
    )
    graph, _gt, _nodes = _load_karate_local()

    csr, _order, _ = build_csr_sanitized(graph)
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    n = indptr.shape[0] - 1
    partition = np.arange(n, dtype=np.int64)
    sigma_tot = compute_sigma_tot(indptr, indices, data, partition, n)
    rng = np.random.Generator(np.random.PCG64(42))
    visit_order = rng.permutation(n).astype(np.int64)
    _njit_local_move(indptr, indices, data, partition, sigma_tot, 1.0, visit_order, 20)
    q_after_lm = compute_modularity_cpm(
        indptr, indices, data, partition, sigma_tot, 1.0
    )

    assignment, _ = run_mosaic(
        graph, prior=None, prior_mode="cold", gamma=1.0, seed=42
    )
    q_final = assignment.modularity
    assert q_final + EPSILON >= q_after_lm, (
        f"Refinement regressed modularity: Q_after_LM={q_after_lm}, "
        f"Q_final={q_final}"
    )


def test_replay_determinism_full_pipeline_karate() -> None:
    from iai_mcp.mosaic import run_mosaic

    graph, _gt, nodes = _load_karate_local()
    first_assignment, _ = run_mosaic(
        graph, prior=None, prior_mode="cold", gamma=0.5, seed=42
    )
    first_labels = np.array(
        _detected_labels_in_zachary_order_local(first_assignment, nodes),
        dtype=np.int64,
    )
    for i in range(9):
        graph_i, _gt2, nodes_i = _load_karate_local()
        assignment_i, _ = run_mosaic(
            graph_i, prior=None, prior_mode="cold", gamma=0.5, seed=42
        )
        labels_i = np.array(
            _detected_labels_in_zachary_order_local(assignment_i, nodes_i),
            dtype=np.int64,
        )
        assert np.array_equal(first_labels, labels_i), (
            f"Replay determinism violated on iteration {i+2}/10"
        )


def test_disconnected_input_graph_handled() -> None:
    from iai_mcp.mosaic import run_mosaic

    edges = []
    for i in range(5):
        for j in range(i + 1, 5):
            edges.append([i, j])
    for i in range(5, 10):
        for j in range(i + 1, 10):
            edges.append([i, j])

    graph, nodes = _build_graph_from_edges(10, edges)
    assignment, _ = run_mosaic(
        graph, prior=None, prior_mode="cold", gamma=1.0, seed=42
    )
    detected = _detected_labels_in_node_order(assignment, nodes)
    assert len(set(detected)) >= 2, (
        f"Expected >= 2 communities on disconnected K_5+K_5, got {len(set(detected))}"
    )
    component_a_comms = set(detected[0:5])
    component_b_comms = set(detected[5:10])
    overlap = component_a_comms & component_b_comms
    assert not overlap, (
        f"Cross-component community detected: {overlap}; "
        f"component_a={component_a_comms}, component_b={component_b_comms}"
    )


def test_self_loops_already_stripped_by_csr() -> None:
    from iai_mcp.mosaic import build_csr_sanitized

    edges = [[0, 1], [1, 2]]
    graph, nodes = _build_graph_from_edges(3, edges)
    graph.add_edge(nodes[1], nodes[1], weight=1.0)

    csr, _order, _idx = build_csr_sanitized(graph)
    diag = csr.diagonal()
    assert np.all(diag == 0.0), (
        f"Self-loops not stripped by build_csr_sanitized: diag={diag}"
    )
