from __future__ import annotations

import json
import time
from pathlib import Path
from uuid import UUID, uuid5

import numpy as np
import pytest

from iai_mcp.graph import MemoryGraph


def _emb(seed: int, dim: int = 384) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.random(dim).tolist()


def _load_karate() -> tuple[MemoryGraph, list[int], list[UUID]]:
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


def _build_two_clique_bridge(k: int = 5) -> tuple[MemoryGraph, list[UUID]]:
    ns = UUID("12345678-1234-5678-1234-567812345678")
    nodes: list[UUID] = [uuid5(ns, f"clique-{i}") for i in range(2 * k)]
    g = MemoryGraph()
    for i, u in enumerate(nodes):
        g.add_node(u, community_id=None, embedding=_emb(i))
    for i in range(k):
        for j in range(i + 1, k):
            g.add_edge(nodes[i], nodes[j], weight=1.0)
    for i in range(k, 2 * k):
        for j in range(i + 1, 2 * k):
            g.add_edge(nodes[i], nodes[j], weight=1.0)
    g.add_edge(nodes[0], nodes[k], weight=1.0)
    return g, nodes


def _build_dense_random_graph(n: int = 30, seed: int = 42, p: float = 0.5) -> tuple[MemoryGraph, list[UUID]]:
    rng = np.random.default_rng(seed)
    ns = UUID("12345678-1234-5678-1234-567812345678")
    nodes: list[UUID] = [uuid5(ns, f"dense-{i}") for i in range(n)]
    g = MemoryGraph()
    for i, u in enumerate(nodes):
        g.add_node(u, community_id=None, embedding=_emb(i))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                g.add_edge(nodes[i], nodes[j], weight=1.0)
    return g, nodes


def _detected_labels_in_canonical_order(assignment, nodes_in_canonical_order: list[UUID]) -> list[int]:
    uuid_to_label: dict[UUID, int] = {}
    nl = 0
    out: list[int] = []
    for u in nodes_in_canonical_order:
        cuuid = assignment.node_to_community[u]
        if cuuid not in uuid_to_label:
            uuid_to_label[cuuid] = nl
            nl += 1
        out.append(uuid_to_label[cuuid])
    return out


def test_super_merge_symbol_exists() -> None:
    from iai_mcp.mosaic import _super_level_merge
    assert callable(_super_level_merge)


def test_super_merge_closes_karate_gap() -> None:
    pytest.importorskip("leidenalg")
    pytest.importorskip("igraph")
    pytest.importorskip("sklearn")
    import leidenalg
    import igraph as ig
    from sklearn.metrics import normalized_mutual_info_score

    from iai_mcp.mosaic import run_mosaic

    fixture_path = Path(__file__).parent / "fixtures" / "leiden" / "karate_club.json"
    data = json.loads(fixture_path.read_text())

    g_ig = ig.Graph()
    g_ig.add_vertices(data["n"])
    g_ig.add_edges([tuple(e) for e in data["edges"]])
    ref_partition = leidenalg.find_partition(
        g_ig, leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=0.5, seed=42,
    )
    leidenalg_labels = list(ref_partition.membership)

    graph, _ground_truth, nodes = _load_karate()
    assignment, _lineage = run_mosaic(
        graph, prior=None, prior_mode="cold", gamma=0.5, seed=42
    )

    detected = _detected_labels_in_canonical_order(assignment, nodes)
    nmi = normalized_mutual_info_score(leidenalg_labels, detected)
    assert nmi >= 0.90, (
        f"Karate NMI(custom, leidenalg) {nmi:.4f} below 0.90 gate; "
        f"super-merge should close the 0.7753 -> >= 0.90 gap. "
        f"detected={detected[:5]}... leidenalg={leidenalg_labels[:5]}..."
    )


def test_super_merge_idempotent_on_optimal_partition() -> None:
    from iai_mcp.mosaic import (
        run_mosaic, build_csr_sanitized, _super_level_merge,
        compute_sigma_tot,
    )

    graph, nodes = _build_two_clique_bridge(k=5)

    assignment, _ = run_mosaic(
        graph, prior=None, prior_mode="cold", gamma=1.0, seed=42,
    )
    csr, order, _idx_map = build_csr_sanitized(graph)
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data_arr = np.ascontiguousarray(csr.data, dtype=np.float64)
    n = len(order)
    uuid_to_label: dict[UUID, int] = {}
    nl = 0
    partition = np.zeros(n, dtype=np.int64)
    for i, u in enumerate(order):
        c = assignment.node_to_community[u]
        if c not in uuid_to_label:
            uuid_to_label[c] = nl
            nl += 1
        partition[i] = uuid_to_label[c]

    assert nl == 2, (
        f"Pre-condition violated: expected custom_leiden to find 2 "
        f"communities on K_5-bridge-K_5, got {nl}. Refinement may be "
        f"under-converged; this fixture is supposed to be unambiguous."
    )

    k = nl
    sigma_tot = compute_sigma_tot(indptr, indices, data_arr, partition, k)
    partition_before = partition.copy()

    n_merges = _super_level_merge(
        csr, partition, sigma_tot, gamma=1.0, seed=42, lineage_tracker=None,
    )

    assert n_merges == 0, (
        f"super-merge should NOT consolidate already-optimal partition "
        f"on K_5-bridge-K_5 (got {n_merges} merges)."
    )
    np.testing.assert_array_equal(
        partition, partition_before,
        err_msg="partition was mutated despite zero accepted merges",
    )


def test_super_merge_deterministic() -> None:
    from iai_mcp.mosaic import run_mosaic

    graph_a, _, nodes_a = _load_karate()
    graph_b, _, nodes_b = _load_karate()

    assignment_a, _ = run_mosaic(
        graph_a, prior=None, prior_mode="cold", gamma=0.5, seed=42,
    )
    assignment_b, _ = run_mosaic(
        graph_b, prior=None, prior_mode="cold", gamma=0.5, seed=42,
    )

    labels_a = _detected_labels_in_canonical_order(assignment_a, nodes_a)
    labels_b = _detected_labels_in_canonical_order(assignment_b, nodes_b)

    def _canonical(seq: list[int]) -> list[int]:
        remap: dict[int, int] = {}
        nxt = 0
        out: list[int] = []
        for v in seq:
            if v not in remap:
                remap[v] = nxt
                nxt += 1
            out.append(remap[v])
        return out

    assert _canonical(labels_a) == _canonical(labels_b), (
        f"Determinism violated: same seed produced different partitions.\n"
        f"  run_a labels[:10] = {labels_a[:10]}\n"
        f"  run_b labels[:10] = {labels_b[:10]}"
    )


def test_super_merge_lineage_events_recorded() -> None:
    from iai_mcp.mosaic import (
        build_csr_sanitized, _super_level_merge, compute_sigma_tot,
    )
    from iai_mcp.mosaic_lineage import LineageTracker
    from uuid import uuid4

    graph, _, _ = _load_karate()
    csr, order, _idx_map = build_csr_sanitized(graph)
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data_arr = np.ascontiguousarray(csr.data, dtype=np.float64)
    n = len(order)

    from iai_mcp.mosaic import _run_one_leiden_pass
    import scipy.sparse
    csr_re = scipy.sparse.csr_matrix(
        (data_arr, indices, indptr), shape=(n, n)
    )
    init = np.arange(n, dtype=np.int64)
    init_sigma = compute_sigma_tot(indptr, indices, data_arr, init, n)
    # gamma=1.0 (the canonical CPM resolution) fragments Karate into a
    # multi-community pre-merge partition so the coarser super-merge below has
    # something to collapse. At gamma=0.5 the local move already recovers the
    # true 2-faction split, leaving super-merge nothing to do.
    pre_merge_partition, _q, _stats = _run_one_leiden_pass(
        csr_re, init, init_sigma, gamma=1.0, seed=42,
    )
    partition = pre_merge_partition.copy()
    k = int(len(np.unique(partition)))
    assert k >= 3, (
        f"Pre-condition: expected pre-super-merge to find >=3 "
        f"communities on Karate(gamma=1.0); got {k}."
    )

    sigma_tot = compute_sigma_tot(indptr, indices, data_arr, partition, k)

    label_to_uuid = {i: uuid4() for i in range(k)}

    fresh_tracker = LineageTracker()
    n_merges = _super_level_merge(
        csr, partition, sigma_tot, gamma=0.5, seed=42,
        lineage_tracker=fresh_tracker, label_to_uuid=label_to_uuid,
    )

    assert n_merges >= 1, (
        f"super-merge should accept >= 1 merge on the pre-super-merge "
        f"Karate partition at gamma=0.5 (analytical sim: 2 merges); "
        f"got {n_merges}. k_before={k}."
    )

    merge_events = [
        e for e in fresh_tracker.report().events if e.event_type == "merge"
    ]
    assert len(merge_events) >= n_merges, (
        f"Expected at least {n_merges} 'merge' lineage events recorded "
        f"by super-merge; got {len(merge_events)}. Lineage hook missing?"
    )


def test_super_merge_terminates_within_max_iter() -> None:
    from iai_mcp.mosaic import (
        run_mosaic, build_csr_sanitized, _super_level_merge,
        compute_sigma_tot,
    )

    graph, _ = _build_dense_random_graph(n=30, seed=42, p=0.5)
    assignment, _ = run_mosaic(
        graph, prior=None, prior_mode="cold", gamma=1.0, seed=42,
    )
    csr, order, _ = build_csr_sanitized(graph)
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data_arr = np.ascontiguousarray(csr.data, dtype=np.float64)
    n = len(order)

    uuid_to_label: dict[UUID, int] = {}
    nl = 0
    partition = np.zeros(n, dtype=np.int64)
    for i, u in enumerate(order):
        c = assignment.node_to_community[u]
        if c not in uuid_to_label:
            uuid_to_label[c] = nl
            nl += 1
        partition[i] = uuid_to_label[c]

    k = nl
    sigma_tot = compute_sigma_tot(indptr, indices, data_arr, partition, k)

    t0 = time.monotonic()
    n_merges = _super_level_merge(
        csr, partition, sigma_tot, gamma=1.0, seed=42, lineage_tracker=None,
        max_iter=5,
    )
    wall_s = time.monotonic() - t0

    assert wall_s < 5.0, (
        f"super-merge wall-time {wall_s:.2f}s exceeded 5s budget on a "
        f"30-node dense graph (n_merges={n_merges}); possible infinite "
        f"loop bug or pathological pair-enumeration."
    )
    assert 0 <= n_merges < k, (
        f"super-merge merge count {n_merges} out of bounds [0, {k-1}]"
    )
