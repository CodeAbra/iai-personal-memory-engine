from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

import numpy as np
import pytest

sklearn_nmi = pytest.importorskip("sklearn.metrics").normalized_mutual_info_score


from iai_mcp.mosaic import (
    EPSILON,
    build_csr_sanitized,
    compute_modularity_cpm,
    compute_sigma_tot,
    run_mosaic,
)
from iai_mcp.mosaic_policy import (
    all_communities_connected,
)
from iai_mcp.graph import MemoryGraph


_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "leiden"


def _emb(seed: int, dim: int = 384) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.random(dim).tolist()


def _load_seeds() -> dict[str, Any]:
    return json.loads((_FIXTURE_DIR / "lfr_seeds.json").read_text())


def _detected_labels_in_canonical_order(
    assignment, ordered_nodes: list[UUID]
) -> list[int]:
    uuid_to_label: dict[UUID, int] = {}
    next_label = 0
    detected: list[int] = []
    for u in ordered_nodes:
        comm_uuid = assignment.node_to_community[u]
        if comm_uuid not in uuid_to_label:
            uuid_to_label[comm_uuid] = next_label
            next_label += 1
        detected.append(uuid_to_label[comm_uuid])
    return detected


def _canonical_uuid_sorted_nodes(graph: MemoryGraph) -> list[UUID]:
    uuid_strings = [(str(u), u) for u in graph.iter_nodes()]
    uuid_strings.sort(key=lambda pair: pair[0])
    return [u for _s, u in uuid_strings]


def _load_karate() -> tuple[MemoryGraph, list[int], list[UUID]]:
    data = json.loads((_FIXTURE_DIR / "karate_club.json").read_text())
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


def _load_football() -> tuple[MemoryGraph, list[int], list[UUID]]:
    data = json.loads((_FIXTURE_DIR / "football.json").read_text())
    g = MemoryGraph()
    football_ns = UUID("87654321-4321-8765-4321-876543218765")
    nodes: list[UUID] = [
        uuid5(football_ns, f"football-{i}") for i in range(data["n"])
    ]
    for i, u in enumerate(nodes):
        g.add_node(u, community_id=None, embedding=_emb(i + 10000))
    for u, v in data["edges"]:
        g.add_edge(nodes[u], nodes[v], weight=1.0)
    return g, data["ground_truth"], nodes


def _load_lfr_variant(name: str) -> tuple[MemoryGraph, list[int], list[UUID]]:
    from tests.fixtures.leiden.lfr_generator import generate_lfr_like  # noqa: PLC0415

    seeds = _load_seeds()
    variant = next(v for v in seeds["variants"] if v["name"] == name)
    g, planted = generate_lfr_like(
        n=variant["n"],
        avg_degree=variant["avg_degree"],
        max_degree=variant["max_degree"],
        mu=variant["mu"],
        n_communities=variant["n_communities"],
        min_community=variant["min_community"],
        max_community=variant["max_community"],
        seed=variant["seed"],
    )
    canonical_nodes = _canonical_uuid_sorted_nodes(g)
    return g, planted, canonical_nodes


def _load_ba_n5000(seed: int = 42) -> tuple[MemoryGraph, None, list[UUID]]:
    import networkx as nx  # noqa: PLC0415

    nx_g = nx.barabasi_albert_graph(5000, 5, seed=seed)
    n = nx_g.number_of_nodes()
    uuids = [UUID(int=seed * 10**12 + i) for i in range(n)]
    g = MemoryGraph()
    for i, u in enumerate(uuids):
        g.add_node(u, community_id=None, embedding=_emb(i + 50000))
    for u_idx, v_idx in nx_g.edges():
        g.add_edge(uuids[u_idx], uuids[v_idx], weight=1.0, edge_type="hebbian")
    canonical = _canonical_uuid_sorted_nodes(g)
    return g, None, canonical


GRAPH_VARIANTS: list[dict[str, Any]] = [
    {
        "name": "karate",
        "loader": _load_karate,
        "nmi_min": 0.50,
        "warm_wall_time_s_max": None,
    },
    {
        "name": "football",
        "loader": _load_football,
        "nmi_min": 0.70,
        "warm_wall_time_s_max": None,
    },
    {
        "name": "lfr_n1000_mu01",
        "loader": lambda: _load_lfr_variant("lfr_n1000_mu01"),
        "nmi_min": 0.90,
        "warm_wall_time_s_max": None,
    },
    {
        "name": "lfr_n1000_mu03",
        "loader": lambda: _load_lfr_variant("lfr_n1000_mu03"),
        "nmi_min": 0.80,
        "warm_wall_time_s_max": None,
    },
    {
        "name": "lfr_n1000_mu05",
        "loader": lambda: _load_lfr_variant("lfr_n1000_mu05"),
        "nmi_min": 0.65,
        "warm_wall_time_s_max": None,
    },
    {
        "name": "lfr_n5000_mu01",
        "loader": lambda: _load_lfr_variant("lfr_n5000_mu01"),
        "nmi_min": 0.90,
        "warm_wall_time_s_max": 5.0,
    },
    {
        "name": "ba_n5000_m5",
        "loader": _load_ba_n5000,
        "nmi_min": None,
        "warm_wall_time_s_max": None,
    },
]


def test_gauntlet_config_well_formed():
    seeds = _load_seeds()
    assert "variants" in seeds
    required_keys = {
        "name", "n", "avg_degree", "max_degree", "mu", "n_communities",
        "min_community", "max_community", "seed", "nmi_min",
    }
    for variant in seeds["variants"]:
        missing = required_keys - set(variant.keys())
        assert not missing, (
            f"variant {variant.get('name')} missing keys: {missing}"
        )
        assert 0.0 <= variant["mu"] <= 1.0
        assert variant["n"] > 0
        assert variant["n_communities"] > 0
        assert variant["nmi_min"] >= 0.0


def test_generator_deterministic():
    from tests.fixtures.leiden.lfr_generator import generate_lfr_like  # noqa: PLC0415

    g1, labels1 = generate_lfr_like(
        n=200, avg_degree=8, max_degree=20, mu=0.2,
        n_communities=5, min_community=20, max_community=80, seed=42,
    )
    g2, labels2 = generate_lfr_like(
        n=200, avg_degree=8, max_degree=20, mu=0.2,
        n_communities=5, min_community=20, max_community=80, seed=42,
    )
    assert labels1 == labels2, "planted_labels not deterministic"
    edges1 = sorted(
        tuple(sorted((str(u), str(v)))) for u, v, _w in g1.iter_edges_with_weight()
    )
    edges2 = sorted(
        tuple(sorted((str(u), str(v)))) for u, v, _w in g2.iter_edges_with_weight()
    )
    assert edges1 == edges2, "edge list not deterministic"


def test_generator_no_self_loops():
    from tests.fixtures.leiden.lfr_generator import generate_lfr_like  # noqa: PLC0415

    g, _labels = generate_lfr_like(
        n=200, avg_degree=8, max_degree=20, mu=0.2,
        n_communities=5, min_community=20, max_community=80, seed=42,
    )
    for u, v, _w in g.iter_edges_with_weight():
        assert u != v, f"self-loop found: ({u}, {v})"


def test_generator_planted_labels_length_matches_n():
    seeds = _load_seeds()
    from tests.fixtures.leiden.lfr_generator import generate_lfr_like  # noqa: PLC0415

    for variant in seeds["variants"]:
        if variant["n"] > 1000:
            continue
        g, planted = generate_lfr_like(
            n=variant["n"],
            avg_degree=variant["avg_degree"],
            max_degree=variant["max_degree"],
            mu=variant["mu"],
            n_communities=variant["n_communities"],
            min_community=variant["min_community"],
            max_community=variant["max_community"],
            seed=variant["seed"],
        )
        assert len(planted) == variant["n"], (
            f"{variant['name']}: planted_labels {len(planted)} != n {variant['n']}"
        )
        assert g.node_count() == variant["n"]


def test_generator_planted_communities_within_size_bounds():
    from tests.fixtures.leiden.lfr_generator import generate_lfr_like  # noqa: PLC0415

    n_communities = 10
    min_size = 20
    max_size = 100
    _g, planted = generate_lfr_like(
        n=500, avg_degree=10, max_degree=30, mu=0.2,
        n_communities=n_communities, min_community=min_size, max_community=max_size,
        seed=42,
    )
    sizes = [planted.count(c) for c in range(n_communities)]
    assert all(s >= min_size for s in sizes), f"undersized community: {sizes}"
    assert all(s <= int(max_size * 1.5) for s in sizes), (
        f"oversized community: {sizes}"
    )


@pytest.mark.parametrize(
    "variant",
    GRAPH_VARIANTS,
    ids=[v["name"] for v in GRAPH_VARIANTS],
)
def test_partition_nmi_vs_ground_truth(variant: dict[str, Any]) -> None:
    if variant["nmi_min"] is None:
        pytest.skip(f"{variant['name']} has no ground truth")

    graph, ground, ordered = variant["loader"]()
    assert ground is not None, f"{variant['name']} loader returned no ground truth"

    assignment, _ = run_mosaic(graph, prior=None, prior_mode="cold", seed=42)
    detected = _detected_labels_in_canonical_order(assignment, ordered)

    nmi = sklearn_nmi(ground, detected, average_method="arithmetic")
    n_detected_comms = len(set(detected))
    n_ground_comms = len(set(ground))
    threshold = variant["nmi_min"]
    print(
        f"\n{variant['name']}: NMI={nmi:.4f} (threshold={threshold}) "
        f"detected={n_detected_comms} comms, ground={n_ground_comms} comms, "
        f"Q={assignment.modularity:.4f}"
    )
    assert nmi >= threshold, (
        f"{variant['name']} NMI={nmi:.4f} < threshold={threshold}. "
        f"detected={n_detected_comms} communities, "
        f"ground_truth={n_ground_comms} communities. "
        f"For Karate, see 21-03/21-05 Known Gap; raise threshold when "
        f"super-level pairwise merge follow-up plan lands."
    )


@pytest.mark.parametrize(
    "variant",
    GRAPH_VARIANTS,
    ids=[v["name"] for v in GRAPH_VARIANTS],
)
def test_all_communities_connected(variant: dict[str, Any]) -> None:
    graph, _ground, ordered = variant["loader"]()
    assignment, _ = run_mosaic(graph, prior=None, prior_mode="cold", seed=42)

    csr, _order, _idx = build_csr_sanitized(graph)
    canonical = _canonical_uuid_sorted_nodes(graph)
    detected = _detected_labels_in_canonical_order(assignment, canonical)
    partition = np.asarray(detected, dtype=np.int64)

    if csr.nnz == 0:
        pytest.skip(f"{variant['name']} has zero edges after sanitisation")

    connected = all_communities_connected(csr, partition)
    assert connected, (
        f"{variant['name']}: at least one community induces a disconnected "
        f"subgraph -- connectivity invariant violated."
    )


@pytest.mark.parametrize(
    "variant",
    GRAPH_VARIANTS,
    ids=[v["name"] for v in GRAPH_VARIANTS],
)
def test_modularity_monotonicity_q_final_ge_q_singleton(
    variant: dict[str, Any],
) -> None:
    graph, _ground, _ordered = variant["loader"]()

    csr, _order, _ = build_csr_sanitized(graph)
    if csr.nnz == 0:
        pytest.skip(f"{variant['name']} has zero edges after sanitisation")

    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    n = indptr.shape[0] - 1
    singleton = np.arange(n, dtype=np.int64)
    sigma_singleton = compute_sigma_tot(indptr, indices, data, singleton, n)
    q_singleton = compute_modularity_cpm(
        indptr, indices, data, singleton, sigma_singleton, 1.0
    )

    assignment, _ = run_mosaic(graph, prior=None, prior_mode="cold", seed=42)
    q_final = assignment.modularity

    assert q_final + EPSILON >= q_singleton, (
        f"{variant['name']}: modularity monotonicity violated -- "
        f"Q_final={q_final:.4f} < Q_singleton={q_singleton:.4f}"
    )


@pytest.mark.parametrize(
    "variant",
    GRAPH_VARIANTS,
    ids=[v["name"] for v in GRAPH_VARIANTS],
)
def test_replay_determinism_5x(variant: dict[str, Any]) -> None:
    graph, _ground, _ordered = variant["loader"]()
    canonical = _canonical_uuid_sorted_nodes(graph)

    n_runs = 2 if variant["name"] in ("ba_n5000_m5", "lfr_n5000_mu01") else 5

    partitions: list[list[int]] = []
    for _ in range(n_runs):
        assignment, _ = run_mosaic(graph, prior=None, prior_mode="cold", seed=42)
        partitions.append(_detected_labels_in_canonical_order(assignment, canonical))

    reference = partitions[0]
    for i, p in enumerate(partitions[1:], start=1):
        assert p == reference, (
            f"{variant['name']} run {i+1} differs from run 1 -- determinism "
            f"violated."
        )


@pytest.mark.parametrize(
    "variant",
    [v for v in GRAPH_VARIANTS if v.get("warm_wall_time_s_max") is not None],
    ids=lambda v: v["name"],
)
def test_warm_wall_time_under_budget(variant: dict[str, Any]) -> None:
    graph, _ground, _ordered = variant["loader"]()
    budget = variant["warm_wall_time_s_max"]

    run_mosaic(graph, prior=None, prior_mode="cold", seed=42)

    warm_times: list[float] = []
    for _ in range(3):
        t0 = time.monotonic()
        run_mosaic(graph, prior=None, prior_mode="cold", seed=42)
        warm_times.append(time.monotonic() - t0)

    median = sorted(warm_times)[1]
    print(
        f"\n{variant['name']}: warm wall-times = "
        f"[{warm_times[0]:.2f}s, {warm_times[1]:.2f}s, {warm_times[2]:.2f}s], "
        f"median={median:.2f}s, budget={budget}s"
    )
    assert median <= budget, (
        f"{variant['name']}: median warm wall-time {median:.2f}s exceeds "
        f"budget {budget}s. Times = {warm_times}."
    )


_CROSS_PROCESS_VARIANTS = [
    v for v in GRAPH_VARIANTS if v["name"] in ("ba_n5000_m5", "lfr_n5000_mu01")
]


@pytest.mark.parametrize(
    "variant",
    _CROSS_PROCESS_VARIANTS,
    ids=lambda v: v["name"],
)
def test_cross_process_replay(variant: dict[str, Any], tmp_path: Path) -> None:
    graph, _ground, _ordered = variant["loader"]()
    canonical = _canonical_uuid_sorted_nodes(graph)

    assignment, _ = run_mosaic(graph, prior=None, prior_mode="cold", seed=42)
    in_proc = _detected_labels_in_canonical_order(assignment, canonical)

    repo_root = Path(__file__).parent.parent
    name = variant["name"]
    if name == "ba_n5000_m5":
        construct_block = (
            "import networkx as nx\n"
            "from uuid import UUID\n"
            "from iai_mcp.graph import MemoryGraph\n"
            "nx_g = nx.barabasi_albert_graph(5000, 5, seed=42)\n"
            "n = nx_g.number_of_nodes()\n"
            "uuids = [UUID(int=42*10**12 + i) for i in range(n)]\n"
            "g = MemoryGraph()\n"
            "for i, u in enumerate(uuids):\n"
            "    g.add_node(u, community_id=None, embedding=[0.1]*384)\n"
            "for u_idx, v_idx in nx_g.edges():\n"
            "    g.add_edge(uuids[u_idx], uuids[v_idx],"
            " weight=1.0, edge_type='hebbian')\n"
        )
    elif name == "lfr_n5000_mu01":
        construct_block = (
            "from tests.fixtures.leiden.lfr_generator import generate_lfr_like\n"
            "g, _ = generate_lfr_like(n=5000, avg_degree=20, max_degree=100, "
            "mu=0.1, n_communities=50, min_community=50, max_community=200, seed=42)\n"
        )
    else:
        pytest.skip(f"cross-process replay not configured for {name}")

    script = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        f"sys.path.insert(0, {str(repo_root / 'src')!r})\n"
        "from uuid import UUID\n"
        "from iai_mcp.mosaic import run_mosaic\n"
        f"{construct_block}"
        "assignment, _ = run_mosaic(g, prior=None, prior_mode='cold', seed=42)\n"
        "# Canonical str-sorted UUID order.\n"
        "ordered = sorted(str(uuid_) for uuid_ in g.iter_nodes())\n"
        "uuid_to_label = {}\n"
        "labels = []\n"
        "next_label = 0\n"
        "for s in ordered:\n"
        "    u = UUID(s)\n"
        "    cu = assignment.node_to_community[u]\n"
        "    if cu not in uuid_to_label:\n"
        "        uuid_to_label[cu] = next_label\n"
        "        next_label += 1\n"
        "    labels.append(uuid_to_label[cu])\n"
        "print(json.dumps(labels))\n"
    )

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "99"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        pytest.fail(
            f"subprocess failed (rc={result.returncode}):\n"
            f"STDOUT: {result.stdout[-2000:]}\n"
            f"STDERR: {result.stderr[-2000:]}"
        )
    lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
    assert lines, f"subprocess stdout empty:\nSTDOUT={result.stdout}\nSTDERR={result.stderr}"
    sub_proc = json.loads(lines[-1])

    assert in_proc == sub_proc, (
        f"{name}: cross-process partition mismatch. "
        f"In-process unique communities: {len(set(in_proc))}, "
        f"Subprocess unique communities: {len(set(sub_proc))}. "
        f"PYTHONHASHSEED-independent contract violated."
    )


def test_ba_n5000_m5_modularity_above_threshold() -> None:
    graph, _ground, _ordered = _load_ba_n5000()
    assignment, _ = run_mosaic(graph, prior=None, prior_mode="cold", seed=42)

    assert assignment.backend in ("leiden-custom", "flat"), (
        f"unexpected backend {assignment.backend}"
    )
    if assignment.backend == "flat":
        pytest.fail(
            "BA n=5000 m=5 fell back to flat -- partition has 1 community; "
            "modularity-threshold test does not apply. This is a regression "
            "signal to investigate."
        )
    print(
        f"\nba_n5000_m5: Q={assignment.modularity:.4f}, "
        f"backend={assignment.backend}, "
        f"n_communities={len({u for u in assignment.node_to_community.values()})}"
    )
    assert assignment.modularity > 0.3, (
        f"BA(N=5000, m=5) CPM-Q = {assignment.modularity:.4f} below "
        f"0.3 threshold."
    )
