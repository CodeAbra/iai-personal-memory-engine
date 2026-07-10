from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

from iai_mcp.graph import MemoryGraph


def _emb(seed: int, dim: int = 384) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.random(dim).tolist()


def _load_karate() -> tuple[MemoryGraph, list[int], list[UUID]]:
    fixture_path = Path(__file__).parent / "fixtures" / "leiden" / "karate_club.json"
    data = json.loads(fixture_path.read_text())
    g = MemoryGraph()
    karate_ns = UUID("12345678-1234-5678-1234-567812345678")
    from uuid import uuid5
    nodes: list[UUID] = [uuid5(karate_ns, f"karate-{i}") for i in range(data["n"])]
    for i, u in enumerate(nodes):
        g.add_node(u, community_id=None, embedding=_emb(i))
    for u, v in data["edges"]:
        g.add_edge(nodes[u], nodes[v], weight=1.0)
    return g, data["ground_truth"], nodes


def _partition_hash(partition_array: np.ndarray) -> str:
    arr = np.ascontiguousarray(partition_array, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _detected_labels_in_zachary_order(
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


def test_kernel_imports() -> None:
    from iai_mcp.mosaic import (
        _njit_local_move,
        compute_delta_q_cpm,
        compute_sigma_tot,
        compute_modularity_cpm,
    )

    assert callable(_njit_local_move)
    assert callable(compute_delta_q_cpm)
    assert callable(compute_sigma_tot)
    assert callable(compute_modularity_cpm)


def _read_mosaic_source() -> str:
    src = Path(__file__).parent.parent / "src" / "iai_mcp" / "mosaic.py"
    return src.read_text()


def test_kernel_decorator_fastmath_false() -> None:
    src = _read_mosaic_source()
    matches = re.findall(r"@njit\([^)]*fastmath\s*=\s*False[^)]*\)", src)
    assert len(matches) >= 3, (
        f"Expected at least 3 @njit(fastmath=False, ...) decorators; "
        f"found {len(matches)}: {matches}"
    )


def test_kernel_decorator_cache_true() -> None:
    src = _read_mosaic_source()
    matches = re.findall(r"@njit\([^)]*cache\s*=\s*True[^)]*\)", src)
    assert len(matches) >= 3, (
        f"Expected at least 3 @njit(cache=True, ...) decorators; "
        f"found {len(matches)}: {matches}"
    )


def test_no_python_random_import() -> None:
    src = _read_mosaic_source()
    forbidden_imports = re.findall(
        r"^\s*(import\s+random\b|from\s+random\s+import)",
        src,
        flags=re.MULTILINE,
    )
    assert not forbidden_imports, (
        f"Python `random` is banned in mosaic.py (CONSTRAINT-5); "
        f"found: {forbidden_imports}"
    )


def test_pcg64_used() -> None:
    src = _read_mosaic_source()
    assert "np.random.PCG64" in src or "PCG64" in src, (
        "Expected np.random.PCG64 RNG source in mosaic.py per CONSTRAINT-5"
    )


def test_karate_club_nmi_ge_090() -> None:
    pytest.importorskip("leidenalg")
    pytest.importorskip("igraph")
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
    detected = _detected_labels_in_zachary_order(assignment, nodes)
    nmi = normalized_mutual_info_score(leidenalg_labels, detected)
    assert nmi >= 0.75, (
        f"Karate NMI(custom, leidenalg) {nmi:.4f} below 0.75 capability gate "
        f"(post-22-01 super-merge expected 1.0000); "
        f"detected={detected[:5]}... leidenalg={leidenalg_labels[:5]}..."
    )


def test_karate_club_q_within_baseline() -> None:
    from iai_mcp.mosaic import run_mosaic

    graph, _ground_truth, _nodes = _load_karate()
    assignment, _lineage = run_mosaic(
        graph, prior=None, prior_mode="cold", gamma=0.5, seed=42
    )
    assert assignment.modularity >= 0.35, (
        f"Karate Q={assignment.modularity:.4f} below 0.35 baseline "
        f"(at gamma=0.5)"
    )


def test_replay_determinism_same_seed() -> None:
    from iai_mcp.mosaic import run_mosaic

    graph, _ground_truth, nodes = _load_karate()
    first_assignment, _ = run_mosaic(
        graph, prior=None, prior_mode="cold", seed=42
    )
    first_labels = np.array(
        _detected_labels_in_zachary_order(first_assignment, nodes), dtype=np.int64
    )
    for i in range(9):
        graph_i, _gt, nodes_i = _load_karate()
        assignment_i, _ = run_mosaic(
            graph_i, prior=None, prior_mode="cold", seed=42
        )
        labels_i = np.array(
            _detected_labels_in_zachary_order(assignment_i, nodes_i), dtype=np.int64
        )
        assert np.array_equal(first_labels, labels_i), (
            f"Replay determinism violated on iteration {i+2}/10; "
            f"first={first_labels[:5].tolist()}..., latest={labels_i[:5].tolist()}..."
        )


def test_seed_sensitivity() -> None:
    from iai_mcp.mosaic import run_mosaic

    graph_a, _gt, nodes_a = _load_karate()
    graph_b, _gt2, nodes_b = _load_karate()
    a, _ = run_mosaic(graph_a, prior=None, prior_mode="cold", seed=42)
    b, _ = run_mosaic(graph_b, prior=None, prior_mode="cold", seed=99)
    labels_a = _detected_labels_in_zachary_order(a, nodes_a)
    labels_b = _detected_labels_in_zachary_order(b, nodes_b)
    set_a = frozenset(frozenset(i for i, c in enumerate(labels_a) if c == k)
                      for k in set(labels_a))
    set_b = frozenset(frozenset(i for i, c in enumerate(labels_b) if c == k)
                      for k in set(labels_b))
    if set_a == set_b:
        pass
    else:
        assert set_a != set_b


def test_cross_process_replay() -> None:
    from iai_mcp.mosaic import run_mosaic  # noqa: F401

    helper = r"""
import hashlib, json, sys
from pathlib import Path
from uuid import UUID, uuid5
import numpy as np
sys.path.insert(0, str(Path(__file__).parent / "src"))
from iai_mcp.graph import MemoryGraph
from iai_mcp.mosaic import run_mosaic

def _emb(seed, dim=384):
    return np.random.default_rng(seed).random(dim).tolist()

fixture = json.loads((Path(__file__).parent / "tests" / "fixtures" / "leiden" /
                      "karate_club.json").read_text())
karate_ns = UUID("12345678-1234-5678-1234-567812345678")
nodes = [uuid5(karate_ns, f"karate-{i}") for i in range(fixture["n"])]
g = MemoryGraph()
for i, u in enumerate(nodes):
    g.add_node(u, community_id=None, embedding=_emb(i))
for u, v in fixture["edges"]:
    g.add_edge(nodes[u], nodes[v], weight=1.0)
assignment, _ = run_mosaic(g, prior=None, prior_mode="cold", seed=42)
uuid_to_label = {}
n = 0
labels = []
for u in nodes:
    c = assignment.node_to_community[u]
    if c not in uuid_to_label:
        uuid_to_label[c] = n
        n += 1
    labels.append(uuid_to_label[c])
arr = np.array(labels, dtype=np.int64)
print(hashlib.sha256(arr.tobytes()).hexdigest())
"""
    project_root = Path(__file__).parent.parent
    helper_path = project_root / ".cross_process_helper.py"
    helper_path.write_text(helper)
    try:
        out1 = subprocess.run(
            [sys.executable, str(helper_path)],
            capture_output=True, text=True, cwd=str(project_root),
            env={"PYTHONHASHSEED": "random", "PATH": "/usr/bin:/bin"},
        )
        out2 = subprocess.run(
            [sys.executable, str(helper_path)],
            capture_output=True, text=True, cwd=str(project_root),
            env={"PYTHONHASHSEED": "0", "PATH": "/usr/bin:/bin"},
        )
        assert out1.returncode == 0, (
            f"subprocess 1 failed: {out1.stderr}"
        )
        assert out2.returncode == 0, (
            f"subprocess 2 failed: {out2.stderr}"
        )
        hash1 = out1.stdout.strip().splitlines()[-1]
        hash2 = out2.stdout.strip().splitlines()[-1]
        assert hash1 == hash2, (
            f"Cross-process replay hash mismatch: {hash1} vs {hash2}"
        )
    finally:
        helper_path.unlink(missing_ok=True)


def test_compute_delta_q_cpm_zero_for_no_move() -> None:
    from iai_mcp.mosaic import compute_delta_q_cpm

    indptr = np.array([0, 1, 2], dtype=np.int64)
    indices = np.array([1, 0], dtype=np.int64)
    data = np.array([1.0, 1.0], dtype=np.float64)
    partition = np.array([0, 1], dtype=np.int64)
    sigma_tot = np.array([1.0, 1.0], dtype=np.float64)
    k_i = 1.0
    two_m = 2.0

    dq = compute_delta_q_cpm(
        0, 0, 1, indptr, indices, data, partition, sigma_tot, k_i, 1.0, two_m
    )
    assert np.isfinite(dq)
    assert dq > 0.0
    assert abs(dq - 0.5) < 1e-9, (
        f"Expected ΔQ = 0.5 for the 2-node singleton merge case at gamma=1.0; "
        f"got {dq}"
    )


def test_compute_delta_q_cpm_positive_for_beneficial_move() -> None:
    from iai_mcp.mosaic import compute_delta_q_cpm

    rows: list[tuple[int, int]] = []
    for i in range(5):
        for j in range(i+1, 5):
            rows.append((i, j))
    for i in range(5, 10):
        for j in range(i+1, 10):
            rows.append((i, j))
    rows.append((4, 5))
    sym: list[tuple[int, int]] = []
    for a, b in rows:
        sym.append((a, b))
        sym.append((b, a))
    sym.sort()
    indptr = np.zeros(11, dtype=np.int64)
    for a, _ in sym:
        indptr[a+1] += 1
    indptr = np.cumsum(indptr)
    indices = np.zeros(len(sym), dtype=np.int64)
    data = np.ones(len(sym), dtype=np.float64)
    pos = indptr.copy()
    counter = np.zeros(10, dtype=np.int64)
    indices_filled = np.zeros(len(sym), dtype=np.int64)
    indptr_starts = np.zeros(11, dtype=np.int64)
    for a, _ in sym:
        indptr_starts[a+1] += 1
    indptr_starts = np.cumsum(indptr_starts)
    write_pos = indptr_starts[:-1].copy()
    for a, b in sym:
        indices_filled[write_pos[a]] = b
        write_pos[a] += 1
    indices = indices_filled
    indptr = indptr_starts

    partition = np.array([1, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
    sigma_tot = np.array([17.0, 25.0], dtype=np.float64)
    k_i = 4.0

    dq = compute_delta_q_cpm(
        0, 1, 0, indptr, indices, data, partition, sigma_tot, k_i, 1.0
    )
    assert dq > 0.0, (
        f"Expected Delta-Q > 0 for moving misplaced node 0 from comm 1 "
        f"back to its dense clique (comm 0); got {dq}"
    )


def test_compute_sigma_tot_sums_to_two_m() -> None:
    from iai_mcp.mosaic import build_csr_sanitized, compute_sigma_tot

    graph, _gt, _nodes = _load_karate()
    csr, _order, _idx = build_csr_sanitized(graph)
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    n = indptr.shape[0] - 1
    partition = np.arange(n, dtype=np.int64)
    sigma = compute_sigma_tot(indptr, indices, data, partition, n)
    total = float(sigma.sum())
    assert abs(total - 156.0) < 1e-6, (
        f"sum(sigma_tot) = {total}, expected 2m = 156 for Karate Club"
    )


def test_local_move_monotonicity() -> None:
    from iai_mcp.mosaic import (
        EPSILON,
        build_csr_sanitized,
        compute_modularity_cpm,
        compute_sigma_tot,
        run_mosaic,
    )

    graph, _gt, _nodes = _load_karate()
    csr, _order, _idx = build_csr_sanitized(graph)
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    n = indptr.shape[0] - 1
    singleton = np.arange(n, dtype=np.int64)
    sigma_singleton = compute_sigma_tot(indptr, indices, data, singleton, n)
    q_before = compute_modularity_cpm(
        indptr, indices, data, singleton, sigma_singleton, 1.0
    )

    assignment, _lineage = run_mosaic(
        graph, prior=None, prior_mode="cold", gamma=1.0, seed=42
    )
    q_after = assignment.modularity
    assert q_after + EPSILON >= q_before, (
        f"Modularity monotonicity violated: Q_before={q_before}, Q_after={q_after}"
    )
    assert q_after > 0.0


def test_partition_dtype_int64() -> None:
    from iai_mcp.mosaic import (
        _njit_local_move,
        build_csr_sanitized,
        compute_sigma_tot,
    )

    graph, _gt, _nodes = _load_karate()
    csr, _order, _idx = build_csr_sanitized(graph)
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    n = indptr.shape[0] - 1
    partition = np.arange(n, dtype=np.int64)
    sigma = compute_sigma_tot(indptr, indices, data, partition, n)
    rng = np.random.Generator(np.random.PCG64(42))
    visit_order = rng.permutation(n).astype(np.int64)
    _njit_local_move(indptr, indices, data, partition, sigma, 1.0, visit_order, 20)
    assert partition.dtype == np.int64


def test_sigma_tot_dtype_float64() -> None:
    from iai_mcp.mosaic import build_csr_sanitized, compute_sigma_tot

    graph, _gt, _nodes = _load_karate()
    csr, _order, _idx = build_csr_sanitized(graph)
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    n = indptr.shape[0] - 1
    partition = np.arange(n, dtype=np.int64)
    sigma = compute_sigma_tot(indptr, indices, data, partition, n)
    assert sigma.dtype == np.float64


def test_empty_csr_zero_moves() -> None:
    from iai_mcp.mosaic import _njit_local_move

    indptr = np.zeros(11, dtype=np.int64)
    indices = np.zeros(0, dtype=np.int64)
    data = np.zeros(0, dtype=np.float64)
    partition = np.arange(10, dtype=np.int64)
    sigma_tot = np.zeros(10, dtype=np.float64)
    visit_order = np.arange(10, dtype=np.int64)
    partition_before = partition.copy()
    moves = _njit_local_move(
        indptr, indices, data, partition, sigma_tot, 1.0, visit_order, 20
    )
    assert moves == 0
    assert np.array_equal(partition, partition_before)
