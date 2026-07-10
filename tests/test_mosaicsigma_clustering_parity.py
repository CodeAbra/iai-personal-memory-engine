from __future__ import annotations

import json
import pathlib
import subprocess

import pytest


def _native_available() -> bool:
    try:
        import iai_mcp_native  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _native_available(), reason="iai_mcp_native wheel not installed"
)

pytest.importorskip("networkx")
pytest.importorskip("numpy")

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

FIXTURE_PATH = (
    pathlib.Path(__file__).parent / "fixtures" / "sigma_baseline.json"
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_fixtures() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["fixtures"]


def _edges_to_csr(n_nodes: int, edges: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    adj: list[set[int]] = [set() for _ in range(n_nodes)]
    for u, v in edges:
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    indices_list: list[int] = []
    for u in range(n_nodes):
        sorted_nbrs = sorted(adj[u])
        indices_list.extend(sorted_nbrs)
        indptr[u + 1] = indptr[u] + len(sorted_nbrs)
    indices = np.asarray(indices_list, dtype=np.int64)
    return indptr, indices


def _nx_graph(n_nodes: int, edges: list[tuple[int, int]]) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    g.add_edges_from((u, v) for u, v in edges if u != v)
    return g


def _ours(n_nodes: int, edges: list[tuple[int, int]]) -> float:
    from iai_mcp_native import graph
    indptr, indices = _edges_to_csr(n_nodes, edges)
    return float(graph.average_clustering(indptr, indices, n_nodes))


REFERENCE_FIXTURES = [
    "karate",
    "les_miserables",
    "er_200",
    "er_500",
    "er_1000",
    "tiny_10_ws_k4",
    "tiny_20_ws_p010",
    "ws_2500_k4_p0",
]


def test_local_clustering_matches_networkx_on_fixtures() -> None:
    fixtures = _load_fixtures()
    failures: list[tuple[str, float, float, float]] = []
    for name in REFERENCE_FIXTURES:
        f = fixtures[name]
        n = int(f["n"])
        edges = [(int(u), int(v)) for u, v in f["edges"]]
        g_nx = _nx_graph(n, edges)
        oracle = float(nx.average_clustering(g_nx))
        ours = _ours(n, edges)
        delta = abs(ours - oracle)
        if delta > 1e-9:
            failures.append((name, ours, oracle, delta))
    assert not failures, (
        f"{len(failures)} clustering parity failures:\n"
        + "\n".join(
            f"  {name}: ours={ours:.15g} oracle={oracle:.15g} |delta|={delta:.3e}"
            for name, ours, oracle, delta in failures
        )
    )


def test_global_transitivity_not_used_in_source() -> None:
    target_dir = REPO_ROOT / "rust" / "iai_mcp_graph_core" / "src"
    result = subprocess.run(
        ["grep", "-rE", "rustworkx_core::transitivity", str(target_dir)],
        check=False,
        capture_output=True,
    )
    assert result.returncode == 1, (
        "transitivity import found in graph_core source -- forbidden. "
        "grep output: "
        + result.stdout.decode(errors="replace")[:500]
    )


def test_sanity_band_ring_lattice_k4_nonzero_clustering() -> None:
    fixtures = _load_fixtures()
    f = fixtures["tiny_10_ws_k4"]
    n = int(f["n"])
    edges = [(int(u), int(v)) for u, v in f["edges"]]
    ours = _ours(n, edges)
    assert abs(ours - 0.5) <= 1e-9, (
        f"WS(10, k=4, p=0) sanity band: expected 0.5, got {ours!r}"
    )


def test_sanity_band_complete_graph_unity_clustering() -> None:
    n = 5
    edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    ours = _ours(n, edges)
    assert ours == 1.0, (
        f"K5 complete-graph clustering: expected exactly 1.0, got {ours!r}"
    )


def test_disconnected_graph_clustering_well_defined() -> None:
    edges = [
        (0, 1), (0, 2), (1, 2),
        (3, 4), (3, 5), (4, 5),
    ]
    n = 6
    oracle = float(nx.average_clustering(_nx_graph(n, edges)))
    ours = _ours(n, edges)
    assert oracle == 1.0
    assert ours == 1.0, (
        f"Disjoint K3s clustering: expected exactly 1.0, got {ours!r}"
    )


def test_isolated_and_degree_one_nodes_contribute_zero() -> None:
    edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
    n = 6
    oracle = float(nx.average_clustering(_nx_graph(n, edges)))
    ours = _ours(n, edges)
    assert oracle == 0.0
    assert ours == 0.0, (
        f"Star+isolate clustering: expected exactly 0.0, got {ours!r}"
    )
