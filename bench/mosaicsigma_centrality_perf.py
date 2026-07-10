from __future__ import annotations

import json
import os
import pathlib
import statistics
import sys
import time
from uuid import UUID

import numpy as np


_SRC_PATH = str(pathlib.Path(__file__).resolve().parent.parent / "src")
_ROOT_PATH = str(pathlib.Path(__file__).resolve().parent.parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)

from iai_mcp.graph import MemoryGraph  # noqa: E402


ITERATIONS = 30
SCALES = [5000, 10000]
P50_THRESHOLD_MS = 30.0
P95_THRESHOLD_MS = 80.0

SNAPSHOT_PATH = pathlib.Path(
    "bench/snapshots/live_n2000.npz"
)
DECISION_JSON_PATH = pathlib.Path(
    ".planning/phases/"
    "50-mosaicsigma-networkx-custom-rust-pyo3-graph-engine-lilli-gra/"
    "50-10-perf-gate.json"
)


def _build_memory_graph_from_edges(
    n_nodes: int, edges: list[tuple[int, int, float]]
) -> MemoryGraph:
    ids = [UUID(int=i) for i in range(n_nodes)]
    mg = MemoryGraph()
    for nid in ids:
        mg.add_node(nid, community_id=None, embedding=[0.0] * 384)
    for u, v, w in edges:
        if u == v:
            continue
        mg.add_edge(ids[u], ids[v], weight=float(w))
    return mg


def materialize_graph_at_scale(n_target: int) -> tuple[MemoryGraph, bool]:
    if not SNAPSHOT_PATH.exists():
        print(
            f"WARNING: live snapshot missing at {SNAPSHOT_PATH}; "
            f"falling back to synthetic gnm baseline at N={n_target}",
            file=sys.stderr,
        )
        import networkx as nx
        gr = nx.gnm_random_graph(n_target, n_target * 2, seed=42)
        edges = [(u, v, 1.0) for u, v in gr.edges()]
        return _build_memory_graph_from_edges(n_target, edges), True

    data = np.load(SNAPSHOT_PATH, allow_pickle=False)
    base_indptr = data["indptr"].astype(np.int64)
    base_indices = data["indices"].astype(np.int64)
    base_data = data["data"].astype(np.float64) if "data" in data.files else None
    base_n = len(base_indptr) - 1

    base_edges: list[tuple[int, int, float]] = []
    for u in range(base_n):
        s, e = int(base_indptr[u]), int(base_indptr[u + 1])
        for k in range(s, e):
            v = int(base_indices[k])
            if u >= v:
                continue
            w = float(base_data[k]) if base_data is not None else 1.0
            base_edges.append((u, v, w))

    if n_target <= base_n:
        return _build_memory_graph_from_edges(n_target, base_edges[: 2 * n_target]), False

    import math
    k_tiles = math.ceil(n_target / base_n)
    tiled_edges: list[tuple[int, int, float]] = []
    for tile_idx in range(k_tiles):
        offset = tile_idx * base_n
        for u, v, w in base_edges:
            u2 = u + offset
            v2 = v + offset
            if u2 >= n_target or v2 >= n_target:
                continue
            tiled_edges.append((u2, v2, w))
    return _build_memory_graph_from_edges(n_target, tiled_edges), False


def measure_p50_p95(mg: MemoryGraph, iterations: int) -> tuple[float, float]:
    os.environ["IAI_MCP_CENTRALITY_CACHE"] = "off"
    for _ in range(3):
        mg.centrality()
    latencies_ms: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        mg.centrality()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    p50 = float(statistics.median(latencies_ms))
    p95 = float(np.percentile(latencies_ms, 95))
    return p50, p95


def main() -> int:
    results: dict[str, dict] = {}
    any_fallback = False
    for n in SCALES:
        mg, used_fallback = materialize_graph_at_scale(n)
        if used_fallback:
            any_fallback = True
        p50, p95 = measure_p50_p95(mg, ITERATIONS)
        passed = (p50 <= P50_THRESHOLD_MS) and (p95 <= P95_THRESHOLD_MS)
        results[f"n_{n}"] = {
            "n_nodes": n,
            "p50_ms": p50,
            "p95_ms": p95,
            "pass": passed,
            "used_synthetic_fallback": used_fallback,
        }
    overall_pass_thresholds = all(r["pass"] for r in results.values())

    if any_fallback:
        action = "keep"
        reason = (
            "synthetic gnm fallback used at one or more scales "
            "(live snapshot missing); cannot decide drop because the "
            "synthetic Erdos-Renyi baseline is uniformly sparse and "
            "real memory graphs are heavy-tailed. Conservative default."
        )
    elif overall_pass_thresholds:
        action = "drop"
        reason = (
            f"real-graph perf PASS at both scales "
            f"(p50<={P50_THRESHOLD_MS}ms, p95<={P95_THRESHOLD_MS}ms)"
        )
    else:
        action = "keep"
        reason = (
            "real-graph perf threshold FAIL — cache stays on as the "
            "conservative default"
        )

    decision = {
        "schema_version": "1",
        "results": results,
        "thresholds": {
            "p50_ms": P50_THRESHOLD_MS,
            "p95_ms": P95_THRESHOLD_MS,
        },
        "any_fallback_used": any_fallback,
        "overall_pass_thresholds": overall_pass_thresholds,
        "action": action,
        "reason": reason,
    }

    DECISION_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    DECISION_JSON_PATH.write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
