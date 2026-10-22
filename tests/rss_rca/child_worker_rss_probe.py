"""Committed child-worker RSS decomposition probe.

Decomposes `runtime_graph_cache_worker._worker_entry`'s compute sequence
(`MemoryGraph.clear_and_rebuild` -> `detect_communities` ->
`centrality_for_runtime` -> `rich_club_nodes`) into per-stage RSS deltas,
answers whether the dominant fixed cost is one-time-per-process or per-call,
and anchors the in-process staged total against a real spawned child's own
peak RSS so the decomposition is not measured in a vacuum.

Three arms, each run in its OWN freshly-spawned subprocess (`ru_maxrss` is a
monotonic per-process high-water mark; a shared-process loop across arms
would contaminate later readings with earlier peaks):

  staged                -- four `RUSAGE_SELF` checkpoints across the
                            worker's own compute sequence (build / community
                            / centrality / rich-club), on a synthetic graph,
                            no store.
  one_time_vs_per_call  -- calls `detect_communities` TWICE on the same
                            graph in one process; the second call's
                            `RUSAGE_SELF` increment answers whether the
                            fixed cost is JIT/import residency (near-zero
                            second call) or per-call allocation.
  real_spawn            -- builds a real hermetic store, runs the actual
                            `_rebuild_and_save_rgc` DB->child rebuild, and
                            reads `RUSAGE_CHILDREN` before/after (macOS
                            aggregates a reaped child's `ru_maxrss` into the
                            parent's `RUSAGE_CHILDREN`) -- the anchor the
                            staged arm is checked against.

Full-scale sweeps (N=15000, N=30000, both arms) run in the background with
their output redirected to a file; they are not a per-commit gate. Example::

    python tests/rss_rca/child_worker_rss_probe.py --n 15000 30000 \\
        > bench/results/child-rss-sweep.log 2>&1 &

This is a measurement tool. It changes no production source. The `staged`
and `one_time_vs_per_call` arms never touch a store; `real_spawn` redirects
HOME/IAI_MCP_STORE under a tmp root before any `iai_mcp` import.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np

# Ensure the repo root is importable so ``tests.rss_rca.*`` resolves whether
# this script is run as a module or by absolute path.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_RESULT_SENTINEL = "CHILD_RSS_PROBE_RESULT_JSON "
_ARM_NAMES = ("staged", "one_time_vs_per_call", "real_spawn")
_CLUSTER_SIZE = 10  # records per synthetic cluster (bounded, mirrors corpus edges)
# Relative divergence between the in-process staged total and the real-spawn
# child delta above which the decomposition is flagged as unmoored rather
# than silently reported.
_DIVERGENCE_WARN_FRACTION = 0.5


# ---------------------------------------------------------------------------
# RSS reading -- ru_maxrss is bytes on macOS, KiB on Linux.
# ---------------------------------------------------------------------------


def _ru_maxrss_mib(raw: int) -> float:
    if sys.platform == "darwin":
        return raw / (1024.0 * 1024.0)
    return raw / 1024.0


def _self_peak_mib() -> float:
    return _ru_maxrss_mib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _children_peak_mib() -> float:
    return _ru_maxrss_mib(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)


# ---------------------------------------------------------------------------
# Synthetic graph inputs -- mirrors `_worker_entry`'s node/edge shape without
# a store, so the `staged` and `one_time_vs_per_call` arms measure the
# worker's own compute, not store/DB streaming cost.
# ---------------------------------------------------------------------------


def _synthetic_node_id(rng: "np.random.Generator") -> UUID:
    return UUID(bytes=rng.integers(0, 256, size=16, dtype=np.uint8).tobytes())


def _synthetic_nodes(
    n: int, seed: int, dim: int,
) -> list[tuple[UUID, None, "np.ndarray", dict]]:
    """Node tuples in the exact shape `_worker_entry` accumulates in `nodes_buf`."""
    rng = np.random.default_rng(seed)
    now = datetime.now(timezone.utc).isoformat()
    nodes: list[tuple[UUID, None, np.ndarray, dict]] = []
    for _ in range(n):
        vec = rng.standard_normal(dim).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        nodes.append((_synthetic_node_id(rng), None, vec, {"created_at": now}))
    return nodes


def _synthetic_edges(
    ids: list[UUID], edge_factor: float,
) -> list[tuple[UUID, UUID, float, str]]:
    """Ring + chord edges within bounded clusters (mirrors the corpus harness's
    bounded-cluster topology without opening a store)."""
    n = len(ids)
    target = int(round(edge_factor * n))
    pairs: list[tuple[UUID, UUID, float, str]] = []
    for start in range(0, n, _CLUSTER_SIZE):
        cluster = ids[start:start + _CLUSTER_SIZE]
        c = len(cluster)
        if c < 2:
            continue
        for i in range(c):
            pairs.append((cluster[i], cluster[(i + 1) % c], 0.1, "hebbian"))
        extra = max(0, int(round((edge_factor - 1.0) * c)))
        for k in range(extra):
            i = k % c
            j = (i + 2) % c
            if i != j:
                pairs.append((cluster[i], cluster[j], 0.1, "hebbian"))
        if len(pairs) >= target:
            break
    return pairs[:target] if len(pairs) > target else pairs


def _build_synthetic_graph(n: int, seed: int, dim: int, edge_factor: float):
    from iai_mcp.graph import MemoryGraph

    nodes = _synthetic_nodes(n, seed, dim)
    ids = [nid for nid, _parent, _emb, _meta in nodes]
    edges = _synthetic_edges(ids, edge_factor)
    graph = MemoryGraph()
    graph.clear_and_rebuild(nodes, edges)
    return graph


# ---------------------------------------------------------------------------
# Arm (i): staged in-process decomposition.
# ---------------------------------------------------------------------------


def _arm_staged(n: int, seed: int, dim: int, edge_factor: float) -> dict[str, Any]:
    from iai_mcp.centrality_approx import centrality_for_runtime
    from iai_mcp.community import detect_communities
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.richclub import rich_club_nodes

    baseline_mib = _self_peak_mib()

    nodes = _synthetic_nodes(n, seed, dim)
    ids = [nid for nid, _parent, _emb, _meta in nodes]
    edges = _synthetic_edges(ids, edge_factor)

    graph = MemoryGraph()
    graph.clear_and_rebuild(nodes, edges)
    after_build_mib = _self_peak_mib()

    assignment = detect_communities(graph, prior=None, prior_mode="cold")
    after_community_mib = _self_peak_mib()

    centrality = centrality_for_runtime(graph)
    after_centrality_mib = _self_peak_mib()

    rich_club_nodes(graph, centrality=centrality)
    after_rich_club_mib = _self_peak_mib()

    return {
        "arm": "staged",
        "n": n,
        "baseline_mib": round(baseline_mib, 2),
        "after_build_mib": round(after_build_mib, 2),
        "after_community_mib": round(after_community_mib, 2),
        "after_centrality_mib": round(after_centrality_mib, 2),
        "after_rich_club_mib": round(after_rich_club_mib, 2),
        "delta_build_mib": round(after_build_mib - baseline_mib, 2),
        "delta_community_mib": round(after_community_mib - after_build_mib, 2),
        "delta_centrality_mib": round(after_centrality_mib - after_community_mib, 2),
        "delta_rich_club_mib": round(after_rich_club_mib - after_centrality_mib, 2),
        "staged_total_delta_mib": round(after_rich_club_mib - baseline_mib, 2),
        "n_communities": len(set(assignment.node_to_community.values())),
    }


# ---------------------------------------------------------------------------
# Arm (ii): per-spawn-vs-one-time decider.
# ---------------------------------------------------------------------------


def _arm_one_time_vs_per_call(
    n: int, seed: int, dim: int, edge_factor: float,
) -> dict[str, Any]:
    from iai_mcp.community import detect_communities

    graph = _build_synthetic_graph(n, seed, dim, edge_factor)

    detect_communities(graph, prior=None, prior_mode="cold")
    after_first_mib = _self_peak_mib()
    detect_communities(graph, prior=None, prior_mode="cold")
    after_second_mib = _self_peak_mib()

    return {
        "arm": "one_time_vs_per_call",
        "n": n,
        "after_first_call_mib": round(after_first_mib, 2),
        "after_second_call_mib": round(after_second_mib, 2),
        "second_call_delta_mib": round(after_second_mib - after_first_mib, 2),
    }


# ---------------------------------------------------------------------------
# Arm (iii): real-spawn anchor.
# ---------------------------------------------------------------------------


def _arm_real_spawn(n: int, seed: int, edge_factor: float) -> dict[str, Any]:
    """Real hermetic store + `_rebuild_and_save_rgc`; `RUSAGE_CHILDREN` delta.

    Redirects HOME/IAI_MCP_STORE under a tmp root BEFORE any `iai_mcp` import
    (model on `phys_footprint_harness._worker_run`) so this arm never opens
    the real store.
    """
    tmp_root = Path(tempfile.mkdtemp(prefix="iai-child-rss-probe-"))
    prev_env = {
        k: os.environ.get(k)
        for k in (
            "HOME", "IAI_MCP_STORE", "IAI_MCP_USER_MODEL_PATH",
            "IAI_MCP_CRYPTO_PASSPHRASE", "PYTHON_KEYRING_BACKEND",
        )
    }
    os.environ["HOME"] = str(tmp_root)
    os.environ["IAI_MCP_STORE"] = str(tmp_root / ".iai-mcp" / "hippo")
    os.environ["IAI_MCP_USER_MODEL_PATH"] = str(
        tmp_root / ".iai-mcp" / "user_model.json"
    )
    if not prev_env["IAI_MCP_CRYPTO_PASSPHRASE"]:
        os.environ["IAI_MCP_CRYPTO_PASSPHRASE"] = (
            "iai-mcp-child-rss-probe-deterministic-2026"
        )
    os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
    (tmp_root / ".iai-mcp").mkdir(parents=True, exist_ok=True)

    store = None
    try:
        from iai_mcp import runtime_graph_cache
        from iai_mcp.store import MemoryStore
        from iai_mcp.store._buffers import flush_edge_buffer, flush_record_buffer

        from tests.rss_rca.corpus import build_corpus
        from tests.rss_rca.phys_footprint_harness import _build_synthetic_edges

        store_path = Path(os.environ["IAI_MCP_STORE"])
        store_path.mkdir(parents=True, exist_ok=True)
        store = MemoryStore(path=store_path)
        eff_dim = int(store.embed_dim)

        build_corpus(
            store, n=n, seed=seed, dim=eff_dim,
            varied_tags=True, with_hv_payload=False,
        )
        flush_record_buffer(store)
        _build_synthetic_edges(store, n, edge_factor, seed)
        flush_edge_buffer(store)

        before_children_mib = _children_peak_mib()
        result = runtime_graph_cache._rebuild_and_save_rgc(store)
        after_children_mib = _children_peak_mib()

        return {
            "arm": "real_spawn",
            "n": n,
            "rebuilt": bool(result.get("rebuilt")),
            "before_children_mib": round(before_children_mib, 2),
            "after_children_mib": round(after_children_mib, 2),
            "child_rss_delta_mib": round(after_children_mib - before_children_mib, 2),
        }
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:  # noqa: BLE001
                pass
        for k, v in prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(tmp_root, ignore_errors=True)


def _run_worker_arm(
    arm: str, n: int, seed: int, dim: int, edge_factor: float,
) -> dict[str, Any]:
    if arm == "staged":
        return _arm_staged(n, seed, dim, edge_factor)
    if arm == "one_time_vs_per_call":
        return _arm_one_time_vs_per_call(n, seed, dim, edge_factor)
    if arm == "real_spawn":
        return _arm_real_spawn(n, seed, edge_factor)
    raise ValueError(f"unknown arm: {arm!r}")


# ---------------------------------------------------------------------------
# Parent -- spawn one fresh subprocess per arm, aggregate, check divergence.
# ---------------------------------------------------------------------------


def _spawn_arm(
    arm: str, n: int, seed: int, dim: int, edge_factor: float, timeout_sec: float,
) -> dict[str, Any]:
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker-arm", arm,
        "--n", str(n), "--seed", str(seed),
        "--dim", str(dim), "--edge-factor", str(edge_factor),
    ]
    repo_root = Path(__file__).resolve().parents[2]
    try:
        proc = subprocess.run(
            cmd, cwd=str(repo_root), capture_output=True, text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "arm": arm, "n": n, "ok": False,
            "error": f"timed out after {timeout_sec}s",
            "stdout_tail": (exc.stdout or "")[-2000:] if exc.stdout else "",
        }
    for line in (proc.stdout or "").splitlines():
        if line.startswith(_RESULT_SENTINEL):
            try:
                return json.loads(line[len(_RESULT_SENTINEL):])
            except json.JSONDecodeError:
                break
    return {
        "arm": arm, "n": n, "ok": False,
        "error": f"no parseable result (rc={proc.returncode})",
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    }


def run_all_arms(
    n: int, *, seed: int = 42, dim: int = 384, edge_factor: float = 1.3,
    timeout_sec: float = 280.0,
) -> dict[str, Any]:
    """Spawn all three arms (each its own fresh subprocess) and aggregate.

    Returns each arm's result plus a `divergence` block comparing the staged
    in-process total against the real-spawn child delta.
    """
    staged = _spawn_arm("staged", n, seed, dim, edge_factor, timeout_sec)
    one_time = _spawn_arm(
        "one_time_vs_per_call", n, seed, dim, edge_factor, timeout_sec,
    )
    real_spawn = _spawn_arm("real_spawn", n, seed, dim, edge_factor, timeout_sec)

    divergence: dict[str, Any] = {"comparable": False, "warning": None}
    staged_total = staged.get("staged_total_delta_mib")
    real_delta = real_spawn.get("child_rss_delta_mib")
    if (
        isinstance(staged_total, (int, float))
        and isinstance(real_delta, (int, float))
        and real_delta > 0
    ):
        frac = abs(staged_total - real_delta) / real_delta
        divergence = {
            "comparable": True,
            "staged_total_delta_mib": staged_total,
            "real_spawn_child_delta_mib": real_delta,
            "relative_divergence": round(frac, 3),
            "threshold": _DIVERGENCE_WARN_FRACTION,
            "warning": None,
        }
        if frac > _DIVERGENCE_WARN_FRACTION:
            divergence["warning"] = (
                f"WARNING: staged in-process total ({staged_total:.1f} MiB) "
                f"diverges from the real-spawn child delta ({real_delta:.1f} "
                f"MiB) by {frac * 100:.0f}% (> "
                f"{_DIVERGENCE_WARN_FRACTION * 100:.0f}%) -- the in-process "
                "replication does not match the real spawn child; the "
                "decomposition above is suspect."
            )

    return {
        "n": n,
        "staged": staged,
        "one_time_vs_per_call": one_time,
        "real_spawn": real_spawn,
        "divergence": divergence,
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="child_worker_rss_probe",
        description=(
            "Decomposes the DB->child community-detection/centrality "
            "worker's peak RSS into per-stage deltas and anchors the "
            "in-process replication against a real spawned child."
        ),
    )
    parser.add_argument(
        "--n", type=int, nargs="+", default=[3000],
        help="record count(s) to probe; multiple values run sequentially",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--dim", type=int, default=384, help="embedding dimension")
    parser.add_argument(
        "--edge-factor", type=float, default=1.3,
        help="edges = round(edge_factor * N) (default 1.3)",
    )
    parser.add_argument(
        "--timeout-sec", type=float, default=280.0,
        help="per-arm subprocess timeout in seconds (default 280)",
    )
    # Worker-mode flag (set by the parent when spawning one arm's subprocess).
    parser.add_argument(
        "--worker-arm", choices=_ARM_NAMES, default=None, help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    if args.worker_arm is not None:
        n = args.n[0]
        try:
            result = _run_worker_arm(
                args.worker_arm, n, args.seed, args.dim, args.edge_factor,
            )
            result["ok"] = True
        except Exception as exc:  # noqa: BLE001 -- report failure as a result line
            result = {"arm": args.worker_arm, "n": n, "ok": False, "error": repr(exc)}
        sys.stdout.write(_RESULT_SENTINEL + json.dumps(result, default=str) + "\n")
        sys.stdout.flush()
        return 0 if result.get("ok") else 1

    for n in args.n:
        print(f"\n[child-worker RSS probe] N={n}")
        t0 = time.perf_counter()
        summary = run_all_arms(
            n, seed=args.seed, dim=args.dim, edge_factor=args.edge_factor,
            timeout_sec=args.timeout_sec,
        )
        wall = time.perf_counter() - t0
        print(json.dumps(summary, indent=2, default=str))
        warning = summary["divergence"].get("warning")
        if warning:
            print(warning)
        print(f"[child-worker RSS probe] N={n} completed in {wall:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
