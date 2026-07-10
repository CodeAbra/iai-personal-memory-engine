"""Unit tests for the spawn-context graph-rebuild worker.

The tests drive `_worker_entry` directly with an in-process Pipe end to verify
the recv-then-compute-then-stream protocol without paying the spawn cost.
The import-surface guard test (`test_worker_module_does_not_import_aes_surface`)
spawns a clean subprocess and asserts the AES-key-carrying modules are absent
from `sys.modules` even after the worker's unconditional late imports execute.
"""
from __future__ import annotations

import multiprocessing as mp
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp import runtime_graph_cache_worker
from iai_mcp.community import detect_communities
from iai_mcp.graph import MemoryGraph
from iai_mcp.richclub import rich_club_nodes


_DIM = 384


def _emb_bytes(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(_DIM).astype(np.float32)
    return vec.tobytes()


def _seed_minimal(n: int = 5, seed: int = 7) -> tuple[list[tuple[str, bytes]], list[tuple[str, str, float]]]:
    ids = [uuid4() for _ in range(n)]
    nodes = [(str(uid), _emb_bytes(seed + i)) for i, uid in enumerate(ids)]
    edges = [
        (str(ids[i]), str(ids[(i + 1) % n]), 1.0 + i * 0.1)
        for i in range(n - 1)
    ]
    return nodes, edges


def _drive_worker(node_chunks, edge_chunks):
    """Run `_worker_entry` against an in-process pipe, return the messages drained.

    The worker runs in a daemon thread (same process, so the abort/raise path
    is observable). Sends and drains happen concurrently so neither side
    blocks on a full kernel pipe buffer.
    """
    parent_conn, child_conn = mp.Pipe(duplex=True)
    worker_error: list = []
    messages: list = []

    def _run_worker():
        try:
            runtime_graph_cache_worker._worker_entry(child_conn)
        except BaseException as exc:  # noqa: BLE001
            worker_error.append(exc)

    def _drain():
        while True:
            try:
                if not parent_conn.poll(1.0):
                    if not th.is_alive():
                        # Worker finished; drain anything left.
                        while parent_conn.poll(0.1):
                            try:
                                messages.append(parent_conn.recv())
                            except EOFError:
                                return
                        return
                    continue
                msg = parent_conn.recv()
                messages.append(msg)
                if msg[0] == "done":
                    return
            except (EOFError, OSError):
                return

    th = threading.Thread(target=_run_worker, daemon=True)
    drain_th = threading.Thread(target=_drain, daemon=True)
    th.start()
    drain_th.start()

    for chunk in node_chunks:
        parent_conn.send(("nodes", chunk))
    parent_conn.send(("nodes_end", None))
    for chunk in edge_chunks:
        parent_conn.send(("edges", chunk))
    parent_conn.send(("edges_end", None))

    th.join(timeout=120.0)
    assert not th.is_alive(), "worker thread did not exit within 120s"
    drain_th.join(timeout=10.0)
    if worker_error:
        raise worker_error[0]

    parent_conn.close()
    return messages


def _build_reference(node_chunks, edge_chunks):
    nodes = []
    for chunk in node_chunks:
        for id_str, emb_blob in chunk:
            vec = np.frombuffer(emb_blob, dtype=np.float32).tolist()
            nodes.append((UUID(id_str), None, vec, {}))
    edges = []
    for chunk in edge_chunks:
        for src_str, dst_str, weight in chunk:
            edges.append((UUID(src_str), UUID(dst_str), float(weight), "hebbian"))
    graph = MemoryGraph()
    graph.clear_and_rebuild(nodes, edges)
    assignment = detect_communities(graph, prior=None, prior_mode="cold")
    rc = rich_club_nodes(graph)
    max_degree = max((d for _, d in graph.degrees()), default=0)
    return assignment, rc, max_degree


def _decode_result(messages):
    """Reassemble a streamed worker result from drained messages."""
    community_table_uuids: list[UUID] = []
    community_centroids: dict[UUID, list[float] | None] = {}
    assignments: dict[UUID, int] = {}
    backend: str | None = None
    top_communities: list[UUID] = []
    mid_regions: dict[UUID, list[UUID]] = {}
    rich_club: list[UUID] = []
    max_degree: int | None = None
    done = False
    for kind, payload in messages:
        if kind == "community_table":
            for comm_bytes, centroid_bytes in payload:
                cu = UUID(bytes=comm_bytes)
                community_table_uuids.append(cu)
                if centroid_bytes is None:
                    community_centroids[cu] = None
                else:
                    community_centroids[cu] = np.frombuffer(
                        centroid_bytes, dtype=np.float32
                    ).tolist()
        elif kind == "assign":
            for node_bytes, comm_idx in payload:
                assignments[UUID(bytes=node_bytes)] = int(comm_idx)
        elif kind == "assign_end":
            pass
        elif kind == "backend":
            backend = str(payload)
        elif kind == "top_communities":
            top_communities = [UUID(bytes=b) for b in payload]
        elif kind == "mid_regions":
            for comm_bytes, member_bytes_list in payload:
                mid_regions[UUID(bytes=comm_bytes)] = [
                    UUID(bytes=mb) for mb in member_bytes_list
                ]
        elif kind == "rich_club":
            rich_club = [UUID(bytes=b) for b in payload]
        elif kind == "max_degree":
            max_degree = int(payload)
        elif kind == "done":
            done = True
        elif kind == "error":
            raise RuntimeError(f"worker reported error: {payload!r}")
    node_to_community: dict[UUID, UUID] = {}
    for node_uuid, idx in assignments.items():
        node_to_community[node_uuid] = community_table_uuids[idx]
    return {
        "node_to_community": node_to_community,
        "community_centroids": community_centroids,
        "backend": backend,
        "top_communities": top_communities,
        "mid_regions": mid_regions,
        "rich_club": rich_club,
        "max_degree": max_degree,
        "done": done,
    }


def _partition(node_to_community: dict) -> frozenset:
    groups: dict = {}
    for node_id, community_id in node_to_community.items():
        groups.setdefault(community_id, set()).add(str(node_id))
    return frozenset(frozenset(members) for members in groups.values())


def test_worker_round_trip_minimal():
    nodes, edges = _seed_minimal(n=5, seed=11)
    messages = _drive_worker([nodes], [edges])

    result = _decode_result(messages)
    assert result["done"] is True
    assert result["backend"] is not None

    ref_assignment, ref_rc, ref_max_degree = _build_reference([nodes], [edges])
    assert _partition(result["node_to_community"]) == _partition(
        ref_assignment.node_to_community
    )
    assert frozenset(result["rich_club"]) == frozenset(ref_rc)
    assert result["max_degree"] == ref_max_degree
    assert result["backend"] == ref_assignment.backend


def test_worker_empty_graph_round_trip():
    messages = _drive_worker([[]], [[]])

    result = _decode_result(messages)
    assert result["done"] is True
    assert result["backend"] == "flat"
    assert result["rich_club"] == []
    assert result["max_degree"] == 0
    assert result["node_to_community"] == {}


def test_worker_aborts_on_unknown_kind():
    parent_conn, child_conn = mp.Pipe(duplex=True)
    parent_conn.send(("garbage", None))
    with pytest.raises(RuntimeError, match="unknown chunk kind"):
        runtime_graph_cache_worker._worker_entry(child_conn)
    # Drain any error message the worker sent before raising.
    if parent_conn.poll(0.1):
        kind, payload = parent_conn.recv()
        assert kind == "error"
        assert "unknown chunk kind" in str(payload)
    parent_conn.close()


def test_worker_chunked_assignment_envelope():
    n = 5000
    ids = [uuid4() for _ in range(n)]
    rng_seed = 3
    node_chunks = []
    chunk = []
    for i, uid in enumerate(ids):
        chunk.append((str(uid), _emb_bytes(rng_seed + i)))
        if len(chunk) == 1000:
            node_chunks.append(chunk)
            chunk = []
    if chunk:
        node_chunks.append(chunk)
    # A linear chain so detection isn't trivial.
    edges = [(str(ids[i]), str(ids[i + 1]), 1.0) for i in range(n - 1)]
    messages = _drive_worker(node_chunks, [edges])

    assign_msgs = [m for m in messages if m[0] == "assign"]
    assert assign_msgs, "worker must emit at least one assign chunk"
    seen_nodes: set[UUID] = set()
    for _, payload in assign_msgs:
        assert len(payload) <= 2000, (
            f"assign chunk too large: {len(payload)} > 2000"
        )
        for node_bytes, _idx in payload:
            uid = UUID(bytes=node_bytes)
            assert uid not in seen_nodes, "duplicate node in assign stream"
            seen_nodes.add(uid)
    assert seen_nodes == set(ids), "assign stream must cover every node exactly once"


_GUARD_SCRIPT = textwrap.dedent(
    """
    import sys
    import multiprocessing as mp
    from iai_mcp import runtime_graph_cache_worker

    parent_conn, child_conn = mp.Pipe(duplex=True)
    parent_conn.send(("abort", None))
    runtime_graph_cache_worker._worker_entry(child_conn)

    forbidden = ("iai_mcp.hippo", "iai_mcp.store", "iai_mcp.daemon", "iai_mcp.crypto")
    offenders = [m for m in sys.modules if any(f in m for f in forbidden)]
    if offenders:
        print("OFFENDERS:" + ",".join(sorted(offenders)))
        sys.exit(2)
    sys.exit(0)
    """
).strip()


def test_worker_module_does_not_import_aes_surface(tmp_path: Path):
    proc = subprocess.run(
        [sys.executable, "-c", _GUARD_SCRIPT],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"AES-fence violated. exitcode={proc.returncode}\n"
        f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )


_COMMUNITY_GUARD_SCRIPT = textwrap.dedent(
    """
    import sys
    import multiprocessing as mp
    from iai_mcp import runtime_graph_cache_worker

    parent_conn, child_conn = mp.Pipe(duplex=True)
    parent_conn.send(("abort", None))
    runtime_graph_cache_worker._community_only_worker_entry(child_conn)

    forbidden = ("iai_mcp.hippo", "iai_mcp.store", "iai_mcp.daemon", "iai_mcp.crypto")
    offenders = [m for m in sys.modules if any(f in m for f in forbidden)]
    if offenders:
        print("OFFENDERS:" + ",".join(sorted(offenders)))
        sys.exit(2)
    sys.exit(0)
    """
).strip()


def test_community_only_worker_does_not_import_aes_surface(tmp_path: Path):
    proc = subprocess.run(
        [sys.executable, "-c", _COMMUNITY_GUARD_SCRIPT],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"AES-fence violated. exitcode={proc.returncode}\n"
        f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
