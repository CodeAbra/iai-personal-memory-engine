from __future__ import annotations

import json
import logging
import math
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from cryptography.exceptions import InvalidTag

logger = logging.getLogger(__name__)

from iai_mcp.crypto import (
    CryptoKey,
    decrypt_field,
    encrypt_field,
    is_encrypted,
)
from iai_mcp.types import SCHEMA_VERSION_CURRENT

preload_ready: threading.Event = threading.Event()

# Guards lazy installation of each store's structural-decode single-flight
# lock (see load_recall_structural), so two racing threads never install two
# different locks on the same store instance.
_STRUCTURAL_DECODE_LOCK_GUARD = threading.Lock()

rebuild_ready: threading.Event = threading.Event()


CACHE_VERSION: str = "62-02-v8"

_STALENESS_WINDOW: int = 10


def _quantize_count(count: int) -> int:
    """Quantize a corpus count for the cache key: fixed 10-wide steps below
    200, then geometric ~5%-wide buckets (monotone across the seam).

    A FIXED step at production scale flips the key every few ambient captures,
    and every flip re-keys the warm bundle and spawns a full-corpus background
    refresh — the daemon burns a core rebuilding a graph nobody asked to
    change. Geometric buckets keep the flip rate proportional to how much the
    corpus actually changed at any scale.
    """
    if count < 0:
        return count
    if count < 200:
        return count // _STALENESS_WINDOW
    return 20 + int(math.log(count / 200.0) / math.log(1.05))
LEGACY_CACHE_VERSION_PLAINTEXT: str = "06-02-v1"

_CACHE_AAD: bytes = b"runtime-graph-cache:v3"

CACHE_FILENAME: str = "runtime_graph_cache.json"


_FUSE_MAX_AGE_SECONDS: float = 25.0 * 3600.0

_FUSE_DIRTY_THRESHOLD: int = 50


def _fuse_dirty_threshold(store: Any) -> int:
    """Corpus-scaled dirty fuse: 50 writes is a rewrite at test scale but a
    normal hour of ambient churn at 37k records — a fixed fuse there declares
    the warm bundle stale over and over for a corpus that barely moved. ~2% of
    the corpus, floored at the base constant so small corpora keep
    byte-identical fuse behavior."""
    try:
        _cc = getattr(store, "_corpus_count_cache", None)
        n = _cc.get("active") if _cc is not None else None
        if n is None:
            n = int(store.active_records_count())
        return max(_FUSE_DIRTY_THRESHOLD, int(n) // 50)
    except Exception:  # noqa: BLE001 -- fuse sizing must never break a read
        return _FUSE_DIRTY_THRESHOLD


_dirty_counter: int = 0
_DIRTY_COUNTER_LOCK = threading.Lock()


def increment_dirty_counter() -> None:
    global _dirty_counter  # noqa: PLW0603
    with _DIRTY_COUNTER_LOCK:
        _dirty_counter += 1


def reset_dirty_counter() -> None:
    global _dirty_counter  # noqa: PLW0603
    with _DIRTY_COUNTER_LOCK:
        _dirty_counter = 0


def get_dirty_counter() -> int:
    with _DIRTY_COUNTER_LOCK:
        return _dirty_counter


# One shared graph instance reused across refreshes so the allocator footprint
# stays bounded (a fresh instance per cycle fragments the heap arenas). The lock
# serializes concurrent refreshes so adjacency cannot be corrupted mid-rebuild.
_persistent_graph = None
_PERSISTENT_GRAPH_LOCK = threading.Lock()


def _get_persistent_graph():
    """Module-level reusable graph instance.

    No longer fed by the periodic rebuild path — the rebuild runs in a child
    process and reclaims its own address space. Kept callable for
    backward-compatibility with existing fixture-reset tests and any future
    in-parent graph consumers.
    """
    global _persistent_graph  # noqa: PLW0603
    if _persistent_graph is None:
        from iai_mcp.graph import MemoryGraph
        _persistent_graph = MemoryGraph()
    return _persistent_graph


# Worker timeouts. The rebuild itself is a background sleep-time operation
# and recall is served from the last-good snapshot throughout; the watchdog
# exists to catch a hung worker, not a slow-but-progressing one. The
# centrality + rich_club + community-detection compute scales super-linearly
# with graph size, so we use a base allowance plus a per-1k-nodes ramp:
#   timeout = base + per_1k * (active_records_count / 1000)
# capped at WORKER_TIMEOUT_MAX_S. First spawn after daemon boot uses a
# slightly larger base to absorb numba JIT cold-start. All reads and writes
# of `_first_spawn_seen` happen under `_PERSISTENT_GRAPH_LOCK`, which
# already serializes rebuilds — no new lock.
_WORKER_TIMEOUT_BASE_S: float = 60.0
_WORKER_TIMEOUT_FIRST_BASE_S: float = 120.0
# Coefficient calibrated to the measured per-1k-nodes cost of
# `MemoryGraph.centrality()` (Brandes betweenness, O(V*E)) on this hardware
# plus `detect_communities` overhead. Capped at WORKER_TIMEOUT_MAX_S so a
# truly hung worker is still caught in finite time.
_WORKER_TIMEOUT_PER_1K_NODES_S: float = 35.0
_WORKER_TIMEOUT_MAX_S: float = 3600.0
_first_spawn_seen: bool = False


_STREAM_CHUNK: int = 2000


class WorkerCrashedError(RuntimeError):
    """Child worker exited with a non-zero exit code."""


class WorkerTimeoutError(RuntimeError):
    """Child worker did not produce a complete result within the timeout."""


def _worker_entry_indirection(conn) -> None:
    """Picklable spawn target.

    The worker module is imported only inside the child after spawn, so the
    parent process itself never loads the worker module until spawn time.
    """
    from iai_mcp.runtime_graph_cache_worker import _worker_entry
    _worker_entry(conn)


def _community_only_worker_indirection(conn) -> None:
    """Picklable spawn target for the community-only worker.

    Imports the worker module only inside the child after spawn, so the parent
    never loads it (and never loads numba via that path).
    """
    from iai_mcp.runtime_graph_cache_worker import _community_only_worker_entry
    _community_only_worker_entry(conn)


def _resolve_timeout(active_records_count: int = 0) -> float:
    """Size-scaled watchdog timeout.

    Base plus a per-1k-active-records ramp, capped at
    `_WORKER_TIMEOUT_MAX_S`. First spawn after daemon boot uses a larger
    base to absorb numba JIT cold-start.
    """
    base = _WORKER_TIMEOUT_FIRST_BASE_S if not _first_spawn_seen else _WORKER_TIMEOUT_BASE_S
    ramp = _WORKER_TIMEOUT_PER_1K_NODES_S * (max(0, active_records_count) / 1000.0)
    return min(base + ramp, _WORKER_TIMEOUT_MAX_S)


def _graph_node_count(graph) -> int:
    """Node count for any graph object, independent of its concrete type.

    The runtime graph exposes `node_count()`; other graph objects that may reach
    the timeout sizing (degraded bundles built outside the runtime path) expose
    `number_of_nodes()` or only a length-able node view. Probe in that order so
    the timeout sizing never hard-depends on a single graph implementation.
    """
    node_count = getattr(graph, "node_count", None)
    if callable(node_count):
        return int(node_count())
    number_of_nodes = getattr(graph, "number_of_nodes", None)
    if callable(number_of_nodes):
        return int(number_of_nodes())
    return int(len(graph.nodes()))


def _terminate_worker(process) -> None:
    """Idempotent terminate-then-kill of the worker process."""
    try:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=2.0)
    except Exception:  # noqa: BLE001 -- worker cleanup must not raise
        pass


def _drain_worker_result(parent_conn, timeout: float) -> dict:
    """Drain the chunked compact result envelope into a parent-side dict.

    Raises WorkerTimeoutError if the worker has not emitted the `done`
    terminator within `timeout` seconds. Any `error` envelope is converted
    into a RuntimeError so the caller can dispose.
    """
    import time
    from uuid import UUID

    import numpy as np

    deadline = time.perf_counter() + timeout
    community_table_uuids: list = []
    community_centroids: dict = {}
    assignments: dict = {}
    backend: str | None = None
    top_communities: list = []
    mid_regions: dict = {}
    modularity: float = 0.0
    rich_club: list = []
    max_degree: int = 0
    node_degrees: dict = {}
    done = False

    while not done:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise WorkerTimeoutError(
                f"worker did not complete within {timeout:.1f}s"
            )
        if not parent_conn.poll(min(remaining, 1.0)):
            continue
        envelope = parent_conn.recv()
        kind, payload = envelope
        if kind == "community_table":
            for comm_bytes, centroid_bytes in payload:
                cu = UUID(bytes=comm_bytes)
                community_table_uuids.append(cu)
                if centroid_bytes is None:
                    community_centroids[cu] = []
                else:
                    community_centroids[cu] = np.frombuffer(
                        centroid_bytes, dtype=np.float32
                    ).tolist()
        elif kind == "assign":
            for node_bytes, comm_idx in payload:
                assignments[UUID(bytes=node_bytes)] = int(comm_idx)
        elif kind == "assign_end":
            continue
        elif kind == "backend":
            backend = str(payload)
        elif kind == "top_communities":
            top_communities = [UUID(bytes=b) for b in payload]
        elif kind == "mid_regions":
            for comm_bytes, member_bytes_list in payload:
                mid_regions[UUID(bytes=comm_bytes)] = [
                    UUID(bytes=mb) for mb in member_bytes_list
                ]
        elif kind == "modularity":
            modularity = float(payload)
        elif kind == "rich_club":
            rich_club = [UUID(bytes=b) for b in payload]
        elif kind == "max_degree":
            max_degree = int(payload)
        elif kind == "node_degrees":
            # Payload: [(node_uuid_bytes, int_degree), ...]
            for node_bytes, deg_val in payload:
                try:
                    node_degrees[UUID(bytes=node_bytes)] = int(deg_val)
                except (ValueError, TypeError):
                    pass
        elif kind == "done":
            done = True
        elif kind == "error":
            raise RuntimeError(f"worker reported error: {payload!r}")
        else:
            raise RuntimeError(f"worker emitted unknown envelope kind: {kind!r}")

    node_to_community: dict = {}
    for node_uuid, idx in assignments.items():
        node_to_community[node_uuid] = community_table_uuids[idx]

    return {
        "node_to_community": node_to_community,
        "community_centroids": community_centroids,
        "backend": backend if backend is not None else "flat",
        "top_communities": top_communities,
        "mid_regions": mid_regions,
        "modularity": modularity,
        "rich_club": rich_club,
        "max_degree": max_degree,
        "node_degrees": node_degrees,
    }


def _drain_community_only_result(parent_conn, timeout: float) -> dict:
    """Drain the community-only result envelope into a parent-side dict.

    A trimmed `_drain_worker_result`: it handles the community/assign/backend/
    top/mid envelopes and drops the `rich_club` / `max_degree` branches, which
    the community-only worker never sends. It also accepts the optional
    `("centrality", ...)` chunks the worker streams when centrality was
    requested, reassembling them into a `node_uuid -> float` map. Raises
    WorkerTimeoutError on a hung worker; any `error` envelope becomes a
    RuntimeError.
    """
    import time
    from uuid import UUID

    import numpy as np

    deadline = time.perf_counter() + timeout
    community_table_uuids: list = []
    community_centroids: dict = {}
    assignments: dict = {}
    backend: str | None = None
    top_communities: list = []
    mid_regions: dict = {}
    modularity: float = 0.0
    centrality: dict = {}
    done = False

    while not done:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise WorkerTimeoutError(
                f"worker did not complete within {timeout:.1f}s"
            )
        if not parent_conn.poll(min(remaining, 1.0)):
            continue
        envelope = parent_conn.recv()
        kind, payload = envelope
        if kind == "community_table":
            for comm_bytes, centroid_bytes in payload:
                cu = UUID(bytes=comm_bytes)
                community_table_uuids.append(cu)
                if centroid_bytes is None:
                    community_centroids[cu] = []
                else:
                    community_centroids[cu] = np.frombuffer(
                        centroid_bytes, dtype=np.float32
                    ).tolist()
        elif kind == "assign":
            for node_bytes, comm_idx in payload:
                assignments[UUID(bytes=node_bytes)] = int(comm_idx)
        elif kind == "assign_end":
            continue
        elif kind == "backend":
            backend = str(payload)
        elif kind == "top_communities":
            top_communities = [UUID(bytes=b) for b in payload]
        elif kind == "mid_regions":
            for comm_bytes, member_bytes_list in payload:
                mid_regions[UUID(bytes=comm_bytes)] = [
                    UUID(bytes=mb) for mb in member_bytes_list
                ]
        elif kind == "modularity":
            modularity = float(payload)
        elif kind == "centrality":
            for node_bytes, value in payload:
                centrality[UUID(bytes=node_bytes)] = float(value)
        elif kind == "done":
            done = True
        elif kind == "error":
            raise RuntimeError(f"worker reported error: {payload!r}")
        else:
            raise RuntimeError(f"worker emitted unknown envelope kind: {kind!r}")

    node_to_community: dict = {}
    for node_uuid, idx in assignments.items():
        node_to_community[node_uuid] = community_table_uuids[idx]

    return {
        "node_to_community": node_to_community,
        "community_centroids": community_centroids,
        "backend": backend if backend is not None else "flat",
        "top_communities": top_communities,
        "mid_regions": mid_regions,
        "modularity": modularity,
        "centrality": centrality,
    }


def _stream_graph_to_child(parent_conn, graph) -> None:
    """Ship the in-parent graph (nodes + edges) to the child over `parent_conn`.

    Only plaintext `(uuid, embedding_blob)` node tuples and `(src, dst, weight)`
    edge tuples cross the Pipe — no storage handle and no encryption key. The
    sort is defensive normalization for a reproducible envelope; both the
    community kernel and the CSR build re-canonicalize order internally.
    """
    import numpy as np

    node_chunk: list = []
    for uid in sorted(graph.iter_nodes(), key=lambda u: u.bytes):
        emb = graph.get_embedding(uid) or []
        node_chunk.append(
            (str(uid), np.asarray(emb, dtype=np.float32).tobytes())
        )
        if len(node_chunk) >= _STREAM_CHUNK:
            parent_conn.send(("nodes", node_chunk))
            node_chunk = []
    if node_chunk:
        parent_conn.send(("nodes", node_chunk))
    parent_conn.send(("nodes_end", None))

    edge_chunk: list = []
    for src, dst, weight, edge_type in sorted(
        graph.iter_edges_with_weight_and_type(),
        key=lambda e: (e[0].bytes, e[1].bytes),
    ):
        edge_chunk.append((str(src), str(dst), float(weight), edge_type))
        if len(edge_chunk) >= _STREAM_CHUNK:
            parent_conn.send(("edges", edge_chunk))
            edge_chunk = []
    if edge_chunk:
        parent_conn.send(("edges", edge_chunk))
    parent_conn.send(("edges_end", None))


def _stream_topology_to_child(parent_conn, graph) -> None:
    """Ship only node ids and edges to the child — no embedding blobs.

    Betweenness centrality depends on graph topology (node ids + edges) only;
    embedding vectors are not needed and not sent. Transmitting less data over
    the Pipe is safe: the AES fence is preserved because no key or store handle
    crosses the boundary, and the child builds nodes with empty embeddings that
    are sufficient for the centrality computation.
    """
    node_chunk: list = []
    for uid in sorted(graph.iter_nodes(), key=lambda u: u.bytes):
        node_chunk.append(str(uid))
        if len(node_chunk) >= _STREAM_CHUNK:
            parent_conn.send(("nodes_topology", node_chunk))
            node_chunk = []
    if node_chunk:
        parent_conn.send(("nodes_topology", node_chunk))
    parent_conn.send(("nodes_end", None))

    edge_chunk: list = []
    for src, dst, weight, edge_type in sorted(
        graph.iter_edges_with_weight_and_type(),
        key=lambda e: (e[0].bytes, e[1].bytes),
    ):
        edge_chunk.append((str(src), str(dst), float(weight), edge_type))
        if len(edge_chunk) >= _STREAM_CHUNK:
            parent_conn.send(("edges", edge_chunk))
            edge_chunk = []
    if edge_chunk:
        parent_conn.send(("edges", edge_chunk))
    parent_conn.send(("edges_end", None))


def compute_assignment_in_child(
    graph,
    *,
    prior_mode: str = "seeded",
    timeout_s: float | None = None,
    with_centrality: bool = False,
):
    """Run community detection on `graph` in a spawn-context child process.

    The parent serializes the exact in-parent graph it already holds (node and
    edge tuples) and ships it to the child; the child rebuilds the graph and
    runs `detect_communities`. The returned partition (which nodes share a
    community) is identical to the in-process call; community UUIDs may differ
    (the flat fallback mints fresh ones), which is why callers compare
    partitions, not raw UUIDs.

    When `with_centrality` is True the same child also computes the full
    betweenness centrality on the graph it just built and returns it alongside
    the assignment as `(assignment, centrality_map)` — folding both heavy
    computations into ONE child graph-build so the parent retains neither the
    detection arenas nor the betweenness intermediate. When False the function
    returns just the assignment (the historical contract).

    The child reclaims its numba JIT arenas on exit, so the long-lived parent
    never accumulates them. Only plaintext `(uuid, embedding_blob)` node tuples
    and `(src, dst, weight)` edge tuples cross the Pipe — no storage handle and
    no encryption key.
    """
    import multiprocessing

    from iai_mcp.community import CommunityAssignment

    global _first_spawn_seen  # noqa: PLW0603

    if timeout_s is not None:
        timeout = timeout_s
    else:
        timeout = _resolve_timeout(_graph_node_count(graph))

    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    process = ctx.Process(
        target=_community_only_worker_indirection,
        args=(child_conn,),
        daemon=True,
    )
    process.start()
    child_conn.close()

    try:
        try:
            parent_conn.send((
                "config",
                {"prior_mode": prior_mode, "with_centrality": bool(with_centrality)},
            ))
            _stream_graph_to_child(parent_conn, graph)

            result = _drain_community_only_result(parent_conn, timeout=timeout)

            process.join(timeout=5.0)
            # A fully-drained result IS success. exitcode None just means the
            # child is still tearing down (graph + JIT teardown outgrows the
            # join window as the corpus scales) — the finally-terminate reaps
            # it. Raising here threw away a GOOD result and retried the whole
            # compute every cycle: a permanent full-CPU rebuild loop.
            if process.exitcode not in (0, None):
                raise WorkerCrashedError(
                    f"worker exited with code {process.exitcode}"
                )
        except (EOFError, BrokenPipeError, ConnectionError) as exc:
            raise WorkerCrashedError(f"worker pipe failed: {exc!r}") from exc

        _first_spawn_seen = True

        assignment = CommunityAssignment(
            node_to_community=result["node_to_community"],
            community_centroids=result["community_centroids"],
            modularity=float(result.get("modularity", 0.0)),
            backend=result["backend"],
            top_communities=result["top_communities"],
            mid_regions=result["mid_regions"],
            lineage_report=None,
        )
        if with_centrality:
            return assignment, result.get("centrality", {})
        return assignment
    finally:
        try:
            parent_conn.close()
        except Exception:  # noqa: BLE001
            pass
        _terminate_worker(process)


def compute_centrality_in_child(
    graph,
    *,
    timeout_s: float | None = None,
) -> dict:
    """Compute full betweenness centrality on `graph` in a spawn-context child.

    Used by the cache-hit path that already holds a community assignment but
    still needs the centrality recomputed — running it in the child keeps the
    betweenness intermediate out of the long-lived parent. The child skips
    community detection entirely (`centrality_only`) and streams back only the
    `node_uuid -> float` centrality map.

    Same AES fence as `compute_assignment_in_child`: only `(uuid, embedding)`
    node tuples and `(src, dst, weight)` edges cross the Pipe.
    """
    import multiprocessing

    global _first_spawn_seen  # noqa: PLW0603

    if timeout_s is not None:
        timeout = timeout_s
    else:
        timeout = _resolve_timeout(_graph_node_count(graph))

    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    process = ctx.Process(
        target=_community_only_worker_indirection,
        args=(child_conn,),
        daemon=True,
    )
    process.start()
    child_conn.close()

    try:
        try:
            parent_conn.send(("config", {"centrality_only": True}))
            # Centrality depends only on topology; send node ids and edges
            # without embedding blobs to reduce pipe traffic.
            _stream_topology_to_child(parent_conn, graph)

            result = _drain_community_only_result(parent_conn, timeout=timeout)

            process.join(timeout=5.0)
            # A fully-drained result IS success. exitcode None just means the
            # child is still tearing down (graph + JIT teardown outgrows the
            # join window as the corpus scales) — the finally-terminate reaps
            # it. Raising here threw away a GOOD result and retried the whole
            # compute every cycle: a permanent full-CPU rebuild loop.
            if process.exitcode not in (0, None):
                raise WorkerCrashedError(
                    f"worker exited with code {process.exitcode}"
                )
        except (EOFError, BrokenPipeError, ConnectionError) as exc:
            raise WorkerCrashedError(f"worker pipe failed: {exc!r}") from exc

        _first_spawn_seen = True
        return result.get("centrality", {})
    finally:
        try:
            parent_conn.close()
        except Exception:  # noqa: BLE001
            pass
        _terminate_worker(process)


MAX_CACHE_BYTES: int = 10 * 1024 * 1024
_SNAPSHOT_NEAR_LIMIT_FRACTION: float = 0.80
_snapshot_near_limit_last_gen: int = -1


def _cache_path(store: Any) -> Path:
    root = getattr(store, "root", None)
    if root is None:
        root = Path.cwd()
    return Path(root) / CACHE_FILENAME


def _cache_encryption_key(store: Any) -> bytes:
    cached_via_store = getattr(store, "_crypto_key", None)
    if isinstance(cached_via_store, (bytes, bytearray)) and len(cached_via_store) == 32:
        return bytes(cached_via_store)
    if hasattr(store, "_key") and callable(store._key):
        try:
            key = store._key()
            if isinstance(key, (bytes, bytearray)) and len(key) == 32:
                return bytes(key)
        except (OSError, ValueError, RuntimeError):
            pass
    user_id = getattr(store, "user_id", "default") or "default"
    root = getattr(store, "root", None)
    if root is None:
        raise RuntimeError(
            "cannot resolve a cache encryption key: the store provides no store root "
            "and no direct key source (_crypto_key / _key()); at least one is required "
            "so the key is bound to the correct store context (no store root)"
        )
    return CryptoKey(user_id=user_id, store_root=Path(root)).get_or_create()


def _cache_key(store: Any) -> tuple:
    # Attempt to read the three counts directly from the store-level corpus-count
    # cache (CorpusCountCache) WITHOUT going through the store count methods.  When
    # the cache is warm (all three keys present) this avoids three method calls and
    # any per-call overhead, so a read-only recall burst that calls _cache_key
    # multiple times pays one SQL COUNT on the first miss and O(1) dict reads on
    # every subsequent call.  The store methods remain the authoritative fallback:
    # a cache miss (cold start, or right after a write-invalidation) falls through
    # to the normal store.active_records_count() / etc. path, which re-fills the
    # cache as a side-effect so the next call is served from it.
    _cc = getattr(store, "_corpus_count_cache", None)
    if _cc is not None:
        try:
            _cached_active = _cc.get("active")
            _cached_edges = _cc.get("edges")
        except Exception:  # noqa: BLE001 -- cache access must never break key derivation
            _cached_active = _cached_edges = None
    else:
        _cached_active = _cached_edges = None

    if _cached_active is not None:
        records_count = int(_cached_active)
    else:
        try:
            records_count = int(store.active_records_count())
        except (OSError, ValueError, KeyError, AttributeError):
            try:
                records_count = int(store.db.open_table("records").count_rows())
            except (OSError, ValueError, KeyError, AttributeError):
                records_count = -1

    if _cached_edges is not None:
        edges_count = int(_cached_edges)
    else:
        try:
            edges_count = int(store.edges_count())
        except (OSError, ValueError, KeyError, AttributeError):
            try:
                edges_count = int(store.db.open_table("edges").count_rows())
            except (OSError, ValueError, KeyError, AttributeError):
                edges_count = -1

    # Pending rows are deliberately NOT part of the key: they are never graph
    # nodes (the active predicate excludes them), so a fresh capture landing
    # as pending changes nothing the graph serves — keying on it re-keyed the
    # warm bundle on EVERY ambient capture. The pending->active flip that DOES
    # change the graph invalidates explicitly at the data-operation boundary
    # (the wake sequence unlinks this cache), which no key term can miss.
    embed_dim = int(getattr(store, "embed_dim", 0))
    return (
        _quantize_count(records_count),
        _quantize_count(edges_count),
        SCHEMA_VERSION_CURRENT,
        embed_dim,
        CACHE_VERSION,
    )


def _parity_components(store: Any) -> tuple:
    embed_dim = int(getattr(store, "embed_dim", 0))
    return (SCHEMA_VERSION_CURRENT, embed_dim, CACHE_VERSION)


class _OverlayBypass:
    __slots__ = ("reason", "age_ms")

    def __init__(self, reason: str, age_ms: int = 0) -> None:
        self.reason = reason
        self.age_ms = age_ms

    def __repr__(self) -> str:  # pragma: no cover
        return f"_OverlayBypass(reason={self.reason!r}, age_ms={self.age_ms})"


def _check_snapshot_invariants(data: dict) -> bool:
    assignment_raw = data.get("assignment")
    if not isinstance(assignment_raw, dict):
        return False
    node_to_community = assignment_raw.get("node_to_community") or {}
    if not isinstance(node_to_community, dict):
        return False
    n_communities = len(set(node_to_community.values()))
    if n_communities == 0 and len(node_to_community) > 0:
        return False
    if n_communities > 100_000:
        return False
    rich_club_raw = data.get("rich_club") or []
    if isinstance(rich_club_raw, list) and rich_club_raw:
        node_ids = set(node_to_community.keys())
        for rc_id in rich_club_raw:
            if rc_id not in node_ids:
                return False
    try:
        modularity = float(assignment_raw.get("modularity", 0.0) or 0.0)
        if not (-1.0 <= modularity <= 1.0):
            return False
    except (TypeError, ValueError):
        return False
    return True


def consult_overlay(store: Any) -> "tuple | _OverlayBypass":
    data = _load_and_decrypt_cache(store)
    if data is None:
        return _OverlayBypass("no_snapshot")

    if data.get("cache_version") != CACHE_VERSION:
        return _OverlayBypass("parity_mismatch")

    # Parity rides the TAIL of the key (schema version, embed dim, cache
    # version) whatever the key's arity — positional indexing here is how a
    # key-shape change silently bypassed the overlay on every recall.
    saved_key = tuple(data.get("key", []))
    if len(saved_key) < 3:
        return _OverlayBypass("parity_mismatch")
    if tuple(saved_key[-3:]) != tuple(_parity_components(store)):
        return _OverlayBypass("parity_mismatch")

    snapshot_generation = data.get("generation", 0)
    if not isinstance(snapshot_generation, int):
        return _OverlayBypass("epoch_mismatch")
    current_gen = get_current_generation()
    if current_gen == 0 or snapshot_generation == 0 or snapshot_generation != current_gen:
        return _OverlayBypass("epoch_mismatch")

    rebuild_ts_str = data.get("rebuild_timestamp")
    age_ms = 0
    if rebuild_ts_str:
        try:
            rebuild_dt = datetime.fromisoformat(str(rebuild_ts_str))
            if rebuild_dt.tzinfo is None:
                rebuild_dt = rebuild_dt.replace(tzinfo=timezone.utc)
            age_sec = (datetime.now(timezone.utc) - rebuild_dt).total_seconds()
            age_ms = max(0, int(age_sec * 1000))
        except (TypeError, ValueError):
            age_sec = _FUSE_MAX_AGE_SECONDS + 1.0
            age_ms = int(age_sec * 1000)
    else:
        age_sec = 0.0
        age_ms = 0

    dirty = get_dirty_counter()
    if age_sec > _FUSE_MAX_AGE_SECONDS or dirty > _fuse_dirty_threshold(store):
        _emit_freshness_fuse_tripped(store, age_ms=age_ms)
        return _OverlayBypass("fuse_tripped", age_ms=age_ms)

    if not _check_snapshot_invariants(data):
        return _OverlayBypass("invariant_failure")

    # Decode memo keyed on snapshot file identity: the freshness fuse and the
    # invariant checks above run on EVERY call (cheap once the decrypt is
    # memoized), but the UUID-heavy decode is a pure function of file content
    # and must not be re-run per recall for a file that has not changed.
    identity = _cache_file_identity(store)
    if identity is not None:
        decode_memo = getattr(store, "_decoded_overlay_pair_memo", None)
        if decode_memo is not None and decode_memo[0] == identity:
            return decode_memo[1]

    try:
        assignment = _decode_assignment(data["assignment"])
        rich_club = _decode_rich_club(data.get("rich_club"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.debug("runtime_graph_cache overlay decode failed: %s", exc)
        return _OverlayBypass("invariant_failure")

    if identity is not None:
        try:
            store._decoded_overlay_pair_memo = (identity, (assignment, rich_club))
        except (AttributeError, TypeError):
            pass
    return assignment, rich_club


def _emit_freshness_fuse_tripped(store: Any, *, age_ms: int) -> None:
    try:
        from iai_mcp.events import (
            TELEMETRY_FRESHNESS_FUSE_TRIPPED,
            write_event,
        )
        from iai_mcp.store import MemoryStore

        if not isinstance(store, MemoryStore):
            return
        write_event(
            store,
            TELEMETRY_FRESHNESS_FUSE_TRIPPED,
            {"age_ms": int(age_ms), "pending_rebuild": False},
            severity="info",
            buffered=True,
        )
    except Exception:  # noqa: BLE001 -- telemetry must never break recall
        pass


_current_generation: int = 0
#: When non-zero, save() stamps THIS epoch into the snapshot instead of the
#: module generation. save_with_generation sets it to the next epoch before
#: writing and commits the module generation to the same value only after the
#: snapshot is durable — the on-disk epoch and the module epoch must never
#: diverge, or every overlay read bypasses on epoch_mismatch.
_generation_override: int = 0
_GEN_LOCK = threading.Lock()


def _snapshot_generation() -> int:
    with _GEN_LOCK:
        return _generation_override if _generation_override else _current_generation


def get_current_generation() -> int:
    with _GEN_LOCK:
        return _current_generation


def advance_generation() -> int:
    global _current_generation  # noqa: PLW0603
    with _GEN_LOCK:
        _current_generation += 1
        return _current_generation


def load_current_generation_from_snapshot(store: Any) -> int:
    data = _load_and_decrypt_cache(store)
    if data is None:
        return 0
    if data.get("cache_version") != CACHE_VERSION:
        return 0
    gen = data.get("generation", 0)
    try:
        result = int(gen)
        global _current_generation  # noqa: PLW0603
        with _GEN_LOCK:
            if result > _current_generation:
                _current_generation = result
        return result
    except (TypeError, ValueError):
        return 0


def _encode_assignment(assignment: Any) -> dict:
    return {
        "node_to_community": {
            str(leaf): str(comm)
            for leaf, comm in getattr(assignment, "node_to_community", {}).items()
        },
        "community_centroids": {
            str(comm): list(vec)
            for comm, vec in getattr(assignment, "community_centroids", {}).items()
        },
        "modularity": float(getattr(assignment, "modularity", 0.0)),
        "backend": str(getattr(assignment, "backend", "flat")),
        "top_communities": [str(c) for c in getattr(assignment, "top_communities", [])],
        "mid_regions": {
            str(comm): [str(m) for m in members]
            for comm, members in getattr(assignment, "mid_regions", {}).items()
        },
    }


def _decode_assignment(raw: dict) -> Any:
    from iai_mcp.community import CommunityAssignment

    return CommunityAssignment(
        node_to_community={
            UUID(leaf): UUID(comm)
            for leaf, comm in raw.get("node_to_community", {}).items()
        },
        community_centroids={
            UUID(comm): list(vec)
            for comm, vec in raw.get("community_centroids", {}).items()
        },
        modularity=float(raw.get("modularity", 0.0)),
        backend=str(raw.get("backend", "flat")),
        top_communities=[UUID(c) for c in raw.get("top_communities", [])],
        mid_regions={
            UUID(comm): [UUID(m) for m in members]
            for comm, members in raw.get("mid_regions", {}).items()
        },
    )


def _encode_rich_club(rich_club: Any) -> list[str]:
    return [str(u) for u in (rich_club or [])]


def _decode_rich_club(raw: Any) -> list[UUID]:
    return [UUID(u) for u in (raw or [])]


_JSON_DICT_ENTRY_OVERHEAD: int = 4
# 384-dim float vector dominates: 384*24=9216 + structural ~1024
_NODE_PAYLOAD_BYTES_PER_RECORD: int = 10240
# 384-dim float same calculus as node_payload embedding -> 9216 + UUID
_CENTROID_BYTES_PER_RECORD: int = 9472

_MID_REGION_BYTES_PER_RECORD: int = 1280

_RICH_CLUB_BYTES_PER_ENTRY: int = 38

# 36-char UUID key + JSON float value + ", " separators and quotes — overshoot.
_CENTRALITY_BYTES_PER_ENTRY: int = 70

_BASE_SCAFFOLD_BYTES: int = 4096


def _estimate_serialised_bytes(data: dict) -> int:
    total = _BASE_SCAFFOLD_BYTES

    np_block = data.get("node_payload") or {}
    if isinstance(np_block, dict):
        total += len(np_block) * (
            _NODE_PAYLOAD_BYTES_PER_RECORD + _JSON_DICT_ENTRY_OVERHEAD + 38
        )

    # Compact centrality map: a 36-char UUID key + a JSON float value (~24 chars
    # worst case) + structural punctuation. Overshoot at ~70 bytes/entry so the
    # cap accounting stays honest now that this map survives the node_payload
    # shedding.
    centrality_block = data.get("centrality") or {}
    if isinstance(centrality_block, dict):
        total += len(centrality_block) * (_CENTRALITY_BYTES_PER_ENTRY)

    assignment_block = data.get("assignment") or {}
    if isinstance(assignment_block, dict):
        ntc = assignment_block.get("node_to_community") or {}
        if isinstance(ntc, dict):
            total += len(ntc) * 50

        centroids = assignment_block.get("community_centroids") or {}
        if isinstance(centroids, dict):
            total += len(centroids) * (
                _CENTROID_BYTES_PER_RECORD + _JSON_DICT_ENTRY_OVERHEAD
            )

        mid = assignment_block.get("mid_regions") or {}
        if isinstance(mid, dict):
            total += len(mid) * (
                _MID_REGION_BYTES_PER_RECORD + _JSON_DICT_ENTRY_OVERHEAD
            )

        top = assignment_block.get("top_communities") or []
        if isinstance(top, list):
            total += len(top) * 16

    rich_club = data.get("rich_club") or []
    if isinstance(rich_club, list):
        total += len(rich_club) * _RICH_CLUB_BYTES_PER_ENTRY

    return total


def try_load(store: Any) -> tuple | None:
    path = _cache_path(store)
    if not path.exists():
        return None
    try:
        raw_text = path.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        logger.debug("runtime_graph_cache read failed: %s", exc)
        return None

    legacy_v2_plaintext = False
    if is_encrypted(raw_text):
        try:
            key = _cache_encryption_key(store)
            plaintext_json = decrypt_field(raw_text, key, _CACHE_AAD)
            data = json.loads(plaintext_json)
        except (InvalidTag, OSError, ValueError, KeyError, RuntimeError) as exc:
            try:
                sys.stderr.write(
                    '{"event":"runtime_graph_cache_decrypt_failed","error":'
                    + json.dumps(str(exc) or type(exc).__name__)
                    + '}\n'
                )
            except (OSError, ValueError):
                pass
            return None
    else:
        try:
            data = json.loads(raw_text)
        except (ValueError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        if data.get("cache_version") == LEGACY_CACHE_VERSION_PLAINTEXT:
            legacy_v2_plaintext = True
        else:
            return None

    if not isinstance(data, dict):
        return None
    if not legacy_v2_plaintext and data.get("cache_version") != CACHE_VERSION:
        return None
    saved_key = tuple(data.get("key", []))
    current_key = _cache_key(store)
    if legacy_v2_plaintext:
        expected_legacy_key = tuple(
            list(current_key)[:-1] + [LEGACY_CACHE_VERSION_PLAINTEXT]
        )
        if saved_key != expected_legacy_key:
            return None
    else:
        if saved_key != current_key:
            return None

    try:
        assignment = _decode_assignment(data["assignment"])
        rich_club = _decode_rich_club(data.get("rich_club"))
        node_payload_raw = data.get("node_payload")
        node_payload: dict[str, dict] | None
        if isinstance(node_payload_raw, dict):
            node_payload = {}
            drop_count = 0
            for k, v in node_payload_raw.items():
                if not isinstance(v, dict):
                    continue
                surface = v.get("surface")
                if surface in (None, "") or v.get("_decrypt_failed"):
                    drop_count += 1
                    continue
                node_payload[str(k)] = dict(v)
            if drop_count > 0:
                try:
                    sys.stderr.write(
                        '{"event":"runtime_graph_cache_drop_poisoned_entry","count":'
                        + str(drop_count)
                        + '}\n'
                    )
                except OSError:
                    pass
        else:
            node_payload = None
        try:
            max_degree = int(data.get("max_degree", 0) or 0)
        except (TypeError, ValueError):
            max_degree = 0
        # Per-node degree map: {UUID: int}.  Absent in older cache formats —
        # callers must handle an empty dict gracefully (cold-cache degrade).
        node_degrees_raw = data.get("node_degrees")
        node_degrees: dict[UUID, int] = {}
        if isinstance(node_degrees_raw, dict):
            for k, v in node_degrees_raw.items():
                try:
                    node_degrees[UUID(k)] = int(v)
                except (ValueError, TypeError):
                    pass
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.debug("runtime_graph_cache decode failed: %s", exc)
        return None

    if legacy_v2_plaintext:
        try:
            save(
                store, assignment, rich_club,
                node_payload=node_payload, max_degree=max_degree,
            )
        except (OSError, ValueError) as exc:
            logger.debug("runtime_graph_cache legacy re-save failed: %s", exc)

    return assignment, rich_club, node_payload, max_degree, node_degrees


def _load_and_decrypt_cache(store: Any) -> "dict | None":
    # File-identity memo: the read + AES decrypt + JSON parse below is a pure
    # function of the file's content, and EVERY structural branch (the overlay
    # freshness probe, try_load, last_good, the max_degree re-read) calls this
    # per recall — without the memo a warm recall pays the full decrypt of a
    # multi-hundred-KB snapshot several times per dispatch. The save path
    # replaces the file atomically, so `(mtime_ns, size)` is an exact key.
    identity = _cache_file_identity(store)
    if identity is not None:
        memo = getattr(store, "_decrypted_cache_memo", None)
        if memo is not None and memo[0] == identity:
            return memo[1]

    path = _cache_path(store)
    if not path.exists():
        return None
    try:
        raw_text = path.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        logger.debug("runtime_graph_cache read failed: %s", exc)
        return None
    if not is_encrypted(raw_text):
        return None
    try:
        key = _cache_encryption_key(store)
        plaintext_json = decrypt_field(raw_text, key, _CACHE_AAD)
        data = json.loads(plaintext_json)
    except (InvalidTag, OSError, ValueError, KeyError, RuntimeError) as exc:
        try:
            sys.stderr.write(
                '{"event":"runtime_graph_cache_decrypt_failed","error":'
                + json.dumps(str(exc) or type(exc).__name__)
                + '}\n'
            )
        except (OSError, ValueError):
            pass
        return None
    if not isinstance(data, dict):
        return None
    if identity is not None:
        try:
            store._decrypted_cache_memo = (identity, data)
        except (AttributeError, TypeError):
            pass
    return data


def try_load_cache_results(store: Any) -> "tuple[dict[UUID, float], int] | None":
    """Load the compact, expensive-to-recompute cache results.

    Returns `(centrality_map, payload_record_count)` where `centrality_map` is a
    `{node_id: centrality}` dict keyed by UUID and `payload_record_count` is the
    corpus size the cache was built at. These two compact fields survive the
    size-cap shedding that drops the large `node_payload`, so the warm read path
    can reuse the cached betweenness centrality (skipping the O(V*E) recompute)
    even when the full payload was shed.

    Returns None when the cache is absent, fails the same key/version gate as
    `try_load`, or predates this format (an old cache without the compact fields
    — one rebuild then writes the new format). The returned centrality map may
    be empty when the cache was built from an empty node_payload; callers treat
    an empty map as "no cached centrality" and rebuild.
    """
    data = _load_and_decrypt_cache(store)
    if data is None:
        return None
    if data.get("cache_version") != CACHE_VERSION:
        return None
    saved_key = tuple(data.get("key", []))
    current_key = _cache_key(store)
    if saved_key != current_key:
        return None

    centrality_raw = data.get("centrality")
    if not isinstance(centrality_raw, dict):
        # Predates the compact-results format — force one rebuild that writes it.
        return None
    payload_record_count_raw = data.get("payload_record_count")
    if not isinstance(payload_record_count_raw, int):
        return None

    try:
        centrality_map: dict[UUID, float] = {
            UUID(str(node_id)): float(value)
            for node_id, value in centrality_raw.items()
        }
    except (TypeError, ValueError) as exc:
        logger.debug("runtime_graph_cache centrality decode failed: %s", exc)
        return None

    return centrality_map, int(payload_record_count_raw)


def _cache_file_identity(store: Any) -> "tuple[int, int] | None":
    """The on-disk cache file's `(mtime_ns, size)` identity, or None when
    absent/unstattable. The save path replaces the file atomically (tmp +
    rename), so a changed identity is the staleness signal for any decode
    memo keyed on it."""
    try:
        st = _cache_path(store).stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def load_last_good_structural(store: Any) -> "tuple | None":
    # File-identity memo + per-store single-flight: the decrypt + JSON +
    # assignment decode below is a few hundred milliseconds of pure-Python
    # work, and this loader is the fallback every recall takes while corpus
    # counts drift between cache saves (a write burst) — without the memo a
    # concurrent recall burst pays one full decode PER RECALL. The decode is
    # a pure function of the cache file's content (the parity gate below is
    # process-constant), so `(mtime_ns, size)` identity is an exact key.
    identity = _cache_file_identity(store)
    if identity is not None:
        memo = getattr(store, "_last_good_structural_memo", None)
        if memo is not None and memo[0] == identity:
            return memo[1]

    with _STRUCTURAL_DECODE_LOCK_GUARD:
        decode_lock = getattr(store, "_structural_decode_lock", None)
        if decode_lock is None:
            decode_lock = threading.Lock()
            store._structural_decode_lock = decode_lock
    with decode_lock:
        identity = _cache_file_identity(store)
        if identity is not None:
            memo = getattr(store, "_last_good_structural_memo", None)
            if memo is not None and memo[0] == identity:
                return memo[1]
        result = _load_last_good_structural_uncached(store)
        if identity is not None:
            store._last_good_structural_memo = (identity, result)
        return result


def _load_last_good_structural_uncached(store: Any) -> "tuple | None":
    data = _load_and_decrypt_cache(store)
    if data is None:
        return None
    if data.get("cache_version") != CACHE_VERSION:
        return None
    # Parity is the key's TAIL whatever its arity (see consult_overlay).
    saved_key = tuple(data.get("key", []))
    if len(saved_key) < 3:
        return None
    if tuple(saved_key[-3:]) != tuple(_parity_components(store)):
        return None
    try:
        assignment = _decode_assignment(data["assignment"])
        rich_club = _decode_rich_club(data.get("rich_club"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.debug("runtime_graph_cache last_good decode failed: %s", exc)
        return None
    return assignment, rich_club


def load_last_good_centrality(store: Any) -> "dict[UUID, float] | None":
    """Load the last-good centrality map gated on parity alone, not the full key.

    `try_load_cache_results` gates on the full `_cache_key`, which folds in a
    windowed record/edge count — so a corpus that has drifted past the staleness
    window returns None even when a perfectly usable centrality map is on disk.
    The bounded warm-path degrade needs the most recent centrality regardless of
    that count drift: a stale-but-real centrality signal is safer than a fresh
    one that never completes. This loader gates only on the schema/embed_dim/
    cache_version parity (the same fence `load_last_good_structural` uses), so a
    child centrality timeout can reuse the prior cycle's result instead of
    recomputing exact betweenness in the long-lived parent.

    Returns the `{node_id: centrality}` map, or None when the cache is absent,
    fails the parity gate, predates the compact-centrality format, or carries an
    empty map.
    """
    data = _load_and_decrypt_cache(store)
    if data is None:
        return None
    if data.get("cache_version") != CACHE_VERSION:
        return None
    # Parity is the key's TAIL whatever its arity (see consult_overlay).
    saved_key = tuple(data.get("key", []))
    if len(saved_key) < 3:
        return None
    if tuple(saved_key[-3:]) != tuple(_parity_components(store)):
        return None
    centrality_raw = data.get("centrality")
    if not isinstance(centrality_raw, dict) or not centrality_raw:
        return None
    try:
        centrality_map: dict[UUID, float] = {
            UUID(str(node_id)): float(value)
            for node_id, value in centrality_raw.items()
        }
    except (TypeError, ValueError) as exc:
        logger.debug(
            "runtime_graph_cache last_good centrality decode failed: %s", exc
        )
        return None
    return centrality_map


def load_recall_structural(store: Any) -> "tuple":
    from iai_mcp.community import CommunityAssignment

    if get_current_generation() == 0:
        load_current_generation_from_snapshot(store)
    try:
        # The overlay freshness/parity gates run on every call (cheap: the
        # snapshot decrypt and the UUID-heavy decode are both memoized by
        # file identity at their own layers), so no gate is ever bypassed.
        overlay_result = consult_overlay(store)
        if not isinstance(overlay_result, _OverlayBypass):
            ov_assignment, ov_rich_club = overlay_result
            # (max_degree, node_degrees) decode memo — pure function of file
            # content like the assignment decode; the node_degrees loop alone
            # constructs one UUID per corpus record on every call otherwise.
            nd_identity = _cache_file_identity(store)
            nd_memo = getattr(store, "_decoded_degrees_memo", None)
            if nd_identity is not None and nd_memo is not None and nd_memo[0] == nd_identity:
                ov_max_degree, ov_node_degrees = nd_memo[1]
            else:
                data = _load_and_decrypt_cache(store)
                ov_max_degree = 0
                ov_node_degrees = {}
                if data is not None:
                    try:
                        ov_max_degree = int(data.get("max_degree", 0) or 0)
                    except (TypeError, ValueError):
                        ov_max_degree = 0
                    nd_raw = data.get("node_degrees")
                    if isinstance(nd_raw, dict):
                        for k, v in nd_raw.items():
                            try:
                                ov_node_degrees[UUID(k)] = int(v)
                            except (ValueError, TypeError):
                                pass
                if nd_identity is not None:
                    store._decoded_degrees_memo = (
                        nd_identity, (ov_max_degree, ov_node_degrees),
                    )
            return ov_assignment, ov_rich_club, ov_max_degree, "overlay", ov_node_degrees
    except Exception:  # noqa: BLE001 -- overlay errors must never break recall
        pass

    # --- In-process decoded-assignment memo -----------------------------------
    # The structural-graph assignment (_decode_assignment) is expensive: it
    # constructs UUID objects for every node_to_community / community_centroids /
    # top_communities / mid_regions entry — the majority of the per-recall UUID
    # construction overhead (~0.275 s/recall on the profiled store).  The decoded
    # tuple is keyed on the cache FILE identity: the decode is a pure function
    # of file content, and count-based keys churn under ambient cycle writes,
    # re-decoding identical content on every recall.  This introduces no
    # effective staleness beyond the existing policy — when try_load rejects a
    # count-drifted cache the flow serves last_good, which decodes the SAME
    # file with no count gate at all, so the served content is identical
    # either way and only the redundant decode is skipped.
    #
    # The memo is stored as a store-instance attribute so two distinct open
    # stores never share a decoded assignment.
    current_key = _cache_file_identity(store)
    existing_memo: "tuple | None" = getattr(store, "_decoded_structural_memo", None)
    if current_key is not None and existing_memo is not None:
        memo_key, memo_decoded = existing_memo
        if memo_key == current_key:
            # Cache hit: return the previously decoded tuple without re-reading
            # or re-decoding the on-disk cache.  The rich_club, max_degree, and
            # node_degrees fields are included so the full return tuple is correct.
            return memo_decoded

    # Single-flight the miss-path decode: a burst of concurrent recalls right
    # after a write (the write changed the counts, so every one of them misses
    # the memo simultaneously) must run the expensive decode ONCE — the losers
    # wait on the lock, then take the winner's freshly-stored memo on the
    # double-check instead of each burning a whole decode on the GIL. The lock
    # is per-store (stored lazily on the store instance; setdefault-style via
    # a module lock so two threads never install different locks).
    with _STRUCTURAL_DECODE_LOCK_GUARD:
        decode_lock = getattr(store, "_structural_decode_lock", None)
        if decode_lock is None:
            decode_lock = threading.Lock()
            store._structural_decode_lock = decode_lock
    with decode_lock:
        # Double-check: a peer may have decoded and memoized while this caller
        # waited. Re-derive the key too — the peer's decode is only reusable
        # for the SAME file content this caller would decode.
        existing_memo = getattr(store, "_decoded_structural_memo", None)
        current_key = _cache_file_identity(store)
        if current_key is not None and existing_memo is not None:
            memo_key, memo_decoded = existing_memo
            if memo_key == current_key:
                return memo_decoded

        cached = try_load(store)
        if cached is not None:
            assignment, rich_club, _node_payload, max_degree, node_degrees = cached
            decoded_tuple = (assignment, rich_club, int(max_degree or 0), "normal", node_degrees)
            # Keyed on file identity: only a rewritten cache file re-decodes.
            if current_key is not None:
                store._decoded_structural_memo = (current_key, decoded_tuple)
            return decoded_tuple

    last_good = load_last_good_structural(store)
    if last_good is not None:
        assignment, rich_club = last_good
        return assignment, rich_club, 0, "last_good", {}

    empty_assignment = CommunityAssignment(
        node_to_community={},
        community_centroids={},
        modularity=0.0,
        backend="cold-degrade",
        top_communities=[],
        mid_regions={},
    )
    return empty_assignment, [], 0, "cold_degrade", {}


_rebuild_timestamp_override: str = ""


def _maybe_emit_snapshot_near_limit(store: Any, estimated_bytes: int) -> None:
    """One-shot per-generation telemetry when the snapshot is about to degrade.

    Rate-limited to one emission per `_GEN_LOCK` window so the event remains
    informative even when many saves fire in quick succession.
    """
    global _snapshot_near_limit_last_gen  # noqa: PLW0603
    threshold = int(MAX_CACHE_BYTES * _SNAPSHOT_NEAR_LIMIT_FRACTION)
    if estimated_bytes < threshold:
        return
    with _GEN_LOCK:
        current_gen = _current_generation
        if _snapshot_near_limit_last_gen == current_gen:
            return
        _snapshot_near_limit_last_gen = current_gen
    try:
        from iai_mcp.events import (
            TELEMETRY_RGC_SNAPSHOT_NEAR_LIMIT,
            emit_best_effort,
        )
        from iai_mcp.store import MemoryStore

        # Guard against test doubles that look like a store but cannot supply
        # the encryption key the events writer needs.
        if not isinstance(store, MemoryStore):
            return
        emit_best_effort(
            store,
            TELEMETRY_RGC_SNAPSHOT_NEAR_LIMIT,
            {
                "estimated_bytes": int(estimated_bytes),
                "max_cache_bytes": int(MAX_CACHE_BYTES),
                "fraction": round(estimated_bytes / max(MAX_CACHE_BYTES, 1), 3),
            },
            severity="info",
        )
    except Exception:  # noqa: BLE001 -- telemetry must never break save
        pass


def save(
    store: Any,
    assignment: Any,
    rich_club: Any,
    node_payload: "dict[str, dict] | None" = None,
    max_degree: int = 0,
    node_degrees: "dict | None" = None,
) -> bool:
    path = _cache_path(store)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    encoded_node_payload: dict[str, dict] | None = None
    # The betweenness centrality is expensive to recompute (O(V*E) Brandes on
    # the whole corpus) but tiny to store: one float per node. The node_payload
    # that carries it is large (a 384-dim embedding + decrypted surface per node)
    # and is cheap to rebuild by streaming the store, so it is the first thing
    # shed when the cache exceeds its size cap. To keep the expensive centrality
    # from being lost together with the large rebuildable payload, it is lifted
    # into a compact top-level map that survives the shedding.
    centrality_map: dict[str, float] = {}
    payload_record_count = 0
    if node_payload:
        encoded_node_payload = {}
        for k, v in node_payload.items():
            if not isinstance(v, dict):
                continue
            raw_emb = v.get("embedding") or []
            raw_tags = v.get("tags") or []
            node_centrality = float(v.get("centrality") or 0.0)
            encoded_node_payload[str(k)] = {
                "embedding": [float(x) for x in raw_emb],
                "surface": str(v.get("surface", "")),
                "centrality": node_centrality,
                "tier": str(v.get("tier", "episodic")),
                "pinned": bool(v.get("pinned", False)),
                "tags": [str(t) for t in raw_tags if t is not None],
                "language": str(v.get("language", "en") or "en"),
            }
            centrality_map[str(k)] = node_centrality
        payload_record_count = len(encoded_node_payload)

    # Serialize per-node degree map — compact (one int per node) so it is kept
    # through the size-cap shedding that may drop node_payload.  Stored as a
    # flat {str(node_id): int} map; an absent or empty map degrades gracefully
    # at load time (callers fall back to the bounded-traversal count).
    serialized_node_degrees: dict[str, int] = {}
    if node_degrees:
        for nid, deg in node_degrees.items():
            try:
                serialized_node_degrees[str(nid)] = int(deg)
            except (TypeError, ValueError):
                pass

    data = {
        "cache_version": CACHE_VERSION,
        "key": list(_cache_key(store)),
        "assignment": _encode_assignment(assignment),
        "rich_club": _encode_rich_club(rich_club),
        "node_payload": encoded_node_payload or {},
        # Compact, expensive-to-recompute results that must outlive the size-cap
        # shedding of node_payload. The map is a flat {node_id: centrality}; the
        # count records the corpus size the cache was built at so drift can be
        # measured even after node_payload is dropped.
        "centrality": centrality_map,
        "payload_record_count": int(payload_record_count),
        "max_degree": int(max_degree or 0),
        # Per-node true hebbian degree map (node_id -> int).  Enables bounded
        # traversal fan-out without clamping the ranking degree numerator.
        "node_degrees": serialized_node_degrees,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "generation": int(_snapshot_generation()),
        "rebuild_timestamp": _rebuild_timestamp_override or "",
    }

    estimated_bytes = _estimate_serialised_bytes(data)
    _maybe_emit_snapshot_near_limit(store, estimated_bytes)
    # Shedding order: drop the large rebuildable payload first, then the
    # assignment's centroids/mid_regions if still over cap. The compact
    # `centrality` map and `payload_record_count` are NEVER shed — they are the
    # results the warm read path reuses to skip the betweenness recompute.
    if estimated_bytes > MAX_CACHE_BYTES:
        data["node_payload"] = {}
    if _estimate_serialised_bytes(data) > MAX_CACHE_BYTES:
        if isinstance(data.get("assignment"), dict):
            data["assignment"]["community_centroids"] = {}
    if _estimate_serialised_bytes(data) > MAX_CACHE_BYTES:
        if isinstance(data.get("assignment"), dict):
            data["assignment"]["mid_regions"] = {}
    if _estimate_serialised_bytes(data) > MAX_CACHE_BYTES:
        return False

    serialised = json.dumps(data, ensure_ascii=False)

    try:
        key = _cache_encryption_key(store)
        ciphertext = encrypt_field(serialised, key, _CACHE_AAD)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.debug("runtime_graph_cache encrypt failed: %s", exc)
        try:
            sys.stderr.write(
                '{"event":"runtime_graph_cache_encrypt_failed"}\n'
            )
        except OSError:
            pass
        return False

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="ascii") as f:
            f.write(ciphertext)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(path))
        return True
    except OSError as exc:
        logger.debug("runtime_graph_cache write failed: %s", exc)
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        return False


def save_with_generation(
    store: Any,
    assignment: Any,
    rich_club: Any,
    node_payload: "dict[str, dict] | None" = None,
    max_degree: int = 0,
    node_degrees: "dict | None" = None,
) -> bool:
    ts_iso = datetime.now(timezone.utc).isoformat()
    global _rebuild_timestamp_override, _generation_override  # noqa: PLW0603
    global _current_generation  # noqa: PLW0603
    with _GEN_LOCK:
        _rebuild_timestamp_override = ts_iso
        next_gen = _current_generation + 1
        _generation_override = next_gen
    result = False
    try:
        result = save(
            store, assignment, rich_club,
            node_payload=node_payload, max_degree=max_degree,
            node_degrees=node_degrees,
        )
    finally:
        # The snapshot on disk carries next_gen; the module epoch must commit
        # to the SAME value, and only once the bytes are durable. A failed or
        # raising save leaves the module epoch untouched (no epoch running
        # ahead of disk) and must not zero the dirty counter.
        with _GEN_LOCK:
            _rebuild_timestamp_override = ""
            _generation_override = 0
            if result and next_gen > _current_generation:
                _current_generation = next_gen
    if result:
        reset_dirty_counter()
    return result


def invalidate_at_root(root: Any) -> None:
    """Delete the warm-graph snapshot under ``root`` so the next build re-streams
    the full corpus.

    Takes a store root path directly (not a store handle) so the storage layer
    can invalidate the snapshot at the moment a row's embedding lands without
    holding a store reference. Unlinking is key-free — the AES fence is untouched.
    """
    if root is None:
        return
    path = Path(root) / CACHE_FILENAME
    try:
        if path.exists():
            path.unlink()
    except OSError as exc:
        logger.debug("runtime_graph_cache invalidate failed: %s", exc)


def invalidate(store: Any) -> None:
    # Explicit invalidation is a demand for a FRESH build: drop the in-process
    # warm bundle too, so the next build_runtime_graph rebuilds inline instead
    # of serving the stale bundle while refreshing behind it (the
    # stale-while-revalidate path is reserved for ambient churn, where no
    # caller asked for freshness).
    try:
        if hasattr(store, "_warm_graph_bundle"):
            store._warm_graph_bundle = None
    except (AttributeError, TypeError):
        pass
    invalidate_at_root(getattr(store, "root", None) or Path.cwd())


def _rebuild_and_save_rgc(store: Any, *, force: bool = False) -> dict:
    import multiprocessing
    import time

    from iai_mcp.community import CommunityAssignment
    from iai_mcp.events import (
        TELEMETRY_RGC_WORKER_CRASH,
        TELEMETRY_RGC_WORKER_SUCCESS,
        TELEMETRY_RGC_WORKER_TIMEOUT,
        emit_best_effort,
    )
    from iai_mcp.runtime_graph_cache_ro_export import (
        iter_edges_chunks,
        iter_records_chunks,
        open_ro_connection,
        read_transaction,
    )

    global _first_spawn_seen  # noqa: PLW0603

    with _PERSISTENT_GRAPH_LOCK:
        if not force:
            # Skip the rebuild (and its allocation) only when the cached snapshot
            # is still usable for recall. The read path's structural source is
            # the authoritative signal: warm iff overlay/normal, cold otherwise.
            # It already folds in no-snapshot / parity / epoch / generation==0 /
            # age+dirty fuse. The dirty counter is a separate write-volume signal,
            # so a cache can be cold while the counter is zero — gate on both.
            try:
                structural_source = load_recall_structural(store)[3]
            except Exception:  # noqa: BLE001 -- a probe failure must never drop a warm-up
                structural_source = "cold_degrade"  # fail toward rebuilding
            cache_is_warm = structural_source in ("overlay", "normal")
            if cache_is_warm and get_dirty_counter() <= _fuse_dirty_threshold(store):
                return {
                    "rebuilt": False,
                    "skipped": "warm_and_below_dirty_threshold",
                    "structural_source": structural_source,
                    "node_count": 0,
                    "generation": get_current_generation(),
                }

        # Estimate the dataset size up-front so the watchdog timeout can be
        # scaled to the workload. `active_records_count` is the cheap
        # COUNT(*) under the same predicate used by the streaming SELECT.
        try:
            est_node_count = int(store.active_records_count())
        except Exception:  # noqa: BLE001
            est_node_count = 0

        # Spawn the worker. Spawn-context (not fork) so the child re-imports
        # cleanly on macOS and Linux; the child closes its end after start so
        # the parent does not hold a half-of-pipe alive on crash detection.
        first_spawn_flag = not _first_spawn_seen
        timeout_s = _resolve_timeout(est_node_count)
        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        process = ctx.Process(
            target=_worker_entry_indirection,
            args=(child_conn,),
            daemon=True,
        )
        process.start()
        child_conn.close()

        db_path = store.db._hippo_dir / "brain.sqlite3"
        ro_conn = None
        node_count = 0
        t0 = time.perf_counter()
        try:
            # Stream the projection. The dedicated read-only connection means
            # the shared write lock is never held during streaming.
            ro_conn = open_ro_connection(db_path)
            try:
                with read_transaction(ro_conn):
                    for chunk in iter_records_chunks(ro_conn):
                        parent_conn.send(("nodes", chunk))
                        node_count += len(chunk)
                    parent_conn.send(("nodes_end", None))
                    for chunk in iter_edges_chunks(ro_conn):
                        parent_conn.send(("edges", chunk))
                    parent_conn.send(("edges_end", None))
            finally:
                try:
                    ro_conn.close()
                except Exception:  # noqa: BLE001
                    pass
                ro_conn = None

            # Receive the compact result.
            result = _drain_worker_result(parent_conn, timeout=timeout_s)

            process.join(timeout=5.0)
            # A fully-drained result IS success. exitcode None just means the
            # child is still tearing down (graph + JIT teardown outgrows the
            # join window as the corpus scales) — the finally-terminate reaps
            # it. Raising here threw away a GOOD result and retried the whole
            # compute every cycle: a permanent full-CPU rebuild loop.
            if process.exitcode not in (0, None):
                raise WorkerCrashedError(
                    f"worker exited with code {process.exitcode}"
                )

            # Reassemble parent-side.
            assignment = CommunityAssignment(
                node_to_community=result["node_to_community"],
                community_centroids=result["community_centroids"],
                modularity=float(result.get("modularity", 0.0)),
                backend=result["backend"],
                top_communities=result["top_communities"],
                mid_regions=result["mid_regions"],
                lineage_report=None,
            )
            rich_club = result["rich_club"]
            max_degree = int(result["max_degree"])
            node_degrees = result.get("node_degrees", {})

            saved = save_with_generation(
                store, assignment, rich_club, max_degree=max_degree,
                node_degrees=node_degrees,
            )

            duration_s = time.perf_counter() - t0
            _first_spawn_seen = True

            emit_best_effort(
                store,
                TELEMETRY_RGC_WORKER_SUCCESS,
                {
                    "duration_s": round(duration_s, 3),
                    "node_count": int(node_count),
                    "max_degree": int(max_degree),
                    "first_spawn": first_spawn_flag,
                },
            )
            return {
                "rebuilt": True,
                "saved": saved,
                "node_count": int(node_count),
                "generation": get_current_generation(),
            }

        except WorkerTimeoutError as exc:
            _terminate_worker(process)
            emit_best_effort(
                store,
                TELEMETRY_RGC_WORKER_TIMEOUT,
                {
                    "first_spawn": first_spawn_flag,
                    "timeout_s": timeout_s,
                    "node_count": int(node_count),
                },
                severity="warn",
            )
            return {
                "rebuilt": False,
                "error": str(exc)[:200],
                "node_count": int(node_count),
                "generation": get_current_generation(),
            }
        except WorkerCrashedError as exc:
            _terminate_worker(process)
            emit_best_effort(
                store,
                TELEMETRY_RGC_WORKER_CRASH,
                {
                    "exitcode": getattr(process, "exitcode", None),
                    "reason": "nonzero_exit",
                    "first_spawn": first_spawn_flag,
                },
                severity="warn",
            )
            return {
                "rebuilt": False,
                "error": str(exc)[:200],
                "node_count": int(node_count),
                "generation": get_current_generation(),
            }
        except (BrokenPipeError, EOFError) as exc:
            reason = "broken_pipe" if isinstance(exc, BrokenPipeError) else "pipe_eof"
            _terminate_worker(process)
            emit_best_effort(
                store,
                TELEMETRY_RGC_WORKER_CRASH,
                {
                    "exitcode": getattr(process, "exitcode", None) or "unknown",
                    "reason": reason,
                    "first_spawn": first_spawn_flag,
                },
                severity="warn",
            )
            return {
                "rebuilt": False,
                "error": "worker_disconnected",
                "node_count": int(node_count),
                "generation": get_current_generation(),
            }
        finally:
            try:
                parent_conn.close()
            except Exception:  # noqa: BLE001
                pass
            if ro_conn is not None:
                try:
                    ro_conn.close()
                except Exception:  # noqa: BLE001
                    pass
            if process.is_alive():
                _terminate_worker(process)
