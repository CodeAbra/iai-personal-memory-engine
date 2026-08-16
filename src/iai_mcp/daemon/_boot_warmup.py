"""Write-free boot warm-up: builds the cold storage-engine indexes at WAKE.

A cold storage engine pays a per-process index-build cost the first time it
sees each distinct query shape (an ANN candidate fetch, the resident
exact-cosine matrix, an events scan, a corpus COUNT, a paired edge fetch).
Left alone, that cost lands on the user's first real recall. This module
fires those same query shapes once, read-only, as a background task right
after the socket comes up, so the indexes are already warm by the time a
real recall arrives.

Write-free by construction: every shape below calls a read surface on the
store directly. Nothing here goes through the write/dispatch path — no
reinforcement is queued, no event is written, nothing is captured. The only
side effect is a small diagnostics sidecar (elapsed-ms per shape) written at
the store root, which never contains record content or embeddings.

Fail-safe: any exception while probing one shape is recorded and the
remaining shapes still run; an empty or degenerate corpus is a no-op, not a
crash. Set IAI_MCP_BOOT_WARMUP_OFF=1 to skip entirely.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

log = logging.getLogger(__name__)

BOOT_WARMUP_OFF_ENV = "IAI_MCP_BOOT_WARMUP_OFF"

_SIDECAR_FILENAME = ".boot-warmup.json"

_DEFAULT_PROBE_COUNT = 8
_DEFAULT_K = 10


def _sample_probe_embeddings(store: Any, probe_count: int) -> list[list[float]]:
    """Sample up to probe_count embeddings spread across the active vec_label range.

    Reads the plaintext float32 embedding column directly (the same
    access pattern the resident exact-cosine matrix build already uses) —
    no AES decrypt, no record materialization. Skips any decoded vector
    whose length does not match the store's embedding dimension. Returns an
    empty list on a degenerate (empty) corpus; never raises.
    """
    import numpy as np

    db = store.db
    conn_lock = getattr(db, "_conn_lock", None)
    conn = getattr(db, "_conn", None)
    if conn_lock is None or conn is None:
        return []

    embed_dim = getattr(store, "_embed_dim", None)

    # MIN and MAX are queried as two single-aggregate SELECTs rather than one
    # comma-separated multi-aggregate SELECT: the engine parses each
    # individually but not the combined form.
    with conn_lock:
        lo_row = conn.execute(
            "SELECT MIN(vec_label) FROM records"
            " WHERE tombstoned_at IS NULL AND COALESCE(embedding_pending, 0) = 0"
        ).fetchone()
        hi_row = conn.execute(
            "SELECT MAX(vec_label) FROM records"
            " WHERE tombstoned_at IS NULL AND COALESCE(embedding_pending, 0) = 0"
        ).fetchone()

    if lo_row is None or hi_row is None:
        return []
    lo, hi = lo_row[0], hi_row[0]
    if lo is None or hi is None:
        return []
    lo = int(lo)
    hi = int(hi)
    if hi < lo:
        return []

    span = max(hi - lo, 0)
    bucket_count = max(1, int(probe_count))
    boundaries: list[int] = []
    if bucket_count == 1 or span == 0:
        boundaries = [lo]
    else:
        step = span / (bucket_count - 1)
        boundaries = [lo + round(i * step) for i in range(bucket_count)]

    probes: list[list[float]] = []
    seen_labels: set[int] = set()
    for boundary in boundaries:
        with conn_lock:
            row = conn.execute(
                "SELECT vec_label, embedding FROM records"
                " WHERE vec_label >= ? AND tombstoned_at IS NULL"
                " AND COALESCE(embedding_pending, 0) = 0 AND embedding IS NOT NULL"
                " ORDER BY vec_label LIMIT 1",
                (boundary,),
            ).fetchone()
        if row is None:
            continue
        label = int(row[0])
        if label in seen_labels:
            continue
        seen_labels.add(label)
        blob = row[1]
        try:
            if isinstance(blob, (bytes, bytearray, memoryview)):
                decoded = np.frombuffer(blob, dtype=np.float32).tolist()
            else:
                decoded = list(blob)
        except Exception:  # noqa: BLE001 -- a bad row must not abort probing
            continue
        if embed_dim is not None and len(decoded) != int(embed_dim):
            continue
        probes.append(decoded)

    return probes


def warm_dispatch_surface(store: Any) -> dict:
    """Warm the first-dispatch surface: module imports, the embedder model
    load, and the structural-cache decode memo.

    These are one-time per-process costs that otherwise land on the FIRST
    real recall. Runs before the socket binds in the normal arm; only when
    the total pre-bind deadline overflows does it run post-bind instead —
    reachability never waits on it past that deadline. All read-only: an
    embed produces a vector and discards it, and load_recall_structural only
    reads/decodes the on-disk cache. Returns a summary dict; never raises.
    """
    _t0 = time.perf_counter()
    try:
        from iai_mcp import runtime_graph_cache  # noqa: F401
        from iai_mcp.core import _serializers  # noqa: F401
        from iai_mcp.cue_router import _classify_cue  # noqa: F401
        from iai_mcp.embed import embed_query, embedder_for_store
        from iai_mcp.pipeline import K_CANDIDATES  # noqa: F401

        embedder = embedder_for_store(store)
        embed_query(embedder, "boot warm-up probe cue")
        runtime_graph_cache.load_recall_structural(store)
        return {"elapsed_ms": (time.perf_counter() - _t0) * 1000.0}
    except Exception as exc:  # noqa: BLE001 -- warm-up must never break boot
        return {
            "elapsed_ms": (time.perf_counter() - _t0) * 1000.0,
            "error": str(exc)[:200],
        }


def run_boot_warmup(
    store: Any,
    *,
    probe_count: int = _DEFAULT_PROBE_COUNT,
    k: int = _DEFAULT_K,
    warm_dispatch: bool = True,
) -> dict:
    """Fire the real first-recall read shapes once, write-free.

    Returns a summary dict (never raises). When IAI_MCP_BOOT_WARMUP_OFF=1 is
    set, returns {"skipped": True} immediately without any probe or sidecar
    write.
    """
    if os.environ.get(BOOT_WARMUP_OFF_ENV) == "1":
        return {"skipped": True}

    total_start = time.perf_counter()
    shapes: dict[str, dict] = {}
    probe_hit_ids: list[str] = []

    # Writer-side read-model publication FIRST: build (once) and publish the
    # col/id index pairs + committed row counts so every reader-pool slot
    # adopts them from its first borrow. Without this, readers pay a
    # whole-table scan per query until the first post-boot commit publishes.
    _t_pub = time.perf_counter()
    try:
        _raw = getattr(store.db, "_writer_raw_conn", None) or getattr(
            store.db, "_conn", None,
        )
        _publish = getattr(_raw, "publish_read_models", None)
        if callable(_publish):
            _publish()
            shapes["read_model_publish"] = {
                "ms": round((time.perf_counter() - _t_pub) * 1000.0, 1),
            }
    except Exception as exc:  # noqa: BLE001 -- publication must not abort the warm-up
        log.debug("boot_warmup read-model publish failed: %s", exc, exc_info=True)

    if warm_dispatch:
        shapes["dispatch_surface"] = warm_dispatch_surface(store)

    try:
        probes = _sample_probe_embeddings(store, probe_count)
    except Exception as exc:  # noqa: BLE001 -- sampling failure must not abort the warm-up
        log.debug("boot_warmup probe sampling failed: %s", exc, exc_info=True)
        probes = []

    # Shape (a): ANN candidate fetch — one call per sampled probe, covering a
    # representative spread of the vec_label range.
    _t0 = time.perf_counter()
    try:
        for vec in probes:
            hits = store.query_similar(vec, k=k)
            for record, _score in hits:
                rid = getattr(record, "id", None)
                if rid is not None:
                    probe_hit_ids.append(str(rid))
        shapes["ann_query_similar"] = {
            "elapsed_ms": (time.perf_counter() - _t0) * 1000.0,
            "probe_count": len(probes),
        }
    except Exception as exc:  # noqa: BLE001 -- one bad shape must not sink the rest
        shapes["ann_query_similar"] = {
            "elapsed_ms": (time.perf_counter() - _t0) * 1000.0,
            "error": str(exc)[:200],
        }

    # Shape (b): the resident exact-cosine authority matrix (lazy build).
    _t0 = time.perf_counter()
    try:
        if probes:
            store.exact_top_k(probes[0], k=k)
        shapes["exact_top_k"] = {"elapsed_ms": (time.perf_counter() - _t0) * 1000.0}
    except Exception as exc:  # noqa: BLE001
        shapes["exact_top_k"] = {
            "elapsed_ms": (time.perf_counter() - _t0) * 1000.0,
            "error": str(exc)[:200],
        }

    # Shape (c): curiosity/events query shapes.
    _t0 = time.perf_counter()
    try:
        from iai_mcp.curiosity import get_pending_questions

        get_pending_questions(store, limit=2)
        shapes["pending_questions"] = {"elapsed_ms": (time.perf_counter() - _t0) * 1000.0}
    except Exception as exc:  # noqa: BLE001
        shapes["pending_questions"] = {
            "elapsed_ms": (time.perf_counter() - _t0) * 1000.0,
            "error": str(exc)[:200],
        }

    # Shape (d): the three corpus COUNTs.
    _t0 = time.perf_counter()
    try:
        store.active_records_count()
        store.pending_records_count()
        store.edges_count()
        shapes["corpus_counts"] = {"elapsed_ms": (time.perf_counter() - _t0) * 1000.0}
    except Exception as exc:  # noqa: BLE001
        shapes["corpus_counts"] = {
            "elapsed_ms": (time.perf_counter() - _t0) * 1000.0,
            "error": str(exc)[:200],
        }

    # Shape (e): the recall paired-fetch shape (hebbian + contradicts).
    _t0 = time.perf_counter()
    try:
        if probe_hit_ids:
            from uuid import UUID

            capped_ids = [UUID(rid) for rid in probe_hit_ids[:30]]
            store.incident_edges(
                capped_ids, edge_types=["hebbian", "contradicts"], top_k=None,
            )
        shapes["incident_edges"] = {"elapsed_ms": (time.perf_counter() - _t0) * 1000.0}
    except Exception as exc:  # noqa: BLE001
        shapes["incident_edges"] = {
            "elapsed_ms": (time.perf_counter() - _t0) * 1000.0,
            "error": str(exc)[:200],
        }

    summary: dict[str, Any] = {
        "skipped": False,
        "probe_count": len(probes),
        "probe_hit_ids": probe_hit_ids[:30],
        "shapes": shapes,
        "total_elapsed_ms": (time.perf_counter() - total_start) * 1000.0,
    }

    try:
        sidecar_path = store.root / _SIDECAR_FILENAME
        sidecar_payload = {
            "probe_count": summary["probe_count"],
            "shapes": shapes,
            "total_elapsed_ms": summary["total_elapsed_ms"],
        }
        sidecar_path.write_text(json.dumps(sidecar_payload, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 -- sidecar failure must not raise
        log.debug("boot_warmup sidecar write failed: %s", exc, exc_info=True)

    log.info(
        "boot_warmup completed in %.1fms (probes=%d)",
        summary["total_elapsed_ms"],
        summary["probe_count"],
    )
    return summary
