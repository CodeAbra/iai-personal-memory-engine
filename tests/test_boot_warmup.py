"""Contracts for the daemon boot warm-up: write-free, index-warming, fail-safe.

The boot warm-up fires the real first-recall query shapes (ANN candidate
fetch, the exact-cosine authority build, the curiosity/events query, the
three corpus COUNTs, the incident-edges paired fetch) once, in the
background, right after the socket comes up — so a cold storage engine's
per-process indexes are built before the user's first recall pays for it.

It must never write anything to the memory store: no new rows, no new
events, no reinforcement enqueue. Its only artifact is a small diagnostics
sidecar file (timings only) written at the store root.
"""

from __future__ import annotations

import json
import os
import struct
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp.hippo._db import DEFAULT_STORAGE_DRIVER

_EMBED_DIM = 32


@pytest.fixture(autouse=True)
def _small_embed_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Small embedding dimension so no real embedder ever loads."""
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(_EMBED_DIM))


def _embedding_blob(seed: int) -> bytes:
    val = float((seed % 97) - 48) / 48.0
    vec = [val] * _EMBED_DIM
    return struct.pack(f"<{_EMBED_DIM}f", *vec)


def _embedding_list(seed: int) -> list[float]:
    val = float((seed % 97) - 48) / 48.0
    return [val] * _EMBED_DIM


def _seed_store(store, *, n_records: int = 60, n_edges: int = 80, n_questions: int = 3) -> None:
    """Seed a small corpus directly via SQL so no embedder / no dispatch fires."""
    from iai_mcp.hippo import _txn

    db = store.db
    now_iso = datetime.now(timezone.utc).isoformat()
    record_ids = [str(uuid4()) for _ in range(n_records)]

    record_rows = []
    for i, rid in enumerate(record_ids):
        record_rows.append((
            rid,
            "episodic",
            None,
            _embedding_blob(i),
            None,  # tombstoned_at
            0,     # embedding_pending
            now_iso,
            now_iso,
        ))
    sql = (
        "INSERT INTO records "
        "(id, tier, literal_surface, embedding, tombstoned_at, "
        " embedding_pending, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    )
    with db._conn_lock:
        with _txn(db._conn):
            db._conn.executemany(sql, record_rows)

    edge_rows = []
    for i in range(n_edges):
        src = record_ids[i % n_records]
        dst = record_ids[(i * 7 + 3) % n_records]
        edge_type = "hebbian" if i % 2 == 0 else "contradicts"
        weight = round(0.5 + (i % 10) * 0.05, 3)
        edge_rows.append((src, dst, edge_type, weight))
    edge_sql = (
        "INSERT OR IGNORE INTO edges (src, dst, edge_type, weight) "
        "VALUES (?, ?, ?, ?)"
    )
    with db._conn_lock:
        with _txn(db._conn):
            db._conn.executemany(edge_sql, edge_rows)

    from iai_mcp.events import write_event

    for i in range(n_questions):
        write_event(
            store,
            "curiosity_question",
            {
                "question_id": str(uuid4()),
                "text": f"warm-up seed question {i}",
                "entropy": 0.5,
                "tier": "question",
                "triggered_by": [],
            },
        )


def _db_file_stat(store) -> tuple[int, int] | None:
    """(mtime_ns, size) of the underlying store db file, or None if unresolvable."""
    for candidate_attr in ("db_path", "_db_path", "path"):
        candidate = getattr(store.db, candidate_attr, None)
        if isinstance(candidate, (str, Path)) and Path(candidate).is_file():
            st = Path(candidate).stat()
            return st.st_mtime_ns, st.st_size
    # Fall back: scan the store root for a plausible single db file.
    for pattern in ("*.sqlite3", "*.db", "*.lilli"):
        hits = list((store.root / "hippo").glob(pattern)) if (store.root / "hippo").is_dir() else []
        if hits:
            st = hits[0].stat()
            return st.st_mtime_ns, st.st_size
    return None


def test_warmup_is_write_free(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iai_mcp.daemon._boot_warmup import run_boot_warmup
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    try:
        _seed_store(store)

        active_before = store.active_records_count()
        pending_before = store.pending_records_count()
        edges_before = store.edges_count()
        db_stat_before = _db_file_stat(store)

        reinforce_calls: list[list] = []
        original_queue_reinforce = store.queue_reinforce

        def _spy_queue_reinforce(record_ids):
            reinforce_calls.append(list(record_ids))
            return original_queue_reinforce(record_ids)

        monkeypatch.setattr(store, "queue_reinforce", _spy_queue_reinforce)

        summary = run_boot_warmup(store)

        assert store.active_records_count() == active_before
        assert store.pending_records_count() == pending_before
        assert store.edges_count() == edges_before
        assert reinforce_calls == [], "warm-up must never call queue_reinforce"

        db_stat_after = _db_file_stat(store)
        if db_stat_before is not None:
            assert db_stat_after == db_stat_before, (
                "warm-up must not touch the store db file (mtime_ns, size)"
            )

        sidecar = store.root / ".boot-warmup.json"
        assert sidecar.is_file(), "warm-up must write its diagnostics sidecar"
        payload = json.loads(sidecar.read_text())
        assert isinstance(payload, dict)
        assert payload.get("skipped") is not True
    finally:
        store.close()


@pytest.mark.skipif(
    os.environ.get("LILLI_STORAGE_DRIVER", DEFAULT_STORAGE_DRIVER).lower() != "lilli",
    reason="index-warming contract only observable via the lilli engine's full_scan_count",
)
def test_warmup_builds_engine_indexes(tmp_path) -> None:
    from iai_mcp.curiosity import get_pending_questions
    from iai_mcp.daemon._boot_warmup import run_boot_warmup
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    try:
        _seed_store(store)

        conn = getattr(store.db, "_conn", None)
        if conn is None or not hasattr(conn, "full_scan_count"):
            pytest.skip("full_scan_count not available on this connection type")

        summary = run_boot_warmup(store)
        probe_ids = summary.get("probe_hit_ids") or []

        conn.reset_full_scan_count()

        probe_vec = _embedding_list(0)
        store.query_similar(probe_vec, k=10)
        store.active_records_count()
        store.pending_records_count()
        store.edges_count()
        get_pending_questions(store, limit=2)
        if probe_ids:
            from uuid import UUID as _UUID
            ids = [_UUID(str(pid)) for pid in probe_ids[:30]]
            store.incident_edges(ids, edge_types=["hebbian", "contradicts"], top_k=None)

        scans = conn.full_scan_count()
        assert scans == 0, (
            f"warmed query shapes triggered {scans} full scan(s) after run_boot_warmup — "
            "expected the engine indexes to already be warm"
        )
    finally:
        store.close()


def test_warmup_kill_switch(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iai_mcp.daemon._boot_warmup import BOOT_WARMUP_OFF_ENV, run_boot_warmup
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv(BOOT_WARMUP_OFF_ENV, "1")

    store = MemoryStore(path=tmp_path)
    try:
        _seed_store(store)

        probe_calls: list = []
        original_query_similar = store.query_similar

        def _spy_query_similar(*args, **kwargs):
            probe_calls.append((args, kwargs))
            return original_query_similar(*args, **kwargs)

        monkeypatch.setattr(store, "query_similar", _spy_query_similar)

        summary = run_boot_warmup(store)

        assert summary.get("skipped") is True
        assert probe_calls == [], "kill-switch must fire zero probes"
        assert not (store.root / ".boot-warmup.json").exists(), (
            "kill-switch must not write the sidecar"
        )
    finally:
        store.close()


def test_warmup_per_shape_fail_safe(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iai_mcp.daemon._boot_warmup import run_boot_warmup
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    try:
        _seed_store(store)

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated exact_top_k failure")

        monkeypatch.setattr(store, "exact_top_k", _boom)

        summary = run_boot_warmup(store)

        assert isinstance(summary, dict)
        assert summary.get("skipped") is not True
        shapes = summary.get("shapes")
        assert isinstance(shapes, dict) and shapes, "other shapes must still be timed"
        assert any(
            isinstance(v, dict) and "error" in v
            for v in shapes.values()
        ), "the failed shape must be recorded with an error marker"
        assert any(
            isinstance(v, dict) and "error" not in v
            for v in shapes.values()
        ), "at least one other shape must have succeeded"
    finally:
        store.close()


def test_warmup_empty_store(tmp_path) -> None:
    from iai_mcp.daemon._boot_warmup import run_boot_warmup
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    try:
        summary = run_boot_warmup(store)
        assert isinstance(summary, dict)
        assert summary.get("skipped") is not True
    finally:
        store.close()
