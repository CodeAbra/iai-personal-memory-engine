"""Characterization test of the production buffered-write visibility window.

The autouse `_autoflush_lance_buffers` conftest fixture flushes every insert
in every other test, masking the real buffered->flushed->visible window a
sync-path capture goes through before its row lands in SQL. This file opts
out via `IAI_MCP_TEST_NO_AUTOFLUSH=1` and exercises the real window on a
live lilli store, plus the async write-queue's blocks-until-visible contract
(the reason the sync-path buffer gap is not a live recall bug on the daemon
path).

Reader for SQL visibility is `store.db.ro_conn()` — NOT the raw-writer test
helper (that primitive is gated behind an explicit opt-in, and reading raw is
not the point here: this test proves visibility through the store's own
read-only connection).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord


@pytest.fixture(autouse=True)
def _close_stores(monkeypatch):
    """Close every MemoryStore created in this file, draining background queues."""
    created: list[MemoryStore] = []
    orig_init = MemoryStore.__init__

    def _tracking_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(MemoryStore, "__init__", _tracking_init)
    try:
        yield
    finally:
        for store in created:
            try:
                store.close()
            except Exception:  # noqa: BLE001 -- teardown must never raise
                pass


def _random_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _monkeypatch_env(monkeypatch, tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    # Opt out of the autouse autoflush fixture BEFORE any insert runs, so the
    # production buffered window is actually exercised instead of masked.
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")


def _make_role_user(text: str, seed: int) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=_random_vec(seed),
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=["capture", "role:user"],
        language="en",
    )


def _sql_row_exists(store: MemoryStore, record_id) -> bool:
    with store.db.ro_conn() as conn:
        row = conn.execute(
            "SELECT id FROM records WHERE id = ?", (str(record_id),)
        ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Truth 1: sync insert without flush is NOT visible; after flush it IS.
# ---------------------------------------------------------------------------


def test_sync_insert_invisible_until_flush_then_visible(tmp_path, monkeypatch):
    _monkeypatch_env(monkeypatch, tmp_path)
    store_path = tmp_path / "sync-buffer-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    rec = _make_role_user("buffered-not-yet-flushed distinctive record", seed=1)
    store.insert(rec)

    assert not _sql_row_exists(store, rec.id), (
        "a just-inserted record was already visible to a ro_conn reader "
        "before flush_record_buffer — the autoflush opt-out is not exercising "
        "the real buffered window"
    )

    flush_record_buffer(store)

    assert _sql_row_exists(store, rec.id), (
        "record still not visible after flush_record_buffer — the "
        "buffered->flushed->visible contract is broken"
    )


# ---------------------------------------------------------------------------
# Truth 2 (sync freshness coverage): once the recency buffer is warm, a
# still-buffered role:user marker is served by recent_pending_markers without
# requiring its SQL flush.
# ---------------------------------------------------------------------------


def test_sync_warm_recency_serves_still_buffered_marker(tmp_path, monkeypatch):
    _monkeypatch_env(monkeypatch, tmp_path)
    store_path = tmp_path / "sync-warm-recency-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    # Establish a warm buffer BEFORE the buffered insert, so the write-path
    # push lands directly in an already-warm buffer (no lazy-warm re-fill).
    store.warm_recency_buffer()
    assert store._recency_buffer.is_warm

    rec = _make_role_user("still-buffered distinctive recency marker", seed=2)
    store.insert(rec)

    assert not _sql_row_exists(store, rec.id), (
        "setup invariant broken: record must still be unflushed for this "
        "assertion to mean anything"
    )

    markers = store.recent_pending_markers(n=50)
    marker_ids = {str(m.id) for m in markers}
    assert str(rec.id) in marker_ids, (
        "a still-buffered role:user marker was not served by "
        "recent_pending_markers on an already-warm buffer"
    )


# ---------------------------------------------------------------------------
# Truth 3 (MANDATORY): async write-queue insert blocks until the row is
# visible to a fresh reader — no window a later recall can fall into.
# ---------------------------------------------------------------------------


def test_async_insert_blocks_until_visible(tmp_path, monkeypatch):
    _monkeypatch_env(monkeypatch, tmp_path)
    store_path = tmp_path / "async-visible-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    rec = _make_role_user("async blocks-until-visible distinctive record", seed=3)

    async def _run() -> None:
        await store.enable_async_writes()
        # insert() is synchronous even with async writes enabled: it submits
        # to the background write-queue loop and blocks on the per-record
        # future before returning — no explicit flush, no sleep.
        store.insert(rec)

    asyncio.run(_run())

    assert _sql_row_exists(store, rec.id), (
        "async-path insert() returned but the row is not yet visible to a "
        "fresh ro_conn reader — the blocks-until-visible contract is broken: "
        "the write-queue future must resolve only after the SQL add "
        "completes, so no later read can fall into a visibility window"
    )
