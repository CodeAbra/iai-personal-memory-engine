"""In-place snapshot refresh of pooled RO readers.

A stale pooled reader used to pay a whole-connection teardown + reopen (and
the reopened connection's lazy index rebuilds) after EVERY committed write —
including the reinforcement write recall itself issues, so each recall made
the next one cold. The borrow path now re-arms the connection in place via
the engine's read-only snapshot refresh; the close+reopen survives only as
the fallback for adapters without a refresh (stdlib driver) or a refresh
that errors.

Every store here lives under ``tmp_path`` — never a real or prod store.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp.store import MemoryStore
from iai_mcp.store._buffers import flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _require_lilli(monkeypatch) -> None:
    try:
        import iai_mcp_native  # noqa: F401, PLC0415
    except ImportError:
        pytest.skip("iai_mcp_native not built — lilli driver unavailable")
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")


def _make(text: str) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
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
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )


def _insert_flushed(store: MemoryStore, text: str) -> MemoryRecord:
    r = _make(text)
    store.insert(r)
    flush_record_buffer(store)
    return r


def _count_records(conn) -> int:
    row = conn.execute("SELECT COUNT(*) FROM records").fetchone()
    return int(row[0])


def _prefill(pool) -> None:
    # Converge the pool to capacity so later borrows never trigger the
    # one-slot-per-borrow lazy growth that would look like a reopen.
    for _ in range(int(getattr(pool, "_size", 4)) + 1):
        with pool.borrow():
            pass


def _count_opens(pool, monkeypatch) -> list:
    opens: list = []
    original = pool._open_slot

    def _counting_open():
        opens.append(1)
        return original()

    monkeypatch.setattr(pool, "_open_slot", _counting_open)
    return opens


def test_stale_borrow_refreshes_in_place_without_reopen(tmp_path: Path, monkeypatch):
    _require_lilli(monkeypatch)
    store = MemoryStore(path=tmp_path)
    _insert_flushed(store, "refresh-target-1")

    pool = store.db._ro_pool
    _prefill(pool)
    with pool.borrow() as conn:
        before = _count_records(conn)

    opens = _count_opens(pool, monkeypatch)

    _insert_flushed(store, "refresh-target-2")
    with pool.borrow() as conn:
        assert _count_records(conn) == before + 1, (
            "the refreshed reader must see the newly committed row"
        )
    assert not opens, (
        "a stale borrow must re-arm the connection in place — a reopen "
        "means the refresh path did not engage"
    )


def test_refresh_failure_falls_back_to_reopen(tmp_path: Path, monkeypatch):
    _require_lilli(monkeypatch)
    from iai_mcp.lillibrain import connection as _lc

    store = MemoryStore(path=tmp_path)
    _insert_flushed(store, "fallback-target-1")

    pool = store.db._ro_pool
    _prefill(pool)
    with pool.borrow() as conn:
        before = _count_records(conn)

    opens = _count_opens(pool, monkeypatch)

    def _broken_refresh(self) -> bool:
        raise RuntimeError("simulated refresh failure")

    monkeypatch.setattr(_lc._OwnedEngineRawConn, "refresh", _broken_refresh)

    _insert_flushed(store, "fallback-target-2")
    with pool.borrow() as conn:
        assert _count_records(conn) == before + 1, (
            "the reopened reader must see the newly committed row"
        )
    assert opens, "a failing refresh must fall back to the close + reopen path"


def test_many_write_refresh_rounds_stay_exact(tmp_path: Path, monkeypatch):
    _require_lilli(monkeypatch)
    store = MemoryStore(path=tmp_path)
    _insert_flushed(store, "round-0")

    pool = store.db._ro_pool
    _prefill(pool)
    with pool.borrow() as conn:
        base = _count_records(conn)

    for i in range(1, 16):
        _insert_flushed(store, f"round-{i}")
        with pool.borrow() as conn:
            assert _count_records(conn) == base + i, (
                f"round {i}: refreshed reader must serve every committed row"
            )
