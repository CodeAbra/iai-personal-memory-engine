"""Write-fault durability: a failed buffer flush must NOT drop verbatim content.

The record buffer carries verbatim literal_surface. A transient `.add()` fault
(disk error, lock contention) must leave the batch buffered so the next flush
retries — never silently drop the captured records on the floor. Clear-after-
success: rows are removed from the buffer only after the write succeeds.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture(autouse=True)
def _opt_out_of_buffer_autoflush(monkeypatch):
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")


def _make_record(literal_surface: str):
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=literal_surface,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=1,
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


def _clear_buffer(store) -> None:
    from iai_mcp import store as store_mod

    store_mod._record_buffer.pop(id(store), None)
    store_mod._record_last_flush_at.pop(id(store), None)
    store_mod._edge_buffer.pop(id(store), None)
    store_mod._edge_last_flush_at.pop(id(store), None)


def test_record_buffer_survives_add_failure_then_persists_byte_exact(tmp_path: Path):
    from iai_mcp import store as store_mod
    from iai_mcp.store import RECORDS_TABLE, MemoryStore, flush_record_buffer

    # non-English fixture data: multilingual behavior under test — keep
    surfaces = [
        "verbatim content one — must survive a transient write fault",
        # non-English fixture data: multilingual behavior under test — keep
        "雪 второй verbatim 🧠 unicode literal_surface",
        '{"json": "verbatim payload", "n": 3}',
    ]

    with MemoryStore(path=tmp_path) as store:
        _clear_buffer(store)

        for s in surfaces:
            store.insert(_make_record(s))

        buf = store_mod._record_buffer.get(id(store), [])
        assert len(buf) == len(surfaces), "all records must be buffered after insert"

        real_open_table = store.db.open_table

        class _RaisingTable:
            def __init__(self, inner):
                self._inner = inner

            def add(self, rows):
                raise RuntimeError("simulated transient write fault")

            def __getattr__(self, name):
                return getattr(self._inner, name)

        # First flush: .add() raises. The batch MUST stay buffered (no loss).
        store.db.open_table = lambda name: (
            _RaisingTable(real_open_table(name)) if name == RECORDS_TABLE
            else real_open_table(name)
        )
        flushed = flush_record_buffer(store)
        assert flushed == 0, "failed flush must report 0 persisted"

        buf_after_fail = store_mod._record_buffer.get(id(store), [])
        assert len(buf_after_fail) == len(surfaces), (
            "buffered records MUST still be present after a failed flush — "
            "verbatim content must never be dropped on a write fault"
        )

        # Restore the real engine and flush again: now it succeeds.
        store.db.open_table = real_open_table
        flushed2 = flush_record_buffer(store)
        assert flushed2 == len(surfaces), "retry flush must persist the full batch"

        assert not store_mod._record_buffer.get(id(store)), (
            "buffer must be empty after the successful retry"
        )

        # Read back from SQLite via the store codec: byte-exact, no duplication.
        recovered = {r.literal_surface for r in store.all_records()}
        for s in surfaces:
            assert s in recovered, (
                f"verbatim literal_surface lost or corrupted: {s!r} not in store"
            )

        # No duplication: each surface appears exactly once.
        all_surfaces = [r.literal_surface for r in store.all_records()]
        for s in surfaces:
            assert all_surfaces.count(s) == 1, (
                f"record duplicated after retry flush: {s!r}"
            )


def test_record_buffer_keeps_concurrently_appended_row_on_retry(tmp_path: Path):
    """A row appended AFTER the snapshot must not be dropped by clear-after-success."""
    from iai_mcp import store as store_mod
    from iai_mcp.store import RECORDS_TABLE, MemoryStore, flush_record_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_buffer(store)

        snap_record = _make_record("snapshotted row that flushes first")
        store.insert(snap_record)

        real_open_table = store.db.open_table
        later_row = store._to_row(_make_record("appended-after-snapshot row"))

        class _AppendingTable:
            def __init__(self, inner):
                self._inner = inner

            def add(self, rows):
                # Simulate a concurrent append landing during the write window,
                # then persist the snapshotted rows for real.
                store_mod._record_buffer.setdefault(id(store), []).append(later_row)
                return self._inner.add(rows)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        store.db.open_table = lambda name: (
            _AppendingTable(real_open_table(name)) if name == RECORDS_TABLE
            else real_open_table(name)
        )
        flushed = flush_record_buffer(store)
        store.db.open_table = real_open_table

        assert flushed == 1, "only the single snapshotted record was written"
        remaining = store_mod._record_buffer.get(id(store), [])
        assert len(remaining) == 1, (
            "concurrently-appended row must remain buffered, not be cleared"
        )
        assert remaining[0] is later_row, (
            "the surviving buffered row must be the concurrently-appended one"
        )


def test_edge_buffer_survives_execute_failure(tmp_path: Path):
    from iai_mcp import store as store_mod
    from iai_mcp.store import EDGES_TABLE, MemoryStore, flush_edge_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_buffer(store)

        edge_rows = [
            {"src": str(uuid4()), "dst": str(uuid4()), "edge_type": "hebbian", "weight": 1.0},
            {"src": str(uuid4()), "dst": str(uuid4()), "edge_type": "hebbian", "weight": 1.0},
        ]
        store_mod._edge_buffer.setdefault(id(store), []).extend(edge_rows)

        real_open_table = store.db.open_table

        class _RaisingMerge:
            def execute(self, rows):
                raise RuntimeError("simulated transient edge write fault")

        class _RaisingEdgeTable:
            def __init__(self, inner):
                self._inner = inner

            def merge_insert(self, keys):
                return _RaisingMerge()

            def __getattr__(self, name):
                return getattr(self._inner, name)

        store.db.open_table = lambda name: (
            _RaisingEdgeTable(real_open_table(name)) if name == EDGES_TABLE
            else real_open_table(name)
        )
        flushed = flush_edge_buffer(store)
        assert flushed == 0, "failed edge flush must report 0 persisted"
        assert len(store_mod._edge_buffer.get(id(store), [])) == len(edge_rows), (
            "edge batch must stay buffered after a failed flush"
        )

        store.db.open_table = real_open_table
        flushed2 = flush_edge_buffer(store)
        assert flushed2 == len(edge_rows), "retry edge flush must persist the batch"
        assert not store_mod._edge_buffer.get(id(store)), (
            "edge buffer must be empty after successful retry"
        )
