from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

_skip_on_lilli = pytest.mark.skipif(
    os.environ.get("LILLI_STORAGE_DRIVER", "stdlib").lower() == "lilli",
    reason=(
        "Asserts SQLite WAL cross-visibility between two concurrently-open "
        "connections; lilli uses independent pagers per open store and has no "
        "multi-connection snapshot analog (and read_consistency_interval has no "
        "production consumer). Lilli's durable at-rest persistence across reopen is "
        "proven by test_store_raw.py::test_open_store_raw_lilli_returns_engine_conn."
    ),
)

def _make_record(store):
    from iai_mcp.types import MemoryRecord

    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface="f-05 regression probe",
        aaak_index="",
        embedding=[0.0] * store.embed_dim,
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
        created_at=now,
        updated_at=now,
        language="en",
    )

@pytest.fixture
def tmp_store_env(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")
    return tmp_path

@_skip_on_lilli
def test_reader_with_strong_consistency_sees_writer_insert(tmp_store_env):
    from iai_mcp.daemon import _store_is_empty
    from iai_mcp.store import MemoryStore

    reader = MemoryStore(read_consistency_interval=timedelta(seconds=0))
    assert _store_is_empty(reader) is True

    writer = MemoryStore()
    writer.insert(_make_record(writer))

    assert _store_is_empty(reader) is False

@_skip_on_lilli
def test_default_connection_is_snapshot_pinned(tmp_store_env):
    from iai_mcp.store import MemoryStore

    reader = MemoryStore()
    records_tbl = reader.db.open_table("records")
    assert records_tbl.count_rows() == 0

    writer = MemoryStore()
    writer.insert(_make_record(writer))

    assert records_tbl.count_rows() == 1

def test_kwarg_is_persisted_for_introspection(tmp_store_env):
    from iai_mcp.store import MemoryStore

    default_store = MemoryStore()
    assert default_store._read_consistency_interval is None

    strong_store = MemoryStore(read_consistency_interval=timedelta(seconds=0))
    assert strong_store._read_consistency_interval == timedelta(seconds=0)

    eventual_store = MemoryStore(read_consistency_interval=timedelta(seconds=30))
    assert eventual_store._read_consistency_interval == timedelta(seconds=30)
