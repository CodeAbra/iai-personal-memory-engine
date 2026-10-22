"""Wiring + byte-safety check for the RO connection pager cache bound.

Proves the ``PRAGMA cache_size`` setter added at the ``_OwnedEngineRawConn``
choke point reaches the compiled engine binding on both the raw RO path and
the pooled RO path without error, and that queries through a bounded RO
connection return byte-identical rows to the same queries through the
owning ``HippoDB`` writer connection (which shares the already-bounded main
pager and is never wrapped in ``_OwnedEngineRawConn``).

The native engine's ``PRAGMA cache_size`` has no getter form (the engine's
PRAGMA handler only implements the ``=`` setter; a bare readback returns no
row), so this module does not assert a readback value -- only that the
setter does not raise and that rows read through the bounded connection are
correct.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp.hippo._ro_pool import RoConnPool
from iai_mcp.lillibrain.connection import get_lilli_raw_conn
from iai_mcp.store import MemoryStore
from iai_mcp.store._buffers import flush_edge_buffer, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord

_EDGES_SQL = (
    "SELECT src, dst, edge_type, weight FROM edges "
    "WHERE src IN (?, ?, ?) OR dst IN (?, ?, ?) ORDER BY src, dst"
)
_RECORDS_SQL = "SELECT id, vec_label FROM records ORDER BY vec_label"


def _require_lilli(monkeypatch) -> None:
    try:
        import iai_mcp_native  # noqa: F401, PLC0415
    except ImportError:
        pytest.skip("iai_mcp_native not built -- lilli driver unavailable")
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


def _build_store(tmp_path: Path):
    store = MemoryStore(path=tmp_path)
    recs = [_insert_flushed(store, f"cache-bound-{i}") for i in range(3)]
    store.boost_edges(
        [(recs[0].id, recs[1].id), (recs[1].id, recs[2].id)],
        delta=1.0,
        edge_type="hebbian",
    )
    flush_edge_buffer(store)
    return store, recs


def _edges_params(recs) -> tuple:
    ids = [str(r.id) for r in recs]
    return (*ids, *ids)


def _as_tuples(rows) -> list[tuple]:
    return [tuple(r) for r in rows]


def test_raw_ro_conn_opens_with_bound_and_matches_writer(tmp_path: Path, monkeypatch):
    _require_lilli(monkeypatch)
    store, recs = _build_store(tmp_path)
    try:
        with store.db._conn_lock:
            expected_edges = _as_tuples(
                store.db._conn.execute(_EDGES_SQL, _edges_params(recs)).fetchall()
            )
            expected_records = _as_tuples(store.db._conn.execute(_RECORDS_SQL).fetchall())

        # Opening the raw RO connection through the real production path must
        # not raise -- the setter reaches the compiled engine binding.
        raw = get_lilli_raw_conn(store.db._db_path, read_only=True)
        assert raw is not None
        try:
            got_edges = _as_tuples(raw.execute(_EDGES_SQL, _edges_params(recs)).fetchall())
            got_records = _as_tuples(raw.execute(_RECORDS_SQL).fetchall())
        finally:
            raw.close()

        assert got_edges == expected_edges
        assert got_records == expected_records
    finally:
        store.close()


def test_pooled_ro_slot_opens_with_bound_and_matches_writer(tmp_path: Path, monkeypatch):
    _require_lilli(monkeypatch)
    store, recs = _build_store(tmp_path)
    try:
        with store.db._conn_lock:
            expected_edges = _as_tuples(
                store.db._conn.execute(_EDGES_SQL, _edges_params(recs)).fetchall()
            )
            expected_records = _as_tuples(store.db._conn.execute(_RECORDS_SQL).fetchall())

        pool = RoConnPool(store.db._db_path, size=1)
        try:
            with pool.borrow() as conn:
                got_edges = _as_tuples(conn.execute(_EDGES_SQL, _edges_params(recs)).fetchall())
                got_records = _as_tuples(conn.execute(_RECORDS_SQL).fetchall())
        finally:
            pool.close()

        assert got_edges == expected_edges
        assert got_records == expected_records
    finally:
        store.close()
