"""The crisis drop phase's IN-UPDATE must ride the engine col-index.

The sleep step clears dropped communities with ``UPDATE records SET
community_id = NULL WHERE community_id IN (...)``. That shape is fast only
while three things hold together: ``idx_records_community`` stays in the
table schema, the engine catalog keeps that (non-partial) index in its
ColIndex, and the UPDATE planner serves col-IN through it. Any of the three
regressing silently returns the step to one full table scan per batch —
results stay correct, cycles take hours. Only the engine's own tests pinned
this before, on a hand-built ColIndex; this one pins it end-to-end on a
store the Python schema created.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord

_N_COMMUNITIES = 6
_ROWS_PER_COMMUNITY = 40
_UPDATE_BATCH = 3  # communities per IN-batch, several batches over the set


def _record(community_id) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="crisis drop scan-free fixture row",
        aaak_index="",
        embedding=[0.25] + [0.0] * (EMBED_DIM - 1),
        community_id=community_id,
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
        tags=[],
        language="en",
    )


def test_crisis_drop_in_update_never_full_scans(tmp_path) -> None:
    store = MemoryStore(path=tmp_path / "store")
    communities = [uuid4() for _ in range(_N_COMMUNITIES)]
    for cid in communities:
        for _ in range(_ROWS_PER_COMMUNITY):
            store.insert(_record(cid))
    flush_record_buffer(store)

    tbl = store.db.open_table("records")
    conn = tbl._conn
    counter = getattr(conn, "full_scan_count", None)
    if counter is None:
        pytest.skip("stdlib driver keeps no scan counter; the guard is engine-only")

    from iai_mcp.hippo import _txn

    baseline = counter()
    lock = tbl._db._conn_lock
    for i in range(0, len(communities), _UPDATE_BATCH):
        batch = communities[i:i + _UPDATE_BATCH]
        placeholders = ", ".join("?" for _ in batch)
        stmt = (
            "UPDATE records SET community_id = NULL "
            f"WHERE community_id IN ({placeholders})"
        )
        with lock:
            with _txn(tbl._conn):
                tbl._conn.execute(stmt, [str(c) for c in batch])

    assert counter() == baseline, (
        "the crisis drop IN-UPDATE fell off the engine col-index onto a full "
        "table scan — check idx_records_community in hippo/_table.py, the "
        "catalog's partial-index filter, and the UPDATE planner's col-IN path"
    )

    with lock:
        with _txn(tbl._conn):
            cur = tbl._conn.execute(
                "SELECT COUNT(*) FROM records WHERE community_id IS NULL"
            )
            cleared = cur.fetchone()[0]
    assert cleared == _N_COMMUNITIES * _ROWS_PER_COMMUNITY
