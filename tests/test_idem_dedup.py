"""Exact-key duplicate hygiene.

Two live records sharing one idem tag are the same captured event stored
twice — historical check-then-insert races plus tag-index holes let them
accumulate. Guards here pin the two halves of the fix: the buffer-aware tag
lookup (no new dups minted between an insert and its flush) and the
idem-dedup sweep (existing extras collapse to the earliest copy).
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _rec(text: str, i: int, tags: list[str]) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    emb = [0.0] * EMBED_DIM
    emb[i % EMBED_DIM] = 1.0
    emb[(i * 31 + 7) % EMBED_DIM] = 0.5
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=emb,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.5,
        difficulty=0.3,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=tags,
        language="en",
    )


def test_find_record_by_tag_sees_buffered_rows(tmp_path):
    """The exact-key check must see a record still sitting in the insert
    buffer — the same-batch blind spot minted duplicates."""
    store = MemoryStore(path=tmp_path)
    idem = "idem:" + "c" * 64
    rec = _rec("buffered row", 1, ["capture", idem])
    store.insert(rec)

    found = store.find_record_by_tag(idem)
    assert found == rec.id, (
        "tag lookup missed the buffered row — the dedup blind spot is back"
    )


def test_idem_dedup_collapses_extras_to_earliest(tmp_path):
    from iai_mcp.migrate import cleanup_idem_duplicates
    from iai_mcp.store._buffers import flush_record_buffer

    store = MemoryStore(path=tmp_path)
    idem = "idem:" + "d" * 64
    first = _rec("same event", 1, ["capture", idem])
    second = _rec("same event", 2, ["capture", idem])
    second.created_at = first.created_at.replace(
        microsecond=(first.created_at.microsecond + 1) % 1000000
    )
    lone = _rec("different event", 3, ["capture", "idem:" + "e" * 64])
    for r in (first, second, lone):
        store.insert(r)
    flush_record_buffer(store)

    dry = cleanup_idem_duplicates(store, apply=False)
    assert dry["groups"] == 1
    assert dry["extra_copies"] == 1
    assert dry["tombstoned"] == 0

    applied = cleanup_idem_duplicates(store, apply=True, store_path=tmp_path)
    assert applied["tombstoned"] == 1

    with store.db._conn_lock:
        rows = store.db._conn.execute(
            "SELECT id, tombstoned_at FROM records"
        ).fetchall()
    state = {str(r["id"]): r["tombstoned_at"] for r in rows}
    assert state[str(first.id)] is None, "keeper (earliest) must stay live"
    assert state[str(second.id)] is not None, "extra copy must be tombstoned"
    assert state[str(lone.id)] is None, "unrelated record must be untouched"

    rerun = cleanup_idem_duplicates(store, apply=True, store_path=tmp_path)
    assert rerun["groups"] == 0, "sweep must be idempotent"
