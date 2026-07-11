"""Read-after-write visibility through the pooled RO readers.

A pooled read-only reader is a snapshot-at-open. The pool's staleness
generation used to be bumped only by corpus-count-changing writes
(insert-flush / pending-row paths), so a plain UPDATE — provenance append,
reinforcement, tombstone flip — left every warm reader serving the pre-write
field values indefinitely. The writer connection now signals EVERY mutating
statement (and every commit) to all pools registered for the store path, so
the next borrow refreshes the snapshot.

Every store here lives under ``tmp_path`` — never a real or prod store.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp.store import RECORDS_TABLE, MemoryStore, _uuid_literal
from iai_mcp.store._buffers import flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401, PLC0415
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


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


def _warm_reader(store: MemoryStore, record_id) -> None:
    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.search().where(f"id = '{_uuid_literal(record_id)}'").limit(1).to_pandas()
    assert len(df) == 1, "warm-up read must see the flushed row"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_update_visible_after_warm_reader(driver, tmp_path: Path, monkeypatch):
    """The exact stale-read shape: read (warm a reader), UPDATE, read again."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    r = _insert_flushed(store, "write-visibility-target")
    _warm_reader(store, r.id)

    store.append_provenance(r.id, {"ts": "2026-07-05T00:00:00Z", "cue": "wv"})

    got = store.get(r.id)
    assert got is not None
    assert any(p.get("cue") == "wv" for p in got.provenance), (
        f"[{driver}] a warm pooled reader served the pre-UPDATE snapshot"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_raw_writer_update_visible_through_ro_conn(driver, tmp_path: Path, monkeypatch):
    """A raw UPDATE through db._conn must invalidate pooled readers too."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    r = _insert_flushed(store, "raw-update-target")
    _warm_reader(store, r.id)

    with store.db._conn_lock:
        store.db._conn.execute(
            "UPDATE records SET detail_level = 5 WHERE id = ?", [str(r.id)]
        )

    with store.db.ro_conn() as conn:
        row = conn.execute(
            "SELECT detail_level FROM records WHERE id = ?", [str(r.id)]
        ).fetchone()
    assert row is not None
    assert int(row[0]) == 5, (
        f"[{driver}] ro_conn served a stale detail_level after a raw UPDATE"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_second_store_instance_write_invalidates_first_pool(
    driver, tmp_path: Path, monkeypatch
):
    """Cross-instance in-process: a write via instance B refreshes A's readers."""
    _select_driver(driver, monkeypatch)
    store_a = MemoryStore(path=tmp_path)
    r = _insert_flushed(store_a, "cross-instance-target")
    _warm_reader(store_a, r.id)

    store_b = MemoryStore(path=tmp_path)
    with store_b.db._conn_lock:
        store_b.db._conn.execute(
            "UPDATE records SET detail_level = 4 WHERE id = ?", [str(r.id)]
        )

    with store_a.db.ro_conn() as conn:
        row = conn.execute(
            "SELECT detail_level FROM records WHERE id = ?", [str(r.id)]
        ).fetchone()
    assert row is not None
    assert int(row[0]) == 4, (
        f"[{driver}] instance A's pooled reader missed instance B's committed UPDATE"
    )
