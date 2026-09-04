"""Proof that a store predating the salience_level column migrates cleanly.

Builds a genuine pre-migration schema (the records table without the
salience_level column, exactly as an on-disk store created before this
change would have it), inserts rows against that schema, then reopens the
same store path with the real schema-reconcile code to trigger the actual
ALTER TABLE. Existing rows must read back salience_level == "unflagged" on
both storage drivers, and the lilli-engine ALTER itself must not raise.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

import iai_mcp.hippo._db as _db_mod
import iai_mcp.store._store as _store_mod
from iai_mcp.hippo._db import _txn
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _pre_migration_to_row(self, r):
    """Reimplementation of MemoryStore._to_row with the salience_level key
    omitted, matching what code genuinely predating this migration would
    have written -- old rows never carried the column in the first place."""
    row = _store_mod.MemoryStore._to_row_real(self, r)
    row.pop("salience_level", None)
    return row


def _pre_migration_ensure_tables(self) -> None:
    """Reimplementation of HippoDB._ensure_tables with the salience_level
    column and its reconcile entry both omitted, to build a genuine
    pre-migration fixture on disk."""
    from iai_mcp.hippo._table import (
        _DDL_RECORDS,
        _DDL_RECORDS_INDEXES,
        _DDL_EDGES,
        _DDL_EDGES_INDEXES,
        _DDL_EVENTS,
        _DDL_EVENTS_INDEXES,
        _DDL_BUDGET_LEDGER,
        _DDL_BUDGET_LEDGER_INDEXES,
        _DDL_RATELIMIT_LEDGER,
        _DDL_HIPPO_META,
        _DDL_RECORD_TAGS,
        _DDL_RECORD_TAGS_INDEXES,
    )

    ddl_records_no_salience = "\n".join(
        line for line in _DDL_RECORDS.splitlines() if "salience_level" not in line
    )

    conn = self._conn
    with self._conn_lock, _txn(conn):
        conn.execute(ddl_records_no_salience)
        self._reconcile_columns(
            "records",
            [
                ("wing", "TEXT"),
                ("room", "TEXT"),
                ("drawer", "TEXT"),
                ("valence", "REAL DEFAULT 0.0"),
                ("hv_tier", "TEXT NOT NULL DEFAULT 'bsc'"),
                ("structure_hv_payload", "BLOB NOT NULL DEFAULT x''"),
                ("embedding_pending", "INTEGER NOT NULL DEFAULT 0"),
                ("role", "TEXT"),
                ("epistemic_status", "TEXT NOT NULL DEFAULT 'unknown'"),
                ("live", "INTEGER"),
            ],
        )
        for idx in _DDL_RECORDS_INDEXES:
            conn.execute(idx)

        conn.execute(_DDL_EDGES)
        for idx in _DDL_EDGES_INDEXES:
            conn.execute(idx)

        conn.execute(_DDL_EVENTS)
        for idx in _DDL_EVENTS_INDEXES:
            conn.execute(idx)

        conn.execute(_DDL_BUDGET_LEDGER)
        for idx in _DDL_BUDGET_LEDGER_INDEXES:
            conn.execute(idx)

        conn.execute(_DDL_RATELIMIT_LEDGER)

        conn.execute(_DDL_HIPPO_META)
        conn.execute(
            "INSERT OR IGNORE INTO _hippo_meta (key, value) VALUES (?, ?)",
            ("schema_version", "1"),
        )
        conn.execute(
            "INSERT OR IGNORE INTO _hippo_meta (key, value) VALUES (?, ?)",
            ("embed_dim", str(self._embed_dim)),
        )

        conn.execute(_DDL_RECORD_TAGS)
        for idx in _DDL_RECORD_TAGS_INDEXES:
            conn.execute(idx)

    self._heal_null_vec_labels()


def _record(*, salience_level: str | None = None, text: str = "the sky is blue") -> MemoryRecord:
    kwargs = dict(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.01] * EMBED_DIM,
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
    if salience_level is not None:
        kwargs["salience_level"] = salience_level
    return MemoryRecord(**kwargs)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_salience_level_round_trips_explicit_value(tmp_path, monkeypatch, driver):
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", driver)
    store = MemoryStore(path=tmp_path)

    rec = _record(salience_level="critical")
    store.insert(rec)

    got = store.get(rec.id)
    assert got is not None
    assert got.salience_level == "critical"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_salience_level_defaults_to_unflagged(tmp_path, monkeypatch, driver):
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", driver)
    store = MemoryStore(path=tmp_path)

    rec = _record()
    store.insert(rec)

    got = store.get(rec.id)
    assert got is not None
    assert got.salience_level == "unflagged"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_pre_migration_store_backfills_unflagged_on_reopen(tmp_path, monkeypatch, driver):
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", driver)

    real_ensure_tables = _db_mod.HippoDB._ensure_tables
    real_to_row = _store_mod.MemoryStore._to_row
    _db_mod.HippoDB._ensure_tables = _pre_migration_ensure_tables
    _store_mod.MemoryStore._to_row_real = real_to_row
    _store_mod.MemoryStore._to_row = _pre_migration_to_row
    try:
        store = MemoryStore(path=tmp_path)

        with store.db._conn_lock:
            cols = {
                row["name"]
                for row in store.db._conn.execute("PRAGMA table_info(records)").fetchall()
            }
        assert "salience_level" not in cols, (
            "pre-migration fixture is vacuous: salience_level already present"
        )

        rec_a = _record(text="alice's first note")
        rec_b = _record(text="alice's second note")
        store.insert(rec_a)
        store.insert(rec_b)
        store.close()
    finally:
        _db_mod.HippoDB._ensure_tables = real_ensure_tables
        _store_mod.MemoryStore._to_row = real_to_row
        del _store_mod.MemoryStore._to_row_real

    # Reopen with the real reconcile code -- this must trigger the actual
    # ALTER TABLE ... ADD COLUMN salience_level ... and must not raise.
    store2 = MemoryStore(path=tmp_path)

    with store2.db._conn_lock:
        cols2 = {
            row["name"]
            for row in store2.db._conn.execute("PRAGMA table_info(records)").fetchall()
        }
    assert "salience_level" in cols2

    with store2.db._conn_lock:
        raw_rows = store2.db._conn.execute(
            "SELECT salience_level FROM records WHERE id = ?", (str(rec_a.id),)
        ).fetchall()
    raw_value = raw_rows[0]["salience_level"] if raw_rows else None
    # Empirical evidence only: either the driver backfilled the DEFAULT
    # literal or left the pre-existing row NULL -- both are acceptable,
    # because correctness is guaranteed by _from_row's coercion below, not
    # by this raw value.
    assert raw_value in (None, "unflagged"), raw_value

    got_a = store2.get(rec_a.id)
    got_b = store2.get(rec_b.id)
    assert got_a is not None and got_b is not None
    assert got_a.salience_level == "unflagged"
    assert got_b.salience_level == "unflagged"
    assert got_a.literal_surface == rec_a.literal_surface
    assert got_b.literal_surface == rec_b.literal_surface


def test_lilli_reconcile_alter_does_not_raise(tmp_path, monkeypatch):
    """Standalone probe: the lilli engine accepts the exact ALTER TABLE
    grammar this migration relies on, against a genuinely pre-migration
    on-disk schema (not merely a fresh CREATE TABLE that already has the
    column). A raise here means STOP and report -- do not attempt a Rust
    source change."""
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")

    real_ensure_tables = _db_mod.HippoDB._ensure_tables
    _db_mod.HippoDB._ensure_tables = _pre_migration_ensure_tables
    try:
        store = MemoryStore(path=tmp_path)
        with store.db._conn_lock:
            cols = {
                row["name"]
                for row in store.db._conn.execute("PRAGMA table_info(records)").fetchall()
            }
        assert "salience_level" not in cols
    finally:
        _db_mod.HippoDB._ensure_tables = real_ensure_tables

    store2 = MemoryStore(path=tmp_path)
    with store2.db._conn_lock:
        cols2 = {
            row["name"]
            for row in store2.db._conn.execute("PRAGMA table_info(records)").fetchall()
        }
    assert "salience_level" in cols2
