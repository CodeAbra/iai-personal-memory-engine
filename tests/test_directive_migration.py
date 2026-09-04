"""Proof that a store predating the directive column migrates cleanly.

Builds a genuine pre-migration schema (the records table without the
directive column, exactly as an on-disk store created before this change
would have it), inserts rows against that schema, then reopens the same
store path with the real schema-reconcile code to trigger the actual
ALTER TABLE / add_columns call. Existing rows must read back
directive == False on both storage drivers, and the reconcile call itself
must not raise.
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


def _ddl_without_column(ddl: str, column: str) -> str:
    lines = [line for line in ddl.splitlines() if column not in line]
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == ")":
            continue
        if lines[i].rstrip().endswith(","):
            lines[i] = lines[i].rstrip()[:-1]
        break
    return "\n".join(lines)


def _pre_migration_to_row(self, r):
    row = _store_mod.MemoryStore._to_row_real(self, r)
    row.pop("directive", None)
    return row


def _pre_migration_ensure_tables(self) -> None:
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

    ddl_records_no_directive = _ddl_without_column(_DDL_RECORDS, "directive")

    conn = self._conn
    with self._conn_lock, _txn(conn):
        conn.execute(ddl_records_no_directive)
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
                ("salience_level", "TEXT NOT NULL DEFAULT 'unflagged'"),
                ("live", "INTEGER"),
            ],
        )
        for idx in _DDL_RECORDS_INDEXES:
            if "idx_records_directive" in idx:
                continue
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


def _record(*, directive: bool | None = None, text: str = "the sky is blue") -> MemoryRecord:
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
    if directive is not None:
        kwargs["directive"] = directive
    return MemoryRecord(**kwargs)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_directive_flag_round_trips_explicit_true(tmp_path, monkeypatch, driver):
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", driver)
    store = MemoryStore(path=tmp_path)

    rec = _record(directive=True)
    store.insert(rec)

    got = store.get(rec.id)
    assert got is not None
    assert got.directive is True


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_directive_flag_defaults_to_false(tmp_path, monkeypatch, driver):
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", driver)
    store = MemoryStore(path=tmp_path)

    rec = _record()
    store.insert(rec)

    got = store.get(rec.id)
    assert got is not None
    assert got.directive is False


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_pre_migration_store_backfills_false_on_reopen(tmp_path, monkeypatch, driver):
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
        assert "directive" not in cols, (
            "pre-migration fixture is vacuous: directive already present"
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
    # column-add and must not raise.
    store2 = MemoryStore(path=tmp_path)

    with store2.db._conn_lock:
        cols2 = {
            row["name"]
            for row in store2.db._conn.execute("PRAGMA table_info(records)").fetchall()
        }
    assert "directive" in cols2

    got_a = store2.get(rec_a.id)
    got_b = store2.get(rec_b.id)
    assert got_a is not None and got_b is not None
    assert got_a.directive is False
    assert got_b.directive is False
    assert got_a.literal_surface == rec_a.literal_surface
    assert got_b.literal_surface == rec_b.literal_surface


def test_lilli_reconcile_add_column_does_not_raise(tmp_path, monkeypatch):
    """Standalone probe: the lilli engine accepts add_columns against a
    genuinely pre-migration on-disk schema (not merely a fresh CREATE TABLE
    that already has the column). A raise here means STOP and report -- do
    not attempt a Rust source change."""
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
        assert "directive" not in cols
    finally:
        _db_mod.HippoDB._ensure_tables = real_ensure_tables

    store2 = MemoryStore(path=tmp_path)
    with store2.db._conn_lock:
        cols2 = {
            row["name"]
            for row in store2.db._conn.execute("PRAGMA table_info(records)").fetchall()
        }
    assert "directive" in cols2
