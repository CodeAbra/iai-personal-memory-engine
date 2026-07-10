"""ALTER TABLE ADD COLUMN parity across storage drivers.

A repeated ADD COLUMN must raise an error whose text contains
"duplicate column name" on BOTH drivers. Every migration idempotency guard
keys on that text (see migrate/_role_column.py, _live_flag_backfill.py,
_hv_codec.py); an engine that silently accepts the repeat double-appends the
column and desyncs the row codec.
"""
from __future__ import annotations

import sqlite3

import pytest


def _open(driver: str, tmp_path):
    if driver == "stdlib":
        return sqlite3.connect(str(tmp_path / "t.sqlite3"))
    pytest.importorskip("iai_mcp_native")
    from iai_mcp_native import engine  # noqa: PLC0415

    return engine.Connection.open(str(tmp_path / "t.lilli"), 0)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_repeat_add_column_raises_duplicate_column_name(driver, tmp_path):
    conn = _open(driver, tmp_path)
    try:
        conn.execute("CREATE TABLE records ( id TEXT , n INTEGER )")
        conn.execute("ALTER TABLE records ADD COLUMN role TEXT")
        with pytest.raises(Exception) as exc_info:
            conn.execute("ALTER TABLE records ADD COLUMN role TEXT")
        assert "duplicate column name" in str(exc_info.value).lower(), (
            f"[{driver}] migration guards key on this text, got: {exc_info.value}"
        )
        # The rejected repeat must not have widened the schema.
        cols = [
            (r["name"] if hasattr(r, "keys") else r[1])
            for r in conn.execute("PRAGMA table_info(records)").fetchall()
        ]
        assert cols.count("role") == 1
    finally:
        conn.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_add_column_duplicate_of_create_time_column(driver, tmp_path):
    conn = _open(driver, tmp_path)
    try:
        conn.execute("CREATE TABLE records ( id TEXT , n INTEGER )")
        with pytest.raises(Exception) as exc_info:
            conn.execute("ALTER TABLE records ADD COLUMN n INTEGER")
        assert "duplicate column name" in str(exc_info.value).lower()
    finally:
        conn.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_migration_guard_pattern_is_idempotent(driver, tmp_path):
    """The exact except-shape used by the migration helpers no-ops a repeat."""
    conn = _open(driver, tmp_path)
    try:
        conn.execute("CREATE TABLE records ( id TEXT )")
        unexpected: list[Exception] = []
        for _ in range(3):
            try:
                conn.execute("ALTER TABLE records ADD COLUMN role TEXT")
            except Exception as exc:  # noqa: BLE001 -- mirrors migration guards
                if "duplicate column name" not in str(exc).lower():
                    unexpected.append(exc)
        assert not unexpected, f"[{driver}] unexpected errors: {unexpected}"
        cols = [
            (r["name"] if hasattr(r, "keys") else r[1])
            for r in conn.execute("PRAGMA table_info(records)").fetchall()
        ]
        assert cols.count("role") == 1
    finally:
        conn.close()
