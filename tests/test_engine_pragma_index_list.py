"""PRAGMA index_list parity: engine rows match the stdlib sqlite3 oracle.

Scope: explicitly created indexes only. sqlite3 additionally materialises
``sqlite_autoindex_*`` rows for UNIQUE column constraints; the engine enforces
uniqueness without autoindex rows, so those oracle rows are filtered before
comparison.
"""
from __future__ import annotations

import sqlite3

import pytest

_DDL = "CREATE TABLE records ( id TEXT , n INTEGER , label TEXT )"
_INDEXES = (
    "CREATE INDEX idx_n ON records ( n )",
    "CREATE UNIQUE INDEX idx_id ON records ( id )",
    "CREATE INDEX idx_part ON records ( label ) WHERE label = 'x'",
)


def _rows(conn) -> list[tuple]:
    fetched = conn.execute("PRAGMA index_list(records)").fetchall()
    return [tuple(r) for r in fetched]


def test_index_list_matches_sqlite3_oracle(tmp_path):
    pytest.importorskip("iai_mcp_native")
    from iai_mcp_native import engine  # noqa: PLC0415

    oracle = sqlite3.connect(":memory:")
    eng = engine.Connection.open(str(tmp_path / "t.lilli"), 0)
    try:
        for conn in (oracle, eng):
            conn.execute(_DDL)
            for ddl in _INDEXES:
                conn.execute(ddl)

        oracle_rows = [r for r in _rows(oracle) if not r[1].startswith("sqlite_autoindex")]
        engine_rows = _rows(eng)
        assert engine_rows == oracle_rows, (
            f"engine {engine_rows} != sqlite3 oracle {oracle_rows}"
        )
    finally:
        oracle.close()
        eng.close()


def test_index_list_unknown_table_empty(tmp_path):
    pytest.importorskip("iai_mcp_native")
    from iai_mcp_native import engine  # noqa: PLC0415

    eng = engine.Connection.open(str(tmp_path / "t.lilli"), 0)
    try:
        eng.execute(_DDL)
        assert eng.execute("PRAGMA index_list(nosuch)").fetchall() == []
    finally:
        eng.close()
