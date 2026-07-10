from __future__ import annotations

from pathlib import Path

import pytest

from iai_mcp.lillibrain.connection import (
    LilliBrainConnection,
    LilliBrainCursor,
    _open_lilli_db,
)


def test_cursor_executemany_inserts_full_batch(tmp_path: Path) -> None:
    """conn.cursor().executemany(insert, N rows) inserts the whole batch.

    Builds a pure engine connection on a tmp file (no encryption, no ANN),
    creates a small table, runs a batch insert through the cursor delegate,
    then proves a follow-up COUNT(*) equals the batch size.
    """
    pager, catalog, root_map = _open_lilli_db(tmp_path / "engine.db")
    conn = LilliBrainConnection(pager, catalog, root_map)
    try:
        conn.execute("CREATE TABLE t (a TEXT, b INTEGER)")
        rows = [("alice", 1), ("bob", 2), ("carol", 3), ("dave", 4)]

        cur = conn.cursor()
        cur.executemany("INSERT INTO t (a, b) VALUES (?, ?)", rows)

        count = conn.execute("SELECT COUNT(*) AS n FROM t").fetchall()
        assert count[0]["n"] == len(rows)

        names = {
            r["a"] for r in conn.execute("SELECT a FROM t").fetchall()
        }
        assert names == {"alice", "bob", "carol", "dave"}
    finally:
        conn.close()


def test_cursor_executemany_returns_self(tmp_path: Path) -> None:
    """The executemany delegate returns the same cursor object (DB-API contract)."""
    pager, catalog, root_map = _open_lilli_db(tmp_path / "engine.db")
    conn = LilliBrainConnection(pager, catalog, root_map)
    try:
        conn.execute("CREATE TABLE t (a TEXT, b INTEGER)")
        cur = conn.cursor()
        returned = cur.executemany(
            "INSERT INTO t (a, b) VALUES (?, ?)", [("alice", 1), ("bob", 2)]
        )
        assert returned is cur
    finally:
        conn.close()


def test_unbound_cursor_executemany_raises() -> None:
    """An unbound cursor (no connection) raises rather than silently no-op'ing."""
    cur = LilliBrainCursor([], [])
    with pytest.raises(RuntimeError):
        cur.executemany("INSERT INTO t (a, b) VALUES (?, ?)", [("alice", 1)])
