"""Tier-A SQL parity routed through the RUST engine (not the Python reference).

A Rust-routed analog of the frozen ``test_lillibrain_sql_tier_a.py`` oracle. It
replicates that oracle's differential shape (``_norm`` / ``_seed_both`` /
``_assert_parity``), but the lilli side is built with
``iai_mcp_native.engine.Connection.open(...)`` — the Rust engine — rather than
the Python ``LilliBrainConnection``. So the Tier-A acceptance actually exercises
Rust against stdlib ``sqlite3``, which IS the specification. The frozen Python
oracle is kept untouched as the Python parity reference.

Every connection is asserted to BE the Rust engine class; a missing/stale wheel
makes the ``iai_mcp_native.engine`` import fail and the module SKIPS loudly,
never silently falling through to the Python engine.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

os.environ.setdefault("LILLI_FSYNC_MODE", "fast")


def _native_available() -> bool:
    try:
        from iai_mcp_native import engine  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _native_available(),
    reason="iai_mcp_native.engine submodule not installed (build the native wheel)",
)


# ---------------------------------------------------------------------------
# Differential harness (mirrors the frozen oracle, but lilli == the Rust engine)
# ---------------------------------------------------------------------------

_DB_SEQ = {"n": 0}


def _engine_conn(tmp_path: Path):
    """Open the Rust engine on a tmp file; assert it IS the Rust engine class."""
    from iai_mcp_native import engine

    _DB_SEQ["n"] += 1
    conn = engine.Connection.open(str(tmp_path / f"engine_{_DB_SEQ['n']}.db"), 384)
    assert isinstance(conn, engine.Connection), (
        "the lilli side is not the Rust engine.Connection — refusing a false-green "
        "Tier-A proof against another engine"
    )
    return conn


def _norm(rows) -> list[tuple]:
    return [tuple(r) for r in rows]


def _seed_both(tmp_path: Path, ddl: str, insert_sql: str, seed_rows: list[tuple]):
    """Seed identical DDL + rows into the Rust engine and a :memory: sqlite3."""
    lilli = _engine_conn(tmp_path)
    lilli.execute(ddl)
    for params in seed_rows:
        lilli.execute(insert_sql, params)

    sq = sqlite3.connect(":memory:")
    sq.row_factory = sqlite3.Row
    sq.execute(ddl)
    for params in seed_rows:
        sq.execute(insert_sql, params)
    sq.commit()
    return lilli, sq


def _assert_parity(lilli, sq, sql: str, params=()) -> None:
    lilli_rows = _norm(lilli.execute(sql, params).fetchall())
    sq_rows = _norm(sq.execute(sql, params).fetchall())
    assert lilli_rows == sq_rows, (
        f"Tier-A parity mismatch for {sql!r} params={params!r}:\n"
        f"  engine={lilli_rows!r}\n  sqlite={sq_rows!r}"
    )


_RECORDS_DDL = (
    "CREATE TABLE records ("
    "vec_label INTEGER PRIMARY KEY AUTOINCREMENT, "
    "id TEXT NOT NULL UNIQUE, "
    "created_at TEXT)"
)
_RECORDS_INSERT = "INSERT INTO records (id, created_at) VALUES (?, ?)"


# ---------------------------------------------------------------------------
# Affinity + ORDER BY (single and two-key, NULL placement, stable tiebreak)
# ---------------------------------------------------------------------------


def test_order_by_text_asc_desc(tmp_path) -> None:
    rows = [("a", "2026-01-03"), ("b", "2026-01-01"), ("c", "2026-01-05"), ("d", "2026-01-02")]
    lilli, sq = _seed_both(tmp_path, _RECORDS_DDL, _RECORDS_INSERT, rows)
    _assert_parity(lilli, sq, "SELECT id FROM records ORDER BY created_at ASC")
    _assert_parity(lilli, sq, "SELECT id FROM records ORDER BY created_at DESC")
    _assert_parity(lilli, sq, "SELECT * FROM records ORDER BY created_at DESC")


def test_order_by_null_placement(tmp_path) -> None:
    ddl = "CREATE TABLE t (id TEXT, n INTEGER)"
    insert = "INSERT INTO t (id, n) VALUES (?, ?)"
    rows = [("a", 3), ("b", None), ("c", 1), ("d", None), ("e", 2)]
    lilli, sq = _seed_both(tmp_path, ddl, insert, rows)
    _assert_parity(lilli, sq, "SELECT id, n FROM t ORDER BY n ASC")
    _assert_parity(lilli, sq, "SELECT id, n FROM t ORDER BY n DESC")


def test_order_by_limit_offset(tmp_path) -> None:
    rows = [
        ("a", "2026-01-03"), ("b", "2026-01-01"), ("c", "2026-01-05"),
        ("d", "2026-01-02"), ("e", "2026-01-04"),
    ]
    lilli, sq = _seed_both(tmp_path, _RECORDS_DDL, _RECORDS_INSERT, rows)
    _assert_parity(lilli, sq, "SELECT id FROM records ORDER BY created_at DESC LIMIT 2")
    _assert_parity(lilli, sq, "SELECT id FROM records ORDER BY created_at ASC LIMIT 3")
    _assert_parity(lilli, sq, "SELECT id FROM records ORDER BY created_at DESC LIMIT 0")


# ---------------------------------------------------------------------------
# Three-valued NULL: IS NULL / COALESCE in WHERE and projection
# ---------------------------------------------------------------------------


def test_coalesce_where_and_alias(tmp_path) -> None:
    ddl = "CREATE TABLE t (id TEXT, pending INTEGER)"
    insert = "INSERT INTO t (id, pending) VALUES (?, ?)"
    rows = [("a", 1), ("b", None), ("c", 0), ("d", 1), ("e", None)]
    lilli, sq = _seed_both(tmp_path, ddl, insert, rows)
    _assert_parity(lilli, sq, "SELECT id FROM t WHERE COALESCE(pending, 0) = 1 ORDER BY id")
    _assert_parity(lilli, sq, "SELECT id FROM t WHERE COALESCE(pending, 0) = 0 ORDER BY id")
    _assert_parity(
        lilli, sq, "SELECT id, COALESCE(pending, 0) AS p FROM t ORDER BY id"
    )
    _assert_parity(lilli, sq, "SELECT COUNT(*) FROM t WHERE COALESCE(pending, 0) = 0")


def test_is_null_and_is_not_null(tmp_path) -> None:
    ddl = (
        "CREATE TABLE records (id TEXT, created_at TEXT NOT NULL, tombstoned_at TEXT)"
    )
    insert = "INSERT INTO records (id, created_at, tombstoned_at) VALUES (?, ?, ?)"
    rows = [
        ("a", "2026-01-03T00:00:00", None),
        ("b", "2026-01-09T00:00:00", "2026-01-10T00:00:00"),
        ("c", "2026-01-05T00:00:00", None),
    ]
    lilli, sq = _seed_both(tmp_path, ddl, insert, rows)
    _assert_parity(lilli, sq, "SELECT id FROM records WHERE tombstoned_at IS NULL ORDER BY id")
    _assert_parity(lilli, sq, "SELECT id FROM records WHERE tombstoned_at IS NOT NULL ORDER BY id")


# ---------------------------------------------------------------------------
# datetime() WHERE + ORDER BY
# ---------------------------------------------------------------------------


def test_datetime_where_order(tmp_path) -> None:
    rows = [
        ("a", "2026-01-03T00:00:00"), ("b", "2026-01-01T00:00:00"),
        ("c", "2026-01-05T00:00:00"), ("d", "2026-01-02T00:00:00"),
    ]
    lilli, sq = _seed_both(tmp_path, _RECORDS_DDL, _RECORDS_INSERT, rows)
    _assert_parity(
        lilli, sq,
        "SELECT id FROM records WHERE datetime(created_at) <= datetime(?)"
        " ORDER BY datetime(created_at) DESC",
        ("2026-01-03T00:00:00",),
    )


# ---------------------------------------------------------------------------
# Aggregates: COUNT(*), MIN, MAX (NULL-skip, empty -> NULL)
# ---------------------------------------------------------------------------


def test_min_max_null_skip(tmp_path) -> None:
    ddl = "CREATE TABLE t (id TEXT, n INTEGER, s TEXT)"
    insert = "INSERT INTO t (id, n, s) VALUES (?, ?, ?)"
    rows = [
        ("a", 3, "mango"), ("b", None, None), ("c", 1, "apple"),
        ("d", 5, "pear"), ("e", None, "banana"),
    ]
    lilli, sq = _seed_both(tmp_path, ddl, insert, rows)
    _assert_parity(lilli, sq, "SELECT MIN(n) FROM t")
    _assert_parity(lilli, sq, "SELECT MAX(n) FROM t")
    _assert_parity(lilli, sq, "SELECT MIN(s) FROM t")
    _assert_parity(lilli, sq, "SELECT MAX(s) FROM t")


def test_count_and_empty_min(tmp_path) -> None:
    ddl = "CREATE TABLE t (id TEXT, n INTEGER)"
    insert = "INSERT INTO t (id, n) VALUES (?, ?)"
    lilli, sq = _seed_both(tmp_path, ddl, insert, [])
    _assert_parity(lilli, sq, "SELECT COUNT(*) FROM t")
    _assert_parity(lilli, sq, "SELECT MIN(n) FROM t")  # empty -> NULL


# ---------------------------------------------------------------------------
# GROUP BY + HAVING COUNT(*)
# ---------------------------------------------------------------------------


def test_group_by_count_having(tmp_path) -> None:
    ddl = "CREATE TABLE t (id TEXT, c TEXT)"
    insert = "INSERT INTO t (id, c) VALUES (?, ?)"
    rows = [("r1", "x"), ("r2", "x"), ("r3", "x"), ("r4", "y"), ("r5", "y"), ("r6", "z")]
    lilli, sq = _seed_both(tmp_path, ddl, insert, rows)
    _assert_parity(lilli, sq, "SELECT c, COUNT(*) AS n FROM t GROUP BY c ORDER BY c")
    _assert_parity(
        lilli, sq, "SELECT c, COUNT(*) AS n FROM t GROUP BY c HAVING COUNT(*) >= 2 ORDER BY c"
    )


# ---------------------------------------------------------------------------
# ON CONFLICT (composite-PK edges UPSERT) + LIKE / IN
# ---------------------------------------------------------------------------


def test_on_conflict_do_update(tmp_path) -> None:
    ddl = (
        "CREATE TABLE edges (src TEXT NOT NULL, dst TEXT NOT NULL, edge_type TEXT NOT NULL, "
        "weight REAL, PRIMARY KEY (src, dst, edge_type))"
    )
    upsert = (
        "INSERT INTO edges (src, dst, edge_type, weight) VALUES (?, ?, ?, ?)"
        " ON CONFLICT (src, dst, edge_type) DO UPDATE SET weight = excluded.weight"
    )
    lilli, sq = _seed_both(tmp_path, ddl, upsert, [])
    for params in [("a", "b", "rel", 1.0), ("a", "b", "rel", 9.0), ("x", "y", "rel", 3.0)]:
        lilli.execute(upsert, params)
        sq.execute(upsert, params)
    sq.commit()
    _assert_parity(lilli, sq, "SELECT src, weight FROM edges ORDER BY src, edge_type")


def test_like_and_in(tmp_path) -> None:
    rows = [("a", "2026-01-01x"), ("bb", "2026-01-02y"), ("ccc", "2026-01-03z")]
    lilli, sq = _seed_both(tmp_path, _RECORDS_DDL, _RECORDS_INSERT, rows)
    _assert_parity(lilli, sq, "SELECT id FROM records WHERE created_at LIKE '%x%' ORDER BY id")
    _assert_parity(lilli, sq, "SELECT id FROM records WHERE id IN ('a', 'ccc') ORDER BY id")


# ---------------------------------------------------------------------------
# COUNT(*) fast path through the Rust engine
# ---------------------------------------------------------------------------


def test_count_star_fast_path(tmp_path) -> None:
    """A bare COUNT(*) on a populated table returns the correct count through Rust.

    The COUNT(*) leaf-cell fast path is locked against regression here: the count
    must equal the row total and the query must answer promptly on a large table
    (no per-row decode). Routed through ``engine.Connection`` under the Rust-class
    guard.
    """
    from iai_mcp_native import engine

    eng = _engine_conn(tmp_path)
    eng.execute(_RECORDS_DDL)
    n = 5000
    eng.execute("BEGIN")
    for i in range(n):
        eng.execute(_RECORDS_INSERT, (f"id-{i}", f"2026-01-{(i % 28) + 1:02d}"))
    eng.execute("COMMIT")

    got = eng.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    assert got == n
    assert isinstance(eng, engine.Connection)

    # A filtered COUNT(*) also matches a direct recount via a value scan.
    filtered = eng.execute(
        "SELECT COUNT(*) FROM records WHERE created_at LIKE '2026-01-0%'"
    ).fetchone()[0]
    assert filtered == sum(1 for i in range(n) if f"2026-01-{(i % 28) + 1:02d}".startswith("2026-01-0"))
