"""Differential parity tests for the vendored SQL engine's Tier-A correctness.

The vendored engine (``iai_mcp.lillibrain.sql``) must return byte-identical
rows-and-order to stdlib ``sqlite3`` on the same DDL + seed rows, because
stdlib ``sqlite3`` IS the specification for the supported SQL subset.

Each parity test seeds an identical schema and the identical rows into BOTH a
raw lilli engine connection (opened on a tmp file — pure engine, no encryption,
no ANN) and a fresh ``sqlite3.connect(":memory:")`` connection, runs the same
SQL on each, and asserts the normalised result rows are equal.
"""
from __future__ import annotations

import sqlite3

from iai_mcp import errors
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from iai_mcp.lillibrain.connection import LilliBrainConnection, _open_lilli_db
from iai_mcp.lillibrain.sql.ast_nodes import (
    BinOp,
    ColumnRef,
    FuncCall,
    NamedParam,
    SelectStmt,
)
from iai_mcp.lillibrain.sql.catalog import CatalogError
from iai_mcp.lillibrain.sql.lexer import TT, tokenize
from iai_mcp.lillibrain.sql.parser import ParseError, parse_sql


# ---------------------------------------------------------------------------
# Shared differential harness (reused by every Tier-A parity test)
# ---------------------------------------------------------------------------


def _norm(rows) -> list[tuple]:
    """Map engine / sqlite3 result rows to plain comparable tuples.

    Both ``LilliBrainRow`` and ``sqlite3.Row`` iterate values in column
    declaration order, and ``LilliBrainRow`` normalises SqlNull/None to None at
    construction, so ``tuple(row)`` yields directly comparable value tuples.
    """
    return [tuple(r) for r in rows]


def _seed_both(
    tmp_path: Path, ddl: str, insert_sql: str, seed_rows: list[tuple]
) -> tuple[LilliBrainConnection, sqlite3.Connection]:
    """Open a lilli engine connection and a :memory: sqlite3 connection seeded
    with the identical DDL + rows, and return both.

    ``ddl`` is a single CREATE TABLE statement; ``insert_sql`` is a single
    parameterised INSERT; ``seed_rows`` is the list of parameter tuples to
    insert into each driver in order.
    """
    pager, catalog, root_map = _open_lilli_db(tmp_path / "engine.db")
    lilli = LilliBrainConnection(pager, catalog, root_map)
    lilli.execute(ddl)
    for params in seed_rows:
        lilli.execute(insert_sql, params)

    sqlite_conn = sqlite3.connect(":memory:")
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_conn.execute(ddl)
    for params in seed_rows:
        sqlite_conn.execute(insert_sql, params)
    sqlite_conn.commit()

    return lilli, sqlite_conn


def _assert_parity(
    lilli_conn: LilliBrainConnection,
    sqlite_conn: sqlite3.Connection,
    sql: str,
    params: tuple = (),
) -> None:
    """Assert the lilli engine and sqlite3 return identical normalised rows."""
    lilli_rows = _norm(lilli_conn.execute(sql, params).fetchall())
    sqlite_rows = _norm(sqlite_conn.execute(sql, params).fetchall())
    assert lilli_rows == sqlite_rows, (
        f"parity mismatch for {sql!r} params={params!r}:\n"
        f"  lilli={lilli_rows!r}\n  sqlite={sqlite_rows!r}"
    )


def _assert_parity_named(
    lilli_conn: LilliBrainConnection,
    sqlite_conn: sqlite3.Connection,
    sql: str,
    params_dict: dict,
) -> None:
    """Assert lilli and sqlite3 return identical rows for a dict-bound query.

    sqlite3 binds a mapping param natively (``:name`` resolves from the dict by
    name); the engine must produce the byte-identical result for the same dict.
    """
    lilli_rows = _norm(lilli_conn.execute(sql, params_dict).fetchall())
    sqlite_rows = _norm(sqlite_conn.execute(sql, params_dict).fetchall())
    assert lilli_rows == sqlite_rows, (
        f"named-param parity mismatch for {sql!r} params={params_dict!r}:\n"
        f"  lilli={lilli_rows!r}\n  sqlite={sqlite_rows!r}"
    )


# A records-shaped schema covering the columns the consumer actually orders by.
_RECORDS_DDL = (
    "CREATE TABLE records ("
    "vec_label INTEGER PRIMARY KEY AUTOINCREMENT, "
    "id TEXT NOT NULL UNIQUE, "
    "created_at TEXT)"
)
_RECORDS_INSERT = "INSERT INTO records (id, created_at) VALUES (?, ?)"


# ---------------------------------------------------------------------------
# Parse-level guarantees (EOF assert + ORDER BY no longer dropped)
# ---------------------------------------------------------------------------


def test_trailing_tokens_raise() -> None:
    """A third ORDER BY key and any stray trailing tail fail closed, not silently."""
    # Two-key ORDER BY now parses; a THIRD comma-separated key is left
    # unconsumed and the EOF assert rejects it instead of silently dropping it.
    with pytest.raises(ParseError):
        parse_sql("SELECT id FROM records ORDER BY a, b, c")

    # A bare stray trailing token after an otherwise-complete statement raises.
    with pytest.raises(ParseError):
        parse_sql("SELECT id FROM records LIMIT 5 garbage")


def test_order_by_parsed_not_dropped() -> None:
    """ORDER BY + LIMIT parse into the AST — the LIMIT is no longer discarded."""
    stmt = parse_sql("SELECT id FROM records ORDER BY created_at DESC LIMIT 10")
    assert isinstance(stmt, SelectStmt)
    assert stmt.order_by_col == "created_at"
    assert stmt.order_by_desc is True
    assert stmt.limit == 10


def test_order_by_asc_default() -> None:
    """No direction keyword defaults to ASC; LIMIT absent stays None."""
    stmt = parse_sql("SELECT id FROM records ORDER BY created_at")
    assert isinstance(stmt, SelectStmt)
    assert stmt.order_by_col == "created_at"
    assert stmt.order_by_desc is False
    assert stmt.limit is None


def test_order_by_explicit_asc() -> None:
    """An explicit ASC parses with order_by_desc False."""
    stmt = parse_sql("SELECT id FROM records ORDER BY vec_label ASC")
    assert isinstance(stmt, SelectStmt)
    assert stmt.order_by_col == "vec_label"
    assert stmt.order_by_desc is False


# ---------------------------------------------------------------------------
# datetime(<expr>) scalar function — parse in WHERE and ORDER BY positions
# ---------------------------------------------------------------------------


def _find_funccall(node) -> "FuncCall | None":
    """Return the first FuncCall found in an expression tree, depth-first."""
    if isinstance(node, FuncCall):
        return node
    for child_attr in ("left", "right", "operand"):
        child = getattr(node, child_attr, None)
        if child is not None:
            found = _find_funccall(child)
            if found is not None:
                return found
    for list_attr in ("args", "items"):
        children = getattr(node, list_attr, None)
        if children:
            for child in children:
                found = _find_funccall(child)
                if found is not None:
                    return found
    return None


def test_datetime_in_where_parses() -> None:
    """``WHERE datetime(col) <= datetime(?)`` parses into a FuncCall over the column."""
    stmt = parse_sql(
        "SELECT id FROM records WHERE datetime(created_at) <= datetime(?)"
        " ORDER BY datetime(created_at) DESC LIMIT ?"
    )
    assert isinstance(stmt, SelectStmt)
    fc = _find_funccall(stmt.where)
    assert fc is not None
    assert fc.name == "datetime"
    assert isinstance(fc.args[0], ColumnRef)
    assert fc.args[0].name == "created_at"
    # The ORDER BY key carries the underlying column name plus the datetime flag.
    assert stmt.order_by_col == "created_at"
    assert stmt.order_by_desc is True
    assert stmt.order_by_datetime is True
    assert stmt.limit_is_param is True


def test_datetime_tombstone_disjunction_parses() -> None:
    """The tombstone disjunction with a nested datetime(?) parses without error."""
    stmt = parse_sql(
        "SELECT id FROM records WHERE"
        " (tombstoned_at IS NULL OR datetime(tombstoned_at) > datetime(?))"
    )
    assert isinstance(stmt, SelectStmt)
    fc = _find_funccall(stmt.where)
    assert fc is not None and fc.name == "datetime"


def test_plain_order_by_not_datetime_flagged() -> None:
    """A plain ``ORDER BY <col>`` leaves the datetime flag unset."""
    stmt = parse_sql("SELECT id FROM records ORDER BY created_at DESC")
    assert isinstance(stmt, SelectStmt)
    assert stmt.order_by_col == "created_at"
    assert stmt.order_by_datetime is False


def test_datetime_deep_nesting_trips_depth_guard() -> None:
    """A pathologically nested datetime(datetime(...)) trips the nesting bound."""
    expr = "created_at"
    for _ in range(60):
        expr = f"datetime({expr})"
    with pytest.raises(ParseError):
        parse_sql(f"SELECT id FROM records WHERE {expr} <= datetime(?)")


def test_datetime_wrong_arity_rejected() -> None:
    """``datetime(a, b)`` (two args) fails closed rather than silently truncating."""
    with pytest.raises(ParseError):
        parse_sql("SELECT id FROM records WHERE datetime(created_at, updated_at) <= ?")


# ---------------------------------------------------------------------------
# ORDER BY result parity (executor sort vs sqlite3)
# ---------------------------------------------------------------------------


def test_order_by_matches_sqlite(tmp_path) -> None:
    """ASC and DESC on a TEXT column return sqlite3's row order on both drivers."""
    rows = [
        ("a", "2026-01-03"),
        ("b", "2026-01-01"),
        ("c", "2026-01-05"),
        ("d", "2026-01-02"),
    ]
    lilli, sq = _seed_both(tmp_path, _RECORDS_DDL, _RECORDS_INSERT, rows)
    try:
        _assert_parity(lilli, sq, "SELECT id FROM records ORDER BY created_at ASC")
        _assert_parity(lilli, sq, "SELECT id FROM records ORDER BY created_at DESC")
        # SELECT * carries the sort column in the projection and still matches.
        _assert_parity(lilli, sq, "SELECT * FROM records ORDER BY created_at DESC")
    finally:
        lilli.close()
        sq.close()


def test_order_by_null_ordering(tmp_path) -> None:
    """A nullable column sorts NULLs FIRST on ASC and LAST on DESC, like sqlite3."""
    ddl = "CREATE TABLE t (id TEXT, n INTEGER)"
    insert = "INSERT INTO t (id, n) VALUES (?, ?)"
    rows = [
        ("a", 3),
        ("b", None),
        ("c", 1),
        ("d", None),
        ("e", 2),
    ]
    lilli, sq = _seed_both(tmp_path, ddl, insert, rows)
    try:
        _assert_parity(lilli, sq, "SELECT id, n FROM t ORDER BY n ASC")
        _assert_parity(lilli, sq, "SELECT id, n FROM t ORDER BY n DESC")
    finally:
        lilli.close()
        sq.close()


def test_order_by_limit_and_unprojected(tmp_path) -> None:
    """ORDER BY a non-projected column with LIMIT sorts before the limit (== sqlite3)."""
    rows = [
        ("a", "2026-01-03"),
        ("b", "2026-01-01"),
        ("c", "2026-01-05"),
        ("d", "2026-01-02"),
        ("e", "2026-01-04"),
    ]
    lilli, sq = _seed_both(tmp_path, _RECORDS_DDL, _RECORDS_INSERT, rows)
    try:
        # created_at is NOT in the projection — the projected-scan path must still
        # decode it for the sort, and the LIMIT must apply AFTER the sort.
        _assert_parity(
            lilli, sq,
            "SELECT id FROM records ORDER BY created_at DESC LIMIT 2",
        )
        _assert_parity(
            lilli, sq,
            "SELECT id FROM records ORDER BY created_at ASC LIMIT 3",
        )
        # LIMIT 0 yields no rows on both drivers.
        _assert_parity(
            lilli, sq,
            "SELECT id FROM records ORDER BY created_at DESC LIMIT 0",
        )
    finally:
        lilli.close()
        sq.close()


def test_order_by_stable_tiebreak(tmp_path) -> None:
    """Rows with duplicate sort values keep insertion order (stable), like sqlite3."""
    rows = [
        ("a", "2026-01-01"),
        ("b", "2026-01-01"),
        ("c", "2026-01-02"),
        ("d", "2026-01-01"),
        ("e", "2026-01-02"),
    ]
    lilli, sq = _seed_both(tmp_path, _RECORDS_DDL, _RECORDS_INSERT, rows)
    try:
        _assert_parity(lilli, sq, "SELECT id FROM records ORDER BY created_at ASC")
        _assert_parity(lilli, sq, "SELECT id FROM records ORDER BY created_at DESC")
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# MIN(col) / MAX(col) aggregates (storage-class order, NULL-skip, empty -> NULL)
# ---------------------------------------------------------------------------


def test_min_max_matches_sqlite(tmp_path) -> None:
    """MIN/MAX on INT and TEXT columns match sqlite3 (value AND label), NULL-skip."""
    ddl = "CREATE TABLE t (id TEXT, n INTEGER, s TEXT)"
    insert = "INSERT INTO t (id, n, s) VALUES (?, ?, ?)"
    rows = [
        ("a", 3, "mango"),
        ("b", None, None),
        ("c", 1, "apple"),
        ("d", 5, "pear"),
        ("e", None, "banana"),
    ]
    lilli, sq = _seed_both(tmp_path, ddl, insert, rows)
    try:
        # INTEGER column: min and max ignore NULL rows.
        _assert_parity(lilli, sq, "SELECT MIN(n) FROM t")
        _assert_parity(lilli, sq, "SELECT MAX(n) FROM t")
        # TEXT column: storage-class compare orders lexically.
        _assert_parity(lilli, sq, "SELECT MIN(s) FROM t")
        _assert_parity(lilli, sq, "SELECT MAX(s) FROM t")
        # The default column label is MIN(col) / MAX(col) — assert it directly.
        cur = lilli.execute("SELECT MAX(n) FROM t")
        assert cur.description[0][0] == "MAX(n)"
        # LIMIT 0 yields no rows but keeps the aggregate label.
        _assert_parity(lilli, sq, "SELECT MAX(n) FROM t LIMIT 0")
        _assert_parity(lilli, sq, "SELECT MIN(s) FROM t LIMIT 0")
    finally:
        lilli.close()
        sq.close()


def test_min_max_empty_and_all_null(tmp_path) -> None:
    """MIN/MAX over an empty set or an all-NULL column return a single NULL row."""
    ddl = "CREATE TABLE t (id TEXT, n INTEGER)"
    insert = "INSERT INTO t (id, n) VALUES (?, ?)"
    # Empty table: MIN/MAX return one row whose value is NULL.
    lilli, sq = _seed_both(tmp_path, ddl, insert, [])
    try:
        _assert_parity(lilli, sq, "SELECT MIN(n) FROM t")
        _assert_parity(lilli, sq, "SELECT MAX(n) FROM t")
    finally:
        lilli.close()
        sq.close()
    # All-NULL column: same — every value skipped, result NULL.
    all_null = [("a", None), ("b", None), ("c", None)]
    lilli, sq = _seed_both(tmp_path, ddl, insert, all_null)
    try:
        _assert_parity(lilli, sq, "SELECT MIN(n) FROM t")
        _assert_parity(lilli, sq, "SELECT MAX(n) FROM t")
    finally:
        lilli.close()
        sq.close()


def test_max_created_at_parity(tmp_path) -> None:
    """The consumer's real form: MAX(created_at) WHERE tombstoned_at IS NULL."""
    ddl = (
        "CREATE TABLE records ("
        "vec_label INTEGER PRIMARY KEY AUTOINCREMENT, "
        "id TEXT NOT NULL UNIQUE, "
        "created_at TEXT NOT NULL, "
        "tombstoned_at TEXT)"
    )
    insert = (
        "INSERT INTO records (id, created_at, tombstoned_at) VALUES (?, ?, ?)"
    )
    rows = [
        ("a", "2026-01-03T00:00:00", None),
        ("b", "2026-01-09T00:00:00", "2026-01-10T00:00:00"),  # tombstoned — excluded
        ("c", "2026-01-05T00:00:00", None),
        ("d", "2026-01-01T00:00:00", None),
    ]
    lilli, sq = _seed_both(tmp_path, ddl, insert, rows)
    try:
        _assert_parity(
            lilli, sq,
            "SELECT MAX(created_at) FROM records WHERE tombstoned_at IS NULL",
        )
        # row[0] positional access returns the newest non-tombstoned created_at.
        got = lilli.execute(
            "SELECT MAX(created_at) FROM records WHERE tombstoned_at IS NULL"
        ).fetchone()
        assert got[0] == "2026-01-05T00:00:00"
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# rowid -> vec_label resolution (selectable + orderable; HWM, never the btree key)
# ---------------------------------------------------------------------------


def test_rowid_select_matches_sqlite(tmp_path) -> None:
    """SELECT rowid (and rowid, id) surface rowid == vec_label, equal to sqlite3."""
    rows = [
        ("a", "2026-01-03"),
        ("b", "2026-01-01"),
        ("c", "2026-01-05"),
    ]
    lilli, sq = _seed_both(tmp_path, _RECORDS_DDL, _RECORDS_INSERT, rows)
    try:
        _assert_parity(lilli, sq, "SELECT rowid FROM records")
        _assert_parity(lilli, sq, "SELECT rowid, id FROM records")
        # rowid equals the row's vec_label on a fresh insert sequence (1..3).
        _assert_parity(lilli, sq, "SELECT rowid, vec_label FROM records")
    finally:
        lilli.close()
        sq.close()


def test_rowid_keyset_matches_sqlite(tmp_path) -> None:
    """ORDER BY rowid [DESC] LIMIT n keyset returns sqlite3's vec_label order."""
    rows = [
        ("a", "2026-01-03"),
        ("b", "2026-01-01"),
        ("c", "2026-01-05"),
        ("d", "2026-01-02"),
    ]
    lilli, sq = _seed_both(tmp_path, _RECORDS_DDL, _RECORDS_INSERT, rows)
    try:
        _assert_parity(
            lilli, sq, "SELECT id FROM records ORDER BY rowid DESC LIMIT 2"
        )
        _assert_parity(
            lilli, sq, "SELECT id FROM records ORDER BY rowid ASC LIMIT 3"
        )
        # rowid both selected AND ordered — the column must decode on the
        # projected-scan path (it is not in the catalog by name).
        _assert_parity(
            lilli, sq, "SELECT id, rowid FROM records ORDER BY rowid DESC"
        )
    finally:
        lilli.close()
        sq.close()


def test_rowid_after_delete_insert(tmp_path) -> None:
    """After INSERT a,b,c / DELETE c / INSERT d, rowid == HWM vec_label (not reused key)."""
    rows = [
        ("a", "2026-01-03"),
        ("b", "2026-01-01"),
        ("c", "2026-01-05"),
    ]
    lilli, sq = _seed_both(tmp_path, _RECORDS_DDL, _RECORDS_INSERT, rows)
    try:
        # Run the identical churn on BOTH drivers: drop the highest-key row,
        # then insert a new one. sqlite3 AUTOINCREMENT and the engine HWM both
        # assign the new row rowid/vec_label = 4 (monotonic), NOT the reused 3.
        for conn in (lilli, sq):
            conn.execute("DELETE FROM records WHERE id = ?", ("c",))
            conn.execute(_RECORDS_INSERT, ("d", "2026-01-02"))
        sq.commit()
        _assert_parity(lilli, sq, "SELECT id, rowid FROM records ORDER BY rowid")
        _assert_parity(
            lilli, sq, "SELECT id, rowid FROM records ORDER BY rowid DESC LIMIT 1"
        )
        # d's rowid is the never-reused HWM (4), proving rowid != the btree key.
        got = lilli.execute(
            "SELECT rowid FROM records ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        assert got[0] == 4
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# COALESCE(col, literal) in WHERE (parses + filters, no silent zero-row drop)
# ---------------------------------------------------------------------------


def test_coalesce_where_matches_sqlite(tmp_path) -> None:
    """``WHERE COALESCE(col, lit) = ?`` returns the right rows (== sqlite3), not [].

    Covers both a row WITH a non-null value and a row WITH a NULL value, so the
    fallback path (NULL → literal) is exercised against the oracle.
    """
    ddl = "CREATE TABLE t (id TEXT, pending INTEGER)"
    insert = "INSERT INTO t (id, pending) VALUES (?, ?)"
    rows = [
        ("a", 1),        # non-null, matches = 1
        ("b", None),     # null → COALESCE fallback 0, matches = 0
        ("c", 0),        # non-null zero, matches = 0
        ("d", 1),        # non-null, matches = 1
        ("e", None),     # null → fallback 0, matches = 0
    ]
    lilli, sq = _seed_both(tmp_path, ddl, insert, rows)
    try:
        # Literal comparison value.
        _assert_parity(
            lilli, sq,
            "SELECT id FROM t WHERE COALESCE(pending, 0) = 1",
        )
        _assert_parity(
            lilli, sq,
            "SELECT id FROM t WHERE COALESCE(pending, 0) = 0",
        )
        # Parameterised comparison value — the '?' after the COALESCE binds.
        _assert_parity(
            lilli, sq,
            "SELECT id FROM t WHERE COALESCE(pending, 0) = ?",
            (1,),
        )
        # COUNT(*) over the COALESCE predicate (the consumer's boot query shape).
        _assert_parity(
            lilli, sq,
            "SELECT COUNT(*) FROM t WHERE COALESCE(pending, 0) = 0",
        )
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# COALESCE(col, literal) AS alias in the SELECT list (projection-alias path)
# ---------------------------------------------------------------------------


def test_coalesce_as_alias(tmp_path) -> None:
    """``SELECT COALESCE(col, lit) AS name`` projects under the alias (== sqlite3).

    Covers the value (NULL → fallback, non-NULL passthrough) AND the output
    column label, which the consumer reads by name from the alias.
    """
    ddl = "CREATE TABLE t (id TEXT, pending INTEGER)"
    insert = "INSERT INTO t (id, pending) VALUES (?, ?)"
    rows = [
        ("a", 1),
        ("b", None),   # NULL → fallback 0 under the alias
        ("c", 0),
    ]
    lilli, sq = _seed_both(tmp_path, ddl, insert, rows)
    try:
        sql = (
            "SELECT id, COALESCE(pending, 0) AS pending_flag FROM t"
        )
        _assert_parity(lilli, sq, sql)
        # The output column label is the alias on both drivers.
        lilli_desc = [d[0] for d in lilli.execute(sql).description]
        sqlite_desc = [d[0] for d in sq.execute(sql).description]
        assert lilli_desc == sqlite_desc == ["id", "pending_flag"]
        # COALESCE alias as the SOLE projection (no leading plain column).
        _assert_parity(
            lilli, sq,
            "SELECT COALESCE(pending, 0) AS pending_flag FROM t",
        )
    finally:
        lilli.close()
        sq.close()


def test_select_bare_column_alias(tmp_path) -> None:
    """``SELECT col AS alias`` projects a plain column under the alias (== sqlite3)."""
    rows = [
        ("a", "2026-01-03"),
        ("b", "2026-01-01"),
    ]
    lilli, sq = _seed_both(tmp_path, _RECORDS_DDL, _RECORDS_INSERT, rows)
    try:
        sql = "SELECT id AS record_id, created_at AS made_at FROM records"
        _assert_parity(lilli, sq, sql)
        lilli_desc = [d[0] for d in lilli.execute(sql).description]
        sqlite_desc = [d[0] for d in sq.execute(sql).description]
        assert lilli_desc == sqlite_desc == ["record_id", "made_at"]
        # A mixed list: one plain column, one aliased — source order preserved.
        _assert_parity(
            lilli, sq,
            "SELECT id, created_at AS made_at FROM records",
        )
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# End-to-end real-consumer parity: ORDER BY + rowid + COALESCE composed on the
# exact query shapes the store / recall paths issue per cycle.
# ---------------------------------------------------------------------------


# A records-shaped schema covering the columns the consumer queries touch. The
# real read projects 24 plain columns + COALESCE(embedding_pending, 0) AS
# embedding_pending as the 25th; this subset keeps the columns these shapes
# reference (vec_label PK, id, tier, created_at, tombstoned_at,
# embedding_pending) so the COALESCE-in-WHERE, COALESCE-AS, ORDER BY, and rowid
# behaviour is exercised end-to-end against the oracle.
_CONSUMER_DDL = (
    "CREATE TABLE records ("
    "vec_label INTEGER PRIMARY KEY AUTOINCREMENT, "
    "id TEXT NOT NULL UNIQUE, "
    "tier TEXT, "
    "created_at TEXT, "
    "tombstoned_at TEXT, "
    "embedding_pending INTEGER)"
)
_CONSUMER_INSERT = (
    "INSERT INTO records (id, tier, created_at, tombstoned_at, embedding_pending)"
    " VALUES (?, ?, ?, ?, ?)"
)
# Mix: pending=1, pending=0, pending=NULL (COALESCE fallback), one tombstoned,
# and two equal created_at values (stable-tiebreak exercise).
_CONSUMER_ROWS = [
    ("a", "episodic", "2026-01-03T00:00:00", None, 1),
    ("b", "episodic", "2026-01-01T00:00:00", None, None),    # NULL pending
    ("c", "semantic", "2026-01-05T00:00:00", "2026-01-06T00:00:00", 0),  # tombstoned
    ("d", "episodic", "2026-01-02T00:00:00", None, 1),
    ("e", "episodic", "2026-01-01T00:00:00", None, 0),       # equal created_at to b
]

# The real consumer projection's leading columns + the COALESCE-AS 25th column,
# trimmed to the seeded schema. Keeps the mixed plain + aliased shape so source
# column order (positional access) is asserted against sqlite3.
_CONSUMER_COLS = (
    "id, tier, created_at, tombstoned_at,"
    " COALESCE(embedding_pending, 0) AS embedding_pending"
)


def test_keyset_pending_parity(tmp_path) -> None:
    """Keyset-pending: COALESCE-in-WHERE + COALESCE-AS + ORDER BY rowid DESC LIMIT ?."""
    lilli, sq = _seed_both(
        tmp_path, _CONSUMER_DDL, _CONSUMER_INSERT, _CONSUMER_ROWS
    )
    try:
        sql = (
            f"SELECT {_CONSUMER_COLS} FROM records"
            " WHERE COALESCE(embedding_pending, 0) = 1"
            " ORDER BY rowid DESC LIMIT ?"
        )
        _assert_parity(lilli, sq, sql, (10,))
        # LIMIT smaller than the match count keeps the top-N of the sorted set.
        _assert_parity(lilli, sq, sql, (1,))
        # Positional + by-name access on the lilli rows matches sqlite3's shape:
        # row[0] is id (TEXT), the last column is the COALESCE alias (INTEGER 1).
        lilli_first = lilli.execute(sql, (10,)).fetchone()
        sqlite_first = sq.execute(sql, (10,)).fetchone()
        assert lilli_first[0] == sqlite_first[0]              # positional id
        assert (
            lilli_first["embedding_pending"]
            == sqlite_first["embedding_pending"]
            == 1
        )
        # rowid DESC returns the highest vec_label first among pending rows (d > a).
        assert lilli_first[0] == "d"
    finally:
        lilli.close()
        sq.close()


def test_recency_parity(tmp_path) -> None:
    """Recency: COALESCE-AS + WHERE tombstoned_at IS NULL + ORDER BY created_at DESC."""
    lilli, sq = _seed_both(
        tmp_path, _CONSUMER_DDL, _CONSUMER_INSERT, _CONSUMER_ROWS
    )
    try:
        sql = (
            f"SELECT {_CONSUMER_COLS} FROM records"
            " WHERE tombstoned_at IS NULL ORDER BY created_at DESC"
        )
        _assert_parity(lilli, sq, sql)
        # The LIMIT ? recency variant the daemon-down path issues.
        _assert_parity(
            lilli, sq,
            sql + " LIMIT ?",
            (3,),
        )
        # The output column label is the COALESCE alias on both drivers.
        lilli_desc = [d[0] for d in lilli.execute(sql).description]
        sqlite_desc = [d[0] for d in sq.execute(sql).description]
        assert lilli_desc == sqlite_desc
        assert lilli_desc[-1] == "embedding_pending"
    finally:
        lilli.close()
        sq.close()


def test_reembed_parity(tmp_path) -> None:
    """Reembed: COALESCE-in-WHERE + AND tombstoned_at IS NULL + ORDER BY rowid."""
    lilli, sq = _seed_both(
        tmp_path, _CONSUMER_DDL, _CONSUMER_INSERT, _CONSUMER_ROWS
    )
    try:
        _assert_parity(
            lilli, sq,
            "SELECT id FROM records"
            " WHERE COALESCE(embedding_pending, 0) = 1 AND tombstoned_at IS NULL"
            " ORDER BY rowid",
        )
    finally:
        lilli.close()
        sq.close()


def test_consumer_forms_after_delete_insert(tmp_path) -> None:
    """Keyset + recency survive an identical DELETE+INSERT churn on both drivers.

    Proves the rowid keyset (vec_label high-water mark, never the reused btree
    key) and the COALESCE projection still match sqlite3 after the row set
    changes — the per-cycle reality the consumer runs against.
    """
    lilli, sq = _seed_both(
        tmp_path, _CONSUMER_DDL, _CONSUMER_INSERT, _CONSUMER_ROWS
    )
    try:
        # Identical churn on both connections: drop one pending row, insert a new
        # pending row. sqlite3 AUTOINCREMENT and the engine HWM both assign the
        # new row the never-reused next vec_label.
        for conn in (lilli, sq):
            conn.execute("DELETE FROM records WHERE id = ?", ("a",))
            conn.execute(
                _CONSUMER_INSERT, ("f", "episodic", "2026-01-07T00:00:00", None, 1)
            )
        sq.commit()

        keyset = (
            f"SELECT {_CONSUMER_COLS} FROM records"
            " WHERE COALESCE(embedding_pending, 0) = 1"
            " ORDER BY rowid DESC LIMIT ?"
        )
        _assert_parity(lilli, sq, keyset, (10,))
        # The newest pending row (f) has the high-water vec_label and sorts first.
        assert lilli.execute(keyset, (10,)).fetchone()[0] == "f"

        recency = (
            f"SELECT {_CONSUMER_COLS} FROM records"
            " WHERE tombstoned_at IS NULL ORDER BY created_at DESC"
        )
        _assert_parity(lilli, sq, recency)
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# Single-column ORDER BY on a non-records table (the events read shape)
# ---------------------------------------------------------------------------


def test_events_order_by_parity(tmp_path) -> None:
    """``SELECT id FROM events ORDER BY ts DESC LIMIT ?`` matches sqlite3.

    A minimal two-column table with no INTEGER PRIMARY KEY exercises the
    single-column ORDER BY + LIMIT ? path on an ordinary (non-records) table,
    the shape the event read issues.
    """
    ddl = "CREATE TABLE events (id TEXT, ts TEXT)"
    insert = "INSERT INTO events (id, ts) VALUES (?, ?)"
    rows = [
        ("e1", "2026-01-03T00:00:00"),
        ("e2", "2026-01-01T00:00:00"),
        ("e3", "2026-01-05T00:00:00"),
        ("e4", "2026-01-02T00:00:00"),
        ("e5", "2026-01-05T00:00:00"),   # equal ts to e3 — stable tiebreak
    ]
    lilli, sq = _seed_both(tmp_path, ddl, insert, rows)
    try:
        _assert_parity(
            lilli, sq, "SELECT id FROM events ORDER BY ts DESC LIMIT ?", (3,)
        )
        _assert_parity(
            lilli, sq, "SELECT id FROM events ORDER BY ts DESC LIMIT ?", (10,)
        )
        # The WHERE ts > ? + ORDER BY ts DESC + LIMIT ? full event-read shape.
        _assert_parity(
            lilli, sq,
            "SELECT id FROM events WHERE ts > ? ORDER BY ts DESC LIMIT ?",
            ("2026-01-01T00:00:00", 10),
        )
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# Named parameter (:name) lexer + parser + bind parity
# ---------------------------------------------------------------------------


def test_named_param_tokenizes() -> None:
    """``:pat`` lexes as a single NAMED_PARAM token carrying the ':pat' text."""
    toks = tokenize("WHERE tags_json LIKE :pat")
    named = [t for t in toks if t.type == TT.NAMED_PARAM]
    assert len(named) == 1
    assert named[0].value == ":pat"


def test_named_param_underscore_and_digits_tokenize() -> None:
    """An identifier-charset name (``:a_b1``) lexes as one NAMED_PARAM token."""
    toks = tokenize("SELECT id FROM records WHERE created_at LIKE :a_b1")
    named = [t for t in toks if t.type == TT.NAMED_PARAM]
    assert len(named) == 1
    assert named[0].value == ":a_b1"


def test_named_param_bare_colon_rejected() -> None:
    """A bare ``:`` with no name body fails closed at the scanner remainder."""
    with pytest.raises(SyntaxError):
        tokenize(":")


def test_named_param_parses_to_node() -> None:
    """The WHERE right-hand side of a ``:name`` predicate is a NamedParam node."""
    stmt = parse_sql(
        "SELECT id FROM records WHERE tags_json LIKE :pat"
    )
    assert isinstance(stmt, SelectStmt)
    assert isinstance(stmt.where, BinOp)
    assert stmt.where.op == "LIKE"
    assert isinstance(stmt.where.right, NamedParam)
    assert stmt.where.right.name == "pat"


def test_named_param_like_parity(tmp_path: Path) -> None:
    """``WHERE col LIKE :pat`` dict-bound returns rows byte-identical to sqlite3."""
    lilli, sq = _seed_both(
        tmp_path, _RECORDS_DDL, _RECORDS_INSERT,
        [("a", "2026-01-01T00:00:00x"), ("bb", "2026-01-02T00:00:00y")],
    )
    try:
        _assert_parity_named(
            lilli, sq,
            "SELECT id FROM records WHERE created_at LIKE :pat",
            {"pat": "%x%"},
        )
        # A pattern matching nothing returns the empty set on both.
        _assert_parity_named(
            lilli, sq,
            "SELECT id FROM records WHERE created_at LIKE :pat",
            {"pat": "%zzz%"},
        )
    finally:
        lilli.close()
        sq.close()


def test_named_param_two_keys(tmp_path: Path) -> None:
    """Two named params in one WHERE both bind from the dict by name."""
    lilli, sq = _seed_both(
        tmp_path, _RECORDS_DDL, _RECORDS_INSERT,
        [("a", "2026-01-01T00:00:00x"), ("bb", "2026-01-02T00:00:00y")],
    )
    try:
        _assert_parity_named(
            lilli, sq,
            "SELECT id FROM records WHERE created_at LIKE :pat AND id = :who",
            {"pat": "%x%", "who": "a"},
        )
    finally:
        lilli.close()
        sq.close()


def test_named_param_tuple_rejected(tmp_path: Path) -> None:
    """A named query supplied a tuple param raises on the engine (fail-closed).

    stdlib sqlite3 on Python 3.12 only emits a DeprecationWarning here (it
    raises starting 3.14); the engine raises now, which is the stricter,
    forward-compatible behaviour the named-param contract requires.
    """
    lilli, sq = _seed_both(
        tmp_path, _RECORDS_DDL, _RECORDS_INSERT,
        [("a", "2026-01-01T00:00:00x")],
    )
    try:
        sql = "SELECT id FROM records WHERE created_at LIKE :pat"
        with pytest.raises(Exception):
            lilli.execute(sql, ("%x%",)).fetchall()
    finally:
        lilli.close()
        sq.close()


def test_named_param_mixing_rejected(tmp_path: Path) -> None:
    """Mixing ``:name`` and positional ``?`` in one statement raises on lilli."""
    lilli, sq = _seed_both(
        tmp_path, _RECORDS_DDL, _RECORDS_INSERT,
        [("a", "2026-01-01T00:00:00x")],
    )
    try:
        sql = "SELECT id FROM records WHERE created_at LIKE :pat AND id = ?"
        with pytest.raises(Exception):
            sq.execute(sql, {"pat": "%x%", "1": "a"}).fetchall()
        with pytest.raises(Exception):
            lilli.execute(sql, {"pat": "%x%"}).fetchall()
    finally:
        lilli.close()
        sq.close()


def test_named_param_missing_key_rejected(tmp_path: Path) -> None:
    """A dict missing a referenced key raises a clear bind error (no silent None)."""
    lilli, sq = _seed_both(
        tmp_path, _RECORDS_DDL, _RECORDS_INSERT,
        [("a", "2026-01-01T00:00:00x")],
    )
    try:
        sql = "SELECT id FROM records WHERE created_at LIKE :missing"
        with pytest.raises(Exception):
            sq.execute(sql, {"pat": "%x%"}).fetchall()
        with pytest.raises(Exception):
            lilli.execute(sql, {"pat": "%x%"}).fetchall()
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# Hex-BLOB literal (x'...') lexer + parser + INSERT-store parity
# ---------------------------------------------------------------------------

_BLOB_DDL = "CREATE TABLE t (id TEXT, b BLOB)"


def test_blob_literal_tokenizes() -> None:
    """``x''`` lexes as ONE TT.BLOB token, not IDENT('x') + STRING('')."""
    toks = tokenize("INSERT INTO t (id, b) VALUES ('a', x'')")
    blobs = [t for t in toks if t.type == TT.BLOB]
    assert len(blobs) == 1
    assert blobs[0].value == "x''"
    # No stray IDENT 'x' leaked through ahead of the empty STRING.
    assert all(t.value != "x" for t in toks if t.type == TT.IDENT)


def test_blob_literal_hex_tokenizes() -> None:
    """``x'48656c6c6f'`` lexes as one TT.BLOB token carrying the full literal."""
    toks = tokenize("INSERT INTO t (id, b) VALUES ('a', x'48656c6c6f')")
    blobs = [t for t in toks if t.type == TT.BLOB]
    assert len(blobs) == 1
    assert blobs[0].value == "x'48656c6c6f'"


def test_blob_literal_uppercase_x_tokenizes() -> None:
    """An uppercase ``X'00FF'`` literal also lexes as one TT.BLOB token."""
    toks = tokenize("INSERT INTO t (id, b) VALUES ('a', X'00FF')")
    blobs = [t for t in toks if t.type == TT.BLOB]
    assert len(blobs) == 1
    assert blobs[0].value == "X'00FF'"


def test_blob_literal_empty_parity(tmp_path: Path) -> None:
    """``x''`` in INSERT VALUES stores b'' on lilli, byte-identical to sqlite3."""
    insert = "INSERT INTO t (id, b) VALUES ('a', x'')"
    lilli, sq = _seed_both(tmp_path, _BLOB_DDL, insert, [])
    try:
        lilli.execute(insert)
        sq.execute(insert)
        sq.commit()
        _assert_parity(lilli, sq, "SELECT id, b FROM t")
        got = lilli.execute("SELECT b FROM t").fetchone()
        assert got[0] == b""
    finally:
        lilli.close()
        sq.close()


def test_blob_literal_hex_parity(tmp_path: Path) -> None:
    """``x'48656c6c6f'`` stores b'Hello' (5 bytes) on lilli == sqlite3."""
    insert = "INSERT INTO t (id, b) VALUES ('a', x'48656c6c6f')"
    lilli, sq = _seed_both(tmp_path, _BLOB_DDL, insert, [])
    try:
        lilli.execute(insert)
        sq.execute(insert)
        sq.commit()
        _assert_parity(lilli, sq, "SELECT id, b FROM t")
        got = lilli.execute("SELECT b FROM t").fetchone()
        assert got[0] == b"Hello"
        assert len(got[0]) == 5
    finally:
        lilli.close()
        sq.close()


def test_blob_literal_binary_parity(tmp_path: Path) -> None:
    """``x'00ff'`` stores the non-UTF8 bytes b'\\x00\\xff' on lilli == sqlite3."""
    insert = "INSERT INTO t (id, b) VALUES ('a', x'00ff')"
    lilli, sq = _seed_both(tmp_path, _BLOB_DDL, insert, [])
    try:
        lilli.execute(insert)
        sq.execute(insert)
        sq.commit()
        _assert_parity(lilli, sq, "SELECT id, b FROM t")
        got = lilli.execute("SELECT b FROM t").fetchone()
        assert got[0] == b"\x00\xff"
    finally:
        lilli.close()
        sq.close()


def test_blob_literal_odd_length_fails_closed(tmp_path: Path) -> None:
    """A malformed odd-length hex body (``x'0'``) fails closed, not silently accepted."""
    insert = "INSERT INTO t (id, b) VALUES ('a', x'0')"
    lilli, sq = _seed_both(tmp_path, _BLOB_DDL, insert, [])
    try:
        with pytest.raises(Exception):
            lilli.execute(insert)
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# GROUP BY + COUNT(*) [AS n] + HAVING (one row per group, grouped after WHERE)
# ---------------------------------------------------------------------------

_GROUP_DDL = "CREATE TABLE t (id TEXT, c TEXT, k INTEGER)"
_GROUP_INSERT = "INSERT INTO t (id, c, k) VALUES (?, ?, ?)"
# Three groups by c: 'x' (3 rows), 'y' (2 rows), 'z' (1 row); k mirrors group size.
_GROUP_ROWS = [
    ("r1", "x", 1),
    ("r2", "x", 1),
    ("r3", "x", 1),
    ("r4", "y", 2),
    ("r5", "y", 2),
    ("r6", "z", 3),
]


def test_group_by_count_parity(tmp_path) -> None:
    """``SELECT c, COUNT(*) FROM t GROUP BY c`` returns one row per group == sqlite3."""
    lilli, sq = _seed_both(tmp_path, _GROUP_DDL, _GROUP_INSERT, _GROUP_ROWS)
    try:
        sql = "SELECT c, COUNT(*) FROM t GROUP BY c"
        _assert_parity(lilli, sq, sql)
        # The unaliased count column is labelled COUNT(*) on both drivers.
        lilli_desc = [d[0] for d in lilli.execute(sql).description]
        sqlite_desc = [d[0] for d in sq.execute(sql).description]
        assert lilli_desc == sqlite_desc == ["c", "COUNT(*)"]
    finally:
        lilli.close()
        sq.close()


def test_group_by_count_alias_parity(tmp_path) -> None:
    """``SELECT c, COUNT(*) AS n FROM t GROUP BY c`` labels the count column n == sqlite3."""
    lilli, sq = _seed_both(tmp_path, _GROUP_DDL, _GROUP_INSERT, _GROUP_ROWS)
    try:
        sql = "SELECT c, COUNT(*) AS n FROM t GROUP BY c"
        _assert_parity(lilli, sq, sql)
        lilli_desc = [d[0] for d in lilli.execute(sql).description]
        sqlite_desc = [d[0] for d in sq.execute(sql).description]
        assert lilli_desc == sqlite_desc == ["c", "n"]
    finally:
        lilli.close()
        sq.close()


def test_group_by_count_after_where_parity(tmp_path) -> None:
    """Grouping happens AFTER the WHERE filter == sqlite3."""
    rows = _GROUP_ROWS + [("r7", None, 9)]  # a NULL c row, excluded by the WHERE
    lilli, sq = _seed_both(tmp_path, _GROUP_DDL, _GROUP_INSERT, rows)
    try:
        _assert_parity(
            lilli, sq,
            "SELECT c, COUNT(*) AS n FROM t WHERE c IS NOT NULL GROUP BY c",
        )
    finally:
        lilli.close()
        sq.close()


def test_group_by_having_parity(tmp_path) -> None:
    """``GROUP BY c HAVING COUNT(*) >= 3`` filters groups by count == sqlite3."""
    lilli, sq = _seed_both(tmp_path, _GROUP_DDL, _GROUP_INSERT, _GROUP_ROWS)
    try:
        # >= keeps only the 3-row group.
        _assert_parity(
            lilli, sq,
            "SELECT c, COUNT(*) AS n FROM t GROUP BY c HAVING COUNT(*) >= 3",
        )
        # > 1 keeps the 3-row and 2-row groups.
        _assert_parity(
            lilli, sq,
            "SELECT c, COUNT(*) AS n FROM t GROUP BY c HAVING COUNT(*) > 1",
        )
        # = 1 keeps only the singleton group.
        _assert_parity(
            lilli, sq,
            "SELECT c, COUNT(*) AS n FROM t GROUP BY c HAVING COUNT(*) = 1",
        )
    finally:
        lilli.close()
        sq.close()


def test_group_by_null_key_parity(tmp_path) -> None:
    """A NULL group key buckets as its own group, ordered by storage class == sqlite3."""
    rows = _GROUP_ROWS + [("r7", None, 9), ("r8", None, 9)]
    lilli, sq = _seed_both(tmp_path, _GROUP_DDL, _GROUP_INSERT, rows)
    try:
        _assert_parity(lilli, sq, "SELECT c, COUNT(*) AS n FROM t GROUP BY c")
    finally:
        lilli.close()
        sq.close()


def test_group_by_empty_table_parity(tmp_path) -> None:
    """GROUP BY over an empty table yields zero group rows == sqlite3."""
    lilli, sq = _seed_both(tmp_path, _GROUP_DDL, _GROUP_INSERT, [])
    try:
        _assert_parity(lilli, sq, "SELECT c, COUNT(*) AS n FROM t GROUP BY c")
    finally:
        lilli.close()
        sq.close()


def test_group_by_bare_columns_parity(tmp_path) -> None:
    """``SELECT id, c FROM t GROUP BY c`` projects the first-row value per group == sqlite3.

    The migrate collapsed-timestamps query selects a bare column (id) not in the
    GROUP BY; sqlite3 returns the value from the first row of each group.
    """
    lilli, sq = _seed_both(tmp_path, _GROUP_DDL, _GROUP_INSERT, _GROUP_ROWS)
    try:
        _assert_parity(lilli, sq, "SELECT id, c FROM t GROUP BY c")
        _assert_parity(
            lilli, sq,
            "SELECT id, c FROM t GROUP BY c HAVING COUNT(*) >= 3",
        )
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# Multi-key ORDER BY on a grouped result (the crisis shape) + mixed direction
# ---------------------------------------------------------------------------


def test_group_by_multikey_order_by_parity(tmp_path) -> None:
    """``GROUP BY c ORDER BY n ASC, c ASC`` sorts lexicographically == sqlite3.

    The exact crisis sizing shape: order by the COUNT alias first, then the
    group column, both ASC.
    """
    # Two groups tie on count (x:2, w:2) so the secondary key (c) breaks the tie.
    rows = [
        ("r1", "x", 0),
        ("r2", "x", 0),
        ("r3", "w", 0),
        ("r4", "w", 0),
        ("r5", "y", 0),
        ("r6", "z", 0),
        ("r7", "z", 0),
        ("r8", "z", 0),
    ]
    lilli, sq = _seed_both(tmp_path, _GROUP_DDL, _GROUP_INSERT, rows)
    try:
        _assert_parity(
            lilli, sq,
            "SELECT c, COUNT(*) AS n FROM t GROUP BY c ORDER BY n ASC, c ASC",
        )
    finally:
        lilli.close()
        sq.close()


def test_order_by_multikey_mixed_direction_parity(tmp_path) -> None:
    """``ORDER BY n DESC, c ASC`` composes per-element direction == sqlite3.

    Guards against a blanket reverse: the primary key descends while the
    secondary key ascends, with correct per-element NULL placement.
    """
    rows = [
        ("r1", "x", 0),
        ("r2", "x", 0),
        ("r3", "w", 0),
        ("r4", "w", 0),
        ("r5", "y", 0),
        ("r6", "z", 0),
        ("r7", "z", 0),
        ("r8", "z", 0),
    ]
    lilli, sq = _seed_both(tmp_path, _GROUP_DDL, _GROUP_INSERT, rows)
    try:
        _assert_parity(
            lilli, sq,
            "SELECT c, COUNT(*) AS n FROM t GROUP BY c ORDER BY n DESC, c ASC",
        )
        _assert_parity(
            lilli, sq,
            "SELECT c, COUNT(*) AS n FROM t GROUP BY c ORDER BY n ASC, c DESC",
        )
    finally:
        lilli.close()
        sq.close()


def test_group_by_multikey_order_by_null_element_parity(tmp_path) -> None:
    """A NULL second-element group key lands per-element NULL-first/last == sqlite3."""
    rows = [
        ("r1", None, 0),
        ("r2", None, 0),
        ("r3", "a", 0),
        ("r4", "a", 0),
        ("r5", "b", 0),
        ("r6", "b", 0),
    ]
    lilli, sq = _seed_both(tmp_path, _GROUP_DDL, _GROUP_INSERT, rows)
    try:
        # All groups have count 2, so the secondary group-column key (with a NULL
        # group) drives the order — NULL first on ASC, last on DESC.
        _assert_parity(
            lilli, sq,
            "SELECT c, COUNT(*) AS n FROM t GROUP BY c ORDER BY n ASC, c ASC",
        )
        _assert_parity(
            lilli, sq,
            "SELECT c, COUNT(*) AS n FROM t GROUP BY c ORDER BY n ASC, c DESC",
        )
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# IN (...) list — parameterised and literal (the migrate follow-up shape)
# ---------------------------------------------------------------------------


def test_in_param_list_parity(tmp_path) -> None:
    """``WHERE created_at IN (?, ?, ?)`` bound params match sqlite3 (migrate follow-up)."""
    rows = [
        ("a", "2026-01-01"),
        ("b", "2026-01-02"),
        ("c", "2026-01-03"),
        ("d", "2026-01-04"),
    ]
    lilli, sq = _seed_both(tmp_path, _RECORDS_DDL, _RECORDS_INSERT, rows)
    try:
        _assert_parity(
            lilli, sq,
            "SELECT id FROM records WHERE created_at IN (?, ?, ?)",
            ("2026-01-01", "2026-01-03", "2026-01-09"),
        )
        # A single-element parameterised IN list.
        _assert_parity(
            lilli, sq,
            "SELECT id FROM records WHERE created_at IN (?)",
            ("2026-01-02",),
        )
    finally:
        lilli.close()
        sq.close()


def test_in_literal_list_parity(tmp_path) -> None:
    """``WHERE created_at IN ('a','b')`` literal list matches sqlite3 (defensive)."""
    rows = [
        ("a", "2026-01-01"),
        ("b", "2026-01-02"),
        ("c", "2026-01-03"),
    ]
    lilli, sq = _seed_both(tmp_path, _RECORDS_DDL, _RECORDS_INSERT, rows)
    try:
        _assert_parity(
            lilli, sq,
            "SELECT id FROM records WHERE created_at IN ('2026-01-01', '2026-01-03')",
        )
        # An IN list matching nothing returns the empty set on both.
        _assert_parity(
            lilli, sq,
            "SELECT id FROM records WHERE created_at IN ('zzz', 'yyy')",
        )
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# End-to-end consumer-form parity: the exact crisis sizing + migrate queries,
# composing every GROUP BY feature on one statement, against the oracle.
# ---------------------------------------------------------------------------

# A records schema covering the columns the two consumer queries reference.
_CRISIS_DDL = (
    "CREATE TABLE records ("
    "vec_label INTEGER PRIMARY KEY AUTOINCREMENT, "
    "id TEXT NOT NULL UNIQUE, "
    "tier TEXT, "
    "created_at TEXT, "
    "tombstoned_at TEXT, "
    "community_id TEXT)"
)
_CRISIS_INSERT = (
    "INSERT INTO records (id, tier, created_at, tombstoned_at, community_id)"
    " VALUES (?, ?, ?, ?, ?)"
)


def test_crisis_sizing_consumer_parity(tmp_path) -> None:
    """The exact crisis sizing query (GROUP BY + COUNT(*) AS n + multi-key ORDER BY)."""
    # Communities of varying size, plus NULL community rows excluded by the WHERE.
    rows = [
        ("a", "episodic", "2026-01-01", None, "c1"),
        ("b", "episodic", "2026-01-01", None, "c1"),
        ("c", "episodic", "2026-01-01", None, "c1"),   # c1: 3
        ("d", "episodic", "2026-01-01", None, "c2"),
        ("e", "episodic", "2026-01-01", None, "c2"),   # c2: 2
        ("f", "episodic", "2026-01-01", None, "c3"),   # c3: 1
        ("g", "episodic", "2026-01-01", None, "c4"),
        ("h", "episodic", "2026-01-01", None, "c4"),   # c4: 2 — ties c2 on count
        ("i", "episodic", "2026-01-01", None, None),   # NULL community — excluded
    ]
    lilli, sq = _seed_both(tmp_path, _CRISIS_DDL, _CRISIS_INSERT, rows)
    try:
        sizing = (
            "SELECT community_id, COUNT(*) AS n FROM records "
            "WHERE community_id IS NOT NULL "
            "GROUP BY community_id ORDER BY n ASC, community_id ASC"
        )
        _assert_parity(lilli, sq, sizing)
        # The count column is labelled n on both drivers.
        lilli_desc = [d[0] for d in lilli.execute(sizing).description]
        sqlite_desc = [d[0] for d in sq.execute(sizing).description]
        assert lilli_desc == sqlite_desc == ["community_id", "n"]
    finally:
        lilli.close()
        sq.close()


def test_migrate_collapsed_timestamps_consumer_parity(tmp_path) -> None:
    """The exact migrate GROUP BY + HAVING query + the parameterised IN follow-up."""
    # One collapsed group of 4 (qualifies, >= 3), one of 2 (excluded), one of 1.
    rows = [
        ("a", "episodic", "2026-01-01T00:00:00", None, None),
        ("b", "episodic", "2026-01-01T00:00:00", None, None),
        ("c", "episodic", "2026-01-01T00:00:00", None, None),
        ("d", "episodic", "2026-01-01T00:00:00", None, None),   # group of 4
        ("e", "episodic", "2026-01-02T00:00:00", None, None),
        ("f", "episodic", "2026-01-02T00:00:00", None, None),   # group of 2
        ("g", "episodic", "2026-01-03T00:00:00", None, None),   # group of 1
        ("h", "semantic", "2026-01-01T00:00:00", None, None),   # wrong tier — excluded
        ("i", "episodic", "2026-01-01T00:00:00", "2026-01-09", None),  # tombstoned
    ]
    lilli, sq = _seed_both(tmp_path, _CRISIS_DDL, _CRISIS_INSERT, rows)
    try:
        collapsed = (
            "SELECT id, created_at FROM records"
            " WHERE tier = 'episodic'"
            "   AND tombstoned_at IS NULL"
            " GROUP BY created_at"
            " HAVING COUNT(*) >= 3"
        )
        _assert_parity(lilli, sq, collapsed)
        # The parameterised IN follow-up over the surviving created_at set.
        candidate_created_ats = [
            row[1] for row in sq.execute(collapsed).fetchall()
        ]
        placeholders = ",".join("?" * len(candidate_created_ats))
        followup = (
            "SELECT id FROM records"
            " WHERE tier = 'episodic'"
            "   AND tombstoned_at IS NULL"
            f"   AND created_at IN ({placeholders})"
        )
        _assert_parity(lilli, sq, followup, tuple(candidate_created_ats))
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# Bracket-quoted identifiers + ALTER TABLE RENAME TO + DROP TABLE
# (the re-embed table-swap + rollback consumer surface)
# ---------------------------------------------------------------------------


def test_bracket_ident_tokenizes() -> None:
    """A [bracket]-quoted identifier lexes as an IDENT with the brackets stripped."""
    toks = tokenize("ALTER TABLE [records] RENAME TO [dest]")
    idents = [t.value for t in toks if t.type == TT.IDENT]
    # The bracketed names appear as IDENTs with the brackets stripped (the
    # bare RENAME / TO words also lex as IDENTs — they are not reserved here).
    assert "records" in idents
    assert "dest" in idents
    # No bracket character survives into any token value.
    assert all("[" not in t.value and "]" not in t.value for t in toks)


def test_alter_rename_parity(tmp_path) -> None:
    """ALTER TABLE [records] RENAME TO records_old: the rows follow the new name;
    the old name errors no-such-table — on both drivers."""
    rows = [("a1", "2026-01-01"), ("a2", "2026-01-02")]
    lilli, sq = _seed_both(tmp_path, _RECORDS_DDL, _RECORDS_INSERT, rows)
    try:
        rename = "ALTER TABLE [records] RENAME TO [records_old]"
        lilli.execute(rename)
        sq.execute(rename)
        sq.commit()
        # The renamed table carries the seeded rows on both drivers.
        _assert_parity(
            lilli, sq, "SELECT id, created_at FROM records_old ORDER BY id"
        )
        # The old name no longer resolves: both drivers raise no-such-table.
        with pytest.raises(CatalogError):
            lilli.execute("SELECT id FROM records").fetchall()
        with pytest.raises(sqlite3.OperationalError):
            sq.execute("SELECT id FROM records").fetchall()
    finally:
        lilli.close()
        sq.close()


def test_alter_rename_bare_ident_parity(tmp_path) -> None:
    """A bare (non-bracketed) ALTER TABLE records RENAME TO dest also parses +
    applies — the rename classification keys off RENAME, not the brackets."""
    rows = [("b1", "2026-02-01")]
    lilli, sq = _seed_both(tmp_path, _RECORDS_DDL, _RECORDS_INSERT, rows)
    try:
        rename = "ALTER TABLE records RENAME TO records_new"
        lilli.execute(rename)
        sq.execute(rename)
        sq.commit()
        _assert_parity(lilli, sq, "SELECT id, created_at FROM records_new")
    finally:
        lilli.close()
        sq.close()


def test_alter_rename_survives_reopen(tmp_path) -> None:
    """The rename DDL + the new-name root mapping replay on re-open: a renamed
    table is still present + populated after close + re-open."""
    db_path = tmp_path / "rename_reopen.db"
    pager, catalog, root_map = _open_lilli_db(db_path)
    conn = LilliBrainConnection(pager, catalog, root_map)
    conn.execute(_RECORDS_DDL)
    conn.execute(_RECORDS_INSERT, ("r1", "2026-03-01"))
    conn.execute(_RECORDS_INSERT, ("r2", "2026-03-02"))
    conn.execute("ALTER TABLE [records] RENAME TO [records_old]")
    conn.close()

    # Re-open the same file: meta replay must reconstruct the renamed table.
    pager2, catalog2, root_map2 = _open_lilli_db(db_path)
    conn2 = LilliBrainConnection(pager2, catalog2, root_map2)
    try:
        out = _norm(
            conn2.execute(
                "SELECT id, created_at FROM records_old ORDER BY id"
            ).fetchall()
        )
        assert out == [("r1", "2026-03-01"), ("r2", "2026-03-02")]
        # The old name is gone after re-open as well.
        with pytest.raises(CatalogError):
            conn2.execute("SELECT id FROM records").fetchall()
    finally:
        conn2.close()


def test_drop_table_if_exists_parity(tmp_path) -> None:
    """DROP TABLE IF EXISTS removes a present table; the name then errors
    no-such-table on both drivers."""
    rows = [("d1", "2026-04-01")]
    lilli, sq = _seed_both(tmp_path, _RECORDS_DDL, _RECORDS_INSERT, rows)
    try:
        drop = "DROP TABLE IF EXISTS records"
        lilli.execute(drop)
        sq.execute(drop)
        sq.commit()
        with pytest.raises(CatalogError):
            lilli.execute("SELECT id FROM records").fetchall()
        with pytest.raises(sqlite3.OperationalError):
            sq.execute("SELECT id FROM records").fetchall()
    finally:
        lilli.close()
        sq.close()


def test_drop_table_absent_noop(tmp_path) -> None:
    """DROP TABLE IF EXISTS on an absent table is a no-op (no error) on both
    drivers; DROP without IF EXISTS on a missing table errors on both."""
    rows = [("d2", "2026-04-02")]
    lilli, sq = _seed_both(tmp_path, _RECORDS_DDL, _RECORDS_INSERT, rows)
    try:
        # IF EXISTS + absent table: no error on either driver.
        lilli.execute("DROP TABLE IF EXISTS does_not_exist")
        sq.execute("DROP TABLE IF EXISTS does_not_exist")
        sq.commit()
        # Without IF EXISTS, a missing table errors on both drivers.
        with pytest.raises(CatalogError):
            lilli.execute("DROP TABLE does_not_exist")
        with pytest.raises(sqlite3.OperationalError):
            sq.execute("DROP TABLE does_not_exist")
    finally:
        lilli.close()
        sq.close()


def test_drop_table_survives_reopen(tmp_path) -> None:
    """A drop persists: after close + re-open the dropped table stays absent
    (the meta replay does not resurrect it)."""
    db_path = tmp_path / "drop_reopen.db"
    pager, catalog, root_map = _open_lilli_db(db_path)
    conn = LilliBrainConnection(pager, catalog, root_map)
    conn.execute(_RECORDS_DDL)
    conn.execute(_RECORDS_INSERT, ("x1", "2026-05-01"))
    conn.execute("DROP TABLE IF EXISTS records")
    conn.close()

    pager2, catalog2, root_map2 = _open_lilli_db(db_path)
    conn2 = LilliBrainConnection(pager2, catalog2, root_map2)
    try:
        assert "records" not in root_map2
        with pytest.raises(CatalogError):
            conn2.execute("SELECT id FROM records").fetchall()
    finally:
        conn2.close()


def test_drop_then_create_same_name_lands_in_fresh_root(tmp_path) -> None:
    """DROP-then-CREATE of the same name (the re-embed staging pattern): after a
    drop + re-create + insert, rows land in the FRESH root, not a stale mapping —
    proven across a re-open."""
    db_path = tmp_path / "drop_recreate.db"
    pager, catalog, root_map = _open_lilli_db(db_path)
    conn = LilliBrainConnection(pager, catalog, root_map)
    conn.execute(_RECORDS_DDL)
    conn.execute(_RECORDS_INSERT, ("old", "2026-06-01"))
    conn.execute("DROP TABLE IF EXISTS records")
    # Re-create the same name and insert a single fresh row.
    conn.execute(_RECORDS_DDL)
    conn.execute(_RECORDS_INSERT, ("fresh", "2026-06-02"))
    conn.close()

    # Re-open: the stale pre-drop root must not shadow the fresh table.
    pager2, catalog2, root_map2 = _open_lilli_db(db_path)
    conn2 = LilliBrainConnection(pager2, catalog2, root_map2)
    try:
        out = _norm(
            conn2.execute("SELECT id FROM records ORDER BY id").fetchall()
        )
        assert out == [("fresh",)]
    finally:
        conn2.close()


# ---------------------------------------------------------------------------
# == operator (an exact alias for =, normalised at the lexer to the = value)
# ---------------------------------------------------------------------------


def test_double_equals_tokenizes_to_single_equals() -> None:
    """``==`` lexes to ONE OP token whose value is ``=`` — no stray second ``=``."""
    toks = tokenize("WHERE id == 'a'")
    ops = [t for t in toks if t.type == TT.OP]
    assert len(ops) == 1
    assert ops[0].value == "="


def test_operators_unchanged() -> None:
    """``!=``/``<>``/``<=``/``>=``/``<``/``>``/``=`` lex unchanged by the == rule."""
    for src, expected in [
        ("a != b", "!="),
        ("a <> b", "<>"),
        ("a <= b", "<="),
        ("a >= b", ">="),
        ("a < b", "<"),
        ("a > b", ">"),
        ("a = b", "="),
    ]:
        ops = [t for t in tokenize(src) if t.type == TT.OP]
        assert len(ops) == 1, f"{src!r} produced {len(ops)} OP tokens"
        assert ops[0].value == expected, f"{src!r} -> {ops[0].value!r}"


def test_double_equals_operator_parity(tmp_path) -> None:
    """``WHERE id == 'a'`` returns rows identical to ``= 'a'`` and to sqlite3."""
    rows = [("a", "2026-01-01"), ("b", "2026-01-02")]
    lilli, sq = _seed_both(tmp_path, _RECORDS_DDL, _RECORDS_INSERT, rows)
    try:
        # == matches = on the lilli driver and is byte-identical to sqlite3.
        _assert_parity(lilli, sq, "SELECT id FROM records WHERE id == 'a'")
        eq2 = _norm(
            lilli.execute("SELECT id FROM records WHERE id == 'a'").fetchall()
        )
        eq1 = _norm(
            lilli.execute("SELECT id FROM records WHERE id = 'a'").fetchall()
        )
        assert eq2 == eq1 == [("a",)]
    finally:
        lilli.close()
        sq.close()


def test_double_equals_compound_parity(tmp_path) -> None:
    """A compound ``WHERE a == 'x' AND b == 'y'`` (the multi-== shape) == sqlite3."""
    ddl = "CREATE TABLE pairs (a TEXT, b TEXT)"
    insert = "INSERT INTO pairs (a, b) VALUES (?, ?)"
    rows = [("x", "y"), ("x", "z"), ("p", "y")]
    lilli, sq = _seed_both(tmp_path, ddl, insert, rows)
    try:
        _assert_parity(
            lilli, sq,
            "SELECT a, b FROM pairs WHERE a == 'x' AND b == 'y'",
        )
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# Scalar (ungrouped) COUNT(*) AS alias — the sole-aggregate alias label path
# ---------------------------------------------------------------------------


def test_scalar_count_star_as_alias_parity(tmp_path) -> None:
    """``SELECT COUNT(*) AS n FROM records`` labels the count column n == sqlite3.

    The sole (ungrouped) aggregate alias path — distinct from the grouped
    ``GROUP BY c`` count alias. Covers the count value, the alias label the
    consumer reads the row by, AND the parser accepting ``AS alias`` after the
    closing paren.
    """
    rows = [("a", "2026-01-01"), ("b", "2026-01-02"), ("c", "2026-01-03")]
    lilli, sq = _seed_both(tmp_path, _RECORDS_DDL, _RECORDS_INSERT, rows)
    try:
        sql = "SELECT COUNT(*) AS n FROM records"
        _assert_parity(lilli, sq, sql)
        lilli_desc = [d[0] for d in lilli.execute(sql).description]
        sqlite_desc = [d[0] for d in sq.execute(sql).description]
        assert lilli_desc == sqlite_desc == ["n"]
        # The scalar aggregate with a WHERE filter (the consumer's exact shape).
        where_sql = "SELECT COUNT(*) AS n FROM records WHERE id = 'a'"
        _assert_parity(lilli, sq, where_sql)
        # An unaliased scalar COUNT(*) keeps the default COUNT(*) label.
        bare = "SELECT COUNT(*) FROM records"
        _assert_parity(lilli, sq, bare)
        lilli_bare = [d[0] for d in lilli.execute(bare).description]
        sqlite_bare = [d[0] for d in sq.execute(bare).description]
        assert lilli_bare == sqlite_bare == ["COUNT(*)"]
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# INSERT OR REPLACE — replace-on-conflict upsert parity vs sqlite3
# ---------------------------------------------------------------------------


_REPLACE_DDL = "CREATE TABLE kv (k INTEGER PRIMARY KEY, v TEXT)"
_REPLACE_INSERT = "INSERT INTO kv (k, v) VALUES (?, ?)"


def test_insert_or_replace_overwrites_on_conflict(tmp_path) -> None:
    """INSERT OR REPLACE on an existing primary key overwrites the row == sqlite3."""
    rows = [(1, "first"), (2, "second")]
    lilli, sq = _seed_both(tmp_path, _REPLACE_DDL, _REPLACE_INSERT, rows)
    try:
        replace_sql = "INSERT OR REPLACE INTO kv (k, v) VALUES (1, 'overwritten')"
        lilli.execute(replace_sql)
        sq.execute(replace_sql)
        sq.commit()
        _assert_parity(lilli, sq, "SELECT k, v FROM kv ORDER BY k")
        # Exactly one row at the conflicting key, carrying the new value.
        result = _norm(
            lilli.execute("SELECT k, v FROM kv WHERE k = 1").fetchall()
        )
        assert result == [(1, "overwritten")]
    finally:
        lilli.close()
        sq.close()


def test_insert_or_replace_inserts_on_miss(tmp_path) -> None:
    """INSERT OR REPLACE with a new key inserts a fresh row == sqlite3."""
    rows = [(1, "first"), (2, "second")]
    lilli, sq = _seed_both(tmp_path, _REPLACE_DDL, _REPLACE_INSERT, rows)
    try:
        replace_sql = "INSERT OR REPLACE INTO kv (k, v) VALUES (3, 'third')"
        lilli.execute(replace_sql)
        sq.execute(replace_sql)
        sq.commit()
        _assert_parity(lilli, sq, "SELECT k, v FROM kv ORDER BY k")
        # Three rows total after the miss-insert.
        assert len(_norm(lilli.execute("SELECT k FROM kv").fetchall())) == 3
    finally:
        lilli.close()
        sq.close()


def test_insert_or_replace_not_null_parity(tmp_path) -> None:
    """A REPLACE that would null a NOT NULL column fails in both drivers.

    sqlite3 raises sqlite3.IntegrityError; the engine raises ValueError. The
    parity that matters is the outcome (both reject the write), and that the
    engine does not leave a half-open transaction behind — a subsequent valid
    write must still succeed.
    """
    ddl = "CREATE TABLE nn (k INTEGER PRIMARY KEY, v TEXT NOT NULL)"
    insert = "INSERT INTO nn (k, v) VALUES (?, ?)"
    rows = [(1, "present")]
    lilli, sq = _seed_both(tmp_path, ddl, insert, rows)
    try:
        bad_replace = "INSERT OR REPLACE INTO nn (k) VALUES (1)"
        with pytest.raises(sqlite3.IntegrityError):
            sq.execute(bad_replace)
        with pytest.raises(ValueError):
            lilli.execute(bad_replace)
        # The failed replace must not wedge the engine: a valid replace works.
        lilli.execute("INSERT OR REPLACE INTO nn (k, v) VALUES (1, 'next')")
        result = _norm(lilli.execute("SELECT k, v FROM nn ORDER BY k").fetchall())
        assert result == [(1, "next")]
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# Cursor iteration — `for row in conn.execute(...)` parity vs sqlite3
# ---------------------------------------------------------------------------


def test_cursor_iteration_parity(tmp_path) -> None:
    """Iterating a result cursor row-by-row matches sqlite3.Cursor iteration.

    Drives __iter__ (not fetchall) so the iteration path is exercised, and
    confirms the engine yields the same rows in the same order as stdlib.
    """
    rows = [("a", "2026-01-01"), ("b", "2026-01-02"), ("c", "2026-01-03")]
    lilli, sq = _seed_both(tmp_path, _RECORDS_DDL, _RECORDS_INSERT, rows)
    try:
        sql = "SELECT id, created_at FROM records ORDER BY id"
        lilli_iter = [tuple(row) for row in lilli.execute(sql)]
        sqlite_iter = [tuple(row) for row in sq.execute(sql)]
        assert lilli_iter == sqlite_iter
        # Iteration consumes the shared offset: fetchall after a full iter is empty.
        cur = lilli.execute(sql)
        _ = [row for row in cur]
        assert cur.fetchall() == []
    finally:
        lilli.close()
        sq.close()


def test_pragma_table_info_iteration_parity(tmp_path) -> None:
    """`{row[1] for row in conn.execute('PRAGMA table_info(records)')}` parity.

    This is the exact comprehension three harness sites use; it must yield the
    same column-name set on the engine cursor as on stdlib sqlite3.
    """
    rows = [("a", "2026-01-01")]
    lilli, sq = _seed_both(tmp_path, _RECORDS_DDL, _RECORDS_INSERT, rows)
    try:
        pragma = "PRAGMA table_info(records)"
        lilli_cols = {row[1] for row in lilli.execute(pragma)}
        sqlite_cols = {row[1] for row in sq.execute(pragma)}
        assert lilli_cols == sqlite_cols
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# GAP 1 — integer literal (and bare scalar) in SELECT list
# ---------------------------------------------------------------------------


def test_integer_literal_in_select_list_parity(tmp_path) -> None:
    """SELECT 1 FROM records WHERE ... returns 1 per matching row == sqlite3.

    This is the exact shape has_pending_rows() uses:
      SELECT 1 FROM records WHERE COALESCE(embedding_pending, 0) = 1 LIMIT 1
    """
    ddl = (
        "CREATE TABLE records ("
        "vec_label INTEGER PRIMARY KEY AUTOINCREMENT, "
        "id TEXT NOT NULL, "
        "embedding_pending INTEGER)"
    )
    insert = "INSERT INTO records (id, embedding_pending) VALUES (?, ?)"
    seed = [("r1", 1), ("r2", 0), ("r3", 1)]

    pager, catalog, root_map = _open_lilli_db(tmp_path / "engine.db")
    lilli = LilliBrainConnection(pager, catalog, root_map)
    lilli.execute(ddl)
    for p in seed:
        lilli.execute(insert, p)

    sq = sqlite3.connect(":memory:")
    sq.row_factory = sqlite3.Row
    sq.execute(ddl)
    for p in seed:
        sq.execute(insert, p)
    sq.commit()

    try:
        # Production query shape from has_pending_rows().
        q1 = "SELECT 1 FROM records WHERE COALESCE(embedding_pending, 0) = 1 LIMIT 1"
        lilli_rows = _norm(lilli.execute(q1).fetchall())
        sqlite_rows = _norm(sq.execute(q1).fetchall())
        assert lilli_rows == sqlite_rows, (
            f"SELECT 1 LIMIT 1: lilli={lilli_rows!r} sq={sqlite_rows!r}"
        )
        # Without LIMIT — returns one row per matching record.
        q2 = "SELECT 1 FROM records WHERE COALESCE(embedding_pending, 0) = 1"
        lilli_rows2 = _norm(lilli.execute(q2).fetchall())
        sqlite_rows2 = _norm(sq.execute(q2).fetchall())
        assert lilli_rows2 == sqlite_rows2, (
            f"SELECT 1 (all): lilli={lilli_rows2!r} sq={sqlite_rows2!r}"
        )
        # Integer literal in a multi-column projection.
        q3 = "SELECT id, 1 FROM records WHERE embedding_pending = 1"
        lilli_rows3 = _norm(lilli.execute(q3).fetchall())
        sqlite_rows3 = _norm(sq.execute(q3).fetchall())
        assert lilli_rows3 == sqlite_rows3, (
            f"SELECT id, 1: lilli={lilli_rows3!r} sq={sqlite_rows3!r}"
        )
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# GAP 2 — CREATE TABLE <t> AS SELECT ... (CTAS)
# ---------------------------------------------------------------------------


def test_ctas_creates_and_populates_parity(tmp_path) -> None:
    """CREATE TABLE backup AS SELECT ... creates the table and populates it == sqlite3.

    Mirrors the migrate backup-table path:
      CREATE TABLE records_v4_backup AS SELECT col1, col2, ... FROM records
    """
    ddl = (
        "CREATE TABLE records ("
        "vec_label INTEGER PRIMARY KEY AUTOINCREMENT, "
        "id TEXT NOT NULL, "
        "tier TEXT, "
        "created_at TEXT)"
    )
    insert = "INSERT INTO records (id, tier, created_at) VALUES (?, ?, ?)"
    seed = [
        ("id-1", "episodic", "2026-01-01"),
        ("id-2", "semantic", "2026-01-02"),
        ("id-3", "episodic", "2026-01-03"),
    ]

    pager, catalog, root_map = _open_lilli_db(tmp_path / "engine.db")
    lilli = LilliBrainConnection(pager, catalog, root_map)
    lilli.execute(ddl)
    for p in seed:
        lilli.execute(insert, p)

    sq = sqlite3.connect(":memory:")
    sq.row_factory = sqlite3.Row
    sq.execute(ddl)
    for p in seed:
        sq.execute(insert, p)
    sq.commit()

    ctas = "CREATE TABLE records_backup AS SELECT id, tier, created_at FROM records"

    try:
        lilli.execute(ctas)
        sq.execute(ctas)

        lilli_rows = _norm(
            lilli.execute("SELECT id, tier, created_at FROM records_backup ORDER BY id").fetchall()
        )
        sq_rows = _norm(
            sq.execute("SELECT id, tier, created_at FROM records_backup ORDER BY id").fetchall()
        )
        assert lilli_rows == sq_rows, (
            f"CTAS parity: lilli={lilli_rows!r} sq={sq_rows!r}"
        )
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# GAP 2b — CTAS over a table with a BLOB column (memoryview encode)
# ---------------------------------------------------------------------------


def test_ctas_blob_column_round_trips(tmp_path) -> None:
    """CTAS over a table with a BLOB column must not raise on memoryview cells.

    The engine's full-scan SELECT returns BLOB cells as memoryview slices
    (zero-copy page reads). The CTAS re-insert path feeds those values back
    through encode_record, which must accept memoryview just as it accepts bytes.
    Round-trip: the backed-up BLOB must decode to the original bytes.
    """
    ddl = (
        "CREATE TABLE blobs ("
        "id TEXT NOT NULL, "
        "payload BLOB NOT NULL)"
    )
    blob_a = b"\x00\x01\x02\x03\xff"
    blob_b = b"\xde\xad\xbe\xef"

    pager, catalog, root_map = _open_lilli_db(tmp_path / "blob_engine.db")
    lilli = LilliBrainConnection(pager, catalog, root_map)
    lilli.execute(ddl)
    lilli.execute("INSERT INTO blobs (id, payload) VALUES (?, ?)", ("a", blob_a))
    lilli.execute("INSERT INTO blobs (id, payload) VALUES (?, ?)", ("b", blob_b))

    ctas = "CREATE TABLE blobs_backup AS SELECT id, payload FROM blobs"
    lilli.execute(ctas)

    rows = lilli.execute("SELECT id, payload FROM blobs_backup ORDER BY id").fetchall()
    try:
        got = {str(r["id"]): bytes(r["payload"]) for r in rows}
    finally:
        lilli.close()

    assert got == {"a": blob_a, "b": blob_b}, (
        f"CTAS BLOB round-trip failed: {got!r}"
    )


# ---------------------------------------------------------------------------
# GAP 3 — EXPLAIN / EXPLAIN QUERY PLAN passthrough
# ---------------------------------------------------------------------------


def test_explain_does_not_crash(tmp_path) -> None:
    """EXPLAIN and EXPLAIN QUERY PLAN return a non-crashing result == sqlite3.

    The consumer (test_store_get_batch.py) only checks that EXPLAIN doesn't
    raise.  A non-empty plan rowset is a nice-to-have but not required — the
    engine returns an empty rowset as a passthrough; both sides don't crash.
    """
    rows = [("a", "2026-01-01"), ("b", "2026-01-02")]
    lilli, sq = _seed_both(tmp_path, _RECORDS_DDL, _RECORDS_INSERT, rows)
    try:
        inner = (
            "SELECT id, created_at FROM records"
            " WHERE id = 'a' ORDER BY created_at DESC LIMIT 10"
        )
        explain_qp = f"EXPLAIN QUERY PLAN {inner}"
        explain_bare = f"EXPLAIN {inner}"

        # Neither should raise.
        lilli.execute(explain_qp).fetchall()
        lilli.execute(explain_bare).fetchall()
        sq.execute(explain_qp).fetchall()
        sq.execute(explain_bare).fetchall()
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# Named-parameter parity for INSERT / UPDATE / DELETE (HR-01 guard)
# ---------------------------------------------------------------------------

_NAMED_DML_DDL = "CREATE TABLE t (id TEXT NOT NULL, n INTEGER)"
_NAMED_DML_INSERT_POS = "INSERT INTO t (id, n) VALUES (?, ?)"


def test_named_insert_parity(tmp_path: Path) -> None:
    """INSERT ... VALUES (:a, :b) with a dict binds correctly — no NULL corruption."""
    lilli, sq = _seed_both(tmp_path, _NAMED_DML_DDL, _NAMED_DML_INSERT_POS, [])
    try:
        # Insert via named params on both drivers.
        lilli.execute("INSERT INTO t (id, n) VALUES (:id, :n)", {"id": "x", "n": 5})
        sq.execute("INSERT INTO t (id, n) VALUES (:id, :n)", {"id": "x", "n": 5})
        sq.commit()

        # Both should see the row with the bound values, not NULLs.
        _assert_parity(lilli, sq, "SELECT id, n FROM t WHERE id = 'x'")
        lilli_rows = _norm(lilli.execute("SELECT id, n FROM t WHERE id = 'x'").fetchall())
        assert lilli_rows == [("x", 5)], (
            f"named INSERT stored wrong values: {lilli_rows!r}"
        )
    finally:
        lilli.close()
        sq.close()


def test_named_insert_missing_key_raises(tmp_path: Path) -> None:
    """Named INSERT with a missing key raises rather than silently storing NULL."""
    lilli, sq = _seed_both(tmp_path, _NAMED_DML_DDL, _NAMED_DML_INSERT_POS, [])
    try:
        with pytest.raises(Exception):
            lilli.execute("INSERT INTO t (id, n) VALUES (:id, :n)", {"id": "x"})
    finally:
        lilli.close()
        sq.close()


def test_named_update_parity(tmp_path: Path) -> None:
    """UPDATE SET col=:val WHERE id=:wid with a dict binds correctly."""
    seed = [("alice", 1), ("bob", 2)]
    lilli, sq = _seed_both(tmp_path, _NAMED_DML_DDL, _NAMED_DML_INSERT_POS, seed)
    try:
        update_sql = "UPDATE t SET n = :newn WHERE id = :wid"
        lilli.execute(update_sql, {"newn": 99, "wid": "alice"})
        sq.execute(update_sql, {"newn": 99, "wid": "alice"})
        sq.commit()

        _assert_parity(lilli, sq, "SELECT id, n FROM t ORDER BY id")
        lilli_rows = _norm(lilli.execute("SELECT id, n FROM t WHERE id = 'alice'").fetchall())
        assert lilli_rows == [("alice", 99)], (
            f"named UPDATE stored wrong value: {lilli_rows!r}"
        )
        # bob should be untouched.
        bob_rows = _norm(lilli.execute("SELECT id, n FROM t WHERE id = 'bob'").fetchall())
        assert bob_rows == [("bob", 2)]
    finally:
        lilli.close()
        sq.close()


def test_named_delete_parity(tmp_path: Path) -> None:
    """DELETE FROM t WHERE id=:wid with a dict removes the right row."""
    seed = [("alice", 1), ("bob", 2)]
    lilli, sq = _seed_both(tmp_path, _NAMED_DML_DDL, _NAMED_DML_INSERT_POS, seed)
    try:
        delete_sql = "DELETE FROM t WHERE id = :wid"
        lilli.execute(delete_sql, {"wid": "alice"})
        sq.execute(delete_sql, {"wid": "alice"})
        sq.commit()

        _assert_parity(lilli, sq, "SELECT id, n FROM t ORDER BY id")
        remaining = _norm(lilli.execute("SELECT id, n FROM t").fetchall())
        assert remaining == [("bob", 2)], (
            f"named DELETE left unexpected rows: {remaining!r}"
        )
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# CR-01 guard: negative integers — signed encode+decode round-trip + parity
# ---------------------------------------------------------------------------


def test_signed_int_codec_roundtrip() -> None:
    """encode_record / decode_record round-trips the full signed-64 boundary set.

    Proves the signed encode+decode pair is byte-faithful: every boundary value
    round-trips to itself with zero mismatches. This is the direct codec-level
    proof the review required.
    """
    from iai_mcp.lillibrain.record import encode_record, decode_record

    boundary_values = [
        0, 1, -1, 2, -2,
        127, 128, -128, -129,
        255, 256, -256,
        32767, 32768, -32768, -32769,
        65535, 65536, -65536,
        2**31 - 1, 2**31, -(2**31), -(2**31) - 1,
        2**63 - 1, -(2**63),
        -3, -10,
    ]
    for v in boundary_values:
        encoded = encode_record([v])
        decoded, _ = decode_record(encoded, 0)
        assert decoded == [v], (
            f"round-trip failed for {v}: got {decoded[0]!r}"
        )


def test_negative_int_order_by_min_max_parity(tmp_path) -> None:
    """ORDER BY / MIN / MAX over negative INTEGER values match sqlite3.

    Proves the signed codec fix closes the gap the review identified: before
    the fix, -3 decoded as 2**64-3 and sorted after every positive, making
    MIN and MAX return wrong values. Positives must be unaffected.
    """
    ddl = "CREATE TABLE t (id TEXT, n INTEGER)"
    insert = "INSERT INTO t (id, n) VALUES (?, ?)"
    # Seeds include negatives, zero, and positives at signed-8-bit boundaries.
    rows = [
        ("a", 5),
        ("b", -3),
        ("c", 0),
        ("d", -10),
        ("e", 127),
        ("f", -128),
        ("g", 128),
    ]
    lilli, sq = _seed_both(tmp_path, ddl, insert, rows)
    try:
        _assert_parity(lilli, sq, "SELECT MIN(n) FROM t")
        _assert_parity(lilli, sq, "SELECT MAX(n) FROM t")
        _assert_parity(lilli, sq, "SELECT id FROM t ORDER BY n ASC")
        _assert_parity(lilli, sq, "SELECT id FROM t ORDER BY n DESC")
        # Spot-check: MIN must be -128, MAX must be 128.
        min_row = lilli.execute("SELECT MIN(n) FROM t").fetchone()
        max_row = lilli.execute("SELECT MAX(n) FROM t").fetchone()
        assert min_row[0] == -128, f"MIN wrong: {min_row[0]!r}"
        assert max_row[0] == 128, f"MAX wrong: {max_row[0]!r}"
    finally:
        lilli.close()
        sq.close()


def test_positive_int_encoding_unaffected(tmp_path) -> None:
    """Positive integers (0, 1, 127, 128, 255, 256) encode identically after fix.

    The signed codec change must not regress any existing positive-integer
    value: 128 is now stored as a signed 16-bit (2 bytes) instead of an
    unsigned 8-bit, but decodes to the same Python int 128.
    """
    ddl = "CREATE TABLE t (id TEXT, n INTEGER)"
    insert = "INSERT INTO t (id, n) VALUES (?, ?)"
    rows = [
        ("a", 0), ("b", 1), ("c", 127), ("d", 128),
        ("e", 255), ("f", 256), ("g", 65535), ("h", 65536),
    ]
    lilli, sq = _seed_both(tmp_path, ddl, insert, rows)
    try:
        _assert_parity(lilli, sq, "SELECT id, n FROM t ORDER BY n ASC")
        _assert_parity(lilli, sq, "SELECT MIN(n) FROM t")
        _assert_parity(lilli, sq, "SELECT MAX(n) FROM t")
        # Individual spot-checks for critical boundary values.
        for id_val, expected in [("a", 0), ("b", 1), ("c", 127), ("d", 128),
                                  ("e", 255), ("f", 256)]:
            row = lilli.execute(
                "SELECT n FROM t WHERE id = ?", (id_val,)
            ).fetchone()
            assert row[0] == expected, (
                f"positive value id={id_val!r} round-trips wrong: {row[0]!r} != {expected}"
            )
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# MR-01 guard: non-grouped two-key ORDER BY == sqlite3
# ---------------------------------------------------------------------------


def test_non_grouped_multikey_order_by_parity(tmp_path) -> None:
    """Non-grouped ``ORDER BY a ASC, b DESC`` matches sqlite3 (MR-01 fix).

    Uses rows where all a-values are equal so the tiebreaker b drives the
    order — if the second key is dropped the result stays in insertion order
    and differs from sqlite3.
    """
    ddl = "CREATE TABLE t (id TEXT, a INTEGER, b INTEGER)"
    insert = "INSERT INTO t (id, a, b) VALUES (?, ?, ?)"
    rows = [
        ("r1", 0, 9),
        ("r2", 0, 3),
        ("r3", 0, 7),
        ("r4", 0, 5),
    ]
    db1 = tmp_path / "db1"
    db1.mkdir()
    lilli, sq = _seed_both(db1, ddl, insert, rows)
    try:
        # All a-values equal; b DESC drives the order: 9,7,5,3 → r1,r3,r4,r2
        _assert_parity(lilli, sq, "SELECT id FROM t ORDER BY a ASC, b DESC")
        _assert_parity(lilli, sq, "SELECT id FROM t ORDER BY a ASC, b ASC")
    finally:
        lilli.close()
        sq.close()
    # Mixed: primary non-trivial, secondary as tiebreaker among ties.
    rows2 = [
        ("s1", 1, 5), ("s2", 2, 3), ("s3", 1, 2), ("s4", 2, 7),
    ]
    db2 = tmp_path / "db2"
    db2.mkdir()
    lilli2, sq2 = _seed_both(db2, ddl, insert, rows2)
    try:
        _assert_parity(lilli2, sq2, "SELECT id FROM t ORDER BY a ASC, b DESC")
        _assert_parity(lilli2, sq2, "SELECT id FROM t ORDER BY a DESC, b ASC")
    finally:
        lilli2.close()
        sq2.close()


def test_non_grouped_multikey_order_by_with_null(tmp_path) -> None:
    """Non-grouped two-key ORDER BY with NULL in one key matches sqlite3."""
    ddl = "CREATE TABLE t (id TEXT, a INTEGER, b INTEGER)"
    insert = "INSERT INTO t (id, a, b) VALUES (?, ?, ?)"
    rows = [
        ("r1", None, 3),
        ("r2", 1, 5),
        ("r3", None, 1),
        ("r4", 1, 2),
    ]
    lilli, sq = _seed_both(tmp_path, ddl, insert, rows)
    try:
        # NULL first on ASC, then b sorts among ties; NULL last on DESC.
        _assert_parity(lilli, sq, "SELECT id FROM t ORDER BY a ASC, b ASC")
        _assert_parity(lilli, sq, "SELECT id FROM t ORDER BY a DESC, b DESC")
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# MR-02 guard: rowid on a non-vec_label table == sqlite3
# ---------------------------------------------------------------------------


def test_rowid_non_vec_label_table_parity(tmp_path) -> None:
    """SELECT rowid + ORDER BY rowid on a table without vec_label matches sqlite3.

    A plain two-column table (no INTEGER PRIMARY KEY) has a standard B+tree
    rowid. Before the fix, row.get('vec_label') returned None/SqlNull for every
    row, so ORDER BY rowid was a no-op and SELECT rowid returned None.
    """
    ddl = "CREATE TABLE ev (k TEXT, v INTEGER)"
    insert = "INSERT INTO ev (k, v) VALUES (?, ?)"
    rows = [("a", 10), ("b", 20), ("c", 30)]
    lilli, sq = _seed_both(tmp_path, ddl, insert, rows)
    try:
        # rowid must be 1, 2, 3 on both drivers.
        _assert_parity(lilli, sq, "SELECT rowid, k FROM ev ORDER BY rowid")
        _assert_parity(lilli, sq, "SELECT k FROM ev ORDER BY rowid DESC")
        # Direct rowid value check: must be integers, not None.
        first = lilli.execute("SELECT rowid FROM ev ORDER BY rowid").fetchone()
        assert first[0] == 1, f"rowid on plain table should be 1, got {first[0]!r}"
        last = lilli.execute(
            "SELECT rowid FROM ev ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        assert last[0] == 3, f"last rowid should be 3, got {last[0]!r}"
    finally:
        lilli.close()
        sq.close()


def test_rowid_vec_label_table_unaffected(tmp_path) -> None:
    """rowid on a vec_label AUTOINCREMENT table still uses the HWM, not btree key."""
    rows = [("a", "2026-01-01"), ("b", "2026-01-02"), ("c", "2026-01-03")]
    lilli, sq = _seed_both(tmp_path, _RECORDS_DDL, _RECORDS_INSERT, rows)
    try:
        _assert_parity(lilli, sq, "SELECT rowid, id FROM records ORDER BY rowid")
        # HWM after DELETE+INSERT of the last row is 4, not the reused 3.
        lilli.execute("DELETE FROM records WHERE id = ?", ("c",))
        sq.execute("DELETE FROM records WHERE id = ?", ("c",))
        lilli.execute(_RECORDS_INSERT, ("d", "2026-01-04"))
        sq.execute(_RECORDS_INSERT, ("d", "2026-01-04"))
        sq.commit()
        _assert_parity(lilli, sq, "SELECT rowid, id FROM records ORDER BY rowid")
        hwm = lilli.execute(
            "SELECT rowid FROM records ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        assert hwm[0] == 4, f"HWM after delete+insert must be 4, got {hwm[0]!r}"
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# LR-01 guard: LIMIT n OFFSET m — parse + apply, parity with sqlite3
# ---------------------------------------------------------------------------


def test_limit_offset_parity(tmp_path) -> None:
    """``LIMIT n OFFSET m`` skips m rows then takes n, byte-identical to sqlite3."""
    ddl = "CREATE TABLE t (id TEXT, v INTEGER)"
    insert = "INSERT INTO t (id, v) VALUES (?, ?)"
    rows = [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)]
    lilli, sq = _seed_both(tmp_path, ddl, insert, rows)
    try:
        _assert_parity(lilli, sq, "SELECT id FROM t ORDER BY v ASC LIMIT 2 OFFSET 1")
        _assert_parity(lilli, sq, "SELECT id FROM t ORDER BY v ASC LIMIT 3 OFFSET 2")
        # OFFSET past the end of the set returns empty on both.
        _assert_parity(lilli, sq, "SELECT id FROM t ORDER BY v ASC LIMIT 2 OFFSET 10")
        # OFFSET 0 is identical to no OFFSET.
        _assert_parity(lilli, sq, "SELECT id FROM t ORDER BY v ASC LIMIT 2 OFFSET 0")
    finally:
        lilli.close()
        sq.close()


def test_limit_offset_param_parity(tmp_path) -> None:
    """``LIMIT ? OFFSET ?`` with bound params matches sqlite3."""
    ddl = "CREATE TABLE t (id TEXT, v INTEGER)"
    insert = "INSERT INTO t (id, v) VALUES (?, ?)"
    rows = [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)]
    lilli, sq = _seed_both(tmp_path, ddl, insert, rows)
    try:
        _assert_parity(
            lilli, sq, "SELECT id FROM t ORDER BY v ASC LIMIT ? OFFSET ?", (2, 1)
        )
        _assert_parity(
            lilli, sq, "SELECT id FROM t ORDER BY v ASC LIMIT ? OFFSET ?", (3, 0)
        )
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# HR-01: RENAME tombstones the old-name root meta row so re-open never
# re-aliases it — and a later re-CREATE of that name registers correctly.
# ---------------------------------------------------------------------------


def test_rename_tombstones_old_root_no_stale_alias(tmp_path) -> None:
    """After RENAME A→B + close + reopen: A is gone from both catalog and
    root_map; B is present and queryable; CREATE A works without bricking."""
    db_path = tmp_path / "rename_tombstone.db"

    # --- Session 1: create records, rename to records_old ---
    pager, catalog, root_map = _open_lilli_db(db_path)
    conn = LilliBrainConnection(pager, catalog, root_map)
    conn.execute(_RECORDS_DDL)
    conn.execute(_RECORDS_INSERT, ("r1", "2026-01-01"))
    conn.execute(_RECORDS_INSERT, ("r2", "2026-01-02"))
    conn.execute("ALTER TABLE [records] RENAME TO [records_old]")
    conn.close()

    # --- Session 2: re-open; verify stale alias is gone ---
    pager2, catalog2, root_map2 = _open_lilli_db(db_path)
    conn2 = LilliBrainConnection(pager2, catalog2, root_map2)
    try:
        # records_old is present and populated.
        out = _norm(
            conn2.execute(
                "SELECT id FROM records_old ORDER BY id"
            ).fetchall()
        )
        assert out == [("r1",), ("r2",)]

        # The old name is absent from both catalog and root_map.
        assert "records" not in root_map2, (
            f"stale 'records' alias survived re-open: root_map={root_map2}"
        )
        with pytest.raises(CatalogError):
            conn2.execute("SELECT id FROM records").fetchall()

        # Creating a NEW table named 'records' registers correctly — no brick.
        conn2.execute(_RECORDS_DDL)
        conn2.execute(_RECORDS_INSERT, ("fresh", "2026-06-01"))
        fresh = _norm(
            conn2.execute("SELECT id FROM records ORDER BY id").fetchall()
        )
        assert fresh == [("fresh",)], (
            f"new 'records' table bricked after RENAME tombstone: {fresh!r}"
        )
    finally:
        conn2.close()


def test_rename_reopen_then_create_same_name(tmp_path) -> None:
    """Mirror the reembed double-swap: records→records_old, then NEW records.
    After a reopen between the first and second swap the new 'records' must
    register and accept INSERTs without CatalogError."""
    db_path = tmp_path / "reembed_swap.db"

    # --- Session 1: records → records_old ---
    pager, catalog, root_map = _open_lilli_db(db_path)
    conn = LilliBrainConnection(pager, catalog, root_map)
    conn.execute(_RECORDS_DDL)
    conn.execute(_RECORDS_INSERT, ("old1", "2026-01-01"))
    conn.execute("ALTER TABLE [records] RENAME TO [records_old]")
    conn.close()

    # --- Session 2: reopen + create a fresh 'records' table ---
    pager2, catalog2, root_map2 = _open_lilli_db(db_path)
    conn2 = LilliBrainConnection(pager2, catalog2, root_map2)
    try:
        # records_old is intact.
        assert _norm(
            conn2.execute("SELECT id FROM records_old ORDER BY id").fetchall()
        ) == [("old1",)]

        # Create a new 'records' table and insert into it — must not brick.
        conn2.execute(_RECORDS_DDL)
        conn2.execute(_RECORDS_INSERT, ("new1", "2026-06-02"))
        rows = _norm(
            conn2.execute("SELECT id FROM records ORDER BY id").fetchall()
        )
        assert rows == [("new1",)], f"INSERT into re-created records bricked: {rows!r}"
    finally:
        conn2.close()


# ---------------------------------------------------------------------------
# MR-01: bracket-quoted CREATE TABLE name is accepted (parser parity with
# RENAME / DROP / SELECT which already accept [brackets]).
# ---------------------------------------------------------------------------


def test_bracket_create_table_registers(tmp_path) -> None:
    """``CREATE TABLE [tbl] (...)`` is accepted and the table is queryable."""
    pager, catalog, root_map = _open_lilli_db(tmp_path / "bracket_create.db")
    conn = LilliBrainConnection(pager, catalog, root_map)
    try:
        conn.execute("CREATE TABLE [staging] (id TEXT, v INTEGER)")
        conn.execute("INSERT INTO staging (id, v) VALUES (?, ?)", ("a", 1))
        rows = _norm(conn.execute("SELECT id, v FROM [staging] ORDER BY id").fetchall())
        assert rows == [("a", 1)], f"bracket-CREATE table not queryable: {rows!r}"
    finally:
        conn.close()


def test_bracket_create_if_not_exists(tmp_path) -> None:
    """``CREATE TABLE IF NOT EXISTS [tbl] (...)`` is accepted."""
    pager, catalog, root_map = _open_lilli_db(tmp_path / "bracket_ine.db")
    conn = LilliBrainConnection(pager, catalog, root_map)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS [staging] (id TEXT)")
        conn.execute("INSERT INTO staging (id) VALUES (?)", ("x",))
        rows = _norm(conn.execute("SELECT id FROM [staging]").fetchall())
        assert rows == [("x",)]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# LR-01: param-type mismatch for :name queries raises sqlite3.ProgrammingError
# (not ValueError) so callers catching sqlite3.Error see the same exception
# class as stdlib sqlite3.
# ---------------------------------------------------------------------------


def test_named_param_tuple_raises_programming_error(tmp_path) -> None:
    """Supplying a tuple instead of a dict for a :name query raises
    sqlite3.ProgrammingError, not ValueError."""
    pager, catalog, root_map = _open_lilli_db(tmp_path / "lr01.db")
    conn = LilliBrainConnection(pager, catalog, root_map)
    try:
        conn.execute("CREATE TABLE t (id TEXT, v INTEGER)")
        with pytest.raises(errors.ProgrammingError):
            conn.execute("SELECT id FROM t WHERE id = :x", ("a",)).fetchall()
    finally:
        conn.close()


def test_missing_named_param_raises_programming_error(tmp_path) -> None:
    """A :name placeholder with no value in the dict raises ProgrammingError."""
    pager, catalog, root_map = _open_lilli_db(tmp_path / "lr01b.db")
    conn = LilliBrainConnection(pager, catalog, root_map)
    try:
        conn.execute("CREATE TABLE t (id TEXT, v INTEGER)")
        with pytest.raises(errors.ProgrammingError):
            conn.execute("SELECT id FROM t WHERE id = :x", {}).fetchall()
    finally:
        conn.close()


def test_mixed_named_positional_raises_programming_error(tmp_path) -> None:
    """Mixing :name and ? in one statement raises ProgrammingError."""
    pager, catalog, root_map = _open_lilli_db(tmp_path / "lr01c.db")
    conn = LilliBrainConnection(pager, catalog, root_map)
    try:
        conn.execute("CREATE TABLE t (id TEXT, v INTEGER)")
        with pytest.raises(errors.ProgrammingError):
            conn.execute(
                "SELECT id FROM t WHERE id = :x AND v = ?", {"x": "a"}
            ).fetchall()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# MR-01: INSERT positional-arity mismatch raises ProgrammingError on both
# drivers (differential parity — lilli must match sqlite3's contract).
# ---------------------------------------------------------------------------


def test_insert_too_few_params_raises_programming_error(tmp_path) -> None:
    """INSERT VALUES (?,?) with one param raises ProgrammingError on both drivers."""
    pager, catalog, root_map = _open_lilli_db(tmp_path / "mr01a.db")
    lilli = LilliBrainConnection(pager, catalog, root_map)
    sq = sqlite3.connect(":memory:")
    ddl = "CREATE TABLE t (id TEXT, n INTEGER)"
    try:
        lilli.execute(ddl)
        sq.execute(ddl)
        # sqlite3 raises ProgrammingError for too-few params
        with pytest.raises(sqlite3.ProgrammingError):
            sq.execute("INSERT INTO t (id, n) VALUES (?, ?)", ("bad",))
        # lilli must raise the same exception type
        with pytest.raises(errors.ProgrammingError):
            lilli.execute("INSERT INTO t (id, n) VALUES (?, ?)", ("bad",))
    finally:
        lilli.close()
        sq.close()


def test_insert_too_many_params_raises_programming_error(tmp_path) -> None:
    """INSERT VALUES (?,?) with three params raises ProgrammingError on both drivers."""
    pager, catalog, root_map = _open_lilli_db(tmp_path / "mr01b.db")
    lilli = LilliBrainConnection(pager, catalog, root_map)
    sq = sqlite3.connect(":memory:")
    ddl = "CREATE TABLE t (id TEXT, n INTEGER)"
    try:
        lilli.execute(ddl)
        sq.execute(ddl)
        # sqlite3 raises ProgrammingError for too-many params
        with pytest.raises(sqlite3.ProgrammingError):
            sq.execute("INSERT INTO t (id, n) VALUES (?, ?)", ("a", 1, 999))
        # lilli must raise the same exception type
        with pytest.raises(errors.ProgrammingError):
            lilli.execute("INSERT INTO t (id, n) VALUES (?, ?)", ("a", 1, 999))
    finally:
        lilli.close()
        sq.close()


def test_insert_correct_arity_succeeds(tmp_path) -> None:
    """INSERT with exactly the right number of params succeeds on both drivers."""
    pager, catalog, root_map = _open_lilli_db(tmp_path / "mr01c.db")
    lilli = LilliBrainConnection(pager, catalog, root_map)
    sq = sqlite3.connect(":memory:")
    ddl = "CREATE TABLE t (id TEXT, n INTEGER)"
    try:
        lilli.execute(ddl)
        sq.execute(ddl)
        lilli.execute("INSERT INTO t (id, n) VALUES (?, ?)", ("ok", 42))
        sq.execute("INSERT INTO t (id, n) VALUES (?, ?)", ("ok", 42))
        lilli_rows = [tuple(r) for r in lilli.execute("SELECT id, n FROM t").fetchall()]
        sq_rows = [tuple(r) for r in sq.execute("SELECT id, n FROM t").fetchall()]
        assert lilli_rows == sq_rows == [("ok", 42)]
    finally:
        lilli.close()
        sq.close()


def test_insert_executemany_too_few_raises_programming_error(tmp_path) -> None:
    """executemany INSERT with a wrong-arity row raises ProgrammingError — both
    drivers raise on the bad row (the key contract for exception-type parity).
    Note: post-failure row state diverges intentionally: sqlite3 keeps rows
    committed before the bad entry; lilli rolls back the whole batch. That
    per-batch rollback is an engine-internal invariant, not a parity gap."""
    pager, catalog, root_map = _open_lilli_db(tmp_path / "mr01d.db")
    lilli = LilliBrainConnection(pager, catalog, root_map)
    sq = sqlite3.connect(":memory:")
    ddl = "CREATE TABLE t (id TEXT, n INTEGER)"
    try:
        lilli.execute(ddl)
        sq.execute(ddl)
        # Both engines must raise ProgrammingError on the bad row.
        with pytest.raises(sqlite3.ProgrammingError):
            sq.executemany(
                "INSERT INTO t (id, n) VALUES (?, ?)", [("a", 1), ("bad",)]
            )
        with pytest.raises(errors.ProgrammingError):
            lilli.executemany(
                "INSERT INTO t (id, n) VALUES (?, ?)", [("a", 1), ("bad",)]
            )
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# datetime(<expr>) bind + eval parity against the sqlite3 oracle
# ---------------------------------------------------------------------------

# A records-shaped schema with a tombstone column for the temporal predicate.
_TEMPORAL_DDL = (
    "CREATE TABLE records ("
    "vec_label INTEGER PRIMARY KEY AUTOINCREMENT, "
    "id TEXT NOT NULL UNIQUE, "
    "created_at TEXT, "
    "tombstoned_at TEXT, "
    "embedding_pending INTEGER)"
)
_TEMPORAL_INSERT = (
    "INSERT INTO records (id, created_at, tombstoned_at, embedding_pending)"
    " VALUES (?, ?, ?, ?)"
)
# A deliberately MIXED corpus: T-form, space-form, and date-only created_at
# strings, so only the datetime() evaluator (not any storage change) makes the
# inequality format-agnostic. One tombstoned row, one un-tombstoned.
_TEMPORAL_ROWS = [
    ("t_form", "2026-06-24T10:00:00.123456+00:00", None, 0),
    ("space_form", "2026-06-24 14:00:00.654321+00:00", None, 0),
    ("date_only", "2026-06-23", None, 0),
    ("tombstoned_late", "2026-06-24 09:00:00+00:00", "2026-06-24 20:00:00+00:00", 0),
    ("pending_row", "2026-06-24 11:00:00+00:00", None, 1),
]


def test_datetime_bind_inside_call_parity(tmp_path) -> None:
    """A ``?`` bound INSIDE datetime(?) binds in order; the trailing LIMIT ? is not desynced.

    This is the loud-failure tripwire for the binder: dropping the _bind_params
    FuncCall arm makes the inner ``?`` survive into eval (fromisoformat("?") crash)
    or desyncs the params queue, so this parity assertion fails.
    """
    lilli, sq = _seed_both(tmp_path, _TEMPORAL_DDL, _TEMPORAL_INSERT, _TEMPORAL_ROWS)
    try:
        boundary = "2026-06-24 12:00:00+00:00"
        # The inner ? binds to `boundary`, the trailing ? binds to the LIMIT.
        _assert_parity(
            lilli, sq,
            "SELECT id FROM records WHERE datetime(created_at) <= datetime(?)"
            " ORDER BY datetime(created_at) DESC LIMIT ?",
            (boundary, 10),
        )
        # A LIMIT smaller than the match count keeps the top-N of the sorted set;
        # if the params desynced, the LIMIT would consume the wrong value.
        _assert_parity(
            lilli, sq,
            "SELECT id FROM records WHERE datetime(created_at) <= datetime(?)"
            " ORDER BY datetime(created_at) DESC LIMIT ?",
            (boundary, 2),
        )
    finally:
        lilli.close()
        sq.close()


def test_as_of_no_vec_sql_shape_parity(tmp_path) -> None:
    """The EXACT no-vec as_of SQL shape with params (as_of, as_of, k) parities end-to-end.

    Both datetime(?) placeholders bind BEFORE the LIMIT ?; all three pop in order
    through the FuncCall bind arm.
    """
    lilli, sq = _seed_both(tmp_path, _TEMPORAL_DDL, _TEMPORAL_INSERT, _TEMPORAL_ROWS)
    try:
        as_of = "2026-06-24 12:00:00+00:00"
        sql = (
            "SELECT id FROM records"
            " WHERE (tombstoned_at IS NULL OR datetime(tombstoned_at) > datetime(?))"
            "   AND COALESCE(embedding_pending, 0) = 0"
            "   AND datetime(created_at) <= datetime(?)"
            " ORDER BY datetime(created_at) DESC"
            " LIMIT ?"
        )
        _assert_parity(lilli, sq, sql, (as_of, as_of, 50))
        # A boundary that places the tombstone strictly after as_of keeps the
        # tombstoned_late row in the point-in-time view on both engines.
        early = "2026-06-24 09:30:00+00:00"
        _assert_parity(lilli, sq, sql, (early, early, 50))
    finally:
        lilli.close()
        sq.close()


def test_datetime_where_mixed_corpus_parity(tmp_path) -> None:
    """``WHERE datetime(created_at) <op> datetime(?)`` over a mixed corpus == sqlite3.

    The corpus mixes T-form, space-form, and date-only created_at strings; only
    the datetime() evaluator makes the comparison separator-agnostic.
    """
    lilli, sq = _seed_both(tmp_path, _TEMPORAL_DDL, _TEMPORAL_INSERT, _TEMPORAL_ROWS)
    try:
        for boundary in (
            "2026-06-24 12:00:00+00:00",
            "2026-06-23 00:00:00+00:00",
            "2026-06-24 14:00:00+00:00",
        ):
            for op in ("<=", "<", ">", ">="):
                _assert_parity(
                    lilli, sq,
                    f"SELECT id FROM records WHERE datetime(created_at) {op} datetime(?)",
                    (boundary,),
                )
    finally:
        lilli.close()
        sq.close()


def test_datetime_value_parity_each_input_shape(tmp_path) -> None:
    """The evaluator output equals sqlite3 datetime() for T-form, space-form, and date-only.

    A SELECT-list ``datetime()`` projection is outside the supported subset, so
    the evaluator is asserted via WHERE-result id sets: for each input shape, an
    equality boundary set to that row's known normalised value selects exactly
    that row on both engines, pinning the evaluator's output byte-for-byte
    against the oracle.
    """
    # (id, stored created_at in its shape, the exact sqlite3 datetime() output).
    shape_rows = [
        ("row_t", "2026-06-24T10:00:00.123456+00:00", "2026-06-24 10:00:00"),
        ("row_space", "2026-06-24 14:00:00.654321+00:00", "2026-06-24 14:00:00"),
        ("row_date", "2026-06-23", "2026-06-23 00:00:00"),
    ]
    seed = [(rid, raw, None, 0) for rid, raw, _ in shape_rows]
    lilli, sq = _seed_both(tmp_path, _TEMPORAL_DDL, _TEMPORAL_INSERT, seed)
    try:
        for rid, _raw, normalised in shape_rows:
            # datetime(created_at) = datetime(<known normalised value>) selects
            # exactly the matching row on both engines — proving the evaluator
            # emits that normalised string for this input shape.
            _assert_parity(
                lilli, sq,
                "SELECT id FROM records WHERE datetime(created_at) = datetime(?)",
                (normalised,),
            )
            got = lilli.execute(
                "SELECT id FROM records WHERE datetime(created_at) = datetime(?)",
                (normalised,),
            ).fetchall()
            assert [r[0] for r in got] == [rid]
    finally:
        lilli.close()
        sq.close()


def test_datetime_null_propagation_parity(tmp_path) -> None:
    """datetime(NULL) is NULL: it is excluded by ``<=`` and included by the IS NULL OR form."""
    rows = [
        ("has_ts", "2026-06-24 10:00:00+00:00", None, 0),
        ("null_ts", None, None, 0),
    ]
    lilli, sq = _seed_both(tmp_path, _TEMPORAL_DDL, _TEMPORAL_INSERT, rows)
    try:
        boundary = "2026-06-24 12:00:00+00:00"
        # datetime(NULL) <= ? is NULL → the null_ts row is excluded on both.
        _assert_parity(
            lilli, sq,
            "SELECT id FROM records WHERE datetime(created_at) <= datetime(?)",
            (boundary,),
        )
        # The IS NULL OR form includes the null_ts row on both.
        _assert_parity(
            lilli, sq,
            "SELECT id FROM records WHERE created_at IS NULL"
            " OR datetime(created_at) <= datetime(?)",
            (boundary,),
        )
    finally:
        lilli.close()
        sq.close()


def test_datetime_order_by_mixed_corpus_parity(tmp_path) -> None:
    """``ORDER BY datetime(created_at) DESC`` over the mixed corpus matches sqlite3's order."""
    lilli, sq = _seed_both(tmp_path, _TEMPORAL_DDL, _TEMPORAL_INSERT, _TEMPORAL_ROWS)
    try:
        _assert_parity(
            lilli, sq,
            "SELECT id FROM records ORDER BY datetime(created_at) DESC",
        )
        _assert_parity(
            lilli, sq,
            "SELECT id FROM records ORDER BY datetime(created_at) ASC",
        )
    finally:
        lilli.close()
        sq.close()


def test_direct_temporal_sql_shape_parity(tmp_path) -> None:
    """Execute the degraded-recall builder's exact datetime(?) SQL shape via the engine.

    This guard executes (not just parses) the third builder's WHERE/ORDER-BY/LIMIT
    construct with params (as_of, as_of, limit); a binder regression fails here
    loudly even though the degraded_temporal_recall integration test would still
    pass green via its Python fallback.
    """
    lilli, sq = _seed_both(tmp_path, _TEMPORAL_DDL, _TEMPORAL_INSERT, _TEMPORAL_ROWS)
    try:
        as_of = "2026-06-24 12:00:00+00:00"
        sql = (
            "SELECT id FROM records"
            " WHERE (tombstoned_at IS NULL OR datetime(tombstoned_at) > datetime(?))"
            "   AND datetime(created_at) <= datetime(?)"
            " ORDER BY datetime(created_at) DESC"
            " LIMIT ?"
        )
        _assert_parity(lilli, sq, sql, (as_of, as_of, 10))
    finally:
        lilli.close()
        sq.close()


# An unparseable created_at on EVERY one of these shapes must yield datetime() ==
# NULL on both engines (sqlite3 never errors on a bad timestamp argument — it
# degrades that one value to NULL). The bind ``?`` form proves the scalar over a
# literal; the column form proves the scalar over a stored row.
_UNPARSEABLE_TS = [
    "garbage",
    "",
    "2026-13-99",
    "2026-06-24T25:00:00",
]


def test_datetime_unparseable_arg_is_null_parity(tmp_path) -> None:
    """datetime(<unparseable>) is NULL on both engines, not an error.

    A SELECT-list ``datetime()`` projection is outside the supported subset, so
    the scalar is pinned via WHERE-result id sets (the same technique the shape
    tests use). One corpus carries a distinct row for EACH unparseable shape plus
    a single valid row: every unparseable row is selected by
    ``datetime(created_at) IS NULL`` and excluded by ``datetime(created_at) <= ?``
    on BOTH drivers. sqlite3 IS the oracle, so any byte divergence fails here.
    """
    boundary = "2026-06-24 12:00:00+00:00"
    # A distinct id per unparseable shape, seeded once (the lilli engine persists
    # to its file, so re-seeding the same path would accumulate rows; seed-once +
    # id-set assertion is the file's established idiom for this).
    bad_ids = [f"bad_{i}" for i, _ in enumerate(_UNPARSEABLE_TS)]
    rows = [
        (rid, bad, None, 0) for rid, bad in zip(bad_ids, _UNPARSEABLE_TS)
    ]
    rows.append(("good_row", "2026-06-24 10:00:00+00:00", None, 0))
    lilli, sq = _seed_both(tmp_path, _TEMPORAL_DDL, _TEMPORAL_INSERT, rows)
    try:
        # Every unparseable row is selected by datetime(col) IS NULL on both
        # engines (sqlite3 degrades each to NULL; the engine must too).
        _assert_parity(
            lilli, sq,
            "SELECT id FROM records WHERE datetime(created_at) IS NULL"
            " ORDER BY id ASC",
        )
        got_null = lilli.execute(
            "SELECT id FROM records WHERE datetime(created_at) IS NULL"
            " ORDER BY id ASC"
        ).fetchall()
        assert [r[0] for r in got_null] == sorted(bad_ids), (
            "engine did not NULL every unparseable row"
        )
        # The same rows are excluded by the <= boundary (NULL <= x is NULL);
        # only the good row survives, byte-identical to sqlite3.
        _assert_parity(
            lilli, sq,
            "SELECT id FROM records WHERE datetime(created_at) <= datetime(?)",
            (boundary,),
        )
        got_le = lilli.execute(
            "SELECT id FROM records WHERE datetime(created_at) <= datetime(?)",
            (boundary,),
        ).fetchall()
        assert [r[0] for r in got_le] == ["good_row"], (
            "an unparseable row leaked through the <= boundary"
        )
    finally:
        lilli.close()
        sq.close()


def test_datetime_unparseable_arg_literal_is_null_parity(tmp_path) -> None:
    """datetime('<unparseable>') as a literal argument is NULL on both engines.

    Drives the scalar over a string LITERAL (not a column) by comparing a valid
    stored created_at against ``datetime('garbage')``: the boundary is NULL, so
    ``created_at <= NULL`` is NULL and the row is excluded on both drivers — the
    literal-form proof that the helper returns NULL rather than raising.
    """
    rows = [("a", "2026-06-24 10:00:00+00:00", None, 0)]
    lilli, sq = _seed_both(tmp_path, _TEMPORAL_DDL, _TEMPORAL_INSERT, rows)
    try:
        for bad in _UNPARSEABLE_TS:
            literal = bad.replace("'", "''")
            _assert_parity(
                lilli, sq,
                "SELECT id FROM records WHERE"
                f" datetime(created_at) <= datetime('{literal}')",
            )
            got = lilli.execute(
                "SELECT id FROM records WHERE"
                f" datetime(created_at) <= datetime('{literal}')"
            ).fetchall()
            assert got == [], f"literal {bad!r}: boundary did not degrade to NULL"
    finally:
        lilli.close()
        sq.close()


def test_no_vec_recall_shape_degrades_on_corrupt_created_at(tmp_path) -> None:
    """The no-vec recall SQL shape drops a corrupt created_at row instead of crashing.

    This mirrors the daemon-down point-in-time rail's exact WHERE/ORDER-BY/LIMIT
    construct. A genuinely unparseable created_at (external corruption / manual
    edit) must NOT raise out of the engine — sqlite3 NULL-degrades the one bad
    row and returns the rest; the engine must be byte-identical, and must not
    abort the whole recall.
    """
    rows = [
        ("good_early", "2026-06-24 09:00:00+00:00", None, 0),
        ("corrupt", "garbage", None, 0),
        ("good_late", "2026-06-24 11:00:00+00:00", None, 0),
        ("empty_ts", "", None, 0),
    ]
    lilli, sq = _seed_both(tmp_path, _TEMPORAL_DDL, _TEMPORAL_INSERT, rows)
    try:
        as_of = "2026-06-24 12:00:00+00:00"
        sql = (
            "SELECT id FROM records"
            " WHERE (tombstoned_at IS NULL OR datetime(tombstoned_at) > datetime(?))"
            "   AND COALESCE(embedding_pending, 0) = 0"
            "   AND datetime(created_at) <= datetime(?)"
            " ORDER BY datetime(created_at) DESC"
            " LIMIT ?"
        )
        # Does not raise (the pre-fix engine threw ValueError out of recall here),
        # and the corrupt + empty rows are dropped exactly as sqlite3 drops them.
        _assert_parity(lilli, sq, sql, (as_of, as_of, 50))
        got = lilli.execute(sql, (as_of, as_of, 50)).fetchall()
        assert [r[0] for r in got] == ["good_late", "good_early"], (
            "no-vec recall did not degrade the corrupt rows like sqlite3"
        )
    finally:
        lilli.close()
        sq.close()


# ---------------------------------------------------------------------------
# datetime value encoder storage shape (space-form, sqlite3 default-adapter parity)
# ---------------------------------------------------------------------------


def test_datetime_value_stored_space_form(tmp_path) -> None:
    """A datetime object binds and stores in space-form (index 10 == ' '), == sqlite3's adapter.

    Binding an actual datetime object routes through the encoder's datetime
    branch; the stored TEXT must carry a space separator at index 10 and equal
    the same datetime's ``isoformat(sep=" ")`` byte-for-byte, with the sizing site
    and the body site writing the identical string (a successful raw SELECT plus
    the length check rule out sizing/body drift).
    """
    pager, catalog, root_map = _open_lilli_db(tmp_path / "dtshape.db")
    lilli = LilliBrainConnection(pager, catalog, root_map)
    try:
        lilli.execute("CREATE TABLE t (id TEXT, ts TEXT)")
        dt = datetime(2026, 6, 24, 19, 40, 5, 651369, tzinfo=timezone.utc)
        lilli.execute("INSERT INTO t (id, ts) VALUES (?, ?)", ("a", dt))
        stored = lilli.execute("SELECT ts FROM t").fetchone()[0]
        expected = dt.isoformat(sep=" ")
        assert stored[10] == " ", f"separator at index 10 must be a space, got {stored[10]!r}"
        assert stored == expected, f"stored {stored!r} != isoformat(sep=' ') {expected!r}"
        # Sizing == body: the declared TEXT length matches the written bytes.
        assert len(stored) == len(expected)
    finally:
        lilli.close()


def test_date_value_stored_without_separator(tmp_path) -> None:
    """A plain date (not a datetime) stores as the bare 'YYYY-MM-DD' (no time separator)."""
    pager, catalog, root_map = _open_lilli_db(tmp_path / "dateshape.db")
    lilli = LilliBrainConnection(pager, catalog, root_map)
    try:
        lilli.execute("CREATE TABLE t (id TEXT, d TEXT)")
        d = date(2026, 6, 23)
        lilli.execute("INSERT INTO t (id, d) VALUES (?, ?)", ("a", d))
        stored = lilli.execute("SELECT d FROM t").fetchone()[0]
        assert stored == "2026-06-23"
        assert len(stored) == len(d.isoformat())
    finally:
        lilli.close()
