"""Differential parity for the RUST SQL engine against stdlib sqlite3.

This harness drives the SAME SQL + parameters through stdlib ``sqlite3`` AND the
Rust engine ``iai_mcp_native.engine.Connection`` (opened on a tmp file via
``engine.Connection.open(path, embed_dim)``), and asserts the two return
identical result sets — rows, **order**, AND **types** (the storage class each
value crosses the boundary as: None / int / float / str / bytes). stdlib
``sqlite3`` IS the specification for the supported subset, so any divergence is a
Rust parity bug.

This is the primary Rust-routed parity acceptance: the connection under test is
the Rust engine class, asserted per case. A missing or stale native wheel makes
the ``iai_mcp_native.engine`` import fail, so the whole module SKIPS loudly — it
can never silently fall through to the Python engine.

The corpus covers the supported subset end to end: the storage table DDLs and
their representative INSERT / ON CONFLICT UPSERT / SELECT / UPDATE / DELETE
shapes, affinity, three-valued NULL logic, ``datetime()`` in WHERE, ORDER BY,
the scalar aggregates, GROUP BY + HAVING, named and positional binds, the
verbatim byte corpus (NUL / emoji / CJK / RTL / trailing whitespace, as TEXT and
BLOB), and the fail-loud rejections of out-of-subset SQL.
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
# Differential harness
# ---------------------------------------------------------------------------


# A per-test counter so two engines opened in one test get distinct files.
_DB_SEQ = {"n": 0}


def _engine_conn(tmp_path: Path):
    """Open a Rust engine connection on a tmp file and assert it IS the Rust class.

    The per-connection class assertion is the false-green guard: were a stale
    wheel to expose a non-Rust ``Connection``, this trips loudly rather than
    quietly exercising a different engine.
    """
    from iai_mcp_native import engine

    _DB_SEQ["n"] += 1
    conn = engine.Connection.open(str(tmp_path / f"engine_{_DB_SEQ['n']}.db"), 384)
    assert isinstance(conn, engine.Connection), (
        "the connection under test is not the Rust engine.Connection — "
        "refusing to run a false-green parity proof against another engine"
    )
    return conn


def _norm(rows) -> list[tuple]:
    """Map engine / sqlite3 result rows to plain comparable value tuples."""
    return [tuple(r) for r in rows]


def _types(rows) -> list[tuple]:
    """The per-cell Python type of every value, row by row.

    Asserting these alongside the values pins the storage class each value
    crosses the boundary as (an ``int`` 1 must not arrive as the str '1', a NULL
    must arrive as None), which a value-only compare would miss when both sides
    stringify the same.
    """
    return [tuple(type(v) for v in row) for row in rows]


def _seed(conn, ddl: str, insert_sql: str | None, rows: list[tuple]) -> None:
    """Apply the DDL and seed the rows into one connection."""
    conn.execute(ddl)
    if insert_sql is not None:
        for params in rows:
            conn.execute(insert_sql, params)
    if isinstance(conn, sqlite3.Connection):
        conn.commit()


def _both(tmp_path: Path, ddl: str, insert_sql: str | None, rows: list[tuple]):
    """Return a seeded (rust_engine, sqlite3) pair over the identical DDL + rows."""
    eng = _engine_conn(tmp_path)
    _seed(eng, ddl, insert_sql, rows)

    sq = sqlite3.connect(":memory:")
    sq.row_factory = sqlite3.Row
    _seed(sq, ddl, insert_sql, rows)
    return eng, sq


def _assert_parity(eng, sq, sql: str, params=()) -> None:
    """Assert the Rust engine and sqlite3 return identical rows, order, and types."""
    eng_rows = _norm(eng.execute(sql, params).fetchall())
    sq_rows = _norm(sq.execute(sql, params).fetchall())
    assert eng_rows == sq_rows, (
        f"value parity mismatch for {sql!r} params={params!r}:\n"
        f"  engine={eng_rows!r}\n  sqlite={sq_rows!r}"
    )
    assert _types(eng_rows) == _types(sq_rows), (
        f"type parity mismatch for {sql!r} params={params!r}:\n"
        f"  engine={_types(eng_rows)!r}\n  sqlite={_types(sq_rows)!r}"
    )


# The six storage DDLs the engine must accept and round-trip.
_DDL_RECORDS = (
    "CREATE TABLE records ("
    "vec_label INTEGER PRIMARY KEY AUTOINCREMENT, "
    "id TEXT NOT NULL UNIQUE, "
    "tier TEXT, "
    "created_at TEXT, "
    "tombstoned_at TEXT, "
    "score REAL, "
    "embedding_pending INTEGER)"
)
_RECORDS_INSERT = (
    "INSERT INTO records (id, tier, created_at, tombstoned_at, score, embedding_pending)"
    " VALUES (?, ?, ?, ?, ?, ?)"
)
_RECORDS_ROWS = [
    ("a", "episodic", "2026-01-03T00:00:00", None, 0.9, 1),
    ("b", "episodic", "2026-01-01T00:00:00", None, 0.5, None),
    ("c", "semantic", "2026-01-05T00:00:00", "2026-01-06T00:00:00", 0.1, 0),
    ("d", "episodic", "2026-01-02T00:00:00", None, 0.7, 1),
    ("e", "episodic", "2026-01-01T00:00:00", None, 0.7, 0),
]

_DDL_EDGES = (
    "CREATE TABLE edges ("
    "src TEXT NOT NULL, dst TEXT NOT NULL, edge_type TEXT NOT NULL, "
    "weight REAL NOT NULL DEFAULT 0.0, updated_at TEXT, "
    "PRIMARY KEY (src, dst, edge_type))"
)
_DDL_EVENTS = (
    "CREATE TABLE events ("
    "id TEXT PRIMARY KEY, kind TEXT NOT NULL, ts TEXT NOT NULL, data_json TEXT)"
)


# ---------------------------------------------------------------------------
# DDL + INSERT + SELECT round-trip, projection, aliases, types
# ---------------------------------------------------------------------------


def test_all_storage_ddls_accepted(tmp_path) -> None:
    """All six storage-shaped DDLs apply on the Rust engine without error."""
    eng = _engine_conn(tmp_path)
    for ddl in (
        _DDL_RECORDS,
        _DDL_EDGES,
        _DDL_EVENTS,
        "CREATE TABLE budget_ledger (date TEXT, usd_spent REAL, kind TEXT, ts TEXT)",
        "CREATE TABLE ratelimit_ledger (ts TEXT, status_code INTEGER, endpoint TEXT)",
        "CREATE TABLE _hippo_meta (key TEXT PRIMARY KEY, value TEXT)",
    ):
        eng.execute(ddl)


def test_select_projection_and_types(tmp_path) -> None:
    """SELECT *, a column subset, and aliases match sqlite3 in value AND type."""
    eng, sq = _both(tmp_path, _DDL_RECORDS, _RECORDS_INSERT, _RECORDS_ROWS)
    _assert_parity(eng, sq, "SELECT * FROM records ORDER BY id")
    _assert_parity(eng, sq, "SELECT id, tier, score, embedding_pending FROM records ORDER BY id")
    _assert_parity(
        eng, sq, "SELECT id AS rid, score AS s FROM records ORDER BY id"
    )
    # The REAL / INTEGER / TEXT / NULL storage classes round-trip with their types.
    _assert_parity(eng, sq, "SELECT score, embedding_pending FROM records ORDER BY id")


def test_named_and_positional_binds(tmp_path) -> None:
    """Both ``?`` and ``:name`` binds resolve identically to sqlite3."""
    eng, sq = _both(tmp_path, _DDL_RECORDS, _RECORDS_INSERT, _RECORDS_ROWS)
    _assert_parity(eng, sq, "SELECT id FROM records WHERE tier = ? ORDER BY id", ("episodic",))
    _assert_parity(
        eng, sq,
        "SELECT id FROM records WHERE tier = :t AND id LIKE :p ORDER BY id",
        {"t": "episodic", "p": "%"},
    )


# ---------------------------------------------------------------------------
# Three-valued NULL logic, IN / LIKE / IS NULL / COALESCE
# ---------------------------------------------------------------------------


def test_three_valued_null_logic(tmp_path) -> None:
    """IS NULL / IS NOT NULL / COALESCE behave with sqlite3's three-valued logic."""
    eng, sq = _both(tmp_path, _DDL_RECORDS, _RECORDS_INSERT, _RECORDS_ROWS)
    _assert_parity(eng, sq, "SELECT id FROM records WHERE tombstoned_at IS NULL ORDER BY id")
    _assert_parity(eng, sq, "SELECT id FROM records WHERE tombstoned_at IS NOT NULL ORDER BY id")
    _assert_parity(
        eng, sq,
        "SELECT id FROM records WHERE COALESCE(embedding_pending, 0) = 1 ORDER BY id",
    )
    _assert_parity(
        eng, sq,
        "SELECT id, COALESCE(embedding_pending, 0) AS p FROM records ORDER BY id",
    )


def test_like_and_in(tmp_path) -> None:
    """LIKE patterns and IN lists match sqlite3."""
    eng, sq = _both(tmp_path, _DDL_RECORDS, _RECORDS_INSERT, _RECORDS_ROWS)
    _assert_parity(eng, sq, "SELECT id FROM records WHERE id LIKE 'a%' ORDER BY id")
    _assert_parity(eng, sq, "SELECT id FROM records WHERE tier IN ('episodic') ORDER BY id")


def test_id_in_params_differential(tmp_path) -> None:
    """``WHERE id IN (?, ...)`` returns identical rows, order, and types as sqlite3.

    Covers a bare id-only projection and a wide multi-column projection, each
    with a present-only param tuple and a tuple that includes an absent id.
    """
    eng, sq = _both(tmp_path, _DDL_RECORDS, _RECORDS_INSERT, _RECORDS_ROWS)

    # Bare projection, present-only ids.
    _assert_parity(
        eng, sq,
        "SELECT id FROM records WHERE id IN (?, ?) ORDER BY id",
        ("a", "c"),
    )

    # Bare projection, mixed present/absent.
    _assert_parity(
        eng, sq,
        "SELECT id FROM records WHERE id IN (?, ?, ?) ORDER BY id",
        ("b", "MISSING", "e"),
    )

    # Wide projection, present-only.
    _assert_parity(
        eng, sq,
        "SELECT id, tier, score FROM records WHERE id IN (?, ?) ORDER BY id",
        ("a", "d"),
    )

    # Wide projection, mixed present/absent.
    _assert_parity(
        eng, sq,
        "SELECT id, tier, score FROM records WHERE id IN (?, ?, ?) ORDER BY id",
        ("c", "ABSENT", "e"),
    )


def test_veclabel_in_params_differential(tmp_path) -> None:
    """``WHERE vec_label IN (int, ...)`` returns identical rows, order, and types as sqlite3.

    vec_label is the INTEGER PRIMARY KEY so each IN value is a direct B-tree key.
    Covers present-only, mixed present/absent, AND-tail, and duplicate labels.
    """
    eng, sq = _both(tmp_path, _DDL_RECORDS, _RECORDS_INSERT, _RECORDS_ROWS)

    # All rows inserted by _both get vec_label 1..len(rows).
    # Present-only: labels 1 and 3.
    _assert_parity(
        eng, sq,
        "SELECT * FROM records WHERE vec_label IN (?, ?) ORDER BY vec_label",
        (1, 3),
    )

    # Mixed present/absent: label 0 is never assigned by AUTOINCREMENT.
    _assert_parity(
        eng, sq,
        "SELECT * FROM records WHERE vec_label IN (?, ?, ?) ORDER BY vec_label",
        (2, 0, 4),
    )

    # AND-tail: the predicate re-apply must enforce the extra conjunct.
    _assert_parity(
        eng, sq,
        "SELECT * FROM records WHERE vec_label IN (?, ?, ?) AND tombstoned_at IS NULL"
        " ORDER BY vec_label",
        (1, 2, 3),
    )

    # Duplicate label: must appear at most once, matching sqlite3.
    _assert_parity(
        eng, sq,
        "SELECT * FROM records WHERE vec_label IN (?, ?) ORDER BY vec_label",
        (1, 1),
    )


# ---------------------------------------------------------------------------
# datetime() WHERE + ORDER BY (the temporal false-pass guard) and recall shape
# ---------------------------------------------------------------------------


def test_datetime_where_and_order(tmp_path) -> None:
    """datetime(col) in WHERE and ORDER BY normalizes and orders like sqlite3."""
    eng, sq = _both(tmp_path, _DDL_RECORDS, _RECORDS_INSERT, _RECORDS_ROWS)
    _assert_parity(
        eng, sq,
        "SELECT id FROM records WHERE datetime(created_at) <= datetime(?)"
        " ORDER BY datetime(created_at) DESC",
        ("2026-01-03T00:00:00",),
    )


def test_datetime_literal_iso_in_where_and_order_is_unchanged(tmp_path) -> None:
    """The single-argument datetime() over a literal ISO timestamp is unchanged.

    The store / recall path issues only datetime(?) / datetime('<ISO literal>') in
    a WHERE / ORDER BY context (the engine evaluates datetime() there, not in the
    projection list). This pins that the literal-ISO form the host actually uses
    stays bit-identical to sqlite3 after the modifier handling was added.
    """
    eng, sq = _both(tmp_path, _DDL_RECORDS, _RECORDS_INSERT, _RECORDS_ROWS)
    # datetime(literal) compared in WHERE — the store's timestamp-bound filter.
    _assert_parity(
        eng, sq,
        "SELECT id FROM records"
        " WHERE datetime(created_at) <= datetime('2026-01-03T00:00:00') ORDER BY id",
    )
    # datetime(col) in ORDER BY — the recall-shaped temporal read.
    _assert_parity(
        eng, sq,
        "SELECT id FROM records ORDER BY datetime(created_at) ASC, id ASC",
    )


# A table for exercising datetime() modifiers over real rows in WHERE / ORDER BY.
_DDL_TIMES = "CREATE TABLE times (id TEXT, epoch INTEGER, iso TEXT)"
_TIMES_INSERT = "INSERT INTO times (id, epoch, iso) VALUES (?, ?, ?)"
_TIMES_ROWS = [
    ("a", 0, "1970-01-01T00:00:00"),
    ("b", 1_700_000_000, "2023-11-14T22:13:20"),
    ("c", 1_000_000_000, "2001-09-09T01:46:40"),
]


def test_datetime_unixepoch_modifier_matches_sqlite3(tmp_path) -> None:
    """datetime(epoch, 'unixepoch') in a WHERE comparison matches sqlite3.

    The modifier form is evaluated in the WHERE predicate (the engine's ORDER BY
    key grammar is single-column and does not carry a modifier, which the store /
    recall path never needs). Each comparison proves the epoch decode matches
    sqlite3 second for second.
    """
    eng, sq = _both(tmp_path, _DDL_TIMES, _TIMES_INSERT, _TIMES_ROWS)
    # Decoded epoch compared to a literal ISO timestamp selects the same row.
    _assert_parity(
        eng, sq,
        "SELECT id FROM times"
        " WHERE datetime(epoch, 'unixepoch') = datetime('2023-11-14T22:13:20')",
    )
    _assert_parity(
        eng, sq,
        "SELECT id FROM times"
        " WHERE datetime(epoch, 'unixepoch') = datetime('1970-01-01T00:00:00')",
    )
    # A range comparison over the decoded epochs matches sqlite3's row selection.
    _assert_parity(
        eng, sq,
        "SELECT id FROM times"
        " WHERE datetime(epoch, 'unixepoch') <= datetime('2010-01-01T00:00:00') ORDER BY id",
    )


def test_datetime_now_modifier_is_non_null_in_where(tmp_path) -> None:
    """datetime('now') resolves to a real timestamp usable in a WHERE comparison.

    'now' is non-deterministic, so the differential asserts a stable relational
    fact rather than an exact value: every seeded row's past timestamp is <=
    datetime('now') on both drivers, which holds only if datetime('now') returns a
    real current timestamp (not NULL) on the engine, matching sqlite3.
    """
    eng, sq = _both(tmp_path, _DDL_TIMES, _TIMES_INSERT, _TIMES_ROWS)
    _assert_parity(
        eng, sq,
        "SELECT id FROM times"
        " WHERE datetime(iso) <= datetime('now') ORDER BY id",
    )


def test_recall_shaped_order_by_score(tmp_path) -> None:
    """The recall read shape: WHERE not-tombstoned ORDER BY a score DESC LIMIT k."""
    eng, sq = _both(tmp_path, _DDL_RECORDS, _RECORDS_INSERT, _RECORDS_ROWS)
    _assert_parity(
        eng, sq,
        "SELECT id, score FROM records WHERE tombstoned_at IS NULL"
        " ORDER BY score DESC, id ASC LIMIT ?",
        (3,),
    )


def test_expr_projection_order_by_limit_returns_sorted_top_n(tmp_path) -> None:
    """ORDER BY + LIMIT on an EXPRESSION projection returns the sorted top-N.

    A projection carrying a COALESCE / alias / expression must still honor ORDER
    BY before LIMIT, returning the correctly-sorted top-N rather than the first
    rows in scan order. sqlite3 is the oracle, including DESC, the datetime
    normalization, and the secondary key.
    """
    eng, sq = _both(tmp_path, _DDL_RECORDS, _RECORDS_INSERT, _RECORDS_ROWS)
    # COALESCE projection, ORDER BY a source column ASC, LIMIT — top-N by created_at.
    _assert_parity(
        eng, sq,
        "SELECT id, COALESCE(embedding_pending, 0) AS p FROM records"
        " ORDER BY created_at ASC LIMIT 3",
    )
    # DESC ordering on the same expression projection.
    _assert_parity(
        eng, sq,
        "SELECT id, COALESCE(embedding_pending, 0) AS p FROM records"
        " ORDER BY created_at DESC LIMIT 2",
    )
    # ORDER BY a column NOT in the projection, with an aliased expression projected.
    _assert_parity(
        eng, sq,
        "SELECT COALESCE(tier, 'none') AS t FROM records"
        " ORDER BY score DESC LIMIT 2",
    )
    # Composite ORDER BY (two keys), datetime-normalized, with OFFSET + LIMIT.
    _assert_parity(
        eng, sq,
        "SELECT id, COALESCE(tier, 'none') AS t FROM records"
        " ORDER BY datetime(created_at) ASC, id DESC LIMIT 2 OFFSET 1",
    )
    # The full sorted projection with no LIMIT still matches (no truncation path).
    _assert_parity(
        eng, sq,
        "SELECT id, COALESCE(embedding_pending, 0) AS p FROM records"
        " ORDER BY score DESC, id ASC",
    )


def test_expr_projection_no_order_by_unchanged(tmp_path) -> None:
    """An expression projection with NO ORDER BY keeps the scan-order fast path.

    Without ORDER BY the projection returns its rows in the engine's scan order
    with LIMIT applied early; this pins that the fast path is preserved (the
    result still matches sqlite3 for the unordered LIMIT-less and the recall-shape
    cases the engine actually issues).
    """
    eng, sq = _both(tmp_path, _DDL_RECORDS, _RECORDS_INSERT, _RECORDS_ROWS)
    # No ORDER BY, no LIMIT — full set, both drivers in their natural row order.
    _assert_parity(
        eng, sq,
        "SELECT id, COALESCE(embedding_pending, 0) AS p FROM records ORDER BY id",
    )
    _assert_parity(
        eng, sq,
        "SELECT id FROM records ORDER BY created_at DESC LIMIT 2 OFFSET 1",
    )


# ---------------------------------------------------------------------------
# Aggregates + GROUP BY / HAVING
# ---------------------------------------------------------------------------


def test_aggregates(tmp_path) -> None:
    """COUNT(*), SUM, MIN, MAX (NULL-skip, empty -> NULL) match sqlite3."""
    eng, sq = _both(tmp_path, _DDL_RECORDS, _RECORDS_INSERT, _RECORDS_ROWS)
    _assert_parity(eng, sq, "SELECT COUNT(*) FROM records")
    _assert_parity(eng, sq, "SELECT COUNT(*) FROM records WHERE tombstoned_at IS NULL")
    _assert_parity(eng, sq, "SELECT MIN(score) FROM records")
    _assert_parity(eng, sq, "SELECT MAX(score) FROM records")
    _assert_parity(eng, sq, "SELECT SUM(embedding_pending) FROM records")
    _assert_parity(eng, sq, "SELECT MAX(created_at) FROM records WHERE tombstoned_at IS NULL")


def test_sum_over_real_column_promotes_to_float(tmp_path) -> None:
    """SUM over a REAL column returns a float total including the float rows.

    sqlite3 keeps an integer running total integral while every addend is an
    integer and promotes it to REAL the moment a REAL is summed, returning REAL
    if any input is REAL. A SUM that dropped the float rows (or returned Int)
    would diverge here — the guard for the score / weight / usd_spent REAL columns.
    """
    eng, sq = _both(tmp_path, _DDL_RECORDS, _RECORDS_INSERT, _RECORDS_ROWS)
    # score is REAL: SUM must include every float row and arrive as a float.
    _assert_parity(eng, sq, "SELECT SUM(score) FROM records")
    _assert_parity(
        eng, sq, "SELECT SUM(score) FROM records WHERE tombstoned_at IS NULL"
    )
    # An all-int column still sums to an int (no spurious promotion).
    _assert_parity(eng, sq, "SELECT SUM(embedding_pending) FROM records")

    # A REAL column summed over edges' weight (NOT NULL DEFAULT 0.0).
    upsert = (
        "INSERT INTO edges (src, dst, edge_type, weight, updated_at)"
        " VALUES (?, ?, ?, ?, ?)"
    )
    eng2, sq2 = _both(tmp_path, _DDL_EDGES, None, [])
    for params in [
        ("a", "b", "rel", 1.5, "t1"),
        ("c", "d", "rel", 2.25, "t1"),
        ("e", "f", "rel", 4.0, "t1"),
    ]:
        eng2.execute(upsert, params)
        sq2.execute(upsert, params)
    sq2.commit()
    _assert_parity(eng2, sq2, "SELECT SUM(weight) FROM edges")


def test_real_into_text_renders_with_decimal_point(tmp_path) -> None:
    """A REAL coerced into a TEXT column renders like sqlite3 (1.0 -> '1.0').

    sqlite3 stores a whole-valued REAL bound into a TEXT-affinity column with its
    trailing '.0'; the engine must match, not drop it to '1'. The same rendering
    feeds a float LHS in LIKE.
    """
    ddl = "CREATE TABLE tf (id TEXT, label TEXT, val REAL)"
    insert = "INSERT INTO tf (id, label, val) VALUES (?, ?, ?)"
    rows = [
        ("a", 1.0, 1.0),    # a whole float into a TEXT column -> '1.0'
        ("b", 0.5, 0.5),
        ("c", 100.0, 100.0),
        ("d", 9.0, 9.0),
    ]
    eng, sq = _both(tmp_path, ddl, insert, rows)
    # The TEXT column carries the float's sqlite3 text form, type str both sides.
    _assert_parity(eng, sq, "SELECT id, label FROM tf ORDER BY id")
    # A float LHS in LIKE is stringified the same way ('9.0' LIKE '9%').
    _assert_parity(eng, sq, "SELECT id FROM tf WHERE val LIKE '1%' ORDER BY id")
    _assert_parity(eng, sq, "SELECT id FROM tf WHERE label LIKE '1.0' ORDER BY id")


def test_affinity_on_write_matches_sqlite3_storage_class(tmp_path) -> None:
    """Write-time affinity coercion stores the same storage class as sqlite3.

    sqlite3 applies column affinity on write: a whole-valued REAL bound into an
    INTEGER column stores as an integer, an int bound into a REAL column stores as
    a real, and a numeric string bound into a TEXT column stays text. The engine
    must round-trip the identical storage class — pinned here against the sqlite3
    oracle so the coercion can never silently diverge.
    """
    ddl = "CREATE TABLE ta (id TEXT, i INTEGER, r REAL, x TEXT)"
    insert = "INSERT INTO ta (id, i, r, x) VALUES (?, ?, ?, ?)"
    rows = [
        # 1.0 -> INTEGER col stores int 1; 5 -> REAL col stores 5.0;
        # '123' -> TEXT col stays the string '123'.
        ("row1", 1.0, 5, "123"),
        ("row2", 42, 2, "abc"),
    ]
    eng, sq = _both(tmp_path, ddl, insert, rows)
    # Value AND type parity pins the storage class each cell round-trips as.
    _assert_parity(eng, sq, "SELECT id, i, r, x FROM ta ORDER BY id")


def test_comparison_honors_column_affinity_like_sqlite3(tmp_path) -> None:
    """WHERE comparison of a typed column to a literal honors column affinity.

    sqlite3 applies the column's affinity to the literal: a TEXT-affinity column
    compared to an integer literal takes TEXT affinity, so a '05' cell does not
    equal 5 (only a '5' cell does). The converse INTEGER-affinity column matches a
    numeric string. The engine must produce the same row selection.
    """
    ddl = "CREATE TABLE af (id TEXT, c TEXT, n INTEGER)"
    insert = "INSERT INTO af (id, c, n) VALUES (?, ?, ?)"
    rows = [
        ("a", "05", 5),
        ("b", "5", 5),
        ("c", "50", 50),
    ]
    eng, sq = _both(tmp_path, ddl, insert, rows)
    # TEXT column vs integer literal: '05' must NOT match 5; only '5' matches.
    _assert_parity(eng, sq, "SELECT id FROM af WHERE c = 5 ORDER BY id")
    # The literal on the left side of a TEXT column behaves the same.
    _assert_parity(eng, sq, "SELECT id FROM af WHERE 5 = c ORDER BY id")
    # A TEXT column vs a text literal compares as text (unchanged path).
    _assert_parity(eng, sq, "SELECT id FROM af WHERE c = '05' ORDER BY id")
    # The converse: an INTEGER column vs a numeric string still matches (the int
    # column path is not regressed).
    _assert_parity(eng, sq, "SELECT id FROM af WHERE n = '5' ORDER BY id")
    _assert_parity(eng, sq, "SELECT id FROM af WHERE n = 50 ORDER BY id")
    # A range comparison on the TEXT column is lexicographic under TEXT affinity.
    _assert_parity(eng, sq, "SELECT id FROM af WHERE c < 5 ORDER BY id")


def test_group_by_having(tmp_path) -> None:
    """GROUP BY tier + COUNT(*) + HAVING matches sqlite3 one row per group."""
    eng, sq = _both(tmp_path, _DDL_RECORDS, _RECORDS_INSERT, _RECORDS_ROWS)
    _assert_parity(eng, sq, "SELECT tier, COUNT(*) AS n FROM records GROUP BY tier ORDER BY tier")
    _assert_parity(
        eng, sq,
        "SELECT tier, COUNT(*) AS n FROM records GROUP BY tier HAVING COUNT(*) > 1 ORDER BY tier",
    )


# ---------------------------------------------------------------------------
# ON CONFLICT / excluded — composite-PK edges UPSERT
# ---------------------------------------------------------------------------


def test_on_conflict_do_update_and_do_nothing(tmp_path) -> None:
    """ON CONFLICT DO UPDATE (excluded.col) and DO NOTHING match sqlite3 row-for-row."""
    upsert = (
        "INSERT INTO edges (src, dst, edge_type, weight, updated_at)"
        " VALUES (?, ?, ?, ?, ?)"
        " ON CONFLICT (src, dst, edge_type) DO UPDATE SET"
        " weight = excluded.weight, updated_at = excluded.updated_at"
    )
    eng, sq = _both(tmp_path, _DDL_EDGES, None, [])
    seq = [
        ("a", "b", "rel", 1.0, "t1"),
        ("a", "b", "rel", 9.0, "t2"),  # same key → DO UPDATE in place
        ("a", "b", "other", 2.0, "t1"),
        ("x", "y", "rel", 3.0, "t1"),
    ]
    for params in seq:
        eng.execute(upsert, params)
        sq.execute(upsert, params)
    sq.commit()
    # The supported ORDER BY is up to two keys; (src, edge_type) totally orders
    # this seed (no two edges share both).
    _assert_parity(
        eng, sq,
        "SELECT src, dst, edge_type, weight, updated_at FROM edges ORDER BY src, edge_type",
    )

    # DO NOTHING on an all-key table keeps the first write.
    eng2, sq2 = _both(
        tmp_path, "CREATE TABLE tags (a TEXT, b TEXT, PRIMARY KEY (a, b))", None, []
    )
    nothing = "INSERT INTO tags (a, b) VALUES (?, ?) ON CONFLICT (a, b) DO NOTHING"
    for params in [("x", "y"), ("x", "y"), ("x", "z")]:
        eng2.execute(nothing, params)
        sq2.execute(nothing, params)
    sq2.commit()
    _assert_parity(eng2, sq2, "SELECT a, b FROM tags ORDER BY a, b")


def test_do_update_rewriting_conflict_key_matches_sqlite3(tmp_path) -> None:
    """A DO UPDATE that rewrites a conflict-key column re-keys conflict detection.

    After the key column moves from one value to another, inserting the OLD value
    must insert fresh (no false conflict) and inserting the NEW value must conflict
    and update in place. sqlite3 is the oracle; the engine must produce the same
    final row set.
    """
    ddl = (
        "CREATE TABLE keyed ("
        "k INTEGER PRIMARY KEY, label TEXT)"
    )
    upsert = (
        "INSERT INTO keyed (k, label) VALUES (?, ?)"
        " ON CONFLICT (k) DO UPDATE SET label = excluded.label"
    )
    eng, sq = _both(tmp_path, ddl, None, [])
    for conn in (eng, sq):
        # Seed one row at k=1.
        conn.execute(upsert, (1, "first"))
        # Rewrite the conflict-key column k from 1 to 2 via a literal SET.
        conn.execute(
            "INSERT INTO keyed (k, label) VALUES (?, ?)"
            " ON CONFLICT (k) DO UPDATE SET k = 2, label = excluded.label",
            (1, "moved"),
        )
        # Inserting the OLD key value (1) must insert fresh — no false conflict.
        conn.execute(upsert, (1, "fresh_old"))
        # Inserting the NEW key value (2) must conflict and update in place.
        conn.execute(upsert, (2, "updated_new"))
    sq.commit()
    _assert_parity(eng, sq, "SELECT k, label FROM keyed ORDER BY k")


def test_do_update_onto_third_row_key_raises_unique(tmp_path) -> None:
    """A DO UPDATE that moves the conflict key onto a THIRD row's key raises UNIQUE.

    With rows at k=1 and k=3, a DO UPDATE that rewrites row 1's key to 3 collides
    with the existing row 3. sqlite3 raises 'UNIQUE constraint failed: keyed.k'
    rather than letting two rows share the key; the engine must reject it the same
    way and leave both rows untouched, byte for byte.
    """
    ddl = "CREATE TABLE keyed (k INTEGER PRIMARY KEY, label TEXT)"
    upsert = (
        "INSERT INTO keyed (k, label) VALUES (?, ?)"
        " ON CONFLICT (k) DO UPDATE SET label = excluded.label"
    )
    rewrite = (
        "INSERT INTO keyed (k, label) VALUES (?, ?)"
        " ON CONFLICT (k) DO UPDATE SET k = 3, label = excluded.label"
    )
    eng, sq = _both(tmp_path, ddl, None, [])
    for conn in (eng, sq):
        conn.execute(upsert, (1, "first"))
        conn.execute(upsert, (3, "third"))
    # The engine is an autocommit store (each statement commits); commit sqlite3's
    # seeds to the same point so a later rollback discards only the failed rewrite.
    sq.commit()
    for conn in (eng, sq):
        # Rewriting row 1's key onto row 3's existing key must raise on both drivers.
        with pytest.raises(Exception) as exc:
            conn.execute(rewrite, (1, "moved"))
        assert "unique" in str(exc.value).lower(), (
            f"a key rewrite onto a third row must raise UNIQUE, got {exc.value!r}"
        )
    sq.rollback()  # discard sqlite3's failed-statement implicit transaction
    # Both original rows survive unchanged on both drivers.
    _assert_parity(eng, sq, "SELECT k, label FROM keyed ORDER BY k")


def test_executemany_failing_tail_rolls_back_atomically(tmp_path) -> None:
    """A batch whose tail row violates a constraint leaves no surviving rows.

    The engine's executemany wraps the whole batch in one store transaction and
    rolls it back atomically when any row fails, so none of the batch's rows
    survive and no transaction is left open. stdlib sqlite3 reaches the same
    surviving-row state once the failed batch's open implicit transaction is rolled
    back (sqlite3 leaves the predecessor rows in an open transaction until then).
    Rolling sqlite3 back to that point makes the two drivers' committed state
    identical, which is what the host (an autocommit store driver) observes: a
    failed bulk write leaves the store exactly as it was, byte for byte.
    """
    ddl = (
        "CREATE TABLE batch ("
        "id TEXT NOT NULL UNIQUE, n INTEGER)"
    )
    insert = "INSERT INTO batch (id, n) VALUES (?, ?)"
    seq = [
        ("alpha", 1),
        ("beta", 2),
        (None, 3),  # id is NOT NULL → the tail row fails
    ]
    eng, sq = _both(tmp_path, ddl, None, [])
    with pytest.raises(Exception):
        eng.executemany(insert, seq)
    with pytest.raises(Exception):
        sq.executemany(insert, seq)
    # The engine already rolled the batch back internally; bring sqlite3 to the
    # same committed state by rolling back its still-open implicit transaction.
    sq.rollback()

    # Neither the failed row nor its predecessors survive on either driver.
    _assert_parity(eng, sq, "SELECT id, n FROM batch ORDER BY id")
    _assert_parity(eng, sq, "SELECT id FROM batch WHERE id = 'alpha'")
    _assert_parity(eng, sq, "SELECT COUNT(*) FROM batch")


# ---------------------------------------------------------------------------
# UPDATE / DELETE
# ---------------------------------------------------------------------------


def test_update_and_delete(tmp_path) -> None:
    """UPDATE ... WHERE and DELETE ... WHERE leave the same surviving rows as sqlite3."""
    eng, sq = _both(tmp_path, _DDL_RECORDS, _RECORDS_INSERT, _RECORDS_ROWS)
    for conn in (eng, sq):
        conn.execute("UPDATE records SET tier = ? WHERE tier = ?", ("ep", "episodic"))
        conn.execute("DELETE FROM records WHERE id = ?", ("c",))
    sq.commit()
    _assert_parity(eng, sq, "SELECT id, tier FROM records ORDER BY id")
    _assert_parity(eng, sq, "SELECT COUNT(*) FROM records WHERE tier = 'ep'")


# ---------------------------------------------------------------------------
# UPDATE SET <column-referencing RHS> — must read the row's pre-update value,
# not evaluate against an empty row.
# ---------------------------------------------------------------------------

_SET_COL_REF_DDL = (
    "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, c TEXT, "
    "d INTEGER, ts TEXT)"
)
_SET_COL_REF_INSERT = "INSERT INTO t (id, a, b, c, d, ts) VALUES (?, ?, ?, ?, ?, ?)"
_SET_COL_REF_ROW = [(1, 5, None, "hi", None, "2026-01-01T00:00:00")]


@pytest.mark.parametrize(
    "update_sql, select_sql",
    [
        ("UPDATE t SET b = a WHERE id = 1", "SELECT b FROM t WHERE id = 1"),
        ("UPDATE t SET b = t.a WHERE id = 1", "SELECT b FROM t WHERE id = 1"),
        ("UPDATE t SET b = COALESCE(a, 0) WHERE id = 1", "SELECT b FROM t WHERE id = 1"),
        ("UPDATE t SET b = COALESCE(a, d) WHERE id = 1", "SELECT b FROM t WHERE id = 1"),
        ("UPDATE t SET c = DATETIME(ts) WHERE id = 1", "SELECT c FROM t WHERE id = 1"),
        ("UPDATE t SET b = (a) WHERE id = 1", "SELECT b FROM t WHERE id = 1"),
        ("UPDATE t SET b = 5 WHERE id = 1", "SELECT b FROM t WHERE id = 1"),
    ],
)
def test_update_set_col_ref_class(tmp_path, update_sql, select_sql) -> None:
    """Every column-referencing SET RHS form the parser accepts (bare col, dotted
    col, COALESCE over a col, DATETIME(col), a parenthesized col) reads the row's
    current value, matching sqlite3; the literal RHS case guards against
    regressing the already-correct path. Arithmetic BinOp RHS forms (`a+1`) fail
    loud at parse on this engine and are out of scope.
    """
    eng, sq = _both(tmp_path, _SET_COL_REF_DDL, _SET_COL_REF_INSERT, _SET_COL_REF_ROW)
    for conn in (eng, sq):
        conn.execute(update_sql)
    sq.commit()
    _assert_parity(eng, sq, select_sql)


def test_update_set_col_ref_multi_assignment_swap(tmp_path) -> None:
    """SET a = b, b = a swaps using PRE-update values on both engines (sqlite parity),
    not a partially-applied read of `updated`."""
    rows = [(1, 7, 3, "hi", None, "2026-01-01T00:00:00")]
    eng, sq = _both(tmp_path, _SET_COL_REF_DDL, _SET_COL_REF_INSERT, rows)
    for conn in (eng, sq):
        conn.execute("UPDATE t SET a = b, b = a WHERE id = 1")
    sq.commit()
    _assert_parity(eng, sq, "SELECT a, b FROM t WHERE id = 1")


@pytest.mark.parametrize("rowid_spelling", ["rowid", "ROWID", "RowId"])
def test_update_set_col_ref_rowid_heal_damaged_shape(tmp_path, rowid_spelling) -> None:
    """SET vec_label = rowid resolves to the raw storage key on the damaged shape
    (vec_label a plain non-PK INTEGER, NULL) — the boot heal's actual production
    shape (`_db.py` `_heal_null_vec_labels`). The expected value is sourced from
    stdlib's own rowid, never from lilli's own SELECT rowid — on this shape
    `resolve_rowid` returns the (still NULL, pre-heal) vec_label instead of the
    raw key, a separately reported divergence, not fixed here. Any case spelling
    of `rowid` must resolve identically (SQL identifiers are case-insensitive).
    """
    ddl = "CREATE TABLE t (vec_label INTEGER, id TEXT NOT NULL UNIQUE)"
    insert_sql = "INSERT INTO t (vec_label, id) VALUES (?, ?)"
    eng, sq = _both(tmp_path, ddl, insert_sql, [(None, "r1")])
    sq.execute("UPDATE t SET vec_label = rowid WHERE vec_label IS NULL")
    eng.execute(f"UPDATE t SET vec_label = {rowid_spelling} WHERE vec_label IS NULL")
    sq.commit()
    expected = sq.execute("SELECT vec_label FROM t WHERE id = 'r1'").fetchone()[0]
    assert expected is not None
    got = eng.execute("SELECT vec_label FROM t WHERE id = 'r1'").fetchone()[0]
    assert got == expected, (
        f"damaged-shape rowid heal mismatch ({rowid_spelling=}): "
        f"lilli={got!r} stdlib={expected!r}"
    )


def test_update_set_col_ref_rowid_heal_normal_pk_shape(tmp_path) -> None:
    """SET vec_label = rowid on a normal INTEGER PRIMARY KEY row is a lilli-only
    self-consistency check, NOT a stdlib-parity lock: lilli does not alias
    INTEGER PRIMARY KEY to the storage key (`next_vec_label` high-water mark vs
    `max_key()+1` are independent allocators), so a divergence from stdlib here
    is expected, not a regression.
    """
    ddl = "CREATE TABLE t (vec_label INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE)"
    eng = _engine_conn(tmp_path)
    eng.execute(ddl)
    eng.execute("INSERT INTO t (id) VALUES (?)", ("r1",))
    eng.execute("UPDATE t SET vec_label = rowid WHERE id = 'r1'")
    got = eng.execute("SELECT vec_label FROM t WHERE id = 'r1'").fetchone()[0]
    assert got == 1, (
        f"a single fresh insert's raw storage key is the 1-based insertion "
        f"ordinal (1); got vec_label={got!r}"
    )


@pytest.mark.parametrize(
    "update_sql, select_sql, pre_update_value",
    [
        ("UPDATE t SET b = COALESCE(rowid, 0) WHERE id = 1", "SELECT b FROM t WHERE id = 1", None),
        ("UPDATE t SET c = DATETIME(rowid) WHERE id = 1", "SELECT c FROM t WHERE id = 1", "hi"),
    ],
)
def test_update_set_wrapped_rowid_fails_loud(
    tmp_path, update_sql, select_sql, pre_update_value
) -> None:
    """A `rowid` reference wrapped in COALESCE/DATETIME on a SET RHS raises rather
    than silently writing NULL — only the bare top-level `rowid` form resolves to
    the raw storage key. The target column must be left holding its pre-update
    value, not overwritten with NULL before the raise.
    """
    eng = _engine_conn(tmp_path)
    _seed(eng, _SET_COL_REF_DDL, _SET_COL_REF_INSERT, _SET_COL_REF_ROW)
    with pytest.raises(Exception) as exc:
        eng.execute(update_sql)
    assert "rowid" in str(exc.value).lower(), (
        f"the rejection message {str(exc.value)!r} does not name rowid"
    )
    survived = eng.execute(select_sql).fetchone()[0]
    assert survived == pre_update_value, (
        f"a rejected wrapped-rowid SET must leave the column at its pre-update "
        f"value {pre_update_value!r}, got {survived!r} — the reject must be "
        f"data-independent (pre-`TxnGuard`), never a partial write"
    )


# ---------------------------------------------------------------------------
# Error-shape parity: read-only mount, transaction-control guards, CREATE TABLE
# ---------------------------------------------------------------------------


def test_query_only_off_cannot_flip_a_readonly_mount(tmp_path) -> None:
    """A connection opened read-only stays read-only even after query_only=OFF.

    A mount opened read-only is permanently read-only; clearing the mutable
    query_only flag does not grant write access. A write still fails with the
    read-only error, matching sqlite3 on a database opened with mode=ro.
    """
    from iai_mcp_native import engine

    path = str(tmp_path / "ro.db")
    # Create + seed via a normal writable connection, then close it.
    w = engine.Connection.open(path, 384)
    w.execute(_DDL_RECORDS)
    w.execute(_RECORDS_INSERT, _RECORDS_ROWS[0])
    w.close()

    # Reopen read-only; clearing query_only must NOT make it writable.
    ro = engine.Connection.open_read_only(path, 384)
    ro.execute("PRAGMA query_only=OFF")
    with pytest.raises(Exception) as exc:
        ro.execute(_RECORDS_INSERT, ("z", "episodic", "2026-02-01T00:00:00", None, 0.3, 0))
    assert "readonly" in str(exc.value).lower(), (
        f"a write on a read-only mount must report the read-only error, got {exc.value!r}"
    )
    # The read path still works on the read-only mount.
    rows = _norm(ro.execute("SELECT id FROM records ORDER BY id").fetchall())
    assert rows == [("a",)]


def test_commit_or_rollback_with_no_active_transaction_errors_like_sqlite3(tmp_path) -> None:
    """A SQL-string COMMIT / ROLLBACK with no open transaction errors like sqlite3.

    sqlite3 raises 'cannot commit - no transaction is active' /
    'cannot rollback - no transaction is active'. The engine must surface the
    same no-transaction error rather than driving an illegal pager call.
    """
    for verb in ("COMMIT", "ROLLBACK"):
        eng = _engine_conn(tmp_path)
        eng.execute(_DDL_RECORDS)
        sq = sqlite3.connect(":memory:")
        sq.isolation_level = None  # autocommit so no implicit transaction is open
        sq.execute(_DDL_RECORDS)

        with pytest.raises(Exception) as sq_exc:
            sq.execute(verb)
        with pytest.raises(Exception) as eng_exc:
            eng.execute(verb)
        assert "no transaction" in str(eng_exc.value).lower(), (
            f"{verb} with no active transaction must report the no-transaction error, "
            f"got engine={eng_exc.value!r} sqlite={sq_exc.value!r}"
        )


def test_create_table_on_existing_errors_unless_if_not_exists(tmp_path) -> None:
    """A second CREATE TABLE of an existing table errors unless IF NOT EXISTS.

    sqlite3 raises 'table X already exists' on a bare repeat CREATE TABLE, and
    no-ops when IF NOT EXISTS is given. The engine must match both.
    """
    ddl = "CREATE TABLE dup (id TEXT, n INTEGER)"
    # Bare repeat → both drivers error.
    eng = _engine_conn(tmp_path)
    eng.execute(ddl)
    sq = sqlite3.connect(":memory:")
    sq.execute(ddl)
    with pytest.raises(Exception) as sq_exc:
        sq.execute(ddl)
    with pytest.raises(Exception) as eng_exc:
        eng.execute(ddl)
    assert "already exists" in str(eng_exc.value).lower(), (
        f"a repeat CREATE TABLE must report table-already-exists, "
        f"got engine={eng_exc.value!r} sqlite={sq_exc.value!r}"
    )

    # IF NOT EXISTS → both no-op (no error) on the existing table.
    eng2 = _engine_conn(tmp_path)
    eng2.execute(ddl)
    eng2.execute("CREATE TABLE IF NOT EXISTS dup (id TEXT, n INTEGER)")
    sq2 = sqlite3.connect(":memory:")
    sq2.execute(ddl)
    sq2.execute("CREATE TABLE IF NOT EXISTS dup (id TEXT, n INTEGER)")


# ---------------------------------------------------------------------------
# The verbatim byte corpus — literal_surface parity through the Rust engine
# ---------------------------------------------------------------------------

_VERBATIM = [
    ("nul", "a\x00b"),
    ("emoji", "memory \U0001f9e0 recall"),
    ("cjk", "記憶を思い出す"),
    ("rtl", "مرحبا"),
    ("trail_ws", "value with trailing   "),
    ("mixed", "\U0001f300\x00中文  "),
]


def test_verbatim_text_corpus(tmp_path) -> None:
    """NUL / emoji / CJK / RTL / trailing-whitespace TEXT round-trips byte-identical."""
    ddl = "CREATE TABLE t (id TEXT, s TEXT)"
    insert = "INSERT INTO t (id, s) VALUES (?, ?)"
    eng, sq = _both(tmp_path, ddl, insert, _VERBATIM)
    _assert_parity(eng, sq, "SELECT id, s FROM t ORDER BY id")
    # Each value is recovered exactly as stored (str type preserved).
    for rid, raw in _VERBATIM:
        got = eng.execute("SELECT s FROM t WHERE id = ?", (rid,)).fetchone()[0]
        assert got == raw


def test_verbatim_blob_corpus(tmp_path) -> None:
    """Arbitrary BLOB bytes (incl. NUL and non-UTF8) round-trip byte-identical."""
    ddl = "CREATE TABLE b (id TEXT, payload BLOB)"
    insert = "INSERT INTO b (id, payload) VALUES (?, ?)"
    blobs = [
        ("empty", b""),
        ("nul", b"\x00\x00\x00"),
        ("binary", b"\x00\xff\xfe\x80\x7f"),
        ("emoji_bytes", "\U0001f9e0".encode("utf-8")),
        ("trailing", b"abc   "),
    ]
    eng, sq = _both(tmp_path, ddl, insert, blobs)
    _assert_parity(eng, sq, "SELECT id, payload FROM b ORDER BY id")
    for bid, raw in blobs:
        got = eng.execute("SELECT payload FROM b WHERE id = ?", (bid,)).fetchone()[0]
        assert got == raw
        assert isinstance(got, bytes)


# ---------------------------------------------------------------------------
# Fail-loud: out-of-subset SQL rejected with the same message the engine surfaces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql, needle",
    [
        ("SELECT DISTINCT id FROM records", "DISTINCT"),
        ("SELECT id FROM records UNION SELECT id FROM records", "UNION"),
    ],
)
def test_unsupported_sql_fails_loud(tmp_path, sql, needle) -> None:
    """Out-of-subset SQL the engine does not implement raises (never silently wrong).

    sqlite3 accepts these, so this is not a parity-with-sqlite3 case but a
    fail-closed contract: the Rust engine rejects the grammar it cannot honor
    with a message naming the unsupported construct, rather than returning a
    quietly different result. Asserting the message keeps the rejection wording
    aligned with what the engine surfaces to callers.
    """
    eng = _engine_conn(tmp_path)
    eng.execute(_DDL_RECORDS)
    with pytest.raises(Exception) as exc:
        eng.execute(sql).fetchall()
    assert needle in str(exc.value), (
        f"the rejection message {str(exc.value)!r} does not name {needle!r}"
    )
