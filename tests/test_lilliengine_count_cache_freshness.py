"""Filtered-`COUNT(*)` freshness after a non-indexed-column UPDATE (Rust engine).

The engine's filtered-COUNT cache is fenced by the col-indexed write-generation,
which an UPDATE only advances when the SET touches a col-indexed column. A
tombstone/restore toggle (`UPDATE records SET tombstoned_at = ?`) changes a
predicate's row membership without touching any col-indexed column, so a
generation-only fence under-fences it: the cache can serve a stale over-count
that a live row scan (`SELECT COUNT(*)` vs `SELECT id ... `) would disagree with.
These tests lock `COUNT(*) == row scan` across predicate shapes, after both a
same-connection populate-then-mutate sequence and a cross-connection RO refresh.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


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


def _records_ddl() -> tuple[str, list[str]]:
    from iai_mcp.hippo import _table

    return _table._DDL_RECORDS, list(_table._DDL_RECORDS_INDEXES)


_INSERT_SQL = (
    "INSERT INTO records (id, tier, created_at, embedding, tombstoned_at, embedding_pending)"
    " VALUES (?, ?, ?, ?, ?, ?)"
)

_N_ROWS = 12


def _rows() -> list[tuple]:
    return [
        (f"r{i}", "episodic", f"2026-01-{i + 1:02d}T00:00:00", b"\x00\x01", None, i % 2)
        for i in range(_N_ROWS)
    ]


def _seed(conn, rows: list[tuple]) -> None:
    ddl, indexes = _records_ddl()
    conn.execute(ddl)
    for stmt in indexes:
        conn.execute(stmt)
    for row in rows:
        conn.execute(_INSERT_SQL, row)
    if isinstance(conn, sqlite3.Connection):
        conn.commit()


def _tombstone_then_restore(conn) -> None:
    """The exact non-indexed-column mutation shape: tombstone 3 rows, restore 1.

    `tombstoned_at` carries only a partial CREATE INDEX (`WHERE tombstoned_at
    IS NOT NULL`), which the engine's ColIndex set omits — so this UPDATE never
    advances `col_generation`, the property under test.
    """
    conn.execute(
        "UPDATE records SET tombstoned_at = ? WHERE id IN ('r0', 'r1', 'r2')",
        ("2026-02-01T00:00:00",),
    )
    conn.execute("UPDATE records SET tombstoned_at = NULL WHERE id = 'r1'")
    if isinstance(conn, sqlite3.Connection):
        conn.commit()


def _count(conn, where_sql: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM records WHERE {where_sql}").fetchone()[0]


def _scan_count(conn, where_sql: str) -> int:
    return len(conn.execute(f"SELECT id FROM records WHERE {where_sql}").fetchall())


_PREDICATE_SHAPES = {
    "single": "tombstoned_at IS NULL",
    "conjunction": "tombstoned_at IS NULL AND COALESCE(embedding_pending, 0) = 0",
    "disjunction": "tombstoned_at IS NOT NULL OR embedding_pending = 1",
    "coalesce_wrapped": "COALESCE(embedding_pending, 0) = 0",
    "early_column": "tombstoned_at IS NULL AND embedding_pending = 0",
    "trailing_column": "embedding_pending = 0 AND tombstoned_at IS NULL",
    "operand_swapped": "COALESCE(embedding_pending, 0) = 0 AND tombstoned_at IS NULL",
}


def _engine_conn(path: Path):
    from iai_mcp_native import engine

    conn = engine.Connection.open(str(path), 384)
    assert isinstance(conn, engine.Connection), (
        "the connection under test is not the Rust engine.Connection — "
        "refusing to run a false-green freshness proof against another engine"
    )
    return conn


@pytest.mark.parametrize("shape_name", sorted(_PREDICATE_SHAPES))
def test_filtered_count_matches_row_scan_after_non_indexed_update(tmp_path, shape_name) -> None:
    where_sql = _PREDICATE_SHAPES[shape_name]
    rows = _rows()

    eng = _engine_conn(tmp_path / f"{shape_name}.db")
    _seed(eng, rows)
    sq = sqlite3.connect(":memory:")
    sq.row_factory = sqlite3.Row
    _seed(sq, rows)

    # Cache-engagement precondition: a repeat of the identical filtered COUNT
    # must not re-scan (a cache hit), or the DDL failed to arm the cache and
    # every assertion below would pass vacuously on a broken AND a fixed engine.
    eng.reset_full_scan_count()
    _count(eng, where_sql)
    first_scan_count = eng.full_scan_count()
    _count(eng, where_sql)
    second_scan_delta = eng.full_scan_count() - first_scan_count
    assert second_scan_delta == 0, (
        f"the filtered-COUNT cache never armed for shape {shape_name!r} "
        f"(second identical COUNT re-scanned {second_scan_delta} times) — "
        "the DDL/predicate shape does not satisfy filtered_count_shape's "
        "ColIndex precondition, so this test cannot exercise the staleness bug"
    )

    _tombstone_then_restore(eng)
    _tombstone_then_restore(sq)

    lilli_count = _count(eng, where_sql)
    lilli_scan = _scan_count(eng, where_sql)
    stdlib_count = _count(sq, where_sql)
    assert lilli_count == lilli_scan, (
        f"stale filtered COUNT for shape {shape_name!r}: "
        f"COUNT(*)={lilli_count} row-scan={lilli_scan}"
    )
    assert lilli_count == stdlib_count, (
        f"lilli/stdlib COUNT mismatch for shape {shape_name!r}: "
        f"lilli={lilli_count} stdlib={stdlib_count}"
    )


def test_bare_count_stays_correct_after_non_indexed_update(tmp_path) -> None:
    """A1: bare COUNT(*) (fenced on col_generation only, unchanged by this fix)
    stays correct — an UPDATE never changes total row count, and INSERT/DELETE
    already bump col_generation under has_col_index."""
    rows = _rows()
    eng = _engine_conn(tmp_path / "bare.db")
    _seed(eng, rows)
    sq = sqlite3.connect(":memory:")
    sq.row_factory = sqlite3.Row
    _seed(sq, rows)

    eng.execute("SELECT COUNT(*) FROM records").fetchone()
    _tombstone_then_restore(eng)
    _tombstone_then_restore(sq)

    lilli_bare = eng.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    lilli_scan = len(eng.execute("SELECT id FROM records").fetchall())
    stdlib_bare = sq.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    assert lilli_bare == lilli_scan == stdlib_bare == _N_ROWS


def test_minmax_stays_correct_after_in_place_update_of_ordered_column(tmp_path) -> None:
    """A2: MIN() over a column with a non-partial declared index (eligible for
    the ordered-index fast path) reflects an in-place UPDATE of that same
    column. A failure here is a SEPARATE finding (ordered-index staleness),
    not fixed by this plan -- reported, not silently weakened.
    """
    rows = _rows()
    eng = _engine_conn(tmp_path / "minmax.db")
    _seed(eng, rows)
    sq = sqlite3.connect(":memory:")
    sq.row_factory = sqlite3.Row
    _seed(sq, rows)

    for conn in (eng, sq):
        conn.execute("SELECT MIN(created_at) FROM records")  # warm any ordered path
        conn.execute("UPDATE records SET created_at = ? WHERE id = 'r5'", ("2020-01-01T00:00:00",))
        if isinstance(conn, sqlite3.Connection):
            conn.commit()

    lilli_min = eng.execute("SELECT MIN(created_at) FROM records").fetchone()[0]
    stdlib_min = sq.execute("SELECT MIN(created_at) FROM records").fetchone()[0]
    scan_min = min(r[0] for r in eng.execute("SELECT created_at FROM records").fetchall())
    assert lilli_min == scan_min == stdlib_min == "2020-01-01T00:00:00"


def test_ro_mount_sees_fresh_filtered_count_after_refresh(tmp_path) -> None:
    """A RO mount's cached filtered count is fresh after `.refresh()` following a
    RW non-indexed UPDATE + commit. `.refresh()` MUST report `True` (the
    snapshot actually advanced) -- a `False` return would make the freshness
    check below pass vacuously on the OLD snapshot.
    """
    from iai_mcp_native import engine

    path = tmp_path / "ro_probe.db"
    rows = _rows()
    rw = _engine_conn(path)
    _seed(rw, rows)

    where_sql = _PREDICATE_SHAPES["conjunction"]
    ro = engine.Connection.open_read_only(str(path), 384)
    assert isinstance(ro, engine.Connection)
    ro_before = _count(ro, where_sql)
    assert ro_before == _scan_count(ro, where_sql)

    _tombstone_then_restore(rw)

    advanced = ro.raw_conn(read_only=True).refresh()
    assert advanced is True, (
        "refresh_read_view reported no snapshot advance after a committed RW "
        "write -- the freshness check below would pass vacuously on the stale "
        "pre-mutation snapshot"
    )

    ro_after = _count(ro, where_sql)
    ro_after_scan = _scan_count(ro, where_sql)
    assert ro_after == ro_after_scan, (
        f"RO mount served a stale filtered COUNT after refresh(): "
        f"COUNT(*)={ro_after} row-scan={ro_after_scan}"
    )
