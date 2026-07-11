"""Regression: ``WHERE id IN (...)`` on the records table resolves through the
warm id-index (no ColIndex cold-build full scan).

Drives ``iai_mcp_native.engine.Connection`` directly to prove:

1. Full-scan count is zero: with the id-index warmed by a prior point-lookup
   the first ``SELECT ... WHERE id IN (?, ?, ?)`` pays zero full-tree scans
   (no ColIndex cold-build rebuild).  Before the fix the first ``id IN``
   triggered a ColIndex cold-build scan → full_scan_count() >= 1.

2. Scan equality: the rows the ``id IN (...)`` path returns are byte-identical
   to what stdlib sqlite3 returns for the same SQL + params (value, order
   under any surrounding clause, and Python type).  Both a present-only id
   list and a mixed present/absent id list are covered.

3. Any projection: the zero-scan + parity guarantee holds for both a bare
   ``SELECT id FROM records WHERE id IN (...)`` and a wide multi-column
   ``SELECT id, tier, centrality, ... FROM records WHERE id IN (...)``.

4. Write-boundary safety: a non-id UPDATE followed by a COMMIT does not
   invalidate the id-index, so the next ``id IN`` still pays zero full scans
   and returns correct post-commit rows.
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
# DDL and seed helpers
# ---------------------------------------------------------------------------

_DB_SEQ = {"n": 0}

# The records-table schema the engine tests use. Must match the production DDL
# shape (vec_label INTEGER PRIMARY KEY, id TEXT NOT NULL).
DDL = (
    "CREATE TABLE records ("
    "vec_label INTEGER PRIMARY KEY, "
    "id TEXT NOT NULL, "
    "tier TEXT, "
    "centrality REAL DEFAULT 0.0, "
    "embedding_pending INTEGER, "
    "payload TEXT"
    ")"
)

_INSERT = (
    "INSERT INTO records (vec_label, id, tier, centrality, embedding_pending, payload)"
    " VALUES (?, ?, ?, ?, ?, ?)"
)


def _engine_conn(tmp_path: Path):
    """Open a Rust engine connection and assert it IS the Rust class."""
    from iai_mcp_native import engine

    _DB_SEQ["n"] += 1
    conn = engine.Connection.open(str(tmp_path / f"idin_{_DB_SEQ['n']}.db"), 384)
    assert isinstance(conn, engine.Connection), (
        "connection under test is not the Rust engine.Connection — "
        "refusing to run a false-green proof against a different engine"
    )
    return conn


def _seed(conn, rows: list[tuple]) -> None:
    """Apply the DDL and seed (vec_label, id, tier, centrality, embedding_pending, payload)."""
    conn.execute(DDL, [])
    conn.execute("BEGIN", [])
    for row in rows:
        conn.execute(_INSERT, list(row))
    conn.execute("COMMIT", [])


def _make_rows(n: int = 300) -> list[tuple]:
    """Create `n` seed rows with distinct text ids."""
    return [
        (i, f"rec-{i:04d}", "episodic", float(i) * 0.1, i % 2, f"payload-{i}")
        for i in range(n)
    ]


def _norm(rows) -> list[tuple]:
    return [tuple(r) for r in rows]


def _types(rows) -> list[tuple]:
    return [tuple(type(v) for v in row) for row in rows]


def _assert_parity(eng, sq, sql: str, params=()) -> None:
    """Assert value + type parity between the Rust engine and stdlib sqlite3."""
    eng_rows = _norm(eng.execute(sql, list(params)).fetchall())
    sq_rows = _norm(sq.execute(sql, params).fetchall())
    assert eng_rows == sq_rows, (
        f"value parity mismatch for {sql!r} params={params!r}:\n"
        f"  engine={eng_rows!r}\n  sqlite={sq_rows!r}"
    )
    assert _types(eng_rows) == _types(sq_rows), (
        f"type parity mismatch for {sql!r} params={params!r}:\n"
        f"  engine types={_types(eng_rows)!r}\n  sqlite types={_types(sq_rows)!r}"
    )


# ---------------------------------------------------------------------------
# Test 1: warm-IdIndex id IN pays no full scan — bare projection
# ---------------------------------------------------------------------------


def test_idin_bare_projection_no_full_scan_warm(tmp_path: Path) -> None:
    """``SELECT id FROM records WHERE id IN (?, ?, ?)`` pays zero full scans.

    The id-index is warmed by a prior ``id = ?`` point lookup.  After that,
    any ``id IN (literals)`` must resolve through the warm id-index (O(1) per
    value) rather than triggering a cold ColIndex rebuild scan.

    Present-only and mixed present/absent id lists are both covered.
    """
    conn = _engine_conn(tmp_path)
    rows = _make_rows(300)
    _seed(conn, rows)

    # Warm the id-index with one point-lookup.
    conn.execute("SELECT id FROM records WHERE id = ? LIMIT 1", ["rec-0050"])
    conn.reset_full_scan_count()

    # Bare projection, present-only list — params form (post-binder: each ? → Literal).
    result = conn.execute(
        "SELECT id FROM records WHERE id IN (?, ?, ?)",
        ["rec-0010", "rec-0020", "rec-0030"],
    ).fetchall()
    assert conn.full_scan_count() == 0, (
        f"bare projection, present-only, params: expected full_scan_count==0 "
        f"(warm id-index lookup); got {conn.full_scan_count()} — "
        "the id IN path is triggering a ColIndex cold-build scan"
    )
    assert len(result) == 3

    # Mixed present/absent list.
    conn.reset_full_scan_count()
    result2 = conn.execute(
        "SELECT id FROM records WHERE id IN (?, ?, ?)",
        ["rec-0005", "rec-MISS", "rec-0015"],
    ).fetchall()
    assert conn.full_scan_count() == 0, (
        f"bare projection, mixed list, params: expected full_scan_count==0; "
        f"got {conn.full_scan_count()}"
    )
    # Two present, one absent.
    assert len(result2) == 2

    # Literal-interpolated form (no ? placeholders — tests the InList literal path).
    conn.reset_full_scan_count()
    result3 = conn.execute(
        "SELECT id FROM records WHERE id IN ('rec-0007', 'rec-0008', 'rec-NONE')",
        [],
    ).fetchall()
    assert conn.full_scan_count() == 0, (
        f"bare projection, literal IN (no params): expected full_scan_count==0; "
        f"got {conn.full_scan_count()}"
    )
    assert len(result3) == 2


# ---------------------------------------------------------------------------
# Test 2: warm-IdIndex id IN pays no full scan — wide projection
# ---------------------------------------------------------------------------


def test_idin_wide_projection_no_full_scan_warm(tmp_path: Path) -> None:
    """``SELECT id, tier, centrality, ... WHERE id IN (...)`` also pays no full scan."""
    conn = _engine_conn(tmp_path)
    rows = _make_rows(300)
    _seed(conn, rows)

    # Warm the id-index.
    conn.execute("SELECT id FROM records WHERE id = ? LIMIT 1", ["rec-0001"])
    conn.reset_full_scan_count()

    result = conn.execute(
        "SELECT id, tier, centrality, embedding_pending, payload"
        " FROM records WHERE id IN (?, ?, ?)",
        ["rec-0100", "rec-0200", "rec-ABSENT"],
    ).fetchall()
    assert conn.full_scan_count() == 0, (
        f"wide projection, mixed list: expected full_scan_count==0; "
        f"got {conn.full_scan_count()}"
    )
    # Two present, one absent.
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Test 3: scan-equality — id IN result is byte-identical to stdlib sqlite3
# ---------------------------------------------------------------------------


def test_idin_scan_equality_vs_sqlite3(tmp_path: Path) -> None:
    """The id-index-resolved ``id IN (...)`` result is byte-identical to stdlib sqlite3.

    Both a present-only id list and a mixed present/absent id list are compared
    for value and type parity.  The scan equality is the correctness bar: the
    index may only narrow which rows decode, not change the result.
    """
    rows = _make_rows(300)

    eng = _engine_conn(tmp_path)
    _seed(eng, rows)

    sq = sqlite3.connect(":memory:")
    sq.execute(DDL)
    for row in rows:
        sq.execute(_INSERT, row)
    sq.commit()

    # Warm the engine id-index.
    eng.execute("SELECT id FROM records WHERE id = ? LIMIT 1", ["rec-0001"])

    # Bare projection, present-only, params.
    _assert_parity(
        eng, sq,
        "SELECT id FROM records WHERE id IN (?, ?) ORDER BY id",
        ("rec-0010", "rec-0020"),
    )

    # Bare projection, mixed present/absent, params.
    _assert_parity(
        eng, sq,
        "SELECT id FROM records WHERE id IN (?, ?, ?) ORDER BY id",
        ("rec-0050", "rec-MISS", "rec-0100"),
    )

    # Wide projection, present-only, params.
    _assert_parity(
        eng, sq,
        "SELECT id, tier, centrality FROM records WHERE id IN (?, ?) ORDER BY id",
        ("rec-0003", "rec-0007"),
    )

    # Wide projection, mixed present/absent, literal IN (no params).
    _assert_parity(
        eng, sq,
        "SELECT id, tier FROM records"
        " WHERE id IN ('rec-0042', 'rec-NOWHERE', 'rec-0099') ORDER BY id",
    )


# ---------------------------------------------------------------------------
# Test 4: write-boundary safety — non-id UPDATE + COMMIT keeps id-index warm
# ---------------------------------------------------------------------------


def test_idin_after_non_id_update_commit(tmp_path: Path) -> None:
    """After a non-id UPDATE + COMMIT the id-index stays warm for id IN queries.

    A non-id column update (e.g. centrality) must not invalidate the id-index.
    The post-commit ``id IN (...)`` must still pay zero full scans and return
    the correct (post-update) rows.
    """
    rows = _make_rows(300)

    eng = _engine_conn(tmp_path)
    _seed(eng, rows)

    sq = sqlite3.connect(":memory:")
    sq.execute(DDL)
    for row in rows:
        sq.execute(_INSERT, row)
    sq.commit()

    # Warm the id-index.
    eng.execute("SELECT id FROM records WHERE id = ? LIMIT 1", ["rec-0001"])

    # Non-id UPDATE + COMMIT on both engines.
    eng.execute("BEGIN", [])
    eng.execute(
        "UPDATE records SET centrality = ? WHERE id = ?",
        [999.9, "rec-0050"],
    )
    eng.execute("COMMIT", [])

    sq.execute("UPDATE records SET centrality = ? WHERE id = ?", (999.9, "rec-0050"))
    sq.commit()

    # After the commit the id-index must still be warm.
    eng.reset_full_scan_count()
    eng.execute(
        "SELECT id, centrality FROM records WHERE id IN (?, ?)",
        ["rec-0050", "rec-0100"],
    ).fetchall()
    assert eng.full_scan_count() == 0, (
        f"post-commit id IN: expected full_scan_count==0 (id-index still warm); "
        f"got {eng.full_scan_count()}"
    )

    # Parity: post-commit rows are correct (post-update values visible).
    _assert_parity(
        eng, sq,
        "SELECT id, centrality FROM records WHERE id IN (?, ?) ORDER BY id",
        ("rec-0050", "rec-0100"),
    )
