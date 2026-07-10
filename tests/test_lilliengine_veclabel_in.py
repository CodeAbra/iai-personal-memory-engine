"""Batch-get via ``WHERE vec_label IN (...)`` — correctness and scan-path harness.

Drives ``iai_mcp_native.engine.Connection`` directly to prove:

1. Index-path gate: ``SELECT * FROM records WHERE vec_label IN (pk1, pk2, pk3)``
   resolves via the catalog ColIndex value-based path when
   ``idx_records_vec_label`` is declared on the connection (``full_scan_count()
   == 0`` on the warm second query).  The DDL in this module's seed helper
   declares the index — matching the production schema — so the ColIndex covers
   ``vec_label`` from the first query.

2. Differential parity: ``SELECT * FROM records WHERE vec_label IN (...)`` returns
   byte-identical rows, order, and types vs stdlib sqlite3 for both a
   present-only label list and a mixed present/absent list. The differential bar
   covers the production shape from the hnsw → metadata batch-get.

3. AND-tail parity: the production batch-get appends extra WHERE clauses
   (``AND (tombstoned_at IS NULL)``) to the ``vec_label IN`` base. This shape must
   also be byte-identical to stdlib sqlite3, proving the full predicate is
   re-applied to candidates.

The suite skips loudly when the native wheel is absent — it can never silently
fall through to a different engine class.
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

# Minimal records-shaped DDL with vec_label as INTEGER PRIMARY KEY AUTOINCREMENT,
# matching the production shape where the hnsw ANN labels map to vec_label.
DDL = (
    "CREATE TABLE records ("
    "vec_label INTEGER PRIMARY KEY AUTOINCREMENT, "
    "id TEXT NOT NULL, "
    "tier TEXT, "
    "centrality REAL DEFAULT 0.0, "
    "tombstoned_at TEXT, "
    "embedding_pending INTEGER"
    ")"
)

# Matches the production idx_records_vec_label declaration so the ColIndex
# covers vec_label from the first query on this connection.
_DDL_VEC_LABEL_INDEX = (
    "CREATE INDEX idx_records_vec_label ON records(vec_label)"
)

_INSERT = (
    "INSERT INTO records (id, tier, centrality, tombstoned_at, embedding_pending)"
    " VALUES (?, ?, ?, ?, ?)"
)


def _engine_conn(tmp_path: Path):
    """Open a Rust engine connection and assert it IS the Rust class."""
    from iai_mcp_native import engine

    _DB_SEQ["n"] += 1
    conn = engine.Connection.open(str(tmp_path / f"veclabel_{_DB_SEQ['n']}.db"), 384)
    assert isinstance(conn, engine.Connection), (
        "connection under test is not the Rust engine.Connection — "
        "refusing to run a false-green proof against a different engine"
    )
    return conn


def _seed_engine(conn, n: int = 300) -> list[int]:
    """Seed ``n`` rows without naming vec_label (AUTOINCREMENT assigns 1..n).

    Declares ``idx_records_vec_label`` before the INSERT batch, matching the
    production schema, so the catalog ColIndex covers ``vec_label`` from the
    first query.  Returns the list of assigned vec_label values (1-based) so
    tests can probe specific labels without hard-coding them.
    """
    conn.execute(DDL)
    conn.execute(_DDL_VEC_LABEL_INDEX)
    conn.execute("BEGIN")
    for i in range(n):
        tier = "episodic" if i % 2 == 0 else "semantic"
        tomb = "2026-01-01 00:00:00.000000" if i % 10 == 0 else None
        conn.execute(_INSERT, [f"rec-{i:04d}", tier, float(i) * 0.1, tomb, i % 2])
    conn.execute("COMMIT")
    # AUTOINCREMENT starts at 1; return the assigned range.
    return list(range(1, n + 1))


def _seed_sqlite3(n: int = 300) -> tuple[sqlite3.Connection, list[int]]:
    """Seed stdlib sqlite3 in-memory with the same DDL and rows."""
    sq = sqlite3.connect(":memory:")
    sq.row_factory = sqlite3.Row
    sq.execute(DDL)
    for i in range(n):
        tier = "episodic" if i % 2 == 0 else "semantic"
        tomb = "2026-01-01 00:00:00.000000" if i % 10 == 0 else None
        sq.execute(_INSERT, (f"rec-{i:04d}", tier, float(i) * 0.1, tomb, i % 2))
    sq.commit()
    return sq, list(range(1, n + 1))


def _norm(rows) -> list[tuple]:
    return [tuple(r) for r in rows]


def _types(rows) -> list[tuple]:
    return [tuple(type(v) for v in row) for row in rows]


def _assert_parity(eng, sq, sql: str, params=()) -> None:
    """Assert the Rust engine and stdlib sqlite3 return identical rows, order, and types."""
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
# Test 1 — index-path gate: SELECT * WHERE vec_label IN resolves via ColIndex
# ---------------------------------------------------------------------------


def test_veclabel_in_uses_index_path(tmp_path: Path) -> None:
    """``SELECT * FROM records WHERE vec_label IN (pk1, pk2, pk3)`` is index-served.

    When ``idx_records_vec_label`` is declared on the connection (matching the
    production schema), the catalog ColIndex covers ``vec_label`` and routes
    ``vec_label IN (...)`` through the value-based ColIndex path rather than a
    full-table scan.  The warm second query must produce ``full_scan_count() == 0``,
    confirming the index-served path is active and correct.

    The correctness of the returned rows is verified by the differential-parity
    tests below.
    """
    conn = _engine_conn(tmp_path)
    labels = _seed_engine(conn, 300)

    # First read: builds the ColIndex warm state from the declared index.
    probe_labels = [labels[9], labels[49], labels[149]]  # vec_label 10, 50, 150
    conn.execute(
        "SELECT * FROM records WHERE vec_label IN (?, ?, ?)",
        probe_labels,
    ).fetchall()
    conn.reset_full_scan_count()

    # Warm second read: must be index-served (ColIndex covers vec_label).
    conn.execute(
        "SELECT * FROM records WHERE vec_label IN (?, ?, ?)",
        probe_labels,
    ).fetchall()

    assert conn.full_scan_count() == 0, (
        f"expected full_scan_count==0 for warm vec_label IN (index-served via "
        f"idx_records_vec_label); got {conn.full_scan_count()} — "
        "the ColIndex is not routing vec_label IN through the catalog value-based path"
    )


# ---------------------------------------------------------------------------
# Test 2 — differential parity: vec_label IN result byte-identical to sqlite3
# ---------------------------------------------------------------------------


def test_veclabel_in_index_equality_vs_sqlite3(tmp_path: Path) -> None:
    """``SELECT * FROM records WHERE vec_label IN (...)`` is byte-identical to sqlite3.

    Covers a present-only label list and a mixed present/absent list. The
    ORDER BY vec_label anchor ensures identical row order on both drivers, so
    the assertion pins value, order, and type parity.  The index is declared in
    the seed DDL, so the engine resolves the query via the ColIndex path.
    """
    n = 300
    eng = _engine_conn(tmp_path)
    labels = _seed_engine(eng, n)

    sq, _ = _seed_sqlite3(n)

    # Present-only list: labels at positions 10, 20, 30 (vec_label 11, 21, 31).
    present = [labels[10], labels[20], labels[30]]
    _assert_parity(
        eng, sq,
        "SELECT * FROM records WHERE vec_label IN (?, ?, ?) ORDER BY vec_label",
        present,
    )

    # Mixed present/absent list: one absent label (0 is never assigned by AUTOINCREMENT).
    mixed = [labels[5], 0, labels[100]]
    _assert_parity(
        eng, sq,
        "SELECT * FROM records WHERE vec_label IN (?, ?, ?) ORDER BY vec_label",
        mixed,
    )

    # Single present label.
    _assert_parity(
        eng, sq,
        "SELECT * FROM records WHERE vec_label IN (?) ORDER BY vec_label",
        [labels[0]],
    )

    # All-absent list.
    _assert_parity(
        eng, sq,
        "SELECT * FROM records WHERE vec_label IN (?, ?) ORDER BY vec_label",
        [0, 99999],
    )


# ---------------------------------------------------------------------------
# Test 3 — AND-tail parity: production batch-get shape is byte-identical
# ---------------------------------------------------------------------------


def test_veclabel_in_with_and_tail(tmp_path: Path) -> None:
    """The production batch-get shape with an AND-tail is byte-identical to sqlite3.

    The real hnsw → metadata batch-get appends extra WHERE clauses to the
    ``vec_label IN (...)`` base, e.g. ``AND (tombstoned_at IS NULL)``. This test
    asserts that the engine re-applies the full predicate to ColIndex candidates so
    the AND-tail filters correctly — not merely returning all index-matched rows
    without the extra conjunct.
    """
    n = 300
    eng = _engine_conn(tmp_path)
    labels = _seed_engine(eng, n)

    sq, _ = _seed_sqlite3(n)

    # Row at index 0 has tombstoned_at set (i%10==0); AND-tail must exclude it.
    # labels[0]=1 (tombstoned), labels[1]=2 (live).
    probe = [labels[0], labels[1], labels[2]]

    _assert_parity(
        eng, sq,
        "SELECT * FROM records WHERE vec_label IN (?, ?, ?) AND (tombstoned_at IS NULL)"
        " ORDER BY vec_label",
        probe,
    )

    # Wider probe across tombstoned and live rows.
    wide = [labels[i] for i in range(0, 30, 3)]  # every 3rd label, including tombstoned
    placeholders = ", ".join("?" for _ in wide)
    _assert_parity(
        eng, sq,
        f"SELECT * FROM records WHERE vec_label IN ({placeholders}) AND (tombstoned_at IS NULL)"
        " ORDER BY vec_label",
        wide,
    )
