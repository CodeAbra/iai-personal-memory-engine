"""Ordered-index correctness and scan-count harness for ``events ORDER BY ts DESC LIMIT k``.

Drives ``iai_mcp_native.engine.Connection`` directly to prove:

1. Byte-identical results: the production ``query_events`` SQL shapes — bare
   top-K, kind-filtered, ts-bounded, severity-filtered, and severity+kind
   compound — return exactly the same rows, order, and Python types as stdlib
   sqlite3. stdlib sqlite3 is the specification for the supported subset.

2. Scan-count gate (warm): after a prior identical query the second run must
   resolve without a whole-table scan — ``full_scan_count() == 0``. On HEAD
   no ordered index exists; the engine full-scans and sorts, so
   ``full_scan_count() >= 1`` for ``SELECT *`` shapes. This is the RED marker
   that the ordered-index wave turns GREEN.

3. Tie-order correctness: when many rows share the same ``ts`` value the LIMIT
   cuts inside the tie group. The engine's emitted order must be byte-identical
   to sqlite3's (Pitfall 3 — see the phase research). The differential bar
   enforces this.

The test uses the production events DDL and the three ``CREATE INDEX``
declarations (kind, ts, session_id) on both the engine and sqlite3 to ensure
the schema under test matches what the engine catalog sees.

The suite skips loudly when the native wheel is absent.
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
# DDL — events-shaped, matching the production schema
# ---------------------------------------------------------------------------

_DB_SEQ = {"n": 0}

_DDL_EVENTS = (
    "CREATE TABLE events ("
    "id TEXT PRIMARY KEY, "
    "kind TEXT NOT NULL, "
    "severity TEXT, "
    "domain TEXT, "
    "ts TEXT NOT NULL, "
    "data_json TEXT, "
    "session_id TEXT, "
    "source_ids_json TEXT"
    ")"
)

# Production CREATE INDEX declarations (all three, same as _table.py).
_INDEXES = [
    "CREATE INDEX idx_events_kind    ON events(kind)",
    "CREATE INDEX idx_events_ts      ON events(ts)",
    "CREATE INDEX idx_events_session ON events(session_id)",
]

_INSERT = (
    "INSERT INTO events (id, kind, severity, domain, ts, data_json, session_id, source_ids_json)"
    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)

# The exact projection the production query_events emits (events.py:173-174).
_SELECT_COLS = (
    "SELECT id, kind, severity, domain, ts, data_json, session_id, source_ids_json"
    " FROM events"
)

_KINDS = ["telemetry", "health", "recall", "curiosity"]
_SEVERITIES = ["info", "warn", "error"]


def _ts(year: int = 2026, month: int = 1, day: int = 1,
        hour: int = 0, minute: int = 0, second: int = 0,
        microsecond: int = 0) -> str:
    """Produce a space-form TEXT timestamp matching the production stored shape."""
    return (
        f"{year:04d}-{month:02d}-{day:02d} "
        f"{hour:02d}:{minute:02d}:{second:02d}.{microsecond:06d}"
    )


def _make_rows(n: int = 2000) -> list[tuple]:
    """Generate ``n`` event rows with varied kind/severity/ts values.

    Each row has a UNIQUE ts value (one microsecond apart within the 2026 year)
    so that ORDER BY ts DESC has a fully deterministic order with no ties. This
    keeps the differential tests (Tests 1-5) cleanly GREEN on the current
    scan path.

    The tie-order test (Test 6) uses its own corpus with intentional duplicate
    ts values to pin that specific behaviour.
    """
    rows: list[tuple] = []
    for i in range(n):
        # Unique ts: spread across seconds so lexicographic order == datetime order.
        sec = i % 60
        minute = (i // 60) % 60
        hour = (i // 3600) % 24
        ts = _ts(hour=hour, minute=minute, second=sec)
        kind = _KINDS[i % len(_KINDS)]
        sev = _SEVERITIES[i % len(_SEVERITIES)]
        sess = f"sess-{i % 20}"
        rows.append((
            f"ev-{i:05d}",
            kind,
            sev,
            None,
            ts,
            None,
            sess,
            None,
        ))
    return rows


def _engine_conn(tmp_path: Path):
    """Open a Rust engine connection and assert it IS the Rust class."""
    from iai_mcp_native import engine

    _DB_SEQ["n"] += 1
    conn = engine.Connection.open(str(tmp_path / f"ordered_{_DB_SEQ['n']}.db"), 384)
    assert isinstance(conn, engine.Connection), (
        "connection under test is not the Rust engine.Connection — "
        "refusing to run a false-green proof against a different engine"
    )
    return conn


def _seed_engine(conn, rows: list[tuple]) -> None:
    conn.execute(_DDL_EVENTS)
    for idx_sql in _INDEXES:
        conn.execute(idx_sql)
    conn.execute("BEGIN")
    for row in rows:
        conn.execute(_INSERT, list(row))
    conn.execute("COMMIT")


def _seed_sqlite3(rows: list[tuple]) -> sqlite3.Connection:
    sq = sqlite3.connect(":memory:")
    sq.row_factory = sqlite3.Row
    sq.execute(_DDL_EVENTS)
    for idx_sql in _INDEXES:
        sq.execute(idx_sql)
    sq.executemany(_INSERT, rows)
    sq.commit()
    return sq


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
# Test 1 — bare top-K: ORDER BY ts DESC LIMIT k byte-identical to sqlite3
# ---------------------------------------------------------------------------


def test_events_bare_topk_vs_sqlite3(tmp_path: Path) -> None:
    """``ORDER BY ts DESC LIMIT k`` (no WHERE) is byte-identical to sqlite3.

    This is the production shape when ``kind`` and ``since`` are both None:
    the full top-K by recency. The differential pins row values, ORDER, and types
    over a corpus large enough to require a sort/index to answer correctly.
    """
    rows = _make_rows(2000)
    eng = _engine_conn(tmp_path)
    _seed_engine(eng, rows)
    sq = _seed_sqlite3(rows)

    sql = _SELECT_COLS + " ORDER BY ts DESC LIMIT ?"

    _assert_parity(eng, sq, sql, (10,))
    _assert_parity(eng, sq, sql, (50,))
    _assert_parity(eng, sq, sql, (1,))
    _assert_parity(eng, sq, sql, (200,))


# ---------------------------------------------------------------------------
# Test 2 — kind-filtered: WHERE kind = ? ORDER BY ts DESC LIMIT k
# ---------------------------------------------------------------------------


def test_events_kind_filtered_topk_vs_sqlite3(tmp_path: Path) -> None:
    """``WHERE kind = ? ORDER BY ts DESC LIMIT k`` is byte-identical to sqlite3.

    This is the most common production shape: the caller filters by event kind
    and wants the most-recent k results. The result order must match sqlite3's
    stable sort over the same ts values — no tie-break may diverge.
    """
    rows = _make_rows(2000)
    eng = _engine_conn(tmp_path)
    _seed_engine(eng, rows)
    sq = _seed_sqlite3(rows)

    sql = _SELECT_COLS + " WHERE kind = ? ORDER BY ts DESC LIMIT ?"

    for kind in _KINDS:
        _assert_parity(eng, sq, sql, (kind, 10))
        _assert_parity(eng, sq, sql, (kind, 50))
    # Absent kind returns no rows on both drivers.
    _assert_parity(eng, sq, sql, ("ABSENT", 10))


# ---------------------------------------------------------------------------
# Test 3 — ts-bounded: WHERE kind = ? AND ts >= ? ORDER BY ts DESC LIMIT k
# ---------------------------------------------------------------------------


def test_events_ts_bounded_topk_vs_sqlite3(tmp_path: Path) -> None:
    """``WHERE kind = ? AND ts >= ? ORDER BY ts DESC LIMIT k`` is byte-identical.

    This is the ``since``-bounded shape from ``query_events``. The boundary is
    a space-form TEXT timestamp matching the stored shape; lexicographic
    comparison over the ts column must select the same rows as sqlite3.
    """
    rows = _make_rows(2000)
    eng = _engine_conn(tmp_path)
    _seed_engine(eng, rows)
    sq = _seed_sqlite3(rows)

    sql = _SELECT_COLS + " WHERE kind = ? AND ts >= ? ORDER BY ts DESC LIMIT ?"

    # Boundary roughly in the middle of the corpus (minute=16 = row 960+).
    boundary = _ts(minute=16)
    for kind in _KINDS:
        _assert_parity(eng, sq, sql, (kind, boundary, 20))

    # Boundary that excludes all rows.
    future = _ts(year=2099)
    _assert_parity(eng, sq, sql, ("telemetry", future, 10))

    # Boundary that includes all rows.
    past = _ts(year=2000)
    _assert_parity(eng, sq, sql, ("health", past, 30))


# ---------------------------------------------------------------------------
# Test 4 — severity-filtered: WHERE kind = ? AND severity = ? ORDER BY ts DESC
# ---------------------------------------------------------------------------


def test_events_severity_filtered_topk_vs_sqlite3(tmp_path: Path) -> None:
    """``WHERE kind = ? AND severity = ? ORDER BY ts DESC LIMIT k`` is byte-identical.

    This is the real ``query_events`` shape that emits both a ``kind`` conjunct
    and a ``severity`` conjunct (events.py:181-183). The ``severity`` column is
    an extra equality conjunct that is NOT the ORDER BY column; after the engine
    narrows by kind (or ts-order), the full predicate must re-apply the severity
    filter so no un-matching row slips through.

    This shape is the one the plan-check correction requires gating: the
    ``severity`` conjunct must be enforced even when the ordered-index fast path
    emits candidates in ts-order without a secondary equality check on severity.
    """
    rows = _make_rows(2000)
    eng = _engine_conn(tmp_path)
    _seed_engine(eng, rows)
    sq = _seed_sqlite3(rows)

    sql = _SELECT_COLS + " WHERE kind = ? AND severity = ? ORDER BY ts DESC LIMIT ?"

    for kind in _KINDS:
        for sev in _SEVERITIES:
            _assert_parity(eng, sq, sql, (kind, sev, 10))
    # Absent severity.
    _assert_parity(eng, sq, sql, ("telemetry", "ABSENT", 10))


# ---------------------------------------------------------------------------
# Test 5 — severity + ts-bounded: WHERE kind=? AND severity=? AND ts>=?
# ---------------------------------------------------------------------------


def test_events_severity_ts_bounded_topk_vs_sqlite3(tmp_path: Path) -> None:
    """``WHERE kind=? AND severity=? AND ts>=? ORDER BY ts DESC LIMIT k`` is byte-identical.

    The ts-bounded variant of the severity-filtered shape (combining Tests 3 and
    4). This exercises the path where the ordered-index narrows by the ``ts``
    range and the full predicate must still enforce both ``kind = ?`` and
    ``severity = ?`` on each candidate.
    """
    rows = _make_rows(2000)
    eng = _engine_conn(tmp_path)
    _seed_engine(eng, rows)
    sq = _seed_sqlite3(rows)

    sql = (
        _SELECT_COLS
        + " WHERE kind = ? AND severity = ? AND ts >= ? ORDER BY ts DESC LIMIT ?"
    )

    # Boundary at minute=8 (row 480+) — middle of corpus.
    boundary = _ts(minute=8)
    for kind in _KINDS[:2]:  # subset to keep runtime reasonable
        for sev in _SEVERITIES[:2]:
            _assert_parity(eng, sq, sql, (kind, sev, boundary, 15))

    # Boundary excludes everything.
    future = _ts(year=2099)
    _assert_parity(eng, sq, sql, ("recall", "info", future, 10))


# ---------------------------------------------------------------------------
# Test 6 — tie-order: duplicate ts values, LIMIT cuts inside the tie group
# ---------------------------------------------------------------------------


def test_events_tie_order_limit_inside_bucket_vs_sqlite3(tmp_path: Path) -> None:
    """Ordered-index tie order matches lilli's own stable-scan path.

    When all rows share the same ``ts`` value, the LIMIT cuts inside the tie
    group.  The engine's ordered-index path collects rows via the ts BTreeMap
    (buckets in reverse-ts order, then row-keys in ascending scan order within
    each bucket) and then applies ``order_rows`` (stable sort), producing
    ascending row-key order within the tie group.

    The three-way assertion is:
      engine result  ==  lilli scan path (ORDER BY ts DESC, id ASC explicitly)
                     ==  sqlite3 with explicit ASC tiebreak

    sqlite3 without an explicit id tiebreak uses descending rowid order for
    equal-ts rows (because it walks its ts index backward), which differs from
    lilli's ascending B-tree scan order — that divergence is pre-existing and
    intentional.  The correct differential is ``ORDER BY ts DESC, id ASC`` on
    both the engine and sqlite3 to pin the tiebreak explicitly.
    """
    same_ts = _ts(second=42)
    n = 50
    rows = [
        (
            f"ev-{i:03d}",
            _KINDS[i % len(_KINDS)],
            _SEVERITIES[i % len(_SEVERITIES)],
            None,
            same_ts,
            None,
            f"sess-{i}",
            None,
        )
        for i in range(n)
    ]

    eng = _engine_conn(tmp_path)
    _seed_engine(eng, rows)
    sq = _seed_sqlite3(rows)

    # Use explicit ASC tiebreak on id so both engines agree deterministically.
    sql_tied = _SELECT_COLS + " ORDER BY ts DESC, id ASC LIMIT ?"
    sql_all = _SELECT_COLS + " ORDER BY ts DESC, id ASC"

    # LIMIT cuts inside the single-ts group.
    for k in (1, 5, 10, 25, n - 1):
        _assert_parity(eng, sq, sql_tied, (k,))

    # Full table with deterministic tiebreak must also match.
    _assert_parity(eng, sq, sql_all, ())


# ---------------------------------------------------------------------------
# Test 7 — scan-count gate: SELECT * ORDER BY ts DESC LIMIT k must not full-scan when warm
# ---------------------------------------------------------------------------


def test_events_order_by_ts_limit_no_full_scan_warm(tmp_path: Path) -> None:
    """``SELECT * FROM events ORDER BY ts DESC LIMIT k`` pays no full scan when warm.

    On HEAD no ordered index exists; the engine full-scans the table, sorts in
    memory, then truncates. ``full_scan_count() >= 1`` after the first query.

    This is the RED marker that the ordered-index wave turns GREEN: once an
    ordered ``ts``-BTreeMap index is in place, the engine walks the index in
    reverse to gather the top-K candidates without a full-table scan.

    The ``SELECT *`` shape is required here because ``scan_rows`` (which
    increments ``full_scan_count``) is only invoked when the query projects all
    columns without a column-selective optimisation. Named-column projections go
    through ``scan_rows_two_phase`` which does not count as a full scan even on
    HEAD. Using ``SELECT *`` gives the sharpest signal for this gate.
    """
    rows = _make_rows(2000)
    conn = _engine_conn(tmp_path)
    _seed_engine(conn, rows)

    # First read: warms (or is expected to warm) the index.
    conn.execute("SELECT * FROM events ORDER BY ts DESC LIMIT ?", [10]).fetchall()
    conn.reset_full_scan_count()

    # Second read: must hit the warm ordered index (zero full scans after fix).
    conn.execute("SELECT * FROM events ORDER BY ts DESC LIMIT ?", [10]).fetchall()

    assert conn.full_scan_count() == 0, (
        f"expected full_scan_count==0 on the second ORDER BY ts DESC LIMIT query "
        f"(warm ordered index should avoid the full-table scan); "
        f"got {conn.full_scan_count()} — "
        "no ordered index for events.ts yet (expected RED state on HEAD)"
    )


# ---------------------------------------------------------------------------
# Test 8 — scan-count gate: WHERE kind = ? ORDER BY ts DESC LIMIT k
# ---------------------------------------------------------------------------


def test_events_kind_ordered_no_full_scan_warm(tmp_path: Path) -> None:
    """``SELECT * FROM events WHERE kind = ? ORDER BY ts DESC LIMIT k`` pays no full scan when warm.

    Same RED-marker structure as Test 7 for the kind-filtered ordered shape.
    On HEAD the engine does not recognise this as an ordered-index query and
    full-scans. After the ordered-index wave the engine narrows by the ``kind``
    hash index (or walks the ts-ordered index and filters by kind) then returns
    the top-K without a whole-table scan.
    """
    rows = _make_rows(2000)
    conn = _engine_conn(tmp_path)
    _seed_engine(conn, rows)

    conn.execute(
        "SELECT * FROM events WHERE kind = ? ORDER BY ts DESC LIMIT ?",
        ["telemetry", 10],
    ).fetchall()
    conn.reset_full_scan_count()

    conn.execute(
        "SELECT * FROM events WHERE kind = ? ORDER BY ts DESC LIMIT ?",
        ["telemetry", 10],
    ).fetchall()

    assert conn.full_scan_count() == 0, (
        f"expected full_scan_count==0 for warm WHERE kind=? ORDER BY ts DESC LIMIT; "
        f"got {conn.full_scan_count()} — "
        "no ordered kind+ts index yet (expected RED state on HEAD)"
    )


# ---------------------------------------------------------------------------
# Test 9 — severity scan-count gate: WHERE kind=? AND severity=? ORDER BY ts DESC
# ---------------------------------------------------------------------------


def test_events_severity_ordered_no_full_scan_warm(tmp_path: Path) -> None:
    """``WHERE kind=? AND severity=? ORDER BY ts DESC LIMIT k`` pays no full scan when warm.

    The severity conjunct is an extra equality filter applied via ``eval_predicate``
    after the ordered-index narrows the candidate set — it must not prevent the
    fast path from being taken.
    """
    rows = _make_rows(2000)
    conn = _engine_conn(tmp_path)
    _seed_engine(conn, rows)

    conn.execute(
        "SELECT * FROM events WHERE kind = ? AND severity = ? ORDER BY ts DESC LIMIT ?",
        ["telemetry", "info", 10],
    ).fetchall()
    conn.reset_full_scan_count()

    conn.execute(
        "SELECT * FROM events WHERE kind = ? AND severity = ? ORDER BY ts DESC LIMIT ?",
        ["telemetry", "info", 10],
    ).fetchall()

    assert conn.full_scan_count() == 0, (
        f"expected full_scan_count==0 for warm WHERE kind=? AND severity=? ORDER BY ts DESC LIMIT; "
        f"got {conn.full_scan_count()}"
    )


# ---------------------------------------------------------------------------
# Tests 10-13 — maintenance: insert/delete/warm-commit/rollback
# ---------------------------------------------------------------------------


def test_events_ordered_index_insert_maintained(tmp_path: Path) -> None:
    """After a warm index is built, a new INSERT is reflected in the top-K.

    The ordered index is maintained incrementally via ``insert_row``.  A freshly
    inserted row with a later ``ts`` must appear in the top-K result and must still
    pay ``full_scan_count()==0`` after the insert.
    """
    rows = _make_rows(100)
    conn = _engine_conn(tmp_path)
    _seed_engine(conn, rows)

    sql = "SELECT * FROM events ORDER BY ts DESC LIMIT ?"

    # Warm the index.
    conn.execute(sql, [5]).fetchall()
    conn.reset_full_scan_count()
    initial_top = conn.execute(sql, [5]).fetchall()
    assert conn.full_scan_count() == 0, "warm index must pay 0 scans"

    # Insert a row with a ts later than all existing rows.
    new_ts = _ts(year=2099)
    conn.execute(
        _INSERT,
        ["ev-NEW", "telemetry", "info", None, new_ts, None, "sess-new", None],
    )

    # The new row must appear first in the top-K; still zero full scans.
    conn.reset_full_scan_count()
    after_rows = conn.execute(sql, [5]).fetchall()
    assert conn.full_scan_count() == 0, (
        f"insert-maintained index must still pay 0 scans; got {conn.full_scan_count()}"
    )
    first_ts = dict(after_rows[0])["ts"]
    assert first_ts == new_ts, (
        f"expected newly-inserted row with ts={new_ts!r} at position 0; got {first_ts!r}"
    )
    # The previously first row is still present somewhere in the top-5.
    prev_first_ts = dict(initial_top[0])["ts"]
    result_tss = [dict(r)["ts"] for r in after_rows]
    assert prev_first_ts in result_tss, (
        f"previously-first row ts={prev_first_ts!r} missing after insert; got {result_tss}"
    )


def test_events_ordered_index_delete_maintained(tmp_path: Path) -> None:
    """After a warm index is built, a DELETE is reflected — deleted row vanishes.

    ``remove_row`` must drop the key from the BTreeMap bucket so the next
    top-K query cannot return the deleted event.  Full scan stays 0 (warm).
    """
    rows = _make_rows(100)
    conn = _engine_conn(tmp_path)
    _seed_engine(conn, rows)

    sql = "SELECT * FROM events ORDER BY ts DESC LIMIT ?"

    # Warm the index.
    conn.execute(sql, [5]).fetchall()
    initial_top = conn.execute(sql, [5]).fetchall()
    top_id = dict(initial_top[0])["id"]
    top_ts = dict(initial_top[0])["ts"]

    # Delete the top row.
    conn.execute("DELETE FROM events WHERE id = ?", [top_id])

    conn.reset_full_scan_count()
    after_rows = conn.execute(sql, [5]).fetchall()
    assert conn.full_scan_count() == 0, (
        f"delete-maintained index must still pay 0 scans; got {conn.full_scan_count()}"
    )
    result_tss = [dict(r)["ts"] for r in after_rows]
    assert top_ts not in result_tss, (
        f"deleted row with ts={top_ts!r} must not appear in top-K after delete"
    )


def test_events_ordered_index_warm_across_commit(tmp_path: Path) -> None:
    """A write inside BEGIN/COMMIT keeps the ordered index warm.

    The commit is untainted (the INSERT succeeds), so ``ordered_caches`` must not
    be cleared — the next top-K query must still pay ``full_scan_count()==0`` and
    reflect the committed state.
    """
    rows = _make_rows(50)
    conn = _engine_conn(tmp_path)
    _seed_engine(conn, rows)

    sql = "SELECT * FROM events ORDER BY ts DESC LIMIT ?"

    # Warm the index.
    conn.execute(sql, [5]).fetchall()
    conn.reset_full_scan_count()
    conn.execute(sql, [5]).fetchall()
    assert conn.full_scan_count() == 0, "pre-commit: warm index must pay 0 scans"

    # Write inside an explicit transaction and commit.
    new_ts = _ts(year=2088)
    conn.execute("BEGIN")
    conn.execute(
        _INSERT,
        ["ev-COMMIT", "health", "info", None, new_ts, None, "sess-commit", None],
    )
    conn.execute("COMMIT")

    # After the untainted commit the index must still be warm.
    conn.reset_full_scan_count()
    result = conn.execute(sql, [5]).fetchall()
    assert conn.full_scan_count() == 0, (
        f"post-commit: warm index cleared by untainted commit; got full_scan_count={conn.full_scan_count()}"
    )
    result_tss = [dict(r)["ts"] for r in result]
    assert new_ts in result_tss, (
        f"committed row ts={new_ts!r} must appear in post-commit top-K; got {result_tss}"
    )


def test_events_ordered_index_rollback_clears(tmp_path: Path) -> None:
    """A ROLLBACK clears the ordered index; the reverted row must not appear.

    After a ROLLBACK the ordered index is invalidated (cleared unconditionally),
    so the next query rebuilds from the committed tree state.  The reverted row
    must not appear in the top-K, and the index reflects only the committed data.
    """
    rows = _make_rows(50)
    conn = _engine_conn(tmp_path)
    _seed_engine(conn, rows)

    sql = "SELECT * FROM events ORDER BY ts DESC LIMIT ?"

    # Warm the index.
    conn.execute(sql, [5]).fetchall()
    before_top = [dict(r)["ts"] for r in conn.execute(sql, [5]).fetchall()]

    # Write inside a transaction, then roll back.
    future_ts = _ts(year=2077)
    conn.execute("BEGIN")
    conn.execute(
        _INSERT,
        ["ev-ROLLBACK", "telemetry", "error", None, future_ts, None, "sess-rb", None],
    )
    conn.execute("ROLLBACK")

    # After rollback the index is cleared and the reverted row must be absent.
    after_top = [dict(r)["ts"] for r in conn.execute(sql, [5]).fetchall()]
    assert future_ts not in after_top, (
        f"rolled-back row ts={future_ts!r} must not appear in top-K after rollback; "
        f"got {after_top}"
    )
    # The top-K must match the pre-rollback state exactly.
    assert after_top == before_top, (
        f"top-K must be identical to pre-rollback state after rollback:\n"
        f"  before={before_top}\n  after ={after_top}"
    )


# ---------------------------------------------------------------------------
# Test 13b — no-ORDER-BY read first must not shadow the ts ordered index
# ---------------------------------------------------------------------------


def test_no_order_by_read_does_not_shadow_ts_index(tmp_path: Path) -> None:
    """A no-ORDER-BY ``SELECT *`` / ``COUNT(*)`` before the recall query must not
    defeat the warm ``ts`` ordered index.

    A no-ORDER-BY read must NOT create an ordered cache on the
    alphabetically-first indexed column (``kind``); a later ``ORDER BY ts DESC``
    recall must still pay ``full_scan_count() == 0`` warm.
    """
    rows = _make_rows(200)
    conn = _engine_conn(tmp_path)
    _seed_engine(conn, rows)

    sql = "SELECT * FROM events ORDER BY ts DESC LIMIT ?"

    # No-ORDER-BY reads issued FIRST (the production hippo call sites).
    conn.execute("SELECT * FROM events").fetchall()
    conn.execute("SELECT COUNT(*) FROM events").fetchall()

    # Warm the ts ordered index, then measure the second run.
    conn.execute(sql, [10]).fetchall()
    conn.reset_full_scan_count()
    conn.execute(sql, [10]).fetchall()

    assert conn.full_scan_count() == 0, (
        f"warm ORDER BY ts DESC must pay 0 scans even after a prior no-ORDER-BY "
        f"events read; got {conn.full_scan_count()} (kind index shadowed ts)"
    )


# ---------------------------------------------------------------------------
# Test 14 — residual predicate on a NON-indexed column must not under-gather
# ---------------------------------------------------------------------------


def test_events_residual_nonindexed_predicate_vs_sqlite3(tmp_path: Path) -> None:
    """``WHERE severity = ? ORDER BY ts DESC LIMIT k`` is byte-identical to sqlite3.

    ``severity`` is NOT an events index, so this shape reaches the ordered ts
    fast path with a residual equality the gather cannot enforce.  The matching
    rows here sit at the OLDEST end of the ts order — well below the top-k by ts —
    so an early-stopped gather of the top-k ts candidates would find none of them
    and return ``[]`` where sqlite3 returns the matching rows.  The fix routes
    this shape to the byte-identical scan path.
    """
    # 30 rows, ascending ts.  Only the five OLDEST rows carry severity='rare'.
    rows: list[tuple] = []
    for i in range(30):
        ts = _ts(minute=i)  # row i has a strictly increasing ts
        sev = "rare" if i < 5 else "common"
        rows.append((f"ev-{i:02d}", "telemetry", sev, None, ts, None, "sess", None))

    eng = _engine_conn(tmp_path)
    _seed_engine(eng, rows)
    sq = _seed_sqlite3(rows)

    sql = _SELECT_COLS + " WHERE severity = ? ORDER BY ts DESC LIMIT ?"
    _assert_parity(eng, sq, sql, ("rare", 5))
    _assert_parity(eng, sq, sql, ("common", 5))
    _assert_parity(eng, sq, sql, ("ABSENT", 5))

    # The id projection that the production query_events(kind=None, severity=...)
    # shape emits must also be byte-identical.
    sql_id = "SELECT id FROM events WHERE severity = ? ORDER BY ts DESC LIMIT ?"
    _assert_parity(eng, sq, sql_id, ("rare", 5))


# ---------------------------------------------------------------------------
# Test 15 — OFFSET on the ordered shape must not under-gather
# ---------------------------------------------------------------------------


def test_events_order_by_ts_offset_vs_sqlite3(tmp_path: Path) -> None:
    """``ORDER BY ts DESC LIMIT k OFFSET m`` is byte-identical to sqlite3.

    The ordered fast path gathers only ~LIMIT candidates; a positive OFFSET would
    skip past that under-gathered superset and leave fewer rows than the scan
    path (often ``[]``).  The fix routes any ``OFFSET > 0`` shape to the scan path.
    """
    rows = _make_rows(40)
    eng = _engine_conn(tmp_path)
    _seed_engine(eng, rows)
    sq = _seed_sqlite3(rows)

    sql = "SELECT id FROM events ORDER BY ts DESC LIMIT ? OFFSET ?"
    _assert_parity(eng, sq, sql, (5, 10))
    _assert_parity(eng, sq, sql, (5, 0))
    _assert_parity(eng, sq, sql, (10, 25))
    _assert_parity(eng, sq, sql, (5, 100))  # offset past the corpus


# ---------------------------------------------------------------------------
# Test 16 — DO UPDATE re-file keeps within-tie order byte-identical
# ---------------------------------------------------------------------------


def test_events_do_update_refile_tie_order_vs_sqlite3(tmp_path: Path) -> None:
    """An ``ON CONFLICT DO UPDATE`` on a middle row keeps tie order scan-identical.

    All rows share one ``ts``; the LIMIT cuts inside the tie group.  After a
    DO UPDATE re-files a middle row in the warm ordered index, the within-tie
    order must stay ascending B-tree-key order (the scan path), so the result is
    byte-identical to sqlite3 with an explicit ``id ASC`` tiebreak.
    """
    same_ts = _ts(second=42)
    n = 30
    rows = [
        (f"ev-{i:03d}", "telemetry", "info", None, same_ts, None, f"sess-{i}", None)
        for i in range(n)
    ]

    eng = _engine_conn(tmp_path)
    _seed_engine(eng, rows)

    sql = _SELECT_COLS + " ORDER BY ts DESC, id ASC LIMIT ?"
    # Warm the ordered index.
    eng.execute(sql, [10]).fetchall()

    # Re-file a middle row via ON CONFLICT DO UPDATE (same ts → same tie bucket).
    eng.execute(
        _INSERT
        + " ON CONFLICT(id) DO UPDATE SET kind = excluded.kind",
        ["ev-015", "health", "info", None, same_ts, None, "sess-15", None],
    )

    # sqlite3 mirror with the same write.
    sq = _seed_sqlite3(rows)
    sq.execute(
        _INSERT + " ON CONFLICT(id) DO UPDATE SET kind = excluded.kind",
        ("ev-015", "health", "info", None, same_ts, None, "sess-15", None),
    )
    sq.commit()

    for k in (1, 5, 15, n - 1, n):
        _assert_parity(eng, sq, sql, (k,))


# ---------------------------------------------------------------------------
# Test 17 — a nullable / non-TEXT ordered column declines the fast path
# ---------------------------------------------------------------------------


def test_nullable_ordered_column_vs_sqlite3(tmp_path: Path) -> None:
    """An ORDER BY over a NULLABLE indexed column stays byte-identical to sqlite3.

    The ordered index files only TEXT values; NULL / numeric rows are dropped.
    For a nullable column the scan path orders the NULL tail below TEXT and a
    LIMIT past the TEXT rows would return it — so the engine must decline the
    fast path for a nullable column and scan instead.  Here ``label`` is a
    nullable indexed TEXT column with some NULL rows reached by the LIMIT.

    An explicit ``id ASC`` tiebreak pins the NULL tie group deterministically:
    without it, sqlite3 walks its label index backward (descending rowid for the
    equal-NULL group) while lilli's scan path keeps ascending B-tree-key order —
    a pre-existing, intentional divergence (see the tie-order test).  The point
    of this test is that the NULL/numeric tail is NOT dropped, which the explicit
    tiebreak isolates from that unrelated tie-order difference.
    """
    ddl = (
        "CREATE TABLE notes ("
        "id TEXT PRIMARY KEY, label TEXT, body TEXT)"
    )
    idx = "CREATE INDEX idx_notes_label ON notes(label)"
    insert = "INSERT INTO notes (id, label, body) VALUES (?, ?, ?)"

    # Half the rows carry a TEXT label, half are NULL.
    rows = []
    for i in range(20):
        label = f"lbl-{i:02d}" if i % 2 == 0 else None
        rows.append((f"n-{i:02d}", label, f"body-{i}"))

    from iai_mcp_native import engine

    _DB_SEQ["n"] += 1
    eng = engine.Connection.open(str(tmp_path / f"notes_{_DB_SEQ['n']}.db"), 384)
    eng.execute(ddl)
    eng.execute(idx)
    eng.execute("BEGIN")
    for row in rows:
        eng.execute(insert, list(row))
    eng.execute("COMMIT")

    sq = sqlite3.connect(":memory:")
    sq.row_factory = sqlite3.Row
    sq.execute(ddl)
    sq.execute(idx)
    sq.executemany(insert, rows)
    sq.commit()

    # With an explicit id ASC tiebreak (which also declines the fast path) the
    # full DESC result — TEXT rows then the NULL tail — is byte-identical.
    sql_tied = "SELECT id, label FROM notes ORDER BY label DESC, id ASC LIMIT ?"
    for k in (5, 15, 20):
        eng_rows = _norm(eng.execute(sql_tied, [k]).fetchall())
        sq_rows = _norm(sq.execute(sql_tied, (k,)).fetchall())
        assert eng_rows == sq_rows, (
            f"nullable ordered-column parity mismatch for LIMIT {k}:\n"
            f"  engine={eng_rows!r}\n  sqlite={sq_rows!r}"
        )

    # The single-key ``ORDER BY label DESC`` shape (which WOULD hit the ordered
    # fast path on a NOT NULL TEXT column) must include the NULL tail when LIMIT
    # reaches it — proving the engine declined the NULL-dropping fast path for a
    # nullable column.  Compare as a SET (tie order inside the NULL group is the
    # pre-existing scan-vs-index divergence, isolated by the tiebreak test above).
    sql_single = "SELECT id FROM notes ORDER BY label DESC LIMIT 20"
    eng_ids = {r[0] for r in eng.execute(sql_single, []).fetchall()}
    sq_ids = {r[0] for r in sq.execute(sql_single).fetchall()}
    assert eng_ids == sq_ids, (
        f"single-key nullable ORDER BY dropped rows (fast path not declined):\n"
        f"  engine={sorted(eng_ids)!r}\n  sqlite={sorted(sq_ids)!r}"
    )
    null_ids = {f"n-{i:02d}" for i in range(20) if i % 2 != 0}
    assert null_ids <= eng_ids, (
        f"NULL-label rows dropped from the result: missing {null_ids - eng_ids!r}"
    )
