"""The live-rows recall filter must resolve by index, not a linear O(corpus)
sweep.

``tombstoned_at IS NULL`` was the predicate the three hottest recall queries
issued (``_DIRECT_RECENCY_SQL`` / ``_DIRECT_RECENCY_SQL_LIMITED`` /
``_DIRECT_TEMPORAL_SQL`` in ``iai_mcp.hippo._recall``). The lilli engine's
ordered-top-k fast path (``exec.rs`` ``strict_ordered_lower_bound``) rejects
``IS NULL`` as a residual predicate, so a query issued directly against that
timestamp column falls through to ``scan_rows_two_phase``, a
column-selective streaming sweep that visits EVERY physical row (live +
tombstoned), not just the live subset the ``LIMIT`` needs.

The fix is a denormalized indexed ``live`` boolean column, kept in permanent
agreement with ``tombstoned_at IS NULL`` on every write path. The rewritten
recency query filters on ``live = 1`` instead: a bare column-equals-literal
predicate the catalog's indexed candidate-row path resolves directly (a
union of O(1) index probes plus one B-tree point-get per candidate), so the
query visits only the live subset the ``LIMIT`` needs -- not the whole
physical table.

Oracle -- the direct cost observable, ``cells_visited_count()``:
  The engine counts every leaf cell a streaming scan touches
  (``scan_cells_with``). An index-served query (a point lookup, or an indexed
  live-flag equality) touches at most ``O(LIMIT)`` cells -- often near zero,
  since the indexed candidate-row path resolves each row via a direct B-tree
  point-get rather than a streaming scan. A linear column-selective sweep
  touches ``O(corpus)`` cells instead. This is the honest DIRECT measure of
  the leak, replacing the structurally-unreachable ``full_scan_count()``
  oracle (that counter instruments only the whole-table ``full_scan()`` call
  sites, NOT the streaming-scan or indexed-candidate-row paths the aliased
  ``SELECT ... COALESCE(...) AS ...`` recency shape actually takes, so it
  stays at 0 regardless of whether the predicate is index-served -- it can
  never distinguish an indexed read from a sweep for this query shape).

Both drivers are exercised: the lilli arm is the real regression (skipped
without the native wheel); the stdlib arm is a no-op correctness control
(sqlite3 has no cells-visited counter, so only the byte-identical assertion
applies there).
"""
from __future__ import annotations

import sqlite3
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def _native_available() -> bool:
    try:
        from iai_mcp_native import engine  # noqa: F401

        return True
    except ImportError:
        return False


_HAVE_CELLS_COUNTER = False
if _native_available():
    try:
        from iai_mcp_native import engine as _probe_engine

        _c = _probe_engine.Connection.open(
            str(Path(tempfile.mkdtemp()) / "probe.db"), 384
        )
        _HAVE_CELLS_COUNTER = hasattr(_c, "cells_visited_count")
    except Exception:  # noqa: BLE001 -- probe failure => treat counter as absent
        _HAVE_CELLS_COUNTER = False


_DDL_RECORDS = (
    "CREATE TABLE records ("
    "vec_label INTEGER PRIMARY KEY AUTOINCREMENT, "
    "id TEXT NOT NULL, "
    "tier TEXT, "
    "created_at TEXT, "
    "tombstoned_at TEXT, "
    "live INTEGER, "
    "embedding_pending INTEGER"
    ")"
)

_INSERT = (
    "INSERT INTO records (id, tier, created_at, tombstoned_at, live, embedding_pending)"
    " VALUES (?, ?, ?, ?, ?, ?)"
)

# The EXACT production recency shape: an aliased ``COALESCE(...) AS ...`` column
# beside plain columns, ``WHERE live = 1``, ordered by created_at DESC, capped
# by a LIMIT. This is the shape ``_DIRECT_RECENCY_SQL_LIMITED`` issues (a subset
# of columns -- the WHERE/ORDER BY/LIMIT/alias shape is what governs which
# engine path is taken). ``live`` is a denormalized column kept in permanent
# agreement with ``tombstoned_at IS NULL`` -- it is set on every write path
# and rides the catalog's indexed col=literal reducibility arm.
_LIVE_ROWS_SQL_LIMITED = (
    "SELECT id, tier, created_at,"
    " COALESCE(embedding_pending, 0) AS embedding_pending"
    " FROM records WHERE live = 1"
    " ORDER BY created_at DESC LIMIT ?"
)

# An unambiguous full-corpus reference over the identical logical predicate,
# expressed over the legacy ``tombstoned_at`` timestamp column rather than the
# indexed ``live`` flag the query under test now reads. The engine's minimal
# SQL parser has no subquery support, so the reference conjoins a sentinel
# inequality (``id != '<never matches>'``) with the live-rows predicate.
# ``!=`` has no equality/IN index path in the engine, so this conjunct forces
# a residual per-row evaluation regardless of which path the query-under-test
# takes -- a different SQL shape (and a different backing column) driving the
# identical logical predicate, so it cannot share a false-positive index hit
# with the query under test.
_FULL_SCAN_REFERENCE_SQL = (
    "SELECT id, tier, created_at,"
    " COALESCE(embedding_pending, 0) AS embedding_pending"
    " FROM records"
    " WHERE tombstoned_at IS NULL AND id != 'sentinel-never-matches-0000'"
    " ORDER BY created_at DESC LIMIT ?"
)

_LIMIT = 50


def _make_rows(n_live: int = 2400, n_tombstoned: int = 600) -> list[tuple]:
    """A deterministic mix of live + tombstoned rows with distinct created_at.

    ~20% tombstoned. created_at is strictly increasing per insertion index so
    ORDER BY created_at DESC has a unique, reproducible ordering (no ties). The
    corpus is large enough that an O(corpus) sweep (visiting every row) is
    unmistakably distinct from an O(LIMIT) index-served read. The seeded
    ``live`` column always agrees with ``tombstoned_at IS NULL`` -- the same
    invariant every production write path holds.
    """
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows: list[tuple] = []
    total = n_live + n_tombstoned
    for i in range(total):
        rid = str(uuid.uuid4())
        created_at = (base + timedelta(seconds=i)).strftime("%Y-%m-%d %H:%M:%S")
        # Interleave live/tombstoned so the live set is not contiguous (every
        # 5th row tombstoned => ~20%).
        tombstoned_at = "2020-01-01 00:00:00" if (i % 5 == 0) else None
        live = 0 if tombstoned_at is not None else 1
        rows.append((rid, "episodic", created_at, tombstoned_at, live, 0))
    return rows


def _engine_conn(tmp_path: Path):
    from iai_mcp_native import engine

    conn = engine.Connection.open(str(tmp_path / "live_rows.db"), 384)
    assert isinstance(conn, engine.Connection), (
        "connection under test is not the Rust engine.Connection -- "
        "refusing to run a false-green proof against a different engine"
    )
    return conn


def _seed_engine(conn, rows: list[tuple]) -> None:
    conn.execute(_DDL_RECORDS)
    # A created_at index -- present in production; the ordered-top-k fast path
    # can only ever engage when this exists, so seeding it makes the RED gate
    # measure the IS NULL rejection specifically, not a missing index.
    conn.execute("CREATE INDEX idx_records_created_at ON records(created_at)")
    # The indexed live-flag column -- present in production. This is the
    # index the rewritten recency query rides via the catalog's col=literal
    # reducibility arm.
    conn.execute("CREATE INDEX idx_records_live ON records(live)")
    conn.execute("BEGIN")
    for row in rows:
        conn.execute(_INSERT, list(row))
    conn.execute("COMMIT")


def _seed_sqlite3(rows: list[tuple]) -> sqlite3.Connection:
    sq = sqlite3.connect(":memory:")
    sq.row_factory = sqlite3.Row
    sq.execute(_DDL_RECORDS)
    sq.execute("CREATE INDEX idx_records_created_at ON records(created_at)")
    sq.execute("CREATE INDEX idx_records_live ON records(live)")
    sq.executemany(_INSERT, rows)
    sq.commit()
    return sq


def _norm(rows) -> list[tuple]:
    return [tuple(r) for r in rows]


# ---------------------------------------------------------------------------
# Test 1 (lilli) -- byte-identical differential (PASSES on HEAD: correctness
# is already right; only the cost is the problem).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _native_available(),
    reason="iai_mcp_native.engine submodule not installed (build the native wheel)",
)
def test_live_rows_filter_byte_identical_to_full_scan_reference(tmp_path: Path) -> None:
    rows = _make_rows()
    eng = _engine_conn(tmp_path)
    _seed_engine(eng, rows)

    under_test = _norm(eng.execute(_LIVE_ROWS_SQL_LIMITED, [_LIMIT]).fetchall())
    reference = _norm(eng.execute(_FULL_SCAN_REFERENCE_SQL, [_LIMIT]).fetchall())

    assert under_test == reference, (
        "the live-rows filter (live = 1, ORDER BY created_at DESC) must return "
        "byte-identical rows to the full-corpus reference over the same "
        "logical predicate expressed against tombstoned_at -- the correctness "
        "bar, independent of which path (sweep or index) resolves the query"
    )
    assert len(under_test) == _LIMIT, (
        "expected exactly LIMIT live rows returned from the seeded corpus"
    )


@pytest.mark.skipif(
    not _native_available(),
    reason="iai_mcp_native.engine submodule not installed (build the native wheel)",
)
def test_live_rows_filter_byte_identical_to_sqlite3(tmp_path: Path) -> None:
    """Cross-driver correctness: same predicate, engine vs stdlib sqlite3."""
    rows = _make_rows()
    eng = _engine_conn(tmp_path)
    _seed_engine(eng, rows)
    sq = _seed_sqlite3(rows)

    eng_rows = _norm(eng.execute(_LIVE_ROWS_SQL_LIMITED, [_LIMIT]).fetchall())
    sq_rows = _norm(sq.execute(_LIVE_ROWS_SQL_LIMITED, (_LIMIT,)).fetchall())

    assert eng_rows == sq_rows, (
        f"engine and sqlite3 disagree on the live-rows filter result set:\n"
        f"  engine only : {[r for r in eng_rows if r not in sq_rows]}\n"
        f"  sqlite3 only: {[r for r in sq_rows if r not in eng_rows]}"
    )


# ---------------------------------------------------------------------------
# Test 2 (lilli) -- point-lookup control: the counter DOES drop to ~0 when a
# query is genuinely index-served. Proves the oracle discriminates (it is not
# stuck-high for every query), so the RED marker below is meaningful.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_native_available() and _HAVE_CELLS_COUNTER),
    reason="native engine with cells_visited_count() required",
)
def test_cells_visited_counter_discriminates_index_served(tmp_path: Path) -> None:
    """An id point-lookup (index-served via tree.get) visits ~0 cells.

    This is the control proving the cells-visited oracle actually
    discriminates index-served from linear-sweep resolution -- a stuck-high
    counter would make the RED marker meaningless.
    """
    rows = _make_rows()
    conn = _engine_conn(tmp_path)
    conn.execute(_DDL_RECORDS)
    conn.execute("CREATE INDEX idx_records_id ON records(id)")
    conn.execute("BEGIN")
    for row in rows:
        conn.execute(_INSERT, list(row))
    conn.execute("COMMIT")

    target_id = rows[len(rows) // 2][0]
    sql = "SELECT id, tier FROM records WHERE id = ? LIMIT 1"
    conn.execute(sql, [target_id]).fetchall()  # warm the id index
    conn.reset_cells_visited_count()
    result = conn.execute(sql, [target_id]).fetchall()

    assert len(result) == 1
    visited = conn.cells_visited_count()
    assert visited < len(rows) // 10, (
        f"an id point-lookup must be index-served (near-zero cells visited), "
        f"but visited {visited} of {len(rows)} cells -- the cells-visited "
        f"oracle is not discriminating index-served from a linear sweep"
    )


# ---------------------------------------------------------------------------
# Test 3 (lilli) -- the load-bearing cost bar: the live-rows recency query
# must visit O(LIMIT) cells, never O(corpus).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_native_available() and _HAVE_CELLS_COUNTER),
    reason="native engine with cells_visited_count() required",
)
def test_live_rows_filter_visits_only_limit_cells_warm(tmp_path: Path) -> None:
    """The warm live-rows recency query must visit O(LIMIT) cells, not O(corpus).

    The indexed ``live = 1`` equality resolves through the catalog's indexed
    candidate-row path: a union of O(1) index probes over the live rows plus
    one B-tree point-get per candidate, re-applying the full predicate to
    each decoded row. Point-gets are not counted by the streaming-scan cells
    counter at all, so a genuinely index-served read stays far under the
    generous ceiling below regardless of corpus size.
    """
    rows = _make_rows()
    conn = _engine_conn(tmp_path)
    _seed_engine(conn, rows)

    corpus = len(rows)

    # First read: warms the catalog's live-column index.
    conn.execute(_LIVE_ROWS_SQL_LIMITED, [_LIMIT]).fetchall()
    conn.reset_cells_visited_count()

    # Second (warm) read: must touch only ~LIMIT cells once index-served.
    conn.execute(_LIVE_ROWS_SQL_LIMITED, [_LIMIT]).fetchall()
    visited = conn.cells_visited_count()

    # Generous ceiling: 4x LIMIT absorbs any tie-group over-gather; still far
    # below the O(corpus) sweep (4*LIMIT << corpus at this corpus size).
    ceiling = _LIMIT * 4
    assert visited <= ceiling, (
        f"the warm live-rows recency query visited {visited} cells out of a "
        f"{corpus}-row corpus (LIMIT={_LIMIT}); an index-served query must "
        f"visit <= {ceiling} (O(LIMIT)). Visiting ~corpus means the live-flag "
        f"equality is still forcing a linear column-selective sweep instead "
        f"of resolving through the indexed candidate-row path."
    )


@pytest.mark.skipif(
    not (_native_available() and _HAVE_CELLS_COUNTER),
    reason="native engine with cells_visited_count() required",
)
def test_live_rows_filter_no_longer_visits_full_corpus(tmp_path: Path) -> None:
    """The live-rows query no longer walks the whole physical table.

    This is the inverse framing of the cost bar above -- a plain, always-run
    pin that positively records the query staying UNDER the full corpus size.
    It fails if a future change regresses the indexed path back to a linear
    sweep (e.g. the live column stops riding the catalog index).
    """
    rows = _make_rows()
    conn = _engine_conn(tmp_path)
    _seed_engine(conn, rows)

    corpus = len(rows)
    conn.execute(_LIVE_ROWS_SQL_LIMITED, [_LIMIT]).fetchall()
    conn.reset_cells_visited_count()
    conn.execute(_LIVE_ROWS_SQL_LIMITED, [_LIMIT]).fetchall()
    visited = conn.cells_visited_count()

    # The indexed live-rows query must visit strictly fewer cells than a full
    # sweep of the corpus -- otherwise the live column is not riding the
    # catalog index and the query has regressed to a linear scan.
    assert visited < corpus, (
        f"expected the live-rows recency query to resolve through the "
        f"indexed candidate-row path (fewer than the {corpus}-row corpus "
        f"visited); it visited {visited}. This means the live-flag equality "
        f"regressed to a linear column-selective sweep."
    )


# ---------------------------------------------------------------------------
# Test 4 (stdlib no-op arm) -- correctness control, no cells-visited concept.
# ---------------------------------------------------------------------------


def test_live_rows_filter_stdlib_noop_arm() -> None:
    """stdlib sqlite3 has no cells-visited counter; this pins correctness only.

    Runs unconditionally (no native-wheel dependency) so the module always has
    at least one collected, always-green arm proving the query shape itself
    (predicate + ORDER BY + LIMIT + COALESCE alias) is well-formed and returns
    the expected live-only rows regardless of engine availability.
    """
    rows = _make_rows()
    sq = _seed_sqlite3(rows)

    result = sq.execute(_LIVE_ROWS_SQL_LIMITED, (_LIMIT,)).fetchall()

    n_live = sum(1 for r in rows if r[3] is None)
    assert len(result) == min(n_live, _LIMIT)
    # Every returned row must genuinely be live (tombstoned_at IS NULL, which
    # is exactly the rows the seeded live column marks as 1).
    live_ids = {r[0] for r in rows if r[3] is None}
    for row in result:
        assert row["id"] in live_ids, (
            f"row {row['id']!r} returned by the live-rows filter is not in "
            f"the seeded live set"
        )


# ---------------------------------------------------------------------------
# Test 5 (lilli) -- the fresh-migrate cutover shape: a store copied verbatim
# from a source that predates the live column must land it populated on the
# FIRST open, before the first recall ever runs.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _native_available(),
    reason="iai_mcp_native.engine submodule not installed (build the native wheel)",
)
def test_fresh_migrate_lands_live_column_populated_before_first_recall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh-migrated store must ride the live-flag index on its first open.

    Simulates the cutover shape: a source store written before the live
    column existed (every row's live cell is NULL, as a verbatim copy of such
    a source would land), migrated verbatim into a fresh lilli store, then
    opened as a full store. The boot-migration sequence backfills live from
    each row's own tombstoned_at value unconditionally on the first open (no
    stamp short-circuits before the very first run), so the live column is
    fully populated and the rewritten recency query already resolves through
    the indexed candidate-row path -- never a NULL-column full scan -- on the
    very first recall after migration.
    """
    from iai_mcp.migrate import migrate_sqlite_to_lilli

    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    src_root.mkdir()

    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    monkeypatch.setenv("IAI_MCP_STORE", str(src_root))
    from iai_mcp.store import MemoryStore, flush_record_buffer
    from iai_mcp.types import MemoryRecord
    import numpy as np

    src_store = MemoryStore(src_root)
    rng = np.random.RandomState(7)
    base = datetime(2026, 2, 1, tzinfo=timezone.utc)
    n_records = 40
    for i in range(n_records):
        rec = MemoryRecord(
            id=uuid.uuid4(),
            tier="episodic",
            literal_surface=f"pre-live-column capture number {i}",
            aaak_index="",
            embedding=rng.randn(384).tolist(),
            community_id=None,
            centrality=0.0,
            detail_level=1,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[{"session_id": "sess", "role": "user"}],
            created_at=base + timedelta(seconds=i),
            updated_at=base + timedelta(seconds=i),
            tags=["role:user"],
            language="en",
        )
        src_store.insert(rec)
    flush_record_buffer(src_store)

    # Simulate a source written before the live column existed: every row's
    # live cell lands NULL, as a verbatim copy of such a pre-column source
    # would produce (the column would not exist at all on the true legacy
    # source; NULLing it here on a schema that already carries the column is
    # the equivalent post-copy state, since a copy of a source without the
    # column leaves the destination's column at its NULL default). The
    # backfill-completion stamp is removed too -- a genuine legacy source
    # predating the live column would never have written this key (the
    # migrator that writes it did not exist yet), so leaving the stamp in
    # place here would produce a false pass: the boot migration would skip
    # on a stamp that a real legacy source could never carry.
    with src_store.db._conn_lock:
        src_store.db._conn.execute("UPDATE records SET live = NULL")
        src_store.db._conn.execute(
            "DELETE FROM _hippo_meta WHERE key = 'live_flag_backfill_done'"
        )
        src_store.db._conn.commit()
    src_db_path = str(src_root / "hippo" / "brain.sqlite3")
    src_store.db.close()

    report = migrate_sqlite_to_lilli(src_db_path, dst_root, batch=20)
    assert report.rows_copied.get("records") == n_records

    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    monkeypatch.setenv("IAI_MCP_STORE", str(dst_root))
    dst_store = MemoryStore(dst_root)
    try:
        with dst_store.db._conn_lock:
            unset_rows = dst_store.db._conn.execute(
                "SELECT COUNT(*) FROM records WHERE live IS NULL"
            ).fetchone()
            live_rows = dst_store.db._conn.execute(
                "SELECT COUNT(*) FROM records WHERE live = 1"
            ).fetchone()
        assert int(unset_rows[0]) == 0, (
            f"expected zero rows with live IS NULL after the first open of a "
            f"fresh-migrated store; found {unset_rows[0]} -- the boot backfill "
            f"did not populate the live column before the first recall"
        )
        assert int(live_rows[0]) == n_records, (
            f"expected all {n_records} migrated rows to be live (none "
            f"tombstoned); found {live_rows[0]} rows with live = 1"
        )

        conn = dst_store.db._conn
        if hasattr(conn, "cells_visited_count"):
            conn.execute(_LIVE_ROWS_SQL_LIMITED, [_LIMIT]).fetchall()
            conn.reset_cells_visited_count()
            conn.execute(_LIVE_ROWS_SQL_LIMITED, [_LIMIT]).fetchall()
            visited = conn.cells_visited_count()
            assert visited < n_records, (
                f"the first recall on a fresh-migrated store visited "
                f"{visited} cells out of a {n_records}-row corpus -- the "
                f"live column was not populated in time to ride the index, "
                f"defeating the fix on exactly the cutover corpus shape"
            )
    finally:
        dst_store.db.close()


# ---------------------------------------------------------------------------
# Test 6 -- an already-drifted store (a row tombstoned without deriving live
# in the same write, and its symmetric reverse) self-corrects on the next
# reconciliation pass, and a second pass is a stamped no-op.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _native_available(),
    reason="iai_mcp_native.engine submodule not installed (build the native wheel)",
)
def test_drifted_live_flag_reconciles_and_second_pass_is_noop(
    tmp_path: Path,
) -> None:
    """A store carrying pre-existing live/tombstoned drift self-corrects.

    Simulates the shape an affected maintenance verb leaves behind: a row
    with ``tombstoned_at`` set but ``live`` still 1 (the verb wrote only the
    timestamp), plus the symmetric reverse case (``tombstoned_at`` unset but
    ``live`` still 0) to cover both directional UPDATEs. The reconciliation
    pass must correct both rows and stamp completion; a second pass must be
    a no-op that reports zero further updates.
    """
    from iai_mcp.migrate._live_flag_backfill import (
        _RECONCILE_MARKER_KEY,
        reconcile_live_flag_drift,
    )
    from iai_mcp.store import MemoryStore, flush_record_buffer
    from iai_mcp.types import MemoryRecord
    import numpy as np

    store = MemoryStore(tmp_path)
    try:
        rng = np.random.RandomState(11)
        base = datetime(2026, 3, 1, tzinfo=timezone.utc)

        drifted_id = uuid.uuid4()
        reverse_id = uuid.uuid4()
        for rid, text in (
            (drifted_id, "quarantined blob still marked live"),
            (reverse_id, "revived row still marked dead"),
        ):
            rec = MemoryRecord(
                id=rid,
                tier="episodic",
                literal_surface=text,
                aaak_index="",
                embedding=rng.randn(384).tolist(),
                community_id=None,
                centrality=0.0,
                detail_level=1,
                pinned=False,
                stability=0.0,
                difficulty=0.0,
                last_reviewed=None,
                never_decay=False,
                never_merge=False,
                provenance=[{"session_id": "sess", "role": "user"}],
                created_at=base,
                updated_at=base,
                tags=["role:user"],
                language="en",
            )
            store.insert(rec)
        flush_record_buffer(store)

        # Simulate the drift the affected verbs leave: tombstoned_at set
        # without deriving live in the same write (forward drift), and the
        # symmetric reverse drift (tombstoned_at cleared without deriving
        # live) to exercise the other directional UPDATE.
        with store.db._conn_lock:
            store.db._conn.execute(
                "UPDATE records SET tombstoned_at = ?, live = 1 WHERE id = ?",
                (base.isoformat(), str(drifted_id)),
            )
            store.db._conn.execute(
                "UPDATE records SET tombstoned_at = NULL, live = 0 WHERE id = ?",
                (str(reverse_id),),
            )
            # The store's own construction already ran reconciliation once
            # (with zero rows drifted at that point) and set the stamp. A
            # store that predates this code would never have written this
            # key -- delete it to model that shape, otherwise the drift
            # introduced above would never be examined.
            store.db._conn.execute(
                "DELETE FROM _hippo_meta WHERE key = ?",
                (_RECONCILE_MARKER_KEY,),
            )
            store.db._conn.commit()

            forward_drift_before = store.db._conn.execute(
                "SELECT COUNT(*) FROM records "
                "WHERE tombstoned_at IS NOT NULL AND live = 1"
            ).fetchone()
        assert int(forward_drift_before[0]) > 0, (
            "expected the simulated forward drift to be present before "
            "reconciliation"
        )

        report = reconcile_live_flag_drift(store)
        assert report["updated"] >= 2
        assert report["dry_run"] is False

        with store.db._conn_lock:
            row_drifted = store.db._conn.execute(
                "SELECT live FROM records WHERE id = ?", (str(drifted_id),)
            ).fetchone()
            row_reverse = store.db._conn.execute(
                "SELECT live FROM records WHERE id = ?", (str(reverse_id),)
            ).fetchone()
            forward_drift_after = store.db._conn.execute(
                "SELECT COUNT(*) FROM records "
                "WHERE tombstoned_at IS NOT NULL AND live = 1"
            ).fetchone()
            reverse_drift_after = store.db._conn.execute(
                "SELECT COUNT(*) FROM records "
                "WHERE tombstoned_at IS NULL AND live = 0"
            ).fetchone()
            stamp = store.db._conn.execute(
                "SELECT value FROM _hippo_meta WHERE key = ?",
                (_RECONCILE_MARKER_KEY,),
            ).fetchone()

        live_drifted = (
            row_drifted["live"] if hasattr(row_drifted, "keys") else row_drifted[0]
        )
        live_reverse = (
            row_reverse["live"] if hasattr(row_reverse, "keys") else row_reverse[0]
        )
        assert int(live_drifted) == 0, (
            "expected the drifted tombstoned row to land live = 0 after "
            "reconciliation"
        )
        assert int(live_reverse) == 1, (
            "expected the reverse-drift row to land live = 1 after "
            "reconciliation"
        )
        assert int(forward_drift_after[0]) == 0, (
            "expected zero rows with tombstoned_at set and live = 1 after "
            "reconciliation"
        )
        assert int(reverse_drift_after[0]) == 0, (
            "expected zero rows with tombstoned_at unset and live = 0 after "
            "reconciliation"
        )
        assert stamp is not None, (
            "expected the reconciliation completion stamp to be written"
        )

        # Idempotency: a second pass is a stamped no-op -- no further rows
        # change and the report carries zero updates.
        second_report = reconcile_live_flag_drift(store)
        assert second_report["updated"] == 0
        assert second_report.get("note") == "already reconciled"
    finally:
        store.db.close()
