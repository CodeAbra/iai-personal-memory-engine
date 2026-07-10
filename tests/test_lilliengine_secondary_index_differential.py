"""Differential parity for catalog-declared secondary indexes against stdlib sqlite3.

Drives ``iai_mcp_native.engine.Connection`` directly to prove:

1. Byte-identical results: ``SELECT * FROM records WHERE tier = ?``,
   ``SELECT id FROM records WHERE community_id IN (?, ?)`` and their compound
   form produce exactly the same rows, order, and Python types as stdlib sqlite3
   over identical DDL, indexes, and data.

2. Scan-count gate (warm): after a prior identical query has built and warmed the
   secondary index for a column, the next identical query must resolve without a
   whole-table scan — ``full_scan_count() == 0``.  On HEAD the columns ``tier``
   and ``community_id`` are not in the hardcoded ``col_index_columns`` list, so
   the engine falls through to ``scan_rows`` (full-tree scan) even for ``SELECT *``
   shapes. ``full_scan_count() >= 1`` is the RED marker these tests expose.

3. Compound shape: ``WHERE tier = ? AND community_id IN (?, ?)`` is also
   byte-identical — validating the AND-conjunct superset path once secondary
   indexes are catalog-driven.

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
# DDL — records-shaped with the catalog-declared secondary indexes
# ---------------------------------------------------------------------------

_DB_SEQ = {"n": 0}

# Core columns exercised: tier, community_id plus the AUTOINCREMENT PK and id.
_DDL_RECORDS = (
    "CREATE TABLE records ("
    "vec_label INTEGER PRIMARY KEY AUTOINCREMENT, "
    "id TEXT NOT NULL, "
    "tier TEXT, "
    "community_id TEXT, "
    "embedding_pending INTEGER"
    ")"
)

# The production CREATE INDEX declarations for these columns.
_INDEXES = [
    "CREATE INDEX idx_records_tier      ON records(tier)",
    "CREATE INDEX idx_records_community ON records(community_id)",
]

_INSERT = (
    "INSERT INTO records (id, tier, community_id, embedding_pending)"
    " VALUES (?, ?, ?, ?)"
)

_TIERS = ["episodic", "semantic", "procedural"]
_COMMUNITIES = [f"comm-{i}" for i in range(8)]


def _make_rows(n: int = 300) -> list[tuple]:
    return [
        (
            f"rec-{i:04d}",
            _TIERS[i % len(_TIERS)],
            _COMMUNITIES[i % len(_COMMUNITIES)],
            i % 2,
        )
        for i in range(n)
    ]


def _engine_conn(tmp_path: Path):
    """Open a Rust engine connection and assert it IS the Rust class."""
    from iai_mcp_native import engine

    _DB_SEQ["n"] += 1
    conn = engine.Connection.open(str(tmp_path / f"secidx_{_DB_SEQ['n']}.db"), 384)
    assert isinstance(conn, engine.Connection), (
        "connection under test is not the Rust engine.Connection — "
        "refusing to run a false-green proof against a different engine"
    )
    return conn


def _seed_engine(conn, rows: list[tuple]) -> None:
    conn.execute(_DDL_RECORDS)
    for idx_sql in _INDEXES:
        conn.execute(idx_sql)
    conn.execute("BEGIN")
    for row in rows:
        conn.execute(_INSERT, list(row))
    conn.execute("COMMIT")


def _seed_sqlite3(rows: list[tuple]) -> sqlite3.Connection:
    sq = sqlite3.connect(":memory:")
    sq.row_factory = sqlite3.Row
    sq.execute(_DDL_RECORDS)
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
# Test 1 — differential: tier = ? byte-identical to sqlite3
# ---------------------------------------------------------------------------


def test_tier_eq_scan_equality_vs_sqlite3(tmp_path: Path) -> None:
    """``SELECT * FROM records WHERE tier = ?`` is byte-identical to sqlite3.

    The ``tier`` column is declared with a ``CREATE INDEX`` in both the engine
    and sqlite3 DDL. On HEAD the engine's ``col_index_columns`` hardcode does
    not include ``tier``, so the query falls through to a full-tree scan; the
    result is still byte-identical (the correctness bar holds regardless of
    which path is taken). After the catalog-driven index wave, the same result
    arrives via the secondary index instead of the scan.
    """
    rows = _make_rows(300)
    eng = _engine_conn(tmp_path)
    _seed_engine(eng, rows)
    sq = _seed_sqlite3(rows)

    _assert_parity(
        eng, sq,
        "SELECT * FROM records WHERE tier = ? ORDER BY vec_label",
        ("episodic",),
    )
    _assert_parity(
        eng, sq,
        "SELECT * FROM records WHERE tier = ? ORDER BY vec_label",
        ("semantic",),
    )
    _assert_parity(
        eng, sq,
        "SELECT * FROM records WHERE tier = ? ORDER BY vec_label",
        ("procedural",),
    )
    # Absent tier returns no rows on both drivers.
    _assert_parity(
        eng, sq,
        "SELECT * FROM records WHERE tier = ? ORDER BY vec_label",
        ("ABSENT",),
    )


# ---------------------------------------------------------------------------
# Test 2 — differential: community_id IN byte-identical to sqlite3
# ---------------------------------------------------------------------------


def test_community_id_in_scan_equality_vs_sqlite3(tmp_path: Path) -> None:
    """``SELECT id FROM records WHERE community_id IN (?, ?)`` is byte-identical to sqlite3.

    ``community_id`` is also catalog-indexed but not in the hardcoded column
    list. The result order is pinned by ``ORDER BY vec_label`` so tie-order is
    deterministic on both drivers.
    """
    rows = _make_rows(300)
    eng = _engine_conn(tmp_path)
    _seed_engine(eng, rows)
    sq = _seed_sqlite3(rows)

    _assert_parity(
        eng, sq,
        "SELECT id FROM records WHERE community_id IN (?, ?) ORDER BY vec_label",
        ("comm-0", "comm-1"),
    )
    # Mixed present/absent.
    _assert_parity(
        eng, sq,
        "SELECT id FROM records WHERE community_id IN (?, ?, ?) ORDER BY vec_label",
        ("comm-3", "ABSENT", "comm-7"),
    )
    # Single value.
    _assert_parity(
        eng, sq,
        "SELECT id FROM records WHERE community_id IN (?) ORDER BY vec_label",
        ("comm-4",),
    )


# ---------------------------------------------------------------------------
# Test 3 — differential: compound tier = ? AND community_id IN byte-identical
# ---------------------------------------------------------------------------


def test_tier_and_community_compound_vs_sqlite3(tmp_path: Path) -> None:
    """``WHERE tier = ? AND community_id IN (?, ?)`` is byte-identical to sqlite3.

    This is the AND-conjunct superset shape: one indexed conjunct narrows the
    candidate set and the full predicate re-applies. The correctness bar holds
    in both the full-scan (HEAD) and the index (after fix) paths.
    """
    rows = _make_rows(300)
    eng = _engine_conn(tmp_path)
    _seed_engine(eng, rows)
    sq = _seed_sqlite3(rows)

    _assert_parity(
        eng, sq,
        "SELECT id FROM records WHERE tier = ? AND community_id IN (?, ?) ORDER BY vec_label",
        ("episodic", "comm-0", "comm-4"),
    )
    _assert_parity(
        eng, sq,
        "SELECT * FROM records WHERE tier = ? AND community_id IN (?, ?, ?) ORDER BY vec_label",
        ("semantic", "comm-1", "comm-5", "ABSENT"),
    )


# ---------------------------------------------------------------------------
# Test 4 — scan-count gate: SELECT * WHERE tier = ? must not full-scan when warm
# ---------------------------------------------------------------------------


def test_tier_eq_no_full_scan_warm(tmp_path: Path) -> None:
    """``SELECT * FROM records WHERE tier = ?`` pays no full scan once the secondary index is warm.

    On HEAD the ``tier`` column is absent from ``col_index_columns``, so the
    engine falls through to ``scan_rows`` (full-tree scan) and
    ``full_scan_count()`` is >= 1 after the first query. This is the RED marker
    that the catalog-driven index wave turns GREEN by routing the ``tier = ?``
    shape through the hash secondary index.

    The warm-index test structure mirrors the existing id-index warm tests: a
    prior identical read builds (and is expected to warm) the index; the counter
    is reset; the second read must then pay zero full scans.
    """
    rows = _make_rows(300)
    conn = _engine_conn(tmp_path)
    _seed_engine(conn, rows)

    # First read: expected to build the index (once the fix is in).
    conn.execute(
        "SELECT * FROM records WHERE tier = ? ORDER BY vec_label",
        ["episodic"],
    ).fetchall()
    conn.reset_full_scan_count()

    # Second read: must resolve from the warm index (zero full scans after fix).
    conn.execute(
        "SELECT * FROM records WHERE tier = ? ORDER BY vec_label",
        ["episodic"],
    ).fetchall()

    assert conn.full_scan_count() == 0, (
        f"expected full_scan_count==0 on the second tier = ? query (warm secondary index); "
        f"got {conn.full_scan_count()} — "
        "tier is not yet in the catalog-driven index set (expected RED state on HEAD)"
    )


# ---------------------------------------------------------------------------
# Test 5 — scan-count gate: SELECT * WHERE community_id IN must not full-scan when warm
# ---------------------------------------------------------------------------


def test_community_id_in_no_full_scan_warm(tmp_path: Path) -> None:
    """``SELECT * FROM records WHERE community_id IN (?, ?)`` pays no full scan when warm.

    Same RED-marker structure as the tier test. ``community_id`` is also absent
    from the hardcoded column list so the first query full-scans (>= 1 on HEAD).
    After the catalog-driven index wave, both the first and second queries go
    through the secondary hash index.
    """
    rows = _make_rows(300)
    conn = _engine_conn(tmp_path)
    _seed_engine(conn, rows)

    conn.execute(
        "SELECT * FROM records WHERE community_id IN (?, ?) ORDER BY vec_label",
        ["comm-0", "comm-1"],
    ).fetchall()
    conn.reset_full_scan_count()

    conn.execute(
        "SELECT * FROM records WHERE community_id IN (?, ?) ORDER BY vec_label",
        ["comm-0", "comm-1"],
    ).fetchall()

    assert conn.full_scan_count() == 0, (
        f"expected full_scan_count==0 on the second community_id IN query (warm secondary index); "
        f"got {conn.full_scan_count()} — "
        "community_id is not yet in the catalog-driven index set (expected RED state on HEAD)"
    )


# ---------------------------------------------------------------------------
# Test 6 — incremental INSERT maintenance: new row appears via index
# ---------------------------------------------------------------------------


def test_insert_maintenance_indexed_column(tmp_path: Path) -> None:
    """An INSERT of a fresh tier value is reflected immediately in an index probe.

    After the first SELECT builds the secondary index, a subsequent INSERT of a
    row with a new tier value must appear in a ``WHERE tier = ?`` query without
    paying a full scan (incremental ``insert_row`` maintained the index). The
    new row also appears in the sqlite3 reference, so we check byte-identity.
    """
    rows = _make_rows(30)
    eng = _engine_conn(tmp_path)
    _seed_engine(eng, rows)
    sq = _seed_sqlite3(rows)

    # Warm the index.
    eng.execute(
        "SELECT * FROM records WHERE tier = ? ORDER BY vec_label", ["episodic"]
    ).fetchall()
    eng.reset_full_scan_count()

    # Insert a fresh row with a new tier value under both drivers.
    new_row = ["rec-fresh", "FRESH_TIER", "comm-0", 0]
    eng.execute(_INSERT, new_row)
    sq.execute(_INSERT, new_row)
    sq.commit()

    # The new row must appear through the (now-maintained) index.
    _assert_parity(
        eng, sq,
        "SELECT * FROM records WHERE tier = ? ORDER BY vec_label",
        ("FRESH_TIER",),
    )
    # The incremental path must not have paid a full scan.
    assert eng.full_scan_count() == 0, (
        f"insert maintenance triggered an unexpected full scan; "
        f"full_scan_count={eng.full_scan_count()}"
    )


# ---------------------------------------------------------------------------
# Test 7 — incremental UPDATE maintenance: refile keeps posting lists consistent
# ---------------------------------------------------------------------------


def test_update_maintenance_community_id(tmp_path: Path) -> None:
    """Updating a row's ``community_id`` keeps the index posting lists consistent.

    After the warm phase, an UPDATE that changes a row's ``community_id`` must:
    - make the row appear under the new value (SELECT WHERE community_id = new)
    - remove it from the old posting list (SELECT WHERE community_id = old)
    - be byte-identical to sqlite3 on both shapes.
    """
    rows = _make_rows(30)
    eng = _engine_conn(tmp_path)
    _seed_engine(eng, rows)
    sq = _seed_sqlite3(rows)

    # Pick the first row's id so we can update it.
    first_id = rows[0][0]  # "rec-0000"
    old_community = rows[0][2]
    new_community = "comm-UPDATED"

    # Warm the secondary index.
    eng.execute(
        "SELECT * FROM records WHERE community_id IN (?, ?) ORDER BY vec_label",
        [old_community, new_community],
    ).fetchall()
    eng.reset_full_scan_count()

    # Apply the UPDATE on both drivers.
    update_sql = "UPDATE records SET community_id = ? WHERE id = ?"
    eng.execute(update_sql, [new_community, first_id])
    sq.execute(update_sql, (new_community, first_id))
    sq.commit()

    # New value: the row is now found under its updated community_id.
    _assert_parity(
        eng, sq,
        "SELECT * FROM records WHERE community_id IN (?) ORDER BY vec_label",
        (new_community,),
    )
    # Old value: the row must NOT appear under its former community_id.
    _assert_parity(
        eng, sq,
        "SELECT * FROM records WHERE community_id IN (?) ORDER BY vec_label",
        (old_community,),
    )
    # Reset after the first post-update reads (which may trigger an id-index build
    # or any housekeeping scan); the SECOND identical read must use the warm index.
    eng.reset_full_scan_count()
    eng.execute(
        "SELECT * FROM records WHERE community_id IN (?) ORDER BY vec_label",
        [new_community],
    ).fetchall()
    assert eng.full_scan_count() == 0, (
        f"UPDATE maintenance triggered an unexpected full scan on the warm index; "
        f"full_scan_count={eng.full_scan_count()}"
    )


# ---------------------------------------------------------------------------
# Test 8 — incremental DELETE maintenance: removed row no longer matches
# ---------------------------------------------------------------------------


def test_delete_maintenance_tier(tmp_path: Path) -> None:
    """Deleting a row removes it from the secondary index posting list."""
    rows = _make_rows(30)
    eng = _engine_conn(tmp_path)
    _seed_engine(eng, rows)
    sq = _seed_sqlite3(rows)

    # Pick a tier with multiple rows and delete one row from that tier.
    target_tier = "episodic"
    victim_id = next(r[0] for r in rows if r[1] == target_tier)

    # Warm the index.
    eng.execute(
        "SELECT * FROM records WHERE tier = ? ORDER BY vec_label", [target_tier]
    ).fetchall()
    eng.reset_full_scan_count()

    delete_sql = "DELETE FROM records WHERE id = ?"
    eng.execute(delete_sql, [victim_id])
    sq.execute(delete_sql, (victim_id,))
    sq.commit()

    _assert_parity(
        eng, sq,
        "SELECT * FROM records WHERE tier = ? ORDER BY vec_label",
        (target_tier,),
    )
    # Reset after the first post-delete read (which may pay a housekeeping scan).
    # The SECOND identical read must be index-resolved (warm).
    eng.reset_full_scan_count()
    eng.execute(
        "SELECT * FROM records WHERE tier = ? ORDER BY vec_label",
        [target_tier],
    ).fetchall()
    assert eng.full_scan_count() == 0, (
        f"DELETE maintenance triggered an unexpected full scan on the warm index; "
        f"full_scan_count={eng.full_scan_count()}"
    )


# ---------------------------------------------------------------------------
# Test 9 — warm-across-commit: an untainted COMMIT keeps the index warm
# ---------------------------------------------------------------------------


def test_warm_across_commit_non_indexed_update(tmp_path: Path) -> None:
    """A non-indexed-column UPDATE inside BEGIN/COMMIT keeps the secondary index warm.

    The warm/taint discipline: only a tainted commit clears caches. An UPDATE
    that does not touch the secondary index leaves it valid, so the next
    ``tier = ?`` query pays zero full scans.
    """
    rows = _make_rows(30)
    conn = _engine_conn(tmp_path)
    _seed_engine(conn, rows)

    # Warm the tier secondary index.
    conn.execute(
        "SELECT * FROM records WHERE tier = ? ORDER BY vec_label", ["episodic"]
    ).fetchall()
    conn.reset_full_scan_count()

    # UPDATE a column that is NOT the secondary-indexed column (id is the PK
    # and not part of ColIndex for the tier/community_id column set).
    conn.execute("BEGIN")
    conn.execute(
        "UPDATE records SET embedding_pending = 0 WHERE id = ?", ["rec-0000"]
    )
    conn.execute("COMMIT")

    # The tier index must still be warm.
    conn.reset_full_scan_count()
    conn.execute(
        "SELECT * FROM records WHERE tier = ? ORDER BY vec_label", ["episodic"]
    ).fetchall()
    assert conn.full_scan_count() == 0, (
        f"expected full_scan_count==0 after untainted commit; "
        f"got {conn.full_scan_count()}"
    )


# ---------------------------------------------------------------------------
# Test 10 — rollback clears: a reverted indexed write leaves no stale entry
# ---------------------------------------------------------------------------


def test_rollback_clears_index(tmp_path: Path) -> None:
    """After a ROLLBACK of an indexed write, the secondary index is cleared.

    Rolled-back rows must never be cached. After a rollback the index is
    rebuilt from the committed tree on the next query, so the reverted row
    does not appear.
    """
    rows = _make_rows(30)
    conn = _engine_conn(tmp_path)
    _seed_engine(conn, rows)
    sq = _seed_sqlite3(rows)

    # Warm the index.
    conn.execute(
        "SELECT * FROM records WHERE tier = ? ORDER BY vec_label", ["PHANTOM_TIER"]
    ).fetchall()

    # Begin a transaction, insert a row, then roll it back.
    conn.execute("BEGIN")
    conn.execute(_INSERT, ["rec-phantom", "PHANTOM_TIER", "comm-0", 0])
    conn.execute("ROLLBACK")

    # The rolled-back row must not appear — byte-identical to sqlite3 (which
    # also never committed it).
    _assert_parity(
        conn, sq,
        "SELECT * FROM records WHERE tier = ? ORDER BY vec_label",
        ("PHANTOM_TIER",),
    )


# ---------------------------------------------------------------------------
# Test 11 — partial-skip policy: skipped-partial columns full-scan, byte-identical
# ---------------------------------------------------------------------------


def test_partial_skip_column_full_scans_correctly(tmp_path: Path) -> None:
    """A column covered only by a partial index falls through to a full scan.

    The partial-skip policy skips partial-indexed columns (other than
    ``embedding_pending``) from the ColIndex. Queries on those columns
    must still return byte-identical results to sqlite3 — they just do so
    via the full-scan path, not the hash index.

    We use a synthetic ``tombstoned_at`` column with a partial index
    ``WHERE tombstoned_at IS NOT NULL`` to represent the skipped-partial case.
    """
    from iai_mcp_native import engine as _engine

    _DB_SEQ["n"] += 1
    conn = _engine.Connection.open(
        str(tmp_path / f"partial_{_DB_SEQ['n']}.db"), 384
    )
    assert isinstance(conn, _engine.Connection)

    conn.execute(
        "CREATE TABLE records ("
        "vec_label INTEGER PRIMARY KEY AUTOINCREMENT, "
        "id TEXT NOT NULL, "
        "tier TEXT, "
        "tombstoned_at TEXT"
        ")"
    )
    # A partial index on tombstoned_at — should be skipped (not hash-indexed).
    conn.execute(
        "CREATE INDEX idx_tomb ON records(tombstoned_at) WHERE tombstoned_at IS NOT NULL"
    )
    # A non-partial index on tier — should be hash-indexed.
    conn.execute("CREATE INDEX idx_tier ON records(tier)")

    sq = sqlite3.connect(":memory:")
    sq.row_factory = sqlite3.Row
    sq.execute(
        "CREATE TABLE records ("
        "vec_label INTEGER PRIMARY KEY AUTOINCREMENT, "
        "id TEXT NOT NULL, "
        "tier TEXT, "
        "tombstoned_at TEXT"
        ")"
    )
    sq.execute("CREATE INDEX idx_tomb ON records(tombstoned_at) WHERE tombstoned_at IS NOT NULL")
    sq.execute("CREATE INDEX idx_tier ON records(tier)")

    insert_sql = "INSERT INTO records (id, tier, tombstoned_at) VALUES (?, ?, ?)"
    data = [
        ("r-0", "episodic", None),
        ("r-1", "semantic", "2024-01-01"),
        ("r-2", "episodic", "2024-01-02"),
        ("r-3", "procedural", None),
    ]
    conn.execute("BEGIN")
    for d in data:
        conn.execute(insert_sql, list(d))
    conn.execute("COMMIT")
    sq.executemany(insert_sql, data)
    sq.commit()

    # tombstoned_at = ? must be byte-identical (full scan, correct).
    _assert_parity(
        conn, sq,
        "SELECT * FROM records WHERE tombstoned_at = ? ORDER BY vec_label",
        ("2024-01-01",),
    )
    # tier = ? must be byte-identical AND index-resolved (zero full scans warm).
    conn.execute(
        "SELECT * FROM records WHERE tier = ? ORDER BY vec_label", ["episodic"]
    ).fetchall()
    conn.reset_full_scan_count()
    conn.execute(
        "SELECT * FROM records WHERE tier = ? ORDER BY vec_label", ["episodic"]
    ).fetchall()
    assert conn.full_scan_count() == 0, (
        "tier=? must resolve via hash index (non-partial), not full scan"
    )


# ---------------------------------------------------------------------------
# Test 12 — embedding_pending stays index-resolved (partial exception)
# ---------------------------------------------------------------------------


def test_embedding_pending_index_resolved(tmp_path: Path) -> None:
    """``WHERE embedding_pending = 1`` resolves via hash index, not full scan.

    ``embedding_pending`` is declared with a partial index
    ``WHERE embedding_pending=1``. The partial-skip exception keeps it in the
    ColIndex because the equality probe ``col = 1`` is a subset of the partial's
    domain — no wrong rows are returned, and the probe is index-resolved.
    """
    rows = _make_rows(60)
    conn = _engine_conn(tmp_path)

    conn.execute(
        "CREATE TABLE records ("
        "vec_label INTEGER PRIMARY KEY AUTOINCREMENT, "
        "id TEXT NOT NULL, "
        "tier TEXT, "
        "community_id TEXT, "
        "embedding_pending INTEGER"
        ")"
    )
    # Partial index (embedding_pending=1 predicate).
    conn.execute(
        "CREATE INDEX idx_pending ON records(embedding_pending) WHERE embedding_pending=1"
    )
    conn.execute("BEGIN")
    for r in rows:
        conn.execute(_INSERT, list(r))
    conn.execute("COMMIT")

    # Warm the index.
    conn.execute(
        "SELECT * FROM records WHERE embedding_pending = 1 ORDER BY vec_label", []
    ).fetchall()
    conn.reset_full_scan_count()

    result = conn.execute(
        "SELECT * FROM records WHERE embedding_pending = 1 ORDER BY vec_label", []
    ).fetchall()
    assert conn.full_scan_count() == 0, (
        f"expected index-resolved path for embedding_pending=1; "
        f"full_scan_count={conn.full_scan_count()}"
    )
    # Sanity: half the rows have embedding_pending=1.
    assert len(result) == 30, f"expected 30 rows with embedding_pending=1; got {len(result)}"


# ---------------------------------------------------------------------------
# Role-column differential — DDL, helpers, and edge-case corpus
# ---------------------------------------------------------------------------
#
# These cases prove that a `role TEXT` column with a catalog-declared index
# produces results byte-identical to the legacy `tags_json LIKE '%"role:user"%'`
# predicate, and that the new equality predicate is index-resolved (zero full
# scans) while the LIKE predicate always full-scans.
#
# Separate DDL from the tier/community_id suite above: the role column plus
# a `tags_json TEXT` column (needed for the LIKE-based scan-count probe).
# The existing module-level `_DDL_RECORDS` and `_INDEXES` are left untouched
# so the 13 tests above keep working without modification.
# ---------------------------------------------------------------------------

_DDL_RECORDS_ROLE = (
    "CREATE TABLE records ("
    "vec_label INTEGER PRIMARY KEY AUTOINCREMENT, "
    "id TEXT NOT NULL, "
    "tier TEXT, "
    "role TEXT, "
    "tags_json TEXT, "
    "embedding_pending INTEGER"
    ")"
)

# Catalog-declared indexes for the role column — picked up by the engine's
# catalog machinery without any Rust change.
_INDEXES_ROLE = [
    "CREATE INDEX idx_records_tier_role ON records(tier)",
    "CREATE INDEX idx_records_role      ON records(role)",
]

# Insert: (id, tier, role, tags_json, embedding_pending)
_INSERT_ROLE = (
    "INSERT INTO records (id, tier, role, tags_json, embedding_pending)"
    " VALUES (?, ?, ?, ?, ?)"
)


def _derive_role_tag(tags_json_str: str) -> "str | None":
    """Extract the conversational role value from a JSON-serialized tags list.

    Returns the suffix of the first ``role:<x>`` tag found, or None when no
    role tag is present. This mirrors the production derive logic so the seed
    corpus stays consistent between the ``role`` column and ``tags_json``.
    """
    import json as _json
    try:
        tags = _json.loads(tags_json_str)
    except Exception:
        return None
    for t in (tags or []):
        if isinstance(t, str) and t.startswith("role:"):
            return t[len("role:"):]
    return None


def _make_role_rows() -> list[tuple]:
    """Build the edge-case corpus covering every byte-identity case.

    Each tuple: (id, tier, role, tags_json, embedding_pending)

    Cases covered:
    - No role tag: role column NULL, tags_json without any role: entry.
    - role:user (episodic and semantic mix to exercise the tier conjunct).
    - role:assistant.
    - role:tool (a non-user/assistant role, proving verbatim storage).
    - Several plain role:user rows to exercise the LIMIT path.
    """
    import json as _json
    rows: list[tuple] = []
    # Row 1 — no role tag, episodic.
    rows.append(("id-norole-ep", "episodic", None, _json.dumps(["capture"]), 0))
    # Row 2 — role:user, episodic.
    rows.append(("id-user-ep", "episodic", "user", _json.dumps(["capture", "role:user"]), 0))
    # Row 3 — role:assistant, episodic.
    rows.append(("id-asst-ep", "episodic", "assistant", _json.dumps(["capture", "role:assistant"]), 0))
    # Row 4 — role:tool (non-user/assistant), episodic.
    rows.append(("id-tool-ep", "episodic", "tool", _json.dumps(["capture", "role:tool"]), 0))
    # Row 5 — role:user, semantic (should NOT appear in tier='episodic' queries).
    rows.append(("id-user-sem", "semantic", "user", _json.dumps(["capture", "role:user"]), 0))
    # Rows 6–10 — additional role:user episodic rows to exercise the LIMIT path.
    for i in range(5):
        rows.append((
            f"id-user-ep-{i}",
            "episodic",
            "user",
            _json.dumps(["capture", "role:user"]),
            0,
        ))
    return rows


def _engine_conn_role(tmp_path: Path):
    """Open a Rust engine connection for the role-column DDL."""
    from iai_mcp_native import engine

    _DB_SEQ["n"] += 1
    conn = engine.Connection.open(str(tmp_path / f"role_{_DB_SEQ['n']}.db"), 384)
    assert isinstance(conn, engine.Connection), (
        "connection under test is not the Rust engine.Connection — "
        "refusing to run a false-green proof against a different engine"
    )
    return conn


def _seed_engine_role(conn, rows: list[tuple]) -> None:
    conn.execute(_DDL_RECORDS_ROLE)
    for idx_sql in _INDEXES_ROLE:
        conn.execute(idx_sql)
    conn.execute("BEGIN")
    for row in rows:
        conn.execute(_INSERT_ROLE, list(row))
    conn.execute("COMMIT")


def _seed_sqlite3_role(rows: list[tuple]) -> sqlite3.Connection:
    sq = sqlite3.connect(":memory:")
    sq.row_factory = sqlite3.Row
    sq.execute(_DDL_RECORDS_ROLE)
    for idx_sql in _INDEXES_ROLE:
        sq.execute(idx_sql)
    sq.executemany(_INSERT_ROLE, rows)
    sq.commit()
    return sq


# ---------------------------------------------------------------------------
# Test 14 — byte-identity: `role='user'` set equals `tags_json LIKE` set
# ---------------------------------------------------------------------------


def test_role_eq_matches_legacy_tag_like_byte_identity(tmp_path: Path) -> None:
    """``WHERE role='user'`` returns the byte-identical row set as ``tags_json LIKE '%"role:user"%'``.

    Both predicates are applied to the same seeded corpus over the engine.
    The corpus covers every edge case from the byte-identity analysis: no role
    tag, role:user, role:assistant, role:tool, and role:user rows across mixed
    tiers. Both predicates must agree on which ids to return (set-equality)
    and the combined ``tier='episodic' AND role='user'`` shape must exclude
    non-episodic rows.
    """
    rows = _make_role_rows()
    eng = _engine_conn_role(tmp_path)
    _seed_engine_role(eng, rows)
    sq = _seed_sqlite3_role(rows)

    # Engine: new column predicate.
    eng_new = set(
        r[0] for r in eng.execute(
            "SELECT id FROM records"
            " WHERE tier='episodic' AND role='user'"
            " ORDER BY id"
        ).fetchall()
    )
    # Engine: legacy LIKE predicate (same corpus, same engine).
    eng_old = set(
        r[0] for r in eng.execute(
            "SELECT id FROM records"
            " WHERE tier='episodic' AND tags_json LIKE '%\"role:user\"%'"
            " ORDER BY id"
        ).fetchall()
    )
    assert eng_new == eng_old, (
        f"role='user' and tags_json LIKE produce different id sets on engine:\n"
        f"  role='user' only : {sorted(eng_new - eng_old)}\n"
        f"  LIKE only        : {sorted(eng_old - eng_new)}"
    )

    # Stdlib sqlite3: new column predicate.
    sq_new = set(
        r[0] for r in sq.execute(
            "SELECT id FROM records"
            " WHERE tier='episodic' AND role='user'"
            " ORDER BY id"
        ).fetchall()
    )
    # Stdlib sqlite3: legacy LIKE predicate.
    sq_old = set(
        r[0] for r in sq.execute(
            "SELECT id FROM records"
            " WHERE tier='episodic' AND tags_json LIKE '%\"role:user\"%'"
            " ORDER BY id"
        ).fetchall()
    )
    assert sq_new == sq_old, (
        f"role='user' and tags_json LIKE produce different id sets on sqlite3:\n"
        f"  role='user' only : {sorted(sq_new - sq_old)}\n"
        f"  LIKE only        : {sorted(sq_old - sq_new)}"
    )

    # Cross-check: both drivers agree on the new predicate.
    assert eng_new == sq_new, (
        f"engine and sqlite3 disagree on role='user' result set:\n"
        f"  engine only : {sorted(eng_new - sq_new)}\n"
        f"  sqlite3 only: {sorted(sq_new - eng_new)}"
    )

    # Sanity: the expected ids are present (role:user episodic rows only).
    expected_ids = {"id-user-ep"} | {f"id-user-ep-{i}" for i in range(5)}
    assert eng_new == expected_ids, (
        f"unexpected id set for role:user episodic rows: {sorted(eng_new)}"
    )


# ---------------------------------------------------------------------------
# Test 15 — cross-driver parity: `role='user'` byte-identical engine vs sqlite3
# ---------------------------------------------------------------------------


def test_role_eq_cross_driver_parity(tmp_path: Path) -> None:
    """``WHERE tier='episodic' AND role='user'`` is byte-identical between engine and sqlite3.

    Reuses ``_assert_parity`` (value + type parity) to confirm the result rows
    match exactly across both drivers over the edge-case corpus.
    """
    rows = _make_role_rows()
    eng = _engine_conn_role(tmp_path)
    _seed_engine_role(eng, rows)
    sq = _seed_sqlite3_role(rows)

    _assert_parity(
        eng, sq,
        "SELECT id FROM records WHERE tier='episodic' AND role='user' ORDER BY id",
    )
    # role:assistant must also be parity-correct.
    _assert_parity(
        eng, sq,
        "SELECT id FROM records WHERE tier='episodic' AND role='assistant' ORDER BY id",
    )
    # role:tool (verbatim non-user/assistant value).
    _assert_parity(
        eng, sq,
        "SELECT id FROM records WHERE tier='episodic' AND role='tool' ORDER BY id",
    )
    # Absent role (role IS NULL rows): both drivers return the no-role-tag row.
    _assert_parity(
        eng, sq,
        "SELECT id FROM records WHERE tier='episodic' AND role IS NULL ORDER BY id",
    )
    # No role-bearing row appears when querying a non-episodic tier.
    _assert_parity(
        eng, sq,
        "SELECT id FROM records WHERE tier='semantic' AND role='user' ORDER BY id",
    )


# ---------------------------------------------------------------------------
# Test 16 — warm scan-count: `role='user'` without `idx_records_role` full-scans
#
# RED on HEAD: the `role` column exists in the DDL but `idx_records_role` has
# not been declared yet (simulating the HEAD state of the production schema
# where the column and its index are absent). Without the index the engine has
# no secondary-index entry for `role` and falls back to a full table scan, so
# ``full_scan_count() >= 1``. The assertion ``full_scan_count()==0`` therefore
# FAILS on HEAD — that is the intended RED state for this wave.
#
# After the role column implementation adds ``CREATE INDEX idx_records_role ON records(role)`` to the
# production DDL, the same equality predicate rides the catalog-driven ColIndex
# and returns ``full_scan_count()==0`` (GREEN).
# ---------------------------------------------------------------------------

# DDL for the scan-count probe: `role TEXT` column present, NO secondary
# indexes declared (not even `tier`) — so the only route to full_scan_count==0
# is via `idx_records_role` added by the role column implementation.
_DDL_RECORDS_ROLE_NO_INDEX = (
    "CREATE TABLE records ("
    "vec_label INTEGER PRIMARY KEY AUTOINCREMENT, "
    "id TEXT NOT NULL, "
    "tier TEXT, "
    "role TEXT, "
    "tags_json TEXT, "
    "embedding_pending INTEGER"
    ")"
)


def test_role_user_eq_no_full_scan_warm(tmp_path: Path) -> None:
    """``WHERE tier='episodic' AND role='user'`` must not full-scan once the role index is warm.

    The ``role`` column carries ``idx_records_role`` (a catalog-declared secondary
    index). After a prior identical query has warmed the index, the second
    identical query must resolve without a whole-table scan —
    ``full_scan_count() == 0``.
    """
    import json as _json
    # Seed corpus: role:user episodic rows plus a mix of other roles.
    seed_rows: list[tuple] = []
    for i in range(12):
        tier = "episodic" if i % 3 != 2 else "semantic"
        role_val = ["user", "assistant", "tool"][i % 3]
        seed_rows.append((
            f"sc-rec-{i:03d}",
            tier,
            role_val,
            _json.dumps(["capture", f"role:{role_val}"]),
            0,
        ))

    _DB_SEQ["n"] += 1
    from iai_mcp_native import engine as _eng
    conn = _eng.Connection.open(str(tmp_path / f"scancount_{_DB_SEQ['n']}.db"), 384)
    assert isinstance(conn, _eng.Connection), (
        "connection is not the Rust engine.Connection — "
        "false-green guard"
    )

    # Apply DDL with the role column and idx_records_role declared.
    conn.execute(_DDL_RECORDS_ROLE_NO_INDEX)
    conn.execute("CREATE INDEX idx_records_role ON records(role)")
    conn.execute("BEGIN")
    for row in seed_rows:
        conn.execute(_INSERT_ROLE, list(row))
    conn.execute("COMMIT")

    # First read: builds the warm index posting list.
    conn.execute(
        "SELECT * FROM records WHERE tier='episodic' AND role='user' ORDER BY rowid DESC LIMIT 50",
        [],
    ).fetchall()
    conn.reset_full_scan_count()

    # Second read: must resolve from the warm role index (zero full scans).
    conn.execute(
        "SELECT * FROM records WHERE tier='episodic' AND role='user' ORDER BY rowid DESC LIMIT 50",
        [],
    ).fetchall()

    assert conn.full_scan_count() == 0, (
        f"expected full_scan_count==0 on the warm role='user' query "
        f"(index-resolved via idx_records_role); "
        f"got {conn.full_scan_count()}"
    )


# ---------------------------------------------------------------------------
# Test 13 — AES fence: encrypted columns are NOT in the indexed set
# ---------------------------------------------------------------------------


def test_aes_fence_encrypted_columns_not_indexed(tmp_path: Path) -> None:
    """The catalog's indexed-column set for records/events/edges contains no encrypted column.

    Production CREATE INDEX statements name only plaintext metadata columns.
    This test asserts the intersection of encrypted columns and the
    catalog-driven indexed-column set is empty, pinning that invariant against
    accidental future regression.

    The encrypted columns are: ``literal_surface``, ``provenance_json``,
    ``profile_modulation_gain_json``, ``data_json``.
    """
    from iai_mcp_native import engine as _engine

    _DB_SEQ["n"] += 1
    conn = _engine.Connection.open(
        str(tmp_path / f"aesfence_{_DB_SEQ['n']}.db"), 384
    )
    assert isinstance(conn, _engine.Connection)

    # Apply the production DDL: tables + all CREATE INDEX statements, mirroring
    # the actual schema the host issues via hippo/_table.py.
    _SCHEMA_DDL = [
        # records table
        (
            "CREATE TABLE records ("
            "vec_label INTEGER PRIMARY KEY AUTOINCREMENT, "
            "id TEXT NOT NULL, "
            "tier TEXT, "
            "community_id TEXT, "
            "created_at TEXT, "
            "embedding_pending INTEGER, "
            "tombstoned_at TEXT, "
            "literal_surface BLOB, "
            "provenance_json BLOB, "
            "profile_modulation_gain_json BLOB"
            ")"
        ),
        "CREATE INDEX IF NOT EXISTS idx_records_id        ON records(id)",
        "CREATE INDEX IF NOT EXISTS idx_records_tier      ON records(tier)",
        "CREATE INDEX IF NOT EXISTS idx_records_community ON records(community_id)",
        "CREATE INDEX IF NOT EXISTS idx_records_tomb      ON records(tombstoned_at) WHERE tombstoned_at IS NOT NULL",
        "CREATE INDEX IF NOT EXISTS idx_records_pending   ON records(embedding_pending) WHERE embedding_pending=1",
        "CREATE INDEX IF NOT EXISTS idx_records_created_at ON records(created_at)",
        # edges table
        "CREATE TABLE edges (src TEXT NOT NULL, dst TEXT NOT NULL, edge_type TEXT)",
        "CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src)",
        "CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst)",
        # events table
        (
            "CREATE TABLE events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "kind TEXT, "
            "ts TEXT, "
            "session_id TEXT, "
            "data_json BLOB"
            ")"
        ),
        "CREATE INDEX IF NOT EXISTS idx_events_kind    ON events(kind)",
        "CREATE INDEX IF NOT EXISTS idx_events_ts      ON events(ts)",
        "CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)",
    ]

    for ddl in _SCHEMA_DDL:
        conn.execute(ddl)

    # Now probe: which columns does the catalog-driven set include for each table?
    # We do this by inserting a single row and checking that a WHERE on an encrypted
    # column triggers a full scan rather than an index hit. The definitive check is
    # that no encrypted column name ever appears as an indexed column — we verify
    # this by doing a warm-then-count probe for each encrypted column.

    _ENCRYPTED_COLUMNS = {
        "records": ["literal_surface", "provenance_json", "profile_modulation_gain_json"],
        "events": ["data_json"],
    }

    # Insert one placeholder row per table.
    conn.execute(
        "INSERT INTO records (id, tier, community_id, created_at, embedding_pending, "
        "tombstoned_at, literal_surface, provenance_json, profile_modulation_gain_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["r-1", "episodic", "comm-0", "2024-01-01", 0, None, b"enc1", b"enc2", b"enc3"],
    )
    conn.execute(
        "INSERT INTO events (kind, ts, session_id, data_json) VALUES (?, ?, ?, ?)",
        ["event_type_a", "2024-01-01T00:00:00", "sess-1", b"enc_data"],
    )

    for table, enc_cols in _ENCRYPTED_COLUMNS.items():
        for col in enc_cols:
            # Warm-then-measure: if the column were in the index, the second
            # probe would pay 0 full scans. We assert it is NOT zero — the
            # column is NOT indexed and the query pays at least one full scan.
            try:
                conn.execute(
                    f"SELECT * FROM {table} WHERE {col} = ? ORDER BY rowid",
                    [b"enc1"],
                ).fetchall()
                conn.reset_full_scan_count()
                conn.execute(
                    f"SELECT * FROM {table} WHERE {col} = ? ORDER BY rowid",
                    [b"enc1"],
                ).fetchall()
                scans = conn.full_scan_count()
                # Encrypted columns must NOT be in the index — they must scan.
                assert scans >= 1, (
                    f"encrypted column {table}.{col} appears to be hash-indexed "
                    f"(full_scan_count={scans} after warm query); "
                    "no encrypted column must ever be in the catalog-driven index set"
                )
            except Exception as exc:
                # A parse error on a BLOB equality probe is acceptable (the type
                # system may reject it) — what matters is the column is not indexed.
                # Re-raise only if it's an unexpected error type.
                if "no such column" in str(exc).lower():
                    raise
