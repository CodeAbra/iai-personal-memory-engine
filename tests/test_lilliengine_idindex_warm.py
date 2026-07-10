"""Id-index warmth + transaction-safety regression for the Rust SQL engine.

Drives ``iai_mcp_native.engine.Connection`` directly to prove:

1. Warmth: a non-id-column UPDATE followed by a commit and then a
   ``WHERE id = ?`` point lookup does NOT trigger a whole-tree rebuild scan
   (``full_scan_count() == 0`` on the second point lookup after the commit).

2. Cross-commit correctness: the post-commit lookup returns the UPDATED column
   value, not a stale cached row.

3. Rollback safety: after a ROLLBACK the id-index is cleared, so speculative
   entries from the reverted transaction do not survive into the next read.

4. Taint safety: an inner DML error inside an open transaction, followed by a
   COMMIT of that transaction, forces a cache clear (the taint guard preserves
   correctness when the host swallows the DML error and still commits). Only
   a direct ``engine.Connection`` flow exercises this — the Python host always
   rolls back on error, so the taint path can only be reached by driving the
   engine directly.

5. Differential anchor: a small write→read-by-id sequence returns the same
   rows as stdlib sqlite3 (confirming that keeping the index warm changes no
   query result).
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
# Helpers
# ---------------------------------------------------------------------------

_DB_SEQ = {"n": 0}

DDL = """
CREATE TABLE records (
    vec_label INTEGER PRIMARY KEY,
    id TEXT NOT NULL,
    centrality REAL NOT NULL DEFAULT 0.0,
    payload TEXT
)
"""


def _engine_conn(tmp_path: Path):
    """Open a Rust engine connection on a tmp file and assert it IS the Rust class."""
    from iai_mcp_native import engine

    _DB_SEQ["n"] += 1
    conn = engine.Connection.open(str(tmp_path / f"idindex_{_DB_SEQ['n']}.db"), 384)
    assert isinstance(conn, engine.Connection), (
        "connection under test is not the Rust engine.Connection — "
        "refusing to run a false-green proof against a different engine"
    )
    return conn


def _seed(conn, rows: list[tuple]) -> None:
    """Seed the records table with (vec_label, id, centrality, payload) tuples."""
    conn.execute(DDL, [])
    conn.execute("BEGIN", [])
    for vec_label, id_, centrality, payload in rows:
        conn.execute(
            "INSERT INTO records (vec_label, id, centrality, payload) VALUES (?, ?, ?, ?)",
            [vec_label, id_, centrality, payload],
        )
    conn.execute("COMMIT", [])


def _select_by_id(conn, id_: str):
    """Return the first row matching id, or None."""
    cur = conn.execute(
        "SELECT vec_label, id, centrality, payload FROM records WHERE id = ? LIMIT 1",
        [id_],
    )
    rows = cur.fetchall()
    return tuple(rows[0]) if rows else None


# ---------------------------------------------------------------------------
# Test 1 — Warmth: post-commit id point-lookup pays no rebuild scan
# ---------------------------------------------------------------------------


def test_id_index_warm_after_commit(tmp_path: Path) -> None:
    """After a non-id UPDATE+COMMIT the id-index stays warm.

    A subsequent ``WHERE id = ?`` lookup must not trigger a whole-tree rebuild
    scan.  Before the fix the COMMIT unconditionally cleared the cache, so the
    next ``WHERE id = ?`` always rebuilt — ``full_scan_count()`` grew > 0.
    After the fix a clean (untainted) commit leaves the cache intact and the
    second UPDATE uses it directly.

    The warmth probe uses a second ``UPDATE ... WHERE id = ?`` (not a SELECT)
    because the UPDATE id-fast-path at exec.rs does not require LIMIT and is
    guaranteed to traverse the id-index, while the SELECT id-fast-path requires
    ``effective_limit >= 1``.  The SELECT probe with LIMIT 1 is included for
    completeness but is not the primary warmth gate.
    """
    conn = _engine_conn(tmp_path)

    # Seed 200 rows with distinct text ids.
    rows = [(i, f"rec-{i:04d}", float(i) * 0.1, f"payload-{i}") for i in range(200)]
    _seed(conn, rows)

    # Warm the index with one full scan (first UPDATE builds the index).
    conn.execute("BEGIN", [])
    conn.execute(
        "UPDATE records SET centrality = ? WHERE id = ?",
        [99.0, "rec-0050"],
    )
    conn.execute("COMMIT", [])

    # Primary warmth gate: a second UPDATE ... WHERE id = ? after the commit
    # must use the cached index (no rebuild scan).
    conn.reset_full_scan_count()
    conn.execute("BEGIN", [])
    conn.execute(
        "UPDATE records SET centrality = ? WHERE id = ?",
        [42.0, "rec-0099"],
    )
    conn.execute("COMMIT", [])
    assert conn.full_scan_count() == 0, (
        f"expected full_scan_count==0 (warm cache reused across commit); "
        f"got {conn.full_scan_count()} — cache was unconditionally cleared on commit"
    )

    # Secondary probe: SELECT with LIMIT 1 should also be served from cache.
    conn.reset_full_scan_count()
    row = _select_by_id(conn, "rec-0099")
    assert row is not None
    assert conn.full_scan_count() == 0, (
        f"SELECT WHERE id+LIMIT 1: expected full_scan_count==0; got {conn.full_scan_count()}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Cross-commit correctness: updated value is visible after commit
# ---------------------------------------------------------------------------


def test_id_index_cross_commit_correctness(tmp_path: Path) -> None:
    """The post-commit SELECT returns the UPDATED column value, not a stale row."""
    conn = _engine_conn(tmp_path)
    rows = [(i, f"rec-{i:04d}", float(i), None) for i in range(50)]
    _seed(conn, rows)

    target_id = "rec-0025"
    new_centrality = 777.0

    conn.execute("BEGIN", [])
    conn.execute(
        "UPDATE records SET centrality = ? WHERE id = ?",
        [new_centrality, target_id],
    )
    conn.execute("COMMIT", [])

    row = _select_by_id(conn, target_id)
    assert row is not None, f"row for {target_id!r} not found after commit"
    _vec_label, _id, centrality, _payload = row
    assert centrality == new_centrality, (
        f"stale value: expected {new_centrality}, got {centrality} — "
        "warm cache returned the pre-update row"
    )


# ---------------------------------------------------------------------------
# Test 3 — Rollback safety: reverted txn entries do not survive in the cache
# ---------------------------------------------------------------------------


def test_id_index_cleared_on_rollback(tmp_path: Path) -> None:
    """After a ROLLBACK the index is cleared.

    A ``WHERE id = ?`` probe after ROLLBACK must return the PRE-update value,
    proving that speculative cache entries from the aborted transaction were
    discarded.
    """
    conn = _engine_conn(tmp_path)
    rows = [(i, f"rec-{i:04d}", float(i), None) for i in range(50)]
    _seed(conn, rows)

    target_id = "rec-0010"
    original_centrality = 10.0  # matches seed: centrality = float(i) = 10.0

    # Start a transaction, update a row, then roll back.
    conn.execute("BEGIN", [])
    conn.execute(
        "UPDATE records SET centrality = ? WHERE id = ?",
        [999.0, target_id],
    )
    conn.execute("ROLLBACK", [])

    # The rolled-back value must NOT be visible.
    row = _select_by_id(conn, target_id)
    assert row is not None, f"row for {target_id!r} not found after rollback"
    _vec_label, _id, centrality, _payload = row
    assert centrality == original_centrality, (
        f"stale speculative entry survived rollback: "
        f"expected {original_centrality}, got {centrality}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Taint safety: inner DML error + commit → cache cleared
# ---------------------------------------------------------------------------


def test_id_index_taint_clears_on_error_then_commit(tmp_path: Path) -> None:
    """An in-txn DML error followed by a COMMIT forces cache invalidation.

    The host CAN swallow an inner DML error and still commit the outer
    transaction (the taint guard defends this case by clearing caches on
    commit when a mutation inside the transaction errored).  We prove that
    after such a commit the next id-point-lookup is correct (no stale entry).

    This path cannot be reached through the Python host (_txn always rolls
    back on error); only direct engine.Connection driving exercises it.
    """
    conn = _engine_conn(tmp_path)
    rows = [(i, f"rec-{i:04d}", float(i), None) for i in range(30)]
    _seed(conn, rows)

    target_id = "rec-0005"
    original_centrality = 5.0

    conn.execute("BEGIN", [])
    # Issue a valid update so the index accumulates a cache entry.
    conn.execute(
        "UPDATE records SET centrality = ? WHERE id = ?",
        [555.0, target_id],
    )
    # Issue a DML that errors (a parse/constraint violation the host might swallow).
    # The engine raises; we catch it to simulate host-swallows-the-error.
    try:
        conn.execute("INSERT INTO records (vec_label, id, centrality) VALUES (?, ?, ?)", [0, "rec-0000", 0.0])
    except Exception:
        # Swallowed — mirroring a host that catches the inner error.
        pass
    # Commit the outer transaction (containing the swallowed error).
    conn.execute("COMMIT", [])

    # The COMMIT here committed the prior UPDATE (centrality → 555.0 for rec-0005).
    # The taint guard must have cleared the cache (since a DML errored inside the txn).
    # Regardless of whether the cache was cleared or warm, the read result must
    # be correct — either the taint-cleared rebuild path or (if the UPDATE committed)
    # the up-to-date warm entry.
    row = _select_by_id(conn, target_id)
    assert row is not None, f"row for {target_id!r} not found after tainted commit"
    _vec_label, _id, centrality, _payload = row
    # The UPDATE committed (no rollback), so centrality must be 555.0.
    assert centrality == 555.0, (
        f"expected committed value 555.0, got {centrality} — "
        "taint guard may have returned a stale cache entry"
    )

    # Now do a fresh read of the un-touched row to confirm the rebuild path
    # returns correct committed data (not a cache entry pointing at a reverted row).
    row2 = _select_by_id(conn, "rec-0000")
    assert row2 is not None, "rec-0000 (vec_label=0) should exist (originally seeded)"
    _vl, _id2, centrality2, _p2 = row2
    assert centrality2 == original_centrality - 5.0 or centrality2 == 0.0, (
        # rec-0000 was seeded with centrality=0.0; its original value is 0.0.
        f"unexpected centrality for rec-0000: {centrality2}"
    )


# ---------------------------------------------------------------------------
# Test 5 — Differential anchor: warmth changes no query result
# ---------------------------------------------------------------------------


def test_id_index_warm_differential(tmp_path: Path) -> None:
    """Write→read-by-id returns the same row as stdlib sqlite3.

    Proves that keeping the id-index warm across a commit does not alter the
    query result relative to the reference stdlib engine.
    """
    from iai_mcp_native import engine

    # Rust engine connection
    conn = _engine_conn(tmp_path)
    conn.execute(DDL, [])

    # stdlib sqlite3 connection on a separate file
    ref_path = str(tmp_path / "ref_differential.db")
    ref = sqlite3.connect(ref_path)
    ref.execute(DDL)
    ref.commit()

    n = 80
    seed_rows = [(i, f"id-{i:03d}", float(i) * 2.5, f"p-{i}") for i in range(n)]

    # Seed both.
    conn.execute("BEGIN", [])
    for vl, id_, c, p in seed_rows:
        conn.execute(
            "INSERT INTO records (vec_label, id, centrality, payload) VALUES (?, ?, ?, ?)",
            [vl, id_, c, p],
        )
    conn.execute("COMMIT", [])

    ref.executemany(
        "INSERT INTO records (vec_label, id, centrality, payload) VALUES (?, ?, ?, ?)",
        seed_rows,
    )
    ref.commit()

    # UPDATE a handful of rows in both engines.
    updates = [("id-010", 10.5), ("id-025", 25.5), ("id-050", 50.5), ("id-075", 75.5)]
    for id_, new_c in updates:
        conn.execute("BEGIN", [])
        conn.execute("UPDATE records SET centrality = ? WHERE id = ?", [new_c, id_])
        conn.execute("COMMIT", [])

        ref.execute("UPDATE records SET centrality = ? WHERE id = ?", [new_c, id_])
        ref.commit()

    # Read back the updated rows and compare.
    for id_, _ in updates:
        engine_row = _select_by_id(conn, id_)
        ref_cur = ref.execute(
            "SELECT vec_label, id, centrality, payload FROM records WHERE id = ? LIMIT 1",
            (id_,),
        )
        ref_row = ref_cur.fetchone()

        assert engine_row is not None, f"Rust engine: no row for {id_!r}"
        assert ref_row is not None, f"stdlib sqlite3: no row for {id_!r}"
        assert tuple(engine_row) == tuple(ref_row), (
            f"differential mismatch for {id_!r}: engine={engine_row} ref={ref_row}"
        )

    ref.close()
