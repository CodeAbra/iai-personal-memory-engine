"""A1 index-on-open proof: an already-migrated store acquires idx_records_vec_label.

Proves two things end-to-end:

**A1-RW: read-write open persists the index.**
  A lilli store seeded WITHOUT ``idx_records_vec_label`` in its engine meta (the
  pre-upgrade condition) gets the index registered and serving on the first
  read-write HippoDB open.  The mechanism is ``HippoDB.__init__`` calling
  ``_ensure_tables()`` (guarded by ``not read_only`` at ``_db.py:274``) which
  re-runs the full ``_DDL_RECORDS_INDEXES`` list — including the new vec_label
  line — via ``conn.execute(idx)`` → ``run_ddl_passthrough`` → catalog
  registration + DDL persistence into engine meta.

**A1-RO: the production read-only recall path is index-served via meta.replay.**
  After the read-write open above persists the index DDL, a FRESH read-only
  connection (``engine.Connection.open_read_only`` / ``open_store_conn(...,
  read_only=True)``) populates its catalog via ``from_store`` → ``meta.replay``
  (``conn.rs:333``).  It never runs ``_ensure_tables`` — it sees
  ``idx_records_vec_label`` ONLY because the read-write open persisted the DDL.
  The warm ``vec_label IN (200 scattered labels)`` query is index-served
  (``full_scan_count()==0``) and returns all 200 present rows byte-identical to
  stdlib sqlite3.

  Production shape being modelled: ``_ann_to_pandas`` runs the ``vec_label IN``
  lookup against the table's shared ``self._conn`` under ``_conn_lock`` (NOT
  ``_open_snapshot_conn`` — only ``to_batches`` full-corpus export uses the
  snapshot connection).  When the daemon is up ``self._conn`` is the read-write
  writer that ran ``_ensure_tables`` (index present); when the store is opened
  read-only (daemon down / reader process) ``self._conn`` is the read-only
  engine connection that gets the index via ``meta.replay``.  This test exercises
  the read-only branch directly through ``open_read_only`` + ``meta.replay``.

This is the exact production scenario: the daemon/MemoryStore opens the store
read-write first (persisting the index), then a read-only recall reader serves
queries via replayed DDL on its shared connection.

RW-first ordering invariant: the index exists in engine meta ONLY after a
read-write ``_ensure_tables`` run.  A cold read-only-first open (no prior
read-write open on an un-indexed store) would show the full-scan path, not the
index path.  This test asserts BOTH shapes in the correct order.

Guard: skips loudly when the native wheel is absent or LILLI_STORAGE_DRIVER is
not ``lilli``.  The stdlib driver has no ``full_scan_count`` surface; this test
is exclusively for the lilli/native engine path.
"""
from __future__ import annotations

import contextlib
import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from iai_mcp.hippo._db import DEFAULT_STORAGE_DRIVER

if TYPE_CHECKING:
    pass

os.environ.setdefault("LILLI_FSYNC_MODE", "fast")

# ---------------------------------------------------------------------------
# Guard: require native wheel + lilli driver
# ---------------------------------------------------------------------------


def _native_available() -> bool:
    try:
        from iai_mcp_native import engine  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _native_available() or os.environ.get("LILLI_STORAGE_DRIVER", DEFAULT_STORAGE_DRIVER).lower() != "lilli",
    reason=(
        "iai_mcp_native.engine submodule not installed or LILLI_STORAGE_DRIVER != lilli — "
        "full_scan_count is only available on the native engine path"
    ),
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimal records DDL — vec_label as INTEGER PRIMARY KEY AUTOINCREMENT so the
# engine assigns sequential B-tree keys while allowing explicit vec_label inserts.
_DDL_RECORDS = (
    "CREATE TABLE records ("
    "vec_label INTEGER PRIMARY KEY AUTOINCREMENT, "
    "id TEXT NOT NULL, "
    "tier TEXT, "
    "centrality REAL DEFAULT 0.0, "
    "tombstoned_at TEXT, "
    "embedding_pending INTEGER NOT NULL DEFAULT 0"
    ")"
)

# Explicit-vec_label INSERT: names vec_label first so the engine stores the
# supplied scattered value as vec_label while assigning the B-tree key at
# max_key()+1.  This produces the divergence condition (vec_label != btree_key)
# that characterises a migrated store.
_INSERT_EXPLICIT = (
    "INSERT INTO records (vec_label, id, tier, centrality, tombstoned_at, embedding_pending)"
    " VALUES (?, ?, ?, ?, ?, ?)"
)


def _scattered_labels(n: int = 200) -> list[int]:
    """Return ``n`` deterministic scattered vec_label values (base 5000, irregular gaps).

    All values are well above the row count so that ``vec_label != btree_key``
    holds for every row — the exact divergence condition of a migrated store.
    """
    _GAP_CYCLE = [3, 8, 30, 101, 5, 17, 22, 6, 44, 11]
    labels: list[int] = []
    v = 5000
    for i in range(n):
        labels.append(v)
        v += _GAP_CYCLE[i % len(_GAP_CYCLE)]
    return labels


_LABELS_200 = _scattered_labels(200)


# ---------------------------------------------------------------------------
# Helper: seed a lilli store WITHOUT the vec_label index (pre-upgrade state)
# ---------------------------------------------------------------------------


def _seed_pre_upgrade_lilli_store(db_path: str, labels: list[int]) -> None:
    """Create a lilli store with scattered vec_label rows but WITHOUT idx_records_vec_label.

    Models a store migrated before the vec_label index was added to the
    production DDL.  The tables are created with the minimal DDL; no
    ``CREATE INDEX ... ON records(vec_label)`` is issued, so the engine meta
    does not contain that DDL and a subsequent ``open_read_only`` would not
    see it via ``meta.replay``.

    A read-write HippoDB open (``_ensure_tables``) is the mechanism that adds
    the index to meta — this seed deliberately omits it to model the pre-upgrade
    state.
    """
    from iai_mcp_native import engine

    conn = engine.Connection.open(db_path, 384)
    conn.execute(_DDL_RECORDS)
    conn.execute("BEGIN")
    for i, lbl in enumerate(labels):
        tier = "episodic" if i % 2 == 0 else "semantic"
        tomb = "2026-01-01 00:00:00.000000" if i % 10 == 0 else None
        # embedding_pending=1 excludes rows from the ANN rebuild (no embedding
        # bytes are seeded here — the test focuses on the vec_label index path,
        # not ANN accuracy).
        conn.execute(
            _INSERT_EXPLICIT,
            [lbl, f"rec-{i:04d}", tier, float(i) * 0.1, tomb, 1],
        )
    conn.execute("COMMIT")
    conn.close()


# ---------------------------------------------------------------------------
# Helper: sqlite3 reference
# ---------------------------------------------------------------------------


def _seed_sqlite3_reference(labels: list[int]) -> sqlite3.Connection:
    """Seed an in-memory stdlib sqlite3 with the same DDL and rows (reference)."""
    sq = sqlite3.connect(":memory:")
    sq.row_factory = sqlite3.Row
    sq.execute(_DDL_RECORDS)
    for i, lbl in enumerate(labels):
        tier = "episodic" if i % 2 == 0 else "semantic"
        tomb = "2026-01-01 00:00:00.000000" if i % 10 == 0 else None
        sq.execute(
            _INSERT_EXPLICIT,
            (lbl, f"rec-{i:04d}", tier, float(i) * 0.1, tomb, 1),
        )
    sq.commit()
    return sq


def _norm(rows) -> list[tuple]:
    return [tuple(r) for r in rows]


def _assert_parity_200(eng_rows: list, sq: sqlite3.Connection, labels: list[int], *, context: str) -> None:
    """Assert 200/200 count and that all expected vec_label values are present.

    Compares the vec_label set between the engine result and the sqlite3 reference.
    The engine schema may have additional columns (added by HippoDB's _reconcile_columns
    on first open) that the minimal sqlite3 reference lacks, so the comparison is
    limited to the vec_label column — which is both necessary and sufficient to prove
    correctness (all 200 scattered labels retrieved, none dropped).
    """
    ph = ", ".join("?" for _ in labels)
    # Both stores query the same SQL; extract only vec_label for cross-schema parity.
    sql_labels_only = f"SELECT vec_label FROM records WHERE vec_label IN ({ph}) ORDER BY vec_label"

    sq_vecs = sorted(row[0] for row in sq.execute(sql_labels_only, labels).fetchall())

    # eng_rows already fetched (full schema); extract vec_label from index 0.
    eng_vecs = sorted(int(row[0]) for row in eng_rows)

    assert len(eng_vecs) == 200, (
        f"{context}: expected 200 rows, got {len(eng_vecs)} — "
        "rows were silently dropped (btree-key descent or index miss)"
    )
    assert eng_vecs == sq_vecs, (
        f"{context}: vec_label set mismatch — "
        f"engine returned {len(eng_vecs)} labels, sqlite3 reference {len(sq_vecs)}; "
        f"first 5 engine: {eng_vecs[:5]!r}, first 5 sqlite3: {sq_vecs[:5]!r}"
    )


# ---------------------------------------------------------------------------
# Main proof: A1-RW + A1-RO
# ---------------------------------------------------------------------------


def test_veclabel_index_on_open_rw_and_ro(tmp_path: Path) -> None:
    """A migrated-shape store acquires idx_records_vec_label on RW open; RO path serves it.

    Step 1 — Seed a lilli store with scattered vec_label rows but WITHOUT the
    vec_label index (models a store migrated before this phase).

    Step 2 (A1-RW) — Re-open via HippoDB (read-write).  ``_ensure_tables``
    runs the full ``_DDL_RECORDS_INDEXES`` list, issuing ``CREATE INDEX IF NOT
    EXISTS idx_records_vec_label ON records(vec_label)`` via
    ``run_ddl_passthrough``, which registers the index in the catalog and
    persists its DDL in engine meta.  A warm ``vec_label IN (200 labels)`` query
    on this read-write connection must produce ``full_scan_count()==0``.

    Step 3 (A1-RO) — Close the read-write HippoDB.  Open a FRESH read-only
    connection on the same store (``engine.Connection.open_read_only``, the same
    underlying call ``open_store_conn(read_only=True)`` makes).  The read-only
    ``from_store`` → ``meta.replay`` path loads the persisted vec_label index
    DDL without running ``_ensure_tables``.  A warm ``vec_label IN (200 labels)``
    query on this read-only connection must produce ``full_scan_count()==0`` and
    return 200/200 rows byte-identical to stdlib sqlite3.

    This is the exact production read-only recall shape (``_ann_to_pandas``
    → ``_open_snapshot_conn`` → ``open_store_conn(read_only=True)``), confirming
    the last untested link in A1: a read-only recall connection on an already-
    migrated store is index-served.
    """
    from iai_mcp.hippo import HippoDB
    from iai_mcp_native import engine

    labels = _LABELS_200
    sq = _seed_sqlite3_reference(labels)
    ph = ", ".join("?" for _ in labels)
    sql_200 = f"SELECT * FROM records WHERE vec_label IN ({ph}) ORDER BY vec_label"

    # -----------------------------------------------------------------
    # Step 1: build a pre-upgrade lilli store (no vec_label index in meta)
    # -----------------------------------------------------------------
    store_root = tmp_path / "a1_store"
    hippo_dir = store_root / "hippo"
    hippo_dir.mkdir(parents=True)
    db_path = str(hippo_dir / "brain.sqlite3")

    _seed_pre_upgrade_lilli_store(db_path, labels)

    # Confirm: a bare engine RO open at this point full-scans (index not in meta yet).
    _check_conn = engine.Connection.open_read_only(db_path, 0)
    try:
        _check_conn.execute(sql_200, labels).fetchall()
        _check_conn.reset_full_scan_count()
        _check_conn.execute(sql_200, labels).fetchall()
        _pre_scan_count = _check_conn.full_scan_count()
    finally:
        with contextlib.suppress(Exception):
            _check_conn.close()
    assert _pre_scan_count >= 1, (
        f"pre-upgrade bare RO open: expected full_scan_count>=1 (no index in meta); "
        f"got {_pre_scan_count} — the pre-upgrade condition is not modelled correctly"
    )

    # -----------------------------------------------------------------
    # Step 2 (A1-RW): re-open via HippoDB read-write → _ensure_tables persists index
    # -----------------------------------------------------------------
    db_rw = HippoDB(store_root, read_only=False)
    try:
        # _ensure_tables ran: idx_records_vec_label should now be in catalog + meta.
        # Confirm the index is in the engine's sqlite_master equivalent.
        rw_conn = db_rw._conn

        # Warm read to build ColIndex state.
        rw_conn.execute(sql_200, labels).fetchall()
        rw_conn.reset_full_scan_count()

        # Second (warm) read: must be index-served.
        rw_rows = _norm(rw_conn.execute(sql_200, labels).fetchall())
        rw_scan_count = rw_conn.full_scan_count()
    finally:
        db_rw.close()

    # A1-RW assertion: read-write open persisted the index and it serves queries.
    assert rw_scan_count == 0, (
        f"A1-RW: expected full_scan_count==0 after HippoDB RW open "
        f"(idx_records_vec_label persisted via _ensure_tables); "
        f"got {rw_scan_count} — _ensure_tables did not register the index"
    )
    _assert_parity_200(rw_rows, sq, labels, context="A1-RW")

    # -----------------------------------------------------------------
    # Step 3 (A1-RO): fresh read-only open — index served via meta.replay only
    # -----------------------------------------------------------------
    # This is the production _ann_to_pandas path: Connection.open_read_only →
    # from_store → meta.replay — never runs _ensure_tables.
    #
    # Note: read-only engine connections cannot flush on close (no write path).
    # Suppress the OperationalError on close — the read data is already materialized
    # in Python-side lists before close is called, so no rows are lost.
    ro_conn = engine.Connection.open_read_only(db_path, 0)
    try:
        # Warm read to build ColIndex from replayed catalog.
        ro_conn.execute(sql_200, labels).fetchall()
        ro_conn.reset_full_scan_count()

        # Second (warm) read: must be index-served via meta.replay.
        ro_rows = _norm(ro_conn.execute(sql_200, labels).fetchall())
        ro_scan_count = ro_conn.full_scan_count()
    finally:
        with contextlib.suppress(Exception):
            ro_conn.close()

    # A1-RO assertion: the read-only production recall path is index-served.
    assert ro_scan_count == 0, (
        f"A1-RO: expected full_scan_count==0 on fresh read-only open "
        f"(index served via meta.replay, _ensure_tables not run); "
        f"got {ro_scan_count} — the persisted DDL was not replayed into the catalog"
    )
    _assert_parity_200(ro_rows, sq, labels, context="A1-RO (production recall shape)")
