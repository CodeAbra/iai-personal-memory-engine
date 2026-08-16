"""Backfill migrator for the ``live`` column on the ``records`` table.

Idempotent: only updates rows where ``live IS NULL``. The column and its
index are added by ``_ensure_tables`` / ``_reconcile_columns`` on store open,
so the ``ALTER TABLE`` guard here is a belt-and-suspenders safety net for
stores opened before that path ran.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iai_mcp.store import MemoryStore

log = logging.getLogger(__name__)


# Persisted idempotency stamp written to ``_hippo_meta`` once the backfill has
# run to completion. A keyed lookup of this row is O(1) (point read on the meta
# table), so a post-backfill open never touches ``records`` again. The marker is
# written inside the same connection/transaction as the final backfill commit so
# a crash mid-backfill leaves it unset and the next open re-runs the backfill.
_BACKFILL_MARKER_KEY = "live_flag_backfill_done"

# Separate stamp for the drift-reconciliation pass below. Distinct from
# ``_BACKFILL_MARKER_KEY`` because the two passes cover disjoint row sets:
# the backfill only ever touches ``live IS NULL`` rows, while reconciliation
# corrects rows that already carry a (wrong) ``live`` value.
_RECONCILE_MARKER_KEY = "live_flag_reconcile_done"


def _backfill_marker_set(db) -> bool:
    """Return True when the one-time live-flag backfill stamp is present.

    A single keyed read on ``_hippo_meta`` — never scans ``records``. The
    column predicate ``key = ?`` rides the meta table's primary-key access,
    so this stays O(1) regardless of corpus size.
    """
    with db._conn_lock:
        row = db._conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = ?",
            (_BACKFILL_MARKER_KEY,),
        ).fetchone()
    return row is not None


def migrate_live_flag_backfill(
    store: "MemoryStore",
    dry_run: bool = False,
) -> dict:
    """Backfill the ``live`` column from ``tombstoned_at`` for every unset row.

    The column is added by ``_ensure_tables`` on store open. This migrator
    only fills in rows with ``live IS NULL``: ``live = 1`` when
    ``tombstoned_at`` is unset, else ``live = 0``. Rows that already carry a
    correct ``live`` value are skipped. A fresh-migrated store's rows land
    with ``live`` NULL (the copied ``tombstoned_at`` predates the column), so
    this backfill runs on the FIRST open of a fresh-migrated store — the
    boot-migration sequence is unconditional (no stamp short-circuit before
    the first run), so the live-rows recall query rides the index from the
    first recall.

    Gated by a persisted ``_hippo_meta`` stamp: once the backfill has run to
    completion the stamp is set and every subsequent open short-circuits on a
    single keyed meta lookup — no scan of ``records``. The stamp is written in
    the same transaction as the final backfill commit, so a crash mid-backfill
    leaves it unset and the next open re-runs the (idempotent) backfill.

    Returns a dict with ``updated``, ``duration_ms``, and ``dry_run`` keys.
    An empty store, or a store whose rows are all already backfilled, returns
    ``updated=0``.
    """
    from iai_mcp.hippo import HippoDB
    from iai_mcp.events import write_event

    db = store.db
    if not isinstance(db, HippoDB):
        return {
            "dry_run": dry_run,
            "updated": 0,
            "duration_ms": 0.0,
            "note": "non-hippo backend; skipped",
        }

    # Fast path: the persisted stamp means the backfill already completed. A
    # single keyed meta lookup — no ``records`` touch, no scan. A dry-run still
    # falls through so callers can probe the would-update count on demand.
    if not dry_run and _backfill_marker_set(db):
        return {
            "dry_run": False,
            "updated": 0,
            "duration_ms": 0.0,
            "note": "already migrated",
        }

    t0 = time.time()

    # Belt-and-suspenders: ensure the column exists for stores opened before
    # the canonical add-path (_reconcile_columns) ran. The existence check is
    # mandatory, not an optimization: a bare ALTER ADD COLUMN is only rejected
    # as a duplicate by backends that diff the live schema. Backends that apply
    # the column blindly would append a SECOND ``live`` column, desyncing the
    # row codec (column count no longer matches stored row width). PRAGMA
    # table_info is the portable existence probe across every backend.
    with db._conn_lock:
        pragma_rows = db._conn.execute("PRAGMA table_info(records)").fetchall()
        existing_cols = {
            (r["name"] if hasattr(r, "keys") else r[1]) for r in pragma_rows
        }
        if "live" not in existing_cols:
            try:
                db._conn.execute("ALTER TABLE records ADD COLUMN live INTEGER")
            except Exception as exc:  # noqa: BLE001
                if "duplicate column name" not in str(exc).lower():
                    log.warning("unexpected error adding live column: %s", exc)

    # Every row with an unset live flag is backfilled by deriving from its own
    # tombstoned_at value — a row currently tombstoned lands live=0, a live row
    # lands live=1. This is the exact same rule every write-path seam applies
    # at write time, so the backfilled column agrees with the
    # `tombstoned_at IS NULL` recall predicate for every row, byte-identically.
    with db._conn_lock:
        pending = db._conn.execute(
            "SELECT id, tombstoned_at FROM records WHERE live IS NULL"
        ).fetchall()

    updated = 0

    for row in pending:
        row_id = row["id"] if hasattr(row, "keys") else row[0]
        tombstoned_at = row["tombstoned_at"] if hasattr(row, "keys") else row[1]
        live_val = 0 if tombstoned_at else 1

        if dry_run:
            updated += 1
            continue

        with db._conn_lock:
            db._conn.execute(
                "UPDATE records SET live = ? WHERE id = ?",
                (live_val, row_id),
            )

        updated += 1

    if not dry_run:
        # Safe by statement ordering + idempotency, not by transactional
        # atomicity: each per-row UPDATE above already committed independently
        # under autocommit. The stamp below must remain the last statement —
        # a crash before it leaves the stamp unset, so the next open re-runs
        # the (idempotent) backfill against the remaining live IS NULL rows.
        with db._conn_lock:
            db._conn.execute(
                "INSERT OR REPLACE INTO _hippo_meta (key, value) VALUES (?, ?)",
                (_BACKFILL_MARKER_KEY, "1"),
            )
            db._conn.commit()

    duration_ms = (time.time() - t0) * 1000.0

    if not dry_run and updated > 0:
        try:
            write_event(
                store,
                kind="migration_live_flag_backfill",
                data={
                    "updated": updated,
                    "duration_ms": duration_ms,
                },
                severity="info",
            )
        except Exception:  # noqa: BLE001 — telemetry must not abort migration
            pass

    return {
        "dry_run": dry_run,
        "updated": updated,
        "duration_ms": duration_ms,
    }


def reconcile_live_flag_drift(
    store: "MemoryStore",
    dry_run: bool = False,
) -> dict:
    """Reconcile rows where ``live`` disagrees with ``tombstoned_at``.

    Distinct from ``migrate_live_flag_backfill``: the backfill only fills in
    rows with ``live IS NULL``. This pass corrects rows whose ``live`` value
    disagrees with their own ``tombstoned_at``. Two directional single-table
    UPDATEs (the engine rejects JOIN and subquery-IN):

    ``tombstoned_at IS NOT NULL AND live = 1`` -> ``live = 0``
    ``tombstoned_at IS NULL AND live = 0`` -> ``live = 1``

    Gated by its own persisted ``_hippo_meta`` stamp, written in the same
    transaction as the final UPDATE so a crash mid-pass leaves it unset and
    the next open re-runs.

    Returns a dict with ``updated``, ``duration_ms``, and ``dry_run`` keys.
    """
    from iai_mcp.hippo import HippoDB
    from iai_mcp.hippo._db import _txn
    from iai_mcp.events import write_event

    db = store.db
    if not isinstance(db, HippoDB):
        return {
            "dry_run": dry_run,
            "updated": 0,
            "duration_ms": 0.0,
            "note": "non-hippo backend; skipped",
        }

    # Fast path: the persisted stamp means reconciliation already completed.
    # A single keyed meta lookup -- no ``records`` touch, no scan. A dry-run
    # still falls through so callers can probe the would-update count.
    if not dry_run:
        with db._conn_lock:
            row = db._conn.execute(
                "SELECT value FROM _hippo_meta WHERE key = ?",
                (_RECONCILE_MARKER_KEY,),
            ).fetchone()
        if row is not None:
            return {
                "dry_run": False,
                "updated": 0,
                "duration_ms": 0.0,
                "note": "already reconciled",
            }

    t0 = time.time()

    # Counted before the UPDATEs run (rather than trusting engine-reported
    # rowcount, which is not uniformly available across drivers) so the
    # returned ``updated`` figure is exact on both drivers.
    with db._conn_lock:
        drifted_tombstoned = db._conn.execute(
            "SELECT COUNT(*) FROM records "
            "WHERE tombstoned_at IS NOT NULL AND live = 1"
        ).fetchone()
        drifted_live = db._conn.execute(
            "SELECT COUNT(*) FROM records "
            "WHERE tombstoned_at IS NULL AND live = 0"
        ).fetchone()
    updated = int(drifted_tombstoned[0]) + int(drifted_live[0])

    if not dry_run:
        # The two directional UPDATEs and the completion stamp run inside a
        # real transaction: the stamp is durable only if the reconciliation
        # it certifies is durable. A crash before COMMIT rolls back all three
        # statements, leaving the stamp unset -> next open re-runs.
        with db._conn_lock, _txn(db._conn):
            db._conn.execute(
                "UPDATE records SET live = 0 "
                "WHERE tombstoned_at IS NOT NULL AND live = 1"
            )
            db._conn.execute(
                "UPDATE records SET live = 1 "
                "WHERE tombstoned_at IS NULL AND live = 0"
            )
            db._conn.execute(
                "INSERT OR REPLACE INTO _hippo_meta (key, value) VALUES (?, ?)",
                (_RECONCILE_MARKER_KEY, "1"),
            )

    duration_ms = (time.time() - t0) * 1000.0

    if not dry_run and updated > 0:
        try:
            write_event(
                store,
                kind="migration_live_flag_reconcile",
                data={
                    "updated": updated,
                    "duration_ms": duration_ms,
                },
                severity="info",
            )
        except Exception:  # noqa: BLE001 — telemetry must not abort migration
            pass

    return {
        "dry_run": dry_run,
        "updated": updated,
        "duration_ms": duration_ms,
    }
