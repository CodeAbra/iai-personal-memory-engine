"""Backfill migrator for the ``role`` column on the ``records`` table.

Idempotent: only updates rows where ``role IS NULL``. The column and its
index are added by ``_ensure_tables`` / ``_reconcile_columns`` on store open,
so the ``ALTER TABLE`` guard here is a belt-and-suspenders safety net for
stores opened before that path ran.
"""
from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iai_mcp.store import MemoryStore

log = logging.getLogger(__name__)


def _derive_role_from_tags_json(tags_json: str | None) -> str | None:
    """Extract the conversational role suffix from a JSON-encoded tags list.

    Returns the verbatim suffix after the first ``role:`` prefix found, or
    None when the input is empty, unparseable, or carries no role tag. This
    is the exact same logic as ``_derive_role`` in the store layer, kept
    here as a private copy so the migrator has no import-time dependency on
    the store internals.
    """
    if not tags_json:
        return None
    try:
        tags = json.loads(tags_json)
    except Exception:  # noqa: BLE001
        return None
    for t in (tags or []):
        if isinstance(t, str) and t.startswith("role:"):
            return t[len("role:"):]
    return None


# Persisted idempotency stamp written to ``_hippo_meta`` once the backfill has
# run to completion. A keyed lookup of this row is O(1) (point read on the meta
# table), so a post-backfill open never touches ``records`` again. The marker is
# written inside the same connection/transaction as the final backfill commit so
# a crash mid-backfill leaves it unset and the next open re-runs the backfill.
_BACKFILL_MARKER_KEY = "role_backfill_done"


def _backfill_marker_set(db) -> bool:
    """Return True when the one-time role backfill stamp is present.

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


def migrate_role_column(
    store: "MemoryStore",
    dry_run: bool = False,
) -> dict:
    """Backfill the ``role`` column from ``tags_json`` for every unset row.

    The column is added by ``_ensure_tables`` on store open. This migrator
    only fills in the value for rows that pre-date the write-path change
    (``role IS NULL``). Rows written after the change already carry the
    correct value and are skipped.

    Gated by a persisted ``_hippo_meta`` stamp: once the backfill has run to
    completion the stamp is set and every subsequent open short-circuits on a
    single keyed meta lookup — no scan of ``records``. The stamp is written in
    the same transaction as the final backfill commit, so a crash mid-backfill
    leaves it unset and the next open re-runs the (idempotent) backfill.

    Returns a dict with ``updated``, ``skipped``, ``duration_ms``, and
    ``dry_run`` keys. An empty store returns ``updated=0, skipped=0``.
    """
    from iai_mcp.hippo import HippoDB
    from iai_mcp.events import write_event

    db = store.db
    if not isinstance(db, HippoDB):
        return {
            "dry_run": dry_run,
            "updated": 0,
            "skipped": 0,
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
            "skipped": 0,
            "duration_ms": 0.0,
            "note": "already migrated",
        }

    t0 = time.time()

    # Belt-and-suspenders: ensure the column exists for stores opened before
    # the canonical add-path (_reconcile_columns) ran. The existence check is
    # mandatory, not an optimization: a bare ALTER ADD COLUMN is only rejected
    # as a duplicate by backends that diff the live schema. Backends that apply
    # the column blindly would append a SECOND ``role`` column, desyncing the
    # row codec (column count no longer matches stored row width). PRAGMA
    # table_info is the portable existence probe across every backend.
    with db._conn_lock:
        pragma_rows = db._conn.execute("PRAGMA table_info(records)").fetchall()
        existing_cols = {
            (r["name"] if hasattr(r, "keys") else r[1]) for r in pragma_rows
        }
        if "role" not in existing_cols:
            try:
                db._conn.execute("ALTER TABLE records ADD COLUMN role TEXT")
            except Exception as exc:  # noqa: BLE001
                if "duplicate column name" not in str(exc).lower():
                    log.warning("unexpected error adding role column: %s", exc)

    # Fetch every row where role IS NULL and tags_json is non-empty.
    with db._conn_lock:
        pending = db._conn.execute(
            "SELECT id, tags_json FROM records WHERE role IS NULL AND tags_json IS NOT NULL"
        ).fetchall()

    updated = 0
    skipped = 0

    for row in pending:
        row_id = row["id"] if hasattr(row, "keys") else row[0]
        tags_json_str = row["tags_json"] if hasattr(row, "keys") else row[1]
        role_val = _derive_role_from_tags_json(tags_json_str)

        if role_val is None:
            # No role tag — leave role as NULL; not counted as skipped since
            # NULL is the correct value for records without a conversational role.
            continue

        if dry_run:
            updated += 1
            continue

        with db._conn_lock:
            db._conn.execute(
                "UPDATE records SET role = ? WHERE id = ?",
                (role_val, row_id),
            )

        updated += 1

    if not dry_run:
        # The backfill UPDATEs and the completion stamp commit together: the
        # stamp is durable only if the backfill it certifies is durable. A crash
        # before this commit leaves the stamp unset → next open re-runs.
        with db._conn_lock:
            db._conn.execute(
                "INSERT OR REPLACE INTO _hippo_meta (key, value) VALUES (?, ?)",
                (_BACKFILL_MARKER_KEY, "1"),
            )
            db._conn.commit()

    # Count of rows now carrying a non-NULL role — reported once, on the run
    # that actually performs the backfill (subsequent opens take the fast path).
    with db._conn_lock:
        already_set_count = db._conn.execute(
            "SELECT COUNT(*) FROM records WHERE role IS NOT NULL"
        ).fetchone()
    skipped = (already_set_count[0] if already_set_count else 0)

    duration_ms = (time.time() - t0) * 1000.0

    if not dry_run and updated > 0:
        try:
            write_event(
                store,
                kind="migration_role_column_backfill",
                data={
                    "updated": updated,
                    "skipped": skipped,
                    "duration_ms": duration_ms,
                },
                severity="info",
            )
        except Exception:  # noqa: BLE001 — telemetry must not abort migration
            pass

    return {
        "dry_run": dry_run,
        "updated": updated,
        "skipped": skipped,
        "duration_ms": duration_ms,
    }
