"""One-time backfill migrator for the ``record_tags`` index table.

``record_tags`` is created by ``HippoDB._ensure_tables`` on every store open
(``CREATE TABLE IF NOT EXISTS`` — always present after this phase). This
migrator only populates it for a store that existed BEFORE the table did: it
scans ``records.tags_json`` once and upserts every tag into ``record_tags``.

Idempotent: gated by a persisted ``_hippo_meta`` stamp, mirroring
``migrate_role_column``'s backfill-marker pattern. A crash mid-backfill
leaves the stamp unset, so the next open re-runs the (idempotent, merge-insert
semantics) backfill rather than silently leaving the index half-populated.

A backfill failure logs and leaves ``find_record_by_tag``'s LIKE-scan fallback
gate set, rather than raising into ``MemoryStore.__init__`` and breaking
store open.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iai_mcp.store import MemoryStore

log = logging.getLogger(__name__)

_BACKFILL_MARKER_KEY = "record_tags_backfill_done"


def _backfill_marker_set(db) -> bool:
    """Return True when the one-time record_tags backfill stamp is present.

    A single keyed read on ``_hippo_meta`` — never scans ``records`` or
    ``record_tags``. O(1) regardless of corpus size.
    """
    with db._conn_lock:
        row = db._conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = ?",
            (_BACKFILL_MARKER_KEY,),
        ).fetchone()
    return row is not None


def migrate_record_tags_backfill(store: "MemoryStore") -> dict:
    """Backfill ``record_tags`` from ``records.tags_json`` exactly once.

    Runs at ``MemoryStore`` construction, before the daemon serves sockets.
    Gated by a persisted ``_hippo_meta`` stamp so a post-backfill open pays a
    single keyed meta lookup, never a scan of ``records``.

    On success, returns ``{"ok": True, "upserted": N, "duration_ms": ...}``.
    On failure, the exception is caught, logged, and
    ``{"ok": False, "error": ...}`` is returned — the caller keeps the
    LIKE-scan fallback active for ``find_record_by_tag`` rather than treating
    this as fatal to store open.
    """
    from iai_mcp.hippo import HippoDB
    from iai_mcp.hippo._table import _upsert_record_tags

    db = store.db
    if not isinstance(db, HippoDB):
        return {"ok": True, "upserted": 0, "duration_ms": 0.0, "note": "non-hippo backend; skipped"}

    if _backfill_marker_set(db):
        return {"ok": True, "upserted": 0, "duration_ms": 0.0, "note": "already migrated"}

    t0 = time.time()
    try:
        with db._conn_lock:
            # record_tags is created by _ensure_tables before this migrator ever
            # runs (both are part of MemoryStore.__init__'s boot sequence), but
            # an empty-vs-nonempty check keeps this migrator a true no-op on a
            # store that already has a populated table from normal writes (e.g.
            # a store opened once on a pre-backfill binary, written to, then
            # reopened -- record_tags already reflects those writes and a
            # redundant full backfill would be wasted work, though harmless
            # since the upsert is idempotent).
            existing = db._conn.execute("SELECT 1 FROM record_tags LIMIT 1").fetchone()
            if existing is not None:
                db._conn.execute(
                    "INSERT OR REPLACE INTO _hippo_meta (key, value) VALUES (?, ?)",
                    (_BACKFILL_MARKER_KEY, "1"),
                )
                db._conn.commit()
                return {
                    "ok": True,
                    "upserted": 0,
                    "duration_ms": (time.time() - t0) * 1000.0,
                    "note": "record_tags already populated",
                }

            rows = db._conn.execute(
                "SELECT id, tags_json FROM records WHERE tags_json IS NOT NULL"
            ).fetchall()

            upserted = 0
            for row in rows:
                row_id = row["id"] if hasattr(row, "keys") else row[0]
                tags_json_str = row["tags_json"] if hasattr(row, "keys") else row[1]
                if not tags_json_str:
                    continue
                _upsert_record_tags(db._conn, str(row_id), tags_json_str)
                upserted += 1

            db._conn.execute(
                "INSERT OR REPLACE INTO _hippo_meta (key, value) VALUES (?, ?)",
                (_BACKFILL_MARKER_KEY, "1"),
            )
            db._conn.commit()

        duration_ms = (time.time() - t0) * 1000.0
        if upserted > 0:
            try:
                from iai_mcp.events import write_event
                write_event(
                    store,
                    kind="migration_record_tags_backfill",
                    data={"upserted": upserted, "duration_ms": duration_ms},
                    severity="info",
                )
            except Exception:  # noqa: BLE001 -- telemetry must not abort migration
                pass
        return {"ok": True, "upserted": upserted, "duration_ms": duration_ms}
    except Exception as exc:  # noqa: BLE001 -- backfill failure must not break store open
        log.warning("record_tags backfill failed (non-fatal, LIKE-scan fallback active): %s", exc)
        return {"ok": False, "error": str(exc), "duration_ms": (time.time() - t0) * 1000.0}
