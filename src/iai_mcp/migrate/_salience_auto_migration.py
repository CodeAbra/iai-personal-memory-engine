"""Marker-gated one-time activation of ``backfill_salience`` from the sleep
pipeline.

Not a second producer: the payload is the unmodified, already-shipped
``backfill_salience``. This module adds only the provenance gate (a
born-native ``_hippo_meta`` stamp, same pattern as ``_live_flag_backfill.py``)
and a bounded attempt counter that caps the disk cost of the backfill's
unconditional pre-write snapshot on a persistently-failing store.

The native stamp is re-evaluated on every store open, not written once at
construction: it checks whether ANY row (live or tombstoned) currently
exists in ``records``, which is a monotone-safe existence probe, not a
birth-time fact.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from iai_mcp.store import MemoryStore

logger = logging.getLogger(__name__)


_DONE_MARKER_KEY = "salience_auto_migration_done"
_NATIVE_KEY = "salience_auto_migration_native"
_ATTEMPTS_KEY = "salience_auto_migration_attempts"
_LAST_ERROR_KEY = "salience_auto_migration_last_error"

_SNAPSHOT_GLOB = "hippo-pre-salience-backfill-*"

MAX_ATTEMPTS = 5


def _marker_present(db: Any, key: str) -> bool:
    """O(1) keyed ``_hippo_meta`` read -- mirrors ``_backfill_marker_set``."""
    with db._conn_lock:
        row = db._conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = ?",
            (key,),
        ).fetchone()
    return row is not None


def _read_attempts(db: Any) -> int:
    with db._conn_lock:
        row = db._conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = ?",
            (_ATTEMPTS_KEY,),
        ).fetchone()
    if row is None:
        return 0
    value = row["value"] if hasattr(row, "keys") else row[0]
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _write_meta(db: Any, key: str, value: str) -> None:
    with db._conn_lock:
        db._conn.execute(
            "INSERT OR REPLACE INTO _hippo_meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        db._conn.commit()


def _clear_meta(db: Any, key: str) -> None:
    with db._conn_lock:
        db._conn.execute("DELETE FROM _hippo_meta WHERE key = ?", (key,))
        db._conn.commit()


def resolve_migration_state(
    *, done: bool, native: bool, attempts: int, max_attempts: int,
) -> str:
    """Single precedence for the done/native/gave-up/pending states -- every
    caller that decides what to report or whether to write a marker MUST go
    through this, never re-derive the done-vs-native order independently.
    ``done`` and ``native`` are meant to be mutually exclusive (both mean
    "no legacy migration needed," differing only in why); done wins if a
    bug ever leaves both set, so a completed run is never hidden behind a
    stale native stamp."""
    if done:
        return "done"
    if native:
        return "native"
    if attempts >= max_attempts:
        return "gave_up"
    return "pending"


def _combined_backfill_errors(summary: dict) -> list:
    """Union of ``summary["errors"]`` and ``summary["entity_backfill"]["errors"]``
    -- every caller deciding whether a backfill run is "done" MUST use this,
    never ``summary["errors"]`` alone, or the auto and manual gates diverge."""
    entity_errors = (summary.get("entity_backfill") or {}).get("errors") or []
    return list(summary.get("errors") or []) + list(entity_errors)


def _list_snapshot_dirs(root: Path) -> set[str]:
    """Raises ``OSError`` on a genuine enumeration failure -- callers decide
    fail-open vs fail-closed; this helper does neither."""
    return {p.name for p in root.glob(_SNAPSHOT_GLOB) if p.is_dir()}


def _cleanup_new_snapshot_dirs(root: Path, before: set[str] | None) -> None:
    """Remove only the snapshot directories that appeared during THIS call.

    A pre-existing (manual) snapshot is never touched, and a snapshot from a
    backfill that already succeeded is never passed here -- that path
    returns before reaching the caller's cleanup. ``before is None`` means
    the pre-run enumeration itself failed and what existed beforehand is
    unknown -- cleanup is skipped entirely rather than risk treating every
    existing snapshot as newly created.
    """
    if before is None:
        return
    try:
        after = _list_snapshot_dirs(root)
    except OSError:
        return
    for name in after - before:
        shutil.rmtree(root / name, ignore_errors=True)


def stamp_native_if_fresh(store: "MemoryStore") -> None:
    """Stamp a store that has never held any record (live or tombstoned) as
    native.

    Native means this store was created under a binary where capture-time
    salience classification is already active, so it will never carry
    legacy unclassified rows. Provenance only, never a classification-value
    probe: a store that has ever held a row predates the producer (this
    stamp did not exist when the row was written) and is left un-stamped so
    the one-time backfill still runs for it. The probe checks for ANY row,
    not just live ones, so a row that is transiently tombstoned and later
    restored (e.g. ``compact_hippo``'s pinned/never-decay untombstone) is
    never missed by a reopen landing in the tombstoned window.
    """
    from iai_mcp.hippo import HippoDB

    db = store.db
    if not isinstance(db, HippoDB):
        return

    if _marker_present(db, _DONE_MARKER_KEY) or _marker_present(db, _NATIVE_KEY):
        return

    with db._conn_lock:
        row = db._conn.execute(
            "SELECT 1 FROM records LIMIT 1",
        ).fetchone()
    if row is not None:
        return

    _write_meta(db, _NATIVE_KEY, "1")


def mark_migration_done(store: "MemoryStore") -> None:
    """Shared success marker for both the sleep-step auto path and the
    manual operator ``--apply`` path, so the two converge on one stamp and
    neither re-triggers the other's work."""
    from iai_mcp.hippo import HippoDB

    db = store.db
    if not isinstance(db, HippoDB):
        return
    _write_meta(db, _DONE_MARKER_KEY, "1")


def run_salience_auto_migration(store: "MemoryStore") -> dict:
    """Gate order: done marker -> born-native stamp -> attempts ceiling ->
    the actual backfill.

    The gate consults ONLY the two provenance/marker keys, never
    ``salience_level`` -- that classification value cannot distinguish a
    never-migrated legacy row from a legitimately-unflagged modern one.
    """
    from iai_mcp.hippo import HippoDB
    from iai_mcp.migrate._salience_backfill import backfill_salience

    db = store.db
    if not isinstance(db, HippoDB):
        return {"status": "skipped", "note": "non-hippo backend"}

    attempts = _read_attempts(db)
    state = resolve_migration_state(
        done=_marker_present(db, _DONE_MARKER_KEY),
        native=_marker_present(db, _NATIVE_KEY),
        attempts=attempts,
        max_attempts=MAX_ATTEMPTS,
    )
    if state == "done":
        return {"status": "already_migrated"}
    if state == "native":
        return {"status": "not_needed"}
    if state == "gave_up":
        return {"status": "gave_up", "attempts": attempts}

    # Opposite-ordering rule for the two keys: the failure counter is
    # written BEFORE the work (must survive a hard crash mid-snapshot, so a
    # persistently-failing store still converges on MAX_ATTEMPTS instead of
    # re-snapshotting forever); the success marker is written AFTER the work
    # (must NOT survive an incomplete run, so a crash always leaves the next
    # nightly cycle free to retry the idempotent backfill).
    _write_meta(db, _ATTEMPTS_KEY, str(attempts + 1))

    root = Path(store.root)
    try:
        before = _list_snapshot_dirs(root)
    except OSError:
        before = None  # None means unknown; never substitute an empty set here.

    try:
        summary = backfill_salience(store, apply=True, store_path=store.root)
    except Exception as exc:  # noqa: BLE001 -- one failed attempt must not abort the cycle
        _write_meta(db, _LAST_ERROR_KEY, f"{type(exc).__name__}: {exc}")
        # Scoped strictly to THIS call's own failure.
        _cleanup_new_snapshot_dirs(root, before)
        return {"status": "failed", "error": str(exc)}

    errors = _combined_backfill_errors(summary)
    if errors:
        # Snapshot is the rollback point for a partial success -- never cleaned up here.
        _write_meta(
            db,
            _LAST_ERROR_KEY,
            f"{len(errors)} record(s) failed during backfill: {errors[0]}",
        )
        return {
            "status": "failed",
            "error": f"{len(errors)} record(s) failed during backfill",
            **summary,
        }

    mark_migration_done(store)
    _clear_meta(db, _ATTEMPTS_KEY)
    _clear_meta(db, _LAST_ERROR_KEY)
    return {"status": "migrated", **summary}
