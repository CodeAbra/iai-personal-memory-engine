"""Episodic capture de-duplication.

Tombstones duplicate episodic records produced by the capture_turn()
check-then-insert race (fixed by _CAPTURE_DEDUP_LOCK in capture.py):
concurrent daemon RPC threads draining overlapping Stop-hook full-transcript
replays could each see a tag as absent and insert their own copy before the
fix landed. This is the one-off cleanup for records inserted before then.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from iai_mcp.events import write_event
from iai_mcp.store import MemoryStore

log = logging.getLogger(__name__)


def migrate_dedupe_episodic_captures(
    store: "MemoryStore",
    *,
    dry_run: bool = False,
) -> dict:
    """Tombstone duplicate episodic records that share an idem-tag.

    Groups non-tombstoned episodic records by their ``idem:<hash>`` tag --
    the same identity key capture_turn() checks before inserting -- and
    keeps exactly one record per group. Duplicates are soft-deleted via
    ``tombstoned_at`` (the same column the erasure-agent sleep step uses),
    never hard-deleted: literal_surface, provenance, and embeddings are
    left untouched on every record.

    Safe to call multiple times (idempotent): once a group has one survivor,
    re-running finds no group with more than one non-tombstoned member.
    """
    from iai_mcp.hippo import HippoDB

    db = store.db
    if not isinstance(db, HippoDB):
        return {"groups": 0, "tombstoned": 0, "dry_run": dry_run}

    with db._conn_lock:
        rows = db._conn.execute(
            "SELECT id FROM records"
            " WHERE tier = 'episodic' AND tombstoned_at IS NULL"
        ).fetchall()
    record_ids = [row[0] for row in rows]

    groups: dict[str, list[str]] = {}
    for rid_str in record_ids:
        try:
            rec = store.get(UUID(rid_str))
        except (ValueError, TypeError):
            continue
        if rec is None:
            continue
        idem_tag = next(
            (t for t in (rec.tags or []) if t.startswith("idem:")), None
        )
        if idem_tag is None:
            continue
        groups.setdefault(idem_tag, []).append(rid_str)

    dup_groups = {tag: ids for tag, ids in groups.items() if len(ids) > 1}

    to_tombstone: list[str] = []
    for ids in dup_groups.values():
        # Keep the lexicographically-smallest id as the canonical survivor --
        # deterministic and idempotent across re-runs. created_at is
        # identical within a group (they all resolve to the same true
        # transcript turn), so it can't break ties meaningfully.
        ids_sorted = sorted(ids)
        to_tombstone.extend(ids_sorted[1:])

    if not dry_run and to_tombstone:
        now = datetime.now(timezone.utc)
        with db._conn_lock:
            db._conn.executemany(
                "UPDATE records SET tombstoned_at = ?, live = 0 WHERE id = ?",
                [(now, rid) for rid in to_tombstone],
            )
        # The tombstone pass moved rows out of the active set: the active count
        # decreased.  Invalidate the cached active count so the next
        # active_records_count() recomputes from the post-tombstone SQL state.
        # Dedupe does not cascade to edges here, so only the active key is cleared.
        # Best-effort guarded: a store without the cache (or a broken cache) must
        # never fail the migration.
        try:
            _inv = getattr(store, "_invalidate_corpus_count", None)
            if _inv is not None:
                _inv("active")
        except Exception:  # noqa: BLE001 -- invalidation must not crash migration
            pass
        # Invalidate the resident exact-cosine matrix so a just-tombstoned
        # duplicate never lingers as a recallable id — the same discipline the
        # recency-buffer eviction below applies for its own tier.
        try:
            _inv_x = getattr(store, "invalidate_exact_index", None)
            if callable(_inv_x):
                _inv_x()
        except Exception:  # noqa: BLE001 -- invalidation must not crash migration
            pass
        # Evict the just-tombstoned ids from the recency buffer so a dead row
        # never lingers as a stale marker until the next boot rebuild — the SQL
        # authority filters tombstoned rows, so the buffer must too for them to
        # stay byte-identical. Best-effort: a missing buffer or absent id is a
        # no-op and must never fail the migration.
        buf = getattr(store, "_recency_buffer", None)
        if buf is not None:
            for rid in to_tombstone:
                try:
                    buf.evict(str(rid))
                except Exception as exc:  # noqa: BLE001 -- eviction is best-effort
                    log.debug("recency buffer evict failed for %s: %s", rid, exc)
        try:
            write_event(
                store,
                "migration_dedupe_episodic_captures",
                {"groups": len(dup_groups), "tombstoned": len(to_tombstone)},
            )
        except (OSError, ValueError, RuntimeError) as exc:
            log.error(
                "migration_dedupe_episodic_captures event write failed: %s", exc
            )

    return {
        "groups": len(dup_groups),
        "tombstoned": len(to_tombstone),
        "dry_run": dry_run,
    }
