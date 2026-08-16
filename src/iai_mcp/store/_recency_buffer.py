"""In-process recency buffer for recent role:user markers.

Holds a bounded, embedding-free snapshot of the most-recent role:user and
pending markers in volatile RAM so the recall pipeline can serve the recency
set without decoding the full corpus on every read.

The buffer is a cache/index — the store remains the lossless authority.  A
buffer miss degrades to the SQL path, never to data loss.  The buffer is never
serialized to disk.

Thread safety: all mutation is guarded by an internal ``threading.Lock``.
Snapshot-reads (``markers()``) copy the entry set under the lock then sort
outside it so they never block a concurrent write for longer than a dict copy.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator

__all__ = ["RecencyMarker", "RecencyBuffer"]

_MIN_DT = datetime.min.replace(tzinfo=timezone.utc)


def _as_utc(dt: datetime | None) -> datetime:
    """Normalize a datetime to tz-aware UTC for ordering.

    A naive ``datetime`` is treated as UTC (``replace(tzinfo=utc)``), NOT as
    host-local time. ``datetime.timestamp()`` on a naive value silently uses
    the host timezone, which would offset the buffer epoch relative to the
    warm/pending feeds (tz-aware UTC) and relative to the SQL authority (which
    normalizes naive → UTC in ``_from_row``). Normalizing here keeps the buffer
    ordering byte-identical to the authority and reproducible across machines.
    ``None`` maps to ``datetime.min`` UTC.
    """
    if dt is None:
        return _MIN_DT
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


@dataclass
class RecencyMarker:
    """Lightweight marker for a single role:user or pending record.

    Carries only the fields the recall pipeline consumer reads; the embedding
    BLOB and other heavy columns are intentionally excluded.

    ``source_rowid`` is the SQLite rowid of the records-table row this marker
    came from.  It is the tie-break authority that reproduces the SQL union's
    ``rowid DESC`` ordering when ``created_at`` values are equal — buffer push
    order is NOT a substitute (push order diverges from rowid order after a
    reembed or bulk-copy re-key).

    When a write-path feed cannot resolve the rowid immediately it stores ``-1``
    as a sentinel.  The next boot rebuild re-keys every entry from the true
    rowid, correcting any ``-1`` sentinels.
    """

    id: str
    literal_surface: str
    created_at: datetime | None
    session_id: str | None
    embedding_pending: int
    role: str | None
    source_rowid: int  # SQLite rowid; -1 = unresolved (will be corrected at boot-rebuild)
    tier: str = "episodic"  # mirrors the SQL authority's role-branch tier='episodic' predicate


class RecencyBuffer:
    """Bounded in-process buffer of recent role:user and pending markers.

    Internal storage is a ``dict[str, RecencyMarker]`` keyed by marker id so
    that push-by-id updates in place and ``reconcile_reembed`` flips the flag
    on the existing entry without duplicating it.

    Capacity policy: last-N by created_at.  When the dict exceeds ``maxlen``
    the entry with the oldest ``created_at`` is evicted.  This reproduces the
    ``ORDER BY created_at DESC LIMIT N`` semantics of the SQL read path.

    The buffer is never serialized.  No ``__reduce__``, pickle, or file I/O
    methods exist — by design.
    """

    def __init__(self, maxlen: int = 200) -> None:
        if maxlen < 1:
            raise ValueError(f"maxlen must be >= 1, got {maxlen}")
        self._maxlen = maxlen
        self._entries: dict[str, RecencyMarker] = {}
        self._lock = threading.Lock()
        self._warm = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_warm(self) -> bool:
        """True after a boot rebuild has populated the buffer from SQL."""
        return self._warm

    @is_warm.setter
    def is_warm(self, value: bool) -> None:
        self._warm = value

    def push(self, entry: RecencyMarker) -> None:
        """Insert or update an entry by id; evict oldest when over capacity.

        Updates in place if the id already exists (so an edited marker
        refreshes its buffered state without duplicating).  After inserting,
        if the dict exceeds ``maxlen``, drops the entry with the oldest
        ``created_at`` (``datetime.min`` sentinel for null timestamps).
        """
        with self._lock:
            self._entries[entry.id] = entry
            if len(self._entries) > self._maxlen:
                self._evict_oldest()

    def reconcile_reembed(self, entry_id: str) -> None:
        """Flip ``embedding_pending`` to 0 on the entry with this id.

        Called when ``reembed_pending_rows`` successfully embeds a row.  The
        entry moves from the pending branch to the role branch without being
        dropped or duplicated.  A no-op when the id is absent (the entry will
        be picked up on the next boot rebuild).
        """
        with self._lock:
            existing = self._entries.get(entry_id)
            if existing is not None:
                # Replace with a new dataclass instance (frozen-style update)
                self._entries[entry_id] = RecencyMarker(
                    id=existing.id,
                    literal_surface=existing.literal_surface,
                    created_at=existing.created_at,
                    session_id=existing.session_id,
                    embedding_pending=0,
                    role=existing.role,
                    source_rowid=existing.source_rowid,
                    tier=existing.tier,
                )

    def markers(self, n: int) -> list[RecencyMarker]:
        """Return up to n markers ordered to reproduce the SQL union order.

        Filters to entries where ``role == 'user'`` or ``embedding_pending == 1``
        (matching the SQL union predicate), then sorts by the exact composite
        key that reproduces the SQL union's stable-sort order:

          1. ``created_at DESC`` (primary — ``datetime.min`` UTC for null)
          2. ``branch_rank ASC`` (pending=0 sorts before role=1, matching the
             pending-branch-first insertion order into the SQL union's ``seen`` dict)
          3. ``source_rowid DESC`` (tie-break within a branch, mirrors the
             SQL ``ORDER BY rowid DESC`` within each branch query)

        The sort is implemented as a single key tuple with negated components
        for the descending dimensions:
        ``(-created_at_epoch, branch_rank, -source_rowid)``

        Push order is never used as a tie-break; rowid order is the authority.
        """
        with self._lock:
            snapshot = list(self._entries.values())

        # Eligibility mirrors the SQL authority's union exactly:
        #   pending branch: embedding_pending == 1 (tier-agnostic, role-agnostic)
        #   role branch:    role == 'user' AND tier == 'episodic'
        # A non-episodic role:user row is excluded here, matching the authority's
        # role-branch tier='episodic' predicate.
        eligible = [
            e for e in snapshot
            if e.embedding_pending == 1
            or (e.role == "user" and e.tier == "episodic")
        ]

        def _sort_key(r: RecencyMarker):
            # Normalize naive datetimes to UTC so the epoch matches the warm/
            # pending feeds and the SQL authority (which compares UTC datetimes);
            # a naive .timestamp() would silently use host-local time.
            epoch = _as_utc(r.created_at).timestamp()
            branch_rank = 0 if r.embedding_pending == 1 else 1
            return (-epoch, branch_rank, -r.source_rowid)

        eligible.sort(key=_sort_key)
        return eligible[:n]

    def evict(self, entry_id: str) -> None:
        """Drop the entry with this id, if present.

        Used when a row is soft-tombstoned after being fed: a dead row must
        not linger in the buffer until the next boot rebuild, or the buffer
        would diverge from the SQL authority (which filters tombstoned rows).
        A no-op when the id is absent.
        """
        with self._lock:
            self._entries.pop(entry_id, None)

    def replace_all(self, entries: list[RecencyMarker]) -> None:
        """Atomically swap the entire entry set under a single lock acquisition.

        Builds the new id-keyed map fully, then installs it and sets the warm
        flag in ONE lock hold — never clear-then-fill across separate
        acquisitions. This is what makes a concurrent warm race-free: a reader
        holding the lock sees either the complete old set or the complete new
        set, never an empty or half-filled intermediate.

        Honors the capacity bound: when the input exceeds ``maxlen`` the oldest
        entries (by ``created_at``) are evicted, matching ``push`` semantics.
        """
        with self._lock:
            self._entries = {}
            for e in entries:
                self._entries[e.id] = e
                if len(self._entries) > self._maxlen:
                    self._evict_oldest()
            self._warm = True

    def merge_warm(self, entries: list[RecencyMarker]) -> None:
        """Merge a SQL-warmed set into the buffer, preserving unflushed pushes.

        A cold-buffer lazy warm cannot rebuild an entry the write path just
        pushed (``source_rowid == -1``) because that SQL query does not see
        the row yet. ``replace_all`` would wipe that entry; this instead
        builds the new SQL-warmed map first, then unions in every currently
        buffered sentinel whose id is NOT already present in the SQL set (a
        same-id SQL row is the flushed version and wins over its sentinel).
        Sentinels are exempt from capacity eviction — only non-sentinel
        entries are evicted down to ``maxlen`` when the merged set exceeds
        it, so a sentinel can never be evicted before it is superseded by
        its own flush. With no sentinels present this produces the same
        result as ``replace_all``.

        Single lock acquisition, matching ``replace_all``'s race-free swap
        discipline — never clear-then-fill across separate holds.
        """
        with self._lock:
            sentinels = {
                eid: e
                for eid, e in self._entries.items()
                if e.source_rowid == -1
            }
            new_entries: dict[str, RecencyMarker] = {}
            for e in entries:
                new_entries[e.id] = e
                if len(new_entries) > self._maxlen:
                    oldest_id = min(
                        new_entries,
                        key=lambda eid: _as_utc(new_entries[eid].created_at),
                    )
                    del new_entries[oldest_id]
            for eid, sentinel in sentinels.items():
                if eid not in new_entries:
                    new_entries[eid] = sentinel
            self._entries = new_entries
            self._warm = True

    def clear(self) -> None:
        """Empty the buffer and reset the warm flag."""
        with self._lock:
            self._entries.clear()
            self._warm = False

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __iter__(self) -> Iterator[RecencyMarker]:
        """Snapshot-iterate over all entries (no lock held during iteration)."""
        with self._lock:
            snapshot = list(self._entries.values())
        return iter(snapshot)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_oldest(self) -> None:
        """Remove the entry with the oldest created_at.  Called under the lock."""
        oldest_id = min(
            self._entries,
            key=lambda eid: _as_utc(self._entries[eid].created_at),
        )
        del self._entries[oldest_id]
