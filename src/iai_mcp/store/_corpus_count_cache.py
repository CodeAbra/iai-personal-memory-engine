"""In-process, volatile count cache for the three corpus counts.

Holds cached integer values for the three corpus counts (active, pending,
edges) that feed the runtime-graph cache key.  The store remains the lossless
authority; a miss returns ``None`` and the caller degrades to the live
FILTERED SQL COUNT, never to a wrong answer.

The cache is invalidated on every count-changing write so the cached value is
exact between writes.  On a read-only recall burst the cache turns three
full-scan filtered COUNTs into O(1) dict reads.

Thread safety: all mutation and read are guarded by an internal
``threading.Lock``.  The lock is held only for the dict operation — never
across any I/O or SQL COUNT (that lives in the store method, never here).

The cache is never serialized, never persisted.  No ``__reduce__``, pickle,
or file I/O methods exist — by design.

No-raise contract: ``get`` uses ``dict.get`` (not subscript) so it returns
``None`` for a miss or an unknown key and can never raise ``KeyError``.
``invalidate(*keys)`` uses ``dict.pop(key, None)`` so an absent key is a
no-op.  Nothing in this module raises on a normal call.

Recompute race: a ``get`` miss runs a live SQL COUNT *outside* this lock,
then stores the result.  A concurrent ``invalidate`` between the COUNT and
the store would otherwise be lost, repopulating a stale value over a fresh
invalidation.  A monotonic generation counter closes that window: every
``invalidate``/``clear`` bumps the generation; the caller snapshots the
generation before the COUNT via ``generation()`` and stores with
``put_if_gen`` — a compare-and-set that refuses the store if the generation
advanced, so a stale value can never overwrite a newer invalidation.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

__all__ = ["CorpusCountCache"]

_log = logging.getLogger(__name__)

# Reference keys; an unknown key is always a miss — ``get`` returns None.
COUNT_KEYS: tuple[str, ...] = ("active", "pending", "edges")


class CorpusCountCache:
    """Write-invalidated in-process cache of the three corpus COUNT values.

    Internal state: a ``threading.Lock`` and a ``dict[str, int]`` payload.
    The three canonical keys are ``"active"``, ``"pending"``, and ``"edges"``;
    an unknown key is treated as a miss (``get`` returns ``None``).

    Lock discipline: the lock guards only the dict access.  The caller is
    responsible for running the live SQL COUNT *outside* this lock so the
    cache lock and ``_conn_lock`` are never held simultaneously.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, int] = {}
        # Monotonic generation, bumped on every invalidate/clear.  A recompute
        # snapshots it before the live COUNT and stores via ``put_if_gen`` so a
        # value computed against a since-invalidated state is discarded instead
        # of overwriting the newer invalidation.
        self._gen: int = 0
        # Optional no-raise hook fired by invalidate()/clear() AFTER the dict
        # lock is released — the union point every committed corpus-changing
        # write reaches (including the buffered flush, which never calls the
        # store's invalidation funnel).  None until a caller wires one (e.g.
        # MemoryStore.__init__ wiring db.mark_ro_pool_stale).  A hook failure
        # is swallowed so it can never break this cache's no-raise contract.
        self.on_invalidate: Callable[[], None] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> int | None:
        """Return the cached value for *key*, or ``None`` on a miss.

        Always returns ``None`` for an unknown key — uses ``dict.get``, not
        subscript, so a ``KeyError`` can never escape.  The no-raise contract
        makes it safe for the caller to branch on ``None`` without a
        ``try/except``.
        """
        with self._lock:
            return self._data.get(key)

    def put(self, key: str, value: int) -> None:
        """Store a freshly-computed *value* for *key* unconditionally."""
        with self._lock:
            self._data[key] = value

    def generation(self) -> int:
        """Return the current generation counter.

        Snapshot this *before* running the live COUNT, then pass it to
        ``put_if_gen`` so an intervening ``invalidate``/``clear`` causes the
        store to be refused (compare-and-set).
        """
        with self._lock:
            return self._gen

    def put_if_gen(self, key: str, value: int, gen: int) -> None:
        """Store *value* for *key* only if the generation has not advanced.

        Compare-and-set: if a concurrent ``invalidate``/``clear`` bumped the
        generation after the caller snapshotted *gen* (i.e. during the live
        COUNT), the *value* is stale relative to the latest invalidation and
        is discarded.  Otherwise it is installed.  Both the comparison and the
        store happen under the same lock acquisition so the check cannot tear.
        """
        with self._lock:
            if gen == self._gen:
                self._data[key] = value

    def adjust(self, key: str, delta: int) -> None:
        """Shift a cached count by a known-exact *delta* instead of dropping it.

        The hot write paths (capture flush, pending insert, reembed flip) know
        their exact count delta; dropping the value there forces the next read
        into a full filtered COUNT — on the lilli engine a scan of every leaf
        page, paid over and over under ambient write churn. A miss stays a
        miss (there is nothing to shift; the next read recomputes).

        The generation is bumped either way: an in-flight recompute snapshotted
        its generation BEFORE this write committed, so its COUNT may predate
        the write — ``put_if_gen`` must refuse it exactly as it would after an
        ``invalidate``. Fires ``on_invalidate`` after the lock is released, so
        downstream staleness signals (RO pools) see adjusted writes exactly as
        they see invalidating ones.
        """
        with self._lock:
            self._gen += 1
            if key in self._data:
                self._data[key] += delta
        self._fire_on_invalidate()

    def invalidate(self, *keys: str) -> None:
        """Remove the named *keys* from the cache.

        An absent key is a no-op (``dict.pop(key, None)``).  Accepts
        multiple keys so the caller can invalidate a set atomically under a
        single lock acquisition.  Bumps the generation so an in-flight
        recompute storing via ``put_if_gen`` is refused.

        Fires ``on_invalidate`` (if registered) AFTER the lock is released —
        never while holding ``self._lock`` — so a hook that itself touches
        another lock can never create a new lock-ordering edge through this
        cache. The hook is called inside try/except so a hook failure can
        never break this method's own no-raise contract.
        """
        with self._lock:
            self._gen += 1
            for key in keys:
                self._data.pop(key, None)
        self._fire_on_invalidate()

    def clear(self) -> None:
        """Remove all cached values and bump the generation."""
        with self._lock:
            self._gen += 1
            self._data.clear()
        self._fire_on_invalidate()

    def _fire_on_invalidate(self) -> None:
        hook = self.on_invalidate
        if hook is None:
            return
        try:
            hook()
        except Exception as exc:  # noqa: BLE001 -- hook failure MUST NOT propagate
            _log.debug("CorpusCountCache.on_invalidate hook failed: %s", exc)

    def __repr__(self) -> str:
        with self._lock:
            snapshot = dict(self._data)
        return f"CorpusCountCache({snapshot!r})"
