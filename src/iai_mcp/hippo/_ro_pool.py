"""Pooled read-only reader connections for the lilli storage engine.

Wraps the lock-free ``get_lilli_raw_conn(path, read_only=True)`` primitive in a
fixed-size pool so the recall read path never contends on the writer's
``_conn_lock``. On the stdlib driver this module is not constructed at all —
``HippoDB.ro_conn()`` falls back to the existing shared-connection path.

Staleness model: each pool slot is a snapshot-at-open reader (the native
``open_read_only`` overlays committed WAL frames at open time; it does not
live-track the writer afterward). That snapshot-at-open property is the
HAZARD this module closes, not the observed behavior of a borrowed slot: a
slot is refreshed (closed and reopened) the next time it is borrowed after
the pool's generation counter has been bumped by any mutating statement on
the writer connection (not only a corpus-count-changing one) — never
eagerly, never on a read. On the writer-proxy path, already-committed data
is never stale: a slot refreshes on its next borrow after any commit. This
keeps read-heavy windows on a warm parse/col-cache connection while bounding
staleness to "at most one borrow behind the last commit".
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time as _time
from typing import Any

from iai_mcp import errors

_log = logging.getLogger(__name__)


def _slow_ro_execute_log_ms() -> float:
    try:
        return float(os.environ.get("IAI_MCP_SLOW_RO_EXECUTE_LOG_MS", "400"))
    except ValueError:
        return 400.0

#: Default fixed pool size; operator-overridable, matching the existing
#: IAI_MCP_* env-override convention used elsewhere in this codebase.
RO_POOL_SIZE_DEFAULT = 8

#: The read-only-snapshot fence (crates/lillibrain/src/pager.rs:322-338
#: validate_ro_snapshot_fence) surfaces as a GENERIC errors.OperationalError
#: (crates/lilliengine/src/py.rs:39-55 to_pyerr maps StoreError::Integrity to
#: it) -- there is NO distinct catchable Python fence class. The message
#: substring below (pager.rs:329 "main file size changed" variant / pager.rs:335
#: "checkpoint generation advanced" variant, both prefixed with this exact
#: phrase) is the ONLY discriminator between a checkpoint-invalidated snapshot
#: and a genuine OperationalError. If that Rust message ever changes, this
#: constant must be updated in lockstep or every fence occurrence silently
#: stops being retried (falls through to the non-fence re-raise branch).
_RO_SNAPSHOT_FENCE_MARKER = "read-only snapshot invalidated"

try:
    # The native engine raises this dedicated OperationalError subclass for a
    # snapshot fence, so callers match the fault STRUCTURALLY instead of on the
    # message text. It is unavailable on the stdlib driver / a Python-only build.
    from iai_mcp_native.engine import SnapshotFenceError as _SnapshotFenceError
except Exception:  # noqa: BLE001 -- native engine absent; substring is the fallback
    _SnapshotFenceError = ()


def _is_snapshot_fence(exc: BaseException) -> bool:
    """True when ``exc`` is a retryable read-only snapshot fence.

    Prefers a structural ``isinstance`` against the engine's dedicated
    ``SnapshotFenceError`` (immune to message-text drift); falls back to the
    message marker for the stdlib driver and any path where the class is
    unavailable.
    """
    if _SnapshotFenceError and isinstance(exc, _SnapshotFenceError):
        return True
    return _RO_SNAPSHOT_FENCE_MARKER in str(exc)

#: Bound on how many times borrow() will close+reopen a slot in response to
#: the fence before giving up and raising (so ro_conn() falls back to the
#: writer path). Guards against a pathologically-flapping checkpoint spinning
#: forever; a real invalidation is resolved by a single reopen in practice.
#: The bound is generous because a reopen is cheap (the decoded column-index
#: sidecar is served from the process-wide cache, so a slot reopen is
#: sub-millisecond) while the fallback is expensive: it re-routes a read onto
#: the shared writer connection, where it serializes behind background write
#: activity — under a write-concurrent burst (drain + reinforcements), a
#: too-small bound pushed EVERY read through the writer and starved
#: concurrent recalls behind the writer's lock.
_MAX_FENCE_REOPEN_ATTEMPTS = 8


def _pool_size() -> int:
    raw = os.environ.get("IAI_MCP_RO_POOL_SIZE")
    if raw is None:
        return RO_POOL_SIZE_DEFAULT
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return RO_POOL_SIZE_DEFAULT
    return val if val > 0 else RO_POOL_SIZE_DEFAULT


def _pool_off() -> bool:
    return os.environ.get("IAI_MCP_RO_POOL_OFF", "").strip() in ("1", "true", "TRUE", "yes")


class _Slot:
    """One pool slot: a lazily-opened RO connection plus its open generation."""

    __slots__ = ("conn", "generation")

    def __init__(self, conn: Any, generation: int) -> None:
        self.conn = conn
        self.generation = generation


class RoConnPool:
    """Fixed-size pool of lock-free read-only lilli engine connections.

    Backing structure is a ``queue.Queue(maxsize=size)`` — thread-safe,
    blocks correctly under saturation, no new synchronization primitive.
    Slots are opened LAZILY (first borrow pays the open cost, not
    construction) so a freshly-booted daemon pays nothing until the first
    recall actually borrows a slot.
    """

    def __init__(self, path: str, size: int | None = None) -> None:
        self._path = path
        self._size = size if size is not None and size > 0 else _pool_size()
        self._queue: "queue.Queue[_Slot]" = queue.Queue(maxsize=self._size)
        self._filled = 0
        self._fill_lock = threading.Lock()
        self._gen_lock = threading.Lock()
        self._generation = 0
        self._closed = False
        # Observability for the two failure lanes the pool exists to avoid:
        # fence reopens (normal under checkpoints, should stay cheap) and
        # writer-path fallbacks (each one serializes a recall read behind
        # background writes — a rising count means the pool is not delivering).
        self.fence_reopen_count = 0
        self.slot_refresh_count = 0
        self.slot_reopen_count = 0
        self.writer_fallback_count = 0

    @staticmethod
    def _fence_backoff(attempts: int) -> None:
        """Brief growing pause between fence reopens: a checkpoint flap that
        defeats back-to-back reopen+probe cycles usually clears within
        milliseconds, and one landed retry keeps the read OFF the writer lock
        (the expensive fallback). Worst case across the bound is ~180ms."""
        import time as _time

        _time.sleep(min(0.05, 0.005 * attempts))

    # ------------------------------------------------------------------
    # Slot lifecycle
    # ------------------------------------------------------------------

    def _open_slot(self) -> Any:
        from iai_mcp.lillibrain.connection import get_lilli_raw_conn

        conn = get_lilli_raw_conn(self._path, read_only=True)
        if conn is None:
            raise RuntimeError(
                f"RoConnPool: no lilli engine connection available for {self._path!r}"
            )
        return conn

    def _close_slot_conn(self, conn: Any) -> None:
        close = getattr(conn, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 -- best-effort close only
                pass

    def _current_generation(self) -> int:
        with self._gen_lock:
            return self._generation

    def _ensure_filled(self) -> None:
        """Open exactly ONE additional slot, lazily, so a borrower is never
        blocked behind the cost of filling every other slot it does not need.

        Called from every ``borrow()`` while ``_filled < _size``, independent
        of whether the queue is currently empty -- growth must not stall once
        the pool has produced its first released slot. A purely
        queue-empty-gated growth condition converges to exactly ONE slot under
        a SEQUENTIAL borrow/release workload (single caller, no concurrency):
        after the first borrow's slot is released it sits in the queue, so the
        queue is never empty again for any later borrow, and a pool that only
        grows on "queue empty" would then never open its second slot even
        though ``_filled`` never reached ``_size``. Growth is instead driven
        purely by ``_filled < _size`` here; ``borrow()`` decides separately
        whether the freshly-opened slot serves the CURRENT caller (queue was
        empty) or is simply queued for a future borrower (queue was not empty)
        -- so a caller that finds a released slot waiting still never pays for
        opening another one itself; the incremental open (when it happens)
        always happens on the very call that needs to grow the pool, one slot
        at a time, never blocking one caller behind filling the whole pool.
        Guarded by ``_fill_lock`` so concurrent first-borrows do not each try
        to open a slot independently and overshoot ``_size``.
        """
        with self._fill_lock:
            if self._filled >= self._size:
                return
            gen = self._current_generation()
            slot = _Slot(self._open_slot(), gen)
            self._queue.put_nowait(slot)
            self._filled += 1

    # ------------------------------------------------------------------
    # Borrow / release
    # ------------------------------------------------------------------

    class _FenceRetryingConn:
        """Thin ``execute()`` wrapper closing the TOCTOU window between the
        borrow-time probe and the caller's own query.

        ``borrow()``'s proactive probe only proves the slot was fence-free at
        the moment of handout — a checkpoint that commits in the window
        between handout and the caller's real ``execute()`` still raises the
        fence on that real call. Retrying it here (bounded, same marker/bound
        as ``borrow()``'s own probe retry) means every ``ro_conn()`` caller is
        covered transparently, not just call sites that happen to re-borrow.
        Every non-fence error, and fence-retry exhaustion, propagates
        unchanged — this wrapper only ever narrows what is retried, never
        widens what is swallowed.
        """

        __slots__ = ("_pool", "_borrowed")

        def __init__(self, pool: "RoConnPool", borrowed: "RoConnPool._Borrowed") -> None:
            self._pool = pool
            self._borrowed = borrowed

        def execute(self, sql: str, params=()):
            # Wrapping execute() alone is COMPLETE fence coverage: the engine
            # cursor materializes the full result set inside execute
            # (crates/lilliengine/src/py.rs Cursor — "cursor over a
            # materialized result set"), so fetchone/fetchall never read a
            # page and can never hit the fence.
            attempts = 0
            while True:
                try:
                    _conn = self._borrowed._slot.conn
                    _t0 = _time.perf_counter()
                    _cells0: "int | None" = None
                    try:
                        _cells0 = int(_conn.cells_visited_count())
                    except Exception:  # noqa: BLE001 — introspection best-effort
                        pass
                    result = _conn.execute(sql, params)
                    _ms = (_time.perf_counter() - _t0) * 1000.0
                    if _ms >= _slow_ro_execute_log_ms():
                        _idx = "n/a"
                        try:
                            _cells = (
                                int(_conn.cells_visited_count()) - _cells0
                                if _cells0 is not None
                                else -1
                            )
                            _idx = (
                                f"id_ready={_conn.id_index_ready('records')} "
                                f"col_ready={_conn.col_index_ready('records')} "
                                f"cells={_cells}"
                            )
                        except Exception:  # noqa: BLE001 — introspection best-effort
                            pass
                        _log.warning(
                            "slow RO execute: %.0fms pid=%d thread=%s %s sql=%s",
                            _ms, os.getpid(), threading.current_thread().name,
                            _idx, sql[:120],
                        )
                    return result
                except errors.OperationalError as exc:
                    if not _is_snapshot_fence(exc):
                        raise
                    attempts += 1
                    if attempts > _MAX_FENCE_REOPEN_ATTEMPTS:
                        raise
                    _log.debug(
                        "RoConnPool: fence hit on caller's own query (attempt "
                        "%d/%d), reopening slot: %s",
                        attempts,
                        _MAX_FENCE_REOPEN_ATTEMPTS,
                        exc,
                    )
                    self._pool.fence_reopen_count += 1
                    self._pool._fence_backoff(attempts)
                    self._pool._refresh_or_reopen_slot(
                        self._borrowed._slot, self._pool._current_generation()
                    )

        def __getattr__(self, name: str):
            return getattr(self._borrowed._slot.conn, name)

    class _Borrowed:
        """Context manager yielding the borrowed slot's connection."""

        def __init__(self, pool: "RoConnPool", slot: _Slot) -> None:
            self._pool = pool
            self._slot = slot

        def __enter__(self) -> Any:
            return RoConnPool._FenceRetryingConn(self._pool, self)

        def __exit__(self, exc_type, exc, tb) -> bool:
            self._pool._release(self._slot)
            return False

    def borrow(self) -> "RoConnPool._Borrowed":
        """Check out a slot, refreshing it first if it is stale.

        Also self-heals against the read-only-snapshot fence: a WAL
        auto-checkpoint triggered by a concurrent writer can invalidate an
        already-open RO snapshot even when the pool's own generation counter
        was never bumped. (On lilli the mutation-signalling proxy DOES bump
        the generation for every mutating statement including reinforce/boost
        UPDATEs; the probe still covers writers that bypass the proxy and the
        checkpoint-only case.) A
        proactive ``SELECT 1`` probe forces that fence check on the slot
        BEFORE handout, so a caller never receives a slot that raises on its
        first real query. Only the narrowed fence error (message contains
        ``_RO_SNAPSHOT_FENCE_MARKER``) is retried (close + reopen + re-probe);
        every other ``OperationalError`` re-raises immediately. The
        reopen is bounded (``_MAX_FENCE_REOPEN_ATTEMPTS``) so a genuinely
        broken store cannot spin -- exhaustion raises here too.

        Raises on total failure (no lilli connection available at all, a
        non-fence OperationalError, or fence-retry exhaustion) so the caller
        (``HippoDB.ro_conn()``) can catch and fall back to the writer path —
        this method itself never silently degrades.
        """
        # Grow by exactly one slot per call while under the configured size --
        # gating on "_filled < _size" alone, NOT on queue emptiness. A
        # queue-empty-only gate converges to just ONE slot under a SEQUENTIAL
        # borrow/release workload: the released slot sits in the queue, so the
        # queue is never empty again for any later borrow, and the pool never
        # grows past its first slot even though it never reached `_size`. This
        # still never blocks one caller behind opening the WHOLE pool --
        # `_ensure_filled` opens exactly one slot per invocation -- so this
        # borrow only ever pays for at most one incremental open, same as
        # before; convergence to `_size` just takes `_size` borrow calls
        # instead of relying on concurrent demand alone to trigger it. Once
        # the pool is at capacity this is a cheap no-op check every borrow.
        if self._filled < self._size:
            self._ensure_filled()
        slot = self._queue.get()
        # Any raise between the get() above and the _Borrowed handoff below
        # abandons a dequeued slot: it is never requeued and _filled never
        # shrinks, so each raising borrow permanently removes one slot — after
        # _size of them over a daemon lifetime the next get() blocks FOREVER
        # and recall hangs with no fallback (the writer-fallback lane catches
        # raises, not blocks). Give the slot back to the accounting on every
        # raise; its connection may be torn, so close it and let the next
        # fill reopen fresh.
        try:
            return self._borrow_prepared(slot)
        except BaseException:
            self._close_slot_conn(slot.conn)
            with self._fill_lock:
                self._filled = max(0, self._filled - 1)
            raise

    def _borrow_prepared(self, slot: "_Slot") -> "RoConnPool._Borrowed":
        current_gen = self._current_generation()
        if slot.generation < current_gen:
            self._refresh_or_reopen_slot(slot, current_gen)

        attempts = 0
        while True:
            try:
                # The fence (pager.rs:322-338) only re-validates on a genuine
                # main-file-fallback PAGE READ of content that moved between
                # generations -- it is invisible to any page the RO snapshot
                # already has cached (e.g. the header page, or a page never
                # touched since open). reinforce_record/boost_edges rewrite
                # EXISTING `records` rows in place, so the probe must read
                # from `records` itself (the same table those writers touch,
                # and the same table every recall read serves from) to force
                # a genuine re-validation. A bare "SELECT 1" (no FROM) is
                # rejected by the lilli SQL parser, so this is the cheapest
                # FROM-bearing probe that actually exercises the fence.
                slot.conn.execute("SELECT 1 FROM records LIMIT 1")
                break
            except errors.OperationalError as exc:
                if _RO_SNAPSHOT_FENCE_MARKER not in str(exc):
                    raise
                attempts += 1
                if attempts > _MAX_FENCE_REOPEN_ATTEMPTS:
                    _log.debug(
                        "RoConnPool.borrow: fence retry exhausted after "
                        "%d attempts, raising for writer-path fallback: %s",
                        attempts - 1,
                        exc,
                    )
                    raise
                _log.debug(
                    "RoConnPool.borrow: read-only snapshot fence hit "
                    "(attempt %d/%d), reopening slot: %s",
                    attempts,
                    _MAX_FENCE_REOPEN_ATTEMPTS,
                    exc,
                )
                self.fence_reopen_count += 1
                self._fence_backoff(attempts)
                self._refresh_or_reopen_slot(slot, self._current_generation())

        return RoConnPool._Borrowed(self, slot)

    def _refresh_or_reopen_slot(self, slot: "_Slot", target_gen: int) -> None:
        """Bring a stale slot to the newest committed snapshot.

        Prefers the engine's in-place snapshot refresh — the connection keeps
        its adopted indexes for every table whose committed write-generation
        did not move, so a capture-sized write no longer costs this reader a
        whole-connection teardown plus lazy index rebuilds. Any refresh
        failure falls back to the always-correct close + reopen.
        """
        refreshed = False
        _refresh = getattr(slot.conn, "refresh", None)
        if callable(_refresh):
            try:
                _refresh()
                refreshed = True
            except Exception as exc:  # noqa: BLE001 -- reopen is the fallback
                _log.debug(
                    "RoConnPool: slot refresh failed, reopening instead: %s", exc
                )
        # Health counters: a pool that silently degrades to reopen-per-borrow
        # (e.g. a stale .so without the engine refresh) is indistinguishable
        # from a healthy one without these — and reopen-per-borrow is the
        # exact storm the refresh exists to kill.
        if refreshed:
            self.slot_refresh_count += 1
        else:
            self.slot_reopen_count += 1
            self._close_slot_conn(slot.conn)
            slot.conn = self._open_slot()
        slot.generation = target_gen

    def _release(self, slot: _Slot) -> None:
        if self._closed:
            # The pool closed while this slot was checked out: close its
            # connection instead of returning it to a drained queue, where it
            # would never be closed (a leaked connection at shutdown).
            self._close_slot_conn(slot.conn)
            return
        try:
            self._queue.put_nowait(slot)
        except queue.Full:  # pragma: no cover -- defensive; sized to filled count
            self._close_slot_conn(slot.conn)
            with self._fill_lock:
                self._filled = max(0, self._filled - 1)

    # ------------------------------------------------------------------
    # Staleness + recycling
    # ------------------------------------------------------------------

    def mark_stale(self) -> None:
        """Bump the generation counter. No-raise, trivial lock only."""
        with self._gen_lock:
            self._generation += 1

    def recycle(self) -> None:
        """Bump the generation and proactively drain/reopen every idle slot.

        Checked-out slots are not touched here — they refresh on their next
        borrow (the generation bump alone covers them). Never raises.
        """
        try:
            self.mark_stale()
            gen = self._current_generation()
            drained: list[_Slot] = []
            while True:
                try:
                    drained.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            for slot in drained:
                try:
                    self._close_slot_conn(slot.conn)
                    slot.conn = self._open_slot()
                    slot.generation = gen
                except Exception as exc:  # noqa: BLE001 -- recycle must never raise
                    _log.debug("RoConnPool.recycle: slot reopen failed: %s", exc)
                    with self._fill_lock:
                        self._filled = max(0, self._filled - 1)
                    continue
                try:
                    self._queue.put_nowait(slot)
                except queue.Full:  # pragma: no cover -- defensive
                    # A concurrent _release() can refill the queue between our
                    # drain and re-put; closing this slot WITHOUT shrinking
                    # _filled would leave the pool believing it holds a live slot
                    # it doesn't -- that leak drains the pool over a daemon
                    # lifetime until borrow()'s get() blocks forever. Decrement
                    # _filled exactly as _release() does.
                    self._close_slot_conn(slot.conn)
                    with self._fill_lock:
                        self._filled = max(0, self._filled - 1)
        except Exception as exc:  # noqa: BLE001 -- recycle must never raise
            _log.debug("RoConnPool.recycle failed: %s", exc)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Best-effort close of every slot currently in the queue."""
        if self._closed:
            return
        self._closed = True
        while True:
            try:
                slot = self._queue.get_nowait()
            except queue.Empty:
                break
            self._close_slot_conn(slot.conn)
        self._filled = 0
