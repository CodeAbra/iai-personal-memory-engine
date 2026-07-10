"""Background write queue for deferred reinforcement writes.

Moves the per-hit reconsolidation write (labile_until UPDATE + edge boost)
off the recall response thread.  The recall handler enqueues record ids and
returns immediately; this queue drains them on a background daemon thread,
calling store.reinforce_record(rid, is_retrieval=True) per id.

Design mirrors ProvenanceWriteQueue (provenance_queue.py):
  - bounded stdlib queue.Queue, daemon thread, coalesced batch drain
  - enqueue is non-blocking: on a full queue the batch is logged+dropped
    (reinforcement is best-effort — a dropped boost degrades gracefully)
  - worker never crashes on a per-id error (catch-log-continue)
  - atexit flush+stop
  - flush(timeout) and stop() for lifecycle management
"""
from __future__ import annotations

import atexit
import logging
import queue
import sys
import threading
import time
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from iai_mcp.store import MemoryStore

logger = logging.getLogger(__name__)

_STOP = object()
_FLUSH = object()

_WORKER_IDLE_POLL_S = 5.0


class ReinforceWriteQueue:
    """Deferred, background reinforce writer.

    Accepts batches of record ids via enqueue() and drains them on a single
    daemon thread, calling store.reinforce_record(rid, is_retrieval=True)
    per id.  The caller is never blocked — on queue overflow the batch is
    dropped with a stderr marker (reinforcement is best-effort).
    """

    def __init__(
        self,
        store: "MemoryStore",
        *,
        coalesce_ms: int = 50,
        max_queue_size: int = 4096,
        max_batch: int = 256,
    ) -> None:
        self._store = store
        self._coalesce_s = max(1, int(coalesce_ms)) / 1000.0
        self._max_batch = int(max_batch)
        self._q: queue.Queue = queue.Queue(maxsize=int(max_queue_size))
        self._thread: threading.Thread | None = None
        self._started = False
        self._stop_requested = False
        self._flush_event = threading.Event()
        self._atexit_registered = False
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._stop_requested = False
            self._thread = threading.Thread(
                target=self._run,
                name="iai-mcp-reinforce-queue",
                daemon=True,
            )
            self._thread.start()
            if not self._atexit_registered:
                atexit.register(self._atexit_flush)
                self._atexit_registered = True

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            self._stop_requested = True
            try:
                self._q.put_nowait(_STOP)
            except queue.Full:
                try:
                    self._q.get_nowait()
                    self._q.put_nowait(_STOP)
                except queue.Empty:
                    pass
            t = self._thread
        if t is not None:
            t.join(timeout=5.0)
        with self._lock:
            self._started = False
            self._thread = None

    def flush(self, timeout: float = 2.0) -> None:
        if not self._started:
            return
        self._flush_event.clear()
        try:
            self._q.put(_FLUSH, timeout=timeout)
        except queue.Full:
            return
        self._flush_event.wait(timeout=timeout)

    def enqueue(self, record_ids: "list[UUID]") -> None:
        """Enqueue record ids for deferred reinforcement.

        Non-blocking: on a full queue the batch is dropped with a log/stderr
        marker.  Reinforcement is best-effort — a missed boost degrades
        gracefully (the edge weight is not updated, labile_until is not set
        for this recall, but the store remains consistent).
        """
        if not record_ids:
            return
        batch = list(record_ids)
        try:
            self._q.put_nowait(batch)
            # Emit a deferred-enqueue signal so the eventual-consistency window
            # is observable (stderr JSON line, one per batch).
            try:
                sys.stderr.write(
                    '{"event":"reinforce_deferred","n_ids":'
                    + str(len(batch))
                    + "}\n"
                )
            except Exception:  # noqa: BLE001 -- stderr emission is best-effort
                pass
            return
        except queue.Full:
            pass
        # Queue overflow: log + drop.  Reinforcement is best-effort.
        logger.debug(
            "reinforce_queue_overflow",
            extra={"n_ids": len(batch)},
        )
        try:
            sys.stderr.write(
                '{"event":"reinforce_queue_overflow","n_ids":'
                + str(len(batch))
                + "}\n"
            )
        except Exception:  # noqa: BLE001 -- stderr emission is best-effort
            pass

    def _run(self) -> None:
        while True:
            try:
                item = self._q.get(timeout=_WORKER_IDLE_POLL_S)
            except queue.Empty:
                # A stop requested while the queue was empty (the _STOP sentinel
                # was already consumed, e.g. drained behind a _FLUSH) must still
                # end the worker rather than idle forever.
                if self._stop_requested:
                    return
                continue
            except Exception:  # noqa: BLE001 -- worker must never die
                logger.debug("reinforce_queue_get_error", exc_info=True)
                continue
            if item is _STOP:
                self._drain_remaining()
                return
            if item is _FLUSH:
                # A _STOP may be sitting behind this _FLUSH (stop() enqueues
                # _STOP right after flush() enqueues _FLUSH).  _drain_remaining
                # reports whether it consumed that _STOP so the worker exits
                # instead of swallowing the stop and idling forever.
                saw_stop = self._drain_remaining()
                self._flush_event.set()
                if saw_stop:
                    return
                continue
            ids: list = list(item)
            while len(ids) < self._max_batch:
                try:
                    nxt = self._q.get(timeout=self._coalesce_s)
                except queue.Empty:
                    break
                if nxt is _STOP:
                    self._flush_ids(ids)
                    self._drain_remaining()
                    return
                if nxt is _FLUSH:
                    self._flush_ids(ids)
                    saw_stop = self._drain_remaining()
                    self._flush_event.set()
                    if saw_stop:
                        return
                    ids = []
                    break
                ids.extend(nxt)
            if ids:
                self._flush_ids(ids)

    def _drain_remaining(self) -> bool:
        """Drain pending items, running any queued batches.  Returns True if a
        _STOP sentinel was consumed, so the caller can exit the worker loop
        rather than discard the stop signal and idle forever.
        """
        ids: list = []
        saw_flush = False
        saw_stop = False
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item is _STOP:
                saw_stop = True
                continue
            if item is _FLUSH:
                saw_flush = True
                continue
            ids.extend(item)
        if ids:
            self._flush_ids(ids)
        if saw_flush:
            self._flush_event.set()
        return saw_stop

    # Per-transaction chunk size for a reinforcement batch. Each chunk runs
    # under ONE connection-lock hold and ONE commit, so a backlog of hundreds
    # of queued reinforcements costs a handful of bounded lock windows (and
    # fsyncs) instead of one lock/commit per record — a per-record cadence
    # lets this background worker monopolize the shared writer connection in
    # a lock ping-pong that starves concurrent foreground recalls for the
    # whole backlog. Between chunks the worker yields the OS scheduler slot so
    # a waiting recall acquires the lock promptly.
    _TXN_CHUNK = 16

    def _flush_ids(self, ids: list) -> None:
        if not ids:
            return
        n_ok = 0
        for start in range(0, len(ids), self._TXN_CHUNK):
            chunk = ids[start : start + self._TXN_CHUNK]
            n_ok += self._reinforce_chunk(chunk)
            time.sleep(0)  # real scheduler handoff between lock windows
        # Emit a drain-completion signal so the eventual-consistency window
        # close is observable.
        try:
            sys.stderr.write(
                '{"event":"reinforce_drained","n_drained":'
                + str(n_ok)
                + ',"n_total":'
                + str(len(ids))
                + "}\n"
            )
        except Exception:  # noqa: BLE001 -- stderr emission is best-effort
            pass

    def _reinforce_chunk(self, chunk: list) -> int:
        """Reinforce one chunk under a single lock hold + transaction.

        Falls back to the per-record path (own lock/commit per call) when the
        batched transaction seam is unavailable or the wrapping itself fails —
        reinforcement is best-effort and a batching problem must degrade to
        the slower correct path, never drop the chunk. A per-record failure
        inside the batch skips only that record.
        """
        n_ok = 0
        try:
            from iai_mcp.hippo._db import _txn

            conn_lock = self._store.db._conn_lock
            conn = self._store.db._conn
            with conn_lock, _txn(conn):
                for rid in chunk:
                    try:
                        self._store.reinforce_record(rid, is_retrieval=True)
                        n_ok += 1
                    except Exception:  # noqa: BLE001 -- worker must never die
                        logger.debug(
                            "reinforce_queue_item_failed",
                            extra={"rid": str(rid)},
                            exc_info=True,
                        )
            return n_ok
        except Exception:  # noqa: BLE001 -- batching seam failure: degrade to
            # the per-record path below rather than dropping the chunk.
            logger.debug("reinforce_chunk_txn_failed", exc_info=True)
        n_ok = 0
        for rid in chunk:
            try:
                self._store.reinforce_record(rid, is_retrieval=True)
                n_ok += 1
            except Exception:  # noqa: BLE001 -- worker must never die
                logger.debug(
                    "reinforce_queue_item_failed",
                    extra={"rid": str(rid)},
                    exc_info=True,
                )
        return n_ok

    def _atexit_flush(self) -> None:
        try:
            if self._started:
                self.flush(timeout=2.0)
                self.stop()
        except Exception:  # noqa: BLE001 -- atexit must never raise
            pass
