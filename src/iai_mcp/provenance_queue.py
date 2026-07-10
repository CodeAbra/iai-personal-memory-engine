from __future__ import annotations

import atexit
import json
import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from iai_mcp.store import MemoryStore


_STOP = object()
_FLUSH = object()

OVERFLOW_DIR_NAME = ".provenance-overflow"
_WORKER_IDLE_POLL_S = 5.0


class ProvenanceWriteQueue:

    def __init__(
        self,
        store: "MemoryStore",
        *,
        coalesce_ms: int = 50,
        max_queue_size: int = 4096,
        max_batch_pairs: int = 256,
        overflow_dir: "Path | None" = None,
    ) -> None:
        self._store = store
        # Bind the overflow directory to THIS queue's own store root once, at
        # construction. It must never be re-derived from a process-global path at
        # poll time: multiple queues in one process would otherwise all resolve to
        # the same directory and drain each other's spilled batches into the wrong
        # store. The store root already anchors at the install home, so the resolved
        # path is unchanged in normal operation.
        if overflow_dir is not None:
            self._overflow_dir = Path(overflow_dir)
        else:
            self._overflow_dir = Path(store.root) / OVERFLOW_DIR_NAME
        self._coalesce_s = max(1, int(coalesce_ms)) / 1000.0
        self._max_batch = int(max_batch_pairs)
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
                name="iai-mcp-provenance-queue",
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


    def enqueue(self, pairs: "list[tuple[UUID, dict]]") -> None:
        if not pairs:
            return
        try:
            self._q.put_nowait(list(pairs))
            return
        except queue.Full:
            pass
        self._spill_to_disk(list(pairs))
        try:
            sys.stderr.write(
                '{"event":"provenance_queue_overflow_spill","n_pairs":'
                + str(len(pairs))
                + "}\n"
            )
        except Exception:  # noqa: BLE001 -- stderr emission is best-effort
            pass


    def _spill_to_disk(self, pairs: list) -> None:
        if not pairs:
            return
        try:
            overflow_dir = self._overflow_dir
            overflow_dir.mkdir(parents=True, exist_ok=True)
            ts_ms = int(time.time() * 1000)
            fpath = overflow_dir / f"{ts_ms}-{len(pairs)}-{id(pairs) & 0xFFFF:04x}.jsonl"
            tmp_path = fpath.with_suffix(fpath.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as fh:
                for rid, entry in pairs:
                    fh.write(json.dumps({"id": str(rid), "entry": entry}) + "\n")
            tmp_path.rename(fpath)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("provenance_queue_spill_failed", extra={"err": str(exc)[:200], "n_pairs": len(pairs)})
            try:
                sys.stderr.write(
                    '{"event":"provenance_queue_spill_failed","error":'
                    + _json_str(str(exc))
                    + ',"n_pairs":' + str(len(pairs)) + '}\n'
                )
            except Exception:  # noqa: BLE001 -- stderr emission is best-effort
                pass

    def _drain_overflow_dir(self) -> int:
        overflow_dir = self._overflow_dir
        if not overflow_dir.exists():
            return 0
        n_re_enqueued = 0
        for fpath in sorted(overflow_dir.glob("*.jsonl")):
            try:
                pairs: list = []
                with fpath.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        pairs.append((UUID(obj["id"]), obj["entry"]))
                if not pairs:
                    fpath.unlink()
                    continue
                try:
                    self._q.put(pairs, timeout=0.5)
                except queue.Full:
                    return n_re_enqueued
                fpath.unlink()
                n_re_enqueued += len(pairs)
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning("provenance_queue_spill_drain_failed", extra={"err": str(exc)[:200]})
                try:
                    failed = fpath.with_suffix(f".failed-{int(time.time())}.jsonl")
                    fpath.rename(failed)
                    sys.stderr.write(
                        '{"event":"provenance_queue_spill_drain_failed","error":'
                        + _json_str(str(exc)) + '}\n'
                    )
                except Exception:  # noqa: BLE001 -- stderr + rename is best-effort
                    pass
        return n_re_enqueued


    def _run(self) -> None:
        while True:
            try:
                item = self._q.get(timeout=_WORKER_IDLE_POLL_S)
            except queue.Empty:
                try:
                    self._drain_overflow_dir()
                except Exception:  # noqa: BLE001 -- worker must never die (Rule 1)
                    logger.debug("provenance_overflow_drain_error", exc_info=True)
                continue
            except Exception:  # noqa: BLE001 -- worker must never die (Rule 1)
                logger.debug("provenance_queue_get_error", exc_info=True)
                continue
            if item is _STOP:
                self._drain_remaining()
                return
            if item is _FLUSH:
                self._drain_remaining()
                self._flush_event.set()
                continue
            pairs: list = list(item)
            while len(pairs) < self._max_batch:
                try:
                    nxt = self._q.get(timeout=self._coalesce_s)
                except queue.Empty:
                    break
                if nxt is _STOP:
                    self._flush_batch(pairs)
                    self._drain_remaining()
                    return
                if nxt is _FLUSH:
                    self._flush_batch(pairs)
                    self._drain_remaining()
                    self._flush_event.set()
                    pairs = []
                    break
                pairs.extend(nxt)
            if pairs:
                self._flush_batch(pairs)

    def _drain_remaining(self) -> None:
        pairs: list = []
        saw_flush = False
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item is _STOP:
                continue
            if item is _FLUSH:
                saw_flush = True
                continue
            pairs.extend(item)
        if pairs:
            self._flush_batch(pairs)
        if saw_flush:
            self._flush_event.set()

    def _flush_batch(self, pairs: list) -> None:
        if not pairs:
            return
        try:
            self._store.append_provenance_batch(pairs, records_cache=None)
        except Exception as exc:  # noqa: BLE001 -- Rule 1: recall must never fail
            logger.warning("provenance_queue_flush_failed", extra={"n_pairs": len(pairs), "err": str(exc)[:200]})
            try:
                sys.stderr.write(
                    '{"event":"provenance_queue_flush_failed","n_pairs":'
                    + str(len(pairs))
                    + ',"error":'
                    + _json_str(str(exc))
                    + "}\n"
                )
            except Exception:  # noqa: BLE001 -- stderr emission is best-effort
                pass

    def _atexit_flush(self) -> None:
        try:
            if self._started:
                self.flush(timeout=2.0)
                self.stop()
        except Exception:  # noqa: BLE001 -- atexit must never raise
            pass


def _json_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
