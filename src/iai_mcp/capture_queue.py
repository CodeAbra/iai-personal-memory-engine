from __future__ import annotations

import errno
import fcntl
import json
import os
import secrets
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_QUEUE_DIR: Path = Path.home() / ".iai-mcp" / "pending"
"""Production location for the persistent queue."""

DEFAULT_MAX_SIZE: int = 10_000
"""Default ceiling before ``prune_oldest`` kicks in."""

_PRUNE_BATCH_HEADROOM: int = 100

SCHEMA_VERSION: int = 1
"""Bumped only when the on-disk pending-<ulid>.json layout changes."""

_AUDIT_LOG_NAME: str = ".overflow-audit.log"

_CROCKFORD: str = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


class CaptureQueueError(Exception):
    pass


class CaptureQueueSchemaError(CaptureQueueError):
    pass


class CaptureQueueLocked(CaptureQueueError):
    pass


_ulid_lock = threading.Lock()
_last_ms: int = 0


def _now_ms() -> int:
    return int(time.time() * 1000)


def _b32_encode(data: bytes, length: int) -> str:
    n = int.from_bytes(data, "big")
    out = []
    for _ in range(length):
        out.append(_CROCKFORD[n & 0x1F])
        n >>= 5
    return "".join(reversed(out))


def generate_ulid() -> str:
    global _last_ms
    with _ulid_lock:
        ms = _now_ms()
        if ms <= _last_ms:
            ms = _last_ms + 1
        _last_ms = ms

    ts_bytes = ms.to_bytes(6, "big")
    rand_bytes = secrets.token_bytes(10)
    return _b32_encode(ts_bytes, 10) + _b32_encode(rand_bytes, 16)


class CaptureQueue:

    def __init__(
        self,
        queue_dir: Path | None = None,
        max_size: int = DEFAULT_MAX_SIZE,
    ) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self._queue_dir = (
            Path(queue_dir) if queue_dir is not None else DEFAULT_QUEUE_DIR
        )
        self._queue_dir.mkdir(parents=True, exist_ok=True)
        self._max_size = max_size
        self._audit_log = self._queue_dir / _AUDIT_LOG_NAME


    @property
    def queue_dir(self) -> Path:
        return self._queue_dir

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def audit_log_path(self) -> Path:
        return self._audit_log

    def pending_count(self) -> int:
        return sum(1 for _ in self._iter_pending_files())

    def list_pending(self) -> list[Path]:
        return sorted(self._iter_pending_files(), key=lambda p: p.name)

    def _iter_pending_files(self):
        for entry in self._queue_dir.iterdir():
            name = entry.name
            if (
                entry.is_file()
                and name.startswith("pending-")
                and name.endswith(".json")
                and not name.endswith(".json.tmp")
            ):
                yield entry


    def append(self, record: dict) -> str:
        if not isinstance(record, dict):
            raise TypeError(f"record must be a dict, got {type(record).__name__}")

        ulid = generate_ulid()
        appended_at = datetime.now(timezone.utc).isoformat()
        envelope: dict = {
            "ulid": ulid,
            "appended_at": appended_at,
            "record": record,
            "schema_version": SCHEMA_VERSION,
        }

        final_path = self._queue_dir / f"pending-{ulid}.json"
        tmp_path = self._queue_dir / f"pending-{ulid}.json.tmp"

        fd = os.open(
            str(tmp_path),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            payload = json.dumps(
                envelope,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            os.write(fd, payload)
            os.fsync(fd)
        except (OSError, TypeError, ValueError):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        finally:
            os.close(fd)

        os.replace(tmp_path, final_path)

        if self.pending_count() > self._max_size:
            target = max(0, self._max_size - _PRUNE_BATCH_HEADROOM)
            self.prune_oldest(target_size=target)

        return ulid


    def ingest_pending(self, handler: Callable[[dict], None]) -> int:
        if not callable(handler):
            raise TypeError("handler must be callable")

        ingested = 0
        for pending_path in self.list_pending():
            ulid = self._ulid_from_path(pending_path)
            lock_path = self._queue_dir / f"pending-{ulid}.lock"

            try:
                lock_fd = os.open(
                    str(lock_path),
                    os.O_WRONLY | os.O_CREAT,
                    0o600,
                )
            except OSError:
                continue

            try:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                        continue
                    raise

                if not pending_path.exists():
                    continue

                envelope = self._read_envelope(pending_path)
                version = envelope.get("schema_version")
                if version != SCHEMA_VERSION:
                    raise CaptureQueueSchemaError(
                        f"unsupported schema_version={version!r} in "
                        f"{pending_path.name}; expected {SCHEMA_VERSION}",
                    )

                record = envelope["record"]
                handler(record)

                try:
                    os.unlink(pending_path)
                except FileNotFoundError:
                    pass
                ingested += 1
            finally:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(lock_fd)
                if not pending_path.exists():
                    try:
                        os.unlink(lock_path)
                    except FileNotFoundError:
                        pass

        return ingested


    def prune_oldest(self, target_size: int | None = None) -> int:
        if target_size is None:
            target_size = self._max_size
        if target_size < 0:
            raise ValueError(f"target_size must be >= 0, got {target_size}")

        oldest_first = self.list_pending()
        excess = len(oldest_first) - target_size
        if excess <= 0:
            return 0

        queue_size_before = len(oldest_first)
        dropped = 0
        for pending_path in oldest_first[:excess]:
            ulid = self._ulid_from_path(pending_path)
            try:
                envelope = self._read_envelope(pending_path)
                appended_at = envelope.get("appended_at", "")
            except (FileNotFoundError, json.JSONDecodeError, CaptureQueueError):
                appended_at = ""

            try:
                os.unlink(pending_path)
            except FileNotFoundError:
                continue

            self._audit_drop(
                dropped_ulid=ulid,
                appended_at=appended_at,
                queue_size_before_prune=queue_size_before,
            )
            dropped += 1
        return dropped


    @staticmethod
    def _ulid_from_path(path: Path) -> str:
        return path.stem[len("pending-"):]

    @staticmethod
    def _read_envelope(path: Path) -> dict:
        with path.open("rb") as f:
            raw = f.read()
        return json.loads(raw.decode("utf-8"))

    def _audit_drop(
        self,
        *,
        dropped_ulid: str,
        appended_at: str,
        queue_size_before_prune: int,
    ) -> None:
        line = (
            json.dumps(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "dropped_ulid": dropped_ulid,
                    "appended_at": appended_at,
                    "reason": "queue_overflow",
                    "queue_size_before_prune": queue_size_before_prune,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        try:
            fd = os.open(
                str(self._audit_log),
                os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                0o600,
            )
        except OSError:
            return
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                os.write(fd, line.encode("utf-8"))
                os.fsync(fd)
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            os.close(fd)


__all__ = [
    "CaptureQueue",
    "CaptureQueueError",
    "CaptureQueueLocked",
    "CaptureQueueSchemaError",
    "DEFAULT_MAX_SIZE",
    "DEFAULT_QUEUE_DIR",
    "SCHEMA_VERSION",
    "generate_ulid",
]
