from __future__ import annotations

import json
import logging
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

log = logging.getLogger(__name__)


def _default_lock_path() -> Path:
    env_path = os.environ.get("IAI_MCP_STORE")
    root = Path(env_path) if env_path else (Path.home() / ".iai-mcp")
    return root / ".locked"


DEFAULT_LOCK_PATH: Path = _default_lock_path()

SCHEMA_VERSION: int = 1


class LifecycleLockConflict(RuntimeError):

    def __init__(self, message: str, existing: "LockPayload | None" = None) -> None:
        super().__init__(message)
        self.existing = existing


class LockPayload(TypedDict):

    pid: int
    hostname: str
    started_at: str
    schema_version: int


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _current_hostname() -> str:
    return socket.gethostname()


def pid_exists(pid: int) -> bool:
    """Generic pid liveness, safe on every platform.

    psutil-first: the POSIX ``os.kill(pid, 0)`` idiom raises ``OSError``
    (WinError 87) for a HEALTHY pid on Windows, so a kill-0-only probe reads
    live processes as dead вЂ” lock stealers, crash-quarantine misjudgments,
    scanner crashes. The kill-0 fallback survives only for a stripped
    environment and never declares death on an unreliable probe result.
    """
    if pid <= 0:
        return False
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            return bool(psutil.pid_exists(pid))
        except Exception:  # noqa: BLE001 -- defensive against backend quirks
            return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False

    try:
        import psutil
    except ImportError:
        psutil = None

    if psutil is None:
        # POSIX-only probe: on Windows os.kill(pid, 0) raises OSError
        # (WinError 87) for a HEALTHY pid вЂ” a healthy daemon would read as
        # dead. psutil (a hard dependency) is the reliable path; this branch
        # survives only for a stripped environment.
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            # Unreliable probe result вЂ” never declare death on it.
            return True
        log.debug(
            "lifecycle_lock: psutil unavailable; falling back to "
            "os.kill-only liveness for pid=%d",
            pid,
        )
        return True

    try:
        if not psutil.pid_exists(pid):
            return False
    except Exception:  # noqa: BLE001 -- defensive against psutil backend quirks
        log.debug(
            "lifecycle_lock: psutil.pid_exists(%d) raised; assuming live",
            pid,
            exc_info=True,
        )
        return True

    try:
        proc = psutil.Process(pid)
        cmdline = " ".join(proc.cmdline() or [])
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False
    except Exception:  # noqa: BLE001 -- defensive against psutil backend quirks
        log.debug(
            "lifecycle_lock: psutil.Process(%d).cmdline() raised "
            "unexpectedly; assuming live",
            pid,
            exc_info=True,
        )
        return True

    return "iai_mcp.daemon" in cmdline


def _validate_payload(raw: object) -> LockPayload:
    if not isinstance(raw, dict):
        raise ValueError(
            f"lockfile payload must be a JSON object, got {type(raw).__name__}"
        )
    pid = raw.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        raise ValueError(f"lockfile.pid must be a positive int, got {pid!r}")
    hostname = raw.get("hostname")
    if not isinstance(hostname, str) or not hostname:
        raise ValueError(
            f"lockfile.hostname must be a non-empty string, got {hostname!r}"
        )
    started_at = raw.get("started_at")
    if not isinstance(started_at, str) or not started_at:
        raise ValueError(
            f"lockfile.started_at must be a non-empty string, got {started_at!r}"
        )
    sv = raw.get("schema_version")
    if not isinstance(sv, int) or sv <= 0:
        raise ValueError(
            f"lockfile.schema_version must be a positive int, got {sv!r}"
        )
    return {
        "pid": pid,
        "hostname": hostname,
        "started_at": started_at,
        "schema_version": sv,
    }


class LifecycleLock:

    def __init__(self, lock_path: Path | None = None) -> None:
        self._lock_path = (
            lock_path if lock_path is not None else _default_lock_path()
        )

    @property
    def lock_path(self) -> Path:
        return self._lock_path

    def read(self) -> LockPayload | None:
        if not self._lock_path.exists():
            return None
        try:
            raw = json.loads(self._lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        try:
            return _validate_payload(raw)
        except ValueError:
            return None

    def is_held_by_self(self) -> bool:
        payload = self.read()
        if payload is None:
            return False
        return (
            payload["pid"] == os.getpid()
            and payload["hostname"] == _current_hostname()
        )

    def acquire(self) -> None:
        # Atomic claim: O_CREAT|O_EXCL means exactly ONE racer creates the
        # file. The prior read->check->replace shape let two daemons both
        # pass the liveness check and both install their payload вЂ” a
        # double-writer on the same store. A stale lock (dead pid, corrupt
        # payload) is unlinked and the claim retried; only one racer wins the
        # recreate.
        payload: LockPayload = {
            "pid": os.getpid(),
            "hostname": _current_hostname(),
            "started_at": _utc_now_iso(),
            "schema_version": SCHEMA_VERSION,
        }
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)

        for _attempt in range(3):
            try:
                fd = os.open(
                    str(self._lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                existing = self.read()
                if existing is not None:
                    if existing["hostname"] == _current_hostname() and _is_pid_alive(
                        existing["pid"]
                    ):
                        raise LifecycleLockConflict(
                            f"daemon already running: pid={existing['pid']} "
                            f"hostname={existing['hostname']} "
                            f"started_at={existing['started_at']}",
                            existing=existing,
                        )
                # Stale or unreadable: clear it and retry the atomic claim.
                try:
                    os.unlink(self._lock_path)
                except FileNotFoundError:
                    pass
                continue
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
            except (OSError, TypeError, ValueError):
                try:
                    os.unlink(self._lock_path)
                except OSError:
                    pass
                raise
            return

        raise LifecycleLockConflict(
            "lifecycle lock claim lost three consecutive races; "
            "another daemon is starting on this store",
            existing=self.read(),
        )

    def release(self) -> None:
        try:
            self._lock_path.unlink()
        except FileNotFoundError:
            return

    def force_unlock(self) -> LockPayload | None:
        previous = self.read()
        try:
            self._lock_path.unlink()
        except FileNotFoundError:
            pass
        return previous

