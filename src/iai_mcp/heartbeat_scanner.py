from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


DEFAULT_STALE_THRESHOLD_SEC = 90

IDLE_WINDOW_SEC = 30 * 60

_HEARTBEAT_GLOB = "heartbeat-*.json"


class HeartbeatStatus(Enum):

    FRESH = "fresh"
    STALE = "stale"
    ORPHAN = "orphan"


@dataclass
class HeartbeatEntry:

    path: Path
    pid: int
    uuid: str
    last_refresh: datetime
    status: HeartbeatStatus


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    # NOT os.kill(pid, 0) first: on Windows os.kill rejects signal 0 with
    # OSError [WinError 87] (invalid parameter) even for a live PID, which made
    # this scanner (and doctor check (m)) fail against a healthy daemon.
    # psutil.pid_exists is correct and cross-platform (psutil is a hard dep).
    try:
        import psutil

        return psutil.pid_exists(pid)
    except Exception:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _parse_heartbeat_file(path: Path) -> tuple[int, str, datetime] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return _fallback_parse_from_filename(path)

    pid = payload.get("pid")
    uuid_str = payload.get("uuid", "")
    last_refresh_raw = payload.get("last_refresh")

    if not isinstance(pid, int) or not isinstance(uuid_str, str):
        return _fallback_parse_from_filename(path)
    if not isinstance(last_refresh_raw, str):
        return _fallback_parse_from_filename(path)

    try:
        normalized = last_refresh_raw.replace("Z", "+00:00")
        last_refresh = datetime.fromisoformat(normalized)
    except ValueError:
        return _fallback_parse_from_filename(path)

    if last_refresh.tzinfo is None:
        last_refresh = last_refresh.replace(tzinfo=timezone.utc)
    else:
        last_refresh = last_refresh.astimezone(timezone.utc)

    return pid, uuid_str, last_refresh


def _fallback_parse_from_filename(path: Path) -> tuple[int, str, datetime] | None:
    name = path.stem
    parts = name.split("-", 2)
    if len(parts) != 3 or parts[0] != "heartbeat":
        return None
    try:
        pid = int(parts[1])
    except ValueError:
        return None
    uuid_str = parts[2]
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return None
    return pid, uuid_str, datetime.fromtimestamp(mtime, tz=timezone.utc)


class HeartbeatScanner:

    def __init__(
        self,
        wrappers_dir: Path,
        stale_threshold_sec: int = DEFAULT_STALE_THRESHOLD_SEC,
    ) -> None:
        self._wrappers_dir = wrappers_dir
        self._stale_threshold_sec = stale_threshold_sec
        self._last_scan: list[HeartbeatEntry] = []


    def scan(self) -> list[HeartbeatEntry]:
        entries: list[HeartbeatEntry] = []
        if not self._wrappers_dir.exists():
            self._last_scan = entries
            return entries

        try:
            candidates = list(self._wrappers_dir.glob(_HEARTBEAT_GLOB))
        except OSError:
            self._last_scan = entries
            return entries

        now = datetime.now(timezone.utc)
        for path in candidates:
            parsed = _parse_heartbeat_file(path)
            if parsed is None:
                continue
            pid, uuid_str, last_refresh = parsed

            age_sec = (now - last_refresh).total_seconds()
            is_alive = _is_pid_alive(pid)

            if age_sec > self._stale_threshold_sec:
                status = HeartbeatStatus.STALE
            elif not is_alive:
                status = HeartbeatStatus.ORPHAN
            else:
                status = HeartbeatStatus.FRESH

            entries.append(
                HeartbeatEntry(
                    path=path,
                    pid=pid,
                    uuid=uuid_str,
                    last_refresh=last_refresh,
                    status=status,
                )
            )

        self._last_scan = entries
        return entries


    def fresh_count(self) -> int:
        return sum(1 for e in self.scan() if e.status is HeartbeatStatus.FRESH)

    def is_active(self) -> bool:
        return self.fresh_count() >= 1

    def heartbeat_idle_30min(self) -> bool:
        return self.fresh_count() == 0


    def cleanup_stale_orphans(self) -> int:
        deleted = 0
        for entry in self.scan():
            if entry.status is HeartbeatStatus.FRESH:
                continue
            try:
                entry.path.unlink()
                deleted += 1
            except FileNotFoundError:
                deleted += 1
            except OSError:
                continue
        return deleted
