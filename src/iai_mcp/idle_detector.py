from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


_IOREG_BIN = "/usr/sbin/ioreg"

_PMSET_BIN = "/usr/bin/pmset"

_BUSCTL_BIN = "/usr/bin/busctl"

_IOREG_TIMEOUT_SEC = 5

_PMSET_TIMEOUT_SEC = 10

_BUSCTL_TIMEOUT_SEC = 5

_PMSET_TAIL_LINES = 200

_HID_IDLE_RE = re.compile(r'"HIDIdleTime"\s*=\s*(\d+)')

_PMSET_SLEEP_MARKERS = ("System Sleep", "Display is turned off")

_PMSET_DEFAULT_WINDOW_MIN = 5

_HID_IDLE_THRESHOLD_SEC = 30 * 60

_PMSET_TS_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+([+-]\d{4})"
)

_PMSET_TS_FMT = "%Y-%m-%d %H:%M:%S"


@dataclass
class IdleStatus:

    hid_idle_sec: int | None = None
    pmset_recent_sleep: bool = False
    available_signals: list[str] = field(default_factory=list)


class IdleDetector:


    def hid_idle_time_sec(self) -> int | None:
        try:
            result = subprocess.run(
                [_IOREG_BIN, "-c", "IOHIDSystem"],
                capture_output=True,
                text=True,
                timeout=_IOREG_TIMEOUT_SEC,
                check=False,
            )
        except FileNotFoundError:
            return None
        except subprocess.TimeoutExpired:
            return None
        except OSError:
            return None

        if result.returncode != 0:
            return None

        match = _HID_IDLE_RE.search(result.stdout or "")
        if match is None:
            return None
        try:
            ns = int(match.group(1))
        except ValueError:
            return None
        if ns < 0:
            return None
        return ns // 1_000_000_000


    def pmset_recent_sleep(
        self, window_min: int = _PMSET_DEFAULT_WINDOW_MIN
    ) -> bool:
        try:
            result = subprocess.run(
                [_PMSET_BIN, "-g", "log"],
                capture_output=True,
                text=True,
                timeout=_PMSET_TIMEOUT_SEC,
                check=False,
            )
        except FileNotFoundError:
            return False
        except subprocess.TimeoutExpired:
            return False
        except OSError:
            return False

        if result.returncode != 0:
            return False

        return self._scan_pmset_lines(result.stdout or "", window_min)

    @staticmethod
    def _scan_pmset_lines(stdout: str, window_min: int) -> bool:
        if window_min <= 0:
            return False
        now_utc = datetime.now(timezone.utc)
        cutoff = now_utc - timedelta(minutes=window_min)

        lines = stdout.splitlines()
        tail = lines[-_PMSET_TAIL_LINES:] if len(lines) > _PMSET_TAIL_LINES else lines

        for line in tail:
            if not any(marker in line for marker in _PMSET_SLEEP_MARKERS):
                continue
            ts = _parse_pmset_timestamp(line)
            if ts is None:
                continue
            if ts >= cutoff:
                return True
        return False


    def _busctl_json(self, *args: str) -> object | None:
        try:
            result = subprocess.run(
                [_BUSCTL_BIN, "--json=short", *args],
                capture_output=True,
                text=True,
                timeout=_BUSCTL_TIMEOUT_SEC,
                check=False,
            )
        except FileNotFoundError:
            return None
        except subprocess.TimeoutExpired:
            return None
        except OSError:
            return None

        if result.returncode != 0:
            return None
        try:
            return json.loads(result.stdout or "")
        except json.JSONDecodeError:
            return None

    def _logind_session_paths(self) -> list[str]:
        # Enumerate sessions rather than resolving "the calling process's own
        # session" (e.g. via GetSessionByPID): the daemon runs as a systemd
        # user service, not attached to any interactive session's cgroup, so
        # a self-lookup by PID never resolves. Instead, find every one of the
        # caller's user's seat-attached (interactive, non-headless) sessions
        # -- there can be more than one on real multi-seat hardware, and
        # picking just one arbitrarily risks reading the wrong seat's idle
        # state while another seat is actively in use.
        payload = self._busctl_json(
            "--system", "call",
            "org.freedesktop.login1", "/org/freedesktop/login1",
            "org.freedesktop.login1.Manager", "ListSessions",
        )
        if not isinstance(payload, dict):
            return []
        try:
            entries = payload["data"][0]
        except (KeyError, IndexError, TypeError):
            return []
        if not isinstance(entries, list):
            return []

        target_uid = os.getuid()
        paths: list[str] = []
        for entry in entries:
            try:
                _session_id, uid, _user, seat, path = entry
            except (ValueError, TypeError):
                continue
            if uid == target_uid and seat:
                paths.append(path)
        return paths

    def _logind_get_property(self, session_path: str, prop: str) -> object | None:
        payload = self._busctl_json(
            "--system", "get-property",
            "org.freedesktop.login1", session_path,
            "org.freedesktop.login1.Session", prop,
        )
        if not isinstance(payload, dict):
            return None
        return payload.get("data")

    def _logind_idle_from_session(self, session_path: str) -> int | None:
        idle_hint = self._logind_get_property(session_path, "IdleHint")
        if idle_hint is not True:
            return None

        idle_since_usec = self._logind_get_property(session_path, "IdleSinceHint")
        if not isinstance(idle_since_usec, int) or idle_since_usec <= 0:
            return None

        now_usec = int(datetime.now(timezone.utc).timestamp() * 1_000_000)
        return max(0, (now_usec - idle_since_usec) // 1_000_000)

    def _logind_aggregate_idle(self, session_paths: list[str]) -> int | None:
        # "Idle" means every seat-attached session is idle -- a single
        # active seat means the user isn't idle overall, regardless of how
        # long any other seat has been untouched.
        if not session_paths:
            return None
        idle_times: list[int] = []
        for path in session_paths:
            idle_sec = self._logind_idle_from_session(path)
            if idle_sec is None:
                return None
            idle_times.append(idle_sec)
        return min(idle_times)


    def os_idle_time_sec(self) -> tuple[int | None, str | None]:
        """Platform dispatcher for OS-level idle time.

        Senses the current OS and queries whichever idle source it
        supports. Returns ``(idle_seconds, source_name)``. Functional
        consumers that only need ``idle_seconds`` (e.g. ``sleep_eligible``)
        are fully platform-agnostic -- they never inspect ``source_name``.
        ``source_name`` itself is a platform-specific label (``"HIDIdleTime"``,
        ``"logind"``), set whenever the underlying source was reachable even
        if ``idle_seconds`` is ``None`` (session not currently idle); it
        exists for diagnostic/reporting consumers (``status()``,
        ``describe()``), which is also where any source-name-specific
        formatting belongs -- not in a caller of this class.
        """
        system = platform.system()
        if system == "Darwin":
            idle_sec = self.hid_idle_time_sec()
            return idle_sec, ("HIDIdleTime" if idle_sec is not None else None)
        if system == "Linux":
            session_paths = self._logind_session_paths()
            if not session_paths:
                return None, None
            return self._logind_aggregate_idle(session_paths), "logind"
        return None, None


    def sleep_eligible(self, heartbeat_idle_30min: bool) -> bool:
        if heartbeat_idle_30min:
            return True

        idle_sec, _source = self.os_idle_time_sec()
        if idle_sec is not None and idle_sec >= _HID_IDLE_THRESHOLD_SEC:
            return True

        if platform.system() == "Darwin":
            return self.pmset_recent_sleep()
        return False


    def status(self) -> IdleStatus:
        idle_sec, source = self.os_idle_time_sec()

        signals: list[str] = []
        if source is not None:
            signals.append(source)

        pmset_seen = False
        if platform.system() == "Darwin":
            pmset_seen = self.pmset_recent_sleep()
            if _pmset_responsive():
                signals.append("pmset")

        return IdleStatus(
            hid_idle_sec=idle_sec,
            pmset_recent_sleep=pmset_seen,
            available_signals=signals,
        )


    def describe(self) -> tuple[str, str]:
        """Human-readable ``(detail, status)`` for doctor's idle-source
        health check. Callers don't need to know which platform-specific
        backend (HIDIdleTime, logind, ...) is actually in use -- that
        knowledge stays here, next to the platform dispatch itself.
        """
        status = self.status()

        def idle_str(none_label: str) -> str:
            return (
                f"{status.hid_idle_sec}s"
                if status.hid_idle_sec is not None
                else none_label
            )

        signals_str = (
            ",".join(status.available_signals) if status.available_signals else "none"
        )
        pmset_str = "recent-sleep" if status.pmset_recent_sleep else "clean"

        if "HIDIdleTime" in status.available_signals:
            detail = (
                f"HIDIdleTime: {idle_str('unavailable')}, pmset: {pmset_str}, "
                f"available: {signals_str}"
            )
            return detail, "PASS"
        if "logind" in status.available_signals:
            detail = (
                f"logind IdleHint: {idle_str('not idle')}, available: {signals_str}"
            )
            return detail, "PASS"
        detail = (
            f"HIDIdleTime: {idle_str('unavailable')}, pmset: {pmset_str}, "
            f"available: {signals_str}; L6 will fall back to heartbeat-idle only"
        )
        return detail, "WARN"


def _parse_pmset_timestamp(line: str) -> datetime | None:
    m = _PMSET_TS_RE.match(line)
    if m is None:
        return None
    ts_str, offset_str = m.group(1), m.group(2)
    try:
        naive = datetime.strptime(ts_str, _PMSET_TS_FMT)
    except ValueError:
        return None
    sign = 1 if offset_str[0] == "+" else -1
    try:
        hours = int(offset_str[1:3])
        minutes = int(offset_str[3:5])
    except ValueError:
        return None
    offset = timedelta(hours=hours, minutes=minutes) * sign
    return (naive - offset).replace(tzinfo=timezone.utc)


def _pmset_responsive() -> bool:
    try:
        result = subprocess.run(
            [_PMSET_BIN, "-g"],
            capture_output=True,
            text=True,
            timeout=_PMSET_TIMEOUT_SEC,
            check=False,
        )
    except FileNotFoundError:
        return False
    except subprocess.TimeoutExpired:
        return False
    except OSError:
        return False
    return result.returncode == 0
