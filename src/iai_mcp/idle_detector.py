from __future__ import annotations

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

_LOGIND_SESSION_PATH_RE = re.compile(r'"(/org/freedesktop/login1/session/[^"]+)"')

_LOGIND_BOOL_RE = re.compile(r"\bb\s+(true|false)\b")

_LOGIND_UINT64_RE = re.compile(r"\bt\s+(\d+)")

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


    def _logind_session_path(self) -> str | None:
        try:
            result = subprocess.run(
                [
                    _BUSCTL_BIN, "--system", "call",
                    "org.freedesktop.login1", "/org/freedesktop/login1",
                    "org.freedesktop.login1.Manager", "GetSessionByPID",
                    "u", str(os.getpid()),
                ],
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
        match = _LOGIND_SESSION_PATH_RE.search(result.stdout or "")
        return match.group(1) if match else None

    def _logind_get_property(self, session_path: str, prop: str) -> str | None:
        try:
            result = subprocess.run(
                [
                    _BUSCTL_BIN, "--system", "get-property",
                    "org.freedesktop.login1", session_path,
                    "org.freedesktop.login1.Session", prop,
                ],
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
        return result.stdout or ""

    def _logind_idle_from_session(self, session_path: str) -> int | None:
        hint_out = self._logind_get_property(session_path, "IdleHint")
        if hint_out is None:
            return None
        hint_match = _LOGIND_BOOL_RE.search(hint_out)
        if hint_match is None or hint_match.group(1) != "true":
            return None

        since_out = self._logind_get_property(session_path, "IdleSinceHint")
        if since_out is None:
            return None
        since_match = _LOGIND_UINT64_RE.search(since_out)
        if since_match is None:
            return None
        idle_since_usec = int(since_match.group(1))
        if idle_since_usec <= 0:
            return None

        now_usec = int(datetime.now(timezone.utc).timestamp() * 1_000_000)
        return max(0, (now_usec - idle_since_usec) // 1_000_000)

    def logind_idle_time_sec(self) -> int | None:
        session_path = self._logind_session_path()
        if session_path is None:
            return None
        return self._logind_idle_from_session(session_path)


    def os_idle_time_sec(self) -> tuple[int | None, str | None]:
        """Platform dispatcher for OS-level idle time.

        Senses the current OS and queries whichever idle source it
        supports, so callers never need their own platform checks. Returns
        ``(idle_seconds, source_name)``. ``source_name`` is set whenever the
        underlying source was reachable, even if ``idle_seconds`` is ``None``
        (session not currently idle) -- this lets callers distinguish
        "signal available" from "signal unavailable" without knowing which
        platform they're on.
        """
        system = platform.system()
        if system == "Darwin":
            idle_sec = self.hid_idle_time_sec()
            return idle_sec, ("HIDIdleTime" if idle_sec is not None else None)
        if system == "Linux":
            session_path = self._logind_session_path()
            if session_path is None:
                return None, None
            return self._logind_idle_from_session(session_path), "logind"
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
