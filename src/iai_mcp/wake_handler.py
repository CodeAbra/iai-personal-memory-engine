from __future__ import annotations

import platform
import time
from pathlib import Path


__all__ = ["WakeHandler"]

_IS_WINDOWS: bool = platform.system() == "Windows"

# Windows os.unlink fails with PermissionError (WinError 5/ACCESS_DENIED or
# 32/SHARING_VIOLATION) when another process momentarily holds the target
# open -- e.g. the CLI/hook atomically replacing the signal, or `daemon
# status`/the MCP server reading it, at the same instant the daemon consumes
# it. Python's open() on Windows does not request FILE_SHARE_DELETE, so any
# concurrent handle transiently blocks the unlink. The handle is held only
# briefly, so a few short retries resolve it. POSIX unlink never sees this.
# Mirrors iai_mcp.daemon_state._atomic_replace.
_UNLINK_RETRY_ATTEMPTS: int = 10
_UNLINK_RETRY_SLEEP_SEC: float = 0.05


class WakeHandler:

    def __init__(self, wake_signal_path: Path) -> None:
        self._wake_signal_path = wake_signal_path

    def consume_wake_signal(self) -> bool:
        for attempt in range(_UNLINK_RETRY_ATTEMPTS):
            try:
                self._wake_signal_path.unlink()
                return True
            except FileNotFoundError:
                return False
            except PermissionError:
                # Transient Windows sharing violation: the signal exists but a
                # concurrent writer/reader holds it open. Returning False here
                # would drop a real wake signal and strand the daemon in
                # HIBERNATION with an unserved socket, so retry briefly. On
                # POSIX a PermissionError is a genuine permission fault -- do
                # not retry.
                if not _IS_WINDOWS or attempt == _UNLINK_RETRY_ATTEMPTS - 1:
                    return False
                time.sleep(_UNLINK_RETRY_SLEEP_SEC)
            except OSError:
                return False
        return False

    def signal_wake(self) -> bool:
        """Create the wake signal so the next daemon boot enters WAKE.

        Symmetric counterpart to ``consume_wake_signal``. Whoever brings the
        daemon up from a hibernated (process-exited) state — the CLI
        start/install path, or the per-turn capture hook — must create this
        signal *before* the kickstart, so the booting daemon transitions
        HIBERNATION -> WAKE and serves its socket, instead of re-reading the
        persisted HIBERNATION state and immediately hibernate-exiting (which
        closes the socket and leaves recall unserved). Idempotent.
        """
        try:
            self._wake_signal_path.parent.mkdir(parents=True, exist_ok=True)
            self._wake_signal_path.touch()
        except OSError:
            return False
        return True

    def has_pending_wake(self) -> bool:
        try:
            return self._wake_signal_path.is_file()
        except OSError:
            return False
