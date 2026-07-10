from __future__ import annotations

import os
from pathlib import Path


__all__ = ["WakeHandler", "wake_signal_path"]

WAKE_SIGNAL_FILENAME: str = "wake.signal"


def wake_signal_path(store_root: Path | str | None = None) -> Path:
    """Resolve the wake-signal file path under the active store root.

    Mirrors ``lifecycle_state_path``/``daemon_state_path``: an explicit
    ``store_root`` wins; absent that, the ``IAI_MCP_STORE`` environment
    variable is honored (read at call time); absent both, the home-rooted
    default is returned, keeping default resolution byte-identical to the
    pre-existing hard-coded path.
    """
    if store_root is not None:
        return Path(store_root) / WAKE_SIGNAL_FILENAME
    env = os.environ.get("IAI_MCP_STORE")
    if env:
        return Path(env) / WAKE_SIGNAL_FILENAME
    return Path.home() / ".iai-mcp" / WAKE_SIGNAL_FILENAME


class WakeHandler:

    def __init__(self, wake_signal_path: Path) -> None:
        self._wake_signal_path = wake_signal_path

    def consume_wake_signal(self) -> bool:
        try:
            self._wake_signal_path.unlink()
        except FileNotFoundError:
            return False
        except OSError:
            return False
        return True

    def has_pending_wake(self) -> bool:
        try:
            return self._wake_signal_path.is_file()
        except OSError:
            return False
