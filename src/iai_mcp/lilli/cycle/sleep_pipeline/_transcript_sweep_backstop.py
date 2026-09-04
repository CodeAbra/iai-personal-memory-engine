from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep

logger = logging.getLogger(__name__)


def step_transcript_sweep_backstop(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    from iai_mcp.cli._cowork import _sweeper_enabled_flag_path

    if not _sweeper_enabled_flag_path(Path.home()).exists():
        return True, {"transcript_sweep_backstop": "disabled"}

    if self._check_interrupt(SleepStep.TRANSCRIPT_SWEEP_BACKSTOP, 0, interrupt_check):
        return False, {}

    from iai_mcp.capture import drain_capture_backlog
    from iai_mcp.transcript_sweep import sweep_once

    try:
        summary = sweep_once()
        drain_counts = drain_capture_backlog(self._store)
    except Exception as exc:  # noqa: BLE001 -- a backstop failure must never fail the cycle
        logger.warning("transcript_sweep_backstop step degraded: %s", exc)
        return True, {"transcript_sweep_backstop_error": type(exc).__name__}

    return True, {
        "files_seen": summary.get("files_seen", 0),
        "sessions_staged": summary.get("sessions_staged", 0),
        "lines_staged": summary.get("lines_staged", 0),
        "drain_counts": drain_counts,
    }
