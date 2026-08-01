from __future__ import annotations

import logging
import os
from typing import Any, Callable

from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep

logger = logging.getLogger(__name__)


def step_curiosity_mine(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    if os.environ.get("IAI_MCP_CURIOSITY_MINE_OFF") == "true":
        return True, {"curiosity_mine": "disabled"}
    if self._check_interrupt(SleepStep.CURIOSITY_MINE, 0, interrupt_check):
        return False, {}
    from iai_mcp.curiosity import process_deferred_inputs

    try:
        result = process_deferred_inputs(self._store)
    except Exception as exc:  # noqa: BLE001 -- advisory step must never fail the cycle
        logger.warning("curiosity_mine step degraded: %s", exc)
        return True, {"curiosity_mine_error": type(exc).__name__}
    return True, result
