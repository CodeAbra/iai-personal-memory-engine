from __future__ import annotations

import logging
import os
from typing import Any, Callable

from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep

logger = logging.getLogger(__name__)


def step_entity_link(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    if os.environ.get("IAI_MCP_ENTITY_LINK_OFF") == "true":
        return True, {"entity_link": "disabled"}
    if self._check_interrupt(SleepStep.ENTITY_LINK, 0, interrupt_check):
        return False, {}
    from iai_mcp.entity_link import mine_entity_edges

    try:
        result = mine_entity_edges(self._store)
    except Exception as exc:  # noqa: BLE001 -- enrichment step must never fail the cycle
        logger.warning("entity_link step degraded: %s", exc)
        return True, {"entity_link_error": type(exc).__name__}
    return True, result
