from __future__ import annotations

import logging
from typing import Any, Callable

from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep

logger = logging.getLogger(__name__)


def step_salience_auto_migration(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    # Mirrors every sibling step's interrupt contract: the prior step's own
    # _save_progress already pinned the resume index, so (False, {}) re-enters
    # only this tail step next cycle -- (True, ...) here would advance the
    # saved index past this step without having run it.
    if self._check_interrupt(SleepStep.SALIENCE_AUTO_MIGRATION, 0, interrupt_check):
        return False, {}

    if self._store is None:
        return True, {"status": "no_store"}

    from iai_mcp.migrate._salience_auto_migration import run_salience_auto_migration

    try:
        result = run_salience_auto_migration(self._store)
    except Exception as exc:  # noqa: BLE001 -- must never raise into the pipeline (3-strike quarantine)
        logger.warning("salience auto-migration step failed: %s", exc)
        return True, {"status": "failed", "error": str(exc)}

    return True, {
        "status": result.get("status"),
        "records_scanned": result.get("records_scanned", 0),
        "salience_promoted": result.get("salience_promoted", 0),
    }
