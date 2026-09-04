from __future__ import annotations

import logging
import uuid as _uuid
from typing import Any, Callable

from iai_mcp.exceptions import StoreError
from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep
from iai_mcp.lilli.ops.reconsolidation import (
    STABILITY_BOOST_ON_RECALL,
    ReconsolidationBuffer,
)

logger = logging.getLogger(__name__)


def step_reconsolidation_valence(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    if self._check_interrupt(
        SleepStep.RECONSOLIDATION_VALENCE, 0, interrupt_check,
    ):
        return False, {}

    from iai_mcp.daemon_config import _load_reconsolidation_valence_config
    cfg = _load_reconsolidation_valence_config()
    if not cfg.enabled:
        return True, {"candidates_labile": 0, "valence_writes": 0, "valence_saturated": 0}

    now = self._now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    buffer = ReconsolidationBuffer()
    candidates = 0
    try:
        for chunk_idx, row in enumerate(
            self._store.iter_record_columns(
                ["id"],
                batch_size=1024,
                where=f"labile_until > '{now_str}'",
            ),
            start=1,
        ):
            if self._check_interrupt(
                SleepStep.RECONSOLIDATION_VALENCE, chunk_idx, interrupt_check,
            ):
                return False, {}
            candidates += 1
            try:
                rid = _uuid.UUID(str(row["id"]))
            except (TypeError, ValueError):
                continue
            buffer.enter_labile(rid, context="retrieval reinforcement")
            buffer.modify_valence(
                rid, STABILITY_BOOST_ON_RECALL, reason="retrieval reinforcement",
            )
    except (OSError, ValueError, RuntimeError, StoreError) as exc:
        logger.debug("reconsolidation_valence labile query failed: %s", exc)

    writes, saturated = buffer.persist_valence(self._store)
    buffer.close_expired()

    return True, {
        "candidates_labile": int(candidates),
        "valence_writes": int(writes),
        "valence_saturated": int(saturated),
    }
