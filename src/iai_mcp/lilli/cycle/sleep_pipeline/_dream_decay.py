from __future__ import annotations

import logging
from typing import Any, Callable

from iai_mcp.exceptions import StoreError
from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep

logger = logging.getLogger(__name__)


def step_dream_decay(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    from iai_mcp.sleep import _decay_edges

    if self._check_interrupt(SleepStep.DREAM_DECAY, 0, interrupt_check):
        return False, {}
    _plasticity = 1.0
    try:
        from iai_mcp.user_model import load as _load_um
        _um = _load_um()
        _plasticity = getattr(_um, "plasticity_gain", 1.0) or 1.0
    except (OSError, ValueError, RuntimeError, StoreError, AttributeError) as exc:
        logger.debug("non-critical plasticity_gain load failed: %s", exc)

    # Deferral between chunks reuses the standard bookkeeping; committed
    # chunks stay applied (their edges carry a fresh updated_at, so the
    # resumed run skips them via the grace window — restart-idempotent).
    def _defer(chunk_no: int) -> bool:
        return self._check_interrupt(
            SleepStep.DREAM_DECAY, chunk_no, interrupt_check,
        )

    result = _decay_edges(
        self._store, plasticity_gain=_plasticity, should_defer=_defer,
    )
    if isinstance(result, dict):
        if result.get("interrupted"):
            return False, {}
        return True, {
            "decayed": int(result.get("decayed", 0) or 0),
            "pruned": int(result.get("pruned", 0) or 0),
        }
    return True, {}
