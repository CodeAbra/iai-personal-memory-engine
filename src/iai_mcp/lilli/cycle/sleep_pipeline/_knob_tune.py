from __future__ import annotations

import logging
from typing import Any, Callable

from iai_mcp.exceptions import StoreError
from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep

logger = logging.getLogger(__name__)


def step_knob_tune(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    from iai_mcp.profile import PROFILE_KNOBS, default_state

    knob_names = sorted(PROFILE_KNOBS.keys())
    snapshot = default_state()
    for chunk_idx, name in enumerate(knob_names):
        if self._check_interrupt(
            SleepStep.KNOB_TUNE, chunk_idx, interrupt_check,
        ):
            return False, {}
        _ = snapshot.get(name)

    try:
        from iai_mcp.user_model import load as _load_um, save as _save_um
        tbl = self._store.db.open_table("edges")
        total_edges = tbl.count_rows()
        curiosity_count = tbl.count_rows("edge_type = 'curiosity_bridge'") if total_edges > 0 else 0
        curiosity_ratio = curiosity_count / max(total_edges, 1)
        # Zero bridges = no signal (generator quiet), not "low curiosity";
        # a dead subsystem must never steer a live knob.
        if curiosity_count > 0 and (curiosity_ratio > 0.1 or curiosity_ratio < 0.02):
            um = _load_um()
            if curiosity_ratio > 0.1:
                um.soft_knobs["monotropism"] = 1.5
            elif curiosity_ratio < 0.02:
                um.soft_knobs["monotropism"] = 0.8
            _save_um(um)
    except (OSError, ValueError, RuntimeError, KeyError, StoreError) as exc:
        logger.debug("non-critical soft_knobs auto-write failed: %s", exc)

    return True, {"knobs_tuned": len(knob_names)}
