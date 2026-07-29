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
        target: float | None = None
        if curiosity_ratio > 0.1:
            target = 1.5
        elif curiosity_ratio < 0.02:
            target = 0.8
        if target is not None:
            um = _load_um()
            # Idempotent write: only persist when the knob actually moves.
            # The prior code re-saved the user model on every sleep cycle
            # because a corpus with no ``curiosity_bridge`` edges keeps
            # ``curiosity_ratio`` pinned at 0.0 (< 0.02) forever, so the knob
            # was rewritten to 0.8 every cycle -- churn the nightly report kept
            # surfacing. Guarding on the current value stops the rewrite once
            # the knob has settled.
            if um.soft_knobs.get("monotropism") != target:
                um.soft_knobs["monotropism"] = target
                _save_um(um)
    except (OSError, ValueError, RuntimeError, KeyError, StoreError) as exc:
        logger.debug("non-critical soft_knobs auto-write failed: %s", exc)

    try:
        from iai_mcp.gaba_annealing import compute_annealed_k, should_normalize
        cycle_count = self._cycle_counter
        annealed_k = compute_annealed_k(cycle_count)
        if should_normalize(cycle_count):
            logger.debug("GABA: k=%d at cycle %d, normalization due", annealed_k, cycle_count)
    except (ImportError, AttributeError, TypeError) as exc:
        logger.debug("GABA annealing skipped: %s", exc)

    return True, {"knobs_tuned": len(knob_names)}
