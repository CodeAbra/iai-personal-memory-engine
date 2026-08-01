from __future__ import annotations

import logging
from typing import Any, Callable

from iai_mcp.exceptions import StoreError
from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep

logger = logging.getLogger(__name__)

def step_user_model_update(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    if self._check_interrupt(
        SleepStep.USER_MODEL_UPDATE, 0, interrupt_check,
    ):
        return False, {}

    from iai_mcp.daemon_config import _load_user_model_config
    from iai_mcp.user_model import UserModelAggregator, save
    from iai_mcp.events import write_event

    cfg = _load_user_model_config()
    agg = UserModelAggregator()
    model = agg.aggregate(
        self._store, window_days=cfg.aggregation_window_days,
    )

    if not cfg.dry_run:
        try:
            save(model)
        except (OSError, ValueError, RuntimeError, StoreError) as exc:
            logger.warning("user_model_update save failed: %s", exc, exc_info=True)
            write_event(
                self._store,
                "user_model_aggregate_pass",
                {
                    "topics_count": int(len(model.top_recent_topics)),
                    "tools_count": int(len(model.tool_usage_freq)),
                    "hours_count": int(len(model.time_of_day_pattern)),
                                        "window_days": int(cfg.aggregation_window_days),
                    "dry_run_mode": False,
                    "persist_error": str(exc)[:500],
                },
                severity="warning",
            )
            return True, {
                "topics_count": int(len(model.top_recent_topics)),
                "dry_run": False,
                "persist_error": True,
            }

    write_event(
        self._store,
        "user_model_aggregate_pass",
        {
            "topics_count": int(len(model.top_recent_topics)),
            "tools_count": int(len(model.tool_usage_freq)),
            "hours_count": int(len(model.time_of_day_pattern)),
                        "window_days": int(cfg.aggregation_window_days),
            "dry_run_mode": bool(cfg.dry_run),
        },
        severity="info",
    )

    return True, {
        "topics_count": int(len(model.top_recent_topics)),
        "dry_run": bool(cfg.dry_run),
    }
