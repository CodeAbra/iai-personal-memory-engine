from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Callable

from iai_mcp.exceptions import StoreError
from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep

logger = logging.getLogger(__name__)


def step_erasure_agent(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    if self._check_interrupt(
        SleepStep.ERASURE_AGENT, 0, interrupt_check,
    ):
        return False, {}

    from iai_mcp.daemon_config import _load_erasure_config
    try:
        from iai_mcp.sleep_wal import SleepWAL
        _wal = SleepWAL()
    except ImportError:
        _wal = None
    cfg = _load_erasure_config()
    threshold = cfg.centrality_threshold
    age_days = cfg.age_days
    window_days = cfg.retrieval_window_days
    dry_run = cfg.dry_run

    now = self._now()
    age_cutoff = now - timedelta(days=age_days)
    window_cutoff = now - timedelta(days=window_days)

    from iai_mcp.store import RECORDS_TABLE
    tbl = self._store.db.open_table(RECORDS_TABLE)

    window_cutoff_str = window_cutoff.strftime("%Y-%m-%d %H:%M:%S")
    age_cutoff_str = age_cutoff.strftime("%Y-%m-%d %H:%M:%S")
    eligibility_where = (
        f"centrality < {threshold} "
        f"AND (last_reviewed IS NULL OR "
        f"last_reviewed < '{window_cutoff_str}') "
        f"AND created_at < '{age_cutoff_str}' "
        f"AND pinned = false "
        f"AND never_decay = false "
        f"AND tombstoned_at IS NULL"
    )

    try:
        count_quarantined = int(tbl.count_rows(filter=eligibility_where))
    except (OSError, ValueError, RuntimeError, StoreError) as exc:
        logger.debug("erasure_agent count_rows failed: %s", exc)
        count_quarantined = 0
    total_records_after = int(tbl.count_rows())

    from iai_mcp.events import query_events, write_event
    prior_drops = query_events(
        self._store, kind="erasure_optimize_drops", limit=1,
    )
    count_dropped = 0
    if prior_drops:
        prior_body = prior_drops[0].get("data") or {}
        count_dropped = int(prior_body.get("count_dropped", 0) or 0)

    if not dry_run and count_quarantined > 0:
        try:
            tbl.update(
                where=eligibility_where,
                values={"tombstoned_at": now, "live": 0},
            )
        except Exception as exc:  # noqa: BLE001 -- visibility over crash
            logger.error("erasure_agent tombstone mutation failed: %s", exc, exc_info=True)
            write_event(
                self._store,
                "erasure_agent_pass",
                {
                    "count_quarantined": int(count_quarantined),
                    "count_dropped": int(count_dropped),
                    "total_records_after": int(total_records_after),
                    "threshold_used": float(threshold),
                    "dry_run_mode": bool(dry_run),
                    "mutation_error": str(exc)[:500],
                },
                severity="warning",
            )
            raise
        # The tombstone succeeded: rows moved from active to tombstoned, so the
        # active count decreased.  Invalidate the cached active count so the next
        # active_records_count() call recomputes from the post-tombstone SQL state.
        try:
            _inv = getattr(self._store, "_invalidate_corpus_count", None)
            if _inv is not None:
                _inv("active")
        except Exception:  # noqa: BLE001 -- invalidation must not crash a sleep step
            pass
        # Rows just moved from active to tombstoned: invalidate the resident
        # exact-cosine matrix so it never continues to serve a now-tombstoned
        # id — the next exact_top_k rebuilds tombstone-filtered.
        try:
            _inv_x = getattr(self._store, "invalidate_exact_index", None)
            if callable(_inv_x):
                _inv_x()
        except Exception:  # noqa: BLE001 -- invalidation must not crash a sleep step
            pass

    write_event(
        self._store,
        "erasure_agent_pass",
        {
            "count_quarantined": int(count_quarantined),
            "count_dropped": int(count_dropped),
            "total_records_after": int(total_records_after),
            "threshold_used": float(threshold),
            "dry_run_mode": bool(dry_run),
        },
        severity="info",
    )

    return True, {
        "count_quarantined": int(count_quarantined),
        "dry_run": bool(dry_run),
    }
