from __future__ import annotations

import asyncio

from iai_mcp.events import write_event
from iai_mcp.guard import BudgetLedger, RateLimitLedger
from iai_mcp.schema import induce_schemas_tier1
from iai_mcp.sleep import SleepConfig, run_heavy_consolidation

REM_CYCLE_MAX_SEC: int = 15 * 60


async def _emit(store, kind: str, data: dict, severity: str | None = None) -> None:
    if severity is None:
        await asyncio.to_thread(write_event, store, kind, data)
    else:
        await asyncio.to_thread(write_event, store, kind, data, severity=severity)


async def run_rem_cycle(
    store,
    cycle_num: int,
    total_cycles: int,
    session_id: str,
    *,
    is_last: bool,
    claude_enabled: bool,
) -> dict:
    await _emit(store, "rem_cycle_started", {"n": cycle_num, "of": total_cycles})

    result: dict = {
        "cycle": cycle_num,
        "summaries_created": 0,
        "schemas_induced": 0,
        "schema_candidates": 0,
        "claude_call_used": False,
        "main_insight_text": None,
        "timed_out": False,
    }

    try:
        async with asyncio.timeout(REM_CYCLE_MAX_SEC):
            cfg = SleepConfig(llm_enabled=False)
            heavy = await asyncio.to_thread(
                run_heavy_consolidation,
                store, session_id, cfg,
                BudgetLedger(store), RateLimitLedger(store),
                False,
            )
            if isinstance(heavy, dict):
                result["summaries_created"] = int(heavy.get("summaries_created", 0) or 0)
                result["schemas_induced"] = int(heavy.get("schemas_induced", 0) or 0)

            candidates = await asyncio.to_thread(
                induce_schemas_tier1,
                store, BudgetLedger(store), RateLimitLedger(store), False,
            )
            result["schema_candidates"] = len(candidates) if candidates else 0

            if is_last and claude_enabled:
                from iai_mcp.insight import generate_overnight_insight

                insight = await generate_overnight_insight(store, session_id)
                if isinstance(insight, dict) and insight.get("ok"):
                    result["claude_call_used"] = True
                    result["main_insight_text"] = insight.get("text")
                elif isinstance(insight, dict):
                    # A skipped/failed insight must say WHY in the cycle
                    # record, or a silent night is indistinguishable from a
                    # gated one.
                    result["insight_skip_reason"] = str(
                        insight.get("reason") or "unknown"
                    )[:200]

    except asyncio.TimeoutError:
        result["timed_out"] = True
        await _emit(
            store,
            "rem_cycle_timeout",
            {"cycle": cycle_num},
            severity="warning",
        )
    except Exception as exc:  # noqa: BLE001 -- daemon must never die on cycle error
        await _emit(
            store,
            "rem_cycle_error",
            {"cycle": cycle_num, "error": str(exc)[:500]},
            severity="critical",
        )

    await _emit(store, "rem_cycle_completed", result)
    return result
