from __future__ import annotations

import asyncio
import logging
import os

from iai_mcp.events import write_event
from iai_mcp.s5 import detect_drift_anomaly
from iai_mcp.sigma import compute_and_emit

logger = logging.getLogger(__name__)

AUDIT_INTERVAL_SEC: int = 60 * 60


def _first_iter_grace_sec(effective_interval: float) -> float:
    raw = os.environ.get("IAI_MCP_AUDIT_FIRST_ITER_GRACE_SEC")
    if raw is None:
        return max(0.0, float(effective_interval))
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return max(0.0, float(effective_interval))


async def continuous_audit(
    store,
    shutdown: asyncio.Event,
    *,
    interval_sec: float | None = None,
) -> None:
    effective_interval: float = (
        float(interval_sec) if interval_sec is not None else float(AUDIT_INTERVAL_SEC)
    )
    first_grace = _first_iter_grace_sec(effective_interval)
    if first_grace > 0:
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=first_grace)
            return
        except asyncio.TimeoutError:
            pass

    while not shutdown.is_set():
        try:
            await asyncio.to_thread(detect_drift_anomaly, store, 5)
        except Exception as exc:  # noqa: BLE001 -- daemon must never die
            logger.warning("identity_audit_s5_failed", extra={"err": str(exc)[:200]})
            try:
                await asyncio.to_thread(
                    write_event,
                    store,
                    "identity_audit_error",
                    {"stage": "s5", "error": str(exc)[:500]},
                    severity="warning",
                )
            except Exception:  # noqa: BLE001 -- event write failure is non-fatal
                pass

        try:
            await asyncio.to_thread(compute_and_emit, store)
        except Exception as exc:  # noqa: BLE001 -- daemon must never die
            logger.warning("identity_audit_sigma_failed", extra={"err": str(exc)[:200]})
            try:
                await asyncio.to_thread(
                    write_event,
                    store,
                    "identity_audit_error",
                    {"stage": "sigma", "error": str(exc)[:500]},
                    severity="warning",
                )
            except Exception:  # noqa: BLE001 -- event write failure is non-fatal
                pass

        try:
            await asyncio.wait_for(shutdown.wait(), timeout=effective_interval)
            break
        except asyncio.TimeoutError:
            continue
