from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from iai_mcp.events import write_event
from iai_mcp.lifecycle_event_log import _LIVENESS_SPEC_VERSION, LifecycleEventLog
from iai_mcp.s5 import detect_drift_anomaly
from iai_mcp.sigma import compute_and_emit

logger = logging.getLogger(__name__)

AUDIT_INTERVAL_SEC: int = 60 * 60


def _resolve_log_dir() -> Path:
    env = os.environ.get("IAI_MCP_STORE")
    return (Path(env) if env else Path.home() / ".iai-mcp") / "logs"


def _stamp_identity_audit_liveness(
    alerts: list[dict], candidates: int, processed: int,
) -> None:
    try:
        LifecycleEventLog(log_dir=_resolve_log_dir()).append({
            "event": "promise_liveness",
            "promise": "identity_audit",
            "liveness_candidates": candidates,
            "liveness_processed": processed,
            "liveness_spec_version": _LIVENESS_SPEC_VERSION,
        })
    except Exception as exc:  # noqa: BLE001 -- diagnostic instrumentation must never break the audit
        logger.debug("identity_audit_liveness_append_failed: %s", exc)


async def continuous_audit(
    store,
    shutdown: asyncio.Event,
    *,
    interval_sec: float | None = None,
) -> None:
    while not shutdown.is_set():
        effective_interval: float = (
            float(interval_sec) if interval_sec is not None else float(AUDIT_INTERVAL_SEC)
        )
        try:
            alerts, candidates, processed = await asyncio.to_thread(
                detect_drift_anomaly, store, 5,
            )
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
        else:
            await asyncio.to_thread(
                _stamp_identity_audit_liveness, alerts, candidates, processed,
            )

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
