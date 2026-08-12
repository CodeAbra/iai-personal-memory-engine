from __future__ import annotations

import logging
from typing import Any, Callable

from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep

logger = logging.getLogger(__name__)


def step_embedding_integrity(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    if self._check_interrupt(SleepStep.EMBEDDING_INTEGRITY, 0, interrupt_check):
        return False, {}

    try:
        from iai_mcp.embed import (
            EmbedderConfigError,
            EmbedIdentityMismatch,
            embedder_for_store,
        )
        from iai_mcp.events import write_event

        store = self._store

        # The identity-guarded embedder drives BOTH lanes on success. On an
        # identity mismatch Lane B is skipped — comparing a stored vector
        # against an unconfirmed embedder identity could migrate a good vector
        # into the wrong space. Lane A re-resolves with the guard bypassed: a
        # zero vector is "no answer was ever written", not "a different model's
        # answer", so any resolvable embedder filling it is a strict
        # improvement even under an unconfirmed identity. The bypass is for
        # Lane A write-back only and must never widen to Lane B.
        embedder: Any = None
        embedder_for_zero_lane: Any = None
        try:
            embedder = embedder_for_store(store)
            embedder_for_zero_lane = embedder
        except (EmbedIdentityMismatch, EmbedderConfigError) as exc:
            logger.info(
                "embed_integrity: identity-guarded embedder unavailable (%s); "
                "drift lane skipped, zero lane retries with the guard bypassed",
                type(exc).__name__,
            )
            embedder = None
            try:
                embedder_for_zero_lane = embedder_for_store(
                    store, allow_identity_mismatch=True,
                )
            except Exception as exc2:  # noqa: BLE001 -- detection still runs
                logger.info(
                    "embed_integrity: bypassed embedder also unavailable (%s); "
                    "zero lane detects only this cycle",
                    type(exc2).__name__,
                )
                embedder_for_zero_lane = None

        result = store.db.selfheal_degraded_embeddings(
            embedder,
            embedder_for_zero_lane=embedder_for_zero_lane,
            interrupt_check=interrupt_check,
        )

        # info only when everything found this cycle was healed and no operator
        # signal is pending; warning whenever something was found-but-not-healed
        # (mass mismatch, a low-sample defer, an unresolvable embedder, a skipped
        # or deferred drift lane, or a budget-truncated write-back) — anything
        # carrying rows into the next cycle.
        fully_healed = (
            result.get("zero_found", 0) == result.get("zero_healed", 0)
            and result.get("sample_flagged", 0) == result.get("sample_healed", 0)
            and not result.get("mass_mismatch_detected", False)
            and not result.get("low_sample", False)
            and not result.get("zero_lane_embedder_unavailable", False)
            and not result.get("sample_lane_skipped", False)
            and not result.get("sample_lane_deferred", False)
        )
        severity = "info" if fully_healed else "warning"

        try:
            write_event(
                store,
                "embed_integrity_selfheal",
                {**result, "step": SleepStep.EMBEDDING_INTEGRITY.name},
                severity=severity,
            )
        except Exception as exc:  # noqa: BLE001 -- telemetry is best-effort
            logger.debug("embed_integrity_selfheal event write failed: %s", exc)

        return True, result

    except Exception as exc:  # noqa: BLE001 -- step must not crash the pipeline
        logger.warning(
            "embedding_integrity step failed: %s", exc, exc_info=True,
        )
        return True, {"error": str(exc)[:200], "healed": 0}
