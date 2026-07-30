from __future__ import annotations

import logging
from typing import Any, Callable

from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep

logger = logging.getLogger(__name__)


def step_dmn_reflection(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    from iai_mcp.daemon_config import _load_dmn_config
    from iai_mcp.dmn_reflection import MetaAnalyst, ReflectionAgent
    from iai_mcp.events import write_event

    # Curiosity consumer. Recall only records a cheap ``deferred_curiosity_input``
    # event on the hot path; this REM reflection pass is where those inputs are
    # replayed into ``curiosity_question`` events and ``curiosity_bridge`` edges
    # via ``fire_curiosity``. Self-contained and fail-soft -- it never alters the
    # DMN result and never aborts the step, so a curiosity hiccup cannot stall
    # reflection or trip sleep-cycle quarantine.
    try:
        from iai_mcp.curiosity import drain_deferred_curiosity

        _drain = drain_deferred_curiosity(self._store)
        if _drain.get("drained"):
            write_event(
                self._store,
                "curiosity_drain_pass",
                _drain,
                severity="info",
            )
    except Exception as exc:  # noqa: BLE001 -- curiosity drain never aborts DMN
        logger.debug("curiosity_drain within dmn failed: %s", exc)

    meta_analyst_emitted = False
    reflection_synthesized = False
    try:
        cfg = _load_dmn_config()

        if cfg.meta_analyst_enabled:
            snapshot = MetaAnalyst().snapshot(
                self._store, cfg.reflection_window_hours,
            )
            snapshot["dry_run_mode"] = bool(cfg.dry_run)
            write_event(
                self._store,
                "system_health_report",
                snapshot,
                severity="info",
            )
            meta_analyst_emitted = True

        if self._check_interrupt(
            SleepStep.DMN_REFLECTION, 0, interrupt_check,
        ):
            return False, {}

        synth_record = ReflectionAgent().synthesize(
            self._store, cfg.reflection_window_hours,
        )
        prov = (synth_record.provenance or [{}])[0]
        if prov.get("embed_failed"):
            # A reflection that could not be embedded is junk in the ANN
            # (cosine ~0 to every cue); skip the insert, next cycle retries.
            return True, {
                "meta_analyst_emitted": meta_analyst_emitted,
                "reflection_synthesized": False,
                "reflection_skipped_embed_failed": True,
                "dry_run_mode": bool(cfg.dry_run),
            }
        empty_window = (
            int(prov.get("captured_count") or 0) == 0
            and int(prov.get("recalled_count") or 0) == 0
        )
        if empty_window:
            # A genuinely quiet window writes NOTHING: a real-embedded
            # "captured 0 turns" record is a valid ANN neighbor for vague
            # cues — junk that can actually surface on degraded recall.
            return True, {
                "meta_analyst_emitted": meta_analyst_emitted,
                "reflection_synthesized": False,
                "reflection_skipped_empty": True,
                "dry_run_mode": bool(cfg.dry_run),
            }
        if not cfg.dry_run:
            self._store.insert(synth_record)
            reflection_synthesized = True

        return True, {
            "meta_analyst_emitted": meta_analyst_emitted,
            "reflection_synthesized": reflection_synthesized,
            "dry_run_mode": bool(cfg.dry_run),
        }
    except Exception as exc:  # noqa: BLE001 -- non-critical DMN pass
        logger.warning("dmn_reflection step failed: %s", exc, exc_info=True)
        try:
            write_event(
                self._store,
                "dmn_reflection_pass",
                {
                    "meta_analyst_emitted": meta_analyst_emitted,
                    "reflection_synthesized": reflection_synthesized,
                    "persist_error": str(exc)[:500],
                },
                severity="warning",
            )
        except (OSError, ValueError) as inner_exc:
            logger.debug("best-effort dmn_reflection_pass event failed: %s", inner_exc)
        return True, {
            "meta_analyst_emitted": meta_analyst_emitted,
            "reflection_synthesized": reflection_synthesized,
            "persist_error": True,
        }
