from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypedDict

from iai_mcp.events import TELEMETRY_MEMORY_RELIEF
from iai_mcp.exceptions import (
    SleepCheckpointError,
    SleepPipelineError,
    SleepQuarantineError,
    SleepStepError,
    StoreError,
)
from iai_mcp.lifecycle_event_log import _LIVENESS_SPEC_VERSION

if TYPE_CHECKING:
    from iai_mcp.lifecycle_event_log import LifecycleEventLog
    from iai_mcp.lifecycle_state import (
        LifecycleStateRecord,
        Quarantine,
        SleepCycleProgress,
    )

logger = logging.getLogger(__name__)


QUARANTINE_TTL_HOURS_DEFAULT: float = float(
    os.environ.get("IAI_MCP_SLEEP_QUARANTINE_TTL_HOURS", "24"),
)


class SleepStep(Enum):

    SCHEMA_MINE = 1
    KNOB_TUNE = 2
    DREAM_DECAY = 3
    OPTIMIZE_HIPPO = 4
    HIPPO_CLEANUP = 5
    ERASURE_AGENT = 6
    CLUSTER_REPLAY = 7
    CRISIS_RECLUSTER = 8
    RECONSOLIDATION = 9
    USER_MODEL_UPDATE = 10
    DMN_REFLECTION = 11
    CLUSTER_SUMMARY = 12
    RECALL_INDEX_REBUILD = 13
    ENTITY_LINK = 14
    CURIOSITY_MINE = 15
    EMBEDDING_INTEGRITY = 16
    COMMUNITY_NAMING = 17


class SleepPhase(Enum):

    NREM = "NREM"
    REM = "REM"


STEP_PHASE: dict[SleepStep, SleepPhase] = {
    SleepStep.SCHEMA_MINE: SleepPhase.NREM,
    SleepStep.KNOB_TUNE: SleepPhase.NREM,
    SleepStep.OPTIMIZE_HIPPO: SleepPhase.NREM,
    SleepStep.HIPPO_CLEANUP: SleepPhase.NREM,
    SleepStep.DREAM_DECAY: SleepPhase.REM,
    SleepStep.ERASURE_AGENT: SleepPhase.REM,
    SleepStep.CLUSTER_REPLAY: SleepPhase.REM,
    SleepStep.RECONSOLIDATION: SleepPhase.REM,
    SleepStep.USER_MODEL_UPDATE: SleepPhase.REM,
    SleepStep.DMN_REFLECTION: SleepPhase.REM,
    SleepStep.CRISIS_RECLUSTER: SleepPhase.REM,
    SleepStep.CLUSTER_SUMMARY: SleepPhase.REM,
    SleepStep.RECALL_INDEX_REBUILD: SleepPhase.REM,
    SleepStep.ENTITY_LINK: SleepPhase.REM,
    SleepStep.CURIOSITY_MINE: SleepPhase.REM,
    SleepStep.EMBEDDING_INTEGRITY: SleepPhase.NREM,
    SleepStep.COMMUNITY_NAMING: SleepPhase.REM,
}


#: step -> (candidate payload key, processed payload key), each None when the
#: step's real return payload carries no independent count. Exhaustive over
#: SleepStep (test-enforced); a lookup for an unmapped step must degrade to
#: (None, None), never raise.
_LIVENESS_SPEC: dict[SleepStep, tuple[str | None, str | None]] = {
    SleepStep.SCHEMA_MINE: ("schemas_induced", "schemas_persisted"),
    SleepStep.KNOB_TUNE: (None, None),
    SleepStep.DREAM_DECAY: (None, None),
    SleepStep.OPTIMIZE_HIPPO: (None, None),
    SleepStep.HIPPO_CLEANUP: ("edges_scanned", "orphan_edges_deleted"),
    SleepStep.ERASURE_AGENT: (None, "count_quarantined"),
    SleepStep.CLUSTER_REPLAY: ("clusters_replayed", "sequential_pairs"),
    SleepStep.CRISIS_RECLUSTER: ("total_communities", "communities_dropped"),
    SleepStep.RECONSOLIDATION: ("records_scanned", "records_reconsolidated"),
    SleepStep.USER_MODEL_UPDATE: (None, None),
    SleepStep.DMN_REFLECTION: (None, None),
    SleepStep.CLUSTER_SUMMARY: (None, None),
    SleepStep.RECALL_INDEX_REBUILD: (None, None),
    SleepStep.ENTITY_LINK: (None, None),
    SleepStep.CURIOSITY_MINE: (None, None),
    SleepStep.EMBEDDING_INTEGRITY: (None, None),
    SleepStep.COMMUNITY_NAMING: ("communities_seen", "names_persisted"),
}


MAX_PAIRS_PER_CLUSTER: int = 100


class SleepPipelineResult(TypedDict, total=False):

    completed_steps: list[SleepStep]
    failed_step: SleepStep | None
    error: str | None
    duration_sec: float
    quarantine_triggered: bool
    interrupted: bool


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


class SleepPipeline:

    def __init__(
        self,
        store: Any,
        lifecycle_state_path: Path | None = None,
        event_log: Any | None = None,
        quarantine_ttl_hours: float | None = None,
        s2_coordinator: Any | None = None,
        loop: Any | None = None,
        *,
        lifecycle_state_machine: Any | None = None,
        lifecycle_event_log: Any | None = None,
    ) -> None:
        self._store = store

        self._lifecycle_state_path: Path | None = lifecycle_state_path

        self._lel: Any | None = lifecycle_event_log if lifecycle_event_log is not None else event_log

        self._quarantine_ttl_hours = (
            float(quarantine_ttl_hours)
            if quarantine_ttl_hours is not None
            else QUARANTINE_TTL_HOURS_DEFAULT
        )
        self._s2_coordinator = s2_coordinator
        self._loop = loop

        # Holds the per-step resident-set snapshot taken at step start so the
        # completion event can report the gross (pre-relief) delta.
        self._current_step_rss_before: dict[str, Any] | None = None

    def _get_state_path(self) -> Path:
        if self._lifecycle_state_path is not None:
            return self._lifecycle_state_path
        from iai_mcp.lifecycle_state import LIFECYCLE_STATE_PATH
        return LIFECYCLE_STATE_PATH

    def _get_event_log(self) -> Any:
        if self._lel is not None:
            return self._lel
        from iai_mcp.lifecycle_event_log import LifecycleEventLog
        self._lel = LifecycleEventLog()
        return self._lel

    @property
    def _event_log(self) -> Any:
        return self._get_event_log()


    def _load_state_record(self) -> Any:
        from iai_mcp.lifecycle_state import load_state
        return load_state(self._get_state_path())

    def _save_state_record(self, record: Any) -> None:
        from iai_mcp.lifecycle_state import save_state
        save_state(record, self._get_state_path())

    def _load_quarantine(self) -> Quarantine | None:
        return self._load_state_record().get("quarantine")

    def _set_quarantine(self, reason: str) -> Quarantine:
        now = _utc_now()
        until = now + timedelta(hours=self._quarantine_ttl_hours)
        quarantine: Quarantine = {
            "until_ts": until.isoformat(),
            "reason": reason,
            "since_ts": now.isoformat(),
        }
        record = self._load_state_record()
        record["quarantine"] = quarantine
        self._save_state_record(record)
        try:
            self._event_log.append({
                "event": "quarantine_entered",
                "reason": reason,
                "until_ts": quarantine["until_ts"],
                "ttl_hours": self._quarantine_ttl_hours,
            })
        except (OSError, ValueError) as exc:
            logger.debug("best-effort quarantine_entered event failed: %s", exc)
        return quarantine

    def _clear_quarantine(self, *, reason: str = "manual_reset") -> None:
        record = self._load_state_record()
        prior_quarantine = record.get("quarantine")
        record["quarantine"] = None
        progress = record.get("sleep_cycle_progress")
        if progress is not None:
            progress["attempt"] = 0
            record["sleep_cycle_progress"] = progress
        self._save_state_record(record)
        try:
            self._event_log.append({
                "event": "quarantine_lifted",
                "reason": reason,
                "prior_until_ts": (
                    prior_quarantine["until_ts"] if prior_quarantine else None
                ),
            })
        except (OSError, ValueError) as exc:
            logger.debug("best-effort quarantine_lifted event failed: %s", exc)

    def is_quarantined(self) -> bool:
        quarantine = self._load_quarantine()
        if quarantine is None:
            return False
        try:
            until = datetime.fromisoformat(quarantine["until_ts"])
        except (TypeError, ValueError):
            return False
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        return _utc_now() < until

    def reset_quarantine(self) -> None:
        self._clear_quarantine(reason="manual_reset")


    def _load_progress(self) -> Any:
        progress = self._load_state_record().get("sleep_cycle_progress")
        if progress is None:
            return None
        if (
            "last_completed_step" in progress
            and "last_completed_index" not in progress
        ):
            legacy = int(progress.pop("last_completed_step", 0))
            try:
                legacy_step = SleepStep(legacy)
                progress["last_completed_index"] = self._STEP_ORDER.index(
                    legacy_step,
                )
            except (ValueError, KeyError):
                progress["last_completed_index"] = -1
        return progress

    def _save_progress(
        self,
        last_completed_index: int,
        attempt: int,
        last_error: str | None,
        *,
        started_at: str | None = None,
    ) -> SleepCycleProgress:
        record = self._load_state_record()
        prior = record.get("sleep_cycle_progress") or {}
        progress: dict = {
            "last_completed_index": last_completed_index,
            "attempt": attempt,
            "last_error": last_error,
            "started_at": (
                started_at
                if started_at is not None
                else prior.get("started_at", _utc_now_iso())
            ),
        }
        record["sleep_cycle_progress"] = progress
        self._save_state_record(record)
        return progress

    def _clear_progress(self) -> None:
        record = self._load_state_record()
        record["sleep_cycle_progress"] = None
        self._save_state_record(record)


    def _emit_step_started(self, step: SleepStep) -> None:
        from iai_mcp.lilli.cycle.sleep_pipeline._rss_probe import capture_step_snapshot

        # RSS + pool + NRT only (no vmmap subprocess) — the per-step path must
        # stay microsecond-cheap; the region-count trend is sampled by the
        # watchdog breadcrumb on its own cadence.
        snap_before = capture_step_snapshot(include_vmmap=False)
        self._current_step_rss_before = snap_before
        try:
            self._event_log.append({
                "event": "sleep_step_started",
                "step": step.name,
                "step_num": step.value,
                "rss_kib": snap_before.get("rss_kib"),
                "vmmap_region_count": snap_before.get("vmmap_region_count"),
                "vm_allocate_kib": snap_before.get("vm_allocate_kib"),
                "numba_nrt_alloc_count": snap_before.get("numba_nrt_alloc_count"),
            })
        except (OSError, ValueError) as exc:
            logger.debug("best-effort sleep_step_started event failed: %s", exc)

    def _emit_step_completed(
        self, step: SleepStep, duration_sec: float, **payload: Any,
    ) -> None:
        from iai_mcp.lilli.cycle.sleep_pipeline._rss_probe import capture_step_snapshot

        # The after-snapshot is the gross, pre-relief resident set: the
        # page-reclaim postlude runs after this hook, and its net figure is
        # carried by the separate memory_relief event (joinable on step).
        snap_after = capture_step_snapshot(include_vmmap=False)
        snap_before = getattr(self, "_current_step_rss_before", None) or {}
        rss_before = snap_before.get("rss_kib")
        rss_after = snap_after.get("rss_kib")
        rss_delta_kib = (
            rss_after - rss_before
            if isinstance(rss_before, int) and isinstance(rss_after, int)
            else None
        )
        cand_field, proc_field = _LIVENESS_SPEC.get(step, (None, None))
        cand = payload.get(cand_field) if cand_field else None
        proc = payload.get(proc_field) if proc_field else None
        try:
            self._event_log.append({
                "event": "sleep_step_completed",
                "step": step.name,
                "step_num": step.value,
                "duration_sec": round(duration_sec, 3),
                "rss_kib_before": rss_before,
                "rss_kib_after": rss_after,
                "rss_delta_kib": rss_delta_kib,
                "vmmap_region_count_before": snap_before.get("vmmap_region_count"),
                "vmmap_region_count_after": snap_after.get("vmmap_region_count"),
                "vm_allocate_kib_before": snap_before.get("vm_allocate_kib"),
                "vm_allocate_kib_after": snap_after.get("vm_allocate_kib"),
                "numba_nrt_alloc_count_before": snap_before.get("numba_nrt_alloc_count"),
                "numba_nrt_alloc_count_after": snap_after.get("numba_nrt_alloc_count"),
                # liveness triple injected after **payload: a step payload key
                # named liveness_* can never shadow the normalized pair.
                **payload,
                "liveness_candidates": cand,
                "liveness_processed": proc,
                "liveness_spec_version": _LIVENESS_SPEC_VERSION,
            })
        except (OSError, ValueError) as exc:
            logger.debug("best-effort sleep_step_completed event failed: %s", exc)
        finally:
            self._current_step_rss_before = None

    def _check_interrupt(
        self,
        step: SleepStep,
        chunk_idx: int,
        interrupt_check: Callable[[], bool] | None,
    ) -> bool:
        if interrupt_check is None:
            return False
        try:
            should = bool(interrupt_check())
        except Exception as exc:  # noqa: BLE001 -- caller predicate may raise anything
            logger.debug("interrupt_check predicate raised: %s", exc)
            should = False
        if not should:
            return False
        # Capture exception context if an exception was active in this frame's
        # caller — that's the real signal future triage needs. Without this,
        # every report of this class arrives with "no traceback".
        import sys
        exc_str = ""
        exc_info = sys.exc_info()
        if exc_info[0] is not None and exc_info[1] is not None:
            exc_type = exc_info[0].__name__
            exc_msg = repr(exc_info[1])[:200]
            exc_str = f" caused_by={exc_type}: {exc_msg}"
        prior = self._load_progress() or {}
        last_completed_index = self._STEP_ORDER.index(step) - 1
        attempt = int(prior.get("attempt", 0))
        last_error = f"deferred:step={step.name}:chunk_idx={chunk_idx}{exc_str}"
        self._save_progress(
            last_completed_index=last_completed_index,
            attempt=attempt,
            last_error=last_error,
        )
        logger.warning(
            "sleep_step_deferred step=%s chunk_idx=%d%s",
            step.name, chunk_idx, exc_str,
        )
        return True

    def _now(self) -> datetime:
        # re-fetch the module helper per call so monkeypatches stay visible
        from iai_mcp.lilli.cycle import sleep_pipeline as _pkg
        return _pkg._utc_now()

    @property
    def _step_methods(
        self,
    ) -> dict[
        SleepStep,
        Callable[
            [Callable[[], bool] | None],
            "tuple[bool, dict[str, Any]]",
        ],
    ]:
        return {
            SleepStep.SCHEMA_MINE: self._step_schema_mine,
            SleepStep.KNOB_TUNE: self._step_knob_tune,
            SleepStep.DREAM_DECAY: self._step_dream_decay,
            SleepStep.ERASURE_AGENT: self._step_erasure_agent,
            SleepStep.OPTIMIZE_HIPPO: self._step_optimize_hippo,
            SleepStep.HIPPO_CLEANUP: self._step_hippo_cleanup,
            SleepStep.CLUSTER_REPLAY: self._step_cluster_replay,
            SleepStep.RECONSOLIDATION: self._step_reconsolidation,
            SleepStep.USER_MODEL_UPDATE: self._step_user_model_update,
            SleepStep.DMN_REFLECTION: self._step_dmn_reflection,
            SleepStep.CRISIS_RECLUSTER: self._step_crisis_recluster,
            SleepStep.CLUSTER_SUMMARY: self._step_cluster_summary,
            SleepStep.RECALL_INDEX_REBUILD: self._step_recall_index_rebuild,
            SleepStep.ENTITY_LINK: self._step_entity_link,
            SleepStep.CURIOSITY_MINE: self._step_curiosity_mine,
            SleepStep.EMBEDDING_INTEGRITY: self._step_embedding_integrity,
            SleepStep.COMMUNITY_NAMING: self._step_community_naming,
        }


    _STEP_ORDER: tuple[SleepStep, ...] = (
        SleepStep.SCHEMA_MINE,
        SleepStep.KNOB_TUNE,
        SleepStep.OPTIMIZE_HIPPO,
        SleepStep.HIPPO_CLEANUP,
        SleepStep.DREAM_DECAY,
        SleepStep.ERASURE_AGENT,
        SleepStep.CLUSTER_REPLAY,
        SleepStep.RECONSOLIDATION,
        SleepStep.USER_MODEL_UPDATE,
        SleepStep.DMN_REFLECTION,
        SleepStep.CRISIS_RECLUSTER,
        SleepStep.CLUSTER_SUMMARY,
        SleepStep.RECALL_INDEX_REBUILD,
        # Appended at the TAIL: WAL recovery persists step POSITIONS, so
        # every pre-existing step keeps its index and in-flight cycles from
        # an older binary resume correctly. Entity edges land at cycle end;
        # the next graph build reads them from the edges table.
        SleepStep.ENTITY_LINK,
        SleepStep.CURIOSITY_MINE,
        SleepStep.EMBEDDING_INTEGRITY,
        # Runs after RECALL_INDEX_REBUILD so the warm lexical index is
        # current for the keyphrase scoring.
        SleepStep.COMMUNITY_NAMING,
    )

    # Steps whose bodies materialize large transients (clustering / columnar
    # to-pandas paths). After each, a memory-relief postlude hands idle
    # allocator pages back to the OS. Not a step — a loop-tail call gated on
    # this set, so the step order and the WAL recovery index stay frozen.
    HEAVY_RELIEF_STEPS: frozenset[SleepStep] = frozenset({
        SleepStep.CRISIS_RECLUSTER,
        SleepStep.RECONSOLIDATION,
        SleepStep.CLUSTER_REPLAY,
        SleepStep.OPTIMIZE_HIPPO,
    })

    _QUARANTINE_STRIKE_THRESHOLD: int = 3

    def run(
        self, interrupt_check: Callable[[], bool] | None = None,
    ) -> SleepPipelineResult:
        return self._run_internal(
            interrupt_check, force=False,
        )

    def force_run(
        self, interrupt_check: Callable[[], bool] | None = None,
    ) -> SleepPipelineResult:
        return self._run_internal(
            interrupt_check, force=True,
        )

    def _run_internal(
        self,
        interrupt_check: Callable[[], bool] | None,
        *,
        force: bool,
    ) -> SleepPipelineResult:
        t0 = time.monotonic()
        completed_steps: list[SleepStep] = []

        if not force and self._check_and_maybe_auto_recover_quarantine():
            return {
                "completed_steps": [],
                "failed_step": None,
                "error": None,
                "duration_sec": round(time.monotonic() - t0, 3),
                "quarantine_triggered": True,
                "interrupted": False,
            }

        try:
            self._run_essential_variable_tracker_hook()
        except Exception as exc:  # noqa: BLE001 -- tracker is best-effort observer
            logger.warning("essential_variable_tracker hook failed: %s", exc, exc_info=True)

        progress = self._load_progress()
        last_completed_index = (
            int(progress.get("last_completed_index", -1))
            if progress is not None
            else -1
        )
        if last_completed_index >= len(self._STEP_ORDER) - 1:
            last_completed_index = -1
        resume_step_index = last_completed_index + 1

        step_payloads: dict[SleepStep, dict] = {}

        for step in self._STEP_ORDER:
            if self._STEP_ORDER.index(step) < resume_step_index:
                continue

            self._emit_step_started(step)
            step_t0 = time.monotonic()
            method = self._step_methods[step]
            try:
                done, payload = method(interrupt_check)
            except Exception as exc:  # noqa: BLE001 -- 3-strike + quarantine flow
                logger.error("sleep step %s failed: %s", step.name, exc, exc_info=True)
                err_str = str(exc)[:500]
                prior = self._load_progress() or {}
                prior_last_index = int(prior.get("last_completed_index", -1))
                step_idx = self._STEP_ORDER.index(step)
                if prior_last_index == step_idx - 1:
                    new_attempt = int(prior.get("attempt", 0)) + 1
                else:
                    new_attempt = 1
                self._save_progress(
                    last_completed_index=step_idx - 1,
                    attempt=new_attempt,
                    last_error=err_str,
                )
                self._emit_step_completed(
                    step,
                    duration_sec=time.monotonic() - step_t0,
                    error=err_str,
                    attempt=new_attempt,
                )
                quarantine_triggered = False
                if new_attempt >= self._QUARANTINE_STRIKE_THRESHOLD:
                    self._set_quarantine(
                        reason=(
                            f"sleep step {step.value} ({step.name}) "
                            f"failed {new_attempt}x"
                        ),
                    )
                    quarantine_triggered = True
                return {
                    "completed_steps": completed_steps,
                    "failed_step": step,
                    "error": err_str,
                    "duration_sec": round(time.monotonic() - t0, 3),
                    "quarantine_triggered": quarantine_triggered,
                    "interrupted": False,
                }

            if not done:
                return {
                    "completed_steps": completed_steps,
                    "failed_step": None,
                    "error": None,
                    "duration_sec": round(time.monotonic() - t0, 3),
                    "quarantine_triggered": False,
                    "interrupted": True,
                }

            self._save_progress(
                last_completed_index=self._STEP_ORDER.index(step),
                attempt=0,
                last_error=None,
            )
            self._emit_step_completed(
                step,
                duration_sec=time.monotonic() - step_t0,
                **payload,
            )
            completed_steps.append(step)
            step_payloads[step] = payload

            if step in self.HEAVY_RELIEF_STEPS:
                try:
                    from iai_mcp.lilli.cycle.sleep_pipeline._memory_relief import (
                        _step_memory_relief,
                    )

                    relief = _step_memory_relief(label=step.name)
                    self._event_log.append({
                        "event": TELEMETRY_MEMORY_RELIEF,
                        "step": step.name,
                        "step_num": step.value,
                        **relief,
                    })
                except Exception as exc:  # noqa: BLE001 -- relief never aborts a cycle
                    logger.debug(
                        "best-effort memory relief after %s failed: %s",
                        step.name, exc,
                    )

        try:
            from iai_mcp.sleep import _emit_cls_consolidation_run

            _decay_payload = step_payloads.get(SleepStep.DREAM_DECAY, {})
            _schema_payload = step_payloads.get(SleepStep.SCHEMA_MINE, {})
            _cluster_payload = step_payloads.get(SleepStep.CLUSTER_SUMMARY, {})

            _emit_cls_consolidation_run(
                self._store,
                "system",
                summaries_created=int(_cluster_payload.get("summaries_created", 0)),
                decay_result={
                    "decayed": int(_decay_payload.get("decayed", 0)),
                    "pruned": int(_decay_payload.get("pruned", 0)),
                },
                schema_candidates=int(_schema_payload.get("schemas_induced", 0)),
                schemas_induced=int(_schema_payload.get("schemas_persisted", 0)),
            )
        except Exception as exc:  # noqa: BLE001 -- cls emit is best-effort introspection
            logger.debug("pipeline-level cls_consolidation_run emit failed: %s", exc)

        self._clear_progress()
        return {
            "completed_steps": completed_steps,
            "failed_step": None,
            "error": None,
            "duration_sec": round(time.monotonic() - t0, 3),
            "quarantine_triggered": False,
            "interrupted": False,
        }

    def _check_and_maybe_auto_recover_quarantine(self) -> bool:
        quarantine = self._load_quarantine()
        if quarantine is None:
            return False
        try:
            until = datetime.fromisoformat(quarantine["until_ts"])
        except (TypeError, ValueError):
            self._clear_quarantine(reason="auto_recovery_malformed_ts")
            return False
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        if _utc_now() >= until:
            self._clear_quarantine(reason="auto_recovery_after_ttl")
            return False
        return True


# Per-step bodies live in sibling sub-modules as free functions taking self.
# Imported after the class is defined so each sub-module's top-level import of
# the spine names resolves against the already-populated package, then bound as
# class attributes (the descriptor protocol turns them back into methods, so
# method identity and instance/class patching are preserved).
from iai_mcp.lilli.cycle.sleep_pipeline import (  # noqa: E402
    _schema_mine, _knob_tune, _dream_decay, _erasure, _optimize, _compact,
    _cluster_replay, _reconsolidation, _user_model, _dmn, _crisis,
    _cluster_summary, _recall_index, _essential_variable, _entity_link,
    _curiosity_mine, _embed_integrity, _topic_naming,
)

SleepPipeline._step_schema_mine = _schema_mine.step_schema_mine
SleepPipeline._step_knob_tune = _knob_tune.step_knob_tune
SleepPipeline._step_dream_decay = _dream_decay.step_dream_decay
SleepPipeline._step_erasure_agent = _erasure.step_erasure_agent
SleepPipeline._step_compact_hippo = _optimize.step_compact_hippo
SleepPipeline._step_optimize_hippo = _optimize.step_optimize_hippo
SleepPipeline._step_hippo_cleanup_noop = _compact.step_hippo_cleanup_noop
SleepPipeline._step_hippo_cleanup = _compact.step_hippo_cleanup
SleepPipeline._step_cluster_replay = _cluster_replay.step_cluster_replay
SleepPipeline._step_reconsolidation = _reconsolidation.step_reconsolidation
SleepPipeline._step_user_model_update = _user_model.step_user_model_update
SleepPipeline._step_dmn_reflection = _dmn.step_dmn_reflection
SleepPipeline._step_crisis_recluster = _crisis.step_crisis_recluster
SleepPipeline._step_cluster_summary = _cluster_summary.step_cluster_summary
SleepPipeline._step_recall_index_rebuild = _recall_index.step_recall_index_rebuild
SleepPipeline._step_entity_link = _entity_link.step_entity_link
SleepPipeline._step_curiosity_mine = _curiosity_mine.step_curiosity_mine
SleepPipeline._step_embedding_integrity = _embed_integrity.step_embedding_integrity
SleepPipeline._step_community_naming = _topic_naming.step_community_naming
SleepPipeline._run_essential_variable_tracker_hook = _essential_variable.run_essential_variable_tracker_hook
SleepPipeline._clear_crisis_mode_via_s2_or_fallback = _essential_variable.clear_crisis_mode_via_s2_or_fallback
SleepPipeline._set_crisis_mode_via_s2_or_fallback = _essential_variable.set_crisis_mode_via_s2_or_fallback


__all__ = [
    "SleepPipeline", "SleepStep", "SleepPhase", "SleepPipelineResult",
    "STEP_PHASE", "QUARANTINE_TTL_HOURS_DEFAULT", "MAX_PAIRS_PER_CLUSTER",
    "_utc_now", "_utc_now_iso",
]
