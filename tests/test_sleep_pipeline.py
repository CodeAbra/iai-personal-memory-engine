from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from iai_mcp.lifecycle_event_log import LifecycleEventLog
from iai_mcp.lifecycle_state import (
    LifecycleState,
    default_state,
    load_state,
    save_state,
)
from iai_mcp.lilli.cycle.sleep_pipeline import (
    SleepPipeline,
    SleepPipelineResult,
    SleepStep,
)

@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "lifecycle_state.json"

@pytest.fixture
def event_log_dir(tmp_path: Path) -> Path:
    d = tmp_path / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d

@pytest.fixture
def event_log(event_log_dir: Path) -> LifecycleEventLog:
    return LifecycleEventLog(log_dir=event_log_dir)

@pytest.fixture
def pipeline(state_path: Path, event_log: LifecycleEventLog) -> SleepPipeline:
    return SleepPipeline(
        store=None,
        lifecycle_state_path=state_path,
        event_log=event_log,
        quarantine_ttl_hours=24.0,
    )

def _patch_steps_to_noop(
    pipeline: SleepPipeline, monkeypatch: pytest.MonkeyPatch,
    *,
    record: list[SleepStep] | None = None,
    payloads: dict[SleepStep, dict] | None = None,
) -> list[SleepStep]:
    calls = record if record is not None else []
    payloads = payloads or {}

    def _make_step(step: SleepStep):
        payload = payloads.get(step, {})

        def _noop(_interrupt_check):
            calls.append(step)
            return True, dict(payload)

        return _noop

    monkeypatch.setattr(
        pipeline, "_step_schema_mine", _make_step(SleepStep.SCHEMA_MINE),
    )
    monkeypatch.setattr(
        pipeline, "_step_knob_tune", _make_step(SleepStep.KNOB_TUNE),
    )
    monkeypatch.setattr(
        pipeline, "_step_dream_decay", _make_step(SleepStep.DREAM_DECAY),
    )
    monkeypatch.setattr(
        pipeline, "_step_erasure_agent",
        _make_step(SleepStep.ERASURE_AGENT),
    )
    monkeypatch.setattr(
        pipeline, "_step_optimize_hippo", _make_step(SleepStep.OPTIMIZE_HIPPO),
    )
    monkeypatch.setattr(
        pipeline, "_step_hippo_cleanup",
        _make_step(SleepStep.HIPPO_CLEANUP),
    )
    monkeypatch.setattr(
        pipeline, "_step_cluster_replay",
        _make_step(SleepStep.CLUSTER_REPLAY),
    )
    monkeypatch.setattr(
        pipeline, "_step_reconsolidation",
        _make_step(SleepStep.RECONSOLIDATION),
    )
    monkeypatch.setattr(
        pipeline, "_step_user_model_update",
        _make_step(SleepStep.USER_MODEL_UPDATE),
    )
    monkeypatch.setattr(
        pipeline, "_step_dmn_reflection",
        _make_step(SleepStep.DMN_REFLECTION),
    )
    monkeypatch.setattr(
        pipeline, "_step_crisis_recluster",
        _make_step(SleepStep.CRISIS_RECLUSTER),
    )
    monkeypatch.setattr(
        pipeline, "_step_cluster_summary",
        _make_step(SleepStep.CLUSTER_SUMMARY),
    )
    monkeypatch.setattr(
        pipeline, "_step_recall_index_rebuild",
        _make_step(SleepStep.RECALL_INDEX_REBUILD),
    )
    return calls

def test_pipeline_runs_9_steps_in_order(
    pipeline: SleepPipeline, monkeypatch: pytest.MonkeyPatch,
):
    calls = _patch_steps_to_noop(pipeline, monkeypatch)

    result: SleepPipelineResult = pipeline.run()

    assert calls == [
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
    ]
    assert result["completed_steps"] == calls
    assert result["failed_step"] is None
    assert result["error"] is None
    assert result["quarantine_triggered"] is False
    assert result.get("interrupted", False) is False

def test_pipeline_clears_progress_on_success(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    _patch_steps_to_noop(pipeline, monkeypatch)
    pipeline.run()
    record = load_state(state_path)
    assert record["sleep_cycle_progress"] is None

def test_pipeline_emits_started_and_completed_events(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    event_log: LifecycleEventLog,
):
    _patch_steps_to_noop(pipeline, monkeypatch)
    pipeline.run()
    events = event_log.read_all()
    started = [e for e in events if e["event"] == "sleep_step_started"]
    completed = [e for e in events if e["event"] == "sleep_step_completed"]
    assert len(started) == 13
    assert len(completed) == 13
    assert [e["step"] for e in started] == [
        s.name for s in (
            SleepStep.SCHEMA_MINE, SleepStep.KNOB_TUNE,
            SleepStep.OPTIMIZE_HIPPO, SleepStep.HIPPO_CLEANUP,
            SleepStep.DREAM_DECAY, SleepStep.ERASURE_AGENT,
            SleepStep.CLUSTER_REPLAY,
            SleepStep.RECONSOLIDATION,
            SleepStep.USER_MODEL_UPDATE,
            SleepStep.DMN_REFLECTION,
            SleepStep.CRISIS_RECLUSTER,
            SleepStep.CLUSTER_SUMMARY,
            SleepStep.RECALL_INDEX_REBUILD,
        )
    ]

def test_pipeline_resume_from_step_N(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    record = default_state()
    record["sleep_cycle_progress"] = {
        "last_completed_index": SleepPipeline._STEP_ORDER.index(
            SleepStep.KNOB_TUNE,
        ),
        "attempt": 0,
        "last_error": None,
        "started_at": "2026-05-02T00:00:00+00:00",
    }
    save_state(record, state_path)

    calls = _patch_steps_to_noop(pipeline, monkeypatch)
    pipeline.run()

    assert calls == [
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
    ]

def test_pipeline_resume_after_cycle_complete_treated_as_fresh(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    record = default_state()
    record["sleep_cycle_progress"] = {
        "last_completed_index": len(SleepPipeline._STEP_ORDER) - 1,
        "attempt": 0,
        "last_error": None,
        "started_at": "2026-05-02T00:00:00+00:00",
    }
    save_state(record, state_path)

    calls = _patch_steps_to_noop(pipeline, monkeypatch)
    pipeline.run()

    assert calls == [
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
    ]

def _patch_step_to_raise(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    failing_step: SleepStep,
    *,
    error_msg: str = "synthetic failure",
) -> None:
    _patch_steps_to_noop(pipeline, monkeypatch)
    method_name = {
        SleepStep.SCHEMA_MINE: "_step_schema_mine",
        SleepStep.KNOB_TUNE: "_step_knob_tune",
        SleepStep.DREAM_DECAY: "_step_dream_decay",
        SleepStep.ERASURE_AGENT: "_step_erasure_agent",
        SleepStep.OPTIMIZE_HIPPO: "_step_optimize_hippo",
        SleepStep.HIPPO_CLEANUP: "_step_hippo_cleanup",
        SleepStep.CLUSTER_REPLAY: "_step_cluster_replay",
        SleepStep.RECONSOLIDATION: "_step_reconsolidation",
        SleepStep.USER_MODEL_UPDATE: "_step_user_model_update",
        SleepStep.DMN_REFLECTION: "_step_dmn_reflection",
        SleepStep.CRISIS_RECLUSTER: "_step_crisis_recluster",
    }[failing_step]

    def _raiser(_interrupt_check):
        raise RuntimeError(error_msg)

    monkeypatch.setattr(pipeline, method_name, _raiser)

def test_pipeline_failure_persists_progress(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    _patch_step_to_raise(pipeline, monkeypatch, SleepStep.DREAM_DECAY)

    result = pipeline.run()

    assert result["failed_step"] == SleepStep.DREAM_DECAY
    assert result["error"] is not None
    assert "synthetic failure" in result["error"]
    assert result["completed_steps"] == [
        SleepStep.SCHEMA_MINE,
        SleepStep.KNOB_TUNE,
        SleepStep.OPTIMIZE_HIPPO,
        SleepStep.HIPPO_CLEANUP,
    ]

    record = load_state(state_path)
    progress = record["sleep_cycle_progress"]
    assert progress is not None
    assert progress["last_completed_index"] == SleepPipeline._STEP_ORDER.index(
        SleepStep.HIPPO_CLEANUP,
    )
    assert progress["attempt"] == 1
    assert "synthetic failure" in (progress["last_error"] or "")

def test_pipeline_resume_then_fail_again_increments_attempt(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    _patch_step_to_raise(pipeline, monkeypatch, SleepStep.DREAM_DECAY)

    pipeline.run()
    pipeline.run()

    record = load_state(state_path)
    progress = record["sleep_cycle_progress"]
    assert progress is not None
    assert progress["last_completed_index"] == SleepPipeline._STEP_ORDER.index(
        SleepStep.HIPPO_CLEANUP,
    )
    assert progress["attempt"] == 2
    assert record["quarantine"] is None

def test_pipeline_3_strike_quarantine(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    _patch_step_to_raise(pipeline, monkeypatch, SleepStep.OPTIMIZE_HIPPO)

    pipeline.run()
    pipeline.run()
    result = pipeline.run()

    assert result["quarantine_triggered"] is True
    assert result["failed_step"] == SleepStep.OPTIMIZE_HIPPO

    record = load_state(state_path)
    assert record["quarantine"] is not None
    quarantine = record["quarantine"]
    assert "OPTIMIZE_HIPPO" in quarantine["reason"]
    assert "3x" in quarantine["reason"]
    until = datetime.fromisoformat(quarantine["until_ts"])
    since = datetime.fromisoformat(quarantine["since_ts"])
    delta = until - since
    assert timedelta(hours=23, minutes=59) <= delta <= timedelta(
        hours=24, minutes=1,
    )

def test_pipeline_quarantined_run_short_circuits(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    now = datetime.now(timezone.utc)
    record = default_state()
    record["quarantine"] = {
        "until_ts": (now + timedelta(hours=12)).isoformat(),
        "reason": "manual seed",
        "since_ts": now.isoformat(),
    }
    save_state(record, state_path)

    calls = _patch_steps_to_noop(pipeline, monkeypatch)
    result = pipeline.run()

    assert result["quarantine_triggered"] is True
    assert result["completed_steps"] == []
    assert calls == []

def test_pipeline_quarantine_auto_recovery_after_ttl(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
    event_log: LifecycleEventLog,
):
    now = datetime.now(timezone.utc)
    record = default_state()
    record["quarantine"] = {
        "until_ts": (now - timedelta(hours=1)).isoformat(),
        "reason": "expired seed",
        "since_ts": (now - timedelta(hours=25)).isoformat(),
    }
    save_state(record, state_path)

    calls = _patch_steps_to_noop(pipeline, monkeypatch)
    result = pipeline.run()

    assert result["quarantine_triggered"] is False
    assert calls == [
        SleepStep.SCHEMA_MINE, SleepStep.KNOB_TUNE,
        SleepStep.OPTIMIZE_HIPPO, SleepStep.HIPPO_CLEANUP,
        SleepStep.DREAM_DECAY, SleepStep.ERASURE_AGENT,
        SleepStep.CLUSTER_REPLAY,
        SleepStep.RECONSOLIDATION,
        SleepStep.USER_MODEL_UPDATE,
        SleepStep.DMN_REFLECTION,
        SleepStep.CRISIS_RECLUSTER,
        SleepStep.CLUSTER_SUMMARY,
        SleepStep.RECALL_INDEX_REBUILD,
    ]
    record_after = load_state(state_path)
    assert record_after["quarantine"] is None
    events = event_log.read_all()
    lifted = [e for e in events if e["event"] == "quarantine_lifted"]
    assert len(lifted) >= 1
    assert lifted[0]["reason"] == "auto_recovery_after_ttl"

def test_pipeline_reset_quarantine_clears(
    pipeline: SleepPipeline,
    state_path: Path,
):
    now = datetime.now(timezone.utc)
    record = default_state()
    record["quarantine"] = {
        "until_ts": (now + timedelta(hours=12)).isoformat(),
        "reason": "stuck",
        "since_ts": now.isoformat(),
    }
    record["sleep_cycle_progress"] = {
        "last_completed_index": SleepPipeline._STEP_ORDER.index(
            SleepStep.DREAM_DECAY,
        ),
        "attempt": 3,
        "last_error": "boom",
        "started_at": now.isoformat(),
    }
    save_state(record, state_path)

    assert pipeline.is_quarantined() is True
    pipeline.reset_quarantine()
    assert pipeline.is_quarantined() is False

    record_after = load_state(state_path)
    assert record_after["quarantine"] is None
    progress = record_after["sleep_cycle_progress"]
    assert progress is not None
    assert progress["attempt"] == 0
    assert progress["last_completed_index"] == SleepPipeline._STEP_ORDER.index(
        SleepStep.DREAM_DECAY,
    )

def test_recover_expired_quarantine_clears_without_sleep_cycle(
    pipeline: SleepPipeline,
    state_path: Path,
):
    # An expired quarantine must be clearable WITHOUT attempting a sleep cycle
    # (the daemon calls this at boot). Without it, a daemon that never sleeps
    # carries a zombie quarantine entry indefinitely.
    now = datetime.now(timezone.utc)
    record = default_state()
    record["quarantine"] = {
        "until_ts": (now - timedelta(hours=1)).isoformat(),
        "reason": "sleep step 3 (DREAM_DECAY) failed 3x",
        "since_ts": (now - timedelta(hours=25)).isoformat(),
    }
    save_state(record, state_path)

    cleared = pipeline.recover_expired_quarantine()

    assert cleared is True
    assert load_state(state_path)["quarantine"] is None
    assert pipeline.is_quarantined() is False


def test_recover_expired_quarantine_keeps_active_quarantine(
    pipeline: SleepPipeline,
    state_path: Path,
):
    # An ACTIVE quarantine must survive the boot-time recovery probe.
    now = datetime.now(timezone.utc)
    record = default_state()
    record["quarantine"] = {
        "until_ts": (now + timedelta(hours=12)).isoformat(),
        "reason": "still cooling down",
        "since_ts": now.isoformat(),
    }
    save_state(record, state_path)

    cleared = pipeline.recover_expired_quarantine()

    assert cleared is False
    assert load_state(state_path)["quarantine"] is not None
    assert pipeline.is_quarantined() is True


def test_recover_expired_quarantine_noop_when_absent(
    pipeline: SleepPipeline,
    state_path: Path,
):
    save_state(default_state(), state_path)
    assert pipeline.recover_expired_quarantine() is False
    assert load_state(state_path)["quarantine"] is None


def test_pipeline_force_run_ignores_quarantine(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    now = datetime.now(timezone.utc)
    record = default_state()
    record["quarantine"] = {
        "until_ts": (now + timedelta(hours=12)).isoformat(),
        "reason": "stuck",
        "since_ts": now.isoformat(),
    }
    save_state(record, state_path)

    calls = _patch_steps_to_noop(pipeline, monkeypatch)
    result = pipeline.force_run()

    assert result["quarantine_triggered"] is False
    assert len(calls) == 13
    record_after = load_state(state_path)
    assert record_after["quarantine"] is not None

def test_pipeline_bounded_deferral_persists_chunk_idx(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    _patch_steps_to_noop(pipeline, monkeypatch)

    real_dream = SleepPipeline._step_dream_decay.__get__(pipeline)
    monkeypatch.setattr(pipeline, "_step_dream_decay", real_dream)

    call_counter = {"n": 0}

    def _trigger():
        call_counter["n"] += 1
        return True

    result = pipeline.run(interrupt_check=_trigger)

    assert result.get("interrupted") is True
    assert result["completed_steps"] == [
        SleepStep.SCHEMA_MINE,
        SleepStep.KNOB_TUNE,
        SleepStep.OPTIMIZE_HIPPO,
        SleepStep.HIPPO_CLEANUP,
    ]
    assert result["failed_step"] is None

    record = load_state(state_path)
    progress = record["sleep_cycle_progress"]
    assert progress is not None
    assert progress["last_completed_index"] == SleepPipeline._STEP_ORDER.index(
        SleepStep.HIPPO_CLEANUP,
    )
    err = progress["last_error"] or ""
    assert err.startswith("deferred:")
    assert "DREAM_DECAY" in err
    assert "chunk_idx=0" in err

def test_check_interrupt_captures_active_exception(
    pipeline: SleepPipeline,
    state_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    """When _check_interrupt fires while an exception is live in its caller's
    frame, the persisted `last_error` must include `caused_by=ExcType: repr`
    so future triage has actionable context. The bare string-only format
    (the bug Issue #17-A surfaces) makes every deferred report opaque."""
    import logging
    caplog.set_level(logging.WARNING, logger="iai_mcp.lilli.cycle.sleep_pipeline")
    try:
        raise ValueError("synthetic step failure")
    except ValueError:
        result = pipeline._check_interrupt(
            SleepStep.DREAM_DECAY, chunk_idx=7, interrupt_check=lambda: True,
        )
    assert result is True

    record = load_state(state_path)
    progress = record["sleep_cycle_progress"]
    err = progress["last_error"] or ""
    assert err.startswith("deferred:step=DREAM_DECAY:chunk_idx=7"), (
        f"prefix wrong: {err!r}"
    )
    assert "caused_by=ValueError" in err, (
        f"expected ValueError caused_by tag in last_error, got {err!r}"
    )
    assert "synthetic step failure" in err, (
        f"expected exception repr in last_error, got {err!r}"
    )
    warning_msgs = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("sleep_step_deferred" in m for m in warning_msgs), warning_msgs


def test_check_interrupt_no_exception_still_records_deferred(
    pipeline: SleepPipeline,
    state_path: Path,
):
    """Symmetric case: when no exception is active, the persisted last_error
    keeps the bare `deferred:step=X:chunk_idx=Y` shape (backwards compatible)."""
    result = pipeline._check_interrupt(
        SleepStep.SCHEMA_MINE, chunk_idx=0, interrupt_check=lambda: True,
    )
    assert result is True

    record = load_state(state_path)
    progress = record["sleep_cycle_progress"]
    err = progress["last_error"] or ""
    assert err == "deferred:step=SCHEMA_MINE:chunk_idx=0", err


def test_pipeline_resumes_after_deferral(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    _patch_steps_to_noop(pipeline, monkeypatch)
    real_dream = SleepPipeline._step_dream_decay.__get__(pipeline)
    monkeypatch.setattr(pipeline, "_step_dream_decay", real_dream)
    pipeline.run(interrupt_check=lambda: True)

    calls: list[SleepStep] = []
    _patch_steps_to_noop(pipeline, monkeypatch, record=calls)
    pipeline.run()
    assert calls == [
        SleepStep.DREAM_DECAY,
        SleepStep.ERASURE_AGENT,
        SleepStep.CLUSTER_REPLAY,
        SleepStep.RECONSOLIDATION,
        SleepStep.USER_MODEL_UPDATE,
        SleepStep.DMN_REFLECTION,
        SleepStep.CRISIS_RECLUSTER,
        SleepStep.CLUSTER_SUMMARY,
        SleepStep.RECALL_INDEX_REBUILD,
    ]

def test_pipeline_deferral_does_not_increment_attempt(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    _patch_steps_to_noop(pipeline, monkeypatch)
    real_dream = SleepPipeline._step_dream_decay.__get__(pipeline)
    monkeypatch.setattr(pipeline, "_step_dream_decay", real_dream)

    pipeline.run(interrupt_check=lambda: True)
    pipeline.run(interrupt_check=lambda: True)
    pipeline.run(interrupt_check=lambda: True)

    record = load_state(state_path)
    progress = record["sleep_cycle_progress"]
    assert progress is not None
    assert progress["attempt"] == 0
    assert record["quarantine"] is None

def test_pipeline_atomic_no_corruption_on_step_crash(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    _patch_step_to_raise(
        pipeline, monkeypatch, SleepStep.OPTIMIZE_HIPPO,
        error_msg="lance shard corrupt",
    )
    pipeline.run()

    record = load_state(state_path)
    assert record["sleep_cycle_progress"] is not None
    progress = record["sleep_cycle_progress"]
    assert progress["last_completed_index"] == SleepPipeline._STEP_ORDER.index(
        SleepStep.KNOB_TUNE,
    )
    assert progress["attempt"] == 1
    assert record["current_state"] == LifecycleState.WAKE.value
    assert record["shadow_run"] is False

def test_pipeline_run_does_not_mutate_other_state_fields(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    record = default_state()
    record["current_state"] = LifecycleState.SLEEP.value
    record["wrapper_event_seq"] = 42
    save_state(record, state_path)

    _patch_steps_to_noop(pipeline, monkeypatch)
    pipeline.run()

    after = load_state(state_path)
    assert after["current_state"] == LifecycleState.SLEEP.value
    assert after["wrapper_event_seq"] == 42
    assert after["sleep_cycle_progress"] is None

def test_is_quarantined_false_when_no_record(
    pipeline: SleepPipeline,
):
    assert pipeline.is_quarantined() is False

def test_is_quarantined_false_for_malformed_until_ts(
    pipeline: SleepPipeline,
    state_path: Path,
):
    record = default_state()
    record["quarantine"] = {
        "until_ts": "not a timestamp",
        "reason": "test",
        "since_ts": "also not a timestamp",
    }
    save_state(record, state_path)
    assert pipeline.is_quarantined() is False

def test_pipeline_resume_after_cycle_complete_wraps_to_schema_mine(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    record = default_state()
    record["sleep_cycle_progress"] = {
        "last_completed_index": len(SleepPipeline._STEP_ORDER) - 1,
        "attempt": 0,
        "last_error": None,
        "started_at": "2026-05-02T00:00:00+00:00",
    }
    save_state(record, state_path)

    calls = _patch_steps_to_noop(pipeline, monkeypatch)
    pipeline.run()

    assert calls[0] == SleepStep.SCHEMA_MINE

def test_pipeline_failed_erasure_agent_resumes_at_erasure_agent(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    _patch_step_to_raise(pipeline, monkeypatch, SleepStep.ERASURE_AGENT)
    pipeline.run()

    record = load_state(state_path)
    progress = record["sleep_cycle_progress"]
    assert progress is not None
    assert progress["last_completed_index"] == SleepPipeline._STEP_ORDER.index(
        SleepStep.DREAM_DECAY,
    )

    pipeline.run()
    record_after = load_state(state_path)
    progress_after = record_after["sleep_cycle_progress"]
    assert progress_after is not None
    assert progress_after["attempt"] == 2

def test_pipeline_three_strike_quarantine_on_erasure_agent(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    _patch_step_to_raise(pipeline, monkeypatch, SleepStep.ERASURE_AGENT)
    pipeline.run()
    pipeline.run()
    result = pipeline.run()

    assert result["quarantine_triggered"] is True
    assert result["failed_step"] == SleepStep.ERASURE_AGENT
    record = load_state(state_path)
    assert record["quarantine"] is not None
    assert "ERASURE_AGENT" in record["quarantine"]["reason"]

def test_pipeline_legacy_last_completed_step_field_migrated(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    record = default_state()
    record["sleep_cycle_progress"] = {
        "last_completed_step": 4,
        "attempt": 0,
        "last_error": None,
        "started_at": "2026-05-02T00:00:00+00:00",
    }
    save_state(record, state_path)

    calls = _patch_steps_to_noop(pipeline, monkeypatch)
    pipeline.run()

    assert calls == [
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
    ]

    calls2: list[SleepStep] = []
    _patch_steps_to_noop(pipeline, monkeypatch, record=calls2)
    pipeline.run()
    assert calls2 == [
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
    ]

def test_pipeline_legacy_last_completed_step_zero_starts_fresh(
    pipeline: SleepPipeline,
    monkeypatch: pytest.MonkeyPatch,
    state_path: Path,
):
    record = default_state()
    record["sleep_cycle_progress"] = {
        "last_completed_step": 0,
        "attempt": 0,
        "last_error": None,
        "started_at": "2026-05-02T00:00:00+00:00",
    }
    save_state(record, state_path)

    calls = _patch_steps_to_noop(pipeline, monkeypatch)
    pipeline.run()

    assert calls == [
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
    ]
