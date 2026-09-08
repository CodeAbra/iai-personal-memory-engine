"""A completed sleep cycle records a 'consolidated up to episodic record X'
watermark that survives _clear_progress() -- it is a top-level sibling key
on LifecycleStateRecord, not nested inside sleep_cycle_progress (which
_clear_progress() nulls on every clean finish).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from iai_mcp import store_watermark
from iai_mcp.lifecycle_event_log import LifecycleEventLog
from iai_mcp.lifecycle_state import default_state, load_state, save_state
from iai_mcp.lilli.cycle.sleep_pipeline import (
    SleepPipeline,
    SleepPipelineResult,
    SleepStep,
)

from test_sleep_pipeline import _patch_steps_to_noop  # noqa: E402 -- reuse the noop-step harness


class _StubStore:
    """Exposes only what _read_pre_cycle_watermark's idiom needs -- .db is
    None so getattr(store.db, "_hippo_dir", store.root / "hippo") falls
    through to the .root-derived path, matching production's fallback arm.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.db = None


@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "lifecycle_state.json"


@pytest.fixture
def event_log(tmp_path: Path) -> LifecycleEventLog:
    d = tmp_path / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return LifecycleEventLog(log_dir=d)


@pytest.fixture
def stub_store(tmp_path: Path) -> _StubStore:
    return _StubStore(tmp_path / "store-root")


def test_clean_cycle_sets_consolidated_watermark_surviving_clear_progress(
    state_path, event_log, stub_store, monkeypatch,
):
    pre_cycle_ts = "2026-09-06T10:00:00+00:00"
    store_watermark.emit(stub_store.root / "hippo", pre_cycle_ts)

    pipeline = SleepPipeline(
        store=stub_store,
        lifecycle_state_path=state_path,
        event_log=event_log,
        quarantine_ttl_hours=24.0,
    )
    _patch_steps_to_noop(pipeline, monkeypatch)

    result: SleepPipelineResult = pipeline.run()
    assert result["failed_step"] is None
    assert result["interrupted"] is False

    record = load_state(state_path)
    assert record.get("consolidated_watermark") == pre_cycle_ts, (
        f"consolidated_watermark must equal the pre-cycle store watermark, "
        f"got {record.get('consolidated_watermark')!r}"
    )
    # The property under test: sleep_cycle_progress IS nulled by the same
    # clean finish, yet consolidated_watermark survives it -- proven in one
    # assertion pair against the same loaded record.
    assert record.get("sleep_cycle_progress") is None, (
        "sleep_cycle_progress must still be nulled on a clean finish"
    )


def test_watermark_not_advanced_by_writes_during_the_cycle(
    state_path, event_log, stub_store, monkeypatch,
):
    """The persisted value is the PRE-cycle watermark (input to the
    completed cycle), not whatever the sidecar reads as of cycle end --
    a later sidecar advance (e.g. from this cycle's own writes) must not
    retroactively change what was already captured at t0.
    """
    pre_cycle_ts = "2026-09-06T10:00:00+00:00"
    store_watermark.emit(stub_store.root / "hippo", pre_cycle_ts)

    pipeline = SleepPipeline(
        store=stub_store,
        lifecycle_state_path=state_path,
        event_log=event_log,
        quarantine_ttl_hours=24.0,
    )
    _patch_steps_to_noop(pipeline, monkeypatch)

    # Simulate a same-cycle write advancing the sidecar mid-run: patch the
    # noop schema_mine step to bump the sidecar, mimicking a step's own
    # write landing before the cycle finishes.
    original = pipeline._step_schema_mine

    def _advancing_step(interrupt_check):
        store_watermark.emit(stub_store.root / "hippo", "2026-09-06T23:00:00+00:00")
        return original(interrupt_check)

    monkeypatch.setattr(pipeline, "_step_schema_mine", _advancing_step)

    pipeline.run()

    record = load_state(state_path)
    assert record.get("consolidated_watermark") == pre_cycle_ts, (
        "consolidated_watermark must stay pinned to the t0 pre-cycle value, "
        "not a value advanced mid-cycle"
    )


def test_watermark_pinned_across_interrupted_then_resumed_cycle(
    state_path, event_log, stub_store, monkeypatch,
):
    """A step fails once (attempt-tracked, quarantine trips only at 3
    strikes) and the cycle returns without calling _clear_progress(). The
    NEXT run() call -- possibly hours later, after ordinary capture
    activity has advanced the sidecar -- resumes and SKIPS the
    already-completed step. consolidated_watermark on the eventual clean
    finish must equal the FIRST call's t0, not the resume-time sidecar
    value: the skipped step never saw anything past the original t0.
    """
    t0 = "2026-09-06T10:00:00+00:00"
    t1 = "2026-09-07T10:00:00+00:00"
    store_watermark.emit(stub_store.root / "hippo", t0)

    pipeline = SleepPipeline(
        store=stub_store,
        lifecycle_state_path=state_path,
        event_log=event_log,
        quarantine_ttl_hours=24.0,
    )
    _patch_steps_to_noop(pipeline, monkeypatch)

    succeed = pipeline._step_knob_tune
    attempts = {"count": 0}

    def _fail_once_then_succeed(interrupt_check):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("synthetic transient failure")
        return succeed(interrupt_check)

    monkeypatch.setattr(pipeline, "_step_knob_tune", _fail_once_then_succeed)

    first = pipeline.run()
    assert first["failed_step"] == SleepStep.KNOB_TUNE
    record_after_first = load_state(state_path)
    assert record_after_first.get("consolidated_watermark") is None, (
        "an interrupted cycle must never advance consolidated_watermark"
    )

    store_watermark.emit(stub_store.root / "hippo", t1)

    second = pipeline.run()
    assert second["failed_step"] is None
    assert second["interrupted"] is False
    assert SleepStep.SCHEMA_MINE not in second["completed_steps"], (
        "the resumed call must SKIP the already-completed step, not re-run it"
    )

    record = load_state(state_path)
    assert record.get("consolidated_watermark") == t0, (
        f"resumed cycle must certify only the ORIGINAL t0 ({t0!r}), not the "
        f"resume-time sidecar value; got {record.get('consolidated_watermark')!r}"
    )


def test_wrap_to_fresh_does_not_inherit_stale_pinned_watermark(
    state_path, event_log, stub_store, monkeypatch,
):
    """A prior cycle completed every step (last_completed_index at the
    tail) but crashed before _clear_progress -- e.g. between the final
    _save_progress and the cls-emit block -- leaving sleep_cycle_progress
    non-null with a PINNED t0. The next run() call wraps to fresh
    (resume_step_index == 0, nothing skipped, every step re-runs against
    today's data) and must re-read the sidecar for THIS run's t0, not
    inherit the stale pinned one from the crashed prior cycle.
    """
    old_t0 = "2026-09-01T00:00:00+00:00"
    record = default_state()
    record["sleep_cycle_progress"] = {
        "last_completed_index": len(SleepPipeline._STEP_ORDER) - 1,
        "attempt": 0,
        "last_error": None,
        "started_at": "2026-09-01T00:00:00+00:00",
        "pre_cycle_watermark": old_t0,
    }
    save_state(record, state_path)

    fresh_t0 = "2026-09-06T10:00:00+00:00"
    store_watermark.emit(stub_store.root / "hippo", fresh_t0)

    pipeline = SleepPipeline(
        store=stub_store,
        lifecycle_state_path=state_path,
        event_log=event_log,
        quarantine_ttl_hours=24.0,
    )
    calls = _patch_steps_to_noop(pipeline, monkeypatch)

    result = pipeline.run()
    assert result["failed_step"] is None
    assert calls and calls[0] == SleepStep.SCHEMA_MINE, (
        "wrap-to-fresh must re-run every step, not skip any"
    )

    record = load_state(state_path)
    assert record.get("consolidated_watermark") == fresh_t0, (
        f"a wrap-to-fresh cycle must certify THIS run's t0 ({fresh_t0!r}), "
        f"not the prior cycle's stale pinned watermark ({old_t0!r}); "
        f"got {record.get('consolidated_watermark')!r}"
    )


def test_old_record_without_the_key_loads_cleanly(state_path):
    """A lifecycle_state.json predating this key must load via the .get
    idiom without a KeyError -- the NotRequired/additive contract.
    """
    state_path.write_text(json.dumps({
        "current_state": "WAKE",
        "since_ts": "2026-09-06T00:00:00+00:00",
        "last_activity_ts": "2026-09-06T00:00:00+00:00",
        "wrapper_event_seq": 0,
        "sleep_cycle_progress": None,
        "quarantine": None,
        "shadow_run": False,
        "crisis_mode": False,
    }), encoding="utf-8")

    record = load_state(state_path)
    assert record.get("consolidated_watermark") is None
