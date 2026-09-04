"""The sleep-pipeline backstop step: reruns the courier's own transcript-sweep
producer during consolidation, gated by the courier's enablement flag -- and
proves it is idempotent against the courier through the shared per-file
sweep state, plus the WAL step-order/value stability this insertion must
never disturb."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from iai_mcp.lifecycle_event_log import LifecycleEventLog
from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline, SleepStep


def _write_transcript(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for obj in lines:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _user_line(text: str, *, uuid: str, ts: str) -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "timestamp": ts,
        "message": {"role": "user", "content": text},
    }


def _assistant_line(text: str, *, uuid: str, ts: str) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "timestamp": ts,
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _write_fixture_transcript(home: Path, *, session_id: str, nonce: str) -> Path:
    transcript_path = home / ".claude" / "projects" / "-alice-work" / f"{session_id}.jsonl"
    _write_transcript(
        transcript_path,
        [
            _user_line(nonce, uuid="u-1", ts="2026-09-03T20:00:00.000000+00:00"),
            _assistant_line(
                "acknowledged the backstop tracer request",
                uuid="a-1", ts="2026-09-03T20:00:01.000000+00:00",
            ),
        ],
    )
    return transcript_path


def _enable_flag(home: Path) -> Path:
    from iai_mcp.cli._cowork import _sweeper_enabled_flag_path

    flag = _sweeper_enabled_flag_path(home)
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.touch()
    return flag


def _open_store():
    from iai_mcp.store import MemoryStore
    return MemoryStore()


def _make_pipeline(store, tmp_path: Path) -> SleepPipeline:
    return SleepPipeline(
        store=store,
        lifecycle_state_path=tmp_path / "lifecycle_state.json",
        event_log=LifecycleEventLog(log_dir=tmp_path / "logs"),
    )


# ---------------------------------------------------------------------------
# Task 1: enabled runs the shared producer then drains; disabled is a no-op;
# a producer failure is isolated, never crashing the cycle.
# ---------------------------------------------------------------------------


def test_backstop_enabled_runs_producer_then_drain(hermetic_store, tmp_path):
    session_id = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    nonce = "backstop tracer unique nonce seven seven seven"
    home = Path.home()
    _write_fixture_transcript(home, session_id=session_id, nonce=nonce)
    _enable_flag(home)

    store = _open_store()
    try:
        pipeline = _make_pipeline(store, tmp_path)
        done, payload = pipeline._step_transcript_sweep_backstop(None)

        assert done is True
        assert payload["sessions_staged"] == 1
        assert payload["lines_staged"] == 2

        turns = store.recent_user_turns(50, session_id=session_id)
        texts = [t.literal_surface for t in turns]
        assert any(nonce in (t or "") for t in texts), (
            f"nonce not found in recent_user_turns; got: {texts!r}"
        )
    finally:
        store.close()


def test_backstop_disabled_stages_and_drains_nothing(hermetic_store, tmp_path):
    session_id = "bbbbbbbb-cccc-4ddd-8eee-ffffffffffff"
    nonce = "backstop disabled tracer nonce eight eight eight"
    home = Path.home()
    _write_fixture_transcript(home, session_id=session_id, nonce=nonce)
    # deliberately no _enable_flag(home) call -- flag absent

    store = _open_store()
    try:
        pipeline = _make_pipeline(store, tmp_path)
        done, payload = pipeline._step_transcript_sweep_backstop(None)

        assert done is True
        assert payload == {"transcript_sweep_backstop": "disabled"}

        turns = store.recent_user_turns(50, session_id=session_id)
        assert turns == []

        from iai_mcp.capture import deferred_captures_dir

        spool_dir = deferred_captures_dir()
        staged_files = list(spool_dir.glob("*")) if spool_dir.is_dir() else []
        assert staged_files == []
    finally:
        store.close()


def test_backstop_error_isolated_does_not_abort_cycle(
    hermetic_store, tmp_path, monkeypatch,
):
    from tests.test_sleep_pipeline import _patch_steps_to_noop

    home = Path.home()
    _enable_flag(home)

    def _raiser(*_args, **_kwargs):
        raise RuntimeError("synthetic sweep failure")

    monkeypatch.setattr("iai_mcp.transcript_sweep.sweep_once", _raiser)

    store = _open_store()
    try:
        pipeline = _make_pipeline(store, tmp_path)

        done, payload = pipeline._step_transcript_sweep_backstop(None)
        assert done is True
        assert payload["transcript_sweep_backstop_error"] == "RuntimeError"

        # Whole-cycle proof: the raising producer never propagates out of the
        # step, so a full pipeline run completes past it rather than
        # aborting. Every OTHER step is noop'd; this step's real handler
        # (with the raising sweep_once still monkeypatched above) stays live.
        _patch_steps_to_noop(pipeline, monkeypatch)
        real_backstop = SleepPipeline._step_transcript_sweep_backstop.__get__(pipeline)
        monkeypatch.setattr(pipeline, "_step_transcript_sweep_backstop", real_backstop)

        result = pipeline.run()

        assert result["failed_step"] is None
        assert result["error"] is None
        assert SleepStep.TRANSCRIPT_SWEEP_BACKSTOP in result["completed_steps"]
    finally:
        store.close()


def test_backstop_source_imports_shared_producer_not_a_copy():
    from iai_mcp.lilli.cycle.sleep_pipeline import _transcript_sweep_backstop as mod

    source = inspect.getsource(mod)
    assert "from iai_mcp.transcript_sweep import sweep_once" in source
    assert "def sweep_once" not in source


# ---------------------------------------------------------------------------
# Task 2: the backstop after the courier is a no-op through the shared
# per-file sweep state, and every step's WAL value/order is stable.
# ---------------------------------------------------------------------------


def test_backstop_after_courier_is_noop_via_shared_state(hermetic_store, tmp_path):
    from iai_mcp.capture import drain_capture_backlog
    from iai_mcp.transcript_sweep import sweep_once

    session_id = "cccccccc-dddd-4eee-8fff-000000000000"
    nonce = "shared state idempotency nonce nine nine nine"
    home = Path.home()
    # The shared state root both triggers read/write through: the same
    # capture-state directory under this one hermetic home.
    shared_state_root = home / ".iai-mcp"
    _write_fixture_transcript(home, session_id=session_id, nonce=nonce)
    _enable_flag(home)

    store = _open_store()
    try:
        # Courier pass: stage, then drain (as the daemon's own regular drain
        # would, independent of the backstop).
        courier_summary = sweep_once()
        assert courier_summary["sessions_staged"] == 1
        drain_capture_backlog(store)

        rows_before = len(store.recent_user_turns(50, session_id=session_id))
        assert rows_before == 2  # one user turn, one assistant turn

        pipeline = _make_pipeline(store, tmp_path)
        done, payload = pipeline._step_transcript_sweep_backstop(None)

        assert done is True
        assert payload["sessions_staged"] == 0, (
            f"backstop re-staged a file the courier already swept via "
            f"{shared_state_root}: {payload!r}"
        )

        rows_after = len(store.recent_user_turns(50, session_id=session_id))
        assert rows_after == rows_before
    finally:
        store.close()


# Every step's integer VALUE and _STEP_ORDER position as they exist after
# this insertion. A change to any pre-existing entry here is exactly the
# WAL-recovery corruption this plan's insertion must never cause.
_EXPECTED_VALUE_AND_POSITION: "list[tuple[SleepStep, int, int]]" = [
    (SleepStep.SCHEMA_MINE, 1, 0),
    (SleepStep.KNOB_TUNE, 2, 1),
    (SleepStep.OPTIMIZE_HIPPO, 4, 2),
    (SleepStep.HIPPO_CLEANUP, 5, 3),
    (SleepStep.DREAM_DECAY, 3, 4),
    (SleepStep.ERASURE_AGENT, 6, 5),
    (SleepStep.CLUSTER_REPLAY, 7, 6),
    (SleepStep.RECONSOLIDATION, 9, 7),
    (SleepStep.USER_MODEL_UPDATE, 10, 8),
    (SleepStep.DMN_REFLECTION, 11, 9),
    (SleepStep.CRISIS_RECLUSTER, 8, 10),
    (SleepStep.CLUSTER_SUMMARY, 12, 11),
    (SleepStep.RECALL_INDEX_REBUILD, 13, 12),
    (SleepStep.ENTITY_LINK, 14, 13),
    (SleepStep.CURIOSITY_MINE, 15, 14),
    (SleepStep.EMBEDDING_INTEGRITY, 16, 15),
    (SleepStep.COMMUNITY_NAMING, 17, 16),
    (SleepStep.RECONSOLIDATION_VALENCE, 18, 17),
    (SleepStep.PROC_MINE, 19, 18),
    (SleepStep.TRANSCRIPT_SWEEP_BACKSTOP, 20, 19),
]


def test_wal_value_and_order_stable_for_every_step():
    order = SleepPipeline._STEP_ORDER
    assert len(order) == len(_EXPECTED_VALUE_AND_POSITION)

    for step, expected_value, expected_index in _EXPECTED_VALUE_AND_POSITION:
        assert step.value == expected_value, (
            f"{step.name} integer VALUE moved: expected {expected_value}, "
            f"got {step.value}"
        )
        assert order.index(step) == expected_index, (
            f"{step.name} _STEP_ORDER position moved: expected "
            f"{expected_index}, got {order.index(step)}"
        )

    all_values = [s.value for s in SleepStep]
    assert len(all_values) == len(set(all_values)), (
        "duplicate integer VALUE across SleepStep"
    )


def test_wal_round_trip_resumes_new_tail_step_after_legacy_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A progress record minted by a binary that predates this step (its
    last completed step was PROC_MINE, value 19) must resume straight into
    the newly appended tail step on the next cycle."""
    from iai_mcp.lifecycle_state import default_state, save_state

    from tests.test_sleep_pipeline import _patch_steps_to_noop

    state_path = tmp_path / "lifecycle_state.json"
    record = default_state()
    record["sleep_cycle_progress"] = {
        "last_completed_step": SleepStep.PROC_MINE.value,
        "attempt": 0,
        "last_error": None,
        "started_at": "2026-09-03T00:00:00+00:00",
    }
    save_state(record, state_path)

    pipeline = SleepPipeline(
        store=None,
        lifecycle_state_path=state_path,
        event_log=LifecycleEventLog(log_dir=tmp_path / "logs"),
    )
    calls = _patch_steps_to_noop(pipeline, monkeypatch)
    pipeline.run()

    assert calls == [SleepStep.TRANSCRIPT_SWEEP_BACKSTOP]
