"""The lifecycle idle countdown must watch real work, not only the heartbeat.

Regression coverage for the bug where the FSM forced itself to SLEEP after the
idle timer expired even though the daemon was still draining a continuously-fed
backlog: the only refresh of ``_last_active_monotonic`` was gated on the Node
wrapper heartbeat file, which can be stale (empty wrappers dir) while a drain
kicked off by earlier RPC traffic is still hammering the store. Advancing to
SLEEP there escalates to an EXCLUSIVE store lock mid-drain -> lock contention.

These tests pin the two added signals -- an in-flight drain and recent RPC
activity -- and the regression guard that a genuinely idle daemon still sleeps
(so crisis re-arming, which only runs in SLEEP, keeps working).
"""

from __future__ import annotations

import asyncio
import platform
import threading

import pytest


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="daemon module is POSIX-only on this project",
)


# Inputs that, on their own, would push the daemon all the way to SLEEP: well
# past the sleep threshold, stale RPC, sleep-eligible.
# Each test flips exactly one signal to prove it holds the countdown open.
_SLEEPY = dict(
    scanner_active=False,
    seconds_since_rpc=10_000.0,
    idle_elapsed=10_000.0,
    sleep_eligible=True,
    recent_rpc_window_sec=30.0,
    drowsy_after_sec=300.0,
    sleep_heartbeat_idle_sec=1800.0,
)


def test_drain_in_progress_blocks_advance_toward_sleep():
    from iai_mcp.daemon import (
        IDLE_DECISION_ACTIVE,
        IDLE_DECISION_SLEEP,
        _idle_countdown_decision,
    )

    decision = _idle_countdown_decision(drain_in_progress=True, **_SLEEPY)

    assert decision != IDLE_DECISION_SLEEP, (
        "idle countdown advanced toward SLEEP while a drain was in progress"
    )
    assert decision == IDLE_DECISION_ACTIVE


def test_recent_rpc_blocks_advance_toward_sleep():
    from iai_mcp.daemon import IDLE_DECISION_ACTIVE, _idle_countdown_decision

    inputs = dict(_SLEEPY)
    inputs["seconds_since_rpc"] = 5.0  # within recent_rpc_window_sec (30s)

    decision = _idle_countdown_decision(drain_in_progress=False, **inputs)

    assert decision == IDLE_DECISION_ACTIVE


def test_active_request_blocks_advance_toward_sleep():
    from iai_mcp.daemon import IDLE_DECISION_ACTIVE, _idle_countdown_decision

    decision = _idle_countdown_decision(
        drain_in_progress=False,
        active_requests=1,
        **_SLEEPY,
    )

    assert decision == IDLE_DECISION_ACTIVE


def test_mark_sleep_requests_honored_clears_pending_flags():
    from iai_mcp.daemon import _mark_sleep_requests_honored

    state = {
        "force_rem_request": {"pending": True, "ts": "2026-01-01T00:00:00+00:00"},
        "user_sleep_request": {"pending": True, "reason": "manual"},
        "unrelated": "kept",
    }

    updated = _mark_sleep_requests_honored(
        state,
        force_rem=True,
        user_sleep=True,
        honored_at="2026-01-01T00:00:30+00:00",
    )

    assert updated["force_rem_request"]["pending"] is False
    assert updated["force_rem_request"]["honored_at"] == "2026-01-01T00:00:30+00:00"
    assert updated["user_sleep_request"]["pending"] is False
    assert updated["user_sleep_request"]["honored_at"] == "2026-01-01T00:00:30+00:00"
    assert updated["unrelated"] == "kept"
    assert state["force_rem_request"]["pending"] is True


def test_clear_sleep_requests_already_honored_only_when_sleeping():
    from iai_mcp.daemon import _clear_sleep_requests_already_honored

    state = {"force_rem_request": {"pending": True}}

    awake_state, awake_changed = _clear_sleep_requests_already_honored(
        state,
        lifecycle_state="WAKE",
        honored_at="2026-01-01T00:00:30+00:00",
    )
    sleep_state, sleep_changed = _clear_sleep_requests_already_honored(
        state,
        lifecycle_state="SLEEP",
        honored_at="2026-01-01T00:00:30+00:00",
    )

    assert awake_changed is False
    assert awake_state["force_rem_request"]["pending"] is True
    assert sleep_changed is True
    assert sleep_state["force_rem_request"]["pending"] is False
    assert sleep_state["force_rem_request"]["honored_at"] == "2026-01-01T00:00:30+00:00"


def test_truly_idle_still_advances_to_sleep():
    # Regression guard: with NO activity of any kind, a quiet daemon must still
    # reach SLEEP, otherwise crisis re-arming (SLEEP-only) never runs again.
    from iai_mcp.daemon import IDLE_DECISION_SLEEP, _idle_countdown_decision

    decision = _idle_countdown_decision(drain_in_progress=False, **_SLEEPY)

    assert decision == IDLE_DECISION_SLEEP


def test_idle_past_drowsy_but_not_sleep_eligible_goes_drowsy_only():
    from iai_mcp.daemon import IDLE_DECISION_DROWSY, _idle_countdown_decision

    inputs = dict(_SLEEPY)
    inputs["idle_elapsed"] = 600.0      # past drowsy_after_sec (300) ...
    inputs["sleep_eligible"] = False    # ... but not sleep-eligible yet

    decision = _idle_countdown_decision(drain_in_progress=False, **inputs)

    assert decision == IDLE_DECISION_DROWSY


def test_within_drowsy_window_holds():
    from iai_mcp.daemon import IDLE_DECISION_HOLD, _idle_countdown_decision

    inputs = dict(_SLEEPY)
    inputs["idle_elapsed"] = 60.0       # under drowsy_after_sec (300)
    inputs["sleep_eligible"] = False

    decision = _idle_countdown_decision(drain_in_progress=False, **inputs)

    assert decision == IDLE_DECISION_HOLD


def test_scanner_active_alone_does_not_block_sleep():
    from iai_mcp.daemon import IDLE_DECISION_SLEEP, _idle_countdown_decision

    inputs = dict(_SLEEPY)
    inputs["scanner_active"] = True

    decision = _idle_countdown_decision(drain_in_progress=False, **inputs)

    assert decision == IDLE_DECISION_SLEEP


def test_sleep_wakes_on_real_activity():
    from iai_mcp.daemon import (
        IDLE_DECISION_ACTIVE,
        IDLE_DECISION_SLEEP,
        STATE_SLEEP,
        _sleep_should_wake_for_activity,
    )

    assert _sleep_should_wake_for_activity(
        current_state=STATE_SLEEP,
        idle_decision=IDLE_DECISION_ACTIVE,
    )
    assert _sleep_should_wake_for_activity(
        current_state=STATE_SLEEP,
        idle_decision=IDLE_DECISION_SLEEP,
        sleep_interrupted=True,
    )
    assert not _sleep_should_wake_for_activity(
        current_state=STATE_SLEEP,
        idle_decision=IDLE_DECISION_SLEEP,
    )
    assert not _sleep_should_wake_for_activity(
        current_state="WAKE",
        idle_decision=IDLE_DECISION_ACTIVE,
    )


def test_sleep_cycle_hibernates_with_idle_connected_wrapper():
    from iai_mcp.daemon import _sleep_cycle_still_idle_for_hibernation

    assert _sleep_cycle_still_idle_for_hibernation(
        drain_in_progress=False,
        seconds_since_rpc=10_000.0,
        recent_rpc_window_sec=30.0,
    )


def test_idle_connected_wrapper_sleep_cycle_exits_sleep_state(tmp_path):
    from iai_mcp.daemon import _sleep_cycle_still_idle_for_hibernation
    from iai_mcp.lifecycle import LifecycleEvent, LifecycleState, LifecycleStateMachine
    from iai_mcp.lifecycle_event_log import LifecycleEventLog
    from iai_mcp.lifecycle_state import default_state, load_state, save_state
    from iai_mcp.s2_coordinator import S2Coordinator

    state_path = tmp_path / "lifecycle_state.json"
    record = default_state()
    record["current_state"] = LifecycleState.SLEEP.value
    save_state(record, state_path)
    machine = LifecycleStateMachine(
        state_path=state_path,
        event_log=LifecycleEventLog(log_dir=tmp_path / "logs"),
        coordinator=S2Coordinator(
            store=None,
            state_path=state_path,
            min_interval_sec=0.0,
        ),
    )

    still_idle = _sleep_cycle_still_idle_for_hibernation(
        drain_in_progress=False,
        seconds_since_rpc=10_000.0,
        recent_rpc_window_sec=30.0,
    )
    asyncio.run(
        machine.dispatch(LifecycleEvent.SLEEP_CYCLE_DONE, still_idle=still_idle)
    )

    assert load_state(state_path)["current_state"] == LifecycleState.HIBERNATION.value


def test_sleep_cycle_does_not_hibernate_during_real_activity():
    from iai_mcp.daemon import _sleep_cycle_still_idle_for_hibernation

    assert not _sleep_cycle_still_idle_for_hibernation(
        drain_in_progress=True,
        seconds_since_rpc=10_000.0,
        recent_rpc_window_sec=30.0,
    )
    assert not _sleep_cycle_still_idle_for_hibernation(
        drain_in_progress=False,
        seconds_since_rpc=5.0,
        recent_rpc_window_sec=30.0,
    )


# --- capture wiring: the production drain path actually flips the flag --------


def test_drain_deferred_marks_in_progress_and_holds_off_sleep(monkeypatch):
    from iai_mcp import capture
    from iai_mcp.daemon import (
        IDLE_DECISION_ACTIVE,
        IDLE_DECISION_SLEEP,
        _idle_countdown_decision,
    )

    seen: dict[str, object] = {}

    def spy_impl(store, counts):
        # Mid-drain the flag must be set, and the idle countdown -- fed the flag
        # -- must refuse to advance toward SLEEP on otherwise-sleepy inputs.
        seen["in_progress"] = capture.is_drain_in_progress()
        seen["decision"] = _idle_countdown_decision(
            drain_in_progress=capture.is_drain_in_progress(), **_SLEEPY,
        )
        return {"files_drained": 0, "files_failed": 0}

    monkeypatch.setattr(capture, "_drain_deferred_captures_locked", spy_impl)

    assert capture.is_drain_in_progress() is False
    capture.drain_deferred_captures(store=None)

    assert seen["in_progress"] is True
    assert seen["decision"] == IDLE_DECISION_ACTIVE
    assert seen["decision"] != IDLE_DECISION_SLEEP
    # Released once the drain returns: a quiet daemon can sleep again.
    assert capture.is_drain_in_progress() is False


def test_drain_active_live_marks_in_progress(monkeypatch):
    from iai_mcp import capture

    seen: dict[str, object] = {}

    def spy_impl(store, *, exclude_session_id):
        seen["in_progress"] = capture.is_drain_in_progress()
        seen["sid"] = exclude_session_id
        return {"files_drained": 0}

    monkeypatch.setattr(capture, "_drain_active_live_captures_impl", spy_impl)

    capture.drain_active_live_captures(store=None, exclude_session_id="sess-1")

    assert seen["in_progress"] is True
    assert seen["sid"] == "sess-1"
    assert capture.is_drain_in_progress() is False


def test_guard_releases_on_exception(monkeypatch):
    from iai_mcp import capture

    def boom(store, counts):
        raise RuntimeError("drain blew up")

    monkeypatch.setattr(capture, "_drain_deferred_captures_locked", boom)

    assert capture.is_drain_in_progress() is False
    with pytest.raises(RuntimeError):
        capture.drain_deferred_captures(store=None)
    # A failed drain must not leak the in-progress count, else the daemon would
    # never sleep again.
    assert capture.is_drain_in_progress() is False


def test_overlapping_drains_tracked_by_depth():
    # The counter (not a boolean) keeps overlapping drains across threads
    # correct: the flag stays True until the LAST one releases.
    from iai_mcp import capture

    assert capture.is_drain_in_progress() is False
    with capture._drain_in_progress_guard():
        assert capture.is_drain_in_progress() is True
        with capture._drain_in_progress_guard():
            assert capture.is_drain_in_progress() is True
        assert capture.is_drain_in_progress() is True  # outer still holds
    assert capture.is_drain_in_progress() is False


def test_is_drain_in_progress_true_while_other_thread_drains(monkeypatch):
    from iai_mcp import capture

    entered = threading.Event()
    release = threading.Event()

    def slow_impl(store, counts):
        entered.set()
        release.wait(timeout=5.0)
        return {"files_drained": 0, "files_failed": 0}

    monkeypatch.setattr(capture, "_drain_deferred_captures_locked", slow_impl)

    worker = threading.Thread(
        target=lambda: capture.drain_deferred_captures(store=None),
    )
    worker.start()
    try:
        assert entered.wait(timeout=5.0)
        # Observed from a DIFFERENT thread than the one draining.
        assert capture.is_drain_in_progress() is True
    finally:
        release.set()
        worker.join(timeout=5.0)

    assert capture.is_drain_in_progress() is False
