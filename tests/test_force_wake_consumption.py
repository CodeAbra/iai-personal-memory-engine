from __future__ import annotations

import asyncio

import pytest


class _RecordingStateMachine:
    def __init__(self):
        self.dispatched: list[tuple] = []

    async def dispatch(self, event, *, reason=None, **payload):
        self.dispatched.append((event, reason))
        return "WAKE"


@pytest.fixture
def state_file(tmp_path, monkeypatch):
    from iai_mcp import daemon_state

    path = tmp_path / ".daemon-state.json"
    monkeypatch.setattr(daemon_state, "STATE_PATH", path)
    return path


def test_pending_force_wake_dispatches_wake_signal_and_marks_honored(state_file):
    from iai_mcp.daemon import _consume_force_wake
    from iai_mcp.daemon_state import load_state
    from iai_mcp.lifecycle import LifecycleEvent

    sm = _RecordingStateMachine()
    ds = {
        "fsm_state": "SLEEP",
        "force_wake_request": {"ts": "2026-07-08T10:00:00+00:00", "pending": True},
    }
    updated = asyncio.run(_consume_force_wake(ds, sm))

    assert sm.dispatched == [(LifecycleEvent.WAKE_SIGNAL, "force_wake_request")]
    req = updated["force_wake_request"]
    assert req["pending"] is False
    assert req["honored_at"]

    persisted = load_state()
    assert persisted["force_wake_request"]["pending"] is False


def test_no_pending_request_is_a_noop(state_file):
    from iai_mcp.daemon import _consume_force_wake

    sm = _RecordingStateMachine()
    ds = {"fsm_state": "WAKE"}
    updated = asyncio.run(_consume_force_wake(ds, sm))

    assert updated is ds
    assert sm.dispatched == []
    assert not state_file.exists()


def test_s2_blocked_wake_stays_pending(state_file):
    """When the S2 oscillation guard refuses the wake, the request must NOT
    be marked honored — reporting success for a refused wake is the control
    lying; the next tick retries after the min-interval clears."""
    from iai_mcp.daemon import _consume_force_wake
    from iai_mcp.s2_coordinator import S2OscillationBlocked

    class _BlockingStateMachine:
        async def dispatch(self, event, *, reason=None, **payload):
            raise S2OscillationBlocked(
                first_transition={"from_state": "WAKE", "to_state": "SLEEP"},
                second_transition={"from_state": "SLEEP", "to_state": "WAKE"},
                interval_sec=1.0,
            )

    ds = {
        "fsm_state": "SLEEP",
        "force_wake_request": {"ts": "2026-07-09T04:00:00+00:00", "pending": True},
    }
    updated = asyncio.run(_consume_force_wake(ds, _BlockingStateMachine()))
    assert updated is ds
    assert ds["force_wake_request"]["pending"] is True
    assert not state_file.exists()


def test_already_honored_request_never_refires(state_file):
    from iai_mcp.daemon import _consume_force_wake

    sm = _RecordingStateMachine()
    ds = {
        "fsm_state": "WAKE",
        "force_wake_request": {
            "ts": "2026-07-06T10:00:00+00:00",
            "pending": False,
            "honored_at": "2026-07-06T10:00:05+00:00",
        },
    }
    updated = asyncio.run(_consume_force_wake(ds, sm))
    assert updated is ds
    assert sm.dispatched == []
