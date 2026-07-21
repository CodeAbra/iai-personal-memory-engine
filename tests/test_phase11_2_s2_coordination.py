from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from iai_mcp.lifecycle_state import (
    LifecycleState,
    default_state,
    load_state,
    save_state,
)
from iai_mcp.s2_coordinator import (
    S2Coordinator,
    S2OscillationBlocked,
    S2OscillationConflict,
)

class _CapturedEvents:

    def __init__(self) -> None:
        self.events: list[dict] = []

    def __call__(self, store, kind, data, *, severity=None, **_kw):
        self.events.append(
            {"kind": kind, "data": dict(data), "severity": severity}
        )
        return None

    def by_kind(self, kind: str) -> list[dict]:
        return [e for e in self.events if e["kind"] == kind]

@pytest.fixture
def captured_events():
    cap = _CapturedEvents()
    with patch("iai_mcp.s2_coordinator.write_event", cap):
        yield cap

@pytest.fixture
def seeded_state(tmp_path: Path) -> Path:
    state_path = tmp_path / "lifecycle_state.json"
    save_state(default_state(), state_path)
    return state_path

def _reseed_state(state_path: Path, target: LifecycleState) -> None:
    rec = default_state()
    rec["current_state"] = target.value
    save_state(rec, state_path)

def _make_coord(
    state_path: Path,
    *,
    min_interval_sec: float = 5.0,
    dry_run: bool = False,
) -> S2Coordinator:
    return S2Coordinator(
        store=None,
        state_path=state_path,
        min_interval_sec=min_interval_sec,
        dry_run=dry_run,
    )

def test_single_transition_happy_path_emits_attempt_event(
    captured_events, seeded_state
):
    coord = _make_coord(seeded_state)

    new = asyncio.run(
        coord.transition(
            LifecycleState.WAKE,
            LifecycleState.DROWSY,
            "drowsy_on_idle_5min",
        )
    )

    assert new == LifecycleState.DROWSY
    assert coord.version == 1

    rec = load_state(seeded_state)
    assert rec["current_state"] == "DROWSY"

    attempts = captured_events.by_kind("s2_transition_attempt")
    assert len(attempts) == 1
    body = attempts[0]["data"]
    assert set(body.keys()) == {
        "from_state",
        "to_state",
        "reason",
        "version_before",
        "version_after",
        "succeeded",
        "conflict_reason",
    }
    assert body["from_state"] == "WAKE"
    assert body["to_state"] == "DROWSY"
    assert body["reason"] == "drowsy_on_idle_5min"
    assert body["version_before"] == 0
    assert body["version_after"] == 1
    assert body["succeeded"] is True
    assert body["conflict_reason"] is None
    assert captured_events.by_kind("s2_oscillation_blocked") == []

def test_concurrent_transitions_one_wins_one_raises_conflict(
    captured_events, seeded_state
):
    coord = _make_coord(seeded_state)

    async def race():
        return await asyncio.gather(
            coord.transition(
                LifecycleState.WAKE, LifecycleState.DROWSY, "race_a"
            ),
            coord.transition(
                LifecycleState.WAKE, LifecycleState.SLEEP, "race_b"
            ),
            return_exceptions=True,
        )

    results = asyncio.run(race())

    successes = [r for r in results if isinstance(r, LifecycleState)]
    conflicts = [r for r in results if isinstance(r, S2OscillationConflict)]
    assert len(successes) == 1, results
    assert len(conflicts) == 1, results

    winner = successes[0]
    assert conflicts[0].actual_state == winner, (
        f"conflict.actual_state {conflicts[0].actual_state} should "
        f"equal winner {winner}"
    )
    assert conflicts[0].attempted_from == LifecycleState.WAKE
    assert coord.version == 1

    rec = load_state(seeded_state)
    assert rec["current_state"] == winner.value

    attempts = captured_events.by_kind("s2_transition_attempt")
    assert len(attempts) == 2
    succeeded_flags = sorted(a["data"]["succeeded"] for a in attempts)
    assert succeeded_flags == [False, True]
    rejected = [a for a in attempts if a["data"]["succeeded"] is False]
    assert rejected[0]["data"]["conflict_reason"] == "cas_mismatch"
    assert captured_events.by_kind("s2_oscillation_blocked") == []

def test_rapid_oscillation_raises_blocked(captured_events, seeded_state):
    _reseed_state(seeded_state, LifecycleState.SLEEP)
    coord = _make_coord(seeded_state, min_interval_sec=60.0)

    asyncio.run(
        coord.transition(
            LifecycleState.SLEEP,
            LifecycleState.WAKE,
            "wake_on_signal_consumed",
        )
    )

    with pytest.raises(S2OscillationBlocked) as exc_info:
        asyncio.run(
            coord.transition(
                LifecycleState.WAKE,
                LifecycleState.SLEEP,
                "sleep_on_idle_30min",
            )
        )
    assert 0.0 <= exc_info.value.interval_sec < 60.0

    rec = load_state(seeded_state)
    assert rec["current_state"] == "WAKE"
    assert coord.version == 1

    blocks = captured_events.by_kind("s2_oscillation_blocked")
    assert len(blocks) == 1
    body = blocks[0]["data"]
    assert set(body.keys()) == {
        "first_transition",
        "second_transition",
        "interval_sec",
        "dry_run_mode",
    }
    assert body["first_transition"]["from_state"] == "SLEEP"
    assert body["first_transition"]["to_state"] == "WAKE"
    assert body["second_transition"]["from_state"] == "WAKE"
    assert body["second_transition"]["to_state"] == "SLEEP"
    assert body["dry_run_mode"] is False
    assert 0.0 <= body["interval_sec"] < 60.0

    attempts = captured_events.by_kind("s2_transition_attempt")
    rejected = [a for a in attempts if a["data"]["succeeded"] is False]
    assert any(
        a["data"]["conflict_reason"] == "oscillation_blocked"
        for a in rejected
    )

def test_dry_run_permits_oscillation_emits_event(
    captured_events, seeded_state
):
    _reseed_state(seeded_state, LifecycleState.SLEEP)
    coord = _make_coord(seeded_state, min_interval_sec=60.0, dry_run=True)

    asyncio.run(
        coord.transition(
            LifecycleState.SLEEP,
            LifecycleState.WAKE,
            "wake_on_signal_consumed",
        )
    )
    new = asyncio.run(
        coord.transition(
            LifecycleState.WAKE,
            LifecycleState.SLEEP,
            "sleep_on_idle_30min",
        )
    )

    assert new == LifecycleState.SLEEP
    assert coord.version == 2

    rec = load_state(seeded_state)
    assert rec["current_state"] == "SLEEP"

    blocks = captured_events.by_kind("s2_oscillation_blocked")
    assert len(blocks) == 1
    assert blocks[0]["data"]["dry_run_mode"] is True

    attempts = captured_events.by_kind("s2_transition_attempt")
    assert len(attempts) == 2
    assert all(a["data"]["succeeded"] is True for a in attempts)

def test_event_body_shape_contract(captured_events, seeded_state):
    coord = _make_coord(seeded_state, min_interval_sec=60.0)

    asyncio.run(
        coord.transition(LifecycleState.WAKE, LifecycleState.DROWSY, "r1")
    )
    with pytest.raises(S2OscillationConflict):
        asyncio.run(
            coord.transition(
                LifecycleState.WAKE, LifecycleState.SLEEP, "r2"
            )
        )
    asyncio.run(
        coord.transition(LifecycleState.DROWSY, LifecycleState.SLEEP, "r3")
    )
    with pytest.raises(S2OscillationBlocked):
        asyncio.run(
            coord.transition(
                LifecycleState.SLEEP, LifecycleState.DROWSY, "r4"
            )
        )

    attempts = captured_events.by_kind("s2_transition_attempt")
    assert len(attempts) == 4, [a["data"] for a in attempts]
    expected_attempt_keys = {
        "from_state",
        "to_state",
        "reason",
        "version_before",
        "version_after",
        "succeeded",
        "conflict_reason",
    }
    for a in attempts:
        body = a["data"]
        assert set(body.keys()) == expected_attempt_keys
        assert isinstance(body["from_state"], str)
        assert isinstance(body["to_state"], str)
        assert isinstance(body["reason"], str)
        assert isinstance(body["version_before"], int)
        assert isinstance(body["version_after"], int)
        assert isinstance(body["succeeded"], bool)
        assert body["conflict_reason"] is None or isinstance(
            body["conflict_reason"], str
        )

    blocks = captured_events.by_kind("s2_oscillation_blocked")
    assert len(blocks) == 1, [b["data"] for b in blocks]
    expected_block_keys = {
        "first_transition",
        "second_transition",
        "interval_sec",
        "dry_run_mode",
    }
    body = blocks[0]["data"]
    assert set(body.keys()) == expected_block_keys
    assert isinstance(body["first_transition"], dict)
    assert isinstance(body["second_transition"], dict)
    assert isinstance(body["interval_sec"], float)
    assert isinstance(body["dry_run_mode"], bool)
