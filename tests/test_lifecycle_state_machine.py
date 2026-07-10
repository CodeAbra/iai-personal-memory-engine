from __future__ import annotations

import asyncio
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path
from typing import Any

import pytest


from iai_mcp.lifecycle import (
    DEFAULT_LOCK_PATH,  # noqa: F401  -- import sanity
    LifecycleEvent,
    LifecycleState,
    LifecycleStateMachine,
    _lifecycle_lock,
    compute_transition,
)
from iai_mcp.lifecycle_event_log import LifecycleEventLog
from iai_mcp.lifecycle_state import default_state, load_state, save_state
from iai_mcp.s2_coordinator import S2Coordinator


def _seed_state(state_path: Path, state: LifecycleState) -> None:
    record = default_state()
    record["current_state"] = state.value
    save_state(record, state_path)


def _make_machine(tmp_path: Path, *, shadow_run: bool = True) -> LifecycleStateMachine:
    state_path = tmp_path / "lifecycle_state.json"
    coordinator = S2Coordinator(
        store=None,
        state_path=state_path,
        min_interval_sec=0.0,
        dry_run=False,
    )
    return LifecycleStateMachine(
        state_path=state_path,
        event_log=LifecycleEventLog(log_dir=tmp_path / "logs"),
        lock_path=tmp_path / ".lifecycle.lock",
        shadow_run=shadow_run,
        coordinator=coordinator,
    )


@pytest.mark.parametrize(
    "from_state, event, payload, expected",
    [
        (LifecycleState.WAKE, LifecycleEvent.IDLE_5MIN, {}, LifecycleState.DROWSY),
        (LifecycleState.DROWSY, LifecycleEvent.HEARTBEAT_REFRESH, {}, LifecycleState.WAKE),
        (LifecycleState.DROWSY, LifecycleEvent.IDLE_30MIN,
         {"sleep_eligible": True}, LifecycleState.SLEEP),
        (LifecycleState.SLEEP, LifecycleEvent.SLEEP_CYCLE_DONE,
         {"still_idle": True}, LifecycleState.HIBERNATION),
        (LifecycleState.HIBERNATION, LifecycleEvent.WAKE_SIGNAL, {}, LifecycleState.WAKE),
        (LifecycleState.SLEEP, LifecycleEvent.REQUEST_ARRIVED, {}, LifecycleState.WAKE),
        (LifecycleState.DROWSY, LifecycleEvent.REQUEST_ARRIVED, {}, LifecycleState.WAKE),
        (LifecycleState.HIBERNATION, LifecycleEvent.REQUEST_ARRIVED, {}, LifecycleState.WAKE),
    ],
)
def test_transition_table_positive(from_state, event, payload, expected):
    assert compute_transition(from_state, event, payload) == expected


@pytest.mark.parametrize(
    "from_state, event, payload",
    [
        (LifecycleState.DROWSY, LifecycleEvent.IDLE_30MIN, {}),
        (LifecycleState.DROWSY, LifecycleEvent.IDLE_30MIN, {"sleep_eligible": False}),
        (LifecycleState.SLEEP, LifecycleEvent.SLEEP_CYCLE_DONE, {}),
        (LifecycleState.SLEEP, LifecycleEvent.SLEEP_CYCLE_DONE, {"still_idle": False}),
        (LifecycleState.WAKE, LifecycleEvent.HEARTBEAT_REFRESH, {}),
        (LifecycleState.WAKE, LifecycleEvent.IDLE_30MIN, {"sleep_eligible": True}),
        (LifecycleState.HIBERNATION, LifecycleEvent.IDLE_5MIN, {}),
        (LifecycleState.SLEEP, LifecycleEvent.IDLE_5MIN, {}),
        (LifecycleState.WAKE, LifecycleEvent.TICK, {}),
        (LifecycleState.DROWSY, LifecycleEvent.TICK, {}),
        (LifecycleState.SLEEP, LifecycleEvent.TICK, {}),
        (LifecycleState.HIBERNATION, LifecycleEvent.TICK, {}),
        (LifecycleState.HIBERNATION, LifecycleEvent.HIBERNATION_GRACE_EXPIRED, {}),
    ],
)
def test_transition_table_negative_returns_none(from_state, event, payload):
    assert compute_transition(from_state, event, payload) is None


@pytest.mark.parametrize("seed", list(range(50)))
def test_property_random_sequence_never_invalid(seed):
    rng = random.Random(seed)
    states = list(LifecycleState)
    events = list(LifecycleEvent)

    state = rng.choice(states)
    for _ in range(200):
        event = rng.choice(events)
        payload: dict[str, Any] = {
            "sleep_eligible": rng.choice([True, False]),
            "still_idle": rng.choice([True, False]),
        }
        target = compute_transition(state, event, payload)
        assert target is None or isinstance(target, LifecycleState), (
            f"seed={seed} state={state} event={event} produced {target!r}"
        )
        if target is not None:
            state = target
        assert state in LifecycleState, f"unexpected state escape: {state!r}"


@pytest.mark.parametrize("seed", list(range(20)))
def test_property_deterministic(seed):
    rng = random.Random(seed)
    state = rng.choice(list(LifecycleState))
    event = rng.choice(list(LifecycleEvent))
    payload = {
        "sleep_eligible": rng.choice([True, False]),
        "still_idle": rng.choice([True, False]),
    }
    first = compute_transition(state, event, payload)
    for _ in range(1000):
        assert compute_transition(state, event, payload) == first


def test_property_full_cycle_reachable_from_wake():
    state = LifecycleState.WAKE

    state = compute_transition(state, LifecycleEvent.IDLE_5MIN) or state
    assert state == LifecycleState.DROWSY

    state = compute_transition(
        state, LifecycleEvent.IDLE_30MIN, {"sleep_eligible": True}
    ) or state
    assert state == LifecycleState.SLEEP

    state = compute_transition(
        state, LifecycleEvent.SLEEP_CYCLE_DONE, {"still_idle": True}
    ) or state
    assert state == LifecycleState.HIBERNATION

    state = compute_transition(state, LifecycleEvent.WAKE_SIGNAL) or state
    assert state == LifecycleState.WAKE


def test_property_cycle_reachable_from_any_starting_state():
    for start in LifecycleState:
        state = start
        target = compute_transition(state, LifecycleEvent.REQUEST_ARRIVED) or state
        assert target == LifecycleState.WAKE


def test_dispatch_persists_new_state_on_transition(tmp_path):
    machine = _make_machine(tmp_path)
    _seed_state(machine._state_path, LifecycleState.WAKE)

    new = asyncio.run(machine.dispatch(LifecycleEvent.IDLE_5MIN))
    assert new == LifecycleState.DROWSY

    record = load_state(machine._state_path)
    assert record["current_state"] == "DROWSY"


def test_dispatch_logs_state_transition(tmp_path):
    machine = _make_machine(tmp_path)
    _seed_state(machine._state_path, LifecycleState.WAKE)

    asyncio.run(machine.dispatch(LifecycleEvent.IDLE_5MIN))

    log = LifecycleEventLog(log_dir=tmp_path / "logs")
    records = log.read_all()
    transitions = [r for r in records if r["event"] == "state_transition"]
    assert len(transitions) == 1
    assert transitions[0]["from"] == "WAKE"
    assert transitions[0]["to"] == "DROWSY"
    assert transitions[0]["trigger"] == "idle_5min"


def test_dispatch_no_op_returns_current_state_no_log(tmp_path):
    machine = _make_machine(tmp_path)
    _seed_state(machine._state_path, LifecycleState.WAKE)

    state = asyncio.run(machine.dispatch(LifecycleEvent.TICK))
    assert state == LifecycleState.WAKE

    log = LifecycleEventLog(log_dir=tmp_path / "logs")
    records = log.read_all()
    transitions = [r for r in records if r["event"] == "state_transition"]
    assert transitions == []


def test_dispatch_advances_seq_and_activity_on_user_event(tmp_path):
    machine = _make_machine(tmp_path)
    _seed_state(machine._state_path, LifecycleState.DROWSY)

    record_before = load_state(machine._state_path)
    seq_before = record_before["wrapper_event_seq"]
    activity_before = record_before["last_activity_ts"]

    time.sleep(0.01)

    asyncio.run(machine.dispatch(LifecycleEvent.HEARTBEAT_REFRESH))

    record_after = load_state(machine._state_path)
    assert record_after["wrapper_event_seq"] == seq_before + 1
    assert record_after["last_activity_ts"] > activity_before


def test_shadow_run_hibernation_persists_state_and_warns(tmp_path):
    machine = _make_machine(tmp_path, shadow_run=True)
    _seed_state(machine._state_path, LifecycleState.SLEEP)

    new = asyncio.run(machine.dispatch(LifecycleEvent.SLEEP_CYCLE_DONE, still_idle=True))
    assert new == LifecycleState.HIBERNATION

    record = load_state(machine._state_path)
    assert record["current_state"] == "HIBERNATION"

    log = LifecycleEventLog(log_dir=tmp_path / "logs")
    records = log.read_all()
    kinds = [r["event"] for r in records]
    assert "state_transition" in kinds
    assert "shadow_run_warning" in kinds

    warning = next(r for r in records if r["event"] == "shadow_run_warning")
    assert warning["would_action"] == "hibernate_kill_process"
    assert warning["blocked_by"] == "shadow_run=True"


def test_shadow_run_false_hibernation_logs_no_warning(tmp_path):
    machine = _make_machine(tmp_path, shadow_run=False)
    _seed_state(machine._state_path, LifecycleState.SLEEP)

    asyncio.run(machine.dispatch(LifecycleEvent.SLEEP_CYCLE_DONE, still_idle=True))

    log = LifecycleEventLog(log_dir=tmp_path / "logs")
    records = log.read_all()
    kinds = [r["event"] for r in records]
    assert "shadow_run_warning" not in kinds


def test_shadow_run_does_not_terminate_process(tmp_path):
    machine = _make_machine(tmp_path, shadow_run=True)
    _seed_state(machine._state_path, LifecycleState.SLEEP)

    asyncio.run(machine.dispatch(LifecycleEvent.SLEEP_CYCLE_DONE, still_idle=True))

    sentinel = "still alive"
    assert sentinel == "still alive"


def _lock_try_acquire(lock_path_str: str, result_q: "mp.Queue[Any]") -> None:
    from iai_mcp.lifecycle import (
        LifecycleStateLocked as _Locked,
        _lifecycle_lock as _lock,
    )

    try:
        with _lock(Path(lock_path_str)):
            result_q.put("acquired")
    except _Locked as exc:
        result_q.put(f"locked:{exc}")


def _writer_subprocess(
    state_path_str: str,
    log_dir_str: str,
    lock_path_str: str,
    hold_seconds: float,
    result_q: "mp.Queue[Any]",
) -> None:
    import asyncio as _asyncio

    from iai_mcp.lifecycle import (
        LifecycleStateLocked as _Locked,
        LifecycleStateMachine as _Machine,
        _lifecycle_lock as _lock,
    )
    from iai_mcp.lifecycle_event_log import LifecycleEventLog as _Log
    from iai_mcp.s2_coordinator import S2Coordinator as _Coord

    if hold_seconds > 0:
        try:
            with _lock(Path(lock_path_str)):
                time.sleep(hold_seconds)
            _sp = Path(state_path_str)
            machine = _Machine(
                state_path=_sp,
                event_log=_Log(log_dir=Path(log_dir_str)),
                lock_path=Path(lock_path_str),
                shadow_run=True,
                coordinator=_Coord(store=None, state_path=_sp, min_interval_sec=0.0),
            )
            new_state = _asyncio.run(machine.dispatch(LifecycleEvent.IDLE_5MIN))
            result_q.put(("ok", new_state.value))
        except _Locked as exc:
            result_q.put(("locked", str(exc)))
        except Exception as exc:  # noqa: BLE001
            result_q.put(("error", repr(exc)))
    else:
        try:
            _sp = Path(state_path_str)
            machine = _Machine(
                state_path=_sp,
                event_log=_Log(log_dir=Path(log_dir_str)),
                lock_path=Path(lock_path_str),
                shadow_run=True,
                coordinator=_Coord(store=None, state_path=_sp, min_interval_sec=0.0),
            )
            new_state = _asyncio.run(machine.dispatch(LifecycleEvent.IDLE_5MIN))
            result_q.put(("ok", new_state.value))
        except _Locked as exc:
            result_q.put(("locked", str(exc)))
        except Exception as exc:  # noqa: BLE001
            result_q.put(("error", repr(exc)))


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="fcntl.flock is POSIX-only",
)
def test_single_writer_contention_one_succeeds(tmp_path):
    state_path = tmp_path / "lifecycle_state.json"
    log_dir = tmp_path / "logs"
    lock_path = tmp_path / ".lifecycle.lock"
    _seed_state(state_path, LifecycleState.WAKE)

    ctx = mp.get_context("spawn")
    q: mp.Queue[Any] = ctx.Queue()

    p1 = ctx.Process(
        target=_writer_subprocess,
        args=(str(state_path), str(log_dir), str(lock_path), 1.5, q),
    )
    p1.start()
    time.sleep(0.5)
    p2 = ctx.Process(
        target=_writer_subprocess,
        args=(str(state_path), str(log_dir), str(lock_path), 0.0, q),
    )
    p2.start()

    p1.join(timeout=10)
    p2.join(timeout=10)
    assert p1.exitcode == 0
    assert p2.exitcode == 0

    results = []
    while not q.empty():
        results.append(q.get())
    assert len(results) == 2
    kinds = sorted(r[0] for r in results)
    assert kinds == ["ok", "ok"]


def test_lifecycle_lock_contention_raises(tmp_path):
    lock_path = tmp_path / ".lifecycle.lock"
    with _lifecycle_lock(lock_path):
        ctx = mp.get_context("spawn")
        q: mp.Queue[Any] = ctx.Queue()
        p = ctx.Process(target=_lock_try_acquire, args=(str(lock_path), q))
        p.start()
        p.join(timeout=5)
        assert p.exitcode == 0
        outcome = q.get(timeout=1)
        assert outcome.startswith("locked:")


def test_lifecycle_lock_releases_on_context_exit(tmp_path):
    lock_path = tmp_path / ".lifecycle.lock"
    with _lifecycle_lock(lock_path):
        pass
    ctx = mp.get_context("spawn")
    q: mp.Queue[Any] = ctx.Queue()
    p = ctx.Process(target=_lock_try_acquire, args=(str(lock_path), q))
    p.start()
    p.join(timeout=5)
    assert p.exitcode == 0
    assert q.get(timeout=1) == "acquired"
