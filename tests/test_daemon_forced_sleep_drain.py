from __future__ import annotations

import asyncio
import inspect
import platform
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="daemon module is POSIX-only on this project",
)


def _states():
    from iai_mcp.lifecycle_state import LifecycleState
    return LifecycleState


def test_main_tick_calls_the_lifecycle_drain_helper():
    import iai_mcp.daemon as daemon_mod

    source = inspect.getsource(daemon_mod.main)
    assert "_maybe_drain_on_lifecycle_edge(" in source, (
        "main() must call the shared drain-gate helper, not an inline copy "
        "of the drain predicate"
    )


def test_cold_force_wake_to_sleep_drains_once():
    from iai_mcp.daemon import _maybe_drain_on_lifecycle_edge
    L = _states()

    calls: list[int] = []
    events: list[tuple] = []

    def write_event(store, kind, data, severity="info"):
        events.append((kind, data, severity))

    def drain_fn(store):
        calls.append(1)
        return {"files_drained": 1, "files_failed": 0}

    def queue_drain_fn(store, *, write_event_fn, phase=None):
        return None

    fired = asyncio.run(
        _maybe_drain_on_lifecycle_edge(
            L.WAKE,
            L.SLEEP,
            SimpleNamespace(),
            drain_fn=drain_fn,
            write_event_fn=write_event,
            queue_drain_fn=queue_drain_fn,
        )
    )

    assert fired is True
    assert len(calls) == 1, calls
    assert len(events) == 1, events
    kind, data, _severity = events[0]
    assert kind == "deferred_drain_sleep_edge", (
        "a forced WAKE->SLEEP collapse must not be labeled as a drowsy-edge "
        f"drain: {events}"
    )
    assert data.get("phase") == "sleep_edge", data


def test_dwell_then_force_drains_on_drowsy_and_sleep_edges():
    from iai_mcp.daemon import _maybe_drain_on_lifecycle_edge
    L = _states()

    def drain_fn(store):
        return {"files_drained": 0, "files_failed": 0}

    def write_event(store, kind, data, severity="info"):
        pass

    def queue_drain_fn(store, *, write_event_fn, phase=None):
        return None

    trajectory = [L.WAKE, L.DROWSY, L.DROWSY, L.SLEEP]
    triggers = 0
    prev = trajectory[0]
    for cur in trajectory[1:]:
        fired = asyncio.run(
            _maybe_drain_on_lifecycle_edge(
                prev,
                cur,
                SimpleNamespace(),
                drain_fn=drain_fn,
                write_event_fn=write_event,
                queue_drain_fn=queue_drain_fn,
            )
        )
        if fired:
            triggers += 1
        prev = cur

    assert triggers == 2, triggers


def test_natural_path_second_pass_is_quiet_zero_work():
    from iai_mcp.daemon import _maybe_drain_on_lifecycle_edge
    L = _states()

    events: list[tuple] = []
    call_results = iter(
        [
            {"files_drained": 1, "files_failed": 0},
            {"files_drained": 0, "files_failed": 0},
        ]
    )

    def drain_fn(store):
        return next(call_results)

    def write_event(store, kind, data, severity="info"):
        events.append((kind, data, severity))

    def queue_drain_fn(store, *, write_event_fn, phase=None):
        return None

    asyncio.run(
        _maybe_drain_on_lifecycle_edge(
            L.WAKE,
            L.DROWSY,
            SimpleNamespace(),
            drain_fn=drain_fn,
            write_event_fn=write_event,
            queue_drain_fn=queue_drain_fn,
        )
    )
    asyncio.run(
        _maybe_drain_on_lifecycle_edge(
            L.DROWSY,
            L.SLEEP,
            SimpleNamespace(),
            drain_fn=drain_fn,
            write_event_fn=write_event,
            queue_drain_fn=queue_drain_fn,
        )
    )

    drowsy_events = [e for e in events if e[0] == "deferred_drain_drowsy"]
    assert len(drowsy_events) == 1, events


def test_drain_failure_does_not_crash_helper():
    from iai_mcp.daemon import _maybe_drain_on_lifecycle_edge
    L = _states()

    events: list[tuple] = []

    def write_event(store, kind, data, severity="info"):
        events.append((kind, data, severity))

    def failing_drain(store):
        raise RuntimeError("drain blew up")

    def queue_drain_fn(store, *, write_event_fn, phase=None):
        return None

    fired = asyncio.run(
        _maybe_drain_on_lifecycle_edge(
            L.WAKE,
            L.SLEEP,
            SimpleNamespace(),
            drain_fn=failing_drain,
            write_event_fn=write_event,
            queue_drain_fn=queue_drain_fn,
        )
    )

    assert fired is True
    kinds = [e[0] for e in events]
    assert "deferred_drain_failed" in kinds, events
