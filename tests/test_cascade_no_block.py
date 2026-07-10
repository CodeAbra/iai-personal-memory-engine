from __future__ import annotations

import asyncio
import concurrent.futures
import time
from unittest.mock import patch


def test_concurrent_coroutine_completes_under_100ms_while_cascade_sleeps_5s(monkeypatch):
    asyncio.run(_concurrent_coroutine_completes_under_100ms_body(monkeypatch))


async def _concurrent_coroutine_completes_under_100ms_body(monkeypatch):
    sleep_duration = 5.0
    sentinel_assignment = type("Asgmt", (), {"top_communities": [], "mid_regions": {}})()

    def slow_blocking_stub(store):
        time.sleep(sleep_duration)
        return (None, sentinel_assignment, [])

    def fast_cascade_stub(store, assignment, **kwargs):
        return ([], [])

    state_holder = {
        "fsm_state": "WAKE",
        "hippea_cascade_request": {"pending": True, "session_id": "test"},
    }

    def load_state_stub():
        return dict(state_holder)

    def save_state_stub(state):
        state_holder.clear()
        state_holder.update(state)

    def write_event_stub(*args, **kwargs):
        return None

    shutdown = asyncio.Event()

    import iai_mcp.daemon as daemon_mod
    monkeypatch.setattr(daemon_mod, "_last_cascade_completed_at", 0.0)

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="cascade-test"
    )
    monkeypatch.setattr(daemon_mod, "_cascade_executor", executor)

    with patch("iai_mcp.retrieve.build_runtime_graph", slow_blocking_stub), \
         patch("iai_mcp.hippea_cascade.compute_and_fetch_warm", fast_cascade_stub), \
         patch("iai_mcp.daemon_state.load_state", load_state_stub), \
         patch("iai_mcp.daemon_state.save_state", save_state_stub), \
         patch("iai_mcp.daemon.write_event", write_event_stub):

        cascade_task = asyncio.create_task(
            daemon_mod._hippea_cascade_loop(store=None, shutdown=shutdown),
        )

        await asyncio.sleep(0.2)

        t_start = time.monotonic()
        await asyncio.sleep(0.01)
        await asyncio.sleep(0.01)
        elapsed = time.monotonic() - t_start

        shutdown.set()
        try:
            await asyncio.wait_for(cascade_task, timeout=sleep_duration + 2.0)
        except asyncio.TimeoutError:
            cascade_task.cancel()
            try:
                await cascade_task
            except (asyncio.CancelledError, Exception):
                pass

    executor.shutdown(wait=False)

    assert elapsed < 0.1, (
        f"R1 FAIL: event loop pinned for {elapsed:.3f}s while cascade body "
        f"was running. Expected <100ms. Did the "
        f"`await asyncio.to_thread(retrieve.build_runtime_graph, store)` "
        f"wrap land in src/iai_mcp/daemon/_watchdog.py::_hippea_cascade_loop?"
    )
