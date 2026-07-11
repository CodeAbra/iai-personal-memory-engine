"""A health/liveness status probe over the daemon socket must not be counted
as external memory-operation activity.

The daemon's liveness watchdog opens a real unix connection and sends
``{"type": "status"}`` on a fixed cadence. The sleep pipeline's interrupt-check
defers the current step whenever ``now - last_activity_ts`` is within the recent
activity window. If the watchdog's own probe refreshes ``last_activity_ts``, the
window never elapses and the pipeline never completes a single step -- the daemon
starves its own consolidation by counting its own health probe as user activity.

Genuine external activity (a real JSON-RPC ``memory_recall``/``memory_capture``
request from a client session) MUST still refresh the timestamp so an active user
is never interrupted by a sleep cycle.
"""
from __future__ import annotations

import asyncio
import json

import pytest


class _FakeReader:
    """Feeds a fixed list of already-framed lines, then EOF."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)

    def at_eof(self) -> bool:
        return not self._lines

    async def readline(self) -> bytes:
        if not self._lines:
            return b""
        return self._lines.pop(0)


class _FakeWriter:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.chunks.append(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        return None

    async def wait_closed(self) -> None:
        return None


def _make_server():
    from iai_mcp.socket_server import SocketServer

    # A minimal state so the control-plane fork is wired; the status handler
    # only needs the control-plane dispatch to accept the message.
    state = {"fsm_state": "WAKE"}
    return SocketServer(store=None, idle_secs=99999, state=state)


def _run_one_line(srv, line: bytes) -> None:
    reader = _FakeReader([line])
    writer = _FakeWriter()
    asyncio.run(srv.handle(reader, writer))  # type: ignore[arg-type]


def test_status_probe_does_not_refresh_last_activity() -> None:
    srv = _make_server()
    # Push the baseline far into the past so any refresh is unmistakable.
    srv.last_activity_ts = 1.0
    before = srv.last_activity_ts

    _run_one_line(srv, (json.dumps({"type": "status"}) + "\n").encode("utf-8"))

    assert srv.last_activity_ts == before, (
        "a {'type': 'status'} health probe refreshed last_activity_ts; the "
        "daemon's own liveness watchdog would therefore be counted as user "
        "activity and would starve the sleep pipeline's interrupt-check"
    )


def test_real_jsonrpc_request_still_refreshes_last_activity() -> None:
    srv = _make_server()
    srv.last_activity_ts = 1.0
    before = srv.last_activity_ts

    # A real MCP client request. It will error internally (store is None), but
    # the activity timestamp must be refreshed regardless -- the point is that a
    # genuine external memory operation counts as activity.
    _run_one_line(
        srv,
        (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "memory_recall",
                    "params": {"cue": "x"},
                }
            )
            + "\n"
        ).encode("utf-8"),
    )

    assert srv.last_activity_ts > before, (
        "a genuine JSON-RPC client request did not refresh last_activity_ts; an "
        "actively-working user could be interrupted by a sleep cycle"
    )


def test_interrupt_window_elapses_despite_periodic_status_probes() -> None:
    """End-to-end at the signal level: with only status probes arriving, an
    interrupt-check built on last_activity_ts must eventually report 'no recent
    activity' so the sleep pipeline can advance."""
    import time

    from iai_mcp.daemon import INTERRUPT_RECENT_ACTIVITY_WINDOW_SEC

    srv = _make_server()
    # Simulate a daemon that has been idle (no external client) longer than the
    # window, then receives a health probe (as the watchdog does every ~30s).
    srv.last_activity_ts = time.monotonic() - (
        INTERRUPT_RECENT_ACTIVITY_WINDOW_SEC + 5.0
    )
    _run_one_line(srv, (json.dumps({"type": "status"}) + "\n").encode("utf-8"))

    def _interrupt_check() -> bool:
        if srv.active_connections > 0:
            return True
        elapsed = time.monotonic() - srv.last_activity_ts
        return elapsed < INTERRUPT_RECENT_ACTIVITY_WINDOW_SEC

    assert _interrupt_check() is False, (
        "interrupt-check reports recent activity after only a health probe; the "
        "sleep pipeline would defer forever on an idle daemon"
    )


def test_real_watchdog_probe_over_live_socket_does_not_block_interrupt_window() -> None:
    """End-to-end through the production code path: a REAL SocketServer, the REAL
    watchdog ``_probe_status_roundtrip``, and the REAL interrupt-check closure.

    The watchdog probe is the exact function the daemon's liveness thread runs.
    After it round-trips against the live socket, the interrupt-check (built on
    ``SocketServer.last_activity_ts``) must still report 'no recent external
    activity' once a short window has elapsed -- proving the sleep pipeline would
    advance instead of deferring forever.
    """
    import shutil
    import tempfile
    import time
    from pathlib import Path

    from iai_mcp.daemon._watchdog import _probe_status_roundtrip
    from iai_mcp.socket_server import SocketServer

    # A short, independently allocated dir -- tmp_path is too long for the
    # macOS sun_path 104-byte cap.
    sock_dir = Path(tempfile.mkdtemp(prefix="iai-wdprobe-"))
    sock_path = sock_dir / "d.sock"
    assert len(str(sock_path).encode("utf-8")) < 104, "socket path too long"

    # A short stand-in for the real 30s window/cadence: the watchdog probes
    # FASTER than the window closes (probe_period < window), exactly as the real
    # daemon (both ~30s but the probe keeps landing inside the window). Without
    # the fix each probe refreshes last_activity_ts, so the window NEVER elapses
    # across many probes -> perpetual defer. With the fix, probes don't refresh
    # it, so the window elapses despite continuous probing.
    window_sec = 0.4
    probe_period = 0.15
    total_probes = 8  # spans ~1.2s >> window; without the fix the check never clears

    async def _drive() -> bool:
        srv = SocketServer(store=None, idle_secs=99999, state={"fsm_state": "WAKE"})
        server_task = asyncio.create_task(srv.serve(socket_path=sock_path))
        try:
            for _ in range(500):
                if sock_path.exists():
                    break
                await asyncio.sleep(0.01)
            assert sock_path.exists(), "socket never bound"

            # Establish an idle baseline older than the window (no external
            # client has touched the socket).
            srv.last_activity_ts = time.monotonic() - (window_sec + 1.0)

            def _interrupt_check() -> bool:
                if srv.active_connections > 0:
                    return True
                elapsed = time.monotonic() - srv.last_activity_ts
                return elapsed < window_sec

            # Fire the REAL watchdog probe repeatedly, faster than the window,
            # exactly as the liveness thread does over the life of the daemon.
            # Track whether the interrupt-check EVER clears while probes keep
            # arriving -- that is the property the sleep pipeline needs.
            cleared_while_probing = False
            for _ in range(total_probes):
                probe_ok = await _probe_status_roundtrip(
                    str(sock_path), read_timeout=5.0
                )
                assert probe_ok, "watchdog status probe did not get a response"
                await asyncio.sleep(probe_period)
                if not _interrupt_check():
                    cleared_while_probing = True

            # Return True iff the pipeline would be STARVED (never cleared).
            return not cleared_while_probing
        finally:
            srv.shutdown_event.set()
            try:
                await asyncio.wait_for(server_task, timeout=5)
            except Exception:
                pass

    try:
        starved = asyncio.run(_drive())
    finally:
        shutil.rmtree(sock_dir, ignore_errors=True)
    assert starved is False, (
        "the interrupt-check NEVER cleared across continuous REAL watchdog "
        "status probes over the live socket; the sleep pipeline would be "
        "perpetually starved by the daemon's own liveness probe"
    )


def test_sleep_pipeline_completes_all_steps_under_continuous_watchdog_probes(
    tmp_path, monkeypatch,
) -> None:
    """The dominant end-to-end guard.

    A REAL ``SleepPipeline`` (steps stubbed to no-ops so this stays fast and
    hermetic) is driven by the REAL interrupt-check closure bound to a REAL
    ``SocketServer`` that receives continuous REAL watchdog status probes for the
    whole run. The pipeline must advance through every step -- SCHEMA_MINE ...
    RECALL_INDEX_REBUILD (the final, sidecar-persisting step) -- despite the
    probing. Before the fix, each watchdog probe refreshed ``last_activity_ts``
    and the interrupt-check deferred SCHEMA_MINE (step 0) forever; the pipeline
    never completed a single step.
    """
    import shutil
    import tempfile
    import time
    from pathlib import Path

    from iai_mcp.daemon._watchdog import _probe_status_roundtrip
    from iai_mcp.lifecycle_event_log import LifecycleEventLog
    from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline, SleepStep
    from iai_mcp.socket_server import SocketServer

    sock_dir = Path(tempfile.mkdtemp(prefix="iai-wdpipe-"))
    sock_path = sock_dir / "d.sock"
    assert len(str(sock_path).encode("utf-8")) < 104, "socket path too long"

    state_path = tmp_path / "lifecycle_state.json"
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    pipeline = SleepPipeline(
        store=None,
        lifecycle_state_path=state_path,
        event_log=LifecycleEventLog(log_dir=log_dir),
        quarantine_ttl_hours=24.0,
    )

    completed: list[SleepStep] = []

    def _make_noop(step: SleepStep):
        # Mirror the real step bodies: honor the interrupt-check as the first
        # action (returning (False, {}) when interrupted), then no-op the work.
        def _noop(interrupt_check):
            if pipeline._check_interrupt(step, 0, interrupt_check):
                return False, {}
            completed.append(step)
            return True, {}

        return _noop

    for attr, step in (
        ("_step_schema_mine", SleepStep.SCHEMA_MINE),
        ("_step_knob_tune", SleepStep.KNOB_TUNE),
        ("_step_optimize_hippo", SleepStep.OPTIMIZE_HIPPO),
        ("_step_hippo_cleanup", SleepStep.HIPPO_CLEANUP),
        ("_step_dream_decay", SleepStep.DREAM_DECAY),
        ("_step_erasure_agent", SleepStep.ERASURE_AGENT),
        ("_step_cluster_replay", SleepStep.CLUSTER_REPLAY),
        ("_step_reconsolidation", SleepStep.RECONSOLIDATION),
        ("_step_user_model_update", SleepStep.USER_MODEL_UPDATE),
        ("_step_dmn_reflection", SleepStep.DMN_REFLECTION),
        ("_step_crisis_recluster", SleepStep.CRISIS_RECLUSTER),
        ("_step_cluster_summary", SleepStep.CLUSTER_SUMMARY),
        ("_step_recall_index_rebuild", SleepStep.RECALL_INDEX_REBUILD),
    ):
        monkeypatch.setattr(pipeline, attr, _make_noop(step))

    # A shortened interrupt window keeps the test fast; the mechanism is
    # identical to the shipped 30s value (the interrupt-check closure below uses
    # it exactly as daemon/__init__.py does with
    # INTERRUPT_RECENT_ACTIVITY_WINDOW_SEC). Kept comfortably larger than the
    # per-tick scheduling/to_thread overhead so a tainted timestamp reliably
    # lands inside the window (reproducing the bug) rather than racing it.
    window_sec = 0.6

    async def _drive() -> None:
        srv = SocketServer(store=None, idle_secs=99999, state={"fsm_state": "WAKE"})
        server_task = asyncio.create_task(srv.serve(socket_path=sock_path))
        for _ in range(500):
            if sock_path.exists():
                break
            await asyncio.sleep(0.01)
        assert sock_path.exists(), "socket never bound"

        # Baseline older than the window: no external client has touched it.
        srv.last_activity_ts = time.monotonic() - (window_sec + 1.0)

        # The REAL interrupt-check closure (mirrors daemon/__init__.py:1326-1332).
        def _interrupt_check() -> bool:
            if srv.active_connections > 0:
                return True
            elapsed = time.monotonic() - srv.last_activity_ts
            return elapsed < window_sec

        try:
            # Drive the pipeline across ticks the way the daemon does: run() may
            # return interrupted; on the next tick it resumes from where it
            # deferred. Before EACH tick, fire a REAL watchdog status probe over
            # the live socket -- the shipped regime (a probe lands ~every 30s,
            # i.e. within the interrupt window of each tick). Bound the loop so a
            # STARVED pipeline (the bug) exits via deadline instead of hanging.
            deadline = time.monotonic() + 6.0
            while time.monotonic() < deadline:
                # A watchdog probe lands right before this tick's interrupt-check.
                probe_ok = await _probe_status_roundtrip(
                    str(sock_path), read_timeout=5.0
                )
                assert probe_ok, "watchdog status probe did not get a response"
                # Yield so the server task's handle() coroutine processes the
                # probe line (and, before the fix, refreshes last_activity_ts)
                # BEFORE this tick's interrupt-check reads it.
                await asyncio.sleep(0.05)

                res = await asyncio.to_thread(pipeline.run, _interrupt_check)
                if not res.get("interrupted"):
                    break
                # Without the fix, the probe above refreshed last_activity_ts so
                # the interrupt-check inside run() saw fresh activity and deferred
                # step 0 -- and every subsequent tick repeats it -> perpetual
                # starvation (the loop exits only via the deadline, leaving the
                # pipeline incomplete). With the fix, the probe does NOT refresh
                # the timestamp, so the (already-elapsed) window lets run()
                # advance a step each tick until completion.
        finally:
            srv.shutdown_event.set()
            try:
                await asyncio.wait_for(server_task, timeout=5)
            except Exception:
                pass

    try:
        asyncio.run(_drive())
    finally:
        shutil.rmtree(sock_dir, ignore_errors=True)

    expected_order = [
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
    assert completed == expected_order, (
        "the sleep pipeline did not run to completion under continuous watchdog "
        f"probing; completed steps: {[s.name for s in completed]} (the final "
        "RECALL_INDEX_REBUILD step re-persists the colindex sidecar and must be "
        "reached)"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
