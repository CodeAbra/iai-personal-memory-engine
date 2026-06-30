from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest

from iai_mcp._ipc import IS_WINDOWS, open_ipc_connection


def _endpoint_ready_path(sock_path: Path) -> Path:
    """Path that exists once the control socket has bound: the unix socket on
    POSIX, the TCP port file (``<sock_path>.port``) on Windows."""
    return Path(f"{sock_path}.port") if IS_WINDOWS else sock_path


@pytest.fixture
def short_socket_paths(tmp_path, monkeypatch):
    from iai_mcp import concurrency, daemon_state

    state_path = tmp_path / ".daemon-state.json"

    with tempfile.TemporaryDirectory(prefix="iai-sock-") as sock_dir_name:
        sock_dir = Path(sock_dir_name)
        sock_path = sock_dir / "d.sock"
        monkeypatch.setattr(concurrency, "SOCKET_PATH", sock_path)
        monkeypatch.setattr(daemon_state, "STATE_PATH", state_path)
        # Per-test endpoint isolation (unix socket on POSIX; TCP port file on
        # Windows) via the env var start_ipc_server/open_ipc_connection honor.
        monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(sock_path))

        try:
            yield None, sock_path, state_path
        finally:
            try:
                if sock_path.exists():
                    sock_path.unlink()
            except OSError:
                pass


async def _send_ndjson(sock_path: Path, message: dict, *, timeout: float = 5.0) -> dict:
    reader, writer = await open_ipc_connection(timeout=timeout)
    try:
        writer.write((json.dumps(message) + "\n").encode("utf-8"))
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
    if not line:
        raise AssertionError("daemon closed without reply")
    return json.loads(line.decode("utf-8"))


async def _with_real_dispatcher(sock_path: Path, state: dict, coro_fn):
    from iai_mcp.concurrency import serve_control_socket

    shutdown = asyncio.Event()
    server_task = asyncio.create_task(
        serve_control_socket(
            store=None,
            state=state,
            shutdown=shutdown,
            socket_path=sock_path,
        ),
    )
    ready_path = _endpoint_ready_path(sock_path)
    for _ in range(250):
        if ready_path.exists():
            break
        await asyncio.sleep(0.01)
    if not ready_path.exists():
        shutdown.set()
        await asyncio.wait_for(server_task, timeout=5)
        raise AssertionError("socket never bound")

    try:
        result = await coro_fn(sock_path, state)
    finally:
        shutdown.set()
        try:
            await asyncio.wait_for(server_task, timeout=5)
        except Exception:
            pass
    return result


def test_status_returns_version_and_full_snapshot(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    from iai_mcp import __version__ as pkg_version

    state = {
        "fsm_state": "WAKE",
        "daemon_started_at": "2026-04-18T00:00:00+00:00",
        "last_tick_at": "2026-04-18T12:30:00+00:00",
        "quiet_window": [44, 16],
        "pending_digest": {
            "rem_cycles_completed": 2,
            "episodes_processed": 15,
            "schemas_induced_tier0": 3,
            "claude_call_used": True,
            "main_insight_text": "deeply long verbose insight text " * 50,
        },
        "scheduler_paused": False,
    }

    async def _runner(sock_path, state):
        return await _send_ndjson(sock_path, {"type": "status"})

    resp = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))

    assert resp["ok"] is True
    assert resp["state"] == "WAKE"
    assert isinstance(resp["uptime_sec"], (int, float))
    assert resp["version"] == pkg_version
    assert resp["fsm_state"] == "WAKE"
    assert resp["last_tick_at"] == "2026-04-18T12:30:00+00:00"
    assert resp["quiet_window"] == [44, 16]
    assert resp["daemon_started_at"] == "2026-04-18T00:00:00+00:00"
    assert resp["scheduler_paused"] is False
    pd = resp["pending_digest"]
    assert pd["rem_cycles_completed"] == 2
    assert pd["episodes_processed"] == 15
    assert pd["schemas_induced_tier0"] == 3
    assert pd["claude_call_used"] is True
    assert "main_insight_text" not in pd, (
        "truncated digest leaked verbose text over the socket"
    )


def test_user_initiated_sleep_sets_pending_flag(short_socket_paths):
    _, sock_path, state_path = short_socket_paths
    state = {"fsm_state": "WAKE"}

    async def _runner(sock_path, state):
        return await _send_ndjson(
            sock_path,
            {
                "type": "user_initiated_sleep",
                "reason": "I am going to bed",
                "ts": "2026-04-18T23:00:00+00:00",
            },
        )

    resp = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))

    assert resp == {"ok": True, "state": "TRANSITIONING"}

    from iai_mcp.daemon_state import load_state
    loaded = load_state()
    req = loaded["user_sleep_request"]
    assert req["pending"] is True
    assert req["reason"] == "I am going to bed"
    assert req["ts"] == "2026-04-18T23:00:00+00:00"


def test_user_initiated_sleep_rejects_when_already_sleeping(short_socket_paths):
    _, sock_path, state_path = short_socket_paths
    state = {"fsm_state": "DREAMING"}

    async def _runner(sock_path, state):
        return await _send_ndjson(
            sock_path,
            {
                "type": "user_initiated_sleep",
                "reason": "redundant",
                "ts": "2026-04-18T23:00:00+00:00",
            },
        )

    resp = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))

    assert resp == {"ok": False, "reason": "already_sleeping"}

    from iai_mcp.daemon_state import load_state
    loaded = load_state()
    assert "user_sleep_request" not in loaded


def test_force_wake_queues_flag(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    state = {"fsm_state": "DREAMING"}

    async def _runner(sock_path, state):
        return await _send_ndjson(
            sock_path,
            {"type": "force_wake", "ts": "2026-04-18T23:45:00+00:00"},
        )

    resp = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))
    assert resp == {"ok": True, "reason": "wake_queued"}

    from iai_mcp.daemon_state import load_state
    loaded = load_state()
    assert loaded["force_wake_request"]["pending"] is True
    assert loaded["force_wake_request"]["ts"] == "2026-04-18T23:45:00+00:00"


def test_force_rem_queues_flag(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    state = {"fsm_state": "WAKE"}

    async def _runner(sock_path, state):
        return await _send_ndjson(
            sock_path,
            {"type": "force_rem", "ts": "2026-04-18T10:00:00+00:00"},
        )

    resp = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))
    assert resp == {"ok": True, "reason": "rem_queued"}

    from iai_mcp.daemon_state import load_state
    loaded = load_state()
    assert loaded["force_rem_request"]["pending"] is True
    assert loaded["force_rem_request"]["ts"] == "2026-04-18T10:00:00+00:00"


def test_pause_then_resume_flips_flag(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    state = {"fsm_state": "WAKE"}

    async def _runner(sock_path, state):
        r1 = await _send_ndjson(sock_path, {"type": "pause"})
        r2 = await _send_ndjson(sock_path, {"type": "resume"})
        return r1, r2

    r1, r2 = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))

    assert r1 == {"ok": True, "paused": True}
    assert r2 == {"ok": True, "paused": False}

    from iai_mcp.daemon_state import load_state
    loaded = load_state()
    assert loaded["scheduler_paused"] is False


def test_pause_persists_True_before_resume(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    state = {"fsm_state": "WAKE"}

    async def _runner(sock_path, state):
        return await _send_ndjson(sock_path, {"type": "pause"})

    resp = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))
    assert resp == {"ok": True, "paused": True}

    from iai_mcp.daemon_state import load_state
    loaded = load_state()
    assert loaded["scheduler_paused"] is True


def test_unknown_message_type_returns_error(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    state = {"fsm_state": "WAKE"}

    async def _runner(sock_path, state):
        return await _send_ndjson(
            sock_path,
            {"type": "nuke_from_orbit", "ts": "whatever"},
        )

    resp = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))

    assert resp["ok"] is False
    assert resp["reason"] == "unknown_message_type"
    assert resp["type"] == "nuke_from_orbit"


def test_invalid_message_missing_ts_on_force_wake(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    state = {"fsm_state": "WAKE"}

    async def _runner(sock_path, state):
        return await _send_ndjson(sock_path, {"type": "force_wake"})

    resp = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))

    assert resp["ok"] is False
    assert resp["reason"] == "invalid_message"
    assert "ts" in resp["error"]


def test_invalid_message_wrong_type_user_sleep(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    state = {"fsm_state": "WAKE"}

    async def _runner(sock_path, state):
        return await _send_ndjson(
            sock_path,
            {"type": "user_initiated_sleep", "reason": 42, "ts": "x"},
        )

    resp = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))

    assert resp["ok"] is False
    assert resp["reason"] == "invalid_message"
    assert "reason" in resp["error"]


def test_invalid_message_non_string_type(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    state = {"fsm_state": "WAKE"}

    async def _runner(sock_path, state):
        return await _send_ndjson(sock_path, {"type": 42})

    resp = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))
    assert resp["ok"] is False
    assert resp["reason"] == "invalid_message"


def test_invalid_message_pause_wrong_seconds_type(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    state = {"fsm_state": "WAKE"}

    async def _runner(sock_path, state):
        return await _send_ndjson(sock_path, {"type": "pause", "seconds": "forever"})

    resp = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))
    assert resp["ok"] is False
    assert resp["reason"] == "invalid_message"
    assert "seconds" in resp["error"]


def test_dispatcher_does_not_transition_fsm_directly(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    state = {"fsm_state": "WAKE"}

    async def _runner(sock_path, state):
        await _send_ndjson(
            sock_path,
            {
                "type": "user_initiated_sleep",
                "reason": "night",
                "ts": "2026-04-18T23:00:00+00:00",
            },
        )
        return state["fsm_state"]

    fsm_after = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))
    assert fsm_after == "WAKE"


def test_user_initiated_sleep_reason_clipped(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    state = {"fsm_state": "WAKE"}

    long_reason = "x" * 5000

    async def _runner(sock_path, state):
        return await _send_ndjson(
            sock_path,
            {
                "type": "user_initiated_sleep",
                "reason": long_reason,
                "ts": "2026-04-18T23:00:00+00:00",
            },
        )

    resp = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))
    assert resp == {"ok": True, "state": "TRANSITIONING"}

    from iai_mcp.daemon_state import load_state
    loaded = load_state()
    assert len(loaded["user_sleep_request"]["reason"]) == 500


def test_concurrent_clients_both_succeed(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    state = {"fsm_state": "WAKE"}

    async def _runner(sock_path, state):
        coro1 = _send_ndjson(
            sock_path,
            {"type": "force_rem", "ts": "2026-04-18T01:00:00+00:00"},
        )
        coro2 = _send_ndjson(sock_path, {"type": "pause"})
        results = await asyncio.gather(coro1, coro2)
        return results

    r1, r2 = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))

    assert r1 == {"ok": True, "reason": "rem_queued"}
    assert r2 == {"ok": True, "paused": True}

    from iai_mcp.daemon_state import load_state
    loaded = load_state()
    assert loaded["force_rem_request"]["pending"] is True
    assert loaded["scheduler_paused"] is True


def test_full_message_type_matrix_end_to_end(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    state = {
        "fsm_state": "WAKE",
        "daemon_started_at": "2026-04-18T00:00:00+00:00",
    }

    async def _runner(sock_path, state):
        out = {}
        out["status"] = await _send_ndjson(sock_path, {"type": "status"})
        out["user_initiated_sleep"] = await _send_ndjson(
            sock_path,
            {
                "type": "user_initiated_sleep",
                "reason": "bedtime",
                "ts": "2026-04-18T23:30:00+00:00",
            },
        )
        out["force_rem"] = await _send_ndjson(
            sock_path,
            {"type": "force_rem", "ts": "2026-04-18T23:31:00+00:00"},
        )
        out["force_wake"] = await _send_ndjson(
            sock_path,
            {"type": "force_wake", "ts": "2026-04-18T23:32:00+00:00"},
        )
        out["pause"] = await _send_ndjson(sock_path, {"type": "pause"})
        out["resume"] = await _send_ndjson(sock_path, {"type": "resume"})
        return out

    results = asyncio.run(_with_real_dispatcher(sock_path, state, _runner))

    assert results["status"]["ok"] is True
    assert results["status"]["fsm_state"] == "WAKE"
    assert results["user_initiated_sleep"] == {"ok": True, "state": "TRANSITIONING"}
    assert results["force_rem"] == {"ok": True, "reason": "rem_queued"}
    assert results["force_wake"] == {"ok": True, "reason": "wake_queued"}
    assert results["pause"] == {"ok": True, "paused": True}
    assert results["resume"] == {"ok": True, "paused": False}

    from iai_mcp.daemon_state import load_state
    loaded = load_state()
    assert loaded["user_sleep_request"]["pending"] is True
    assert loaded["force_rem_request"]["pending"] is True
    assert loaded["force_wake_request"]["pending"] is True
    assert loaded["scheduler_paused"] is False
