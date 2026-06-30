"""Regression: the socket server must track *real* memory traffic only.

`SocketServer.last_activity_ts` feeds the daemon's `_interrupt_check`: the sleep
/ consolidation pipeline defers whenever activity is recent (< 30s). The watchdog
probes daemon liveness with a `{"type": "status"}` control message every 7-30s
(`daemon/_watchdog.py::_probe_status_roundtrip`). When *every* inbound line —
including that probe — refreshed `last_activity_ts`, `_interrupt_check` was
perpetually True, so the cycle never completed, the daemon never hibernated, and
the wake-hook re-ran every tick (a ~200% CPU churn on any long-lived deployment).

The fix: refresh `last_activity_ts` only for dispatched JSON-RPC method calls
(recall/capture/etc.), never for control-plane messages. These tests lock that in.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest

from iai_mcp._ipc import IS_WINDOWS, open_ipc_connection


def _endpoint_ready_path(sock_path: Path) -> Path:
    """Path that exists once SocketServer has bound: the unix socket on POSIX,
    the TCP port file (``<sock_path>.port``) on Windows."""
    return Path(f"{sock_path}.port") if IS_WINDOWS else sock_path


@pytest.fixture
def short_socket_paths(tmp_path, monkeypatch):
    from iai_mcp import concurrency, daemon_state

    state_path = tmp_path / ".daemon-state.json"
    store_root = tmp_path / "store_root"
    store_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    with tempfile.TemporaryDirectory(prefix="iai-sock-") as sock_dir_name:
        sock_dir = Path(sock_dir_name)
        sock_path = sock_dir / "d.sock"
        monkeypatch.setattr(concurrency, "SOCKET_PATH", sock_path)
        monkeypatch.setattr(daemon_state, "STATE_PATH", state_path)
        # Per-test endpoint isolation (unix socket on POSIX; TCP port file on
        # Windows) via the env var both serve() and open_ipc_connection() honor.
        monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(sock_path))

        try:
            yield sock_path
        finally:
            try:
                if sock_path.exists():
                    sock_path.unlink()
            except OSError:
                pass


async def _send_line(sock_path: Path, payload: dict, *, timeout: float = 10.0) -> dict:
    reader, writer = await open_ipc_connection(timeout=timeout)
    try:
        writer.write((json.dumps(payload) + "\n").encode("utf-8"))
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass
    if not line:
        raise AssertionError(f"daemon closed without reply (payload={payload})")
    return json.loads(line.decode("utf-8"))


async def _serve(sock_path: Path, store, coro_fn):
    from iai_mcp.socket_server import SocketServer

    srv = SocketServer(store, idle_secs=99999)
    server_task = asyncio.create_task(srv.serve())
    ready_path = _endpoint_ready_path(sock_path)
    for _ in range(250):
        if ready_path.exists():
            break
        await asyncio.sleep(0.01)
    if not ready_path.exists():
        srv.shutdown_event.set()
        raise AssertionError("socket never bound")
    try:
        return await coro_fn(srv)
    finally:
        srv.shutdown_event.set()
        try:
            await asyncio.wait_for(server_task, timeout=5)
        except Exception:  # noqa: BLE001
            pass


def test_status_probe_does_not_refresh_last_activity(short_socket_paths):
    sock_path = short_socket_paths
    from iai_mcp.store import MemoryStore

    store = MemoryStore()

    async def _runner(srv):
        # Sentinel baseline so any refresh is detectable.
        srv.last_activity_ts = 0.0
        resp = await _send_line(sock_path, {"type": "status"})
        # The control message is answered (round-trip), proving it was received...
        assert resp is not None, resp
        # ...yet it must NOT have been counted as memory activity: the watchdog's
        # periodic probe would otherwise keep the daemon awake forever.
        assert srv.last_activity_ts == 0.0, (
            "status liveness probe wrongly refreshed last_activity_ts"
        )

    asyncio.run(_serve(sock_path, store, _runner))


def test_jsonrpc_method_refreshes_last_activity(short_socket_paths):
    sock_path = short_socket_paths
    from iai_mcp.store import MemoryStore

    store = MemoryStore()

    async def _runner(srv):
        srv.last_activity_ts = 0.0
        resp = await _send_line(
            sock_path,
            {"jsonrpc": "2.0", "id": 1, "method": "session_start_payload", "params": {}},
        )
        assert "result" in resp, resp
        # Real recall/capture traffic MUST refresh the activity clock so the sleep
        # pipeline correctly defers while the user is actively using memory.
        assert srv.last_activity_ts > 0.0, (
            "dispatched JSON-RPC method did not refresh last_activity_ts"
        )

    asyncio.run(_serve(sock_path, store, _runner))
