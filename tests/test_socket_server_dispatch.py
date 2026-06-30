from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest

from iai_mcp._ipc import IS_WINDOWS, open_ipc_connection


def _endpoint_ready_path(sock_path: Path) -> Path:
    """Path that exists once the SocketServer has bound its endpoint.
    POSIX: the unix socket file. Windows: the TCP port file written alongside
    it (``<sock_path>.port``, see iai_mcp._ipc._port_file_path)."""
    return Path(f"{sock_path}.port") if IS_WINDOWS else sock_path


@pytest.fixture
def short_socket_paths(tmp_path, monkeypatch):
    from iai_mcp import concurrency, daemon_state

    state_path = tmp_path / ".daemon-state.json"
    monkeypatch.setattr(daemon_state, "STATE_PATH", state_path)
    store_root = tmp_path / "store_root"
    store_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    with tempfile.TemporaryDirectory(prefix="iai-sock-") as sock_dir_name:
        sock_dir = Path(sock_dir_name)
        sock_path = sock_dir / "d.sock"
        monkeypatch.setattr(concurrency, "SOCKET_PATH", sock_path)
        # Isolate the IPC endpoint per-test. POSIX uses this as the unix socket
        # path; Windows persists the ephemeral TCP port to "<sock_path>.port".
        # Both SocketServer.serve() and open_ipc_connection() resolve through it,
        # so concurrent tests never collide on the shared default endpoint.
        monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(sock_path))

        try:
            yield None, sock_path, state_path
        finally:
            try:
                if sock_path.exists():
                    sock_path.unlink()
            except OSError:
                pass

async def _send_jsonrpc(
    sock_path: Path,
    method: str,
    params: dict | None = None,
    req_id: int | str = 1,
    *,
    timeout: float = 10.0,
) -> dict:
    reader, writer = await open_ipc_connection(timeout=timeout)
    try:
        envelope: dict = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            envelope["params"] = params
        writer.write((json.dumps(envelope) + "\n").encode("utf-8"))
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
    if not line:
        raise AssertionError(f"daemon closed without reply (method={method})")
    return json.loads(line.decode("utf-8"))

async def _send_raw(sock_path: Path, raw_bytes: bytes, *, timeout: float = 5.0) -> dict:
    reader, writer = await open_ipc_connection(timeout=timeout)
    try:
        writer.write(raw_bytes)
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

async def _with_socket_server(sock_path: Path, store, coro_fn):
    from iai_mcp.socket_server import SocketServer

    srv = SocketServer(store, idle_secs=99999)
    # No socket_path: serve() resolves the endpoint from IAI_DAEMON_SOCKET_PATH
    # (set by the fixture) — a unix socket on POSIX, TCP loopback on Windows.
    server_task = asyncio.create_task(srv.serve())

    ready_path = _endpoint_ready_path(sock_path)
    for _ in range(250):
        if ready_path.exists():
            break
        await asyncio.sleep(0.01)
    if not ready_path.exists():
        srv.shutdown_event.set()
        try:
            await asyncio.wait_for(server_task, timeout=5)
        except Exception:
            pass
        raise AssertionError("socket never bound")

    try:
        result = await coro_fn(sock_path, store)
    finally:
        srv.shutdown_event.set()
        try:
            await asyncio.wait_for(server_task, timeout=5)
        except Exception:
            pass
    return result

def test_memory_recall_routed_over_socket(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    from iai_mcp.store import MemoryStore

    store = MemoryStore()

    async def _runner(sock_path, store):
        return await _send_jsonrpc(
            sock_path, "memory_recall",
            {"cue": "phase 7 done", "budget_tokens": 400},
            req_id=1,
        )

    resp = asyncio.run(_with_socket_server(sock_path, store, _runner))

    assert resp["jsonrpc"] == "2.0", resp
    assert resp["id"] == 1, resp
    assert "result" in resp, resp
    assert "hits" in resp["result"], resp["result"]

def test_session_start_payload_routed(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    from iai_mcp.store import MemoryStore

    store = MemoryStore()

    async def _runner(sock_path, store):
        return await _send_jsonrpc(
            sock_path, "session_start_payload", {}, req_id=2,
        )

    resp = asyncio.run(_with_socket_server(sock_path, store, _runner))

    assert resp["jsonrpc"] == "2.0", resp
    assert resp["id"] == 2, resp
    assert "result" in resp, resp
    result = resp["result"]
    assert "l0" in result and "l1" in result, result
    assert "wake_depth" in result, result

def test_profile_get_routed(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    from iai_mcp.store import MemoryStore

    store = MemoryStore()

    async def _runner(sock_path, store):
        return await _send_jsonrpc(
            sock_path, "profile_get", {"knob": None}, req_id=3,
        )

    resp = asyncio.run(_with_socket_server(sock_path, store, _runner))

    assert resp["jsonrpc"] == "2.0", resp
    assert resp["id"] == 3, resp
    assert "result" in resp, resp
    result = resp["result"]
    assert isinstance(result, dict), result
    assert "live" in result, result
    assert "literal_preservation" in result["live"], result

def test_topology_routed(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    from iai_mcp.store import MemoryStore

    store = MemoryStore()

    async def _runner(sock_path, store):
        return await _send_jsonrpc(sock_path, "topology", {}, req_id=4)

    resp = asyncio.run(_with_socket_server(sock_path, store, _runner))

    assert resp["jsonrpc"] == "2.0", resp
    assert resp["id"] == 4, resp
    assert "result" in resp, resp
    result = resp["result"]
    assert "regime" in result or "sigma" in result, result

def test_invalid_jsonrpc_returns_minus_32600(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    from iai_mcp.store import MemoryStore

    store = MemoryStore()

    async def _runner(sock_path, store):
        bad = {"jsonrpc": "1.0", "id": 1, "method": "memory_recall"}
        return await _send_raw(
            sock_path, (json.dumps(bad) + "\n").encode("utf-8"),
        )

    resp = asyncio.run(_with_socket_server(sock_path, store, _runner))

    assert resp["jsonrpc"] == "2.0", resp
    assert "error" in resp, resp
    assert resp["error"]["code"] == -32600, resp

def test_unknown_method_returns_minus_32601(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    from iai_mcp.store import MemoryStore

    store = MemoryStore()

    async def _runner(sock_path, store):
        return await _send_jsonrpc(
            sock_path, "not_a_real_method", {}, req_id=5,
        )

    resp = asyncio.run(_with_socket_server(sock_path, store, _runner))

    assert resp["jsonrpc"] == "2.0", resp
    assert resp["id"] == 5, resp
    assert "error" in resp, resp
    assert "result" not in resp, resp
    assert resp["error"]["code"] == -32601, resp
    assert "not_a_real_method" in resp["error"]["message"], resp

def test_missing_required_param_returns_minus_32602(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    from iai_mcp.store import MemoryStore

    store = MemoryStore()

    async def _runner(sock_path, store):
        return await _send_jsonrpc(
            sock_path, "memory_recall", {}, req_id=6,
        )

    resp = asyncio.run(_with_socket_server(sock_path, store, _runner))

    assert resp["jsonrpc"] == "2.0", resp
    assert resp["id"] == 6, resp
    assert "error" in resp, resp
    assert "result" not in resp, resp
    assert resp["error"]["code"] == -32602, resp
    msg = resp["error"]["message"]
    assert "missing required param" in msg, resp
    assert "cue" in msg, resp

def test_id_echoed_unchanged(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    from iai_mcp.store import MemoryStore

    store = MemoryStore()

    async def _runner(sock_path, store):
        r1 = await _send_jsonrpc(
            sock_path, "session_start_payload", {}, req_id=1,
        )
        r2 = await _send_jsonrpc(
            sock_path, "session_start_payload", {}, req_id=999,
        )
        r3 = await _send_jsonrpc(
            sock_path, "session_start_payload", {}, req_id="some-string-id",
        )
        return r1, r2, r3

    r1, r2, r3 = asyncio.run(_with_socket_server(sock_path, store, _runner))

    assert r1["id"] == 1, r1
    assert r2["id"] == 999, r2
    assert r3["id"] == "some-string-id", r3

def test_unknown_method_does_not_crash_server(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    from iai_mcp.store import MemoryStore

    store = MemoryStore()

    async def _runner(sock_path, store):
        r_bad = await _send_jsonrpc(
            sock_path, "definitely_not_a_method", {}, req_id=100,
        )
        r_good = await _send_jsonrpc(
            sock_path, "session_start_payload", {}, req_id=101,
        )
        return r_bad, r_good

    r_bad, r_good = asyncio.run(_with_socket_server(sock_path, store, _runner))

    assert r_bad["id"] == 100, r_bad
    assert "error" in r_bad, r_bad
    assert "result" not in r_bad, r_bad
    assert r_good["id"] == 101, r_good
    assert "result" in r_good, r_good

def test_parse_error_returns_minus_32700(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    from iai_mcp.store import MemoryStore

    store = MemoryStore()

    async def _runner(sock_path, store):
        return await _send_raw(sock_path, b"not valid json\n")

    resp = asyncio.run(_with_socket_server(sock_path, store, _runner))

    assert resp["jsonrpc"] == "2.0", resp
    assert resp["id"] is None, resp
    assert "error" in resp, resp
    assert resp["error"]["code"] == -32700, resp

def test_empty_params_defaults_to_empty_dict(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    from iai_mcp.store import MemoryStore

    store = MemoryStore()

    async def _runner(sock_path, store):
        return await _send_jsonrpc(
            sock_path, "session_start_payload", None, req_id=200,
        )

    resp = asyncio.run(_with_socket_server(sock_path, store, _runner))

    assert resp["jsonrpc"] == "2.0", resp
    assert resp["id"] == 200, resp
    assert "result" in resp, resp
