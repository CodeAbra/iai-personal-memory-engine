from __future__ import annotations

import asyncio
import json
import os
import platform
import socket
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="AF_UNIX inherited-fd protocol is POSIX-only in this test scope",
)

@contextmanager
def _bind_to_fd_3(sock_path: Path) -> Iterator[socket.socket]:
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(sock_path))
    listener.listen(128)
    try:
        try:
            saved_fd = os.dup(3)
        except OSError:
            saved_fd = None
        try:
            os.dup2(listener.fileno(), 3)
            yield listener
        finally:
            if saved_fd is not None:
                try:
                    os.dup2(saved_fd, 3)
                finally:
                    os.close(saved_fd)
            else:
                try:
                    os.close(3)
                except OSError:
                    pass
    finally:
        try:
            listener.close()
        except OSError:
            pass

def _short_sock_path(suffix: str) -> Path:
    sock_dir = Path(tempfile.mkdtemp(prefix=f"iai-sock-{suffix}-"))
    return sock_dir / "d.sock"

def _cleanup_sock(sock_path: Path) -> None:
    try:
        if sock_path.exists():
            sock_path.unlink()
    except OSError:
        pass
    try:
        sock_path.parent.rmdir()
    except OSError:
        pass

def test_inherit_returns_none_when_env_missing(monkeypatch):
    from iai_mcp.socket_server import _inherit_activated_socket

    monkeypatch.delenv("LISTEN_FDS", raising=False)
    monkeypatch.delenv("LISTEN_PID", raising=False)

    assert _inherit_activated_socket() is None

def test_inherit_returns_none_when_pid_mismatch(monkeypatch):
    from iai_mcp.socket_server import _inherit_activated_socket

    monkeypatch.setenv("LISTEN_FDS", "1")
    monkeypatch.setenv("LISTEN_PID", "999999")

    assert _inherit_activated_socket() is None

def test_inherit_returns_none_when_fds_zero(monkeypatch):
    from iai_mcp.socket_server import _inherit_activated_socket

    monkeypatch.setenv("LISTEN_FDS", "0")
    monkeypatch.setenv("LISTEN_PID", str(os.getpid()))

    assert _inherit_activated_socket() is None

def test_inherit_returns_none_on_non_integer(monkeypatch):
    from iai_mcp.socket_server import _inherit_activated_socket

    monkeypatch.setenv("LISTEN_FDS", "foo")
    monkeypatch.setenv("LISTEN_PID", str(os.getpid()))

    result = _inherit_activated_socket()
    assert result is None

def test_inherit_returns_socket_when_env_correct_simulated(monkeypatch):
    from iai_mcp.socket_server import _inherit_activated_socket

    sock_path = _short_sock_path("e")
    try:
        with _bind_to_fd_3(sock_path):
            monkeypatch.setenv("LISTEN_FDS", "1")
            monkeypatch.setenv("LISTEN_PID", str(os.getpid()))

            inherited = _inherit_activated_socket()
            assert inherited is not None, "should have returned the inherited socket"
            try:
                assert inherited.getsockname() == str(sock_path), (
                    f"expected bound path {sock_path}, got {inherited.getsockname()}"
                )
                assert inherited.getblocking() is False, (
                    "inherited socket must be non-blocking"
                )
            finally:
                try:
                    inherited.close()
                except OSError:
                    pass
    finally:
        _cleanup_sock(sock_path)

async def _connect_and_send_jsonrpc(
    sock_path: Path, method: str, *, timeout: float = 5.0,
) -> dict:
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(path=str(sock_path)),
        timeout=timeout,
    )
    try:
        envelope = {"jsonrpc": "2.0", "id": 42, "method": method, "params": {}}
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
        raise AssertionError("daemon closed without reply")
    return json.loads(line.decode("utf-8"))

def test_serve_uses_inherited_socket_path(monkeypatch, tmp_path):
    store_root = tmp_path / "lancedb_root"
    store_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    from iai_mcp.socket_server import SocketServer
    from iai_mcp.store import MemoryStore

    sock_path = _short_sock_path("f")

    async def _runner() -> dict:
        return await _connect_and_send_jsonrpc(sock_path, "definitely_not_a_real_method")

    async def _scenario() -> dict:
        store = MemoryStore()
        srv = SocketServer(store, idle_secs=99999)
        os.environ["LISTEN_FDS"] = "1"
        os.environ["LISTEN_PID"] = str(os.getpid())
        try:
            server_task = asyncio.create_task(srv.serve(socket_path=sock_path))
            await asyncio.sleep(0.2)
            try:
                resp = await asyncio.wait_for(_runner(), timeout=5.0)
            finally:
                srv.shutdown_event.set()
                try:
                    await asyncio.wait_for(server_task, timeout=5)
                except Exception:
                    pass
            return resp
        finally:
            os.environ.pop("LISTEN_FDS", None)
            os.environ.pop("LISTEN_PID", None)

    try:
        with _bind_to_fd_3(sock_path):
            resp = asyncio.run(_scenario())
    finally:
        _cleanup_sock(sock_path)

    assert resp["jsonrpc"] == "2.0", resp
    assert resp["id"] == 42, resp
    assert "error" in resp, resp
    assert "result" not in resp, resp
    assert resp["error"]["code"] == -32601, resp
    assert "definitely_not_a_real_method" in resp["error"]["message"], resp
