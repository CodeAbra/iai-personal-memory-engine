"""Proves serve() never touches the POSIX transport on the Windows branch."""

from __future__ import annotations

import asyncio

import pytest


def test_serve_does_not_touch_start_unix_server_on_windows(monkeypatch):
    import iai_mcp._ipc as ipc_mod
    from iai_mcp.socket_server import SocketServer

    monkeypatch.setattr(ipc_mod, "IS_WINDOWS", True)
    # If any code path still references this attribute, it fails loudly.
    monkeypatch.delattr(asyncio, "start_unix_server", raising=False)

    called = {}

    class _FakeServer:
        def close(self) -> None:
            pass

        async def wait_closed(self) -> None:
            return None

    async def _fake_start_ipc_server(handler, socket_path, limit=None):
        called["hit"] = True
        return _FakeServer(), ("127.0.0.1", 0), False

    monkeypatch.setattr(ipc_mod, "start_ipc_server", _fake_start_ipc_server)
    monkeypatch.setattr(ipc_mod, "shutdown_ipc", lambda: None)

    state = {"fsm_state": "WAKE"}
    srv = SocketServer(store=None, idle_secs=99999, state=state)
    srv.shutdown_event.set()

    async def _scenario() -> None:
        await asyncio.wait_for(srv.serve(), timeout=5.0)

    asyncio.run(_scenario())

    assert called.get("hit") is True
