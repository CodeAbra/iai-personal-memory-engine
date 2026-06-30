from __future__ import annotations

import asyncio
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iai_mcp._ipc import IS_WINDOWS
from iai_mcp.cli import _send_jsonrpc_request

# This module asserts *which unix-socket path* the client routes to by spying on
# asyncio.open_unix_connection — a POSIX-only mechanism (Windows routes over TCP
# loopback via a port file, and open_unix_connection doesn't exist there). The
# equivalent Windows endpoint routing/isolation is covered by the _ipc
# port-file tests and the IAI_DAEMON_SOCKET_PATH isolation in PR #6.
pytestmark = pytest.mark.skipif(
    IS_WINDOWS,
    reason="POSIX unix-socket-path routing hermeticity; Windows routes via TCP port file (covered elsewhere)",
)

def _capture_stdout(fn) -> tuple[str, int]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = fn()
    return buf.getvalue(), rc

def _make_connected_asyncmock():
    foreign_response = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "result": {"events": [], "count": 0}}
    ).encode() + b"\n"

    reader = AsyncMock()
    reader.readline = AsyncMock(return_value=foreign_response)

    writer = AsyncMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()

    spy = AsyncMock(return_value=(reader, writer))
    return spy

class TestIsCustomStore:

    def test_custom_store_returns_true(self, tmp_path, monkeypatch):
        from iai_mcp.cli import _is_custom_store

        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
        monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)
        assert _is_custom_store() is True

    def test_unset_store_returns_false(self, monkeypatch):
        from iai_mcp.cli import _is_custom_store

        monkeypatch.delenv("IAI_MCP_STORE", raising=False)
        monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)
        assert _is_custom_store() is False

    def test_store_set_to_default_path_returns_false(self, monkeypatch):
        from iai_mcp.cli import _is_custom_store
        from iai_mcp import store as _store

        monkeypatch.setenv("IAI_MCP_STORE", str(_store.DEFAULT_STORAGE_PATH))
        monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)
        assert _is_custom_store() is False

    def test_socket_override_set_does_not_affect_predicate(self, tmp_path, monkeypatch):
        from iai_mcp.cli import _is_custom_store

        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
        monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", "/tmp/fake.sock")
        assert _is_custom_store() is True

class TestCustomStoreNoSocketSkipsConnect:

    def _seed_store(self, tmp_path):
        from iai_mcp.events import write_event
        from iai_mcp.store import MemoryStore

        store = MemoryStore(path=tmp_path)
        write_event(
            store,
            kind="s5_invariant_update",
            data={"anchor_id": "x", "new_record_id": "y"},
            severity="info",
            session_id="s1",
        )
        store.close()

    def test_audit_never_probes_default_socket(self, tmp_path, monkeypatch):
        from iai_mcp.cli import main as cli_main

        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
        monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)
        self._seed_store(tmp_path)

        spy = _make_connected_asyncmock()
        with patch("asyncio.open_unix_connection", spy):
            out, rc = _capture_stdout(lambda: cli_main(["audit"]))

        spy.assert_not_called()
        assert rc == 0
        assert "s5_invariant_update" in out

    def test_trajectory_never_probes_default_socket(self, tmp_path, monkeypatch):
        from iai_mcp.cli import main as cli_main

        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
        monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)

        spy = _make_connected_asyncmock()
        with patch("asyncio.open_unix_connection", spy):
            out, rc = _capture_stdout(lambda: cli_main(["trajectory"]))

        spy.assert_not_called()
        assert rc == 0

class TestDefaultStoreStillProbes:

    def test_default_store_probes_default_socket(self, monkeypatch):
        monkeypatch.delenv("IAI_MCP_STORE", raising=False)
        monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)

        spy = AsyncMock(side_effect=FileNotFoundError("no socket"))
        with patch("asyncio.open_unix_connection", spy):
            result = _send_jsonrpc_request("events_query", {"kind": "llm_health", "limit": 1})

        spy.assert_called_once()
        assert result is None

class TestSocketOverrideWins:

    def test_socket_override_with_custom_store_still_probes(self, tmp_path, monkeypatch):
        override_sock = "/tmp/iai-test-override.sock"
        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
        monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", override_sock)

        spy = AsyncMock(side_effect=FileNotFoundError("no socket"))
        with patch("asyncio.open_unix_connection", spy):
            result = _send_jsonrpc_request("events_query", {"kind": "llm_health", "limit": 1})

        spy.assert_called_once_with(override_sock)
        assert result is None

class TestSendSocketRequestOverride:

    def test_socket_request_uses_env_override(self, monkeypatch):
        from iai_mcp.cli import _send_socket_request

        override_sock = "/tmp/iai-control-override.sock"
        monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", override_sock)

        spy = AsyncMock(side_effect=FileNotFoundError("no socket"))
        with patch("asyncio.open_unix_connection", spy):
            result = _send_socket_request({"jsonrpc": "2.0", "id": 1, "method": "status", "params": {}})

        spy.assert_called_once_with(override_sock)
        assert result is None

    def test_socket_request_default_without_override(self, monkeypatch):
        from iai_mcp._ipc import SOCKET_PATH
        from iai_mcp.cli import _send_socket_request

        monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)

        spy = AsyncMock(side_effect=FileNotFoundError("no socket"))
        with patch("asyncio.open_unix_connection", spy):
            result = _send_socket_request({"jsonrpc": "2.0", "id": 1, "method": "status", "params": {}})

        spy.assert_called_once_with(str(SOCKET_PATH))
        assert result is None
