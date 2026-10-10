"""The `tools_exec_client` Windows-branch pieces: TCP-loopback connect gated
behind a mandatory auth-token handshake, run over a real local stub TCP
server. Forces the branch via a monkeypatched `IS_WINDOWS` rather than
requiring `sys.platform` to actually be `win32`, matching the pattern
already used by `tests/test_ipc_transport.py`."""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest


class _StubTcpServer:
    """Accepts one connection at a time, records the exact bytes received
    up to and including the first newline, and optionally replies."""

    def __init__(self, reply: bytes | None = None) -> None:
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)
        self._listener.settimeout(5.0)
        self.port = self._listener.getsockname()[1]
        self.first_lines: list[bytes] = []
        self._reply = reply
        self._stopped = False
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _accept_loop(self) -> None:
        while not self._stopped:
            try:
                conn, _ = self._listener.accept()
            except OSError:
                return
            with conn:
                conn.settimeout(5.0)
                first_line = self._read_line(conn)
                self.first_lines.append(first_line)
                if self._reply is not None:
                    self._read_line(conn)  # the JSON-RPC request line, discarded
                    try:
                        conn.sendall(self._reply)
                    except OSError:
                        pass

    @staticmethod
    def _read_line(conn: socket.socket) -> bytes:
        buf = b""
        try:
            while not buf.endswith(b"\n"):
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
        except OSError:
            pass
        return buf

    def close(self) -> None:
        self._stopped = True
        self._listener.close()
        self._thread.join(timeout=2.0)


@pytest.fixture
def windows_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from iai_mcp import tools_exec_client

    sock_path = tmp_path / ".execd.sock"
    monkeypatch.setenv("IAI_TOOLS_ROOT", str(tmp_path / "iai-tools-root"))
    monkeypatch.setenv("IAI_TOOLS_SOCKET_PATH", str(sock_path))
    monkeypatch.setattr(tools_exec_client, "IS_WINDOWS", True)
    yield {"sock_path": sock_path}


def _write_port(sock_path: Path, port: int) -> None:
    Path(f"{sock_path}.port").write_text(str(port), encoding="utf-8")


def _write_token(sock_path: Path, token: str) -> None:
    Path(f"{sock_path}.token").write_text(token, encoding="utf-8")


def test_windows_connect_sends_ondisk_token_as_mandatory_first_line(
    windows_env: dict,
) -> None:
    from iai_mcp import tools_exec_client

    sock_path = windows_env["sock_path"]
    token = "a" * 64
    server = _StubTcpServer()
    try:
        _write_token(sock_path, token)
        _write_port(sock_path, server.port)

        sock = tools_exec_client._connect(timeout=1.0)
        try:
            time.sleep(0.1)  # let the stub server's accept thread record the line
        finally:
            sock.close()

        assert server.first_lines == [f"{token}\n".encode("utf-8")]
    finally:
        server.close()


def test_windows_connect_refuses_without_token_zero_bytes_sent(
    windows_env: dict,
) -> None:
    from iai_mcp import tools_exec_client

    sock_path = windows_env["sock_path"]
    server = _StubTcpServer()
    try:
        # No token file written -- port file alone must not be enough.
        _write_port(sock_path, server.port)

        with pytest.raises(FileNotFoundError):
            tools_exec_client._connect(timeout=1.0)

        time.sleep(0.1)
        assert server.first_lines == [b""], "no bytes may reach the stub server without a token"
    finally:
        server.close()


def test_windows_connect_round_trip_returns_result_on_correct_token(
    windows_env: dict,
) -> None:
    from iai_mcp import tools_exec_client

    sock_path = windows_env["sock_path"]
    token = "b" * 64
    reply = b'{"jsonrpc": "2.0", "id": 1, "result": {"ok": true}}\n'
    server = _StubTcpServer(reply=reply)
    try:
        _write_token(sock_path, token)
        _write_port(sock_path, server.port)

        result = tools_exec_client._call("tool_run", {"command": "echo"})

        assert result == {"ok": True}
        assert server.first_lines[0].startswith(token.encode("utf-8"))
    finally:
        server.close()


def test_windows_connect_sends_token_read_from_file_not_a_cached_value(
    windows_env: dict,
) -> None:
    """The client must send whatever is currently on disk, never a value it
    generated or cached itself -- proves it authenticates against whichever
    daemon actually won the bind."""
    from iai_mcp import tools_exec_client

    sock_path = windows_env["sock_path"]
    on_disk_token = "c" * 64
    server = _StubTcpServer()
    try:
        _write_token(sock_path, on_disk_token)
        _write_port(sock_path, server.port)

        sock = tools_exec_client._connect(timeout=1.0)
        try:
            time.sleep(0.1)
        finally:
            sock.close()

        assert server.first_lines == [f"{on_disk_token}\n".encode("utf-8")]
    finally:
        server.close()


def test_posix_path_unchanged_when_windows_branch_is_off(monkeypatch, tmp_path: Path) -> None:
    """With IS_WINDOWS left at its real (non-Windows) value on this dev box,
    _connect must still take the AF_UNIX branch -- proves the new Windows
    branch is additive, not a replacement of the POSIX path."""
    from iai_mcp import tools_exec_client

    assert tools_exec_client.IS_WINDOWS is False

    sock_path = tmp_path / ".execd.sock"
    monkeypatch.setenv("IAI_TOOLS_SOCKET_PATH", str(sock_path))

    calls: list[int] = []
    real_socket = socket.socket

    def _tracking_socket(family, *args, **kwargs):
        calls.append(family)
        return real_socket(family, *args, **kwargs)

    monkeypatch.setattr(socket, "socket", _tracking_socket)

    with pytest.raises((ConnectionRefusedError, FileNotFoundError, OSError)):
        tools_exec_client._connect(timeout=0.2)

    assert socket.AF_INET not in calls, "POSIX path must never open an AF_INET socket"


class _FakeSpawnedProcess:
    pid = 424242


def test_windows_spawn_hands_fresh_token_via_env_and_clears_stale_files(
    windows_env: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iai_mcp import tools_exec_client

    sock_path = windows_env["sock_path"]
    _write_port(sock_path, 9999)
    _write_token(sock_path, "stale-token-from-a-crashed-daemon")

    captured: dict = {}

    def _fake_popen(args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs.get("env")
        return _FakeSpawnedProcess()

    monkeypatch.setattr(tools_exec_client.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(tools_exec_client, "_resolve_execd_bin", lambda: "/fake/iai_tools_execd.exe")

    tools_exec_client._spawn_daemon()

    assert not Path(f"{sock_path}.port").exists(), "stale port file must be cleared before spawn"
    assert not Path(f"{sock_path}.token").exists(), "stale token file must be cleared before spawn"
    token_env = captured["env"]["IAI_TOOLS_TOKEN"]
    assert len(token_env) == 64
    int(token_env, 16)  # must be valid hex
    assert token_env != "stale-token-from-a-crashed-daemon"


def test_posix_spawn_never_sets_token_env_or_clears_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from iai_mcp import tools_exec_client

    assert tools_exec_client.IS_WINDOWS is False

    sock_path = tmp_path / ".execd.sock"
    monkeypatch.setenv("IAI_TOOLS_SOCKET_PATH", str(sock_path))
    Path(f"{sock_path}.port").write_text("1234", encoding="utf-8")

    captured: dict = {}

    def _fake_popen(args, **kwargs):
        captured["env"] = kwargs.get("env")
        return _FakeSpawnedProcess()

    monkeypatch.setattr(tools_exec_client.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(tools_exec_client, "_resolve_execd_bin", lambda: "/fake/iai_tools_execd")

    tools_exec_client._spawn_daemon()

    assert "IAI_TOOLS_TOKEN" not in captured["env"]
    assert Path(f"{sock_path}.port").exists(), "POSIX spawn must not clear the Windows-only files"


def test_resolve_execd_bin_appends_exe_suffix_on_win32(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from iai_mcp import tools_exec_client

    monkeypatch.delenv("IAI_TOOLS_EXECD_BIN", raising=False)
    monkeypatch.setattr(tools_exec_client.shutil, "which", lambda name: None)
    monkeypatch.setattr(tools_exec_client.sys, "platform", "win32")
    monkeypatch.setattr(tools_exec_client, "_repo_root", lambda: tmp_path)

    candidate = tmp_path / "target" / "release" / "iai_tools_execd.exe"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("", encoding="utf-8")

    assert tools_exec_client._resolve_execd_bin() == str(candidate)


def test_resolve_execd_bin_no_exe_suffix_off_win32(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from iai_mcp import tools_exec_client

    monkeypatch.delenv("IAI_TOOLS_EXECD_BIN", raising=False)
    monkeypatch.setattr(tools_exec_client.shutil, "which", lambda name: None)
    monkeypatch.setattr(tools_exec_client.sys, "platform", "darwin")
    monkeypatch.setattr(tools_exec_client, "_repo_root", lambda: tmp_path)

    candidate = tmp_path / "target" / "release" / "iai_tools_execd"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("", encoding="utf-8")

    assert tools_exec_client._resolve_execd_bin() == str(candidate)
