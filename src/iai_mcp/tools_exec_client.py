"""Thin socket client for the `iai_tools_execd` executor daemon.

Both `tools_cli.py` (the `iai-tools` console script) and `cli/_tools.py`
(the `iai-mcp tools` operator CLI) import this module for their
`run`/`fetch`/`batch-run` subcommands. Mirrors `_ipc.py`'s connect/retry
shape by STRUCTURE only -- must never import iai_mcp.hippo, iai_mcp.store,
iai_mcp.lilli, iai_mcp.capture, or iai_mcp_native. Stdlib only.
"""

from __future__ import annotations

import itertools
import json
import os
import platform
import secrets
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_ID_COUNTER = itertools.count(1)
_CONNECT_TIMEOUT_SECS = 5.0
_CALL_TIMEOUT_SECS = 300.0

IS_WINDOWS: bool = platform.system() == "Windows"


class ToolExecError(RuntimeError):
    """Raised when the daemon returns a JSON-RPC error response."""


def _tools_root() -> Path:
    env = os.environ.get("IAI_TOOLS_ROOT")
    if env:
        return Path(env)
    return Path.home() / ".iai-tools"


def _socket_path() -> Path:
    env = os.environ.get("IAI_TOOLS_SOCKET_PATH")
    if env:
        return Path(env)
    return _tools_root() / ".execd.sock"


def _port_file_path() -> Path:
    """`<socket-path>.port` -- the file the Rust daemon writes on Windows,
    strictly after the token file, so its presence is a valid readiness edge."""
    return Path(f"{_socket_path()}.port")


def _token_file_path() -> Path:
    """`<socket-path>.token` -- the file the Rust daemon writes on Windows,
    strictly before the port file."""
    return Path(f"{_socket_path()}.token")


def _read_port() -> int | None:
    try:
        return int(_port_file_path().read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError, OSError):
        return None


def _read_token() -> str | None:
    try:
        return _token_file_path().read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_execd_bin() -> str:
    """Resolves the `iai_tools_execd` binary path.

    `IAI_TOOLS_EXECD_BIN` is the explicit override tests always set (pytest
    has no `CARGO_BIN_EXE_*`). PATH lookup and the cargo target-dir guess
    below are a convenience for interactive/dev use, never load-bearing.
    """
    env = os.environ.get("IAI_TOOLS_EXECD_BIN")
    if env:
        return env
    on_path = shutil.which("iai_tools_execd")
    if on_path:
        return on_path
    bin_name = "iai_tools_execd.exe" if sys.platform == "win32" else "iai_tools_execd"
    for profile in ("release", "debug"):
        candidate = _repo_root() / "target" / profile / bin_name
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "iai_tools_execd binary not found; set IAI_TOOLS_EXECD_BIN or build it via "
        "'cargo build -p iai_mcp_tools_core --bins'"
    )


def _execd_log_path() -> Path:
    """Where a spawned daemon's stderr is captured."""
    return _tools_root() / "execd-stderr.log"


def _clear_stale_transport_files() -> None:
    """Removes a stale `<socket-path>.port`/`.token` pair left by a crashed
    prior daemon, before spawning a fresh one -- mirrors the Unix arm
    unlinking a stale socket file, and restores the port file appearing as
    a valid fresh-daemon readiness edge for the connect retry loop."""
    for path in (_port_file_path(), _token_file_path()):
        try:
            path.unlink()
        except (FileNotFoundError, OSError):
            pass


def _spawn_daemon() -> None:
    bin_path = _resolve_execd_bin()
    _tools_root().mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if IS_WINDOWS:
        _clear_stale_transport_files()
        env["IAI_TOOLS_TOKEN"] = secrets.token_hex(32)
    log_path = _execd_log_path()
    # O_CREAT's mode applies at creation time, closing the window a
    # create-then-chmod sequence would leave the file briefly world-readable
    # under. The chmod below additionally self-heals a log file left behind
    # by a build predating this restriction.
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.chmod(log_path, 0o600)
    except OSError:
        pass
    with os.fdopen(fd, "ab") as log_file:
        subprocess.Popen(
            [bin_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=log_file,
            env=env,
            start_new_session=True,
        )


_RETRY_POLL_SECS = 0.02
_RETRY_ATTEMPT_CAP_SECS = 0.5


def _send_token_or_raise(sock: socket.socket) -> None:
    """Sends the on-disk auth token as the mandatory first line of a Windows
    connection. Never a bare-TCP fallback: if the token cannot be read, the
    socket is closed and the failure raises without sending a single byte."""
    token = _read_token()
    if token is None:
        sock.close()
        raise FileNotFoundError(
            f"Executor daemon auth token not found: {_token_file_path()} missing."
        )
    sock.sendall((token + "\n").encode("utf-8"))


def _tcp_connect_only(port: int, timeout: float | None) -> socket.socket | None:
    """Opens a TCP-loopback connection with no token handshake. Returns
    `None` on a retryable connect failure so callers can distinguish that
    from a token failure, which must never be treated as retryable."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if timeout is not None:
        sock.settimeout(timeout)
    try:
        sock.connect(("127.0.0.1", port))
    except OSError:
        sock.close()
        return None
    sock.settimeout(None)
    return sock


def _connect_windows(timeout: float) -> socket.socket:
    port = _read_port()
    if port is not None:
        sock = _tcp_connect_only(port, timeout=None)
        if sock is not None:
            _send_token_or_raise(sock)
            return sock

    _spawn_daemon()

    # The port file is the readiness edge: the daemon writes the (fully
    # ACL-tightened) token file strictly before the port file, so a port
    # file that exists implies the token file is already complete.
    deadline = time.monotonic() + timeout
    last_err: OSError | None = None
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        port = _read_port()
        if port is None:
            last_err = FileNotFoundError(f"{_port_file_path()} not found")
            time.sleep(_RETRY_POLL_SECS)
            continue
        sock = _tcp_connect_only(port, timeout=min(_RETRY_ATTEMPT_CAP_SECS, max(remaining, 0.0)))
        if sock is None:
            last_err = ConnectionRefusedError(f"127.0.0.1:{port} refused connection")
            time.sleep(_RETRY_POLL_SECS)
            continue
        _send_token_or_raise(sock)
        return sock
    raise last_err or ConnectionRefusedError("executor daemon at 127.0.0.1 never became reachable")


def _connect(timeout: float = _CONNECT_TIMEOUT_SECS) -> socket.socket:
    if IS_WINDOWS:
        return _connect_windows(timeout)

    path = _socket_path()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(str(path))
        return sock
    except (ConnectionRefusedError, FileNotFoundError):
        sock.close()

    _spawn_daemon()

    # connect() is the only real accept-readiness test -- path.exists() is
    # true for both a bind-before-listen gap and a stale unlinked socket.
    deadline = time.monotonic() + timeout
    last_err: OSError | None = None
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(min(_RETRY_ATTEMPT_CAP_SECS, max(remaining, 0.0)))
        try:
            sock.connect(str(path))
            sock.settimeout(None)
            return sock
        except OSError as e:
            last_err = e
            sock.close()
            time.sleep(_RETRY_POLL_SECS)
    raise last_err or ConnectionRefusedError(f"daemon at {path} never became reachable")


def _call(method: str, params: dict[str, Any]) -> Any:
    sock = _connect()
    try:
        sock.settimeout(_CALL_TIMEOUT_SECS)
        request_id = next(_ID_COUNTER)
        request = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        sock.sendall((json.dumps(request) + "\n").encode("utf-8"))

        buf = b""
        while not buf.endswith(b"\n"):
            chunk = sock.recv(65536)
            if not chunk:
                break
            buf += chunk
        response = json.loads(buf.decode("utf-8"))
    finally:
        sock.close()

    if "error" in response:
        err = response["error"]
        raise ToolExecError(f"{err.get('code')}: {err.get('message')}")
    return response["result"]


def run(
    command: str,
    args: list[str] | None = None,
    *,
    timeout: float | None = None,
    background: bool = False,
) -> dict[str, Any]:
    """Runs `command` via the daemon; returns {run_id, state, bounded_tail}."""
    params: dict[str, Any] = {
        "command": command,
        "args": list(args or []),
        "background": background,
    }
    if timeout is not None:
        params["timeout"] = timeout
    return _call("tool_run", params)


def fetch(run_id: str, *, bound: int | None = None) -> dict[str, Any]:
    """Re-serves a prior run's next bounded output window without re-running it.

    `bound`, when given, overrides the daemon's default byte cap per stream
    for this fetch only; the persisted offset still advances by exactly what
    was served, so a later fetch continues from there.
    """
    params: dict[str, Any] = {"run_id": run_id}
    if bound is not None:
        params["bound"] = bound
    return _call("result_fetch", params)


def batch_run(
    commands: list[dict[str, Any]],
    *,
    concurrency: int = 1,
    timeout_ms: int | None = None,
) -> list[dict[str, Any]]:
    """Runs a list of `{"command": ..., "args": [...]}` dicts via the daemon.

    Returns a list aligned 1:1 with `commands` in input order.
    """
    params: dict[str, Any] = {"commands": commands, "concurrency": concurrency}
    if timeout_ms is not None:
        params["timeout_ms"] = timeout_ms
    return _call("tool_batch_run", params)
