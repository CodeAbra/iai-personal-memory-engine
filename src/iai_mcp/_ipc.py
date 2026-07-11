"""
Platform-agnostic IPC transport layer.

POSIX:   Unix-domain socket  →  ~/.iai-mcp/.daemon.sock
         Access control is provided by the socket file's filesystem permissions.

Windows: TCP loopback         →  127.0.0.1:<ephemeral port>
         Port is persisted in ~/.iai-mcp/.daemon.port.
         Because loopback TCP is reachable by any local process, an
         auth-token handshake is layered on top: the daemon generates a
         32-byte random hex token on start, writes it to
         ~/.iai-mcp/.daemon.token (ACL-restricted to the current user via
         icacls), and requires every client to send that token as the
         first line of each connection.  Connections that send the wrong
         token are closed immediately without processing any requests.
"""
from __future__ import annotations

import asyncio
import inspect
import os
import platform
import secrets
import socket
import subprocess
from pathlib import Path
from typing import Any

IS_WINDOWS: bool = platform.system() == "Windows"

_BASE_DIR: Path = Path.home() / ".iai-mcp"
SOCKET_PATH: Path = _BASE_DIR / ".daemon.sock"  # POSIX only — kept for compatibility
PORT_FILE: Path = _BASE_DIR / ".daemon.port"     # Windows only
TOKEN_FILE: Path = _BASE_DIR / ".daemon.token"   # Windows only — auth secret

_TOKEN_BYTES = 32  # 256-bit random token → 64 hex chars on the wire


# ---------------------------------------------------------------------------
# Port file helpers (Windows only)
# ---------------------------------------------------------------------------

def _port_file_path() -> Path:
    """Resolve the Windows port-file location at call time.

    Mirrors the POSIX ``IAI_DAEMON_SOCKET_PATH`` override (see ``ipc_address``)
    so a daemon bound to a non-default endpoint — a custom ``IAI_MCP_STORE``,
    or an isolated test harness — persists its port *alongside* that socket
    path (``<socket-path>.port``) instead of always clobbering the shared
    ``~/.iai-mcp/.daemon.port``. Without this, every Windows daemon (and every
    test) raced for one global port file. Resolved dynamically, not as a module
    constant, because tests set the env var after import.
    """
    env = os.environ.get("IAI_DAEMON_SOCKET_PATH")
    if env:
        return Path(f"{env}.port")
    return PORT_FILE


def _read_port() -> int | None:
    try:
        return int(_port_file_path().read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError, OSError):
        return None


def _write_port(port: int) -> None:
    path = _port_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(port), encoding="utf-8")


def _remove_port_file() -> None:
    try:
        _port_file_path().unlink()
    except (FileNotFoundError, OSError):
        pass


# ---------------------------------------------------------------------------
# Token file helpers (Windows only)
# ---------------------------------------------------------------------------

def _restrict_token_file(path: Path) -> None:
    """Restrict token file to current user only via icacls (Windows equivalent of chmod 0o600)."""
    username = os.environ.get("USERNAME", "")
    if username:
        subprocess.run(
            ["icacls", str(path), "/inheritance:d", "/grant:r", f"{username}:F"],
            check=False,
            capture_output=True,
        )


def _token_file_path() -> Path:
    """Resolve the Windows auth-token file at call time, mirroring
    ``_port_file_path`` so the token is per-endpoint (an isolated test harness
    or a custom ``IAI_MCP_STORE``) rather than a single shared
    ``~/.iai-mcp/.daemon.token`` that every daemon and test would clobber."""
    env = os.environ.get("IAI_DAEMON_SOCKET_PATH")
    if env:
        return Path(f"{env}.token")
    return TOKEN_FILE


def _generate_token() -> str:
    """Generate a fresh 32-byte random token and persist it to the token file."""
    token = secrets.token_hex(_TOKEN_BYTES)
    path = _token_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token, encoding="utf-8")
    _restrict_token_file(path)
    return token


def _read_token() -> str | None:
    try:
        return _token_file_path().read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return None


def _remove_token_file() -> None:
    try:
        _token_file_path().unlink()
    except (FileNotFoundError, OSError):
        pass


# ---------------------------------------------------------------------------
# Auth-wrapping helpers (Windows only)
# ---------------------------------------------------------------------------

def _make_authenticated_handler(handler: Any, token: str) -> Any:
    """
    Wrap *handler* so that the first line received on each connection must be
    the auth token.  If it matches, the connection proceeds normally.
    If it doesn't, the connection is closed immediately.
    """
    async def _auth_handler(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=5.0)
        except (asyncio.TimeoutError, OSError):
            writer.close()
            return
        received = line.decode("utf-8", errors="replace").strip()
        if not secrets.compare_digest(received, token):
            writer.close()
            return
        await handler(reader, writer)

    return _auth_handler


async def _send_token_async(writer: asyncio.StreamWriter) -> None:
    """Send the auth token as the first line on a Windows client connection."""
    token = _read_token()
    if token is None:
        raise FileNotFoundError(
            f"Daemon auth token not found: {_token_file_path()} missing."
        )
    writer.write((token + "\n").encode("utf-8"))
    await writer.drain()


def _send_token_sync(sock: socket.socket) -> None:
    """Send the auth token as the first line on a synchronous Windows client socket."""
    token = _read_token()
    if token is None:
        raise FileNotFoundError(
            f"Daemon auth token not found: {_token_file_path()} missing."
        )
    sock.sendall((token + "\n").encode("utf-8"))


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def ipc_address() -> str | tuple[str, int]:
    """
    Return the current IPC endpoint.
    POSIX: Unix socket path string.
    Windows: ("127.0.0.1", port) tuple.
    """
    if not IS_WINDOWS:
        env = os.environ.get("IAI_DAEMON_SOCKET_PATH")
        return env if env else str(SOCKET_PATH)
    port = _read_port()
    if port is None:
        raise FileNotFoundError(
            "Daemon not running: ~/.iai-mcp/.daemon.port not found."
        )
    return ("127.0.0.1", port)


async def open_ipc_connection(
    addr: str | tuple[str, int] | None = None,
    *,
    timeout: float | None = None,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """
    Open a client connection to the daemon.

    On POSIX wraps asyncio.open_unix_connection; on Windows wraps
    asyncio.open_connection over TCP loopback and performs the auth-token
    handshake before returning.

    The *addr* parameter is ignored on Windows (always uses port file).
    """
    coro: Any
    if IS_WINDOWS:
        port = _read_port()
        if port is None:
            raise FileNotFoundError(
                f"Daemon not running: {_port_file_path()} not found."
            )
        coro = asyncio.open_connection("127.0.0.1", port)
    else:
        if addr is None:
            env = os.environ.get("IAI_DAEMON_SOCKET_PATH")
            addr = env if env else str(SOCKET_PATH)
        coro = asyncio.open_unix_connection(str(addr))

    if timeout is not None:
        reader, writer = await asyncio.wait_for(coro, timeout=timeout)
    else:
        reader, writer = await coro

    if IS_WINDOWS:
        await _send_token_async(writer)

    return reader, writer


async def start_ipc_server(
    handler: Any,
    addr: str | Path | None = None,
) -> tuple[asyncio.AbstractServer, str | tuple[str, int], bool]:
    """
    Start the daemon server.

    Returns ``(server, actual_addr, needs_manual_cleanup)`` where:
    - *actual_addr* is the socket path (POSIX) or ("127.0.0.1", port) (Windows).
    - *needs_manual_cleanup* is True if the caller must call ``shutdown_ipc``
      in its finally block (i.e. asyncio will NOT clean up automatically).

    On Windows a fresh auth token is generated and written to TOKEN_FILE, and
    the port is written to PORT_FILE immediately after bind.
    """
    if IS_WINDOWS:
        token = _generate_token()
        authenticated_handler = _make_authenticated_handler(handler, token)
        server = await asyncio.start_server(authenticated_handler, "127.0.0.1", 0)
        port: int = server.sockets[0].getsockname()[1]
        _write_port(port)
        return server, ("127.0.0.1", port), True

    # POSIX: try to use asyncio's built-in cleanup_socket (Python 3.12+)
    if addr is None:
        env = os.environ.get("IAI_DAEMON_SOCKET_PATH")
        path_str = env if env else str(SOCKET_PATH)
    else:
        path_str = str(addr)

    sig = inspect.signature(asyncio.start_unix_server)
    supports_cleanup = "cleanup_socket" in sig.parameters
    kwargs: dict[str, Any] = {"cleanup_socket": True} if supports_cleanup else {}

    server = await asyncio.start_unix_server(handler, path=path_str, **kwargs)
    return server, path_str, not supports_cleanup


def cleanup_ipc_address(addr: str | Path | None = None) -> None:
    """
    Remove a stale socket file before binding (POSIX only). No-op on Windows.
    """
    if IS_WINDOWS:
        return
    if addr is None:
        env = os.environ.get("IAI_DAEMON_SOCKET_PATH")
        path = Path(env) if env else SOCKET_PATH
    else:
        path = Path(addr)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        try:
            path.unlink()
        except OSError:
            pass


def shutdown_ipc(addr: str | tuple[str, int] | None = None) -> None:
    """
    Clean up after daemon shutdown.
    POSIX: unlink the socket file (idempotent).
    Windows: remove the port file and the token file.
    """
    if IS_WINDOWS:
        _remove_port_file()
        _remove_token_file()
        return
    if addr is None or isinstance(addr, tuple):
        env = os.environ.get("IAI_DAEMON_SOCKET_PATH")
        path = Path(env) if env else SOCKET_PATH
    else:
        path = Path(addr)
    try:
        path.unlink()
    except (FileNotFoundError, OSError):
        pass


def make_sync_ipc_socket() -> tuple[socket.socket, str | tuple[str, int]]:
    """
    Create a synchronous (blocking) client socket and the address to connect to.

    Returns ``(sock, addr)`` where *addr* is a string path (POSIX) or
    ``("127.0.0.1", port)`` tuple (Windows).  Caller is responsible for
    ``settimeout``, ``connect``, and ``close``.

    On Windows the caller must also call ``send_sync_auth_token(sock)`` after
    ``connect()`` and before sending any application messages.
    """
    if IS_WINDOWS:
        port = _read_port()
        if port is None:
            raise FileNotFoundError(
                "Daemon not running: ~/.iai-mcp/.daemon.port not found."
            )
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        return s, ("127.0.0.1", port)

    env = os.environ.get("IAI_DAEMON_SOCKET_PATH")
    path = env if env else str(SOCKET_PATH)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    return s, path


def send_sync_auth_token(sock: socket.socket) -> None:
    """
    Send the Windows auth token on a synchronous socket immediately after connect().
    No-op on POSIX.
    """
    if IS_WINDOWS:
        _send_token_sync(sock)
