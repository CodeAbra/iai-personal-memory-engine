"""The IPC seam's Windows-transport pieces that are testable on any platform:
the token handshake wrapper runs over plain TCP loopback, and the token/port
file helpers are pure path logic. POSIX behavior of the seam (unix sockets)
is exercised by every existing daemon/relay suite."""
from __future__ import annotations

import asyncio

import pytest


def test_token_and_port_files_scope_to_the_socket_env(monkeypatch, tmp_path):
    from iai_mcp import _ipc

    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "x.sock"))
    assert _ipc._token_file_path() == tmp_path / "x.sock.token"
    assert _ipc._port_file_path() == tmp_path / "x.sock.port"

    monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)
    assert _ipc._token_file_path() == _ipc.TOKEN_FILE
    assert _ipc._port_file_path() == _ipc.PORT_FILE


def test_generated_token_round_trips_and_is_256_bit(monkeypatch, tmp_path):
    from iai_mcp import _ipc

    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "y.sock"))
    token = _ipc._generate_token()
    assert len(token) == 64 and int(token, 16) >= 0
    assert _ipc._read_token() == token
    _ipc._remove_token_file()
    assert _ipc._read_token() is None


def test_auth_handler_gates_the_connection(monkeypatch, tmp_path):
    """The authenticated wrapper must (a) pass a correct first-line token
    through to the real handler, (b) close a wrong-token connection without
    ever invoking it. Runs over TCP loopback — platform-neutral."""
    from iai_mcp import _ipc

    async def scenario() -> tuple[str, bytes]:
        served: list[str] = []

        async def real_handler(reader, writer):
            line = await reader.readline()
            served.append(line.decode().strip())
            writer.write(b"ok\n")
            await writer.drain()
            writer.close()

        token = "a" * 64
        wrapped = _ipc._make_authenticated_handler(real_handler, token)
        server = await asyncio.start_server(wrapped, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        # Correct token: the request reaches the handler and gets a reply.
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write((token + "\n").encode())
        writer.write(b"hello\n")
        await writer.drain()
        reply = await asyncio.wait_for(reader.readline(), timeout=5)
        writer.close()

        # Wrong token: the connection dies without a byte of response and
        # the handler never runs.
        reader2, writer2 = await asyncio.open_connection("127.0.0.1", port)
        writer2.write(b"wrong-token\n")
        writer2.write(b"stealth\n")
        await writer2.drain()
        rejected = await asyncio.wait_for(reader2.read(), timeout=5)
        writer2.close()

        server.close()
        await server.wait_closed()
        assert served == ["hello"], "handler must see only authenticated traffic"
        return reply.decode().strip(), rejected

    reply, rejected = asyncio.run(scenario())
    assert reply == "ok"
    assert rejected == b"", "a wrong token must get nothing back"


def test_flock_shim_is_posix_passthrough():
    import fcntl as _real

    from iai_mcp import _flock

    assert _flock.SHARED_IS_EXCLUSIVE is False
    assert _flock.flock is _real.flock
    assert _flock.LOCK_SH == _real.LOCK_SH
    assert _flock.LOCK_EX == _real.LOCK_EX


def test_posix_ipc_address_honors_env(monkeypatch, tmp_path):
    from iai_mcp import _ipc

    if _ipc.IS_WINDOWS:
        pytest.skip("POSIX address resolution")
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "z.sock"))
    assert _ipc.ipc_address() == str(tmp_path / "z.sock")
