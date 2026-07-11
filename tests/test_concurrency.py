from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def socket_path(tmp_path, monkeypatch):
    from iai_mcp import concurrency
    sock_dir = Path(f"/tmp/iai-{os.getpid()}-{id(tmp_path)}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"
    monkeypatch.setattr(concurrency, "SOCKET_PATH", sock_path)
    try:
        yield sock_path
    finally:
        try:
            if sock_path.exists():
                sock_path.unlink()
        except OSError:
            pass
        try:
            sock_dir.rmdir()
        except OSError:
            pass


def test_socket_status_round_trip(socket_path):
    from iai_mcp.concurrency import serve_control_socket

    state = {"fsm_state": "WAKE", "daemon_started_at": "2026-04-18T00:00:00+00:00"}

    async def runner():
        shutdown = asyncio.Event()
        server_task = asyncio.create_task(
            serve_control_socket(store=None, state=state, shutdown=shutdown,
                                 socket_path=socket_path)
        )
        for _ in range(100):
            if socket_path.exists():
                break
            await asyncio.sleep(0.02)
        assert socket_path.exists(), "socket never bound"

        reader, writer = await asyncio.open_unix_connection(path=str(socket_path))
        writer.write(b'{"type":"status"}\n')
        await writer.drain()
        line = await reader.readline()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

        shutdown.set()
        await asyncio.wait_for(server_task, timeout=5)
        return json.loads(line)

    resp = asyncio.run(runner())

    assert resp["ok"] is True
    assert resp["state"] == "WAKE"
    assert isinstance(resp["uptime_sec"], (int, float))


def test_socket_injected_dispatcher(socket_path):
    from iai_mcp.concurrency import serve_control_socket

    received: list[dict] = []

    async def custom_dispatcher(req: dict) -> dict:
        received.append(req)
        return {"ok": True, "seen": req.get("type")}

    requests = [
        {"type": "pause", "seconds": 60},
        {"type": "force_rem"},
        {"type": "tail_logs", "n": 10},
    ]

    async def runner():
        shutdown = asyncio.Event()
        server_task = asyncio.create_task(
            serve_control_socket(
                store=None, state={}, shutdown=shutdown,
                dispatcher=custom_dispatcher, socket_path=socket_path,
            )
        )
        for _ in range(100):
            if socket_path.exists():
                break
            await asyncio.sleep(0.02)
        assert socket_path.exists()

        responses = []
        for req in requests:
            r, w = await asyncio.open_unix_connection(path=str(socket_path))
            w.write((json.dumps(req) + "\n").encode())
            await w.drain()
            line = await r.readline()
            responses.append(json.loads(line))
            w.close()
            try:
                await w.wait_closed()
            except Exception:
                pass

        shutdown.set()
        await asyncio.wait_for(server_task, timeout=5)
        return responses

    responses = asyncio.run(runner())

    assert received == requests, f"dispatcher saw {received!r}"
    for resp, req in zip(responses, requests):
        assert resp == {"ok": True, "seen": req["type"]}


def test_stale_socket_cleanup(socket_path):
    from iai_mcp.concurrency import serve_control_socket

    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.write_text("stale")
    assert socket_path.exists()

    async def runner():
        shutdown = asyncio.Event()
        server_task = asyncio.create_task(
            serve_control_socket(store=None, state={}, shutdown=shutdown,
                                 socket_path=socket_path)
        )
        for _ in range(100):
            if socket_path.exists() and socket_path.stat().st_size == 0:
                break
            await asyncio.sleep(0.02)
        r, w = await asyncio.open_unix_connection(path=str(socket_path))
        w.write(b'{"type":"status"}\n')
        await w.drain()
        line = await r.readline()
        w.close()
        try:
            await w.wait_closed()
        except Exception:
            pass
        shutdown.set()
        await asyncio.wait_for(server_task, timeout=5)
        return json.loads(line)

    resp = asyncio.run(runner())

    assert resp.get("ok") is True


def test_socket_permissions_user_only(socket_path):
    from iai_mcp.concurrency import serve_control_socket

    async def runner():
        shutdown = asyncio.Event()
        server_task = asyncio.create_task(
            serve_control_socket(store=None, state={}, shutdown=shutdown,
                                 socket_path=socket_path)
        )
        for _ in range(100):
            if socket_path.exists():
                break
            await asyncio.sleep(0.02)
        sock_mode = socket_path.stat().st_mode & 0o777
        shutdown.set()
        await asyncio.wait_for(server_task, timeout=5)
        return sock_mode

    sock_mode = asyncio.run(runner())
    assert sock_mode == 0o600, f"socket mode is {oct(sock_mode)}, expected 0o600"
