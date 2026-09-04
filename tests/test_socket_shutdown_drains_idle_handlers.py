from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import time
from pathlib import Path

import pytest

from iai_mcp.socket_server import SocketServer

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="binds a unix socket",
)


def test_shutdown_drains_idle_handler_within_timeout() -> None:
    asyncio.run(_run())


async def _run() -> None:
    # macOS caps AF_UNIX paths at ~104 bytes; a short /tmp dir avoids it.
    sock_dir = Path(f"/tmp/iai-shdrain-{os.getpid()}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"

    srv = SocketServer(None, idle_secs=99999)
    server_task = asyncio.create_task(srv.serve(socket_path=sock_path))

    client: socket.socket | None = None
    try:
        for _ in range(250):
            if sock_path.exists():
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("socket never bound")

        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(sock_path))
        # Deliberately send nothing: the handler parks on reader.readline().

        for _ in range(200):
            if srv.active_connections > 0:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("handler task never started")

        srv.shutdown_event.set()
        server_task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(server_task, return_exceptions=True), 10,
            )
        except asyncio.TimeoutError:
            pytest.fail(
                "serve() did not return within 10s while an idle client "
                "was attached",
            )
    finally:
        # Close only after the assertion: closing early sends EOF and
        # makes the handler exit on its own, hiding the bug.
        if client is not None:
            client.close()
        try:
            if sock_path.exists():
                sock_path.unlink()
        except OSError:
            pass
        try:
            sock_dir.rmdir()
        except OSError:
            pass


def test_shutdown_waits_for_in_flight_dispatch_then_drains_idle_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(_run_busy_and_idle(monkeypatch))


async def _run_busy_and_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    import iai_mcp.core as core_mod

    def _slow_dispatch(store: object, method: str, params: dict) -> dict:
        # Plain blocking sleep: this runs in the asyncio.to_thread worker
        # thread, not on the event loop.
        time.sleep(1.0)
        return {"ok": True}

    monkeypatch.setattr(core_mod, "dispatch", _slow_dispatch)

    sock_dir = Path(f"/tmp/iai-shdrain-busy-{os.getpid()}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"

    srv = SocketServer(None, idle_secs=99999)
    server_task = asyncio.create_task(srv.serve(socket_path=sock_path))

    busy_client: socket.socket | None = None
    idle_client: socket.socket | None = None
    try:
        for _ in range(250):
            if sock_path.exists():
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("socket never bound")

        loop = asyncio.get_running_loop()

        busy_client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        busy_client.connect(str(sock_path))
        req = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "whatever", "params": {},
        }) + "\n"
        busy_client.sendall(req.encode("utf-8"))

        idle_client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        idle_client.connect(str(sock_path))
        # Deliberately send nothing: this handler parks on reader.readline().

        for _ in range(200):
            if srv._busy_handlers:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("busy handler never entered its dispatch window")

        for _ in range(200):
            if srv.active_connections >= 2:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("idle handler task never started")

        start = time.monotonic()
        srv.shutdown_event.set()
        server_task.cancel()

        # The idle sibling must be cancelled near-instantly, independent of
        # how long the busy dispatch takes to finish.
        idle_client.setblocking(False)
        idle_data = await asyncio.wait_for(loop.sock_recv(idle_client, 4096), 1.0)
        idle_elapsed = time.monotonic() - start
        assert idle_data == b"", "idle handler must be cancelled, not answer"
        assert idle_elapsed < 0.9, (
            f"idle sibling took {idle_elapsed:.2f}s to be cancelled -- "
            "expected near-instant cancellation, independent of the busy "
            "dispatch"
        )

        try:
            await asyncio.wait_for(
                asyncio.gather(server_task, return_exceptions=True), 10,
            )
        except asyncio.TimeoutError:
            pytest.fail(
                "serve() did not return within 10s while a busy dispatch "
                "was in flight",
            )
        elapsed = time.monotonic() - start

        assert elapsed >= 0.9, (
            f"shutdown returned in {elapsed:.2f}s -- the busy dispatch was "
            "cancelled instead of allowed to finish naturally"
        )
        assert elapsed < 3.5, (
            f"shutdown took {elapsed:.2f}s -- looks like it waited for the "
            "full busy-drain timeout instead of the dispatch finishing"
        )

        busy_client.setblocking(True)
        busy_client.settimeout(2.0)
        raw = busy_client.recv(4096)
        assert raw, "busy handler must write its response before serve() returns"
        resp = json.loads(raw.decode("utf-8").strip())
        assert resp["result"] == {"ok": True}
    finally:
        if busy_client is not None:
            busy_client.close()
        if idle_client is not None:
            idle_client.close()
        try:
            if sock_path.exists():
                sock_path.unlink()
        except OSError:
            pass
        try:
            sock_dir.rmdir()
        except OSError:
            pass


def test_shutdown_reaps_busy_handler_promptly_once_it_goes_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(_run_busy_then_finishes_during_shutdown(monkeypatch))


async def _run_busy_then_finishes_during_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iai_mcp.core as core_mod

    _fast_dispatch_secs = 0.3

    def _fast_dispatch(store: object, method: str, params: dict) -> dict:
        time.sleep(_fast_dispatch_secs)
        return {"ok": True}

    monkeypatch.setattr(core_mod, "dispatch", _fast_dispatch)

    sock_dir = Path(f"/tmp/iai-shdrain-reap-{os.getpid()}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"

    srv = SocketServer(None, idle_secs=99999)
    server_task = asyncio.create_task(srv.serve(socket_path=sock_path))

    busy_client: socket.socket | None = None
    try:
        for _ in range(250):
            if sock_path.exists():
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("socket never bound")

        busy_client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        busy_client.connect(str(sock_path))
        req = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "whatever", "params": {},
        }) + "\n"
        busy_client.sendall(req.encode("utf-8"))

        for _ in range(200):
            if srv._busy_handlers:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("busy handler never entered its dispatch window")

        # busy_client stays connected (no EOF) after its response, so the
        # handler task only exits via the shutdown-time break in handle()'s
        # read loop. Without that break the task loops back to readline(),
        # never becomes .done(), and _drain_handler_tasks()'s busy-wait
        # burns the full _DRAIN_BUSY_TIMEOUT_SEC (5.0s) before force-
        # cancelling it -- this assertion (< 2.0s) fails in that case.
        start = time.monotonic()
        srv.shutdown_event.set()
        server_task.cancel()

        try:
            await asyncio.wait_for(
                asyncio.gather(server_task, return_exceptions=True), 10,
            )
        except asyncio.TimeoutError:
            pytest.fail(
                "serve() did not return within 10s after the busy dispatch "
                "finished mid-shutdown",
            )
        elapsed = time.monotonic() - start

        assert elapsed < 2.0, (
            f"shutdown took {elapsed:.2f}s -- expected the handler to exit "
            "promptly once its dispatch finished, well under "
            "_DRAIN_BUSY_TIMEOUT_SEC=5.0s, not burn the full busy-drain "
            "timeout"
        )
        # start is captured after the poll that confirms the handler is
        # already busy, so some of the dispatch's 0.3s may have already
        # elapsed -- assert against a fraction, not the full duration, to
        # avoid a timing-jitter false negative while still distinguishing
        # "finished naturally" from "cancelled outright" (near-0s).
        assert elapsed >= _fast_dispatch_secs * 0.5, (
            f"shutdown returned in {elapsed:.2f}s, too fast for the "
            f"{_fast_dispatch_secs}s dispatch to have finished naturally -- "
            "the busy handler was likely cancelled outright instead"
        )
        assert srv.active_connections == 0
        assert not srv._handler_tasks

        busy_client.setblocking(True)
        busy_client.settimeout(2.0)
        raw = busy_client.recv(4096)
        assert raw, "busy handler must write its response before serve() returns"
        resp = json.loads(raw.decode("utf-8").strip())
        assert resp["result"] == {"ok": True}
    finally:
        if busy_client is not None:
            busy_client.close()
        try:
            if sock_path.exists():
                sock_path.unlink()
        except OSError:
            pass
        try:
            sock_dir.rmdir()
        except OSError:
            pass


def test_shutdown_survives_second_cancellation_during_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(_run_double_cancel(monkeypatch))


async def _run_double_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    import iai_mcp.core as core_mod

    def _slow_dispatch(store: object, method: str, params: dict) -> dict:
        time.sleep(0.5)
        return {"ok": True}

    monkeypatch.setattr(core_mod, "dispatch", _slow_dispatch)

    sock_dir = Path(f"/tmp/iai-shdrain-dblcancel-{os.getpid()}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"

    srv = SocketServer(None, idle_secs=99999)
    server_task = asyncio.create_task(srv.serve(socket_path=sock_path))

    busy_client: socket.socket | None = None
    try:
        for _ in range(250):
            if sock_path.exists():
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("socket never bound")

        busy_client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        busy_client.connect(str(sock_path))
        req = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "whatever", "params": {},
        }) + "\n"
        busy_client.sendall(req.encode("utf-8"))

        for _ in range(200):
            if srv._busy_handlers:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("busy handler never entered its dispatch window")

        srv.shutdown_event.set()
        server_task.cancel()
        # Fire a second cancellation while the drain (bounded wait for the
        # busy dispatch) is still in progress -- must not truncate it.
        await asyncio.sleep(0.1)
        server_task.cancel()

        try:
            await asyncio.wait_for(
                asyncio.gather(server_task, return_exceptions=True), 10,
            )
        except asyncio.TimeoutError:
            pytest.fail(
                "serve() did not finish within 10s after a second "
                "cancellation delivered mid-drain",
            )

        assert server_task.done()
        # The discriminating check: without shielding, the second
        # cancellation aborts the drain wait early and serve() returns
        # while the busy handler is still mid-dispatch.
        assert srv.active_connections == 0, (
            "busy handler must finish before serve() returns even when a "
            "second cancellation lands mid-drain"
        )
        assert not srv._handler_tasks
        assert not sock_path.exists(), (
            "socket path must be cleaned up even after a second "
            "cancellation lands mid-teardown"
        )
    finally:
        if busy_client is not None:
            busy_client.close()
        try:
            if sock_path.exists():
                sock_path.unlink()
        except OSError:
            pass
        try:
            sock_dir.rmdir()
        except OSError:
            pass
