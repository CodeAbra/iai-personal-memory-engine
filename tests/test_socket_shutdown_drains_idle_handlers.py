"""Cancelling serve() must not hang on connections parked in readline().

Regression test for the shutdown hang: with an idle client attached, the
daemon took the full systemd stop timeout and died by SIGKILL on every stop.

The mechanism is a CPython behaviour change. ``Server.wait_closed()`` returns
only once every active connection has been dropped (3.12.1+; before that it
returned immediately even with connections open). A handler parked on
``reader.readline()`` for a client that sends nothing never drops on its own,
so the teardown blocked there forever.

Cancellation is the part that makes this subtle, and the reason this test
cancels rather than setting ``shutdown_event``. ``daemon.main()`` sets the
event and then immediately cancels the serve task, so the cancellation lands
while ``serve`` is still parked on ``shutdown_event.wait()``. Any cleanup
written in the body of an ``async with server:`` is skipped in that case, and
``__aexit__`` runs ``close()`` + ``wait_closed()`` with the handlers still
live. A fix that drains handlers in the body passes a test that only sets the
event, and still hangs in production.
"""

from __future__ import annotations

import asyncio
import socket
import sys
import tempfile
from pathlib import Path

import pytest

if sys.platform == "win32":  # pragma: no cover -- POSIX socket path only
    pytest.skip("unix socket path", allow_module_level=True)


def test_cancelled_serve_drains_idle_connection() -> None:
    from iai_mcp.socket_server import SocketServer

    async def _drive() -> None:
        with tempfile.TemporaryDirectory() as td:
            sock_path = Path(td) / "d.sock"
            srv = SocketServer(store=None, idle_secs=99999, state={"fsm_state": "WAKE"})
            server_task = asyncio.create_task(srv.serve(socket_path=sock_path))

            for _ in range(500):
                if sock_path.exists():
                    break
                await asyncio.sleep(0.01)
            assert sock_path.exists(), "socket never bound"

            # Connect and send nothing: this is what parks handle() forever.
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(str(sock_path))
            try:
                for _ in range(500):
                    if srv.active_connections > 0:
                        break
                    await asyncio.sleep(0.01)
                assert srv.active_connections > 0, "handler never started"

                # Exactly what daemon.main() does at teardown.
                srv.shutdown_event.set()
                server_task.cancel()

                # Ten seconds is far below the 60s stop timeout that exposed
                # this, and far above the ~0s a correct teardown needs.
                try:
                    await asyncio.wait_for(
                        asyncio.gather(server_task, return_exceptions=True),
                        timeout=10,
                    )
                except asyncio.TimeoutError:  # pragma: no cover -- the bug
                    pytest.fail(
                        "serve() did not finish within 10s of cancellation: the "
                        "idle connection's handler was never drained, so "
                        "wait_closed() blocked"
                    )
            finally:
                client.close()

    asyncio.run(_drive())
