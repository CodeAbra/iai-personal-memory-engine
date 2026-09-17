"""Regression witnesses for the three Windows daemon-boot defects fixed in
fix/windows-daemon-boot. All platform-neutral: the Windows conditions are
simulated (SIGHUP deleted, start_unix_server deleted, IS_WINDOWS forced), so
the suite exercises them on POSIX CI too.

The original failure chain on a real Windows 11 host, in order:

1. ``_install_boot_signal_trace`` named ``signal.SIGHUP`` inside a tuple --
   built before the loop body, so the per-signal AttributeError guard never
   ran and the daemon died at boot, before anything else.
2. ``SocketServer.serve`` probed ``asyncio.start_unix_server`` above its
   ``IS_WINDOWS`` branch. asyncio exposes that symbol only where AF_UNIX
   exists, so the probe raised before the loopback bind. serve() runs as a
   fire-and-forget task, so the failure was SILENT: the daemon booted,
   warmed, and ticked its FSM while serving nothing; the port file was never
   written, every client reported "daemon not running", and the watchdog
   killed the process as wedged when the cold-start grace expired -- on
   every boot.
3. The Task Scheduler XML had no <UserId> in its LogonTrigger or Principal,
   which Windows reads as "at logon of ANY user" -- a machine-wide
   registration needing elevation, so ``schtasks /Create`` failed with
   "Access is denied" from the unelevated shell the installer runs in.
"""

from __future__ import annotations

import asyncio
import signal


def test_boot_signal_trace_survives_missing_sighup(monkeypatch, tmp_path):
    """Windows has no signal.SIGHUP: installing the boot trace must skip it
    rather than raise while building the signal tuple."""
    from iai_mcp.daemon import _install_boot_signal_trace

    monkeypatch.setattr(
        "iai_mcp.daemon._watchdog._watchdog_log_path",
        lambda: tmp_path / "watchdog.log",
    )
    monkeypatch.delattr("signal.SIGHUP", raising=False)

    registered: list[int] = []
    monkeypatch.setattr(
        "signal.signal", lambda sig, handler: registered.append(sig)
    )

    _install_boot_signal_trace()  # must not raise

    assert set(registered) == {signal.SIGTERM, signal.SIGINT}


def test_serve_windows_path_never_touches_start_unix_server(monkeypatch):
    """The POSIX capability probe must sit below the Windows return: on a
    platform without asyncio.start_unix_server the loopback path has to bind
    without evaluating the symbol at all."""
    from iai_mcp.socket_server import SocketServer

    monkeypatch.delattr("asyncio.start_unix_server", raising=False)
    monkeypatch.setattr("iai_mcp._ipc.IS_WINDOWS", True)

    served: list[bool] = []

    class _FakeServer:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def close(self):
            pass

        async def wait_closed(self):
            pass

    async def fake_start_ipc_server(handler, addr=None, *, limit=None):
        served.append(True)
        return _FakeServer(), ("127.0.0.1", 12345), True

    monkeypatch.setattr("iai_mcp._ipc.start_ipc_server", fake_start_ipc_server)
    monkeypatch.setattr("iai_mcp._ipc.shutdown_ipc", lambda addr=None: None)

    server = SocketServer.__new__(SocketServer)
    server.shutdown_event = asyncio.Event()
    server.handle = None
    server.shutdown_event.set()  # serve() should bind, observe shutdown, exit

    asyncio.run(server.serve())  # AttributeError here is the regression

    assert served == [True], "the Windows branch must reach the loopback bind"


def test_windows_task_xml_pins_trigger_and_principal_to_user(monkeypatch):
    """Without an explicit <UserId>, Task Scheduler treats the logon trigger
    as any-user and demands elevation; the rendered XML must pin both the
    trigger and the principal to the installing account."""
    import defusedxml.ElementTree as ET

    from iai_mcp.cli import _daemon

    monkeypatch.setattr(_daemon, "_windows_task_user_id", lambda: "DOM\\alice")
    rendered = _daemon._render_windows_task_xml()
    assert "{USER_ID}" not in rendered

    root = ET.fromstring(rendered)
    ns = "{http://schemas.microsoft.com/windows/2004/02/mit/task}"
    trigger_user = root.find(f"{ns}Triggers/{ns}LogonTrigger/{ns}UserId")
    principal_user = root.find(f"{ns}Principals/{ns}Principal/{ns}UserId")
    assert trigger_user is not None and trigger_user.text == "DOM\\alice"
    assert principal_user is not None and principal_user.text == "DOM\\alice"


def test_windows_task_user_id_resolves_from_environment(monkeypatch):
    from iai_mcp.cli import _daemon

    monkeypatch.setenv("USERNAME", "alice")
    monkeypatch.setenv("USERDOMAIN", "DOM")
    assert _daemon._windows_task_user_id() == "DOM\\alice"

    monkeypatch.delenv("USERDOMAIN")
    assert _daemon._windows_task_user_id() == "alice"
