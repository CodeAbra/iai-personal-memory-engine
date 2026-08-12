"""End-to-end witness for the operator daemon lifecycle: an isolated daemon
starts, answers status with ok:True and a fresh heartbeat, is stopped, and
leaves no socket behind -- with the real prod daemon (if any) proven
untouched throughout.
"""
from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

import pytest

from iai_mcp.cli import _send_socket_request
from tests._warm_recall_repro_support import await_socket, spawn_isolated_daemon


def _real_home() -> Path:
    """The real (non-test) HOME, independent of any HOME override already in
    this process's environment."""
    try:
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (KeyError, ImportError, AttributeError):
        return Path.home()


def _prod_daemon_pid() -> "int | None":
    state_path = _real_home() / ".iai-mcp" / ".daemon-state.json"
    if not state_path.exists():
        return None
    try:
        data = json.loads(state_path.read_text())
    except (OSError, ValueError):
        return None
    pid = data.get("daemon_pid")
    return int(pid) if isinstance(pid, int) else None


def _assert_prod_daemon_alive_if_present(pid: "int | None") -> None:
    if pid is None:
        return
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pytest.fail(
            f"the real prod daemon (pid {pid}) is no longer alive after "
            f"this test ran an isolated daemon -- possible cross-contamination"
        )
    except PermissionError:
        pass  # alive, just not signalable by this uid -- fine
    except OSError:
        pass


def test_isolated_daemon_start_status_stop_socket_gone(tmp_path, monkeypatch):
    try:
        import iai_mcp_native  # noqa: F401
    except ImportError:
        pytest.skip(
            "iai_mcp_native is unavailable; isolated daemon lifecycle "
            "requires the native engine extension"
        )

    prod_pid = _prod_daemon_pid()

    base = tmp_path / "iso-base"
    scratch_home = tmp_path / "scratch-home"
    base.mkdir(parents=True)

    handle = spawn_isolated_daemon(
        base,
        "lilli",
        scratch_home=scratch_home,
        extra_env={"IAI_MCP_CRYPTO_PASSPHRASE": "uat-daemon-lifecycle-passphrase"},
    )
    try:
        await_socket(handle.socket_path, timeout=30.0)

        # Route the CLI's own status entry point at the isolated socket,
        # never the prod one.
        monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(handle.socket_path))

        deadline = time.monotonic() + 20.0
        resp: "dict | None" = None
        while time.monotonic() < deadline:
            resp = _send_socket_request({"type": "status"}, timeout=5.0)
            if resp is not None and resp.get("ok") is True:
                break
            time.sleep(0.5)

        assert resp is not None, "isolated daemon never answered a status request"
        assert resp.get("ok") is True, f"status ok!=True: {resp!r}"

        # Fresh heartbeat: last_tick_at once the FSM has ticked, or
        # daemon_started_at as the process-level liveness proof before its
        # first tick -- either proves the daemon just booted and is live.
        heartbeat_present = bool(resp.get("last_tick_at") or resp.get("daemon_started_at"))
        assert heartbeat_present, f"no heartbeat field present in status: {resp!r}"

        # Stop: SIGTERM the daemon process directly and confirm the socket
        # is gone because the daemon itself unlinked it on shutdown -- not
        # because DaemonHandle.terminate()'s later rmtree deleted the
        # containing directory (that would make the assertion vacuous).
        handle.proc.send_signal(signal.SIGTERM)
        handle.proc.wait(timeout=15.0)
        assert not handle.socket_path.exists(), (
            f"socket {handle.socket_path} still present after daemon "
            f"SIGTERM+exit"
        )
    finally:
        handle.terminate()
        _assert_prod_daemon_alive_if_present(prod_pid)
