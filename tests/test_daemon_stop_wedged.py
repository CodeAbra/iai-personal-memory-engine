from __future__ import annotations

import os
import signal
import subprocess
import sys
import time


from iai_mcp.lifecycle_lock import _is_pid_alive


_WEDGED_CHILD = r"""
import asyncio, os, signal, sys, time

ready_path = sys.argv[1]
sock_path = sys.argv[2]

async def _main():
    loop = asyncio.get_running_loop()
    serviced = {"sigterm": False}
    loop.add_signal_handler(signal.SIGTERM, lambda: serviced.__setitem__("sigterm", True))
    with open(ready_path, "w") as fh:
        fh.write(str(os.getpid()))
        fh.flush()
        os.fsync(fh.fileno())
    time.sleep(3600)

asyncio.run(_main())
"""


def _spawn_wedged_child(tmp_path):
    ready_path = tmp_path / "child_ready"
    sock_path = tmp_path / "test_daemon.sock"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _WEDGED_CHILD,
            str(ready_path),
            str(sock_path),
            "iai_mcp.daemon",
        ],
    )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if ready_path.exists():
            return proc
        if proc.poll() is not None:
            raise AssertionError(
                f"wedged child exited early rc={proc.returncode}"
            )
        time.sleep(0.02)
    raise AssertionError("wedged child did not signal readiness within 5s")


def test_wedged_loop_survives_sigterm_then_killed_by_escalation(tmp_path):
    proc = _spawn_wedged_child(tmp_path)
    pid = proc.pid
    try:
        assert _is_pid_alive(pid), "child not recognised as alive by the gate"

        os.kill(pid, signal.SIGTERM)
        sigterm_bound = time.monotonic() + 1.5
        while time.monotonic() < sigterm_bound:
            if not _is_pid_alive(pid):
                break
            time.sleep(0.05)
        assert _is_pid_alive(pid), (
            "wedged child died from SIGTERM alone -- the in-loop handler gap "
            "is not being reproduced (the loop must absorb the SIGTERM)"
        )

        os.kill(pid, signal.SIGKILL)
        kill_bound = time.monotonic() + 5.0
        dead = False
        while time.monotonic() < kill_bound:
            if not _is_pid_alive(pid):
                dead = True
                break
            time.sleep(0.02)
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass
        assert dead or proc.poll() is not None, (
            "SIGKILL escalation did not terminate the wedged child within "
            "the bound"
        )
        assert proc.poll() is not None, "child still running after SIGKILL"
        assert proc.returncode in (-signal.SIGKILL, signal.SIGKILL, 137), (
            f"unexpected child returncode {proc.returncode}"
        )
    finally:
        if proc.poll() is None:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass


def test_escalation_only_targets_the_spawned_child_pid(tmp_path):
    proc = _spawn_wedged_child(tmp_path)
    pid = proc.pid
    try:
        assert pid != os.getpid()
        assert _is_pid_alive(pid)
        os.kill(pid, signal.SIGKILL)
        proc.wait(timeout=5.0)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            try:
                os.kill(pid, signal.SIGKILL)
                proc.wait(timeout=5.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
