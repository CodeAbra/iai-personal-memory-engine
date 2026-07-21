from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest


@pytest.fixture
def isolated_daemon_paths(tmp_path, monkeypatch):
    iai_dir = tmp_path / ".iai-mcp"
    iai_dir.mkdir(parents=True, exist_ok=True)

    state_path = iai_dir / ".daemon-state.json"
    lock_path = iai_dir / ".lock"
    store_dir = iai_dir / "store"
    store_dir.mkdir(parents=True, exist_ok=True)

    sock_dir = Path(f"/tmp/iai-rec-{os.getpid()}-{id(tmp_path)}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"

    real_hf_home = Path.home() / ".cache" / "huggingface"

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(real_hf_home))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(sock_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(store_dir))
    monkeypatch.setenv("IAI_DAEMON_IDLE_SHUTDOWN_SECS", "99999")
    monkeypatch.setenv(
        "PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring"
    )
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-recovery-passphrase")
    import keyring.core

    keyring.core._keyring_backend = None

    from iai_mcp import cli, daemon_state

    monkeypatch.setattr(daemon_state, "STATE_PATH", state_path)
    monkeypatch.setattr(cli, "LOCK_PATH", lock_path)
    monkeypatch.setattr(cli, "SOCKET_PATH", sock_path)

    try:
        yield sock_path, state_path, store_dir, lock_path
    finally:
        _kill_test_daemons(sock_path)
        try:
            if sock_path.exists():
                sock_path.unlink()
        except OSError:
            pass
        try:
            sock_dir.rmdir()
        except OSError:
            pass
        import keyring.core

        keyring.core._keyring_backend = None


def _spawn_daemon(sock_path: Path, store_dir: Path, home: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["IAI_DAEMON_SOCKET_PATH"] = str(sock_path)
    env["IAI_MCP_STORE"] = str(store_dir)
    env["IAI_DAEMON_IDLE_SHUTDOWN_SECS"] = "99999"
    env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
    env["IAI_MCP_CRYPTO_PASSPHRASE"] = "test-recovery-passphrase"
    return subprocess.Popen(
        [sys.executable, "-m", "iai_mcp.daemon"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_socket_and_pid(
    sock_path: Path, state_path: Path, expected_pid: int, timeout_sec: float = 30.0
) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if sock_path.exists() and state_path.exists():
            try:
                state = json.loads(state_path.read_text())
                if state.get("daemon_pid") == expected_pid:
                    return True
            except (OSError, json.JSONDecodeError):
                pass
        time.sleep(0.1)
    return False


def _wait_for_socket_only(sock_path: Path, timeout_sec: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if sock_path.exists():
            return True
        time.sleep(0.1)
    return False


def _kill_test_daemons(sock_path: Path) -> None:
    target = str(sock_path)
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cl = " ".join(p.info.get("cmdline") or [])
            if "iai_mcp.daemon" not in cl:
                continue
            try:
                env = p.environ()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
            if env.get("IAI_DAEMON_SOCKET_PATH") == target:
                try:
                    p.send_signal(signal.SIGTERM)
                    p.wait(timeout=3)
                except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                    try:
                        p.send_signal(signal.SIGKILL)
                    except psutil.NoSuchProcess:
                        pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


@pytest.mark.slow
def test_apply_yes_recovers_from_kill(isolated_daemon_paths):
    sock_path, state_path, store_dir, _ = isolated_daemon_paths

    proc = _spawn_daemon(sock_path, store_dir, home=Path(os.environ["HOME"]))
    try:
        assert _wait_for_socket_and_pid(
            sock_path, state_path, proc.pid, timeout_sec=30
        ), (
            f"daemon never bound socket + stamped daemon_pid={proc.pid} within 30s"
        )

        original_pid = proc.pid

        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=5)
        time.sleep(0.5)

        from iai_mcp.doctor import cmd_doctor, run_diagnosis

        pre_results = run_diagnosis()
        pre_fail_names = [r.name for r in pre_results if not r.passed]
        assert "(a) daemon process alive" in pre_fail_names, (
            f"after kill, check (a) should FAIL; got fails: {pre_fail_names}"
        )
        assert "(b) socket file fresh" in pre_fail_names, (
            f"after kill, check (b) should FAIL; got fails: {pre_fail_names}"
        )

        t0 = time.monotonic()
        args = argparse.Namespace(apply=True, yes=True)
        rc = cmd_doctor(args)
        elapsed = time.monotonic() - t0

        assert rc == 0, (
            f"doctor recovery returned rc={rc}, elapsed={elapsed:.2f}s "
            "— expected exit 0 (all PASS after recovery)"
        )
        assert elapsed < 15.0, (
            f"doctor recovery took {elapsed:.2f}s, exceeds 15s safety budget"
        )

        assert state_path.exists(), "respawned daemon never wrote state file"
        s2 = json.loads(state_path.read_text())
        new_pid = s2.get("daemon_pid")
        assert new_pid is not None, "respawned daemon did not stamp daemon_pid"
        assert new_pid != original_pid, (
            f"daemon was not actually respawned: same PID {new_pid} after recovery"
        )

        post_results = run_diagnosis()
        post_fails = [r.name for r in post_results if not r.passed]
        assert post_fails == [], f"post-recovery FAILs remain: {post_fails}"

        # Stop the respawned daemon so we can open the store for event audit.
        # Use the PID from the state file for a reliable kill.
        _respawned_pid = new_pid
        try:
            os.kill(_respawned_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _kill_test_daemons(sock_path)

        # Poll until the hippo flock is released (kernel releases on
        # process death, but reaping may take a moment).
        import errno as _errno
        from iai_mcp import _flock as _fcntl
        _hippo_lock = store_dir / "hippo" / ".lock"
        _flock_deadline = time.monotonic() + 8.0
        while time.monotonic() < _flock_deadline:
            if not _hippo_lock.exists():
                break
            _probe_fd = os.open(str(_hippo_lock), os.O_RDONLY)
            try:
                _fcntl.flock(_probe_fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
                _fcntl.flock(_probe_fd, _fcntl.LOCK_UN)
                break
            except OSError as _e:
                if _e.errno not in (_errno.EAGAIN, _errno.EWOULDBLOCK):
                    raise
            finally:
                os.close(_probe_fd)
            time.sleep(0.2)

        from iai_mcp.events import query_events
        from iai_mcp.store import MemoryStore

        with MemoryStore() as store:
            recent = query_events(store, kind="doctor_action", limit=10)
        assert len(recent) >= 1, (
            "doctor_action events not written to ledger after --apply"
        )
        action_labels = {e["data"].get("action") for e in recent}
        # unlink_stale_socket is written before the daemon respawns (store
        # still idle). daemon_respawned_by_doctor is written by the daemon
        # itself on boot (via IAI_DAEMON_RESPAWN_BY env marker), so it
        # does not contend with the doctor for the EXCLUSIVE flock.
        assert "unlink_stale_socket" in action_labels, (
            f"unlink_stale_socket event missing; saw actions: {action_labels}"
        )
        assert "daemon_respawned_by_doctor" in action_labels, (
            f"daemon_respawned_by_doctor event missing; saw actions: {action_labels}"
        )
    finally:
        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGKILL)
                proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                pass


def test_apply_no_yes_skips_destructive_action_on_n_response(
    isolated_daemon_paths, monkeypatch
):
    sock_path, _, _, _ = isolated_daemon_paths

    import psutil

    class _FakeProc:
        def __init__(self, pid: int, cmdline: list[str]):
            self.info = {"pid": pid, "cmdline": cmdline}

    fake = _FakeProc(99_999, ["python", "-m", "iai_mcp.core"])
    monkeypatch.setattr(psutil, "process_iter", lambda *a, **kw: [fake])

    monkeypatch.setattr("builtins.input", lambda *a, **kw: "n")

    from iai_mcp.doctor import cmd_doctor

    args = argparse.Namespace(apply=True, yes=False)
    rc = cmd_doctor(args)

    assert rc == 2, (
        f"declining destructive action should leave FAILs unfixed → rc=2; got {rc}"
    )
