from __future__ import annotations

import asyncio
import json
import os
import signal
import socket as sk
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil
import pytest

from _socket_test_helpers import (
    daemon_endpoint,
    daemon_endpoint_ready_path,
    new_daemon_client_socket,
    send_daemon_token,
)

@pytest.fixture
def short_socket_paths(tmp_path):
    lock_path = tmp_path / ".lock"
    state_path = tmp_path / ".daemon-state.json"

    with tempfile.TemporaryDirectory(prefix="iai-sock-") as sock_dir_name:
        sock_dir = Path(sock_dir_name)
        sock_path = sock_dir / "d.sock"
        try:
            yield lock_path, sock_path, state_path
        finally:
            try:
                if sock_path.exists():
                    sock_path.unlink()
            except OSError:
                pass

def _count_iai_mcp_processes() -> dict[str, int]:
    counts = {"core": 0, "daemon": 0}
    for p in psutil.process_iter(["cmdline"]):
        try:
            cl = p.info.get("cmdline") or []
            if not cl:
                continue
            joined = " ".join(c or "" for c in cl)
            if "iai_mcp.core" in joined:
                counts["core"] += 1
            if "iai_mcp.daemon" in joined:
                counts["daemon"] += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return counts

def _spawn_daemon_for_test(sock_path: Path, store_root: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["IAI_DAEMON_SOCKET_PATH"] = str(sock_path)
    env["IAI_MCP_STORE"] = str(store_root)
    env["IAI_DAEMON_IDLE_SHUTDOWN_SECS"] = "99999"
    return subprocess.Popen(
        [sys.executable, "-m", "iai_mcp.daemon"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def _wait_for_socket(sock_path: Path, timeout_sec: float = 30.0) -> bool:
    ready = daemon_endpoint_ready_path(sock_path)
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if ready.exists():
            return True
        time.sleep(0.1)
    return False

def test_kill_daemon_midcall_no_orphan_core_spawn(short_socket_paths, tmp_path):
    _, sock_path, _ = short_socket_paths
    store_root = tmp_path / "store"
    store_root.mkdir(parents=True, exist_ok=True)

    baseline = _count_iai_mcp_processes()

    proc = _spawn_daemon_for_test(sock_path, store_root)
    try:
        assert _wait_for_socket(sock_path, timeout_sec=30), (
            "daemon never bound socket within 30s"
        )

        before = _count_iai_mcp_processes()
        assert before["daemon"] >= baseline["daemon"] + 1, (
            f"our daemon not visible in process list: baseline={baseline}, before={before}"
        )
        before_delta = before["core"] - baseline["core"]
        assert before_delta == 0, (
            f"our daemon spawned {before_delta} iai_mcp.core processes BEFORE kill "
            f"(baseline={baseline}, before={before}) — singleton invariant violated"
        )

        proc.kill()
        proc.wait(timeout=5)

        time.sleep(0.5)

        after = _count_iai_mcp_processes()
        after_delta = after["core"] - baseline["core"]
        assert after_delta <= 0, (
            f"FAIL-LOUD VIOLATION: our daemon spawned {after_delta} new "
            f"iai_mcp.core processes after kill (baseline={baseline}, after={after}) "
            "— invariant: the daemon must never spawn a second core."
        )

        s = new_daemon_client_socket()
        s.settimeout(0.5)
        err_kind = None
        try:
            s.connect(daemon_endpoint(sock_path))
            err_kind = "no_error"
        except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
            err_kind = type(e).__name__
        finally:
            try:
                s.close()
            except OSError:
                pass
        assert err_kind in (
            "ConnectionRefusedError", "FileNotFoundError", "OSError", "TimeoutError",
        ), f"unexpected post-kill connect outcome: {err_kind}"
    finally:
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        try:
            if sock_path.exists():
                sock_path.unlink()
        except OSError:
            pass

def test_kill_daemon_during_active_connection(short_socket_paths, tmp_path):
    _, sock_path, _ = short_socket_paths
    store_root = tmp_path / "store"
    store_root.mkdir(parents=True, exist_ok=True)

    proc = _spawn_daemon_for_test(sock_path, store_root)
    try:
        assert _wait_for_socket(sock_path, timeout_sec=30), (
            "daemon never bound socket within 30s"
        )

        s = new_daemon_client_socket()
        s.settimeout(15)
        s.connect(daemon_endpoint(sock_path))
        send_daemon_token(s, sock_path)  # Windows handshake; no-op on POSIX
        msg = (json.dumps({"type": "status"}) + "\n").encode("utf-8")
        s.sendall(msg)

        first_response = b""
        while not first_response.endswith(b"\n"):
            chunk = s.recv(4096)
            if not chunk:
                break
            first_response += chunk
        assert first_response, "daemon never replied to initial status"
        decoded = json.loads(first_response.decode("utf-8"))
        assert decoded.get("ok") is True, decoded

        proc.kill()
        proc.wait(timeout=5)

        s.settimeout(2.0)
        eof_or_error = False
        try:
            chunk = s.recv(4096)
            if chunk == b"":
                eof_or_error = True
        except (ConnectionResetError, BrokenPipeError, OSError):
            eof_or_error = True
        finally:
            try:
                s.close()
            except OSError:
                pass
        assert eof_or_error, (
            "daemon kill did not surface as EOF / OSError on open connection — "
            "wrapper-side daemon_unreachable translation would silently hang"
        )

        s2 = new_daemon_client_socket()
        s2.settimeout(0.5)
        err_kind = None
        try:
            s2.connect(daemon_endpoint(sock_path))
            err_kind = "no_error"
        except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
            err_kind = type(e).__name__
        finally:
            try:
                s2.close()
            except OSError:
                pass
        assert err_kind in (
            "ConnectionRefusedError", "FileNotFoundError", "OSError", "TimeoutError",
        ), f"unexpected post-kill connect outcome: {err_kind}"
    finally:
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        try:
            if sock_path.exists():
                sock_path.unlink()
        except OSError:
            pass
