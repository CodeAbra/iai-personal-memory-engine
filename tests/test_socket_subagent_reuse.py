from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from tests.conftest_shared import pids_bound_to_unix_socket

REPO = Path(__file__).resolve().parent.parent
WRAPPER = REPO / "mcp-wrapper"

@pytest.fixture(scope="module")
def built_wrapper() -> Path:
    if not (WRAPPER / "node_modules").exists():
        subprocess.run(["npm", "install"], cwd=WRAPPER, check=True)
    subprocess.run(["npm", "run", "build"], cwd=WRAPPER, check=True)
    dist = WRAPPER / "dist" / "index.js"
    assert dist.exists(), "npm run build should have produced dist/index.js"
    return dist

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

def _daemon_pids_on_socket(sock_path: Path) -> set[int]:
    """Return PIDs of iai_mcp.daemon processes bound to exactly this socket.

    Scoped by the test's own unique socket path so the count ignores the
    always-warm production daemon and any daemon from another test/session that
    happens to start or exit during the window — those bind different sockets.
    A daemon spawned by the sub-agent path would bind THIS socket, so the count
    still detects a real singleton violation.
    """
    pids = pids_bound_to_unix_socket(sock_path)
    daemon_pids: set[int] = set()
    for pid in pids:
        try:
            if "iai_mcp.daemon" in " ".join(psutil.Process(pid).cmdline()):
                daemon_pids.add(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return daemon_pids

def _kill_test_daemons(sock_path: Path) -> None:
    for pid in pids_bound_to_unix_socket(sock_path):
        try:
            cl = " ".join(psutil.Process(pid).cmdline())
            if "iai_mcp.daemon" in cl:
                psutil.Process(pid).send_signal(signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

def _quick_recall_via_wrapper(
    built_wrapper: Path, env_overrides: dict[str, str], cue: str,
) -> dict:
    env = os.environ.copy()
    env["IAI_MCP_PYTHON"] = sys.executable
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.update(env_overrides)
    proc = subprocess.Popen(
        ["node", str(built_wrapper)],
        cwd=str(REPO),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        init = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "subagent-reuse-test", "version": "0.0"},
            },
        }
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write((json.dumps(init) + "\n").encode("utf-8"))
        proc.stdin.flush()
        init_line = proc.stdout.readline()
        if not init_line:
            raise RuntimeError(f"sub-agent wrapper closed stdout before initialize (cue={cue!r})")
        init_resp = json.loads(init_line.decode("utf-8"))
        assert "result" in init_resp, f"initialize failed: {init_resp}"
        note = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        proc.stdin.write((json.dumps(note) + "\n").encode("utf-8"))
        proc.stdin.flush()

        recall = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "memory_recall",
                "arguments": {"cue": cue, "budget_tokens": 50},
            },
        }
        proc.stdin.write((json.dumps(recall) + "\n").encode("utf-8"))
        proc.stdin.flush()
        deadline = time.monotonic() + 5.0
        line = b""
        while time.monotonic() < deadline:
            readable, _, _ = select.select([proc.stdout], [], [], 0.5)
            if readable:
                line = proc.stdout.readline()
                break
        if not line:
            raise RuntimeError(f"sub-agent recall timed out (cue={cue!r})")
        return json.loads(line.decode("utf-8"))
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

def _wait_for_daemon_socket(sock_path: Path, timeout_sec: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if sock_path.exists():
            return True
        time.sleep(0.1)
    return False

def _spawn_daemon_in_background(
    sock_path: Path, store_dir: Path, idle_secs: int = 120,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["IAI_DAEMON_SOCKET_PATH"] = str(sock_path)
    env["IAI_MCP_STORE"] = str(store_dir)
    env["IAI_DAEMON_IDLE_SHUTDOWN_SECS"] = str(idle_secs)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        [sys.executable, "-m", "iai_mcp.daemon"],
        cwd=str(REPO),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def test_subagent_spawns_zero_new_processes(built_wrapper, tmp_path):
    sock_dir = Path(f"/tmp/iai-subagent-{os.getpid()}-{id(tmp_path)}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"
    store_dir = sock_dir / "store"
    store_dir.mkdir(parents=True, exist_ok=True)
    assert not sock_path.exists()

    env_overrides = {
        "IAI_DAEMON_SOCKET_PATH": str(sock_path),
        "IAI_MCP_STORE": str(store_dir),
        "IAI_DAEMON_IDLE_SHUTDOWN_SECS": "120",
    }

    daemon_proc = _spawn_daemon_in_background(sock_path, store_dir)
    try:
        assert _wait_for_daemon_socket(sock_path, timeout_sec=30.0), (
            f"daemon did not bind socket {sock_path} within 30s"
        )
        time.sleep(0.3)

        first_resp = _quick_recall_via_wrapper(
            built_wrapper, env_overrides, cue="bootstrap subagent test",
        )
        assert "result" in first_resp or "error" in first_resp, first_resp

        before = _count_iai_mcp_processes()
        before_daemons = _daemon_pids_on_socket(sock_path)
        assert len(before_daemons) >= 1, (
            f"bootstrap did not leave a running daemon on {sock_path}: "
            f"{before_daemons}"
        )

        for i in range(3):
            resp = _quick_recall_via_wrapper(
                built_wrapper, env_overrides, cue=f"subagent recall #{i + 1}",
            )
            assert "result" in resp or "error" in resp, (
                f"sub-agent #{i + 1} response shape unexpected: {resp}"
            )
            time.sleep(0.3)

        time.sleep(0.5)

        after = _count_iai_mcp_processes()
        after_daemons = _daemon_pids_on_socket(sock_path)

        core_delta = after["core"] - before["core"]
        assert core_delta <= 0, (
            f"FAIL-LOUD: sub-agent path spawned iai_mcp.core "
            f"(before={before['core']} after={after['core']} delta={core_delta})"
        )

        # Scope the singleton check to daemons bound to THIS test's socket so an
        # unrelated daemon (the production daemon, or another test's) starting or
        # exiting during the window cannot perturb the count. A sub-agent that
        # wrongly spawned its own daemon would bind this same socket and appear
        # as a new PID here.
        new_daemons = after_daemons - before_daemons
        assert not new_daemons, (
            f"singleton violated: sub-agent path spawned an extra daemon on "
            f"{sock_path} (before={sorted(before_daemons)} "
            f"after={sorted(after_daemons)} new={sorted(new_daemons)})"
        )
    finally:
        try:
            daemon_proc.terminate()
            daemon_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            daemon_proc.kill()
        _kill_test_daemons(sock_path)
        time.sleep(0.5)
        try:
            sock_path.unlink()
        except OSError:
            pass
