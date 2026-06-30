from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil
import pytest

REPO = Path(__file__).resolve().parent.parent
WRAPPER = REPO / "mcp-wrapper"


def _wrapper_ready() -> bool:
    return (WRAPPER / "dist" / "index.js").exists()


@pytest.fixture(scope="module")
def built_wrapper() -> Path:
    if not (WRAPPER / "node_modules").exists():
        subprocess.run(["npm", "install"], cwd=WRAPPER, check=True)
    subprocess.run(["npm", "run", "build"], cwd=WRAPPER, check=True)
    dist = WRAPPER / "dist" / "index.js"
    assert dist.exists(), "npm run build should have produced dist/index.js"
    return dist


@pytest.fixture(scope="module")
def daemon_sock() -> "Path":
    sock_dir_ctx = tempfile.TemporaryDirectory(prefix="iai-sock-")
    sock_dir = Path(sock_dir_ctx.name)
    sock_path = sock_dir / "d.sock"
    store_dir = Path(tempfile.mkdtemp(prefix="iai-store-"))
    store_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["IAI_DAEMON_SOCKET_PATH"] = str(sock_path)
    env["IAI_MCP_STORE"] = str(store_dir)
    env.setdefault(
        "IAI_MCP_CRYPTO_PASSPHRASE",
        "iai-mcp-test-passphrase-2026-04-30",
    )
    env["IAI_DAEMON_IDLE_SHUTDOWN_SECS"] = "300"
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    daemon_proc = subprocess.Popen(
        [sys.executable, "-m", "iai_mcp.daemon"],
        cwd=str(REPO),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if sock_path.exists():
            break
        time.sleep(0.1)
    else:
        try:
            daemon_proc.kill()
        except OSError:
            pass
        raise RuntimeError(f"test daemon did not bind socket {sock_path} within 30s")

    yield sock_path

    try:
        daemon_proc.terminate()
        daemon_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        daemon_proc.kill()
    sock_str = str(sock_path)
    for p in psutil.process_iter(["cmdline", "environ"]):
        try:
            cl = " ".join(p.info.get("cmdline") or [])
            if "iai_mcp.daemon" not in cl:
                continue
            penv = p.info.get("environ") or {}
            if penv.get("IAI_DAEMON_SOCKET_PATH") == sock_str:
                p.send_signal(signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    time.sleep(0.3)
    try:
        sock_path.unlink()
    except OSError:
        pass
    try:
        sock_dir_ctx.cleanup()
        shutil.rmtree(store_dir, ignore_errors=True)
    except OSError:
        pass


def _mcp_call(proc: subprocess.Popen, method: str, params: dict, rpc_id: int) -> dict:
    req = {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params}
    assert proc.stdin is not None
    proc.stdin.write((json.dumps(req) + "\n").encode())
    proc.stdin.flush()
    assert proc.stdout is not None
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("wrapper closed stdout before replying")
    return json.loads(line.decode())


def _spawn_wrapper(built_wrapper: Path, daemon_sock: Path | None = None) -> subprocess.Popen:
    env = os.environ.copy()
    env["IAI_MCP_PYTHON"] = sys.executable
    if daemon_sock is not None:
        env["IAI_DAEMON_SOCKET_PATH"] = str(daemon_sock)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        ["node", str(built_wrapper)],
        cwd=str(REPO),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _initialize(proc: subprocess.Popen, rpc_id: int = 1) -> None:
    resp = _mcp_call(
        proc,
        "initialize",
        {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "iai-mcp-test", "version": "0.1.0"},
        },
        rpc_id,
    )
    assert "result" in resp, f"initialize failed: {resp}"
    assert proc.stdin is not None
    note = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    proc.stdin.write((json.dumps(note) + "\n").encode())
    proc.stdin.flush()


def test_wrapper_lists_twelve_tools(built_wrapper: Path, daemon_sock: Path) -> None:
    proc = _spawn_wrapper(built_wrapper, daemon_sock)
    try:
        _initialize(proc, 1)
        resp = _mcp_call(proc, "tools/list", {}, 2)
        assert "result" in resp, f"tools/list error: {resp}"
        tools = resp["result"]["tools"]
        names = {t["name"] for t in tools}
        assert names == {
            "memory_recall",
            "memory_reinforce",
            "memory_contradict",
            "memory_consolidate",
            "profile_get_set",
            "curiosity_pending",
            "schema_list",
            "events_query",
            "memory_recall_structural",
            "topology",
            "camouflaging_status",
            "memory_capture",
            "episodes_recent",
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_wrapper_profile_get_returns_live_knobs(built_wrapper: Path, daemon_sock: Path) -> None:
    proc = _spawn_wrapper(built_wrapper, daemon_sock)
    try:
        _initialize(proc, 1)
        resp = _mcp_call(
            proc,
            "tools/call",
            {"name": "profile_get_set", "arguments": {"operation": "get"}},
            2,
        )
        assert "result" in resp, f"tools/call error: {resp}"
        content = resp["result"]["content"][0]["text"]
        payload = json.loads(content)
        assert payload["live"]["literal_preservation"] == "strong"
        assert payload["live"]["masking_off"] is True
        assert payload["live"]["task_support"] == "cued_recognition"
        assert payload["live"]["scene_construction_scaffold"] is True
        assert len(payload["live"]) == 11
        assert len(payload["deferred"]) == 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_wrapper_memory_consolidate_runs_heavy(built_wrapper: Path, daemon_sock: Path) -> None:
    proc = _spawn_wrapper(built_wrapper, daemon_sock)
    try:
        _initialize(proc, 1)
        resp = _mcp_call(
            proc,
            "tools/call",
            {"name": "memory_consolidate", "arguments": {}},
            2,
        )
        assert "result" in resp, f"tools/call error: {resp}"
        content = resp["result"]["content"][0]["text"]
        payload = json.loads(content)
        assert payload["mode"] == "heavy"
        assert payload["tier"] in ("tier0", "tier1")
        assert "summaries_created" in payload
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
