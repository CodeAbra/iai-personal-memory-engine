from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WRAPPER = REPO / "mcp-wrapper"


def _wrapper_dist() -> Path:
    return WRAPPER / "dist" / "index.js"


@pytest.fixture(scope="module")
def built_wrapper() -> Path:
    if not (WRAPPER / "node_modules").exists():
        subprocess.run(["npm", "install"], cwd=WRAPPER, check=True)
    subprocess.run(["npm", "run", "build"], cwd=WRAPPER, check=True)
    dist = _wrapper_dist()
    assert dist.exists(), "npm run build should have produced dist/index.js"
    return dist


def _spawn_wrapper_no_daemon(
    built_wrapper: Path,
    nonexistent_sock: Path,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["IAI_MCP_PYTHON"] = sys.executable
    env["IAI_DAEMON_SOCKET_PATH"] = str(nonexistent_sock)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        ["node", str(built_wrapper)],
        cwd=str(REPO),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _send_rpc(proc: subprocess.Popen, method: str, params: dict, rpc_id: int) -> None:
    req = {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params}
    assert proc.stdin is not None
    proc.stdin.write((json.dumps(req) + "\n").encode())
    proc.stdin.flush()


def _send_notification(proc: subprocess.Popen, method: str) -> None:
    note = {"jsonrpc": "2.0", "method": method}
    assert proc.stdin is not None
    proc.stdin.write((json.dumps(note) + "\n").encode())
    proc.stdin.flush()


def _read_response_with_id(
    proc: subprocess.Popen,
    rpc_id: int,
    timeout_sec: float,
) -> dict:
    assert proc.stdout is not None
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            stderr_tail = b""
            try:
                stderr_tail = proc.stderr.read() if proc.stderr else b""
            except Exception:
                pass
            raise RuntimeError(
                f"wrapper closed stdout before responding to id={rpc_id}; "
                f"stderr tail: {stderr_tail.decode(errors='replace')[-2000:]}"
            )
        line_str = line.decode(errors="replace").strip()
        if not line_str:
            continue
        try:
            msg = json.loads(line_str)
        except json.JSONDecodeError:
            continue
        if isinstance(msg, dict) and msg.get("id") == rpc_id:
            return msg
    raise TimeoutError(
        f"wrapper did not respond to id={rpc_id} within {timeout_sec}s"
    )


def test_tools_list_returns_without_daemon(
    built_wrapper: Path,
    tmp_path: Path,
) -> None:
    nonexistent_sock = tmp_path / "iai-mcp-this-socket-does-not-exist.sock"
    assert not nonexistent_sock.exists()

    proc = _spawn_wrapper_no_daemon(built_wrapper, nonexistent_sock)
    try:
        t0 = time.monotonic()
        _send_rpc(
            proc,
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "iai-mcp-no-daemon-test", "version": "0.1.0"},
            },
            1,
        )
        init_resp = _read_response_with_id(proc, 1, timeout_sec=2.0)
        assert "result" in init_resp, f"initialize failed: {init_resp}"

        _send_notification(proc, "notifications/initialized")

        _send_rpc(proc, "tools/list", {}, 2)
        list_resp = _read_response_with_id(proc, 2, timeout_sec=2.0)
        elapsed = time.monotonic() - t0

        assert "result" in list_resp, f"tools/list error: {list_resp}"
        tools = list_resp["result"]["tools"]
        names = {t["name"] for t in tools}
        expected = {
            "memory_recall",
        "memory_search",
            "memory_recall_structural",
            "memory_reinforce",
            "memory_contradict",
            "memory_capture",
            "memory_consolidate",
            "profile_get_set",
            "curiosity_pending",
            "schema_list",
            "events_query",
            "topology",
            "episodes_recent",
            "memory_temporal_recall",
        }
        assert names == expected, (
            f"tools/list returned {len(names)} tools, expected {len(expected)}: "
            f"missing={expected - names}, extra={names - expected}"
        )
        assert elapsed < 4.0, (
            f"tools/list took {elapsed:.2f}s with no daemon — pre-fix "
            f"regression: wrapper is blocking on bridge.start() before "
            f"serving tools/list."
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
