"""batch_execute tool end-to-end through the real resident executor daemon.

Follows tests/test_tools_mcp_server_execute.py's shape: a real subprocess
against a temp IAI_TOOLS_ROOT/socket, never a live ~/.iai-tools/. Skips (does
not fail) only when iai_tools_execd truly cannot be found -- this test's
whole purpose is exercising the real executor, never a stub.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WRAPPER = REPO / "mcp-wrapper"


def _execd_bin() -> str:
    for profile in ("release", "debug"):
        candidate = REPO / "target" / profile / "iai_tools_execd"
        if candidate.exists():
            return str(candidate)
    pytest.skip("iai_tools_execd not built; run: cargo build -p iai_mcp_tools_core --bins")


def _wrapper_dist() -> Path:
    return WRAPPER / "dist" / "tools-index.js"


@pytest.fixture(scope="module")
def built_wrapper() -> Path:
    if not (WRAPPER / "node_modules").exists():
        subprocess.run(["npm", "install"], cwd=WRAPPER, check=True)
    subprocess.run(["npm", "run", "build"], cwd=WRAPPER, check=True)
    dist = _wrapper_dist()
    assert dist.exists(), "npm run build should have produced dist/tools-index.js"
    return dist


_TEST_IDLE_SECS = 1.0


@pytest.fixture
def exec_env(tmp_path: Path):
    root = tmp_path / "iai-tools-root"
    # AF_UNIX's sun_path is capped at ~104 bytes on macOS -- pytest's own
    # tmp_path nests several directories deep and overflows that, so the
    # socket lives in its own short-path directory directly under /tmp.
    sock_dir = Path(tempfile.mkdtemp(prefix="iai-tools-mcp-", dir="/tmp"))
    sock_path = sock_dir / ".execd.sock"
    try:
        yield {"root": root, "sock_path": sock_path}
    finally:
        shutil.rmtree(sock_dir, ignore_errors=True)


def _assert_daemon_exited(sock_path: Path, idle_secs: float = _TEST_IDLE_SECS, margin: float = 3.0) -> None:
    time.sleep(idle_secs + margin)
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.connect(str(sock_path))
    except (ConnectionRefusedError, FileNotFoundError):
        return
    probe.close()
    raise AssertionError(
        f"daemon at {sock_path} did not self-terminate within {idle_secs + margin}s"
    )


def _spawn_server(built_wrapper: Path, home_dir: Path, exec_env: dict) -> subprocess.Popen:
    env = os.environ.copy()
    env["IAI_MCP_PYTHON"] = sys.executable
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["HOME"] = str(home_dir)
    env["IAI_TOOLS_ROOT"] = str(exec_env["root"])
    env["IAI_TOOLS_SOCKET_PATH"] = str(exec_env["sock_path"])
    env["IAI_TOOLS_EXECD_BIN"] = _execd_bin()
    env["IAI_TOOLS_IDLE_SECS"] = str(_TEST_IDLE_SECS)
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


def _read_response_with_id(proc: subprocess.Popen, rpc_id: int, timeout_sec: float) -> dict:
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
                f"server closed stdout before responding to id={rpc_id}; "
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
    raise TimeoutError(f"server did not respond to id={rpc_id} within {timeout_sec}s")


def test_batch_execute_returns_results_in_input_order_bounded_and_failure_as_result(
    built_wrapper: Path,
    tmp_path: Path,
    exec_env: dict,
) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    sock_path = exec_env["sock_path"]
    assert not sock_path.exists(), "no daemon must be running before this test's first call"

    proc = _spawn_server(built_wrapper, home_dir, exec_env)
    try:
        _send_rpc(
            proc,
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "iai-tools-batch-execute-test", "version": "0.1.0"},
            },
            1,
        )
        init_resp = _read_response_with_id(proc, 1, timeout_sec=5.0)
        assert "result" in init_resp, f"initialize failed: {init_resp}"

        _send_notification(proc, "notifications/initialized")

        emitted_bytes = 200_000
        commands = [
            {"command": "sh", "args": ["-c", "sleep 0.3; echo first"]},
            {"command": "echo", "args": ["second"]},
            {"command": "python3", "args": ["-c", f"print('x' * {emitted_bytes})"]},
            {"command": "false"},
        ]
        _send_rpc(
            proc,
            "tools/call",
            {"name": "batch_execute", "arguments": {"commands": commands, "concurrency": 2}},
            2,
        )
        call_resp = _read_response_with_id(proc, 2, timeout_sec=30.0)
        assert "result" in call_resp, f"tools/call error: {call_resp}"
        result = call_resp["result"]
        assert not result.get("isError"), f"batch_execute returned an error: {result}"
        structured = result.get("structuredContent") or json.loads(result["content"][0]["text"])

        results = structured["results"]
        assert len(results) == 4, results
        for element in results:
            # exact key set -- also proves no auto-index field (indexed/refused/index_error) leaks in
            assert set(element.keys()) == {"run_id", "state", "stdout", "stderr"}, element

        # input order, not completion order -- the sleepy command stays first
        assert "first" in results[0]["stdout"], results[0]
        assert "second" in results[1]["stdout"], results[1]

        # bounded tail: non-empty but shorter than what the command emitted
        large_stdout = results[2]["stdout"]
        assert large_stdout, "expected non-empty bounded output"
        assert len(large_stdout) < emitted_bytes, (
            f"expected the tail to be bounded below {emitted_bytes} bytes, got {len(large_stdout)}"
        )

        # a per-command failure surfaces as a result element, not an MCP error
        assert results[3]["state"] == "exited:1", results[3]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    _assert_daemon_exited(sock_path)
