from __future__ import annotations

import json
import os
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


def _kill_test_daemons(sock_path: Path) -> None:
    target = str(sock_path)
    res = subprocess.run(
        ["lsof", "-U", "-F", "pn"],
        capture_output=True, text=True, check=False,
    )
    current: int | None = None
    pids: set[int] = set()
    for line in res.stdout.splitlines():
        if line.startswith("p"):
            try:
                current = int(line[1:])
            except ValueError:
                current = None
        elif line.startswith("n") and current is not None and line[1:] == target:
            pids.add(current)
    for pid in pids:
        try:
            cl = " ".join(psutil.Process(pid).cmdline())
            if "iai_mcp.daemon" in cl:
                psutil.Process(pid).send_signal(signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def _spawn_wrapper(
    built_wrapper: Path,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["IAI_MCP_PYTHON"] = sys.executable
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    if env_overrides:
        env.update(env_overrides)
    return subprocess.Popen(
        ["node", str(built_wrapper)],
        cwd=str(REPO),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _spawn_daemon_in_background(
    sock_path: Path, store_dir: Path
) -> subprocess.Popen:
    env = os.environ.copy()
    env["IAI_DAEMON_SOCKET_PATH"] = str(sock_path)
    env["IAI_MCP_STORE"] = str(store_dir)
    env["IAI_DAEMON_IDLE_SHUTDOWN_SECS"] = "120"
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        [sys.executable, "-m", "iai_mcp.daemon"],
        cwd=str(REPO),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _initialize(proc: subprocess.Popen, rpc_id: int = 1) -> dict:
    assert proc.stdin is not None and proc.stdout is not None
    init = {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "iai-mcp-bridge-no-spawn-test", "version": "0.1.0"},
        },
    }
    proc.stdin.write((json.dumps(init) + "\n").encode("utf-8"))
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("wrapper closed stdout before initialize reply")
    resp = json.loads(line.decode("utf-8"))
    note = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    proc.stdin.write((json.dumps(note) + "\n").encode("utf-8"))
    proc.stdin.flush()
    return resp


def _call_memory_recall(
    proc: subprocess.Popen,
    cue: str,
    rpc_id: int = 2,
    *,
    timeout_sec: float = 10.0,
) -> tuple[float, dict]:
    assert proc.stdin is not None and proc.stdout is not None
    req = {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "method": "tools/call",
        "params": {
            "name": "memory_recall",
            "arguments": {"cue": cue, "budget_tokens": 100},
        },
    }
    t0 = time.monotonic()
    proc.stdin.write((json.dumps(req) + "\n").encode("utf-8"))
    proc.stdin.flush()
    import select
    deadline = time.monotonic() + timeout_sec
    line = b""
    while time.monotonic() < deadline:
        readable, _, _ = select.select([proc.stdout], [], [], 0.5)
        if readable:
            line = proc.stdout.readline()
            break
    elapsed = time.monotonic() - t0
    if not line:
        raise RuntimeError(
            f"no response within {timeout_sec}s "
            f"(stderr: {proc.stderr.read1(2000) if proc.stderr else b'?'!r})"
        )
    return elapsed, json.loads(line.decode("utf-8"))


def _wait_for_daemon_socket(sock_path: Path, timeout_sec: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if sock_path.exists():
            return True
        time.sleep(0.1)
    return False


def test_start_throws_DaemonUnreachableError_when_socket_missing(
    built_wrapper, tmp_path
):
    sock_dir_ctx = tempfile.TemporaryDirectory(prefix="iai-sock-")
    sock_dir = Path(sock_dir_ctx.name)
    sock_path = sock_dir / "d.sock"
    store_dir = tmp_path / "store"
    store_dir.mkdir(parents=True, exist_ok=True)

    assert not sock_path.exists(), f"tmp socket pre-exists: {sock_path}"

    baseline = _count_iai_mcp_processes()
    daemon_baseline = baseline["daemon"]
    core_baseline = baseline["core"]

    env_overrides = {
        "IAI_DAEMON_SOCKET_PATH": str(sock_path),
        "IAI_MCP_STORE": str(store_dir),
    }
    wrapper_proc = _spawn_wrapper(built_wrapper, env_overrides)
    try:
        init_resp = _initialize(wrapper_proc, rpc_id=1)
        assert "result" in init_resp, f"initialize failed: {init_resp}"

        list_req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }
        wrapper_proc.stdin.write((json.dumps(list_req) + "\n").encode("utf-8"))
        wrapper_proc.stdin.flush()
        list_t0 = time.monotonic()
        line = wrapper_proc.stdout.readline()
        list_elapsed = time.monotonic() - list_t0
        assert line, "wrapper closed stdout before tools/list reply"
        list_resp = json.loads(line.decode("utf-8"))
        assert "result" in list_resp, f"tools/list error: {list_resp}"
        tools = list_resp["result"]["tools"]
        names = {t["name"] for t in tools}
        assert len(names) == 13, (
            f"tools/list returned {len(names)} tools, expected 13. "
            f"names={sorted(names)}"
        )
        assert list_elapsed < 4.0, (
            f"tools/list took {list_elapsed:.2f}s with no daemon — "
            f"regression: wrapper is blocking server.connect on "
            f"bridge.start (the mcp-tools-list-empty-cache bug)."
        )

        time.sleep(7.0)
        assert wrapper_proc.poll() is None, (
            f"wrapper exited (rc={wrapper_proc.returncode}) past the "
            f"5s bridge connect timeout — fire-and-forget bridge.start "
            f"chain is leaking the rejection. The .catch(() => {{}}) on "
            f"the top-level chain in index.ts must absorb "
            f"DaemonUnreachableError."
        )

        call_req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "memory_recall",
                "arguments": {"cue": "no-daemon test"},
            },
        }
        wrapper_proc.stdin.write((json.dumps(call_req) + "\n").encode("utf-8"))
        wrapper_proc.stdin.flush()
        import select as _select
        deadline = time.monotonic() + 12.0
        call_line = b""
        while time.monotonic() < deadline:
            readable, _, _ = _select.select([wrapper_proc.stdout], [], [], 0.5)
            if readable:
                call_line = wrapper_proc.stdout.readline()
                break
        assert call_line, "wrapper did not respond to tools/call within 12s"
        call_resp = json.loads(call_line.decode("utf-8"))
        assert "result" in call_resp, f"tools/call missing result: {call_resp}"
        result = call_resp["result"]
        is_error = result.get("isError") is True
        content_text = ""
        if isinstance(result.get("content"), list) and result["content"]:
            content_text = result["content"][0].get("text", "") or ""
        assert is_error or content_text, (
            f"tools/call returned neither an error nor usable content when "
            f"daemon is missing — silent-fail invariant violated. result={result}"
        )

        time.sleep(1.0)
        after = _count_iai_mcp_processes()
        daemon_delta = after["daemon"] - daemon_baseline
        assert daemon_delta == 0, (
            f"REGRESSION: wrapper spawned {daemon_delta} new iai_mcp.daemon "
            f"process(es) (baseline={daemon_baseline}, after={after['daemon']}). "
            f"Wrappers MUST NOT spawn the daemon — the spawn-fallback "
            f"chain in bridge.ts has been re-introduced."
        )
        core_delta = after["core"] - core_baseline
        assert core_delta == 0, (
            f"wrapper spawned {core_delta} iai_mcp.core process(es) "
            f"(baseline={core_baseline}, after={after['core']})"
        )
    finally:
        if wrapper_proc.poll() is None:
            try:
                wrapper_proc.terminate()
                wrapper_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                wrapper_proc.kill()
        _kill_test_daemons(sock_path)
        time.sleep(0.3)
        try:
            sock_path.unlink()
        except OSError:
            pass
        sock_dir_ctx.cleanup()


def test_start_succeeds_with_warm_daemon_no_extra_spawn(built_wrapper, tmp_path):
    sock_dir_ctx = tempfile.TemporaryDirectory(prefix="iai-sock-")
    sock_dir = Path(sock_dir_ctx.name)
    sock_path = sock_dir / "d.sock"
    store_dir = tmp_path / "store"
    store_dir.mkdir(parents=True, exist_ok=True)
    assert not sock_path.exists()

    daemon_proc = _spawn_daemon_in_background(sock_path, store_dir)
    try:
        assert _wait_for_daemon_socket(sock_path, timeout_sec=30.0), (
            f"daemon did not bind socket {sock_path} within 30s"
        )

        baseline = _count_iai_mcp_processes()
        daemon_baseline = baseline["daemon"]
        core_baseline = baseline["core"]

        env_overrides = {
            "IAI_DAEMON_SOCKET_PATH": str(sock_path),
            "IAI_MCP_STORE": str(store_dir),
        }
        wrapper_proc = _spawn_wrapper(built_wrapper, env_overrides)
        try:
            init_resp = _initialize(wrapper_proc, rpc_id=1)
            assert "result" in init_resp, f"initialize failed: {init_resp}"

            elapsed, recall_resp = _call_memory_recall(
                wrapper_proc, cue="phase 7.1 warm-daemon test",
                rpc_id=2, timeout_sec=10.0,
            )
            assert "result" in recall_resp or "error" in recall_resp, recall_resp

            assert elapsed < 2.0, (
                f"warm-daemon memory_recall took {elapsed:.2f}s, exceeds "
                f"2.0s safety budget"
            )

            time.sleep(0.5)
            after = _count_iai_mcp_processes()

            daemon_delta = after["daemon"] - daemon_baseline
            assert daemon_delta == 0, (
                f"REGRESSION: wrapper spawned a second daemon during boot "
                f"(baseline={daemon_baseline}, after={after['daemon']}, "
                f"delta={daemon_delta}). Wrappers MUST be pure "
                f"connectors."
            )
            core_delta = after["core"] - core_baseline
            assert core_delta == 0, (
                f"wrapper spawned iai_mcp.core (delta={core_delta})"
            )
        finally:
            try:
                wrapper_proc.terminate()
                wrapper_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                wrapper_proc.kill()
    finally:
        try:
            daemon_proc.terminate()
            daemon_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            daemon_proc.kill()
        _kill_test_daemons(sock_path)
        time.sleep(0.3)
        try:
            sock_path.unlink()
        except OSError:
            pass
        sock_dir_ctx.cleanup()
