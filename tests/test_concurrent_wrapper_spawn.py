from __future__ import annotations

import json
import os
import platform
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

REPO = Path(__file__).resolve().parent.parent
WRAPPER = REPO / "mcp-wrapper"

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="LaunchAgent + launchctl is macOS-only",
)


@pytest.fixture(scope="module")
def built_wrapper() -> Path:
    if not (WRAPPER / "node_modules").exists():
        subprocess.run(["npm", "install"], cwd=WRAPPER, check=True)
    subprocess.run(["npm", "run", "build"], cwd=WRAPPER, check=True)
    dist = WRAPPER / "dist" / "index.js"
    assert dist.exists(), "npm run build should have produced dist/index.js"
    return dist


@pytest.fixture
def test_launchagent(tmp_path):
    if os.environ.get("IAI_MCP_SKIP_LAUNCHCTL_TESTS") == "1":
        pytest.skip("IAI_MCP_SKIP_LAUNCHCTL_TESTS=1")

    sock_dir = Path(f"/tmp/iai-cspawn-{os.getpid()}-{id(tmp_path) & 0xFFFFFF:x}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"
    if sock_path.exists():
        sock_path.unlink()

    store_dir = tmp_path / "store"
    store_dir.mkdir(parents=True, exist_ok=True)

    label = f"com.iai-mcp.daemon.test-{os.getpid()}-{id(tmp_path) & 0xFFFFFF:x}"

    plist_dir = tmp_path / "Library" / "LaunchAgents"
    plist_dir.mkdir(parents=True, exist_ok=True)
    plist_path = plist_dir / f"{label}.plist"

    template = (REPO / "scripts" / "com.iai-mcp.daemon.plist.template").read_text()
    label_old_xml = "<string>com.iai-mcp.daemon</string>"
    label_new_xml = f"<string>{label}</string>"
    if template.count(label_old_xml) != 1:
        pytest.fail(
            f"plist template invariant broken: expected exactly one "
            f"<string>com.iai-mcp.daemon</string> occurrence (the "
            f"<key>Label</key> binding); found "
            f"{template.count(label_old_xml)}",
        )
    rendered = (
        template
        .replace("{PYTHON_PATH}", sys.executable)
        .replace("{HOME}", str(Path.home()))
        .replace(label_old_xml, label_new_xml)
        .replace(
            f"{Path.home()}/.iai-mcp/.daemon.sock",
            str(sock_path),
        )
        .replace(
            "<key>IAI_MCP_LAUNCHD_MANAGED</key>\n    <string>1</string>",
            "<key>IAI_MCP_LAUNCHD_MANAGED</key>\n    <string>1</string>\n"
            f"    <key>IAI_DAEMON_SOCKET_PATH</key>\n    <string>{sock_path}</string>\n"
            f"    <key>PYTHONPATH</key>\n    <string>{REPO / 'src'}</string>\n"
            f"    <key>IAI_MCP_STORE</key>\n    <string>{store_dir}</string>\n"
            "    <key>IAI_MCP_CRYPTO_PASSPHRASE</key>\n    <string>test-cspawn-key</string>",
        )
    )
    plist_path.write_text(rendered)

    subprocess.run(
        ["launchctl", "unload", "-w", str(plist_path)],
        capture_output=True, check=False,
    )

    res = subprocess.run(
        ["launchctl", "load", "-w", str(plist_path)],
        capture_output=True, text=True, check=False,
    )
    if res.returncode != 0:
        pytest.skip(f"launchctl load failed (rc={res.returncode}): {res.stderr.strip()}")

    list_res = subprocess.run(
        ["launchctl", "list"], capture_output=True, text=True, check=False,
    )
    if label not in list_res.stdout:
        subprocess.run(
            ["launchctl", "unload", "-w", str(plist_path)],
            capture_output=True, check=False,
        )
        pytest.fail(
            f"LaunchAgent {label!r} not present in `launchctl list` after load",
        )

    env = {
        **os.environ,
        "IAI_MCP_PYTHON": sys.executable,
        "PYTHONPATH": str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "IAI_DAEMON_SOCKET_PATH": str(sock_path),
        "IAI_MCP_STORE": str(store_dir),
        "IAI_MCP_CRYPTO_PASSPHRASE": "test-cspawn-key",
    }

    try:
        yield sock_path, plist_path, label, env, store_dir
    finally:
        subprocess.run(
            ["launchctl", "unload", "-w", str(plist_path)],
            capture_output=True, check=False,
        )
        target = str(sock_path)
        term_pids: set[int] = set()
        lsof_res = subprocess.run(
            ["lsof", "-U", "-F", "pn"],
            capture_output=True, text=True, check=False,
        )
        current: int | None = None
        for line in lsof_res.stdout.splitlines():
            if line.startswith("p"):
                try:
                    current = int(line[1:])
                except ValueError:
                    current = None
            elif line.startswith("n") and current is not None and line[1:] == target:
                term_pids.add(current)
        for pid in list(term_pids):
            try:
                cl = " ".join(psutil.Process(pid).cmdline())
                if "iai_mcp.daemon" not in cl:
                    term_pids.discard(pid)
                    continue
                psutil.Process(pid).send_signal(signal.SIGTERM)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                term_pids.discard(pid)
        time.sleep(0.5)
        for pid in term_pids:
            try:
                if psutil.pid_exists(pid):
                    cl = " ".join(psutil.Process(pid).cmdline())
                    if "iai_mcp.daemon" in cl:
                        psutil.Process(pid).kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        try:
            sock_path.unlink()
        except (FileNotFoundError, OSError):
            pass
        try:
            sock_dir.rmdir()
        except OSError:
            pass


def _spawn_wrapper_send_initialize(
    built_wrapper: Path, env: dict,
) -> subprocess.Popen:
    proc = subprocess.Popen(
        ["node", str(built_wrapper)],
        cwd=str(REPO),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "concurrent-spawn-test", "version": "0.0"},
        },
    }
    try:
        assert proc.stdin is not None
        proc.stdin.write((json.dumps(init_req) + "\n").encode("utf-8"))
        proc.stdin.flush()
    except BrokenPipeError:
        pass
    return proc


def _read_initialize_response(
    proc: subprocess.Popen, timeout_sec: float = 2.0,
) -> dict | None:
    if proc.stdout is None:
        return None
    try:
        ready, _, _ = select.select([proc.stdout], [], [], timeout_sec)
        if not ready:
            return None
        line = proc.stdout.readline()
        if not line:
            return None
        return json.loads(line.decode("utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _count_daemons_for_socket(sock_path: Path, store_dir: Path) -> int:
    store_str = str(store_dir.resolve())
    count = 0
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cl = " ".join(proc.info.get("cmdline") or [])
            if "iai_mcp.daemon" not in cl:
                continue
            for f in proc.open_files():
                if str(Path(f.path).resolve()).startswith(store_str):
                    count += 1
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return count


def _count_binders(sock_path: Path) -> int:
    res = subprocess.run(
        ["lsof", "-U", "-F", "pn"],
        capture_output=True, text=True, check=False,
    )
    pids: set[int] = set()
    current: int | None = None
    target = str(sock_path)
    for line in res.stdout.splitlines():
        if line.startswith("p"):
            try:
                current = int(line[1:])
            except ValueError:
                current = None
        elif line.startswith("n") and current is not None and line[1:] == target:
            pids.add(current)
    return len(pids)


def test_5_concurrent_wrapper_cold_starts_yield_singleton(
    built_wrapper, test_launchagent,
):
    sock_path, plist_path, label, env, store_dir = test_launchagent

    initial_daemon_count = _count_daemons_for_socket(sock_path, store_dir)
    assert initial_daemon_count <= 1, (
        f"expected <= 1 daemon before test, found {initial_daemon_count} "
        f"(stale daemons from earlier test? cleanup leak?)"
    )

    procs: list[subprocess.Popen] = []
    stagger_intervals = [0.0, 0.05, 0.05, 0.05, 0.05]
    for delay in stagger_intervals:
        if delay > 0:
            time.sleep(delay)
        procs.append(_spawn_wrapper_send_initialize(built_wrapper, env))

    time.sleep(15)

    init_responses: list[dict | None] = [
        _read_initialize_response(p, timeout_sec=2.0) for p in procs
    ]

    daemon_count = _count_daemons_for_socket(sock_path, store_dir)
    binder_count = _count_binders(sock_path)

    for proc in procs:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

    assert daemon_count == 1, (
        f"singleton invariant violated: {daemon_count} daemons bound to "
        f"{sock_path} after 5 concurrent wrapper cold-starts. "
        f"contract: launchd handles the spawn-once; all wrappers join "
        f"the same daemon. Pre-Phase-7.1 baseline reproduces 2-5 daemons "
        f"via TOCTOU race in bridge.ts spawn-fallback."
    )
    assert binder_count <= 1, (
        f"lsof reports {binder_count} binders for {sock_path}; "
        f"expected <= 1 (singleton)"
    )
    success_count = sum(
        1 for r in init_responses if r is not None and "result" in r
    )
    assert success_count == 5, (
        f"only {success_count}/5 wrappers received successful initialize "
        f"response. Responses: {init_responses}"
    )
