from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WRAPPER = REPO / "mcp-wrapper"

def _wrapper_ready() -> bool:
    return (WRAPPER / "dist" / "index.js").exists()

@pytest.fixture(scope="module")
def built_wrapper() -> Path:
    if not _wrapper_ready():
        if not (WRAPPER / "node_modules").exists():
            subprocess.run(["npm", "install"], cwd=WRAPPER, check=True)
        subprocess.run(["npm", "run", "build"], cwd=WRAPPER, check=True)
    dist = WRAPPER / "dist" / "index.js"
    if not dist.exists():
        pytest.skip(f"mcp-wrapper not built; missing {dist}")
    return dist

_FAKE_DAEMON_SCRIPT = r"""
import json, os, socket, sys, threading

sock_path = sys.argv[1]
try:
    os.unlink(sock_path)
except FileNotFoundError:
    pass

srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(sock_path)
srv.listen(8)

state_lock = threading.Lock()
live_conns = []

sys.stdout.write("BOUND\n")
sys.stdout.flush()

def serve(conn):
    buf = b""
    try:
        while True:
            data = conn.recv(65536)
            if not data:
                break
            buf += data
            while b"\n" in buf:
                line, _, buf = buf.partition(b"\n")
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                rid = req.get("id")
                method = req.get("method", "")
                resp = {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {
                        "ok": True,
                        "method": method,
                        "fake_daemon": True,
                    },
                }
                try:
                    conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
                except Exception:
                    return
    except Exception:
        pass
    finally:
        with state_lock:
            try:
                live_conns.remove(conn)
            except ValueError:
                pass
        try:
            conn.close()
        except Exception:
            pass

def stdin_reader():
    for raw in sys.stdin:
        cmd = raw.strip()
        if cmd == "DROP":
            with state_lock:
                victims = list(live_conns)
                live_conns.clear()
            for c in victims:
                try:
                    c.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    c.close()
                except Exception:
                    pass
            sys.stdout.write("DROPPED\n")
            sys.stdout.flush()
        elif cmd == "QUIT":
            break

threading.Thread(target=stdin_reader, daemon=True).start()

while True:
    try:
        conn, _ = srv.accept()
    except Exception:
        break
    with state_lock:
        live_conns.append(conn)
    threading.Thread(target=serve, args=(conn,), daemon=True).start()
"""

def _spawn_fake_daemon(sock_path: Path) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-c", _FAKE_DAEMON_SCRIPT, str(sock_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 10.0
    assert proc.stdout is not None
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if line.strip() == b"BOUND":
            return proc
        if proc.poll() is not None:
            err = proc.stderr.read() if proc.stderr is not None else b""
            raise RuntimeError(
                f"fake daemon exited before binding: {err.decode(errors='replace')}"
            )
    proc.kill()
    raise RuntimeError("fake daemon did not bind within 10s")

def _drop_fake_daemon_conn(proc: subprocess.Popen) -> None:
    assert proc.stdin is not None
    proc.stdin.write(b"DROP\n")
    proc.stdin.flush()
    assert proc.stdout is not None
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if line.strip() == b"DROPPED":
            return
    raise RuntimeError("fake daemon did not ack DROP within 5s")

@pytest.fixture
def fake_daemon():
    sock_dir = Path(f"/tmp/iai-mcp-disconnect-{os.getpid()}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"

    proc = _spawn_fake_daemon(sock_path)

    def drop_connections() -> None:
        _drop_fake_daemon_conn(proc)

    yield {"path": sock_path, "proc": proc, "drop_connections": drop_connections}

    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        sock_path.unlink()
    except OSError:
        pass
    try:
        shutil.rmtree(sock_dir, ignore_errors=True)
    except OSError:
        pass

def _spawn_wrapper(
    built_wrapper: Path,
    daemon_sock: Path,
    reconnect_delay_ms: int = 1000,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["IAI_MCP_PYTHON"] = sys.executable
    tmpdir = tempfile.mkdtemp(prefix="iai-mcp-disconnect-test-")
    env["IAI_MCP_STORE"] = tmpdir
    env["IAI_DAEMON_SOCKET_PATH"] = str(daemon_sock)
    env["IAI_MCP_RECONNECT_TEST_DELAY_MS"] = str(reconnect_delay_ms)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        ["node", str(built_wrapper)],
        cwd=str(REPO),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

def _mcp_call(
    proc: subprocess.Popen,
    method: str,
    params: dict,
    rpc_id: int = 99,
    timeout_s: float = 10.0,
) -> dict:
    req = {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params}
    assert proc.stdin is not None
    proc.stdin.write((json.dumps(req) + "\n").encode())
    proc.stdin.flush()
    assert proc.stdout is not None
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("wrapper closed stdout before replying")
        try:
            return json.loads(line.decode())
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"timeout waiting for {method} response")

def _initialize(proc: subprocess.Popen, rpc_id: int = 1) -> None:
    resp = _mcp_call(
        proc,
        "initialize",
        {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "iai-mcp-disconnect-test", "version": "0.1.0"},
        },
        rpc_id,
    )
    assert "result" in resp, f"initialize failed: {resp}"
    assert proc.stdin is not None
    note = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    proc.stdin.write((json.dumps(note) + "\n").encode())
    proc.stdin.flush()

def test_call_during_socket_death_resolves_after_reconnect(
    built_wrapper: Path,
    fake_daemon: dict,
) -> None:
    sock_path = fake_daemon["path"]
    wrapper = _spawn_wrapper(built_wrapper, sock_path)
    try:
        _initialize(wrapper)

        r1 = _mcp_call(
            wrapper,
            "tools/call",
            {"name": "topology", "arguments": {}},
            rpc_id=2,
        )
        err_str_1 = json.dumps(r1)
        assert "daemon_unreachable" not in err_str_1, (
            f"baseline call already broken: {r1}"
        )

        fake_daemon["drop_connections"]()

        time.sleep(0.05)

        r2 = _mcp_call(
            wrapper,
            "tools/call",
            {"name": "topology", "arguments": {}},
            rpc_id=3,
            timeout_s=20.0,
        )
        err_str_2 = json.dumps(r2)
        assert "daemon_unreachable" not in err_str_2, (
            f"V3-05 race not closed: {r2}"
        )
    finally:
        try:
            wrapper.terminate()
            wrapper.wait(timeout=5)
        except subprocess.TimeoutExpired:
            wrapper.kill()
