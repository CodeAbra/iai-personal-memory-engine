"""Hermetic reproduction + hardening tests for `tools_exec_client._connect`.

Two stub binaries stand in for `iai_tools_execd`: one exits without ever
binding a socket (the never-comes-up case), one binds after an injected
delay (the slow-but-healthy case). Most tests pre-create a stale socket
file the way a crashed daemon leaves one -- bound, listened, closed,
never unlinked; a few instead leave no socket file at all, the shape a
daemon that unlinked cleanly on its own idle exit leaves behind. Neither
scenario spawns a real daemon process or touches the live
`~/.iai-tools/` store.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _execd_bin() -> str:
    """Resolves the built `iai_tools_execd` binary for tests.

    Prefers the `IAI_TOOLS_EXECD_BIN` override CI always sets; falls back to
    a cargo target-dir guess for local dev use, `.exe`-suffixed on win32 --
    without that suffix this guess never matches on Windows and the caller
    self-skips, which would make a windows-latest job falsely green.
    """
    env = os.environ.get("IAI_TOOLS_EXECD_BIN")
    if env:
        return env
    bin_name = "iai_tools_execd.exe" if sys.platform == "win32" else "iai_tools_execd"
    for profile in ("release", "debug"):
        candidate = REPO / "target" / profile / bin_name
        if candidate.exists():
            return str(candidate)
    pytest.skip("iai_tools_execd not built; run: cargo build -p iai_mcp_tools_core --bins")


_STUB_NEVER_BINDS = "import sys\nsys.exit(0)\n"

_STUB_WRITES_STDERR_THEN_EXITS = (
    'import sys\n'
    'sys.stderr.write("stub startup failure\\n")\n'
    'sys.stderr.flush()\n'
    'sys.exit(1)\n'
)

_STUB_BINDS_AFTER_DELAY = """
import os
import socket
import time

delay = float(os.environ.get("STUB_BIND_DELAY_SECS", "0.2"))
hold = float(os.environ.get("STUB_HOLD_SECS", "1.0"))
time.sleep(delay)

path = os.environ["IAI_TOOLS_SOCKET_PATH"]
try:
    os.remove(path)
except FileNotFoundError:
    pass

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.bind(path)
sock.listen(1)
time.sleep(hold)
"""


def _write_stub(path: Path, body: str) -> Path:
    path.write_text(f"#!/usr/bin/env python3\n{body}\n")
    path.chmod(0o755)
    return path


def _make_stale_socket(sock_path: Path) -> None:
    """Leaves a bound-and-listened-then-closed socket at `sock_path`,
    unlinked -- mirrors `std::process::exit(0)` skipping cleanup."""
    try:
        sock_path.unlink()
    except FileNotFoundError:
        pass
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(sock_path))
    sock.listen(1)
    sock.close()
    assert sock_path.exists(), "stale socket construction must leave the path behind"


@pytest.fixture
def exec_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "iai-tools-root"
    # AF_UNIX's sun_path is capped at ~104 bytes on macOS -- pytest's own
    # tmp_path nests several directories deep and overflows that, so the
    # socket lives in its own short-path directory directly under /tmp.
    sock_dir = Path(tempfile.mkdtemp(prefix="iai-tools-", dir="/tmp"))
    sock_path = sock_dir / ".execd.sock"
    monkeypatch.setenv("IAI_TOOLS_ROOT", str(root))
    monkeypatch.setenv("IAI_TOOLS_SOCKET_PATH", str(sock_path))

    spawned_pids: list[int] = []
    real_popen = subprocess.Popen

    def _tracking_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        spawned_pids.append(proc.pid)
        return proc

    monkeypatch.setattr(subprocess, "Popen", _tracking_popen)

    try:
        yield {"root": root, "sock_path": sock_path, "spawned_pids": spawned_pids}
    finally:
        for pid in spawned_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        shutil.rmtree(sock_dir, ignore_errors=True)


def test_stale_socket_with_never_binding_respawn_still_fails_loud(
    exec_env: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iai_mcp import tools_exec_client

    sock_path = exec_env["sock_path"]
    _make_stale_socket(sock_path)

    stub = _write_stub(tmp_path / "stub_never_binds.py", _STUB_NEVER_BINDS)
    monkeypatch.setenv("IAI_TOOLS_EXECD_BIN", str(stub))

    # Harness sanity -- distinguishes a real refutation from a broken fixture.
    assert sock_path == tools_exec_client._socket_path()
    assert sock_path.exists()
    assert os.access(stub, os.X_OK)

    with pytest.raises(ConnectionRefusedError):
        tools_exec_client._connect(timeout=0.5)


def test_stale_socket_recovers_once_respawn_binds_within_deadline(
    exec_env: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iai_mcp import tools_exec_client

    sock_path = exec_env["sock_path"]
    stub = _write_stub(tmp_path / "stub_binds_after_delay.py", _STUB_BINDS_AFTER_DELAY)
    monkeypatch.setenv("IAI_TOOLS_EXECD_BIN", str(stub))
    monkeypatch.setenv("STUB_BIND_DELAY_SECS", "0.2")
    monkeypatch.setenv("STUB_HOLD_SECS", "1.0")

    # Repeated across runs -- proves the recovery is stable, not a fluke.
    for _ in range(10):
        _make_stale_socket(sock_path)
        sock = tools_exec_client._connect(timeout=1.0)
        sock.close()


def test_connect_retry_spawns_daemon_exactly_once(
    exec_env: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iai_mcp import tools_exec_client

    sock_path = exec_env["sock_path"]
    _make_stale_socket(sock_path)

    stub = _write_stub(tmp_path / "stub_never_binds_spawn_count.py", _STUB_NEVER_BINDS)
    monkeypatch.setenv("IAI_TOOLS_EXECD_BIN", str(stub))

    calls = {"count": 0}
    real_spawn = tools_exec_client._spawn_daemon

    def _counting_spawn() -> None:
        calls["count"] += 1
        real_spawn()

    monkeypatch.setattr(tools_exec_client, "_spawn_daemon", _counting_spawn)

    with pytest.raises(ConnectionRefusedError):
        tools_exec_client._connect(timeout=0.3)

    assert calls["count"] == 1, "the retry loop must never re-spawn -- one spawn per _connect() call"


def test_spawn_daemon_captures_child_stderr_to_log_file(
    exec_env: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves the redirect actually receives child output -- a log file
    that merely exists (e.g. because the redirect silently failed and the
    file was only ever opened) would also satisfy a weaker existence-only
    check."""
    from iai_mcp import tools_exec_client

    stub = _write_stub(tmp_path / "stub_writes_stderr.py", _STUB_WRITES_STDERR_THEN_EXITS)
    monkeypatch.setenv("IAI_TOOLS_EXECD_BIN", str(stub))

    tools_exec_client._spawn_daemon()

    log_path = tools_exec_client._execd_log_path()
    deadline = time.monotonic() + 2.0
    content = ""
    while time.monotonic() < deadline:
        if log_path.exists():
            content = log_path.read_text(encoding="utf-8")
            if content:
                break
        time.sleep(0.02)

    assert "stub startup failure" in content, (
        f"the child's stderr must reach the log file, got: {content!r}"
    )


def test_connect_fails_loud_after_deadline_when_daemon_never_binds(
    exec_env: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iai_mcp import tools_exec_client

    sock_path = exec_env["sock_path"]
    _make_stale_socket(sock_path)

    stub = _write_stub(tmp_path / "stub_never_binds_timed.py", _STUB_NEVER_BINDS)
    monkeypatch.setenv("IAI_TOOLS_EXECD_BIN", str(stub))

    started = time.monotonic()
    with pytest.raises(ConnectionRefusedError):
        tools_exec_client._connect(timeout=0.5)
    elapsed = time.monotonic() - started
    margin = 1.0
    assert 0.5 <= elapsed <= 0.5 + margin, (
        f"expected the raise only after the bounded deadline, took {elapsed}s"
    )


def test_missing_socket_fails_loud_with_file_not_found_after_deadline(
    exec_env: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No socket file exists at all (the now-dominant shape once a clean
    idle-exit unlinks its own socket) -- the retry loop must still bound
    the wait and raise the specific exception a fresh `connect()` against
    an absent path actually produces, not merely some `OSError`."""
    from iai_mcp import tools_exec_client

    sock_path = exec_env["sock_path"]
    assert not sock_path.exists(), "harness sanity: no stale socket must be present"

    stub = _write_stub(tmp_path / "stub_never_binds_missing_socket.py", _STUB_NEVER_BINDS)
    monkeypatch.setenv("IAI_TOOLS_EXECD_BIN", str(stub))

    started = time.monotonic()
    with pytest.raises(FileNotFoundError):
        tools_exec_client._connect(timeout=0.5)
    elapsed = time.monotonic() - started
    margin = 1.0
    assert 0.5 <= elapsed <= 0.5 + margin, (
        f"expected the raise only after the bounded deadline, took {elapsed}s"
    )


def test_missing_socket_recovers_once_respawn_binds_within_deadline(
    exec_env: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mirrors the stale-socket recovery test, but for a socket path that
    never existed at all -- the retry loop must treat the initial
    `FileNotFoundError` as recoverable and keep polling until the
    freshly-spawned daemon binds, not raise immediately."""
    from iai_mcp import tools_exec_client

    sock_path = exec_env["sock_path"]
    assert not sock_path.exists(), "harness sanity: no stale socket must be present"

    stub = _write_stub(tmp_path / "stub_binds_after_delay_missing_socket.py", _STUB_BINDS_AFTER_DELAY)
    monkeypatch.setenv("IAI_TOOLS_EXECD_BIN", str(stub))
    monkeypatch.setenv("STUB_BIND_DELAY_SECS", "0.2")
    monkeypatch.setenv("STUB_HOLD_SECS", "1.0")

    sock = tools_exec_client._connect(timeout=1.0)
    sock.close()


def test_daemon_unlinks_socket_on_idle_exit(
    exec_env: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iai_mcp import tools_exec_client

    bin_path = _execd_bin()
    sock_path = exec_env["sock_path"]
    idle_secs = 1.0
    monkeypatch.setenv("IAI_TOOLS_EXECD_BIN", bin_path)
    monkeypatch.setenv("IAI_TOOLS_IDLE_SECS", str(idle_secs))

    result = tools_exec_client.run("echo", ["idle-unlink-probe"])
    assert result["run_id"]
    assert sock_path.exists(), "daemon must have bound its socket before idle-exit"

    # Sleep past the idle window WITHOUT touching the socket -- any probe
    # connect would itself count as activity and reset the idle clock.
    time.sleep(idle_secs + 3.0)
    assert not sock_path.exists(), "daemon must unlink its socket on idle self-exit"


@pytest.mark.perf
def test_execd_spawn_latency_stays_under_the_connect_deadline() -> None:
    from iai_mcp import tools_exec_client

    bin_path = _execd_bin()
    samples: list[float] = []
    iterations = 20

    for i in range(iterations):
        root = Path(tempfile.mkdtemp(prefix="iai-tools-latency-root-"))
        sock_dir = Path(tempfile.mkdtemp(prefix="iai-tools-lat-", dir="/tmp"))
        sock_path = sock_dir / ".execd.sock"
        env = os.environ.copy()
        env["IAI_TOOLS_ROOT"] = str(root)
        env["IAI_TOOLS_SOCKET_PATH"] = str(sock_path)
        env["IAI_TOOLS_IDLE_SECS"] = "1.0"

        started = time.monotonic()
        proc = subprocess.Popen(
            [bin_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        try:
            connect_deadline = time.monotonic() + 10.0
            connected = False
            while time.monotonic() < connect_deadline:
                probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    probe.connect(str(sock_path))
                    connected = True
                    probe.close()
                    break
                except (ConnectionRefusedError, FileNotFoundError):
                    probe.close()
                    time.sleep(0.005)
            elapsed = time.monotonic() - started
            assert connected, f"iteration {i}: daemon never became reachable within 10s"
            samples.append(elapsed)
        finally:
            proc.kill()
            proc.wait()
            shutil.rmtree(root, ignore_errors=True)
            shutil.rmtree(sock_dir, ignore_errors=True)

    samples.sort()
    worst = samples[-1]

    print(f"spawn latency samples (s): {samples}")
    print(f"max={worst:.4f}s n={len(samples)}")

    # A near-zero max means every iteration hit a fast path (e.g. a shared
    # socket path with a still-live daemon) instead of a real cold spawn.
    # At this sample count a percentile assertion would just re-check the
    # max under a different name, so only max is asserted.
    assert worst >= 0.001, "suspiciously fast measurement -- fixture likely reused a live daemon"
    assert worst < tools_exec_client._CONNECT_TIMEOUT_SECS, (
        f"max spawn latency {worst:.3f}s exceeded the "
        f"{tools_exec_client._CONNECT_TIMEOUT_SECS}s connect deadline"
    )
