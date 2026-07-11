from __future__ import annotations

import multiprocessing as mp
import os
import platform
import socket
import subprocess
import time
from pathlib import Path

import pytest


# No integration test for two daemons binding one store: the single-owner
# lifecycle lock makes that state impossible to construct (the second daemon
# fails the lock and exits before it can bind). The detection path is covered
# by the _extract_binder_pids and check_g unit tests below, which fabricate
# lsof output and bind real sockets directly without spawning two daemons.


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX AF_UNIX required (lsof -U + multiprocessing socket binders)",
)


def test_extract_binder_pids_parses_lsof_output():
    from iai_mcp.doctor import _extract_binder_pids

    target = Path("/tmp/iai-test/d.sock")
    lsof_output = "\n".join([
        "p12345",
        f"n{target}",
        "p67890",
        f"n{target}",
        "p99999",
        "n/tmp/other-app/socket",
    ])

    pids = _extract_binder_pids(lsof_output, target)

    assert pids == {12345, 67890}, f"expected {{12345, 67890}}, got {pids}"


def test_extract_binder_pids_skips_unrelated_sockets():
    from iai_mcp.doctor import _extract_binder_pids

    target = Path("/tmp/iai-test/d.sock")
    lsof_output = "\n".join([
        "p1001",
        "n/var/run/some-other-daemon.sock",
        "p2002",
        f"n{target}",
        "p3003",
        "n/tmp/X11-unix/X0",
        "p4004",
        f"n{target}",
        "n/some/extra/name/for/p4004",
    ])

    pids = _extract_binder_pids(lsof_output, target)

    assert pids == {2002, 4004}, f"expected {{2002, 4004}}, got {pids}"


def test_extract_binder_pids_handles_empty_output():
    from iai_mcp.doctor import _extract_binder_pids

    target = Path("/tmp/anywhere.sock")
    assert _extract_binder_pids("", target) == set()
    assert _extract_binder_pids("\n\n\n", target) == set()
    assert _extract_binder_pids("p123\nXgarbage\np\n", target) == set()


def test_extract_binder_pids_ss_parses_ss_output():
    from iai_mcp.doctor import _extract_binder_pids_ss

    target = Path("/tmp/iai-test/d.sock")
    ss_output = "\n".join([
        f"u_str LISTEN 0 5 {target} 12345 * 0 users:((\"python3\",pid=12345,fd=3))",
        f"u_str LISTEN 0 5 {target} 67890 * 0 users:((\"python3\",pid=67890,fd=4))",
        "u_str LISTEN 0 5 /tmp/other.sock 99999 * 0 users:((\"nginx\",pid=99999,fd=8))",
    ])

    pids = _extract_binder_pids_ss(ss_output, target)

    assert pids == {12345, 67890}, f"expected {{12345, 67890}}, got {pids}"


def test_extract_binder_pids_ss_handles_empty_output():
    from iai_mcp.doctor import _extract_binder_pids_ss

    target = Path("/tmp/anywhere.sock")
    assert _extract_binder_pids_ss("", target) == set()
    assert _extract_binder_pids_ss("\n\n\n", target) == set()
    assert _extract_binder_pids_ss("u_str LISTEN 0 5 /tmp/other.sock 1 * 0 users:((\"x\",pid=1,fd=0))", target) == set()


def test_extract_binder_pids_ss_exact_token_no_false_positive():
    # A line whose path is a *superset* of the target (e.g. .daemon.sock-old,
    # .daemon.sock2) must NOT match — substring containment would trigger on these.
    from iai_mcp.doctor import _extract_binder_pids_ss

    target = Path("/home/u/.iai-mcp/.daemon.sock")
    decoy_lines = "\n".join([
        f'u_str LISTEN 0 5 {target}-old 9001 * 0 users:(("x",pid=9001,fd=3))',
        f'u_str LISTEN 0 5 {target}2 9002 * 0 users:(("x",pid=9002,fd=4))',
        f'u_str LISTEN 0 5 {target}.bak 9003 * 0 users:(("x",pid=9003,fd=5))',
    ])
    assert _extract_binder_pids_ss(decoy_lines, target) == set(), (
        "paths that contain the target as a substring must not match"
    )

    # The exact target path DOES match.
    exact_line = (
        f'u_str LISTEN 0 5 {target} 8000 * 0 users:(("python3",pid=8000,fd=3))'
    )
    assert _extract_binder_pids_ss(exact_line, target) == {8000}, (
        "the exact target path must match"
    )


@pytest.fixture
def short_socket_path(tmp_path, monkeypatch):
    sock_dir = Path(f"/tmp/iai-mb-{os.getpid()}-{id(tmp_path)}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(sock_path))
    try:
        yield sock_path
    finally:
        try:
            if sock_path.exists():
                sock_path.unlink()
        except OSError:
            pass
        try:
            sock_dir.rmdir()
        except OSError:
            pass


def test_check_g_no_socket_skips(short_socket_path, monkeypatch):
    from iai_mcp.doctor import check_g_no_dup_binders

    assert not short_socket_path.exists()

    result = check_g_no_dup_binders()

    assert result.passed is True
    assert "no socket file" in result.detail


def _bind_socket_worker(sock_path_str: str, ready_event: mp.Event, exit_event: mp.Event) -> None:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.bind(sock_path_str)
        s.listen(5)
        ready_event.set()
        exit_event.wait(timeout=30)
    finally:
        try:
            s.close()
        except OSError:
            pass


def test_check_g_single_binder_passes(short_socket_path):
    from iai_mcp.doctor import check_g_no_dup_binders

    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    exit_signal = ctx.Event()
    worker = ctx.Process(
        target=_bind_socket_worker,
        args=(str(short_socket_path), ready, exit_signal),
    )
    worker.start()
    try:
        assert ready.wait(timeout=10), "binder worker never signaled ready"
        time.sleep(0.2)

        result = check_g_no_dup_binders()

        assert result.passed is True, (
            f"single-binder scenario should PASS; got detail={result.detail!r}"
        )
        assert "1 binder" in result.detail, f"unexpected detail: {result.detail!r}"
    finally:
        exit_signal.set()
        worker.join(timeout=5)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=2)


def test_check_g_two_binders_fails(short_socket_path):
    from iai_mcp.doctor import (
        _extract_binder_pids,
        _extract_binder_pids_ss,
        check_g_no_dup_binders,
    )

    ctx = mp.get_context("spawn")

    ready1 = ctx.Event()
    exit1 = ctx.Event()
    w1 = ctx.Process(
        target=_bind_socket_worker,
        args=(str(short_socket_path), ready1, exit1),
    )
    w1.start()

    ready2 = ctx.Event()
    exit2 = ctx.Event()
    w2 = None
    try:
        assert ready1.wait(timeout=10), "worker 1 never signaled ready"
        try:
            short_socket_path.unlink()
        except OSError:
            pass
        w2 = ctx.Process(
            target=_bind_socket_worker,
            args=(str(short_socket_path), ready2, exit2),
        )
        w2.start()
        assert ready2.wait(timeout=10), "worker 2 never signaled ready"
        time.sleep(0.3)

        if platform.system() == "Linux":
            ss_out = subprocess.run(
                ["ss", "-lxp"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout
            binder_pids = _extract_binder_pids_ss(ss_out, short_socket_path)
        else:
            lsof_out = subprocess.run(
                ["lsof", "-U", "-F", "pn"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout
            binder_pids = _extract_binder_pids(lsof_out, short_socket_path)
        assert {w1.pid, w2.pid}.issubset(binder_pids), (
            f"socket-binder probe should report both worker PIDs; got {binder_pids} "
            f"(workers: {w1.pid}, {w2.pid})"
        )

        result = check_g_no_dup_binders()

        assert result.passed is False, (
            f"two-binder scenario should FAIL; got detail={result.detail!r}"
        )
        assert str(w1.pid) in result.detail, f"detail missing PID {w1.pid}: {result.detail!r}"
        assert str(w2.pid) in result.detail, f"detail missing PID {w2.pid}: {result.detail!r}"
    finally:
        exit1.set()
        if w2 is not None:
            exit2.set()
        for proc in (w1, w2):
            if proc is None:
                continue
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
