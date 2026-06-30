from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from iai_mcp._ipc import IS_WINDOWS
from _socket_test_helpers import bind_fake_daemon_socket


@pytest.fixture
def short_socket_paths(tmp_path, monkeypatch):
    lock_path = tmp_path / ".lock"
    state_path = tmp_path / ".daemon-state.json"
    store_dir = tmp_path / "store"
    store_dir.mkdir(parents=True, exist_ok=True)

    from iai_mcp import cli, daemon_state

    with tempfile.TemporaryDirectory(prefix="iai-sock-") as sock_dir_name:
        sock_dir = Path(sock_dir_name)
        sock_path = sock_dir / "d.sock"
        monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(sock_path))
        monkeypatch.setenv("IAI_MCP_STORE", str(store_dir))
        monkeypatch.setattr(daemon_state, "STATE_PATH", state_path)
        monkeypatch.setattr(cli, "LOCK_PATH", lock_path)
        monkeypatch.setattr(cli, "SOCKET_PATH", sock_path)

        try:
            yield lock_path, sock_path, state_path
        finally:
            try:
                if sock_path.exists():
                    sock_path.unlink()
            except OSError:
                pass


def test_clean_environment_yields_check_a_fail_exit_1(short_socket_paths, capsys):
    from iai_mcp.doctor import cmd_doctor

    args = argparse.Namespace(apply=False, yes=False)
    rc = cmd_doctor(args)
    captured = capsys.readouterr()

    assert rc == 1, f"expected 1 (FAIL no --apply), got {rc}"
    assert "iai doctor" in captured.out
    assert "(a) daemon process alive" in captured.out
    assert "ABSENT" in captured.out, "check (a) should say ABSENT when no daemon_pid"


@pytest.mark.parametrize(
    "scenario,expected_fail_check",
    [
        ("no_daemon_pid", "(a) daemon process alive"),
        ("dead_pid_in_state", "(a) daemon process alive"),
        ("stale_socket_unconnectable", "(b) socket file fresh"),
        ("orphan_core_procs", "(d) no orphan iai_mcp.core procs"),
        ("corrupt_state_fsm", "(e) daemon state file valid"),
    ],
)
def test_individual_failure_modes(
    scenario, expected_fail_check, short_socket_paths, monkeypatch
):
    _, sock_path, state_path = short_socket_paths

    if scenario == "no_daemon_pid":
        pass

    elif scenario == "dead_pid_in_state":
        state_path.write_text(json.dumps({"daemon_pid": 2_000_000, "fsm_state": "WAKE"}))

    elif scenario == "stale_socket_unconnectable":
        sock_path.write_text("")

    elif scenario == "orphan_core_procs":
        import psutil

        class _FakeProc:
            def __init__(self, pid: int, cmdline: list[str]):
                self.info = {"pid": pid, "cmdline": cmdline}

        fake = _FakeProc(99_999, ["python", "-m", "iai_mcp.core"])
        monkeypatch.setattr(
            psutil, "process_iter", lambda *a, **kw: [fake]
        )

    elif scenario == "corrupt_state_fsm":
        state_path.write_text(json.dumps({"fsm_state": "INVALID_STATE_VALUE"}))

    from iai_mcp.doctor import run_diagnosis

    results = run_diagnosis()
    fail_names = [r.name for r in results if not r.passed]
    assert expected_fail_check in fail_names, (
        f"Expected FAIL on '{expected_fail_check}' for scenario '{scenario}'; "
        f"got fails: {fail_names}"
    )


def test_print_checklist_format_six_rows(short_socket_paths, monkeypatch, capsys):
    from iai_mcp import doctor

    forced_results = [
        doctor.CheckResult("(a) daemon process alive", True, "PID 99999 (iai_mcp.daemon)"),
        doctor.CheckResult("(b) socket file fresh", True, "connected in 5 ms"),
        doctor.CheckResult("(c) lock file healthy", True, "acquirable"),
        doctor.CheckResult("(d) no orphan iai_mcp.core procs", True, "0 found"),
        doctor.CheckResult("(e) daemon state file valid", True, "fsm_state=WAKE"),
        doctor.CheckResult("(f) hippo storage readable", True, "Hippo storage opens without error"),
    ]
    doctor.print_checklist(forced_results)
    out = capsys.readouterr().out

    assert "iai doctor" in out
    assert out.count("[PASS]") == 6
    assert out.count("[FAIL]") == 0


def test_all_pass_returns_exit_0(short_socket_paths, monkeypatch, capsys):
    from iai_mcp import doctor

    forced_pass = [
        doctor.CheckResult(name, True, "synthetic pass") for name in (
            "(a) daemon process alive",
            "(b) socket file fresh",
            "(c) lock file healthy",
            "(d) no orphan iai_mcp.core procs",
            "(e) daemon state file valid",
            "(f) hippo storage readable",
        )
    ]
    monkeypatch.setattr(doctor, "run_diagnosis", lambda: forced_pass)

    args = argparse.Namespace(apply=False, yes=False)
    rc = doctor.cmd_doctor(args)
    out = capsys.readouterr().out

    assert rc == 0
    assert "All checks passed" in out


def test_apply_without_yes_warns_when_yes_alone(short_socket_paths, monkeypatch, capsys):
    from iai_mcp import doctor

    args = argparse.Namespace(apply=False, yes=True)
    rc = doctor.cmd_doctor(args)
    captured = capsys.readouterr()

    assert "--yes without --apply is meaningless" in captured.err
    assert rc in (0, 1)


def test_exit_code_2_when_apply_cannot_fix(short_socket_paths, monkeypatch, capsys):
    _, _, state_path = short_socket_paths
    state_path.write_text(json.dumps({"fsm_state": "TOTALLY_BOGUS"}))

    from iai_mcp import doctor

    def _forced_fail_e_only():
        return [
            doctor.CheckResult("(a) daemon process alive", True, "synthetic"),
            doctor.CheckResult("(b) socket file fresh", True, "synthetic"),
            doctor.CheckResult("(c) lock file healthy", True, "synthetic"),
            doctor.CheckResult("(d) no orphan iai_mcp.core procs", True, "synthetic"),
            doctor.CheckResult(
                "(e) daemon state file valid",
                False,
                "fsm_state='TOTALLY_BOGUS' not in [...]",
            ),
            doctor.CheckResult("(f) hippo storage readable", True, "synthetic"),
        ]

    monkeypatch.setattr(doctor, "run_diagnosis", _forced_fail_e_only)

    args = argparse.Namespace(apply=True, yes=True)
    rc = doctor.cmd_doctor(args)
    out = capsys.readouterr().out

    assert rc == 2, f"expected 2 (--apply tried but FAIL persists), got {rc}"
    assert "STILL BROKEN" in out
    assert "(e) daemon state file valid" in out


def test_check_b_returns_fail_when_socket_missing(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    if sock_path.exists():
        sock_path.unlink()

    from iai_mcp.doctor import check_b_socket_fresh

    result = check_b_socket_fresh()
    assert result.passed is False
    assert "does not exist" in result.detail


def test_check_e_passes_when_state_file_absent(short_socket_paths):
    _, _, state_path = short_socket_paths
    if state_path.exists():
        state_path.unlink()

    from iai_mcp.doctor import check_e_state_file_valid

    result = check_e_state_file_valid()
    assert result.passed is True
    assert "no state file" in result.detail


def test_check_b_passes_against_silent_listening_socket(short_socket_paths):
    import socket as _socket
    import threading
    import time as _time

    _, sock_path, _ = short_socket_paths
    if sock_path.exists():
        sock_path.unlink()

    server = bind_fake_daemon_socket(sock_path)
    stop = threading.Event()
    accepted: list = []

    def _accept_loop():
        while not stop.is_set():
            try:
                server.settimeout(0.05)
                try:
                    conn, _ = server.accept()
                except (_socket.timeout, OSError):
                    continue
                accepted.append(conn)
            except Exception:
                break

    th = threading.Thread(target=_accept_loop, daemon=True)
    th.start()

    try:
        _time.sleep(0.05)
        from iai_mcp.doctor import check_b_socket_fresh

        result = check_b_socket_fresh()
        assert result.passed is True, (
            f"check_b should PASS against a silent listening socket; "
            f"got: {result.detail}"
        )
        assert "connected" in result.detail
    finally:
        stop.set()
        try:
            for c in accepted:
                try:
                    c.close()
                except OSError:
                    pass
        finally:
            try:
                server.close()
            except OSError:
                pass
        th.join(timeout=1.0)


@pytest.mark.skipif(
    IS_WINDOWS,
    reason="regular-file-where-a-socket-should-be is an AF_UNIX concept; Windows uses a TCP port file",
)
def test_check_b_fails_when_socket_is_regular_file(short_socket_paths):
    _, sock_path, _ = short_socket_paths
    if sock_path.exists():
        sock_path.unlink()
    sock_path.write_text("")

    from iai_mcp.doctor import check_b_socket_fresh

    result = check_b_socket_fresh()
    assert result.passed is False
    assert "unreachable" in result.detail
