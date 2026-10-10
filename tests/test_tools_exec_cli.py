"""`tools_exec_client` + `iai-tools`/`iai-mcp tools` run/fetch/batch-run subcommands.

Follows `tests/test_tools_cli_tracer.py`'s naming/shape: real subprocesses
against a temp `IAI_TOOLS_ROOT`/socket, never a live `~/.iai-tools/`.
"""

from __future__ import annotations

import argparse
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


def _execd_bin() -> str:
    for profile in ("release", "debug"):
        candidate = REPO / "target" / profile / "iai_tools_execd"
        if candidate.exists():
            return str(candidate)
    pytest.skip("iai_tools_execd not built; run: cargo build -p iai_mcp_tools_core --bins")


def _console_script_bin(name: str) -> str:
    candidate = Path(sys.executable).parent / name
    return str(candidate) if candidate.exists() else name


def _run_console_script(name: str, args: list[str], *, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_console_script_bin(name), *args],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _parse_run_id(stdout: str) -> str:
    for line in stdout.splitlines():
        if line.startswith("run_id:"):
            # "run_id: <id> state: <state>" -- the id is the second token.
            return line.split()[1]
    raise AssertionError(f"no 'run_id:' line found in stdout:\n{stdout}")


_TEST_IDLE_SECS = 1.0


def _assert_daemon_exited(sock_path: Path, idle_secs: float = _TEST_IDLE_SECS, margin: float = 3.0) -> None:
    """Confirms the daemon self-terminated after its idle window.

    Sleeps past `idle_secs` WITHOUT touching the socket first -- each probe
    connection is itself daemon activity (`connection_opened` records it),
    so polling by connecting would reset the idle clock and the daemon
    would never appear idle. Exactly ONE connect attempt, after the sleep,
    checks the outcome.
    """
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
    monkeypatch.setenv("IAI_TOOLS_EXECD_BIN", _execd_bin())
    # Short idle window: every test's own daemon self-terminates quickly so
    # no live process is ever left running after the test returns.
    monkeypatch.setenv("IAI_TOOLS_IDLE_SECS", str(_TEST_IDLE_SECS))
    try:
        yield {"root": root, "sock_path": sock_path}
    finally:
        shutil.rmtree(sock_dir, ignore_errors=True)


def test_client_lazy_spawns_and_runs_echo_from_cold(exec_env: dict) -> None:
    from iai_mcp import tools_exec_client

    sock_path = exec_env["sock_path"]
    assert not sock_path.exists(), "no daemon must be running before this test's first call"

    result = tools_exec_client.run("echo", ["hello-cold"])
    assert result["run_id"]
    assert "hello-cold" in result["bounded_tail"]["stdout"]
    assert result["state"] == "exited:0"

    _assert_daemon_exited(sock_path)


def test_client_fetch_reserves_without_rerunning(exec_env: dict) -> None:
    from iai_mcp import tools_exec_client

    run_result = tools_exec_client.run("echo", ["fetch-me"])
    run_id = run_result["run_id"]

    fetch_result = tools_exec_client.fetch(run_id)
    # tool_run's own initial fetch already served every byte -- a re-fetch
    # must return nothing NEW, proving no second execution.
    assert fetch_result["bounded_tail"]["stdout"] == ""
    assert fetch_result["run_id"] == run_id

    _assert_daemon_exited(exec_env["sock_path"])


def test_client_run_timeout_actually_kills_a_long_running_command(exec_env: dict) -> None:
    from iai_mcp import tools_exec_client

    started = time.monotonic()
    result = tools_exec_client.run("sleep", ["5"], timeout=0.2)
    elapsed = time.monotonic() - started

    assert result["state"] in ("timed_out",) or result["state"].startswith("killed:")
    assert elapsed < 4.0, f"an honored --timeout must not wait out the full 5s sleep, took {elapsed}s"

    _assert_daemon_exited(exec_env["sock_path"])


def test_client_fetch_bound_actually_caps_the_returned_window(exec_env: dict) -> None:
    from iai_mcp import tools_exec_client

    run_result = tools_exec_client.run(
        "sh", ["-c", "sleep 0.3; printf '0123456789'"], background=True
    )
    run_id = run_result["run_id"]

    # Poll until the background child has actually written its output --
    # a fetch racing ahead of the write would see an empty window and the
    # bound's effect would be unobservable.
    stdout_log = exec_env["root"] / "artifacts" / run_id / "stdout.log"
    deadline = time.monotonic() + 3.0
    while stdout_log.stat().st_size < 10 if stdout_log.exists() else True:
        assert time.monotonic() < deadline, "child never wrote its output"
        time.sleep(0.02)

    bounded = tools_exec_client.fetch(run_id, bound=4)
    assert bounded["bounded_tail"]["stdout"] == "0123", "bound=4 must cap the window at 4 bytes"

    rest = tools_exec_client.fetch(run_id, bound=100)
    assert rest["bounded_tail"]["stdout"] == "456789"

    _assert_daemon_exited(exec_env["sock_path"])


_MALICIOUS_RUN_IDS = ("../../etc/passwd", "/etc/passwd", "..", "", "a/b", "1a-2b-с03")


@pytest.mark.parametrize("run_id", _MALICIOUS_RUN_IDS)
def test_client_fetch_rejects_malicious_run_ids(exec_env: dict, run_id: str) -> None:
    from iai_mcp import tools_exec_client

    with pytest.raises(tools_exec_client.ToolExecError):
        tools_exec_client.fetch(run_id)

    _assert_daemon_exited(exec_env["sock_path"])


def test_client_batch_run_round_trips_two_commands(exec_env: dict) -> None:
    from iai_mcp import tools_exec_client

    results = tools_exec_client.batch_run(
        [
            {"command": "echo", "args": ["batch-one"]},
            {"command": "echo", "args": ["batch-two"]},
        ],
        concurrency=2,
    )
    assert len(results) == 2
    assert "batch-one" in results[0]["bounded_tail"]["stdout"]
    assert "batch-two" in results[1]["bounded_tail"]["stdout"]

    _assert_daemon_exited(exec_env["sock_path"])


def test_client_imports_no_brain_module(exec_env: dict) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")

    brain_modules = (
        "iai_mcp.hippo",
        "iai_mcp.lilli",
        "iai_mcp.store",
        "iai_mcp.capture",
        "iai_mcp_native",
        "pandas",
    )
    probe = (
        "import sys, json, iai_mcp.tools_exec_client as c;"
        "result = c.run('echo', ['probe']);"
        f"brain = [m for m in {brain_modules!r} if m in sys.modules];"
        "print(json.dumps({'run_id': result['run_id'], 'brain_modules': brain}))"
    )
    probe_proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert probe_proc.returncode == 0, probe_proc.stderr
    result = json.loads(probe_proc.stdout.strip().splitlines()[-1])
    assert result["run_id"]
    assert result["brain_modules"] == [], (
        f"tools_exec_client must load no brain module, found: {result['brain_modules']}"
    )

    _assert_daemon_exited(exec_env["sock_path"])


def test_tools_cli_run_argparse_parses_flags_and_remainder_args() -> None:
    from iai_mcp.tools_cli import _build_parser

    ns = _build_parser().parse_args(
        ["run", "--timeout", "5", "--background", "echo", "-n", "hi"]
    )
    assert ns.command == "echo"
    assert ns.args == ["-n", "hi"]
    assert ns.timeout == 5.0
    assert ns.background is True


def test_tools_cli_batch_run_argparse_parses_concurrency_and_commands() -> None:
    from iai_mcp.tools_cli import _build_parser

    ns = _build_parser().parse_args(
        ["batch-run", "--concurrency", "3", "--timeout-ms", "2000", "echo one", "echo two"]
    )
    assert ns.concurrency == 3
    assert ns.timeout_ms == 2000
    assert ns.commands == ["echo one", "echo two"]


def test_cli_iai_tools_run_then_fetch_round_trip(exec_env: dict) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")

    run_proc = _run_console_script("iai-tools", ["run", "echo", "iai-tools-round-trip"], env=env)
    assert run_proc.returncode == 0, run_proc.stderr
    assert "iai-tools-round-trip" in run_proc.stdout
    run_id = _parse_run_id(run_proc.stdout)

    fetch_proc = _run_console_script("iai-tools", ["fetch", run_id], env=env)
    assert fetch_proc.returncode == 0, fetch_proc.stderr
    assert f"run_id: {run_id}" in fetch_proc.stdout

    _assert_daemon_exited(exec_env["sock_path"])


def test_cli_iai_tools_batch_run_round_trip(exec_env: dict) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")

    proc = _run_console_script(
        "iai-tools",
        ["batch-run", "--concurrency", "2", "echo cli-batch-one", "echo cli-batch-two"],
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "cli-batch-one" in proc.stdout
    assert "cli-batch-two" in proc.stdout

    _assert_daemon_exited(exec_env["sock_path"])


def test_cli_iai_mcp_tools_run_then_fetch_round_trip(exec_env: dict) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")

    run_proc = _run_console_script(
        "iai-mcp", ["tools", "run", "echo", "iai-mcp-tools-round-trip"], env=env
    )
    assert run_proc.returncode == 0, run_proc.stderr
    assert "iai-mcp-tools-round-trip" in run_proc.stdout
    run_id = _parse_run_id(run_proc.stdout)

    fetch_proc = _run_console_script("iai-mcp", ["tools", "fetch", run_id], env=env)
    assert fetch_proc.returncode == 0, fetch_proc.stderr
    assert f"run_id: {run_id}" in fetch_proc.stdout

    _assert_daemon_exited(exec_env["sock_path"])


@pytest.mark.parametrize("run_id", _MALICIOUS_RUN_IDS)
def test_cli_iai_tools_fetch_rejects_malicious_run_ids(exec_env: dict, run_id: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")

    proc = _run_console_script("iai-tools", ["fetch", run_id], env=env)
    assert proc.returncode != 0, f"a malicious run_id must not be served, got: {proc.stdout!r}"

    _assert_daemon_exited(exec_env["sock_path"])


@pytest.mark.parametrize("run_id", _MALICIOUS_RUN_IDS)
def test_cli_iai_mcp_tools_fetch_rejects_malicious_run_ids(exec_env: dict, run_id: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")

    proc = _run_console_script("iai-mcp", ["tools", "fetch", run_id], env=env)
    assert proc.returncode != 0, f"a malicious run_id must not be served, got: {proc.stdout!r}"

    _assert_daemon_exited(exec_env["sock_path"])


def test_embed_index_background_returns_run_id_without_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import iai_mcp.tools_exec_client as tools_exec_client
    from iai_mcp.tools_cli import _build_parser

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "one.py").write_text("print('one')\n")

    calls: "list[dict]" = []

    def fake_run(command, args=None, *, timeout=None, background=False):
        calls.append({"command": command, "args": list(args or []), "background": background})
        return {
            "run_id": "stub-run-id",
            "state": "detached",
            "bounded_tail": {"stdout": "", "stderr": ""},
        }

    monkeypatch.setattr(tools_exec_client, "run", fake_run)

    ns = _build_parser().parse_args(["embed-index", str(corpus)])
    rc = ns.func(ns)
    assert rc == 0

    out = capsys.readouterr().out
    assert "run_id: stub-run-id" in out
    assert len(calls) == 1, "embed-index must submit exactly one run, never poll/wait on it"
    assert calls[0]["background"] is True
    assert calls[0]["command"] == sys.executable
    assert "iai_mcp.tools.embed_worker" in calls[0]["args"]
    assert str(corpus / "one.py") in calls[0]["args"]

def test_embed_index_cli_never_imports_brain_embedder(tmp_path: Path) -> None:
    """`sys.modules` in THIS pytest process is already polluted by
    conftest.py's own recall-helper imports (unrelated fixtures shared
    across the whole test session), so the in-process assertion above
    cannot observe this invariant. A clean child process (`python -c`,
    no conftest.py) mirrors `test_client_imports_no_brain_module`'s
    precedent instead.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "one.py").write_text("print('one')\n")

    brain_modules = (
        "iai_mcp.hippo",
        "iai_mcp.lilli",
        "iai_mcp.store",
        "iai_mcp.capture",
        "iai_mcp_native",
        "pandas",
    )
    probe = (
        "import json, sys\n"
        "import iai_mcp.tools_exec_client as tools_exec_client\n"
        "def fake_run(command, args=None, *, timeout=None, background=False):\n"
        "    return {'run_id': 'stub-run-id', 'state': 'detached', "
        "'bounded_tail': {'stdout': '', 'stderr': ''}}\n"
        "tools_exec_client.run = fake_run\n"
        "from iai_mcp.tools_cli import _build_parser\n"
        f"ns = _build_parser().parse_args(['embed-index', {str(corpus)!r}])\n"
        "rc = ns.func(ns)\n"
        f"brain = [m for m in {brain_modules!r} if m in sys.modules]\n"
        "print(json.dumps({'rc': rc, 'brain_modules': brain}))\n"
    )
    probe_proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert probe_proc.returncode == 0, probe_proc.stderr
    result = json.loads(probe_proc.stdout.strip().splitlines()[-1])
    assert result["rc"] == 0
    assert result["brain_modules"] == [], (
        f"embed-index must load no brain module, found: {result['brain_modules']}"
    )


def test_embed_index_foreground_flag_runs_synchronously(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import iai_mcp.tools_exec_client as tools_exec_client
    from iai_mcp.tools_cli import _build_parser

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "one.py").write_text("print('one')\n")

    calls: "list[dict]" = []

    def fake_run(command, args=None, *, timeout=None, background=False):
        calls.append({"background": background})
        return {
            "run_id": "stub-run-id",
            "state": "exited:0",
            "bounded_tail": {"stdout": "", "stderr": ""},
        }

    monkeypatch.setattr(tools_exec_client, "run", fake_run)

    ns = _build_parser().parse_args(["embed-index", str(corpus), "--foreground"])
    rc = ns.func(ns)
    assert rc == 0
    assert calls[0]["background"] is False


def test_cli_iai_mcp_tools_batch_run_round_trip(exec_env: dict) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")

    proc = _run_console_script(
        "iai-mcp",
        ["tools", "batch-run", "--concurrency", "2", "echo mcp-batch-one", "echo mcp-batch-two"],
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "mcp-batch-one" in proc.stdout
    assert "mcp-batch-two" in proc.stdout

    _assert_daemon_exited(exec_env["sock_path"])


def test_tools_cli_batch_run_argparse_parses_commands_json() -> None:
    from iai_mcp.tools_cli import _build_parser

    raw = json.dumps([{"command": "echo", "args": ['a b"c']}])
    ns = _build_parser().parse_args(["batch-run", "--commands-json", raw])
    assert ns.commands_json == raw
    assert ns.commands == []


def test_batch_run_commands_json_bypasses_shlex_tokenization(monkeypatch: pytest.MonkeyPatch) -> None:
    """An argument containing a space and an embedded quote reaches the
    shared batch body byte-identical -- proving --commands-json never
    passes through shlex.split, unlike the positional shell-string form.
    """
    from iai_mcp.tools import exec_cli as exec_cli_mod

    captured: "list[list[dict]]" = []

    def fake_batch_run_and_render(commands, *, concurrency=1, timeout_ms=None):
        captured.append(commands)
        return []

    monkeypatch.setattr(exec_cli_mod, "batch_run_and_render", fake_batch_run_and_render)

    raw_arg = 'a b"c'
    commands_json = json.dumps([{"command": "echo", "args": [raw_arg]}])
    ns = argparse.Namespace(
        commands=[], commands_json=commands_json, concurrency=1, timeout_ms=None, json=False
    )
    rc = exec_cli_mod.cmd_batch_run(ns)
    assert rc == 0
    assert captured[0][0]["args"] == [raw_arg]


def test_batch_run_rejects_both_commands_json_and_positional(capsys: pytest.CaptureFixture) -> None:
    from iai_mcp.tools import exec_cli as exec_cli_mod

    ns = argparse.Namespace(
        commands=["echo one"],
        commands_json=json.dumps([{"command": "echo", "args": ["two"]}]),
        concurrency=1,
        timeout_ms=None,
        json=False,
    )
    rc = exec_cli_mod.cmd_batch_run(ns)
    assert rc == 1
    assert "not both" in capsys.readouterr().err


def test_batch_run_rejects_neither_commands_json_nor_positional(capsys: pytest.CaptureFixture) -> None:
    from iai_mcp.tools import exec_cli as exec_cli_mod

    ns = argparse.Namespace(commands=[], commands_json=None, concurrency=1, timeout_ms=None, json=False)
    rc = exec_cli_mod.cmd_batch_run(ns)
    assert rc == 1
    assert "supply --commands-json" in capsys.readouterr().err


def test_batch_run_rejects_empty_commands_json_array(capsys: pytest.CaptureFixture) -> None:
    from iai_mcp.tools import exec_cli as exec_cli_mod

    ns = argparse.Namespace(commands=[], commands_json="[]", concurrency=1, timeout_ms=None, json=False)
    rc = exec_cli_mod.cmd_batch_run(ns)
    assert rc == 1
    assert "non-empty" in capsys.readouterr().err


def test_batch_run_rejects_commands_json_that_is_not_a_list(capsys: pytest.CaptureFixture) -> None:
    from iai_mcp.tools import exec_cli as exec_cli_mod

    ns = argparse.Namespace(
        commands=[],
        commands_json=json.dumps({"command": "echo"}),
        concurrency=1,
        timeout_ms=None,
        json=False,
    )
    rc = exec_cli_mod.cmd_batch_run(ns)
    assert rc == 1
    assert "non-empty JSON array" in capsys.readouterr().err


def test_batch_run_rejects_invalid_json_in_commands_json(capsys: pytest.CaptureFixture) -> None:
    from iai_mcp.tools import exec_cli as exec_cli_mod

    ns = argparse.Namespace(commands=[], commands_json="not json", concurrency=1, timeout_ms=None, json=False)
    rc = exec_cli_mod.cmd_batch_run(ns)
    assert rc == 1
    assert "not valid JSON" in capsys.readouterr().err


def test_batch_run_rejects_a_batch_over_the_size_cap(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    from iai_mcp.tools import exec_cli as exec_cli_mod

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("batch_run_and_render must not be called for an oversized batch")

    monkeypatch.setattr(exec_cli_mod, "batch_run_and_render", fail_if_called)

    oversized = json.dumps(
        [{"command": "echo", "args": [str(i)]} for i in range(exec_cli_mod.MAX_BATCH_COMMANDS + 1)]
    )
    ns = argparse.Namespace(
        commands=[], commands_json=oversized, concurrency=1, timeout_ms=None, json=False
    )
    rc = exec_cli_mod.cmd_batch_run(ns)
    assert rc == 1
    assert "at most" in capsys.readouterr().err


def test_batch_run_rejects_a_positional_batch_over_the_size_cap(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The size cap applies to the positional shell-string form too --
    the aggregate-response risk it guards against is identical regardless
    of which input form produced the oversized command list.
    """
    from iai_mcp.tools import exec_cli as exec_cli_mod

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("batch_run_and_render must not be called for an oversized batch")

    monkeypatch.setattr(exec_cli_mod, "batch_run_and_render", fail_if_called)

    oversized_positional = [f"echo {i}" for i in range(exec_cli_mod.MAX_BATCH_COMMANDS + 1)]
    ns = argparse.Namespace(
        commands=oversized_positional, commands_json=None, concurrency=1, timeout_ms=None, json=False
    )
    rc = exec_cli_mod.cmd_batch_run(ns)
    assert rc == 1
    assert "at most" in capsys.readouterr().err


def test_batch_command_cap_matches_the_wrapper_constant() -> None:
    """Pins the literal value, not just `MAX_BATCH_COMMANDS + 1` -- an
    independent change to either language's constant must fail a test.
    """
    from iai_mcp.tools import exec_cli as exec_cli_mod

    assert exec_cli_mod.MAX_BATCH_COMMANDS == 64


def test_batch_run_dual_surface_parity_for_commands_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both `iai-tools batch-run` and `iai-mcp tools batch-run` delegate to
    the SAME `iai_mcp.tools.exec_cli.cmd_batch_run` -- proven by import
    identity, so `--commands-json` reaches the same shared body from
    either surface.
    """
    from iai_mcp import tools_cli
    from iai_mcp.cli import _tools as cli_tools
    from iai_mcp.tools import exec_cli as exec_cli_mod

    batch_calls: "list[object]" = []
    monkeypatch.setattr(exec_cli_mod, "cmd_batch_run", lambda a: (batch_calls.append(a), 0)[1])

    ns = argparse.Namespace(
        commands=[],
        commands_json=json.dumps([{"command": "echo", "args": ["one"]}]),
        concurrency=1,
        timeout_ms=None,
        json=False,
    )
    assert tools_cli._cmd_batch_run(ns) == 0
    assert cli_tools.cmd_tools_batch_run(ns) == 0
    assert batch_calls == [ns, ns]
