"""Consolidated `run`/`fetch`/`batch-run` CLI surface: --auto-index/--json
on `run`, --json on `fetch`/`batch-run`, --store selector on `search`, and
dual-surface parity between `iai-tools` and `iai-mcp tools`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _execd_bin() -> str:
    for profile in ("release", "debug"):
        candidate = REPO / "target" / profile / "iai_tools_execd"
        if candidate.exists():
            return str(candidate)
    pytest.skip("iai_tools_execd not built; run: cargo build -p iai_mcp_tools_core --bins")


@pytest.fixture
def exec_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "iai-tools-root"
    sock_dir = Path(tempfile.mkdtemp(prefix="iai-tools-", dir="/tmp"))
    sock_path = sock_dir / ".execd.sock"
    monkeypatch.setenv("IAI_TOOLS_ROOT", str(root))
    monkeypatch.setenv("IAI_TOOLS_SOCKET_PATH", str(sock_path))
    monkeypatch.setenv("IAI_TOOLS_EXECD_BIN", _execd_bin())
    monkeypatch.setenv("IAI_TOOLS_IDLE_SECS", "1.0")
    try:
        yield {"root": root, "sock_path": sock_path}
    finally:
        shutil.rmtree(sock_dir, ignore_errors=True)


def test_run_json_shape(exec_env: dict, capsys: pytest.CaptureFixture) -> None:
    from iai_mcp import tools_cli

    rc = tools_cli.main(["run", "--json", "echo", "hello-json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert set(payload.keys()) == {"run_id", "state", "stdout", "stderr"}
    assert "hello-json" in payload["stdout"]

    rc2 = tools_cli.main(["run", "echo", "hello-plain"])
    assert rc2 == 0
    out2 = capsys.readouterr().out
    assert "hello-plain" in out2
    assert "run_id:" in out2 and "state:" in out2


def test_fetch_and_batch_json_shape(exec_env: dict, capsys: pytest.CaptureFixture) -> None:
    from iai_mcp import tools_cli

    run_rc = tools_cli.main(["run", "--json", "echo", "fetch-source"])
    assert run_rc == 0
    run_payload = json.loads(capsys.readouterr().out.strip())
    run_id = run_payload["run_id"]

    fetch_rc = tools_cli.main(["fetch", run_id, "--json"])
    assert fetch_rc == 0
    fetch_payload = json.loads(capsys.readouterr().out.strip())
    assert set(fetch_payload.keys()) == {"run_id", "state", "stdout", "stderr"}
    assert fetch_payload["run_id"] == run_id

    fetch_rc2 = tools_cli.main(["fetch", run_id])
    assert fetch_rc2 == 0
    assert f"run_id: {run_id}" in capsys.readouterr().out

    batch_rc = tools_cli.main(
        ["batch-run", "--concurrency", "2", "--json", "echo batch-one", "echo batch-two"]
    )
    assert batch_rc == 0
    batch_payload = json.loads(capsys.readouterr().out.strip())
    assert isinstance(batch_payload, list)
    assert len(batch_payload) == 2
    for element in batch_payload:
        assert set(element.keys()) == {"run_id", "state", "stdout", "stderr"}
    assert "batch-one" in batch_payload[0]["stdout"]
    assert "batch-two" in batch_payload[1]["stdout"]

    batch_rc2 = tools_cli.main(
        ["batch-run", "--concurrency", "2", "echo batch-one-plain", "echo batch-two-plain"]
    )
    assert batch_rc2 == 0
    out3 = capsys.readouterr().out
    assert "batch-one-plain" in out3 and "batch-two-plain" in out3


def test_dual_surface_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both `iai-tools` and `iai-mcp tools` delegate `run`/`fetch`/
    `batch-run` to the SAME `iai_mcp.tools.exec_cli` entry points -- proven
    by import identity: patching the shared module's function is observed
    by both surfaces' dispatch functions.
    """
    from iai_mcp import tools_cli
    from iai_mcp.cli import _tools as cli_tools
    from iai_mcp.tools import exec_cli as exec_cli_mod

    run_calls: "list[object]" = []
    fetch_calls: "list[object]" = []
    batch_calls: "list[object]" = []

    monkeypatch.setattr(exec_cli_mod, "cmd_run", lambda a: (run_calls.append(a), 0)[1])
    monkeypatch.setattr(exec_cli_mod, "cmd_fetch", lambda a: (fetch_calls.append(a), 0)[1])
    monkeypatch.setattr(exec_cli_mod, "cmd_batch_run", lambda a: (batch_calls.append(a), 0)[1])

    run_ns = argparse.Namespace(
        command="echo", args=["hi"], timeout=None, background=False, auto_index=False, json=False
    )
    assert tools_cli._cmd_run(run_ns) == 0
    assert cli_tools.cmd_tools_run(run_ns) == 0
    assert run_calls == [run_ns, run_ns]

    fetch_ns = argparse.Namespace(run_id="some-id", bound=None, json=False)
    assert tools_cli._cmd_fetch(fetch_ns) == 0
    assert cli_tools.cmd_tools_fetch(fetch_ns) == 0
    assert fetch_calls == [fetch_ns, fetch_ns]

    batch_ns = argparse.Namespace(
        commands=["echo one"], concurrency=1, timeout_ms=None, json=False
    )
    assert tools_cli._cmd_batch_run(batch_ns) == 0
    assert cli_tools.cmd_tools_batch_run(batch_ns) == 0
    assert batch_calls == [batch_ns, batch_ns]


def test_search_store_root_shared_across_surfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both `search` command bodies delegate the `--store` selector to the
    SAME `iai_mcp.tools.exec_cli.search_store_root` -- proven by import
    identity: patching the shared function is observed by both surfaces,
    and the store_root it returns is what actually reaches `search()`.
    """
    import iai_mcp.tools as tools_mod
    from iai_mcp import tools_cli
    from iai_mcp.cli import _tools as cli_tools
    from iai_mcp.tools import exec_cli as exec_cli_mod

    selector_calls: "list[object]" = []
    sentinel = Path("/sentinel-command-output-store")

    def fake_selector(args: object) -> Path:
        selector_calls.append(args)
        return sentinel

    monkeypatch.setattr(exec_cli_mod, "search_store_root", fake_selector)

    search_calls: "list[Path | None]" = []

    def fake_search(query: str, *, k: int = 10, store_root: "Path | None" = None):
        search_calls.append(store_root)
        return []

    monkeypatch.setattr(tools_mod, "search", fake_search)

    ns = argparse.Namespace(query="q", k=10, store="output", json=False)
    assert tools_cli._cmd_search(ns) == 0
    assert cli_tools.cmd_tools_search(ns) == 0

    assert len(selector_calls) == 2
    assert search_calls == [sentinel, sentinel]


def test_run_auto_index_cli_cycle(exec_env: dict, tmp_path: Path) -> None:
    """Subprocess round trip with a fresh HOME/IAI_TOOLS_ROOT so the fresh
    import binds `_DEFAULT_IAI_ROOT`/`default_command_output_store_root()`
    under the tmp HOME -- an in-process monkeypatch of HOME would NOT
    rebind the import-time default (mirrors
    tests/test_tools_cli_session_stats.py:86-140).
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    home = tmp_path / "home"
    home.mkdir()
    env["HOME"] = str(home)
    env["IAI_TOOLS_ROOT"] = str(tmp_path / "tools-root")

    marker = uuid.uuid4().hex

    run_proc = subprocess.run(
        [
            sys.executable, "-m", "iai_mcp.tools_cli",
            "run", "--auto-index", "--json", "echo", marker,
        ],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=60,
    )
    assert run_proc.returncode == 0, run_proc.stderr
    payload = json.loads(run_proc.stdout.strip())
    assert marker in payload["stdout"]
    assert payload["indexed"] >= 1
    assert payload["refused"] == []

    search_proc = subprocess.run(
        [
            sys.executable, "-m", "iai_mcp.tools_cli",
            "search", marker, "--store", "output", "--json",
        ],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=60,
    )
    assert search_proc.returncode == 0, search_proc.stderr
    search_payload = json.loads(search_proc.stdout.strip())
    assert search_payload["hits"], "a later search --store output must retrieve the run's output"

    source_search_proc = subprocess.run(
        [sys.executable, "-m", "iai_mcp.tools_cli", "search", marker, "--json"],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=60,
    )
    assert source_search_proc.returncode == 0, source_search_proc.stderr
    source_payload = json.loads(source_search_proc.stdout.strip())
    assert source_payload["hits"] == [], "the default (code) store must not see the command output"
