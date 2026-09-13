from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest


def test_happy_path_execs_node_with_wrapper_argv(tmp_path, monkeypatch):
    stub = tmp_path / "index.js"
    stub.write_text("// stub\n")
    monkeypatch.setenv("IAI_MCP_WRAPPER_PATH", str(stub))
    monkeypatch.setenv("PATH", "/sentinel/bin")

    captured = {}

    def fake_execvpe(command, args, env):
        captured["command"] = command
        captured["args"] = args
        captured["env"] = env

    import iai_mcp.mcp_server_entry as entry_mod

    monkeypatch.setattr(entry_mod.os, "execvpe", fake_execvpe)

    result = entry_mod.main()

    assert result is None
    assert captured["command"] == "node"
    assert captured["args"] == ["node", str(stub)]


def test_env_merge_preserves_path_and_sets_expected_vars(tmp_path, monkeypatch):
    stub = tmp_path / "index.js"
    stub.write_text("// stub\n")
    monkeypatch.setenv("IAI_MCP_WRAPPER_PATH", str(stub))
    monkeypatch.setenv("PATH", "/sentinel/bin")

    captured = {}

    def fake_execvpe(command, args, env):
        captured["env"] = env

    import iai_mcp.mcp_server_entry as entry_mod

    monkeypatch.setattr(entry_mod.os, "execvpe", fake_execvpe)

    entry_mod.main()

    env = captured["env"]
    assert env["IAI_MCP_PYTHON"] == sys.executable
    assert env["IAI_MCP_STORE"] == str(Path.home() / ".iai-mcp")
    assert "PATH" in env
    assert env["PATH"] == "/sentinel/bin"


def test_editable_tree_wrapper_exists():
    from iai_mcp.cli import _resolve_wrapper_path

    p = _resolve_wrapper_path()
    assert p.exists()
    assert str(p).endswith("mcp-wrapper/dist/index.js")


def test_fail_loud_on_missing_wrapper_returns_1_without_exec(monkeypatch):
    import iai_mcp.mcp_server_entry as entry_mod

    def raiser():
        raise FileNotFoundError("wrapper not found")

    monkeypatch.setattr(
        "iai_mcp.cli._capture._build_iai_mcp_server_entry", raiser
    )

    def fail_if_called(*_args, **_kwargs):
        pytest.fail("os.execvpe must not be called on the fail-loud path")

    monkeypatch.setattr(entry_mod.os, "execvpe", fail_if_called)

    result = entry_mod.main()

    assert result == 1


@pytest.mark.skipif(
    shutil.which("node") is None, reason="node not on PATH"
)
def test_wrapper_liveness_smoke():
    from iai_mcp.cli import _resolve_wrapper_path

    try:
        wrapper = _resolve_wrapper_path()
    except FileNotFoundError:
        pytest.skip("mcp-wrapper not built")

    proc = subprocess.Popen(
        ["node", str(wrapper)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        time.sleep(0.5)
        assert proc.poll() is None
    finally:
        proc.terminate()
        proc.wait(timeout=5)
