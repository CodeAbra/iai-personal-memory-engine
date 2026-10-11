"""A single `iai-mcp capture-hooks install --target codex --with-tools` run
wires the iai-tools MCP server into both the Codex config.toml and the
Claude Code host config, and `uninstall --target codex` reverses both.
"""
from __future__ import annotations

import argparse
import json
import tomllib

import pytest

from iai_mcp.cli._capture import (
    cmd_capture_hooks_install,
    cmd_capture_hooks_status,
    cmd_capture_hooks_uninstall,
)


@pytest.fixture()
def temp_hosts(tmp_path, monkeypatch):
    codex_home = tmp_path / "codex-home"
    claude_home = tmp_path / "claude-home"
    claude_home.mkdir(parents=True)
    monkeypatch.setenv("IAI_MCP_CODEX_HOME", str(codex_home))
    monkeypatch.setenv("HOME", str(claude_home))

    wrapper = tmp_path / "index.js"
    wrapper.write_text("// fake memory wrapper\n", encoding="utf-8")
    monkeypatch.setenv("IAI_MCP_WRAPPER_PATH", str(wrapper))

    tools_wrapper = tmp_path / "tools-index.js"
    tools_wrapper.write_text("// fake tools wrapper\n", encoding="utf-8")
    monkeypatch.setenv("IAI_MCP_TOOLS_WRAPPER_PATH", str(tools_wrapper))
    monkeypatch.setattr("iai_mcp.cli._capture._iai_tools_preflight", lambda: [])

    # Simulate a machine that already ran `capture-hooks install --target
    # claude` for the memory server, so the codex-targeted run below can
    # prove it leaves that pre-existing registration untouched.
    (claude_home / ".claude.json").write_text(
        json.dumps({"mcpServers": {"iai-mcp": {"command": "node", "args": [str(wrapper)]}}}),
        encoding="utf-8",
    )

    return codex_home, claude_home


def _args(target: str) -> argparse.Namespace:
    return argparse.Namespace(target=target, with_tools=True)


class TestInstall:
    def test_registers_iai_tools_for_both_hosts(self, temp_hosts):
        codex_home, claude_home = temp_hosts

        cmd_capture_hooks_install(_args("codex"))

        codex_cfg = tomllib.loads((codex_home / "config.toml").read_text(encoding="utf-8"))
        assert "iai-tools" in codex_cfg["mcp_servers"]

        claude_cfg = json.loads((claude_home / ".claude.json").read_text(encoding="utf-8"))
        assert "iai-tools" in claude_cfg["mcpServers"]

    def test_does_not_disturb_existing_memory_server_registration(self, temp_hosts):
        codex_home, claude_home = temp_hosts

        cmd_capture_hooks_install(_args("codex"))

        claude_cfg = json.loads((claude_home / ".claude.json").read_text(encoding="utf-8"))
        assert "iai-mcp" in claude_cfg["mcpServers"]


class TestUninstall:
    def test_removes_iai_tools_from_both_hosts(self, temp_hosts):
        codex_home, claude_home = temp_hosts

        cmd_capture_hooks_install(_args("codex"))
        cmd_capture_hooks_uninstall(_args("codex"))

        codex_cfg = tomllib.loads((codex_home / "config.toml").read_text(encoding="utf-8"))
        assert "iai-tools" not in codex_cfg.get("mcp_servers", {})

        claude_cfg = json.loads((claude_home / ".claude.json").read_text(encoding="utf-8"))
        assert "iai-tools" not in claude_cfg["mcpServers"]
        assert "iai-mcp" in claude_cfg["mcpServers"]


class TestStatus:
    def test_status_reports_wired_after_install(self, temp_hosts, capsys):
        cmd_capture_hooks_install(_args("codex"))
        capsys.readouterr()

        rc = cmd_capture_hooks_status(_args("codex"))
        out = capsys.readouterr().out

        assert rc == 0
        assert "iai-tools" in out.lower() or "mcp_servers.iai-tools" in out


class TestInstallExitCodeReflectsToolsRegistration:
    def test_install_succeeds_when_claude_json_is_readable(self, temp_hosts):
        assert cmd_capture_hooks_install(_args("codex")) == 0

    def test_install_exit_code_matches_status_on_unreadable_claude_json(
        self, temp_hosts, capsys
    ):
        _codex_home, claude_home = temp_hosts
        (claude_home / ".claude.json").write_text("{not json", encoding="utf-8")

        install_rc = cmd_capture_hooks_install(_args("codex"))
        capsys.readouterr()
        status_rc = cmd_capture_hooks_status(_args("codex"))

        # A scripted install that checks $? must see the same failure state
        # `status` reports for the identical broken ~/.claude.json --
        # install must not silently skip the iai-tools registration at rc 0.
        assert install_rc != 0
        assert status_rc != 0
