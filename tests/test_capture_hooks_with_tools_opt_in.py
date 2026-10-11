"""`capture-hooks install` wires iai-tools only behind `--with-tools`.

A default install leaves no iai-tools hook, script, or MCP registration
behind and reports ACTIVE; `--with-tools` refuses to touch anything when the
tools server or executor is missing, and a plain `uninstall` still removes an
opted-in install.
"""
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import pytest

from iai_mcp.cli import _capture

TOOLS_HOOK_FILES = (
    "iai-tools-session-index.sh",
    "iai-tools-session-capture.sh",
    "iai-tools-enforce.sh",
    "iai-tools-session-hierarchy.sh",
    "iai-tools-agent-inject.sh",
)


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("IAI_MCP_CODEX_HOME", str(tmp_path / "codex-home"))
    tools_wrapper = tmp_path / "tools-index.js"
    tools_wrapper.write_text("// fake tools wrapper\n", encoding="utf-8")
    monkeypatch.setenv("IAI_MCP_TOOLS_WRAPPER_PATH", str(tools_wrapper))
    return home


@pytest.fixture
def tools_ready(monkeypatch):
    monkeypatch.setattr(_capture, "_iai_tools_preflight", lambda: [])


def _settings(home: Path) -> dict:
    return json.loads((home / ".claude" / "settings.json").read_text(encoding="utf-8"))


def _all_commands(data: dict) -> list[str]:
    return [
        h.get("command") or ""
        for entries in data.get("hooks", {}).values()
        for e in entries
        for h in (e.get("hooks") or [])
    ]


def _claude_json_servers(home: Path) -> dict:
    cfg = home / ".claude.json"
    if not cfg.exists():
        return {}
    return json.loads(cfg.read_text(encoding="utf-8")).get("mcpServers", {})


def test_default_install_wires_no_iai_tools(fake_home):
    rc = _capture.cmd_capture_hooks_install(argparse.Namespace())
    assert rc == 0

    commands = _all_commands(_settings(fake_home))
    assert any("iai-mcp-session-capture.sh" in c for c in commands)
    for name in TOOLS_HOOK_FILES:
        assert not any(name in c for c in commands), name
        assert not (fake_home / ".claude" / "hooks" / name).exists(), name
    assert "iai-tools" not in _claude_json_servers(fake_home)


def test_default_install_reports_active_with_tools_off(fake_home, capsys):
    assert _capture.cmd_capture_hooks_install(argparse.Namespace()) == 0
    capsys.readouterr()

    rc = _capture.cmd_capture_hooks_status(argparse.Namespace())
    out = capsys.readouterr().out

    assert rc == 0, out
    assert "status: ACTIVE" in out
    assert "iai-tools: off" in out
    assert "Enforce installed" not in out


def test_with_tools_refuses_and_changes_nothing_when_components_missing(
    fake_home, monkeypatch, capsys
):
    monkeypatch.setattr(
        _capture, "_iai_tools_preflight", lambda: ["iai-tools executor (iai_tools_execd)"]
    )

    rc = _capture.cmd_capture_hooks_install(argparse.Namespace(with_tools=True))
    err = capsys.readouterr().err

    assert rc == 1
    assert "--with-tools" in err and "iai_tools_execd" in err
    assert not (fake_home / ".claude" / "settings.json").exists()
    assert not (fake_home / ".claude" / "hooks").exists()
    assert "iai-tools" not in _claude_json_servers(fake_home)


def test_with_tools_wires_all_tools_hooks_and_server(fake_home, tools_ready, capsys):
    rc = _capture.cmd_capture_hooks_install(argparse.Namespace(with_tools=True))
    assert rc == 0

    commands = _all_commands(_settings(fake_home))
    for name in TOOLS_HOOK_FILES:
        assert any(name in c for c in commands), name
        assert (fake_home / ".claude" / "hooks" / name).exists(), name
    assert "iai-tools" in _claude_json_servers(fake_home)

    capsys.readouterr()
    status_rc = _capture.cmd_capture_hooks_status(argparse.Namespace())
    out = capsys.readouterr().out
    assert status_rc == 0, out
    assert "iai-tools: off" not in out


def test_partial_tools_install_reports_inactive_with_opt_in_hint(
    fake_home, tools_ready, capsys
):
    assert _capture.cmd_capture_hooks_install(argparse.Namespace(with_tools=True)) == 0

    settings_path = fake_home / ".claude" / "settings.json"
    data = _settings(fake_home)
    data["hooks"]["PreToolUse"] = [
        e for e in data["hooks"]["PreToolUse"]
        if not any("iai-tools-enforce.sh" in (h.get("command") or "")
                   for h in (e.get("hooks") or []))
    ]
    settings_path.write_text(json.dumps(data), encoding="utf-8")
    capsys.readouterr()

    rc = _capture.cmd_capture_hooks_status(argparse.Namespace())
    out = capsys.readouterr().out

    assert rc == 1, out
    assert "Run: iai-mcp capture-hooks install --with-tools" in out


def test_plain_uninstall_removes_an_opted_in_install(fake_home, tools_ready):
    assert _capture.cmd_capture_hooks_install(argparse.Namespace(with_tools=True)) == 0

    _capture.cmd_capture_hooks_uninstall(argparse.Namespace())

    settings_path = fake_home / ".claude" / "settings.json"
    commands = _all_commands(json.loads(settings_path.read_text(encoding="utf-8")))
    for name in TOOLS_HOOK_FILES:
        assert not any(name in c for c in commands), name
        assert not (fake_home / ".claude" / "hooks" / name).exists(), name
    assert "iai-tools" not in _claude_json_servers(fake_home)


def test_default_codex_install_registers_no_iai_tools(fake_home, tmp_path, capsys):
    rc = _capture.cmd_capture_hooks_install(argparse.Namespace(target="codex"))
    assert rc == 0

    codex_cfg = tmp_path / "codex-home" / "config.toml"
    if codex_cfg.exists():
        servers = tomllib.loads(codex_cfg.read_text(encoding="utf-8")).get("mcp_servers", {})
        assert "iai-tools" not in servers
    assert "iai-tools" not in _claude_json_servers(fake_home)

    capsys.readouterr()
    status_rc = _capture.cmd_capture_hooks_status(argparse.Namespace(target="codex"))
    out = capsys.readouterr().out
    assert status_rc == 0, out
    assert "Codex iai-tools MCP server: off" in out


def test_preflight_names_each_missing_component(monkeypatch, tmp_path):
    from iai_mcp import tools_exec_client
    from iai_mcp.cli import _codex_mcp

    def _no_wrapper():
        raise FileNotFoundError("no tools-index.js")

    monkeypatch.setattr(_codex_mcp, "_resolve_tools_wrapper_path", _no_wrapper)
    monkeypatch.setattr(
        tools_exec_client, "_resolve_execd_bin", lambda: str(tmp_path / "absent-execd")
    )

    missing = _capture._iai_tools_preflight()

    assert len(missing) == 2
    assert any("tools-index.js" in m for m in missing)
    assert any("iai_tools_execd" in m for m in missing)


def test_preflight_is_empty_when_server_and_executor_exist(monkeypatch, tmp_path):
    from iai_mcp import tools_exec_client
    from iai_mcp.cli import _codex_mcp

    wrapper = tmp_path / "tools-index.js"
    wrapper.write_text("// built\n", encoding="utf-8")
    execd = tmp_path / "iai_tools_execd"
    execd.write_text("", encoding="utf-8")
    monkeypatch.setattr(_codex_mcp, "_resolve_tools_wrapper_path", lambda: wrapper)
    monkeypatch.setattr(tools_exec_client, "_resolve_execd_bin", lambda: str(execd))

    assert _capture._iai_tools_preflight() == []
