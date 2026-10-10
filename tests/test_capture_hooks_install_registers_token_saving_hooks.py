"""Install/uninstall/status registration coverage for the two new
token-saving sibling hooks: iai-tools-session-hierarchy.sh (SessionStart)
and iai-tools-agent-inject.sh (PreToolUse, matcher Agent|Task).

Both hooks are registered via a genuinely new marker constant so the
absent-marker install branch fires on a fresh machine AND on a machine
that already ran the prior enforce-hook install -- the upgrade path.
"""
from __future__ import annotations

import argparse
import json
import re
import stat
from pathlib import Path

import pytest


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".claude").mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def _settings_path(home: Path) -> Path:
    return home / ".claude" / "settings.json"


def _entries(data: dict, event: str, marker: str) -> list:
    event_entries = data.get("hooks", {}).get(event, [])
    return [
        e for e in event_entries
        if any(marker in (h.get("command") or "") for h in (e.get("hooks") or []))
    ]


# --- Fresh install ----------------------------------------------------

def test_fresh_install_registers_session_hierarchy_hook(fake_home):
    from iai_mcp import cli as cli_mod

    rc = cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    assert rc == 0

    dst = fake_home / ".claude" / "hooks" / "iai-tools-session-hierarchy.sh"
    assert dst.exists()
    assert dst.stat().st_mode & stat.S_IXUSR, "deployed hook must be executable"

    data = json.loads(_settings_path(fake_home).read_text())
    matching = _entries(data, "SessionStart", "iai-tools-session-hierarchy.sh")
    assert len(matching) == 1, data.get("hooks", {}).get("SessionStart", [])
    entry = matching[0]
    assert entry.get("matcher") == "startup|resume|clear|compact", entry
    assert entry["hooks"][0].get("timeout") == 5, entry
    assert re.search(r"bash .*iai-tools-session-hierarchy\.sh", entry["hooks"][0]["command"])


def test_fresh_install_registers_agent_inject_hook(fake_home):
    from iai_mcp import cli as cli_mod

    rc = cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    assert rc == 0

    dst = fake_home / ".claude" / "hooks" / "iai-tools-agent-inject.sh"
    assert dst.exists()
    assert dst.stat().st_mode & stat.S_IXUSR, "deployed hook must be executable"

    data = json.loads(_settings_path(fake_home).read_text())
    matching = _entries(data, "PreToolUse", "iai-tools-agent-inject.sh")
    assert len(matching) == 1, data.get("hooks", {}).get("PreToolUse", [])
    entry = matching[0]
    assert entry.get("matcher") == "Agent|Task", entry
    assert entry["hooks"][0].get("timeout") == 5, entry
    assert re.search(r"bash .*iai-tools-agent-inject\.sh", entry["hooks"][0]["command"])


def test_reinstall_is_idempotent_single_entry_each(fake_home):
    from iai_mcp import cli as cli_mod

    cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    cli_mod.cmd_capture_hooks_install(argparse.Namespace())

    data = json.loads(_settings_path(fake_home).read_text())
    assert len(_entries(data, "SessionStart", "iai-tools-session-hierarchy.sh")) == 1
    assert len(_entries(data, "PreToolUse", "iai-tools-agent-inject.sh")) == 1


# --- Upgrade path: machine already has the prior enforce hook installed --

def test_install_on_settings_with_prior_enforce_entry_still_adds_new_hooks(fake_home):
    from iai_mcp import cli as cli_mod
    from iai_mcp.cli import _capture as capture_mod

    _, dst_enforce, settings = capture_mod._enforce_hook_paths()
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps({
        "hooks": {
            "PreToolUse": [
                {"matcher": "Grep|Glob|Bash",
                 "hooks": [{"type": "command", "command": f"bash {dst_enforce}", "timeout": 5}]},
            ]
        }
    }), encoding="utf-8")

    rc = cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    assert rc == 0

    data = json.loads(settings.read_text())
    # The prior enforce entry survives untouched.
    assert len(_entries(data, "PreToolUse", "iai-tools-enforce.sh")) == 1
    # The two new sibling hooks are added by the absent-marker branch.
    assert len(_entries(data, "SessionStart", "iai-tools-session-hierarchy.sh")) == 1
    assert len(_entries(data, "PreToolUse", "iai-tools-agent-inject.sh")) == 1


def test_status_reports_both_new_hooks_wired_after_upgrade_install(fake_home, capsys):
    # Combines the upgrade-install scenario with a status check: confirms
    # WIRED reporting both on a fresh install and on an install over
    # settings.json that already has the prior enforce-hook entry, not
    # just the settings.json content in isolation.
    from iai_mcp import cli as cli_mod
    from iai_mcp.cli import _capture as capture_mod

    _, dst_enforce, settings = capture_mod._enforce_hook_paths()
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps({
        "hooks": {
            "PreToolUse": [
                {"matcher": "Grep|Glob|Bash",
                 "hooks": [{"type": "command", "command": f"bash {dst_enforce}", "timeout": 5}]},
            ]
        }
    }), encoding="utf-8")

    cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    capsys.readouterr()
    rc = cli_mod.cmd_capture_hooks_status(argparse.Namespace())
    out = capsys.readouterr().out
    assert rc == 0, out
    assert re.search(r"session-hierarchy SessionStart:\s+\S+\s+WIRED", out), out
    assert re.search(r"agent-inject PreToolUse:\s+\S+\s+WIRED", out), out


# --- Uninstall ----------------------------------------------------------

def test_uninstall_removes_both_new_hooks_symmetrically(fake_home):
    from iai_mcp import cli as cli_mod

    cli_mod.cmd_capture_hooks_install(argparse.Namespace())

    session_hierarchy_dst = fake_home / ".claude" / "hooks" / "iai-tools-session-hierarchy.sh"
    agent_inject_dst = fake_home / ".claude" / "hooks" / "iai-tools-agent-inject.sh"
    assert session_hierarchy_dst.exists()
    assert agent_inject_dst.exists()

    rc = cli_mod.cmd_capture_hooks_uninstall(argparse.Namespace())
    assert rc == 0

    assert not session_hierarchy_dst.exists()
    assert not agent_inject_dst.exists()

    data = json.loads(_settings_path(fake_home).read_text())
    assert not _entries(data, "SessionStart", "iai-tools-session-hierarchy.sh")
    assert not _entries(data, "PreToolUse", "iai-tools-agent-inject.sh")


def test_uninstall_preserves_foreign_pretooluse_entry(fake_home):
    from iai_mcp import cli as cli_mod

    settings = _settings_path(fake_home)
    foreign_cmd = "bash /Users/alice/.claude/hooks/some-other-guard.sh"
    settings.write_text(json.dumps({
        "hooks": {
            "PreToolUse": [
                {"matcher": "Bash", "hooks": [{"type": "command", "command": foreign_cmd}]},
            ]
        }
    }), encoding="utf-8")

    cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    cli_mod.cmd_capture_hooks_uninstall(argparse.Namespace())

    data = json.loads(settings.read_text())
    pre_entries = data.get("hooks", {}).get("PreToolUse", [])
    assert any(
        foreign_cmd in (h.get("command") or "")
        for e in pre_entries for h in (e.get("hooks") or [])
    ), pre_entries


def test_uninstall_preserves_foreign_sessionstart_entry(fake_home):
    # SessionStart now has multiple owned entries (recall, tools-index,
    # session-hierarchy) each removed by its own reload-filter-write pass --
    # a foreign, non-owned entry must survive all of them.
    from iai_mcp import cli as cli_mod

    settings = _settings_path(fake_home)
    foreign_cmd = "bash /Users/alice/.claude/hooks/some-other-sessionstart.sh"
    settings.write_text(json.dumps({
        "hooks": {
            "SessionStart": [
                {"matcher": "startup", "hooks": [{"type": "command", "command": foreign_cmd}]},
            ]
        }
    }), encoding="utf-8")

    cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    cli_mod.cmd_capture_hooks_uninstall(argparse.Namespace())

    data = json.loads(settings.read_text())
    ss_entries = data.get("hooks", {}).get("SessionStart", [])
    assert any(
        foreign_cmd in (h.get("command") or "")
        for e in ss_entries for h in (e.get("hooks") or [])
    ), ss_entries
    assert not _entries(data, "SessionStart", "iai-tools-session-hierarchy.sh")


# --- Status ---------------------------------------------------------------

def test_status_reports_both_new_hooks_wired(fake_home, capsys):
    from iai_mcp import cli as cli_mod

    cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    capsys.readouterr()
    rc = cli_mod.cmd_capture_hooks_status(argparse.Namespace())
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "iai-tools-session-hierarchy.sh" in out, out
    assert "iai-tools-agent-inject.sh" in out, out
    assert re.search(r"session-hierarchy SessionStart:\s+\S+\s+WIRED", out), out
    assert re.search(r"agent-inject PreToolUse:\s+\S+\s+WIRED", out), out


def test_status_reports_agent_inject_wired_without_enforce_entry(fake_home, capsys):
    # Discriminates a wrong-list read: seed settings.json with ONLY the
    # agent-inject PreToolUse entry (no enforce entry at all) and confirm
    # status still reports agent-inject WIRED from the same pre_list.
    from iai_mcp import cli as cli_mod
    from iai_mcp.cli import _capture as capture_mod

    _, dst_agent_inject, settings = capture_mod._agent_inject_hook_paths()
    dst_agent_inject.parent.mkdir(parents=True, exist_ok=True)
    src_agent_inject, _, _ = capture_mod._agent_inject_hook_paths()
    dst_agent_inject.write_bytes(src_agent_inject.read_bytes())

    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps({
        "hooks": {
            "PreToolUse": [
                {"matcher": "Agent|Task",
                 "hooks": [{"type": "command", "command": f"bash {dst_agent_inject}", "timeout": 5}]},
            ]
        }
    }), encoding="utf-8")

    capsys.readouterr()
    cli_mod.cmd_capture_hooks_status(argparse.Namespace())
    out = capsys.readouterr().out
    assert re.search(r"agent-inject PreToolUse:\s+\S+\s+WIRED", out), out
    assert re.search(r"enforce PreToolUse:\s+\S+\s+NOT WIRED", out), out


def test_status_stays_active_when_new_hook_templates_unshipped(fake_home, monkeypatch, tmp_path):
    from iai_mcp import cli as cli_mod
    from iai_mcp.cli import _capture as capture_mod

    cli_mod.cmd_capture_hooks_install(argparse.Namespace())

    dst_sh = fake_home / ".claude" / "hooks" / "iai-tools-session-hierarchy.sh"
    dst_ai = fake_home / ".claude" / "hooks" / "iai-tools-agent-inject.sh"
    dst_sh.unlink()
    dst_ai.unlink()

    missing_src = tmp_path / "not-shipped-this-build.sh"

    original_sh = capture_mod._session_hierarchy_hook_paths
    original_ai = capture_mod._agent_inject_hook_paths

    def _patched_sh():
        _, real_dst, real_settings = original_sh()
        return missing_src, real_dst, real_settings

    def _patched_ai():
        _, real_dst, real_settings = original_ai()
        return missing_src, real_dst, real_settings

    monkeypatch.setattr(capture_mod, "_session_hierarchy_hook_paths", _patched_sh)
    monkeypatch.setattr(capture_mod, "_agent_inject_hook_paths", _patched_ai)

    rc_status = cli_mod.cmd_capture_hooks_status(argparse.Namespace())
    assert rc_status == 0
