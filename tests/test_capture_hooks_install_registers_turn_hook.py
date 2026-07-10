from __future__ import annotations

import argparse
import json
import platform
import stat
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX exec bit + shell hook layout",
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        "iai_mcp.cli._claude_desktop_config_path",
        lambda: None,
        raising=True,
    )
    return tmp_path


def _install(home: Path) -> int:
    from iai_mcp.cli import cmd_capture_hooks_install
    return cmd_capture_hooks_install(argparse.Namespace())


def _uninstall(home: Path) -> int:
    from iai_mcp.cli import cmd_capture_hooks_uninstall
    return cmd_capture_hooks_uninstall(argparse.Namespace())


def _status(home: Path) -> int:
    from iai_mcp.cli import cmd_capture_hooks_status
    return cmd_capture_hooks_status(argparse.Namespace())


def _entries(settings_path: Path, key: str) -> list:
    if not settings_path.exists():
        return []
    data = json.loads(settings_path.read_text())
    return data.get("hooks", {}).get(key, [])


def _commands_for(entries: list) -> list[str]:
    out = []
    for entry in entries:
        for h in (entry.get("hooks") or []):
            cmd = h.get("command")
            if cmd:
                out.append(cmd)
    return out


def test_install_writes_both_hook_entries(home):
    rc = _install(home)
    assert rc == 0

    settings = home / ".claude" / "settings.json"
    assert settings.exists()

    stop_cmds = _commands_for(_entries(settings, "Stop"))
    submit_cmds = _commands_for(_entries(settings, "UserPromptSubmit"))

    assert any(c.endswith("iai-mcp-session-capture.sh") for c in stop_cmds), stop_cmds
    assert any(c.endswith("iai-mcp-turn-capture.sh") for c in submit_cmds), submit_cmds

    stop_hook = home / ".claude" / "hooks" / "iai-mcp-session-capture.sh"
    turn_hook = home / ".claude" / "hooks" / "iai-mcp-turn-capture.sh"
    assert stop_hook.exists()
    assert turn_hook.exists()


def test_install_idempotent(home):
    _install(home)
    _install(home)

    settings = home / ".claude" / "settings.json"
    stop_entries = _entries(settings, "Stop")
    submit_cmds = _commands_for(_entries(settings, "UserPromptSubmit"))
    assert len(stop_entries) == 1, stop_entries
    # UserPromptSubmit carries two distinct hooks (turn-capture + per-turn
    # recall); idempotency means exactly one entry per marker, not one total.
    assert sum(c.endswith("iai-mcp-turn-capture.sh") for c in submit_cmds) == 1, submit_cmds
    assert sum(c.endswith("iai-mcp-per-turn-recall.sh") for c in submit_cmds) == 1, submit_cmds


def test_uninstall_removes_both(home):
    _install(home)
    _uninstall(home)

    settings = home / ".claude" / "settings.json"
    data = json.loads(settings.read_text()) if settings.exists() else {}
    hooks = data.get("hooks", {})

    assert "Stop" not in hooks or hooks["Stop"] == []
    assert "UserPromptSubmit" not in hooks or hooks["UserPromptSubmit"] == []

    stop_hook = home / ".claude" / "hooks" / "iai-mcp-session-capture.sh"
    turn_hook = home / ".claude" / "hooks" / "iai-mcp-turn-capture.sh"
    assert not stop_hook.exists()
    assert not turn_hook.exists()


def test_status_reports_both_wired(home, capsys):
    _install(home)
    capsys.readouterr()
    rc = _status(home)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "WIRED" in out
    assert "UserPromptSubmit" in out or "iai-mcp-turn-capture.sh" in out


def test_install_copies_turn_hook_script(home):
    _install(home)

    turn_hook = home / ".claude" / "hooks" / "iai-mcp-turn-capture.sh"
    assert turn_hook.exists()
    mode = turn_hook.stat().st_mode
    assert mode & stat.S_IXUSR, oct(mode)
