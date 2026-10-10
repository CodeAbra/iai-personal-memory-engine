from __future__ import annotations

import argparse
import json
import re
import stat
from pathlib import Path

import pytest

HOOK_MARKER = "iai-tools-session-capture.sh"


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


def _post_entries(data: dict) -> list:
    entries = data.get("hooks", {}).get("PostToolUse", [])
    return [
        e for e in entries
        if any(HOOK_MARKER in (h.get("command") or "") for h in (e.get("hooks") or []))
    ]


def _submit_entries(data: dict) -> list:
    entries = data.get("hooks", {}).get("UserPromptSubmit", [])
    return [
        e for e in entries
        if any(HOOK_MARKER in (h.get("command") or "") for h in (e.get("hooks") or []))
    ]


def test_install_deploys_and_wires_both_lifecycles(fake_home):
    from iai_mcp import cli as cli_mod

    rc = cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    assert rc == 0

    dst = fake_home / ".claude" / "hooks" / HOOK_MARKER
    assert dst.exists()
    mode = dst.stat().st_mode
    assert mode & stat.S_IXUSR, "deployed hook must be executable"

    data = json.loads(_settings_path(fake_home).read_text())

    post_matching = _post_entries(data)
    assert len(post_matching) == 1, data.get("hooks", {}).get("PostToolUse", [])
    assert post_matching[0].get("matcher") == "Edit|Write|MultiEdit|TodoWrite|Task|Skill|Bash"
    post_cmd = post_matching[0]["hooks"][0]["command"]
    assert re.search(r"bash .*iai-tools-session-capture\.sh PostToolUse$", post_cmd), post_cmd

    submit_matching = _submit_entries(data)
    assert len(submit_matching) == 1, data.get("hooks", {}).get("UserPromptSubmit", [])
    submit_cmd = submit_matching[0]["hooks"][0]["command"]
    assert re.search(r"bash .*iai-tools-session-capture\.sh UserPromptSubmit$", submit_cmd), submit_cmd

    # the brain's own UserPromptSubmit hooks still coexist in the same array
    submit_entries = data.get("hooks", {}).get("UserPromptSubmit", [])
    assert any(
        "iai-mcp-turn-capture.sh" in (h.get("command") or "")
        for e in submit_entries for h in (e.get("hooks") or [])
    ), submit_entries
    assert any(
        "iai-mcp-per-turn-recall.sh" in (h.get("command") or "")
        for e in submit_entries for h in (e.get("hooks") or [])
    ), submit_entries


def test_reinstall_is_idempotent_single_entry_per_lifecycle(fake_home):
    from iai_mcp import cli as cli_mod

    rc1 = cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    assert rc1 == 0
    rc2 = cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    assert rc2 == 0

    data = json.loads(_settings_path(fake_home).read_text())
    assert len(_post_entries(data)) == 1, data.get("hooks", {}).get("PostToolUse", [])
    assert len(_submit_entries(data)) == 1, data.get("hooks", {}).get("UserPromptSubmit", [])


def test_uninstall_removes_script_and_both_entries_preserves_foreign_entry(fake_home):
    from iai_mcp import cli as cli_mod

    # Hand-seed a foreign PostToolUse entry BEFORE install -- this installer
    # must never touch an entry it does not own. Neutral foreign command,
    # unrelated to any hook this repo ships.
    settings = _settings_path(fake_home)
    foreign_cmd = "node /Users/alice/tools/some-other-hook.mjs"
    settings.write_text(json.dumps({
        "hooks": {
            "PostToolUse": [
                {"matcher": "Edit", "hooks": [{"type": "command", "command": foreign_cmd}]},
            ]
        }
    }), encoding="utf-8")

    cli_mod.cmd_capture_hooks_install(argparse.Namespace())

    dst = fake_home / ".claude" / "hooks" / HOOK_MARKER
    assert dst.exists(), "install did not deploy the session-capture hook"

    rc = cli_mod.cmd_capture_hooks_uninstall(argparse.Namespace())
    assert rc == 0

    assert not dst.exists()

    data = json.loads(settings.read_text())
    assert not _post_entries(data)
    assert not _submit_entries(data)

    post_entries = data.get("hooks", {}).get("PostToolUse", [])
    assert any(
        foreign_cmd in (h.get("command") or "")
        for e in post_entries for h in (e.get("hooks") or [])
    ), post_entries


def test_status_reports_session_capture_wired(fake_home, capsys):
    from iai_mcp import cli as cli_mod

    cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    capsys.readouterr()
    cli_mod.cmd_capture_hooks_status(argparse.Namespace())
    out = capsys.readouterr().out
    assert HOOK_MARKER in out, out
    assert re.search(r"session-capture PostToolUse:\s+\S+\s+WIRED", out), out
    assert re.search(r"session-capture UserPromptSubmit:\s+\S+\s+WIRED", out), out


def test_status_stays_active_when_session_capture_template_unshipped(fake_home, monkeypatch, tmp_path):
    from iai_mcp import cli as cli_mod
    from iai_mcp.cli import _capture as capture_mod

    cli_mod.cmd_capture_hooks_install(argparse.Namespace())

    dst = fake_home / ".claude" / "hooks" / HOOK_MARKER
    dst.unlink()

    original = capture_mod._tools_session_capture_hook_paths
    missing_src = tmp_path / "not-shipped-this-build.sh"

    def _patched():
        _, real_dst, real_settings = original()
        return missing_src, real_dst, real_settings

    monkeypatch.setattr(capture_mod, "_tools_session_capture_hook_paths", _patched)

    rc_status = cli_mod.cmd_capture_hooks_status(argparse.Namespace())
    assert rc_status == 0
