from __future__ import annotations

import argparse
import json
import re
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


def test_install_adds_per_turn_recall_entry_idempotent(fake_home):
    from iai_mcp import cli as cli_mod

    rc1 = cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    assert rc1 == 0
    rc2 = cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    assert rc2 == 0

    data = json.loads(_settings_path(fake_home).read_text())
    submit_entries = data.get("hooks", {}).get("UserPromptSubmit", [])
    matching = [
        e for e in submit_entries
        if any("iai-mcp-per-turn-recall.sh" in (h.get("command") or "")
               for h in (e.get("hooks") or []))
    ]
    assert len(matching) == 1, submit_entries
    cmd = matching[0]["hooks"][0]["command"]
    assert re.search(r"bash .*iai-mcp-per-turn-recall\.sh", cmd), cmd

    assert (fake_home / ".claude" / "hooks" / "iai-mcp-per-turn-recall.sh").exists()


def test_uninstall_removes_per_turn_recall_entry_and_script(fake_home):
    from iai_mcp import cli as cli_mod

    cli_mod.cmd_capture_hooks_install(argparse.Namespace())

    pt_dst = fake_home / ".claude" / "hooks" / "iai-mcp-per-turn-recall.sh"
    assert pt_dst.exists(), "install did not copy per-turn recall hook"

    rc = cli_mod.cmd_capture_hooks_uninstall(argparse.Namespace())
    assert rc == 0

    settings = _settings_path(fake_home)
    if settings.exists():
        data = json.loads(settings.read_text())
        for e in data.get("hooks", {}).get("UserPromptSubmit", []):
            for h in (e.get("hooks") or []):
                assert "iai-mcp-per-turn-recall.sh" not in (h.get("command") or "")

    assert not pt_dst.exists()


def test_uninstall_keeps_turn_capture_removal_independent(fake_home):
    # Both UserPromptSubmit hooks (turn-capture + per-turn recall) must be
    # removed without one filter pass resurrecting or shadowing the other.
    from iai_mcp import cli as cli_mod

    cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    cli_mod.cmd_capture_hooks_uninstall(argparse.Namespace())

    settings = _settings_path(fake_home)
    if settings.exists():
        data = json.loads(settings.read_text())
        for e in data.get("hooks", {}).get("UserPromptSubmit", []):
            for h in (e.get("hooks") or []):
                assert "iai-mcp-turn-capture.sh" not in (h.get("command") or "")
                assert "iai-mcp-per-turn-recall.sh" not in (h.get("command") or "")


def test_status_reports_per_turn_recall(fake_home, capsys):
    from iai_mcp import cli as cli_mod

    cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    capsys.readouterr()
    rc = cli_mod.cmd_capture_hooks_status(argparse.Namespace())
    out = capsys.readouterr().out
    assert "iai-mcp-per-turn-recall.sh" in out, out
    assert rc == 0, out
