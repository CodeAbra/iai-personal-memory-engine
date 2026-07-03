from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pytest

# The installer emits PowerShell hooks on Windows and bash hooks on POSIX.
_HOOK_EXT = "ps1" if os.name == "nt" else "sh"
_HOOK_INTERP = "powershell" if os.name == "nt" else "bash"
_RECALL_HOOK = f"iai-mcp-session-recall.{_HOOK_EXT}"
_CAPTURE_HOOK = f"iai-mcp-session-capture.{_HOOK_EXT}"


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


def test_install_adds_sessionstart_entry_idempotent(fake_home):
    from iai_mcp import cli as cli_mod

    rc1 = cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    assert rc1 == 0
    rc2 = cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    assert rc2 == 0

    data = json.loads(_settings_path(fake_home).read_text(encoding="utf-8"))
    ss_entries = data.get("hooks", {}).get("SessionStart", [])
    matching = [
        e for e in ss_entries
        if any(_RECALL_HOOK in (h.get("command") or "")
               for h in (e.get("hooks") or []))
    ]
    assert len(matching) == 1, ss_entries
    entry = matching[0]
    assert entry.get("matcher") == "startup|resume|clear|compact", entry
    cmd = entry["hooks"][0]["command"]
    assert re.search(
        rf"{_HOOK_INTERP}.*iai-mcp-session-recall\.{_HOOK_EXT}", cmd
    ), cmd

    stop_entries = data.get("hooks", {}).get("Stop", [])
    assert any(
        _CAPTURE_HOOK in (h.get("command") or "")
        for e in stop_entries for h in (e.get("hooks") or [])
    ), stop_entries

    assert (fake_home / ".claude" / "hooks" / _RECALL_HOOK).exists()


def test_uninstall_removes_sessionstart_entry_and_script(fake_home):
    from iai_mcp import cli as cli_mod

    cli_mod.cmd_capture_hooks_install(argparse.Namespace())

    recall_dst = fake_home / ".claude" / "hooks" / _RECALL_HOOK
    assert recall_dst.exists(), "install did not copy recall hook"
    data = json.loads(_settings_path(fake_home).read_text(encoding="utf-8"))
    ss = data.get("hooks", {}).get("SessionStart", [])
    assert any(
        _RECALL_HOOK in (h.get("command") or "")
        for e in ss for h in (e.get("hooks") or [])
    ), ss

    rc = cli_mod.cmd_capture_hooks_uninstall(argparse.Namespace())
    assert rc == 0

    settings = _settings_path(fake_home)
    if settings.exists():
        data = json.loads(settings.read_text(encoding="utf-8"))
        ss = data.get("hooks", {}).get("SessionStart", [])
        for e in ss:
            for h in (e.get("hooks") or []):
                assert _RECALL_HOOK not in (h.get("command") or "")

    assert not recall_dst.exists()


def test_status_reports_session_recall_alongside_stop(fake_home, capsys):
    from iai_mcp import cli as cli_mod

    cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    capsys.readouterr()
    cli_mod.cmd_capture_hooks_status(argparse.Namespace())
    out = capsys.readouterr().out
    assert _RECALL_HOOK in out, out
    assert _CAPTURE_HOOK in out, out
