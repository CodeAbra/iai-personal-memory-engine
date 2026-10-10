from __future__ import annotations

import argparse
import importlib
import json
from unittest import mock

import pytest

from iai_mcp.cli import _capture

_WIN_BASH = "C:\\Program Files\\Git\\bin\\bash.exe"

# (event key, script filename == the marker substring, trailing literal argument)
_BUILDERS = [
    ("Stop", "iai-mcp-session-capture.sh", None),
    ("UserPromptSubmit", "iai-mcp-turn-capture.sh", None),
    ("UserPromptSubmit", "iai-mcp-per-turn-recall.sh", None),
    ("SessionStart", "iai-mcp-session-recall.sh", None),
    ("SessionStart", "iai-tools-session-index.sh", None),
    ("PostToolUse", "iai-tools-session-capture.sh", "PostToolUse"),
    ("UserPromptSubmit", "iai-tools-session-capture.sh", "UserPromptSubmit"),
    ("PreToolUse", "iai-tools-enforce.sh", None),
    ("SessionStart", "iai-tools-session-hierarchy.sh", None),
    ("PreToolUse", "iai-tools-agent-inject.sh", None),
]


@pytest.fixture(autouse=True)
def _restore_capture_constants():
    # _IS_WIN/_BASH are derived at import time, so a reload under a patched
    # platform leaks into every later test in the session unless undone here.
    yield
    importlib.reload(_capture)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        "iai_mcp.cli._claude_desktop_config_path",
        lambda: None,
        raising=True,
    )
    return tmp_path


def _settings_path(home) -> "object":
    return home / ".claude" / "settings.json"


def _stop_command(home) -> str:
    data = json.loads(_settings_path(home).read_text(encoding="utf-8"))
    return data["hooks"]["Stop"][0]["hooks"][0]["command"]


def _commands_by_marker(settings_path, event: str, marker: str) -> list:
    if not settings_path.exists():
        return []
    data = json.loads(settings_path.read_text(encoding="utf-8"))
    entries = data.get("hooks", {}).get(event, [])
    out = []
    for entry in entries:
        for h in (entry.get("hooks") or []):
            cmd = h.get("command") or ""
            if marker in cmd:
                out.append(cmd)
    return out


def _reload_as_windows(monkeypatch, which=None):
    monkeypatch.setattr(
        "iai_mcp.cli._hook_quoting.shutil.which",
        which or (lambda name: _WIN_BASH if name == "bash" else None),
    )
    importlib.reload(_capture)


def test_windows_stop_hook_quotes_the_interpreter_and_the_script_path(home, monkeypatch):
    with mock.patch("platform.system", return_value="Windows"):
        _reload_as_windows(monkeypatch)
        rc = _capture.cmd_capture_hooks_install(argparse.Namespace())

    assert rc == 0
    dst = home / ".claude" / "hooks" / "iai-mcp-session-capture.sh"
    assert _stop_command(home) == f"'{_WIN_BASH}' {dst}"


def test_posix_stop_hook_command_is_byte_identical_to_before(home):
    rc = _capture.cmd_capture_hooks_install(argparse.Namespace())

    assert rc == 0
    dst = home / ".claude" / "hooks" / "iai-mcp-session-capture.sh"
    # Exact full-string equality -- an endswith/contains check would not
    # catch an accidental interpreter-resolution leak onto POSIX.
    assert _stop_command(home) == f"bash {dst}"


@pytest.mark.parametrize(("event", "name", "trailing_arg"), _BUILDERS)
def test_windows_every_builder_quotes_the_interpreter(home, monkeypatch, event, name, trailing_arg):
    with mock.patch("platform.system", return_value="Windows"):
        _reload_as_windows(monkeypatch)
        rc = _capture.cmd_capture_hooks_install(argparse.Namespace())

    assert rc == 0
    dst = home / ".claude" / "hooks" / name
    expected = f"'{_WIN_BASH}' {dst}"
    if trailing_arg:
        expected += f" {trailing_arg}"
    cmds = _commands_by_marker(_settings_path(home), event, name)
    assert expected in cmds, cmds


@pytest.mark.parametrize(("event", "name", "trailing_arg"), _BUILDERS)
def test_posix_every_builder_is_byte_identical_to_before(home, event, name, trailing_arg):
    rc = _capture.cmd_capture_hooks_install(argparse.Namespace())

    assert rc == 0
    dst = home / ".claude" / "hooks" / name
    expected = f"bash {dst}"
    if trailing_arg:
        expected += f" {trailing_arg}"
    cmds = _commands_by_marker(_settings_path(home), event, name)
    assert expected in cmds, cmds


def test_windows_session_capture_hooks_keep_trailing_argument_outside_the_quoted_segment(home, monkeypatch):
    with mock.patch("platform.system", return_value="Windows"):
        _reload_as_windows(monkeypatch)
        rc = _capture.cmd_capture_hooks_install(argparse.Namespace())

    assert rc == 0
    dst = home / ".claude" / "hooks" / "iai-tools-session-capture.sh"
    post_cmds = _commands_by_marker(_settings_path(home), "PostToolUse", "iai-tools-session-capture.sh")
    submit_cmds = _commands_by_marker(_settings_path(home), "UserPromptSubmit", "iai-tools-session-capture.sh")

    # Exact match against the well-formed form: quoted interpreter, a space,
    # the (unquoted, space-free) script path, a space, the bare trailing
    # token -- NOT the broken all-in-one-quoted form
    # "'<bash>' '<dst> PostToolUse'" which a `contains` check would miss.
    assert post_cmds == [f"'{_WIN_BASH}' {dst} PostToolUse"]
    assert submit_cmds == [f"'{_WIN_BASH}' {dst} UserPromptSubmit"]


def test_heal_on_reinstall_rewrites_a_stale_posix_stop_command_in_place(home):
    settings = _settings_path(home)
    settings.parent.mkdir(parents=True, exist_ok=True)
    stale_cmd = "bash /old/stale/path/iai-mcp-session-capture.sh"
    settings.write_text(
        json.dumps({"hooks": {"Stop": [{"hooks": [{"type": "command", "command": stale_cmd, "timeout": 35}]}]}}),
        encoding="utf-8",
    )

    rc = _capture.cmd_capture_hooks_install(argparse.Namespace())
    assert rc == 0

    data = json.loads(settings.read_text(encoding="utf-8"))
    stop_entries = data["hooks"]["Stop"]
    assert len(stop_entries) == 1, stop_entries  # healed in place, never duplicated

    dst = home / ".claude" / "hooks" / "iai-mcp-session-capture.sh"
    assert stop_entries[0]["hooks"][0]["command"] == f"bash {dst}"
    assert stop_entries[0]["hooks"][0]["command"] != stale_cmd


def test_heal_on_reinstall_rewrites_a_stale_unquoted_windows_stop_command_in_place(home, monkeypatch):
    settings = _settings_path(home)
    settings.parent.mkdir(parents=True, exist_ok=True)
    dst = home / ".claude" / "hooks" / "iai-mcp-session-capture.sh"
    stale_cmd = f"C:\\Program Files\\Git\\bin\\bash.exe {dst}"  # old, unquoted
    settings.write_text(
        json.dumps({"hooks": {"Stop": [{"hooks": [{"type": "command", "command": stale_cmd, "timeout": 35}]}]}}),
        encoding="utf-8",
    )

    with mock.patch("platform.system", return_value="Windows"):
        _reload_as_windows(monkeypatch)
        rc = _capture.cmd_capture_hooks_install(argparse.Namespace())

    assert rc == 0
    data = json.loads(settings.read_text(encoding="utf-8"))
    stop_entries = data["hooks"]["Stop"]
    assert len(stop_entries) == 1, stop_entries  # healed in place, never duplicated
    assert stop_entries[0]["hooks"][0]["command"] == f"'{_WIN_BASH}' {dst}"


def test_heal_on_reinstall_does_not_disturb_an_already_correct_command(home):
    rc = _capture.cmd_capture_hooks_install(argparse.Namespace())
    assert rc == 0
    first_pass = _stop_command(home)

    rc = _capture.cmd_capture_hooks_install(argparse.Namespace())
    assert rc == 0

    data = json.loads(_settings_path(home).read_text(encoding="utf-8"))
    stop_entries = data["hooks"]["Stop"]
    assert len(stop_entries) == 1, stop_entries
    assert stop_entries[0]["hooks"][0]["command"] == first_pass
