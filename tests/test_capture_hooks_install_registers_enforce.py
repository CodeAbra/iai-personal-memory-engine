from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

HOOK_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "iai_mcp" / "_deploy" / "hooks" / "iai-tools-enforce.sh"
)


@pytest.fixture(autouse=True)
def _tools_preflight_ready(monkeypatch):
    monkeypatch.setattr("iai_mcp.cli._capture._iai_tools_preflight", lambda: [])


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


def _enforce_entries(data: dict) -> list:
    pre_entries = data.get("hooks", {}).get("PreToolUse", [])
    return [
        e for e in pre_entries
        if any("iai-tools-enforce.sh" in (h.get("command") or "")
               for h in (e.get("hooks") or []))
    ]


def test_install_deploys_and_wires_enforce_hook(fake_home):
    from iai_mcp import cli as cli_mod

    rc = cli_mod.cmd_capture_hooks_install(argparse.Namespace(with_tools=True))
    assert rc == 0

    dst = fake_home / ".claude" / "hooks" / "iai-tools-enforce.sh"
    assert dst.exists()

    mode = dst.stat().st_mode
    assert mode & stat.S_IXUSR, "deployed hook must be executable"

    data = json.loads(_settings_path(fake_home).read_text())
    matching = _enforce_entries(data)
    assert len(matching) == 1, data.get("hooks", {}).get("PreToolUse", [])
    cmd = matching[0]["hooks"][0]["command"]
    assert re.search(r"bash .*iai-tools-enforce\.sh", cmd), cmd
    assert matching[0]["hooks"][0].get("timeout") == 5, matching[0]


def test_install_backs_up_and_warns_on_malformed_settings_json(fake_home, capsys):
    from iai_mcp import cli as cli_mod

    settings = _settings_path(fake_home)
    settings.write_text("{not valid json", encoding="utf-8")

    rc = cli_mod.cmd_capture_hooks_install(argparse.Namespace(with_tools=True))
    assert rc == 0

    out = capsys.readouterr().out
    assert "WARN" in out and "backed up" in out, out

    backups = list(settings.parent.glob("settings.json.bak-*"))
    assert len(backups) == 1, backups
    assert backups[0].read_text(encoding="utf-8") == "{not valid json"

    data = json.loads(settings.read_text(encoding="utf-8"))
    matching = _enforce_entries(data)
    assert len(matching) == 1, data


def test_reinstall_is_idempotent_single_entry(fake_home):
    from iai_mcp import cli as cli_mod

    rc1 = cli_mod.cmd_capture_hooks_install(argparse.Namespace(with_tools=True))
    assert rc1 == 0
    rc2 = cli_mod.cmd_capture_hooks_install(argparse.Namespace(with_tools=True))
    assert rc2 == 0

    data = json.loads(_settings_path(fake_home).read_text())
    matching = _enforce_entries(data)
    assert len(matching) == 1, data.get("hooks", {}).get("PreToolUse", [])


def test_reinstall_heals_missing_timeout_on_stale_registration(fake_home):
    from iai_mcp import cli as cli_mod
    from iai_mcp.cli import _capture as capture_mod

    _, dst_enforce, settings = capture_mod._enforce_hook_paths()
    settings.parent.mkdir(parents=True, exist_ok=True)
    stale_cmd = f"bash {dst_enforce}"
    settings.write_text(json.dumps({
        "hooks": {
            "PreToolUse": [
                {"matcher": "Grep|Glob|Bash",
                 "hooks": [{"type": "command", "command": stale_cmd}]},
            ]
        }
    }), encoding="utf-8")

    rc = cli_mod.cmd_capture_hooks_install(argparse.Namespace(with_tools=True))
    assert rc == 0

    data = json.loads(settings.read_text())
    matching = _enforce_entries(data)
    assert len(matching) == 1, data.get("hooks", {}).get("PreToolUse", [])
    assert matching[0]["hooks"][0].get("timeout") == 5, matching[0]


def test_uninstall_removes_enforce_entry_and_file_preserves_two_foreign_entries(fake_home):
    from iai_mcp import cli as cli_mod

    # This is the installer's FIRST PreToolUse hook -- the real array on a
    # live machine already holds foreign entries (rtk, gsd guards, a
    # personal search-blocking hook). Seed TWO distinct foreign PreToolUse
    # entries before install and assert BOTH survive uninstall by positive
    # match -- a wrong filter here is the highest-risk destructive path in
    # this phase.
    settings = _settings_path(fake_home)
    foreign_cmd_a = "rtk hook claude"
    foreign_cmd_b = "bash /Users/alice/.claude/hooks/some-other-guard.sh"
    settings.write_text(json.dumps({
        "hooks": {
            "PreToolUse": [
                {"matcher": "Bash", "hooks": [{"type": "command", "command": foreign_cmd_a}]},
                {"matcher": "Bash|Edit|Write", "hooks": [{"type": "command", "command": foreign_cmd_b}]},
            ]
        }
    }), encoding="utf-8")

    cli_mod.cmd_capture_hooks_install(argparse.Namespace(with_tools=True))

    enforce_dst = fake_home / ".claude" / "hooks" / "iai-tools-enforce.sh"
    assert enforce_dst.exists(), "install did not deploy the enforce hook"

    rc = cli_mod.cmd_capture_hooks_uninstall(argparse.Namespace())
    assert rc == 0

    assert not enforce_dst.exists()

    data = json.loads(settings.read_text())
    pre_entries = data.get("hooks", {}).get("PreToolUse", [])
    assert not _enforce_entries(data), pre_entries

    # Both foreign, non-owned entries survive by positive match.
    assert any(
        foreign_cmd_a in (h.get("command") or "")
        for e in pre_entries for h in (e.get("hooks") or [])
    ), pre_entries
    assert any(
        foreign_cmd_b in (h.get("command") or "")
        for e in pre_entries for h in (e.get("hooks") or [])
    ), pre_entries


def test_status_reports_enforce_wired(fake_home, capsys):
    from iai_mcp import cli as cli_mod

    cli_mod.cmd_capture_hooks_install(argparse.Namespace(with_tools=True))
    capsys.readouterr()
    cli_mod.cmd_capture_hooks_status(argparse.Namespace())
    out = capsys.readouterr().out
    assert "iai-tools-enforce.sh" in out, out
    assert re.search(r"enforce PreToolUse:\s+\S+\s+WIRED", out), out


def test_status_rc_reflects_broken_enforce_deploy(fake_home):
    from iai_mcp import cli as cli_mod

    rc_install = cli_mod.cmd_capture_hooks_install(argparse.Namespace(with_tools=True))
    assert rc_install == 0

    # Simulate a broken deploy: the packaged template exists (this build
    # ships it) but the installed copy is gone -- the overall verdict must
    # surface this instead of reporting ACTIVE.
    dst = fake_home / ".claude" / "hooks" / "iai-tools-enforce.sh"
    dst.unlink()

    rc_status = cli_mod.cmd_capture_hooks_status(argparse.Namespace())
    assert rc_status == 1


def test_status_stays_active_when_enforce_template_unshipped(fake_home, monkeypatch, tmp_path):
    from iai_mcp import cli as cli_mod
    from iai_mcp.cli import _capture as capture_mod

    cli_mod.cmd_capture_hooks_install(argparse.Namespace(with_tools=True))

    # Soft, coexisting rollout: a machine whose packaged template is absent
    # (not yet shipped this build) must not flip to INACTIVE just because
    # the enforce hook was never installed there.
    dst = fake_home / ".claude" / "hooks" / "iai-tools-enforce.sh"
    dst.unlink()

    original = capture_mod._enforce_hook_paths
    missing_src = tmp_path / "not-shipped-this-build.sh"

    def _patched():
        _, real_dst, real_settings = original()
        return missing_src, real_dst, real_settings

    monkeypatch.setattr(capture_mod, "_enforce_hook_paths", _patched)

    rc_status = cli_mod.cmd_capture_hooks_status(argparse.Namespace())
    assert rc_status == 0


def test_install_prints_project_scope_manual_reapply_note(fake_home, capsys):
    from iai_mcp import cli as cli_mod

    cli_mod.cmd_capture_hooks_install(argparse.Namespace(with_tools=True))
    out = capsys.readouterr().out
    assert "project-scope" in out, out
    assert "re-applied manually" in out, out


# --- Hook-behavior smoke: proves the deployed hook actually denies/allows ---

pytestmark_skip_win = pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX shell hook")


def _run_enforce_hook(payload: dict, *, timeout: float = 10.0):
    env = os.environ.copy()
    # jq must be findable at /usr/bin/jq -- without it on PATH the deny row
    # fails open to empty stdout + exit 0 and reads as a false "allow".
    env["PATH"] = "/usr/bin:/bin"
    return subprocess.run(
        ["bash", str(HOOK_PATH)],
        input=json.dumps(payload),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytestmark_skip_win
def test_deployed_hook_denies_native_search_tool_call():
    assert HOOK_PATH.exists(), f"hook script missing: {HOOK_PATH}"
    proc = _run_enforce_hook({"tool_name": "Grep"})
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny", proc.stdout


@pytestmark_skip_win
def test_deployed_hook_allows_benign_git_status():
    assert HOOK_PATH.exists(), f"hook script missing: {HOOK_PATH}"
    proc = _run_enforce_hook({"tool_name": "Bash", "tool_input": {"command": "git status"}})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", proc.stdout
