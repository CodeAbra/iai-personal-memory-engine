"""Regression tests for the three narrow Codex installer fixes: renderer
companion deploy, element-wise foreign-hook preservation on uninstall, and
registration-only status wording. Kept in a separate file so the existing
tests/test_codex_hooks.py fence stays byte-unmodified.
"""
from __future__ import annotations

import json

import pytest

from iai_mcp.cli._codex_hooks import (
    _RECALL_RENDER_HELPER,
    install_codex_hooks,
    status_codex_hooks,
    uninstall_codex_hooks,
)

_HOST_AUTHORIZATION_CLAIM_TOKENS = ("active", "trusted", "authorized", "will execute")


@pytest.fixture()
def codex_home(tmp_path, monkeypatch):
    home = tmp_path / "codex-home"
    monkeypatch.setenv("IAI_MCP_CODEX_HOME", str(home))
    return home


class TestRendererDeploy:
    def test_install_deploys_recall_render_helper(self, codex_home):
        assert install_codex_hooks() == 0
        dst = codex_home / "hooks" / _RECALL_RENDER_HELPER
        assert dst.exists()
        assert "render_recall_block" in dst.read_text(encoding="utf-8")

    def test_uninstall_removes_recall_render_helper(self, codex_home):
        install_codex_hooks()
        dst = codex_home / "hooks" / _RECALL_RENDER_HELPER
        assert dst.exists()
        uninstall_codex_hooks()
        assert not dst.exists()


class TestForeignHookSurvival:
    def test_uninstall_preserves_foreign_command_sharing_our_entry(self, codex_home):
        codex_home.mkdir(parents=True)
        (codex_home / "hooks.json").write_text(json.dumps({"hooks": {}}))
        install_codex_hooks()

        data = json.loads((codex_home / "hooks.json").read_text())
        own_cmd = data["hooks"]["Stop"][0]["hooks"][0]["command"]
        # Simulate another tool co-locating its own command in the SAME
        # hooks array entry as ours, rather than as a separate array entry.
        data["hooks"]["Stop"][0]["hooks"].append(
            {"type": "command", "command": "echo foreign-in-same-entry"}
        )
        (codex_home / "hooks.json").write_text(json.dumps(data))

        assert uninstall_codex_hooks() == 0

        data2 = json.loads((codex_home / "hooks.json").read_text())
        stop_cmds = [h["command"] for e in data2["hooks"]["Stop"] for h in e["hooks"]]
        assert "echo foreign-in-same-entry" in stop_cmds
        assert own_cmd not in stop_cmds

    def test_uninstall_drops_entry_only_when_nothing_but_ours_remains(self, codex_home):
        install_codex_hooks()
        data = json.loads((codex_home / "hooks.json").read_text())
        assert len(data["hooks"]["Stop"]) == 1
        uninstall_codex_hooks()
        data2 = json.loads((codex_home / "hooks.json").read_text())
        assert "Stop" not in data2["hooks"]

    def test_uninstall_drops_matcher_carrying_entry_when_all_own(self, codex_home):
        # SessionStart wiring carries a matcher alongside our own hook and
        # nothing else -- the whole entry (matcher included) must go, not
        # linger as a dangling {"hooks": [], "matcher": "..."}.
        install_codex_hooks()
        data = json.loads((codex_home / "hooks.json").read_text())
        session_start = data["hooks"]["SessionStart"]
        assert len(session_start) == 1
        assert session_start[0].get("matcher")

        assert uninstall_codex_hooks() == 0

        data2 = json.loads((codex_home / "hooks.json").read_text())
        assert "SessionStart" not in data2["hooks"]

    def test_uninstall_preserves_matcher_entry_with_foreign_hook(self, codex_home):
        # Same matcher-carrying entry, but a foreign hook shares it -- the
        # entry (and its matcher) must survive with only our hook removed.
        install_codex_hooks()
        data = json.loads((codex_home / "hooks.json").read_text())
        entry = data["hooks"]["SessionStart"][0]
        own_command = entry["hooks"][0]["command"]
        entry["hooks"].append({"type": "command", "command": "echo foreign-session-start"})
        (codex_home / "hooks.json").write_text(json.dumps(data))

        assert uninstall_codex_hooks() == 0

        data2 = json.loads((codex_home / "hooks.json").read_text())
        remaining = data2["hooks"]["SessionStart"]
        assert len(remaining) == 1
        assert remaining[0]["matcher"] == entry["matcher"]
        remaining_cmds = [h["command"] for h in remaining[0]["hooks"]]
        assert "echo foreign-session-start" in remaining_cmds
        assert own_command not in remaining_cmds


class TestStatusWordingScope:
    def test_status_makes_no_host_authorization_claim(self, codex_home, capsys):
        install_codex_hooks()
        assert status_codex_hooks() == 0
        out = capsys.readouterr().out.lower()
        for token in _HOST_AUTHORIZATION_CLAIM_TOKENS:
            assert token not in out, f"status text claims host authorization via {token!r}"

    def test_status_still_reports_readiness(self, codex_home, capsys):
        assert status_codex_hooks() == 1
        capsys.readouterr()
        install_codex_hooks()
        assert status_codex_hooks() == 0
        out = capsys.readouterr().out
        assert "REGISTERED" in out
