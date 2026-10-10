"""Cursor host wiring: the core capture scripts plus the Cursor-specific
session-recall and turn-capture wrappers, registered in
~/.cursor/hooks.json.
"""
from __future__ import annotations

import json

import pytest

from iai_mcp.cli._cursor_hooks import (
    _HOOK_SCRIPTS,
    install_cursor_hooks,
    status_cursor_hooks,
    uninstall_cursor_hooks,
)


@pytest.fixture()
def cursor_home(tmp_path, monkeypatch):
    home = tmp_path / "cursor-home"
    monkeypatch.setenv("IAI_MCP_CURSOR_HOME", str(home))
    return home


class TestInstaller:
    def test_install_writes_scripts_and_wires_events(self, cursor_home):
        assert install_cursor_hooks() == 0
        for name in _HOOK_SCRIPTS:
            dst = cursor_home / "iai-hooks" / name
            assert dst.exists(), name
            assert dst.stat().st_mode & 0o100, f"{name} must be executable"

        data = json.loads((cursor_home / "hooks.json").read_text())
        events = data["hooks"]
        assert any(
            "iai-mcp-cursor-session-recall.sh" in e["command"]
            for e in events["sessionStart"]
        )
        assert any(
            "iai-mcp-cursor-turn-capture.sh" in e["command"]
            for e in events["beforeSubmitPrompt"]
        )
        assert any(
            "iai-mcp-session-capture.sh" in e["command"] for e in events["stop"]
        )
        assert any(
            "iai-mcp-session-capture.sh" in e["command"] for e in events["sessionEnd"]
        )

    def test_install_is_idempotent(self, cursor_home):
        install_cursor_hooks()
        first = json.loads((cursor_home / "hooks.json").read_text())
        install_cursor_hooks()
        second = json.loads((cursor_home / "hooks.json").read_text())
        assert first == second

    def test_install_preserves_foreign_hooks(self, cursor_home):
        cursor_home.mkdir(parents=True)
        foreign = {
            "version": 1,
            "hooks": {"stop": [{"command": "echo mine"}]},
        }
        (cursor_home / "hooks.json").write_text(json.dumps(foreign))
        install_cursor_hooks()
        data = json.loads((cursor_home / "hooks.json").read_text())
        cmds = [e["command"] for e in data["hooks"]["stop"]]
        assert "echo mine" in cmds
        assert any("iai-mcp-session-capture.sh" in c for c in cmds)

    def test_uninstall_removes_only_ours(self, cursor_home):
        cursor_home.mkdir(parents=True)
        (cursor_home / "hooks.json").write_text(json.dumps({
            "version": 1,
            "hooks": {"stop": [{"command": "echo mine"}]},
        }))
        install_cursor_hooks()
        assert uninstall_cursor_hooks() == 0
        for name in _HOOK_SCRIPTS:
            assert not (cursor_home / "iai-hooks" / name).exists()
        data = json.loads((cursor_home / "hooks.json").read_text())
        cmds = [e["command"] for e in data["hooks"]["stop"]]
        assert cmds == ["echo mine"]
        assert "sessionStart" not in data["hooks"]

    def test_status_reflects_wiring(self, cursor_home):
        assert status_cursor_hooks() == 1
        install_cursor_hooks()
        assert status_cursor_hooks() == 0
        uninstall_cursor_hooks()
        assert status_cursor_hooks() == 1

    def test_uninstall_preserves_foreign_entry_containing_basename_substring(
        self, cursor_home
    ):
        cursor_home.mkdir(parents=True)
        foreign_cmd = "bash /opt/other-tool/wrap-iai-mcp-session-capture.sh-relay.sh"
        (cursor_home / "hooks.json").write_text(json.dumps({
            "version": 1,
            "hooks": {"stop": [{"command": foreign_cmd}]},
        }))
        install_cursor_hooks()
        # Uninstall cannot tell this foreign wrapper from a hand-edited own
        # entry by content alone, so it takes the safe branch: kept, not
        # silently deleted. The invariant under test is survival, not rc.
        uninstall_cursor_hooks()
        data = json.loads((cursor_home / "hooks.json").read_text())
        cmds = [e["command"] for e in data["hooks"]["stop"]]
        assert foreign_cmd in cmds

    def test_uninstall_removes_own_byte_identical_entry(self, cursor_home):
        install_cursor_hooks()
        exact = f"bash {cursor_home / 'iai-hooks' / 'iai-mcp-session-capture.sh'}"
        data = json.loads((cursor_home / "hooks.json").read_text())
        cmds = [e["command"] for e in data["hooks"]["stop"]]
        assert exact in cmds
        assert uninstall_cursor_hooks() == 0
        data2 = json.loads((cursor_home / "hooks.json").read_text())
        assert "stop" not in data2["hooks"]

    def test_uninstall_refuses_own_near_miss_and_reports(self, cursor_home, capsys):
        install_cursor_hooks()
        data = json.loads((cursor_home / "hooks.json").read_text())
        stop_entry = data["hooks"]["stop"][0]
        near_miss_cmd = stop_entry["command"] + " --extra-flag"
        stop_entry["command"] = near_miss_cmd
        (cursor_home / "hooks.json").write_text(json.dumps(data))

        rc = uninstall_cursor_hooks()
        assert rc == 1
        out = capsys.readouterr().out
        assert "remove" in out.lower()

        data2 = json.loads((cursor_home / "hooks.json").read_text())
        cmds = [e["command"] for e in data2["hooks"]["stop"]]
        assert near_miss_cmd in cmds, "near-miss entry must be kept, not deleted"

    def test_install_skip_check_is_exact_only(self, cursor_home):
        cursor_home.mkdir(parents=True)
        near_miss_cmd = (
            f"bash {cursor_home / 'iai-hooks' / 'iai-mcp-session-capture.sh'} "
            "--wrong-flag"
        )
        (cursor_home / "hooks.json").write_text(json.dumps({
            "version": 1,
            "hooks": {"stop": [{"command": near_miss_cmd}]},
        }))
        install_cursor_hooks()
        data = json.loads((cursor_home / "hooks.json").read_text())
        cmds = [e["command"] for e in data["hooks"]["stop"]]
        exact = f"bash {cursor_home / 'iai-hooks' / 'iai-mcp-session-capture.sh'}"
        assert exact in cmds, "a near-miss entry must not block adding the correct one"
        assert near_miss_cmd in cmds, "install must not remove the pre-existing entry"

    def test_uninstall_refuses_to_touch_unparseable_hooks_json(
        self, cursor_home, capsys
    ):
        cursor_home.mkdir(parents=True)
        (cursor_home / "hooks.json").write_text("{not json")
        rc = uninstall_cursor_hooks()
        assert rc == 1
        out = capsys.readouterr().out
        assert "cannot parse" in out.lower()
        assert (cursor_home / "hooks.json").read_text() == "{not json"
