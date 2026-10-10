"""Antigravity host wiring (POSIX): named group `iai-pme` in
~/.gemini/config/hooks.json. The Windows-specific installer behaviors
(platform-dependent extension/prefix, scheduled task) live in
test_antigravity_hooks_windows.py — this file exercises the POSIX
install/uninstall/status flow and the install-time dedup match.
"""
from __future__ import annotations

import json

import pytest

from iai_mcp.cli._antigravity_hooks import (
    _GROUP,
    _HOOK_SCRIPTS,
    install_antigravity_hooks,
    status_antigravity_hooks,
    uninstall_antigravity_hooks,
)


@pytest.fixture()
def antigravity_config_dir(tmp_path, monkeypatch):
    config_dir = tmp_path / "gemini-config"
    monkeypatch.setenv("IAI_MCP_ANTIGRAVITY_CONFIG_DIR", str(config_dir))
    return config_dir


class TestInstaller:
    def test_install_writes_scripts_and_wires_events(self, antigravity_config_dir):
        assert install_antigravity_hooks() == 0
        for base_name in _HOOK_SCRIPTS:
            dst = antigravity_config_dir / "iai-hooks" / (base_name + ".sh")
            assert dst.exists(), base_name
            assert dst.stat().st_mode & 0o100, f"{base_name} must be executable"

        data = json.loads((antigravity_config_dir / "hooks.json").read_text())
        group = data[_GROUP]
        assert group["enabled"] is True
        assert any(
            "iai-mcp-antigravity-recall.sh" in h["command"]
            for e in group["PreInvocation"] for h in e["hooks"]
        )
        assert any(
            "iai-mcp-antigravity-capture.sh" in h["command"]
            for e in group["Stop"] for h in e["hooks"]
        )

    def test_install_is_idempotent(self, antigravity_config_dir):
        install_antigravity_hooks()
        first = json.loads((antigravity_config_dir / "hooks.json").read_text())
        install_antigravity_hooks()
        second = json.loads((antigravity_config_dir / "hooks.json").read_text())
        assert first == second

    def test_install_preserves_foreign_group(self, antigravity_config_dir):
        antigravity_config_dir.mkdir(parents=True)
        foreign = {
            "other-tool": {
                "enabled": True,
                "Stop": [{"hooks": [{"type": "command", "command": "echo mine"}]}],
            }
        }
        (antigravity_config_dir / "hooks.json").write_text(json.dumps(foreign))
        install_antigravity_hooks()
        data = json.loads((antigravity_config_dir / "hooks.json").read_text())
        assert data["other-tool"] == foreign["other-tool"]
        assert _GROUP in data

    def test_uninstall_preserves_foreign_group_containing_basename_substring(
        self, antigravity_config_dir
    ):
        # A foreign group that happens to mention one of our basenames must
        # survive uninstall — uninstall only ever pops the _GROUP key.
        antigravity_config_dir.mkdir(parents=True)
        foreign_cmd = "bash /opt/other-tool/wrap-iai-mcp-antigravity-capture.sh-relay.sh"
        foreign = {
            "other-tool": {
                "enabled": True,
                "Stop": [
                    {"hooks": [{"type": "command", "command": foreign_cmd}]}
                ],
            }
        }
        (antigravity_config_dir / "hooks.json").write_text(json.dumps(foreign))
        install_antigravity_hooks()
        assert uninstall_antigravity_hooks() == 0
        data = json.loads((antigravity_config_dir / "hooks.json").read_text())
        assert data["other-tool"] == foreign["other-tool"]
        assert _GROUP not in data

    def test_uninstall_removes_our_group(self, antigravity_config_dir):
        install_antigravity_hooks()
        assert uninstall_antigravity_hooks() == 0
        for base_name in _HOOK_SCRIPTS:
            assert not (
                antigravity_config_dir / "iai-hooks" / (base_name + ".sh")
            ).exists()
        data = json.loads((antigravity_config_dir / "hooks.json").read_text())
        assert _GROUP not in data

    def test_status_reflects_wiring(self, antigravity_config_dir):
        assert status_antigravity_hooks() == 1
        install_antigravity_hooks()
        assert status_antigravity_hooks() == 0
        uninstall_antigravity_hooks()
        assert status_antigravity_hooks() == 1

    def test_install_heals_a_near_miss_own_entry_in_place(self, antigravity_config_dir):
        # A hand-edited near-miss own entry (marker substring present, not
        # the exact command install writes) is healed in place on
        # reinstall -- one entry, never a duplicate append.
        antigravity_config_dir.mkdir(parents=True)
        install_antigravity_hooks()
        data = json.loads((antigravity_config_dir / "hooks.json").read_text())
        stop_hook = data[_GROUP]["Stop"][0]["hooks"][0]
        near_miss_cmd = stop_hook["command"] + " --extra-flag"
        stop_hook["command"] = near_miss_cmd
        (antigravity_config_dir / "hooks.json").write_text(json.dumps(data))

        assert install_antigravity_hooks() == 0
        data2 = json.loads((antigravity_config_dir / "hooks.json").read_text())
        cmds = [
            h["command"]
            for e in data2[_GROUP]["Stop"]
            for h in e["hooks"]
        ]
        exact = (
            f"bash "
            f"{antigravity_config_dir / 'iai-hooks' / 'iai-mcp-antigravity-capture.sh'}"
        )
        assert cmds == [exact], "the stale entry must be healed in place, not duplicated"
