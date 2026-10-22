"""Claude Code host registration for the iai-tools server: idempotent
add/remove of the `iai-tools` key in ~/.claude.json, mirroring the
existing memory-server (`iai-mcp`) patcher without touching it.
"""
from __future__ import annotations

import json

import pytest

from iai_mcp.cli._capture import _patch_claude_code_tools_config


@pytest.fixture()
def claude_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_TOOLS_WRAPPER_PATH", str(tmp_path / "tools-index.js"))
    (tmp_path / "tools-index.js").write_text("// fake tools wrapper\n", encoding="utf-8")
    return tmp_path


class TestInstall:
    def test_creates_config_with_iai_tools_when_absent(self, claude_home):
        msg, ok = _patch_claude_code_tools_config("install")
        assert ok is True
        assert "created" in msg or "patched" in msg
        cfg = json.loads((claude_home / ".claude.json").read_text(encoding="utf-8"))
        assert "iai-tools" in cfg["mcpServers"]
        entry = cfg["mcpServers"]["iai-tools"]
        assert entry["command"] == "node"
        assert entry["args"] == [str(claude_home / "tools-index.js")]

    def test_install_is_idempotent(self, claude_home):
        _patch_claude_code_tools_config("install")
        first = (claude_home / ".claude.json").read_bytes()
        msg, ok = _patch_claude_code_tools_config("install")
        second = (claude_home / ".claude.json").read_bytes()
        assert first == second
        assert "no change" in msg
        assert ok is True

    def test_unreadable_config_reports_failure(self, claude_home):
        cfg_path = claude_home / ".claude.json"
        cfg_path.write_text("{not json", encoding="utf-8")
        msg, ok = _patch_claude_code_tools_config("install")
        assert ok is False
        assert "unreadable" in msg

    def test_preserves_foreign_server_entry(self, claude_home):
        cfg_path = claude_home / ".claude.json"
        cfg_path.write_text(
            json.dumps({"mcpServers": {"other-server": {"command": "node", "args": ["/opt/other.js"]}}}),
            encoding="utf-8",
        )
        _patch_claude_code_tools_config("install")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert cfg["mcpServers"]["other-server"] == {"command": "node", "args": ["/opt/other.js"]}
        assert "iai-tools" in cfg["mcpServers"]

    def test_does_not_touch_memory_server_key(self, claude_home):
        cfg_path = claude_home / ".claude.json"
        cfg_path.write_text(
            json.dumps({"mcpServers": {"iai-mcp": {"command": "node", "args": ["/opt/memory.js"]}}}),
            encoding="utf-8",
        )
        _patch_claude_code_tools_config("install")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert cfg["mcpServers"]["iai-mcp"] == {"command": "node", "args": ["/opt/memory.js"]}
        assert "iai-tools" in cfg["mcpServers"]


class TestUninstall:
    def test_removes_only_iai_tools_key(self, claude_home):
        cfg_path = claude_home / ".claude.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "other-server": {"command": "node", "args": ["/opt/other.js"]},
                        "iai-tools": {"command": "node", "args": ["/old/path.js"]},
                    }
                }
            ),
            encoding="utf-8",
        )
        msg, ok = _patch_claude_code_tools_config("uninstall")
        assert ok is True
        assert "removed" in msg
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert "iai-tools" not in cfg["mcpServers"]
        assert cfg["mcpServers"]["other-server"] == {"command": "node", "args": ["/opt/other.js"]}

    def test_uninstall_on_absent_config_is_noop(self, claude_home):
        msg, ok = _patch_claude_code_tools_config("uninstall")
        assert ok is True
        assert "absent" in msg
        assert not (claude_home / ".claude.json").exists()

    def test_uninstall_without_iai_tools_key_is_noop(self, claude_home):
        cfg_path = claude_home / ".claude.json"
        cfg_path.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
        msg, ok = _patch_claude_code_tools_config("uninstall")
        assert ok is True
        assert "no change" in msg

    def test_uninstall_unreadable_config_reports_failure(self, claude_home):
        cfg_path = claude_home / ".claude.json"
        cfg_path.write_text("{not json", encoding="utf-8")
        msg, ok = _patch_claude_code_tools_config("uninstall")
        assert ok is False
        assert "unreadable" in msg
