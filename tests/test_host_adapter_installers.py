from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from iai_mcp.cli._antigravity_hooks import (
    install_antigravity_hooks,
    status_antigravity_hooks,
    uninstall_antigravity_hooks,
)
from iai_mcp.cli._cursor_hooks import (
    install_cursor_hooks,
    status_cursor_hooks,
    uninstall_cursor_hooks,
)
from iai_mcp.cli._hermes_hooks import (
    install_hermes_hooks,
    status_hermes_hooks,
    uninstall_hermes_hooks,
)
from iai_mcp.cli._openclaw_mcp import (
    install_openclaw_mcp,
    status_openclaw_mcp,
    uninstall_openclaw_mcp,
)


class TestCursorInstaller:
    @pytest.fixture(autouse=True)
    def _home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        home = tmp_path / "cursor"
        monkeypatch.setenv("IAI_MCP_CURSOR_HOME", str(home))
        return home

    def test_install_wires_events_and_is_idempotent(self, _home: Path) -> None:
        assert install_cursor_hooks() == 0
        data = json.loads((_home / "hooks.json").read_text())
        assert data["version"] == 1
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
        assert install_cursor_hooks() == 0
        again = json.loads((_home / "hooks.json").read_text())
        assert again == data

    def test_install_preserves_foreign_entries(self, _home: Path) -> None:
        _home.mkdir(parents=True)
        (_home / "hooks.json").write_text(json.dumps({
            "version": 1,
            "hooks": {"preToolUse": [{"command": "other-tool"}]},
        }))
        assert install_cursor_hooks() == 0
        data = json.loads((_home / "hooks.json").read_text())
        assert data["hooks"]["preToolUse"] == [{"command": "other-tool"}]

    def test_uninstall_removes_only_ours(self, _home: Path) -> None:
        _home.mkdir(parents=True)
        (_home / "hooks.json").write_text(json.dumps({
            "version": 1,
            "hooks": {"stop": [{"command": "other-tool"}]},
        }))
        install_cursor_hooks()
        uninstall_cursor_hooks()
        data = json.loads((_home / "hooks.json").read_text())
        assert data["hooks"]["stop"] == [{"command": "other-tool"}]
        assert "sessionStart" not in data["hooks"]
        assert not list((_home / "iai-hooks").glob("*.sh"))

    def test_status_roundtrip(self, _home: Path) -> None:
        assert status_cursor_hooks() == 1
        install_cursor_hooks()
        assert status_cursor_hooks() == 0

    def test_install_refuses_unparseable_config(self, _home: Path) -> None:
        _home.mkdir(parents=True)
        corrupt = '{"version": 1, "hooks": {"stop": [{"command": "theirs"}]}} trailing'
        (_home / "hooks.json").write_text(corrupt)
        assert install_cursor_hooks() == 1
        assert (_home / "hooks.json").read_text() == corrupt


class TestAntigravityInstaller:
    @pytest.fixture(autouse=True)
    def _config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        config = tmp_path / "gemini-config"
        monkeypatch.setenv("IAI_MCP_ANTIGRAVITY_CONFIG_DIR", str(config))
        return config

    def test_install_writes_named_group_and_is_idempotent(
        self, _config: Path,
    ) -> None:
        assert install_antigravity_hooks() == 0
        data = json.loads((_config / "hooks.json").read_text())
        group = data["iai-pme"]
        assert group["enabled"] is True
        assert any(
            "iai-mcp-antigravity-recall.sh" in h["command"]
            for e in group["PreInvocation"] for h in e["hooks"]
        )
        assert any(
            "iai-mcp-antigravity-capture.sh" in h["command"]
            for e in group["Stop"] for h in e["hooks"]
        )
        assert install_antigravity_hooks() == 0
        assert json.loads((_config / "hooks.json").read_text()) == data

    def test_uninstall_removes_group_keeps_foreign(self, _config: Path) -> None:
        _config.mkdir(parents=True)
        (_config / "hooks.json").write_text(json.dumps({
            "other-group": {"enabled": True},
        }))
        install_antigravity_hooks()
        uninstall_antigravity_hooks()
        data = json.loads((_config / "hooks.json").read_text())
        assert "iai-pme" not in data
        assert data["other-group"] == {"enabled": True}

    def test_status_roundtrip(self, _config: Path) -> None:
        assert status_antigravity_hooks() == 1
        install_antigravity_hooks()
        assert status_antigravity_hooks() == 0

    def test_install_refuses_unparseable_config(self, _config: Path) -> None:
        _config.mkdir(parents=True)
        corrupt = '{"other-group": {"enabled": true}'
        (_config / "hooks.json").write_text(corrupt)
        assert install_antigravity_hooks() == 1
        assert (_config / "hooks.json").read_text() == corrupt


class TestHermesInstaller:
    @pytest.fixture(autouse=True)
    def _home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        home = tmp_path / "hermes"
        monkeypatch.setenv("IAI_MCP_HERMES_HOME", str(home))
        return home

    def test_install_appends_block_and_is_idempotent(self, _home: Path) -> None:
        assert install_hermes_hooks() == 0
        text = (_home / "config.yaml").read_text()
        assert "pre_llm_call" in text
        assert "iai-mcp-hermes-recall.sh" in text
        assert "iai-mcp-hermes-capture.sh" in text
        assert install_hermes_hooks() == 0
        assert (_home / "config.yaml").read_text() == text

    def test_install_refuses_foreign_hooks_block(self, _home: Path) -> None:
        _home.mkdir(parents=True)
        original = "model: something\nhooks:\n  pre_tool_call:\n    - command: theirs\n"
        (_home / "config.yaml").write_text(original)
        assert install_hermes_hooks() == 1
        # The safety rule: an unmergeable config is never rewritten.
        assert (_home / "config.yaml").read_text() == original

    def test_install_preserves_foreign_top_level_keys(self, _home: Path) -> None:
        _home.mkdir(parents=True)
        (_home / "config.yaml").write_text("model: something\n")
        assert install_hermes_hooks() == 0
        text = (_home / "config.yaml").read_text()
        assert text.startswith("model: something\n")
        assert "hooks:" in text

    def test_uninstall_removes_exact_block(self, _home: Path) -> None:
        _home.mkdir(parents=True)
        (_home / "config.yaml").write_text("model: something\n")
        install_hermes_hooks()
        assert uninstall_hermes_hooks() == 0
        assert (_home / "config.yaml").read_text() == "model: something\n"

    def test_status_roundtrip(self, _home: Path) -> None:
        assert status_hermes_hooks() == 1
        install_hermes_hooks()
        assert status_hermes_hooks() == 0

    def test_refusal_catches_spaced_hooks_key(self, _home: Path) -> None:
        _home.mkdir(parents=True)
        # `hooks :` (space before colon) is the same YAML key — appending a
        # second hooks mapping would corrupt the file.
        original = "hooks :\n  pre_tool_call:\n    - command: theirs\n"
        (_home / "config.yaml").write_text(original)
        assert install_hermes_hooks() == 1
        assert (_home / "config.yaml").read_text() == original

    def test_recall_timeout_outlives_wrapper_subprocess(self, _home: Path) -> None:
        install_hermes_hooks()
        lines = (_home / "config.yaml").read_text().splitlines()
        idx = next(
            i for i, line in enumerate(lines)
            if "iai-mcp-hermes-recall.sh" in line
        )
        # Host timeout 30 > wrapper subprocess 25 > core CLI 10.
        assert "timeout: 30" in lines[idx + 1]


@pytest.mark.skipif(
    shutil.which("node") is None,
    reason="openclaw registration requires node on PATH",
)
class TestOpenclawInstaller:
    @pytest.fixture(autouse=True)
    def _env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        home = tmp_path / "openclaw"
        monkeypatch.setenv("IAI_MCP_OPENCLAW_HOME", str(home))
        wrapper = tmp_path / "index.js"
        wrapper.write_text("// bundled wrapper stub")
        monkeypatch.setenv("IAI_MCP_WRAPPER_PATH", str(wrapper))
        return home

    def test_install_registers_mcp_server_and_is_idempotent(
        self, _env: Path,
    ) -> None:
        assert install_openclaw_mcp() == 0
        data = json.loads((_env / "openclaw.json").read_text())
        entry = data["mcp"]["servers"]["iai-pme"]
        assert entry["args"][0].endswith("index.js")
        assert install_openclaw_mcp() == 0
        assert json.loads((_env / "openclaw.json").read_text()) == data

    def test_install_preserves_foreign_servers(self, _env: Path) -> None:
        _env.mkdir(parents=True)
        (_env / "openclaw.json").write_text(json.dumps({
            "mcp": {"servers": {"other": {"command": "x"}}},
            "agents": {"main": {}},
        }))
        assert install_openclaw_mcp() == 0
        data = json.loads((_env / "openclaw.json").read_text())
        assert data["mcp"]["servers"]["other"] == {"command": "x"}
        assert data["agents"] == {"main": {}}

    def test_uninstall_removes_only_our_server(self, _env: Path) -> None:
        install_openclaw_mcp()
        assert uninstall_openclaw_mcp() == 0
        data = json.loads((_env / "openclaw.json").read_text())
        assert "iai-pme" not in data["mcp"]["servers"]

    def test_status_roundtrip(self, _env: Path) -> None:
        assert status_openclaw_mcp() == 1
        install_openclaw_mcp()
        assert status_openclaw_mcp() == 0

    def test_install_refuses_unparseable_config(self, _env: Path) -> None:
        _env.mkdir(parents=True)
        corrupt = '{"mcp": {"servers": {"other": '
        (_env / "openclaw.json").write_text(corrupt)
        assert install_openclaw_mcp() == 1
        assert (_env / "openclaw.json").read_text() == corrupt
