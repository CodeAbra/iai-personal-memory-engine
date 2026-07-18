"""`iai-mcp cowork install` must stage the plugin marketplace and wire it
into every Cowork home without clobbering foreign marketplaces or plugins;
uninstall must remove only iai-mcp keys and the staged tree."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest

from iai_mcp.cli._cowork import (
    MARKETPLACE_NAME,
    PLUGIN_KEY,
    PLUGIN_NAME,
    _cowork_session_roots,
    _plugin_marketplace_root,
    cmd_cowork_install,
    cmd_cowork_status,
    cmd_cowork_uninstall,
)

ACCT = "11111111-2222-4333-8444-555555555555"
DEVICE = "66666666-7777-4888-9999-aaaaaaaaaaaa"

FOREIGN_SETTINGS = {
    "extraKnownMarketplaces": {
        "knowledge-work-plugins": {
            "source": {"source": "github", "repo": "anthropics/knowledge-work-plugins"}
        }
    },
    "enabledPlugins": {"productivity@knowledge-work-plugins": True},
}


@pytest.fixture()
def cowork_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("APPDATA", str(home / "AppData" / "Roaming"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setattr(
        "iai_mcp.cli._capture._patch_claude_desktop_config",
        lambda action: "Claude Desktop: stubbed",
    )

    root = _cowork_session_roots()[0]
    dev = root / ACCT / DEVICE
    dev.mkdir(parents=True)
    (dev / "cowork_settings.json").write_text(json.dumps(FOREIGN_SETTINGS))

    # Non-UUID sibling and a bare UUID device without Cowork markers: both
    # must be left untouched by install.
    decoy = root / "skills-plugin" / DEVICE
    decoy.mkdir(parents=True)
    bare = root / ACCT / "bbbbbbbb-cccc-4ddd-8eee-ffffffffffff"
    bare.mkdir()

    return {"home": home, "dev": dev, "decoy": decoy, "bare": bare}


def _settings(dev: Path) -> dict:
    return json.loads((dev / "cowork_settings.json").read_text())


def test_install_stages_plugin_and_wires_home(cowork_env):
    rc = cmd_cowork_install(argparse.Namespace())
    assert rc == 0

    root = _plugin_marketplace_root()
    market = json.loads((root / ".claude-plugin" / "marketplace.json").read_text())
    assert market["name"] == MARKETPLACE_NAME
    assert market["plugins"][0]["name"] == PLUGIN_NAME

    plugin_dir = root / PLUGIN_NAME
    manifest = json.loads((plugin_dir / ".claude-plugin" / "plugin.json").read_text())
    assert manifest["name"] == PLUGIN_NAME

    hooks = json.loads((plugin_dir / "hooks" / "hooks.json").read_text())["hooks"]
    assert set(hooks) == {"SessionStart", "UserPromptSubmit", "Stop"}
    for entries in hooks.values():
        for entry in entries:
            for h in entry["hooks"]:
                assert h["type"] == "command"
                assert "${CLAUDE_PLUGIN_ROOT}/hooks/" in h["command"]
                script = h["command"].split("/hooks/")[1].rstrip('"')
                target = plugin_dir / "hooks" / script
                assert target.exists(), f"hooks.json references missing {script}"
                assert os.access(target, os.X_OK), f"{script} must be executable"

    data = _settings(cowork_env["dev"])
    assert data["extraKnownMarketplaces"][MARKETPLACE_NAME]["source"] == {
        "source": "directory",
        "path": str(root),
    }
    assert data["enabledPlugins"][PLUGIN_KEY] is True
    assert "knowledge-work-plugins" in data["extraKnownMarketplaces"]
    assert data["enabledPlugins"]["productivity@knowledge-work-plugins"] is True

    backup = cowork_env["dev"] / "cowork_settings.json.iai-backup"
    assert json.loads(backup.read_text()) == FOREIGN_SETTINGS


def test_install_is_idempotent(cowork_env):
    assert cmd_cowork_install(argparse.Namespace()) == 0
    first = _settings(cowork_env["dev"])
    backup = cowork_env["dev"] / "cowork_settings.json.iai-backup"
    first_backup = backup.read_text()

    assert cmd_cowork_install(argparse.Namespace()) == 0
    assert _settings(cowork_env["dev"]) == first
    assert backup.read_text() == first_backup, "backup must keep the pre-iai state"


def test_uninstall_removes_only_iai_keys(cowork_env):
    assert cmd_cowork_install(argparse.Namespace()) == 0
    assert cmd_cowork_uninstall(argparse.Namespace()) == 0

    data = _settings(cowork_env["dev"])
    assert MARKETPLACE_NAME not in data["extraKnownMarketplaces"]
    assert PLUGIN_KEY not in data["enabledPlugins"]
    assert "knowledge-work-plugins" in data["extraKnownMarketplaces"]
    assert data["enabledPlugins"]["productivity@knowledge-work-plugins"] is True
    assert not _plugin_marketplace_root().exists()


def test_install_skips_decoys_and_bare_dirs(cowork_env):
    assert cmd_cowork_install(argparse.Namespace()) == 0
    assert not (cowork_env["decoy"] / "cowork_settings.json").exists()
    assert not (cowork_env["bare"] / "cowork_settings.json").exists()


def test_install_creates_settings_for_marker_only_home(cowork_env):
    root = _cowork_session_roots()[0]
    fresh = root / ACCT / "12121212-3434-4565-8787-909090909090"
    fresh.mkdir()
    (fresh / ".claude.json").write_text("{}")

    assert cmd_cowork_install(argparse.Namespace()) == 0
    data = json.loads((fresh / "cowork_settings.json").read_text())
    assert data["enabledPlugins"][PLUGIN_KEY] is True


def test_status_reflects_wiring(cowork_env, monkeypatch, tmp_path):
    desktop_cfg = tmp_path / "claude_desktop_config.json"
    desktop_cfg.write_text(json.dumps({"mcpServers": {"iai-mcp": {}}}))
    monkeypatch.setattr(
        "iai_mcp.cli._claude_desktop_config_path", lambda: desktop_cfg
    )

    assert cmd_cowork_status(argparse.Namespace()) == 1
    assert cmd_cowork_install(argparse.Namespace()) == 0
    assert cmd_cowork_status(argparse.Namespace()) == 0

    assert cmd_cowork_uninstall(argparse.Namespace()) == 0
    assert cmd_cowork_status(argparse.Namespace()) == 1
