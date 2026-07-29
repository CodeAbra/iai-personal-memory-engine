"""The shipped Claude Code plugin must stay consistent with the package.

The plugin carries its own copies of the hook scripts (a plugin has to be
self-contained), so byte-identity with the packaged originals is pinned here:
a fix applied to one copy and not the other is the class of drift these
assertions exist to catch.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN = REPO_ROOT / "plugin"
PACKAGED_HOOKS = REPO_ROOT / "src" / "iai_mcp" / "_deploy" / "hooks"
HOOK_SCRIPTS = (
    "iai-mcp-session-recall.sh",
    "iai-mcp-turn-capture.sh",
    "iai-mcp-per-turn-recall.sh",
    "iai-mcp-session-capture.sh",
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_and_marketplace_agree() -> None:
    manifest = _json(PLUGIN / ".claude-plugin" / "plugin.json")
    marketplace = _json(REPO_ROOT / ".claude-plugin" / "marketplace.json")

    entries = [p for p in marketplace["plugins"] if p["name"] == manifest["name"]]
    assert entries, f"{manifest['name']} is missing from marketplace.json"
    entry = entries[0]
    assert entry["source"] == "./plugin"
    assert entry["description"] == manifest["description"], (
        "the marketplace entry and the manifest describe the same plugin; "
        "keep the description identical so the catalog cannot go stale"
    )


def test_manifest_version_tracks_the_package() -> None:
    manifest = _json(PLUGIN / ".claude-plugin" / "plugin.json")
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for line in pyproject.splitlines():
        if line.startswith("version = "):
            package_version = line.split('"')[1]
            break
    else:  # pragma: no cover - pyproject always declares a version
        pytest.fail("pyproject.toml has no version")
    assert manifest["version"] == package_version, (
        "a version mismatch between the plugin manifest and the package is the "
        "most common reason a marketplace submission is rejected"
    )


@pytest.mark.parametrize("script", HOOK_SCRIPTS)
def test_plugin_hook_scripts_match_the_packaged_originals(script: str) -> None:
    plugin_copy = (PLUGIN / "hooks" / script).read_bytes()
    packaged = (PACKAGED_HOOKS / script).read_bytes()
    assert plugin_copy == packaged, (
        f"{script} differs between plugin/hooks/ and the package; copy the "
        "packaged original over the plugin copy so both ship the same fix"
    )


def test_declared_hooks_exist_and_are_executable() -> None:
    hooks = _json(PLUGIN / "hooks" / "hooks.json")["hooks"]
    referenced = {
        Path(entry["command"].split("/hooks/")[-1].rstrip('"')).name
        for group in hooks.values()
        for matcher in group
        for entry in matcher["hooks"]
    }
    assert referenced == set(HOOK_SCRIPTS), referenced
    for script in HOOK_SCRIPTS:
        path = PLUGIN / "hooks" / script
        assert path.is_file(), path
        assert path.stat().st_mode & stat.S_IXUSR, f"{script} is not executable"


def test_mcp_launcher_is_executable_and_declared() -> None:
    mcp = _json(PLUGIN / ".mcp.json")["mcpServers"]
    assert set(mcp) == {"iai-pme"}, mcp
    command = mcp["iai-pme"]["command"]
    assert command.startswith("${CLAUDE_PLUGIN_ROOT}/"), command
    launcher = PLUGIN / command.split("${CLAUDE_PLUGIN_ROOT}/", 1)[1]
    assert launcher.is_file(), launcher
    assert os.access(launcher, os.X_OK), f"{launcher} is not executable"
