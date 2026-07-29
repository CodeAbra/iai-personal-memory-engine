"""Claude Cowork (desktop local-agent mode) integration commands.

Cowork sessions run with an isolated, per-session config dir, so hooks
installed under ~/.claude/ never fire there. The shared channel Cowork does
load for every session is its plugin system: this module materializes an
iai-mcp plugin (hooks + scripts) into a local plugin marketplace under
~/.iai-mcp/ and registers it in each Cowork home's cowork_settings.json.
"""

from __future__ import annotations

import argparse
import importlib.resources as _res
import json
import logging
import os
import re
import shutil
import stat
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

MARKETPLACE_NAME = "iai-mcp-local"
PLUGIN_NAME = "iai-mcp"
PLUGIN_KEY = f"{PLUGIN_NAME}@{MARKETPLACE_NAME}"

_HOOK_SCRIPTS = (
    "iai-mcp-session-recall.sh",
    "iai-mcp-turn-capture.sh",
    "iai-mcp-per-turn-recall.sh",
    "iai-mcp-session-capture.sh",
)

_UUIDISH = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)


def _plugin_marketplace_root() -> Path:
    return Path.home() / ".iai-mcp" / "claude-plugin"


def _cowork_session_roots() -> list[Path]:
    home = Path.home()
    if sys.platform == "darwin":
        return [home / "Library" / "Application Support" / "Claude" / "local-agent-mode-sessions"]
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else home / "AppData" / "Roaming"
        return [base / "Claude" / "local-agent-mode-sessions"]
    return [home / ".config" / "Claude" / "local-agent-mode-sessions"]


def _looks_like_cowork_home(d: Path) -> bool:
    return (
        (d / "cowork_settings.json").exists()
        or (d / "cowork_plugins").is_dir()
        or (d / ".claude.json").exists()
    )


def _discover_cowork_homes() -> list[Path]:
    homes: list[Path] = []
    for root in _cowork_session_roots():
        if not root.is_dir():
            continue
        try:
            accounts = sorted(root.iterdir())
        except OSError:
            continue
        for acct in accounts:
            if not acct.is_dir() or not _UUIDISH.match(acct.name):
                continue
            try:
                devices = sorted(acct.iterdir())
            except OSError:
                continue
            for dev in devices:
                if not dev.is_dir() or not _UUIDISH.match(dev.name):
                    continue
                if _looks_like_cowork_home(dev):
                    homes.append(dev)
    return homes


def _plugin_version() -> str:
    try:
        from iai_mcp import _distribution_version

        resolved = _distribution_version()
        if resolved != "0+unknown":
            return resolved
    except Exception:
        pass
    return "0.0.0"


def _hook_entry(script: str, timeout: int) -> dict:
    return {
        "type": "command",
        "command": f'bash "${{CLAUDE_PLUGIN_ROOT}}/hooks/{script}"',
        "timeout": timeout,
    }


def _build_hooks_json() -> dict:
    return {
        "hooks": {
            "SessionStart": [
                {
                    "matcher": "startup|resume|clear|compact",
                    "hooks": [_hook_entry("iai-mcp-session-recall.sh", 30)],
                }
            ],
            "UserPromptSubmit": [
                {
                    "hooks": [
                        _hook_entry("iai-mcp-turn-capture.sh", 5),
                        _hook_entry("iai-mcp-per-turn-recall.sh", 5),
                    ]
                }
            ],
            "Stop": [{"hooks": [_hook_entry("iai-mcp-session-capture.sh", 35)]}],
        }
    }


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.parent / (path.name + ".tmp-iai")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def _materialize_plugin() -> Path:
    root = _plugin_marketplace_root()
    plugin_dir = root / PLUGIN_NAME
    (root / ".claude-plugin").mkdir(parents=True, exist_ok=True)
    (plugin_dir / ".claude-plugin").mkdir(parents=True, exist_ok=True)
    (plugin_dir / "hooks").mkdir(parents=True, exist_ok=True)

    _write_json_atomic(
        root / ".claude-plugin" / "marketplace.json",
        {
            "name": MARKETPLACE_NAME,
            "owner": {"name": "iai-mcp"},
            "plugins": [
                {
                    "name": PLUGIN_NAME,
                    "source": f"./{PLUGIN_NAME}",
                    "description": (
                        "Persistent memory for Claude: per-turn context injection "
                        "and ambient capture backed by the iai-mcp store."
                    ),
                }
            ],
        },
    )
    _write_json_atomic(
        plugin_dir / ".claude-plugin" / "plugin.json",
        {
            "name": PLUGIN_NAME,
            "version": _plugin_version(),
            "description": (
                "iai-mcp ambient memory hooks: session-start recall, per-turn "
                "context injection, per-response capture."
            ),
        },
    )
    _write_json_atomic(plugin_dir / "hooks" / "hooks.json", _build_hooks_json())

    for script in _HOOK_SCRIPTS:
        src = _res.files("iai_mcp") / "_deploy" / "hooks" / script
        if not src.exists():
            raise FileNotFoundError(f"hook template missing in package data: {src}")
        dst = plugin_dir / "hooks" / script
        dst.write_bytes(src.read_bytes())
        dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return root


def _plugin_materialized() -> bool:
    root = _plugin_marketplace_root()
    plugin_dir = root / PLUGIN_NAME
    needed = [
        root / ".claude-plugin" / "marketplace.json",
        plugin_dir / ".claude-plugin" / "plugin.json",
        plugin_dir / "hooks" / "hooks.json",
    ] + [plugin_dir / "hooks" / s for s in _HOOK_SCRIPTS]
    return all(p.exists() for p in needed)


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _home_wired(home: Path) -> bool:
    data = _load_json(home / "cowork_settings.json")
    markets = data.get("extraKnownMarketplaces") or {}
    enabled = data.get("enabledPlugins") or {}
    return MARKETPLACE_NAME in markets and bool(enabled.get(PLUGIN_KEY))


def _patch_cowork_settings(home: Path, action: str) -> str:
    settings_path = home / "cowork_settings.json"
    exists = settings_path.exists()

    if action == "uninstall" and not exists:
        return f"{home}: no cowork_settings.json — skipped"

    data = _load_json(settings_path) if exists else {}
    markets = data.setdefault("extraKnownMarketplaces", {})
    enabled = data.setdefault("enabledPlugins", {})

    if action == "uninstall":
        changed = False
        if MARKETPLACE_NAME in markets:
            markets.pop(MARKETPLACE_NAME, None)
            changed = True
        for key in [k for k in enabled if k.endswith(f"@{MARKETPLACE_NAME}")]:
            enabled.pop(key, None)
            changed = True
        if not changed:
            return f"{home}: iai-mcp not wired — no change"
        _write_json_atomic(settings_path, data)
        return f"{home}: iai-mcp plugin removed"

    entry = {
        "source": {
            "source": "directory",
            "path": str(_plugin_marketplace_root()),
        }
    }
    if markets.get(MARKETPLACE_NAME) == entry and enabled.get(PLUGIN_KEY) is True:
        return f"{home}: already wired — no change"

    if exists:
        backup = settings_path.parent / (settings_path.name + ".iai-backup")
        if not backup.exists():
            try:
                backup.write_bytes(settings_path.read_bytes())
            except OSError:
                pass

    markets[MARKETPLACE_NAME] = entry
    enabled[PLUGIN_KEY] = True
    _write_json_atomic(settings_path, data)
    return f"{home}: iai-mcp plugin wired"


def _seed_cli_path_cache() -> None:
    # The hook scripts resolve the CLI through this cache before falling back
    # to a PATH scan; seeding it here makes them work in harnesses that strip
    # the login PATH (Cowork sessions do).
    try:
        cache = Path.home() / ".iai-mcp" / ".cli-path"
        if cache.exists():
            cached = cache.read_text().strip()
            if cached and os.access(cached, os.X_OK):
                return
        candidate = Path(sys.argv[0]).resolve()
        if candidate.name == "iai-mcp" and os.access(candidate, os.X_OK):
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(str(candidate))
    except OSError:
        pass


def cmd_cowork_install(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    from iai_mcp.cli._capture import _patch_claude_desktop_config

    try:
        root = _materialize_plugin()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=_cli.sys.stderr)
        return 1
    print(f"materialized: {root} (marketplace '{MARKETPLACE_NAME}', plugin '{PLUGIN_NAME}')")

    _seed_cli_path_cache()

    homes = _discover_cowork_homes()
    if not homes:
        print(
            "no Cowork homes found — Claude Desktop not installed or Cowork "
            "never used on this machine. The plugin is staged; rerun install "
            "after the first Cowork session."
        )
    for home in homes:
        print(_patch_cowork_settings(home, "install"))

    print(_patch_claude_desktop_config("install"))

    print("\nNext: fully quit + relaunch the Claude desktop app (macOS: `killall Claude`),")
    print("      then start a NEW Cowork session so it loads the plugin.")
    print("Verify: iai-mcp cowork status")
    return 0


def cmd_cowork_uninstall(args: argparse.Namespace) -> int:
    for home in _discover_cowork_homes():
        print(_patch_cowork_settings(home, "uninstall"))

    root = _plugin_marketplace_root()
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
        print(f"removed: {root}")
    else:
        print(f"(not present) {root}")
    # The claude_desktop_config.json MCP entry also serves Claude Desktop chat
    # and Claude Code; removing it belongs to `capture-hooks uninstall`.
    return 0


def cmd_cowork_status(args: argparse.Namespace) -> int:
    materialized = _plugin_materialized()
    root = _plugin_marketplace_root()
    print(f"plugin marketplace:   {root}  {'PRESENT' if materialized else 'MISSING'}")

    homes = _discover_cowork_homes()
    if not homes:
        print("Cowork homes:         none found (desktop app absent or Cowork unused)")
    wired_any = False
    for home in homes:
        wired = _home_wired(home)
        wired_any = wired_any or wired
        print(f"Cowork home:          {home}  {'WIRED' if wired else 'NOT WIRED'}")

    desktop_cfg = None
    try:
        from iai_mcp import cli as _cli

        desktop_cfg = _cli._claude_desktop_config_path()
    except Exception:
        pass
    if desktop_cfg is None or not desktop_cfg.exists():
        print("Claude Desktop MCP:   config absent")
        desktop_wired = False
    else:
        desktop_wired = "iai-mcp" in (_load_json(desktop_cfg).get("mcpServers") or {})
        print(f"Claude Desktop MCP:   {desktop_cfg}  {'WIRED' if desktop_wired else 'NOT WIRED'}")

    if materialized and homes and wired_any and desktop_wired:
        print("\nstatus: ACTIVE — plugin staged and wired into Cowork")
        return 0
    print("\nstatus: INACTIVE — run: iai-mcp cowork install")
    return 1
