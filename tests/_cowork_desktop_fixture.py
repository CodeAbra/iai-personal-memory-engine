"""Desktop-layout fixture builder for the Cowork integration tests.

Builds fake Claude application-support trees under a caller-supplied
``tmp_path`` root, for the three Desktop on-disk layouts this phase must
work against, plus both Desktop session-record shapes and the two
hook-receipt kinds a firing hook writes. Every writer is pure file IO;
nothing here reads or writes a real Claude installation.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

# Fixed UUID-shaped literals so callers can reference the same account and
# user identity across a whole test module without regenerating it.
ACCOUNT_ID = "11111111-2222-4333-8444-555555555555"
USER_ID = "66666666-7777-4888-9999-aaaaaaaaaaaa"


def build_desktop_tree(root: Path, *, layout: str = "current") -> dict:
    """Create a fake Claude application-support tree under ``root``.

    ``layout`` selects one of three on-disk shapes:
      - ``current``: the build with a server-issued ``rpm/manifest.json``
        entry and no legacy marketplaces directory.
      - ``legacy_140``: an older build carrying a stale
        ``cowork_plugins/marketplaces/`` tree and ``known_marketplaces.json``,
        and no ``rpm/`` tree.
      - ``windows``: the same relative structure, rooted under an
        ``AppData/Roaming/Claude/`` prefix, with none of the marker files
        present.

    Returns a dict of the paths created; a key is ``None`` when its artifact
    does not exist for the requested layout.
    """
    if layout not in ("current", "legacy_140", "windows"):
        raise ValueError(f"unknown layout: {layout!r}")

    base = root / "AppData" / "Roaming" / "Claude" if layout == "windows" else root

    agent_root = base / "local-agent-mode-sessions"
    code_root = base / "claude-code-sessions"
    device_home = agent_root / ACCOUNT_ID / USER_ID
    code_session_home = code_root / ACCOUNT_ID / USER_ID
    device_home.mkdir(parents=True, exist_ok=True)
    code_session_home.mkdir(parents=True, exist_ok=True)

    paths: dict = {
        "base": base,
        "device_home": device_home,
        "code_session_home": code_session_home,
        "cowork_settings": None,
        "cowork_plugins": None,
        "rpm_manifest": None,
        "marketplaces_dir": None,
        "known_marketplaces": None,
        "agent_config_dir": None,
    }

    if layout == "windows":
        # Marker-absent by design: neither cowork_settings.json nor
        # cowork_plugins/ nor .claude.json exists under device_home.
        return paths

    settings_path = device_home / "cowork_settings.json"
    settings_path.write_text(
        json.dumps({"extraKnownMarketplaces": {}, "enabledPlugins": {}}),
        encoding="utf-8",
    )
    paths["cowork_settings"] = settings_path

    plugins_dir = device_home / "cowork_plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    paths["cowork_plugins"] = plugins_dir

    agent_config_dir = device_home / "agent" / f"local_ditto_{USER_ID}" / ".claude"
    agent_config_dir.mkdir(parents=True, exist_ok=True)
    paths["agent_config_dir"] = agent_config_dir

    if layout == "current":
        rpm_manifest = device_home / "rpm" / "manifest.json"
        rpm_manifest.parent.mkdir(parents=True, exist_ok=True)
        rpm_manifest.write_text(
            json.dumps({"plugins": [{"id": "plugin_server-issued", "version": "1.0.0"}]}),
            encoding="utf-8",
        )
        paths["rpm_manifest"] = rpm_manifest
    else:  # legacy_140
        marketplaces_dir = plugins_dir / "marketplaces"
        marketplaces_dir.mkdir(parents=True, exist_ok=True)
        paths["marketplaces_dir"] = marketplaces_dir
        known_marketplaces = plugins_dir / "known_marketplaces.json"
        known_marketplaces.write_text(
            json.dumps({"knowledge-work-plugins": {"source": "github"}}),
            encoding="utf-8",
        )
        paths["known_marketplaces"] = known_marketplaces

    return paths


def write_code_session(paths: dict, *, session_id: str, cwd: str, last_activity: datetime) -> Path:
    """Write one Desktop code-session record. No ``sessionType`` key: its
    absence is the discriminator against an agent-session record."""
    code_session_home = paths["code_session_home"]
    code_session_home.mkdir(parents=True, exist_ok=True)
    record = {
        "sessionId": session_id,
        "cwd": cwd,
        "createdAt": last_activity.isoformat(),
        "lastActivityAt": last_activity.isoformat(),
        "title": f"session {session_id}",
    }
    dst = code_session_home / f"local_{session_id}.json"
    dst.write_text(json.dumps(record), encoding="utf-8")
    return dst


def write_agent_session(
    paths: dict,
    *,
    session_id: str,
    cli_session_id: str,
    cwd: str,
    last_activity: datetime,
    plugin_install_paths: list,
) -> Path:
    """Write one Desktop agent ("ditto") session record. Carries a
    ``sessionType`` of ``agent`` -- the discriminator against a code-session
    record."""
    device_home = paths["device_home"]
    agent_dir = device_home / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "sessionId": session_id,
        "cliSessionId": cli_session_id,
        "sessionType": "agent",
        "hostLoopMode": True,
        "vmProcessName": "inspiring-brave-bohr",
        "cwd": cwd,
        "createdAt": last_activity.isoformat(),
        "lastActivityAt": last_activity.isoformat(),
        "pluginInstallPaths": list(plugin_install_paths),
    }
    dst = agent_dir / f"local_ditto_{USER_ID}.json"
    dst.write_text(json.dumps(record), encoding="utf-8")
    return dst


def write_hook_receipt(iai_home: Path, *, kind: str, session_id: str, channel: str, when: datetime) -> Path:
    """Append one hook-receipt line under ``iai_home/logs/``, mirroring the
    daily-log naming and the per-invocation line grammar the packaged hook
    scripts write.

    ``kind`` is ``"recall"`` for a SessionStart receipt or ``"capture"`` for
    a per-turn capture receipt.
    """
    if kind == "recall":
        log_name = f"recall-{when:%Y-%m-%d}.log"
        line = f"{when:%Y-%m-%dT%H:%M:%SZ} session={session_id} source=startup channel={channel}"
    elif kind == "capture":
        log_name = f"turn-capture-{when:%Y-%m-%d}.log"
        line = f"{when:%Y-%m-%dT%H:%M:%SZ} session={session_id} rc=0 channel={channel}"
    else:
        raise ValueError(f"unknown receipt kind: {kind!r}")

    logs_dir = iai_home / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / log_name
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return log_path
