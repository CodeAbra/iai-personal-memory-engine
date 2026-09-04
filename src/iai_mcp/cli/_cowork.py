"""Claude Cowork (desktop local-agent mode) integration commands.

Code sessions run against the shared ~/.claude configuration home, served
by the capture-hooks command. Cowork and other local sessions are served by
a background sweep of the conversation transcripts the app already writes
to disk: `cowork install` sets up a scheduled process that reads new
content on an interval and hands it to the memory system.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from iai_mcp.transcript_sweep import SWEEP_SCHEDULED_ENV_VAR

logger = logging.getLogger(__name__)

MARKETPLACE_NAME = "iai-mcp-local"
PLUGIN_NAME = "iai-mcp"

_UUIDISH = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)

# Same restricted set the hook scripts enforce on session ids before using
# them in a filesystem path (iai-mcp-turn-capture.sh); every id crossing the
# Desktop-record / receipt-log boundary is re-validated against it here.
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _plugin_marketplace_root() -> Path:
    return Path.home() / ".iai-mcp" / "claude-plugin"


def _claude_app_support_roots() -> list[Path]:
    home = Path.home()
    if sys.platform == "darwin":
        return [home / "Library" / "Application Support" / "Claude"]
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else home / "AppData" / "Roaming"
        return [base / "Claude"]
    return [home / ".config" / "Claude"]


def _cowork_session_roots() -> list[Path]:
    return [p / "local-agent-mode-sessions" for p in _claude_app_support_roots()]


# ============================================================================
# The background sweeper: a scheduled process, not a task inside the memory
# daemon. Its only job is to invoke the transcript-sweep producer on an
# interval -- the interval itself lives nowhere but the rendered unit below,
# so there is exactly one authority for the cadence.
# ============================================================================

SWEEPER_LAUNCHD_LABEL = "com.iai-mcp.transcript-sweep"
SWEEPER_SYSTEMD_SERVICE_NAME = "iai-mcp-transcript-sweep.service"
SWEEPER_SYSTEMD_TIMER_NAME = "iai-mcp-transcript-sweep.timer"
SWEEPER_ENABLED_FLAG_NAME = ".cowork-sweep-enabled"

# Seconds between sweeps. Lives only in the rendered unit text below -- no
# environment variable governs it, so a running unit's own file is always
# the single source of truth for how often it fires.
_SWEEPER_INTERVAL_SEC = 120


def _sweeper_enabled_flag_path(home: Path) -> Path:
    return home / ".iai-mcp" / SWEEPER_ENABLED_FLAG_NAME


def _sweeper_unit_paths(home: Path) -> dict:
    """Platform-correct unit file location(s) and identifying label(s) for
    the background sweeper. ``kind`` is ``None`` on a platform with no
    supported scheduler -- callers must decline cleanly rather than write
    anything."""
    if sys.platform == "darwin":
        return {
            "kind": "launchd",
            "label": SWEEPER_LAUNCHD_LABEL,
            "unit": home / "Library" / "LaunchAgents" / f"{SWEEPER_LAUNCHD_LABEL}.plist",
        }
    if sys.platform.startswith("linux"):
        unit_dir = home / ".config" / "systemd" / "user"
        return {
            "kind": "systemd",
            "service_name": SWEEPER_SYSTEMD_SERVICE_NAME,
            "timer_name": SWEEPER_SYSTEMD_TIMER_NAME,
            "service": unit_dir / SWEEPER_SYSTEMD_SERVICE_NAME,
            "timer": unit_dir / SWEEPER_SYSTEMD_TIMER_NAME,
        }
    return {"kind": None}


def _sweeper_unit_targets(paths: dict) -> dict:
    """Every on-disk file `paths` describes, keyed by role -- the set
    install writes and uninstall removes, so the two can never drift."""
    if paths.get("kind") == "launchd":
        return {"unit": paths["unit"]}
    if paths.get("kind") == "systemd":
        return {"service": paths["service"], "timer": paths["timer"]}
    return {}


def _sweeper_program_args() -> list[str]:
    return [sys.executable, "-m", "iai_mcp.cli", "transcript-sweep", "run"]


def _render_sweeper_launchd_plist(home: Path) -> str:
    log_dir = home / ".iai-mcp" / "logs"
    args_xml = "\n".join(f"        <string>{a}</string>" for a in _sweeper_program_args())
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"\n'
        '  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0">\n'
        "<dict>\n"
        "    <key>Label</key>\n"
        f"    <string>{SWEEPER_LAUNCHD_LABEL}</string>\n\n"
        "    <key>ProgramArguments</key>\n"
        "    <array>\n"
        f"{args_xml}\n"
        "    </array>\n\n"
        "    <key>StartInterval</key>\n"
        f"    <integer>{_SWEEPER_INTERVAL_SEC}</integer>\n\n"
        "    <key>RunAtLoad</key>\n"
        "    <true/>\n\n"
        "    <key>StandardOutPath</key>\n"
        f"    <string>{log_dir / 'transcript-sweep-stdout.log'}</string>\n"
        "    <key>StandardErrorPath</key>\n"
        f"    <string>{log_dir / 'transcript-sweep-stderr.log'}</string>\n\n"
        "    <key>EnvironmentVariables</key>\n"
        "    <dict>\n"
        "        <key>HOME</key>\n"
        f"        <string>{home}</string>\n"
        f"        <key>{SWEEP_SCHEDULED_ENV_VAR}</key>\n"
        "        <string>1</string>\n"
        "    </dict>\n"
        "</dict>\n"
        "</plist>\n"
    )


def _render_sweeper_systemd_service(home: Path) -> str:
    log_dir = home / ".iai-mcp" / "logs"
    program = " ".join(_sweeper_program_args())
    return (
        "[Unit]\n"
        "Description=iai-mcp background sweep -- reads new Claude "
        "conversations already on disk\n\n"
        "[Service]\n"
        "Type=oneshot\n"
        f"Environment={SWEEP_SCHEDULED_ENV_VAR}=1\n"
        f"ExecStart={program}\n"
        f"StandardOutput=append:{log_dir / 'transcript-sweep-stdout.log'}\n"
        f"StandardError=append:{log_dir / 'transcript-sweep-stderr.log'}\n"
    )


def _render_sweeper_systemd_timer() -> str:
    return (
        "[Unit]\n"
        "Description=Periodic trigger for the iai-mcp background sweep\n\n"
        "[Timer]\n"
        f"OnUnitActiveSec={_SWEEPER_INTERVAL_SEC}\n"
        f"OnBootSec={_SWEEPER_INTERVAL_SEC}\n"
        "Persistent=true\n"
        f"Unit={SWEEPER_SYSTEMD_SERVICE_NAME}\n\n"
        "[Install]\n"
        "WantedBy=timers.target\n"
    )


def _render_sweeper_unit(home: Path) -> dict:
    """The unit's text, keyed the same way `_sweeper_unit_targets` keys its
    file paths -- a launchd plist on macOS, a systemd service plus timer on
    Linux, empty on an unsupported platform."""
    paths = _sweeper_unit_paths(home)
    if paths["kind"] == "launchd":
        return {"unit": _render_sweeper_launchd_plist(home)}
    if paths["kind"] == "systemd":
        return {
            "service": _render_sweeper_systemd_service(home),
            "timer": _render_sweeper_systemd_timer(),
        }
    return {}


def _write_sweeper_unit(home: Path, paths: dict) -> bool:
    """Write the rendered unit file(s) atomically. Returns True when every
    target already existed before this call -- an idempotent re-install."""
    targets = _sweeper_unit_targets(paths)
    rendered = _render_sweeper_unit(home)
    already_present = bool(targets) and all(t.exists() for t in targets.values())
    for key, target in targets.items():
        content = rendered[key]
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.parent / (target.name + ".tmp-iai")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, target)
        try:
            os.chmod(target, 0o644)
        except OSError:
            pass
    return already_present


def _issue_sweeper_load(paths: dict) -> None:
    """Load the just-written sweeper unit through the platform scheduler.

    A thin process boundary -- tests monkeypatch this whole function rather
    than mocking individual subprocess calls, keeping the unit-render tests
    free of scheduler plumbing.
    """
    if paths.get("kind") == "launchd":
        uid = os.getuid()
        target = paths["unit"]
        subprocess.run(
            ["launchctl", "bootout", f"gui/{uid}", str(target)],
            check=False, capture_output=True,
        )
        subprocess.run(
            ["launchctl", "bootstrap", f"gui/{uid}", str(target)],
            check=False, capture_output=True,
        )
        subprocess.run(
            ["launchctl", "kickstart", f"gui/{uid}/{paths['label']}"],
            check=False, capture_output=True,
        )
    elif paths.get("kind") == "systemd":
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"], check=False, capture_output=True,
        )
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", paths["timer_name"]],
            check=False, capture_output=True,
        )


def _issue_sweeper_unload(paths: dict) -> None:
    """Unload the sweeper unit through the platform scheduler. Every
    failure mode -- the unit already gone, the scheduler binary absent --
    is swallowed here; uninstall must never abort on this step."""
    if paths.get("kind") == "launchd":
        uid = os.getuid()
        target = paths.get("unit")
        subprocess.run(
            ["launchctl", "bootout", f"gui/{uid}", str(target)],
            check=False, capture_output=True,
        )
    elif paths.get("kind") == "systemd":
        subprocess.run(
            ["systemctl", "--user", "disable", "--now", paths["timer_name"]],
            check=False, capture_output=True,
        )
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"], check=False, capture_output=True,
        )


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


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.parent / (path.name + ".tmp-iai")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _parse_iso_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


@dataclass(frozen=True)
class DesktopSession:
    id: str
    kind: str  # "code" or "agent"
    cwd: str | None
    last_activity: datetime | None


def _iter_account_user_dirs(root: Path):
    if not root.is_dir():
        return
    try:
        accounts = sorted(root.iterdir())
    except OSError:
        return
    for acct in accounts:
        if not acct.is_dir() or not _UUIDISH.match(acct.name):
            continue
        try:
            users = sorted(acct.iterdir())
        except OSError:
            continue
        for user in users:
            if user.is_dir() and _UUIDISH.match(user.name):
                yield user


def _iter_local_json_files(user_dir: Path) -> list[Path]:
    try:
        return [p for p in sorted(user_dir.glob("**/local_*.json")) if p.is_file()]
    except OSError:
        return []


def _session_from_record(data: dict) -> DesktopSession | None:
    kind = "agent" if "sessionType" in data else "code"
    # The id a transcript-sweep receipt carries is the CLI-facing session id
    # (the ".jsonl" filename stem). Desktop's own "sessionId" field can carry
    # a "local_" prefix that never matches that stem, on both session shapes
    # -- "cliSessionId" is the field that agrees with it whenever Desktop
    # records one, so it is preferred regardless of kind.
    cli_id = data.get("cliSessionId")
    if isinstance(cli_id, str) and _SAFE_ID.match(cli_id):
        raw_id = cli_id
    else:
        raw_id = data.get("sessionId")
    if not isinstance(raw_id, str) or not _SAFE_ID.match(raw_id):
        return None
    cwd = data.get("cwd")
    cwd = cwd if isinstance(cwd, str) else None
    return DesktopSession(
        id=raw_id,
        kind=kind,
        cwd=cwd,
        last_activity=_parse_iso_datetime(data.get("lastActivityAt")),
    )


def _read_desktop_sessions(roots) -> list[DesktopSession]:
    """Read-only cross-reference to Desktop's own session records.

    Returns an empty list rather than raising for a missing root, an
    unreadable or malformed record, or a directory where a file was
    expected; every id is validated before it is returned.
    """
    sessions: list[DesktopSession] = []
    for base in roots:
        for session_root in (base / "claude-code-sessions", base / "local-agent-mode-sessions"):
            for user_dir in _iter_account_user_dirs(session_root):
                for entry in _iter_local_json_files(user_dir):
                    data = _load_json(entry)
                    record = _session_from_record(data)
                    if record is not None:
                        sessions.append(record)
    return sessions


_RECEIPT_LOG_KINDS = {
    "recall": "recall",
    "turn-capture": "capture",
    "transcript-sweep": "transcript-sweep",
}
_LOG_NAME_RE = re.compile(r"^(recall|turn-capture|transcript-sweep)-\d{4}-\d{2}-\d{2}\.log$")
_TS_FIELD_RE = re.compile(r"^(\S+)")
_SESSION_FIELD_RE = re.compile(r"(?:^|\s)session=(\S+)")
_CHANNEL_FIELD_RE = re.compile(r"(?:^|\s)channel=(\S+)")


@dataclass(frozen=True)
class HookReceipt:
    session_id: str
    channel: str
    timestamp: datetime | None
    kind: str
    log_name: str


def _read_hook_receipts(iai_home: Path, *, days: int) -> list[HookReceipt]:
    """Read-only scan of the daily hook-receipt logs under the iai home.

    Returns an empty list rather than raising for a missing log directory;
    an unparseable or truncated line is skipped, never raised on.
    """
    logs_dir = iai_home / "logs"
    if not logs_dir.is_dir():
        return []
    try:
        log_files = sorted(logs_dir.iterdir())
    except OSError:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    receipts: list[HookReceipt] = []
    for log_path in log_files:
        if not log_path.is_file():
            continue
        name_match = _LOG_NAME_RE.match(log_path.name)
        if not name_match:
            continue
        kind = _RECEIPT_LOG_KINDS[name_match.group(1)]
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            session_match = _SESSION_FIELD_RE.search(line)
            channel_match = _CHANNEL_FIELD_RE.search(line)
            if not session_match or not channel_match:
                continue
            raw_id = session_match.group(1)
            if not _SAFE_ID.match(raw_id):
                continue
            ts_match = _TS_FIELD_RE.match(line)
            timestamp = _parse_iso_datetime(ts_match.group(1)) if ts_match else None
            if timestamp is not None and timestamp < cutoff:
                continue
            receipts.append(
                HookReceipt(
                    session_id=raw_id,
                    channel=channel_match.group(1),
                    timestamp=timestamp,
                    kind=kind,
                    log_name=log_path.name,
                )
            )
    return receipts


_DESKTOP_APP_INFO_PLIST = Path("/Applications/Claude.app/Contents/Info.plist")


def _desktop_build_version() -> str | None:
    if sys.platform != "darwin":
        return None
    try:
        import plistlib

        with _DESKTOP_APP_INFO_PLIST.open("rb") as f:
            data = plistlib.load(f)
    except (OSError, ValueError, KeyError):
        return None
    version = data.get("CFBundleShortVersionString")
    return version if isinstance(version, str) and version else None


TIER_ACTIVE = "ACTIVE"
TIER_STAGED = "STAGED"
TIER_NOT_INSTALLED = "NOT_INSTALLED"

# Absorbs clock skew between a hook's log line and Desktop's own
# lastActivityAt; not env-overridable -- a widenable tolerance is a
# tolerance that can be widened until any receipt matches any session. A
# receipt written by the nightly sleep-pipeline backstop, rather than the
# awake courier, lands hours after lastActivityAt and will not fall inside
# this window -- a backstop-only capture reports STAGED, not ACTIVE, until
# the awake courier or another session refreshes the receipt. This
# understates capture rather than overstating it.
_ACTIVITY_WINDOW_TOLERANCE = timedelta(minutes=10)

# The channel a transcript-sweep receipt line's own channel= field carries
# (iai_mcp.transcript_sweep._RECEIPT_CHANNEL) -- the settled capture
# mechanism for local Cowork and other local-agent sessions.
_AGENT_RECEIPT_CHANNEL = "transcript-sweep"
_AGENT_CHANNEL_LABEL = "transcript-sweep"


@dataclass(frozen=True)
class AgentChannelState:
    tier: str
    build: str | None
    staged_shape: str
    receipt: HookReceipt | None


def _agent_channel_state(
    *,
    staged: bool,
    build: str | None,
    receipts: list[HookReceipt],
    sessions: list[DesktopSession],
) -> AgentChannelState:
    """Pure function from already-gathered evidence to a verdict.

    No filesystem, clock, or environment access -- every branch here is
    unit-testable with hand-built evidence. The transcript-sweep channel
    reads transcript files present on every Desktop build, so no build
    gates it here; a detected build is carried through only as
    informational output on the returned state.
    """
    sessions_by_id = {s.id: s for s in sessions}
    matched: HookReceipt | None = None
    for receipt in receipts:
        if receipt.channel != _AGENT_RECEIPT_CHANNEL:
            continue
        session = sessions_by_id.get(receipt.session_id)
        if session is None or session.last_activity is None or receipt.timestamp is None:
            continue
        if abs(receipt.timestamp - session.last_activity) > _ACTIVITY_WINDOW_TOLERANCE:
            continue
        matched = receipt
        break

    if matched is not None:
        return AgentChannelState(
            tier=TIER_ACTIVE, build=build, staged_shape=_AGENT_CHANNEL_LABEL, receipt=matched
        )
    if staged:
        return AgentChannelState(
            tier=TIER_STAGED, build=build, staged_shape=_AGENT_CHANNEL_LABEL, receipt=None
        )
    return AgentChannelState(
        tier=TIER_NOT_INSTALLED, build=build, staged_shape=_AGENT_CHANNEL_LABEL, receipt=None
    )


def _strip_marketplace_and_plugin(data: dict, *, marketplace_name: str) -> bool:
    """Remove one marketplace's registration and every enabled-plugin key
    that names it, in place. Returns whether anything changed. Every key
    it does not own -- a foreign marketplace, a foreign plugin -- survives
    untouched."""
    changed = False
    markets = data.get("extraKnownMarketplaces")
    if isinstance(markets, dict) and marketplace_name in markets:
        markets.pop(marketplace_name, None)
        changed = True
    enabled = data.get("enabledPlugins")
    if isinstance(enabled, dict):
        for key in [k for k in enabled if k.endswith(f"@{marketplace_name}")]:
            enabled.pop(key, None)
            changed = True
    return changed


def _patch_cowork_settings(home: Path) -> str:
    """Strip the legacy directory-marketplace registration an older version
    wrote into a Cowork home's cowork_settings.json. Removes only keys this
    tool owns; every foreign key survives byte-identically."""
    settings_path = home / "cowork_settings.json"
    if not settings_path.exists():
        return f"{home}: no cowork_settings.json -- skipped"

    data = _load_json(settings_path)
    if not _strip_marketplace_and_plugin(data, marketplace_name=MARKETPLACE_NAME):
        return f"{home}: iai-mcp not wired -- no change"
    _write_json_atomic(settings_path, data)
    return f"{home}: iai-mcp plugin removed"


def _seed_cli_path_cache() -> None:
    # The hook scripts resolve the CLI through this cache before falling back
    # to a PATH scan; seeding it here makes them work in harnesses that strip
    # the login PATH (Cowork sessions do).
    try:
        cache = Path.home() / ".iai-mcp" / ".cli-path"
        if cache.exists():
            cached = cache.read_text(encoding="utf-8").strip()
            if cached and os.access(cached, os.X_OK):
                return
        candidate = Path(sys.argv[0]).resolve()
        if candidate.name == "iai-mcp" and os.access(candidate, os.X_OK):
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(str(candidate), encoding="utf-8")
    except OSError:
        pass


def cmd_cowork_install(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    from iai_mcp.cli._capture import _patch_claude_desktop_config

    home = Path.home()
    paths = _sweeper_unit_paths(home)
    if paths["kind"] is None:
        print(
            "ERROR: no supported background scheduler on this platform "
            "(need launchd on macOS or a systemd user session on Linux) -- "
            "the memory system's own overnight consolidation still sweeps "
            "new conversations in, just not continuously through the day.",
            file=_cli.sys.stderr,
        )
        return 1

    targets = _sweeper_unit_targets(paths)
    already_present = _write_sweeper_unit(home, paths)

    flag = _sweeper_enabled_flag_path(home)
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.touch(exist_ok=True)

    _issue_sweeper_load(paths)

    unit_list = ", ".join(str(t) for t in targets.values())
    if already_present:
        print(f"already installed: {unit_list}")
    else:
        print(f"installed: {unit_list}")

    _seed_cli_path_cache()
    print(_patch_claude_desktop_config("install"))

    print(
        "\nThe memory system will keep reading new Claude conversations in "
        "the background."
    )
    print("Verify: iai-mcp cowork status")
    return 0


# Fixed paths a retired reachability check staged; uninstall still removes
# and detects them. Nothing in this module writes under these names.
_RETIRED_CHECK_PLUGIN_NAME = "iai-mcp-cowork-probe"
_RETIRED_CHECK_LOCAL_MARKETPLACE = "local-desktop-app-uploads"


def _remove_retired_channel_artifacts(home: Path) -> list[str]:
    """Removes every artifact the retired reachability check may have
    staged under a Cowork home. Fixed-path removal only -- the exact-
    content restoration machinery that check used retired along with it,
    so a settings.json entry is stripped by key, never restored from a
    backup sidecar."""
    messages: list[str] = []

    marketplace_dir = (
        home / "cowork_plugins" / "marketplaces" / _RETIRED_CHECK_LOCAL_MARKETPLACE
    )
    if marketplace_dir.exists():
        shutil.rmtree(marketplace_dir, ignore_errors=True)
        messages.append(f"removed: {marketplace_dir}")

    sandbox_plugin_dir = home / "cowork_plugins" / _RETIRED_CHECK_PLUGIN_NAME
    if sandbox_plugin_dir.exists():
        shutil.rmtree(sandbox_plugin_dir, ignore_errors=True)
        messages.append(f"removed: {sandbox_plugin_dir}")

    synced_root = (
        home.parent.parent.parent / "synced_plugins" / home.parent.name
        / _RETIRED_CHECK_PLUGIN_NAME
    )
    if synced_root.exists():
        shutil.rmtree(synced_root, ignore_errors=True)
        messages.append(f"removed: {synced_root}")

    settings_path = home / "cowork_settings.json"
    if settings_path.exists():
        data = _load_json(settings_path)
        if _strip_marketplace_and_plugin(
            data, marketplace_name=_RETIRED_CHECK_LOCAL_MARKETPLACE
        ):
            _write_json_atomic(settings_path, data)
            messages.append(f"{home}: retired reachability-check registration removed")

    return messages


def _detect_retired_channel_hook_mutation(home: Path) -> list[str]:
    """Warns about, but cannot reverse, the retired reachability check's
    agent-settings candidate: an in-place ``hooks.SessionStart`` mutation
    in an agent config's own settings.json. Reversing an unknown hook
    entry safely would require the exact-content backup/restore machinery
    that check used, which retired along with it -- so this only detects
    and reports the condition."""
    agent_dir = home / "agent"
    if not agent_dir.is_dir():
        return []
    try:
        ditto_dirs = sorted(p for p in agent_dir.iterdir() if p.is_dir())
    except OSError:
        return []

    messages: list[str] = []
    for ditto_dir in ditto_dirs:
        settings_path = ditto_dir / ".claude" / "settings.json"
        if not settings_path.exists():
            continue
        data = _load_json(settings_path)
        if not isinstance(data, dict):
            continue
        session_start = data.get("hooks", {})
        session_start = session_start.get("SessionStart", []) if isinstance(session_start, dict) else []
        if not isinstance(session_start, list):
            continue
        referenced = any(
            _RETIRED_CHECK_PLUGIN_NAME in str(hook.get("command", ""))
            for entry in session_start
            if isinstance(entry, dict)
            for hook in entry.get("hooks", [])
            if isinstance(hook, dict)
        )
        if referenced:
            messages.append(
                f"warning: {settings_path} still references the retired "
                f"{_RETIRED_CHECK_PLUGIN_NAME} hook script under "
                "hooks.SessionStart -- this cannot be removed automatically; "
                "edit that file by hand and delete the matching array entry"
            )
    return messages


def cmd_cowork_uninstall(args: argparse.Namespace) -> int:
    """Reverses everything any version of this tool ever wrote: the loaded
    sweeper unit, its unit file, the enablement flag, the legacy directory-
    marketplace and enabled-plugins entries an older version wrote, and the
    retired reachability check's own artifacts. One known exception is not
    reversed but is reported: the retired check's in-place settings.json
    hook mutation. Every failure is reported, never raised -- this command
    always reaches its last line."""
    messages: list[str] = []
    home = Path.home()

    # 1) the loaded sweeper unit
    paths = _sweeper_unit_paths(home)
    targets = _sweeper_unit_targets(paths)
    if targets:
        try:
            _issue_sweeper_unload(paths)
        except OSError as exc:
            messages.append(f"warning: could not unload sweeper unit: {exc}")
    else:
        messages.append("(unsupported platform) no sweeper unit to unload")

    # 2) the unit file(s) on disk
    for target in targets.values():
        if target.exists():
            try:
                target.unlink()
                messages.append(f"removed: {target}")
            except OSError as exc:
                messages.append(f"warning: could not remove {target}: {exc}")
        else:
            messages.append(f"(not present) {target}")

    # 3) the enablement flag
    flag = _sweeper_enabled_flag_path(home)
    if flag.exists():
        try:
            flag.unlink()
            messages.append(f"removed: {flag}")
        except OSError as exc:
            messages.append(f"warning: could not remove {flag}: {exc}")
    else:
        messages.append(f"(not present) {flag}")

    # 4) the legacy directory-marketplace and enabled-plugins entries an
    #    older version wrote, and the retired reachability check's own
    #    per-home artifacts
    for cowork_home in _discover_cowork_homes():
        messages.append(_patch_cowork_settings(cowork_home))
        messages.extend(_remove_retired_channel_artifacts(cowork_home))
        messages.extend(_detect_retired_channel_hook_mutation(cowork_home))

    # 5) any plugin marketplace tree and the retired reachability check's
    #    host-side script tree, each removed if still present
    root = _plugin_marketplace_root()
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
        messages.append(f"removed: {root}")
    else:
        messages.append(f"(not present) {root}")

    check_host_tree = home / ".iai-mcp" / "cowork-probe"
    if check_host_tree.exists():
        shutil.rmtree(check_host_tree, ignore_errors=True)
        messages.append(f"removed: {check_host_tree}")
    else:
        messages.append(f"(not present) {check_host_tree}")

    for m in messages:
        print(m)
    # The claude_desktop_config.json MCP entry also serves Claude Desktop chat
    # and Claude Code; removing it belongs to `capture-hooks uninstall`.
    return 0


_RECEIPT_LOOKBACK_DAYS = 7


def cmd_cowork_status(args: argparse.Namespace) -> int:
    from iai_mcp.cli._capture import cmd_capture_hooks_status

    print("Cowork code sessions (capture-hooks channel -- shared ~/.claude home):")
    cmd_capture_hooks_status(argparse.Namespace())
    print()

    print(f"Cowork agent sessions ({_AGENT_CHANNEL_LABEL} channel):")
    home = Path.home()
    staged = _sweeper_enabled_flag_path(home).exists()
    build = _desktop_build_version()
    sessions = _read_desktop_sessions(_claude_app_support_roots())
    receipts = _read_hook_receipts(home / ".iai-mcp", days=_RECEIPT_LOOKBACK_DAYS)
    state = _agent_channel_state(
        staged=staged,
        build=build,
        receipts=receipts,
        sessions=sessions,
    )

    build_label = state.build if state.build else "unknown (not determined on this platform)"
    print(f"tier:                 {state.tier}")
    print(f"detected build:       {build_label} (informational only -- not gated on)")
    print(f"channel:              {state.staged_shape}")
    if state.tier == TIER_ACTIVE and state.receipt is not None:
        print(
            f"receipt:              {state.receipt.kind} receipt on session "
            f"{state.receipt.session_id}, cross-referenced against a Desktop-recorded session"
        )

    if state.tier == TIER_STAGED:
        print(
            "next: the background sweep is enabled but no receipt has landed "
            "yet -- run a local Cowork session, then check again"
        )
    elif state.tier == TIER_NOT_INSTALLED:
        print("next: run: iai-mcp cowork install")

    print(
        "note: exit status reflects only the agent-session tier above; "
        "check the code-session surface separately with: iai-mcp capture-hooks status"
    )
    return 0 if state.tier == TIER_ACTIVE else 1
