"""Antigravity hook wiring (the `agy` CLI; the IDE shares the config dir).

Registration lives in `~/.gemini/config/hooks.json` as named groups:
`{"iai-pme": {"enabled": true, "<Event>": [{"hooks": [{"type": "command",
"command": ..., "timeout": N}]}]}}`. Only five events exist and none is a
session start, so the recall wrapper multiplexes on the invocation
counter; capture rides Stop through a key-translating wrapper that also
swaps in `transcript_full.jsonl` (the short transcript truncates fields).
"""
from __future__ import annotations

import json
import os
import platform
import stat
from importlib import resources as _res
from pathlib import Path

from iai_mcp.cli._atomic_write import _atomic_write_text
from iai_mcp.cli._hook_quoting import quote_posix, quote_powershell

_GROUP = "iai-pme"

_IS_WIN = platform.system() == "Windows"
_EXT = ".ps1" if _IS_WIN else ".sh"
_CMD_PREFIX = (
    "powershell.exe -ExecutionPolicy Bypass -NoProfile -File " if _IS_WIN else "bash "
)
_quote = quote_powershell if _IS_WIN else quote_posix

# POSIX installs the Antigravity pair PLUS the Claude-Code trio they
# delegate to; the PowerShell pair is self-contained and Windows instead
# needs the Task Scheduler scanner script.
_POSIX_HOOK_SCRIPTS = (
    "iai-mcp-session-capture",
    "iai-mcp-session-recall",
    "iai-mcp-per-turn-recall",
    "iai-mcp-antigravity-recall",
    "iai-mcp-antigravity-capture",
)
_WIN_HOOK_SCRIPTS = (
    "iai-mcp-antigravity-recall",
    "iai-mcp-antigravity-capture",
    "iai-mcp-auto-scan-antigravity",
)
_HOOK_SCRIPTS = _WIN_HOOK_SCRIPTS if _IS_WIN else _POSIX_HOOK_SCRIPTS

_EVENT_WIRING = (
    ("PreInvocation", f"iai-mcp-antigravity-recall{_EXT}", 30),
    ("Stop", f"iai-mcp-antigravity-capture{_EXT}", 35),
)


def _antigravity_config_dir() -> Path:
    env = os.environ.get("IAI_MCP_ANTIGRAVITY_CONFIG_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".gemini" / "config"


def _antigravity_paths() -> "tuple[Path, Path]":
    config_dir = _antigravity_config_dir()
    return config_dir / "iai-hooks", config_dir / "hooks.json"


def _load_hooks_json(path: Path) -> "dict | None":
    """None means the file exists but is unreadable/unparseable — callers
    must refuse to rewrite it, or foreign groups would be silently lost."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _entry_matches_exact(entry: dict, exact_command: str) -> bool:
    return any(
        (h.get("command") or "") == exact_command
        for h in (entry.get("hooks") or [])
        if isinstance(h, dict)
    )


def _find_hook_with_marker(entries: list, marker: str) -> "dict | None":
    """Return the first hook dict whose command contains marker, or None.

    Substring match on the script filename -- the same shape as the
    capture-hooks install path -- so a stale command is found (and can be
    healed) instead of exact-matched away.
    """
    for entry in entries:
        for h in (entry.get("hooks") or []):
            if isinstance(h, dict) and marker in (h.get("command") or ""):
                return h
    return None


def install_antigravity_hooks() -> int:
    hooks_dir, hooks_json = _antigravity_paths()

    templates = _res.files("iai_mcp") / "_deploy" / "hooks"
    missing = [n + _EXT for n in _HOOK_SCRIPTS if not (templates / (n + _EXT)).exists()]
    if missing:
        print(f"ERROR: hook templates missing in package data: {missing}")
        return 1

    hooks_dir.mkdir(parents=True, exist_ok=True)
    for base_name in _HOOK_SCRIPTS:
        name = base_name + _EXT
        dst = hooks_dir / name
        dst.write_bytes((templates / name).read_bytes())
        dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
        print(f"installed: {dst}")

    data = _load_hooks_json(hooks_json)
    if data is None:
        print(f"ERROR: cannot parse {hooks_json} — fix or remove it, then re-run")
        return 1
    group = data.setdefault(_GROUP, {})
    if not isinstance(group, dict):
        print(f"ERROR: {hooks_json} has an unexpected '{_GROUP}' shape — not touching it")
        return 1
    group["enabled"] = True
    changed = False
    for event, marker, timeout in _EVENT_WIRING:
        entries = group.setdefault(event, [])
        exact_command = f"{_CMD_PREFIX}{_quote(str(hooks_dir / marker))}"
        existing = _find_hook_with_marker(entries, marker)
        if existing is not None:
            if existing.get("command") != exact_command:
                existing["command"] = exact_command
                changed = True
                print(f"patched: {hooks_json} ({event}: {marker} command healed)")
            else:
                print(f"hooks.json already wires {marker} on {event} — no change")
            continue
        entries.append({
            "hooks": [{
                "type": "command",
                "command": f"{_CMD_PREFIX}{hooks_dir / marker}",
                "timeout": timeout,
            }]
        })
        changed = True
        print(f"patched: {hooks_json} ({event}: {marker})")

    if changed or not hooks_json.exists():
        hooks_json.parent.mkdir(parents=True, exist_ok=True)
        hooks_json.write_text(json.dumps(data, indent=2), encoding="utf-8")

    if _IS_WIN:
        print("\nSetting up auto-scan for Antigravity GUI (Desktop App) users...")
        scan_script = hooks_dir / "iai-mcp-auto-scan-antigravity.ps1"
        if not scan_script.exists():
            # The copy loop above places it; reaching here means the install
            # is broken and the user must know — a silent skip would report
            # a scheduled scanner that does not exist.
            print(f"ERROR: scanner script missing after install: {scan_script}")
            return 1
        import subprocess

        tmpl = (
            _res.files("iai_mcp") / "_deploy" / "windows"
            / "iai-mcp-antigravity-scan.xml"
        )
        from iai_mcp.cli._daemon import _scope_logon_trigger_to_user

        text = tmpl.read_text(encoding="utf-8")
        text = text.replace("{SCAN_SCRIPT}", str(scan_script))
        # Deliberately NOT caught below: a resolution failure here must
        # raise uncaught (mirrors the daemon install path), before xml_out
        # is written or schtasks runs — a prior unscoped task must never be
        # left registered under an easily-ignored WARNING+rc=0.
        text = _scope_logon_trigger_to_user(text)

        xml_out = Path.home() / ".iai-mcp" / "iai-mcp-antigravity-scan.task.xml"
        xml_out.parent.mkdir(parents=True, exist_ok=True)
        xml_out.write_text(text, encoding="utf-16")

        try:
            subprocess.run(
                ["schtasks", "/Create", "/TN", "IaiMcpAntigravityAutoScan",
                 "/XML", str(xml_out), "/F"],
                check=True, stdout=subprocess.DEVNULL,
            )
            print(
                "status: ACTIVE — Antigravity GUI auto-scanner scheduled in "
                "Windows Task Scheduler (runs every 30m)."
            )
        except Exception as e:  # noqa: BLE001 -- scheduler absence must not fail install
            print(f"WARNING: failed to create scheduled task: {e}")

    print(
        "\nNote: verified against the Antigravity CLI (`agy`) and the "
        "Antigravity GUI (Desktop App)."
        "\nRestart the CLI or let the background scanner handle the GUI logs."
    )
    return 0


def uninstall_antigravity_hooks() -> int:
    hooks_dir, hooks_json = _antigravity_paths()

    for base_name in _HOOK_SCRIPTS:
        name = base_name + _EXT
        dst = hooks_dir / name
        if dst.exists():
            dst.unlink()
            print(f"removed: {dst}")
        else:
            print(f"(not present) {dst}")

    if _IS_WIN:
        import subprocess
        try:
            subprocess.run(
                ["schtasks", "/Delete", "/TN", "IaiMcpAntigravityAutoScan", "/F"],
                check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            print("removed scheduled task: IaiMcpAntigravityAutoScan")
        except Exception:  # noqa: BLE001 -- absent task is a clean uninstall
            pass

    if not hooks_json.exists():
        print(f"(not present) {hooks_json}")
        return 0

    data = _load_hooks_json(hooks_json)
    if data is None:
        print(f"NOT patched: cannot parse {hooks_json} — remove the {_GROUP} group by hand")
        return 1
    if _GROUP in data:
        data.pop(_GROUP, None)
        hooks_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"patched: {hooks_json} ({_GROUP} group removed)")
    else:
        print(f"(no {_GROUP} group to remove) {hooks_json}")
    return 0


def status_antigravity_hooks() -> int:
    hooks_dir, hooks_json = _antigravity_paths()
    templates = _res.files("iai_mcp") / "_deploy" / "hooks"

    all_installed = True
    for base_name in _HOOK_SCRIPTS:
        name = base_name + _EXT
        src_ok = (templates / name).exists()
        dst_ok = (hooks_dir / name).exists()
        all_installed = all_installed and dst_ok
        print(
            f"{name}: template {'PRESENT' if src_ok else 'MISSING'}, "
            f"installed {'PRESENT' if dst_ok else 'MISSING'} ({hooks_dir / name})"
        )

    data = _load_hooks_json(hooks_json)
    if data is None:
        print(f"WARNING: cannot parse {hooks_json}")
        data = {}
    group = data.get(_GROUP, {})
    all_wired = isinstance(group, dict) and bool(group.get("enabled"))
    for event, marker, _timeout in _EVENT_WIRING:
        exact_command = f"{_CMD_PREFIX}{_quote(str(hooks_dir / marker))}"
        wired = isinstance(group, dict) and any(
            marker in (h.get("command") or "")
            for e in group.get(event, [])
            if isinstance(e, dict)
            for h in (e.get("hooks") or [])
            if isinstance(h, dict)
        )
        all_wired = all_wired and wired
        print(
            f"Antigravity hooks.json {event} ({marker}): "
            f"{'WIRED' if wired else 'NOT WIRED'}"
        )

    if all_installed and all_wired:
        print("\nstatus: ACTIVE — Antigravity hooks installed and wired")
        return 0
    print(
        "\nstatus: INACTIVE — run: iai-mcp capture-hooks install --target antigravity"
    )
    return 1
