"""Cursor hook wiring: core scripts plus a session-recall envelope wrapper.

Cursor registers hooks in `~/.cursor/hooks.json`
(`{"version": 1, "hooks": {"<event>": [{"command": ...}]}}`) and its hook
payloads already carry `session_id` and `transcript_path`, so the core
capture scripts run unchanged. Only session-start injection needs the
wrapper: Cursor expects `{"additional_context": ...}` on stdout, not raw
markdown. `beforeSubmitPrompt` cannot inject, so per-turn recall is not
wired on this host — the working-tier slice would be silently discarded.
"""
from __future__ import annotations

import json
import os
import stat
from importlib import resources as _res
from pathlib import Path

_HOOK_SCRIPTS = (
    "iai-mcp-session-capture.sh",
    "iai-mcp-turn-capture.sh",
    "iai-mcp-session-recall.sh",
    "iai-mcp-cursor-session-recall.sh",
    "iai-mcp-cursor-turn-capture.sh",
)

_EVENT_WIRING = (
    ("sessionStart", "iai-mcp-cursor-session-recall.sh"),
    ("beforeSubmitPrompt", "iai-mcp-cursor-turn-capture.sh"),
    ("stop", "iai-mcp-session-capture.sh"),
    ("sessionEnd", "iai-mcp-session-capture.sh"),
)


def _cursor_home() -> Path:
    env = os.environ.get("IAI_MCP_CURSOR_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cursor"


def _cursor_paths() -> "tuple[Path, Path]":
    home = _cursor_home()
    return home / "iai-hooks", home / "hooks.json"


def _load_hooks_json(path: Path) -> "dict | None":
    """None means the file exists but is unreadable/unparseable — callers
    must refuse to rewrite it, or foreign entries would be silently lost."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def install_cursor_hooks() -> int:
    hooks_dir, hooks_json = _cursor_paths()

    templates = _res.files("iai_mcp") / "_deploy" / "hooks"
    missing = [n for n in _HOOK_SCRIPTS if not (templates / n).exists()]
    if missing:
        print(f"ERROR: hook templates missing in package data: {missing}")
        return 1

    hooks_dir.mkdir(parents=True, exist_ok=True)
    for name in _HOOK_SCRIPTS:
        dst = hooks_dir / name
        dst.write_bytes((templates / name).read_bytes())
        dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
        print(f"installed: {dst}")

    data = _load_hooks_json(hooks_json)
    if data is None:
        print(f"ERROR: cannot parse {hooks_json} — fix or remove it, then re-run")
        return 1
    data.setdefault("version", 1)
    events = data.setdefault("hooks", {})
    if not isinstance(events, dict):
        print(f"ERROR: {hooks_json} has an unexpected 'hooks' shape — not touching it")
        return 1
    changed = False
    for event, marker in _EVENT_WIRING:
        entries = events.setdefault(event, [])
        if any(
            marker in (e.get("command") or "")
            for e in entries
            if isinstance(e, dict)
        ):
            print(f"hooks.json already wires {marker} on {event} — no change")
            continue
        entries.append({"command": f"bash {hooks_dir / marker}"})
        changed = True
        print(f"patched: {hooks_json} ({event}: {marker})")

    if changed or not hooks_json.exists():
        hooks_json.parent.mkdir(parents=True, exist_ok=True)
        hooks_json.write_text(json.dumps(data, indent=2))

    print(
        "\nNote: restart Cursor to pick up hooks.json. Per-turn recall is "
        "not available on Cursor (beforeSubmitPrompt cannot inject); "
        "session-start recall and ambient capture are."
    )
    return 0


def uninstall_cursor_hooks() -> int:
    hooks_dir, hooks_json = _cursor_paths()

    for name in _HOOK_SCRIPTS:
        dst = hooks_dir / name
        if dst.exists():
            dst.unlink()
            print(f"removed: {dst}")
        else:
            print(f"(not present) {dst}")

    if not hooks_json.exists():
        print(f"(not present) {hooks_json}")
        return 0

    data = _load_hooks_json(hooks_json)
    if data is None:
        print(f"NOT patched: cannot parse {hooks_json} — remove our entries by hand")
        return 1
    events = data.get("hooks", {})
    changed = False
    if isinstance(events, dict):
        for event in list(events):
            entries = events.get(event, [])
            kept = [
                e for e in entries
                if not (
                    isinstance(e, dict)
                    and any(m in (e.get("command") or "") for m in _HOOK_SCRIPTS)
                )
            ]
            if len(kept) != len(entries):
                if kept:
                    events[event] = kept
                else:
                    events.pop(event, None)
                changed = True
                print(f"patched: {hooks_json} ({event} entry removed)")
    if changed:
        hooks_json.write_text(json.dumps(data, indent=2))
    else:
        print(f"(no hook entry to remove) {hooks_json}")
    return 0


def status_cursor_hooks() -> int:
    hooks_dir, hooks_json = _cursor_paths()
    templates = _res.files("iai_mcp") / "_deploy" / "hooks"

    all_installed = True
    for name in _HOOK_SCRIPTS:
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
    events = data.get("hooks", {})
    all_wired = True
    for event, marker in _EVENT_WIRING:
        wired = isinstance(events, dict) and any(
            marker in (e.get("command") or "")
            for e in events.get(event, [])
            if isinstance(e, dict)
        )
        all_wired = all_wired and wired
        print(f"Cursor hooks.json {event} ({marker}): {'WIRED' if wired else 'NOT WIRED'}")

    if all_installed and all_wired:
        print("\nstatus: ACTIVE — Cursor hooks installed and wired")
        return 0
    print("\nstatus: INACTIVE — run: iai-mcp capture-hooks install --target cursor")
    return 1
