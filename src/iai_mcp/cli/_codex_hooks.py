"""Codex CLI hook wiring: the same four hook scripts, a different host.

Codex's hook protocol mirrors Claude Code's — command hooks receive one
JSON object on stdin carrying `session_id`, `transcript_path`, `cwd`,
plus `prompt` (UserPromptSubmit) and `source` (SessionStart) — so the
shipped hook scripts run unchanged. Only the registration differs:
Codex reads `~/.codex/hooks.json` (event -> matcher group -> handlers).

`transcript_path` may be null on Codex; every script already skips when
it is missing, so absent fields degrade to a no-op, never a crash.
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
    "iai-mcp-per-turn-recall.sh",
)

_EVENT_WIRING = (
    # (event, marker script, timeout seconds, matcher or None)
    ("Stop", "iai-mcp-session-capture.sh", 35, None),
    ("UserPromptSubmit", "iai-mcp-turn-capture.sh", 5, None),
    ("UserPromptSubmit", "iai-mcp-per-turn-recall.sh", 5, None),
    ("SessionStart", "iai-mcp-session-recall.sh", 30, "startup|resume|clear|compact"),
)


def _codex_home() -> Path:
    env = os.environ.get("IAI_MCP_CODEX_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".codex"


def _codex_paths() -> "tuple[Path, Path]":
    home = _codex_home()
    return home / "hooks", home / "hooks.json"


def _load_hooks_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _entry_has_marker(entry: dict, marker: str) -> bool:
    return any(
        marker in (h.get("command") or "")
        for h in (entry.get("hooks") or [])
        if isinstance(h, dict)
    )


def install_codex_hooks() -> int:
    hooks_dir, hooks_json = _codex_paths()

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
    events = data.setdefault("hooks", {})
    changed = False
    for event, marker, timeout, matcher in _EVENT_WIRING:
        entries = events.setdefault(event, [])
        if any(_entry_has_marker(e, marker) for e in entries if isinstance(e, dict)):
            print(f"hooks.json already wires {marker} on {event} — no change")
            continue
        entry: dict = {
            "hooks": [
                {
                    "type": "command",
                    "command": f"bash {hooks_dir / marker}",
                    "timeout": timeout,
                }
            ]
        }
        if matcher:
            entry["matcher"] = matcher
        entries.append(entry)
        changed = True
        print(f"patched: {hooks_json} ({event}: {marker})")

    if changed or not hooks_json.exists():
        hooks_json.parent.mkdir(parents=True, exist_ok=True)
        hooks_json.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print(
        "\nNote: Codex reads hooks.json at session start — restart Codex to "
        "pick this up. Memory tools over MCP are registered separately "
        "(`codex mcp add`)."
    )
    return 0


def uninstall_codex_hooks() -> int:
    hooks_dir, hooks_json = _codex_paths()

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
    events = data.get("hooks", {})
    changed = False
    for event in list(events):
        entries = events.get(event, [])
        kept = [
            e for e in entries
            if not (
                isinstance(e, dict)
                and any(_entry_has_marker(e, m) for m in _HOOK_SCRIPTS)
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
        hooks_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        print(f"(no hook entry to remove) {hooks_json}")
    return 0


def status_codex_hooks() -> int:
    hooks_dir, hooks_json = _codex_paths()
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
    events = data.get("hooks", {})
    all_wired = True
    for event, marker, _timeout, _matcher in _EVENT_WIRING:
        wired = any(
            _entry_has_marker(e, marker)
            for e in events.get(event, [])
            if isinstance(e, dict)
        )
        all_wired = all_wired and wired
        print(f"Codex hooks.json {event} ({marker}): {'WIRED' if wired else 'NOT WIRED'}")

    if all_installed and all_wired:
        print("\nstatus: ACTIVE — Codex hooks installed and wired")
        return 0
    print(
        "\nstatus: INACTIVE — run: iai-mcp capture-hooks install --target codex"
    )
    return 1
