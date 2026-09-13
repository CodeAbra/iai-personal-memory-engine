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

from iai_mcp.cli._atomic_write import _atomic_write_text

_HOOK_SCRIPTS = (
    "iai-mcp-session-capture.sh",
    "iai-mcp-turn-capture.sh",
    "iai-mcp-session-recall.sh",
    "iai-mcp-per-turn-recall.sh",
)

# Deployed alongside the hook scripts, in lockstep, but never wired as an
# event handler itself -- the per-turn-recall hook sys.path-inserts its own
# directory and imports this by name; a stale or missing copy silently
# drops the render step, not the whole accelerator.
_RECALL_RENDER_HELPER = "_recall_render.py"

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


def _load_hooks_json(path: Path) -> "dict | None":
    """None means the file exists but is unreadable/unparseable — callers
    must refuse to rewrite it, or foreign entries would be silently lost."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _own_commands(hooks_dir: Path) -> "dict[str, str]":
    return {name: f"bash {hooks_dir / name}" for name in _HOOK_SCRIPTS}


def _entry_matches_exact(entry: dict, exact_command: str) -> bool:
    return any(
        (h.get("command") or "") == exact_command
        for h in (entry.get("hooks") or [])
        if isinstance(h, dict)
    )


def _entry_is_near_miss(entry: dict, own_commands: "dict[str, str]") -> bool:
    exacts = set(own_commands.values())
    for h in entry.get("hooks") or []:
        if not isinstance(h, dict):
            continue
        cmd = h.get("command") or ""
        if cmd in exacts:
            continue
        if any(basename in cmd for basename in own_commands):
            return True
    return False


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

    render_src = templates / _RECALL_RENDER_HELPER
    if render_src.exists():
        render_dst = hooks_dir / _RECALL_RENDER_HELPER
        render_dst.write_bytes(render_src.read_bytes())
        print(f"installed: {render_dst}")
    else:
        print(f"WARN: recall render helper missing in package data: {render_src}")

    data = _load_hooks_json(hooks_json)
    if data is None:
        print(f"ERROR: cannot parse {hooks_json} — fix or remove it, then re-run")
        return 1
    events = data.setdefault("hooks", {})
    changed = False
    for event, marker, timeout, matcher in _EVENT_WIRING:
        entries = events.setdefault(event, [])
        exact_command = f"bash {hooks_dir / marker}"
        if any(_entry_matches_exact(e, exact_command) for e in entries if isinstance(e, dict)):
            print(f"hooks.json already wires {marker} on {event} — no change")
            continue
        entry: dict = {
            "hooks": [
                {
                    "type": "command",
                    "command": exact_command,
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
        _atomic_write_text(hooks_json, json.dumps(data, indent=2))

    print(
        "\nNote: Codex reads hooks.json at session start — restart Codex to "
        "pick this up. Memory tools over MCP are registered separately "
        "(`codex mcp add`)."
    )
    return 0


def uninstall_codex_hooks() -> int:
    hooks_dir, hooks_json = _codex_paths()

    for name in (*_HOOK_SCRIPTS, _RECALL_RENDER_HELPER):
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
    own_commands = _own_commands(hooks_dir)
    own_exacts = set(own_commands.values())
    events = data.get("hooks", {})
    changed = False
    near_miss_events: "list[str]" = []
    for event in list(events):
        entries = events.get(event, [])
        kept = []
        for e in entries:
            if not isinstance(e, dict):
                kept.append(e)
                continue
            if _entry_is_near_miss(e, own_commands):
                near_miss_events.append(event)
                kept.append(e)
                continue
            hooks = e.get("hooks") or []
            remaining = [
                h for h in hooks
                if not (isinstance(h, dict) and (h.get("command") or "") in own_exacts)
            ]
            if len(remaining) == len(hooks):
                kept.append(e)
                continue
            changed = True
            has_foreign_group_data = bool(set(e) - {"hooks", "matcher"})
            if remaining or has_foreign_group_data:
                kept.append({**e, "hooks": remaining})
        if kept != entries:
            if kept:
                events[event] = kept
            else:
                events.pop(event, None)
            print(f"patched: {hooks_json} ({event} entry removed)")
    if changed:
        _atomic_write_text(hooks_json, json.dumps(data, indent=2))
    else:
        print(f"(no hook entry to remove) {hooks_json}")

    if near_miss_events:
        print(
            f"NOT removed: {hooks_json} has entries under "
            f"{sorted(set(near_miss_events))} that mention our hook scripts "
            "but do not match exactly what this installer writes — remove "
            "them by hand."
        )
        return 1
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
    if data is None:
        print(f"WARNING: cannot parse {hooks_json}")
        data = {}
    events = data.get("hooks", {})
    all_wired = True
    for event, marker, _timeout, _matcher in _EVENT_WIRING:
        exact_command = f"bash {hooks_dir / marker}"
        wired = any(
            _entry_matches_exact(e, exact_command)
            for e in events.get(event, [])
            if isinstance(e, dict)
        )
        all_wired = all_wired and wired
        print(f"Codex hooks.json {event} ({marker}): {'WIRED' if wired else 'NOT WIRED'}")

    if all_installed and all_wired:
        print(
            "\nstatus: REGISTERED — our files and hooks.json entries are in "
            "place; whether Codex actually runs them is a separate question "
            "this local check does not answer."
        )
        return 0
    print(
        "\nstatus: NOT REGISTERED — run: iai-mcp capture-hooks install --target codex"
    )
    return 1
