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
import stat
from importlib import resources as _res
from pathlib import Path

_GROUP = "iai-pme"

_HOOK_SCRIPTS = (
    "iai-mcp-session-capture.sh",
    "iai-mcp-session-recall.sh",
    "iai-mcp-per-turn-recall.sh",
    "iai-mcp-antigravity-recall.sh",
    "iai-mcp-antigravity-capture.sh",
)

_EVENT_WIRING = (
    ("PreInvocation", "iai-mcp-antigravity-recall.sh", 30),
    ("Stop", "iai-mcp-antigravity-capture.sh", 35),
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
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def install_antigravity_hooks() -> int:
    hooks_dir, hooks_json = _antigravity_paths()

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
    group = data.setdefault(_GROUP, {})
    if not isinstance(group, dict):
        print(f"ERROR: {hooks_json} has an unexpected '{_GROUP}' shape — not touching it")
        return 1
    group["enabled"] = True
    changed = False
    for event, marker, timeout in _EVENT_WIRING:
        entries = group.setdefault(event, [])
        if any(
            marker in (h.get("command") or "")
            for e in entries
            if isinstance(e, dict)
            for h in (e.get("hooks") or [])
            if isinstance(h, dict)
        ):
            print(f"hooks.json already wires {marker} on {event} — no change")
            continue
        entries.append({
            "hooks": [{
                "type": "command",
                "command": f"bash {hooks_dir / marker}",
                "timeout": timeout,
            }]
        })
        changed = True
        print(f"patched: {hooks_json} ({event}: {marker})")

    if changed or not hooks_json.exists():
        hooks_json.parent.mkdir(parents=True, exist_ok=True)
        hooks_json.write_text(json.dumps(data, indent=2))

    print(
        "\nNote: verified against the Antigravity CLI (`agy`); the IDE "
        "shares this config dir but is untested. Restart the CLI to pick "
        "up hooks.json."
    )
    return 0


def uninstall_antigravity_hooks() -> int:
    hooks_dir, hooks_json = _antigravity_paths()

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
        print(f"NOT patched: cannot parse {hooks_json} — remove the {_GROUP} group by hand")
        return 1
    if _GROUP in data:
        data.pop(_GROUP, None)
        hooks_json.write_text(json.dumps(data, indent=2))
        print(f"patched: {hooks_json} ({_GROUP} group removed)")
    else:
        print(f"(no {_GROUP} group to remove) {hooks_json}")
    return 0


def status_antigravity_hooks() -> int:
    hooks_dir, hooks_json = _antigravity_paths()
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
    group = data.get(_GROUP, {})
    all_wired = isinstance(group, dict) and bool(group.get("enabled"))
    for event, marker, _timeout in _EVENT_WIRING:
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
