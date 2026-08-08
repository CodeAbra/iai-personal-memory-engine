"""Hermes hook wiring (Nous Research terminal agent, >= 0.5.0).

Hermes registers shell hooks in a `hooks:` block of `~/.hermes/config.yaml`.
There is no YAML dependency in this package, so the installer does a
text-level merge with one hard safety rule: if the config already has a
top-level `hooks:` key that we did not write, the file is left alone and
the block is printed for the user to merge by hand — appending a second
`hooks:` mapping would corrupt valid YAML.

Recall rides `pre_llm_call` ({"context": ...} envelope, first-turn gated
inside the wrapper); capture rides `on_session_end` reading Hermes'
`state.db` — the host keeps no transcript file.
"""
from __future__ import annotations

import os
import re
import stat
from importlib import resources as _res
from pathlib import Path

_MARKER = "iai-mcp-hermes"

_HOOK_SCRIPTS = (
    "iai-mcp-session-recall.sh",
    "iai-mcp-per-turn-recall.sh",
    "iai-mcp-hermes-recall.sh",
    "iai-mcp-hermes-capture.sh",
)


def _hermes_home() -> Path:
    env = os.environ.get("IAI_MCP_HERMES_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".hermes"


def _hermes_paths() -> "tuple[Path, Path]":
    home = _hermes_home()
    return home / "agent-hooks", home / "config.yaml"


def _hooks_block(hooks_dir: Path) -> str:
    return (
        "hooks:\n"
        "  pre_llm_call:\n"
        f"    - command: \"{hooks_dir / 'iai-mcp-hermes-recall.sh'}\"\n"
        # Timeout ladder must stay outermost-largest: host 30 > wrapper
        # subprocess 25 > core CLI 10 — an inverted ladder kills every
        # non-cache recall before it can land.
        "      timeout: 30\n"
        "  on_session_end:\n"
        f"    - command: \"{hooks_dir / 'iai-mcp-hermes-capture.sh'}\"\n"
        "      timeout: 35\n"
    )


def install_hermes_hooks() -> int:
    hooks_dir, config = _hermes_paths()

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

    block = _hooks_block(hooks_dir)
    existing = ""
    if config.exists():
        try:
            existing = config.read_text(encoding="utf-8")
        except OSError as e:
            print(f"ERROR: cannot read {config}: {e}")
            return 1

    if _MARKER in existing:
        print(f"config.yaml already wires {_MARKER} hooks — no change")
    elif any(re.match(r"hooks\s*:", line) for line in existing.splitlines()):
        # A foreign hooks: mapping exists — a second one would corrupt the
        # YAML, and rewriting theirs without a YAML parser is guesswork.
        print(
            f"NOT patched: {config} already has a `hooks:` block.\n"
            "Merge these entries into it by hand:\n\n" + block
        )
        return 1
    else:
        config.parent.mkdir(parents=True, exist_ok=True)
        joiner = "" if (not existing or existing.endswith("\n")) else "\n"
        config.write_text(existing + joiner + block, encoding="utf-8")
        print(f"patched: {config} (hooks block appended)")

    print(
        "\nNote: requires Hermes >= 0.5.0 (earlier builds documented these "
        "events but never fired them). First use prompts for hook consent — "
        "verify with `hermes hooks test pre_llm_call`."
    )
    return 0


def uninstall_hermes_hooks() -> int:
    hooks_dir, config = _hermes_paths()

    for name in _HOOK_SCRIPTS:
        dst = hooks_dir / name
        if dst.exists():
            dst.unlink()
            print(f"removed: {dst}")
        else:
            print(f"(not present) {dst}")

    if not config.exists():
        print(f"(not present) {config}")
        return 0
    try:
        existing = config.read_text(encoding="utf-8")
    except OSError as e:
        print(f"ERROR: cannot read {config}: {e}")
        return 1
    if _MARKER not in existing:
        print(f"(no hook entry to remove) {config}")
        return 0
    block = _hooks_block(_hermes_paths()[0])
    if block in existing:
        config.write_text(existing.replace(block, "", 1), encoding="utf-8")
        print(f"patched: {config} (hooks block removed)")
        return 0
    print(
        f"NOT patched: {config} mentions {_MARKER} but not as the exact "
        "block this installer wrote — remove the entries by hand."
    )
    return 1


def status_hermes_hooks() -> int:
    hooks_dir, config = _hermes_paths()
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

    wired = False
    if config.exists():
        try:
            wired = _MARKER in config.read_text(encoding="utf-8")
        except OSError:
            wired = False
    print(f"Hermes config.yaml ({_MARKER}): {'WIRED' if wired else 'NOT WIRED'}")

    if all_installed and wired:
        print("\nstatus: ACTIVE — Hermes hooks installed and wired (needs Hermes >= 0.5.0)")
        return 0
    print("\nstatus: INACTIVE — run: iai-mcp capture-hooks install --target hermes")
    return 1
