"""Behavior coverage for the iai-tools-session-hierarchy SessionStart hook."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HOOK_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "iai_mcp" / "_deploy" / "hooks" / "iai-tools-session-hierarchy.sh"
)
BASH_PATH = shutil.which("bash") or "/bin/bash"

pytestmark_skip_win = pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX shell hook")

_BLOCK_CHAR_CEILING = 800

_TOOL_NAMES = (
    "mcp__iai-tools__code_search",
    "mcp__iai-tools__execute",
    "mcp__iai-tools__batch_execute",
    "mcp__iai-tools__session_search",
    "mcp__iai-tools__stats",
)


def _run_hook(stdin_text: str, *, path_override: "str | None" = None, timeout: float = 10.0):
    env = os.environ.copy()
    env["PATH"] = path_override if path_override is not None else "/usr/bin:/bin"
    return subprocess.run(
        [BASH_PATH, str(HOOK_PATH)],
        input=stdin_text,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytestmark_skip_win
def test_emits_valid_json_naming_every_tool_and_decision_rule():
    proc = _run_hook(json.dumps({"source": "startup", "cwd": "/tmp"}))
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    block = out["hookSpecificOutput"]["additionalContext"]
    assert out["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    for tool in _TOOL_NAMES:
        assert tool in block, block
    lowered = block.lower()
    assert "code_search" in lowered and "grep" in lowered, block


@pytestmark_skip_win
def test_block_within_char_ceiling():
    proc = _run_hook(json.dumps({"source": "resume"}))
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    block = out["hookSpecificOutput"]["additionalContext"]
    assert len(block) <= _BLOCK_CHAR_CEILING, len(block)


@pytestmark_skip_win
def test_fail_open_on_malformed_json():
    proc = _run_hook("{not valid json")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", proc.stdout


@pytestmark_skip_win
def test_fail_open_on_empty_stdin():
    proc = _run_hook("")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", proc.stdout


@pytestmark_skip_win
def test_fail_open_on_missing_jq(tmp_path):
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    proc = _run_hook(json.dumps({"source": "startup"}), path_override=str(empty_bin))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", proc.stdout
