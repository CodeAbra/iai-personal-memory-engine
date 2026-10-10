"""Table-driven behavior coverage for the iai-tools-agent-inject PreToolUse hook.

Field detection scans an ORDERED candidate-key list -- prompt, request,
objective, question, query, task -- and injects into the FIRST key the
payload actually has. Every row hard-codes the exact stdin payload.
"""
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
    / "src" / "iai_mcp" / "_deploy" / "hooks" / "iai-tools-agent-inject.sh"
)
BASH_PATH = shutil.which("bash") or "/bin/bash"

pytestmark_skip_win = pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX shell hook")

_SENTINEL = "<iai_tools_routing>"


def _run_hook_raw(stdin_text: str, *, path_override: "str | None" = None, timeout: float = 10.0):
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


def _run_hook(payload: dict, **kwargs):
    return _run_hook_raw(json.dumps(payload), **kwargs)


def _assert_allow_no_op(proc: subprocess.CompletedProcess) -> None:
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", proc.stdout


def _assert_injected(proc: subprocess.CompletedProcess, field: str, orig_value: str) -> dict:
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    hso = out["hookSpecificOutput"]
    assert hso["hookEventName"] == "PreToolUse", out
    assert hso["permissionDecision"] == "allow", out
    updated = hso["updatedInput"]
    assert field in updated, updated
    assert updated[field].startswith(orig_value), updated[field]
    assert _SENTINEL in updated[field], updated[field]
    for tool in (
        "mcp__iai-tools__code_search",
        "mcp__iai-tools__execute",
        "mcp__iai-tools__batch_execute",
        "mcp__iai-tools__session_search",
        "mcp__iai-tools__stats",
    ):
        assert tool in updated[field], updated[field]
    return updated


# --- Ordered candidate-key field detection ------------------------------

@pytestmark_skip_win
def test_prompt_key_injected():
    proc = _run_hook({"tool_name": "Task", "tool_input": {"prompt": "do X", "subagent_type": "foo"}})
    updated = _assert_injected(proc, "prompt", "do X")
    # other keys survive untouched
    assert updated["subagent_type"] == "foo", updated


@pytestmark_skip_win
def test_non_prompt_candidate_key_injected_when_prompt_absent():
    proc = _run_hook({"tool_name": "Task", "tool_input": {"request": "do Y"}})
    _assert_injected(proc, "request", "do Y")


@pytestmark_skip_win
def test_earlier_ordered_key_wins_when_multiple_present():
    proc = _run_hook({"tool_name": "Task", "tool_input": {"prompt": "P", "request": "R"}})
    updated = _assert_injected(proc, "prompt", "P")
    # non-matched sibling key is untouched
    assert updated["request"] == "R", updated


@pytestmark_skip_win
@pytest.mark.parametrize("field", ["objective", "question", "query", "task"])
def test_each_remaining_candidate_key_supported(field):
    proc = _run_hook({"tool_name": "Task", "tool_input": {field: "value"}})
    _assert_injected(proc, field, "value")


# --- Fail-open guards -----------------------------------------------------

@pytestmark_skip_win
def test_no_candidate_key_present_fails_open():
    proc = _run_hook({"tool_name": "Task", "tool_input": {"subagent_type": "foo"}})
    _assert_allow_no_op(proc)


@pytestmark_skip_win
def test_matched_key_null_value_fails_open():
    proc = _run_hook({"tool_name": "Task", "tool_input": {"prompt": None}})
    _assert_allow_no_op(proc)


@pytestmark_skip_win
def test_matched_key_non_string_value_fails_open():
    proc = _run_hook({"tool_name": "Task", "tool_input": {"prompt": 123}})
    _assert_allow_no_op(proc)


@pytestmark_skip_win
def test_already_injected_value_no_duplicate():
    already = f"do X\n\n{_SENTINEL}already here</iai_tools_routing>"
    proc = _run_hook({"tool_name": "Task", "tool_input": {"prompt": already}})
    _assert_allow_no_op(proc)


@pytestmark_skip_win
def test_missing_tool_input_fails_open():
    proc = _run_hook({"tool_name": "Task"})
    _assert_allow_no_op(proc)


@pytestmark_skip_win
def test_fail_open_on_malformed_json():
    proc = _run_hook_raw("{not valid json")
    _assert_allow_no_op(proc)


@pytestmark_skip_win
def test_fail_open_on_empty_stdin():
    proc = _run_hook_raw("")
    _assert_allow_no_op(proc)


@pytestmark_skip_win
def test_fail_open_on_missing_jq(tmp_path):
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    proc = _run_hook(
        {"tool_name": "Task", "tool_input": {"prompt": "x"}},
        path_override=str(empty_bin),
    )
    _assert_allow_no_op(proc)


# --- Neutral wording: no evasion/bypass framing ---------------------------

_FORBIDDEN_FRAMING = (
    "ev" + "asion",
    "byp" + "ass",
    "avoid" + " " + "the" + " " + "hook",
    "get" + " " + "around",
)


@pytestmark_skip_win
def test_routing_block_carries_no_forbidden_framing_words():
    proc = _run_hook({"tool_name": "Task", "tool_input": {"prompt": "do X"}})
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    block = out["hookSpecificOutput"]["updatedInput"]["prompt"]
    lowered = block.lower()
    for phrase in _FORBIDDEN_FRAMING:
        assert phrase not in lowered, f"forbidden framing {phrase!r} found in: {block}"
