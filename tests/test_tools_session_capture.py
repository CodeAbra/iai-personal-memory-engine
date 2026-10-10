"""Session-event capture -> store -> query -> snapshot round-trip, and the
brain-import-boundary contract for `iai_mcp.tools.session`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

_BRAIN_MODULES = (
    "iai_mcp.capture",
    "iai_mcp.hippo",
    "iai_mcp.lilli",
    "iai_mcp_native.embed",
    "iai_mcp_native",
)

# One synthetic payload per load-bearing category -- proves the classifier's
# full 14-category coverage through the same capture path a real hook uses.
_ALL_14_CATEGORIES_PAYLOADS = [
    (
        {
            "tool_name": "Edit",
            "file_path": "src/x.py",
            "session_id": "sess-14cats",
            "source_hook": "PostToolUse",
        },
        "file",
    ),
    (
        {
            "tool_name": "TodoWrite",
            "content": "add tests",
            "session_id": "sess-14cats",
            "source_hook": "PostToolUse",
        },
        "task",
    ),
    (
        {
            "tool_name": "Bash",
            "command": "git status",
            "session_id": "sess-14cats",
            "source_hook": "PostToolUse",
        },
        "git",
    ),
    (
        {
            "tool_name": "Bash",
            "command": "cd /repo",
            "session_id": "sess-14cats",
            "source_hook": "PostToolUse",
        },
        "cwd",
    ),
    (
        {
            "tool_name": "Task",
            "content": "spawn reviewer",
            "session_id": "sess-14cats",
            "source_hook": "PostToolUse",
        },
        "subagent",
    ),
    (
        {
            "tool_name": "Skill",
            "content": "i-have-adhd",
            "session_id": "sess-14cats",
            "source_hook": "PostToolUse",
        },
        "skill",
    ),
    (
        {
            "tool_name": "Bash",
            "command": "pytest",
            "is_error": True,
            "session_id": "sess-14cats",
            "source_hook": "PostToolUse",
        },
        "error",
    ),
    (
        {
            "tool_name": "Bash",
            "command": "export FOO=bar",
            "session_id": "sess-14cats",
            "source_hook": "PostToolUse",
        },
        "env",
    ),
    (
        {
            "source_hook": "UserPromptSubmit",
            "prompt": "what does this do",
            "session_id": "sess-14cats",
        },
        "user-prompt",
    ),
    (
        {"prompt": "decision: ship v2", "session_id": "sess-14cats"},
        "decision",
    ),
    (
        {"prompt": "rule: no force push", "session_id": "sess-14cats"},
        "rule",
    ),
    (
        {"prompt": "intent: refactor auth", "session_id": "sess-14cats"},
        "intent",
    ),
    (
        {"prompt": "goal: ship by friday", "session_id": "sess-14cats"},
        "goal",
    ),
    (
        {"prompt": "role: reviewer only", "session_id": "sess-14cats"},
        "role",
    ),
]


def _session_bin() -> str:
    for profile in ("release", "debug"):
        candidate = REPO / "target" / profile / "iai_tools_session"
        if candidate.exists():
            return str(candidate)
    pytest.skip("iai_tools_session not built; run: cargo build -p iai_mcp_tools_core --bins")


def test_one_event_round_trips_through_capture_query_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_TOOLS_ROOT", str(tmp_path))

    from iai_mcp.tools import session

    payload = {
        "tool_name": "Edit",
        "file_path": "src/foo.py",
        "session_id": "sess-test-1",
        "source_hook": "PostToolUse",
    }
    session.capture_event(payload)

    jsonl_path = tmp_path / "sessions" / "sess-test-1.jsonl"
    assert jsonl_path.exists()

    events = session.query_events("sess-test-1", substring="foo.py")
    assert len(events) == 1
    assert events[0]["category"] == "file"
    assert events[0]["data"] == "src/foo.py"

    snap = session.build_snapshot("sess-test-1")
    assert "file" in snap
    assert "--session sess-test-1 --category file" in snap


def test_malformed_hook_payload_is_a_no_op(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_TOOLS_ROOT", str(tmp_path))

    from iai_mcp.tools import session

    session.capture_event({"tool_name": "Bash"})
    session.capture_event({})

    assert not (tmp_path / "sessions").exists()


def test_invalid_session_id_is_rejected_not_rewritten(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_TOOLS_ROOT", str(tmp_path))

    from iai_mcp.tools import session

    session.capture_event(
        {
            "tool_name": "Edit",
            "file_path": "src/foo.py",
            "session_id": "../../etc/passwd",
            "source_hook": "PostToolUse",
        }
    )
    assert not (tmp_path / "sessions").exists()
    assert session.query_events("../../etc/passwd") == []


def test_session_module_imports_no_brain_module():
    script = (
        "import sys\n"
        "import iai_mcp.tools.session\n"
        f"hit = [m for m in {_BRAIN_MODULES!r} if m in sys.modules]\n"
        "print(','.join(hit))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", (
        f"brain module(s) loaded on import: {result.stdout.strip()!r} / {result.stderr}"
    )


def test_session_module_source_has_no_brain_import():
    source = (REPO / "src" / "iai_mcp" / "tools" / "session.py").read_text(encoding="utf-8")
    for line in source.splitlines():
        stripped = line.strip()
        for forbidden in ("import iai_mcp.capture", "import iai_mcp.hippo", "import iai_mcp.lilli"):
            assert not stripped.startswith(forbidden), f"forbidden brain import found: {line!r}"


def test_all_14_categories_round_trip_through_capture_event(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_TOOLS_ROOT", str(tmp_path))

    from iai_mcp.tools import session

    for payload, _expected_category in _ALL_14_CATEGORIES_PAYLOADS:
        session.capture_event(payload)

    events = session.query_events("sess-14cats")
    assert len(events) == len(_ALL_14_CATEGORIES_PAYLOADS)

    expected_categories = {category for _, category in _ALL_14_CATEGORIES_PAYLOADS}
    assert len(expected_categories) == 14
    found_categories = {e["category"] for e in events}
    assert found_categories == expected_categories

    user_prompt_events = [e for e in events if e["category"] == "user-prompt"]
    assert len(user_prompt_events) == 1
    assert "user_prompt" not in found_categories, "hyphenated category must never round-trip as user_prompt"


def test_binary_capture_then_search_round_trip(tmp_path):
    env = os.environ.copy()
    env["IAI_TOOLS_ROOT"] = str(tmp_path)

    payload = json.dumps(
        {
            "tool_name": "Edit",
            "file_path": "src/binary.py",
            "session_id": "sess-binary",
            "source_hook": "PostToolUse",
        }
    )
    capture_proc = subprocess.run(
        [_session_bin(), "capture"],
        input=payload,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert capture_proc.returncode == 0, capture_proc.stderr

    search_proc = subprocess.run(
        [_session_bin(), "search", "--session", "sess-binary", "--category", "file"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert search_proc.returncode == 0, search_proc.stderr
    lines = [line for line in search_proc.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["category"] == "file"
    assert event["data"] == "src/binary.py"


def test_binary_none_classification_is_a_no_op_exit_zero(tmp_path):
    env = os.environ.copy()
    env["IAI_TOOLS_ROOT"] = str(tmp_path)

    payload = json.dumps({"tool_name": "Read", "session_id": "sess-binary-none"})
    capture_proc = subprocess.run(
        [_session_bin(), "capture"],
        input=payload,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert capture_proc.returncode == 0, capture_proc.stderr
    assert not (tmp_path / "sessions").exists()


def test_binary_malformed_stdin_is_a_no_op_exit_zero(tmp_path):
    env = os.environ.copy()
    env["IAI_TOOLS_ROOT"] = str(tmp_path)

    capture_proc = subprocess.run(
        [_session_bin(), "capture"],
        input="not json at all {{{",
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert capture_proc.returncode == 0, capture_proc.stderr
    assert not (tmp_path / "sessions").exists()


def test_binary_not_referenced_by_any_repo_deploy_hook_template():
    """Reads ONLY this repo's own shipped hook templates -- never the
    operator's live `~/.claude/settings.json` or any file under `~/.claude/`.
    """
    hooks_dir = REPO / "src" / "iai_mcp" / "_deploy" / "hooks"
    if not hooks_dir.is_dir():
        pytest.skip("no _deploy/hooks directory shipped in this checkout")

    binary_name = "iai_tools_session"
    offending = []
    for path in sorted(hooks_dir.iterdir()):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if binary_name in text:
            offending.append(str(path))

    assert offending == [], (
        f"{binary_name} referenced by repo-shipped deploy-hook template(s): {offending}"
    )
