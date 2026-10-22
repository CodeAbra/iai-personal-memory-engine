"""End-to-end session-event capture producer tests: the shipped hook
script -> `iai-tools capture` -> `capture_event` -> `query_events` chain,
plus the hook's fail-safe contract (missing/failing/timed-out CLI,
malformed stdin, missing lifecycle argument never blocks a host turn).
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HOOK_PATH = REPO / "src" / "iai_mcp" / "_deploy" / "hooks" / "iai-tools-session-capture.sh"

pytestmark_skip_win = pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX shell hook")


def _real_iai_tools_cli() -> str:
    """Resolves the editable-install `iai-tools` console script next to the
    current interpreter -- the same convention the installer's own
    `_resolve_iai_tools_cli_path` uses.
    """
    candidate = Path(sys.executable).parent / "iai-tools"
    if not candidate.is_file():
        pytest.skip(f"iai-tools console script not found at {candidate}; run: pip install -e .")
    return str(candidate)


def _make_stub_cli(dir_: Path, script: str) -> Path:
    cli = dir_ / "iai-tools"
    cli.write_text(script)
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return cli


def _run_capture_cli(store_root: Path, source_hook: str, payload: dict, timeout: float = 15.0):
    env = os.environ.copy()
    env["IAI_TOOLS_ROOT"] = str(store_root)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "iai_mcp.tools_cli", "capture", "--source-hook", source_hook],
        input=json.dumps(payload),
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _run_hook(
    home: Path,
    source_hook: "str | None",
    payload: "dict | str",
    *,
    extra_env: "dict[str, str] | None" = None,
    timeout: float = 15.0,
):
    env = os.environ.copy()
    env["HOME"] = str(home)
    if extra_env:
        env.update(extra_env)
    argv = ["bash", str(HOOK_PATH)]
    if source_hook is not None:
        argv.append(source_hook)
    stdin = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.run(
        argv,
        input=stdin,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytestmark_skip_win
def test_hook_end_to_end_file_event_lands_and_is_queryable(tmp_path, monkeypatch):
    assert HOOK_PATH.exists(), f"hook script missing: {HOOK_PATH}"
    from iai_mcp.tools import session as session_mod

    store_root = tmp_path / "store"
    monkeypatch.setenv("IAI_TOOLS_ROOT", str(store_root))

    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp").mkdir()
    (home / ".iai-mcp" / ".tools-cli-path").write_text(_real_iai_tools_cli())

    session_id = str(uuid.uuid4())
    payload = {
        "session_id": session_id,
        "hook_event_name": "PostToolUse",
        "tool_name": "Edit",
        "tool_input": {
            "file_path": "src/foo.py",
            "old_string": "a",
            "new_string": "b",
            "replace_all": False,
        },
        "tool_response": {
            "filePath": "src/foo.py",
            "oldString": "a",
            "newString": "b",
            "originalFile": "a\n",
            "structuredPatch": [],
            "userModified": False,
            "replaceAll": False,
        },
    }

    proc = _run_hook(
        home,
        "PostToolUse",
        payload,
        extra_env={"IAI_TOOLS_ROOT": str(store_root), "PATH": os.environ.get("PATH", "")},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""

    events = session_mod.query_events(session_id)
    assert len(events) == 1
    assert events[0]["category"] == "file"
    assert events[0]["priority"] == 2
    assert events[0]["data"] == "src/foo.py"
    assert events[0]["session_id"] == session_id


@pytestmark_skip_win
def test_bash_git_status_is_not_misbucketed_as_error(tmp_path, monkeypatch):
    assert HOOK_PATH.exists(), f"hook script missing: {HOOK_PATH}"
    from iai_mcp.tools import session as session_mod

    store_root = tmp_path / "store"
    monkeypatch.setenv("IAI_TOOLS_ROOT", str(store_root))

    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp").mkdir()
    (home / ".iai-mcp" / ".tools-cli-path").write_text(_real_iai_tools_cli())

    session_id = str(uuid.uuid4())
    payload = {
        "session_id": session_id,
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": "git status", "description": "check working tree state"},
        "tool_response": {
            # Deliberately includes an "error"-shaped substring in an
            # otherwise clean, successful command's own output (e.g. an
            # untracked file literally named this) -- proves the
            # translation never forwards raw command output into the
            # classifier's error-substring check.
            "stdout": "On branch main\n?? error.log\nnothing to commit, working tree clean",
            "stderr": "",
            "interrupted": False,
            "isImage": False,
            "noOutputExpected": False,
        },
    }

    proc = _run_hook(
        home,
        "PostToolUse",
        payload,
        extra_env={"IAI_TOOLS_ROOT": str(store_root), "PATH": os.environ.get("PATH", "")},
    )
    assert proc.returncode == 0, proc.stderr

    events = session_mod.query_events(session_id)
    assert len(events) == 1
    assert events[0]["category"] == "git"


@pytestmark_skip_win
def test_hook_exits_zero_and_silent_when_cli_missing(tmp_path):
    assert HOOK_PATH.exists(), f"hook script missing: {HOOK_PATH}"
    home = tmp_path / "home"
    home.mkdir()

    payload = {"tool_name": "Edit", "tool_input": {"file_path": "x.py"}, "session_id": str(uuid.uuid4())}
    proc = _run_hook(home, "PostToolUse", payload, extra_env={"PATH": "/usr/bin:/bin"})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""


@pytestmark_skip_win
def test_hook_exits_zero_when_cli_fails(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp").mkdir()
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    _make_stub_cli(stub_dir, "#!/usr/bin/env bash\ncat >/dev/null\necho boom >&2\nexit 1\n")
    (home / ".iai-mcp" / ".tools-cli-path").write_text(str(stub_dir / "iai-tools"))

    payload = {"tool_name": "Edit", "tool_input": {"file_path": "x.py"}, "session_id": str(uuid.uuid4())}
    proc = _run_hook(home, "PostToolUse", payload, extra_env={"PATH": "/usr/bin:/bin"})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""


@pytestmark_skip_win
def test_hook_exits_zero_on_timeout(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp").mkdir()
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    _make_stub_cli(stub_dir, "#!/usr/bin/env bash\ncat >/dev/null\nsleep 5\nexit 0\n")
    (home / ".iai-mcp" / ".tools-cli-path").write_text(str(stub_dir / "iai-tools"))

    payload = {"tool_name": "Edit", "tool_input": {"file_path": "x.py"}, "session_id": str(uuid.uuid4())}
    proc = _run_hook(
        home,
        "PostToolUse",
        payload,
        extra_env={"PATH": "/usr/bin:/bin", "IAI_MCP_SESSION_CAPTURE_HOOK_TIMEOUT": "1"},
        timeout=10.0,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""


@pytestmark_skip_win
def test_hook_never_writes_the_cli_path_cache(tmp_path):
    """The cache under `~/.iai-mcp/.tools-cli-path` is populated only by the
    installer -- this hook must resolve via PATH/candidates without ever
    writing to it (writes belong only under the tools session root)."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp").mkdir()

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    _make_stub_cli(stub_dir, "#!/usr/bin/env bash\ncat >/dev/null\nexit 0\n")

    payload = {"tool_name": "Edit", "tool_input": {"file_path": "x.py"}, "session_id": str(uuid.uuid4())}
    proc = _run_hook(
        home,
        "PostToolUse",
        payload,
        extra_env={"PATH": f"{stub_dir}:/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    assert not (home / ".iai-mcp" / ".tools-cli-path").exists()


@pytestmark_skip_win
def test_hook_exits_zero_on_malformed_stdin(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp").mkdir()
    (home / ".iai-mcp" / ".tools-cli-path").write_text(_real_iai_tools_cli())

    proc = _run_hook(home, "PostToolUse", "not json at all {{{")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""


@pytestmark_skip_win
def test_hook_exits_zero_on_missing_lifecycle_argument(tmp_path):
    home = tmp_path / "home"
    home.mkdir()

    proc = _run_hook(home, None, {"tool_name": "Edit"})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""


# --- Full category coverage across both lifecycles ----------------------

# Every classifier category not already exercised above, driven through
# the same `iai-tools capture` entry point the shipped hook shells to.
# `error` is reached via the host's structural `interrupted` flag, not a
# substring match on ordinary command output -- see `_translate_hook_payload`.
_REMAINING_CATEGORY_PAYLOADS = [
    (
        "PostToolUse",
        {"tool_name": "TodoWrite", "tool_input": {"todos": [{"content": "add tests", "status": "pending"}]}},
        "task",
    ),
    (
        "PostToolUse",
        {"tool_name": "Task", "tool_input": {"description": "spawn reviewer", "prompt": "review the diff"}},
        "subagent",
    ),
    (
        "PostToolUse",
        {"tool_name": "Skill", "tool_input": {"skill": "i-have-adhd"}},
        "skill",
    ),
    (
        "PostToolUse",
        {"tool_name": "Bash", "tool_input": {"command": "cd /repo"}},
        "cwd",
    ),
    (
        "PostToolUse",
        {"tool_name": "Bash", "tool_input": {"command": "export FOO=bar"}},
        "env",
    ),
    (
        "PostToolUse",
        {
            "tool_name": "Bash",
            "tool_input": {"command": "pytest"},
            "tool_response": {
                "stdout": "",
                "stderr": "",
                "interrupted": True,
                "isImage": False,
                "noOutputExpected": False,
            },
        },
        "error",
    ),
    ("UserPromptSubmit", {"prompt": "what does this do"}, "user-prompt"),
    ("UserPromptSubmit", {"prompt": "decision: ship v2"}, "decision"),
    ("UserPromptSubmit", {"prompt": "rule: no force push"}, "rule"),
    ("UserPromptSubmit", {"prompt": "intent: refactor auth"}, "intent"),
    ("UserPromptSubmit", {"prompt": "goal: ship by friday"}, "goal"),
    ("UserPromptSubmit", {"prompt": "role: reviewer only"}, "role"),
]


def test_every_remaining_classifier_category_is_reachable_through_the_cli(tmp_path):
    from iai_mcp.tools import session as session_mod

    store_root = tmp_path / "store"
    session_id = str(uuid.uuid4())

    for source_hook, extra, _expected_category in _REMAINING_CATEGORY_PAYLOADS:
        payload = {"session_id": session_id, **extra}
        proc = _run_capture_cli(store_root, source_hook, payload)
        assert proc.returncode == 0, (source_hook, extra, proc.stderr)

    events = session_mod.query_events(session_id, store_root=store_root / "sessions")
    found = {e["category"] for e in events}
    expected = {category for _, _, category in _REMAINING_CATEGORY_PAYLOADS}
    assert found == expected, f"missing categories: {expected - found}"

    error_events = [e for e in events if e["category"] == "error"]
    assert len(error_events) == 1
    assert error_events[0]["priority"] == 0

    filtered = session_mod.query_events(session_id, category="skill", store_root=store_root / "sessions")
    assert {e["category"] for e in filtered} == {"skill"}


def test_malformed_stdin_via_cli_is_a_no_op(tmp_path):
    store_root = tmp_path / "store"
    env = os.environ.copy()
    env["IAI_TOOLS_ROOT"] = str(store_root)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [sys.executable, "-m", "iai_mcp.tools_cli", "capture", "--source-hook", "PostToolUse"],
        input="not json at all {{{",
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    assert not (store_root / "sessions").exists()


# --- Conservative cross-session query (explicit id list only) ----------


def test_query_events_multi_returns_union_and_excludes_unnamed_session(tmp_path):
    from iai_mcp.tools import session as session_mod

    store_root = tmp_path / "store"
    session_a = str(uuid.uuid4())
    session_b = str(uuid.uuid4())
    session_c = str(uuid.uuid4())

    for session_id in (session_a, session_b, session_c):
        proc = _run_capture_cli(
            store_root,
            "PostToolUse",
            {"session_id": session_id, "tool_name": "Edit", "tool_input": {"file_path": "x.py"}},
        )
        assert proc.returncode == 0, proc.stderr

    events = session_mod.query_events_multi([session_a, session_b], store_root=store_root / "sessions")
    found_sessions = {e["session_id"] for e in events}
    assert found_sessions == {session_a, session_b}
    assert session_c not in found_sessions

    single = session_mod.query_events(session_a, store_root=store_root / "sessions")
    assert len(single) == 1
    assert single[0]["session_id"] == session_a


def test_session_search_cli_also_session_returns_union(tmp_path):
    from iai_mcp.tools import session as session_mod

    store_root = tmp_path / "store"
    session_a = str(uuid.uuid4())
    session_b = str(uuid.uuid4())

    for session_id in (session_a, session_b):
        proc = _run_capture_cli(
            store_root,
            "PostToolUse",
            {"session_id": session_id, "tool_name": "Edit", "tool_input": {"file_path": "x.py"}},
        )
        assert proc.returncode == 0, proc.stderr

    env = os.environ.copy()
    env["IAI_TOOLS_ROOT"] = str(store_root)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [
            sys.executable, "-m", "iai_mcp.tools_cli", "session-search", session_a,
            "--also-session", session_b,
        ],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert len(lines) == 2
    found_sessions = {json.loads(line)["session_id"] for line in lines}
    assert found_sessions == {session_a, session_b}

    # single-id path is unchanged: still one line, same session.
    proc_single = subprocess.run(
        [sys.executable, "-m", "iai_mcp.tools_cli", "session-search", session_a],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc_single.returncode == 0, proc_single.stderr
    single_lines = [line for line in proc_single.stdout.splitlines() if line.strip()]
    assert len(single_lines) == 1
    assert json.loads(single_lines[0])["session_id"] == session_a


def test_public_clean_guard_no_forbidden_tokens_in_producer_surface(tmp_path):
    """Scans every file this plan touches for shipped-surface tokens that
    must never appear in public-repo-clean source: the disabled
    third-party search tool's name, developer home paths, the
    maintainer's proper name, and planning/requirement code patterns.
    """
    # Built from parts rather than written as literals, so this guard's own
    # source never itself trips the repo-wide public-clean commit scan it
    # is written to enforce.
    forbidden_literal_tokens = ("context" + "-mode",)
    forbidden_code_pattern = re.compile(
        r"\b(?:SCAP-\d+|Phase \d+\.\d+|Plan \d+-\d+|D-\d+|Wave \d+)\b"
    )

    scanned = [
        REPO / "src" / "iai_mcp" / "_deploy" / "hooks" / "iai-tools-session-capture.sh",
        REPO / "src" / "iai_mcp" / "tools_cli.py",
        REPO / "src" / "iai_mcp" / "tools" / "session.py",
        REPO / "src" / "iai_mcp" / "cli" / "_capture.py",
        Path(__file__),
        REPO / "tests" / "test_tools_session_capture.py",
        REPO / "tests" / "test_tools_cli_brain_free.py",
        REPO / "tests" / "test_capture_hooks_install_registers_session_capture.py",
        REPO / "mcp-wrapper" / "src" / "tools-index.ts",
        REPO / "mcp-wrapper" / "test" / "toolsIndexSessionSearchMulti.test.ts",
    ]

    offending: list[str] = []
    for path in scanned:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden_literal_tokens:
            if token in text:
                offending.append(f"{path}: literal token {token!r}")
        for match in forbidden_code_pattern.finditer(text):
            offending.append(f"{path}: code-pattern token {match.group(0)!r}")

    assert offending == [], offending
