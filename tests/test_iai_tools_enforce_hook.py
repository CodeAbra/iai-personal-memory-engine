"""Table-driven allow/deny coverage for the iai-tools-enforce PreToolUse hook.

Every deny row asserts the hook's decision on a given literal command
string, never what rtk rewrites a command into -- rtk's rewrite is rtk's
own contract, not this hook's, so each row hard-codes the exact string
the hook receives on stdin.
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
    / "src" / "iai_mcp" / "_deploy" / "hooks" / "iai-tools-enforce.sh"
)
BASH_PATH = shutil.which("bash") or "/bin/bash"

pytestmark_skip_win = pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX shell hook")


def _run_hook(
    tool_name: str,
    command: str | None = None,
    *,
    timeout: float = 10.0,
    extra_env: dict | None = None,
):
    env = os.environ.copy()
    # jq must be findable at /usr/bin/jq -- without it on PATH the deny row
    # fails open to empty stdout + exit 0 and reads as a false "allow".
    env["PATH"] = "/usr/bin:/bin"
    if extra_env:
        env.update(extra_env)
    payload: dict = {"tool_name": tool_name}
    if command is not None:
        payload["tool_input"] = {"command": command}
    return subprocess.run(
        [BASH_PATH, str(HOOK_PATH)],
        input=json.dumps(payload),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _make_stub_cli(tmp_path: Path) -> Path:
    """A hermetic stand-in for the resolved iai-tools CLI -- the hook only
    ever rewrites text with this path, never executes it."""
    stub = tmp_path / "stub-iai-tools"
    stub.write_text("#!/bin/sh\nexit 0\n")
    stub.chmod(0o755)
    return stub


def _assert_deny(proc: subprocess.CompletedProcess, expected_tool_substr: str) -> None:
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny", proc.stdout
    reason = out["hookSpecificOutput"]["permissionDecisionReason"]
    assert expected_tool_substr in reason, reason


def _assert_allow(proc: subprocess.CompletedProcess) -> None:
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", proc.stdout


# --- Native tool denial ------------------------------------------------

@pytestmark_skip_win
def test_native_grep_denied():
    _assert_deny(_run_hook("Grep"), "mcp__iai-tools__code_search")


@pytestmark_skip_win
def test_native_glob_denied():
    _assert_deny(_run_hook("Glob"), "mcp__iai-tools__code_search")


# --- Search-tool denial: raw AND rtk-rewritten forms --------------------

@pytestmark_skip_win
@pytest.mark.parametrize(
    "command",
    [
        "grep -rn foo src/",
        "rtk grep -rn foo src/",
        "rg foo src/",
        "ripgrep foo src/",
        "find . -name foo",
        "fd foo",
    ],
)
def test_search_tools_denied_raw_and_rtk_rewritten(command):
    _assert_deny(_run_hook("Bash", command), "mcp__iai-tools__code_search")


# --- Bulk read denial: raw, pipe, and rtk verb-substituted forms --------

@pytestmark_skip_win
@pytest.mark.parametrize(
    "command",
    [
        "cat file.txt",
        "rtk read file.txt",
        "git log | head",
        "rtk git log | head",
        "some_command | tail -50",
    ],
)
def test_bulk_read_denied_raw_pipe_and_rtk_rewritten(command):
    _assert_deny(_run_hook("Bash", command), "mcp__iai-tools__execute")


@pytestmark_skip_win
@pytest.mark.parametrize(
    "command",
    [
        "cat x > y",
        "head -n5 x > y",
        "cat <<EOF > file\nhello\nEOF",
    ],
)
def test_bulk_read_write_shape_allowed(command):
    _assert_allow(_run_hook("Bash", command))


@pytestmark_skip_win
def test_bash_read_builtin_allowed_not_rtk_subcommand():
    _assert_allow(_run_hook("Bash", "while read -r line; do :; done"))


# --- Over-blocking regression: token as a plain argument, not the command --

@pytestmark_skip_win
@pytest.mark.parametrize(
    "command",
    [
        "rm cat",
        "ls tail",
        "mv a head",
        "git add tail",
        "git checkout -b find",
        "echo cat",
        "python3 script.py --grep=x",
        "echo gh pr view 123",
        "echo hi gh api /repos/x/y",
        "myscript.sh gh pr view",
        'git commit -m "use gh pr view"',
    ],
)
def test_deny_token_as_plain_argument_allowed(command):
    _assert_allow(_run_hook("Bash", command))


# --- Newline is a real separator, unlike a plain space -------------------

@pytestmark_skip_win
@pytest.mark.parametrize(
    "command",
    [
        "cd foo\ncat bar.txt",
        "cd foo\ngrep x .",
    ],
)
def test_deny_token_on_second_line_of_multiline_command_denied(command):
    _assert_deny(_run_hook("Bash", command), "mcp__iai-tools__")


# --- Raw gh read denial: each subcommand its own row --------------------

@pytestmark_skip_win
@pytest.mark.parametrize(
    "command",
    [
        "gh pr view 5",
        "rtk gh pr view 5",
        "gh pr list",
        "gh pr show 5",
        "gh pr diff 5",
        "gh pr log",
        "gh issue view 3",
        "gh issue list",
        "gh repo view",
        "gh run list",
        "gh release show v1.0",
        "gh api /repos/x/y",
        "rtk gh api /repos/x/y",
        "gh api /repos/x/y -f name=-Xtra",
        "gh run view 5",
        "x | gh pr list",
    ],
)
def test_gh_read_subcommand_denied(command):
    _assert_deny(_run_hook("Bash", command), "mcp__iai-tools__execute")


@pytestmark_skip_win
@pytest.mark.parametrize(
    "command",
    [
        "gh pr create --title x",
        "gh pr merge 5",
        "gh issue close 3",
        "gh pr comment 5 -b hi",
        "gh pr edit 5",
        "gh pr review 5 --approve",
        "gh api /repos/x/y -X POST",
        "gh api /repos/x/y --method POST",
    ],
)
def test_gh_write_subcommand_allowed(command):
    _assert_allow(_run_hook("Bash", command))


# --- State-mutating commands pass through -------------------------------

@pytestmark_skip_win
@pytest.mark.parametrize(
    "command",
    [
        "git add file.txt",
        "git commit -m msg",
        "git push",
        "git checkout main",
        "git merge branch",
        "git rebase main",
        "git branch",
        "git tag v1",
        "git stash",
        "git reset --hard",
        "git apply patch.diff",
        "mkdir foo",
        "mv a b",
        "cp a b",
        "rm file",
        "touch file",
        "chmod +x file",
        "ln -s a b",
    ],
)
def test_state_mutating_commands_allowed(command):
    _assert_allow(_run_hook("Bash", command))


# --- Short fixed-output commands pass through ---------------------------

@pytestmark_skip_win
@pytest.mark.parametrize(
    "command",
    [
        "pytest tests/test_x.py -x",
        "npm run build",
        "git status",
        "git diff --stat",
        "ls",
        "pwd",
        "echo hi",
        "python3 script.py",
    ],
)
def test_short_fixed_output_commands_allowed(command):
    _assert_allow(_run_hook("Bash", command))


# --- Scope fence: bare unbounded git log/diff/show is a non-goal --------

@pytestmark_skip_win
@pytest.mark.parametrize(
    "command",
    [
        "git log",
        "git diff",
        "git show HEAD",
    ],
)
def test_bare_git_history_inspection_not_denied(command):
    # Non-goal: a bounded, revision/path/date-qualified git wrapper is a
    # separate, already-planned deliverable. This hook must not silently
    # expand its own deny-set to cover bare git log/diff/show.
    _assert_allow(_run_hook("Bash", command))


# --- Noisy-command redirect: bare curl/wget/gradle/mvn/sbt auto-redirect --

@pytestmark_skip_win
def test_bare_curl_redirected_to_bounded_run(tmp_path):
    stub = _make_stub_cli(tmp_path)
    savings_root = tmp_path / "iai-tools-root"
    proc = _run_hook(
        "Bash",
        "curl -s https://example.com",
        extra_env={"IAI_TOOLS_RUN_CLI": str(stub), "IAI_TOOLS_ROOT": str(savings_root)},
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    hook_out = out["hookSpecificOutput"]
    assert hook_out["permissionDecision"] == "allow", proc.stdout
    assert hook_out["updatedInput"]["command"] == f"{stub} run -- curl -s https://example.com"

    savings_log = savings_root / "tools-savings.jsonl"
    assert savings_log.is_file()
    lines = [ln for ln in savings_log.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    json.loads(lines[0])  # well-formed JSONL


@pytestmark_skip_win
@pytest.mark.parametrize(
    "command",
    [
        "wget https://example.com/file.tar.gz",
        "gradle build",
        "gradlew build",
        "./gradlew build",
        "mvn package",
        "mvnw package",
        "./mvnw package",
        "sbt compile",
        "mvn -q package",
        "gradle --console=plain build",
    ],
)
def test_noisy_command_redirected_to_bounded_run(tmp_path, command):
    stub = _make_stub_cli(tmp_path)
    proc = _run_hook("Bash", command, extra_env={"IAI_TOOLS_RUN_CLI": str(stub)})
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    hook_out = out["hookSpecificOutput"]
    assert hook_out["permissionDecision"] == "allow", proc.stdout
    assert hook_out["updatedInput"]["command"] == f"{stub} run -- {command}"


@pytestmark_skip_win
@pytest.mark.parametrize(
    ("command", "stripped"),
    [
        ("rtk curl -s https://example.com", "curl -s https://example.com"),
        ("rtk gradle build", "gradle build"),
    ],
)
def test_rtk_prefixed_noisy_command_redirected_to_bounded_run(tmp_path, command, stripped):
    stub = _make_stub_cli(tmp_path)
    proc = _run_hook("Bash", command, extra_env={"IAI_TOOLS_RUN_CLI": str(stub)})
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    hook_out = out["hookSpecificOutput"]
    assert hook_out["permissionDecision"] == "allow", proc.stdout
    wrapped = hook_out["updatedInput"]["command"]
    assert wrapped == f"{stub} run -- {stripped}"
    assert "rtk" not in wrapped.split("run --", 1)[1]


@pytestmark_skip_win
@pytest.mark.parametrize(
    "command",
    [
        "curl -s https://example.com | jq .",
        "curl -s https://example.com > out.json",
        "wget https://example.com/f > f",
        "gradle build && echo done",
        "(curl -s https://example.com)",
        "curl -d @- https://x < body.txt",
    ],
)
def test_noisy_command_with_shell_operator_passes_through_untouched(tmp_path, command):
    stub = _make_stub_cli(tmp_path)
    proc = _run_hook("Bash", command, extra_env={"IAI_TOOLS_RUN_CLI": str(stub)})
    _assert_allow(proc)


@pytestmark_skip_win
def test_already_wrapped_curl_passes_through_untouched(tmp_path):
    stub = _make_stub_cli(tmp_path)
    command = f"{stub} run -- curl -s https://example.com"
    proc = _run_hook("Bash", command, extra_env={"IAI_TOOLS_RUN_CLI": str(stub)})
    _assert_allow(proc)


# --- Structurally-bounded / state-mutation stay untouched by the redirect --

@pytestmark_skip_win
@pytest.mark.parametrize(
    "command",
    [
        "pwd",
        "git status",
        "curl --version",
        "git add file.txt",
        "git commit -m msg",
        "git push",
        "git checkout main",
        "mkdir foo",
        "mv a b",
        "cp a b",
        "rm file",
        "touch file",
        "chmod +x file",
        "ln -s a b",
    ],
)
def test_bounded_and_state_mutating_commands_never_redirected(tmp_path, command):
    stub = _make_stub_cli(tmp_path)
    proc = _run_hook("Bash", command, extra_env={"IAI_TOOLS_RUN_CLI": str(stub)})
    _assert_allow(proc)


# --- 355's native flooders stay hard-denied, redirect never weakens them --

@pytestmark_skip_win
@pytest.mark.parametrize(
    "command",
    [
        "grep -rn foo src/",
        "rg foo src/",
        "find . -name foo",
        "fd foo",
        "cat file.txt",
        "head file.txt",
        "tail file.txt",
        "gh pr view 5",
        "gh api /repos/x/y",
    ],
)
def test_native_flooders_still_hard_denied_with_redirect_cli_resolvable(tmp_path, command):
    stub = _make_stub_cli(tmp_path)
    proc = _run_hook("Bash", command, extra_env={"IAI_TOOLS_RUN_CLI": str(stub)})
    _assert_deny(proc, "mcp__iai-tools__")


# --- Fail-open: unresolvable CLI never rewrites to a broken binary --------

@pytestmark_skip_win
def test_noisy_command_fail_open_on_unresolvable_cli(tmp_path):
    unresolvable = tmp_path / "does-not-exist"
    proc = _run_hook(
        "Bash",
        "curl -s https://example.com",
        extra_env={"IAI_TOOLS_RUN_CLI": str(unresolvable), "HOME": str(tmp_path / "empty-home")},
    )
    _assert_allow(proc)


# --- Fail-open contract: malformed / empty / missing-jq payloads --------

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


@pytestmark_skip_win
def test_fail_open_on_malformed_json():
    proc = _run_hook_raw("{not valid json")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", proc.stdout


@pytestmark_skip_win
def test_fail_open_on_empty_stdin():
    proc = _run_hook_raw("")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", proc.stdout


@pytestmark_skip_win
def test_fail_open_on_missing_jq(tmp_path):
    # PATH must point at a directory with NO jq at all -- reusing
    # /usr/bin:/bin (where jq lives at /usr/bin/jq) would never exercise
    # this branch.
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    proc = _run_hook_raw(
        json.dumps({"tool_name": "Grep"}), path_override=str(empty_bin)
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", proc.stdout


# --- Neutral deny wording: distinguishing marker, no evasion framing ----

# Forbidden framing words assembled from fragments so this test itself
# never contains the literal anti-pattern phrasing (mirrors the
# fragment-assembly technique used elsewhere in this repo's guard tests).
_FORBIDDEN_FRAMING = (
    "ev" + "asion",
    "byp" + "ass",
    "avoid" + " " + "the" + " " + "hook",
    "get" + " " + "around",
)
_ENFORCE_MARKER = "iai" + "-tools-enforce"


@pytestmark_skip_win
def test_deny_reason_carries_marker_and_no_forbidden_framing_words():
    proc = _run_hook("Grep")
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    reason = out["hookSpecificOutput"]["permissionDecisionReason"]
    assert _ENFORCE_MARKER in reason, reason
    lowered = reason.lower()
    for phrase in _FORBIDDEN_FRAMING:
        assert phrase not in lowered, f"forbidden framing {phrase!r} found in: {reason}"


@pytestmark_skip_win
def test_redirect_output_carries_no_forbidden_framing_words(tmp_path):
    stub = _make_stub_cli(tmp_path)
    proc = _run_hook(
        "Bash", "curl -s https://example.com", extra_env={"IAI_TOOLS_RUN_CLI": str(stub)}
    )
    assert proc.returncode == 0, proc.stderr
    lowered = proc.stdout.lower()
    for phrase in _FORBIDDEN_FRAMING:
        assert phrase not in lowered, f"forbidden framing {phrase!r} found in: {proc.stdout}"
