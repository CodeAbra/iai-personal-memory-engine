"""Shared body for the bounded git-history query wrapper (`git-log`,
`git-blame`, `git-diff`), used by both the `iai-tools` console script and
the `iai-mcp tools` operator CLI so the two entry points cannot drift.

Module-level imports are stdlib only. `iai_mcp.tools.exec_cli` is imported
at call time, so a plain import of this module stays native-extension-free
and brain-free.

Bound-by-construction validation, the leading-dash argument-injection
guard, and the secret-gate scrub all live in this ONE module so no caller
surface (either CLI entry point, or the MCP execute-mode dispatch path)
can bypass them. `_run_and_render_git` is the single render/return choke
point every query shape dispatches through.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

# Emitted verbatim by every query shape's bound-rejection message, so one
# table-driven test can assert a single substring across all shapes.
_BOUND_CLAUSE = "to bound this query"

_REDACTED_LINE = "[redacted: secret-shaped content]"


class GitQueryError(Exception):
    """Raised for a rejected git-query call. Callers print this and return
    exit code 1."""


def _reject_leading_dash(value: "str | None", *, shape: str, label: str) -> None:
    if value is not None and value.startswith("-"):
        raise GitQueryError(f"git-{shape}: {label} value must not start with '-': {value!r}")


def _build_log_argv(args: argparse.Namespace) -> "list[str]":
    """Validates the bound requirement and the injection guard, then
    constructs the discrete git argv for the log shape. Raises
    `GitQueryError` for either failure -- never builds a partial argv."""
    since = getattr(args, "since", None)
    until = getattr(args, "until", None)
    author = getattr(args, "author", None)
    rev_range = getattr(args, "rev_range", None)
    path = getattr(args, "path", None)
    # repo is intentionally unrestricted (git -C <repo>, any path) -- this
    # grants no more access than the pre-existing execute(command="git",
    # args=["-C", ...]) path; it is a convenience parameter, not a sandbox.
    repo = getattr(args, "repo", None)

    if not (since or until or rev_range or path):
        raise GitQueryError(
            f"git-log: supply --since/--until, a revision range, or --path, {_BOUND_CLAUSE}"
        )

    # Asymmetric guard: a revision sits structurally before the `--`
    # separator and so cannot be `--`-protected -- its only protection is
    # this blanket leading-dash rejection. A path gets the same rejection
    # PLUS the `--` separator below.
    _reject_leading_dash(rev_range, shape="log", label="revision")
    _reject_leading_dash(path, shape="log", label="path")
    _reject_leading_dash(repo, shape="log", label="--repo")

    argv: "list[str]" = []
    if repo:
        argv.extend(["-C", repo])
    argv.append("log")
    if since:
        argv.append(f"--since={since}")
    if until:
        argv.append(f"--until={until}")
    if author:
        argv.append(f"--author={author}")
    if rev_range:
        argv.append(rev_range)
    if path:
        argv.extend(["--", path])
    return argv


def _scrub_secrets(text: str) -> str:
    """Line-by-line secret-gate scrub: a matching line is replaced whole,
    surrounding non-secret lines survive untouched."""
    if not text:
        return text

    from iai_mcp.tools.secret_gate import contains_secret_shaped_content

    scrubbed: "list[str]" = []
    for line in text.splitlines(keepends=True):
        if contains_secret_shaped_content(line):
            newline = "\n" if line.endswith("\n") else ""
            scrubbed.append(_REDACTED_LINE + newline)
        else:
            scrubbed.append(line)
    return "".join(scrubbed)


def _redact_secret_shaped_tail(text: str) -> str:
    """The Rust tail-window byte-caps raw output before this module ever
    sees it, and a killed process can leave a short file behind either way
    -- both can cut a credential mid-token, shorter than the length
    `_scrub_secrets`'s line-based match requires, so a straddling fragment
    survives that scrub unredacted. Applied unconditionally (not gated on
    `truncated`/`state`): it only ever replaces a fragment that itself
    looks like the start of a credential, so a genuinely complete, ordinary
    tail is never touched.
    """
    if not text:
        return text

    from iai_mcp.tools.secret_gate import secret_shaped_tail_start

    start = secret_shaped_tail_start(text)
    return text[:start] + _REDACTED_LINE if start is not None else text


def _flatten_git_result(result: "dict[str, Any]") -> "dict[str, Any]":
    """git-query's own superset flatten -- a SUPERSET of
    `exec_cli._flatten_run_result`'s shape, carrying the bound signal
    (`truncated`/`elided_bytes`) through from `bounded_tail`. Never modifies
    `_flatten_run_result` itself, which stays byte-unchanged for
    run/fetch/batch-run. This is the sole render/return choke point every
    query shape dispatches through -- the secret-gate scrub is applied here
    so every shape inherits it.
    """
    tail = result.get("bounded_tail") or {}
    truncated = tail.get("truncated", False)
    stdout = _redact_secret_shaped_tail(_scrub_secrets(tail.get("stdout", "")))
    stderr = _redact_secret_shaped_tail(_scrub_secrets(tail.get("stderr", "")))
    return {
        "run_id": result.get("run_id"),
        "state": result.get("state"),
        "stdout": stdout,
        "stderr": stderr,
        "truncated": truncated,
        "elided_bytes": tail.get("elided_bytes", 0),
    }


def _build_blame_argv(args: argparse.Namespace) -> "list[str]":
    """Validates the bound requirement (a file must be supplied) and the
    injection guard, then constructs the discrete git argv for the blame
    shape. Raises `GitQueryError` for either failure -- never builds a
    partial argv. `start_line`/`end_line` are wrapper-validated integers
    (argparse `type=int`), never a caller-passed-through `-L` flag string."""
    file = getattr(args, "file", None)
    # repo is intentionally unrestricted (git -C <repo>, any path) -- this
    # grants no more access than the pre-existing execute(command="git",
    # args=["-C", ...]) path; it is a convenience parameter, not a sandbox.
    repo = getattr(args, "repo", None)
    start_line = getattr(args, "start_line", None)
    end_line = getattr(args, "end_line", None)

    if not file:
        raise GitQueryError(f"git-blame: supply a file, {_BOUND_CLAUSE}")

    _reject_leading_dash(file, shape="blame", label="file")
    _reject_leading_dash(repo, shape="blame", label="--repo")

    argv: "list[str]" = []
    if repo:
        argv.extend(["-C", repo])
    argv.append("blame")
    if start_line is not None or end_line is not None:
        lo = "" if start_line is None else str(start_line)
        hi = "" if end_line is None else str(end_line)
        argv.append(f"-L{lo},{hi}")
    argv.extend(["--", file])
    return argv


def _build_diff_argv(args: argparse.Namespace) -> "list[str]":
    """Validates the bound requirement (a revision range/pair or a path
    filter) and the injection guard, then constructs the discrete git argv
    for the diff shape. Raises `GitQueryError` for either failure -- never
    builds a partial argv."""
    rev_a = getattr(args, "rev_a", None)
    rev_b = getattr(args, "rev_b", None)
    path = getattr(args, "path", None)
    # repo is intentionally unrestricted (git -C <repo>, any path) -- this
    # grants no more access than the pre-existing execute(command="git",
    # args=["-C", ...]) path; it is a convenience parameter, not a sandbox.
    repo = getattr(args, "repo", None)

    if not (rev_a or rev_b or path):
        raise GitQueryError(
            f"git-diff: supply a revision range, two revisions, or --path, {_BOUND_CLAUSE}"
        )

    _reject_leading_dash(rev_a, shape="diff", label="revision")
    _reject_leading_dash(rev_b, shape="diff", label="revision")
    _reject_leading_dash(path, shape="diff", label="path")
    _reject_leading_dash(repo, shape="diff", label="--repo")

    argv: "list[str]" = []
    if repo:
        argv.extend(["-C", repo])
    argv.append("diff")
    if rev_a:
        argv.append(rev_a)
    if rev_b:
        argv.append(rev_b)
    if path:
        argv.extend(["--", path])
    return argv


def _run_and_render_git(argv: "list[str]") -> "dict[str, Any]":
    """The single render/return choke point every git-query shape dispatches
    through -- reuses `exec_cli.run_and_render` (no auto-index) so output
    bounding comes from the existing tail-window for free."""
    from iai_mcp.tools.exec_cli import run_and_render

    result = run_and_render("git", argv)
    return _flatten_git_result(result)


def _dispatch_git_query(build_argv: "Any", args: argparse.Namespace) -> int:
    """Shared command body every query shape's `cmd_git_*` delegates to:
    build the argv (or reject), dispatch through the one render/return
    choke point, and render the result identically regardless of shape."""
    try:
        argv = build_argv(args)
    except GitQueryError as e:
        print(str(e), file=sys.stderr)
        return 1

    result = _run_and_render_git(argv)

    if getattr(args, "json", False):
        print(json.dumps(result))
        return 0

    sys.stdout.write(result["stdout"])
    sys.stderr.write(result["stderr"])
    print(f"run_id: {result['run_id']} state: {result['state']}")
    return 0


def cmd_git_log(args: argparse.Namespace) -> int:
    """Shared `git-log` command body for both CLI surfaces."""
    return _dispatch_git_query(_build_log_argv, args)


def cmd_git_blame(args: argparse.Namespace) -> int:
    """Shared `git-blame` command body for both CLI surfaces."""
    return _dispatch_git_query(_build_blame_argv, args)


def cmd_git_diff(args: argparse.Namespace) -> int:
    """Shared `git-diff` command body for both CLI surfaces."""
    return _dispatch_git_query(_build_diff_argv, args)
