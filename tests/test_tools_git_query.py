"""`git_query` shared body: bound-by-construction validation, the leading-
dash injection guard, and the default-path secret-gate scrub, exercised
through both `iai-tools`/`iai-mcp tools` CLI surfaces.

Follows `tests/test_tools_exec_cli.py`'s naming/shape: real subprocesses
against a temp `IAI_TOOLS_ROOT`/socket, never a live `~/.iai-tools/`, plus a
hermetic scratch git repo created inside the test's own tmp dir.
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _execd_bin() -> str:
    for profile in ("release", "debug"):
        candidate = REPO / "target" / profile / "iai_tools_execd"
        if candidate.exists():
            return str(candidate)
    pytest.skip("iai_tools_execd not built; run: cargo build -p iai_mcp_tools_core --bins")


_TEST_IDLE_SECS = 2.0


def _assert_daemon_exited(sock_path: Path, idle_secs: float = _TEST_IDLE_SECS, margin: float = 3.0) -> None:
    """Confirms the daemon self-terminated after its idle window. Exactly
    ONE connect attempt, after the sleep -- polling would reset the idle
    clock (each probe connection is itself daemon activity)."""
    time.sleep(idle_secs + margin)
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.connect(str(sock_path))
    except (ConnectionRefusedError, FileNotFoundError):
        return
    probe.close()
    raise AssertionError(
        f"daemon at {sock_path} did not self-terminate within {idle_secs + margin}s"
    )


@pytest.fixture
def exec_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "iai-tools-root"
    # AF_UNIX's sun_path is capped at ~104 bytes on macOS -- pytest's own
    # tmp_path nests several directories deep and overflows that, so the
    # socket lives in its own short-path directory directly under /tmp.
    sock_dir = Path(tempfile.mkdtemp(prefix="iai-tools-git-", dir="/tmp"))
    sock_path = sock_dir / ".execd.sock"
    monkeypatch.setenv("IAI_TOOLS_ROOT", str(root))
    monkeypatch.setenv("IAI_TOOLS_SOCKET_PATH", str(sock_path))
    monkeypatch.setenv("IAI_TOOLS_EXECD_BIN", _execd_bin())
    monkeypatch.setenv("IAI_TOOLS_IDLE_SECS", str(_TEST_IDLE_SECS))
    try:
        yield {"root": root, "sock_path": sock_path}
    finally:
        shutil.rmtree(sock_dir, ignore_errors=True)


def _init_scratch_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "alice"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "alice@example.com"], check=True
    )
    subprocess.run(["git", "-C", str(path), "config", "commit.gpgsign", "false"], check=True)


def _commit_file(repo: Path, name: str, content: str, message: str) -> None:
    (repo / name).write_text(content)
    subprocess.run(["git", "-C", str(repo), "add", name], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", message], check=True)


@pytest.fixture
def scratch_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "scratch-repo"
    repo.mkdir()
    _init_scratch_repo(repo)
    _commit_file(repo, "a.txt", "hello\n", "add a.txt")
    _commit_file(repo, "b.txt", "world\n", "add b.txt")
    return repo


def _log_ns(
    *,
    since: "str | None" = None,
    until: "str | None" = None,
    author: "str | None" = None,
    path: "str | None" = None,
    repo: "str | None" = None,
    rev_range: "str | None" = None,
    json_out: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        since=since,
        until=until,
        author=author,
        path=path,
        repo=repo,
        rev_range=rev_range,
        json=json_out,
    )


# -- Task 1: log end-to-end, bound rejection, bound-crossing --------------


def test_git_log_scoped_by_since_returns_bounded_output_not_truncated(
    exec_env: dict, scratch_repo: Path, capsys: pytest.CaptureFixture
) -> None:
    from iai_mcp.tools.git_query import cmd_git_log

    ns = _log_ns(since="1970-01-01", repo=str(scratch_repo), json_out=True)
    rc = cmd_git_log(ns)
    assert rc == 0

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["truncated"] is False
    assert "add a.txt" in payload["stdout"]
    assert "add b.txt" in payload["stdout"]

    _assert_daemon_exited(exec_env["sock_path"])


def test_git_log_unqualified_is_rejected(capsys: pytest.CaptureFixture) -> None:
    from iai_mcp.tools.git_query import _BOUND_CLAUSE, cmd_git_log

    rc = cmd_git_log(_log_ns())
    assert rc == 1
    assert _BOUND_CLAUSE in capsys.readouterr().err


def test_git_log_dual_surface_parity_rejects_unqualified_query(capsys: pytest.CaptureFixture) -> None:
    """Both CLI surfaces reach the SAME shared-body rejection -- proves the
    MCP execute(mode=...) dispatch path (which delegates identically)
    inherits it too.
    """
    from iai_mcp.cli._tools import cmd_tools_git_log
    from iai_mcp.tools.git_query import _BOUND_CLAUSE
    from iai_mcp.tools_cli import _cmd_git_log

    rc_a = _cmd_git_log(_log_ns())
    err_a = capsys.readouterr().err
    rc_b = cmd_tools_git_log(_log_ns())
    err_b = capsys.readouterr().err

    assert rc_a == 1 and rc_b == 1
    assert _BOUND_CLAUSE in err_a
    assert _BOUND_CLAUSE in err_b


def test_git_log_bound_crossing_returns_truncated_within_cap(
    exec_env: dict, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from iai_mcp.tools.git_query import cmd_git_log

    repo = tmp_path / "big-repo"
    repo.mkdir()
    _init_scratch_repo(repo)
    # A single commit message body over TAIL_BYTE_CAP (64 KiB) reliably
    # crosses the tail-window cap in one call -- faster than many
    # --allow-empty commits crossing the 2000-line cap.
    big_message = "oversized commit body\n" + ("B" * (70 * 1024))
    _commit_file(repo, "f.txt", "x\n", big_message)

    ns = _log_ns(since="1970-01-01", repo=str(repo), json_out=True)
    rc = cmd_git_log(ns)
    assert rc == 0

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["truncated"] is True
    assert payload["elided_bytes"] > 0
    assert len(payload["stdout"]) <= 64 * 1024
    assert len(payload["stdout"]) < len(big_message)

    _assert_daemon_exited(exec_env["sock_path"])


def test_tools_cli_git_log_argparse_parses_flags() -> None:
    from iai_mcp.tools_cli import _build_parser

    ns = _build_parser().parse_args(
        ["git-log", "--since", "2020-01-01", "--path", "src", "HEAD~5..HEAD"]
    )
    assert ns.since == "2020-01-01"
    assert ns.path == "src"
    assert ns.rev_range == "HEAD~5..HEAD"


def test_git_log_dual_surface_parity_delegates_to_shared_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both `iai-tools git-log` and `iai-mcp tools git-log` delegate to the
    SAME `iai_mcp.tools.git_query.cmd_git_log` -- proven by import identity,
    so the two CLIs cannot drift."""
    from iai_mcp.cli import _tools as cli_tools
    from iai_mcp.tools import git_query as git_query_mod
    from iai_mcp import tools_cli

    calls: "list[argparse.Namespace]" = []
    monkeypatch.setattr(git_query_mod, "cmd_git_log", lambda a: (calls.append(a), 0)[1])

    ns = _log_ns(since="1970-01-01")
    assert tools_cli._cmd_git_log(ns) == 0
    assert cli_tools.cmd_tools_git_log(ns) == 0
    assert calls == [ns, ns]


# -- Task 2: argument-injection guard --------------------------------------


def _leading_dash_revision() -> str:
    # Fragment-assembled at runtime -- the test source never carries a
    # verbatim flag-injection literal.
    return "-" + "-" + "upload-pack=/tmp/x"


def _leading_dash_path() -> str:
    return "-" + "-output=/tmp/x"


def test_build_log_argv_places_path_after_separator() -> None:
    from iai_mcp.tools.git_query import _build_log_argv

    argv = _build_log_argv(_log_ns(rev_range="HEAD", path="src/foo.py"))
    assert argv == ["log", "HEAD", "--", "src/foo.py"]


def test_build_log_argv_prefixes_repo_with_dash_c() -> None:
    from iai_mcp.tools.git_query import _build_log_argv

    argv = _build_log_argv(_log_ns(since="2020-01-01", repo="/tmp/some-repo"))
    assert argv[:2] == ["-C", "/tmp/some-repo"]


def test_git_log_rejects_leading_dash_revision(capsys: pytest.CaptureFixture) -> None:
    from iai_mcp.tools.git_query import cmd_git_log

    ns = _log_ns(rev_range=_leading_dash_revision())
    rc = cmd_git_log(ns)
    assert rc == 1
    assert "revision" in capsys.readouterr().err


def test_git_log_rejects_leading_dash_path(capsys: pytest.CaptureFixture) -> None:
    from iai_mcp.tools.git_query import cmd_git_log

    ns = _log_ns(since="1970-01-01", path=_leading_dash_path())
    rc = cmd_git_log(ns)
    assert rc == 1
    assert "path" in capsys.readouterr().err


def test_git_log_valid_revision_and_path_still_work(
    exec_env: dict, scratch_repo: Path, capsys: pytest.CaptureFixture
) -> None:
    from iai_mcp.tools.git_query import cmd_git_log

    ns = _log_ns(repo=str(scratch_repo), rev_range="HEAD", path="a.txt", json_out=True)
    rc = cmd_git_log(ns)
    assert rc == 0

    payload = json.loads(capsys.readouterr().out.strip())
    assert "add a.txt" in payload["stdout"]

    _assert_daemon_exited(exec_env["sock_path"])


# -- Task 3: secret-gate scrub on the default response path ---------------


def _credential_token() -> str:
    # Fragment-assembled so the literal "ghp_"-prefixed shape never appears
    # contiguous in this file's own source (would trip the pre-commit
    # secret scan on the test file itself).
    prefix = "gh" + "p_"
    body = "".join(chr(ord("a") + (i % 26)) for i in range(24))
    return prefix + body


def test_git_log_scrubs_planted_credential_from_default_response(
    exec_env: dict, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from iai_mcp.tools.git_query import cmd_git_log

    repo = tmp_path / "secret-repo"
    repo.mkdir()
    _init_scratch_repo(repo)
    token = _credential_token()
    _commit_file(repo, "c.txt", "clean\n", f"leaked credential {token} in commit message")
    _commit_file(repo, "d.txt", "also clean\n", "add d.txt, legitimate content")

    ns = _log_ns(since="1970-01-01", repo=str(repo), json_out=True)
    rc = cmd_git_log(ns)
    assert rc == 0

    payload = json.loads(capsys.readouterr().out.strip())
    assert token not in payload["stdout"]
    assert "add d.txt, legitimate content" in payload["stdout"]

    _assert_daemon_exited(exec_env["sock_path"])


# -- git-blame and git-diff thin entry points ------------------------------


def _blame_ns(
    *,
    file: "str | None" = None,
    repo: "str | None" = None,
    start_line: "int | None" = None,
    end_line: "int | None" = None,
    json_out: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        file=file,
        repo=repo,
        start_line=start_line,
        end_line=end_line,
        json=json_out,
    )


def _diff_ns(
    *,
    rev_a: "str | None" = None,
    rev_b: "str | None" = None,
    path: "str | None" = None,
    repo: "str | None" = None,
    json_out: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        rev_a=rev_a,
        rev_b=rev_b,
        path=path,
        repo=repo,
        json=json_out,
    )


def test_git_blame_returns_bounded_output_for_file(
    exec_env: dict, scratch_repo: Path, capsys: pytest.CaptureFixture
) -> None:
    from iai_mcp.tools.git_query import cmd_git_blame

    ns = _blame_ns(file="a.txt", repo=str(scratch_repo), json_out=True)
    rc = cmd_git_blame(ns)
    assert rc == 0

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["truncated"] is False
    assert "hello" in payload["stdout"]

    _assert_daemon_exited(exec_env["sock_path"])


def test_git_blame_honors_line_range(
    exec_env: dict, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from iai_mcp.tools.git_query import cmd_git_blame

    repo = tmp_path / "blame-repo"
    repo.mkdir()
    _init_scratch_repo(repo)
    _commit_file(repo, "multi.txt", "line1\nline2\nline3\nline4\n", "add multi.txt")

    ns = _blame_ns(file="multi.txt", repo=str(repo), start_line=2, end_line=3, json_out=True)
    rc = cmd_git_blame(ns)
    assert rc == 0

    payload = json.loads(capsys.readouterr().out.strip())
    lines = [ln for ln in payload["stdout"].splitlines() if ln.strip()]
    assert len(lines) == 2
    assert "line2" in payload["stdout"]
    assert "line3" in payload["stdout"]
    assert "line1" not in payload["stdout"]
    assert "line4" not in payload["stdout"]

    _assert_daemon_exited(exec_env["sock_path"])


def test_git_blame_no_file_is_rejected(capsys: pytest.CaptureFixture) -> None:
    from iai_mcp.tools.git_query import _BOUND_CLAUSE, cmd_git_blame

    rc = cmd_git_blame(_blame_ns())
    assert rc == 1
    assert _BOUND_CLAUSE in capsys.readouterr().err


def test_build_blame_argv_places_file_after_separator() -> None:
    from iai_mcp.tools.git_query import _build_blame_argv

    argv = _build_blame_argv(_blame_ns(file="src/foo.py"))
    assert argv == ["blame", "--", "src/foo.py"]


def test_build_blame_argv_honors_line_range() -> None:
    from iai_mcp.tools.git_query import _build_blame_argv

    argv = _build_blame_argv(_blame_ns(file="a.txt", start_line=2, end_line=5))
    assert argv == ["blame", "-L2,5", "--", "a.txt"]


def test_build_blame_argv_prefixes_repo_with_dash_c() -> None:
    from iai_mcp.tools.git_query import _build_blame_argv

    argv = _build_blame_argv(_blame_ns(file="a.txt", repo="/tmp/some-repo"))
    assert argv[:2] == ["-C", "/tmp/some-repo"]


def test_git_diff_two_revisions_returns_bounded_output(
    exec_env: dict, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from iai_mcp.tools.git_query import cmd_git_diff

    repo = tmp_path / "diff-repo"
    repo.mkdir()
    _init_scratch_repo(repo)
    _commit_file(repo, "e.txt", "one\n", "add e.txt")
    _commit_file(repo, "e.txt", "one\ntwo\n", "extend e.txt")

    ns = _diff_ns(rev_a="HEAD~1", rev_b="HEAD", repo=str(repo), json_out=True)
    rc = cmd_git_diff(ns)
    assert rc == 0

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["truncated"] is False
    assert "+two" in payload["stdout"]

    _assert_daemon_exited(exec_env["sock_path"])


def test_git_diff_path_scoped_returns_bounded_output(
    exec_env: dict, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from iai_mcp.tools.git_query import cmd_git_diff

    repo = tmp_path / "diff-path-repo"
    repo.mkdir()
    _init_scratch_repo(repo)
    _commit_file(repo, "f.txt", "one\n", "add f.txt")
    _commit_file(repo, "g.txt", "two\n", "add g.txt")
    (repo / "f.txt").write_text("one\nmodified\n")

    ns = _diff_ns(path="f.txt", repo=str(repo), json_out=True)
    rc = cmd_git_diff(ns)
    assert rc == 0

    payload = json.loads(capsys.readouterr().out.strip())
    assert "+modified" in payload["stdout"]

    _assert_daemon_exited(exec_env["sock_path"])


def test_git_diff_unqualified_is_rejected(capsys: pytest.CaptureFixture) -> None:
    from iai_mcp.tools.git_query import _BOUND_CLAUSE, cmd_git_diff

    rc = cmd_git_diff(_diff_ns())
    assert rc == 1
    assert _BOUND_CLAUSE in capsys.readouterr().err


def test_build_diff_argv_places_path_after_separator() -> None:
    from iai_mcp.tools.git_query import _build_diff_argv

    argv = _build_diff_argv(_diff_ns(path="src/foo.py"))
    assert argv == ["diff", "--", "src/foo.py"]


def test_build_diff_argv_two_revisions() -> None:
    from iai_mcp.tools.git_query import _build_diff_argv

    argv = _build_diff_argv(_diff_ns(rev_a="HEAD~1", rev_b="HEAD"))
    assert argv == ["diff", "HEAD~1", "HEAD"]


def test_build_diff_argv_prefixes_repo_with_dash_c() -> None:
    from iai_mcp.tools.git_query import _build_diff_argv

    argv = _build_diff_argv(_diff_ns(path="src", repo="/tmp/some-repo"))
    assert argv[:2] == ["-C", "/tmp/some-repo"]


def test_tools_cli_git_blame_argparse_parses_flags() -> None:
    from iai_mcp.tools_cli import _build_parser

    ns = _build_parser().parse_args(
        ["git-blame", "--start-line", "2", "--end-line", "5", "src/foo.py"]
    )
    assert ns.file == "src/foo.py"
    assert ns.start_line == 2
    assert ns.end_line == 5


def test_tools_cli_git_diff_argparse_parses_flags() -> None:
    from iai_mcp.tools_cli import _build_parser

    ns = _build_parser().parse_args(["git-diff", "--path", "src", "HEAD~5", "HEAD"])
    assert ns.path == "src"
    assert ns.rev_a == "HEAD~5"
    assert ns.rev_b == "HEAD"


def test_git_blame_diff_dual_surface_parity_delegates_to_shared_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both `iai-tools`/`iai-mcp tools` blame/diff subcommands delegate to
    the SAME shared-body functions -- proven by import identity, so the two
    CLIs cannot drift."""
    from iai_mcp.cli import _tools as cli_tools
    from iai_mcp.tools import git_query as git_query_mod
    from iai_mcp import tools_cli

    blame_calls: "list[argparse.Namespace]" = []
    diff_calls: "list[argparse.Namespace]" = []
    monkeypatch.setattr(git_query_mod, "cmd_git_blame", lambda a: (blame_calls.append(a), 0)[1])
    monkeypatch.setattr(git_query_mod, "cmd_git_diff", lambda a: (diff_calls.append(a), 0)[1])

    blame_ns = _blame_ns(file="a.txt")
    diff_ns = _diff_ns(path="a.txt")
    assert tools_cli._cmd_git_blame(blame_ns) == 0
    assert cli_tools.cmd_tools_git_blame(blame_ns) == 0
    assert tools_cli._cmd_git_diff(diff_ns) == 0
    assert cli_tools.cmd_tools_git_diff(diff_ns) == 0
    assert blame_calls == [blame_ns, blame_ns]
    assert diff_calls == [diff_ns, diff_ns]


# -- bound-by-construction across all shapes, both entrypoints, and the
# per-shape secret-gate scrub ------------------------------------------------


@pytest.mark.parametrize("shape", ["log", "blame", "diff"])
def test_unqualified_query_rejected_across_shapes_both_entrypoints(
    shape: str, capsys: pytest.CaptureFixture
) -> None:
    """An unqualified query is rejected -- non-zero exit plus the shared-
    body rejection's fixed common clause on stderr -- for all three shapes
    (log/blame/diff), via BOTH CLI entrypoints (`tools_cli`, `cli/_tools`),
    which both delegate to the one shared body. Proves the shared-body
    rejection for the two CLI surfaces; the MCP-dispatch-path rejection is
    proven separately by `mcp-wrapper/test/toolsIndexGitQuery.test.ts`."""
    from iai_mcp.cli import _tools as cli_tools
    from iai_mcp.tools.git_query import _BOUND_CLAUSE
    from iai_mcp import tools_cli

    unqualified_ns_by_shape = {"log": _log_ns, "blame": _blame_ns, "diff": _diff_ns}
    entrypoints_by_shape = {
        "log": (tools_cli._cmd_git_log, cli_tools.cmd_tools_git_log),
        "blame": (tools_cli._cmd_git_blame, cli_tools.cmd_tools_git_blame),
        "diff": (tools_cli._cmd_git_diff, cli_tools.cmd_tools_git_diff),
    }

    ns_factory = unqualified_ns_by_shape[shape]
    for entrypoint in entrypoints_by_shape[shape]:
        rc = entrypoint(ns_factory())
        err = capsys.readouterr().err
        assert rc == 1, f"{shape} via {entrypoint}: expected rejection, got rc={rc}"
        assert _BOUND_CLAUSE in err, f"{shape} via {entrypoint}: missing common clause in {err!r}"


def test_git_blame_scrubs_planted_credential(
    exec_env: dict, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from iai_mcp.tools.git_query import cmd_git_blame

    repo = tmp_path / "secret-blame-repo"
    repo.mkdir()
    _init_scratch_repo(repo)
    token = _credential_token()
    _commit_file(repo, "s.txt", f"clean\nleaked {token} here\nlegit line\n", "add s.txt")

    ns = _blame_ns(file="s.txt", repo=str(repo), json_out=True)
    rc = cmd_git_blame(ns)
    assert rc == 0

    payload = json.loads(capsys.readouterr().out.strip())
    assert token not in payload["stdout"]
    assert "legit line" in payload["stdout"]

    _assert_daemon_exited(exec_env["sock_path"])


# -- truncation-boundary secret redaction ----------------------------------


def test_flatten_git_result_redacts_partial_secret_fragment_at_truncation_boundary() -> None:
    """The byte-cap truncation can cut mid-line, mid-token. A secret-shape
    pattern requires a minimum trailing length, so the surviving fragment
    of a cut token is too short to match and would otherwise leak
    unredacted. Only the secret-shaped tail fragment is replaced -- the
    rest of the (truncated) line survives."""
    from iai_mcp.tools.git_query import _flatten_git_result

    token = _credential_token()
    filler = "F" * 50
    raw_stdout = "legit line one\n" + filler + token[:15]  # no trailing newline: cut mid-line

    result = {
        "run_id": "r1",
        "state": "exited:0",
        "bounded_tail": {
            "stdout": raw_stdout,
            "stderr": "",
            "truncated": True,
            "elided_bytes": 999,
        },
    }
    flattened = _flatten_git_result(result)

    assert token not in flattened["stdout"]
    assert token[:15] not in flattened["stdout"]
    assert "legit line one" in flattened["stdout"]
    assert filler in flattened["stdout"]
    assert flattened["truncated"] is True
    assert flattened["elided_bytes"] == 999


def test_flatten_git_result_keeps_complete_lines_when_truncation_lands_on_newline() -> None:
    """A line-cap cut (as opposed to a byte-cap cut) always lands on a
    newline -- nothing partial to drop, so no complete line is lost."""
    from iai_mcp.tools.git_query import _flatten_git_result

    result = {
        "run_id": "r1",
        "state": "exited:0",
        "bounded_tail": {
            "stdout": "line one\nline two\n",
            "stderr": "",
            "truncated": True,
            "elided_bytes": 42,
        },
    }
    flattened = _flatten_git_result(result)

    assert "line one" in flattened["stdout"]
    assert "line two" in flattened["stdout"]


def test_flatten_git_result_preserves_single_huge_line_redacting_only_the_secret_tail() -> None:
    """A truncated window that is one giant line (no `\\n` anywhere -- a
    minified bundle, a lockfile, a generated single-line diff body) must
    not be wiped entirely: the bulk of the line survives, and only a
    trailing secret-shaped fragment is replaced."""
    from iai_mcp.tools.git_query import _flatten_git_result

    token = _credential_token()
    body = "y" * 200_000  # no trigger words, no newlines -- pure filler
    raw_stdout = (
        "diff --git a/bundle.min.js b/bundle.min.js\n"
        "index a1b2c3d..d4e5f6a 100644\n"
        "--- a/bundle.min.js\n"
        "+++ b/bundle.min.js\n"
        "@@ -1 +1 @@\n" + body + token[:15]  # single unterminated line, no \n
    )

    result = {
        "run_id": "r2",
        "state": "exited:0",
        "bounded_tail": {
            "stdout": raw_stdout,
            "stderr": "",
            "truncated": True,
            "elided_bytes": 500,
        },
    }
    flattened = _flatten_git_result(result)

    assert token not in flattened["stdout"]
    assert token[:15] not in flattened["stdout"]
    assert "bundle.min.js" in flattened["stdout"]
    assert flattened["stdout"].count("y") > len(body) - 256
    assert flattened["truncated"] is True


def test_flatten_git_result_preserves_non_secret_tail_regardless_of_truncation_state() -> None:
    """A legitimate final fragment that is not secret-shaped must survive
    byte-for-byte, whether or not it was truncated and whether or not it
    ends on a newline -- only a secret-shaped tail is ever redacted."""
    from iai_mcp.tools.git_query import _flatten_git_result

    no_newline_tail = "legit line\n" + ("z" * 100)  # cut mid-line, not secret-shaped

    truncated_result = {
        "run_id": "r3",
        "state": "exited:0",
        "bounded_tail": {
            "stdout": no_newline_tail,
            "stderr": "",
            "truncated": True,
            "elided_bytes": 10,
        },
    }
    assert _flatten_git_result(truncated_result)["stdout"] == no_newline_tail

    complete_result = {
        "run_id": "r4",
        "state": "exited:0",
        "bounded_tail": {
            "stdout": "line one\nline two\n",
            "stderr": "",
            "truncated": False,
            "elided_bytes": 0,
        },
    }
    assert _flatten_git_result(complete_result)["stdout"] == "line one\nline two\n"


def test_flatten_git_result_redacts_secret_tail_when_process_was_killed_not_truncated() -> None:
    """A process killed by the daemon's watcher mid-write can leave a
    genuinely partial final line with `truncated=False` (the Rust
    tail-window never elided anything -- the file on disk is just short).
    The tail redaction is unconditional, so it still catches a
    secret-shaped fragment even though `truncated` never fired."""
    from iai_mcp.tools.git_query import _flatten_git_result

    raw_stdout = "ordinary output\n" + "sk_live_" + "abc123"  # short of the full-match minimum

    result = {
        "run_id": "r5",
        "state": "killed:9",
        "bounded_tail": {
            "stdout": raw_stdout,
            "stderr": "",
            "truncated": False,
            "elided_bytes": 0,
        },
    }
    flattened = _flatten_git_result(result)

    assert "sk_live_abc123" not in flattened["stdout"]
    assert "ordinary output" in flattened["stdout"]
    assert flattened["truncated"] is False


def _commit_calibrated_boundary_message(repo: Path, token: str, target_start: int) -> None:
    """Adds (then amends) a single-commit message so that `token` begins at
    exactly `target_start` bytes into this repo's raw `git log
    --since=1970-01-01` stdout -- calibrated via a real, non-daemon `git`
    invocation (byte-for-byte identical to what the daemon would run,
    before the tail-window truncates it) so the final daemon-mediated call
    lands the byte-cap cut inside the token, reproducing a live-observed
    leak deterministically."""

    def raw_log_bytes() -> bytes:
        return subprocess.run(
            ["git", "-C", str(repo), "log", "--since=1970-01-01"],
            check=True,
            capture_output=True,
        ).stdout

    filler_len = 0
    _commit_file(repo, "boundary.txt", "x\n", ("F" * filler_len) + token)
    start = raw_log_bytes().find(token.encode())
    assert start != -1

    filler_len += target_start - start
    assert filler_len >= 0
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "commit",
            "--amend",
            "-q",
            "-m",
            ("F" * filler_len) + token,
        ],
        check=True,
    )
    start = raw_log_bytes().find(token.encode())
    assert start == target_start, f"calibration failed: got {start}, wanted {target_start}"


def test_git_log_scrubs_credential_straddling_truncation_boundary(
    exec_env: dict, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Live reproduction of the reviewer's CR-01 scenario: a credential
    token straddles the 64 KiB tail-window cap so only its first 15
    characters survive truncation. Neither the full token nor that partial
    prefix may appear in the response."""
    from iai_mcp.tools.git_query import cmd_git_log

    repo = tmp_path / "boundary-repo"
    repo.mkdir()
    _init_scratch_repo(repo)
    token = _credential_token()
    target_start = 64 * 1024 - 15
    _commit_calibrated_boundary_message(repo, token, target_start)

    ns = _log_ns(since="1970-01-01", repo=str(repo), json_out=True)
    rc = cmd_git_log(ns)
    assert rc == 0

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["truncated"] is True
    assert token not in payload["stdout"]
    assert token[:15] not in payload["stdout"]

    _assert_daemon_exited(exec_env["sock_path"])


def test_git_diff_scrubs_planted_credential(
    exec_env: dict, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from iai_mcp.tools.git_query import cmd_git_diff

    repo = tmp_path / "secret-diff-repo"
    repo.mkdir()
    _init_scratch_repo(repo)
    _commit_file(repo, "t.txt", "one\n", "add t.txt")
    token = _credential_token()
    (repo / "t.txt").write_text(f"one\nleaked {token} in diff hunk\nlegit added line\n")

    ns = _diff_ns(path="t.txt", repo=str(repo), json_out=True)
    rc = cmd_git_diff(ns)
    assert rc == 0

    payload = json.loads(capsys.readouterr().out.strip())
    assert token not in payload["stdout"]
    assert "legit added line" in payload["stdout"]

    _assert_daemon_exited(exec_env["sock_path"])
