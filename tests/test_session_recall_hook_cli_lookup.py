from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="POSIX shell hook"
)

REPO_ROOT = Path(__file__).resolve().parent.parent
HOOK_PATH = (
    REPO_ROOT / "src" / "iai_mcp" / "_deploy" / "hooks" / "iai-mcp-session-recall.sh"
)


def _make_stub_cli(dir_: Path) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    cli = dir_ / "iai-mcp"
    cli.write_text("#!/usr/bin/env bash\necho RECALLED\n")
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return cli


def _run_hook(home: Path, path_entries: list[str]):
    env = os.environ.copy()
    env["HOME"] = str(home)
    # Drop any developer override so the test exercises the built-in lookup.
    env.pop("IAI_MCP_SESSION_RECALL_CLI", None)
    env["PATH"] = os.pathsep.join([*path_entries, "/usr/bin", "/bin"])
    return subprocess.run(
        ["bash", str(HOOK_PATH)],
        input='{"session_id":"x","source":"startup","cwd":"/tmp","transcript_path":""}',
        env=env,
        capture_output=True,
        text=True,
        timeout=15.0,
    )


def test_hook_resolves_cli_from_path(tmp_path):
    """A pip/pipx install that is only reachable via PATH must be found.

    The candidate list cannot enumerate every install layout, so PATH is
    consulted before falling back to it. Without this the hook exits 0 with
    no output and recall is silently disabled.
    """
    home = tmp_path / "home"
    (home / ".iai-mcp").mkdir(parents=True)
    stub_dir = tmp_path / "path-only"
    _make_stub_cli(stub_dir)

    proc = _run_hook(home, [str(stub_dir)])

    assert proc.returncode == 0, proc.stderr
    cache = home / ".iai-mcp" / ".cli-path"
    assert cache.exists(), "PATH-resolved CLI was not cached"
    assert cache.read_text().strip() == str(stub_dir / "iai-mcp")


def test_hook_resolves_cli_from_user_local_bin(tmp_path):
    """`pip install --user` lands in ~/.local/bin; it must be a candidate."""
    home = tmp_path / "home"
    (home / ".iai-mcp").mkdir(parents=True)
    _make_stub_cli(home / ".local" / "bin")

    # Not on PATH — force the candidate-list branch to do the work.
    proc = _run_hook(home, [])

    assert proc.returncode == 0, proc.stderr
    cache = home / ".iai-mcp" / ".cli-path"
    assert cache.exists(), "~/.local/bin CLI was not found"
    assert cache.read_text().strip() == str(home / ".local" / "bin" / "iai-mcp")


def test_hook_still_exits_zero_when_cli_is_absent(tmp_path):
    """Fail-safe is preserved: no CLI anywhere is still a silent exit 0."""
    home = tmp_path / "home"
    (home / ".iai-mcp").mkdir(parents=True)

    proc = _run_hook(home, [])

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ""
    assert not (home / ".iai-mcp" / ".cli-path").exists()
