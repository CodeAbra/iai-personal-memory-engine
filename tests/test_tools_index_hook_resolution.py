"""Hermetic regression coverage for the SessionStart code-index hook's
CLI-resolution ladder (source hook only, never the installed copy)."""

import json
import os
import stat
import subprocess
from pathlib import Path

HOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "iai_mcp"
    / "_deploy"
    / "hooks"
    / "iai-tools-session-index.sh"
)

def _make_executable_stub(path: Path, marker_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/sh\n: > "{marker_path}"\n')
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _make_executable_stub_script(path: Path, script: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _run_hook(cwd: str, home: Path, extra_env: dict) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "HOME": str(home),
        **extra_env,
    }
    env.pop("IAI_MCP_TOOLS_INDEX_CLI", None)
    payload = json.dumps({"cwd": cwd, "source": "resume"})
    return subprocess.run(
        ["sh", str(HOOK_PATH)],
        input=payload,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_trusted_path_file_resolves_cli(tmp_path):
    assert HOOK_PATH.is_file()

    tmp_home = tmp_path / "home"
    tmp_home.mkdir()
    tmp_project = tmp_path / "project"
    tmp_project.mkdir()

    marker_cached = tmp_path / "marker-cached"
    cached_stub = tmp_path / "cached-cli" / "iai-tools"
    _make_executable_stub(cached_stub, marker_cached)

    cache_dir = tmp_home / ".iai-mcp"
    cache_dir.mkdir(parents=True)
    (cache_dir / ".tools-cli-path").write_text(str(cached_stub), encoding="utf-8")

    extra_env = {"PATH": "/usr/bin:/bin"}
    result = _run_hook(str(tmp_project), tmp_home, extra_env)

    assert result.returncode == 0
    assert marker_cached.exists(), "installer-written trusted path-file CLI should have run"


def test_no_venv_stays_fail_safe(tmp_path):
    assert HOOK_PATH.is_file()

    tmp_home = tmp_path / "home"
    tmp_home.mkdir()
    tmp_project = tmp_path / "project-empty"
    tmp_project.mkdir()

    extra_env = {"PATH": "/usr/bin:/bin"}

    result = _run_hook(str(tmp_project), tmp_home, extra_env)

    assert result.returncode == 0

    log_dir = tmp_home / ".iai-mcp" / "logs"
    assert log_dir.is_dir()
    log_files = list(log_dir.glob("tools-index-*.log"))
    assert len(log_files) == 1
    log_content = log_files[0].read_text()
    assert "CLI not found" in log_content


def test_project_local_venv_binary_never_auto_executes(tmp_path):
    """The removed discovery surface must not resurrect itself: an
    executable planted at <project>/.venv/bin/iai-tools is never a trusted
    resolution source, git-tracked or not, symlinked or not."""
    assert HOOK_PATH.is_file()

    tmp_home = tmp_path / "home"
    tmp_home.mkdir()
    tmp_project = tmp_path / "project"
    tmp_project.mkdir()

    marker_planted = tmp_path / "marker-planted"
    planted_stub = tmp_project / ".venv" / "bin" / "iai-tools"
    _make_executable_stub(planted_stub, marker_planted)

    extra_env = {"PATH": "/usr/bin:/bin"}
    result = _run_hook(str(tmp_project), tmp_home, extra_env)

    assert result.returncode == 0
    assert not marker_planted.exists(), "project-tree binary must never auto-execute"

    log_dir = tmp_home / ".iai-mcp" / "logs"
    log_files = list(log_dir.glob("tools-index-*.log"))
    assert len(log_files) == 1
    # Machine-independent: a real system `iai-tools` at one of the baked
    # candidate paths (/opt/homebrew/bin, /usr/local/bin) could still
    # resolve here and make an exact "CLI not found" assertion false for
    # the wrong reason. Assert on the deleted-surface property instead: the
    # log must never reference the project's own .venv path, regardless of
    # which resolution rung (or none) ends up satisfied.
    log_content = log_files[0].read_text()
    assert str(planted_stub) not in log_content, (
        "hook log references the project-local .venv path at all"
    )


def test_non_timeout_failure_backs_off_without_reinvoking(tmp_path):
    """rc -ne 0 (not just rc -eq 124) must stamp the attempt watermark, so a
    plain non-zero exit backs off the same as a timeout instead of
    re-invoking the CLI on every session start."""
    assert HOOK_PATH.is_file()

    tmp_home = tmp_path / "home"
    tmp_home.mkdir()
    tmp_project = tmp_path / "project"
    tmp_project.mkdir()
    (tmp_project / "file.txt").write_text("content\n")

    counter = tmp_path / "calls.log"
    cli_stub = tmp_path / "cli" / "iai-tools"
    _make_executable_stub_script(cli_stub, f'#!/bin/sh\necho hit >> "{counter}"\nexit 1\n')

    cache_dir = tmp_home / ".iai-mcp"
    cache_dir.mkdir(parents=True)
    (cache_dir / ".tools-cli-path").write_text(str(cli_stub), encoding="utf-8")

    extra_env = {"PATH": "/usr/bin:/bin"}

    result1 = _run_hook(str(tmp_project), tmp_home, extra_env)
    assert result1.returncode == 0
    assert counter.read_text().count("hit") == 1

    watermark_dir = tmp_home / ".iai-mcp" / "tools" / ".watermarks"
    attempt_files = list(watermark_dir.glob("*.attempted"))
    attempt_ts_files = list(watermark_dir.glob("*.attempted-ts"))
    assert len(attempt_files) == 1, "rc=1 must stamp the attempt watermark"
    assert len(attempt_ts_files) == 1, "rc=1 must stamp the attempt timestamp"

    log_dir = tmp_home / ".iai-mcp" / "logs"
    log_files = list(log_dir.glob("tools-index-*.log"))
    assert len(log_files) == 1
    assert "rc=1" in log_files[0].read_text()

    result2 = _run_hook(str(tmp_project), tmp_home, extra_env)
    assert result2.returncode == 0
    assert counter.read_text().count("hit") == 1, (
        "backoff failed to converge -- second run re-invoked the CLI"
    )
    assert "backoff" in log_files[0].read_text()
