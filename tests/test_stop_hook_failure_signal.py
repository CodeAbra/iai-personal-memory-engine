"""Stop-hook contract: both silent-failure sites (CLI not found, and
capture-turn-deferred nonzero/timeout) must write a durable marker line
using only POSIX shell primitives -- no dependency on the CLI itself,
since the CLI-not-found site is exactly the case where it is unreachable --
and the hook must still exit 0 in every case (the response must never be
blocked by a capture failure)."""
from __future__ import annotations

import json
import os
import platform
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HOOK = REPO / "src" / "iai_mcp" / "_deploy" / "hooks" / "iai-mcp-session-capture.sh"
MARKER_REL = Path(".iai-mcp") / ".capture-state" / ".stop-hook-failures.jsonl"

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="bash + POSIX shell hook",
)


def _install_shim(home: Path, log_path: Path, exit_code: int = 0) -> Path:
    bin_dir = home / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    shim = bin_dir / "iai-mcp"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        f"echo \"$@\" >> {log_path}\n"
        f"exit {exit_code}\n"
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    cli_cache = home / ".iai-mcp" / ".cli-path"
    cli_cache.parent.mkdir(parents=True, exist_ok=True)
    cli_cache.write_text(str(shim))
    return shim


def _make_transcript(home: Path, session_id: str) -> Path:
    projects = home / ".claude" / "projects" / "fakeproj"
    projects.mkdir(parents=True, exist_ok=True)
    transcript = projects / f"{session_id}.jsonl"
    transcript.write_text(
        json.dumps({"type": "user", "message": {"role": "user", "content": "hi"}}) + "\n"
    )
    return transcript


# A minimal PATH with no chance of resolving a REAL iai-mcp install already
# on the host (e.g. ~/.local/bin) -- the CLI-not-found tests need the
# resolution to genuinely fail, not accidentally find the developer's own
# install via inherited PATH.
_NO_CLI_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"


def _run_hook(
    home: Path,
    session_id: str,
    transcript: Path,
    path_prefix: str = "",
    path_override: str | None = None,
) -> subprocess.CompletedProcess:
    stdin = json.dumps({
        "session_id": session_id,
        "transcript_path": str(transcript),
        "cwd": str(home),
    })
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["PATH"] = path_override if path_override is not None else path_prefix + env.get("PATH", "")
    return subprocess.run(
        ["bash", str(HOOK)],
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        timeout=40,
    )


def _read_marker_lines(home: Path) -> list[dict]:
    marker = home / MARKER_REL
    if not marker.exists():
        return []
    lines = []
    for raw in marker.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        lines.append(json.loads(raw))
    return lines


def test_capture_turn_deferred_nonzero_writes_marker_and_hook_still_exits_0(tmp_path):
    home = tmp_path
    sid = "SESSION-FAIL-NONZERO"
    shim_log = home / "shim.log"
    _install_shim(home, shim_log, exit_code=1)
    transcript = _make_transcript(home, sid)

    result = _run_hook(home, sid, transcript, path_prefix=f"{home}/bin:")

    assert result.returncode == 0, result.stderr

    lines = _read_marker_lines(home)
    assert len(lines) == 1, lines
    entry = lines[0]
    assert entry["session_id"] == sid
    assert entry["rc"] == "1"
    assert entry["channel"] == "settings"
    assert "ts" in entry


def test_cli_not_found_writes_marker_and_hook_still_exits_0(tmp_path):
    home = tmp_path
    sid = "SESSION-FAIL-NO-CLI"
    transcript = _make_transcript(home, sid)

    # No shim installed anywhere on PATH, no cache, no baked-in candidate
    # exists under this fake $HOME -- iai_cli resolution must fail entirely
    # (the pure-shell, no-CLI-dependency property under test).
    result = _run_hook(home, sid, transcript, path_override=_NO_CLI_PATH)

    assert result.returncode == 0, result.stderr

    lines = _read_marker_lines(home)
    assert len(lines) == 1, lines
    entry = lines[0]
    assert entry["session_id"] == sid
    assert entry["rc"] == "cli-not-found"
    assert "ts" in entry


def test_cached_cli_path_pointing_at_missing_file_writes_marker(tmp_path):
    home = tmp_path
    sid = "SESSION-FAIL-STALE-CACHE"
    transcript = _make_transcript(home, sid)

    cli_cache = home / ".iai-mcp" / ".cli-path"
    cli_cache.parent.mkdir(parents=True, exist_ok=True)
    cli_cache.write_text(str(home / "bin" / "iai-mcp"))  # never created

    result = _run_hook(home, sid, transcript, path_override=_NO_CLI_PATH)

    assert result.returncode == 0, result.stderr

    lines = _read_marker_lines(home)
    assert len(lines) == 1, lines
    assert lines[0]["rc"] == "cli-not-found"


def test_successful_capture_writes_no_marker(tmp_path):
    home = tmp_path
    sid = "SESSION-OK"
    shim_log = home / "shim.log"
    _install_shim(home, shim_log, exit_code=0)
    transcript = _make_transcript(home, sid)

    result = _run_hook(home, sid, transcript, path_prefix=f"{home}/bin:")

    assert result.returncode == 0, result.stderr
    assert _read_marker_lines(home) == []


def test_multiple_failures_append_multiple_marker_lines(tmp_path):
    home = tmp_path
    sid = "SESSION-FAIL-REPEAT"
    shim_log = home / "shim.log"
    _install_shim(home, shim_log, exit_code=2)
    transcript = _make_transcript(home, sid)

    for _ in range(2):
        result = _run_hook(home, sid, transcript, path_prefix=f"{home}/bin:")
        assert result.returncode == 0, result.stderr

    lines = _read_marker_lines(home)
    assert len(lines) == 2, lines
    assert all(entry["rc"] == "2" for entry in lines)


def test_session_id_with_quote_does_not_break_marker_json(tmp_path):
    """A crafted session_id containing a JSON-breaking character must not
    corrupt the marker line -- it must be substituted with a safe
    placeholder instead of interpolated verbatim."""
    home = tmp_path
    sid = 'evil"session\\injected'
    shim_log = home / "shim.log"
    _install_shim(home, shim_log, exit_code=1)
    transcript = _make_transcript(home, sid)

    result = _run_hook(home, sid, transcript, path_prefix=f"{home}/bin:")
    assert result.returncode == 0, result.stderr

    lines = _read_marker_lines(home)
    assert len(lines) == 1, lines
    assert lines[0]["session_id"] == "invalid"
    assert lines[0]["rc"] == "1"


def test_marker_file_growth_is_capped(tmp_path):
    """The marker file must not grow without bound -- once it exceeds the
    configured cap, the oldest lines are dropped."""
    home = tmp_path
    sid = "SESSION-FAIL-CAP"
    shim_log = home / "shim.log"
    _install_shim(home, shim_log, exit_code=1)
    transcript = _make_transcript(home, sid)

    marker = home / MARKER_REL
    marker.parent.mkdir(parents=True, exist_ok=True)
    # Pre-seed the marker past the cap so a single append must trigger a
    # truncation, without running the hook thousands of times.
    with marker.open("w", encoding="utf-8") as fh:
        for i in range(2005):
            fh.write(
                json.dumps({
                    "ts": f"seed-{i}", "session_id": "seed",
                    "rc": "1", "channel": "settings",
                }) + "\n"
            )

    result = _run_hook(home, sid, transcript, path_prefix=f"{home}/bin:")
    assert result.returncode == 0, result.stderr

    lines = _read_marker_lines(home)
    assert len(lines) <= 2000, len(lines)
    # The newest entry (this run's) must survive the truncation.
    assert lines[-1]["session_id"] == sid
