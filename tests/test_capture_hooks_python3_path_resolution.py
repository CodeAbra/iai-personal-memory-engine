"""Proves the deploy hooks resolve python3 via PATH, not a fixed absolute path.

An exit-code check would false-pass here: the real /usr/bin/python3 exists on
the macOS/Linux test runner even when a hook still hardcodes that path. The
PATH-shim sentinel makes PATH-resolution itself the thing under test.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

HOOKS_DIR = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "_deploy" / "hooks"

HOOK_NAMES = [
    "iai-mcp-turn-capture.sh",
    "iai-mcp-session-capture.sh",
    "iai-mcp-hermes-capture.sh",
    "iai-mcp-antigravity-capture.sh",
    "iai-mcp-session-recall.sh",
    "iai-mcp-cursor-session-recall.sh",
    "iai-mcp-antigravity-recall.sh",
    "iai-mcp-hermes-recall.sh",
    "iai-tools-session-index.sh",
]


def _make_python3_shim(shim_dir: Path, sentinel: Path) -> None:
    shim = shim_dir / "python3"
    shim.write_text(
        "#!/bin/sh\n"
        f"touch '{sentinel}'\n"
        "exec /usr/bin/python3 \"$@\"\n"
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture(scope="module")
def jq_free_bin(tmp_path_factory) -> Path:
    # Every hook's other shell dependencies (mktemp, stat, sed, ...) live
    # alongside jq in /usr/bin; symlink all of it except jq so the fallback
    # branch is actually exercised instead of merely omitted-by-luck.
    curated = tmp_path_factory.mktemp("usr-bin-no-jq")
    for entry in Path("/usr/bin").iterdir():
        if entry.name == "jq":
            continue
        try:
            (curated / entry.name).symlink_to(entry)
        except OSError:
            pass
    return curated


@pytest.mark.parametrize("hook_name", HOOK_NAMES)
def test_hook_resolves_python3_via_path(hook_name: str, tmp_path: Path, jq_free_bin: Path) -> None:
    hook_path = HOOKS_DIR / hook_name
    assert hook_path.exists(), f"hook script missing: {hook_path}"

    shim_dir = tmp_path / "shim-bin"
    shim_dir.mkdir()
    sentinel = tmp_path / "shim-was-used.marker"
    _make_python3_shim(shim_dir, sentinel)

    home = tmp_path / "home"
    home.mkdir()

    env = os.environ.copy()
    env["HOME"] = str(home)
    # jq must not resolve, so the fallback branch is taken; the shim must be
    # found before any real python3 on PATH.
    env["PATH"] = f"{shim_dir}:{jq_free_bin}:/bin"

    payload = json.dumps({"session_id": "x", "cwd": str(tmp_path)})
    subprocess.run(
        ["bash", str(hook_path)],
        input=payload,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert sentinel.exists(), (
        f"{hook_name} must resolve python3 via PATH, not a hardcoded absolute path"
    )


def _make_wrapper_shim(bin_dir: Path, name: str, sentinel: Path) -> None:
    """A `timeout`/`gtimeout` stand-in: drops its own duration arg and execs
    straight through, so the underlying python3 shim still runs."""
    shim = bin_dir / name
    shim.write_text(
        "#!/bin/sh\n"
        f"touch '{sentinel}'\n"
        "shift\n"
        "exec \"$@\"\n"
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _make_logging_python3_shim(bin_dir: Path, log_path: Path) -> None:
    """Like `_make_python3_shim`, but records every invocation's argv so a
    caller can tell which of several python3 call sites in one hook fired --
    a bare touch-sentinel cannot distinguish "reached this line" from
    "reached some other, earlier-executed python3 call in the same file"."""
    shim = bin_dir / "python3"
    shim.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$*\" >> '{log_path}'\n"
        "exec /usr/bin/python3 \"$@\"\n"
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


_MINIMAL_UTILS = (
    "bash", "cat", "mktemp", "mkdir", "date", "rm", "mv", "touch",
    "find", "sort", "xargs", "cut", "cksum",
)


def _make_minimal_bin(tmp_path_factory) -> Path:
    """PATH dir carrying only an explicit allow-list of coreutils. Unlike
    jq_free_bin (which mirrors all of /usr/bin except jq), this never leaks
    timeout/gtimeout/stat/git/jq in from the host, so command-resolution
    branches that depend on their absence are deterministic across hosts.
    Deliberately never combined with "/bin" in a PATH string: on merged-/usr
    Linux distros /bin is a symlink to /usr/bin, which would re-leak
    whatever this allow-list excludes.
    """
    curated = tmp_path_factory.mktemp("minimal-bin")
    for name in _MINIMAL_UTILS:
        src = Path("/usr/bin") / name
        if not src.exists():
            src = Path("/bin") / name
        if src.exists():
            try:
                (curated / name).symlink_to(src)
            except OSError:
                pass
    return curated


def _turn_capture_payload(tmp_path: Path) -> str:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text('{"type":"user","message":{"content":"hi there"}}\n')
    return json.dumps({
        "session_id": "x",
        "cwd": str(tmp_path),
        "transcript_path": str(transcript),
        "prompt_id": "p1",
    })


@pytest.mark.parametrize(
    ("wrapper_name", "other_wrapper"),
    [("timeout", "gtimeout"), ("gtimeout", "timeout")],
)
def test_turn_capture_reaches_main_invocation_via_wrapper(
    tmp_path: Path, tmp_path_factory, wrapper_name: str, other_wrapper: str,
) -> None:
    # iai-mcp-turn-capture.sh:839 (timeout) / :841 (gtimeout): supplying
    # transcript_path + prompt_id bypasses the early-exit guard at line 59
    # so the main capture invocation actually runs. extract()'s own python3
    # fallback (line 41) always fires first regardless (jq is excluded from
    # PATH here), so a shared touch-sentinel would stay green even if
    # 839/841 hardcoded /usr/bin/python3 -- log argv instead: the main
    # invocation passes prompt_id positionally, extract()'s call passes none.
    hook_path = HOOKS_DIR / "iai-mcp-turn-capture.sh"
    shim_dir = tmp_path / "shim-bin"
    shim_dir.mkdir()
    py_log = tmp_path / "python3-calls.log"
    _make_logging_python3_shim(shim_dir, py_log)
    wrapper_sentinel = tmp_path / f"{wrapper_name}-used.marker"
    _make_wrapper_shim(shim_dir, wrapper_name, wrapper_sentinel)
    assert not (shim_dir / other_wrapper).exists()

    minimal_bin = _make_minimal_bin(tmp_path_factory)
    home = tmp_path / "home"
    home.mkdir()
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["PATH"] = f"{shim_dir}:{minimal_bin}"

    subprocess.run(
        ["bash", str(hook_path)],
        input=_turn_capture_payload(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert wrapper_sentinel.exists(), f"{wrapper_name} wrapper must be reached"
    calls = py_log.read_text().splitlines() if py_log.exists() else []
    assert any("p1" in line for line in calls), (
        "python3 must be PATH-resolved at the main capture invocation, not "
        f"hardcoded (calls observed: {calls!r})"
    )


def test_turn_capture_reaches_main_invocation_without_any_wrapper(
    tmp_path: Path, tmp_path_factory,
) -> None:
    # iai-mcp-turn-capture.sh:843: no timeout/gtimeout on PATH at all -- the
    # plain python3 call must still be PATH-resolved.
    hook_path = HOOKS_DIR / "iai-mcp-turn-capture.sh"
    shim_dir = tmp_path / "shim-bin"
    shim_dir.mkdir()
    py_log = tmp_path / "python3-calls.log"
    _make_logging_python3_shim(shim_dir, py_log)

    minimal_bin = _make_minimal_bin(tmp_path_factory)
    assert not (minimal_bin / "timeout").exists()
    assert not (minimal_bin / "gtimeout").exists()
    home = tmp_path / "home"
    home.mkdir()
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["PATH"] = f"{shim_dir}:{minimal_bin}"

    subprocess.run(
        ["bash", str(hook_path)],
        input=_turn_capture_payload(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    calls = py_log.read_text().splitlines() if py_log.exists() else []
    assert any("p1" in line for line in calls), (
        "python3 must be PATH-resolved at the main capture invocation, not "
        f"hardcoded (calls observed: {calls!r})"
    )


def test_tools_session_index_dir_mtime_python3_fallback(
    tmp_path: Path, tmp_path_factory,
) -> None:
    # iai-tools-session-index.sh:56: dir_mtime()'s python3 fallback only
    # fires when `stat` is unavailable AND the lock mkdir fails (a held
    # lock). Identify THIS call (not extract()'s unrelated fallback at line
    # 34) by logging argv: dir_mtime passes the lock dir path positionally,
    # extract() does not.
    hook_path = HOOKS_DIR / "iai-tools-session-index.sh"
    shim_dir = tmp_path / "shim-bin"
    shim_dir.mkdir()
    py_log = tmp_path / "python3-calls.log"
    _make_logging_python3_shim(shim_dir, py_log)

    minimal_bin = _make_minimal_bin(tmp_path_factory)
    assert not (minimal_bin / "stat").exists()
    assert not (minimal_bin / "git").exists()

    home = tmp_path / "home"
    home.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    env = os.environ.copy()
    env["HOME"] = str(home)
    env["PATH"] = f"{shim_dir}:{minimal_bin}"

    # Mirror the hook's own path_hash computation with the SAME cksum
    # binary the hook resolves, so the lock dir is pre-created at the exact
    # path the hook will probe.
    cksum_bin = str(minimal_bin / "cksum")
    cksum_out = subprocess.run(
        [cksum_bin], input=str(project_dir).encode(),
        capture_output=True, timeout=10,
    ).stdout.decode()
    path_hash = cksum_out.split()[0]
    watermark_dir = home / ".iai-mcp" / "tools" / ".watermarks"
    watermark_dir.mkdir(parents=True)
    lock_dir = watermark_dir / f"{path_hash}.lock"
    lock_dir.mkdir()

    payload = json.dumps({"cwd": str(project_dir)})
    subprocess.run(
        ["bash", str(hook_path)],
        input=payload,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    calls = py_log.read_text().splitlines() if py_log.exists() else []
    assert any(str(lock_dir) in line for line in calls), (
        "dir_mtime() must resolve python3 via PATH when stat is absent and "
        f"the lock is held; calls observed: {calls!r}"
    )
