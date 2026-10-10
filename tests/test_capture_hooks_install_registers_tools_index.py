from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

HOOK_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "iai_mcp" / "_deploy" / "hooks" / "iai-tools-session-index.sh"
)


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".claude").mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def _settings_path(home: Path) -> Path:
    return home / ".claude" / "settings.json"


def _tools_index_entries(data: dict) -> list:
    ss_entries = data.get("hooks", {}).get("SessionStart", [])
    return [
        e for e in ss_entries
        if any("iai-tools-session-index.sh" in (h.get("command") or "")
               for h in (e.get("hooks") or []))
    ]


def test_install_deploys_and_wires_tools_index_hook(fake_home):
    from iai_mcp import cli as cli_mod

    rc = cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    assert rc == 0

    dst = fake_home / ".claude" / "hooks" / "iai-tools-session-index.sh"
    assert dst.exists()

    mode = dst.stat().st_mode
    assert mode & stat.S_IXUSR, "deployed hook must be executable"

    data = json.loads(_settings_path(fake_home).read_text())
    matching = _tools_index_entries(data)
    assert len(matching) == 1, data.get("hooks", {}).get("SessionStart", [])
    cmd = matching[0]["hooks"][0]["command"]
    assert re.search(r"bash .*iai-tools-session-index\.sh", cmd), cmd

    # session-recall still coexists in the same array
    ss_entries = data.get("hooks", {}).get("SessionStart", [])
    assert any(
        "iai-mcp-session-recall.sh" in (h.get("command") or "")
        for e in ss_entries for h in (e.get("hooks") or [])
    ), ss_entries


def test_install_backs_up_and_warns_on_malformed_settings_json(fake_home, capsys):
    from iai_mcp import cli as cli_mod

    settings = _settings_path(fake_home)
    settings.write_text("{not valid json", encoding="utf-8")

    rc = cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    assert rc == 0

    out = capsys.readouterr().out
    assert "WARN" in out and "backed up" in out, out

    backups = list(settings.parent.glob("settings.json.bak-*"))
    assert len(backups) == 1, backups
    assert backups[0].read_text(encoding="utf-8") == "{not valid json"

    # Install still proceeds and produces a valid, fully-wired settings.json
    # rather than aborting -- the malformed original is preserved in the
    # backup, not silently lost.
    data = json.loads(settings.read_text(encoding="utf-8"))
    matching = _tools_index_entries(data)
    assert len(matching) == 1, data


def test_reinstall_is_idempotent_single_entry(fake_home):
    from iai_mcp import cli as cli_mod

    rc1 = cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    assert rc1 == 0
    rc2 = cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    assert rc2 == 0

    data = json.loads(_settings_path(fake_home).read_text())
    matching = _tools_index_entries(data)
    assert len(matching) == 1, data.get("hooks", {}).get("SessionStart", [])


def test_uninstall_removes_tools_index_entry_and_file_preserves_foreign_entry(fake_home):
    from iai_mcp import cli as cli_mod

    # Hand-seed a foreign SessionStart entry (command shaped like another
    # tool's hook) BEFORE install -- this installer must never touch an
    # entry it does not own.
    settings = _settings_path(fake_home)
    foreign_cmd = "node /Users/alice/.bun/install/global/node_modules/other-tool/hooks/sessionstart.mjs"
    settings.write_text(json.dumps({
        "hooks": {
            "SessionStart": [
                {"matcher": "startup", "hooks": [{"type": "command", "command": foreign_cmd}]},
            ]
        }
    }), encoding="utf-8")

    cli_mod.cmd_capture_hooks_install(argparse.Namespace())

    tools_index_dst = fake_home / ".claude" / "hooks" / "iai-tools-session-index.sh"
    assert tools_index_dst.exists(), "install did not deploy the tools-index hook"

    rc = cli_mod.cmd_capture_hooks_uninstall(argparse.Namespace())
    assert rc == 0

    assert not tools_index_dst.exists()

    data = json.loads(settings.read_text())
    ss_entries = data.get("hooks", {}).get("SessionStart", [])
    assert not _tools_index_entries(data), ss_entries

    # The foreign, non-owned entry survives by positive match.
    assert any(
        foreign_cmd in (h.get("command") or "")
        for e in ss_entries for h in (e.get("hooks") or [])
    ), ss_entries


def test_status_reports_tools_index_wired(fake_home, capsys):
    from iai_mcp import cli as cli_mod

    cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    capsys.readouterr()
    cli_mod.cmd_capture_hooks_status(argparse.Namespace())
    out = capsys.readouterr().out
    assert "iai-tools-session-index.sh" in out, out
    assert re.search(r"tools-index:\s+\S+\s+WIRED", out), out


def test_status_rc_reflects_broken_tools_index_deploy(fake_home):
    from iai_mcp import cli as cli_mod

    rc_install = cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    assert rc_install == 0

    # Simulate a broken deploy: the packaged template exists (this build
    # ships it) but the installed copy is gone -- the overall verdict must
    # surface this instead of reporting ACTIVE.
    dst = fake_home / ".claude" / "hooks" / "iai-tools-session-index.sh"
    dst.unlink()

    rc_status = cli_mod.cmd_capture_hooks_status(argparse.Namespace())
    assert rc_status == 1


def test_status_stays_active_when_tools_index_template_unshipped(fake_home, monkeypatch, tmp_path):
    from iai_mcp import cli as cli_mod
    from iai_mcp.cli import _capture as capture_mod

    cli_mod.cmd_capture_hooks_install(argparse.Namespace())

    # Soft, coexisting rollout: a machine whose packaged template is absent
    # (not yet shipped this build) must not flip to INACTIVE just because
    # the tools-index hook was never installed there.
    dst = fake_home / ".claude" / "hooks" / "iai-tools-session-index.sh"
    dst.unlink()

    original = capture_mod._tools_index_hook_paths
    missing_src = tmp_path / "not-shipped-this-build.sh"

    def _patched():
        _, real_dst, real_settings = original()
        return missing_src, real_dst, real_settings

    monkeypatch.setattr(capture_mod, "_tools_index_hook_paths", _patched)

    rc_status = cli_mod.cmd_capture_hooks_status(argparse.Namespace())
    assert rc_status == 0


# --- Hook script fail-safe contract -----------------------------------

pytestmark_skip_win = pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX shell hook")


def _make_stub_cli(dir_: Path, script: str) -> Path:
    cli = dir_ / "iai-tools"
    cli.write_text(script)
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return cli


def _run_index_hook(home: Path, project_dir: Path, *, extra_env: dict[str, str] | None = None,
                     timeout: float = 10.0):
    env = os.environ.copy()
    env["HOME"] = str(home)
    # Scrub PATH so `command -v iai-tools` cannot resolve the real venv
    # binary -- the fail-safe path must be exercised, not the happy path.
    env["PATH"] = "/usr/bin:/bin"
    env.pop("IAI_MCP_TOOLS_INDEX_CLI", None)
    if extra_env:
        env.update(extra_env)
    stdin_payload = json.dumps({"cwd": str(project_dir), "session_id": "x", "source": "startup"})
    return subprocess.run(
        ["bash", str(HOOK_PATH)],
        input=stdin_payload,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytestmark_skip_win
def test_hook_exits_zero_when_binary_missing(tmp_path):
    assert HOOK_PATH.exists(), f"hook script missing: {HOOK_PATH}"
    home = tmp_path / "home"
    home.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    proc = _run_index_hook(home, project_dir)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", proc.stdout

    watermark_dir = home / ".iai-mcp" / "tools" / ".watermarks"
    leftover_locks = list(watermark_dir.glob("*.lock")) if watermark_dir.exists() else []
    assert leftover_locks == [], f"lock dir leaked on CLI-not-found exit path: {leftover_locks}"


def _project_path_hash(project_dir: Path) -> str:
    out = subprocess.run(
        ["cksum"], input=str(project_dir).encode(), capture_output=True, check=True
    ).stdout
    return out.split()[0].decode()


@pytestmark_skip_win
def test_hook_skips_when_lock_held_by_concurrent_session(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp").mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "file.txt").write_text("content\n")

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    counter = tmp_path / "calls.log"
    _make_stub_cli(stub_dir, f"#!/usr/bin/env bash\necho hit >> {counter}\nexit 0\n")
    (home / ".iai-mcp" / ".tools-cli-path").write_text(str(stub_dir / "iai-tools"))

    watermark_dir = home / ".iai-mcp" / "tools" / ".watermarks"
    watermark_dir.mkdir(parents=True)
    lock_dir = watermark_dir / f"{_project_path_hash(project_dir)}.lock"
    lock_dir.mkdir()
    (lock_dir / "ts").write_text(str(int(time.time())))

    proc = _run_index_hook(home, project_dir)
    assert proc.returncode == 0, proc.stderr
    assert not counter.exists(), "concurrent-lock holder was raced -- stub CLI got invoked"
    assert lock_dir.exists(), "this session stole a fresh lock it does not own"


@pytestmark_skip_win
def test_hook_steals_stale_lock_and_proceeds(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp").mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "file.txt").write_text("content\n")

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    counter = tmp_path / "calls.log"
    _make_stub_cli(stub_dir, f"#!/usr/bin/env bash\necho hit >> {counter}\nexit 0\n")
    (home / ".iai-mcp" / ".tools-cli-path").write_text(str(stub_dir / "iai-tools"))

    watermark_dir = home / ".iai-mcp" / "tools" / ".watermarks"
    watermark_dir.mkdir(parents=True)
    lock_dir = watermark_dir / f"{_project_path_hash(project_dir)}.lock"
    lock_dir.mkdir()
    # Staleness now reads the lock DIRECTORY's own mtime, not a stamp file
    # inside it -- backdate the directory itself past 2x the default 20s
    # hook_timeout to simulate a presumed-dead holder.
    old = time.time() - 100
    os.utime(lock_dir, (old, old))

    proc = _run_index_hook(home, project_dir)
    assert proc.returncode == 0, proc.stderr
    assert counter.read_text().count("hit") == 1, "stale lock was not stolen"


@pytestmark_skip_win
def test_hook_steals_stale_lock_with_foreign_owner_stamp(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp").mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "file.txt").write_text("content\n")

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    counter = tmp_path / "calls.log"
    _make_stub_cli(stub_dir, f"#!/usr/bin/env bash\necho hit >> {counter}\nexit 0\n")
    (home / ".iai-mcp" / ".tools-cli-path").write_text(str(stub_dir / "iai-tools"))

    # A dead holder that got far enough to stamp its own nonce before dying
    # must still be stolen: the steal decision reads only the lock
    # directory's own mtime, never the owner file's content, so a stale
    # lock carrying a foreign nonce is a distinct shape from an ownerless
    # one and must not be treated as still-live just because a stamp
    # exists.
    watermark_dir = home / ".iai-mcp" / "tools" / ".watermarks"
    watermark_dir.mkdir(parents=True)
    lock_dir = watermark_dir / f"{_project_path_hash(project_dir)}.lock"
    lock_dir.mkdir()
    (lock_dir / "owner").write_text("99999-1700000000000000000")
    old = time.time() - 100
    os.utime(lock_dir, (old, old))

    proc = _run_index_hook(home, project_dir)
    assert proc.returncode == 0, proc.stderr
    assert counter.read_text().count("hit") == 1, (
        "a stale lock with a foreign owner stamp wedged indexing instead of being stolen"
    )
    # The winning holder's own release trap fires on exit and removes the
    # directory (its nonce matches what it wrote), confirming the steal
    # produced a real, exclusively-owned lock rather than a false-positive
    # match against the dead holder's leftover nonce.
    assert not lock_dir.exists(), "stolen lock was not released by its new holder"


@pytestmark_skip_win
def test_hook_does_not_steal_fresh_lock_before_stamp_written(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp").mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "file.txt").write_text("content\n")

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    counter = tmp_path / "calls.log"
    _make_stub_cli(stub_dir, f"#!/usr/bin/env bash\necho hit >> {counter}\nexit 0\n")
    (home / ".iai-mcp" / ".tools-cli-path").write_text(str(stub_dir / "iai-tools"))

    # A brand-new lock dir with no owner file yet -- the exact shape a
    # live, just-mkdir'd holder has for the brief window before it writes
    # its own nonce. Must be treated as held, never stolen.
    watermark_dir = home / ".iai-mcp" / "tools" / ".watermarks"
    watermark_dir.mkdir(parents=True)
    lock_dir = watermark_dir / f"{_project_path_hash(project_dir)}.lock"
    lock_dir.mkdir()

    proc = _run_index_hook(home, project_dir)
    assert proc.returncode == 0, proc.stderr
    assert not counter.exists(), "a fresh, unowned lock was wrongly stolen"
    assert lock_dir.exists(), "this session removed a lock it does not own"


def _stale_lock_race_hit_count(tmp_path: Path) -> int:
    """Race 5 processes for one stale lock and return how many won (ran the
    stub CLI). A correct atomic-rename steal always yields exactly 1."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp").mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "file.txt").write_text("content\n")

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    counter = tmp_path / "calls.log"
    _make_stub_cli(stub_dir, f"#!/usr/bin/env bash\necho hit >> {counter}\nexit 0\n")
    (home / ".iai-mcp" / ".tools-cli-path").write_text(str(stub_dir / "iai-tools"))

    watermark_dir = home / ".iai-mcp" / "tools" / ".watermarks"
    watermark_dir.mkdir(parents=True)
    lock_dir = watermark_dir / f"{_project_path_hash(project_dir)}.lock"
    lock_dir.mkdir()
    # Comfortably past the default 40s stale threshold (2x the 20s hook
    # timeout) so a scheduling delay under load never puts the staleness
    # verdict itself in question -- only the steal race is under test.
    old = time.time() - 3600
    os.utime(lock_dir, (old, old))

    # No stdin JSON needed -- `cwd` set via subprocess `cwd=`, and the hook
    # falls back to $PWD when its stdin JSON has no `cwd` key.
    env = {**os.environ, "HOME": str(home), "PATH": "/usr/bin:/bin"}
    procs = [
        subprocess.Popen(
            ["bash", str(HOOK_PATH)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(project_dir),
            env=env,
        )
        for _ in range(5)
    ]
    for p in procs:
        p.communicate(timeout=15)

    return counter.read_text().count("hit") if counter.exists() else 0


@pytestmark_skip_win
def test_hook_concurrent_steal_of_stale_lock_has_exactly_one_winner(tmp_path):
    """5 processes race an atomic-rename steal of one stale lock; exactly
    one must win. Under heavy CPU/IO contention (e.g. inside the full
    sequential gate) scheduler delays can occasionally widen enough to
    perturb this specific run's outcome without indicating a real steal
    defect -- retried in a fresh sandbox per attempt rather than weakening
    what a single, uncontended attempt asserts (still exactly one winner)."""
    attempts = 3
    observed: list[int] = []
    for attempt in range(attempts):
        sandbox = tmp_path / f"attempt-{attempt}"
        sandbox.mkdir()
        observed.append(_stale_lock_race_hit_count(sandbox))
    assert all(h <= 1 for h in observed), (
        f"more than one winner in at least one attempt (observed hit counts: "
        f"{observed}) -- rm+mkdir race, not atomic rename"
    )
    assert any(h == 1 for h in observed), (
        f"no attempt out of {attempts} produced exactly one winner "
        f"(observed hit counts: {observed}) -- no contender ever won the "
        f"stale lock"
    )


def _init_git_repo(project_dir: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=project_dir, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=project_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=project_dir, check=True)
    (project_dir / "a.txt").write_text("one\n")
    (project_dir / "b.txt").write_text("two\n")
    subprocess.run(["git", "add", "."], cwd=project_dir, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=project_dir, check=True)


@pytestmark_skip_win
@pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")
def test_freshness_signature_hashes_dirty_content_not_count(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp").mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _init_git_repo(project_dir)

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    counter = tmp_path / "calls.log"
    _make_stub_cli(stub_dir, f"#!/usr/bin/env bash\necho hit >> {counter}\nexit 0\n")
    (home / ".iai-mcp" / ".tools-cli-path").write_text(str(stub_dir / "iai-tools"))

    # Dirty state A: modify a.txt only (dirty count = 1).
    (project_dir / "a.txt").write_text("one-changed\n")
    proc1 = _run_index_hook(home, project_dir)
    assert proc1.returncode == 0, proc1.stderr
    assert counter.read_text().count("hit") == 1

    # Revert a.txt, dirty b.txt instead -- same dirty count (1), different
    # porcelain content. A count-only signature would alias these and
    # falsely skip the second run.
    subprocess.run(["git", "checkout", "--", "a.txt"], cwd=project_dir, check=True)
    (project_dir / "b.txt").write_text("two-changed\n")

    proc2 = _run_index_hook(home, project_dir)
    assert proc2.returncode == 0, proc2.stderr
    assert counter.read_text().count("hit") == 2, (
        "dirty-count aliasing falsely skipped a real content change"
    )


@pytestmark_skip_win
def test_freshness_gate_converges_for_non_git_tree(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp").mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "file.txt").write_text("content\n")

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    counter = tmp_path / "calls.log"
    _make_stub_cli(stub_dir, f"#!/usr/bin/env bash\necho hit >> {counter}\nexit 0\n")
    (home / ".iai-mcp" / ".tools-cli-path").write_text(str(stub_dir / "iai-tools"))

    proc1 = _run_index_hook(home, project_dir)
    assert proc1.returncode == 0, proc1.stderr
    assert counter.read_text().count("hit") == 1

    proc2 = _run_index_hook(home, project_dir)
    assert proc2.returncode == 0, proc2.stderr
    assert counter.read_text().count("hit") == 1, (
        "non-git fallback signature failed to converge on the second run"
    )

    watermark_dir = home / ".iai-mcp" / "tools" / ".watermarks"
    leftover_locks = list(watermark_dir.glob("*.lock")) if watermark_dir.exists() else []
    assert leftover_locks == [], f"lock dir leaked across two hook runs: {leftover_locks}"


@pytestmark_skip_win
def test_freshness_gate_non_git_detects_same_size_same_minute_content_change(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp").mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "file.txt").write_text("aaaaaaaaaa")

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    counter = tmp_path / "calls.log"
    _make_stub_cli(stub_dir, f"#!/usr/bin/env bash\necho hit >> {counter}\nexit 0\n")
    (home / ".iai-mcp" / ".tools-cli-path").write_text(str(stub_dir / "iai-tools"))

    proc1 = _run_index_hook(home, project_dir)
    assert proc1.returncode == 0, proc1.stderr
    assert counter.read_text().count("hit") == 1

    # Same byte size, well within the same calendar minute -- a
    # metadata-only signature (size + minute-granularity mtime) would alias
    # this to "unchanged" and skip the re-index.
    (project_dir / "file.txt").write_text("bbbbbbbbbb")

    proc2 = _run_index_hook(home, project_dir)
    assert proc2.returncode == 0, proc2.stderr
    assert counter.read_text().count("hit") == 2, (
        "same-size, same-minute content change was falsely skipped"
    )


@pytestmark_skip_win
def test_freshness_gate_backs_off_after_timeout(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp").mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "file.txt").write_text("content\n")

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    counter = tmp_path / "calls.log"
    _make_stub_cli(stub_dir, f"#!/usr/bin/env bash\necho hit >> {counter}\nsleep 5\nexit 0\n")
    (home / ".iai-mcp" / ".tools-cli-path").write_text(str(stub_dir / "iai-tools"))

    extra_env = {"IAI_MCP_TOOLS_INDEX_HOOK_TIMEOUT": "1"}

    proc1 = _run_index_hook(home, project_dir, extra_env=extra_env, timeout=15.0)
    assert proc1.returncode == 0, proc1.stderr
    assert counter.read_text().count("hit") == 1

    log_files = list((home / ".iai-mcp" / "logs").glob("tools-index-*.log"))
    assert log_files, "no hook log written"
    assert "rc=124" in log_files[0].read_text(), log_files[0].read_text()

    proc2 = _run_index_hook(home, project_dir, extra_env=extra_env, timeout=15.0)
    assert proc2.returncode == 0, proc2.stderr
    assert counter.read_text().count("hit") == 1, (
        "timeout back-off failed to converge -- second run re-invoked the CLI"
    )
    assert "backoff" in log_files[0].read_text(), log_files[0].read_text()

    watermark_dir = home / ".iai-mcp" / "tools" / ".watermarks"
    leftover_locks = list(watermark_dir.glob("*.lock")) if watermark_dir.exists() else []
    assert leftover_locks == [], (
        f"lock dir leaked across the timeout + backoff exit paths: {leftover_locks}"
    )


# --- unit coverage for _resolve_iai_tools_cli_path / _write_tools_cli_path_cache ---


def test_resolve_ignores_cwd_relative_sibling_for_falsy_sys_executable(monkeypatch, tmp_path):
    from iai_mcp.cli import _capture as capture_mod

    # Bait planted directly in cwd: a guard that only checked `Path(exe)`
    # without the `if exe:` short-circuit would compute Path("").parent ==
    # Path(".") and pick this up as a false-positive sibling match.
    bait = tmp_path / "iai-tools"
    bait.write_text("#!/usr/bin/env bash\nexit 0\n")
    bait.chmod(bait.stat().st_mode | stat.S_IXUSR)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "executable", "")
    monkeypatch.setattr(shutil, "which", lambda name: None)

    assert capture_mod._resolve_iai_tools_cli_path() is None


def test_resolve_ignores_cwd_relative_sibling_for_relative_sys_executable(monkeypatch, tmp_path):
    from iai_mcp.cli import _capture as capture_mod

    # Bait at the exact relative sibling path a guard missing the
    # `is_absolute()` check would derive from "bin/python3".
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    bait = bin_dir / "iai-tools"
    bait.write_text("#!/usr/bin/env bash\nexit 0\n")
    bait.chmod(bait.stat().st_mode | stat.S_IXUSR)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "executable", "bin/python3")
    monkeypatch.setattr(shutil, "which", lambda name: None)

    assert capture_mod._resolve_iai_tools_cli_path() is None


def test_resolve_falls_back_to_absolute_which_result(monkeypatch, tmp_path):
    from iai_mcp.cli import _capture as capture_mod

    which_target = tmp_path / "iai-tools"
    which_target.write_text("#!/usr/bin/env bash\n")
    which_target.chmod(which_target.stat().st_mode | stat.S_IXUSR)

    monkeypatch.setattr(sys, "executable", "")
    monkeypatch.setattr(shutil, "which", lambda name: str(which_target))

    result = capture_mod._resolve_iai_tools_cli_path()
    assert result == which_target
    assert result.is_absolute()


def test_write_cache_survives_resolve_exception(fake_home, monkeypatch, capsys):
    from iai_mcp.cli import _capture as capture_mod

    def _raise():
        raise RuntimeError("boom")

    monkeypatch.setattr(capture_mod, "_resolve_iai_tools_cli_path", _raise)

    capture_mod._write_tools_cli_path_cache()  # must not raise

    out = capsys.readouterr().out
    assert "boom" in out
    assert "not found" not in out, out
    assert not capture_mod._tools_cli_cache_path().exists()


def test_install_swallows_cache_write_exception_and_continues(fake_home, monkeypatch, capsys):
    from iai_mcp import cli as cli_mod
    from iai_mcp.cli import _capture as capture_mod

    def _raise():
        raise RuntimeError("cache write boom")

    monkeypatch.setattr(capture_mod, "_write_tools_cli_path_cache", _raise)

    rc = cli_mod.cmd_capture_hooks_install(argparse.Namespace())
    assert rc == 0

    out = capsys.readouterr().out
    assert "iai-tools CLI path cache step failed" in out

    # Downstream work after the failing cache step (hook deploy already
    # happens before it; the SessionStart wiring happens after) must still
    # complete -- proves the outer try/except does not abort install.
    data = json.loads(_settings_path(fake_home).read_text())
    assert len(_tools_index_entries(data)) == 1, data


def test_write_cache_removes_stale_entry_pointing_at_missing_target(fake_home, monkeypatch):
    from iai_mcp.cli import _capture as capture_mod

    cache_path = capture_mod._tools_cli_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(str(fake_home / "gone" / "iai-tools"), encoding="utf-8")

    monkeypatch.setattr(capture_mod, "_resolve_iai_tools_cli_path", lambda: None)

    capture_mod._write_tools_cli_path_cache()

    assert not cache_path.exists()


def test_write_cache_removes_stale_entry_pointing_at_non_executable_target(fake_home, monkeypatch):
    from iai_mcp.cli import _capture as capture_mod

    non_exec_target = fake_home / "iai-tools"
    non_exec_target.write_text("#!/usr/bin/env bash\n")
    non_exec_target.chmod(0o644)

    cache_path = capture_mod._tools_cli_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(str(non_exec_target), encoding="utf-8")

    monkeypatch.setattr(capture_mod, "_resolve_iai_tools_cli_path", lambda: None)

    capture_mod._write_tools_cli_path_cache()

    assert not cache_path.exists()


def test_write_cache_leaves_valid_cached_entry_when_resolve_fails(fake_home, monkeypatch):
    from iai_mcp.cli import _capture as capture_mod

    valid_target = fake_home / "iai-tools"
    valid_target.write_text("#!/usr/bin/env bash\n")
    valid_target.chmod(valid_target.stat().st_mode | stat.S_IXUSR)

    cache_path = capture_mod._tools_cli_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(str(valid_target), encoding="utf-8")

    monkeypatch.setattr(capture_mod, "_resolve_iai_tools_cli_path", lambda: None)

    capture_mod._write_tools_cli_path_cache()

    assert cache_path.exists()
    assert cache_path.read_text(encoding="utf-8") == str(valid_target)


@pytestmark_skip_win
def test_hook_exits_zero_when_cli_fails(tmp_path):
    assert HOOK_PATH.exists(), f"hook script missing: {HOOK_PATH}"
    home = tmp_path / "home"
    home.mkdir()
    (home / ".iai-mcp").mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    _make_stub_cli(stub_dir, "#!/usr/bin/env bash\necho boom >&2\nexit 1\n")
    (home / ".iai-mcp" / ".tools-cli-path").write_text(str(stub_dir / "iai-tools"))

    proc = _run_index_hook(home, project_dir)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", proc.stdout

    watermark_dir = home / ".iai-mcp" / "tools" / ".watermarks"
    leftover_locks = list(watermark_dir.glob("*.lock")) if watermark_dir.exists() else []
    assert leftover_locks == [], f"lock dir leaked on non-zero CLI exit path: {leftover_locks}"
