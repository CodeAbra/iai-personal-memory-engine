"""Hermetic tests for the dead-PID `.processing-<pid>.jsonl` salvage migration.

Covers the unlock-policy three branches (dead, live-foreign, live-current),
dry-run accounting, idempotency, collision-safe naming, and the CLI surface.

All fixtures live in `tmp_path / ".deferred-captures"`; the live
`~/.iai-mcp/.deferred-captures/` tree is NEVER touched. Dead PIDs are obtained
by spawning a Python child that immediately exits, then waiting on it. Live-
foreign and live-current PIDs are simulated via the explicit ``live_daemon_pid=``
kwarg override on the migration function — no daemon construction required.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="os.rename overwrite semantics differ on Windows; salvage targets POSIX only",
)


# ---- helpers ---------------------------------------------------------------


def _spawn_dead_pid() -> int:
    """Spawn a Python child that exits immediately; return its (now-dead) pid."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _make_locked_file(deferred: Path, basename: str, owner_pid: int) -> Path:
    """Create a fake `<basename>.processing-<owner_pid>.jsonl` file with two lines."""
    deferred.mkdir(parents=True, exist_ok=True)
    path = deferred / f"{basename}.processing-{owner_pid}.jsonl"
    path.write_text(
        '{"event": "fake_capture", "n": 1}\n'
        '{"event": "fake_capture", "n": 2}\n'
    )
    return path


# ---- tests -----------------------------------------------------------------


def test_dead_owner_pid_is_unlocked(tmp_path):
    """Dead owner pid → file renamed to bare `<basename>.jsonl`."""
    from iai_mcp.migrate import migrate_unlock_dead_pid_processing_files

    deferred = tmp_path / ".deferred-captures"
    dead_pid = _spawn_dead_pid()
    locked = _make_locked_file(deferred, "sess-dead", dead_pid)

    result = migrate_unlock_dead_pid_processing_files(
        deferred_dir=deferred,
        live_daemon_pid=os.getpid(),
    )

    target = deferred / "sess-dead.jsonl"
    assert not locked.exists(), "locked file should be gone after rename"
    assert target.exists(), "bare `.jsonl` target should exist"
    assert result["files_scanned"] == 1
    assert result["files_unlocked"] == 1
    assert result["skipped_live_current_daemon"] == 0
    assert result["collision_safe_renames"] == 0
    assert result["dry_run"] is False


def test_live_foreign_pid_is_unlocked(tmp_path):
    """Alive owner pid that is NOT the live daemon → unlocked anyway."""
    from iai_mcp.migrate import migrate_unlock_dead_pid_processing_files

    deferred = tmp_path / ".deferred-captures"
    # We are alive — simulate the daemon being a DIFFERENT alive pid.
    foreign_pid = os.getpid()
    simulated_daemon_pid = foreign_pid + 1  # explicitly NOT us

    locked = _make_locked_file(deferred, "sess-foreign", foreign_pid)

    result = migrate_unlock_dead_pid_processing_files(
        deferred_dir=deferred,
        live_daemon_pid=simulated_daemon_pid,
    )

    target = deferred / "sess-foreign.jsonl"
    assert not locked.exists()
    assert target.exists()
    assert result["files_scanned"] == 1
    assert result["files_unlocked"] == 1
    assert result["skipped_live_current_daemon"] == 0


def test_live_current_daemon_pid_is_preserved(tmp_path):
    """Owner pid == live-daemon pid → file is NEVER touched (protective branch)."""
    from iai_mcp.migrate import migrate_unlock_dead_pid_processing_files

    deferred = tmp_path / ".deferred-captures"
    current_pid = os.getpid()
    locked = _make_locked_file(deferred, "sess-current", current_pid)

    result = migrate_unlock_dead_pid_processing_files(
        deferred_dir=deferred,
        live_daemon_pid=current_pid,
    )

    target = deferred / "sess-current.jsonl"
    assert locked.exists(), "current-daemon-owned file MUST remain locked"
    assert not target.exists(), "no bare target should be created"
    assert result["files_scanned"] == 1
    assert result["files_unlocked"] == 0
    assert result["skipped_live_current_daemon"] == 1


def test_plain_jsonl_files_untouched(tmp_path):
    """Files matching no `_PROCESSING_MARKER_RE` are not even scanned."""
    from iai_mcp.migrate import migrate_unlock_dead_pid_processing_files

    deferred = tmp_path / ".deferred-captures"
    deferred.mkdir(parents=True, exist_ok=True)
    plain = deferred / "sess-X.jsonl"
    live = deferred / "sess-Y.live.jsonl"
    crash = deferred / "sess-Z.crash-1.jsonl"
    for p in (plain, live, crash):
        p.write_text('{"x": 1}\n')

    contents_before = {p.name: p.read_text() for p in (plain, live, crash)}

    result = migrate_unlock_dead_pid_processing_files(
        deferred_dir=deferred,
        live_daemon_pid=os.getpid(),
    )

    assert result["files_scanned"] == 0
    assert result["files_unlocked"] == 0
    for p in (plain, live, crash):
        assert p.exists(), f"{p.name} must remain"
        assert p.read_text() == contents_before[p.name]


def test_dry_run_makes_no_changes(tmp_path):
    """`dry_run=True` reports counts but mutates nothing."""
    from iai_mcp.migrate import migrate_unlock_dead_pid_processing_files

    deferred = tmp_path / ".deferred-captures"
    dead_pid = _spawn_dead_pid()
    locked = _make_locked_file(deferred, "sess-dry", dead_pid)
    target = deferred / "sess-dry.jsonl"

    result = migrate_unlock_dead_pid_processing_files(
        deferred_dir=deferred,
        live_daemon_pid=os.getpid(),
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["files_scanned"] == 1
    assert result["files_unlocked"] == 1, "dry-run reports what would happen"
    assert locked.exists(), "FS must be untouched in dry-run"
    assert not target.exists(), "no bare target should be created in dry-run"


def test_idempotent(tmp_path):
    """After a successful rename, the marker is gone — second call finds nothing."""
    from iai_mcp.migrate import migrate_unlock_dead_pid_processing_files

    deferred = tmp_path / ".deferred-captures"
    dead_pid = _spawn_dead_pid()
    _make_locked_file(deferred, "sess-idem", dead_pid)

    first = migrate_unlock_dead_pid_processing_files(
        deferred_dir=deferred,
        live_daemon_pid=os.getpid(),
    )
    second = migrate_unlock_dead_pid_processing_files(
        deferred_dir=deferred,
        live_daemon_pid=os.getpid(),
    )

    assert first["files_unlocked"] == 1
    assert second["files_scanned"] == 0
    assert second["files_unlocked"] == 0
    assert second["collision_safe_renames"] == 0


def test_collision_safe_naming(tmp_path):
    """When bare `<basename>.jsonl` already exists, recover via
    `<basename>.recovered-<utc_ts>-<unlock_pid>.jsonl` and PRESERVE the
    pre-existing target verbatim."""
    from iai_mcp.migrate import migrate_unlock_dead_pid_processing_files

    deferred = tmp_path / ".deferred-captures"
    deferred.mkdir(parents=True, exist_ok=True)

    pre_existing = deferred / "sess-clash.jsonl"
    pre_existing_content = '{"this": "must-not-be-overwritten"}\n'
    pre_existing.write_text(pre_existing_content)

    dead_pid = _spawn_dead_pid()
    locked = _make_locked_file(deferred, "sess-clash", dead_pid)

    result = migrate_unlock_dead_pid_processing_files(
        deferred_dir=deferred,
        live_daemon_pid=os.getpid(),
    )

    assert result["files_unlocked"] == 1
    assert result["collision_safe_renames"] == 1
    assert not locked.exists()
    assert pre_existing.exists(), "pre-existing target must be untouched"
    assert pre_existing.read_text() == pre_existing_content

    siblings = [
        p for p in deferred.iterdir()
        if p.name.startswith("sess-clash.recovered-") and p.name.endswith(".jsonl")
    ]
    assert len(siblings) == 1, (
        f"expected exactly one `.recovered-` sibling, found {[p.name for p in siblings]}"
    )
    sibling_name = siblings[0].name
    # Suffix shape: sess-clash.recovered-<utc_ts>-<unlock_pid>.jsonl
    assert sibling_name.startswith("sess-clash.recovered-")
    assert sibling_name.endswith(".jsonl")


def test_cli_help_lists_subcommand_and_flags():
    """`iai-mcp deferred-unlock-dead-pids --help` exits 0 and lists --dry-run + --json.
    Pins the CLI surface against accidental rename/regression."""
    result = subprocess.run(
        [sys.executable, "-m", "iai_mcp.cli", "deferred-unlock-dead-pids", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"--help exited non-zero. stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    combined = result.stdout + result.stderr
    assert "--dry-run" in combined
    assert "--json" in combined


# ---- CR-01: atomic no-clobber TOCTOU test ----------------------------------


def test_atomic_no_clobber_when_target_exists_at_link_time(tmp_path):
    """``_atomic_claim_target`` never clobbers a pre-existing file even if the
    caller's ``bare_target.exists()`` check returned False (the TOCTOU window).

    We simulate this by calling ``_atomic_claim_target`` with a bare_target that
    is already on disk — as if the racer wrote it after our pre-check but before
    our ``os.link``.  ``os.link`` raises ``FileExistsError``; the function falls
    back to a ``.recovered-`` name.  The pre-existing file content MUST survive.
    """
    from iai_mcp.migrate._dead_pid_unlock import _atomic_claim_target

    deferred = tmp_path / ".deferred-captures"
    deferred.mkdir(parents=True)

    # Pre-existing file that simulates the "racer wrote it" scenario.
    bare_target = deferred / "sess-race.jsonl"
    pre_existing_content = '{"must": "survive"}\n'
    bare_target.write_text(pre_existing_content)

    # Source to be moved.
    src = deferred / "sess-race.processing-99999.jsonl"
    src_content = '{"salvaged": true}\n'
    src.write_text(src_content)

    # Call _atomic_claim_target; it must NOT clobber bare_target even though it
    # tries to link onto it first and gets FileExistsError from the OS.
    result_path, collision_safe = _atomic_claim_target(
        src,
        bare_target,
        deferred_dir=deferred,
        stripped="sess-race.jsonl",
        unlock_pid=os.getpid(),
    )

    # Pre-existing content MUST be intact.
    assert bare_target.exists(), "pre-existing bare target must still exist"
    assert bare_target.read_text() == pre_existing_content, (
        "pre-existing content was overwritten — atomic no-clobber invariant violated"
    )

    # Salvaged file must have landed on a .recovered- name.
    assert result_path is not None, "salvage should succeed via .recovered- fallback"
    assert collision_safe is True, "collision_safe must be True when fallback was used"
    assert result_path != bare_target, "must NOT have landed on the pre-existing target"
    assert result_path.exists(), "recovered file must exist on disk"
    assert result_path.read_text() == src_content, "salvaged events must be preserved"

    # Source link must be gone after the atomic move.
    assert not src.exists(), "source .processing- file must be gone after atomic move"


def test_atomic_no_clobber_concurrent_double_run(tmp_path):
    """Simulate two concurrent unlock invocations racing for the same bare target.

    Both call ``migrate_unlock_dead_pid_processing_files`` with two different
    locked files mapping to the same stripped name, both with no pre-existing
    bare target.  One lands on the bare name; the other gets a ``.recovered-``
    name.  Neither is lost.
    """
    from iai_mcp.migrate import migrate_unlock_dead_pid_processing_files

    deferred = tmp_path / ".deferred-captures"
    deferred.mkdir(parents=True)

    dead_pid_1 = _spawn_dead_pid()
    dead_pid_2 = _spawn_dead_pid()

    # Two locked files that both strip to the same bare name.
    locked_1 = _make_locked_file(deferred, "sess-twin", dead_pid_1)
    locked_2 = _make_locked_file(deferred, "sess-twin", dead_pid_2)
    assert locked_1 != locked_2, "two distinct .processing- files must exist"

    result = migrate_unlock_dead_pid_processing_files(
        deferred_dir=deferred,
        live_daemon_pid=os.getpid(),
    )

    assert result["files_scanned"] == 2
    assert result["files_unlocked"] == 2, (
        "both files must be salvaged — one to bare name, one to .recovered-"
    )
    assert result["collision_safe_renames"] == 1, (
        "exactly one rename must have used the .recovered- fallback"
    )

    # Both source files must be gone.
    assert not locked_1.exists()
    assert not locked_2.exists()

    # Bare target exists.
    bare = deferred / "sess-twin.jsonl"
    assert bare.exists(), "bare target must exist (first winner)"

    # One .recovered- sibling must exist.
    siblings = [
        p for p in deferred.iterdir()
        if p.name.startswith("sess-twin.recovered-") and p.name.endswith(".jsonl")
    ]
    assert len(siblings) == 1, (
        f"expected exactly 1 .recovered- sibling, found {[p.name for p in siblings]}"
    )


# ---- WR-01: live-daemon guard test -----------------------------------------


def test_cli_refuses_when_live_daemon_running(tmp_path, monkeypatch, capsys):
    """Without --force the command exits 2 when a live daemon is detected.

    ``_read_live_daemon_pid`` in the migrate module is monkeypatched to return
    our own PID (= alive process).  The guard must fire, print a refusal to
    stderr, and return 2 without renaming anything.
    """
    import argparse

    import iai_mcp.migrate._dead_pid_unlock as _unlock_mod
    from iai_mcp.cli._analytics import cmd_deferred_unlock_dead_pids

    deferred = tmp_path / ".deferred-captures"
    deferred.mkdir(parents=True)

    # Plant a locked file to confirm it is NOT renamed when the guard fires.
    dead_pid = _spawn_dead_pid()
    locked = _make_locked_file(deferred, "sess-guard", dead_pid)

    live_pid = os.getpid()
    # Patch the module attribute — the from-import inside cmd_deferred_unlock_dead_pids
    # re-resolves this from the module namespace at call time.
    monkeypatch.setattr(_unlock_mod, "_read_live_daemon_pid", lambda: live_pid)

    args = argparse.Namespace(dry_run=False, force=False, json=False)
    rc = cmd_deferred_unlock_dead_pids(args)

    captured = capsys.readouterr()
    assert rc == 2, (
        f"expected exit 2 (live daemon guard), got {rc}. "
        f"stdout={captured.out!r} stderr={captured.err!r}"
    )
    assert "live daemon" in captured.err.lower(), (
        "refusal message must mention 'live daemon'"
    )

    # The locked file must be completely untouched.
    assert locked.exists(), "live-daemon guard: locked file must NOT be renamed"
    bare = deferred / "sess-guard.jsonl"
    assert not bare.exists(), "no bare target should be created when guard fires"


def test_cli_force_bypasses_live_daemon_guard(tmp_path, monkeypatch):
    """--force lets the command proceed even when a live daemon is detected.

    We monkeypatch both ``_read_live_daemon_pid`` (returns a live pid) and
    ``migrate_unlock_dead_pid_processing_files`` (uses our tmp deferred dir) so
    the test is hermetic and doesn't touch the real deferred-captures dir.
    """
    import argparse

    import iai_mcp.migrate as _migrate_pkg
    import iai_mcp.migrate._dead_pid_unlock as _unlock_mod
    from iai_mcp.cli._analytics import cmd_deferred_unlock_dead_pids
    from iai_mcp.migrate import migrate_unlock_dead_pid_processing_files

    deferred = tmp_path / ".deferred-captures"
    deferred.mkdir(parents=True)

    dead_pid = _spawn_dead_pid()
    _make_locked_file(deferred, "sess-force", dead_pid)

    live_pid = os.getpid()
    monkeypatch.setattr(_unlock_mod, "_read_live_daemon_pid", lambda: live_pid)

    # Wrap migrate to inject our tmp deferred_dir so nothing touches ~/.iai-mcp/.
    _real_migrate = migrate_unlock_dead_pid_processing_files

    def _migrate_tmp(**kwargs):
        kwargs.setdefault("deferred_dir", deferred)
        return _real_migrate(**kwargs)

    monkeypatch.setattr(_migrate_pkg, "migrate_unlock_dead_pid_processing_files",
                        _migrate_tmp)

    args = argparse.Namespace(dry_run=False, force=True, json=False)
    rc = cmd_deferred_unlock_dead_pids(args)

    # --force must bypass the guard and complete with exit 0.
    assert rc == 0, f"--force should bypass the guard; got {rc}"


# ---- WR-02: fail-closed on unknown live daemon PID -------------------------


def test_fail_closed_alive_owner_not_unlocked_when_daemon_pid_unknown(tmp_path):
    """When ``live_daemon_pid=None`` (daemon state absent/corrupt), alive owner PIDs
    must NOT be unlocked — the function must stay fail-closed.

    Dead owner PIDs are still unlocked; only ambiguous-alive ones are skipped.
    """
    from iai_mcp.migrate import migrate_unlock_dead_pid_processing_files

    deferred = tmp_path / ".deferred-captures"

    # An alive foreign owner (us) — should be SKIPPED under None live_daemon_pid.
    alive_foreign_pid = os.getpid()
    locked_alive = _make_locked_file(deferred, "sess-alive-foreign", alive_foreign_pid)

    # A provably dead owner — should still be unlocked.
    dead_pid = _spawn_dead_pid()
    locked_dead = _make_locked_file(deferred, "sess-dead-unknown", dead_pid)

    # Explicit None = "live daemon PID indeterminate" (not the default sentinel).
    result = migrate_unlock_dead_pid_processing_files(
        deferred_dir=deferred,
        live_daemon_pid=None,  # explicitly unknown
    )

    # Alive owner: NOT unlocked (fail-closed).
    assert locked_alive.exists(), (
        "alive owner must NOT be unlocked when live_daemon_pid is None (fail-closed)"
    )
    bare_alive = deferred / "sess-alive-foreign.jsonl"
    assert not bare_alive.exists(), "no bare target for the alive-owner file"
    assert result["skipped_live_unknown"] == 1

    # Dead owner: unlocked normally.
    bare_dead = deferred / "sess-dead-unknown.jsonl"
    assert not locked_dead.exists(), "dead owner file should be renamed"
    assert bare_dead.exists(), "bare target for dead owner should exist"
    assert result["files_unlocked"] == 1
