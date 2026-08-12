"""Sweep of capture-state files no writer will return to.

Session state mutates on every hook fire, so month-old files belong to
sessions that are gone; day-old tmp names are debris of crashed writer
pids. Unknown names must never be touched.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from iai_mcp.capture import (
    CAPTURE_STATE_STALE_SEC,
    CAPTURE_STATE_TMP_STALE_SEC,
    capture_state_dir,
    sweep_capture_state,
)


def _mk(state_dir: Path, name: str, age_sec: float, now: float) -> Path:
    p = state_dir / name
    p.write_text("x", encoding="utf-8")
    os.utime(p, (now - age_sec, now - age_sec))
    return p


def _setup(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    state_dir = capture_state_dir()
    state_dir.mkdir(parents=True)
    return state_dir


def test_sweep_classifies_and_removes(tmp_path, monkeypatch) -> None:
    state_dir = _setup(tmp_path, monkeypatch)
    now = time.time()

    fresh = _mk(state_dir, "live.offset", 3600, now)
    stale = _mk(state_dir, "gone.turnstate.json", CAPTURE_STATE_STALE_SEC + 60, now)
    old_lock = _mk(state_dir, "gone.capture.lock", CAPTURE_STATE_STALE_SEC + 60, now)
    tmp_old = _mk(
        state_dir, "gone.offset.tmp12345", CAPTURE_STATE_TMP_STALE_SEC + 60, now
    )
    tmp_fresh = _mk(state_dir, "live.offset.tmp999", 60, now)
    unknown = _mk(state_dir, "unrelated.bin", CAPTURE_STATE_STALE_SEC * 2, now)

    dry = sweep_capture_state(apply=False, now=now)
    assert dry == {"stale": 1, "tmp": 1, "removed": 0, "kept": 4}
    assert stale.exists() and tmp_old.exists(), "dry run must not delete"

    applied = sweep_capture_state(apply=True, now=now)
    assert applied["removed"] == 2
    assert not stale.exists() and not tmp_old.exists()
    assert old_lock.exists(), (
        "lock files are never swept: flock acquisition does not touch "
        "mtime, so a held lock can look arbitrarily old, and unlinking it "
        "splits the mutual exclusion across two inodes"
    )
    assert fresh.exists() and tmp_fresh.exists() and unknown.exists()


def test_sweep_missing_dir_is_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert sweep_capture_state(apply=True) == {
        "stale": 0, "tmp": 0, "removed": 0, "kept": 0,
    }


def test_capture_hooks_status_flags_stale_installed_hook(
    tmp_path, monkeypatch, capsys
) -> None:
    """PRESENT alone hides a stale install; the status must compare the
    installed hook body against the packaged template."""
    import argparse

    from iai_mcp.cli import cmd_capture_hooks_status
    from iai_mcp.cli._capture import _turn_hook_paths

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    hooks_dir = tmp_path / ".claude" / "hooks"
    hooks_dir.mkdir(parents=True)

    turn_src, turn_dst = _turn_hook_paths()
    turn_dst.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    cmd_capture_hooks_status(argparse.Namespace(target="claude"))
    out = capsys.readouterr().out
    assert "STALE" in out and "rerun capture-hooks install" in out

    turn_dst.write_bytes(Path(str(turn_src)).read_bytes())
    cmd_capture_hooks_status(argparse.Namespace(target="claude"))
    out = capsys.readouterr().out
    assert "Turn installed" in out
    assert "matches template" in out


def test_doctor_row_fails_on_stale_and_heals(tmp_path, monkeypatch) -> None:
    from iai_mcp.doctor import _sweep_capture_state_heal
    from iai_mcp.doctor._storage_checks import check_aa_capture_state_hygiene

    state_dir = _setup(tmp_path, monkeypatch)
    now = time.time()
    _mk(state_dir, "gone.watermark", CAPTURE_STATE_STALE_SEC + 60, now)

    result = check_aa_capture_state_hygiene()
    assert result.passed is False
    assert "month-old" in result.detail

    ok, msg, rc = _sweep_capture_state_heal()
    assert ok and rc == 0 and "1" in msg

    result = check_aa_capture_state_hygiene()
    assert result.passed is True
