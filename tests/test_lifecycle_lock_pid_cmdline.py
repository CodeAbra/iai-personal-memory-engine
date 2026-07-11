from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import psutil as _real_psutil
import pytest

from iai_mcp.lifecycle_lock import (
    LifecycleLock,
    LifecycleLockConflict,
    SCHEMA_VERSION,
)


def _write_lockfile(lock_path: Path, *, pid: int, hostname: str) -> None:
    lock_path.write_text(
        json.dumps(
            {
                "pid": pid,
                "hostname": hostname,
                "started_at": "2026-05-17T10:00:00+00:00",
                "schema_version": SCHEMA_VERSION,
            }
        ),
        encoding="utf-8",
    )


def _install_fake_psutil(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cmdline_value: list[str] | None = None,
    cmdline_exc: Exception | None = None,
) -> MagicMock:
    fake_proc = MagicMock()
    if cmdline_exc is not None:
        fake_proc.cmdline.side_effect = cmdline_exc
    else:
        fake_proc.cmdline.return_value = cmdline_value or []
    fake_psutil = MagicMock()
    fake_psutil.Process.return_value = fake_proc
    fake_psutil.NoSuchProcess = _real_psutil.NoSuchProcess
    fake_psutil.AccessDenied = _real_psutil.AccessDenied
    fake_psutil.ZombieProcess = _real_psutil.ZombieProcess
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    return fake_psutil


def test_acquire_treats_recycled_pid_with_non_daemon_cmdline_as_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / ".locked"
    real_pid = os.getpid()
    _write_lockfile(lock_path, pid=real_pid, hostname="test-host.local")

    import iai_mcp.lifecycle_lock as ll
    monkeypatch.setattr(ll, "_current_hostname", lambda: "test-host.local")
    _install_fake_psutil(monkeypatch, cmdline_value=["bash"])

    lock = LifecycleLock(lock_path)
    lock.acquire()

    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["pid"] == real_pid, (
        f"cue=acquire-overwrote-stale lock pid={payload['pid']} expected={real_pid}"
    )


def test_acquire_treats_daemon_cmdline_as_live(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / ".locked"
    real_pid = os.getpid()
    _write_lockfile(lock_path, pid=real_pid, hostname="test-host.local")

    import iai_mcp.lifecycle_lock as ll
    monkeypatch.setattr(ll, "_current_hostname", lambda: "test-host.local")
    fake_psutil = _install_fake_psutil(
        monkeypatch, cmdline_value=["python3", "-m", "iai_mcp.daemon"]
    )

    lock = LifecycleLock(lock_path)
    with pytest.raises(LifecycleLockConflict) as exc_info:
        lock.acquire()
    assert exc_info.value.existing is not None, "cue=conflict-carries-existing-payload"
    assert exc_info.value.existing["pid"] == real_pid

    assert fake_psutil.Process.called, (
        "cue=psutil-not-consulted; helper still on os.kill-only path"
    )


def test_acquire_treats_psutil_nosuchprocess_as_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / ".locked"
    real_pid = os.getpid()
    _write_lockfile(lock_path, pid=real_pid, hostname="test-host.local")

    import iai_mcp.lifecycle_lock as ll
    monkeypatch.setattr(ll, "_current_hostname", lambda: "test-host.local")
    _install_fake_psutil(
        monkeypatch,
        cmdline_exc=_real_psutil.NoSuchProcess(real_pid),
    )

    lock = LifecycleLock(lock_path)
    lock.acquire()

    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["pid"] == real_pid, (
        f"cue=acquire-overwrote-on-NoSuchProcess pid={payload['pid']} expected={real_pid}"
    )


def test_acquire_falls_back_to_os_kill_when_psutil_unimportable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    lock_path = tmp_path / ".locked"
    real_pid = os.getpid()
    _write_lockfile(lock_path, pid=real_pid, hostname="test-host.local")

    import iai_mcp.lifecycle_lock as ll
    monkeypatch.setattr(ll, "_current_hostname", lambda: "test-host.local")

    monkeypatch.setitem(sys.modules, "psutil", None)

    caplog.set_level(logging.DEBUG, logger="iai_mcp.lifecycle_lock")

    lock = LifecycleLock(lock_path)
    with pytest.raises(LifecycleLockConflict):
        lock.acquire()

    fallback_logged = any(
        "psutil unavailable" in rec.getMessage()
        for rec in caplog.records
        if rec.name == "iai_mcp.lifecycle_lock"
    )
    assert fallback_logged, (
        "cue=psutil-import-fallback-branch-not-executed; "
        f"records={[r.getMessage() for r in caplog.records]}"
    )
