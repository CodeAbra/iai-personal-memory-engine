from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from iai_mcp.lifecycle_lock import (
    LifecycleLock,
    LifecycleLockConflict,
    SCHEMA_VERSION,
)


def test_acquire_in_clean_state(tmp_path: Path) -> None:
    lock_path = tmp_path / ".locked"
    lock = LifecycleLock(lock_path)

    lock.acquire()

    assert lock_path.exists()
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["pid"] == os.getpid()
    assert isinstance(payload["hostname"], str) and payload["hostname"]
    assert isinstance(payload["started_at"], str) and payload["started_at"]
    assert payload["schema_version"] == SCHEMA_VERSION


def test_acquire_when_existing_lock_dead_pid_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / ".locked"
    lock_path.write_text(
        json.dumps(
            {
                "pid": 999_999,
                "hostname": "Some-Other-Mac.local",
                "started_at": "2026-04-30T15:00:00+00:00",
                "schema_version": SCHEMA_VERSION,
            }
        )
    )
    import iai_mcp.lifecycle_lock as ll
    monkeypatch.setattr(ll, "_current_hostname", lambda: "Some-Other-Mac.local")
    monkeypatch.setattr(ll, "_is_pid_alive", lambda pid: False)

    lock = LifecycleLock(lock_path)
    lock.acquire()

    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["pid"] == os.getpid()
    assert payload["hostname"] == "Some-Other-Mac.local"


def test_acquire_when_existing_lock_live_pid_same_host_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / ".locked"
    lock_path.write_text(
        json.dumps(
            {
                "pid": 12_345,
                "hostname": "test-host.local",
                "started_at": "2026-04-30T10:00:00+00:00",
                "schema_version": SCHEMA_VERSION,
            }
        )
    )
    import iai_mcp.lifecycle_lock as ll
    monkeypatch.setattr(ll, "_current_hostname", lambda: "test-host.local")
    monkeypatch.setattr(ll, "_is_pid_alive", lambda pid: True)

    lock = LifecycleLock(lock_path)
    with pytest.raises(LifecycleLockConflict) as exc_info:
        lock.acquire()

    assert exc_info.value.existing is not None
    assert exc_info.value.existing["pid"] == 12_345
    assert exc_info.value.existing["hostname"] == "test-host.local"
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["pid"] == 12_345


def test_acquire_when_existing_lock_different_hostname_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / ".locked"
    lock_path.write_text(
        json.dumps(
            {
                "pid": 12_345,
                "hostname": "Other-Mac.local",
                "started_at": "2026-04-30T10:00:00+00:00",
                "schema_version": SCHEMA_VERSION,
            }
        )
    )
    import iai_mcp.lifecycle_lock as ll
    monkeypatch.setattr(ll, "_current_hostname", lambda: "This-Mac.local")
    monkeypatch.setattr(ll, "_is_pid_alive", lambda pid: True)

    lock = LifecycleLock(lock_path)
    lock.acquire()

    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["pid"] == os.getpid()
    assert payload["hostname"] == "This-Mac.local"


def test_release_deletes_file(tmp_path: Path) -> None:
    lock_path = tmp_path / ".locked"
    lock = LifecycleLock(lock_path)
    lock.acquire()
    assert lock_path.exists()

    lock.release()
    assert not lock_path.exists()

    lock.release()
    assert not lock_path.exists()


def test_is_held_by_self_true_after_acquire(tmp_path: Path) -> None:
    lock_path = tmp_path / ".locked"
    lock = LifecycleLock(lock_path)
    assert lock.is_held_by_self() is False

    lock.acquire()
    assert lock.is_held_by_self() is True


def test_is_held_by_self_false_when_pid_differs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / ".locked"
    lock_path.write_text(
        json.dumps(
            {
                "pid": os.getpid() + 1,
                "hostname": "test-host.local",
                "started_at": "2026-04-30T10:00:00+00:00",
                "schema_version": SCHEMA_VERSION,
            }
        )
    )
    import iai_mcp.lifecycle_lock as ll
    monkeypatch.setattr(ll, "_current_hostname", lambda: "test-host.local")

    lock = LifecycleLock(lock_path)
    assert lock.is_held_by_self() is False


def test_force_unlock_returns_previous_content(tmp_path: Path) -> None:
    lock_path = tmp_path / ".locked"
    lock_path.write_text(
        json.dumps(
            {
                "pid": 4242,
                "hostname": "stale-host.local",
                "started_at": "2026-04-29T08:00:00+00:00",
                "schema_version": SCHEMA_VERSION,
            }
        )
    )

    lock = LifecycleLock(lock_path)
    previous = lock.force_unlock()

    assert previous is not None
    assert previous["pid"] == 4242
    assert previous["hostname"] == "stale-host.local"
    assert not lock_path.exists()


def test_force_unlock_when_no_lockfile(tmp_path: Path) -> None:
    lock_path = tmp_path / ".locked"
    lock = LifecycleLock(lock_path)
    assert lock.force_unlock() is None
    assert not lock_path.exists()


def test_acquire_overwrites_corrupt_lockfile(tmp_path: Path) -> None:
    lock_path = tmp_path / ".locked"
    lock_path.write_text("not-valid-json{{{")

    lock = LifecycleLock(lock_path)
    lock.acquire()

    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["pid"] == os.getpid()


def test_read_returns_none_for_missing_file(tmp_path: Path) -> None:
    lock_path = tmp_path / ".locked"
    lock = LifecycleLock(lock_path)
    assert lock.read() is None


def test_read_returns_none_for_corrupt_json(tmp_path: Path) -> None:
    lock_path = tmp_path / ".locked"
    lock_path.write_text("garbage---")
    lock = LifecycleLock(lock_path)
    assert lock.read() is None


def test_read_returns_none_for_invalid_schema(tmp_path: Path) -> None:
    lock_path = tmp_path / ".locked"
    lock_path.write_text(
        json.dumps({"pid": 1, "hostname": "h", "schema_version": 1})
    )
    lock = LifecycleLock(lock_path)
    assert lock.read() is None


def test_acquire_writes_mode_0600(tmp_path: Path) -> None:
    lock_path = tmp_path / ".locked"
    lock = LifecycleLock(lock_path)
    lock.acquire()

    mode = lock_path.stat().st_mode & 0o777
    assert mode == 0o600, f"expected mode 0o600, got 0o{mode:o}"
