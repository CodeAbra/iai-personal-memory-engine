from __future__ import annotations

import platform
import threading
import time
from pathlib import Path

import pytest

from iai_mcp.wake_handler import WakeHandler

_IS_WINDOWS = platform.system() == "Windows"

@pytest.fixture
def wake_signal_path(tmp_path: Path) -> Path:
    return tmp_path / "wake.signal"

def _write_signal(
    path: Path,
    payload: str = '{"requested_at":"2026-05-02T15:00:00Z"}',
    tmp_suffix: str = ".tmp",
) -> None:
    tmp = path.with_suffix(path.suffix + tmp_suffix)
    tmp.write_text(payload, encoding="utf-8")
    # Model a robust atomic writer: on Windows os.replace (MoveFileEx)
    # transiently fails with PermissionError when the consumer momentarily
    # holds the destination open. Production writers retry the same way (see
    # iai_mcp.daemon_state._atomic_replace); the harness must too, otherwise
    # the writer thread dies early and starves the consumer. POSIX rename is
    # atomic and never sees this.
    attempts = 10 if _IS_WINDOWS else 1
    for i in range(attempts):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.05)

def test_consume_wake_signal_when_present_deletes_and_returns_true(
    wake_signal_path: Path,
) -> None:
    _write_signal(wake_signal_path)
    assert wake_signal_path.is_file()

    handler = WakeHandler(wake_signal_path)
    assert handler.consume_wake_signal() is True
    assert not wake_signal_path.exists()

def test_consume_wake_signal_when_absent_returns_false(
    wake_signal_path: Path,
) -> None:
    assert not wake_signal_path.exists()

    handler = WakeHandler(wake_signal_path)
    assert handler.consume_wake_signal() is False

def test_consume_wake_signal_idempotent(wake_signal_path: Path) -> None:
    _write_signal(wake_signal_path)

    handler = WakeHandler(wake_signal_path)
    assert handler.consume_wake_signal() is True
    assert handler.consume_wake_signal() is False
    assert handler.consume_wake_signal() is False

def test_has_pending_wake_read_only(wake_signal_path: Path) -> None:
    _write_signal(wake_signal_path)

    handler = WakeHandler(wake_signal_path)
    assert handler.has_pending_wake() is True
    assert wake_signal_path.is_file()
    assert handler.has_pending_wake() is True
    assert wake_signal_path.is_file()
    assert handler.consume_wake_signal() is True
    assert handler.has_pending_wake() is False

def test_consume_atomic_no_race(wake_signal_path: Path) -> None:
    """Concurrent writers replacing the signal while a consumer unlinks it must
    never raise, and the consumer must observe at least one signal.

    The consumer polls until it has consumed a few times or a deadline expires,
    while the writers produce continuously until told to stop -- rather than each
    running a fixed iteration count. A fixed count is flaky: the consumer's fast
    unlink loop can finish before any writer's first replace lands (adverse
    thread scheduling on a loaded runner), reading consumed=0. Continuous
    production plus a poll-until-consumed consumer makes the >=1 liveness check
    deterministic without weakening the atomicity intent (no error, no crash
    under concurrent write+consume). Bounded by deadlines so it can never hang.
    """
    handler = WakeHandler(wake_signal_path)
    consumed_truthy_count = 0
    errors: list[BaseException] = []

    stop_writers = threading.Event()

    def writer_loop(writer_id: int) -> None:
        suffix = f".tmp.w{writer_id}"
        try:
            while not stop_writers.is_set():
                _write_signal(wake_signal_path, tmp_suffix=suffix)
        except BaseException as exc:  # pragma: no cover -- defensive
            errors.append(exc)

    def consumer_loop() -> None:
        nonlocal consumed_truthy_count
        deadline = time.monotonic() + 10.0
        try:
            while time.monotonic() < deadline:
                if handler.consume_wake_signal():
                    consumed_truthy_count += 1
                    if consumed_truthy_count >= 5:
                        return
        except BaseException as exc:
            errors.append(exc)

    writers = [
        threading.Thread(target=writer_loop, args=(i,)) for i in range(3)
    ]
    consumer = threading.Thread(target=consumer_loop)
    for w in writers:
        w.start()
    consumer.start()
    consumer.join(timeout=15.0)
    stop_writers.set()
    for w in writers:
        w.join(timeout=10.0)

    assert errors == []
    assert consumed_truthy_count >= 1


def test_consume_retries_transient_permission_error_on_windows(
    wake_signal_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient Windows sharing-violation (PermissionError) from unlink must
    be retried, not swallowed as 'no signal'. Swallowing it drops a real wake
    and strands the daemon in HIBERNATION with an unserved socket. Simulates the
    Windows branch + a flaky unlink so it runs deterministically on any platform.
    """
    import iai_mcp.wake_handler as wh

    monkeypatch.setattr(wh, "_IS_WINDOWS", True)
    monkeypatch.setattr(wh, "_UNLINK_RETRY_SLEEP_SEC", 0.0)

    calls = {"n": 0}

    def flaky_unlink(self, *a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("simulated Windows sharing violation")
        return None

    monkeypatch.setattr(Path, "unlink", flaky_unlink)

    handler = WakeHandler(wake_signal_path)
    # Retries past the two transient failures and consumes on the third attempt.
    assert handler.consume_wake_signal() is True
    assert calls["n"] == 3


def test_consume_does_not_retry_permission_error_on_posix(
    wake_signal_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On POSIX a PermissionError from unlink is a genuine permission fault, not
    a transient sharing violation, so it must NOT be retried -- return False on
    the first failure. Guards against the Windows retry leaking onto POSIX.
    """
    import iai_mcp.wake_handler as wh

    monkeypatch.setattr(wh, "_IS_WINDOWS", False)

    calls = {"n": 0}

    def perm_unlink(self, *a, **k):
        calls["n"] += 1
        raise PermissionError("genuine permission fault")

    monkeypatch.setattr(Path, "unlink", perm_unlink)

    handler = WakeHandler(wake_signal_path)
    assert handler.consume_wake_signal() is False
    assert calls["n"] == 1
