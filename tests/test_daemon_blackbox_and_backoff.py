from __future__ import annotations

import asyncio
import os
import time

import pytest

from iai_mcp import daemon
from iai_mcp.hippo import HippoLockHeldError


DEBOUNCE_N = 3
GRACE = 600.0
NORMAL = 1
WARN = 2
RSS_LOW = 300 * 1024 * 1024


@pytest.fixture
def watchdog_env(tmp_path, monkeypatch):
    log_path = tmp_path / ".daemon-watchdog.log"
    bb_log_path = tmp_path / ".daemon-blackbox.log"
    sock_path = str(tmp_path / ".daemon.sock")

    fd = os.open(str(log_path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    monkeypatch.setattr(daemon, "_WATCHDOG_LOG_FD", fd)

    if hasattr(daemon, "_WATCHDOG_BLACKBOX_EPISODE_FIRED"):
        monkeypatch.setattr(daemon, "_WATCHDOG_BLACKBOX_EPISODE_FIRED", False)

    monkeypatch.setattr(
        daemon, "_daemon_started_monotonic", time.monotonic() - (GRACE + 60.0)
    )

    kill_calls: list[tuple[int, int]] = []

    def _fake_kill(pid, sig):
        kill_calls.append((pid, sig))

    monkeypatch.setattr(daemon.os, "kill", _fake_kill)

    class _Ns:
        pass

    ns = _Ns()
    ns.log_path = log_path
    ns.bb_log_path = bb_log_path
    ns.sock_path = sock_path
    ns.kill_calls = kill_calls
    ns.fd = fd
    yield ns
    try:
        os.close(fd)
    except OSError:
        pass


def _probe(result: bool):
    async def _p(_sock, _timeout):
        return result
    return _p


def _tick(
    env,
    consec: int,
    probe_result: bool,
    *,
    blackbox_fn=None,
) -> tuple[float, int]:
    return daemon._watchdog_tick(
        object(),
        env.sock_path,
        env.log_path,
        consec,
        probe_fn=_probe(probe_result),
        pressure_fn=lambda: NORMAL,
        rss_fn=lambda: RSS_LOW,
        blackbox_fn=blackbox_fn,
    )


class TestBlackboxOnFailedProbe:

    def test_blackbox_called_once_on_first_failing_tick(self, watchdog_env):
        calls: list[int] = []

        def _bb(log_fd, probe_ok, consec, debounce_n):
            calls.append(consec)

        _interval, consec = _tick(watchdog_env, 0, False, blackbox_fn=_bb)
        assert consec == 1
        assert len(calls) == 1, "black box must fire on the first failing tick"

    def test_blackbox_not_called_on_subsequent_failing_ticks_same_episode(
        self, watchdog_env
    ):
        calls: list[int] = []

        def _bb(log_fd, probe_ok, consec, debounce_n):
            calls.append(consec)

        consec = 0
        for _ in range(DEBOUNCE_N - 1):
            _interval, consec = _tick(watchdog_env, consec, False, blackbox_fn=_bb)

        assert len(calls) == 1, (
            "black box must fire exactly once per failure episode, "
            f"but was called {len(calls)} times"
        )
        assert watchdog_env.kill_calls == [], "no kill below debounce threshold"

    def test_blackbox_not_called_on_clean_tick(self, watchdog_env):
        calls: list[int] = []

        def _bb(log_fd, probe_ok, consec, debounce_n):
            calls.append(consec)

        _interval, consec = _tick(watchdog_env, 0, True, blackbox_fn=_bb)
        assert consec == 0
        assert calls == [], "black box must NOT fire on a clean probe tick"

    def test_episode_flag_resets_after_clean_tick(self, watchdog_env):
        calls: list[int] = []

        def _bb(log_fd, probe_ok, consec, debounce_n):
            calls.append(consec)

        _interval, consec = _tick(watchdog_env, 0, False, blackbox_fn=_bb)
        assert len(calls) == 1

        _interval, consec = _tick(watchdog_env, consec, True, blackbox_fn=_bb)
        assert consec == 0
        assert len(calls) == 1, "no extra call on clean tick"

        _interval, consec = _tick(watchdog_env, consec, False, blackbox_fn=_bb)
        assert len(calls) == 2, "black box must re-arm after a clean tick"

    def test_blackbox_never_called_on_kill_tick(self, watchdog_env):
        calls: list[int] = []

        def _bb(log_fd, probe_ok, consec, debounce_n):
            calls.append(consec)

        consec = 0
        for _ in range(DEBOUNCE_N):
            _interval, consec = _tick(watchdog_env, consec, False, blackbox_fn=_bb)

        assert watchdog_env.kill_calls, "kill must fire at debounce_n failures"
        assert len(calls) == 1, (
            "black box must fire exactly once (not on the kill tick)"
        )

    def test_self_kill_not_called_on_non_kill_failing_tick(self, watchdog_env):
        consec = 0
        for _ in range(DEBOUNCE_N - 1):
            _interval, consec = _tick(watchdog_env, consec, False, blackbox_fn=None)

        assert watchdog_env.kill_calls == [], "_self_kill must not fire below debounce"

    def test_returned_tuple_unchanged_with_blackbox(self, watchdog_env):
        _iv1, _c1 = _tick(watchdog_env, 0, False, blackbox_fn=None)

        if hasattr(daemon, "_WATCHDOG_BLACKBOX_EPISODE_FIRED"):
            daemon._WATCHDOG_BLACKBOX_EPISODE_FIRED = False

        calls: list = []

        def _bb(log_fd, probe_ok, consec, debounce_n):
            calls.append(True)

        _iv2, _c2 = _tick(watchdog_env, 0, False, blackbox_fn=_bb)

        assert _iv1 == _iv2, "interval must not change due to black box"
        assert _c1 == _c2, "consecutive_failures must not change due to black box"

    def test_blackbox_fn_receives_correct_args(self, watchdog_env):
        received: list[tuple] = []

        def _bb(log_fd, probe_ok, consec, debounce_n):
            received.append((log_fd, probe_ok, consec, debounce_n))

        _interval, _consec = _tick(watchdog_env, 0, False, blackbox_fn=_bb)

        assert len(received) == 1
        _fd, _probe_ok, _consec_val, _dbn = received[0]
        assert _probe_ok is False
        assert _dbn == daemon.WATCHDOG_FAILURE_DEBOUNCE_N

    def test_no_store_lock_taken_in_blackbox(self, watchdog_env):
        calls: list = []

        def _bb(log_fd, probe_ok, consec, debounce_n):
            calls.append("fired")

        _interval, consec = _tick(watchdog_env, 0, False, blackbox_fn=_bb)
        assert calls == ["fired"], "injected blackbox_fn must be called"


def test_capture_blackbox_writes_real_traceback(tmp_path):
    log_path = tmp_path / "bb.log"
    fd = os.open(str(log_path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        daemon._capture_blackbox(fd, False, 1, daemon.WATCHDOG_FAILURE_DEBOUNCE_N)
    finally:
        os.close(fd)

    content = log_path.read_text(encoding="utf-8", errors="replace")
    assert "pre_kill_forensic_dump" in content, "header must be present"
    assert ('File "' in content) or ("Thread" in content) or ("Stack" in content), (
        "faulthandler all-thread traceback must be present in the dump"
    )
    assert "--- end dump ---" in content, "trailing separator must be present"
    assert "tasks=" in content, "tasks field must appear in the header"


class TestBootLockBackoff:

    def test_backoff_retries_and_succeeds_when_lock_frees(
        self, tmp_path
    ):
        sentinel = object()
        calls: list[int] = []

        def _factory_first_k_fail(k: int):
            def _f():
                calls.append(len(calls) + 1)
                if len(calls) <= k:
                    raise HippoLockHeldError(
                        tmp_path / ".lock", f"attempt-{len(calls)}"
                    )
                return sentinel
            return _f

        result = asyncio.run(daemon._open_exclusive_store_with_backoff(
            _factory_first_k_fail(2),
            max_attempts=5,
            backoff_sec=0.01,
        ))
        assert result is sentinel, "helper must return the store on eventual success"
        assert len(calls) == 3, f"expected 3 attempts, got {len(calls)}"

    def test_backoff_exhausts_and_reraises_when_lock_never_frees(
        self, tmp_path
    ):
        calls: list[int] = []
        lock_path = tmp_path / ".lock"

        def _always_fail():
            calls.append(len(calls) + 1)
            raise HippoLockHeldError(lock_path, "always-held")

        with pytest.raises(HippoLockHeldError):
            asyncio.run(daemon._open_exclusive_store_with_backoff(
                _always_fail,
                max_attempts=3,
                backoff_sec=0.01,
            ))
        assert len(calls) == 3, (
            f"helper must attempt exactly max_attempts times, got {len(calls)}"
        )

    def test_backoff_succeeds_immediately_no_lock_error(self, tmp_path):
        sentinel = object()
        calls: list[int] = []

        def _succeed():
            calls.append(1)
            return sentinel

        result = asyncio.run(daemon._open_exclusive_store_with_backoff(
            _succeed,
            max_attempts=5,
            backoff_sec=0.01,
        ))
        assert result is sentinel
        assert len(calls) == 1, "no retry when first attempt succeeds"

    def test_backoff_non_lock_error_propagates_immediately(self, tmp_path):
        class _UnrelatedError(RuntimeError):
            pass

        calls: list[int] = []

        def _wrong_error():
            calls.append(1)
            raise _UnrelatedError("some other boot failure")

        with pytest.raises(_UnrelatedError):
            asyncio.run(daemon._open_exclusive_store_with_backoff(
                _wrong_error,
                max_attempts=5,
                backoff_sec=0.01,
            ))
        assert len(calls) == 1, "must NOT retry on non-HippoLockHeldError"

    def test_backoff_single_owner_guarantee(self, tmp_path):
        lock_path = tmp_path / ".lock"

        def _always_fail():
            raise HippoLockHeldError(lock_path, "unknown")

        exc: HippoLockHeldError | None = None
        try:
            asyncio.run(daemon._open_exclusive_store_with_backoff(
                _always_fail,
                max_attempts=2,
                backoff_sec=0.01,
            ))
        except HippoLockHeldError as e:
            exc = e

        assert exc is not None, "HippoLockHeldError must surface on exhaustion"
        assert "lock" in str(exc).lower() or str(lock_path) in str(exc)
