"""Deterministic breadcrumb witness for the boot-window false-kill fix.

The traced handler wraps its re-raise in ``try: ... except Exception:
os._exit(...)`` -- any mock that raises inside ``os.kill`` would be
swallowed into a silent runner kill. Every mock here is record-only; all
assertions live in the test body, after the handler has returned.
"""

from __future__ import annotations

import os
import signal


def test_daemon_boot_window_signal_breadcrumb(monkeypatch, tmp_path):
    from iai_mcp.daemon import _install_boot_signal_trace

    watchdog_log = tmp_path / "watchdog.log"
    monkeypatch.setattr(
        "iai_mcp.daemon._watchdog._watchdog_log_path",
        lambda: watchdog_log,
    )

    registered: list[tuple[int, object]] = []

    def signal_recorder(sig, handler):
        registered.append((sig, handler))
        return None

    monkeypatch.setattr("signal.signal", signal_recorder)

    _install_boot_signal_trace()

    registered_sigs = {sig for sig, _handler in registered}
    # SIGHUP exists only on POSIX; Windows registers the other two.
    expected = {signal.SIGTERM, signal.SIGINT}
    if hasattr(signal, "SIGHUP"):
        expected.add(signal.SIGHUP)
    assert registered_sigs == expected
    handlers = {handler for _sig, handler in registered}
    assert len(handlers) == 1, "every signal must map to the same handler"
    trace = registered[0][1]

    # Guard: a failed monkeypatch would otherwise write the breadcrumb to
    # the real watchdog log.
    from iai_mcp.daemon._watchdog import _watchdog_log_path

    assert str(_watchdog_log_path()).startswith(str(tmp_path))

    kill_calls: list[tuple[int, int]] = []
    log_snapshots: list[str] = []
    signal_state_at_kill: list[list[tuple[int, object]]] = []

    def kill_recorder(pid, sig):
        # RECORD-ONLY: never raise, not even on our own bookkeeping. The
        # handler's bare ``except Exception: os._exit(128 + signum)`` would
        # swallow any exception raised here -- including a missing-file
        # read on a regression that skipped the breadcrumb write -- and
        # hard-kill the pytest runner with no summary line.
        try:
            kill_calls.append((pid, sig))
            try:
                snapshot = watchdog_log.read_text()
            except OSError:
                snapshot = ""
            log_snapshots.append(snapshot)
            signal_state_at_kill.append(list(registered))
        except Exception:  # noqa: BLE001 -- bookkeeping must never raise here
            pass

    monkeypatch.setattr("os.kill", kill_recorder)

    trace(signal.SIGTERM, None)

    # write-before-death: the breadcrumb was already on disk at the moment
    # the re-raise (os.kill) was attempted.
    assert len(log_snapshots) == 1
    expected_line = (
        f"daemon_boot_window_signal signal={signal.SIGTERM} pid={os.getpid()}"
    )
    assert expected_line in log_snapshots[0]

    # the re-raise targeted this process with the original signal.
    assert kill_calls == [(os.getpid(), signal.SIGTERM)]

    # ordering: the SIG_DFL restore was recorded before the kill call.
    # (one registration per available signal, plus the restore)
    state_at_kill = signal_state_at_kill[0]
    assert len(state_at_kill) == len(expected) + 1
    assert state_at_kill[-1] == (signal.SIGTERM, signal.SIG_DFL)
