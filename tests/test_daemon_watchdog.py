from __future__ import annotations

import asyncio
import os
import signal
import socket
import threading
import time

import pytest

from iai_mcp import daemon

HARD_CAP = 2_684_354_560
FLOOR = 1_610_612_736
DEBOUNCE_N = 3
GRACE = 600.0
MAX_RECOVERIES = 3
WINDOW = 600.0

RSS_LOW = 300 * 1024 * 1024
RSS_BIG = 2 * 1024 * 1024 * 1024
RSS_LEAK = 3 * 1024 * 1024 * 1024

NORMAL = 1
WARN = 2
CRITICAL = 4


def _evaluate(
    probe_ok=True,
    rss=RSS_LOW,
    pressure=NORMAL,
    uptime=GRACE + 1.0,
    consecutive=0,
    recoveries=None,
    now_wall=1_000_000.0,
):
    return daemon._evaluate_watchdog(
        probe_ok,
        rss,
        pressure,
        uptime,
        consecutive,
        list(recoveries or []),
        now_wall,
        hard_cap=HARD_CAP,
        contributor_floor=FLOOR,
        debounce_n=DEBOUNCE_N,
        cold_start_grace_sec=GRACE,
        max_recoveries=MAX_RECOVERIES,
        recovery_window_sec=WINDOW,
    )


def test_healthy_idle_no_kill():
    assert _evaluate() == ("none", "healthy")


def test_busy_healthy_no_kill_cpu_never_a_signal():
    assert _evaluate(probe_ok=True, rss=RSS_LOW, pressure=NORMAL) == (
        "none",
        "healthy",
    )


def test_single_wedge_blip_does_not_kill_debounce():
    assert _evaluate(probe_ok=False, consecutive=1) == ("none", "debounce")
    assert _evaluate(probe_ok=False, consecutive=DEBOUNCE_N - 1) == (
        "none",
        "debounce",
    )


def test_wedge_after_n_consecutive_failures_kills():
    assert _evaluate(probe_ok=False, consecutive=DEBOUNCE_N) == ("kill", "wedge")


def test_wedge_not_grace_covered():
    assert _evaluate(probe_ok=False, consecutive=DEBOUNCE_N, uptime=1.0) == (
        "kill",
        "wedge",
    )


def test_leak_kills_even_at_normal_pressure():
    assert _evaluate(rss=RSS_LEAK, pressure=NORMAL, consecutive=DEBOUNCE_N) == (
        "kill",
        "leak",
    )


def test_leak_single_tick_kills_immediately_no_debounce():
    # A resident-set reading over the hard cap is a runaway, not a transient.
    # The hard cap must kill on the FIRST over-cap sample — it does not wait
    # for the failure debounce that exists to ride out transient probe blips.
    assert _evaluate(rss=RSS_LEAK, pressure=NORMAL, consecutive=0) == (
        "kill",
        "leak",
    )
    assert _evaluate(rss=RSS_LEAK, pressure=NORMAL, consecutive=1) == (
        "kill",
        "leak",
    )


def test_warn_plus_big_kills_memory():
    assert _evaluate(rss=RSS_BIG, pressure=WARN, consecutive=DEBOUNCE_N) == (
        "kill",
        "memory",
    )


def test_critical_plus_big_also_kills_memory():
    assert _evaluate(rss=RSS_BIG, pressure=CRITICAL, consecutive=DEBOUNCE_N) == (
        "kill",
        "memory",
    )


def test_warn_but_another_process_owns_ram_no_kill():
    assert _evaluate(rss=RSS_LOW, pressure=WARN, consecutive=DEBOUNCE_N) == (
        "none",
        "healthy",
    )


def test_unreadable_pressure_does_not_kill():
    assert _evaluate(rss=RSS_BIG, pressure=None, consecutive=DEBOUNCE_N) == (
        "none",
        "healthy",
    )


def test_unreadable_pressure_leak_backstop_still_fires():
    assert _evaluate(rss=RSS_LEAK, pressure=None, consecutive=DEBOUNCE_N) == (
        "kill",
        "leak",
    )


def test_cold_start_grace_suppresses_memory_trigger():
    assert _evaluate(
        rss=RSS_BIG, pressure=WARN, uptime=10.0, consecutive=DEBOUNCE_N
    ) == ("none", "healthy")


def test_cold_start_grace_does_not_suppress_hard_cap_leak():
    # The cold-start grace suppresses the pressure/contributor path (a booting
    # daemon's warm-up spike is legitimate), but it must NOT blind the hard cap:
    # a runaway during the post-boot drain is exactly what the cap exists to
    # catch. An over-hard-cap reading kills even within the grace window.
    assert _evaluate(
        rss=RSS_LEAK, pressure=NORMAL, uptime=10.0, consecutive=0
    ) == ("kill", "leak")


def test_circuit_breaker_trips_to_needs_operator_not_kill():
    now = 1_000_000.0
    recent = [now - 10, now - 20, now - 30]
    assert _evaluate(
        probe_ok=False, consecutive=DEBOUNCE_N, recoveries=recent, now_wall=now
    ) == ("needs_operator", "circuit_breaker")


def test_circuit_breaker_ignores_out_of_window_recoveries():
    now = 1_000_000.0
    old = [now - WINDOW - 100, now - WINDOW - 200, now - WINDOW - 300]
    assert _evaluate(
        probe_ok=False, consecutive=DEBOUNCE_N, recoveries=old, now_wall=now
    ) == ("kill", "wedge")


def test_circuit_breaker_below_threshold_still_kills():
    now = 1_000_000.0
    recent = [now - 10, now - 20]
    assert _evaluate(
        probe_ok=False, consecutive=DEBOUNCE_N, recoveries=recent, now_wall=now
    ) == ("kill", "wedge")


def test_poll_interval_normal_is_steady():
    assert daemon._next_poll_interval(NORMAL) == daemon.WATCHDOG_LIVENESS_POLL_SEC


def test_poll_interval_none_is_steady():
    assert daemon._next_poll_interval(None) == daemon.WATCHDOG_LIVENESS_POLL_SEC


def test_poll_interval_warn_is_tightened():
    assert daemon._next_poll_interval(WARN) == daemon.WATCHDOG_WARN_POLL_SEC


def test_poll_interval_critical_is_tightened():
    assert daemon._next_poll_interval(CRITICAL) == daemon.WATCHDOG_WARN_POLL_SEC


def test_warn_poll_is_strictly_tighter_than_steady():
    assert daemon.WATCHDOG_WARN_POLL_SEC < daemon.WATCHDOG_LIVENESS_POLL_SEC


@pytest.fixture
def watchdog_env(tmp_path, monkeypatch):
    log_path = tmp_path / ".daemon-watchdog.log"
    sock_path = str(tmp_path / ".daemon.sock")

    fd = os.open(str(log_path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    monkeypatch.setattr(daemon, "_WATCHDOG_LOG_FD", fd)

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


def _read_breadcrumb(log_path):
    return log_path.read_text(encoding="utf-8")


def test_thread_wedge_after_n_consecutive_kills(watchdog_env):
    store = object()
    consec = 0
    for _ in range(DEBOUNCE_N - 1):
        _interval, consec = daemon._watchdog_tick(
            store,
            watchdog_env.sock_path,
            watchdog_env.log_path,
            consec,
            probe_fn=_probe(False),
            pressure_fn=lambda: NORMAL,
            rss_fn=lambda: RSS_LOW,
        )
    assert watchdog_env.kill_calls == []
    _interval, consec = daemon._watchdog_tick(
        store,
        watchdog_env.sock_path,
        watchdog_env.log_path,
        consec,
        probe_fn=_probe(False),
        pressure_fn=lambda: NORMAL,
        rss_fn=lambda: RSS_LOW,
    )
    assert watchdog_env.kill_calls == [(os.getpid(), signal.SIGKILL)]
    crumb = _read_breadcrumb(watchdog_env.log_path)
    assert daemon.DAEMON_WEDGE_KILL in crumb
    assert "reason=wedge" in crumb


def test_thread_healthy_busy_not_killed(watchdog_env):
    store = object()
    consec = 0
    for _ in range(DEBOUNCE_N + 5):
        _interval, consec = daemon._watchdog_tick(
            store,
            watchdog_env.sock_path,
            watchdog_env.log_path,
            consec,
            probe_fn=_probe(True),
            pressure_fn=lambda: NORMAL,
            rss_fn=lambda: RSS_LOW,
        )
    assert watchdog_env.kill_calls == []
    assert consec == 0


def test_thread_warn_plus_big_memory_kill(watchdog_env):
    store = object()
    consec = 0
    for _ in range(DEBOUNCE_N):
        _interval, consec = daemon._watchdog_tick(
            store,
            watchdog_env.sock_path,
            watchdog_env.log_path,
            consec,
            probe_fn=_probe(True),
            pressure_fn=lambda: WARN,
            rss_fn=lambda: RSS_BIG,
        )
    assert watchdog_env.kill_calls == [(os.getpid(), signal.SIGKILL)]
    crumb = _read_breadcrumb(watchdog_env.log_path)
    assert daemon.DAEMON_MEMORY_PRESSURE_KILL in crumb
    assert "reason=memory" in crumb


def test_in_grace_window_suppresses_memory_kill(watchdog_env, monkeypatch):
    # A booting daemon must NOT self-kill while within the cold-start grace
    # window. The tick reads the boot timestamp from the package slot; if it
    # read a stale module-local slot, uptime would collapse to the 1e9 branch,
    # in_grace would be False, and these memory-kill inputs would trigger a
    # spurious kill. Override the boot timestamp to "just started" (uptime ~ 0).
    monkeypatch.setattr(daemon, "_daemon_started_monotonic", time.monotonic())
    store = object()
    consec = 0
    for _ in range(DEBOUNCE_N):
        _interval, consec = daemon._watchdog_tick(
            store,
            watchdog_env.sock_path,
            watchdog_env.log_path,
            consec,
            probe_fn=_probe(True),
            pressure_fn=lambda: WARN,
            rss_fn=lambda: RSS_BIG,
        )
    assert watchdog_env.kill_calls == [], (
        "watchdog self-killed within the cold-start grace window — "
        "_watchdog_tick is reading a stale boot timestamp, not the package slot"
    )


def test_thread_warn_another_process_owns_ram_no_kill(watchdog_env):
    store = object()
    consec = 0
    for _ in range(DEBOUNCE_N + 5):
        _interval, consec = daemon._watchdog_tick(
            store,
            watchdog_env.sock_path,
            watchdog_env.log_path,
            consec,
            probe_fn=_probe(True),
            pressure_fn=lambda: WARN,
            rss_fn=lambda: RSS_LOW,
        )
    assert watchdog_env.kill_calls == []


def test_thread_adaptive_cadence_tightens_under_warn(watchdog_env):
    store = object()
    interval_normal, _ = daemon._watchdog_tick(
        store,
        watchdog_env.sock_path,
        watchdog_env.log_path,
        0,
        probe_fn=_probe(True),
        pressure_fn=lambda: NORMAL,
        rss_fn=lambda: RSS_LOW,
    )
    interval_warn, _ = daemon._watchdog_tick(
        store,
        watchdog_env.sock_path,
        watchdog_env.log_path,
        0,
        probe_fn=_probe(True),
        pressure_fn=lambda: WARN,
        rss_fn=lambda: RSS_LOW,
    )
    assert interval_normal == daemon.WATCHDOG_LIVENESS_POLL_SEC
    assert interval_warn == daemon.WATCHDOG_WARN_POLL_SEC


def test_thread_circuit_breaker_emits_needs_operator_not_kill(
    watchdog_env, monkeypatch
):
    now = time.time()
    lines = [
        f"{daemon.datetime.fromtimestamp(now - 10 * (i + 1), daemon.timezone.utc).isoformat()} "
        f"{daemon.DAEMON_WEDGE_KILL} reason=wedge pid=999\n"
        for i in range(MAX_RECOVERIES)
    ]
    watchdog_env.log_path.write_text("".join(lines), encoding="utf-8")
    os.close(watchdog_env.fd)
    fd = os.open(
        str(watchdog_env.log_path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600
    )
    monkeypatch.setattr(daemon, "_WATCHDOG_LOG_FD", fd)

    emitted: list[tuple] = []

    def _fake_write_event(store, kind, data, **kw):
        emitted.append((kind, data, kw))
        return "evt-id"

    monkeypatch.setattr(daemon, "write_event", _fake_write_event)

    store = object()
    consec = 0
    for _ in range(DEBOUNCE_N):
        _interval, consec = daemon._watchdog_tick(
            store,
            watchdog_env.sock_path,
            watchdog_env.log_path,
            consec,
            probe_fn=_probe(False),
            pressure_fn=lambda: NORMAL,
            rss_fn=lambda: RSS_LOW,
        )
    assert watchdog_env.kill_calls == []
    assert any(k == daemon.DAEMON_WATCHDOG_NEEDS_OPERATOR for k, _d, _kw in emitted)
    try:
        os.close(fd)
    except OSError:
        pass


def test_self_kill_is_unconditional_when_breadcrumb_fails_wedge(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        daemon, "_daemon_started_monotonic", time.monotonic() - (GRACE + 60.0)
    )
    bad_fd = os.open(str(tmp_path / "tmp"), os.O_WRONLY | os.O_CREAT, 0o600)
    os.close(bad_fd)
    monkeypatch.setattr(daemon, "_WATCHDOG_LOG_FD", bad_fd)

    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        daemon.os, "kill", lambda pid, sig: kill_calls.append((pid, sig))
    )

    log_path = tmp_path / ".daemon-watchdog.log"
    store = object()
    consec = 0
    for _ in range(DEBOUNCE_N):
        _interval, consec = daemon._watchdog_tick(
            store,
            str(tmp_path / ".daemon.sock"),
            log_path,
            consec,
            probe_fn=_probe(False),
            pressure_fn=lambda: NORMAL,
            rss_fn=lambda: RSS_LOW,
        )
    assert kill_calls == [(os.getpid(), signal.SIGKILL)]


def test_self_kill_is_unconditional_when_breadcrumb_fails_memory(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        daemon, "_daemon_started_monotonic", time.monotonic() - (GRACE + 60.0)
    )
    bad_fd = os.open(str(tmp_path / "tmp2"), os.O_WRONLY | os.O_CREAT, 0o600)
    os.close(bad_fd)
    monkeypatch.setattr(daemon, "_WATCHDOG_LOG_FD", bad_fd)

    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        daemon.os, "kill", lambda pid, sig: kill_calls.append((pid, sig))
    )

    log_path = tmp_path / ".daemon-watchdog.log"
    store = object()
    consec = 0
    for _ in range(DEBOUNCE_N):
        _interval, consec = daemon._watchdog_tick(
            store,
            str(tmp_path / ".daemon.sock"),
            log_path,
            consec,
            probe_fn=_probe(True),
            pressure_fn=lambda: WARN,
            rss_fn=lambda: RSS_BIG,
        )
    assert kill_calls == [(os.getpid(), signal.SIGKILL)]


def test_self_kill_direct_breadcrumb_failure_still_kills(tmp_path, monkeypatch):

    def _raise(_line):
        raise OSError("simulated blocked/failed breadcrumb sink")

    monkeypatch.setattr(daemon, "_write_breadcrumb", _raise)
    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        daemon.os, "kill", lambda pid, sig: kill_calls.append((pid, sig))
    )
    daemon._self_kill("wedge", daemon.DAEMON_WEDGE_KILL)
    assert kill_calls == [(os.getpid(), signal.SIGKILL)]


def test_probe_returns_false_when_no_socket(tmp_path):
    sock_path = str(tmp_path / "absent.sock")
    assert asyncio.run(daemon._probe_status_roundtrip(sock_path, 0.2)) is False


def test_probe_returns_false_on_connect_but_no_reply(tmp_path, short_socket):
    sock_path = str(short_socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)
    accepted: list = []

    def _accept_and_hang():
        try:
            conn, _ = srv.accept()
            accepted.append(conn)
        except OSError:
            pass

    t = threading.Thread(target=_accept_and_hang, daemon=True)
    t.start()
    try:
        result = asyncio.run(daemon._probe_status_roundtrip(sock_path, 0.3))
        assert result is False
    finally:
        for c in accepted:
            try:
                c.close()
            except OSError:
                pass
        srv.close()


def test_probe_returns_true_on_full_roundtrip(tmp_path, short_socket):
    sock_path = str(short_socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)
    held: list = []

    def _accept_and_reply():
        try:
            conn, _ = srv.accept()
            held.append(conn)
            conn.recv(4096)
            conn.sendall(b'{"ok": true}\n')
        except OSError:
            pass

    t = threading.Thread(target=_accept_and_reply, daemon=True)
    t.start()
    try:
        result = asyncio.run(daemon._probe_status_roundtrip(sock_path, 1.0))
        assert result is True
    finally:
        for c in held:
            try:
                c.close()
            except OSError:
                pass
        srv.close()


def test_load_recovery_timestamps_reads_back_kill_lines(tmp_path):
    log_path = tmp_path / ".daemon-watchdog.log"
    now = time.time()
    iso = lambda ts: daemon.datetime.fromtimestamp(ts, daemon.timezone.utc).isoformat()
    log_path.write_text(
        f"{iso(now - 5)} {daemon.DAEMON_WEDGE_KILL} reason=wedge pid=1\n"
        f"{iso(now - 6)} {daemon.DAEMON_MEMORY_PRESSURE_KILL} reason=memory pid=2\n"
        f"{iso(now - 7)} {daemon.DAEMON_WATCHDOG_NEEDS_OPERATOR} reason=x pid=3\n"
        "garbage line that should be skipped\n",
        encoding="utf-8",
    )
    ts = daemon._load_recovery_timestamps(
        log_path, (daemon.DAEMON_WEDGE_KILL, daemon.DAEMON_MEMORY_PRESSURE_KILL)
    )
    assert len(ts) == 2
    assert abs(ts[0] - (now - 5)) < 1.0
    assert abs(ts[1] - (now - 6)) < 1.0


def test_load_recovery_timestamps_missing_file_is_empty(tmp_path):
    assert daemon._load_recovery_timestamps(
        tmp_path / "nope.log", (daemon.DAEMON_WEDGE_KILL,)
    ) == []
