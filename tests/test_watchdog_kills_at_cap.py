"""The resident-set hard cap must kill PROMPTLY — on the first over-cap sample.

Regression coverage for a backstop that sampled the resident set correctly (in a
dedicated thread, never starved) but acted LATE: a runaway during the post-boot
drain blew the resident set past the hard cap while the cold-start grace blinded
the memory trigger, and once grace expired the failure debounce required three
consecutive over-cap ticks (~90 s at the steady poll cadence) before the kill
landed — long enough for the resident set to keep climbing toward host memory
pressure.

The hard cap is the last line of defence and sits well above the legitimate warm
plateau, so it must fast-path past the grace, the debounce, and the recovery
circuit breaker: a single sample over the cap is a runaway, not a transient.
These tests pin that fast-path behaviour, prove it still fires from the dedicated
watchdog thread while the main thread is busy in a blocking call, and assert the
revert-proof (the pre-fix grace+debounce path would NOT kill within the window).
"""

from __future__ import annotations

import os
import signal
import threading
import time

import pytest

from iai_mcp import daemon
from iai_mcp.daemon import _watchdog as wd

# Mirror the suite's watchdog constants so the test is independent of the live
# cap value (which is operator-tunable). The cap here is deliberately small.
HARD_CAP = 2_684_354_560
FLOOR = 1_610_612_736
DEBOUNCE_N = 3
GRACE = 600.0
MAX_RECOVERIES = 3
WINDOW = 600.0

RSS_LOW = 300 * 1024 * 1024
RSS_OVER_CAP = HARD_CAP + (512 * 1024 * 1024)  # comfortably past the ceiling

NORMAL = 1
WARN = 2


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


# --- decision-level: the hard cap kills on the first over-cap sample ----------


def test_over_cap_kills_on_first_sample_no_debounce():
    # consecutive_failures == 0 is the very first over-cap tick.
    assert _evaluate(rss=RSS_OVER_CAP, consecutive=0) == ("kill", "leak")


def test_over_cap_kills_inside_cold_start_grace():
    # uptime well inside the 600 s grace — the runaway-during-boot scenario.
    assert _evaluate(rss=RSS_OVER_CAP, uptime=5.0, consecutive=0) == (
        "kill",
        "leak",
    )


def test_over_cap_kills_even_with_breaker_tripped():
    # A respawn loop is safer than letting a runaway reach host memory pressure.
    now = 1_000_000.0
    recent = [now - 10, now - 20, now - 30]
    assert _evaluate(
        rss=RSS_OVER_CAP, consecutive=0, recoveries=recent, now_wall=now
    ) == ("kill", "leak")


def test_over_cap_kills_even_when_probe_ok():
    # The daemon answers its socket but its resident set is over the cap.
    assert _evaluate(probe_ok=True, rss=RSS_OVER_CAP, consecutive=0) == (
        "kill",
        "leak",
    )


def test_below_cap_pressure_path_still_debounced_and_graced():
    # The transient paths keep grace + debounce — only the hard cap fast-paths.
    assert _evaluate(rss=FLOOR + 1, pressure=WARN, consecutive=0) == (
        "none",
        "debounce",
    )
    assert _evaluate(
        rss=FLOOR + 1, pressure=WARN, uptime=5.0, consecutive=DEBOUNCE_N
    ) == ("none", "healthy")


# --- integrated tick: sampler over-cap -> SIGKILL, no test process death ------


@pytest.fixture
def tick_env(tmp_path, monkeypatch):
    log_path = tmp_path / ".daemon-watchdog.log"
    sock_path = str(tmp_path / ".daemon.sock")

    fd = os.open(str(log_path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    monkeypatch.setattr(daemon, "_WATCHDOG_LOG_FD", fd)

    # Boot timestamp INSIDE the grace window — proves the cap ignores grace.
    monkeypatch.setattr(daemon, "_daemon_started_monotonic", time.monotonic())

    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        daemon.os, "kill", lambda pid, sig: kill_calls.append((pid, sig))
    )

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


def test_tick_over_cap_kills_on_first_tick_within_grace(tick_env):
    store = object()
    _interval, consec = daemon._watchdog_tick(
        store,
        tick_env.sock_path,
        tick_env.log_path,
        0,  # consecutive_failures starts at zero
        probe_fn=_probe(True),  # socket answers — only the cap matters
        pressure_fn=lambda: NORMAL,
        rss_fn=lambda: daemon.WATCHDOG_RSS_HARD_CAP_BYTES + (512 * 1024 * 1024),
    )
    assert tick_env.kill_calls == [(os.getpid(), signal.SIGKILL)], (
        "an over-hard-cap sample must SIGKILL on the FIRST tick, even within "
        "the cold-start grace and with consecutive_failures == 0"
    )
    crumb = tick_env.log_path.read_text(encoding="utf-8")
    assert daemon.DAEMON_MEMORY_PRESSURE_KILL in crumb
    assert "reason=leak" in crumb


# --- the sampler fires from its own thread while the main thread is BUSY ------


def test_watchdog_thread_kills_while_main_thread_blocked(tmp_path, monkeypatch):
    """The dedicated watchdog thread samples + kills even while the main thread
    is parked in a long blocking call. The watchdog is a real OS thread, not a
    coroutine on the work loop, so a busy main thread cannot starve it."""
    log_path = tmp_path / ".daemon-watchdog.log"
    sock_path = str(tmp_path / ".daemon.sock")

    fd = os.open(str(log_path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    # Package-slot attributes are read via _pkg() (sys.modules["iai_mcp.daemon"]),
    # so they must be set on the package namespace.
    monkeypatch.setattr(daemon, "_WATCHDOG_LOG_FD", fd)
    # Inside grace — the cap must still fire.
    monkeypatch.setattr(daemon, "_daemon_started_monotonic", time.monotonic())
    # _watchdog_tick resolves the poll cadence + sampler helpers from the
    # _watchdog module's own globals (the tick has no probe_fn/rss_fn kwargs
    # when driven by _liveness_watchdog), so patch the module, not the package.
    monkeypatch.setattr(wd, "WATCHDOG_LIVENESS_POLL_SEC", 0.05)
    monkeypatch.setattr(wd, "WATCHDOG_WARN_POLL_SEC", 0.05)

    killed = threading.Event()
    kill_calls: list[tuple[int, int]] = []

    def _fake_kill(pid, sig):
        kill_calls.append((pid, sig))
        killed.set()

    monkeypatch.setattr(wd.os, "kill", _fake_kill)
    # Probe + pressure + the memory readers run inside the watchdog thread; keep
    # them trivial. The kill decision grades the cap against the charged metric
    # (macOS phys_footprint, resident_size on fallback), so the simulated runaway
    # must drive BOTH readers over the cap — not just resident_size.
    monkeypatch.setattr(wd, "_probe_status_roundtrip", _probe(True))
    monkeypatch.setattr(wd, "_vm_pressure_level", lambda: NORMAL)
    monkeypatch.setattr(
        wd,
        "_own_rss_bytes",
        lambda: wd.WATCHDOG_RSS_HARD_CAP_BYTES + (512 * 1024 * 1024),
    )
    monkeypatch.setattr(
        wd,
        "_phys_footprint_bytes",
        lambda: wd.WATCHDOG_RSS_HARD_CAP_BYTES + (512 * 1024 * 1024),
    )

    stop = threading.Event()
    wd_thread = threading.Thread(
        target=daemon._liveness_watchdog,
        args=(object(), stop, sock_path),
        name="test-liveness-watchdog",
        daemon=True,
    )
    wd_thread.start()

    # Simulate the main thread blocked in a long GIL-releasing call (the real
    # drain spends its time in native embed/sqlite that releases the GIL).
    # time.sleep releases the GIL the same way; the watchdog thread runs.
    fired_in_window = killed.wait(timeout=3.0)

    stop.set()
    try:
        os.close(fd)
    except OSError:
        pass

    assert fired_in_window, (
        "the dedicated watchdog thread did not kill within the window while "
        "the main thread was blocked — the backstop is starvable"
    )
    assert kill_calls and kill_calls[0] == (os.getpid(), signal.SIGKILL)


# --- revert-proof: the pre-fix grace+debounce path would NOT kill in window ---


def _evaluate_legacy(
    probe_ok,
    rss,
    pressure_level,
    uptime_sec,
    consecutive_failures,
    recovery_timestamps,
    now_wall,
    *,
    hard_cap,
    contributor_floor,
    debounce_n,
    cold_start_grace_sec,
    max_recoveries,
    recovery_window_sec,
):
    """The decision logic exactly as it stood BEFORE the hard-cap fast path:
    the leak rode through the cold-start grace and the failure debounce. Pinned
    here so the test fails loudly if someone reverts the fix to this shape."""
    recent = [t for t in recovery_timestamps if now_wall - t <= recovery_window_sec]
    breaker_tripped = len(recent) >= max_recoveries

    leak = rss is not None and rss > hard_cap
    pressure = pressure_level is not None and pressure_level >= 2
    big = rss is not None and rss > contributor_floor
    in_grace = uptime_sec < cold_start_grace_sec

    mem_trigger = (not in_grace) and (leak or (pressure and big))
    wedge_trigger = not probe_ok

    if not (mem_trigger or wedge_trigger):
        return ("none", "healthy")
    if consecutive_failures < debounce_n:
        return ("none", "debounce")
    if breaker_tripped:
        return ("needs_operator", "circuit_breaker")
    if wedge_trigger:
        return ("kill", "wedge")
    if leak:
        return ("kill", "leak")
    return ("kill", "memory")


def _legacy(rss, uptime, consecutive):
    return _evaluate_legacy(
        True,
        rss,
        NORMAL,
        uptime,
        consecutive,
        [],
        1_000_000.0,
        hard_cap=HARD_CAP,
        contributor_floor=FLOOR,
        debounce_n=DEBOUNCE_N,
        cold_start_grace_sec=GRACE,
        max_recoveries=MAX_RECOVERIES,
        recovery_window_sec=WINDOW,
    )


def test_revert_proof_legacy_path_misses_the_over_cap_window():
    # Within grace, the legacy path ignores the over-cap leak entirely.
    assert _legacy(RSS_OVER_CAP, uptime=5.0, consecutive=0) == ("none", "healthy")
    # Just out of grace but on the first over-cap tick, the legacy path debounces.
    assert _legacy(RSS_OVER_CAP, uptime=GRACE + 1.0, consecutive=0) == (
        "none",
        "debounce",
    )
    assert _legacy(RSS_OVER_CAP, uptime=GRACE + 1.0, consecutive=DEBOUNCE_N - 1) == (
        "none",
        "debounce",
    )
    # The CURRENT path, given the identical first-tick inputs, kills immediately —
    # this is the behavioural delta the fix introduces.
    assert _evaluate(rss=RSS_OVER_CAP, uptime=5.0, consecutive=0) == ("kill", "leak")
    assert _evaluate(rss=RSS_OVER_CAP, uptime=GRACE + 1.0, consecutive=0) == (
        "kill",
        "leak",
    )
