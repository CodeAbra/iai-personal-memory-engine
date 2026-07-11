"""Hermetic tests for the sleep-cycle staleness predicate and its
`_watchdog_tick` integration.

The predicate `_check_sleep_cycle_staleness(state, now)` decides whether
the daemon's lifecycle has been parked in SLEEP with an unadvancing
consolidation cycle for longer than a configurable threshold. When true,
the watchdog tick fires the operator-facing `daemon_sleep_cycle_stale`
event at most once per stuck cycle. Dedup is durable: the tick queries
the events ledger for a prior event whose payload `sleep_cycle_started_at`
matches the current cycle's identity, so the same wedge does not re-emit
after a daemon restart. An in-process attribute cache primes the fast-path
for subsequent ticks within one process lifetime.

Tests are hermetic: no real `~/.iai-mcp/` interaction. State is injected
via monkeypatch on `iai_mcp.lifecycle_state.load_state`; the watchdog's
in-memory dedup state is reset between cases via `monkeypatch.setattr`
on `daemon._last_sleep_stale_started_at`; the ledger query is intercepted
via monkeypatch on `iai_mcp.daemon.query_events` (alias resolved by the
late import inside the tick). One regression test
(`test_first_tick_on_stale_state_emits_without_attr_init`) deliberately
skips the dedup-reset so the call-site's `getattr(_pkg(), ..., "")`
default-empty contract is pinned against the package-vs-submodule
attribute trap.
"""
from __future__ import annotations

import logging
import os
import signal
import time
from datetime import datetime, timedelta, timezone

import pytest

from iai_mcp import daemon
from iai_mcp.lifecycle_state import LifecycleState

NOW = datetime(2026, 6, 19, 12, 0, 0, tzinfo=timezone.utc)


def _progress(
    started_at: datetime,
    attempt: int = 1,
    last_index: int = 1,
    last_error: str = "deferred:step=OPTIMIZE_HIPPO:chunk_idx=0",
) -> dict:
    return {
        "last_completed_index": last_index,
        "attempt": attempt,
        "last_error": last_error,
        "started_at": started_at.isoformat(),
    }


def _state(
    current_state: str,
    progress: dict | None,
    crisis_mode: bool = False,
) -> dict:
    return {
        "current_state": current_state,
        "since_ts": "2026-06-01T00:00:00+00:00",
        "last_activity_ts": "2026-06-19T11:59:00+00:00",
        "wrapper_event_seq": 0,
        "sleep_cycle_progress": progress,
        "quarantine": None,
        "shadow_run": False,
        "crisis_mode": crisis_mode,
    }


# ---- predicate-only tests ----


class TestPredicate:
    def test_wake_state_is_never_stale(self):
        state = _state(
            LifecycleState.WAKE.value,
            _progress(NOW - timedelta(days=10)),
        )
        is_stale, _ = daemon._check_sleep_cycle_staleness(state, NOW)
        assert is_stale is False

    def test_sleep_with_no_progress_is_not_stale(self):
        state = _state(LifecycleState.SLEEP.value, None)
        assert daemon._check_sleep_cycle_staleness(state, NOW)[0] is False

    def test_sleep_under_threshold_is_not_stale(self):
        state = _state(
            LifecycleState.SLEEP.value,
            _progress(NOW - timedelta(seconds=7199)),
        )
        assert daemon._check_sleep_cycle_staleness(state, NOW)[0] is False

    def test_sleep_just_over_threshold_is_stale(self):
        state = _state(
            LifecycleState.SLEEP.value,
            _progress(NOW - timedelta(seconds=7201)),
        )
        is_stale, ctx = daemon._check_sleep_cycle_staleness(state, NOW)
        assert is_stale is True
        assert ctx["sleep_stuck_sec"] == 7201
        assert ctx["last_completed_index"] == 1
        assert ctx["attempt"] == 1
        assert ctx["crisis_mode"] is False

    def test_attempt_gt_1_is_not_stale(self):
        # attempt >= 2 means the cycle has retried — quarantine/retry band,
        # outside the scope of this predicate.
        state = _state(
            LifecycleState.SLEEP.value,
            _progress(NOW - timedelta(days=10), attempt=2),
        )
        assert daemon._check_sleep_cycle_staleness(state, NOW)[0] is False

    def test_attempt_0_step_body_hang_is_stale(self):
        # attempt=0 is the on-disk shape for a true step-body hang: after the
        # prior step completes _save_progress writes attempt=0, and if the next
        # step's body blocks forever no further save runs, leaving attempt=0.
        # Also covers a clean-defer-loop wedge (no prior exception → attempt
        # preserved at 0 via prior.get("attempt", 0)).
        state = _state(
            LifecycleState.SLEEP.value,
            _progress(NOW - timedelta(days=10), attempt=0),
        )
        is_stale, ctx = daemon._check_sleep_cycle_staleness(state, NOW)
        assert is_stale is True
        assert ctx["attempt"] == 0

    def test_attempt_negative_is_not_stale(self):
        # Defensive: a corrupt negative attempt value must not fire.
        state = _state(
            LifecycleState.SLEEP.value,
            _progress(NOW - timedelta(days=10), attempt=-1),
        )
        assert daemon._check_sleep_cycle_staleness(state, NOW)[0] is False

    def test_live_wedge_shape_matches_stale(self):
        # Exact shape from a real wedged lifecycle_state.json on disk.
        state = _state(
            LifecycleState.SLEEP.value,
            _progress(
                datetime.fromisoformat("2026-06-09T23:42:45.360247+00:00"),
                attempt=1,
                last_index=1,
                last_error="deferred:step=OPTIMIZE_HIPPO:chunk_idx=0",
            ),
            crisis_mode=True,
        )
        is_stale, ctx = daemon._check_sleep_cycle_staleness(state, NOW)
        assert is_stale is True
        assert ctx["crisis_mode"] is True
        assert ctx["last_error"] == "deferred:step=OPTIMIZE_HIPPO:chunk_idx=0"
        assert ctx["sleep_stuck_sec"] > 7200

    def test_malformed_started_at_returns_false(self):
        state = _state(
            LifecycleState.SLEEP.value,
            {"attempt": 1, "started_at": "not-an-iso-string"},
        )
        assert daemon._check_sleep_cycle_staleness(state, NOW)[0] is False

    def test_missing_started_at_returns_false(self):
        state = _state(LifecycleState.SLEEP.value, {"attempt": 1})
        assert daemon._check_sleep_cycle_staleness(state, NOW)[0] is False

    def test_naive_started_at_treated_as_utc(self):
        # Defensive: if a future writer drops the tz suffix, predicate stays sane.
        naive_iso = (NOW - timedelta(days=10)).replace(tzinfo=None).isoformat()
        state = _state(
            LifecycleState.SLEEP.value,
            {"attempt": 1, "started_at": naive_iso},
        )
        assert daemon._check_sleep_cycle_staleness(state, NOW)[0] is True


# ---- emit + dedup integration tests ----


class TestWatchdogTickEmits:
    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        # Mirror tests/test_daemon_watchdog.py:191-222 fixture shape.
        log_path = tmp_path / ".daemon-watchdog.log"
        fd = os.open(
            str(log_path),
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o600,
        )
        monkeypatch.setattr(daemon, "_WATCHDOG_LOG_FD", fd)
        # Bypass cold-start grace (default 600s); 1200s uptime > 600s.
        monkeypatch.setattr(
            daemon, "_daemon_started_monotonic", time.monotonic() - 1200.0,
        )
        monkeypatch.setattr(daemon.os, "kill", lambda p, s: None)
        # Reset the cycle-dedup state between tests so a previous test's
        # successful emit does not suppress the next test's first emit.
        monkeypatch.setattr(daemon, "_last_sleep_stale_started_at", "")

        class _Ns:
            pass

        ns = _Ns()
        ns.tmp_path = tmp_path
        ns.log_path = log_path
        ns.sock_path = str(tmp_path / ".daemon.sock")
        ns.fd = fd
        yield ns
        try:
            os.close(fd)
        except OSError:
            pass

    def test_stale_state_emits_once(self, env, monkeypatch):
        stale_state = _state(
            LifecycleState.SLEEP.value,
            _progress(datetime.now(timezone.utc) - timedelta(days=10)),
            crisis_mode=True,
        )
        monkeypatch.setattr(
            "iai_mcp.lifecycle_state.load_state",
            lambda path=None: stale_state,
        )
        emitted: list = []
        monkeypatch.setattr(
            daemon,
            "write_event",
            lambda store, kind, data, **kw: emitted.append((kind, data, kw)) or "id",
        )

        async def _probe_ok(_sock, _t):
            return True

        for _ in range(5):
            daemon._watchdog_tick(
                object(),
                env.sock_path,
                env.log_path,
                0,
                probe_fn=_probe_ok,
                pressure_fn=lambda: 1,
                rss_fn=lambda: 300 * 1024 * 1024,
            )

        stale_emits = [e for e in emitted if e[0] == "daemon_sleep_cycle_stale"]
        assert len(stale_emits) == 1
        assert stale_emits[0][1]["crisis_mode"] is True
        assert stale_emits[0][2]["severity"] == "critical"

    def test_new_cycle_re_emits(self, env, monkeypatch):
        states = {
            "first": _state(
                LifecycleState.SLEEP.value,
                _progress(datetime.now(timezone.utc) - timedelta(days=10)),
            ),
            "second": _state(
                LifecycleState.SLEEP.value,
                _progress(datetime.now(timezone.utc) - timedelta(days=20)),
            ),
        }
        cycle = {"current": "first"}
        monkeypatch.setattr(
            "iai_mcp.lifecycle_state.load_state",
            lambda path=None: states[cycle["current"]],
        )
        emitted: list = []
        monkeypatch.setattr(
            daemon,
            "write_event",
            lambda store, kind, data, **kw: emitted.append((kind, data, kw)) or "id",
        )

        async def _probe_ok(_sock, _t):
            return True

        daemon._watchdog_tick(
            object(),
            env.sock_path,
            env.log_path,
            0,
            probe_fn=_probe_ok,
            pressure_fn=lambda: 1,
            rss_fn=lambda: 300 * 1024 * 1024,
        )
        cycle["current"] = "second"
        daemon._watchdog_tick(
            object(),
            env.sock_path,
            env.log_path,
            0,
            probe_fn=_probe_ok,
            pressure_fn=lambda: 1,
            rss_fn=lambda: 300 * 1024 * 1024,
        )

        stale_emits = [e for e in emitted if e[0] == "daemon_sleep_cycle_stale"]
        assert len(stale_emits) == 2
        assert (
            stale_emits[0][1]["sleep_cycle_started_at"]
            != stale_emits[1][1]["sleep_cycle_started_at"]
        )

    def test_durable_dedup_suppresses_emit_on_restart(self, env, monkeypatch):
        """Same on-disk wedge (same started_at) must NOT re-emit after a daemon
        restart.  Simulates a restart by clearing the in-process dedup key and
        injecting a ledger that already contains the prior event for that cycle.
        """
        import iai_mcp.events as _events_mod

        started_at = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        stale_state = _state(
            LifecycleState.SLEEP.value,
            {
                "last_completed_index": 1,
                "attempt": 1,
                "last_error": "deferred:step=OPTIMIZE_HIPPO:chunk_idx=0",
                "started_at": started_at,
            },
        )
        monkeypatch.setattr(
            "iai_mcp.lifecycle_state.load_state",
            lambda path=None: stale_state,
        )
        # Ledger already has one event for this cycle (from before the restart).
        prior_ledger = [
            {"data": {"sleep_cycle_started_at": started_at}, "kind": "daemon_sleep_cycle_stale"}
        ]
        monkeypatch.setattr(
            _events_mod, "query_events", lambda store, kind=None, limit=500: prior_ledger,
        )
        emitted: list = []
        monkeypatch.setattr(
            daemon,
            "write_event",
            lambda store, kind, data, **kw: emitted.append((kind, data, kw)) or "id",
        )

        async def _probe_ok(_sock, _t):
            return True

        # Simulate a post-restart tick: in-process dedup was cleared by the
        # fixture's monkeypatch.setattr(daemon, "_last_sleep_stale_started_at", "").
        daemon._watchdog_tick(
            object(),
            env.sock_path,
            env.log_path,
            0,
            probe_fn=_probe_ok,
            pressure_fn=lambda: 1,
            rss_fn=lambda: 300 * 1024 * 1024,
        )

        stale_emits = [e for e in emitted if e[0] == "daemon_sleep_cycle_stale"]
        assert stale_emits == [], "durable dedup must suppress re-emit on restart"

    def test_durable_dedup_primes_inprocess_fastpath(self, env, monkeypatch):
        """After ledger-dedup suppresses the first tick, subsequent ticks must
        NOT re-hit the ledger (in-process fast-path should be primed).
        """
        import iai_mcp.events as _events_mod

        started_at = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        stale_state = _state(
            LifecycleState.SLEEP.value,
            {
                "last_completed_index": 1,
                "attempt": 0,
                "last_error": None,
                "started_at": started_at,
            },
        )
        monkeypatch.setattr(
            "iai_mcp.lifecycle_state.load_state",
            lambda path=None: stale_state,
        )
        prior_ledger = [
            {"data": {"sleep_cycle_started_at": started_at}, "kind": "daemon_sleep_cycle_stale"}
        ]
        ledger_reads: list = []

        def _counting_query(store, kind=None, limit=500):
            ledger_reads.append(1)
            return prior_ledger

        monkeypatch.setattr(_events_mod, "query_events", _counting_query)
        monkeypatch.setattr(
            daemon, "write_event", lambda *a, **kw: "id",
        )

        async def _probe_ok(_sock, _t):
            return True

        for _ in range(3):
            daemon._watchdog_tick(
                object(),
                env.sock_path,
                env.log_path,
                0,
                probe_fn=_probe_ok,
                pressure_fn=lambda: 1,
                rss_fn=lambda: 300 * 1024 * 1024,
            )

        # Ledger should be read exactly once (first tick); ticks 2 and 3 use
        # the in-process fast-path and skip the DB hit.
        assert len(ledger_reads) == 1, (
            f"expected 1 ledger read, got {len(ledger_reads)}"
        )

    def test_emit_failure_does_not_crash_tick(self, env, monkeypatch):
        stale_state = _state(
            LifecycleState.SLEEP.value,
            _progress(datetime.now(timezone.utc) - timedelta(days=10)),
        )
        monkeypatch.setattr(
            "iai_mcp.lifecycle_state.load_state",
            lambda path=None: stale_state,
        )

        def _boom(*a, **kw):
            raise RuntimeError("ledger down")

        monkeypatch.setattr(daemon, "write_event", _boom)

        async def _probe_ok(_sock, _t):
            return True

        # MUST NOT raise.
        daemon._watchdog_tick(
            object(),
            env.sock_path,
            env.log_path,
            0,
            probe_fn=_probe_ok,
            pressure_fn=lambda: 1,
            rss_fn=lambda: 300 * 1024 * 1024,
        )

    def test_corrupted_state_does_not_crash_tick(self, env, monkeypatch):
        def _bad_load(path=None):
            raise RuntimeError("state file decayed")

        monkeypatch.setattr(
            "iai_mcp.lifecycle_state.load_state", _bad_load,
        )

        async def _probe_ok(_sock, _t):
            return True

        # MUST NOT raise.
        daemon._watchdog_tick(
            object(),
            env.sock_path,
            env.log_path,
            0,
            probe_fn=_probe_ok,
            pressure_fn=lambda: 1,
            rss_fn=lambda: 300 * 1024 * 1024,
        )

    def test_kill_path_still_fires_when_socket_down(self, env, monkeypatch):
        """Staleness check must not perturb the existing kill path."""
        stale_state = _state(
            LifecycleState.SLEEP.value,
            _progress(datetime.now(timezone.utc) - timedelta(days=10)),
        )
        monkeypatch.setattr(
            "iai_mcp.lifecycle_state.load_state",
            lambda path=None: stale_state,
        )
        kill_calls: list = []
        monkeypatch.setattr(
            daemon.os,
            "kill",
            lambda p, s: kill_calls.append((p, s)),
        )
        monkeypatch.setattr(
            daemon, "write_event", lambda *a, **kw: "id",
        )

        async def _probe_dead(_sock, _t):
            return False

        consec = 0
        for _ in range(3):  # DEBOUNCE_N
            _interval, consec = daemon._watchdog_tick(
                object(),
                env.sock_path,
                env.log_path,
                consec,
                probe_fn=_probe_dead,
                pressure_fn=lambda: 1,
                rss_fn=lambda: 300 * 1024 * 1024,
            )
        assert kill_calls == [(os.getpid(), signal.SIGKILL)]


# ---- regression: package-vs-submodule attribute trap (PLAN-CHECK HIGH) ----


def test_first_tick_on_stale_state_emits_without_attr_init(
    tmp_path, monkeypatch, caplog,
):
    """First tick on a stale state must emit cleanly even when the dedup
    state attribute has NEVER been written to the package object.

    Pins the call-site's `getattr(_pkg(), "_last_sleep_stale_started_at", "")`
    default-empty contract against the package-vs-submodule attribute trap:
    the new module-scope `_last_sleep_stale_started_at: str = ""` lives in
    `_watchdog.py`, but `_pkg()` returns the `iai_mcp.daemon` PACKAGE — and
    the variable is not visible on the package object until the first
    `setattr(_pkg(), ..., started_at)` call. Without `getattr(..., "")`,
    this FIRST tick would AttributeError, the outer try/except would swallow
    it at DEBUG level, and the alert would silently never emit in production.

    Deliberately does NOT pre-`setattr(daemon, "_last_sleep_stale_started_at", "")`.
    """
    # Set up the watchdog environment minus the dedup-state reset.
    log_path = tmp_path / ".daemon-watchdog.log"
    fd = os.open(
        str(log_path),
        os.O_WRONLY | os.O_APPEND | os.O_CREAT,
        0o600,
    )
    monkeypatch.setattr(daemon, "_WATCHDOG_LOG_FD", fd)
    monkeypatch.setattr(
        daemon, "_daemon_started_monotonic", time.monotonic() - 1200.0,
    )
    monkeypatch.setattr(daemon.os, "kill", lambda p, s: None)

    # CRITICAL: do NOT pre-set _last_sleep_stale_started_at. Cooperate with
    # other tests in the suite by snapshot+restore around this case so any
    # later test's fixture starts from the same baseline. The fixture used
    # by TestWatchdogTickEmits already resets to "", so it self-heals if it
    # runs after this one.
    pkg = daemon
    sentinel = object()
    had_attr = hasattr(pkg, "_last_sleep_stale_started_at")
    prev = getattr(pkg, "_last_sleep_stale_started_at", sentinel)
    if had_attr:
        try:
            delattr(pkg, "_last_sleep_stale_started_at")
        except AttributeError:
            pass

    try:
        stale_state = _state(
            LifecycleState.SLEEP.value,
            _progress(datetime.now(timezone.utc) - timedelta(days=10)),
            crisis_mode=True,
        )
        monkeypatch.setattr(
            "iai_mcp.lifecycle_state.load_state",
            lambda path=None: stale_state,
        )
        emitted: list = []
        monkeypatch.setattr(
            daemon,
            "write_event",
            lambda store, kind, data, **kw: emitted.append((kind, data, kw)) or "id",
        )

        async def _probe_ok(_sock, _t):
            return True

        with caplog.at_level(logging.WARNING, logger="iai_mcp.daemon._watchdog"):
            # SHOULD NOT raise, SHOULD NOT warn-log, SHOULD emit exactly once.
            daemon._watchdog_tick(
                object(),
                str(tmp_path / ".daemon.sock"),
                log_path,
                0,
                probe_fn=_probe_ok,
                pressure_fn=lambda: 1,
                rss_fn=lambda: 300 * 1024 * 1024,
            )

        stale_emits = [e for e in emitted if e[0] == "daemon_sleep_cycle_stale"]
        assert len(stale_emits) == 1
        # No AttributeError warning logged at WARNING+ level.
        attr_err_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and "AttributeError" in r.getMessage()
        ]
        assert attr_err_warnings == []
    finally:
        # Restore the prior attribute state so subsequent tests are not
        # affected by this test's deletion.
        if had_attr and prev is not sentinel:
            setattr(pkg, "_last_sleep_stale_started_at", prev)
        try:
            os.close(fd)
        except OSError:
            pass
