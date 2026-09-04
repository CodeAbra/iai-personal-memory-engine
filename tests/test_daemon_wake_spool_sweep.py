from __future__ import annotations

import platform
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="daemon module is POSIX-only on this project",
)


def _states():
    from iai_mcp.lifecycle_state import LifecycleState
    return LifecycleState


def _predicate(**overrides):
    from iai_mcp.daemon import _should_run_wake_spool_sweep
    L = _states()
    kwargs = dict(
        current=L.WAKE,
        now_mono=100.0,
        last_mono=0.0,
        inflight=False,
        interval_sec=30.0,
        signature=(3, 10, 300),
        last_signature=(2, 5, 200),
    )
    kwargs.update(overrides)
    current = kwargs.pop("current")
    return _should_run_wake_spool_sweep(current, **kwargs)


def test_sweep_runs_in_wake_when_spool_changed():
    assert _predicate() is True


def test_sweep_runs_in_drowsy_and_hibernation_but_never_in_sleep():
    L = _states()
    assert _predicate(current=L.DROWSY) is True
    assert _predicate(current=L.HIBERNATION) is True
    assert _predicate(current=L.SLEEP) is False


def test_sweep_skips_while_inflight_or_before_interval():
    assert _predicate(inflight=True) is False
    assert _predicate(now_mono=20.0, last_mono=0.0, interval_sec=30.0) is False
    assert _predicate(now_mono=30.0, last_mono=0.0, interval_sec=30.0) is True


def test_sweep_skips_unchanged_or_missing_spool():
    assert _predicate(signature=(3, 10, 300), last_signature=(3, 10, 300)) is False
    assert _predicate(signature=None) is False


def test_sweep_disabled_by_zero_interval():
    assert _predicate(interval_sec=0.0) is False


def test_spool_signature_tracks_appends_and_absence(tmp_path: Path):
    from iai_mcp.daemon import _spool_signature

    assert _spool_signature(tmp_path / "missing") is None
    spool = tmp_path / "spool"
    spool.mkdir()
    assert _spool_signature(spool) == (0, 0, 0)

    live = spool / "s1.live.jsonl"
    live.write_text('{"version": 1}\n')
    first = _spool_signature(spool)
    assert first is not None and first[0] == 1 and first[2] == live.stat().st_size

    time.sleep(0.01)
    with live.open("a") as fh:
        fh.write('{"role": "user", "text": "hello there"}\n')
    second = _spool_signature(spool)
    assert second != first
    assert second[0] == 1 and second[2] > first[2]

    (spool / "s1.live-1-2.jsonl").write_text('{"version": 1}\n')
    third = _spool_signature(spool)
    assert third[0] == 2 and third != second


def test_window_open_gates_are_evaluated_before_the_spool_is_sampled():
    from iai_mcp.daemon import _wake_sweep_window_open

    L = _states()
    base = dict(now_mono=100.0, last_mono=0.0, inflight=False, interval_sec=30.0)
    assert _wake_sweep_window_open(L.WAKE, **base) is True
    assert _wake_sweep_window_open(L.SLEEP, **base) is False
    assert _wake_sweep_window_open(L.WAKE, **{**base, "inflight": True}) is False
    assert _wake_sweep_window_open(L.WAKE, **{**base, "now_mono": 10.0}) is False
    assert _wake_sweep_window_open(L.WAKE, **{**base, "interval_sec": 0.0}) is False


def test_run_wake_spool_sweep_failure_writes_event_and_reports_error():
    from iai_mcp.daemon import _run_wake_spool_sweep

    events: list[tuple] = []

    def write_event(store, kind, data, severity="info"):
        events.append((kind, data, severity))

    def failing_drain(store):
        raise RuntimeError("drain blew up")

    result = _run_wake_spool_sweep(
        SimpleNamespace(), drain_fn=failing_drain, write_event_fn=write_event
    )
    assert result == {"error": "drain blew up"}
    assert events and events[0][0] == "deferred_drain_failed"
    assert events[0][1].get("phase") == "wake"


def test_wake_sweep_report_carries_status_counts_and_error():
    from iai_mcp.daemon import _wake_sweep_report

    ok = _wake_sweep_report(
        "ok",
        {"files_drained": 2, "events_inserted": 1,
         "live_files_drained": 1, "live_events_inserted": 3},
    )
    assert ok["status"] == "ok"
    assert ok["files_drained"] == 3 and ok["events_inserted"] == 4, (
        "both promotion lanes must count — a live-only sweep is not zero work"
    )
    assert ok["at"] and "error" not in ok

    busy = _wake_sweep_report("gate_busy")
    assert busy["status"] == "gate_busy" and "files_drained" not in busy

    err = _wake_sweep_report("error", error="x" * 500)
    assert err["status"] == "error" and len(err["error"]) == 200


def test_run_wake_spool_sweep_returns_drain_result_without_event_churn():
    from iai_mcp.daemon import _run_wake_spool_sweep

    events: list[tuple] = []

    def write_event(store, kind, data, severity="info"):
        events.append((kind, data, severity))

    payload = {"files_drained": 1, "events_inserted": 2, "live_events_inserted": 3}
    result = _run_wake_spool_sweep(
        SimpleNamespace(), drain_fn=lambda store: payload, write_event_fn=write_event
    )
    assert result == payload
    assert events == []


def test_sweep_inserted_records_counts_both_lanes():
    from iai_mcp.daemon import _sweep_inserted_records

    assert _sweep_inserted_records({}) == 0
    assert _sweep_inserted_records({"events_inserted": 2}) == 2
    assert _sweep_inserted_records({"live_events_inserted": 3}) == 3
    assert _sweep_inserted_records(
        {"events_inserted": 2, "live_events_inserted": 3}
    ) == 5


def test_precache_refresh_only_after_inserts_and_debounce():
    from iai_mcp.daemon import _should_refresh_precache_after_sweep

    inserted = {"events_inserted": 0, "live_events_inserted": 1}
    assert _should_refresh_precache_after_sweep(
        inserted, now_mono=1000.0, last_mono=0.0, min_interval_sec=300.0
    ) is True
    assert _should_refresh_precache_after_sweep(
        inserted, now_mono=100.0, last_mono=0.0, min_interval_sec=300.0
    ) is False
    assert _should_refresh_precache_after_sweep(
        {"events_inserted": 0}, now_mono=1000.0, last_mono=0.0, min_interval_sec=300.0
    ) is False
    assert _should_refresh_precache_after_sweep(
        {}, now_mono=1000.0, last_mono=0.0, min_interval_sec=300.0
    ) is False


def test_env_knobs_parse_with_defaults(monkeypatch: pytest.MonkeyPatch):
    from iai_mcp.daemon import (
        _precache_refresh_min_sec,
        _wake_spool_sweep_interval_sec,
    )

    monkeypatch.delenv("IAI_MCP_WAKE_SPOOL_SWEEP_SEC", raising=False)
    monkeypatch.delenv("IAI_MCP_PRECACHE_REFRESH_MIN_SEC", raising=False)
    assert _wake_spool_sweep_interval_sec() == 30.0
    assert _precache_refresh_min_sec() == 300.0

    monkeypatch.setenv("IAI_MCP_WAKE_SPOOL_SWEEP_SEC", "0")
    monkeypatch.setenv("IAI_MCP_PRECACHE_REFRESH_MIN_SEC", "60")
    assert _wake_spool_sweep_interval_sec() == 0.0
    assert _precache_refresh_min_sec() == 60.0

    monkeypatch.setenv("IAI_MCP_WAKE_SPOOL_SWEEP_SEC", "not-a-number")
    monkeypatch.setenv("IAI_MCP_PRECACHE_REFRESH_MIN_SEC", "-5")
    assert _wake_spool_sweep_interval_sec() == 30.0
    assert _precache_refresh_min_sec() == 0.0


def test_daemon_status_carries_wake_spool_sweep_report():
    import asyncio

    from iai_mcp.concurrency import _dispatch_socket_request

    report = {"at": "2026-08-17T16:40:00+00:00", "files_drained": 1, "events_inserted": 2}
    state = {"fsm_state": "WAKE", "wake_spool_sweep": report}
    resp = asyncio.run(_dispatch_socket_request({"type": "status"}, SimpleNamespace(), state))
    assert resp["ok"] is True
    assert resp["wake_spool_sweep"] == report

    state["wake_spool_sweep"] = "not-a-dict"
    resp = asyncio.run(_dispatch_socket_request({"type": "status"}, SimpleNamespace(), state))
    assert resp["wake_spool_sweep"] is None
