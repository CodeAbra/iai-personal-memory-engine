from __future__ import annotations

import asyncio

import pytest


def test_continuous_audit_invokes_both_underlying_calls(monkeypatch):
    from iai_mcp import identity_audit

    s5_calls: list = []
    sigma_calls: list = []

    def fake_s5(store, window):
        s5_calls.append((store, window))
        return []

    def fake_sigma(store):
        sigma_calls.append((store,))
        return {"phase": "healthy"}

    monkeypatch.setattr(identity_audit, "detect_drift_anomaly", fake_s5)
    monkeypatch.setattr(identity_audit, "compute_and_emit", fake_sigma)
    monkeypatch.setattr(identity_audit, "AUDIT_INTERVAL_SEC", 0.02)

    async def runner():
        shutdown = asyncio.Event()
        task = asyncio.create_task(
            identity_audit.continuous_audit(object(), shutdown)
        )
        await asyncio.sleep(0.05)
        shutdown.set()
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(runner())

    assert len(s5_calls) >= 1, "detect_drift_anomaly never called"
    assert len(sigma_calls) >= 1, "compute_and_emit never called"
    assert s5_calls[0][1] == 5


def test_audit_runs_even_when_paused(monkeypatch):
    from iai_mcp import identity_audit

    s5_calls: list = []

    def fake_s5(store, window):
        s5_calls.append(1)
        return []

    monkeypatch.setattr(identity_audit, "detect_drift_anomaly", fake_s5)
    monkeypatch.setattr(identity_audit, "compute_and_emit", lambda store: {})
    monkeypatch.setattr(identity_audit, "AUDIT_INTERVAL_SEC", 0.02)

    daemon_state = {
        "paused_until": "2099-01-01T00:00:00+00:00",
        "fsm_state": "WAKE",
    }
    assert "paused_until" in daemon_state

    async def runner():
        shutdown = asyncio.Event()
        task = asyncio.create_task(
            identity_audit.continuous_audit(object(), shutdown)
        )
        await asyncio.sleep(0.05)
        shutdown.set()
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(runner())

    assert len(s5_calls) >= 1, "audit did NOT fire while daemon was 'paused' (C6 violation)"


def test_audit_shuts_down_cleanly(monkeypatch):
    from iai_mcp import identity_audit

    monkeypatch.setattr(identity_audit, "detect_drift_anomaly", lambda s, w: [])
    monkeypatch.setattr(identity_audit, "compute_and_emit", lambda s: {})
    monkeypatch.setattr(identity_audit, "AUDIT_INTERVAL_SEC", 3600)

    async def runner():
        shutdown = asyncio.Event()
        task = asyncio.create_task(
            identity_audit.continuous_audit(object(), shutdown)
        )
        await asyncio.sleep(0.02)
        shutdown.set()
        await asyncio.wait_for(task, timeout=2.0)
        assert task.done()

    asyncio.run(runner())


def test_audit_defers_first_iteration_by_default(monkeypatch):
    from iai_mcp import identity_audit

    s5_calls: list = []

    def fake_s5(store, window):
        s5_calls.append(1)
        return []

    monkeypatch.setattr(identity_audit, "detect_drift_anomaly", fake_s5)
    monkeypatch.setattr(identity_audit, "compute_and_emit", lambda s: {})
    monkeypatch.delenv("IAI_MCP_AUDIT_FIRST_ITER_GRACE_SEC", raising=False)

    async def runner():
        shutdown = asyncio.Event()
        task = asyncio.create_task(
            identity_audit.continuous_audit(object(), shutdown, interval_sec=10.0)
        )
        await asyncio.sleep(0.03)
        shutdown.set()
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(runner())

    assert s5_calls == []


def test_first_iter_grace_default_is_capped(monkeypatch):
    from iai_mcp import identity_audit

    monkeypatch.delenv("IAI_MCP_AUDIT_FIRST_ITER_GRACE_SEC", raising=False)

    assert identity_audit._first_iter_grace_sec(3600.0) == 60.0
    assert identity_audit._first_iter_grace_sec(30.0) == 30.0


def test_first_iter_grace_env_override_wins(monkeypatch):
    from iai_mcp import identity_audit

    monkeypatch.setenv("IAI_MCP_AUDIT_FIRST_ITER_GRACE_SEC", "0.25")

    assert identity_audit._first_iter_grace_sec(3600.0) == 0.25


def test_audit_survives_s5_exception_and_emits_event(monkeypatch):
    from iai_mcp import identity_audit

    s5_calls: list = []
    emitted: list = []

    def flaky_s5(store, window):
        s5_calls.append(1)
        if len(s5_calls) == 1:
            raise RuntimeError("simulated s5 failure")
        return []

    def capture_event(store, kind, data, *, severity=None, **kwargs):
        emitted.append((kind, dict(data), severity))
        return None

    monkeypatch.setattr(identity_audit, "detect_drift_anomaly", flaky_s5)
    monkeypatch.setattr(identity_audit, "compute_and_emit", lambda s: {})
    monkeypatch.setattr(identity_audit, "write_event", capture_event)
    monkeypatch.setattr(identity_audit, "AUDIT_INTERVAL_SEC", 0.01)

    async def runner():
        shutdown = asyncio.Event()
        task = asyncio.create_task(
            identity_audit.continuous_audit(object(), shutdown)
        )
        await asyncio.sleep(0.25)
        shutdown.set()
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(runner())

    s5_err = [e for e in emitted if e[0] == "identity_audit_error" and e[1].get("stage") == "s5"]
    assert len(s5_err) >= 1, f"no s5 error event emitted; emitted={emitted}"
    assert "simulated s5 failure" in s5_err[0][1]["error"]
    assert len(s5_calls) >= 2, (
        f"audit did not continue after s5 exception; calls={len(s5_calls)}"
    )


def test_audit_fires_multiple_times_with_short_interval(monkeypatch):
    from iai_mcp import identity_audit

    s5_calls: list = []

    def fake_s5(store, window):
        s5_calls.append(1)
        return []

    monkeypatch.setattr(identity_audit, "detect_drift_anomaly", fake_s5)
    monkeypatch.setattr(identity_audit, "compute_and_emit", lambda s: {})
    monkeypatch.setattr(identity_audit, "AUDIT_INTERVAL_SEC", 0.03)

    async def runner():
        shutdown = asyncio.Event()
        task = asyncio.create_task(
            identity_audit.continuous_audit(object(), shutdown)
        )
        await asyncio.sleep(0.25)
        shutdown.set()
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(runner())
    assert len(s5_calls) >= 3, (
        f"expected >=3 ticks in 0.25s @ 0.03s interval; got {len(s5_calls)}"
    )
