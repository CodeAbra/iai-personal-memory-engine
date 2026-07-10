from __future__ import annotations

import asyncio



def test_continuous_audit_never_calls_optimize_hippo_storage(monkeypatch):
    from iai_mcp import identity_audit

    optimize_calls: list = []

    def fake_optimize(store):
        optimize_calls.append(store)
        return {}

    try:
        from iai_mcp import maintenance as _maint
        monkeypatch.setattr(_maint, "optimize_hippo_storage", fake_optimize)
    except (ImportError, AttributeError):
        pass

    if hasattr(identity_audit, "optimize_hippo_storage"):
        monkeypatch.setattr(identity_audit, "optimize_hippo_storage", fake_optimize)

    s5_calls: list = []
    sigma_calls: list = []

    def fake_s5(store, window):
        s5_calls.append(store)

    def fake_sigma(store):
        sigma_calls.append(store)

    monkeypatch.setattr(identity_audit, "detect_drift_anomaly", fake_s5)
    monkeypatch.setattr(identity_audit, "compute_and_emit", fake_sigma)

    async def runner():
        shutdown = asyncio.Event()
        task = asyncio.create_task(
            identity_audit.continuous_audit(
                object(), shutdown, interval_sec=0.01,
            )
        )
        await asyncio.sleep(0.06)
        shutdown.set()
        await task

    asyncio.run(runner())

    assert optimize_calls == [], (
        f"optimize_hippo_storage was called {len(optimize_calls)} time(s) "
        "from the audit loop — expected 0"
    )


def test_continuous_audit_read_stages_still_run(monkeypatch):
    from iai_mcp import identity_audit

    s5_calls: list = []
    sigma_calls: list = []

    def fake_s5(store, window):
        s5_calls.append(store)

    def fake_sigma(store):
        sigma_calls.append(store)

    monkeypatch.setattr(identity_audit, "detect_drift_anomaly", fake_s5)
    monkeypatch.setattr(identity_audit, "compute_and_emit", fake_sigma)

    async def runner():
        shutdown = asyncio.Event()
        task = asyncio.create_task(
            identity_audit.continuous_audit(
                object(), shutdown, interval_sec=0.01,
            )
        )
        await asyncio.sleep(0.05)
        shutdown.set()
        await task

    asyncio.run(runner())

    assert len(s5_calls) >= 2, (
        f"detect_drift_anomaly was called {len(s5_calls)} time(s); expected >= 2"
    )
    assert len(sigma_calls) >= 2, (
        f"compute_and_emit was called {len(sigma_calls)} time(s); expected >= 2"
    )


def test_continuous_audit_module_has_no_optimize_import():
    from iai_mcp import identity_audit

    assert not hasattr(identity_audit, "optimize_hippo_storage"), (
        "identity_audit still exposes optimize_hippo_storage; "
        "the Stage-3 compaction removal is incomplete"
    )
    assert not hasattr(identity_audit, "_last_optimize_completed_at"), (
        "identity_audit still carries _last_optimize_completed_at; "
        "the Stage-3 compaction removal is incomplete"
    )
