from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _kr
    fake: dict = {}
    monkeypatch.setattr(_kr, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(_kr, "set_password", lambda s, u, p: fake.__setitem__((s, u), p))
    monkeypatch.setattr(_kr, "delete_password", lambda s, u: fake.pop((s, u), None))
    yield fake


@pytest.fixture
def daemon_store(tmp_path, monkeypatch):
    from iai_mcp import daemon_state
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import MemoryRecord

    state_path = tmp_path / ".daemon-state.json"
    monkeypatch.setattr(daemon_state, "STATE_PATH", state_path)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")

    store = MemoryStore()
    rec = MemoryRecord(
        id=uuid4(), tier="semantic", literal_surface="seed",
        aaak_index="", embedding=[0.0] * store.embed_dim,
        community_id=None, centrality=0.0, detail_level=1,
        pinned=False, stability=0.0, difficulty=0.0,
        last_reviewed=None, never_decay=False, never_merge=False,
        provenance=[], created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc), tags=[], language="en",
    )
    store.insert(rec)

    yield store, state_path, tmp_path


def test_a_tick_body_never_calls_consolidation(daemon_store, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store, state_path, tmp_path = daemon_store

    rem_calls: list = []
    heavy_calls: list = []

    monkeypatch.setattr(
        daemon_mod, "run_rem_cycle",
        AsyncMock(side_effect=lambda *a, **kw: rem_calls.append(1) or {}),
    )

    import iai_mcp.sleep as sleep_mod
    monkeypatch.setattr(
        sleep_mod, "run_heavy_consolidation",
        lambda *a, **kw: heavy_calls.append(1) or {},
    )

    monkeypatch.setattr(daemon_mod, "should_relearn", lambda last, now: False)

    for extra_state in [
        {"force_rem_request": {"pending": True, "ts": "2026-01-01T00:00:00+00:00"}},
        {"user_sleep_request": {"pending": True, "ts": "2026-01-01T00:00:00+00:00"}},
        {},
    ]:
        state = {"fsm_state": "WAKE", **extra_state}
        asyncio.run(daemon_mod._tick_body(store, state))

    assert rem_calls == [], (
        f"_tick_body called run_rem_cycle {len(rem_calls)} times; expected 0"
    )
    assert heavy_calls == [], (
        f"_tick_body called run_heavy_consolidation {len(heavy_calls)} times; expected 0"
    )


def test_b_no_fsm_drift_across_lifecycle_sequence(tmp_path):
    from iai_mcp.fsm_reconcile import _CANONICAL_TO_LEGACY, reconcile_fsm_state
    from iai_mcp.lifecycle_state import LifecycleState

    canon_path = tmp_path / "lifecycle_state.json"
    legacy_path = tmp_path / ".daemon-state.json"

    def _set_canonical(state: LifecycleState):
        canon_path.write_text(json.dumps({"current_state": state.value}))

    def _set_legacy(fsm_state: str):
        legacy_path.write_text(json.dumps({"fsm_state": fsm_state}))

    for canonical_state in [
        LifecycleState.WAKE,
        LifecycleState.DROWSY,
        LifecycleState.SLEEP,
        LifecycleState.WAKE,
    ]:
        _set_canonical(canonical_state)
        _set_legacy("WRONG_STATE")

        report = reconcile_fsm_state(
            canonical_path=canon_path,
            legacy_path=legacy_path,
            auto_correct=True,
        )

        assert report["corrected"] is True, (
            f"expected corrected=True (drift was present): {report}"
        )

        expected_legacy = _CANONICAL_TO_LEGACY[canonical_state.value]
        actual_legacy = json.loads(legacy_path.read_text()).get("fsm_state")
        assert actual_legacy == expected_legacy, (
            f"legacy={actual_legacy!r}, expected={expected_legacy!r} "
            f"for canonical={canonical_state.value}"
        )

        report2 = reconcile_fsm_state(
            canonical_path=canon_path,
            legacy_path=legacy_path,
            auto_correct=False,
        )
        assert report2["drift"] is False, (
            f"post-correct drift still True: {report2}"
        )


def test_c_tick_body_does_not_mutate_canonical_fsm(daemon_store, monkeypatch, tmp_path):
    from iai_mcp import daemon as daemon_mod
    from iai_mcp.lifecycle_state import LifecycleState

    store, state_path, _ = daemon_store

    canon_path = tmp_path / "lifecycle_state.json"
    from iai_mcp.lifecycle_state import default_state as _ds
    _initial = dict(_ds())
    _initial["current_state"] = LifecycleState.WAKE.value
    canon_path.write_text(json.dumps(_initial))

    monkeypatch.setattr(daemon_mod, "should_relearn", lambda last, now: False)

    import iai_mcp.lifecycle_state as ls_mod
    monkeypatch.setattr(ls_mod, "LIFECYCLE_STATE_PATH", canon_path)

    state = {"fsm_state": "WAKE"}
    asyncio.run(daemon_mod._tick_body(store, state))

    after = json.loads(canon_path.read_text())
    assert after["current_state"] == LifecycleState.WAKE.value, (
        f"_tick_body mutated canonical lifecycle_state.json: {after}"
    )


def test_d_force_sleep_reaches_canonical_via_drowsy(tmp_path):
    from iai_mcp.lifecycle import LifecycleEvent, LifecycleState, compute_transition

    FORCE = LifecycleEvent.FORCE_SLEEP

    target = compute_transition(LifecycleState.WAKE, FORCE)
    assert target is LifecycleState.DROWSY, (
        f"Expected WAKE + FORCE_SLEEP -> DROWSY, got {target}"
    )

    target2 = compute_transition(LifecycleState.DROWSY, FORCE)
    assert target2 is LifecycleState.SLEEP, (
        f"Expected DROWSY + FORCE_SLEEP -> SLEEP, got {target2}"
    )

    target3 = compute_transition(LifecycleState.SLEEP, FORCE)
    assert target3 is None, (
        f"Expected SLEEP + FORCE_SLEEP -> None (no-op), got {target3}"
    )

    target4 = compute_transition(LifecycleState.HIBERNATION, FORCE)
    assert target4 is None, (
        f"Expected HIBERNATION + FORCE_SLEEP -> None, got {target4}"
    )

    s = LifecycleState.WAKE
    s = compute_transition(s, FORCE)
    assert s is LifecycleState.DROWSY
    s = compute_transition(s, FORCE)
    assert s is LifecycleState.SLEEP


def test_e_no_double_drain_of_covered_outputs(tmp_path, monkeypatch):
    import iai_mcp.daemon as daemon_mod

    import inspect
    source = inspect.getsource(daemon_mod)

    wake_hook_start = source.find("# --- WAKE hook (UNDER LOCK_EX, BEFORE downgrade) ---")
    downgrade_marker = source.find("# Downgrade EX → SH after the consolidation window.")
    assert wake_hook_start > 0, "Wake hook comment not found in daemon source"
    assert downgrade_marker > 0, "Downgrade marker not found in daemon source"

    wake_hook_block = source[wake_hook_start:downgrade_marker]

    assert "drain_deferred_captures" not in wake_hook_block, (
        "drain_deferred_captures is in the wake hook block — double-invocation risk"
    )
    assert "flush_event_buffer" not in wake_hook_block, (
        "flush_event_buffer is in the wake hook block — double-invocation risk"
    )
    assert "force_rebuild=False" in wake_hook_block, (
        "sleep wake-hook must not force a runtime graph rebuild after consolidation"
    )

    for marker in [
        "_write_session_start_cache",
        "write_processed_salience_top_n",
        "drain_active_live_captures",
        "flush_deferred_provenance",
    ]:
        assert marker in wake_hook_block, (
            f"Wake hook block missing R-output: {marker}"
        )


def test_f_foraging_gated_during_sleep(daemon_store, monkeypatch, tmp_path):
    from iai_mcp import daemon as daemon_mod
    from iai_mcp.lifecycle_state import LifecycleState
    from unittest.mock import patch

    store, state_path, _ = daemon_store

    canon_path = tmp_path / "lifecycle_state.json"

    monkeypatch.setattr(daemon_mod, "should_relearn", lambda last, now: False)

    forage_calls: list = []

    def fake_forage(store, max_edges=3):
        forage_calls.append(1)
        return 0

    def _write_ls(state: LifecycleState) -> None:
        from iai_mcp.lifecycle_state import default_state
        rec = dict(default_state())
        rec["current_state"] = state.value
        canon_path.write_text(json.dumps(rec))

    _write_ls(LifecycleState.SLEEP)

    state_sleep = {
        "fsm_state": "SLEEP",
        "_last_forage_ts": "",
    }
    with patch("iai_mcp.lifecycle_state.LIFECYCLE_STATE_PATH", canon_path), \
         patch("iai_mcp.foraging.forage_for_connections", fake_forage):
        asyncio.run(daemon_mod._tick_body(store, state_sleep))

    assert forage_calls == [], (
        f"foraging was called {len(forage_calls)} time(s) during SLEEP; expected 0"
    )

    _write_ls(LifecycleState.WAKE)
    forage_calls.clear()

    state_wake = {
        "fsm_state": "WAKE",
        "_last_forage_ts": "",
    }
    with patch("iai_mcp.lifecycle_state.LIFECYCLE_STATE_PATH", canon_path), \
         patch("iai_mcp.foraging.forage_for_connections", fake_forage):
        asyncio.run(daemon_mod._tick_body(store, state_wake))

    assert forage_calls == [1], (
        f"foraging was called {len(forage_calls)} time(s) during WAKE; expected 1"
    )
