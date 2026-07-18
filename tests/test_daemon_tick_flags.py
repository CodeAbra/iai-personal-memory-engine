from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def tick_env(tmp_path, monkeypatch):
    from iai_mcp import daemon_state
    from iai_mcp.store import MemoryStore

    state_path = tmp_path / ".daemon-state.json"

    monkeypatch.setattr(daemon_state, "STATE_PATH", state_path)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")

    store = MemoryStore()

    from iai_mcp.types import MemoryRecord
    from uuid import uuid4
    rec = MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface="seed record",
        aaak_index="",
        embedding=[0.0] * store.embed_dim,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )
    store.insert(rec)

    yield store, state_path, tmp_path


def test_scheduler_paused_emits_skip_event_and_returns(tick_env, monkeypatch):
    from iai_mcp import daemon as daemon_mod
    from iai_mcp.events import query_events

    store, state_path, tmp_path = tick_env

    state = {
        "fsm_state": "WAKE",
        "scheduler_paused": True,
    }

    rem_mock = AsyncMock(side_effect=AssertionError("run_rem_cycle must never be called"))
    monkeypatch.setattr(daemon_mod, "run_rem_cycle", rem_mock)

    asyncio.run(daemon_mod._tick_body(store, state))

    assert state.get("last_tick_skipped_reason") == "paused"
    events = query_events(store, kind="daemon_tick_skipped", limit=1)
    assert len(events) == 1
    assert events[0]["data"]["reason"] == "paused"
    assert state["fsm_state"] == "WAKE"


def test_tick_body_never_calls_run_rem_cycle(tick_env, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store, state_path, tmp_path = tick_env

    rem_calls: list = []
    rem_mock = AsyncMock(side_effect=lambda *a, **kw: rem_calls.append(a) or {})
    monkeypatch.setattr(daemon_mod, "run_rem_cycle", rem_mock)
    monkeypatch.setattr(daemon_mod, "should_relearn", lambda last, now: False)

    state = {
        "fsm_state": "WAKE",
        "quiet_window": None,
        "force_rem_request": {
            "ts": "2026-04-18T10:00:00+00:00",
            "pending": True,
        },
        "last_session_ts": datetime.now(timezone.utc).isoformat(),
    }
    asyncio.run(daemon_mod._tick_body(store, state))

    assert rem_calls == [], (
        f"_tick_body called run_rem_cycle {len(rem_calls)} time(s); expected 0 "
        f"(consolidation now routes through lifecycle_tick)"
    )


def test_tick_body_never_calls_run_rem_cycle_user_sleep(tick_env, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store, state_path, tmp_path = tick_env

    rem_calls: list = []
    rem_mock = AsyncMock(side_effect=lambda *a, **kw: rem_calls.append(a) or {})
    monkeypatch.setattr(daemon_mod, "run_rem_cycle", rem_mock)
    monkeypatch.setattr(daemon_mod, "should_relearn", lambda last, now: False)

    state = {
        "fsm_state": "WAKE",
        "quiet_window": None,
        "user_sleep_request": {
            "reason": "bedtime",
            "ts": "2026-04-18T23:00:00+00:00",
            "pending": True,
        },
        "last_session_ts": datetime.now(timezone.utc).isoformat(),
    }
    asyncio.run(daemon_mod._tick_body(store, state))

    assert rem_calls == [], (
        f"_tick_body called run_rem_cycle {len(rem_calls)} time(s); expected 0"
    )


def test_paused_skip_persists_to_disk(tick_env, monkeypatch):
    from iai_mcp import daemon as daemon_mod
    from iai_mcp.daemon_state import load_state, save_state

    store, state_path, tmp_path = tick_env

    # scheduler_paused reaches disk when it is SET (the pause handler owns
    # that key); the tick's own persist must record the skip without
    # erasing it.
    save_state({"fsm_state": "WAKE", "scheduler_paused": True})
    state = {
        "fsm_state": "WAKE",
        "scheduler_paused": True,
    }

    asyncio.run(daemon_mod._tick_body(store, state))

    loaded = load_state()
    assert loaded["last_tick_skipped_reason"] == "paused"
    assert loaded["scheduler_paused"] is True
    datetime.fromisoformat(loaded["last_tick_at"])


def test_tick_updates_last_tick_at(tick_env, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store, state_path, tmp_path = tick_env

    monkeypatch.setattr(daemon_mod, "should_relearn", lambda last, now: False)

    state = {"fsm_state": "WAKE"}
    asyncio.run(daemon_mod._tick_body(store, state))

    assert "last_tick_at" in state
    datetime.fromisoformat(state["last_tick_at"])
