from __future__ import annotations

import asyncio
import time
import types
from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest


@pytest.fixture
def tick_env(tmp_path, monkeypatch):
    from iai_mcp import daemon_state
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import MemoryRecord

    state_path = tmp_path / ".daemon-state.json"

    monkeypatch.setattr(daemon_state, "STATE_PATH", state_path)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")

    store = MemoryStore()
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

    yield store, tmp_path


def test_tick_body_accepts_mcp_socket_kwarg_without_crashing(tick_env, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store, _ = tick_env

    monkeypatch.setattr(daemon_mod, "should_relearn", lambda last, now: False)

    mcp_socket = types.SimpleNamespace(
        active_connections=0,
        last_activity_ts=time.monotonic() - 600.0,
    )

    state = {"fsm_state": "WAKE"}
    asyncio.run(daemon_mod._tick_body(store, state, mcp_socket=mcp_socket))

    assert "last_tick_at" in state


def test_tick_body_never_calls_run_rem_cycle_with_mcp_socket(tick_env, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store, _ = tick_env

    rem_calls: list = []
    monkeypatch.setattr(
        daemon_mod, "run_rem_cycle",
        AsyncMock(side_effect=lambda *a, **kw: rem_calls.append(a) or {}),
    )
    monkeypatch.setattr(daemon_mod, "should_relearn", lambda last, now: False)

    mcp_socket = types.SimpleNamespace(
        active_connections=1,
        last_activity_ts=time.monotonic() - 5.0,
    )

    state = {"fsm_state": "WAKE", "rem_cycle_count": 5}
    asyncio.run(daemon_mod._tick_body(store, state, mcp_socket=mcp_socket))

    assert rem_calls == [], (
        "_tick_body called run_rem_cycle; expected 0 calls (canonical path)"
    )


def test_tick_body_works_with_none_mcp_socket(tick_env, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store, _ = tick_env

    monkeypatch.setattr(daemon_mod, "should_relearn", lambda last, now: False)

    state = {"fsm_state": "WAKE"}
    asyncio.run(daemon_mod._tick_body(store, state, mcp_socket=None))

    assert "last_tick_at" in state
