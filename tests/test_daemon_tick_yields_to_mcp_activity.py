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


def test_mcp_recent_activity_probe():
    from iai_mcp import daemon as daemon_mod

    recent = types.SimpleNamespace(last_activity_ts=time.monotonic() - 2.0)
    stale = types.SimpleNamespace(last_activity_ts=time.monotonic() - 120.0)

    assert daemon_mod._mcp_recent_activity(recent, window_sec=30.0) is True
    assert daemon_mod._mcp_recent_activity(stale, window_sec=30.0) is False
    assert daemon_mod._mcp_recent_activity(None, window_sec=30.0) is False


def test_capture_queue_retry_waits_for_quiet_socket(tick_env, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store, _ = tick_env
    writes: list = []
    handled: list = []

    class _FakeCaptureQueue:
        calls = 0

        def ingest_pending(self, handler):
            self.calls += 1
            handler({"literal_surface": "queued"})
            return 1

        def pending_count(self):
            return 0

    monkeypatch.setattr(daemon_mod, "write_event", lambda *a, **kw: writes.append((a, kw)))
    monkeypatch.setattr(daemon_mod, "save_state", lambda state: None)

    state = {daemon_mod.CAPTURE_QUEUE_RETRY_PENDING_KEY: True}
    queue = _FakeCaptureQueue()
    recent_socket = types.SimpleNamespace(last_activity_ts=time.monotonic())

    asyncio.run(
        daemon_mod._retry_capture_queue_drain_if_due(
            store,
            state,
            capture_queue=queue,
            capture_handler=handled.append,
            mcp_socket=recent_socket,
        )
    )

    assert queue.calls == 0
    assert handled == []
    assert state[daemon_mod.CAPTURE_QUEUE_RETRY_PENDING_KEY] is True
    assert writes == []


def test_capture_queue_retry_drains_when_socket_is_quiet(tick_env, monkeypatch):
    from iai_mcp import daemon as daemon_mod

    store, _ = tick_env
    writes: list = []
    handled: list = []

    class _FakeCaptureQueue:
        calls = 0

        def ingest_pending(self, handler):
            self.calls += 1
            handler({"literal_surface": "queued"})
            return 1

        def pending_count(self):
            return 0

    monkeypatch.setattr(daemon_mod, "write_event", lambda *a, **kw: writes.append((a, kw)))
    monkeypatch.setattr(daemon_mod, "save_state", lambda state: None)

    state = {daemon_mod.CAPTURE_QUEUE_RETRY_PENDING_KEY: True}
    queue = _FakeCaptureQueue()
    stale_socket = types.SimpleNamespace(last_activity_ts=time.monotonic() - 120.0)

    asyncio.run(
        daemon_mod._retry_capture_queue_drain_if_due(
            store,
            state,
            capture_queue=queue,
            capture_handler=handled.append,
            mcp_socket=stale_socket,
        )
    )

    assert queue.calls == 1
    assert handled == [{"literal_surface": "queued"}]
    assert state[daemon_mod.CAPTURE_QUEUE_RETRY_PENDING_KEY] is False
    assert state["_capture_queue_pending_count"] == 0
    assert writes
    assert writes[0][0][1] == "capture_queue_drained"
    assert writes[0][0][2]["phase"] == "tick_retry"
