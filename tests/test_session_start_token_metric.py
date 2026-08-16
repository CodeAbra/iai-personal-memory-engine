"""Percentile-math coverage for compute_session_start_tokens_p90.

Covers the rolling p90 arithmetic only (windowing, outlier resistance,
event-kind filtering, restart persistence). Not a session-start token
budget guard — that role lives in test_session_refresh_rpc.py, against
the real per-turn refresh RPC.
"""

from __future__ import annotations

import time

def _write_session_started(store, tokens: int, session_id: str = "s") -> None:
    from iai_mcp.events import write_event

    write_event(
        store,
        kind="session_started",
        data={
            "session_id": session_id,
            "session_state_hash": "deadbeef",
            "total_cached_tokens": int(tokens),
            "wake_depth": "standard",
            "timestamp": "2026-05-17T00:00:00+00:00",
        },
        severity="info",
        session_id=session_id,
    )

def test_p90_uniform_returns_uniform(tmp_path):
    from iai_mcp.cli import compute_session_start_tokens_p90
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    for _ in range(100):
        _write_session_started(store, 3000)

    result = compute_session_start_tokens_p90(store)
    assert result == {"p90": 3000, "n_samples": 100}, f"got {result}"

def test_p90_with_outlier_shifts(tmp_path):
    from iai_mcp.cli import compute_session_start_tokens_p90
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    for _ in range(99):
        _write_session_started(store, 3000)
    _write_session_started(store, 5000)

    result = compute_session_start_tokens_p90(store)
    assert result == {"p90": 3000, "n_samples": 100}, f"got {result}"

def test_p90_under_filled_window_reports_n_samples(tmp_path):
    from iai_mcp.cli import compute_session_start_tokens_p90
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    for _ in range(10):
        _write_session_started(store, 2500)

    result = compute_session_start_tokens_p90(store)
    assert result == {"p90": 2500, "n_samples": 10}, f"got {result}"

def test_p90_empty_returns_none(tmp_path):
    from iai_mcp.cli import compute_session_start_tokens_p90
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    result = compute_session_start_tokens_p90(store)
    assert result == {"p90": None, "n_samples": 0}, f"got {result}"

def test_p90_survives_restart(tmp_path):
    from iai_mcp.cli import compute_session_start_tokens_p90
    from iai_mcp.store import MemoryStore

    store1 = MemoryStore(path=tmp_path)
    for _ in range(100):
        _write_session_started(store1, 3000)
    first = compute_session_start_tokens_p90(store1)
    assert first == {"p90": 3000, "n_samples": 100}, f"first: {first}"
    del store1

    store2 = MemoryStore(path=tmp_path)
    second = compute_session_start_tokens_p90(store2)
    assert second == first, f"persistence failed: first={first} second={second}"

def test_p90_only_uses_session_started_kind(tmp_path):
    from iai_mcp.cli import compute_session_start_tokens_p90
    from iai_mcp.events import write_event
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    for _ in range(50):
        _write_session_started(store, 3000)
    for _ in range(25):
        write_event(store, kind="s4_contradiction", data={"total_cached_tokens": 99999})
    for _ in range(25):
        write_event(store, kind="migration_v3_to_v4", data={"total_cached_tokens": 99999})

    result = compute_session_start_tokens_p90(store)
    assert result == {"p90": 3000, "n_samples": 50}, f"got {result}"

def test_p90_takes_most_recent_100(tmp_path):
    from iai_mcp.cli import compute_session_start_tokens_p90
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    for _ in range(50):
        _write_session_started(store, 1000)
    time.sleep(0.05)
    for _ in range(100):
        _write_session_started(store, 4000)

    result = compute_session_start_tokens_p90(store)
    assert result == {"p90": 4000, "n_samples": 100}, f"got {result}"
