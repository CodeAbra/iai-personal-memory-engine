from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.events import (
    flush_event_buffer,
    query_events,
    write_event,
)
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

def _make_record(store, text="alice test"):
    now = datetime.now(timezone.utc)
    rec = MemoryRecord(
        id=uuid4(), tier="episodic", literal_surface=text,
        aaak_index="", embedding=[0.1] * EMBED_DIM,
        community_id=None, centrality=0.0, detail_level=1,
        pinned=False, stability=0.0, difficulty=0.0,
        last_reviewed=None, never_decay=False, never_merge=False,
        provenance=[], created_at=now, updated_at=now,
        tags=[], language="en",
    )
    store.insert(rec)
    return rec

def test_buffered_write_does_not_persist_immediately(tmp_path):
    store = MemoryStore(path=tmp_path)
    _make_record(store)

    write_event(store, "test_kind", {"key": "val"}, severity="info", buffered=True)

    events = query_events(store, kind="test_kind", limit=10)
    assert len(events) == 0

def test_flush_persists_buffered_events(tmp_path):
    store = MemoryStore(path=tmp_path)
    _make_record(store)

    write_event(store, "test_kind", {"n": 1}, severity="info", buffered=True)
    write_event(store, "test_kind", {"n": 2}, severity="info", buffered=True)
    write_event(store, "test_kind", {"n": 3}, severity="info", buffered=True)

    count = flush_event_buffer(store)
    assert count == 3

    events = query_events(store, kind="test_kind", limit=10)
    assert len(events) == 3

def test_non_buffered_write_persists_immediately(tmp_path):
    store = MemoryStore(path=tmp_path)
    _make_record(store)

    write_event(store, "immediate_kind", {"x": 1}, severity="info", buffered=False)

    events = query_events(store, kind="immediate_kind", limit=10)
    assert len(events) == 1

def test_flush_empty_buffer_returns_zero(tmp_path):
    store = MemoryStore(path=tmp_path)
    _make_record(store)

    count = flush_event_buffer(store)
    assert count == 0

@pytest.mark.perf
def test_bench_d_speed_still_green(tmp_path):
    from bench.neural_map import run_neural_map_bench, D_SPEED_P95_MS

    from _perf_helpers import best_of_n, skip_if_loaded

    skip_if_loaded()

    counter = {"i": 0}

    def _one_p95() -> float:
        i = counter["i"]
        counter["i"] += 1
        run = run_neural_map_bench(n=100, iterations=10, store_path=tmp_path / f"run{i}")
        return float(run["latency_ms_p95"])

    min_p95 = best_of_n(_one_p95, n=3)
    assert min_p95 < D_SPEED_P95_MS, (
        f"D-SPEED best-of-3 p95={min_p95:.1f}ms >= {D_SPEED_P95_MS}ms"
    )
