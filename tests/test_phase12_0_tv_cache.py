from __future__ import annotations

import time
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.retrieve import (
    _tv_cache_dirty,
    build_temporal_validity_maps,
)
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

def _make_record(text: str = "test record") -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
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
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )

def test_cache_hit_skips_scan(tmp_path):
    store = MemoryStore(path=tmp_path)
    store.insert(_make_record("alice first"))

    result1 = build_temporal_validity_maps(store)
    assert result1 is not None

    t0 = time.perf_counter()
    result2 = build_temporal_validity_maps(store)
    cache_hit_ms = (time.perf_counter() - t0) * 1000

    assert result2 is result1
    assert cache_hit_ms < 1.0

def test_cache_invalidated_on_insert(tmp_path):
    store = MemoryStore(path=tmp_path)
    store.insert(_make_record("bob first"))

    build_temporal_validity_maps(store)
    assert _tv_cache_dirty.get(store) is False

    store.insert(_make_record("bob second"))
    assert _tv_cache_dirty.get(store) is True

    result = build_temporal_validity_maps(store)
    assert result is not None
    assert _tv_cache_dirty.get(store) is False

def test_cache_invalidated_on_contradict(tmp_path):
    store = MemoryStore(path=tmp_path)
    rec = _make_record("alice original fact")
    store.insert(rec)

    build_temporal_validity_maps(store)
    assert _tv_cache_dirty.get(store) is False

    from iai_mcp.retrieve import contradict
    contradict(store, rec.id, "alice corrected fact", [0.2] * EMBED_DIM)

    assert _tv_cache_dirty.get(store) is True
    result = build_temporal_validity_maps(store)
    outgoing, _ = result
    assert len(outgoing) > 0

def test_per_store_isolation(tmp_path):
    store_a = MemoryStore(path=tmp_path / "a")
    store_b = MemoryStore(path=tmp_path / "b")
    store_a.insert(_make_record("alice store a"))
    store_b.insert(_make_record("bob store b"))

    result_a = build_temporal_validity_maps(store_a)
    result_b = build_temporal_validity_maps(store_b)

    store_a.insert(_make_record("alice store a second"))
    assert _tv_cache_dirty.get(store_a) is True
    assert _tv_cache_dirty.get(store_b) is False

    result_b_cached = build_temporal_validity_maps(store_b)
    assert result_b_cached is result_b

@pytest.mark.perf
def test_d_speed_bench_green(tmp_path):
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
        f"D-SPEED best-of-3 p95={min_p95:.1f}ms > {D_SPEED_P95_MS}ms"
    )
