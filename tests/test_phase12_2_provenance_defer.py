from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest


from iai_mcp.provenance_buffer import (
    _BUFFER_FILENAME,
    defer_provenance,
    flush_deferred_provenance,
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

def test_defer_writes_jsonl(tmp_path):
    store = MemoryStore(path=tmp_path)
    rec = _make_record(store, "bob record")

    defer_provenance(store, [(rec.id, "test cue", "session-1")])

    path = Path(store.root) / _BUFFER_FILENAME
    assert path.exists()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["record_id"] == str(rec.id)
    assert entry["cue"] == "test cue"

def test_flush_drains_buffer_to_store(tmp_path):
    store = MemoryStore(path=tmp_path)
    rec = _make_record(store, "alice flush test")

    defer_provenance(store, [(rec.id, "flush cue", "session-2")])
    count = flush_deferred_provenance(store)
    assert count == 1

    loaded = store.get(rec.id)
    assert loaded is not None
    assert len(loaded.provenance) >= 1
    assert loaded.provenance[-1]["cue"] == "flush cue"

def test_flush_empty_buffer_returns_zero(tmp_path):
    store = MemoryStore(path=tmp_path)
    _make_record(store)
    count = flush_deferred_provenance(store)
    assert count == 0

def test_flush_truncates_file(tmp_path):
    store = MemoryStore(path=tmp_path)
    rec = _make_record(store, "bob truncate")

    defer_provenance(store, [(rec.id, "c1", "s1"), (rec.id, "c2", "s1")])
    flush_deferred_provenance(store)

    path = Path(store.root) / _BUFFER_FILENAME
    assert path.read_text().strip() == ""

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
