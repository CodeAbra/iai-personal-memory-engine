"""Direct unit coverage for `_recall_core`'s pool-gap backfill: the
best-effort fetch that resolves a graph pool member whose node has an
embedding but has not yet received its "surface" payload merge -- the
exact split a concurrent reader can observe between
`_make_graph_sync_hook`'s `add_node` and `set_node_payload` calls against
a shared, warm graph. Confirmed unreachable via the sole production
dispatch call path (which always builds a fresh, atomically-populated
graph), but reachable by any direct `_recall_core`/`recall_for_response`
caller wired to a shared graph -- exactly what this file exercises.
"""
from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from bench.neural_map import _BenchEmbedder, _make_record

import iai_mcp.pipeline as _pm
from iai_mcp.pipeline import recall_for_response
from iai_mcp.retrieve import build_runtime_graph
from iai_mcp.store import MemoryStore, flush_edge_buffer, flush_record_buffer


def _build_base_store(tmp_path: Path, embedder: _BenchEmbedder, n: int = 8) -> MemoryStore:
    store = MemoryStore(path=tmp_path / "gap-backfill-store")
    for i in range(n):
        vec = embedder.embed(f"base-{i}")
        rec = _make_record(vec, text=f"base fact {i}", tags=["topic:base"])
        store.insert(rec)
    flush_record_buffer(store)
    flush_edge_buffer(store)
    return store


def test_pool_gap_backfill_serves_split_write_node_bounded(tmp_path):
    """A node whose graph payload carries an embedding but no "surface" key
    is still resolved and served, via exactly one bounded
    store.get_batch call -- proving the backfill fires and stays capped on
    the real split-write shape, not just in the synthetic-pool test below."""
    embedder = _BenchEmbedder(base_seed=0, dim=384)
    store = _build_base_store(tmp_path, embedder)
    graph, assignment, rich_club = build_runtime_graph(store)

    target_vec = embedder.embed("gap-target")
    target_rec = _make_record(target_vec, text="gap target surface text", tags=["topic:gap"])
    store.insert(target_rec)
    flush_record_buffer(store)
    flush_edge_buffer(store)

    # Split-write: only the embedding half of _make_graph_sync_hook's pair
    # fires here -- set_node_payload (the "surface" merge) never runs,
    # mirroring the exact gap a concurrent reader of a warm bundle can see.
    graph.add_node(target_rec.id, community_id=None, embedding=target_rec.embedding)
    assert "surface" not in graph.get_payload(target_rec.id)

    get_batch_calls: list[list[UUID]] = []
    orig_get_batch = MemoryStore.get_batch

    def _spy_get_batch(self, ids, **kwargs):
        get_batch_calls.append(list(ids))
        return orig_get_batch(self, ids, **kwargs)

    MemoryStore.get_batch = _spy_get_batch
    _pm._last_recall_latency_ms = 0.0
    try:
        response = recall_for_response(
            store=store, graph=graph, assignment=assignment, rich_club=rich_club,
            embedder=embedder, cue="gap-target", session_id="pool-gap-backfill-probe",
            budget_tokens=2000, mode="concept",
        )
    finally:
        MemoryStore.get_batch = orig_get_batch

    assert get_batch_calls, "expected the pool-gap backfill to call store.get_batch"
    for call_ids in get_batch_calls:
        assert len(call_ids) <= 1024, (
            f"pool-gap backfill batch size {len(call_ids)} exceeds the 1024 cap"
        )
    assert any(target_rec.id in call_ids for call_ids in get_batch_calls), (
        "the split-write node's id never reached store.get_batch -- backfill did not fire"
    )

    served = {h.record_id: h for h in response.hits}
    assert target_rec.id in served, (
        "the split-write node was never served despite being pool- and store-resolvable"
    )
    assert served[target_rec.id].literal_surface == target_rec.literal_surface


def test_pool_gap_backfill_truncates_oversized_gap(tmp_path, monkeypatch):
    """The backfill batch stays capped even when the pool-vs-cache gap
    itself is far larger than the cap -- proves the slice bound directly,
    independent of whether any individual id happens to resolve."""
    embedder = _BenchEmbedder(base_seed=1, dim=384)
    store = _build_base_store(tmp_path, embedder)
    graph, assignment, rich_club = build_runtime_graph(store)

    fake_pool_ids = [uuid4() for _ in range(1500)]
    fake_pool_embs = np.zeros((len(fake_pool_ids), 384), dtype=np.float32)
    monkeypatch.setattr(
        _pm, "_collect_graph_pool",
        lambda graph_, records_cache, store_: (fake_pool_ids, fake_pool_embs),
    )

    get_batch_calls: list[list[UUID]] = []
    orig_get_batch = MemoryStore.get_batch

    def _spy_get_batch(self, ids, **kwargs):
        get_batch_calls.append(list(ids))
        return orig_get_batch(self, ids, **kwargs)

    monkeypatch.setattr(MemoryStore, "get_batch", _spy_get_batch)
    _pm._last_recall_latency_ms = 0.0

    recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue="oversized-gap-probe", session_id="pool-gap-backfill-truncate-probe",
        budget_tokens=2000, mode="concept",
    )

    assert get_batch_calls, "expected the pool-gap backfill to call store.get_batch"
    max_batch = max(len(c) for c in get_batch_calls)
    assert max_batch <= 1024, (
        f"pool-gap backfill batch size {max_batch} exceeds the 1024 cap "
        f"with a {len(fake_pool_ids)}-id gap"
    )
