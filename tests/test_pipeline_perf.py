from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord

class _PerfEmbedder:

    DIM = EMBED_DIM

    def __init__(self, base_seed: int = 0) -> None:
        self._base_seed = base_seed

    def embed(self, text: str) -> list[float]:
        import hashlib
        import random
        digest = hashlib.sha256(
            f"{self._base_seed}:{text}".encode("utf-8")
        ).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        v = [rng.random() * 2 - 1 for _ in range(self.DIM)]
        norm = sum(x * x for x in v) ** 0.5
        return [x / norm for x in v] if norm > 0 else v

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

def _make_rec(vec: list[float], text: str, tags: list[str]) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=vec,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=tags,
        language="en",
    )

def _seed_store(path, n: int = 100, seed: int = 0):
    from iai_mcp.pipeline import recall_for_response  # noqa: F401
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=path)
    embedder = _PerfEmbedder(base_seed=seed)
    tag_pool = [
        ["topic:auth"], ["topic:db"], ["topic:web"],
        ["topic:net"], ["topic:cli"],
    ]
    for i in range(n):
        vec = embedder.embed(f"seed-{i}")
        tags = list(tag_pool[i % len(tag_pool)])
        rec = _make_rec(vec, text=f"synthetic fact {i}", tags=tags)
        store.insert(rec)
    graph, assignment, rich_club = build_runtime_graph(store)
    return store, embedder, graph, assignment, rich_club

@pytest.mark.perf
def test_recall_for_response_p95_under_threshold(tmp_path):
    import time

    from iai_mcp.pipeline import recall_for_response

    store, embedder, graph, assignment, rich_club = _seed_store(
        tmp_path, n=100, seed=0,
    )
    cues = [
        "what did we cover about auth yesterday?",
        "explain the db migration plan",
        "how does the web cache invalidation work",
        "summary of the cli subcommand changes",
        "recent network stack bug report",
    ]

    latencies: list[float] = []
    for i in range(10):
        cue = cues[i % len(cues)]
        t0 = time.perf_counter()
        recall_for_response(
            store=store,
            graph=graph,
            assignment=assignment,
            rich_club=rich_club,
            embedder=embedder,
            cue=cue,
            session_id="perf_test",
            budget_tokens=1500,
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies.sort()
    p95 = latencies[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[-1]
    assert p95 < 150.0, (
        f"D-SPEED regression: p95={p95:.2f}ms > 150ms at N=100 "
        f"(target <100ms strict, 150ms CI-generous). "
        f"All latencies: {[f'{x:.1f}' for x in latencies]}"
    )

def test_recall_for_response_single_provenance_batch_call(tmp_path, monkeypatch):
    from iai_mcp.pipeline import recall_for_response
    from iai_mcp.store import MemoryStore
    import iai_mcp.provenance_buffer as pb

    store, embedder, graph, assignment, rich_club = _seed_store(
        tmp_path, n=100, seed=0,
    )

    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")

    defer_calls: list[int] = []
    single_calls: list[object] = []
    batch_calls: list[int] = []

    original_defer = pb.defer_provenance
    original_single = MemoryStore.append_provenance
    original_batch = MemoryStore.append_provenance_batch

    def _recorder_defer(store_arg, entries):
        defer_calls.append(len(entries))
        return original_defer(store_arg, entries)

    def _recorder_single(self, record_id, entry):
        single_calls.append((record_id, entry))
        return original_single(self, record_id, entry)

    def _recorder_batch(self, pairs, *args, **kwargs):
        batch_calls.append(len(pairs))
        return original_batch(self, pairs, *args, **kwargs)

    monkeypatch.setattr(pb, "defer_provenance", _recorder_defer)
    monkeypatch.setattr(MemoryStore, "append_provenance", _recorder_single)
    monkeypatch.setattr(MemoryStore, "append_provenance_batch", _recorder_batch)

    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=embedder,
        cue="what about auth",
        session_id="wire_test",
        budget_tokens=1500,
    )

    assert len(resp.hits) >= 1, "pipeline must return at least one hit on seeded store"
    assert len(defer_calls) == 1, (
        f"defer_provenance should be called EXACTLY once; got {len(defer_calls)}"
    )
    assert defer_calls[0] == len(resp.hits), (
        f"defer entries list should have {len(resp.hits)} entries (one per hit); "
        f"got {defer_calls[0]}"
    )
    assert len(single_calls) == 0, (
        f"append_provenance (single) should NOT be called on the hit path; "
        f"got {len(single_calls)} calls"
    )
    assert len(batch_calls) == 0, (
        f"append_provenance_batch should NOT be called on the hit path; "
        f"got {len(batch_calls)} calls"
    )

def test_recall_for_response_mem05_provenance_preserved(tmp_path):
    from iai_mcp.pipeline import recall_for_response
    from iai_mcp.provenance_buffer import flush_deferred_provenance

    store, embedder, graph, assignment, rich_club = _seed_store(
        tmp_path, n=100, seed=0,
    )
    session = "mem05_preserved"

    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=embedder,
        cue="what about auth",
        session_id=session,
        budget_tokens=1500,
    )
    assert len(resp.hits) >= 1

    flush_deferred_provenance(store)

    for h in resp.hits:
        rec = store.get(h.record_id)
        assert rec is not None
        matching = [p for p in rec.provenance if p.get("session_id") == session]
        assert len(matching) == 1, (
            f"record {h.record_id} has {len(matching)} provenance entries "
            f"for session '{session}'; expected exactly 1. prov={rec.provenance}"
        )

def test_recall_for_response_on_read_check_uses_batch_variant(tmp_path, monkeypatch):
    from iai_mcp import s4
    from iai_mcp.pipeline import recall_for_response

    store, embedder, graph, assignment, rich_club = _seed_store(
        tmp_path, n=100, seed=0,
    )

    sentinel = [{"kind": "sentinel_batch_wired", "severity": "info", "source_ids": [], "text": "ok"}]

    def _boom(*a, **kw):
        raise RuntimeError("old on_read_check must NOT be called")

    def _batch_ok(*a, **kw):
        return sentinel

    monkeypatch.setattr(s4, "on_read_check", _boom)
    monkeypatch.setattr(s4, "on_read_check_batch", _batch_ok)

    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=embedder,
        cue="what about auth",
        session_id="batch_wired_test",
        budget_tokens=1500,
    )
    hint_kinds = [h.get("kind") for h in resp.hints]
    assert "sentinel_batch_wired" in hint_kinds, (
        f"expected on_read_check_batch sentinel in hints; got {hint_kinds}"
    )
