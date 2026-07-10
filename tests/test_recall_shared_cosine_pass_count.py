from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import numpy as np

from iai_mcp.community import CommunityAssignment
from iai_mcp.graph import MemoryGraph
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

class _FakeEmbedder:

    DIM = EMBED_DIM

    def __init__(self, vec: list[float] | None = None) -> None:
        self._vec = vec if vec is not None else [1.0] + [0.0] * (EMBED_DIM - 1)

    def embed(self, text: str) -> list[float]:
        return list(self._vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vec) for _ in texts]

def _make(vec: list[float], text: str = "rec", tier: str = "episodic") -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
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
        tags=[],
        language="en",
    )

def _build_store_and_graph(tmp_path, n: int) -> tuple[MemoryStore, MemoryGraph, list[MemoryRecord]]:
    store = MemoryStore(path=tmp_path / "lancedb")
    recs: list[MemoryRecord] = []
    for i in range(n):
        vec = [0.0] * EMBED_DIM
        vec[i % EMBED_DIM] = 1.0
        rec = _make(vec, text=f"rec{i}")
        store.insert(rec)
        recs.append(rec)
    graph = MemoryGraph()
    for rec in recs:
        graph.add_node(
            rec.id, community_id=None, embedding=list(rec.embedding),
        )
        graph.set_node_payload(rec.id, {
            "embedding": list(rec.embedding),
            "surface": f"rec{recs.index(rec)}",
            "centrality": 0.0,
            "tier": rec.tier,
            "tags": [],
            "language": "en",
        })
    return store, graph, recs

def _flat_assignment(recs: list[MemoryRecord]) -> CommunityAssignment:
    cid = uuid4()
    centroid = [1.0] + [0.0] * (EMBED_DIM - 1)
    return CommunityAssignment(
        node_to_community={r.id: cid for r in recs},
        community_centroids={cid: centroid},
        modularity=0.0,
        backend="flat",
        top_communities=[cid],
        mid_regions={cid: [r.id for r in recs]},
    )

def _matmul_with_counter(counter: dict[str, int]):
    orig = np.matmul

    def wrapped(a, b, **kw):
        try:
            if (
                hasattr(a, "shape")
                and hasattr(b, "shape")
                and len(a.shape) == 2
                and len(b.shape) == 1
                and a.shape[1] == b.shape[0]
                and a.shape[0] >= 50
            ):
                counter["count"] = counter.get("count", 0) + 1
        except Exception:
            pass
        return orig(a, b, **kw)

    return wrapped

def test_recall_for_benchmark_runs_one_pool_cosine(tmp_path, monkeypatch):
    from iai_mcp.pipeline import recall_for_benchmark

    store, graph, recs = _build_store_and_graph(tmp_path, n=60)
    assignment = _flat_assignment(recs)
    embedder = _FakeEmbedder()

    counter: dict[str, int] = {"count": 0}
    monkeypatch.setattr(np, "matmul", _matmul_with_counter(counter))

    recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="primary", session_id="s-bench-cosine-1",
        k_hits=10, mode="concept",
    )

    assert counter["count"] == 1, (
        f"Single-pass violation: cue-vs-large-pool matmul fired "
        f"{counter['count']} times via recall_for_benchmark; expected "
        "exactly 1 (the shared cosine pass at the top of _recall_core)."
    )

def test_recall_for_response_runs_one_pool_cosine(tmp_path, monkeypatch):
    from iai_mcp.pipeline import recall_for_response

    store, graph, recs = _build_store_and_graph(tmp_path, n=60)
    assignment = _flat_assignment(recs)
    embedder = _FakeEmbedder()

    counter: dict[str, int] = {"count": 0}
    monkeypatch.setattr(np, "matmul", _matmul_with_counter(counter))

    recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="primary", session_id="s-resp-cosine-2",
        budget_tokens=4000, mode="concept",
    )

    assert counter["count"] == 1, (
        f"Single-pass violation: cue-vs-large-pool matmul fired "
        f"{counter['count']} times via recall_for_response; expected "
        "exactly 1 (the shared cosine pass at the top of _recall_core)."
    )

def test_l0_fastpath_runs_zero_pool_cosines(tmp_path, monkeypatch):
    import iai_mcp.gate as gate_mod
    from iai_mcp.pipeline import recall_for_benchmark

    monkeypatch.setattr(
        gate_mod,
        "should_skip_retrieval",
        lambda cue: (True, "test L0 reason"),
    )

    store, graph, recs = _build_store_and_graph(tmp_path, n=60)
    l0_uuid = UUID("00000000-0000-0000-0000-000000000001")
    now = datetime.now(timezone.utc)
    l0_rec = MemoryRecord(
        id=l0_uuid,
        tier="episodic",
        literal_surface="L0 identity literal",
        aaak_index="",
        embedding=[1.0] + [0.0] * (EMBED_DIM - 1),
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
        tags=[],
        language="en",
    )
    store.insert(l0_rec)
    assignment = _flat_assignment(recs)
    embedder = _FakeEmbedder()

    counter: dict[str, int] = {"count": 0}
    monkeypatch.setattr(np, "matmul", _matmul_with_counter(counter))

    resp = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="hi", session_id="s-l0-fast-3",
        k_hits=10, mode="concept",
    )

    assert len(resp.hits) == 1, (
        f"L0 fast-path should return exactly 1 hit; got {len(resp.hits)}"
    )
    assert resp.hits[0].record_id == l0_uuid, (
        "L0 fast-path returned a non-L0 record; gate fired but pool walk "
        "happened anyway."
    )
    assert counter["count"] == 0, (
        f"L0 fast-path violation: cue-vs-large-pool matmul fired "
        f"{counter['count']} times even though the L0 gate fired; "
        "expected 0 (the L0 path bypasses the pool walk entirely)."
    )
