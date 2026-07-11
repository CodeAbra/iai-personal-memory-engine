from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.community import CommunityAssignment
from iai_mcp.graph import MemoryGraph
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord, RecallResponse

class _FakeEmbedder:

    DIM = EMBED_DIM

    def __init__(self, vec: list[float] | None = None) -> None:
        self._vec = vec if vec is not None else [1.0] + [0.0] * (EMBED_DIM - 1)

    def embed(self, text: str) -> list[float]:
        return list(self._vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vec) for _ in texts]

def _make(
    vec: list[float], text: str = "rec", aaak: str = "", tier: str = "episodic",
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index=aaak,
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

def _build_store_and_graph(
    tmp_path, n: int, surface_len: int = 4,
) -> tuple[MemoryStore, MemoryGraph, list[MemoryRecord]]:
    store = MemoryStore(path=tmp_path / "lancedb")
    recs: list[MemoryRecord] = []
    for i in range(n):
        vec = [0.0] * EMBED_DIM
        vec[i % EMBED_DIM] = 1.0
        text = "x" * surface_len
        rec = _make(vec, text=text)
        store.insert(rec)
        recs.append(rec)
    graph = MemoryGraph()
    for rec in recs:
        graph.add_node(
            rec.id, community_id=None, embedding=list(rec.embedding),
        )
        graph.set_node_payload(rec.id, {
            "embedding": list(rec.embedding),
            "surface": rec.literal_surface,
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

def test_recall_for_benchmark_no_budget_tokens_param(tmp_path) -> None:
    from iai_mcp.pipeline import recall_for_benchmark

    store, graph, recs = _build_store_and_graph(tmp_path, n=5)
    assignment = _flat_assignment(recs)

    with pytest.raises(TypeError):
        recall_for_benchmark(
            store=store, graph=graph, assignment=assignment,
            rich_club=[], embedder=_FakeEmbedder(),
            cue="test", session_id="s6",
            budget_tokens=1500,
        )

def test_recall_for_benchmark_returns_at_most_k_hits(tmp_path) -> None:
    from iai_mcp.pipeline import recall_for_benchmark

    store, graph, recs = _build_store_and_graph(tmp_path, n=12)
    assignment = _flat_assignment(recs)

    resp = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s7", k_hits=5,
    )

    assert isinstance(resp, RecallResponse)
    assert len(resp.hits) == 5

def test_recall_for_benchmark_hits_sorted_by_score_desc(tmp_path) -> None:
    from iai_mcp.pipeline import recall_for_benchmark

    store, graph, recs = _build_store_and_graph(tmp_path, n=8)
    assignment = _flat_assignment(recs)

    resp = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s8", k_hits=10,
    )

    scores = [h.score for h in resp.hits]
    assert scores == sorted(scores, reverse=True), (
        f"recall_for_benchmark hits not sorted desc by score: {scores}"
    )

def test_recall_for_benchmark_returns_fewer_when_pool_is_small(tmp_path) -> None:
    from iai_mcp.pipeline import recall_for_benchmark

    store, graph, recs = _build_store_and_graph(tmp_path, n=8)
    assignment = _flat_assignment(recs)

    resp = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s9", k_hits=20,
    )

    assert len(resp.hits) == 8

def test_recall_for_benchmark_budget_used_is_informational(tmp_path) -> None:
    from iai_mcp.pipeline import recall_for_benchmark

    store, graph, recs = _build_store_and_graph(tmp_path, n=5, surface_len=200)
    assignment = _flat_assignment(recs)

    resp = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s10", k_hits=3,
    )

    assert len(resp.hits) == 3
    assert resp.budget_used == 150

def test_recall_for_benchmark_threads_mode_to_core(tmp_path) -> None:
    from iai_mcp.pipeline import recall_for_benchmark

    store, graph, recs = _build_store_and_graph(tmp_path, n=5)
    assignment = _flat_assignment(recs)

    resp = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s-mode", k_hits=10, mode="concept",
    )
    assert resp.cue_mode == "concept"

def test_recall_for_benchmark_signature_has_no_budget_tokens_param() -> None:
    import inspect
    from iai_mcp.pipeline import recall_for_benchmark

    sig = inspect.signature(recall_for_benchmark)
    assert "k_hits" in sig.parameters
    assert "mode" in sig.parameters
    assert "budget_tokens" not in sig.parameters, (
        "recall_for_benchmark signature must NOT carry a budget_tokens "
        "parameter (contract split — the entry-point split exists so "
        "the two response shapes can never silently swap via an optional kwarg)."
    )

def test_recall_for_benchmark_default_k_hits_10() -> None:
    import inspect
    from iai_mcp.pipeline import recall_for_benchmark

    sig = inspect.signature(recall_for_benchmark)
    assert sig.parameters["k_hits"].default == 10
