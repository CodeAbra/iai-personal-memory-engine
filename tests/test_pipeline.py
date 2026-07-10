from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from iai_mcp.community import CommunityAssignment
from iai_mcp.graph import MemoryGraph
from iai_mcp.pipeline import (
    W_AAAK,
    W_AGE,
    W_COSINE,
    W_DEGREE,
    _aaak_overlap,
    _community_gate,
    _cosine,
    _pick_seeds,
    recall_for_response,
)
from iai_mcp.provenance_buffer import flush_deferred_provenance
from iai_mcp.store import MemoryStore, flush_edge_buffer, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord

class _FakeEmbedder:

    DIM = EMBED_DIM

    def embed(self, text: str) -> list[float]:
        return [1.0] + [0.0] * (EMBED_DIM - 1)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

def _make(vec: list[float], text: str = "rec", aaak: str = "", detail: int = 2) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index=aaak,
        embedding=vec,
        community_id=None,
        centrality=0.0,
        detail_level=detail,
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

def test_community_gate_picks_nearest() -> None:
    c0 = uuid4()
    c1 = uuid4()
    c2 = uuid4()
    centroids = {
        c0: [1.0] + [0.0] * (EMBED_DIM - 1),
        c1: [0.0] * 384,
        c2: [-1.0] + [0.0] * (EMBED_DIM - 1),
    }
    a = CommunityAssignment(community_centroids=centroids)
    cue = [1.0] + [0.0] * (EMBED_DIM - 1)
    gated = _community_gate(cue, a, top_n=1)
    assert len(gated) == 1
    assert gated[0] == c0

def test_community_gate_returns_top_n_in_order() -> None:
    c0 = uuid4()
    c1 = uuid4()
    c2 = uuid4()
    centroids = {
        c0: [1.0] + [0.0] * (EMBED_DIM - 1),
        c1: [0.5, 0.5] + [0.0] * (EMBED_DIM - 2),
        c2: [-1.0] + [0.0] * (EMBED_DIM - 1),
    }
    a = CommunityAssignment(community_centroids=centroids)
    cue = [1.0] + [0.0] * (EMBED_DIM - 1)
    gated = _community_gate(cue, a, top_n=3)
    assert gated == [c0, c1, c2]

def test_aaak_overlap_basic_jaccard() -> None:
    assert _aaak_overlap("", "anything") == 0.0
    assert _aaak_overlap("x", "") == 0.0
    assert _aaak_overlap("a b", "a b") == 1.0
    assert abs(_aaak_overlap("a b", "b c") - 1 / 3) < 1e-9

def test_aaak_overlap_slash_split_symmetric() -> None:
    assert _aaak_overlap("auth/login", "auth/login") == 1.0
    assert abs(_aaak_overlap("auth/login", "auth/logout") - 1 / 3) < 1e-9
    assert _aaak_overlap("AUTH/Login", "auth/login") == 1.0

def test_cosine_basic_properties() -> None:
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert _cosine([1.0, 0.0], [-1.0, 0.0]) == -1.0
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0

def test_score_weight_constants_match_d13() -> None:
    assert W_COSINE == 1.0
    assert W_AAAK == 0.3
    assert W_DEGREE == 0.1
    assert W_AGE == 0.05

def test_pipeline_returns_hits_with_adjacent_suggestions(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    records = [
        _make([1.0] + [0.0] * (EMBED_DIM - 1), text="primary match", aaak="test match"),
        _make([0.9, 0.1] + [0.0] * (EMBED_DIM - 2), text="close match"),
        _make([0.0, 1.0] + [0.0] * (EMBED_DIM - 2), text="orthogonal"),
        _make([-1.0] + [0.0] * (EMBED_DIM - 1), text="opposite"),
        _make([0.5, 0.5] + [0.0] * (EMBED_DIM - 2), text="mid"),
    ]
    for r in records:
        store.insert(r)
    flush_record_buffer(store)
    graph = MemoryGraph()
    for r in records:
        graph.add_node(r.id, community_id=None, embedding=r.embedding)
    for i in range(len(records) - 1):
        graph.add_edge(records[i].id, records[i + 1].id)

    community_id = uuid4()
    assignment = CommunityAssignment(
        node_to_community={r.id: community_id for r in records},
        community_centroids={community_id: [1.0] + [0.0] * (EMBED_DIM - 1)},
        modularity=0.0,
        backend="flat",
        top_communities=[community_id],
        mid_regions={community_id: [r.id for r in records]},
    )

    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=[],
        embedder=_FakeEmbedder(),
        cue="test match",
        session_id="s1",
    )
    assert len(resp.hits) >= 1
    assert any(h.literal_surface == "primary match" for h in resp.hits)
    assert all(h.literal_surface != "opposite" for h in resp.hits[:2])
    for h in resp.hits:
        assert isinstance(h.adjacent_suggestions, list)
    assert len(resp.activation_trace) >= 1

def test_pipeline_provenance_appended_to_every_hit(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    r1 = _make([1.0] + [0.0] * (EMBED_DIM - 1), text="primary")
    store.insert(r1)
    flush_record_buffer(store)
    graph = MemoryGraph()
    graph.add_node(r1.id, community_id=None, embedding=r1.embedding)
    community_id = uuid4()
    assignment = CommunityAssignment(
        node_to_community={r1.id: community_id},
        community_centroids={community_id: [1.0] + [0.0] * (EMBED_DIM - 1)},
        modularity=0.0,
        backend="flat",
        top_communities=[community_id],
        mid_regions={community_id: [r1.id]},
    )
    recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=[],
        embedder=_FakeEmbedder(),
        cue="anything",
        session_id="session-42",
    )
    flush_deferred_provenance(store)
    refreshed = store.get(r1.id)
    assert refreshed is not None
    assert len(refreshed.provenance) == 1
    assert refreshed.provenance[0]["session_id"] == "session-42"
    assert refreshed.provenance[0]["cue"] == "anything"

def test_pipeline_budget_caps_hit_count(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    long_text = "x" * 200
    records = []
    for i in range(5):
        r = _make(
            [1.0, float(i) * 0.001] + [0.0] * (EMBED_DIM - 2),
            text=f"{long_text}-{i}",
        )
        records.append(r)
        store.insert(r)
    flush_record_buffer(store)
    graph = MemoryGraph()
    for r in records:
        graph.add_node(r.id, community_id=None, embedding=r.embedding)
    community_id = uuid4()
    assignment = CommunityAssignment(
        node_to_community={r.id: community_id for r in records},
        community_centroids={community_id: [1.0] + [0.0] * (EMBED_DIM - 1)},
        modularity=0.0,
        backend="flat",
        top_communities=[community_id],
        mid_regions={community_id: [r.id for r in records]},
    )
    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=[],
        embedder=_FakeEmbedder(),
        cue="c",
        session_id="s",
        budget_tokens=60,
    )
    assert len(resp.hits) == 1

def test_pipeline_anti_hits_from_contradicts_edge(tmp_path) -> None:
    from iai_mcp.core import dispatch

    store = MemoryStore(path=tmp_path)
    r1 = _make([1.0] + [0.0] * (EMBED_DIM - 1), text="original")
    store.insert(r1)
    flush_record_buffer(store)
    dispatch(
        store,
        "memory_contradict",
        {
            "id": str(r1.id),
            "new_fact": "refuted version",
            "cue_embedding": r1.embedding,
        },
    )
    flush_record_buffer(store)
    flush_edge_buffer(store)

    graph = MemoryGraph()
    graph.add_node(r1.id, community_id=None, embedding=[1.0] + [0.0] * (EMBED_DIM - 1))
    community_id = uuid4()
    assignment = CommunityAssignment(
        node_to_community={r1.id: community_id},
        community_centroids={community_id: [1.0] + [0.0] * (EMBED_DIM - 1)},
        modularity=0.0,
        backend="flat",
        top_communities=[community_id],
        mid_regions={community_id: [r1.id]},
    )
    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=[],
        embedder=_FakeEmbedder(),
        cue="anything",
        session_id="s1",
    )
    assert len(resp.anti_hits) >= 1
    assert "refuted" in resp.anti_hits[0].literal_surface

def test_pipeline_activation_trace_includes_seeds(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    a = _make([1.0] + [0.0] * (EMBED_DIM - 1), text="A")
    b = _make([0.9, 0.1] + [0.0] * (EMBED_DIM - 2), text="B")
    c = _make([0.0, 1.0] + [0.0] * (EMBED_DIM - 2), text="C")
    for r in (a, b, c):
        store.insert(r)
    flush_record_buffer(store)
    graph = MemoryGraph()
    for r in (a, b, c):
        graph.add_node(r.id, community_id=None, embedding=r.embedding)
    graph.add_edge(a.id, b.id)
    graph.add_edge(b.id, c.id)
    community_id = uuid4()
    assignment = CommunityAssignment(
        node_to_community={r.id: community_id for r in (a, b, c)},
        community_centroids={community_id: [1.0] + [0.0] * (EMBED_DIM - 1)},
        modularity=0.0,
        backend="flat",
        top_communities=[community_id],
        mid_regions={community_id: [a.id, b.id, c.id]},
    )
    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=[],
        embedder=_FakeEmbedder(),
        cue="c",
        session_id="s",
    )
    assert a.id in resp.activation_trace

def test_pick_seeds_ranks_by_blended_score(tmp_path) -> None:
    import numpy as np

    shared_cos = np.array([1.0, 0.7071068], dtype=np.float32)
    centrality_arr = np.array([0.0, 1.0], dtype=np.float32)
    candidate_indices = np.array([0, 1], dtype=np.int64)
    seed_indices = _pick_seeds(
        candidate_indices, shared_cos, centrality_arr, n=1,
    )
    assert list(seed_indices) == [1]

def test_pipeline_core_dispatch_integration(tmp_path, monkeypatch) -> None:
    from iai_mcp.core import dispatch

    store = MemoryStore(path=tmp_path)
    r = _make([1.0] + [0.0] * (EMBED_DIM - 1), text="integration")
    store.insert(r)

    class _StubEmbedder:
        DIM = 384

        def embed(self, text: str) -> list[float]:
            return [1.0] + [0.0] * (EMBED_DIM - 1)

    import iai_mcp.embed as embed_mod
    monkeypatch.setattr(embed_mod, "Embedder", _StubEmbedder)

    resp = dispatch(
        store,
        "memory_recall",
        {"cue": "integration", "session_id": "s-int"},
    )
    assert "hits" in resp
    assert isinstance(resp["hits"], list)
    assert isinstance(resp["activation_trace"], list)
    assert "budget_used" in resp

def test_pipeline_empty_gate_falls_back_to_all_nodes(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    r = _make([1.0] + [0.0] * (EMBED_DIM - 1), text="lonely")
    store.insert(r)
    flush_record_buffer(store)
    graph = MemoryGraph()
    graph.add_node(r.id, community_id=None, embedding=r.embedding)
    assignment = CommunityAssignment(
        node_to_community={},
        community_centroids={},
        modularity=0.0,
        backend="flat",
        top_communities=[],
        mid_regions={},
    )
    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=[],
        embedder=_FakeEmbedder(),
        cue="c",
        session_id="s",
    )
    assert len(resp.hits) == 1
