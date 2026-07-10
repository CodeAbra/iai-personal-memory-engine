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

def test_recall_for_response_no_k_hits_param(tmp_path) -> None:
    from iai_mcp.pipeline import recall_for_response

    store, graph, recs = _build_store_and_graph(tmp_path, n=5)
    assignment = _flat_assignment(recs)

    with pytest.raises(TypeError):
        recall_for_response(
            store=store, graph=graph, assignment=assignment,
            rich_club=[], embedder=_FakeEmbedder(),
            cue="test", session_id="s1",
            k_hits=10,
        )

def test_recall_for_response_returns_recall_response_type(tmp_path) -> None:
    from iai_mcp.pipeline import recall_for_response

    store, graph, recs = _build_store_and_graph(tmp_path, n=5)
    assignment = _flat_assignment(recs)

    resp = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s2",
    )

    assert isinstance(resp, RecallResponse)
    assert isinstance(resp.hits, list)
    assert isinstance(resp.anti_hits, list)
    assert isinstance(resp.activation_trace, list)
    assert isinstance(resp.budget_used, int)
    assert isinstance(resp.hints, list)
    assert isinstance(resp.cue_mode, str)
    assert isinstance(resp.patterns_observed, list)

def test_recall_for_response_packs_under_budget(tmp_path) -> None:
    from iai_mcp.pipeline import recall_for_response

    store, graph, recs = _build_store_and_graph(tmp_path, n=5, surface_len=200)
    assignment = _flat_assignment(recs)

    resp = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s3", budget_tokens=120,
    )

    assert len(resp.hits) == 2
    assert resp.budget_used == 100

def test_recall_for_response_returns_all_with_unlimited_budget(tmp_path) -> None:
    from iai_mcp.pipeline import recall_for_response

    store, graph, recs = _build_store_and_graph(tmp_path, n=5, surface_len=4)
    assignment = _flat_assignment(recs)

    resp = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s4", budget_tokens=10000,
    )

    assert len(resp.hits) == 5

def test_recall_for_response_minimum_one_hit(tmp_path) -> None:
    from iai_mcp.pipeline import recall_for_response

    store, graph, recs = _build_store_and_graph(tmp_path, n=5, surface_len=400)
    assignment = _flat_assignment(recs)

    resp = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s5", budget_tokens=50,
    )

    assert len(resp.hits) == 1

def test_recall_for_response_threads_mode_to_core(tmp_path) -> None:
    from iai_mcp.pipeline import recall_for_response

    store, graph, recs = _build_store_and_graph(tmp_path, n=5)
    assignment = _flat_assignment(recs)

    resp_v = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s5b", budget_tokens=10000,
        mode="verbatim",
    )
    assert resp_v.cue_mode == "verbatim", (
        f"verbatim mode did not propagate; cue_mode={resp_v.cue_mode}"
    )

    resp_c = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s5c", budget_tokens=10000,
        mode="concept",
    )
    assert resp_c.cue_mode == "concept"

def test_recall_for_response_signature_has_no_k_hits_param() -> None:
    import inspect
    from iai_mcp.pipeline import recall_for_response

    sig = inspect.signature(recall_for_response)
    assert "budget_tokens" in sig.parameters
    assert "mode" in sig.parameters
    assert "k_hits" not in sig.parameters, (
        "recall_for_response signature must NOT carry a k_hits parameter "
        "(contract split — the entry-point split exists so the two "
        "response shapes can never silently swap via an optional kwarg)."
    )

def test_recall_for_response_default_budget_tokens_1500() -> None:
    import inspect
    from iai_mcp.pipeline import recall_for_response

    sig = inspect.signature(recall_for_response)
    assert sig.parameters["budget_tokens"].default == 1500

def test_recall_for_response_shares_core_with_benchmark(tmp_path) -> None:
    from iai_mcp.pipeline import recall_for_benchmark, recall_for_response

    store, graph, recs = _build_store_and_graph(tmp_path, n=8, surface_len=4)
    assignment = _flat_assignment(recs)

    resp_y = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s-shared-r",
        budget_tokens=10000,
    )
    resp_b = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s-shared-b",
        k_hits=100,
    )

    assert resp_y.hits[0].record_id == resp_b.hits[0].record_id, (
        "top scored hit (cosine=1.0 cue-match) must be identical across "
        "entry points; only the final pack/cap is supposed to differ"
    )
    r_set = {h.record_id for h in resp_y.hits}
    b_set = {h.record_id for h in resp_b.hits}
    assert r_set == b_set, (
        f"recall_for_response and recall_for_benchmark must surface the "
        f"same record-id set when neither cap binds; got\n"
        f"  response only: {r_set - b_set}\n"
        f"  benchmark only: {b_set - r_set}"
    )

def test_recall_for_response_budget_enforced_over_pending_markers(tmp_path, monkeypatch) -> None:
    from iai_mcp.pipeline import recall_for_response

    store, graph, recs = _build_store_and_graph(tmp_path, n=5, surface_len=4)
    assignment = _flat_assignment(recs)

    _large_surface = "x" * 80
    _fake_pending: list[MemoryRecord] = []
    for _i in range(10):
        _pm = _make(vec=[0.0] * EMBED_DIM, text=_large_surface)
        _fake_pending.append(_pm)

    monkeypatch.setattr(store, "recent_pending_markers", lambda n=50: _fake_pending[:n])

    resp = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(),
        cue="test", session_id="s-pending-budget",
        budget_tokens=50,
    )

    actual_tokens = sum(len(h.literal_surface) // 4 for h in resp.hits)
    assert resp.budget_used == actual_tokens, (
        f"budget_used={resp.budget_used} != actual surface tokens={actual_tokens}; "
        f"the final enforcement pass is not updating budget_used consistently"
    )

    assert len(resp.hits) <= 7, (
        f"len(hits)={len(resp.hits)} exceeds expected cap of 7 "
        f"(5 scored + 2 pending within budget_tokens=50); "
        f"pending markers are not being capped"
    )
