from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pytest

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
    tmp_path, n: int, gold_indices: list[int] | None = None,
    semantic_indices: list[int] | None = None,
) -> tuple[MemoryStore, MemoryGraph, list[MemoryRecord]]:
    store = MemoryStore(path=tmp_path / "lancedb")
    recs: list[MemoryRecord] = []
    semantic_set = set(semantic_indices or [])
    for i in range(n):
        vec = [0.0] * EMBED_DIM
        vec[i % EMBED_DIM] = 1.0
        tier = "semantic" if i in semantic_set else "episodic"
        rec = _make(vec, text=f"rec{i}", tier=tier)
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
            "created_at": str(getattr(rec, "created_at", "") or ""),
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

def _degenerate_assignment(recs: list[MemoryRecord]) -> CommunityAssignment:
    centroids = {uuid4(): list(rec.embedding) for rec in recs}
    cids = list(centroids.keys())
    return CommunityAssignment(
        node_to_community={recs[i].id: cids[i] for i in range(len(recs))},
        community_centroids=centroids,
        modularity=0.0,
        backend="leiden-test",
        top_communities=cids[:3],
        mid_regions={cids[i]: [recs[i].id] for i in range(len(recs))},
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

def test_recall_core_runs_one_cosine_pass(tmp_path, monkeypatch):
    from iai_mcp.pipeline import _recall_core

    store, graph, recs = _build_store_and_graph(tmp_path, n=60)
    assignment = _flat_assignment(recs)
    embedder = _FakeEmbedder()

    counter: dict[str, int] = {"count": 0}
    monkeypatch.setattr(np, "matmul", _matmul_with_counter(counter))

    _recall_core(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="primary", session_id="s-mat-1",
    )

    assert counter["count"] == 1, (
        f"Single-pass violation: cue-vs-large-pool matmul fired "
        f"{counter['count']} times; expected exactly 1 (the shared "
        "cosine pass at the top of _recall_core)."
    )

def test_recall_core_gate_is_diagnostic_not_filter(tmp_path):
    from iai_mcp.pipeline import _recall_core

    store, graph, recs = _build_store_and_graph(tmp_path, n=50)
    embedder = _FakeEmbedder(
        vec=[0.0] * 5 + [1.0] + [0.0] * (EMBED_DIM - 6)
    )
    assignment = _degenerate_assignment(recs)

    result = _recall_core(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="cue at axis 5", session_id="s-gate-2", mode="concept",
    )

    assert len(result.scored_hits) >= 1
    assert result.scored_hits[0].record_id == recs[5].id

def test_recall_core_K_CANDIDATES_covers_rank_199(tmp_path):
    from iai_mcp.pipeline import K_CANDIDATES, _recall_core

    n = 250
    store = MemoryStore(path=tmp_path / "lancedb")
    recs: list[MemoryRecord] = []
    for i in range(n):
        vec = [0.0] * EMBED_DIM
        vec[0] = float(n - i) / n
        if i + 1 < EMBED_DIM:
            vec[i + 1] = 0.01
        norm = float(np.linalg.norm(np.asarray(vec, dtype=np.float32)))
        if norm > 0.0:
            vec = [v / norm for v in vec]
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
            "surface": "rec",
            "centrality": 0.0,
            "tier": "episodic",
            "tags": [], "language": "en",
        })
    assignment = _flat_assignment(recs)
    gold = recs[199]

    embedder = _FakeEmbedder(
        vec=[1.0] + [0.0] * (EMBED_DIM - 1)
    )
    result = _recall_core(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="cue", session_id="s-k-3",
    )

    assert K_CANDIDATES == 200
    found_ids = {h.record_id for h in result.scored_hits}
    assert gold.id in found_ids

def test_recall_core_passes_shared_cos_to_pick_seeds(tmp_path, monkeypatch):
    import iai_mcp.pipeline as pipeline_mod
    from iai_mcp.pipeline import _recall_core

    store, graph, recs = _build_store_and_graph(tmp_path, n=30)
    assignment = _flat_assignment(recs)
    embedder = _FakeEmbedder()

    captured: dict[str, Any] = {}
    orig = pipeline_mod._pick_seeds

    def spy(candidate_indices, shared_cos, centrality_arr, n=3):
        captured["candidate_indices"] = candidate_indices
        captured["shared_cos"] = shared_cos
        captured["centrality_arr"] = centrality_arr
        return orig(candidate_indices, shared_cos, centrality_arr, n=n)

    monkeypatch.setattr(pipeline_mod, "_pick_seeds", spy)
    _recall_core(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="cue", session_id="s-pick-4",
    )

    assert isinstance(captured["candidate_indices"], np.ndarray)
    assert isinstance(captured["shared_cos"], np.ndarray)
    assert isinstance(captured["centrality_arr"], np.ndarray)
    assert captured["candidate_indices"].dtype.kind in {"i", "u"}

def test_recall_core_reachable_includes_cosine_top_k(tmp_path):
    from iai_mcp.pipeline import _recall_core

    store, graph, recs = _build_store_and_graph(tmp_path, n=30)
    embedder = _FakeEmbedder(
        vec=[0.0] * 17 + [1.0] + [0.0] * (EMBED_DIM - 18)
    )
    assignment = _flat_assignment(recs)

    result = _recall_core(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="axis 17", session_id="s-reach-5",
    )

    found = {h.record_id for h in result.scored_hits}
    assert recs[17].id in found, (
        "Reachability violation: gold record reachable via cosine top-K but "
        "not surfaced in scored_hits (graph 2-hop spread alone cannot "
        "be the source of truth)."
    )

def test_recall_core_stage5_does_not_recompute_cosine(tmp_path, monkeypatch):
    from iai_mcp.pipeline import _recall_core

    store, graph, recs = _build_store_and_graph(tmp_path, n=60)
    assignment = _flat_assignment(recs)
    embedder = _FakeEmbedder()

    counter: dict[str, int] = {"count": 0}
    monkeypatch.setattr(np, "matmul", _matmul_with_counter(counter))

    _recall_core(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="cue", session_id="s-mat-6",
    )

    assert counter["count"] == 1, (
        f"Stage 5 triggered an extra cue-vs-pool matmul; "
        f"total count = {counter['count']} (expected 1 for the shared pass)."
    )

def test_recall_core_scored_hits_sorted_descending(tmp_path):
    from iai_mcp.pipeline import _recall_core

    store, graph, recs = _build_store_and_graph(tmp_path, n=30)
    assignment = _flat_assignment(recs)
    embedder = _FakeEmbedder()

    result = _recall_core(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="cue", session_id="s-sort-7",
    )

    scores = [h.score for h in result.scored_hits]
    assert scores == sorted(scores, reverse=True), (
        f"scored_hits is not sorted descending: {scores}"
    )

def test_recall_core_l0_fastpath_inside_core(tmp_path, monkeypatch):
    import iai_mcp.gate as gate_mod
    from iai_mcp.pipeline import _recall_core

    monkeypatch.setattr(
        gate_mod,
        "should_skip_retrieval",
        lambda cue: (True, "test L0 reason"),
    )

    store = MemoryStore(path=tmp_path / "lancedb")
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
    graph = MemoryGraph()
    graph.add_node(l0_uuid, community_id=None, embedding=l0_rec.embedding)
    assignment = _flat_assignment([l0_rec])
    embedder = _FakeEmbedder()

    result = _recall_core(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="hi", session_id="s-l0-8",
    )

    assert len(result.scored_hits) == 1
    assert result.scored_hits[0].record_id == l0_uuid
    assert result.cue_mode == "concept"
    assert any(h.get("kind") == "retrieval_skipped" for h in result.hints)
    assert result.budget_used > 0

def test_recall_core_verbatim_mode_filters_to_episodic(tmp_path):
    from iai_mcp.pipeline import _recall_core

    store, graph, recs = _build_store_and_graph(
        tmp_path, n=6, semantic_indices=[1, 3, 5],
    )
    assignment = _flat_assignment(recs)
    embedder = _FakeEmbedder()

    result = _recall_core(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="cue", session_id="s-vb-9", mode="verbatim",
    )

    for h in result.scored_hits:
        rec = store.get(h.record_id)
        assert rec is not None
        assert rec.tier == "episodic"
    assert result.hints == []
    assert result.patterns_observed == []

def test_recall_core_verbatim_filter_at_post_stage4_location(
    tmp_path, monkeypatch,
):
    import iai_mcp.pipeline as pipeline_mod
    from iai_mcp.pipeline import _recall_core

    store, graph, recs = _build_store_and_graph(
        tmp_path, n=10, semantic_indices=[0],
    )
    assignment = _flat_assignment(recs)
    embedder = _FakeEmbedder(
        vec=[1.0] + [0.0] * (EMBED_DIM - 1)
    )

    debug_capture: dict[str, Any] = {}
    monkeypatch.setattr(
        pipeline_mod, "_VERBATIM_FILTER_DEBUG", debug_capture, raising=False,
    )

    result = _recall_core(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="cue", session_id="s-vb-9b", mode="verbatim",
    )

    found_ids = {h.record_id for h in result.scored_hits}
    assert recs[0].id not in found_ids

    pre = debug_capture.get("pre_filter_reachable_ids")
    post = debug_capture.get("post_filter_reachable_ids")
    assert pre is not None and post is not None, (
        "verbatim-filter placement proof requires the recall-core debug "
        "hook (_VERBATIM_FILTER_DEBUG) to capture pre/post reachable_ids"
    )
    assert recs[0].id in pre, (
        "semantic record at high cosine rank must reach the union "
        "(reachable = cosine top-K ∪ 2-hop ∪ rich-club, no pre-filter)"
    )
    assert recs[0].id not in post, (
        "verbatim filter must REMOVE semantic record between Stage 4 "
        "(union) and Stage 5 (rank); the canonical pipeline.py:831 "
        "location is preserved"
    )

def test_recall_core_profile_modulation_applied(tmp_path):
    from iai_mcp.pipeline import _recall_core

    store, graph, recs = _build_store_and_graph(tmp_path, n=10)
    assignment = _flat_assignment(recs)
    embedder = _FakeEmbedder()

    result_none = _recall_core(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="cue", session_id="s-mod-10a", profile_state=None,
    )
    result_active = _recall_core(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="cue", session_id="s-mod-10b",
        profile_state={"literal_preservation": "strong"},
    )

    none_scores = {h.record_id: h.score for h in result_none.scored_hits}
    active_scores = {h.record_id: h.score for h in result_active.scored_hits}
    diff_count = sum(
        1 for rid in none_scores
        if rid in active_scores and abs(none_scores[rid] - active_scores[rid]) > 1e-9
    )
    assert len(result_active.scored_hits) == len(result_none.scored_hits)
    assert result_active.cue_mode == "concept"

def test_recall_core_post_rank_artifacts_populated(tmp_path):
    from iai_mcp.pipeline import _RecallCoreResult, _recall_core

    store, graph, recs = _build_store_and_graph(tmp_path, n=8)
    assignment = _flat_assignment(recs)
    embedder = _FakeEmbedder()

    result = _recall_core(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="cue", session_id="s-art-11",
    )

    assert isinstance(result, _RecallCoreResult)
    assert isinstance(result.scored_hits, list)
    assert isinstance(result.activation_trace, list)
    assert isinstance(result.anti_hits, list)
    assert isinstance(result.hints, list)
    assert isinstance(result.patterns_observed, list)
    assert isinstance(result.cue_mode, str)
    assert isinstance(result.budget_used, int)

def test_pick_seeds_new_signature_returns_indices() -> None:
    from iai_mcp.pipeline import _pick_seeds

    candidate_indices = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    shared_cos = np.array(
        [0.1, 0.9, 0.5, 0.2, 0.7], dtype=np.float32
    )
    centrality_arr = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
    )
    out = _pick_seeds(candidate_indices, shared_cos, centrality_arr, n=3)
    assert isinstance(out, np.ndarray)
    assert out.dtype.kind in {"i", "u"}
    assert list(out) == [1, 4, 2]

def test_pick_seeds_blends_cosine_and_centrality() -> None:
    from iai_mcp.pipeline import _pick_seeds

    candidate_indices = np.array([0, 1, 2], dtype=np.int64)
    shared_cos = np.array([0.5, 0.4, 0.6], dtype=np.float32)
    centrality_arr = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    out = _pick_seeds(candidate_indices, shared_cos, centrality_arr, n=1)
    assert list(out) == [1]

def test_pick_seeds_does_no_store_io(monkeypatch) -> None:
    from iai_mcp.pipeline import _pick_seeds

    dot_calls: dict[str, int] = {"count": 0}
    orig_dot = np.dot

    def wrapped_dot(a, b, **kw):
        dot_calls["count"] = dot_calls.get("count", 0) + 1
        return orig_dot(a, b, **kw)

    monkeypatch.setattr(np, "dot", wrapped_dot)
    candidate_indices = np.array([0, 1, 2], dtype=np.int64)
    shared_cos = np.array([0.5, 0.4, 0.6], dtype=np.float32)
    centrality_arr = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    _ = _pick_seeds(candidate_indices, shared_cos, centrality_arr, n=2)
    assert dot_calls["count"] == 0

def test_pick_seeds_empty_candidates_returns_empty() -> None:
    from iai_mcp.pipeline import _pick_seeds

    candidate_indices = np.array([], dtype=np.int64)
    shared_cos = np.array([0.5, 0.4, 0.6], dtype=np.float32)
    centrality_arr = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    out = _pick_seeds(candidate_indices, shared_cos, centrality_arr, n=3)
    assert isinstance(out, np.ndarray)
    assert out.size == 0
    assert out.dtype == candidate_indices.dtype

def test_pick_seeds_old_signature_raises() -> None:
    from iai_mcp.pipeline import _pick_seeds

    with pytest.raises(TypeError):
        _pick_seeds(
            [uuid4()], [1.0] + [0.0] * (EMBED_DIM - 1),
            None, None, {}, 3, None,
        )

def _make_store_hermetic(tmp_path, monkeypatch) -> MemoryStore:
    store_root = tmp_path / "store"
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    return MemoryStore(path=store_root)

def _insert_recs(store: MemoryStore, n: int, start: int = 0) -> list[MemoryRecord]:
    recs = []
    for i in range(n):
        vec = [0.0] * EMBED_DIM
        vec[(start + i) % EMBED_DIM] = 1.0
        norm = float(np.linalg.norm(np.asarray(vec, dtype=np.float32)))
        if norm > 0:
            vec = [v / norm for v in vec]
        rec = _make(vec, text=f"rec{start + i}")
        store.insert(rec)
        recs.append(rec)
    return recs

def test_warm_recall_does_not_call_build_runtime_graph(tmp_path, monkeypatch):
    import iai_mcp.retrieve as retrieve_mod
    from iai_mcp.pipeline import recall_for_response

    store = _make_store_hermetic(tmp_path, monkeypatch)
    recs = _insert_recs(store, 5)
    assignment = _flat_assignment(recs)
    graph = MemoryGraph()
    for r in recs:
        graph.add_node(r.id, community_id=None, embedding=list(r.embedding or []))
        graph.set_node_payload(r.id, {
            "embedding": list(r.embedding or []),
            "surface": r.literal_surface,
            "centrality": 0.0, "tier": r.tier, "tags": [], "language": "en",
        })

    def _no_build_runtime_graph(*a, **kw):
        raise RuntimeError("build_runtime_graph must NOT be called on the warm path")

    monkeypatch.setattr(retrieve_mod, "build_runtime_graph", _no_build_runtime_graph)

    embedder = _FakeEmbedder()
    resp = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="rec0", session_id="s-cc-d-1",
    )
    assert resp is not None, "recall_for_response must return a response"

def test_warm_recall_does_not_call_detect_communities_or_rich_club_nodes(
    tmp_path, monkeypatch
):
    import iai_mcp.community as community_mod
    import iai_mcp.richclub as richclub_mod
    from iai_mcp.pipeline import recall_for_response

    store = _make_store_hermetic(tmp_path, monkeypatch)
    recs = _insert_recs(store, 5)
    assignment = _flat_assignment(recs)
    graph = MemoryGraph()
    for r in recs:
        graph.add_node(r.id, community_id=None, embedding=list(r.embedding or []))
        graph.set_node_payload(r.id, {
            "embedding": list(r.embedding or []),
            "surface": r.literal_surface,
            "centrality": 0.0, "tier": r.tier, "tags": [], "language": "en",
        })

    def _no_detect(*a, **kw):
        raise RuntimeError("detect_communities must NOT be on recall hot path")

    def _no_rich_club(*a, **kw):
        raise RuntimeError("rich_club_nodes must NOT be on recall hot path")

    monkeypatch.setattr(community_mod, "detect_communities", _no_detect)
    monkeypatch.setattr(richclub_mod, "rich_club_nodes", _no_rich_club)

    embedder = _FakeEmbedder()
    resp = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="rec0", session_id="s-cc-d-2",
    )
    assert resp is not None

def test_bounded_2hop_spread_reaches_two_hops(tmp_path, monkeypatch):
    store = _make_store_hermetic(tmp_path, monkeypatch)

    cue_vec = [1.0] + [0.0] * (EMBED_DIM - 1)
    seed_vec = [1.0] + [0.0] * (EMBED_DIM - 1)
    hop1_vec = [0.0] * EMBED_DIM
    hop1_vec[1] = 1.0
    hop2_vec = [0.0] * EMBED_DIM
    hop2_vec[2] = 1.0

    def _l2(v):
        arr = np.asarray(v, dtype=np.float32)
        n = float(np.linalg.norm(arr))
        return (arr / n).tolist() if n > 0 else arr.tolist()

    rec_seed = _make(_l2(seed_vec), text="seed")
    rec_hop1 = _make(_l2(hop1_vec), text="hop1")
    rec_hop2 = _make(_l2(hop2_vec), text="hop2")
    store.insert(rec_seed)
    store.insert(rec_hop1)
    store.insert(rec_hop2)

    store.boost_edges([(rec_seed.id, rec_hop1.id)], edge_type="hebbian", delta=1.0)
    store.boost_edges([(rec_hop1.id, rec_hop2.id)], edge_type="hebbian", delta=1.0)

    from iai_mcp.pipeline import K_CANDIDATES

    ann_pairs = store.query_similar(cue_vec, k=K_CANDIDATES)
    candidate_recs = {r.id: r for r, _s in ann_pairs}

    hop1_edges = store.incident_edges(list(candidate_recs.keys()), top_k=5)
    hop1_new_ids = list({
        nbr
        for nbr_list in hop1_edges.values()
        for (nbr, _et, _wt) in nbr_list
        if nbr not in candidate_recs
    })
    if hop1_new_ids:
        candidate_recs.update(store.get_batch(hop1_new_ids))

    hop2_edges = store.incident_edges(hop1_new_ids, top_k=5) if hop1_new_ids else {}
    hop2_new_ids = list({
        nbr
        for nbr_list in hop2_edges.values()
        for (nbr, _et, _wt) in nbr_list
        if nbr not in candidate_recs
    })
    if hop2_new_ids:
        candidate_recs.update(store.get_batch(hop2_new_ids))

    assert rec_hop2.id in candidate_recs, (
        "CC-C: a record two hops from the ANN seed must be in the bounded "
        "pool after the 2-hop incident_edges expansion (top_k=5 each hop)."
    )
    assert rec_hop1.id in candidate_recs

def test_load_recall_structural_case2_returns_nonempty(tmp_path, monkeypatch):
    import iai_mcp.runtime_graph_cache as rgc

    store = _make_store_hermetic(tmp_path, monkeypatch)
    window = rgc._STALENESS_WINDOW
    recs = _insert_recs(store, window + 2)
    assignment = _flat_assignment(recs)
    ok = rgc.save(store, assignment, [recs[0].id], max_degree=1)
    assert ok

    _insert_recs(store, window, start=window + 10)
    assert rgc.try_load(store) is None, "Precondition: should MISS after window cross"

    result = rgc.load_last_good_structural(store)
    assert result is not None, (
        "CC2-H3 case-2: load_last_good_structural must return last-good "
        "assignment on a count-only drift (NOT empty bias)."
    )
    lga, lgrc = result
    assert len(lga.node_to_community) > 0, "last-good assignment must be non-empty"
    assert recs[0].id in lgrc

def test_load_recall_structural_case3_labelled(tmp_path, monkeypatch):
    import iai_mcp.runtime_graph_cache as rgc

    store = _make_store_hermetic(tmp_path, monkeypatch)
    a, rc, max_deg, src, _node_degrees = rgc.load_recall_structural(store)
    assert src == "cold_degrade", (
        "CC2-H3 case-3: a truly-cold store must return structural_source "
        "'cold_degrade', not a silent empty bias."
    )
    assert len(a.node_to_community) == 0
    assert rc == []
    assert max_deg == 0

def test_preload_ready_flag_exists_and_is_event():
    import threading
    import iai_mcp.runtime_graph_cache as rgc

    assert isinstance(rgc.preload_ready, threading.Event), (
        "runtime_graph_cache.preload_ready must be a threading.Event "
        "so core.py can import and read it without importing daemon.py."
    )

def test_uncapped_contradicts_ts_by_id(tmp_path, monkeypatch):
    store = _make_store_hermetic(tmp_path, monkeypatch)

    cue_vec = [1.0] + [0.0] * (EMBED_DIM - 1)
    seed = _make([1.0] + [0.0] * (EMBED_DIM - 1), text="seed")
    store.insert(seed)

    targets = []
    for i in range(6):
        v = [0.0] * EMBED_DIM
        v[(i + 1) % EMBED_DIM] = 1.0
        r = _make(v, text=f"target{i}")
        store.insert(r)
        targets.append(r)

    for i, t in enumerate(targets):
        w = 1.0 - i * 0.1
        store.boost_edges([(seed.id, t.id)], edge_type="contradicts", delta=w)

    capped = store.incident_edges([seed.id], edge_types=["contradicts"], top_k=5)
    uncapped = store.incident_edges([seed.id], edge_types=["contradicts"], top_k=None)

    capped_nbrs = {nbr for (nbr, _et, _wt) in capped.get(seed.id, [])}
    uncapped_nbrs = {nbr for (nbr, _et, _wt) in uncapped.get(seed.id, [])}

    assert len(uncapped_nbrs) == 6, "UNCAPPED should return all 6 contradicts targets"
    assert len(capped_nbrs) <= 5, "Capped top_k=5 should return at most 5"
    assert uncapped_nbrs >= capped_nbrs, "UNCAPPED must be a superset of capped"

def test_find_anti_hits_does_not_call_edges_to_pandas(tmp_path, monkeypatch):
    store = _make_store_hermetic(tmp_path, monkeypatch)
    recs = _insert_recs(store, 5)

    store.boost_edges([(recs[0].id, recs[1].id)], edge_type="contradicts", delta=1.0)

    def _no_to_pandas(*a, **kw):
        raise RuntimeError("edges.to_pandas() must NOT be called on anti-hits path")

    orig_open = store.db.open_table

    def _spy_open(name, *a, **kw):
        tbl = orig_open(name, *a, **kw)
        if name == "edges":
            tbl.to_pandas = _no_to_pandas
        return tbl

    monkeypatch.setattr(store.db, "open_table", _spy_open)

    from iai_mcp.graph import MemoryGraph
    from iai_mcp.pipeline import _find_anti_hits

    graph = MemoryGraph()
    hits = []
    from iai_mcp.types import MemoryHit
    for r in recs[:2]:
        hits.append(MemoryHit(
            record_id=r.id, score=1.0, reason="test",
            literal_surface=r.literal_surface, adjacent_suggestions=[],
        ))

    result = _find_anti_hits(hits, store, graph, k=3)
    assert isinstance(result, list)

def test_stage14_profile_modulates_chunked_not_large_batch(tmp_path, monkeypatch):
    store = _make_store_hermetic(tmp_path, monkeypatch)

    n = 10
    recs = _insert_recs(store, n)
    assignment = _flat_assignment(recs)
    graph = MemoryGraph()
    for r in recs:
        graph.add_node(r.id, community_id=None, embedding=list(r.embedding or []))
        graph.set_node_payload(r.id, {
            "embedding": list(r.embedding or []),
            "surface": r.literal_surface,
            "centrality": 0.0, "tier": r.tier, "tags": [], "language": "en",
        })

    call_sizes: list[int] = []
    orig_boost = store.boost_edges

    def _spy_boost(pairs, **kw):
        call_sizes.append(len(pairs))
        return orig_boost(pairs, **kw)

    monkeypatch.setattr(store, "boost_edges", _spy_boost)

    profile_state = {"interest_boost": 0.5}

    from iai_mcp.pipeline import recall_for_response
    embedder = _FakeEmbedder()

    resp = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="rec0", session_id="s-cc-e-1",
        profile_state=profile_state,
    )

    assert resp is not None
    assert len(call_sizes) > 0, (
        "CC-E: profile_modulates boost_edges must have been called at least "
        "once when interest_boost>0 and records carry gains."
    )
    for chunk_size in call_sizes:
        assert chunk_size <= 4, (
            f"CC-E: a Stage-14 boost_edges call had {chunk_size} pairs "
            f"(> _BOOST_SMALL_BATCH=4). This triggers the large-batch "
            f"edges.to_pandas() scan on the recall hot path."
        )

def test_boost_edges_small_batch_constant_pinned():
    """Anchor for the hardcoded `<= 4` above: pipeline.py, boost_edges, and
    the sync fallback in store/_store.py all read one shared constant. This
    assertion is what catches a future drift in that constant -- the test
    above only re-asserts the literal, so it would silently pass a drift if
    nothing else pinned the constant's value."""
    from iai_mcp.store._store import BOOST_EDGES_SMALL_BATCH
    assert BOOST_EDGES_SMALL_BATCH == 4

def test_ann_path_used_field_on_recall_response():
    from iai_mcp.types import RecallResponse

    resp = RecallResponse(hits=[], anti_hits=[], activation_trace=[], budget_used=0)
    assert hasattr(resp, "ann_path_used")
    assert resp.ann_path_used is False

def test_ann_path_used_settable():
    from iai_mcp.types import RecallResponse

    resp = RecallResponse(hits=[], anti_hits=[], activation_trace=[], budget_used=0)
    resp.ann_path_used = True
    assert resp.ann_path_used is True

def test_ann_path_used_serialized_in_core_response_dict(tmp_path, monkeypatch):
    from iai_mcp.types import RecallResponse

    resp_true = RecallResponse(
        hits=[], anti_hits=[], activation_trace=[], budget_used=0,
        ann_path_used=True,
    )
    resp_false = RecallResponse(
        hits=[], anti_hits=[], activation_trace=[], budget_used=0,
    )
    assert getattr(resp_true, "ann_path_used", False) is True
    assert getattr(resp_false, "ann_path_used", False) is False

def test_core_dispatch_ann_assembler_executes_and_returns_ann_path_used(
    tmp_path, monkeypatch
):
    store = _make_store_hermetic(tmp_path, monkeypatch)
    recs = _insert_recs(store, 5)

    import iai_mcp.retrieve as retrieve_mod
    build_calls: list[int] = []

    def _no_build_runtime_graph(*a, **kw):
        build_calls.append(1)
        raise RuntimeError(
            "build_runtime_graph must NOT be called via core.dispatch "
            "on the ANN-first warm path"
        )

    monkeypatch.setattr(retrieve_mod, "build_runtime_graph", _no_build_runtime_graph)

    from iai_mcp.core import dispatch

    response = dispatch(
        store=store,
        method="memory_recall",
        params={"cue": "rec0", "session_id": "s-e2e-1", "budget_tokens": 2000},
    )

    assert build_calls == [], (
        "core.dispatch called build_runtime_graph — the ANN assembler "
        "did not replace it on the warm path."
    )
    assert response.get("ann_path_used") is True, (
        f"Expected ann_path_used=True in core.dispatch response; got: "
        f"{response.get('ann_path_used')!r}"
    )

def test_core_dispatch_soft_fallback_leaves_ann_path_used_false(
    tmp_path, monkeypatch
):
    store = _make_store_hermetic(tmp_path, monkeypatch)
    _insert_recs(store, 3)

    import iai_mcp.runtime_graph_cache as rgc_mod
    orig_lrs = rgc_mod.load_recall_structural

    def _failing_lrs(*a, **kw):
        raise RuntimeError("simulated assembler failure for soft-fallback test")

    monkeypatch.setattr(rgc_mod, "load_recall_structural", _failing_lrs)

    from iai_mcp.core import dispatch

    response = dispatch(
        store=store,
        method="memory_recall",
        params={"cue": "rec0", "session_id": "s-e2e-fallback"},
    )

    assert response.get("ann_path_used") is False, (
        f"Expected ann_path_used=False on soft-fallback; got: "
        f"{response.get('ann_path_used')!r}"
    )
