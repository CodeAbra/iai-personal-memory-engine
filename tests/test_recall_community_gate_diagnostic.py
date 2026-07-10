from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.community import CommunityAssignment
from iai_mcp.graph import MemoryGraph
from iai_mcp.pipeline import (
    COMMUNITY_BIAS_CONCEPT,
    COMMUNITY_BIAS_VERBATIM,
    _gate_bias_for_mode,
    _recall_core,
    recall_for_benchmark,
)
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

def _make(vec: list[float], text: str = "rec") -> MemoryRecord:
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
        tags=[],
        language="en",
    )

def _build_one_record_per_community(
    tmp_path,
    n: int = 50,
) -> tuple[MemoryStore, MemoryGraph, list[MemoryRecord], CommunityAssignment]:
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
            "surface": rec.literal_surface,
            "centrality": 0.0,
            "tier": rec.tier,
            "tags": [],
            "language": "en",
        })

    cids = [uuid4() for _ in recs]
    centroids = {cids[i]: list(recs[i].embedding) for i in range(len(recs))}
    node_to_community = {recs[i].id: cids[i] for i in range(len(recs))}
    mid_regions = {cids[i]: [recs[i].id] for i in range(len(recs))}
    assignment = CommunityAssignment(
        node_to_community=node_to_community,
        community_centroids=centroids,
        modularity=0.0,
        backend="leiden-test-degenerate",
        top_communities=cids[:3],
        mid_regions=mid_regions,
    )
    return store, graph, recs, assignment

def test_records_outside_gated_communities_surface_via_cosine(tmp_path):
    store, graph, recs, assignment = _build_one_record_per_community(tmp_path, n=50)

    cue_vec = [0.0] * EMBED_DIM
    cue_vec[5] = 1.0
    embedder = _FakeEmbedder(vec=cue_vec)

    resp = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="cue at axis 5", session_id="s-gate-diag-1",
        k_hits=10, mode="concept",
    )

    found_ids = {h.record_id for h in resp.hits}
    assert recs[5].id in found_ids, (
        "Gate violation: gold record (cosine 1.0 to cue, on axis 5) "
        "is NOT in top-10 hits. The gate must NEVER filter — only "
        "bias. If this fails, someone re-introduced the old "
        "hard-filter behavior (candidates restricted to top-3 "
        "gated-community members)."
    )
    assert resp.hits[0].record_id == recs[5].id, (
        f"Gold should be top-1 by cosine alone (1.0 vs ~0); "
        f"got {resp.hits[0].record_id} as top hit. Possible cause: "
        "Stage 5 weights were re-tuned, or community-bias scalar is "
        "being applied multiplicatively/subtractively instead of "
        "additively to records inside the gated set."
    )

def test_mode_bias_verbatim_zero_concept_nonzero(tmp_path):
    assert COMMUNITY_BIAS_VERBATIM == 0.0
    assert COMMUNITY_BIAS_CONCEPT == 0.1
    assert _gate_bias_for_mode("verbatim") == 0.0
    assert _gate_bias_for_mode("concept") == 0.1
    assert _gate_bias_for_mode("unknown") == 0.0

    store, graph, recs, assignment = _build_one_record_per_community(tmp_path, n=50)

    cue_vec = [0.0] * EMBED_DIM
    cue_vec[0] = 1.0
    embedder = _FakeEmbedder(vec=cue_vec)

    rec_GATED = recs[0]
    rec_CONTROL = recs[20]

    result_v = _recall_core(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="cue at axis 0", session_id="s-mode-bias-v",
        mode="verbatim",
    )

    result_c = _recall_core(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="cue at axis 0", session_id="s-mode-bias-c",
        mode="concept",
    )

    verbatim_ids = {h.record_id for h in result_v.scored_hits}
    concept_ids = {h.record_id for h in result_c.scored_hits}
    assert verbatim_ids == concept_ids, (
        "Gate fence: gate must NEVER filter; mode change should not "
        f"alter the record list. verbatim_only={verbatim_ids - concept_ids}, "
        f"concept_only={concept_ids - verbatim_ids}"
    )

    v_gated = next(h for h in result_v.scored_hits if h.record_id == rec_GATED.id)
    c_gated = next(h for h in result_c.scored_hits if h.record_id == rec_GATED.id)
    v_ctrl = next(h for h in result_v.scored_hits if h.record_id == rec_CONTROL.id)
    c_ctrl = next(h for h in result_c.scored_hits if h.record_id == rec_CONTROL.id)

    cos_GATED = 1.0
    expected_bonus = COMMUNITY_BIAS_CONCEPT * cos_GATED
    delta_gated = c_gated.score - v_gated.score
    assert delta_gated == pytest.approx(expected_bonus, abs=1e-4), (
        f"Concept mode: GATED record (rec[0], cosine={cos_GATED}) "
        f"should gain ~{expected_bonus:.4f} from "
        f"COMMUNITY_BIAS_CONCEPT * cos when transitioning verbatim -> "
        f"concept. Got delta = c_gated.score - v_gated.score = "
        f"{delta_gated:.4f}.\n"
        f"v_gated.score = {v_gated.score:.4f}; "
        f"c_gated.score = {c_gated.score:.4f}."
    )

    delta_ctrl = c_ctrl.score - v_ctrl.score
    assert delta_ctrl == pytest.approx(0.0, abs=1e-6), (
        f"Concept mode: CONTROL record (rec[20], cosine 0 to cue, "
        f"in non-gated community c20) must NOT receive the community "
        f"bias. Got delta = c_ctrl.score - v_ctrl.score = {delta_ctrl:.6f}; "
        f"expected 0.0.\n"
        f"v_ctrl.score = {v_ctrl.score:.6f}; "
        f"c_ctrl.score = {c_ctrl.score:.6f}."
    )

    from iai_mcp.pipeline import W_COSINE
    expected_verbatim_delta = W_COSINE * (1.0 - 0.0)
    actual_verbatim_delta = v_gated.score - v_ctrl.score
    assert actual_verbatim_delta == pytest.approx(
        expected_verbatim_delta, abs=1e-4
    ), (
        f"Verbatim mode: gated vs control score delta should be "
        f"W_COSINE * cos_diff = {W_COSINE} * 1.0 = {expected_verbatim_delta:.4f} "
        f"with NO community-bias contribution. Got delta = "
        f"{actual_verbatim_delta:.4f}. If this differs, either "
        f"COMMUNITY_BIAS_VERBATIM is non-zero, or the mode dispatch "
        f"in _gate_bias_for_mode is broken."
    )
