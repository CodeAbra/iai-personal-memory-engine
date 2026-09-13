"""Rank-fusion proof for the salience_level boost: a higher level ranks
strictly above a lower one, and above unflagged, at equal cosine.

Three fixtures share the SAME embedding vector (sidesteps cosine-parity
fragility) and are otherwise identical -- differing only in salience_level.
The control run (IAI_MCP_SALIENCE_BOOST=0) is the mutation check: it must
tie the three scores exactly, proving every non-salience term is neutral
and the boosted run's strict ordering comes from the multiplier alone.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp import pipeline, rank_boost
from iai_mcp.community import CommunityAssignment
from iai_mcp.embed import Embedder
from iai_mcp.graph import MemoryGraph
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord, SALIENCE_LEVEL_RANK


def _random_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _mk_rec(
    text: str, embedding: list[float], salience_level: str = "unflagged", tier: str = "episodic",
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=embedding,
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
        salience_level=salience_level,
    )


def _pool_graph(records: list[MemoryRecord]) -> MemoryGraph:
    # Nodes carry only an embedding, deliberately never a "surface" payload
    # key -- this keeps the scoring-loop candidate view resolved from the
    # real store records (with salience_level readable), not a graph-payload
    # projection that would silently default the field.
    graph = MemoryGraph()
    for rec in records:
        graph.add_node(rec.id, None, rec.embedding)
    return graph


def test_salience_boost_orders_records_strictly_and_ties_at_control(tmp_path, monkeypatch):
    store = MemoryStore(path=tmp_path / "salience-boost-store")
    embedder = Embedder()
    target_vec = list(embedder.embed("a decision worth remembering clearly"))
    shared_text = "the deployment decision alice made this morning"

    unflagged = _mk_rec(shared_text, target_vec, "unflagged")
    notable = _mk_rec(shared_text, target_vec, "notable")
    critical = _mk_rec(shared_text, target_vec, "critical")
    for rec in (unflagged, notable, critical):
        store.insert(rec)

    fillers = [_mk_rec(f"unrelated filler record {i}", _random_vec(4000 + i)) for i in range(12)]
    for f in fillers:
        store.insert(f)

    graph = _pool_graph([unflagged, notable, critical, *fillers])
    assignment = CommunityAssignment()
    target_ids = {unflagged.id, notable.id, critical.id}

    pipeline._last_recall_latency_ms = 0.0
    monkeypatch.setenv("IAI_MCP_SALIENCE_BOOST", "0")
    control = pipeline.recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=[],
        embedder=embedder, cue="an unrelated grocery list cue phrase",
        session_id="s1", budget_tokens=4000, mode="concept",
        cue_embedding=target_vec,
    )
    control_scores = {h.record_id: h.score for h in control.hits if h.record_id in target_ids}
    assert len(control_scores) == 3, (
        f"expected all 3 salience fixtures to surface as hits; got {control_scores} "
        f"from {[(h.record_id, h.reason) for h in control.hits]}"
    )
    # float32 cosine reconstruction carries ~1e-10 noise even for bytewise
    # identical embeddings -- the tie is proven within a tolerance far below
    # any real scoring signal (the boosted run's separation is ~5% per rank).
    control_values = list(control_scores.values())
    assert max(control_values) - min(control_values) < 1e-6, (
        "control run (IAI_MCP_SALIENCE_BOOST=0) must tie (within float32 noise) -- "
        f"this pins the no-boost baseline; got {control_scores}"
    )

    monkeypatch.delenv("IAI_MCP_SALIENCE_BOOST", raising=False)
    pipeline._last_recall_latency_ms = 0.0
    boosted = pipeline.recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=[],
        embedder=embedder, cue="an unrelated grocery list cue phrase",
        session_id="s1", budget_tokens=4000, mode="concept",
        cue_embedding=target_vec,
    )
    boosted_scores = {h.record_id: h.score for h in boosted.hits if h.record_id in target_ids}
    assert len(boosted_scores) == 3, boosted_scores
    assert boosted_scores[critical.id] > boosted_scores[notable.id] > boosted_scores[unflagged.id], (
        f"expected strict critical > notable > unflagged ordering at equal cosine "
        f"under the default (non-zero) boost env, got {boosted_scores}"
    )


def _pool_graph_with_surface(records: list[MemoryRecord]) -> MemoryGraph:
    """Inverse of `_pool_graph`: every node carries a "surface" payload key,
    so the scoring loop's candidate resolves as a `SimpleRecordView`
    (graph-payload projection) instead of the store's own `MemoryRecord` --
    the record-view class whose salience_level the scoring loop must resolve
    correctly for the salience boost to fire."""
    graph = MemoryGraph()
    for rec in records:
        graph.add_node(rec.id, None, rec.embedding)
        graph.set_node_payload(
            rec.id,
            {
                "embedding": rec.embedding,
                "surface": rec.literal_surface,
                "centrality": 0.0,
                "tier": rec.tier,
                "tags": [],
                "language": "en",
                "aaak_index": "",
                "created_at": rec.created_at.isoformat(),
                "stability": 0.5,
            },
        )
    return graph


@pytest.mark.parametrize("use_rust_scorer", [True, False])
def test_salience_boost_survives_graph_payload_candidate_view(tmp_path, monkeypatch, use_rust_scorer):
    """Graph-payload-candidate regression: when the scoring loop's candidate
    resolves as a SimpleRecordView (graph payload carries "surface"), the
    pre-existing salience_multiplier boost must still fire -- on both the
    Rust-scorer winners-tuple path (use_rust_scorer=True) and the non-Rust /
    kill-switch fallback (use_rust_scorer=False)."""
    store = MemoryStore(path=tmp_path / f"salience-graph-view-store-{use_rust_scorer}")
    embedder = Embedder()
    target_vec = list(embedder.embed("a decision worth remembering clearly"))
    shared_text = "the deployment decision alice made this morning"

    unflagged = _mk_rec(shared_text, target_vec, "unflagged")
    notable = _mk_rec(shared_text, target_vec, "notable")
    for rec in (unflagged, notable):
        store.insert(rec)
    fillers = [_mk_rec(f"unrelated filler record {i}", _random_vec(5000 + i)) for i in range(12)]
    for f in fillers:
        store.insert(f)

    graph = _pool_graph_with_surface([unflagged, notable, *fillers])
    assignment = CommunityAssignment()

    # Control run, mirroring test_salience_boost_orders_records_strictly_and_
    # ties_at_control's discipline: with the boost env zeroed, both fixtures
    # must tie within float32 noise -- this rules out a difference in the
    # SEPARATION check below being an artifact of embedding-reconstruction
    # jitter rather than the salience_multiplier actually firing.
    pipeline._last_recall_latency_ms = 0.0
    monkeypatch.setenv("IAI_MCP_SALIENCE_BOOST", "0")
    control = pipeline.recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=[],
        embedder=embedder, cue="an unrelated grocery list cue phrase",
        session_id="s1", budget_tokens=4000, mode="concept",
        cue_embedding=target_vec, use_rust_scorer=use_rust_scorer,
    )
    control_scores = {h.record_id: h.score for h in control.hits if h.record_id in (unflagged.id, notable.id)}
    assert len(control_scores) == 2, (
        f"expected both salience fixtures to surface as hits (use_rust_scorer="
        f"{use_rust_scorer}); got {control_scores} from "
        f"{[(h.record_id, h.reason) for h in control.hits]}"
    )
    assert abs(control_scores[notable.id] - control_scores[unflagged.id]) < 1e-6, (
        f"control run (IAI_MCP_SALIENCE_BOOST=0) must tie within float32 noise "
        f"(use_rust_scorer={use_rust_scorer}): {control_scores}"
    )

    monkeypatch.delenv("IAI_MCP_SALIENCE_BOOST", raising=False)
    pipeline._last_recall_latency_ms = 0.0
    boosted = pipeline.recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=[],
        embedder=embedder, cue="an unrelated grocery list cue phrase",
        session_id="s1", budget_tokens=4000, mode="concept",
        cue_embedding=target_vec, use_rust_scorer=use_rust_scorer,
    )
    boosted_scores = {h.record_id: h.score for h in boosted.hits if h.record_id in (unflagged.id, notable.id)}
    assert len(boosted_scores) == 2, boosted_scores
    # Margin well above the ~1e-6 float noise floor pinned by the control
    # run above, so this cannot pass by embedding-reconstruction jitter.
    assert boosted_scores[notable.id] > boosted_scores[unflagged.id] + 1e-4, (
        f"salience_multiplier must fire for a SimpleRecordView candidate "
        f"(use_rust_scorer={use_rust_scorer}): {boosted_scores}"
    )


@pytest.mark.parametrize("use_rust_scorer", [True, False])
def test_salience_tier_lift_and_multiplier_share_one_source(tmp_path, monkeypatch, use_rust_scorer):
    """tier_multiplier's semantic_lift term and salience_multiplier must both
    read salience_level off the same candidate field. A semantic-tier
    flagged candidate's boost separation must exceed an episodic-tier
    flagged candidate's by roughly the live semantic_lift-driven ratio --
    if the two terms ever resolved salience_level from independent sources
    again, a divergence between those sources would collapse this
    separation even though episodic-only salience_multiplier still fires.
    """
    store = MemoryStore(path=tmp_path / f"salience-tier-source-store-{use_rust_scorer}")
    embedder = Embedder()
    target_vec = list(embedder.embed("a decision worth remembering clearly"))
    shared_text = "the deployment decision alice made this morning"

    episodic_unflagged = _mk_rec(shared_text, target_vec, "unflagged", tier="episodic")
    episodic_critical = _mk_rec(shared_text, target_vec, "critical", tier="episodic")
    semantic_unflagged = _mk_rec(shared_text, target_vec, "unflagged", tier="semantic")
    semantic_critical = _mk_rec(shared_text, target_vec, "critical", tier="semantic")
    fixtures = [episodic_unflagged, episodic_critical, semantic_unflagged, semantic_critical]
    for rec in fixtures:
        store.insert(rec)
    fillers = [_mk_rec(f"unrelated filler record {i}", _random_vec(6000 + i)) for i in range(12)]
    for f in fillers:
        store.insert(f)

    graph = _pool_graph_with_surface([*fixtures, *fillers])
    assignment = CommunityAssignment()

    pipeline._last_recall_latency_ms = 0.0
    result = pipeline.recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=[],
        embedder=embedder, cue="an unrelated grocery list cue phrase",
        session_id="s1", budget_tokens=4000, mode="concept",
        cue_embedding=target_vec, use_rust_scorer=use_rust_scorer,
    )
    fixture_ids = {r.id for r in fixtures}
    scores = {h.record_id: h.score for h in result.hits if h.record_id in fixture_ids}
    assert len(scores) == 4, (
        f"expected all 4 tier/salience fixtures to surface as hits "
        f"(use_rust_scorer={use_rust_scorer}); got {scores}"
    )

    episodic_delta = scores[episodic_critical.id] - scores[episodic_unflagged.id]
    semantic_delta = scores[semantic_critical.id] - scores[semantic_unflagged.id]
    assert episodic_delta > 0.0 and semantic_delta > 0.0, (
        f"salience_multiplier must raise both tiers' flagged score above their "
        f"unflagged sibling (use_rust_scorer={use_rust_scorer}): "
        f"episodic_delta={episodic_delta}, semantic_delta={semantic_delta}"
    )

    # Expected separation ratio computed from the live rank_boost constants
    # (never hardcoded), on shipped defaults (literal_preservation="strong",
    # no env overrides): the episodic pair's delta is driven by
    # salience_multiplier alone, the semantic pair's by tier_multiplier's
    # additive semantic_lift term stacked on top of the same
    # salience_multiplier -- both from ONE salience_level read.
    salience_only = rank_boost.salience_multiplier(
        salience_level="critical", salience_step=rank_boost.SALIENCE_BOOST_STEP_DEFAULT,
    )
    semantic_and_salience = rank_boost.tier_multiplier(
        tier="semantic", tags=[], tier_boost=rank_boost.TIER_KNOWLEDGE_BOOST_DEFAULT,
        literal_preservation_strong=True, salience_level="critical",
        semantic_lift=rank_boost.SEMANTIC_VALUE_LIFT_DEFAULT,
    ) * salience_only
    expected_ratio = (semantic_and_salience - 1.0) / (salience_only - 1.0)
    assert expected_ratio > 1.0, "fixture must exercise tier_multiplier's semantic_lift branch"
    # Half of the live-computed ratio as a floor tolerates base-score noise
    # between the episodic and semantic candidate pairs while still failing
    # hard if the semantic_lift term stops firing off the same source.
    assert semantic_delta > episodic_delta * expected_ratio * 0.5, (
        f"semantic-tier boost separation must exceed episodic-tier separation "
        f"by roughly the live semantic_lift ratio ({expected_ratio:.3f}) -- both "
        f"boost terms must read salience_level from the same source "
        f"(use_rust_scorer={use_rust_scorer}): episodic_delta={episodic_delta}, "
        f"semantic_delta={semantic_delta}"
    )


def test_salience_boost_step_multiplier_monotonic_across_env_magnitudes(monkeypatch):
    from iai_mcp.pipeline import _salience_boost_step

    for env_value in ("0.05", "0.2", "1.0"):
        monkeypatch.setenv("IAI_MCP_SALIENCE_BOOST", env_value)
        step = _salience_boost_step()
        assert step >= 0.0, f"_salience_boost_step must never be negative, got {step}"
        multipliers = [
            1.0 + SALIENCE_LEVEL_RANK[level] * step
            for level in ("unflagged", "notable", "critical")
        ]
        assert multipliers == sorted(multipliers), (
            f"multiplier must never decrease with rank at step={step}: {multipliers}"
        )
        if step > 0.0:
            assert multipliers[0] < multipliers[1] < multipliers[2], (
                f"a positive step must strictly separate every rank: {multipliers}"
            )
