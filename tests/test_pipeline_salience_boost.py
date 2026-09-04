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

from iai_mcp import pipeline
from iai_mcp.community import CommunityAssignment
from iai_mcp.embed import Embedder
from iai_mcp.graph import MemoryGraph
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord, SALIENCE_LEVEL_RANK


def _random_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _mk_rec(text: str, embedding: list[float], salience_level: str = "unflagged") -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
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
