"""Every rank term must be provably ALIVE on production-shaped input.

A formula term that cannot move any live ranking is silent dead weight
in the signal budget. Each test builds a minimal pair where exactly one
signal differs and asserts the gap is a material fraction of that term's
expected contribution — bare ordering would pass on float noise.

Spread-activation liveness is exercised by test_full_recall_spread_profile
(it needs the entity-anchor provenance pipeline); community liveness by
the reason test in test_rank_score_honesty.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import numpy as np

from iai_mcp.community import CommunityAssignment
from iai_mcp.graph import MemoryGraph
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

from tests.test_recall_core_unit import _FakeEmbedder

CUE_VEC = [1.0] + [0.0] * (EMBED_DIM - 1)


def _unit(i: int) -> list[float]:
    v = [0.0] * EMBED_DIM
    v[i % EMBED_DIM] = 1.0
    return v


def _build(tmp_path, specs: list[dict]):
    """Insert records exactly as specified — embeddings chosen BEFORE
    insert, one graph build, no post-hoc mutation of the pool."""
    store = MemoryStore(path=tmp_path / "store")
    recs: list[MemoryRecord] = []
    now = datetime.now(timezone.utc)
    for i, spec in enumerate(specs):
        rec = MemoryRecord(
            id=uuid4(), tier="episodic",
            literal_surface=spec.get("text", f"liveness record {i}"),
            aaak_index=spec.get("aaak", ""),
            embedding=list(spec["vec"]),
            community_id=None, centrality=0.0, detail_level=2,
            pinned=False, stability=spec.get("stability", 0.5),
            difficulty=0.0, last_reviewed=None,
            never_decay=False, never_merge=False, provenance=[],
            created_at=spec.get("created_at", now), updated_at=now,
            tags=list(spec.get("tags", [])), language="en",
        )
        store.insert(rec)
        recs.append(rec)
    graph = MemoryGraph()
    for rec in recs:
        graph.add_node(rec.id, community_id=None, embedding=list(rec.embedding))
        graph.set_node_payload(rec.id, {
            "embedding": list(rec.embedding),
            "surface": rec.literal_surface,
            "centrality": 0.0,
            "tier": rec.tier,
            "tags": list(rec.tags),
            "language": "en",
            "aaak_index": rec.aaak_index,
            "created_at": rec.created_at.isoformat(),
            "stability": rec.stability,
        })
    cid = uuid4()
    assignment = CommunityAssignment(
        node_to_community={r.id: cid for r in recs},
        community_centroids={cid: list(CUE_VEC)},
        modularity=0.0, backend="flat",
        top_communities=[cid],
        mid_regions={cid: [r.id for r in recs]},
    )
    return store, graph, recs, assignment


def _scores(tmp_path, specs, cue="liveness probe"):
    from iai_mcp.pipeline import _recall_core

    store, graph, recs, assignment = _build(tmp_path, specs)
    result = _recall_core(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(vec=list(CUE_VEC)),
        cue=cue, session_id="s-live",
    )
    return recs, {h.record_id: h.score for h in result.scored_hits}, store, graph


def test_cosine_is_alive(tmp_path):
    specs = [{"vec": CUE_VEC}] + [{"vec": _unit(i)} for i in range(1, 6)]
    recs, scores, _, _ = _scores(tmp_path, specs)
    aligned = recs[0]
    others = [s for rid, s in scores.items() if rid != aligned.id]
    assert scores[aligned.id] - max(others) > 0.5, (
        "a record aligned with the cue must outrank orthogonal ones by a "
        "material cosine margin"
    )


def test_aaak_is_alive(tmp_path):
    specs = [
        {"vec": CUE_VEC, "aaak": "W:E/R:abc/E:liveness/T:capture",
         "text": "anchored twin"},
        {"vec": CUE_VEC, "text": "plain twin"},
    ] + [{"vec": _unit(i)} for i in range(1, 5)]
    recs, scores, _, _ = _scores(tmp_path, specs)
    # Anchor overlap: cue {liveness, probe} vs entity {liveness} = 0.5
    # Jaccard; the gap must be a material fraction of W_AAAK * 0.5 = 0.15.
    assert scores[recs[0].id] - scores[recs[1].id] > 0.1, (
        "an entity anchor matching the cue must lift an otherwise "
        f"identical record by ~W_AAAK*overlap; gap "
        f"{scores[recs[0].id] - scores[recs[1].id]:.4f} — the aaak term "
        "has gone dead again"
    )


def test_degree_is_alive_when_cosine_discriminates(tmp_path):
    # A discriminating pool: distinct real similarities, head spread far
    # above the flat-cosine guard, plus one identical-embedding pair that
    # differs only in earned connectivity.
    rng = np.random.default_rng(3)
    shared = np.asarray(CUE_VEC) + rng.normal(scale=0.05, size=EMBED_DIM)
    shared = (shared / np.linalg.norm(shared)).tolist()
    specs = [{"vec": shared, "text": "hub twin"}, {"vec": shared, "text": "leaf twin"}]
    for i in range(2, 10):
        v = np.asarray(CUE_VEC).copy()
        v[i] += 0.15 * i
        specs.append({"vec": (v / np.linalg.norm(v)).tolist()})
    from iai_mcp.pipeline import _recall_core

    store, graph, recs, assignment = _build(tmp_path, specs)
    for other in recs[2:8]:
        store.boost_edges([(recs[0].id, other.id)], edge_type="hebbian", delta=1.0)
        graph.add_edge(recs[0].id, other.id, edge_type="hebbian", weight=1.0)
    result = _recall_core(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(vec=list(CUE_VEC)),
        cue="liveness probe", session_id="s-live-deg",
    )
    scores = {h.record_id: h.score for h in result.scored_hits}
    # The hub's deg_norm is ~1.0 vs the leaf's 0.0; the gap must be a
    # material fraction of the degree weight (0.1 before literal scaling).
    assert scores[recs[0].id] - scores[recs[1].id] > 0.05, (
        "with cosine discriminating, earned connectivity must still lift "
        f"an otherwise identical record; gap "
        f"{scores[recs[0].id] - scores[recs[1].id]:.4f}"
    )


def test_age_is_alive(tmp_path):
    old_ts = datetime.now(timezone.utc) - timedelta(days=400)
    specs = [
        {"vec": CUE_VEC, "text": "fresh twin"},
        {"vec": CUE_VEC, "text": "old twin", "created_at": old_ts},
    ] + [{"vec": _unit(i)} for i in range(1, 5)]
    recs, scores, _, _ = _scores(tmp_path, specs)
    # A 400-day-old record pays the full age penalty W_AGE * 1.0 = 0.05.
    assert scores[recs[0].id] - scores[recs[1].id] > 0.03, (
        "an old record must pay a material age penalty against a fresh "
        f"twin; gap {scores[recs[0].id] - scores[recs[1].id]:.4f}"
    )
