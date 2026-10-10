"""End-to-end value-lever proof: the shipped salience backfill raises
salience_level on a hermetic corpus, selectively (never touching plain
prose), and the served ranking path reads that mark via the additive
semantic-tier lift -- all through already-shipped machinery, zero new
production code.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from iai_mcp import core
from iai_mcp.migrate._salience_backfill import backfill_salience
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord
from tests._helpers import stub_embedder_for_store


def _unit(i: int) -> list[float]:
    v = [0.0] * EMBED_DIM
    v[i] = 1.0
    return v


class _MappedEmbedder:
    """Deterministic embedder: exact vectors for registered surfaces, a far
    orthogonal vector for anything else."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = dict(mapping)

    def embed(self, text: str) -> list[float]:
        vec = self._mapping.get(text)
        if vec is not None:
            return list(vec)
        fallback = [0.0] * EMBED_DIM
        fallback[EMBED_DIM - 1] = 1.0
        return fallback


def _install_embedder(monkeypatch, mapping: dict[str, list[float]]) -> None:
    stub_embedder_for_store(monkeypatch, _MappedEmbedder(mapping))


def _rec(
    text: str,
    emb: list[float],
    *,
    tier: str = "episodic",
    tags: "list[str] | None" = None,
    directive: bool = False,
    epistemic_status: str = "unknown",
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=emb,
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
        tags=list(tags or []),
        language="en",
        directive=directive,
        epistemic_status=epistemic_status,
        salience_level="unflagged",
    )


def _dispatch(store: MemoryStore, cue: str, cue_vec: list[float]) -> dict:
    resp = core.dispatch(store, "memory_recall", {
        "cue": cue,
        "session_id": "salience-backfill-live-shape-test",
        "budget_tokens": 2000,
        "cue_embedding": cue_vec,
    })
    assert "error" not in resp, f"recall dispatch errored: {resp.get('error')}"
    return resp


def test_backfill_raises_salience_selectively_and_served_score_reflects_it(
    tmp_path, monkeypatch, capsys,
):
    """Backfill moves salience_level off "unflagged" only for records the
    classifier promotes (a directive capture, an existing knowledge summary
    via the cluster-size rule), leaves plain-prose controls untouched, never
    rewrites literal_surface -- and a promoted semantic-tier record's served
    score, read through the real recall dispatch, strictly exceeds an
    otherwise-identical unflagged control at equal cosine, proving the
    additive semantic-tier lift is live on the recall path."""
    directive_text = "deploy the release before the end of the day"
    chatter_text_a = "just checking in, nothing urgent here"
    chatter_text_b = "grabbing coffee, back in ten minutes"
    promoted_summary_text = "alice and bob discussed rollout timing across sessions"
    control_summary_text = "a quiet afternoon summary with nothing notable in it"

    shared_semantic_vec = _unit(0)
    mapping = {
        directive_text: _unit(1),
        chatter_text_a: _unit(2),
        chatter_text_b: _unit(3),
        promoted_summary_text: shared_semantic_vec,
        control_summary_text: shared_semantic_vec,
    }
    _install_embedder(monkeypatch, mapping)

    store = MemoryStore(path=tmp_path)

    directive_row = _rec(directive_text, mapping[directive_text], directive=True)
    chatter_a = _rec(chatter_text_a, mapping[chatter_text_a])
    chatter_b = _rec(chatter_text_b, mapping[chatter_text_b])
    promoted_summary = _rec(
        promoted_summary_text, mapping[promoted_summary_text],
        tier="semantic", tags=["semantic", "cls_summary"],
    )
    control_summary = _rec(
        control_summary_text, mapping[control_summary_text], tier="semantic",
    )
    seeded = [directive_row, chatter_a, chatter_b, promoted_summary, control_summary]
    for r in seeded:
        store.insert(r)
    flush_record_buffer(store)

    literal_before = {r.id: r.literal_surface for r in seeded}
    for r in seeded:
        assert store.get(r.id).salience_level == "unflagged", (
            "every seed row must start unflagged before the sweep runs"
        )

    result = backfill_salience(store, apply=True, store_path=tmp_path)

    before_dist = result["salience_distribution_current"]
    after_dist = result["salience_distribution_resulting"]
    assert before_dist == {"unflagged": 5, "notable": 0, "critical": 0}
    assert after_dist == {"unflagged": 3, "notable": 2, "critical": 0}
    assert result["salience_promoted"] == 2

    assert store.get(directive_row.id).salience_level == "notable", (
        "a directive capture must be promoted by the composite"
    )
    assert store.get(promoted_summary.id).salience_level == "notable", (
        "an existing knowledge summary must be promoted by the cluster-size rule"
    )
    for control in (chatter_a, chatter_b, control_summary):
        assert store.get(control.id).salience_level == "unflagged", (
            f"plain-prose / non-promotable controls must stay unflagged: {control.id}"
        )

    for r in seeded:
        row = store.get(r.id)
        assert row.literal_surface == literal_before[r.id], (
            "the sweep must never rewrite literal_surface"
        )

    resp = _dispatch(store, "value lever proof cue", shared_semantic_vec)
    by_id = {h["record_id"]: h for h in resp["hits"]}
    promoted_hit = by_id.get(str(promoted_summary.id))
    control_hit = by_id.get(str(control_summary.id))
    assert promoted_hit is not None and control_hit is not None, resp["hits"]
    assert promoted_hit["score"] > control_hit["score"], (
        "a backfilled semantic-tier record must score strictly higher than an "
        f"identical-cosine unflagged control: promoted={promoted_hit}, "
        f"control={control_hit}"
    )

    with capsys.disabled():
        print()
        print(f"salience distribution before backfill: {before_dist}")
        print(f"salience distribution after backfill:  {after_dist}")
        print(f"records promoted:                      {result['salience_promoted']}")
        print(f"promoted semantic-tier served score:   {promoted_hit['score']:.6f}")
        print(f"unflagged control served score:        {control_hit['score']:.6f}")
