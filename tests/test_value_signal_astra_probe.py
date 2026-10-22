"""Permanent, deterministic regression guard for the recall surface's
working named-entity cue: models the shape of the shipped
knowledge-vs-chatter fixture's floor-drop pair (synthetic knowledge, a
register-matched but uninformative chatter turn, a small-talk query) with a
controlled embedder so the probe always runs, never skips on missing bench
artifacts, and is fully deterministic. `rank_boost` is shared between this
phase and later ranking work -- this guard must stay green through any
change to that module.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from iai_mcp import core
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord
from tests._helpers import stub_embedder_for_store

_KNOWLEDGE_TEXT = (
    "alice's rollout notes: the Meridian planner now ships with an extended "
    "context window for large repositories and a stricter function-schema "
    "validation pass."
)
_CHATTER_TEXT = (
    "yeah so anyway, nothing much new on my end, just been keeping busy with "
    "the usual stuff honestly"
)
_QUERY_TEXT = "so hey, anything new on your end lately, just checking in on how things are going"


def _unit(i: int) -> "list[float]":
    v = [0.0] * EMBED_DIM
    v[i] = 1.0
    return v


def _mix(i: int, j: int, cos_to_i: float) -> "list[float]":
    import math
    v = [0.0] * EMBED_DIM
    v[i] = cos_to_i
    v[j] = math.sqrt(1.0 - cos_to_i * cos_to_i)
    return v


class _MappedEmbedder:
    """Deterministic embedder: exact vectors for registered surfaces, a far
    orthogonal vector for anything else -- no bench artifacts or model
    caches needed, so this probe never skips."""

    def __init__(self, mapping: "dict[str, list[float]]") -> None:
        self._mapping = dict(mapping)

    def embed(self, text: str) -> "list[float]":
        vec = self._mapping.get(text)
        if vec is not None:
            return list(vec)
        fallback = [0.0] * EMBED_DIM
        fallback[EMBED_DIM - 1] = 1.0
        return fallback


def _rec(text: str, emb: "list[float]", *, tier: str = "episodic") -> MemoryRecord:
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
        tags=[],
        language="en",
        salience_level="unflagged",
    )


def test_named_entity_knowledge_surfaces_over_registered_chatter(tmp_path, monkeypatch):
    """Deterministic, always-running guard: a named-entity knowledge record
    must surface ahead of register-matched, register-matched-but-empty
    chatter on the real `memory_recall` dispatch path -- the shared
    recall/`rank_boost` surface must not regress the one cue that already
    works."""
    query_vec = _unit(0)
    knowledge_vec = _mix(0, 1, 0.9)
    chatter_vec = _mix(0, 2, 0.05)
    mapping = {
        _KNOWLEDGE_TEXT: knowledge_vec,
        _CHATTER_TEXT: chatter_vec,
        _QUERY_TEXT: query_vec,
    }
    stub_embedder_for_store(monkeypatch, _MappedEmbedder(mapping))

    store = MemoryStore(path=tmp_path)
    knowledge = _rec(_KNOWLEDGE_TEXT, knowledge_vec, tier="semantic")
    chatter = _rec(_CHATTER_TEXT, chatter_vec)
    store.insert(knowledge)
    store.insert(chatter)
    flush_record_buffer(store)

    resp = core.dispatch(store, "memory_recall", {
        "cue": _QUERY_TEXT,
        "session_id": "astra-probe-test",
        "budget_tokens": 2000,
        "cue_embedding": query_vec,
    })
    assert "error" not in resp, f"recall dispatch errored: {resp.get('error')}"

    hit_ids = [h["record_id"] for h in resp["hits"]]
    assert str(knowledge.id) in hit_ids, (
        f"named-entity knowledge record must surface in memory_recall hits: {hit_ids}"
    )
    knowledge_rank = hit_ids.index(str(knowledge.id))
    if str(chatter.id) in hit_ids:
        chatter_rank = hit_ids.index(str(chatter.id))
        assert knowledge_rank < chatter_rank, (
            f"knowledge must be served ahead of register-matched chatter: "
            f"knowledge_rank={knowledge_rank}, chatter_rank={chatter_rank}, hits={hit_ids}"
        )
