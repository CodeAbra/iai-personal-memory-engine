from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.events import query_events
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

def _rec(
    *,
    text: str = "t",
    tags: list[str] | None = None,
    language: str = "en",
    tier: str = "episodic",
    detail_level: int = 2,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=[1.0] + [0.0] * (EMBED_DIM - 1),
        community_id=None,
        centrality=0.0,
        detail_level=detail_level,
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
        language=language,
    )

@pytest.fixture(autouse=True)
def _patch_embedder(monkeypatch):
    from iai_mcp import embed as embed_mod

    class _FakeEmbedder:
        DIM = EMBED_DIM
        DEFAULT_DIM = EMBED_DIM
        DEFAULT_MODEL_KEY = "fake"

        def __init__(self, *args, **kwargs):
            self.DIM = EMBED_DIM

        def embed(self, text: str) -> list[float]:
            return [1.0] + [0.0] * (EMBED_DIM - 1)

        def embed_batch(self, texts):
            return [self.embed(t) for t in texts]

    monkeypatch.setattr(embed_mod, "Embedder", _FakeEmbedder)
    yield

def test_schema_d21_thresholds_encoded():
    from iai_mcp import schema

    assert schema.AUTO_INDUCT_COOCCURRENCE == 5
    assert schema.AUTO_INDUCT_CONFIDENCE == 0.85
    assert schema.USER_APPROVAL_COOCCURRENCE == 3
    assert schema.USER_APPROVAL_CONFIDENCE == 0.65

def test_induce_schemas_tier0_returns_candidates_at_threshold(tmp_path):
    from iai_mcp.schema import induce_schemas_tier0

    store = MemoryStore(path=tmp_path)
    for i in range(10):
        store.insert(_rec(text=f"r{i}", tags=["meeting", "notes"]))
    candidates = induce_schemas_tier0(store)
    assert len(candidates) >= 1
    hit = [c for c in candidates if c.evidence_count >= 5 and c.confidence >= 0.85]
    assert len(hit) >= 1
    assert hit[0].status == "auto"

def test_induce_schemas_tier0_threshold_lowered_requires_approval(tmp_path):
    from iai_mcp.schema import induce_schemas_tier0

    store = MemoryStore(path=tmp_path)
    for i in range(4):
        store.insert(_rec(text=f"r{i}", tags=["report", "deadline"]))
    candidates = induce_schemas_tier0(store)
    match = [c for c in candidates if c.evidence_count == 4]
    auto_hits = [c for c in candidates if c.status == "auto"]
    assert len(auto_hits) == 0

def test_induce_schemas_tier0_discards_below_threshold(tmp_path):
    from iai_mcp.schema import induce_schemas_tier0

    store = MemoryStore(path=tmp_path)
    for i in range(2):
        store.insert(_rec(text=f"r{i}", tags=["alpha", "beta"]))
    candidates = induce_schemas_tier0(store)
    assert len(candidates) == 0

def test_induce_schemas_tier0_no_llm_call(tmp_path, monkeypatch):
    from iai_mcp.schema import induce_schemas_tier0

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    store = MemoryStore(path=tmp_path)
    for i in range(3):
        store.insert(_rec(text=f"r{i}", tags=["work", "design"]))
    candidates = induce_schemas_tier0(store)
    assert isinstance(candidates, list)

def test_induce_schemas_tier1_falls_back_on_guard_block(tmp_path, monkeypatch):
    from iai_mcp.guard import BudgetLedger, RateLimitLedger
    from iai_mcp.schema import induce_schemas_tier1

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    store = MemoryStore(path=tmp_path)
    for i in range(5):
        store.insert(_rec(text=f"r{i}", tags=["project", "meeting"]))

    budget = BudgetLedger(store)
    rate = RateLimitLedger(store)
    candidates = induce_schemas_tier1(
        store, budget=budget, rate=rate, llm_enabled=False,
    )
    assert isinstance(candidates, list)
    events = query_events(store, kind="llm_health")
    matching = [e for e in events if e["data"].get("component") == "schema_induction"]
    assert len(matching) >= 1

def test_persist_schema_creates_semantic_record(tmp_path):
    from iai_mcp.schema import SchemaCandidate, persist_schema

    store = MemoryStore(path=tmp_path)
    ev_recs = [_rec(text=f"ev{i}", tags=["meeting", "notes"]) for i in range(3)]
    for r in ev_recs:
        store.insert(r)

    cand = SchemaCandidate(
        pattern="tags:meeting+notes",
        confidence=0.88,
        evidence_count=3,
        evidence_ids=[r.id for r in ev_recs],
        status="auto",
    )
    schema_id = persist_schema(store, cand)

    schema_rec = store.get(schema_id)
    assert schema_rec is not None
    assert schema_rec.tier == "semantic"
    assert schema_rec.detail_level == 3
    assert schema_rec.never_decay is True

def test_persist_schema_creates_schema_instance_of_edges(tmp_path):
    from iai_mcp.schema import SchemaCandidate, persist_schema
    from iai_mcp.store import EDGES_TABLE

    store = MemoryStore(path=tmp_path)
    ev_recs = [_rec(text=f"ev{i}", tags=["m", "n"]) for i in range(3)]
    for r in ev_recs:
        store.insert(r)

    cand = SchemaCandidate(
        pattern="tags:m+n",
        confidence=0.9,
        evidence_count=3,
        evidence_ids=[r.id for r in ev_recs],
        status="auto",
    )
    schema_id = persist_schema(store, cand)

    edges_df = store.db.open_table(EDGES_TABLE).to_pandas()
    sio = edges_df[edges_df["edge_type"] == "schema_instance_of"]
    assert len(sio) == 3

def test_provisional_schemas_for_recall_returns_hint(tmp_path):
    from iai_mcp.schema import provisional_schemas_for_recall

    store = MemoryStore(path=tmp_path)
    recs = [_rec(text=f"r{i}", tags=["meeting", "notes"]) for i in range(3)]
    for r in recs:
        store.insert(r)

    class _Hit:
        def __init__(self, rid, score):
            self.record_id = rid
            self.score = score

    hits = [_Hit(recs[i].id, 0.3) for i in range(3)]
    provisionals = provisional_schemas_for_recall(store, hits, entropy_bits=1.5)
    assert isinstance(provisionals, list)
    assert any(p.get("kind") == "provisional_schema" for p in provisionals)

def test_provisional_schemas_below_entropy_empty(tmp_path):
    from iai_mcp.schema import provisional_schemas_for_recall

    store = MemoryStore(path=tmp_path)
    assert provisional_schemas_for_recall(store, [], entropy_bits=0.5) == []

def test_autistic_threshold_stricter_than_nt():
    from iai_mcp.schema import (
        AUTO_INDUCT_COOCCURRENCE,
        AUTO_INDUCT_CONFIDENCE,
        USER_APPROVAL_COOCCURRENCE,
        USER_APPROVAL_CONFIDENCE,
    )

    assert AUTO_INDUCT_COOCCURRENCE >= 5
    assert AUTO_INDUCT_CONFIDENCE >= 0.85
    assert USER_APPROVAL_COOCCURRENCE == 3
    assert USER_APPROVAL_CONFIDENCE == 0.65


def test_user_approval_tier_is_reachable(tmp_path):
    """Moderate co-occurrence (3-4) must land in pending_user_approval, and
    the band between approval and auto (5-8) must not vanish: confidence is
    count/10, so a confidence gate re-checked inside the approval band could
    never pass and the tier was dead code."""
    from iai_mcp.lilli.cycle.schema import induce_schemas_tier0
    from iai_mcp.store import flush_record_buffer

    store = MemoryStore(path=tmp_path)
    try:
        for _ in range(3):
            store.insert(_rec(text="approval band row", tags=["alpha", "beta"]))
        for _ in range(6):
            store.insert(_rec(text="gap band row", tags=["gamma", "delta"]))
        for _ in range(10):
            store.insert(_rec(text="auto band row", tags=["epsi", "zeta"]))
        flush_record_buffer(store)

        by_pattern = {
            c.pattern: c.status for c in induce_schemas_tier0(store)
        }
        assert by_pattern.get("tags:alpha+beta") == "pending_user_approval"
        assert by_pattern.get("tags:delta+gamma") == "pending_user_approval", (
            "counts between the approval and auto bands must reach the human, "
            "not vanish"
        )
        assert by_pattern.get("tags:epsi+zeta") == "auto"
    finally:
        store.close()

def test_idem_tags_never_form_patterns(tmp_path):
    """idem: hashes are per-record-unique; a pattern containing one is noise
    and must never surface as a candidate, even when the same idem tag
    somehow repeats past the auto threshold."""
    from iai_mcp.schema import induce_schemas_tier0

    store = MemoryStore(path=tmp_path)
    shared_idem = "idem:" + "a" * 64
    for i in range(10):
        store.insert(
            _rec(text=f"r{i}", tags=["capture", "role:user", shared_idem])
        )
    candidates = induce_schemas_tier0(store)
    assert candidates, "the legit capture+role pattern must still surface"
    for cand in candidates:
        assert "idem:" not in cand.pattern, (
            f"idem hash leaked into a schema pattern: {cand.pattern}"
        )
