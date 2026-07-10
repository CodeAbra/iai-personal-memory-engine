from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4


from iai_mcp.schema import SchemaCandidate, persist_schema
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

def _rec(*, language: str, text: str = "seed") -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[1.0] + [0.0] * (EMBED_DIM - 1),
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.5,
        difficulty=0.3,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language=language,
    )

def _seed_cluster(
    store: MemoryStore,
    lang_counts: dict[str, int],
) -> list[uuid4]:
    evidence: list = []
    for lang, count in lang_counts.items():
        for i in range(count):
            r = _rec(language=lang, text=f"{lang}_seed_{i}")
            store.insert(r)
            evidence.append(r.id)
    return evidence

def test_persist_schema_derives_language_from_majority_evidence(tmp_path):
    store = MemoryStore(path=tmp_path)
    evidence = _seed_cluster(store, {"ru": 5, "en": 2, "ja": 1})

    cand = SchemaCandidate(
        pattern="tags:tech+python",
        confidence=0.9,
        evidence_count=len(evidence),
        evidence_ids=list(evidence),
        status="auto",
    )
    schema_id = persist_schema(store, cand)

    fresh = store.get(schema_id)
    assert fresh is not None
    assert fresh.language == "ru", (
        f"persist_schema must read majority language from evidence, got {fresh.language!r}"
    )

def test_persist_schema_fallback_en_on_empty_evidence(tmp_path):
    store = MemoryStore(path=tmp_path)
    cand = SchemaCandidate(
        pattern="tags:orphan",
        confidence=0.9,
        evidence_count=0,
        evidence_ids=[],
        status="auto",
    )
    schema_id = persist_schema(store, cand)
    fresh = store.get(schema_id)
    assert fresh is not None
    assert fresh.language == "en"

def test_persist_schema_tie_is_deterministic(tmp_path):
    store = MemoryStore(path=tmp_path)
    evidence = _seed_cluster(store, {"ru": 3, "en": 3})

    cand = SchemaCandidate(
        pattern="tags:tied",
        confidence=0.9,
        evidence_count=len(evidence),
        evidence_ids=list(evidence),
        status="auto",
    )
    schema_id = persist_schema(store, cand)
    fresh = store.get(schema_id)
    assert fresh is not None
    assert fresh.language == "ru"

def test_persist_schema_ignores_missing_evidence_records(tmp_path):
    store = MemoryStore(path=tmp_path)

    surviving = _seed_cluster(store, {"ja": 2})

    phantom_ids = [uuid4() for _ in range(3)]

    cand = SchemaCandidate(
        pattern="tags:graceful",
        confidence=0.85,
        evidence_count=5,
        evidence_ids=list(surviving) + phantom_ids,
        status="auto",
    )
    schema_id = persist_schema(store, cand)
    fresh = store.get(schema_id)
    assert fresh is not None
    assert fresh.language == "ja", (
        f"persist_schema must ignore missing evidence records, got {fresh.language!r}"
    )

def test_persist_schema_no_hardcoded_english(tmp_path):
    import inspect
    from iai_mcp import schema as schema_mod

    src = inspect.getsource(schema_mod.persist_schema)
    assert "language=\"en\"," not in src, (
        "persist_schema still hardcodes language='en'"
    )
    assert "_majority_language" in src, (
        "persist_schema must call _majority_language to derive schema language"
    )
    from iai_mcp.lilli.cycle import schema as cycle_schema
    assert hasattr(cycle_schema, "_majority_language"), (
        "_majority_language helper must exist at lilli.cycle.schema module scope"
    )
