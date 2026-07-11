from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from iai_mcp.schema import (
    SchemaCandidate,
    induce_schemas_tier0,
    persist_schema,
)
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p)
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None)
    )
    yield fake

@pytest.fixture(autouse=True)
def _patch_embedder(monkeypatch: pytest.MonkeyPatch):
    from iai_mcp import embed as embed_mod

    class _FakeEmbedder:
        DIM = EMBED_DIM
        DEFAULT_DIM = EMBED_DIM
        DEFAULT_MODEL_KEY = "fake"

        def __init__(self, *args, **kwargs):  # noqa: ANN001
            self.DIM = EMBED_DIM

        def embed(self, text: str) -> list[float]:
            return [1.0] + [0.0] * (EMBED_DIM - 1)

        def embed_batch(self, texts):  # noqa: ANN001
            return [self.embed(t) for t in texts]

    monkeypatch.setattr(embed_mod, "Embedder", _FakeEmbedder)
    yield

def _rec(
    *,
    text: str = "t",
    tags: list[str] | None = None,
    tier: str = "episodic",
    detail_level: int = 2,
    language: str = "en",
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
        never_decay=(detail_level >= 3),
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=list(tags or []),
        language=language,
    )

@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "lancedb")

def test_induce_schemas_tier0_uses_iter_record_columns_not_all_records(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(5):
        store.insert(_rec(text=f"r{i}", tags=["meeting", "notes"]))

    spy_all = MagicMock(wraps=store.all_records)
    spy_iter = MagicMock(wraps=store.iter_record_columns)
    monkeypatch.setattr(store, "all_records", spy_all)
    monkeypatch.setattr(store, "iter_record_columns", spy_iter)

    induce_schemas_tier0(store)

    assert spy_all.call_count == 0, (
        f"induce_schemas_tier0 must NOT call store.all_records(); "
        f"got {spy_all.call_count} call(s)"
    )
    assert spy_iter.call_count >= 1, (
        f"induce_schemas_tier0 must call store.iter_record_columns() at least "
        f"once; got {spy_iter.call_count} call(s)"
    )

def test_induce_schemas_tier0_zero_decrypt_calls(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(5):
        store.insert(_rec(text=f"r{i}", tags=["meeting", "notes"]))

    decrypt_spy = MagicMock(wraps=store._decrypt_for_record)
    monkeypatch.setattr(store, "_decrypt_for_record", decrypt_spy)

    induce_schemas_tier0(store)

    assert decrypt_spy.call_count == 0, (
        f"induce_schemas_tier0 must NOT trigger ANY _decrypt_for_record "
        f"calls; got {decrypt_spy.call_count} call(s)"
    )

def test_induce_schemas_tier0_byte_identical_to_pre_d26_implementation(
    store: MemoryStore,
) -> None:
    auto_recs: list[MemoryRecord] = []
    for i in range(9):
        r = _rec(text=f"auto-{i}", tags=["meeting", "notes"])
        auto_recs.append(r)
        store.insert(r)
    for i in range(4):
        store.insert(_rec(text=f"low-{i}", tags=["report", "deadline"]))

    from iai_mcp.schema import (
        AUTO_INDUCT_CONFIDENCE,
        AUTO_INDUCT_COOCCURRENCE,
        MAX_EVIDENCE_PER_SCHEMA,
        USER_APPROVAL_COOCCURRENCE,
    )
    from iai_mcp.lilli.cycle.schema import _tag_cooccurrence

    expected_records = store.all_records()
    pair_counts = _tag_cooccurrence(expected_records)
    expected: list[dict] = []
    for pair, evidence in pair_counts.items():
        count = len(evidence)
        confidence = min(1.0, count / 10.0)
        if count >= AUTO_INDUCT_COOCCURRENCE and confidence >= AUTO_INDUCT_CONFIDENCE:
            status = "auto"
        elif count >= USER_APPROVAL_COOCCURRENCE:
            status = "pending_user_approval"
        else:
            continue
        expected.append({
            "pattern": f"tags:{'+'.join(sorted(pair))}",
            "confidence": confidence,
            "evidence_count": count,
            "status": status,
            "evidence_ids_set": set(evidence[:MAX_EVIDENCE_PER_SCHEMA]),
        })

    actual = induce_schemas_tier0(store)

    expected_sorted = sorted(expected, key=lambda d: d["pattern"])
    actual_sorted = sorted(actual, key=lambda c: c.pattern)

    assert len(actual_sorted) == len(expected_sorted), (
        f"candidate count mismatch — expected={len(expected_sorted)} "
        f"actual={len(actual_sorted)}; expected={expected_sorted!r}; "
        f"actual={[(c.pattern, c.evidence_count, c.confidence, c.status) for c in actual_sorted]!r}"
    )
    for e, a in zip(expected_sorted, actual_sorted, strict=True):
        assert a.pattern == e["pattern"]
        assert a.evidence_count == e["evidence_count"]
        assert a.confidence == pytest.approx(e["confidence"])
        assert a.status == e["status"]
        assert set(a.evidence_ids) == e["evidence_ids_set"]

    assert any(c.status == "auto" for c in actual_sorted), (
        f"expected at least one status='auto' candidate on the 9-record "
        f"meeting+notes pair; got {[(c.pattern, c.evidence_count, c.status) for c in actual_sorted]!r}"
    )

def test_induce_schemas_tier0_evidence_ids_are_uuids(
    store: MemoryStore,
) -> None:
    inserted = []
    for i in range(9):
        r = _rec(text=f"r{i}", tags=["meeting", "notes"])
        store.insert(r)
        inserted.append(r.id)

    candidates = induce_schemas_tier0(store)
    auto = [c for c in candidates if c.status == "auto"]
    assert len(auto) >= 1, "expected at least one auto candidate"

    for c in auto:
        for ev_id in c.evidence_ids:
            assert isinstance(ev_id, UUID), (
                f"evidence_ids must be list[UUID]; got {type(ev_id).__name__} "
                f"for {ev_id!r}"
            )
        assert set(c.evidence_ids).issubset(set(inserted))

def test_persist_schema_uses_iter_record_columns_not_all_records_for_keeper_scan(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    ev_recs = [_rec(text=f"ev{i}", tags=["meeting", "notes"]) for i in range(3)]
    for r in ev_recs:
        store.insert(r)

    spy_all = MagicMock(wraps=store.all_records)
    spy_iter = MagicMock(wraps=store.iter_record_columns)
    monkeypatch.setattr(store, "all_records", spy_all)
    monkeypatch.setattr(store, "iter_record_columns", spy_iter)

    cand = SchemaCandidate(
        pattern="tags:meeting+notes",
        confidence=0.9,
        evidence_count=3,
        evidence_ids=[r.id for r in ev_recs],
        status="auto",
    )
    persist_schema(store, cand)

    assert spy_all.call_count == 0, (
        f"persist_schema must NOT call store.all_records(); "
        f"got {spy_all.call_count} call(s)"
    )
    assert spy_iter.call_count >= 1, (
        f"persist_schema must call store.iter_record_columns() at least once "
        f"in the keeper scan; got {spy_iter.call_count} call(s)"
    )

def test_persist_schema_early_exit_on_first_match(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    pattern = "tags:meeting+notes"
    pattern_tag = f"pattern:{pattern}"
    keeper_ids: list[UUID] = []
    for i in range(50):
        r = _rec(
            text=f"schema-{i}",
            tags=["schema", "auto", pattern_tag],
            tier="semantic",
            detail_level=3,
        )
        store.insert(r)
        keeper_ids.append(r.id)

    real_iter = store.iter_record_columns
    yielded = {"count": 0}

    def counting_iter(columns, **kwargs):  # noqa: ANN001
        for row in real_iter(columns, **kwargs):
            yielded["count"] += 1
            yield row

    monkeypatch.setattr(store, "iter_record_columns", counting_iter)

    ev_recs = [_rec(text=f"ev{i}", tags=["meeting", "notes"]) for i in range(3)]
    for r in ev_recs:
        store.insert(r)

    cand = SchemaCandidate(
        pattern=pattern,
        confidence=0.9,
        evidence_count=3,
        evidence_ids=[r.id for r in ev_recs],
        status="auto",
    )
    schema_id = persist_schema(store, cand)

    assert schema_id in keeper_ids, (
        f"persist_schema must return an existing keeper id when a match exists; "
        f"got {schema_id} not in {keeper_ids[:3]}..."
    )

    assert yielded["count"] <= 50 // 2, (
        f"persist_schema must early-exit on first match; iterator yielded "
        f"{yielded['count']} rows on a 50-keeper-row store (expected break "
        f"after the first match — strictly < 50)"
    )

def test_persist_schema_returns_correct_id_when_keeper_is_mid_stream(
    store: MemoryStore,
) -> None:
    pattern = "tags:meeting+notes"
    pattern_tag = f"pattern:{pattern}"

    for i in range(2):
        store.insert(_rec(
            text=f"unrelated-{i}",
            tags=["schema", "auto", "pattern:other"],
            tier="semantic",
            detail_level=3,
        ))
    keeper = _rec(
        text="the-keeper",
        tags=["schema", "auto", pattern_tag],
        tier="semantic",
        detail_level=3,
    )
    store.insert(keeper)
    keeper_id = keeper.id
    for i in range(2):
        store.insert(_rec(
            text=f"trailing-{i}",
            tags=["schema", "auto", "pattern:something-else"],
            tier="semantic",
            detail_level=3,
        ))

    ev_recs = [_rec(text=f"ev{i}", tags=["meeting", "notes"]) for i in range(3)]
    for r in ev_recs:
        store.insert(r)

    cand = SchemaCandidate(
        pattern=pattern,
        confidence=0.9,
        evidence_count=3,
        evidence_ids=[r.id for r in ev_recs],
        status="auto",
    )
    returned_id = persist_schema(store, cand)

    assert returned_id == keeper_id, (
        f"persist_schema must return the matching keeper's UUID; "
        f"got {returned_id} expected {keeper_id}"
    )
    assert isinstance(returned_id, UUID), (
        f"persist_schema must return a UUID, not a string from row['id']; "
        f"got {type(returned_id).__name__}"
    )

def test_persist_schema_falls_through_to_insert_when_no_keeper(
    store: MemoryStore,
) -> None:
    other_ids: list[UUID] = []
    for i in range(5):
        r = _rec(
            text=f"other-{i}",
            tags=["schema", "auto", f"pattern:other-{i}"],
            tier="semantic",
            detail_level=3,
        )
        store.insert(r)
        other_ids.append(r.id)

    ev_recs = [_rec(text=f"ev{i}", tags=["meeting", "notes"]) for i in range(3)]
    for r in ev_recs:
        store.insert(r)

    cand = SchemaCandidate(
        pattern="tags:meeting+notes",
        confidence=0.9,
        evidence_count=3,
        evidence_ids=[r.id for r in ev_recs],
        status="auto",
    )
    schema_id = persist_schema(store, cand)

    assert schema_id not in other_ids, (
        f"persist_schema must insert a new schema when no keeper matches; "
        f"got returned id {schema_id} which equals one of the existing "
        f"non-matching schema ids ({other_ids!r})"
    )
    new_rec = store.get(schema_id)
    assert new_rec is not None
    assert new_rec.tier == "semantic"
    assert new_rec.detail_level == 3
    assert "schema" in (new_rec.tags or [])
    assert f"pattern:{cand.pattern}" in (new_rec.tags or [])
