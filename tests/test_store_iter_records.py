from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp.crypto import CIPHERTEXT_PREFIX
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

def _make(
    tier: str = "episodic",
    text: str = "hello world",
    tags: list[str] | None = None,
    detail: int = 2,
    pinned: bool = False,
    language: str = "en",
) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=detail,
        pinned=pinned,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=(detail >= 3),
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=tags if tags is not None else [],
        language=language,
    )

@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "lancedb")

def test_iter_records_yields_all_inserted(store):
    inserted_ids = set()
    for i in range(10):
        rec = _make(text=f"record-{i}")
        store.insert(rec)
        inserted_ids.add(rec.id)

    iterated = list(store.iter_records())
    assert len(iterated) == 10
    assert {r.id for r in iterated} == inserted_ids

def test_iter_records_yields_correct_dataclass_type(store):
    rec_a = _make(text="alpha", tier="episodic", tags=["t1", "t2"])
    rec_b = _make(text="beta", tier="semantic", tags=["t3"])
    rec_c = _make(text="gamma", tier="episodic", tags=[])
    for r in (rec_a, rec_b, rec_c):
        store.insert(r)

    iterated = list(store.iter_records())
    assert len(iterated) == 3
    by_id = {r.id: r for r in iterated}
    assert isinstance(by_id[rec_a.id], MemoryRecord)
    assert by_id[rec_a.id].literal_surface == "alpha"
    assert by_id[rec_a.id].tier == "episodic"
    assert by_id[rec_a.id].tags == ["t1", "t2"]
    assert by_id[rec_b.id].literal_surface == "beta"
    assert by_id[rec_b.id].tier == "semantic"
    assert by_id[rec_b.id].tags == ["t3"]
    assert by_id[rec_c.id].literal_surface == "gamma"
    assert by_id[rec_c.id].tags == []

def test_iter_records_handles_empty_store(store):
    assert list(store.iter_records()) == []

def test_iter_records_respects_batch_size_one(store):
    for i in range(7):
        store.insert(_make(text=f"row-{i}"))
    iterated = list(store.iter_records(batch_size=1))
    assert len(iterated) == 7

def test_iter_records_respects_batch_size_larger_than_table(store):
    for i in range(5):
        store.insert(_make(text=f"row-{i}"))
    iterated = list(store.iter_records(batch_size=10000))
    assert len(iterated) == 5

def test_iter_records_with_columns_projects_subset(store):
    rec_a = _make(text="alpha", tags=["tag-a"])
    rec_b = _make(text="beta", tags=["tag-b"])
    rec_c = _make(text="gamma", tags=["tag-c"])
    for r in (rec_a, rec_b, rec_c):
        store.insert(r)

    iterated = list(
        store.iter_records(columns=["id", "tags_json", "tier", "embedding"])
    )
    assert len(iterated) == 3
    by_id = {r.id: r for r in iterated}
    assert by_id[rec_a.id].tags == ["tag-a"]
    assert by_id[rec_b.id].tags == ["tag-b"]
    assert by_id[rec_c.id].tags == ["tag-c"]

def test_iter_records_with_where_filter(store):
    rec_e1 = _make(text="ep1", tier="episodic")
    rec_e2 = _make(text="ep2", tier="episodic")
    rec_s1 = _make(text="sem1", tier="semantic")
    for r in (rec_e1, rec_e2, rec_s1):
        store.insert(r)

    iterated = list(store.iter_records(where="tier = 'episodic'"))
    assert len(iterated) == 2
    assert {r.id for r in iterated} == {rec_e1.id, rec_e2.id}
    assert all(r.tier == "episodic" for r in iterated)

def test_iter_record_columns_returns_raw_dicts(store):
    for i in range(3):
        store.insert(_make(text=f"row-{i}", tags=[f"t{i}"]))

    rows = list(store.iter_record_columns(["id", "tags_json"]))
    assert len(rows) == 3
    for row in rows:
        assert isinstance(row, dict)
        assert set(row.keys()) == {"id", "tags_json"}
        assert "literal_surface" not in row

def test_iter_record_columns_passes_ciphertext_through(store):
    rec = _make(text="secret content for ciphertext test")
    store.insert(rec)

    rows = list(store.iter_record_columns(["id", "literal_surface"]))
    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row["literal_surface"], str)
    assert row["literal_surface"].startswith(CIPHERTEXT_PREFIX), (
        f"expected ciphertext prefix {CIPHERTEXT_PREFIX!r}, "
        f"got {row['literal_surface']!r}"
    )

def test_iter_record_columns_handles_empty_store(store):
    assert list(store.iter_record_columns(["id"])) == []

def test_iter_record_columns_with_where_filter(store):
    rec_e1 = _make(text="ep1", tier="episodic")
    rec_e2 = _make(text="ep2", tier="episodic")
    rec_s1 = _make(text="sem1", tier="semantic")
    for r in (rec_e1, rec_e2, rec_s1):
        store.insert(r)

    rows = list(
        store.iter_record_columns(["id", "tier"], where="tier = 'episodic'")
    )
    assert len(rows) == 2
    assert {r["id"] for r in rows} == {str(rec_e1.id), str(rec_e2.id)}
    assert all(r["tier"] == "episodic" for r in rows)

def test_from_row_partial_row_dict_does_not_crash(store):
    minimal_row = {
        "id": str(uuid4()),
        "embedding": [0.0] * EMBED_DIM,
        "tier": "episodic",
        "tags_json": "[]",
    }
    rec = store._from_row(minimal_row)

    assert isinstance(rec, MemoryRecord)
    assert rec.tier == "episodic"
    assert rec.embedding == [0.0] * EMBED_DIM
    assert rec.literal_surface == ""
    assert rec.aaak_index == ""
    assert rec.provenance == []
    assert rec.detail_level == 1
    assert rec.pinned is False
    assert rec.stability == 0.0
    assert rec.difficulty == 0.0
    assert rec.never_decay is False
    assert rec.never_merge is False
    assert rec.tags == []
    assert rec.language == "en"
    assert isinstance(rec.created_at, datetime)
    assert isinstance(rec.updated_at, datetime)

def test_iter_records_does_not_modify_existing_all_records_behaviour(store):
    inserted = []
    for i in range(5):
        rec = _make(text=f"persist-{i}", tags=[f"persist-{i}"])
        store.insert(rec)
        inserted.append(rec)

    before = store.all_records()

    list(store.iter_records())

    after = store.all_records()

    assert len(before) == len(after) == 5
    by_id_before = {r.id: r for r in before}
    by_id_after = {r.id: r for r in after}
    assert set(by_id_before.keys()) == set(by_id_after.keys())
    for rid in by_id_before:
        assert by_id_before[rid].literal_surface == by_id_after[rid].literal_surface
        assert by_id_before[rid].tags == by_id_after[rid].tags
        assert by_id_before[rid].tier == by_id_after[rid].tier
