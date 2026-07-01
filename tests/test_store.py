from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, STRUCTURE_HV_BYTES, MemoryRecord


def _make(
    tier: str = "episodic",
    text: str = "hello world",
    vec: list[float] | None = None,
    detail: int = 2,
    pinned: bool = False,
    never_merge: bool = False,
    language: str = "en",
) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=vec if vec is not None else [0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=detail,
        pinned=pinned,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=(detail >= 3),
        never_merge=never_merge,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language=language,
    )


def test_insert_and_get_preserves_verbatim(tmp_path):
    store = MemoryStore(path=tmp_path)
    verbatim = "Alice said: пусть каждое слово сохранится точно"
    r = _make(text=verbatim)
    store.insert(r)
    got = store.get(r.id)
    assert got is not None
    assert got.literal_surface == verbatim


def test_query_empty_store_returns_empty_list(tmp_path):
    store = MemoryStore(path=tmp_path)
    assert store.query_similar([0.0] * EMBED_DIM, k=5) == []


def test_detail_level_3_forces_never_decay():
    r = _make(detail=3)
    assert r.never_decay is True
    r4 = _make(detail=4)
    assert r4.never_decay is True


def test_detail_level_below_3_keeps_caller_never_decay_false():
    r = _make(detail=2)
    assert r.never_decay is False


def test_missing_embedding_raises():
    with pytest.raises(TypeError):
        MemoryRecord(  # type: ignore[call-arg]
            id=uuid4(),
            tier="episodic",
            literal_surface="hi",
            aaak_index="",
        )


def test_query_returns_top_k(tmp_path):
    store = MemoryStore(path=tmp_path)
    for _ in range(10):
        store.insert(_make())
    results = store.query_similar([0.1] * EMBED_DIM, k=3)
    assert len(results) == 3


def test_invalid_tier_rejected():
    with pytest.raises(ValueError):
        _make(tier="unknown-tier")


def test_persistence_across_store_instances(tmp_path):
    r = _make(text="persistent fact")
    store1 = MemoryStore(path=tmp_path)
    store1.insert(r)
    del store1
    store2 = MemoryStore(path=tmp_path)
    got = store2.get(r.id)
    assert got is not None
    assert got.literal_surface == "persistent fact"


def test_sql_record_projection_preserves_structure_and_pending(tmp_path):
    store = MemoryStore(path=tmp_path)
    structure_hv = b"\x01" * STRUCTURE_HV_BYTES
    payload = b'{"tier":"sparse"}'
    rec = _make(text="structured pending record")
    rec.structure_hv = structure_hv
    rec.hv_tier = "sparse_vsa"
    rec.structure_hv_payload = payload
    rec.embedding_pending = 1
    store.insert(rec)

    batch = store.get_batch([rec.id])

    got = batch[rec.id]
    assert got.structure_hv == structure_hv
    assert got.hv_tier == "sparse_vsa"
    assert got.structure_hv_payload == payload
    assert got.embedding_pending == 1


def test_direct_recency_projection_preserves_structure_metadata(tmp_path):
    from iai_mcp.hippo._recall import direct_recency_rows_from_store

    store = MemoryStore(path=tmp_path)
    structure_hv = b"\x03" * STRUCTURE_HV_BYTES
    payload = b'{"source":"direct-recency"}'
    rec = _make(text="direct recency structured record")
    rec.structure_hv = structure_hv
    rec.hv_tier = "sparse_vsa"
    rec.structure_hv_payload = payload
    rec.embedding_pending = 1
    store.insert(rec)

    rows = direct_recency_rows_from_store(tmp_path, limit=1)

    assert rows
    row = rows[0]
    assert row["id"] == str(rec.id)
    assert bytes(row["structure_hv"]) == structure_hv
    assert row["hv_tier"] == "sparse_vsa"
    assert bytes(row["structure_hv_payload"]) == payload
    assert int(row["embedding_pending"]) == 1


def test_recent_pending_markers_preserve_structure_metadata(tmp_path):
    store = MemoryStore(path=tmp_path)
    structure_hv = b"\x04" * STRUCTURE_HV_BYTES
    payload = b'{"source":"pending-markers"}'
    rec = _make(text="pending marker structured record")
    rec.structure_hv = structure_hv
    rec.hv_tier = "sparse_vsa"
    rec.structure_hv_payload = payload
    rec.embedding_pending = 1
    store.insert(rec)

    markers = store.recent_pending_markers(n=5)

    assert markers
    got = next(marker for marker in markers if marker.id == rec.id)
    assert got.structure_hv == structure_hv
    assert got.hv_tier == "sparse_vsa"
    assert got.structure_hv_payload == payload
    assert got.embedding_pending == 1


def test_uuid_literal_accepts_uuid_and_canonical_str():
    from uuid import UUID

    from iai_mcp.store import _uuid_literal

    u = UUID("11111111-2222-3333-4444-555555555555")
    assert _uuid_literal(u) == "11111111-2222-3333-4444-555555555555"
    assert _uuid_literal(str(u).upper()) == "11111111-2222-3333-4444-555555555555"


def test_uuid_literal_rejects_injection_shapes():
    from iai_mcp.store import _uuid_literal

    injection_attempts = [
        "' OR 1=1 --",
        "abc",
        "11111111-2222-3333-4444-5555555555555",
        "11111111-2222-3333-4444-55555555555",
        "11111111-2222-3333-4444'--",
        "",
    ]
    for bad in injection_attempts:
        with pytest.raises(ValueError):
            _uuid_literal(bad)


def test_append_provenance_uses_validated_uuid(tmp_path):
    store = MemoryStore(path=tmp_path)
    r = _make(text="provenance-target")
    store.insert(r)
    store.append_provenance(r.id, {"ts": "2026-04-16T00:00:00Z", "cue": "test"})
    got = store.get(r.id)
    assert got is not None
    assert any(p.get("cue") == "test" for p in got.provenance)


def test_boost_edges_uses_validated_uuid(tmp_path):
    store = MemoryStore(path=tmp_path)
    a = _make(text="a")
    b = _make(text="b")
    store.insert(a)
    store.insert(b)
    w1 = store.boost_edges([(a.id, b.id)], delta=0.1)
    assert list(w1.values())[0] == pytest.approx(0.1)
    w2 = store.boost_edges([(a.id, b.id)], delta=0.1)
    assert list(w2.values())[0] == pytest.approx(0.2)
