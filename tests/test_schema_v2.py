from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

def _make_v2(
    *,
    language: str = "en",
    s5_trust_score: float = 0.5,
    profile_modulation_gain: dict | None = None,
    schema_version: int = 2,
    literal_surface: str = "hello world",
    tier: str = "episodic",
    embedding_dim: int | None = None,
):
    from iai_mcp.types import MemoryRecord

    if embedding_dim is None:
        from iai_mcp.embed import Embedder
        embedding_dim = Embedder.DEFAULT_DIM if hasattr(Embedder, "DEFAULT_DIM") else 384

    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=literal_surface,
        aaak_index="",
        embedding=[0.1] * embedding_dim,
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
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language=language,
        s5_trust_score=s5_trust_score,
        profile_modulation_gain=profile_modulation_gain or {},
        schema_version=schema_version,
    )

def test_memory_record_has_language_field():
    r = _make_v2(language="en")
    assert r.language == "en"

def test_memory_record_requires_language_field():
    from iai_mcp.types import MemoryRecord

    from iai_mcp.embed import Embedder
    _dim = Embedder.DEFAULT_DIM if hasattr(Embedder, "DEFAULT_DIM") else 384

    with pytest.raises(TypeError):
        MemoryRecord(  # type: ignore[call-arg]
            id=uuid4(),
            tier="episodic",
            literal_surface="hi",
            aaak_index="",
            embedding=[0.0] * _dim,
            community_id=None,
            centrality=0.0,
            detail_level=1,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tags=[],
        )

def test_memory_record_language_must_be_non_empty():
    with pytest.raises(ValueError):
        _make_v2(language="")

def test_memory_record_has_s5_trust_score():
    r = _make_v2(s5_trust_score=0.5)
    assert r.s5_trust_score == 0.5

def test_memory_record_s5_trust_score_default_is_0_5():
    from iai_mcp.types import MemoryRecord
    from iai_mcp.embed import Embedder
    _dim = Embedder.DEFAULT_DIM if hasattr(Embedder, "DEFAULT_DIM") else 384

    r = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="hi",
        aaak_index="",
        embedding=[0.0] * _dim,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )
    assert r.s5_trust_score == 0.5

def test_memory_record_s5_trust_score_rejects_out_of_range():
    with pytest.raises(ValueError):
        _make_v2(s5_trust_score=1.5)
    with pytest.raises(ValueError):
        _make_v2(s5_trust_score=-0.1)

def test_memory_record_s5_trust_score_boundary_values_ok():
    assert _make_v2(s5_trust_score=0.0).s5_trust_score == 0.0
    assert _make_v2(s5_trust_score=1.0).s5_trust_score == 1.0

def test_memory_record_has_profile_modulation_gain():
    r = _make_v2(profile_modulation_gain={"monotropism_depth": 1.3, "interest_boost": 1.5})
    assert r.profile_modulation_gain == {"monotropism_depth": 1.3, "interest_boost": 1.5}

def test_memory_record_profile_modulation_gain_default_empty_dict():
    r = _make_v2()
    assert r.profile_modulation_gain == {}

def test_memory_record_has_schema_version_default_2():
    r = _make_v2()
    assert r.schema_version == 2

def test_memory_record_schema_version_accepts_1_for_migration():
    r = _make_v2(schema_version=1)
    assert r.schema_version == 1

def test_memory_record_schema_version_rejects_other_values():
    with pytest.raises(ValueError):
        _make_v2(schema_version=0)
    with pytest.raises(ValueError):
        _make_v2(schema_version=99)

def test_edge_types_registry_has_11_members():
    from iai_mcp.store import EDGE_TYPES

    expected = {
        "hebbian",
        "contradicts",
        "consolidated_from",
        "schema_instance_of",
        "temporal_next",
        "invariant_anchor",
        "curiosity_bridge",
        "profile_modulates",
        "hebbian_structure",
        "pattern_separation_seed",
        "hebbian_cluster_replay",
    }
    assert EDGE_TYPES == frozenset(expected)

def test_boost_edges_accepts_new_phase2_types(tmp_path):
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    r1 = _make_v2()
    r2 = _make_v2()
    store.insert(r1)
    store.insert(r2)

    for edge_type in (
        "consolidated_from",
        "schema_instance_of",
        "temporal_next",
        "invariant_anchor",
        "curiosity_bridge",
        "profile_modulates",
    ):
        w = store.boost_edges([(r1.id, r2.id)], edge_type=edge_type, delta=1.0)
        assert list(w.values())[0] == pytest.approx(1.0), f"edge_type={edge_type} weight wrong"

def test_boost_edges_phase1_types_still_work(tmp_path):
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    r1 = _make_v2()
    r2 = _make_v2()
    store.insert(r1)
    store.insert(r2)
    w = store.boost_edges([(r1.id, r2.id)], delta=0.1)
    assert list(w.values())[0] == pytest.approx(0.1)

def test_boost_edges_rejects_unknown_edge_type(tmp_path):
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    r1 = _make_v2()
    r2 = _make_v2()
    store.insert(r1)
    store.insert(r2)
    with pytest.raises(ValueError):
        store.boost_edges([(r1.id, r2.id)], edge_type="not_a_real_type")

def test_record_to_from_row_preserves_language(tmp_path):
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    r = _make_v2(language="ru", literal_surface="Hello Russian")
    store.insert(r)
    got = store.get(r.id)
    assert got is not None
    assert got.language == "ru"

def test_record_to_from_row_preserves_s5_trust_score(tmp_path):
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    r = _make_v2(s5_trust_score=0.73)
    store.insert(r)
    got = store.get(r.id)
    assert got is not None
    assert abs(got.s5_trust_score - 0.73) < 1e-5

def test_record_to_from_row_preserves_profile_modulation_gain(tmp_path):
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    gain = {"monotropism_depth": 1.3, "interest_boost": 1.5}
    r = _make_v2(profile_modulation_gain=gain)
    store.insert(r)
    got = store.get(r.id)
    assert got is not None
    assert got.profile_modulation_gain == gain

def test_record_to_from_row_preserves_schema_version(tmp_path):
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    r = _make_v2(schema_version=2)
    store.insert(r)
    got = store.get(r.id)
    assert got is not None
    assert got.schema_version == 2

def test_legacy_record_reads_default_v1_defaults(tmp_path):
    from datetime import datetime, timezone

    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    from iai_mcp.embed import Embedder
    _dim = Embedder.DEFAULT_DIM if hasattr(Embedder, "DEFAULT_DIM") else 384

    store = MemoryStore(path=tmp_path)
    tbl = store.db.open_table(RECORDS_TABLE)
    now = datetime.now(timezone.utc)
    v1_id = uuid4()
    row = {
        "id": str(v1_id),
        "tier": "episodic",
        "literal_surface": "legacy record",
        "aaak_index": "",
        "embedding": [0.0] * _dim,
        "structure_hv": b"",
        "community_id": "",
        "centrality": 0.0,
        "detail_level": 1,
        "pinned": False,
        "stability": 0.0,
        "difficulty": 0.0,
        "last_reviewed": None,
        "never_decay": False,
        "never_merge": False,
        "provenance_json": "[]",
        "created_at": now,
        "updated_at": now,
        "tags_json": "[]",
        "language": "",
        "s5_trust_score": 0.5,
        "profile_modulation_gain_json": "{}",
        "schema_version": 1,
    }
    tbl.add([row])
    got = store.get(v1_id)
    assert got is not None
    assert got.language in ("en", "")
    assert got.schema_version == 1
