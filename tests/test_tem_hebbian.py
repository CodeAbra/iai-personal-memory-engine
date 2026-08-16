from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch):
    import keyring as _keyring

    fake_store: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake_store.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake_store.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake_store.pop((s, u), None))
    yield fake_store

def _make_record(text="x", structure_hv=None, **overrides):
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    base = dict(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
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
        language="en",
    )
    if structure_hv is not None:
        base["structure_hv"] = structure_hv
    base.update(overrides)
    return MemoryRecord(**base)

def test_structural_similarity_identical_hv():
    from iai_mcp.hebbian_structure import structural_similarity
    from iai_mcp.types import STRUCTURE_HV_BYTES

    hv = bytes([0xAA] * STRUCTURE_HV_BYTES)
    assert structural_similarity(hv, hv) == pytest.approx(1.0)

def test_structural_similarity_orthogonal_hv():
    from iai_mcp.hebbian_structure import structural_similarity
    from iai_mcp.types import STRUCTURE_HV_BYTES

    a = bytes([0x00] * STRUCTURE_HV_BYTES)
    b = bytes([0xFF] * STRUCTURE_HV_BYTES)
    assert structural_similarity(a, b) == pytest.approx(0.0)

def test_structural_similarity_handles_empty_inputs():
    from iai_mcp.hebbian_structure import structural_similarity

    assert structural_similarity(b"", b"") == 0.0
    assert structural_similarity(b"abc", b"de") == 0.0

def test_strengthen_structure_edge_writes_with_correct_edge_type(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.hebbian_structure import strengthen_structure_edge
    from iai_mcp.store import EDGES_TABLE, MemoryStore

    store = MemoryStore()
    a, b = _make_record("a"), _make_record("b")
    store.insert(a)
    store.insert(b)

    strengthen_structure_edge(store, a.id, b.id, gain=0.5)

    edges_df = store.db.open_table(EDGES_TABLE).to_pandas()
    structure_edges = edges_df[edges_df["edge_type"] == "hebbian_structure"]
    assert len(structure_edges) == 1
    row = structure_edges.iloc[0]
    assert {row["src"], row["dst"]} == {str(a.id), str(b.id)}
    assert float(row["weight"]) == pytest.approx(0.5)

def test_co_retrieval_trigger_fires_above_threshold(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.hebbian_structure import co_retrieval_trigger
    from iai_mcp.store import EDGES_TABLE, MemoryStore
    from iai_mcp.types import STRUCTURE_HV_BYTES

    store = MemoryStore()
    shared_hv = bytes([0x55] * STRUCTURE_HV_BYTES)
    a = _make_record("a", structure_hv=shared_hv)
    b = _make_record("b", structure_hv=shared_hv)
    c = _make_record("c", structure_hv=shared_hv)
    for r in (a, b, c):
        store.insert(r)

    fired = co_retrieval_trigger(store, [a, b, c])
    assert fired == 3
    edges_df = store.db.open_table(EDGES_TABLE).to_pandas()
    structure_edges = edges_df[edges_df["edge_type"] == "hebbian_structure"]
    assert len(structure_edges) == 3

def test_co_retrieval_trigger_does_not_fire_below_threshold(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.hebbian_structure import co_retrieval_trigger
    from iai_mcp.store import EDGES_TABLE, MemoryStore
    from iai_mcp.types import STRUCTURE_HV_BYTES

    store = MemoryStore()
    a = _make_record("a", structure_hv=bytes([0x00] * STRUCTURE_HV_BYTES))
    b = _make_record("b", structure_hv=bytes([0xFF] * STRUCTURE_HV_BYTES))
    store.insert(a)
    store.insert(b)

    fired = co_retrieval_trigger(store, [a, b])
    assert fired == 0
    edges_df = store.db.open_table(EDGES_TABLE).to_pandas()
    structure_edges = edges_df[edges_df["edge_type"] == "hebbian_structure"]
    assert len(structure_edges) == 0

def test_sleep_decay_sweep_includes_hebbian_structure(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.hebbian_structure import strengthen_structure_edge
    from iai_mcp.sleep import _decay_edges
    from iai_mcp.store import EDGES_TABLE, MemoryStore

    store = MemoryStore()
    a, b = _make_record("a"), _make_record("b")
    store.insert(a)
    store.insert(b)
    strengthen_structure_edge(store, a.id, b.id, gain=1.0)

    edges_tbl = store.db.open_table(EDGES_TABLE)
    backdate = datetime.now(timezone.utc) - timedelta(days=120)
    edges_tbl.update(
        where="edge_type = 'hebbian_structure'",
        values={"updated_at": backdate},
    )

    _decay_edges(store)

    decayed_df = store.db.open_table(EDGES_TABLE).to_pandas()
    structure_edges = decayed_df[decayed_df["edge_type"] == "hebbian_structure"]
    assert len(structure_edges) == 1
    new_weight = float(structure_edges.iloc[0]["weight"])
    expected = 1.0 * (0.9 ** 30)
    assert new_weight == pytest.approx(expected, rel=1e-3)
