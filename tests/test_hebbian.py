from __future__ import annotations


from iai_mcp.core import L0_ID, LIVE_KNOBS, _seed_l0_identity, dispatch
from iai_mcp.store import MemoryStore
from tests.test_store import _make


def test_reinforce_creates_pairwise_edges(tmp_path):
    store = MemoryStore(path=tmp_path)
    recs = [_make() for _ in range(3)]
    for r in recs:
        store.insert(r)
    ids = [str(r.id) for r in recs]
    result = dispatch(store, "memory_reinforce", {"ids": ids})
    assert result["edges_boosted"] == 3


def test_reinforce_twice_doubles_weight(tmp_path):
    store = MemoryStore(path=tmp_path)
    recs = [_make() for _ in range(2)]
    for r in recs:
        store.insert(r)
    ids = [str(r.id) for r in recs]
    dispatch(store, "memory_reinforce", {"ids": ids})
    r2 = dispatch(store, "memory_reinforce", {"ids": ids})
    assert len(r2["new_weights"]) == 1
    key = next(iter(r2["new_weights"]))
    assert abs(r2["new_weights"][key] - 0.2) < 1e-5


def test_l0_identity_seeded(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps(
        {"identity": {"name": "alice", "languages": "en", "role": "developer"}}))
    store = MemoryStore(path=tmp_path)
    _seed_l0_identity(store)
    l0 = store.get(L0_ID)
    assert l0 is not None
    assert l0.pinned is True
    assert l0.never_decay is True
    assert l0.never_merge is True
    assert l0.detail_level == 5
    assert l0.tier == "semantic"
    assert "alice" in l0.literal_surface


def test_l0_seed_is_idempotent(tmp_path):
    store = MemoryStore(path=tmp_path)
    _seed_l0_identity(store)
    _seed_l0_identity(store)
    _seed_l0_identity(store)
    all_records = store.all_records()
    l0_count = sum(1 for r in all_records if r.id == L0_ID)
    assert l0_count == 1


def test_profile_get_returns_live_knobs(tmp_path):
    store = MemoryStore(path=tmp_path)
    result = dispatch(store, "profile_get", {})
    assert result["live"]["literal_preservation"] == "strong"
    assert result["live"]["masking_off"] is True
    assert result["live"]["task_support"] == "cued_recognition"
    assert result["live"]["scene_construction_scaffold"] is True
    assert result["live"]["wake_depth"] == "minimal"
    assert len(result["live"]) == 10
    assert len(result["deferred"]) == 0


def test_profile_get_specific_live_knob(tmp_path):
    store = MemoryStore(path=tmp_path)
    result = dispatch(store, "profile_get", {"knob": "literal_preservation"})
    assert result["knob"] == "literal_preservation"
    assert result["value"] == "strong"


def test_profile_set_live_knob_succeeds(tmp_path):
    store = MemoryStore(path=tmp_path)
    LIVE_KNOBS["literal_preservation"] = "strong"
    result = dispatch(store, "profile_set", {"knob": "literal_preservation", "value": "loose"})
    assert result["status"] == "ok"
    assert LIVE_KNOBS["literal_preservation"] == "loose"
    LIVE_KNOBS["literal_preservation"] = "strong"


def test_memory_consolidate_real(tmp_path):
    store = MemoryStore(path=tmp_path)
    result = dispatch(store, "memory_consolidate", {})
    assert result["mode"] == "heavy"
    assert result["tier"] in ("tier0", "tier1")
    assert "summaries_created" in result
    assert "decay_result" in result
    assert "schema_candidates" in result
