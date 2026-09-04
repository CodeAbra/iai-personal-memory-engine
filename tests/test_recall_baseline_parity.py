from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord

def _make_episodic(text: str) -> MemoryRecord:
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
    )

def _make_schema(text: str, pattern: str) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=text,
        aaak_index="",
        embedding=[1.0] + [0.0] * (EMBED_DIM - 1),
        community_id=None,
        centrality=0.0,
        detail_level=3,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=["schema", "draft", f"pattern:{pattern}"],
        language="en",
    )

def _make_proc_chunk(text: str) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="procedural",
        literal_surface=text,
        aaak_index="",
        embedding=[1.0] + [0.0] * (EMBED_DIM - 1),
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
        created_at=now,
        updated_at=now,
        tags=["chunk", "source:cofire"],
        language="en",
    )

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

def _seed_mixed_tier_store(tmp_path):
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "lancedb")
    episodic_records = [_make_episodic(f"episodic verbatim text {i}") for i in range(3)]
    schema_records = [
        _make_schema(f"schema record {i}", pattern=f"test:r7:{i}")
        for i in range(2)
    ]
    for r in episodic_records:
        store.insert(r)
    for r in schema_records:
        store.insert(r)
    return store, episodic_records, schema_records

def _seed_mixed_tier_store_with_proc_chunk(tmp_path):
    store, episodic_records, schema_records = _seed_mixed_tier_store(tmp_path)
    proc_rec = _make_proc_chunk("procedural chunk content payload r7")
    store.insert(proc_rec)
    return store, episodic_records, schema_records, proc_rec

def test_baseline_recall_default_mode_is_verbatim_per_d14():
    import inspect
    from iai_mcp.retrieve import recall

    sig = inspect.signature(recall)
    assert "mode" in sig.parameters, "retrieve.recall must accept mode kwarg"
    assert sig.parameters["mode"].default == "verbatim", (
        f"retrieve.recall default mode must be 'verbatim', "
        f"got {sig.parameters['mode'].default!r}"
    )

def test_baseline_recall_verbatim_filters_to_episodic_only(tmp_path):
    from iai_mcp.retrieve import recall

    store, episodic_records, schema_records = _seed_mixed_tier_store(tmp_path)
    cue = [1.0] + [0.0] * (EMBED_DIM - 1)

    resp = recall(
        store=store, cue_embedding=cue, cue_text="probe",
        session_id="r7_default", k_hits=5, k_anti=2,
    )
    assert resp.cue_mode == "verbatim", (
        f"baseline default mode must be 'verbatim', got {resp.cue_mode!r}"
    )
    schema_id_set = {r.id for r in schema_records}
    for h in resp.hits:
        assert h.record_id not in schema_id_set, (
            f"verbatim mode baseline must exclude schema records; "
            f"schema {h.record_id} appeared in hits"
        )
        rec = store.get(h.record_id)
        assert rec is not None
        assert rec.tier == "episodic", (
            f"verbatim mode hit {h.record_id} has tier {rec.tier!r}, expected 'episodic'"
        )

def test_baseline_recall_verbatim_excludes_procedural_chunk(tmp_path):
    """Pins retrieve.py's own tier=="episodic" filter applied to its raw
    query_similar candidates under the default verbatim mode: a
    tier="procedural" record is excluded even at cosine 1.0 to the cue.
    The mutant that turns this RED: removing that tier filter from
    retrieve.recall's verbatim branch.
    """
    from iai_mcp.retrieve import recall

    store, episodic_records, schema_records, proc_rec = (
        _seed_mixed_tier_store_with_proc_chunk(tmp_path)
    )
    cue = [1.0] + [0.0] * (EMBED_DIM - 1)

    candidate_ids = {r.id for r, _ in store.query_similar(cue, k=10)}
    assert proc_rec.id in candidate_ids, (
        f"proc chunk {proc_rec.id} must be a real reachable candidate "
        f"(non-vacuity check) before checking its absence from hits; "
        f"candidates: {candidate_ids}"
    )

    # k_hits set to exceed the full 6-record pool (3 episodic + 2 schema +
    # 1 proc, all tied at cosine 1.0) so the filter's exclusion is the ONLY
    # thing keeping proc_rec out of hits -- a k_hits truncation coincidence
    # cannot mask a disabled tier filter here.
    resp = recall(
        store=store, cue_embedding=cue, cue_text="probe",
        session_id="r7_proc_exclude", k_hits=10, k_anti=2,
    )
    assert resp.cue_mode == "verbatim", (
        f"baseline default mode must be 'verbatim', got {resp.cue_mode!r}"
    )
    hit_ids = {h.record_id for h in resp.hits}
    assert proc_rec.id not in hit_ids, (
        f"procedural chunk {proc_rec.id} leaked into verbatim baseline hits: {hit_ids}"
    )

def test_baseline_recall_concept_mode_returns_all_tiers(tmp_path):
    from iai_mcp.retrieve import recall

    store, episodic_records, schema_records = _seed_mixed_tier_store(tmp_path)
    cue = [1.0] + [0.0] * (EMBED_DIM - 1)

    resp = recall(
        store=store, cue_embedding=cue, cue_text="probe",
        session_id="r7_concept", k_hits=5, k_anti=2, mode="concept",
    )
    assert resp.cue_mode == "concept"
    hit_ids = {h.record_id for h in resp.hits}
    schema_id_set = {r.id for r in schema_records}
    assert schema_id_set & hit_ids, (
        f"concept mode baseline must include schema tier (no filter); "
        f"schema_ids={schema_id_set}, hit_ids={hit_ids}"
    )

def test_dispatch_falls_back_to_baseline_on_graph_build_failure(tmp_path, monkeypatch):
    from iai_mcp import core
    from iai_mcp import retrieve as _retrieve_mod

    store, episodic_records, schema_records = _seed_mixed_tier_store(tmp_path)

    def fake_build(*args, **kwargs):
        raise RuntimeError("simulated graph build failure")

    monkeypatch.setattr(_retrieve_mod, "build_runtime_graph", fake_build)

    response = core.dispatch(
        store, "memory_recall",
        {"cue": "verbatim quote about migration",
         "session_id": "r7_fallback",
         "cue_embedding": [1.0] + [0.0] * (EMBED_DIM - 1)},
    )
    assert isinstance(response, dict)
    assert response["cue_mode"] == "verbatim", (
        f"verbatim cue must classify to verbatim even when graph build fails; "
        f"got {response['cue_mode']!r}"
    )
    schema_id_strs = {str(r.id) for r in schema_records}
    for h in response["hits"]:
        assert h["record_id"] not in schema_id_strs, (
            f"fallback path must apply verbatim filter; schema {h['record_id']} "
            f"appeared in hits despite graph build failure + verbatim cue"
        )

def test_recall_topk_stability_smoke(tmp_path):
    import importlib

    mod = importlib.import_module("tests.test_recall_topk_stability")
    assert hasattr(mod, "test_no_literal_surface_mutation"), (
        "regression-fence module must still expose its sentinel test"
    )
    mod.test_no_literal_surface_mutation(tmp_path)
