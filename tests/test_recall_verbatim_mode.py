from __future__ import annotations

import math
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord
from tests._helpers import stub_embedder_for_store

class _ControlledEmbedder:
    DIM = EMBED_DIM

    def __init__(self) -> None:
        self.fixed: dict[str, list[float]] = {}

    def set_fixed(self, text: str, vec: list[float]) -> None:
        self.fixed[text] = list(vec)

    def embed(self, text: str) -> list[float]:
        if text in self.fixed:
            return list(self.fixed[text])
        import hashlib
        import random
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        v = [rng.random() * 2 - 1 for _ in range(self.DIM)]
        norm = sum(x * x for x in v) ** 0.5
        return [x / norm for x in v] if norm > 0 else v

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

def _unit_vector_with_cosine(cue_vec: list[float], target_cos: float) -> list[float]:
    cue = np.asarray(cue_vec, dtype=np.float32)
    cue_norm = float(np.linalg.norm(cue))
    if cue_norm == 0.0:
        raise ValueError("cue_vec must be non-zero")
    cue = cue / cue_norm

    probe = np.zeros(EMBED_DIM, dtype=np.float32)
    probe[1] = 1.0
    if abs(float(np.dot(cue, probe))) > 0.999:
        probe = np.zeros(EMBED_DIM, dtype=np.float32)
        probe[0] = 1.0
    orth = probe - float(np.dot(cue, probe)) * cue
    orth = orth / float(np.linalg.norm(orth))

    alpha = float(target_cos)
    beta = float(math.sqrt(max(0.0, 1.0 - alpha * alpha)))
    v = alpha * cue + beta * orth
    n = float(np.linalg.norm(v))
    if n > 0:
        v = v / n
    return v.astype(np.float32).tolist()

def _make_episodic(vec: list[float], text: str) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=list(vec),
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

def _make_proc_chunk(vec: list[float], text: str) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="procedural",
        literal_surface=text,
        aaak_index="",
        embedding=list(vec),
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

def _make_schema_hub(vec: list[float], text: str, pattern: str) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=text,
        aaak_index="",
        embedding=list(vec),
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

HUB_DEGREE = 8
HUB_COUNT = 10
VERBATIM_COUNT = 5

VERBATIM_CUES = [
    "verbatim recall the migration snapshot text",
    "exact phrase about pre-cleanup snapshot",
    "quote the deg_norm normalization fix",
    "what did the user say on day 17 about literal_preservation",
    'recall the "schema_reinforced event payload" exact wording',
]
VERBATIM_TEXTS = [
    "verbatim record migration snapshot text content payload one",
    "verbatim record pre-cleanup snapshot phrase content payload two",
    "verbatim record deg_norm normalization fix content payload three",
    "verbatim record day 17 literal_preservation content payload four",
    "verbatim record schema_reinforced event payload exact wording five",
]

def _seed_5_verbatim_plus_10_hubs(tmp_path):
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "lancedb")
    embedder = _ControlledEmbedder()

    verbatim_ids_per_cue: dict[str, "uuid.UUID"] = {}
    for cue, text in zip(VERBATIM_CUES, VERBATIM_TEXTS):
        cue_vec = embedder.embed(cue)
        embedder.set_fixed(cue, cue_vec)
        verbatim_vec = _unit_vector_with_cosine(cue_vec, 0.85)
        verbatim_rec = _make_episodic(verbatim_vec, text)
        store.insert(verbatim_rec)
        verbatim_ids_per_cue[cue] = verbatim_rec.id

    hub_ids: list = []
    edge_pairs: list = []
    distractor_idx = 0
    for h in range(HUB_COUNT):
        hub_vec = embedder.embed(f"schema-hub-{h}-distinct-content")
        hub_rec = _make_schema_hub(
            hub_vec, f"schema hub record {h}", pattern=f"hub:r5:{h}"
        )
        store.insert(hub_rec)
        hub_ids.append(hub_rec.id)
        for _ in range(HUB_DEGREE):
            d_vec = embedder.embed(f"r5-distractor-{distractor_idx}")
            d_rec = _make_episodic(d_vec, f"distractor junk {distractor_idx}")
            store.insert(d_rec)
            edge_pairs.append((hub_rec.id, d_rec.id))
            distractor_idx += 1

    store.boost_edges(edge_pairs, edge_type="schema_instance_of", delta=1.0)

    graph, assignment, rich_club = build_runtime_graph(store)
    return (
        store, embedder, graph, assignment, rich_club,
        verbatim_ids_per_cue, hub_ids, VERBATIM_CUES,
    )

def test_recall_response_back_compat_defaults():
    from iai_mcp.types import RecallResponse

    r = RecallResponse(
        hits=[],
        anti_hits=[],
        activation_trace=[],
        budget_used=0,
    )
    assert r.cue_mode == "concept", "cue_mode default must be 'concept'"
    assert r.patterns_observed == [], (
        "patterns_observed default must be [] for back-compat"
    )

def test_recall_for_response_signature_has_mode_kwarg_default_concept():
    import inspect
    from iai_mcp.pipeline import recall_for_response

    sig = inspect.signature(recall_for_response)
    assert "mode" in sig.parameters, "recall_for_response must accept mode kwarg"
    assert sig.parameters["mode"].default == "concept", (
        f"recall_for_response mode default must be 'concept', "
        f"got {sig.parameters['mode'].default!r}"
    )

def test_retrieve_recall_signature_has_mode_kwarg_default_verbatim():
    import inspect
    from iai_mcp.retrieve import recall

    sig = inspect.signature(recall)
    assert "mode" in sig.parameters, "retrieve.recall must accept mode kwarg"
    assert sig.parameters["mode"].default == "verbatim", (
        f"retrieve.recall mode default must be 'verbatim', "
        f"got {sig.parameters['mode'].default!r}"
    )

def test_verbatim_mode_response_carries_cue_mode_and_empty_patterns(tmp_path):
    from iai_mcp.pipeline import recall_for_response

    (store, embedder, graph, assignment, rich_club,
     verbatim_ids_per_cue, hub_ids, cues) = _seed_5_verbatim_plus_10_hubs(tmp_path)

    cue = cues[0]
    resp = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder, cue=cue,
        session_id="r5_test", mode="verbatim",
    )
    assert resp.cue_mode == "verbatim", f"expected cue_mode='verbatim', got {resp.cue_mode!r}"
    assert resp.patterns_observed == [], (
        f"verbatim mode must emit no patterns_observed, got {resp.patterns_observed!r}"
    )
    assert resp.hints == [], (
        f"verbatim mode must emit no hints (S4/curiosity/schema all suppressed), "
        f"got {resp.hints!r}"
    )

def test_verbatim_mode_hits_are_episodic_only(tmp_path):
    from iai_mcp.pipeline import recall_for_response

    (store, embedder, graph, assignment, rich_club,
     verbatim_ids_per_cue, hub_ids, cues) = _seed_5_verbatim_plus_10_hubs(tmp_path)

    resp = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder, cue=cues[0],
        session_id="r5_episodic", mode="verbatim",
    )
    hub_id_set = set(hub_ids)
    for h in resp.hits:
        assert h.record_id not in hub_id_set, (
            f"verbatim mode must EXCLUDE schema hubs from hits[], "
            f"hub {h.record_id} appeared at position "
            f"{[r.record_id for r in resp.hits].index(h.record_id)}"
        )
        rec = store.get(h.record_id)
        assert rec is not None, f"unknown record id {h.record_id} in hits"
        assert rec.tier == "episodic", (
            f"verbatim mode hit {h.record_id} has tier {rec.tier!r}, expected 'episodic'"
        )

def test_verbatim_mode_five_cue_variance_window_position_1_to_3(tmp_path):
    from iai_mcp.pipeline import recall_for_response

    (store, embedder, graph, assignment, rich_club,
     verbatim_ids_per_cue, hub_ids, cues) = _seed_5_verbatim_plus_10_hubs(tmp_path)

    positions: list[int] = []
    for cue in cues:
        resp = recall_for_response(
            store=store, graph=graph, assignment=assignment,
            rich_club=rich_club, embedder=embedder, cue=cue,
            session_id="r5_variance", mode="verbatim",
        )
        verbatim_id = verbatim_ids_per_cue[cue]
        ids = [h.record_id for h in resp.hits]
        assert verbatim_id in ids, (
            f"cue {cue!r}: matching verbatim {verbatim_id} not in hits at all "
            f"(hits ids: {ids})"
        )
        pos = ids.index(verbatim_id)
        positions.append(pos)
        assert pos <= 2, (
            f"cue {cue!r}: verbatim landed at pos {pos}, must be in 0..2 window. "
            f"All hits: {[(str(h.record_id)[:8], h.score) for h in resp.hits]}"
        )

    assert len(positions) == 5
    print(f"R5 variance positions across 5 cues: {positions}")

def test_verbatim_mode_position_1_strict_on_diagnostic_cue(tmp_path):
    from iai_mcp.pipeline import recall_for_response

    (store, embedder, graph, assignment, rich_club,
     verbatim_ids_per_cue, hub_ids, cues) = _seed_5_verbatim_plus_10_hubs(tmp_path)

    cue = cues[0]
    resp = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder, cue=cue,
        session_id="r5_strict", mode="verbatim",
    )
    verbatim_id = verbatim_ids_per_cue[cue]
    assert resp.hits, "verbatim mode produced empty hits"
    assert resp.hits[0].record_id == verbatim_id, (
        f"verbatim record must be at hits[0] (position-1 strict), "
        f"got {resp.hits[0].record_id} at pos 0; "
        f"matching verbatim {verbatim_id} at pos "
        f"{[h.record_id for h in resp.hits].index(verbatim_id) if verbatim_id in [h.record_id for h in resp.hits] else 'MISSING'}"
    )

def test_verbatim_mode_overrides_loose_knob_setting(tmp_path):
    from iai_mcp.pipeline import recall_for_response

    (store, embedder, graph, assignment, rich_club,
     verbatim_ids_per_cue, hub_ids, cues) = _seed_5_verbatim_plus_10_hubs(tmp_path)

    cue = cues[0]
    resp = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder, cue=cue,
        session_id="r5_override", mode="verbatim",
        profile_state={"literal_preservation": "loose"},
    )
    verbatim_id = verbatim_ids_per_cue[cue]
    ids = [h.record_id for h in resp.hits]
    assert verbatim_id in ids, "verbatim record missing under loose knob + verbatim mode"
    pos = ids.index(verbatim_id)
    assert pos <= 2, (
        f"verbatim mode must beat loose knob setting; got pos {pos} (must be 0..2). "
        f"hits: {[(str(h.record_id)[:8], h.score) for h in resp.hits]}"
    )
    hub_id_set = set(hub_ids)
    for h in resp.hits:
        assert h.record_id not in hub_id_set, (
            f"hub {h.record_id} leaked into hits despite verbatim mode override of loose knob"
        )

def test_concept_mode_default_preserves_locked_baseline(tmp_path):
    from iai_mcp.pipeline import recall_for_response

    (store, embedder, graph, assignment, rich_club,
     verbatim_ids_per_cue, hub_ids, cues) = _seed_5_verbatim_plus_10_hubs(tmp_path)

    resp_default = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder, cue=cues[0],
        session_id="r5_default",
    )
    assert resp_default.cue_mode == "concept", (
        "recall_for_response default mode must be 'concept'"
    )

def test_verbatim_mode_excludes_procedural_chunk_from_hits(tmp_path):
    """Pins pipeline.py's episodic_ids narrowing (built when mode=="verbatim",
    applied to reachable_indices before ranking): a tier="procedural" chunk is
    structurally excluded from verbatim hits regardless of tags or cosine.
    The mutant that turns this RED: removing/disabling the episodic_ids
    narrowing at pipeline.py's reachable_indices filter step.
    """
    from iai_mcp.pipeline import recall_for_response
    from iai_mcp.retrieve import build_runtime_graph

    (store, embedder, graph, assignment, rich_club,
     verbatim_ids_per_cue, hub_ids, cues) = _seed_5_verbatim_plus_10_hubs(tmp_path)

    cue = cues[0]
    cue_vec = embedder.embed(cue)
    proc_vec = _unit_vector_with_cosine(cue_vec, 0.85)
    proc_rec = _make_proc_chunk(proc_vec, "procedural chunk content payload r5")
    store.insert(proc_rec)

    graph, assignment, rich_club = build_runtime_graph(store)

    candidate_ids = {r.id for r, _ in store.query_similar(cue_vec, k=32)}
    assert proc_rec.id in candidate_ids, (
        f"proc chunk {proc_rec.id} must be a real reachable candidate "
        f"(non-vacuity check) before checking its absence from hits; "
        f"candidates: {candidate_ids}"
    )

    resp = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder, cue=cue,
        session_id="r5_proc_exclude", mode="verbatim",
    )
    hit_ids = {h.record_id for h in resp.hits}
    assert proc_rec.id not in hit_ids, (
        f"procedural chunk {proc_rec.id} leaked into verbatim hits: {hit_ids}"
    )
    for h in resp.hits:
        rec = store.get(h.record_id)
        assert rec is not None, f"unknown record id {h.record_id} in hits"
        assert rec.tier == "episodic", (
            f"verbatim hit {h.record_id} has tier {rec.tier!r}, expected 'episodic'"
        )

def test_dispatch_verbatim_5_cue_variance_window(tmp_path, monkeypatch):
    from iai_mcp import core

    (store, embedder, graph, assignment, rich_club,
     verbatim_ids_per_cue, hub_ids, cues) = _seed_5_verbatim_plus_10_hubs(tmp_path)
    stub_embedder_for_store(monkeypatch, embedder)

    positions: list[int] = []
    for cue in cues:
        response = core.dispatch(
            store, "memory_recall",
            {"cue": cue, "session_id": "r5_dispatch_variance",
             "cue_embedding": embedder.embed(cue)},
        )
        assert response["cue_mode"] == "verbatim", (
            f"cue {cue!r} should classify to verbatim, got {response['cue_mode']!r}"
        )
        verbatim_id = str(verbatim_ids_per_cue[cue])
        ids = [h["record_id"] for h in response["hits"]]
        assert verbatim_id in ids, (
            f"cue {cue!r}: matching verbatim {verbatim_id} missing from dispatch response. "
            f"hits ids: {ids}"
        )
        pos = ids.index(verbatim_id)
        positions.append(pos)
        assert pos <= 2, (
            f"cue {cue!r}: dispatch verbatim landed at pos {pos}, must be in 0..2 window"
        )

    assert len(positions) == 5
    print(f"R5 dispatch variance positions across 5 cues: {positions}")

def test_dispatch_verbatim_position_1_strict_diagnostic_cue(tmp_path, monkeypatch):
    from iai_mcp import core

    (store, embedder, graph, assignment, rich_club,
     verbatim_ids_per_cue, hub_ids, cues) = _seed_5_verbatim_plus_10_hubs(tmp_path)
    stub_embedder_for_store(monkeypatch, embedder)

    cue = cues[0]
    response = core.dispatch(
        store, "memory_recall",
        {"cue": cue, "session_id": "r5_dispatch_strict",
         "cue_embedding": embedder.embed(cue)},
    )
    assert response["cue_mode"] == "verbatim"
    assert response["hits"], "dispatch produced empty hits for verbatim cue"
    verbatim_id = str(verbatim_ids_per_cue[cue])
    assert response["hits"][0]["record_id"] == verbatim_id, (
        f"verbatim must be at hits[0] (position-1 strict via dispatch); "
        f"got {response['hits'][0]['record_id']} at pos 0"
    )

def test_dispatch_verbatim_overrides_loose_knob_setting(tmp_path, monkeypatch):
    from iai_mcp import core

    (store, embedder, graph, assignment, rich_club,
     verbatim_ids_per_cue, hub_ids, cues) = _seed_5_verbatim_plus_10_hubs(tmp_path)
    stub_embedder_for_store(monkeypatch, embedder)

    original_lp = core._profile_state.get("literal_preservation", "strong")
    core._profile_state["literal_preservation"] = "loose"
    try:
        cue = cues[0]
        response = core.dispatch(
            store, "memory_recall",
            {"cue": cue, "session_id": "r5_dispatch_override",
             "cue_embedding": embedder.embed(cue)},
        )
        assert response["cue_mode"] == "verbatim"
        verbatim_id = str(verbatim_ids_per_cue[cue])
        ids = [h["record_id"] for h in response["hits"]]
        assert verbatim_id in ids, "verbatim missing under loose knob + verbatim cue"
        pos = ids.index(verbatim_id)
        assert pos <= 2, (
            f"verbatim mode via dispatch must override loose knob; got pos {pos}"
        )
        hub_id_strs = {str(h) for h in hub_ids}
        for h in response["hits"]:
            assert h["record_id"] not in hub_id_strs, (
                f"hub {h['record_id']} leaked despite verbatim mode + loose knob"
            )
    finally:
        core._profile_state["literal_preservation"] = original_lp
