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
        tags=["schema", "draft", f"hub:test:{pattern}"],
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
HUB_COUNT = 5
CUE_TEXT = "literal preservation cue marker R3"

def _seed_verbatim_vs_hubs(tmp_path):
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "lancedb")
    embedder = _ControlledEmbedder()

    cue_vec = embedder.embed(CUE_TEXT)
    embedder.set_fixed(CUE_TEXT, cue_vec)

    verbatim_vec = _unit_vector_with_cosine(cue_vec, 0.60)
    verbatim_rec = _make_episodic(
        verbatim_vec, "the exact verbatim quote you are looking for"
    )
    store.insert(verbatim_rec)

    hub_ids: list = []
    edge_pairs: list = []
    distractor_idx = 0
    for h in range(HUB_COUNT):
        hub_vec = _unit_vector_with_cosine(cue_vec, 0.50)
        hub_rec = _make_schema_hub(
            hub_vec, f"schema hub record {h}", pattern=f"hub:test:{h}"
        )
        store.insert(hub_rec)
        hub_ids.append(hub_rec.id)
        for _ in range(HUB_DEGREE):
            d_vec = embedder.embed(f"distractor-{distractor_idx}-far-from-cue")
            d_rec = _make_episodic(d_vec, f"unrelated junk {distractor_idx}")
            store.insert(d_rec)
            edge_pairs.append((hub_rec.id, d_rec.id))
            distractor_idx += 1

    store.boost_edges(edge_pairs, edge_type="schema_instance_of", delta=1.0)

    graph, assignment, rich_club = build_runtime_graph(store)
    return (
        store, embedder, graph, assignment, rich_club,
        verbatim_rec.id, hub_ids, CUE_TEXT,
    )

def _verbatim_position(resp, verbatim_id) -> int | None:
    ids = [h.record_id for h in resp.hits]
    if verbatim_id not in ids:
        return None
    return ids.index(verbatim_id)

def test_scale_constant_keys_match_profile_enum():
    from iai_mcp.pipeline import LITERAL_PRESERVATION_W_DEGREE_SCALE

    assert LITERAL_PRESERVATION_W_DEGREE_SCALE == {
        "strong": 0.3,
        "medium": 1.0,
        "loose": 1.5,
    }, (
        "Scale map must use profile.py:87 enum keys "
        "(`strong|medium|loose`), not `balanced/weak`. "
        f"Got {LITERAL_PRESERVATION_W_DEGREE_SCALE}"
    )

def test_literal_preservation_strong_ranks_verbatim_high(tmp_path):
    from iai_mcp.pipeline import recall_for_response

    (store, embedder, graph, assignment, rich_club,
     verbatim_id, hub_ids, cue_text) = _seed_verbatim_vs_hubs(tmp_path)

    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=embedder,
        cue=cue_text,
        session_id="r3_strong",
        budget_tokens=2000,
        profile_state={"literal_preservation": "strong"},
    )
    pos = _verbatim_position(resp, verbatim_id)
    assert pos is not None, (
        f"verbatim must be in hits with strong scale; "
        f"hits={[h.record_id for h in resp.hits]}"
    )
    assert pos <= 2, (
        f"strong scale: verbatim must rank in top-3 "
        f"(pos≤2); got pos={pos}, hits={[h.record_id for h in resp.hits]}"
    )

def test_literal_preservation_loose_ranks_verbatim_low(tmp_path):
    from iai_mcp.pipeline import recall_for_response

    (store, embedder, graph, assignment, rich_club,
     verbatim_id, hub_ids, cue_text) = _seed_verbatim_vs_hubs(tmp_path)

    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=embedder,
        cue=cue_text,
        session_id="r3_loose",
        budget_tokens=2000,
        profile_state={"literal_preservation": "loose"},
    )
    pos = _verbatim_position(resp, verbatim_id)
    assert pos is not None, (
        f"verbatim must still be in hits with loose scale "
        f"(it's ranked low but not excluded); "
        f"hits={[h.record_id for h in resp.hits]}"
    )
    assert pos >= 4, (
        f"loose scale: verbatim must rank below top-4 "
        f"(pos≥4); got pos={pos}, hits={[h.record_id for h in resp.hits]}"
    )

def test_literal_preservation_knob_moves_verbatim_position(tmp_path):
    from iai_mcp.pipeline import recall_for_response

    (store, embedder, graph, assignment, rich_club,
     verbatim_id, hub_ids, cue_text) = _seed_verbatim_vs_hubs(tmp_path)

    resp_strong = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder, cue=cue_text,
        session_id="r3_delta_strong", budget_tokens=2000,
        profile_state={"literal_preservation": "strong"},
    )
    resp_loose = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder, cue=cue_text,
        session_id="r3_delta_loose", budget_tokens=2000,
        profile_state={"literal_preservation": "loose"},
    )

    pos_strong = _verbatim_position(resp_strong, verbatim_id)
    pos_loose = _verbatim_position(resp_loose, verbatim_id)
    assert pos_strong is not None and pos_loose is not None, (
        f"verbatim must be present in both responses; "
        f"strong_hits={[h.record_id for h in resp_strong.hits]}, "
        f"loose_hits={[h.record_id for h in resp_loose.hits]}"
    )
    delta = pos_loose - pos_strong
    assert delta >= 3, (
        f"R3 acceptance: position delta between strong and loose must be "
        f">= 3. got pos_strong={pos_strong}, pos_loose={pos_loose}, "
        f"delta={delta}"
    )

def test_literal_preservation_medium_is_normalize_only_baseline(tmp_path):
    from iai_mcp.pipeline import recall_for_response

    (store, embedder, graph, assignment, rich_club,
     verbatim_id, hub_ids, cue_text) = _seed_verbatim_vs_hubs(tmp_path)

    resp_strong = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder, cue=cue_text,
        session_id="r3_medium_strong_ref", budget_tokens=2000,
        profile_state={"literal_preservation": "strong"},
    )
    resp_medium = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder, cue=cue_text,
        session_id="r3_medium", budget_tokens=2000,
        profile_state={"literal_preservation": "medium"},
    )
    resp_loose = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder, cue=cue_text,
        session_id="r3_medium_loose_ref", budget_tokens=2000,
        profile_state={"literal_preservation": "loose"},
    )
    pos_s = _verbatim_position(resp_strong, verbatim_id)
    pos_m = _verbatim_position(resp_medium, verbatim_id)
    pos_l = _verbatim_position(resp_loose, verbatim_id)
    assert pos_s is not None and pos_m is not None and pos_l is not None
    assert pos_s <= pos_m <= pos_l, (
        f"medium must be between strong and loose: "
        f"strong={pos_s}, medium={pos_m}, loose={pos_l}"
    )

def test_empty_profile_state_falls_back_to_medium_scale(tmp_path):
    from iai_mcp.pipeline import recall_for_response

    (store, embedder, graph, assignment, rich_club,
     verbatim_id, hub_ids, cue_text) = _seed_verbatim_vs_hubs(tmp_path)

    resp_empty = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder, cue=cue_text,
        session_id="r3_empty", budget_tokens=2000,
        profile_state={},
    )
    resp_medium = recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder, cue=cue_text,
        session_id="r3_medium_ref", budget_tokens=2000,
        profile_state={"literal_preservation": "medium"},
    )
    ids_empty = [h.record_id for h in resp_empty.hits]
    ids_medium = [h.record_id for h in resp_medium.hits]
    assert ids_empty == ids_medium, (
        f"empty profile_state must equal medium baseline. "
        f"empty={ids_empty}, medium={ids_medium}"
    )
    scores_empty = [h.score for h in resp_empty.hits]
    scores_medium = [h.score for h in resp_medium.hits]
    for a, b in zip(scores_empty, scores_medium):
        assert abs(a - b) < 1e-5, (
            f"empty and medium scores must match within float noise; "
            f"empty={scores_empty}, medium={scores_medium}"
        )

def test_dispatch_passes_profile_state_to_recall_for_response(tmp_path, monkeypatch):
    from iai_mcp import core, pipeline as _pipeline_mod
    from iai_mcp.types import RecallResponse

    (store, embedder, graph, assignment, rich_club,
     verbatim_id, hub_ids, cue_text) = _seed_verbatim_vs_hubs(tmp_path)

    captured: dict = {}

    def _capturing_recall(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return RecallResponse(
            hits=[], anti_hits=[], activation_trace=[],
            budget_used=0, hints=[],
        )

    monkeypatch.setattr(_pipeline_mod, "recall_for_response", _capturing_recall)
    monkeypatch.setitem(core._profile_state, "literal_preservation", "strong")

    core.dispatch(
        store, "memory_recall",
        {"cue": cue_text, "session_id": "dispatch_kwarg_capture"},
    )

    assert "kwargs" in captured, "recall_for_response was not called by dispatch"
    kwargs = captured["kwargs"]
    assert "profile_state" in kwargs, (
        f"dispatch must pass profile_state= kwarg; got kwargs={list(kwargs.keys())}"
    )
    ps = kwargs["profile_state"]
    assert isinstance(ps, dict), f"profile_state must be a dict, got {type(ps)}"
    assert "literal_preservation" in ps, (
        f"profile_state must carry literal_preservation; "
        f"got keys={list(ps.keys())}"
    )
    assert ps["literal_preservation"] == "strong", (
        f"dispatch must thread the live knob value; got {ps['literal_preservation']}"
    )

def test_dispatch_end_to_end_knob_moves_verbatim_position(tmp_path, monkeypatch):
    from iai_mcp import core
    from uuid import UUID

    (store, embedder, graph, assignment, rich_club,
     verbatim_id, hub_ids, cue_text) = _seed_verbatim_vs_hubs_separated(tmp_path)

    # The exact-similarity authority pins an exact-cosine match at the head of
    # the response regardless of the knob under test, which would swallow the
    # position delta this test exercises. Disable it so this test keeps
    # exercising the knob's end-to-end effect on the graph-ranked path.
    monkeypatch.setenv("IAI_MCP_EXACT_AUTHORITY_OFF", "1")

    stub_embedder_for_store(monkeypatch, embedder)

    monkeypatch.setitem(core._profile_state, "literal_preservation", "strong")
    resp_strong = core.dispatch(
        store, "memory_recall",
        {"cue": cue_text, "session_id": "e2e_dispatch_strong",
         "budget_tokens": 2000},
    )
    monkeypatch.setitem(core._profile_state, "literal_preservation", "loose")
    resp_loose = core.dispatch(
        store, "memory_recall",
        {"cue": cue_text, "session_id": "e2e_dispatch_loose",
         "budget_tokens": 2000},
    )

    def _ids(resp):
        return [UUID(h["record_id"]) for h in resp["hits"]]

    ids_strong = _ids(resp_strong)
    ids_loose = _ids(resp_loose)
    assert verbatim_id in ids_strong, (
        f"verbatim must appear in strong dispatch response; "
        f"got {ids_strong}"
    )
    assert verbatim_id in ids_loose, (
        f"verbatim must appear in loose dispatch response; "
        f"got {ids_loose}"
    )
    pos_strong = ids_strong.index(verbatim_id)
    pos_loose = ids_loose.index(verbatim_id)
    delta = pos_loose - pos_strong
    assert delta >= 3, (
        f"E2E via dispatch: position delta between strong and loose must "
        f"be >= 3. got pos_strong={pos_strong}, pos_loose={pos_loose}, "
        f"delta={delta}"
    )


HUB_CUE_TEXT = "team onboarding workflow questions"
SEPARATED_FILLER_CLUSTERS = 12
SEPARATED_FILLER_SIZE = 20
SEPARATED_VERBATIM_COS = 0.52


def _seed_verbatim_vs_hubs_separated(tmp_path):
    """Seed a population where the verbatim record lives in its own community,
    distinct from every hub community, while the high-degree hubs share the
    dispatch cue's gated neighborhood.

    The dispatch path derives node degree from `hebbian` edges and selects the
    community gate by max-node cosine to the cue. Two properties make the
    strong-vs-loose position delta provable end-to-end:

    1. Hubs carry real `hebbian` degree, so the W_DEGREE term is live in the
       dispatch path (a `schema_instance_of`-only graph leaves that term inert).
    2. The verbatim record is edgeless, so community detection places it in a
       singleton community that differs from every hub community — it does not
       inherit hub-cluster degree, and the separation is verified fail-loud
       below.

    The node count is pushed past the flat-assignment floor with filler clusters
    that sit far from the cue, so community detection runs for real instead of
    collapsing everything into one community.
    """
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "lancedb")
    embedder = _ControlledEmbedder()

    hub_cue_vec = embedder.embed(HUB_CUE_TEXT)
    embedder.set_fixed(HUB_CUE_TEXT, hub_cue_vec)

    verbatim_vec = _unit_vector_with_cosine(hub_cue_vec, SEPARATED_VERBATIM_COS)
    verbatim_rec = _make_episodic(
        verbatim_vec, "the unrelated paragraph of stored prose"
    )
    store.insert(verbatim_rec)

    edge_pairs: list = []
    hub_ids: list = []
    distractor_idx = 0
    for h in range(HUB_COUNT):
        hub_vec = _unit_vector_with_cosine(hub_cue_vec, 0.50)
        hub_rec = _make_schema_hub(
            hub_vec, f"schema hub record {h}", pattern=f"hub:test:{h}"
        )
        store.insert(hub_rec)
        hub_ids.append(hub_rec.id)
        for _ in range(HUB_DEGREE):
            d_vec = embedder.embed(f"distractor-{distractor_idx}-far-from-cue")
            d_rec = _make_episodic(d_vec, f"unrelated junk {distractor_idx}")
            store.insert(d_rec)
            edge_pairs.append((hub_rec.id, d_rec.id))
            distractor_idx += 1

    filler_idx = 0
    for c in range(SEPARATED_FILLER_CLUSTERS):
        anchor = embedder.embed(f"filler-anchor-{c}")
        center = _make_episodic(anchor, f"filler center {c}")
        store.insert(center)
        for _ in range(SEPARATED_FILLER_SIZE):
            fv = _unit_vector_with_cosine(anchor, 0.9)
            fr = _make_episodic(fv, f"filler leaf {filler_idx}")
            store.insert(fr)
            edge_pairs.append((center.id, fr.id))
            filler_idx += 1

    store.boost_edges(edge_pairs, edge_type="hebbian", delta=1.0)

    graph, assignment, rich_club = build_runtime_graph(store)

    verbatim_comm = assignment.node_to_community.get(verbatim_rec.id)
    hub_comms = {assignment.node_to_community.get(hid) for hid in hub_ids}
    assert verbatim_comm is not None, (
        "verbatim record did not land in any community after graph build; "
        f"node_to_community size={len(assignment.node_to_community)}"
    )
    assert verbatim_comm not in hub_comms, (
        "fixture geometry failed to separate communities: verbatim shares a "
        f"community ({verbatim_comm}) with a hub. hub communities={hub_comms}. "
        "Community detection must place verbatim in its own community so it "
        "does not inherit hub-cluster degree."
    )

    return (
        store, embedder, graph, assignment, rich_club,
        verbatim_rec.id, hub_ids, HUB_CUE_TEXT,
    )
