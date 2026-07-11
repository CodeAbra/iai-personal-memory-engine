from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.store import EDGES_TABLE, MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

def _rec(
    *,
    text: str,
    vec: list[float] | None = None,
    tags: list[str] | None = None,
    language: str = "en",
) -> MemoryRecord:
    if vec is None:
        vec = [1.0] + [0.0] * (EMBED_DIM - 1)
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=vec,
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
        tags=list(tags or []),
        language=language,
    )

def test_profile_modulation_for_record_empty_profile():
    from iai_mcp.profile import profile_modulation_for_record

    rec = _rec(text="hi", tags=["domain:coding"])
    gains = profile_modulation_for_record(rec, profile_state={})
    assert isinstance(gains, dict)
    assert gains == {} or all(v == 1.0 for v in gains.values())

def test_profile_modulation_for_record_monotropism_depth_domain_tag():
    from iai_mcp.profile import profile_modulation_for_record

    rec = _rec(text="deep coding fact", tags=["domain:coding"])
    gains = profile_modulation_for_record(
        rec,
        profile_state={"monotropism_depth": {"coding": 0.9}},
    )
    assert "monotropism_depth" in gains
    assert gains["monotropism_depth"] > 1.0

def test_profile_modulation_for_record_wrong_domain_no_gain():
    from iai_mcp.profile import profile_modulation_for_record

    rec = _rec(text="gardening fact", tags=["domain:gardening"])
    gains = profile_modulation_for_record(
        rec,
        profile_state={"monotropism_depth": {"coding": 0.9}},
    )
    assert "monotropism_depth" not in gains

def test_profile_modulation_for_record_interest_boost():
    from iai_mcp.profile import profile_modulation_for_record

    rec = _rec(text="hi", tags=[])
    gains = profile_modulation_for_record(
        rec,
        profile_state={"interest_boost": 0.5},
    )
    assert "interest_boost" in gains
    assert gains["interest_boost"] > 1.0

@pytest.fixture(autouse=True)
def _patch_embedder(monkeypatch):
    from iai_mcp import embed as embed_mod

    class _FakeEmbedder:
        DIM = EMBED_DIM
        DEFAULT_DIM = EMBED_DIM
        DEFAULT_MODEL_KEY = "fake"

        def __init__(self, *args, **kwargs):
            self.DIM = EMBED_DIM

        def embed(self, text: str) -> list[float]:
            return [1.0] + [0.0] * (EMBED_DIM - 1)

        def embed_batch(self, texts):
            return [self.embed(t) for t in texts]

    monkeypatch.setattr(embed_mod, "Embedder", _FakeEmbedder)
    yield

def test_profile_modulation_edge_created_on_knob_affect(tmp_path, monkeypatch):
    from iai_mcp import retrieve
    from iai_mcp.pipeline import recall_for_response

    from iai_mcp.embed import Embedder as _E

    store = MemoryStore(path=tmp_path)

    r = _rec(text="code fact", tags=["domain:coding"])
    store.insert(r)

    graph, assignment, rc = retrieve.build_runtime_graph(store)
    profile_state = {"monotropism_depth": {"coding": 0.9}}

    recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rc,
        embedder=_E(),
        cue="anything",
        session_id="s1",
        budget_tokens=1500,
        profile_state=profile_state,
    )

    df = store.db.open_table(EDGES_TABLE).to_pandas()
    pm = df[df["edge_type"] == "profile_modulates"]
    assert len(pm) >= 1

def test_profile_modulates_edge_weight_positive(tmp_path):
    from iai_mcp import retrieve
    from iai_mcp.embed import Embedder as _E
    from iai_mcp.pipeline import recall_for_response

    store = MemoryStore(path=tmp_path)
    r = _rec(text="code fact", tags=["domain:coding"])
    store.insert(r)

    graph, assignment, rc = retrieve.build_runtime_graph(store)
    profile_state = {
        "monotropism_depth": {"coding": 0.9},
        "interest_boost": 0.5,
    }

    recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rc,
        embedder=_E(),
        cue="anything",
        session_id="s1",
        profile_state=profile_state,
    )

    df = store.db.open_table(EDGES_TABLE).to_pandas()
    pm = df[df["edge_type"] == "profile_modulates"]
    assert (pm["weight"] > 0).all()

def test_profile_modulation_gain_populates_on_record(tmp_path):
    from iai_mcp import retrieve
    from iai_mcp.embed import Embedder as _E
    from iai_mcp.pipeline import recall_for_response

    store = MemoryStore(path=tmp_path)
    r = _rec(text="code fact", tags=["domain:coding"])
    store.insert(r)

    graph, assignment, rc = retrieve.build_runtime_graph(store)
    profile_state = {"monotropism_depth": {"coding": 0.9}}

    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rc,
        embedder=_E(),
        cue="anything",
        session_id="s1",
        profile_state=profile_state,
    )

    assert len(resp.hits) >= 1
    df = store.db.open_table(EDGES_TABLE).to_pandas()
    pm = df[df["edge_type"] == "profile_modulates"]
    assert len(pm) >= 1

def test_profile_modulation_no_gain_when_state_empty(tmp_path):
    from iai_mcp import retrieve
    from iai_mcp.embed import Embedder as _E
    from iai_mcp.pipeline import recall_for_response

    store = MemoryStore(path=tmp_path)
    r = _rec(text="fact", tags=["domain:coding"])
    store.insert(r)

    graph, assignment, rc = retrieve.build_runtime_graph(store)

    recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rc,
        embedder=_E(),
        cue="anything",
        session_id="s1",
        profile_state={},
    )

    df = store.db.open_table(EDGES_TABLE).to_pandas()
    pm = df[df["edge_type"] == "profile_modulates"]
    assert len(pm) == 0
