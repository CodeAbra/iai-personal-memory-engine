from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

pytestmark = pytest.mark.perf

from iai_mcp import pipeline, retrieve
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord

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

class _FakeEmbedder:

    def __init__(self, dim: int = 384) -> None:
        self.DIM = dim
        self.DEFAULT_DIM = dim
        self.DEFAULT_MODEL_KEY = "test"

    def embed(self, text: str) -> list[float]:
        import hashlib
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        rng = np.random.default_rng(int(digest[:16], 16))
        v = rng.standard_normal(self.DIM).astype(np.float32)
        v /= float(np.linalg.norm(v)) or 1.0
        return v.tolist()

def _make_record(dim: int, seed: int, text: str = "fact") -> MemoryRecord:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= float(np.linalg.norm(v)) or 1.0
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=f"{text}-{seed}",
        aaak_index="",
        embedding=v.tolist(),
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
        tags=["t"],
        language="en",
    )

@pytest.fixture
def seeded_store(tmp_path: Path, request):
    n = getattr(request, "param", 25)
    store = MemoryStore(path=tmp_path / "lancedb")
    store.root = tmp_path
    for i in range(n):
        store.insert(_make_record(store.embed_dim, seed=i + 1))
    return store

def test_R1_vectorized_rank_produces_sorted_descending(seeded_store):
    emb = _FakeEmbedder(dim=seeded_store.embed_dim)
    graph, assignment, rich_club = retrieve.build_runtime_graph(seeded_store)

    resp = pipeline.recall_for_response(
        store=seeded_store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=emb,
        cue="fact-17",
        session_id="t-R1",
        budget_tokens=4000,
    )
    assert len(resp.hits) > 0
    scores = [h.score for h in resp.hits]
    assert scores == sorted(scores, reverse=True), (
        f"hits not sorted desc: {scores}"
    )
    for h in resp.hits:
        assert h.literal_surface
        assert h.reason
        assert isinstance(h.score, float)

def test_R2_no_per_record_cosine_in_rank_loop(seeded_store, monkeypatch):
    emb = _FakeEmbedder(dim=seeded_store.embed_dim)
    graph, assignment, rich_club = retrieve.build_runtime_graph(seeded_store)

    call_count = {"n": 0}
    real_cosine = pipeline._cosine

    def counting_cosine(*a, **kw):
        call_count["n"] += 1
        return real_cosine(*a, **kw)

    monkeypatch.setattr(pipeline, "_cosine", counting_cosine)
    pipeline.recall_for_response(
        store=seeded_store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=emb,
        cue="fact-1",
        session_id="t-R2",
        budget_tokens=4000,
    )
    assert call_count["n"] < 20, (
        f"pipeline._cosine called {call_count['n']} times — "
        "rank or seed stage is still in a per-record loop"
    )

@pytest.mark.parametrize("seeded_store", [300], indirect=True)
def test_R3_rank_stage_latency_under_budget(seeded_store):
    from _perf_helpers import best_of_n, skip_if_loaded

    skip_if_loaded()

    emb = _FakeEmbedder(dim=seeded_store.embed_dim)
    graph, assignment, rich_club = retrieve.build_runtime_graph(seeded_store)

    pipeline.recall_for_response(
        store=seeded_store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=emb, cue="warmup",
        session_id="warmup", budget_tokens=4000,
    )

    from unittest.mock import patch

    def _one_dt_ms() -> float:
        with patch.object(
            seeded_store, "append_provenance_batch", lambda *a, **kw: None
        ):
            t0 = time.perf_counter()
            pipeline.recall_for_response(
                store=seeded_store, graph=graph, assignment=assignment,
                rich_club=rich_club, embedder=emb, cue="fact-17",
                session_id="t-R3", budget_tokens=4000,
            )
            return (time.perf_counter() - t0) * 1000.0

    dt_ms = best_of_n(_one_dt_ms, n=3)
    assert dt_ms < 120.0, (
        f"vectorized rank-stage best-of-3 recall took {dt_ms:.1f} ms at N=300 "
        "(provenance writes mocked)"
    )

def test_R4_empty_reachable_returns_empty_hits(tmp_path: Path):
    store = MemoryStore(path=tmp_path / "lancedb")
    store.root = tmp_path
    emb = _FakeEmbedder(dim=store.embed_dim)
    graph, assignment, rich_club = retrieve.build_runtime_graph(store)
    resp = pipeline.recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=emb, cue="nothing",
        session_id="t-R4", budget_tokens=4000,
    )
    assert resp.hits == []

def test_R5_tie_break_deterministic_by_uuid(tmp_path: Path, monkeypatch):
    import iai_mcp.pipeline as _p
    monkeypatch.setattr(_p, "_age_penalty", lambda _ts: 0.0)

    store = MemoryStore(path=tmp_path / "lancedb")
    store.root = tmp_path
    rng = np.random.default_rng(42)
    v = rng.standard_normal(store.embed_dim).astype(np.float32)
    v /= float(np.linalg.norm(v)) or 1.0
    ids = []
    for i in range(5):
        now = datetime.now(timezone.utc)
        rec = MemoryRecord(
            id=uuid4(),
            tier="episodic",
            literal_surface=f"tie-{i}",
            aaak_index="",
            embedding=v.tolist(),
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
        store.insert(rec)
        ids.append(rec.id)
    emb = _FakeEmbedder(dim=store.embed_dim)
    monkeypatched = v.tolist()
    emb.embed = lambda t, _v=monkeypatched: _v  # type: ignore[method-assign]
    graph, assignment, rich_club = retrieve.build_runtime_graph(store)

    resp1 = pipeline.recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=emb, cue="x",
        session_id="t-R5a", budget_tokens=4000,
    )
    resp2 = pipeline.recall_for_response(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=emb, cue="x",
        session_id="t-R5b", budget_tokens=4000,
    )
    got1 = [h.record_id for h in resp1.hits]
    got2 = [h.record_id for h in resp2.hits]
    assert got1 == got2, "tie-break must be deterministic across calls"

def test_R6_missing_centrality_falls_back_to_zero(seeded_store):
    emb = _FakeEmbedder(dim=seeded_store.embed_dim)
    graph, assignment, rich_club = retrieve.build_runtime_graph(seeded_store)
    for nid in list(graph.iter_nodes()):
        sidecar = graph._node_payload.get(str(nid))
        if sidecar and "centrality" in sidecar:
            del sidecar["centrality"]

    resp = pipeline.recall_for_response(
        store=seeded_store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=emb, cue="fact-3",
        session_id="t-R6", budget_tokens=4000,
    )
    assert len(resp.hits) > 0
