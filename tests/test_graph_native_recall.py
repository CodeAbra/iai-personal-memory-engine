from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest import mock
from uuid import uuid4

import pytest

from iai_mcp import retrieve
from iai_mcp.pipeline import recall_for_response
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


class _DetEmbedder:

    def __init__(self, dim: int = 384) -> None:
        self.DIM = dim
        self.DEFAULT_DIM = dim
        self.DEFAULT_MODEL_KEY = "test"

    def embed(self, text: str) -> list[float]:
        import hashlib
        import random

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        v = [rng.random() * 2 - 1 for _ in range(self.DIM)]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v] if n > 0 else v


def _make_record(vec: list[float], text: str) -> MemoryRecord:
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
        tags=["t"],
        language="en",
    )


@pytest.fixture
def seeded_store(tmp_path: Path) -> tuple[MemoryStore, _DetEmbedder, list[MemoryRecord]]:
    store = MemoryStore(path=tmp_path / "lancedb")
    store.root = tmp_path
    emb = _DetEmbedder(dim=store.embed_dim)
    recs = []
    for i in range(12):
        vec = emb.embed(f"fact-{i}")
        rec = _make_record(vec, f"synthetic fact {i}")
        store.insert(rec)
        recs.append(rec)
    return store, emb, recs


def test_A1_build_runtime_graph_attaches_node_payload(seeded_store):
    store, _emb, recs = seeded_store
    graph, _assignment, _rc = retrieve.build_runtime_graph(store)

    listed = list(graph.iter_nodes())
    assert len(listed) == len(recs)
    for rec in recs:
        payload = graph.get_payload(rec.id)
        assert "embedding" in payload, f"node {rec.id} missing embedding sidecar"
        assert "surface" in payload, f"node {rec.id} missing surface sidecar"
        assert "centrality" in payload, f"node {rec.id} missing centrality sidecar"
        assert "tier" in payload, f"node {rec.id} missing tier sidecar"
        import pytest as _pt
        assert list(payload["embedding"]) == _pt.approx(
            list(rec.embedding), rel=1e-5
        )
        assert payload["surface"] == rec.literal_surface
        assert payload["tier"] == rec.tier


def test_A2_seed_stage_reads_from_graph_not_store(seeded_store):
    store, emb, _recs = seeded_store
    graph, assignment, rich_club = retrieve.build_runtime_graph(store)

    cue = "explain the authentication migration for long-running deployments"

    class _Boom(RuntimeError):
        pass

    original_get = store.get

    def _explode(rid):
        from uuid import UUID
        l0 = UUID("00000000-0000-0000-0000-000000000001")
        if rid == l0:
            return None
        raise _Boom(f"store.get({rid}) — seed stage should not call this")

    with mock.patch.object(MemoryStore, "get", side_effect=_explode):
        resp = recall_for_response(
            store=store,
            graph=graph,
            assignment=assignment,
            rich_club=rich_club,
            embedder=emb,
            cue=cue,
            session_id="s",
            budget_tokens=1500,
        )
    assert len(resp.hits) >= 1


def test_A3_spread_stage_reads_from_graph_not_store(seeded_store):
    store, emb, _recs = seeded_store
    graph, assignment, rich_club = retrieve.build_runtime_graph(store)

    cue = "network stack changes for the web cache"

    class _Boom(RuntimeError):
        pass

    def _explode(rid):
        from uuid import UUID
        l0 = UUID("00000000-0000-0000-0000-000000000001")
        if rid == l0:
            return None
        raise _Boom(f"store.get({rid}) during spread/rank")

    with mock.patch.object(MemoryStore, "get", side_effect=_explode):
        resp = recall_for_response(
            store=store,
            graph=graph,
            assignment=assignment,
            rich_club=rich_club,
            embedder=emb,
            cue=cue,
            session_id="s",
            budget_tokens=1500,
        )
    assert isinstance(resp.hits, list)


def test_A4_verbatim_l0_fast_path_still_calls_store_get(seeded_store):
    store, emb, _recs = seeded_store
    from uuid import UUID
    l0_id = UUID("00000000-0000-0000-0000-000000000001")
    l0_vec = emb.embed("l0-identity")
    now = datetime.now(timezone.utc)
    l0_rec = MemoryRecord(
        id=l0_id,
        tier="semantic",
        literal_surface="L0 identity kernel",
        aaak_index="",
        embedding=l0_vec,
        community_id=None,
        centrality=0.0,
        detail_level=5,
        pinned=True,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=True,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=["identity"],
        language="en",
    )
    store.insert(l0_rec)
    graph, assignment, rich_club = retrieve.build_runtime_graph(store)

    cue = "hi"

    with mock.patch.object(MemoryStore, "get", wraps=store.get) as spy:
        _ = recall_for_response(
            store=store,
            graph=graph,
            assignment=assignment,
            rich_club=rich_club,
            embedder=emb,
            cue=cue,
            session_id="s",
            budget_tokens=1500,
        )
    assert spy.call_count >= 1


def test_A5_missing_node_attr_falls_back_to_store_get(seeded_store):
    store, emb, recs = seeded_store
    graph, assignment, rich_club = retrieve.build_runtime_graph(store)
    victims = [str(r.id) for r in recs[:6]]
    for nid_str in victims:
        sidecar = graph._node_payload.get(nid_str)
        if sidecar and "embedding" in sidecar:
            del sidecar["embedding"]

    cue = "summary of cli subcommand changes for the auth token rotation"
    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=emb,
        cue=cue,
        session_id="s",
        budget_tokens=1500,
    )
    assert len(resp.hits) >= 1


def test_A6_m04_correctness_no_regression(seeded_store):
    store, emb, recs = seeded_store
    graph, assignment, rich_club = retrieve.build_runtime_graph(store)

    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=emb,
        cue="synthetic fact 7",
        session_id="s",
        budget_tokens=1500,
    )
    assert len(resp.hits) >= 1
    seeded_ids = {r.id for r in recs}
    assert all(h.record_id in seeded_ids for h in resp.hits)
