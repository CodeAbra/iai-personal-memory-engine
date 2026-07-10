from __future__ import annotations

import math
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord

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

def test_build_runtime_graph_sets_max_degree_attribute(tmp_path):
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "lancedb")
    embedder = _ControlledEmbedder()
    for i in range(5):
        vec = embedder.embed(f"isolated-{i}")
        store.insert(_make_episodic(vec, f"text {i}"))

    graph, _, _ = build_runtime_graph(store)
    assert hasattr(graph, "_max_degree"), "graph must carry _max_degree attribute"
    assert isinstance(graph._max_degree, int), "_max_degree must be int"
    assert graph._max_degree >= 0

def test_cache_round_trip_preserves_max_degree(tmp_path):
    from iai_mcp import runtime_graph_cache
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "lancedb")
    embedder = _ControlledEmbedder()
    ids = []
    for i in range(6):
        vec = embedder.embed(f"node-{i}")
        rec = _make_episodic(vec, f"surface {i}")
        store.insert(rec)
        ids.append(rec.id)
    store.boost_edges(
        [(ids[0], ids[j]) for j in range(1, 6)],
        edge_type="hebbian",
        delta=1.0,
    )

    graph1, _, _ = build_runtime_graph(store)
    md1 = graph1._max_degree
    assert md1 >= 5, f"expected hub degree >= 5, got {md1}"

    cache = runtime_graph_cache.try_load(store)
    assert cache is not None, "cache must round-trip"
    assert len(cache) == 5, f"try_load must return 5-tuple, got {len(cache)}"
    _assignment, _rich_club, _node_payload, cached_md, _node_degrees = cache
    assert int(cached_md) == md1

    graph2, _, _ = build_runtime_graph(store)
    assert graph2._max_degree == md1

def test_empty_store_max_degree_is_zero(tmp_path):
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "lancedb")
    embedder = _ControlledEmbedder()
    rec = _make_episodic(embedder.embed("only"), "only one")
    store.insert(rec)

    graph, _, _ = build_runtime_graph(store)
    assert graph._max_degree == 0

def _seed_hub_vs_verbatim(tmp_path, hub_degree: int = 64):
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "lancedb")
    embedder = _ControlledEmbedder()

    cue_text = "verbatim cue marker A"
    cue_vec = embedder.embed(cue_text)
    embedder.set_fixed(cue_text, cue_vec)

    hub_vec = _unit_vector_with_cosine(cue_vec, 0.30)
    verbatim_vec = _unit_vector_with_cosine(cue_vec, 0.60)

    hub_rec = _make_episodic(hub_vec, "hub schema record")
    verbatim_rec = _make_episodic(
        verbatim_vec, "the exact verbatim quote you are looking for"
    )
    store.insert(hub_rec)
    store.insert(verbatim_rec)

    distractor_ids = []
    edge_pairs = []
    for i in range(hub_degree):
        d_vec = embedder.embed(f"distractor-{i}-far-from-cue")
        d_rec = _make_episodic(d_vec, f"unrelated junk {i}")
        store.insert(d_rec)
        distractor_ids.append(d_rec.id)
        edge_pairs.append((hub_rec.id, d_rec.id))

    store.boost_edges(edge_pairs, edge_type="hebbian", delta=1.0)

    graph, assignment, rich_club = build_runtime_graph(store)
    return (
        store, embedder, graph, assignment, rich_club,
        hub_rec.id, verbatim_rec.id, cue_text,
    )

def test_normalized_degree_lets_verbatim_outrank_hub(tmp_path):
    from iai_mcp.pipeline import recall_for_response

    (store, embedder, graph, assignment, rich_club,
     hub_id, verbatim_id, cue_text) = _seed_hub_vs_verbatim(tmp_path)

    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=embedder,
        cue=cue_text,
        session_id="r2_acceptance",
        budget_tokens=1500,
    )
    hit_ids = [h.record_id for h in resp.hits]
    assert verbatim_id in hit_ids, f"verbatim must appear in hits; got {hit_ids}"
    if hub_id in hit_ids:
        verb_pos = hit_ids.index(verbatim_id)
        hub_pos = hit_ids.index(hub_id)
        assert verb_pos < hub_pos, (
            f"verbatim must rank above hub under new formula. "
            f"verbatim@{verb_pos} hub@{hub_pos} hits={hit_ids}"
        )
    assert hit_ids[0] == verbatim_id, (
        f"verbatim must be position-0 under new formula; got {hit_ids[0]} "
        f"(verbatim_id={verbatim_id}, hits={hit_ids})"
    )

def test_old_formula_would_have_ranked_hub_above_verbatim(tmp_path):
    from math import log

    (store, embedder, graph, _assignment, _rich_club,
     hub_id, verbatim_id, cue_text) = _seed_hub_vs_verbatim(tmp_path)

    cue_vec = np.asarray(embedder.embed(cue_text), dtype=np.float32)
    cue_vec = cue_vec / float(np.linalg.norm(cue_vec))

    def _live_cos(rid):
        emb = graph.get_embedding(rid)
        v = np.asarray(emb, dtype=np.float32)
        return float(np.dot(cue_vec, v))

    hub_cos = _live_cos(hub_id)
    verbatim_cos = _live_cos(verbatim_id)

    deg_dict = {str(nid): deg for nid, deg in graph.degrees()}
    hub_deg = float(deg_dict.get(str(hub_id), 0))
    verbatim_deg = float(deg_dict.get(str(verbatim_id), 0))

    W_COSINE = 1.0
    W_DEGREE = 0.1
    hub_old = W_COSINE * hub_cos + W_DEGREE * log(1.0 + hub_deg)
    verbatim_old = W_COSINE * verbatim_cos + W_DEGREE * log(1.0 + verbatim_deg)
    assert hub_old > verbatim_old, (
        "OLD formula must rank hub above verbatim — otherwise the R2 "
        "fix would not change ordering and the new test would be vacuous. "
        f"hub_old={hub_old:.4f} verbatim_old={verbatim_old:.4f} "
        f"hub_cos={hub_cos:.4f} verbatim_cos={verbatim_cos:.4f} "
        f"hub_deg={hub_deg} verbatim_deg={verbatim_deg}"
    )

def test_pipeline_reason_contains_deg_norm_not_raw_log(tmp_path):
    from iai_mcp.pipeline import recall_for_response

    (store, embedder, graph, assignment, rich_club,
     _hub_id, _verbatim_id, cue_text) = _seed_hub_vs_verbatim(tmp_path)

    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=embedder,
        cue=cue_text,
        session_id="r2_reason_check",
        budget_tokens=1500,
    )
    assert resp.hits, "fixture must produce at least one hit"
    for h in resp.hits:
        assert "deg_norm" in h.reason, (
            f"reason must contain 'deg_norm'; got: {h.reason!r}"
        )
        assert "log(deg+1)" not in h.reason, (
            f"reason must NOT contain raw 'log(deg+1)'; got: {h.reason!r}"
        )

def test_zero_max_degree_does_not_raise_division_error(tmp_path):
    from iai_mcp.pipeline import recall_for_response
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "lancedb")
    embedder = _ControlledEmbedder()
    cue_text = "cold start cue with no graph topology"
    cue_vec = embedder.embed(cue_text)
    embedder.set_fixed(cue_text, cue_vec)

    for i in range(3):
        v = _unit_vector_with_cosine(cue_vec, 0.5 - 0.1 * i)
        store.insert(_make_episodic(v, f"isolated-cold-{i}"))

    graph, assignment, rich_club = build_runtime_graph(store)
    assert graph._max_degree == 0, (
        f"isolated graph must have max_degree=0, got {graph._max_degree}"
    )

    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=embedder,
        cue=cue_text,
        session_id="cold_start_zero_max_deg",
        budget_tokens=1500,
    )
    assert len(resp.hits) >= 1
