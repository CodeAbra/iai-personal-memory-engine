from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

from iai_mcp.graph import MemoryGraph
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _make(vec: list[float], text: str = "rec") -> MemoryRecord:
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
        tags=[],
        language="en",
    )


def test_collect_graph_pool_returns_aligned_ids_and_embeddings(tmp_path) -> None:
    from iai_mcp.pipeline import _collect_graph_pool

    store = MemoryStore(path=tmp_path / "hippo")
    records: list[MemoryRecord] = []
    for i in range(5):
        vec = [0.0] * EMBED_DIM
        vec[i] = 1.0
        rec = _make(vec, text=f"rec{i}")
        store.insert(rec)
        records.append(rec)
    graph = MemoryGraph()
    for rec in records:
        graph.add_node(rec.id, community_id=None, embedding=list(rec.embedding))
        graph.set_node_payload(rec.id, {"embedding": list(rec.embedding)})

    pool_ids, pool_embs = _collect_graph_pool(graph, None, store)

    assert len(pool_ids) == 5
    assert pool_embs.shape == (5, EMBED_DIM)
    assert pool_embs.dtype == np.float32
    id_to_rec = {r.id: r for r in records}
    for i, rid in enumerate(pool_ids):
        rec = id_to_rec[rid]
        np.testing.assert_allclose(
            pool_embs[i], np.asarray(rec.embedding, dtype=np.float32)
        )


def test_collect_graph_pool_empty_graph(tmp_path) -> None:
    from iai_mcp.pipeline import _collect_graph_pool

    store = MemoryStore(path=tmp_path / "hippo")
    graph = MemoryGraph()
    pool_ids, pool_embs = _collect_graph_pool(graph, None, store)
    assert pool_ids == []
    assert pool_embs.shape == (0, store.embed_dim)
    assert pool_embs.dtype == np.float32


def test_collect_graph_pool_falls_back_to_store_get(tmp_path) -> None:
    from iai_mcp.pipeline import _collect_graph_pool

    store = MemoryStore(path=tmp_path / "hippo")
    vec = [1.0] + [0.0] * (EMBED_DIM - 1)
    rec = _make(vec, text="store-only")
    store.insert(rec)
    graph = MemoryGraph()
    graph.add_node(rec.id, community_id=None, embedding=list(vec))
    sidecar = graph._node_payload.get(str(rec.id))
    if sidecar and "embedding" in sidecar:
        del sidecar["embedding"]

    pool_ids, pool_embs = _collect_graph_pool(graph, None, store)

    assert pool_ids == [rec.id]
    assert pool_embs.shape == (1, EMBED_DIM)
    np.testing.assert_allclose(
        pool_embs[0], np.asarray(vec, dtype=np.float32)
    )


def test_K_CANDIDATES_is_200() -> None:
    from iai_mcp.pipeline import K_CANDIDATES

    assert K_CANDIDATES == 200
    assert isinstance(K_CANDIDATES, int)


def test_COMMUNITY_BIAS_constants_are_mode_dependent() -> None:
    from iai_mcp.pipeline import COMMUNITY_BIAS_CONCEPT, COMMUNITY_BIAS_VERBATIM

    assert COMMUNITY_BIAS_VERBATIM == 0.0
    assert COMMUNITY_BIAS_CONCEPT == 0.1
    assert isinstance(COMMUNITY_BIAS_VERBATIM, float)
    assert isinstance(COMMUNITY_BIAS_CONCEPT, float)


def test_gate_bias_for_mode_returns_correct_value() -> None:
    from iai_mcp.pipeline import _gate_bias_for_mode

    assert _gate_bias_for_mode("verbatim") == 0.0
    assert _gate_bias_for_mode("concept") == 0.1
    assert _gate_bias_for_mode("unknown") == 0.0
    assert _gate_bias_for_mode("") == 0.0
    assert _gate_bias_for_mode("CONCEPT") == 0.0


def test_RecallCoreResult_dataclass_has_required_fields() -> None:
    from iai_mcp.pipeline import _RecallCoreResult

    r = _RecallCoreResult()
    assert r.scored_hits == []
    assert r.activation_trace == []
    assert r.anti_hits == []
    assert r.hints == []
    assert r.patterns_observed == []
    assert r.cue_mode == "concept"
    assert r.budget_used == 0
