from __future__ import annotations

from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.graph import MemoryGraph


def _build_payload(seed: float) -> dict:
    return {
        "embedding": [seed] + [0.0] * 383,
        "surface": f"untangle-surface-{seed}",
        "centrality": 0.125 * seed,
        "tier": "episodic",
        "pinned": False,
        "tags": ["untangle"],
        "language": "en",
    }


@pytest.fixture
def graph_with_records() -> tuple[MemoryGraph, list[UUID]]:
    graph = MemoryGraph()
    ids: list[UUID] = []
    for i in range(5):
        nid = uuid4()
        ids.append(nid)
        payload = _build_payload(seed=float(i + 1))
        graph.add_node(
            nid,
            community_id=None,
            embedding=list(payload["embedding"]),
        )
        graph.set_node_payload(nid, payload)
    graph.add_edge(ids[0], ids[1], weight=0.75)
    graph.add_edge(ids[1], ids[2], weight=0.50)
    return graph, ids


def test_attrs_no_longer_carries_record_payload(graph_with_records) -> None:
    graph, ids = graph_with_records
    for nid in ids:
        attrs_keys = set(graph._attrs[nid].keys())
        assert "embedding" not in attrs_keys, (
            f"_attrs[{nid}] must NOT carry 'embedding' (sidecar lives in "
            f"_node_payload); got keys={attrs_keys}"
        )
        assert attrs_keys <= {"community_id"}, (
            f"_attrs[{nid}] keys must be a subset of {{community_id}}, "
            f"got {attrs_keys}"
        )


def test_sidecar_carries_embedding(graph_with_records) -> None:
    graph, ids = graph_with_records
    for i, nid in enumerate(ids):
        sidecar = graph._node_payload[str(nid)]
        assert "embedding" in sidecar
        expected_first = float(i + 1)
        assert sidecar["embedding"][0] == pytest.approx(expected_first)
        assert sidecar["surface"] == f"untangle-surface-{expected_first}"
        assert sidecar["tier"] == "episodic"
        assert sidecar["language"] == "en"
        assert sidecar["pinned"] is False
        assert sidecar["tags"] == ["untangle"]


def test_get_embedding_reads_from_sidecar() -> None:
    graph = MemoryGraph()
    nid = uuid4()
    real_emb = [0.7] + [0.0] * 383
    graph.add_node(nid, community_id=None, embedding=real_emb)
    assert np.array_equal(
        graph.get_embedding(nid), np.asarray(real_emb, dtype=np.float32)
    )

    graph._attrs[nid]["embedding"] = [0.0] * 384
    assert np.array_equal(
        graph.get_embedding(nid), np.asarray(real_emb, dtype=np.float32)
    ), "get_embedding must read sidecar; _attrs writes have no effect"

    new_emb = [0.1] * 384
    graph.set_node_payload(nid, {"embedding": new_emb})
    assert np.array_equal(
        graph.get_embedding(nid), np.asarray(new_emb, dtype=np.float32)
    )

    assert graph.get_embedding(uuid4()) is None


def test_new_public_api_signatures(graph_with_records) -> None:
    graph, ids = graph_with_records

    assert callable(graph.iter_nodes)
    assert callable(graph.iter_edges_with_weight)
    assert callable(graph.to_csr_arrays)
    assert callable(graph.degrees)
    assert callable(graph.set_node_payload)
    assert callable(graph.get_centrality)
    assert callable(graph.get_payload)

    listed_nodes = list(graph.iter_nodes())
    assert len(listed_nodes) == 5
    assert all(isinstance(u, UUID) for u in listed_nodes)
    assert set(listed_nodes) == set(ids)

    edges = list(graph.iter_edges_with_weight())
    assert len(edges) == 2
    for u, v, w in edges:
        assert isinstance(u, UUID)
        assert isinstance(v, UUID)
        assert isinstance(w, float)

    indptr, indices, data = graph.to_csr_arrays()
    assert isinstance(indptr, np.ndarray)
    assert isinstance(indices, np.ndarray)
    assert isinstance(data, np.ndarray)
    assert indptr.dtype == np.int64
    assert indices.dtype == np.int64
    assert data.dtype == np.float64
    assert len(indptr) == 6

    deg_pairs = list(graph.degrees())
    assert len(deg_pairs) == 5
    for nid, deg in deg_pairs:
        assert isinstance(nid, UUID)
        assert isinstance(deg, int)


def test_set_node_payload_idempotent() -> None:
    graph = MemoryGraph()
    nid = uuid4()
    graph.add_node(nid, community_id=None, embedding=[0.0] * 384)

    payload = {"embedding": [1.0] * 384, "surface": "idempotent"}
    graph.set_node_payload(nid, payload)
    graph.set_node_payload(nid, payload)

    assert str(nid) in graph._node_payload
    assert graph._node_payload[str(nid)]["surface"] == "idempotent"
    assert isinstance(graph._node_payload[str(nid)], dict)


def test_get_centrality_reads_from_sidecar() -> None:
    graph = MemoryGraph()
    nid = uuid4()
    graph.add_node(nid, community_id=None, embedding=[0.0] * 384)
    assert graph.get_centrality(nid) == 0.0
    graph.set_node_payload(nid, {"centrality": 0.875})
    assert graph.get_centrality(nid) == pytest.approx(0.875)
    assert graph.get_centrality(str(nid)) == pytest.approx(0.875)
    assert graph.get_centrality(uuid4()) == 0.0


def test_get_payload_reads_from_sidecar(graph_with_records) -> None:
    graph, ids = graph_with_records
    payload = graph.get_payload(ids[0])
    assert isinstance(payload, dict)
    assert payload["surface"].startswith("untangle-surface-")
    assert "embedding" in payload
    assert graph.get_payload(uuid4()) == {}
