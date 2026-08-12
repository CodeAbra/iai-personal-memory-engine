from __future__ import annotations

import pathlib
from uuid import UUID, uuid4

import numpy as np

from iai_mcp.graph import MemoryGraph


def _make_payload(seed: float) -> dict:
    return {
        "embedding": [seed] + [0.0] * 383,
        "surface": f"probe-surface-{seed}",
        "centrality": 0.0,
        "tier": "episodic",
        "pinned": False,
        "tags": ["probe"],
        "language": "en",
    }


def _payload_embedding(payload: dict) -> list:
    # Mirrors retrieve.py: a graph-owned vector is a buffer with no truth value.
    emb = payload.get("embedding")
    return [] if emb is None else list(emb)


def _drive_cached_path_sidecar_write(
    graph: MemoryGraph, node_id: UUID, payload: dict
) -> None:
    graph.add_node(
        node_id,
        community_id=None,
        embedding=_payload_embedding(payload),
    )
    graph.set_node_payload(
        node_id,
        {
            "embedding": _payload_embedding(payload),
            "surface": payload.get("surface", ""),
            "centrality": float(payload.get("centrality") or 0.0),
            "tier": payload.get("tier", "episodic"),
            "pinned": bool(payload.get("pinned", False)),
            "tags": list(payload.get("tags") or []),
            "language": str(payload.get("language", "en") or "en"),
        },
    )


def _drive_miss_rebuild_path_sidecar_write(
    graph: MemoryGraph, node_id: UUID, payload: dict
) -> None:
    embedding = _payload_embedding(payload)
    graph.add_node(
        node_id,
        community_id=None,
        embedding=embedding,
    )
    graph.set_node_payload(
        node_id,
        {
            "embedding": list(embedding),
            "surface": str(payload.get("surface", "")),
            "centrality": float(payload.get("centrality") or 0.0),
            "tier": str(payload.get("tier", "episodic")),
            "pinned": bool(payload.get("pinned", False)),
            "tags": list(payload.get("tags") or []),
            "language": str(payload.get("language", "en") or "en"),
        },
    )


def _assert_embedding_in_sidecar(
    graph: MemoryGraph, node_id: UUID, source_lines: str
) -> None:
    sidecar_emb = graph._node_payload[str(node_id)]["embedding"]
    api_emb = graph.get_embedding(node_id)
    assert np.array_equal(api_emb, sidecar_emb), (
        f"sidecar-vs-API drift at retrieve.py {source_lines}; "
        f"uuid={node_id}; "
        f"_node_payload[str(uuid)]['embedding']={sidecar_emb!r}; "
        f"get_embedding(uuid)={api_emb!r}"
    )


def test_cached_path_embedding_field_in_sidecar() -> None:
    graph = MemoryGraph()
    node_id = uuid4()
    payload = _make_payload(seed=1.0)

    _drive_cached_path_sidecar_write(graph, node_id, payload)

    _assert_embedding_in_sidecar(graph, node_id, source_lines="lines 770-785")

    assert graph.has_node(node_id)
    assert node_id in graph._attrs
    assert str(node_id) in graph._node_payload


def test_miss_rebuild_path_embedding_field_in_sidecar() -> None:
    graph = MemoryGraph()
    node_id = uuid4()
    payload = _make_payload(seed=2.0)

    _drive_miss_rebuild_path_sidecar_write(graph, node_id, payload)

    _assert_embedding_in_sidecar(graph, node_id, source_lines="lines 865-879")

    assert graph.has_node(node_id)
    assert node_id in graph._attrs
    assert str(node_id) in graph._node_payload


def test_attrs_post_untangle_carries_only_community_id() -> None:
    graph = MemoryGraph()
    node_id = uuid4()
    embedding = [0.5] + [0.0] * 383

    graph.add_node(node_id, community_id=None, embedding=embedding)

    assert node_id in graph._attrs, (
        f"expected uuid={node_id} present in _attrs after add_node; "
        f"keys={list(graph._attrs.keys())}"
    )
    attrs_keys = set(graph._attrs[node_id].keys())
    assert "embedding" not in attrs_keys, (
        f"_attrs[{node_id}] must NOT carry 'embedding' post-untangle; "
        f"actual keys={attrs_keys}"
    )
    assert attrs_keys <= {"community_id"}, (
        f"_attrs[{node_id}] keys must ⊆ {{community_id}} post-untangle; "
        f"actual={attrs_keys}"
    )
    assert str(node_id) in graph._node_payload
    assert np.array_equal(
        graph._node_payload[str(node_id)].get("embedding"),
        np.asarray(embedding, dtype=np.float32),
    )


def test_downstream_hardest_test_files_present() -> None:
    sync_path = pathlib.Path("tests/test_graph_node_payload_sync.py")
    native_recall_path = pathlib.Path("tests/test_graph_native_recall.py")

    assert sync_path.exists(), (
        f"expected hardest downstream test file present: {sync_path} "
        "(refactored by the upcoming atomic untangle wave)"
    )
    assert native_recall_path.exists(), (
        f"expected hardest downstream test file present: {native_recall_path} "
        "(refactored by the upcoming atomic untangle wave)"
    )
