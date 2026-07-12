from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp import retrieve, runtime_graph_cache
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


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(path=tmp_path / "lancedb")
    s.root = tmp_path
    return s


def _make_record(
    store: MemoryStore,
    text: str = "hello",
    vec_seed: float = 0.1,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[vec_seed] * store.embed_dim,
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


def test_B1_insert_updates_graph_node(store):
    seed = _make_record(store, "seed", 0.5)
    store.insert(seed)

    graph, _a, _rc = retrieve.build_runtime_graph(store)
    assert seed.id in set(graph.iter_nodes())
    new_rec = _make_record(store, "freshly-inserted", 0.3)
    store.insert(new_rec)

    assert new_rec.id in set(graph.iter_nodes())
    payload = graph.get_payload(new_rec.id)
    assert payload.get("surface") == "freshly-inserted"
    assert "embedding" in payload


def test_B2_update_mutates_node_payload(store):
    rec = _make_record(store, "before-update", 0.2)
    store.insert(rec)
    graph, _a, _rc = retrieve.build_runtime_graph(store)

    payload_before = graph.get_payload(rec.id)
    assert payload_before["surface"] == "before-update"

    rec.literal_surface = "after-update"
    rec.embedding = [0.9] * store.embed_dim
    store.update(rec)

    payload_after = graph.get_payload(rec.id)
    assert payload_after["surface"] == "after-update"
    assert list(payload_after["embedding"])[0] == pytest.approx(0.9)


def test_B3_delete_removes_node(store):
    rec = _make_record(store, "to-be-deleted", 0.4)
    store.insert(rec)
    graph, _a, _rc = retrieve.build_runtime_graph(store)
    assert rec.id in set(graph.iter_nodes())

    store.delete(rec.id)
    assert rec.id not in set(graph.iter_nodes())


def test_B4_hook_exception_does_not_break_store_insert(store, capsys):
    def _bad_hook(op, record):
        raise RuntimeError("hook is sad")

    store.register_graph_sync_hook(_bad_hook)

    rec = _make_record(store, "store-write-is-authoritative", 0.15)
    store.insert(rec)

    roundtrip = store.get(rec.id)
    assert roundtrip is not None
    assert roundtrip.literal_surface == "store-write-is-authoritative"

    captured = capsys.readouterr()
    assert "graph_sync_failed" in captured.err
    found = False
    for line in captured.err.splitlines():
        try:
            payload = json.loads(line)
            if payload.get("event") == "graph_sync_failed":
                assert payload.get("op") == "insert"
                found = True
                break
        except (ValueError, TypeError):
            continue
    assert found, "expected a JSON graph_sync_failed event on stderr"


def test_B5_cold_start_restores_node_payload_from_cache(store):
    rec = _make_record(store, "cached-payload", 0.25)
    store.insert(rec)

    graph1, _a, _rc = retrieve.build_runtime_graph(store)
    payload1 = graph1.get_payload(rec.id)
    expected_surface = payload1["surface"]
    expected_emb = list(payload1["embedding"])

    loaded = runtime_graph_cache.try_load(store)
    assert loaded is not None, "cache must be loadable"
    _assignment, _rich_club, node_payload, _max_degree, _node_degrees = loaded
    assert node_payload is not None, "cache is missing node_payload blob"
    assert str(rec.id) in node_payload

    graph2, _a, _rc = retrieve.build_runtime_graph(store)
    payload2 = graph2.get_payload(rec.id)
    assert payload2["surface"] == expected_surface
    assert list(payload2["embedding"]) == expected_emb


def test_B6_cache_version_bump_invalidates_old_cache(store):
    cache_path = runtime_graph_cache._cache_path(store)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(
            {
                "cache_version": "05-09-v1",
                "key": [0, 0, 4, store.embed_dim, "05-09-v1"],
                "assignment": {
                    "node_to_community": {},
                    "community_centroids": {},
                    "modularity": 0.0,
                    "backend": "flat",
                    "top_communities": [],
                    "mid_regions": {},
                },
                "rich_club": [],
                "saved_at": "2026-01-01T00:00:00+00:00",
            },
            f,
        )

    assert runtime_graph_cache.CACHE_VERSION == "62-02-v7"

    assert runtime_graph_cache.try_load(store) is None
