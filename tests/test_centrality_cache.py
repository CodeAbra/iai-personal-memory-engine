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


def _make_record(store: MemoryStore, text: str, seed: int) -> MemoryRecord:
    import numpy as np
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(store.embed_dim).astype(np.float32)
    v /= float(np.linalg.norm(v)) or 1.0
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
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
def seeded_store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore(path=tmp_path / "lancedb")
    store.root = tmp_path
    for i in range(15):
        store.insert(_make_record(store, f"fact-{i}", i + 1))
    records = list(store.all_records())
    ids = [r.id for r in records]
    pairs = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
    pairs += [(ids[0], ids[5]), (ids[2], ids[10])]
    store.boost_edges(pairs, delta=0.5)
    return store


def test_C1_every_node_has_centrality_attr(seeded_store):
    graph, _a, _rc = retrieve.build_runtime_graph(seeded_store)
    listed = list(graph.iter_nodes())
    assert len(listed) > 0
    for nid in listed:
        payload = graph.get_payload(nid)
        assert "centrality" in payload, f"node {nid} missing centrality sidecar"
        assert isinstance(payload["centrality"], float), (
            f"centrality on {nid} must be float, "
            f"got {type(payload['centrality'])}"
        )


def test_C2_cache_round_trips_centrality(seeded_store):
    graph, assignment, rich_club = retrieve.build_runtime_graph(seeded_store)

    live_cent = {
        str(nid): graph.get_centrality(nid) for nid in graph.iter_nodes()
    }

    runtime_graph_cache.invalidate(seeded_store)
    graph2, _a2, _rc2 = retrieve.build_runtime_graph(seeded_store)

    cached = runtime_graph_cache.try_load(seeded_store)
    assert cached is not None, "cache should be populated after build"
    _assignment, _rich_club, node_payload, _max_degree, _node_degrees = cached
    assert node_payload is not None and len(node_payload) > 0

    for nid, live in live_cent.items():
        payload = node_payload.get(nid)
        assert payload is not None, f"missing payload for {nid}"
        assert "centrality" in payload, f"payload {nid} missing centrality"
        assert abs(payload["centrality"] - live) < 1e-9, (
            f"centrality drift on {nid}: cache={payload['centrality']} "
            f"live={live}"
        )


def test_C3_missing_centrality_fallback_inline(seeded_store):
    from iai_mcp import pipeline

    class _E:
        DIM = seeded_store.embed_dim
        DEFAULT_DIM = seeded_store.embed_dim
        DEFAULT_MODEL_KEY = "t"

        def embed(self, t):
            import numpy as np
            import hashlib
            rng = np.random.default_rng(
                int(hashlib.sha256(t.encode()).hexdigest()[:16], 16)
            )
            v = rng.standard_normal(self.DIM).astype(np.float32)
            v /= float(np.linalg.norm(v)) or 1.0
            return v.tolist()

    graph, assignment, rich_club = retrieve.build_runtime_graph(seeded_store)
    for nid in list(graph.iter_nodes()):
        sidecar = graph._node_payload.get(str(nid))
        if sidecar and "centrality" in sidecar:
            del sidecar["centrality"]

    resp = pipeline.recall_for_response(
        store=seeded_store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=_E(), cue="fact-3",
        session_id="t-C3", budget_tokens=4000,
    )
    assert resp is not None
    assert isinstance(resp.hits, list)


def test_C4_cache_version_bumped_to_05_13_v1():
    assert runtime_graph_cache.CACHE_VERSION == "62-02-v8"


def test_C4_legacy_cache_invalidated(seeded_store, tmp_path: Path):
    from iai_mcp.crypto import decrypt_field, encrypt_field

    graph, assignment, rich_club = retrieve.build_runtime_graph(seeded_store)
    cache_path = tmp_path / "runtime_graph_cache.json"
    assert cache_path.exists(), "cache not created by build_runtime_graph"

    key = runtime_graph_cache._cache_encryption_key(seeded_store)
    raw_text = cache_path.read_text(encoding="utf-8")
    plaintext = decrypt_field(raw_text, key, runtime_graph_cache._CACHE_AAD)
    raw = json.loads(plaintext)
    raw["cache_version"] = "05-12-v1"
    new_ct = encrypt_field(json.dumps(raw), key, runtime_graph_cache._CACHE_AAD)
    cache_path.write_text(new_ct, encoding="ascii")

    assert runtime_graph_cache.try_load(seeded_store) is None
