from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

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

def _make_record(rid: UUID, surface: str = "topic") -> MemoryRecord:
    rng = random.Random(rid.int)
    raw = [rng.gauss(0.0, 1.0) for _ in range(EMBED_DIM)]
    mag = math.sqrt(sum(x * x for x in raw))
    embedding = [x / mag for x in raw] if mag > 0 else raw
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid,
        tier="episodic",
        literal_surface=surface,
        aaak_index="",
        embedding=embedding,
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

def _write_encrypted_cache(store: MemoryStore, path: Path, data: dict) -> None:
    from iai_mcp import runtime_graph_cache
    from iai_mcp.crypto import encrypt_field

    plaintext = json.dumps(data)
    ciphertext = encrypt_field(
        plaintext,
        store._key(),
        runtime_graph_cache._CACHE_AAD,
    )
    path.write_text(ciphertext, encoding="ascii")

def test_decrypt_failure_skips_cache_write(store, tmp_path):
    from iai_mcp import retrieve, runtime_graph_cache
    from iai_mcp.store import RECORDS_TABLE, _uuid_literal

    r_a = _make_record(uuid4(), "row A — clean surface")
    r_b = _make_record(uuid4(), "row B — to be tampered")
    store.insert(r_a)
    store.insert(r_b)

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    ct_a = df[df["id"] == str(r_a.id)].iloc[0]["literal_surface"]

    tbl.update(
        where=f"id = '{_uuid_literal(r_b.id)}'",
        values={"literal_surface": ct_a},
    )

    graph, assignment, rich_club = retrieve.build_runtime_graph(store)

    loaded = runtime_graph_cache.try_load(store)
    assert loaded is not None, "cache should have been persisted by build_runtime_graph"
    _, _, payload, _, _ = loaded

    assert str(r_a.id) in payload, "clean record must be cached"
    assert str(r_b.id) not in payload, (
        "poisoned record (decrypt-fail) must NOT be in the cache — "
        "an empty surface there is the V2-03 poisoning bug"
    )

def test_pipeline_falls_back_to_store_on_empty_surface(store, tmp_path):
    from iai_mcp import pipeline, retrieve

    rid = uuid4()
    original = "the literal surface that must round-trip"
    store.insert(_make_record(rid, original))

    graph, _assignment, _rich_club = retrieve.build_runtime_graph(store)

    assert rid in set(graph.iter_nodes()), "node should exist post-build"
    graph.set_node_payload(rid, {"surface": ""})

    out = pipeline._read_record_payload(graph, rid, store)
    assert out is not None, "store.get fallback must produce a record"
    assert out.literal_surface == original, (
        "empty-surface graph node must fall through to store.get; "
        "instead got "
        f"{out.literal_surface!r}"
    )

def test_runtime_graph_cache_drops_poisoned_entries_on_load(
    store, tmp_path, capsys
):
    from iai_mcp import runtime_graph_cache

    rid_real = uuid4()
    store.insert(_make_record(rid_real, "real record present in lancedb"))

    good_id = uuid4()
    bad_id = uuid4()

    data = {
        "cache_version": runtime_graph_cache.CACHE_VERSION,
        "key": list(runtime_graph_cache._cache_key(store)),
        "assignment": {
            "node_to_community": {},
            "community_centroids": {},
            "modularity": 0.0,
            "backend": "leiden-test",
            "top_communities": [],
            "mid_regions": {},
        },
        "rich_club": [],
        "node_payload": {
            str(good_id): {
                "embedding": [0.1] * EMBED_DIM,
                "surface": "good record",
                "centrality": 0.0,
                "tier": "episodic",
                "pinned": False,
                "tags": [],
                "language": "en",
            },
            str(bad_id): {
                "embedding": [0.2] * EMBED_DIM,
                "surface": "",
                "centrality": 0.0,
                "tier": "episodic",
                "pinned": False,
                "tags": [],
                "language": "en",
            },
        },
        "max_degree": 1,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    cache_path = tmp_path / "runtime_graph_cache.json"
    _write_encrypted_cache(store, cache_path, data)

    loaded = runtime_graph_cache.try_load(store)
    assert loaded is not None, (
        "outer decode must succeed; if this fails the fixture is wrong, "
        "not the production code"
    )
    _assignment, _rich_club, payload, _max_degree, _node_degrees = loaded

    assert payload is not None
    assert str(good_id) in payload, "well-formed entry must survive rehydrate"
    assert str(bad_id) not in payload, (
        "poisoned (surface='') entry must be dropped — that is V2-03 fix"
    )

    captured = capsys.readouterr()
    assert "runtime_graph_cache_drop_poisoned_entry" in captured.err, (
        "drop must emit a structured stderr event for OPS-12 observability; "
        f"saw stderr={captured.err!r}"
    )
