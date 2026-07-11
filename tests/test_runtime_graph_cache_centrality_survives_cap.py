"""The expensive community + centrality results must survive the cache size cap.

The runtime-graph cache stores two kinds of data: a large `node_payload` (a
384-dim embedding + decrypted surface per node) and the small, expensive-to-
recompute results (the community assignment and the betweenness centrality map).
When the serialized cache exceeds its size cap the large payload is shed first —
it is cheap to rebuild by streaming the store. The expensive centrality must NOT
be shed with it: recomputing betweenness (O(V*E) Brandes on the whole corpus) is
the multi-minute cost the cache exists to avoid.

This module pins three contracts:

  1. A scale cache (one whose node_payload was shed) reused within drift spawns
     NO betweenness child — the regression guard for the "recompute on every
     warm" bug.

  2. The compact centrality map + payload_record_count round-trip through
     save()/try_load_cache_results() intact even when node_payload was dropped.

  3. The graph produced via the cached-results path carries the same per-node
     centrality (within 1e-6) as a full rebuild, and records added after the
     cache was built stay findable via the index recall path.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp import retrieve, runtime_graph_cache
from iai_mcp.community import CommunityAssignment
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import MemoryRecord


@pytest.fixture(autouse=True)
def _crypto_passphrase(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")
    yield


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
    s = MemoryStore(path=tmp_path / "store")
    s.root = tmp_path
    return s


def _make_rec(seed: int, store: MemoryStore) -> MemoryRecord:
    rng = np.random.default_rng(seed)
    vec = rng.random(store.embed_dim).astype(np.float32)
    vec = (vec / np.linalg.norm(vec)).tolist()
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=f"surface number {seed} carrying real text",
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
        tags=[f"tag{seed % 3}"],
        language="en",
    )


def _seed_connected(store: MemoryStore, n: int, seed_base: int = 0) -> list:
    """Seed n records chained into a path graph so interior nodes carry non-zero
    betweenness centrality (the centrality-bearing cache the warm path reuses)."""
    recs = [_make_rec(seed_base + i, store) for i in range(n)]
    for rec in recs:
        store.insert(rec)
    flush_record_buffer(store)
    ids = [rec.id for rec in recs]
    pairs = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
    store.boost_edges(pairs, delta=1.0, edge_type="hebbian")
    return recs


# ---------------------------------------------------------------------------
# Contract 1: a scale cache (node_payload shed) skips the betweenness child.
# ---------------------------------------------------------------------------
def test_scale_cache_hit_skips_betweenness(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
):
    """THE regression guard. A cache whose node_payload was shed at the size cap,
    reused within drift tolerance, spawns ZERO community-detection / centrality
    children — the bug was a full betweenness recompute on every warm because the
    centrality lived only inside the shed node_payload.

    The cap is shrunk so the node_payload (16 × 384-dim embeddings) is shed while
    the compact centrality map survives, reproducing the production-scale state at
    test speed.
    """
    monkeypatch.setenv("IAI_MCP_RGC_DRIFT_ABS", "3")
    monkeypatch.setenv("IAI_MCP_RGC_DRIFT_FRAC", "0.0")
    # Force the node_payload shed: 16 records × 384-dim embeddings far exceed
    # this cap, but the compact centrality map (16 floats) fits comfortably.
    monkeypatch.setattr(runtime_graph_cache, "MAX_CACHE_BYTES", 60_000)

    _seed_connected(store, n=16, seed_base=700)
    runtime_graph_cache.invalidate(store)

    detect_calls = {"n": 0}
    centrality_calls = {"n": 0}

    real_detect = retrieve._detect_communities_isolated

    def _counting_detect(store_arg, graph, **kw):
        detect_calls["n"] += 1
        return real_detect(store_arg, graph, **kw)

    real_centrality_child = runtime_graph_cache.compute_centrality_in_child

    def _counting_centrality_child(graph, **kw):
        centrality_calls["n"] += 1
        return real_centrality_child(graph, **kw)

    monkeypatch.setattr(retrieve, "_detect_communities_isolated", _counting_detect)
    monkeypatch.setattr(
        runtime_graph_cache,
        "compute_centrality_in_child",
        _counting_centrality_child,
    )

    # First build: cold cache → one detection rebuild (centrality folded in).
    graph0, assign0, _rc0 = retrieve.build_runtime_graph(store)
    assert graph0.node_count() == 16
    assert assign0 is not None
    assert detect_calls["n"] == 1
    assert centrality_calls["n"] == 0

    # The node_payload really was shed at the cap — confirm the bug precondition.
    loaded = runtime_graph_cache.try_load(store)
    assert loaded is not None
    _a, _r, shed_node_payload, _m, _nd = loaded
    assert not shed_node_payload, (
        "precondition: node_payload must be shed at the cap so this test "
        "exercises the scale path"
    )
    # ...yet the compact centrality survived.
    results = runtime_graph_cache.try_load_cache_results(store)
    assert results is not None
    cached_centrality, payload_count = results
    assert payload_count == 16
    assert len(cached_centrality) == 16
    assert any(v != 0.0 for v in cached_centrality.values()), (
        "a path graph must produce non-zero interior betweenness"
    )

    # The probe reports warm despite the shed payload — the fix.
    assert not retrieve._runtime_graph_rebuild_needed(store), (
        "a shed-payload cache with surviving centrality must report warm"
    )

    # Add a below-tolerance number of records (drift 3 ≤ floor 3).
    for rec in (_make_rec(800 + i, store) for i in range(3)):
        store.insert(rec)
    flush_record_buffer(store)

    detect_calls["n"] = 0
    centrality_calls["n"] = 0

    assert not retrieve._runtime_graph_rebuild_needed(store)

    graph1, assign1, _rc1 = retrieve.build_runtime_graph(store)

    # THE assertion: no betweenness fired on the scale cache-hit.
    assert detect_calls["n"] == 0, (
        "scale cache-hit must not re-run community detection"
    )
    assert centrality_calls["n"] == 0, (
        "scale cache-hit must not re-run betweenness centrality "
        "(this is the warm-RSS regression)"
    )
    # With a shed payload the warm build re-streams the live store, so the graph
    # reflects the current corpus (16 original + 3 new = 19). The 3 new records
    # are not in the cached centrality map and keep their default 0.0 — bounded
    # staleness of the centrality result, not of node membership.
    assert graph1.node_count() == 19
    assert assign1 is not None
    # The cached centrality was applied to the original (path-graph) nodes.
    nonzero = [
        v for v in (
            graph1._node_payload[str(n)].get("centrality", 0.0)
            for n in graph1.iter_nodes()
        )
        if v != 0.0
    ]
    assert nonzero, "the cached centrality must be applied to the warm graph"


# ---------------------------------------------------------------------------
# Contract 2: compact results round-trip through the size cap.
# ---------------------------------------------------------------------------
def _make_assignment(n_communities: int = 2) -> CommunityAssignment:
    comms = [uuid4() for _ in range(n_communities)]
    nodes = [uuid4() for _ in range(n_communities * 3)]
    return CommunityAssignment(
        node_to_community={nodes[i]: comms[i // 3] for i in range(len(nodes))},
        community_centroids={c: [0.1, 0.2, 0.3] for c in comms},
        modularity=0.42,
        backend="leiden",
        top_communities=comms,
        mid_regions={c: nodes[i * 3:(i + 1) * 3] for i, c in enumerate(comms)},
    )


def _make_node_payload(node_ids, embed_dim: int, base_centrality: float = 0.1):
    return {
        str(nid): {
            "embedding": [0.123456789] * embed_dim,
            "surface": "hello world surface text",
            "centrality": base_centrality + 0.001 * i,
            "tier": "episodic",
            "pinned": False,
            "tags": ["t1", "t2"],
            "language": "en",
        }
        for i, nid in enumerate(node_ids)
    }


def test_centrality_survives_cap_round_trip(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
):
    """save() with an over-cap node_payload sheds the payload but keeps the
    compact centrality map + payload_record_count, and try_load_cache_results()
    returns them intact and correct."""
    monkeypatch.setattr(runtime_graph_cache, "MAX_CACHE_BYTES", 50_000)

    assignment = _make_assignment(n_communities=2)
    node_ids = list(assignment.node_to_community.keys())
    # 30 nodes × 384-dim embeddings far exceed 50 KB — node_payload is shed.
    extra_ids = [uuid4() for _ in range(30)]
    payload = _make_node_payload(extra_ids, store.embed_dim, base_centrality=0.2)
    expected_centrality = {
        nid: float(p["centrality"]) for nid, p in payload.items()
    }

    ok = runtime_graph_cache.save(
        store, assignment, [uuid4()], node_payload=payload, max_degree=4
    )
    assert ok is True

    # node_payload was shed.
    loaded = runtime_graph_cache.try_load(store)
    assert loaded is not None
    _a, _r, shed_payload, _m, _nd = loaded
    assert not shed_payload, "node_payload must be shed over the cap"

    # ...but the compact results survived and decode correctly.
    results = runtime_graph_cache.try_load_cache_results(store)
    assert results is not None
    cached_centrality, payload_count = results
    assert payload_count == len(payload)
    assert len(cached_centrality) == len(expected_centrality)
    for nid_str, expected in expected_centrality.items():
        from uuid import UUID

        nid = UUID(nid_str)
        assert nid in cached_centrality
        assert cached_centrality[nid] == pytest.approx(expected, abs=1e-9)


def test_cache_results_absent_when_no_payload(store: MemoryStore):
    """A cache saved without a node_payload exposes an empty centrality map, so
    the warm probe treats it as needing a rebuild (no expensive result to reuse)."""
    assignment = _make_assignment(n_communities=2)
    ok = runtime_graph_cache.save(store, assignment, [uuid4()])
    assert ok is True
    results = runtime_graph_cache.try_load_cache_results(store)
    assert results is not None
    cached_centrality, payload_count = results
    assert cached_centrality == {}
    assert payload_count == 0


def test_cache_results_none_on_legacy_cache_without_field(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
):
    """A cache that predates the compact-results format (no `centrality` key)
    returns None from try_load_cache_results so exactly one rebuild rewrites the
    new format — backward compatibility with an on-disk cache."""
    import json

    from iai_mcp.crypto import encrypt_field

    # Build a legacy-shaped data dict: same key/version as the live store but
    # WITHOUT the compact centrality / payload_record_count fields.
    legacy = {
        "cache_version": runtime_graph_cache.CACHE_VERSION,
        "key": list(runtime_graph_cache._cache_key(store)),
        "assignment": runtime_graph_cache._encode_assignment(_make_assignment()),
        "rich_club": [],
        "node_payload": {},
        "max_degree": 0,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "generation": int(runtime_graph_cache.get_current_generation()),
        "rebuild_timestamp": "",
    }
    path = runtime_graph_cache._cache_path(store)
    ciphertext = encrypt_field(
        json.dumps(legacy),
        runtime_graph_cache._cache_encryption_key(store),
        runtime_graph_cache._CACHE_AAD,
    )
    path.write_text(ciphertext, encoding="ascii")

    assert runtime_graph_cache.try_load_cache_results(store) is None, (
        "a legacy cache without the compact fields must read as absent"
    )
    # The 4-tuple try_load still works on the legacy cache (no regression).
    assert runtime_graph_cache.try_load(store) is not None


# ---------------------------------------------------------------------------
# Contract 3: centrality parity between the cached-results path and a full
# rebuild, and new records remain index-findable.
# ---------------------------------------------------------------------------
def test_cached_results_path_matches_full_rebuild_centrality(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
):
    """The graph produced via the cached-results (shed-payload) path carries the
    same per-node centrality as a full rebuild, for the nodes present in both."""
    monkeypatch.setenv("IAI_MCP_RGC_DRIFT_ABS", "5")
    monkeypatch.setenv("IAI_MCP_RGC_DRIFT_FRAC", "0.0")

    _seed_connected(store, n=18, seed_base=1100)

    # Full rebuild WITHOUT the cap squeeze: node_payload retained, centrality
    # computed fresh. Capture the per-node centrality as ground truth.
    runtime_graph_cache.invalidate(store)
    graph_full, _assign_full, _rc_full = retrieve.build_runtime_graph(store)
    full_centrality = {
        str(n): float(graph_full._node_payload[str(n)].get("centrality", 0.0))
        for n in graph_full.iter_nodes()
    }
    assert any(v != 0.0 for v in full_centrality.values())

    # Now force the shed-payload state and rebuild from scratch under the cap, so
    # the saved cache carries only the compact centrality.
    monkeypatch.setattr(runtime_graph_cache, "MAX_CACHE_BYTES", 70_000)
    runtime_graph_cache.invalidate(store)
    retrieve.build_runtime_graph(store)
    shed = runtime_graph_cache.try_load(store)
    assert shed is not None and not shed[2], "node_payload must be shed"

    # The warm read takes the cached-results path: stream node_payload + apply
    # cached centrality, no betweenness child.
    graph_warm, _assign_warm, _rc_warm = retrieve.build_runtime_graph(store)
    warm_centrality = {
        str(n): float(graph_warm._node_payload[str(n)].get("centrality", 0.0))
        for n in graph_warm.iter_nodes()
    }

    shared = set(full_centrality) & set(warm_centrality)
    assert shared, "the two builds must share their node set"
    for nid in shared:
        assert warm_centrality[nid] == pytest.approx(
            full_centrality[nid], abs=1e-6
        ), f"centrality parity broken for node {nid}"


def test_new_record_during_shed_window_still_index_findable(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
):
    """A record inserted while the cache results are stale (within drift, with a
    shed payload) stays findable via the index recall path, and the warm rebuild
    reuses the cached centrality without firing a betweenness child.

    With a shed payload the warm path re-streams the live store, so the
    drift-delta record is naturally present in the streamed graph (a strict
    improvement over the frozen-payload case) — but the expensive centrality is
    still taken from the cache, never recomputed."""
    monkeypatch.setenv("IAI_MCP_RGC_DRIFT_ABS", "5")
    monkeypatch.setenv("IAI_MCP_RGC_DRIFT_FRAC", "0.0")
    monkeypatch.setattr(runtime_graph_cache, "MAX_CACHE_BYTES", 60_000)

    _seed_connected(store, n=14, seed_base=1300)
    runtime_graph_cache.invalidate(store)

    centrality_calls = {"n": 0}
    real_centrality_child = runtime_graph_cache.compute_centrality_in_child

    def _counting_centrality_child(graph, **kw):
        centrality_calls["n"] += 1
        return real_centrality_child(graph, **kw)

    retrieve.build_runtime_graph(store)
    # Confirm the shed-payload warm state.
    shed = runtime_graph_cache.try_load(store)
    assert shed is not None and not shed[2]
    assert not retrieve._runtime_graph_rebuild_needed(store)

    new_rec = _make_rec(99999, store)
    store.insert(new_rec)
    flush_record_buffer(store)

    assert not retrieve._runtime_graph_rebuild_needed(store), (
        "a single new record must not force a rebuild under tolerance"
    )

    monkeypatch.setattr(
        runtime_graph_cache,
        "compute_centrality_in_child",
        _counting_centrality_child,
    )
    graph, _assign, _rc = retrieve.build_runtime_graph(store)
    assert centrality_calls["n"] == 0, (
        "the within-drift warm rebuild must not recompute betweenness"
    )

    hits = store.query_similar(list(new_rec.embedding), k=5)
    hit_ids = {str(rec.id) for rec, _score in hits}
    assert str(new_rec.id) in hit_ids, (
        "a record added during the shed-payload stale window must remain "
        "findable via the index recall path"
    )
