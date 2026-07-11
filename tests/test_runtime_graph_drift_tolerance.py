"""Drift tolerance for the runtime-graph community cache.

The community graph is an enhancement layer over the index recall path. A record
is findable by cosine/ANN the instant it lands in the index, well before it is
folded into the community graph, so the community graph may lag the live corpus
by a bounded number of records without losing any recall. This module pins that
contract:

  1. Below-tolerance drift reuses the cached graph and cached centrality and
     spawns no detection/centrality child — no betweenness recompute fires on a
     small count change. Crossing the tolerance triggers exactly one rebuild.

  2. A record inserted after the cache was built — while the community graph is
     deliberately stale within the tolerance window — is still returned by the
     index recall path, proving the bounded-staleness window loses no records.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp import retrieve, runtime_graph_cache
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


def _join_refresher(store: MemoryStore, timeout: float = 120.0) -> None:
    """Join the single-flight background bundle refresher, if one is in flight,
    so heavy-compute call counts are deterministic when asserted."""
    t = getattr(store, "_graph_refresh_thread", None)
    if t is not None and t.is_alive():
        t.join(timeout=timeout)


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
    """Seed n records and chain them into a path graph.

    A path topology gives interior nodes non-zero betweenness centrality, so the
    first build saves a centrality-bearing cache. A subsequent within-tolerance
    build then takes the warm-cache branch and computes no centrality.
    """
    recs = [_make_rec(seed_base + i, store) for i in range(n)]
    for rec in recs:
        store.insert(rec)
    flush_record_buffer(store)
    ids = [rec.id for rec in recs]
    pairs = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
    store.boost_edges(pairs, delta=1.0, edge_type="hebbian")
    return recs


def test_small_drift_does_not_trigger_betweenness_rebuild(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
):
    """A below-tolerance count change reuses the cached community graph and
    centrality and spawns NO child — neither community detection nor the
    standalone centrality pass. Crossing the tolerance triggers exactly one
    rebuild that folds in the accumulated records.

    The tolerance is shrunk via the operator env knobs so a handful of records
    crosses it, keeping the test fast while exercising the production gate.
    """
    # Tiny tolerance: abs floor 3, no proportional band. Drift of 3 stays warm,
    # drift of 4 forces a rebuild.
    monkeypatch.setenv("IAI_MCP_RGC_DRIFT_ABS", "3")
    monkeypatch.setenv("IAI_MCP_RGC_DRIFT_FRAC", "0.0")
    # The refresher joins below must be deterministic, not cooldown-gated.
    monkeypatch.setenv("IAI_MCP_GRAPH_REFRESH_COOLDOWN_S", "0")

    _seed_connected(store, n=16, seed_base=200)
    runtime_graph_cache.invalidate(store)

    # Count every heavy compute path: community detection AND standalone
    # centrality. Either firing means a betweenness pass ran.
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

    # First build: cold cache → one rebuild that detects communities + computes
    # centrality (folded into the detection child).
    graph0, assign0, _rc0 = retrieve.build_runtime_graph(store)
    assert graph0.node_count() == 16
    assert assign0 is not None
    assert detect_calls["n"] == 1, "first build should run detection exactly once"
    # The detection child carries centrality on the cache-miss path, so the
    # standalone centrality child must not have fired.
    assert centrality_calls["n"] == 0
    assert not retrieve._runtime_graph_rebuild_needed(store), (
        "a freshly-built centrality-bearing cache must need no rebuild"
    )

    # Add a below-tolerance number of records (drift 3 ≤ floor 3).
    for rec in (_make_rec(300 + i, store) for i in range(3)):
        store.insert(rec)
    flush_record_buffer(store)

    detect_calls["n"] = 0
    centrality_calls["n"] = 0

    # The probe still reports warm — drift is within tolerance.
    assert not retrieve._runtime_graph_rebuild_needed(store), (
        "below-tolerance drift must keep the cache warm (no rebuild needed)"
    )

    graph1, assign1, _rc1 = retrieve.build_runtime_graph(store)
    _join_refresher(store)

    # The core regression guard: NO heavy compute fired on the small drift —
    # neither on the serving path nor in the background refresher.
    assert detect_calls["n"] == 0, (
        "below-tolerance drift must not re-run community detection"
    )
    assert centrality_calls["n"] == 0, (
        "below-tolerance drift must not re-run betweenness centrality"
    )
    # Node count is deliberately NOT pinned exactly: the cached node set (16)
    # and a live-sync-hook graph that already folded the fresh inserts (19)
    # are both blessed states — the contract is "no heavy compute, no lost
    # records", not a frozen staleness snapshot.
    assert 16 <= graph1.node_count() <= 19
    assert assign1 is not None

    # Now push total drift ABOVE tolerance (4 > floor 3) and confirm exactly one
    # rebuild fires.
    store.insert(_make_rec(400, store))
    flush_record_buffer(store)

    assert retrieve._runtime_graph_rebuild_needed(store), (
        "over-tolerance drift must mark the cache as needing a rebuild"
    )

    detect_calls["n"] = 0
    centrality_calls["n"] = 0

    graph2, assign2, _rc2 = retrieve.build_runtime_graph(store)
    _join_refresher(store)

    assert detect_calls["n"] == 1, (
        "over-tolerance drift must trigger exactly one community-detection "
        "rebuild (inline or via the single-flight background refresher)"
    )
    assert assign2 is not None

    # After the refresher lands, the next build serves the rebuilt bundle: the
    # accumulated records (3 + 1 = 4) are folded in and the community graph
    # covers the full corpus again — with no additional detection pass.
    graph3, assign3, _rc3 = retrieve.build_runtime_graph(store)
    assert graph3.node_count() == 20
    assert assign3 is not None
    assert detect_calls["n"] == 1


def test_record_added_during_stale_window_still_recall_findable(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
):
    """A record inserted while the community graph is deliberately stale (within
    drift tolerance) is still returned by the index recall path.

    This proves the bounded-staleness window loses no records: recall reads the
    index, not the community graph, so a not-yet-in-graph record is found by its
    own embedding.
    """
    monkeypatch.setenv("IAI_MCP_RGC_DRIFT_ABS", "5")
    monkeypatch.setenv("IAI_MCP_RGC_DRIFT_FRAC", "0.0")

    _seed_connected(store, n=14, seed_base=500)
    runtime_graph_cache.invalidate(store)

    # Build a warm, centrality-bearing cache.
    retrieve.build_runtime_graph(store)
    assert not retrieve._runtime_graph_rebuild_needed(store)

    # Insert a NEW record while the community graph stays stale (drift 1 < 5).
    new_rec = _make_rec(9999, store)
    store.insert(new_rec)
    flush_record_buffer(store)

    # The community graph is still warm (no rebuild forced). Whether the new
    # record already appears as a graph node is NOT pinned: the live sync-hook
    # may have folded it in (fresh) or the cached node set may still be served
    # (bounded staleness) — both are blessed, because recall correctness never
    # depends on community-graph membership.
    assert not retrieve._runtime_graph_rebuild_needed(store), (
        "a single new record must not force a rebuild under tolerance"
    )
    retrieve.build_runtime_graph(store)

    # The actual contract: the index recall path returns it — query with the
    # record's own embedding and assert it comes back.
    hits = store.query_similar(list(new_rec.embedding), k=5)
    hit_ids = {str(rec.id) for rec, _score in hits}
    assert str(new_rec.id) in hit_ids, (
        "a record added during the stale community-graph window must remain "
        "findable via the index recall path"
    )


def test_drift_tolerance_env_override_widens_and_narrows(
    monkeypatch: pytest.MonkeyPatch
):
    """The operator env knobs override the absolute floor and proportional band,
    and a malformed/negative override falls back to the default."""
    # Default: max(500, ceil(0.05 * cached)).
    monkeypatch.delenv("IAI_MCP_RGC_DRIFT_ABS", raising=False)
    monkeypatch.delenv("IAI_MCP_RGC_DRIFT_FRAC", raising=False)
    assert retrieve._drift_tolerance(0) == 500
    assert retrieve._drift_tolerance(100) == 500
    assert retrieve._drift_tolerance(40_000) == 2_000  # ceil(0.05*40000)

    # Narrow the floor.
    monkeypatch.setenv("IAI_MCP_RGC_DRIFT_ABS", "10")
    monkeypatch.setenv("IAI_MCP_RGC_DRIFT_FRAC", "0.0")
    assert retrieve._drift_tolerance(100) == 10
    assert retrieve._within_drift_tolerance(100, 110) is True
    assert retrieve._within_drift_tolerance(100, 111) is False

    # Widen the proportional band.
    monkeypatch.setenv("IAI_MCP_RGC_DRIFT_ABS", "0")
    monkeypatch.setenv("IAI_MCP_RGC_DRIFT_FRAC", "0.10")
    assert retrieve._drift_tolerance(1000) == 100  # ceil(0.10 * 1000)

    # Malformed / negative overrides fall back to defaults, never crash.
    monkeypatch.setenv("IAI_MCP_RGC_DRIFT_ABS", "not-an-int")
    monkeypatch.setenv("IAI_MCP_RGC_DRIFT_FRAC", "-1.0")
    assert retrieve._drift_tolerance(0) == 500
