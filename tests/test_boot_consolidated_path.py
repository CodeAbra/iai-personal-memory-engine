"""Tracer: the consolidated daemon boot path.

Boot preload now runs `runtime_graph_cache._rebuild_and_save_rgc(store)` --
DB->child streaming, never a full in-parent graph -- followed by
`retrieve.build_runtime_graph(store)` in the same thread, exactly as
`daemon._boot_preload_body` now does. The persisted compact centrality +
payload_record_count make the reconstruct child-free and
`store._warm_graph_bundle` is populated before the boot preload returns.

Proven on a small synthesized store (never the live store):

  1. `store._warm_graph_bundle` is populated after the sequence.
  2. The persisted cache carries a non-empty centrality map and
     `payload_record_count` equal to the active node count -- the
     discriminator between a working consolidated boot sequence and a
     silently-partial one.
  3. The reconstruct (`build_runtime_graph` after `_rebuild_and_save_rgc`)
     spawns ZERO detection/centrality children.
  4. Per-node betweenness centrality parity against a full in-parent rebuild
     on the same corpus -- the established recall-correctness proxy this
     suite uses (see `test_runtime_graph_cache_centrality_survives_cap.py`).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp import retrieve, runtime_graph_cache
from iai_mcp.store import MemoryStore, flush_record_buffer
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
    s = MemoryStore(path=tmp_path / "store")
    s.root = tmp_path
    return s


def _make_rec(
    seed: int, store: MemoryStore, created_at: "datetime | None" = None
) -> MemoryRecord:
    rng = np.random.default_rng(seed)
    vec = rng.random(store.embed_dim).astype(np.float32)
    vec = (vec / np.linalg.norm(vec)).tolist()
    now = created_at if created_at is not None else datetime.now(timezone.utc)
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
    """Chain n records into a path graph so interior nodes carry non-zero
    betweenness centrality -- an edgeless corpus has all-zero centrality and
    would not discriminate a wiped vs. a populated compact-centrality cache."""
    recs = [_make_rec(seed_base + i, store) for i in range(n)]
    for rec in recs:
        store.insert(rec)
    flush_record_buffer(store)
    ids = [rec.id for rec in recs]
    pairs = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
    store.boost_edges(pairs, delta=1.0, edge_type="hebbian")
    return recs


def _seed_connected_time_spread(
    store: MemoryStore, n: int, seed_base: int = 0
) -> list:
    """Chain n records into a path graph with TIME-SPREAD `created_at` values
    -- half backdated past the rich-club age-decay half-life, half fresh --
    so the recency-decay term actually moves the ranking. A same-instant
    fixture cannot discriminate a boot path that silently drops the decay
    signal: every node would decay by an equal, uniform amount."""
    now = datetime.now(timezone.utc)
    recs = [
        _make_rec(
            seed_base + i,
            store,
            created_at=now - timedelta(days=90) if i % 2 == 0 else now,
        )
        for i in range(n)
    ]
    for rec in recs:
        store.insert(rec)
    flush_record_buffer(store)
    ids = [rec.id for rec in recs]
    pairs = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
    store.boost_edges(pairs, delta=1.0, edge_type="hebbian")
    return recs


def _partition(node_to_community: dict) -> frozenset:
    groups: dict = {}
    for n, c in node_to_community.items():
        groups.setdefault(c, set()).add(str(n))
    return frozenset(frozenset(m) for m in groups.values())


def _install_no_child_guards(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Count entries into the two children the reconstruct must NOT spawn."""
    counts = {"detect": 0, "centrality": 0}

    real_detect = retrieve._detect_communities_isolated
    real_centrality_child = runtime_graph_cache.compute_centrality_in_child

    def _counting_detect(store_arg, graph, **kw):
        counts["detect"] += 1
        return real_detect(store_arg, graph, **kw)

    def _counting_centrality_child(graph, **kw):
        counts["centrality"] += 1
        return real_centrality_child(graph, **kw)

    monkeypatch.setattr(retrieve, "_detect_communities_isolated", _counting_detect)
    monkeypatch.setattr(
        runtime_graph_cache, "compute_centrality_in_child", _counting_centrality_child
    )
    return counts


def test_boot_consolidated_path_end_to_end(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
):
    """The consolidated boot sequence -- _rebuild_and_save_rgc then
    build_runtime_graph, both in the boot thread -- warms the bundle, spawns
    no child on the reconstruct, and preserves centrality parity."""
    _seed_connected(store, n=16, seed_base=3600)

    # Reference: a full in-parent rebuild on the SAME corpus, via today's
    # unmodified _build_runtime_graph_impl cache-miss algorithm -- the
    # recall-correctness ground truth this suite compares against elsewhere.
    graph_ref, assign_ref, _rc_ref = retrieve.build_runtime_graph(store)
    ref_centrality = {
        str(n): float(graph_ref._node_payload[str(n)].get("centrality", 0.0))
        for n in graph_ref.iter_nodes()
    }
    assert any(v != 0.0 for v in ref_centrality.values()), (
        "a path graph must produce non-zero interior betweenness"
    )

    # Reset to cold and run the CONSOLIDATED boot sequence: the DB->child
    # streaming rebuild, then the in-thread reconstruct -- exactly what
    # daemon._boot_preload_body now does.
    runtime_graph_cache.invalidate(store)
    assert getattr(store, "_warm_graph_bundle", None) is None

    result = runtime_graph_cache._rebuild_and_save_rgc(store)
    assert result.get("rebuilt") is True, f"boot rebuild did not complete: {result}"
    assert result.get("saved") is True, "boot rebuild must persist the cache"

    # The discriminator: the reconstruct that follows must be child-free.
    counts = _install_no_child_guards(monkeypatch)
    graph_new, assign_new, _rc_new = retrieve.build_runtime_graph(store)

    assert counts["detect"] == 0, (
        "the post-boot reconstruct must not spawn a community-detection child"
    )
    assert counts["centrality"] == 0, (
        "the post-boot reconstruct must not spawn a betweenness-centrality child"
    )

    # store._warm_graph_bundle populated in-thread, before preload_ready would
    # fire in the real boot sequence.
    assert getattr(store, "_warm_graph_bundle", None) is not None

    # The persisted cache carries a non-empty centrality map and a
    # payload_record_count equal to the active node count -- the discriminator
    # between a working consolidated boot sequence and a silently-partial one.
    cache_results = runtime_graph_cache.try_load_cache_results(store)
    assert cache_results is not None
    cached_centrality, payload_record_count = cache_results
    assert len(cached_centrality) > 0, (
        "the boot-path save must not zero the centrality cache"
    )
    assert payload_record_count == store.active_records_count(), (
        "payload_record_count must equal the streamed active node count"
    )

    # Recall top-k parity proxy: per-node betweenness centrality matches the
    # full-rebuild reference for every node in both builds' node sets.
    # Betweenness depends only on graph topology, so it is unaffected by the
    # unrelated seeded/cold community-detection prior-mode difference between
    # the two worker entry points.
    new_centrality = {
        str(n): float(graph_new._node_payload[str(n)].get("centrality", 0.0))
        for n in graph_new.iter_nodes()
    }
    assert set(new_centrality) == set(ref_centrality)
    for nid in ref_centrality:
        assert new_centrality[nid] == pytest.approx(ref_centrality[nid], abs=1e-6), (
            f"centrality parity broken for node {nid}"
        )

    # Community-partition parity too: _worker_entry detects with
    # prior_mode="cold" while _detect_communities_isolated (the reference
    # path) uses "seeded", but both pass prior=None and this fixture's node
    # count (16) is well under community.SMALL_N_FLAT (200) -- both paths
    # take the identical _flat_assignment(graph, None) branch regardless of
    # prior_mode, so the partitions are guaranteed identical here, not just
    # non-None. This closes the top-k/community-gate axis of the plan's
    # recall-parity must_have (upstream of the centrality-weighted rerank).
    assert assign_new is not None
    assert _partition(assign_ref.node_to_community) == _partition(
        assign_new.node_to_community
    ), "community partition must match the pre-change full-rebuild reference"


def test_boot_consolidated_path_rich_club_matches_reference_with_time_spread_ages(
    store: MemoryStore,
):
    """The boot-routed `rich_club` selection must match the reference
    full-rebuild path's selection when node ages are genuinely spread
    apart, not just on the same-instant fixture the tracer otherwise uses
    (which cannot discriminate a dropped decay signal since every node
    decays by an equal, uniform amount).

    Before the fix, the DB->child worker built every node with an EMPTY
    payload, so `richclub.rich_club_nodes`'s recency-decay term (reading
    `created_at` from the payload) unconditionally no-op'd on the boot path
    -- a pure-centrality selection silently diverging from the reference
    path's age-decayed one whenever real corpus ages differ.
    """
    _seed_connected_time_spread(store, n=16, seed_base=3700)

    # Reference: a full in-parent rebuild on the SAME time-spread corpus.
    _graph_ref, _assign_ref, rc_ref = retrieve.build_runtime_graph(store)
    assert len(rc_ref) > 0

    # Reset to cold and run the CONSOLIDATED boot sequence.
    runtime_graph_cache.invalidate(store)
    result = runtime_graph_cache._rebuild_and_save_rgc(store)
    assert result.get("rebuilt") is True, f"boot rebuild did not complete: {result}"

    # The reconstruct's cache-hit branch carries the boot worker's own
    # rich_club through unchanged (it is recomputed only on a cache miss),
    # so this is exactly the boot-routed selection the daemon would serve.
    _graph_new, _assign_new, rc_new = retrieve.build_runtime_graph(store)

    assert list(rc_new) == list(rc_ref), (
        "boot-path rich_club selection must match the reference full-rebuild "
        "path once node ages are spread apart -- a pure-centrality selection "
        "silently diverging from the age-decayed reference is the regression"
    )
