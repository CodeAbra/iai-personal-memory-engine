"""Regression guard: `_rebuild_and_save_rgc` must not wipe the persisted
centrality cache.

Previously, `_rebuild_and_save_rgc`'s `save_with_generation()` call carried
no `node_payload`, so `save()` unconditionally wrote an EMPTY centrality map
and a `payload_record_count` of 0 -- silently defeating the
`cache_results_fresh` warm path on every DROWSY/WAKE cycle (and the boot
path) until the next full `build_runtime_graph()` self-healed it. This module
pins the fix: the worker's already-computed centrality map now streams back
and survives the save.
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
    """Chain n records into a path graph so interior nodes carry non-zero
    betweenness centrality."""
    recs = [_make_rec(seed_base + i, store) for i in range(n)]
    for rec in recs:
        store.insert(rec)
    flush_record_buffer(store)
    ids = [rec.id for rec in recs]
    pairs = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
    store.boost_edges(pairs, delta=1.0, edge_type="hebbian")
    return recs


def test_rebuild_and_save_rgc_writes_nonempty_centrality(store: MemoryStore):
    """A `_rebuild_and_save_rgc` call on a seeded store writes a NON-empty
    `data["centrality"]` map and `payload_record_count` == the streamed node
    count -- was `{}` / 0 when `node_payload` was omitted from the save
    call."""
    _seed_connected(store, n=14, seed_base=4200)

    result = runtime_graph_cache._rebuild_and_save_rgc(store)
    assert result.get("rebuilt") is True, f"rebuild did not complete: {result}"
    assert result.get("saved") is True

    data = runtime_graph_cache._load_and_decrypt_cache(store)
    assert data is not None
    assert isinstance(data.get("centrality"), dict)
    assert len(data["centrality"]) > 0, (
        "_rebuild_and_save_rgc must not write an empty centrality map"
    )
    assert data.get("payload_record_count") == result["node_count"]
    assert result["node_count"] == store.active_records_count()


def test_rebuild_and_save_rgc_does_not_wipe_prior_good_centrality(
    store: MemoryStore,
):
    """A prior good centrality cache written by a full `build_runtime_graph`
    is NOT zeroed by a subsequent `_rebuild_and_save_rgc` call."""
    _seed_connected(store, n=14, seed_base=4300)

    # Full rebuild: writes node_payload + centrality via the retrieve.py path.
    retrieve.build_runtime_graph(store)
    before = runtime_graph_cache.try_load_cache_results(store)
    assert before is not None
    centrality_before, _count_before = before
    assert len(centrality_before) > 0

    # A DROWSY/WAKE-shaped (or boot-shaped) rebuild must not wipe it.
    result = runtime_graph_cache._rebuild_and_save_rgc(store, force=True)
    assert result.get("saved") is True

    after = runtime_graph_cache.try_load_cache_results(store)
    assert after is not None
    centrality_after, count_after = after
    assert len(centrality_after) > 0, (
        "a prior good centrality cache must survive a _rebuild_and_save_rgc call"
    )
    assert count_after == result["node_count"]


def test_centrality_survives_whole_boot_sequence_no_second_save(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
):
    """After `_rebuild_and_save_rgc` then `build_runtime_graph`, the
    retrieve.py re-save guard is False -- no second `save()` clobbers or
    double-bumps the generation the boot rebuild just committed."""
    _seed_connected(store, n=14, seed_base=4400)

    runtime_graph_cache._rebuild_and_save_rgc(store)
    gen_after_rebuild = runtime_graph_cache.get_current_generation()

    save_calls = {"n": 0}
    real_save = runtime_graph_cache.save

    def _counting_save(*a, **kw):
        save_calls["n"] += 1
        return real_save(*a, **kw)

    # retrieve.py calls runtime_graph_cache.save(...) via the module
    # attribute (no local alias), so patching the module attribute intercepts
    # it.
    monkeypatch.setattr(runtime_graph_cache, "save", _counting_save)

    graph, assignment, _rc = retrieve.build_runtime_graph(store)

    assert save_calls["n"] == 0, (
        "the post-boot reconstruct must not re-save"
    )
    assert runtime_graph_cache.get_current_generation() == gen_after_rebuild
    assert assignment is not None
    assert graph.node_count() == 14


def test_boot_restart_repopulates_wiped_centrality_not_skipped(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
):
    """A decodable, structurally-warm on-disk cache with an
    EMPTY centrality map (the standing legacy state a `save_with_generation`
    call with no `node_payload`/`centrality` writes) must not be inherited
    silently across a daemon restart. The unforced skip gate must also
    require populated centrality, not just structural decodability -- the
    dirty counter alone cannot catch this because it resets to 0 on every
    process start.

    This drives the exact boot call sequence (`_rebuild_and_save_rgc(store)`
    with the default `force=False`, then `build_runtime_graph(store)`) rather
    than passing `force=True` itself, so the test exercises the real restart
    path the boot preload takes.
    """
    _seed_connected(store, n=14, seed_base=4600)

    # Establish a real community assignment, then persist it via the LEGACY
    # call shape (no node_payload, no centrality) -- reproducing the standing
    # pre-existing state a `_rebuild_and_save_rgc` call made before the
    # worker-centrality stream landed.
    _graph, assignment, rc = retrieve.build_runtime_graph(store)
    saved = runtime_graph_cache.save_with_generation(
        store, assignment, rc, max_degree=0, node_degrees={},
    )
    assert saved is True

    fixture_check = runtime_graph_cache.try_load_cache_results(store)
    assert fixture_check is not None
    assert len(fixture_check[0]) == 0, (
        "fixture must start decodable but with an EMPTY centrality map"
    )
    fixture_source = runtime_graph_cache.load_recall_structural(store)[3]
    assert fixture_source in ("overlay", "normal"), (
        "fixture must be structurally warm, not cold -- the fix's "
        "discriminator is a WARM-but-wiped cache, not a cold one"
    )

    # Simulate a daemon restart: the dirty counter has no snapshot-restore
    # and always starts at 0, regardless of what the disk cache holds.
    runtime_graph_cache.reset_dirty_counter()

    centrality_child_calls = {"n": 0}
    detect_calls = {"n": 0}
    real_centrality_child = runtime_graph_cache.compute_centrality_in_child
    real_detect = retrieve._detect_communities_isolated

    def _counting_centrality_child(graph_arg, **kw):
        centrality_child_calls["n"] += 1
        return real_centrality_child(graph_arg, **kw)

    def _counting_detect(store_arg, graph_arg, **kw):
        detect_calls["n"] += 1
        return real_detect(store_arg, graph_arg, **kw)

    monkeypatch.setattr(
        runtime_graph_cache, "compute_centrality_in_child", _counting_centrality_child
    )
    monkeypatch.setattr(retrieve, "_detect_communities_isolated", _counting_detect)

    # The exact boot sequence, unforced.
    result = runtime_graph_cache._rebuild_and_save_rgc(store)
    assert result.get("rebuilt") is True, (
        f"boot's unforced rebuild must not skip a centrality-empty cache: {result}"
    )
    retrieve.build_runtime_graph(store)

    assert centrality_child_calls["n"] == 0, (
        "the post-boot reconstruct must not spawn a second centrality child "
        "-- the boot rebuild's own write must already make the cache fresh"
    )
    assert detect_calls["n"] == 0, (
        "the post-boot reconstruct must not spawn a community-detection child"
    )

    after = runtime_graph_cache.try_load_cache_results(store)
    assert after is not None
    assert len(after[0]) > 0, (
        "boot restart must repopulate centrality, not inherit the wiped cache"
    )


def test_boot_rebuild_reconstruct_is_child_free(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
):
    """After a boot rebuild, `cache_results_fresh` is True and no detection /
    centrality child is spawned on the reconstruct."""
    _seed_connected(store, n=14, seed_base=4500)

    runtime_graph_cache._rebuild_and_save_rgc(store)

    cache_results = runtime_graph_cache.try_load_cache_results(store)
    assert cache_results is not None
    cached_centrality, payload_record_count = cache_results
    records_count = store.active_records_count()
    assert len(cached_centrality) > 0
    assert retrieve._within_drift_tolerance(payload_record_count, records_count)

    detect_calls = {"n": 0}
    centrality_calls = {"n": 0}
    real_detect = retrieve._detect_communities_isolated
    real_centrality_child = runtime_graph_cache.compute_centrality_in_child

    def _counting_detect(store_arg, graph, **kw):
        detect_calls["n"] += 1
        return real_detect(store_arg, graph, **kw)

    def _counting_centrality_child(graph, **kw):
        centrality_calls["n"] += 1
        return real_centrality_child(graph, **kw)

    monkeypatch.setattr(retrieve, "_detect_communities_isolated", _counting_detect)
    monkeypatch.setattr(
        runtime_graph_cache, "compute_centrality_in_child", _counting_centrality_child
    )

    graph, assignment, _rc = retrieve.build_runtime_graph(store)

    assert detect_calls["n"] == 0, (
        "cache_results_fresh must be True: no community-detection child on "
        "the post-boot reconstruct"
    )
    assert centrality_calls["n"] == 0, (
        "cache_results_fresh must be True: no centrality child on the "
        "post-boot reconstruct"
    )
    assert assignment is not None
    assert graph.node_count() == 14
