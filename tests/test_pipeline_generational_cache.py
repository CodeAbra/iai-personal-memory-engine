from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_store import _make  # noqa: E402

import iai_mcp.pipeline as _pipeline_mod  # noqa: E402
from iai_mcp.pipeline import SimpleRecordView, _recall_core  # noqa: E402
from iai_mcp.retrieve import build_runtime_graph  # noqa: E402
from iai_mcp.store import MemoryStore  # noqa: E402
from iai_mcp.types import EMBED_DIM  # noqa: E402


def _unit_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _populate(store: MemoryStore, n: int = 20, seed: int = 20260825) -> None:
    rng = np.random.default_rng(seed)
    for i in range(n):
        v = rng.standard_normal(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        store.insert(_make(text=f"filler record {i}", vec=v.tolist()))


def _call(store, graph, assignment, rich_club, cue_vec, cue="probe cue"):
    return _recall_core(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=None,
        cue=cue,
        session_id="test-session",
        cue_embedding=list(cue_vec),
    )


def test_records_view_cache_hit_skips_rebuild_and_is_value_equal(tmp_path):
    store = MemoryStore(path=str(tmp_path / "store"))
    cue_vec = _unit_vec(1)
    _populate(store, n=25)
    graph, assignment, rich_club = build_runtime_graph(store)

    _call(store, graph, assignment, rich_club, cue_vec)
    cached = getattr(graph, "_records_view_cache", None)
    assert cached is not None, "first call must populate the generational cache"
    version1, base1 = cached
    snapshot1 = {rid: rec.literal_surface for rid, rec in base1.items()}
    assert snapshot1, "cache must be non-empty on a populated store"

    original_get_payload = graph.get_payload
    calls: list = []

    def _counting_get_payload(rid):
        calls.append(rid)
        return original_get_payload(rid)

    graph.get_payload = _counting_get_payload
    try:
        _call(store, graph, assignment, rich_club, cue_vec)
    finally:
        graph.get_payload = original_get_payload

    assert calls == [], (
        "a HIT must not re-walk graph.iter_nodes()/get_payload() -- "
        f"{len(calls)} payload reads happened on the second call"
    )
    version2, base2 = graph._records_view_cache
    assert version2 == version1
    snapshot2 = {rid: rec.literal_surface for rid, rec in base2.items()}
    assert snapshot2 == snapshot1


def test_records_view_cache_miss_after_insert_reflects_new_record(tmp_path):
    store = MemoryStore(path=str(tmp_path / "store"))
    cue_vec = _unit_vec(2)
    _populate(store, n=25)
    graph, assignment, rich_club = build_runtime_graph(store)

    _call(store, graph, assignment, rich_club, cue_vec)
    version1 = graph._records_view_cache[0]

    new_rec = _make(text="freshly inserted record", vec=cue_vec)
    store.insert(new_rec)

    _call(store, graph, assignment, rich_club, cue_vec)
    version2, base2 = graph._records_view_cache
    assert version2 != version1, "an insert must bump the version and invalidate the cache"
    assert new_rec.id in base2, "the rebuilt cache must include the newly inserted record"


def test_records_view_cache_copy_on_serve_isolation(tmp_path, monkeypatch):
    store = MemoryStore(path=str(tmp_path / "store"))
    cue_vec = _unit_vec(3)
    _populate(store, n=25)
    graph, assignment, rich_club = build_runtime_graph(store)

    # A "ghost" node: present on the graph with no resolvable embedding
    # there (embedding=[]), so it is excluded from the initial records_cache
    # build (pipeline.py's "embedding" in node guard) but a real store-
    # backed record. _collect_graph_pool's own store fallback resolves it
    # fresh via store.get_batch and writes it into the per-call records_cache
    # dict (pipeline.py:673-674) -- this isolates that dict-key write site;
    # the field-mutation write sites (community_id, profile_modulation_gain)
    # are covered separately by
    # test_records_view_cache_community_and_gain_never_mutate_shared_object
    # below.
    ghost_rec = _make(text="widened ghost record", vec=list(cue_vec))
    store.insert(ghost_rec)
    ghost_id = ghost_rec.id
    graph.add_node(ghost_id, community_id=None, embedding=[])

    result_a = _call(store, graph, assignment, rich_club, cue_vec, cue="cue A")
    served_a = {h.record_id for h in result_a.scored_hits}
    assert ghost_id in served_a, (
        "the store-fallback write must have actually fired (and the ghost "
        "record scored) for this test to be meaningful"
    )

    base_after_a = graph._records_view_cache[1]
    assert ghost_id not in base_after_a, (
        "cue A's store-fallback write mutated the shared cached base in "
        "place -- copy-on-serve is broken"
    )

    _call(store, graph, assignment, rich_club, cue_vec, cue="cue B")
    base_after_b = graph._records_view_cache[1]
    assert ghost_id not in base_after_b, (
        "cue B's served records_cache carried cue A's store-fallback "
        "record -- cross-call pollution of the cached base"
    )


def test_records_view_cache_community_and_gain_never_mutate_shared_object(tmp_path, monkeypatch):
    """SimpleRecordView is a mutable dataclass; a HIT re-copies the SAME
    object instances that live in graph._records_view_cache. The ranking
    loop must never write community_id / profile_modulation_gain onto those
    shared objects -- doing so would leak one call's per-cue result onto a
    later call that reuses the same cached graph with a different cue.

    Reuses one graph across two cues via the real _recall_core path: cue A
    resolves a community hit for a target record, cue B does not. Asserts
    the shared cached object never carries either field (proven right after
    cue A, before cue B even runs), that cue B's served community_id is not
    cue A's leaked value, and that cue B's result matches a call against a
    genuinely fresh (never-mutated) cache.
    """
    from iai_mcp.lilli.profile.knobs import profile_modulation_for_record

    store = MemoryStore(path=str(tmp_path / "store"))
    cue_vec = _unit_vec(40)
    _populate(store, n=20)
    graph, assignment, rich_club = build_runtime_graph(store)

    fake_community = uuid4()
    profile_state = {"monotropism_depth": {"fake-community-name": 0.5}}
    monkeypatch.setattr(
        "iai_mcp.core.get_community_names",
        lambda: {str(fake_community): "fake-community-name"},
    )

    def _call_with_gate(cue, gated, scores):
        def _fake_gate_scored(cue_emb, assignment_arg, top_n=3, member_embeddings=None):
            return gated, scores

        monkeypatch.setattr(_pipeline_mod, "_community_gate_scored", _fake_gate_scored)
        return _recall_core(
            store=store,
            graph=graph,
            assignment=assignment,
            rich_club=rich_club,
            embedder=None,
            cue=cue,
            session_id="test-session",
            cue_embedding=list(cue_vec),
            profile_state=profile_state,
        )

    # Warm-up call with no community hit -- establishes the base cache.
    _call_with_gate("warm-up cue", [], {})
    base = graph._records_view_cache[1]
    target_id = next(iter(base))
    assignment.mid_regions = {fake_community: [target_id]}

    # Call A: cue that DOES resolve a community hit for target_id.
    result_a = _call_with_gate(
        "cue A community hit", [fake_community], {fake_community: 1.0},
    )
    hit_a = next(h for h in result_a.scored_hits if h.record_id == target_id)
    assert hit_a.community_id == fake_community, (
        "test setup must actually exercise the community-hit branch"
    )
    assert result_a._profile_gains.get(target_id), (
        "test setup must actually exercise the profile-modulation-gain branch"
    )

    # The shared cached object must never be field-mutated -- checked BEFORE
    # cue B runs, isolating "does A leak onto the cache" from "does B read
    # A's leak".
    cached_rec = graph._records_view_cache[1][target_id]
    assert cached_rec.community_id is None, (
        "the shared cached SimpleRecordView must never be field-mutated -- "
        "cue A's community_id leaked onto the object every future call reads"
    )
    assert cached_rec.profile_modulation_gain == {}, (
        "the shared cached SimpleRecordView must never be field-mutated -- "
        "cue A's profile_modulation_gain leaked onto the object"
    )

    # Call B: SAME graph, cache HIT (no mutation between calls), cue that
    # does NOT resolve a community hit for target_id.
    result_b = _call_with_gate("cue B no community hit", [], {})
    hit_b = next(h for h in result_b.scored_hits if h.record_id == target_id)
    assert hit_b.community_id is None, (
        "cue B must not see cue A's stale community_id -- this is the "
        "cross-call field leak this test guards against"
    )
    assert not result_b._profile_gains.get(target_id), (
        "cue B must not see cue A's stale profile_modulation_gain -- this "
        "is the cross-call field leak this test guards against"
    )

    # Non-vacuous against a genuinely fresh cache too: force a rebuild (no
    # prior mutation history on the new objects at all) and confirm cue B's
    # result is unchanged.
    graph._records_view_cache = None
    result_fresh = _call_with_gate("cue B no community hit", [], {})
    hit_fresh = next(h for h in result_fresh.scored_hits if h.record_id == target_id)
    assert hit_b.community_id == hit_fresh.community_id
    assert (result_b._profile_gains.get(target_id) or {}) == (
        result_fresh._profile_gains.get(target_id) or {}
    )

    # Direct proof the downstream reader (knobs.py) sees the call-local
    # value, not a residual object field -- the shared object's field stays
    # None even though this call resolves a real community_id_override.
    assert profile_modulation_for_record(
        cached_rec, profile_state, community_id_override=fake_community,
    ), "profile_modulation_for_record must honor an explicit override"
    assert cached_rec.community_id is None, (
        "calling profile_modulation_for_record with an override must not "
        "itself mutate the record"
    )


def test_records_view_cache_never_caches_store_fallback(tmp_path, monkeypatch):
    store = MemoryStore(path=str(tmp_path / "store"))
    cue_vec = _unit_vec(4)
    _populate(store, n=10)
    graph, assignment, rich_club = build_runtime_graph(store)

    _call(store, graph, assignment, rich_club, cue_vec)
    cached_before = getattr(graph, "_records_view_cache", None)
    assert cached_before is not None, "a normal call must populate the cache first"

    # Bump the version (a genuine MISS trigger) AND force the sweep to
    # resolve zero nodes from the graph on the SAME call -- this reaches the
    # store-fallback branch (pipeline.py) on a call that is NOT a HIT, so a
    # bug that caches the fallback unconditionally on any MISS (rather than
    # only a fully graph-resolved MISS) would still be caught.
    store.insert(_make(text="version bump", vec=list(cue_vec)))
    monkeypatch.setattr(graph, "iter_nodes", lambda: iter(()))
    _call(store, graph, assignment, rich_club, cue_vec)

    assert graph._records_view_cache == cached_before, (
        "a store-fallback (partial/unresolved) sweep must never overwrite "
        "the generational cache -- even though the version had genuinely "
        "moved on, the stale cached entry must be left exactly as it was"
    )


def test_records_view_cache_kill_switch_forces_unconditional_rebuild(tmp_path, monkeypatch):
    store = MemoryStore(path=str(tmp_path / "store"))
    cue_vec = _unit_vec(5)
    _populate(store, n=15)
    graph, assignment, rich_club = build_runtime_graph(store)

    _call(store, graph, assignment, rich_club, cue_vec)
    assert getattr(graph, "_records_view_cache", None) is not None, (
        "a normal call must populate the cache first"
    )

    monkeypatch.setenv("IAI_MCP_GENERATIONAL_CACHE_OFF", "1")

    original_get_payload = graph.get_payload
    calls: list = []

    def _counting_get_payload(rid):
        calls.append(rid)
        return original_get_payload(rid)

    graph.get_payload = _counting_get_payload
    try:
        _call(store, graph, assignment, rich_club, cue_vec)
    finally:
        graph.get_payload = original_get_payload

    assert calls, (
        "the kill-switch must force a full rebuild every call, even when a "
        "live cache entry exists for the current version"
    )


def test_degree_map_cache_retired_no_such_attribute(tmp_path):
    """`_degree_map_cache` is retired (memo-drop): the graph never carries
    this attribute, on the first call or any later one."""
    store = MemoryStore(path=str(tmp_path / "store"))
    cue_vec = _unit_vec(6)
    _populate(store, n=25)
    graph, assignment, rich_club = build_runtime_graph(store)

    _call(store, graph, assignment, rich_club, cue_vec)
    assert not hasattr(graph, "_degree_map_cache"), (
        "_degree_map_cache is retired -- no code path may set it"
    )
    _call(store, graph, assignment, rich_club, cue_vec)
    assert not hasattr(graph, "_degree_map_cache")


def test_degree_recomputed_every_call(tmp_path):
    """Retired (memo-drop): `graph.degrees(...)` runs on every recall, not
    only on a cache miss -- there is no cache left to hit."""
    store = MemoryStore(path=str(tmp_path / "store"))
    cue_vec = _unit_vec(7)
    _populate(store, n=25)
    graph, assignment, rich_club = build_runtime_graph(store)

    _call(store, graph, assignment, rich_club, cue_vec)

    original_degrees = graph.degrees
    calls: list = []

    def _counting_degrees(*args, **kwargs):
        calls.append(1)
        return original_degrees(*args, **kwargs)

    graph.degrees = _counting_degrees
    try:
        _call(store, graph, assignment, rich_club, cue_vec)
    finally:
        graph.degrees = original_degrees

    assert calls, "every recall must recompute degree -- no cache remains to skip it"


def test_degree_map_cache_leaves_global_override_branch_untouched(tmp_path):
    store = MemoryStore(path=str(tmp_path / "store"))
    cue_vec = _unit_vec(9)
    _populate(store, n=10)
    graph, assignment, rich_club = build_runtime_graph(store)

    graph._global_degree = {str(nid): 99 for nid in graph.iter_nodes()}
    graph._max_degree = 99

    original_degrees = graph.degrees
    calls: list = []

    def _counting_degrees(*args, **kwargs):
        calls.append(1)
        return original_degrees(*args, **kwargs)

    graph.degrees = _counting_degrees
    try:
        _call(store, graph, assignment, rich_club, cue_vec)
    finally:
        graph.degrees = original_degrees

    assert calls == [], (
        "the _global_deg_override branch must never call graph.degrees(...) "
        "-- the override is read instead"
    )


def test_degree_stage_bucket_appears_in_stage_profile(tmp_path, monkeypatch):
    store = MemoryStore(path=str(tmp_path / "store"))
    cue_vec = _unit_vec(10)
    _populate(store, n=15)
    graph, assignment, rich_club = build_runtime_graph(store)

    monkeypatch.setenv("IAI_MCP_STAGE_PROFILE", "1")
    _call(store, graph, assignment, rich_club, cue_vec)

    assert "degree" in _pipeline_mod._last_stage_timings_ms, (
        "IAI_MCP_STAGE_PROFILE=1 must record a 'degree' stage bucket"
    )


# ---------------------------------------------------------------------------
# Per-write-surface staleness regression: every surface that mutates the live
# graph must invalidate the cache; every SQL-only surface must be proven
# orthogonal -- serving no view staler than a genuinely fresh rebuild would.
# Each test first proves the cache is a warm HIT before the mutation under
# test, so a passing test cannot be explained by the cache never having been
# warm in the first place.
# ---------------------------------------------------------------------------


def _prove_records_cache_warm_hit(store, graph, assignment, rich_club, cue_vec, cue="probe cue"):
    _call(store, graph, assignment, rich_club, cue_vec, cue=cue)
    version_before, base_before = graph._records_view_cache
    original_get_payload = graph.get_payload
    calls: list = []

    def _counting_get_payload(rid):
        calls.append(rid)
        return original_get_payload(rid)

    graph.get_payload = _counting_get_payload
    try:
        _call(store, graph, assignment, rich_club, cue_vec, cue=cue)
    finally:
        graph.get_payload = original_get_payload
    assert calls == [], (
        "records_cache must already be a warm HIT before the mutation "
        "under test -- otherwise a pass here would not distinguish real "
        "cache behavior from having no cache at all"
    )
    assert graph._records_view_cache[0] == version_before
    return version_before, base_before


def _fresh_degree_map(graph) -> "tuple[dict, float]":
    """`_degree_map_cache` is retired -- degree is recomputed every call, so
    the "before" snapshot these staleness tests compare against is just a
    direct, uncached `graph.degrees(...)` sweep."""
    degree = {
        str(nid): deg
        for nid, deg in graph.degrees(exclude_types=graph.RANKING_DEGREE_EXCLUDED)
    }
    max_deg = float(max(degree.values(), default=0))
    return degree, max_deg


# --- Positive surfaces: bump the version, next recall reflects the change ---


def test_staleness_insert_surface_bumps_and_reflects_new_record(tmp_path):
    store = MemoryStore(path=str(tmp_path / "store"))
    cue_vec = _unit_vec(20)
    _populate(store, n=20)
    graph, assignment, rich_club = build_runtime_graph(store)

    version_before, _base_before = _prove_records_cache_warm_hit(
        store, graph, assignment, rich_club, cue_vec
    )

    new_rec = _make(text="freshly inserted staleness record", vec=cue_vec)
    store.insert(new_rec)
    assert graph._pool_content_version != version_before, (
        "insert -> add_node/set_node_payload must bump _pool_content_version"
    )

    _call(store, graph, assignment, rich_club, cue_vec)
    version_after, base_after = graph._records_view_cache
    assert version_after != version_before, "an insert must invalidate the cache"
    assert new_rec.id in base_after, "the next recall must reflect the inserted record"


def test_staleness_delete_surface_bumps_and_rebuilds_view(tmp_path):
    store = MemoryStore(path=str(tmp_path / "store"))
    cue_vec = _unit_vec(21)
    _populate(store, n=20)
    graph, assignment, rich_club = build_runtime_graph(store)

    version_before, base_before = _prove_records_cache_warm_hit(
        store, graph, assignment, rich_club, cue_vec
    )
    target_id = next(iter(base_before))

    store.delete(target_id)
    assert graph._pool_content_version != version_before, (
        "delete -> remove_node must bump _pool_content_version"
    )

    _call(store, graph, assignment, rich_club, cue_vec)
    version_after, base_after = graph._records_view_cache
    assert version_after != version_before, (
        "a delete must invalidate the cache, not silently keep serving the "
        "pre-delete view"
    )
    # Not a hardcoded presence/absence assertion: the invariant under test is
    # that the cache reflects a genuine rebuild of the CURRENT live graph
    # (tombstoned records can still be served elsewhere, e.g. the
    # live-recency rail, under existing semantics unrelated to this cache).
    # Prove the rebuild is real by comparing directly against the graph's
    # own current node set, which is ground truth for what remove_node did.
    live_ids = set(graph.iter_nodes())
    assert target_id not in live_ids, (
        "remove_node must have actually dropped the deleted id from the "
        "live graph -- otherwise this test proves nothing about staleness"
    )
    assert set(base_after.keys()) <= live_ids, (
        "the rebuilt view must only ever contain ids the live graph "
        "currently carries"
    )


def test_staleness_payload_update_surface_bumps_and_reflects_change(tmp_path):
    store = MemoryStore(path=str(tmp_path / "store"))
    cue_vec = _unit_vec(22)
    _populate(store, n=20)
    graph, assignment, rich_club = build_runtime_graph(store)

    version_before, base_before = _prove_records_cache_warm_hit(
        store, graph, assignment, rich_club, cue_vec
    )
    target_id = next(iter(base_before))
    rec = store.get(target_id)
    assert rec is not None
    assert base_before[target_id].centrality == pytest.approx(rec.centrality)
    assert rec.centrality != pytest.approx(0.987)

    rec.centrality = 0.987
    store.update(rec)
    assert graph._pool_content_version != version_before, (
        "update -> set_node_payload must bump _pool_content_version"
    )

    _call(store, graph, assignment, rich_club, cue_vec)
    version_after, base_after = graph._records_view_cache
    assert version_after != version_before, "an update must invalidate the cache"
    assert base_after[target_id].centrality == pytest.approx(0.987), (
        "the next recall must reflect the updated payload field"
    )


def test_staleness_centrality_surface_bumps_and_reflects_change(tmp_path):
    """Mirrors retrieve.py's `_apply_centrality_map`, which for each node in
    a freshly computed community-rebuild centrality map calls exactly
    `graph.set_node_centrality(rid, float(cval))` -- exercising that same
    call directly proves the cache invalidates on this write surface
    without spinning up a full betweenness-centrality computation."""
    store = MemoryStore(path=str(tmp_path / "store"))
    cue_vec = _unit_vec(23)
    _populate(store, n=20)
    graph, assignment, rich_club = build_runtime_graph(store)

    version_before, base_before = _prove_records_cache_warm_hit(
        store, graph, assignment, rich_club, cue_vec
    )
    target_id = next(iter(base_before))
    assert base_before[target_id].centrality != pytest.approx(0.741)

    graph.set_node_centrality(target_id, 0.741)
    assert graph._pool_content_version != version_before, (
        "set_node_centrality -> set_node_payload must bump _pool_content_version"
    )

    _call(store, graph, assignment, rich_club, cue_vec)
    version_after, base_after = graph._records_view_cache
    assert version_after != version_before, (
        "a community-rebuild centrality write must invalidate the cache"
    )
    assert base_after[target_id].centrality == pytest.approx(0.741), (
        "the next recall must reflect the community-rebuild centrality write"
    )


def test_staleness_edge_add_surface_bumps_degree_cache(tmp_path):
    store = MemoryStore(path=str(tmp_path / "store"))
    cue_vec = _unit_vec(24)
    _populate(store, n=20)
    graph, assignment, rich_club = build_runtime_graph(store)

    version_before = graph._pool_content_version
    degree_before, _max_deg_before = _fresh_degree_map(graph)

    node_ids = list(graph.iter_nodes())
    a, b = node_ids[0], node_ids[1]
    deg_a_before = degree_before.get(str(a), 0)
    graph.add_edge(a, b, weight=1.0, edge_type="hebbian")
    assert graph._pool_content_version != version_before, (
        "add_edge must bump _pool_content_version"
    )

    _call(store, graph, assignment, rich_club, cue_vec)
    degree_after, _max_deg_after = _fresh_degree_map(graph)
    assert degree_after.get(str(a), 0) > deg_a_before, (
        "the degree sweep must reflect the new edge"
    )


# --- Negative surfaces: SQL-only writes, honest orthogonality -------------
#
# None of these writes reach the live graph (_node_payload/_adj) -- they
# never touch _pool_content_version, so a recall (cached or freshly
# rebuilt) is equally unaware of them until the next graph-syncing write.
# The cache introduces ZERO new staleness relative to today's uncached
# path: a store-only write that never mutates the in-memory graph cannot be
# made stale by a cache keyed on the graph's content version.


def test_staleness_boost_edges_surface_is_orthogonal(tmp_path):
    """SQL edges table only (_store.py:3415) -- boost_edges never touches
    the live graph's _adj."""
    store = MemoryStore(path=str(tmp_path / "store"))
    cue_vec = _unit_vec(25)
    _populate(store, n=20)
    graph, assignment, rich_club = build_runtime_graph(store)

    version_before = graph._pool_content_version
    degree_before, _max_deg_before = _fresh_degree_map(graph)

    node_ids = list(graph.iter_nodes())
    a, b = node_ids[0], node_ids[1]
    store.boost_edges([(a, b)], delta=0.5)

    assert graph._pool_content_version == version_before, (
        "boost_edges is SQL-only and must never bump _pool_content_version"
    )
    fresh_degree, _max_deg_after = _fresh_degree_map(graph)
    assert fresh_degree == degree_before, (
        "boost_edges must not change the live graph's adjacency -- a "
        "genuinely fresh degree sweep, taken right now, must equal the "
        "pre-boost value"
    )

    _call(store, graph, assignment, rich_club, cue_vec)


def test_staleness_add_contradicts_edge_surface_is_orthogonal(tmp_path):
    """SQL edges buffer only (_store.py:3713), no graph hook."""
    store = MemoryStore(path=str(tmp_path / "store"))
    cue_vec = _unit_vec(26)
    _populate(store, n=20)
    graph, assignment, rich_club = build_runtime_graph(store)

    version_before = graph._pool_content_version
    degree_before, _max_deg_before = _fresh_degree_map(graph)

    node_ids = list(graph.iter_nodes())
    a, b = node_ids[0], node_ids[1]
    store.add_contradicts_edge(a, b)

    assert graph._pool_content_version == version_before, (
        "add_contradicts_edge is SQL edges-buffer-only and must never bump "
        "_pool_content_version"
    )
    fresh_degree, _max_deg_after = _fresh_degree_map(graph)
    assert fresh_degree == degree_before, (
        "add_contradicts_edge must not change the live graph's adjacency "
        "-- a genuinely fresh degree sweep must equal the pre-write value"
    )

    _call(store, graph, assignment, rich_club, cue_vec)


def test_staleness_add_tags_surface_is_orthogonal(tmp_path):
    """Bare SQL `records` update, no `_fire_graph_sync_hook` (_store.py:1844)
    -- add_tags never reaches graph._node_payload, so any recall (cached or
    freshly rebuilt) is equally blind to the new tag until the next
    graph-syncing write."""
    store = MemoryStore(path=str(tmp_path / "store"))
    cue_vec = _unit_vec(27)
    _populate(store, n=20)
    graph, assignment, rich_club = build_runtime_graph(store)

    version_before, base_before = _prove_records_cache_warm_hit(
        store, graph, assignment, rich_club, cue_vec
    )
    target_id = next(iter(base_before))
    payload_tags_before = list(graph.get_payload(target_id).get("tags") or [])
    assert "freshly_added_tag" not in base_before[target_id].tags

    changed = store.add_tags(target_id, ["freshly_added_tag"])
    assert changed, "the SQL tag write must have actually landed for this test to be meaningful"

    assert graph._pool_content_version == version_before, (
        "add_tags is bare SQL and must never bump _pool_content_version"
    )
    # The live graph payload is the sole source of truth records_cache reads
    # from -- prove it is untouched, so a genuinely fresh rebuild would be
    # equally unaware of the new tag.
    assert graph.get_payload(target_id).get("tags") == payload_tags_before, (
        "add_tags must never write into graph._node_payload"
    )

    _call(store, graph, assignment, rich_club, cue_vec)
    version_after, base_after = graph._records_view_cache
    assert version_after == version_before, "cache must still be a HIT"
    assert "freshly_added_tag" not in base_after[target_id].tags, (
        "the cache must not diverge from a fresh rebuild -- both are "
        "equally unaware of a SQL-only tag write"
    )


def test_staleness_set_aaak_index_surface_is_orthogonal(tmp_path):
    """Bare SQL column update (_store.py:1869), no hook -- the generic
    ``update()`` does not carry aaak_index, and neither does this verb."""
    store = MemoryStore(path=str(tmp_path / "store"))
    cue_vec = _unit_vec(28)
    _populate(store, n=20)
    graph, assignment, rich_club = build_runtime_graph(store)

    version_before, base_before = _prove_records_cache_warm_hit(
        store, graph, assignment, rich_club, cue_vec
    )
    target_id = next(iter(base_before))
    payload_aaak_before = graph.get_payload(target_id).get("aaak_index")
    assert base_before[target_id].aaak_index != "E:freshly_regenerated_index"

    changed = store.set_aaak_index(target_id, "E:freshly_regenerated_index")
    assert changed, "the SQL aaak_index write must have actually landed"

    assert graph._pool_content_version == version_before, (
        "set_aaak_index is a bare SQL column update and must never bump "
        "_pool_content_version"
    )
    assert graph.get_payload(target_id).get("aaak_index") == payload_aaak_before, (
        "set_aaak_index must never write into graph._node_payload"
    )

    _call(store, graph, assignment, rich_club, cue_vec)
    version_after, base_after = graph._records_view_cache
    assert version_after == version_before, "cache must still be a HIT"
    assert base_after[target_id].aaak_index != "E:freshly_regenerated_index", (
        "the cache must not diverge from a fresh rebuild -- both are "
        "equally unaware of a SQL-only aaak_index write"
    )


def test_staleness_remove_tags_surface_is_orthogonal(tmp_path):
    """Bare SQL `records` update (_store.py:1888), no hook."""
    store = MemoryStore(path=str(tmp_path / "store"))
    cue_vec = _unit_vec(29)
    _populate(store, n=20)
    graph, assignment, rich_club = build_runtime_graph(store)

    version_before, base_before = _prove_records_cache_warm_hit(
        store, graph, assignment, rich_club, cue_vec
    )
    target_id = next(iter(base_before))
    # Setup: add a tag via the same SQL-only surface (never touches the
    # graph), so there is something real to remove.
    store.add_tags(target_id, ["to_be_removed"])
    payload_tags_before = list(graph.get_payload(target_id).get("tags") or [])
    assert "to_be_removed" not in payload_tags_before, (
        "the add_tags setup write must not have leaked into the live graph "
        "payload -- otherwise this test cannot isolate remove_tags"
    )

    changed = store.remove_tags(target_id, ["to_be_removed"])
    assert changed, "the SQL tag removal must have actually landed"

    assert graph._pool_content_version == version_before, (
        "remove_tags is bare SQL and must never bump _pool_content_version"
    )
    assert graph.get_payload(target_id).get("tags") == payload_tags_before, (
        "remove_tags must never write into graph._node_payload"
    )

    _call(store, graph, assignment, rich_club, cue_vec)
    version_after, _base_after = graph._records_view_cache
    assert version_after == version_before, "cache must still be a HIT"


def test_staleness_entity_anchor_backfill_surface_is_orthogonal(tmp_path):
    """backfill_entity_anchors (migrate/_entity_backfill.py) receives only
    `store`, never `graph` -- it structurally cannot call a graph mutator.
    It routes exclusively through store.add_tags/store.remove_tags/
    store.set_aaak_index (bare SQL, no hook) and, on a real write, drops
    the on-disk runtime_graph_cache.json rather than touching
    _pool_content_version -- it runs without a live daemon graph in the
    common case."""
    from iai_mcp.migrate._entity_backfill import backfill_entity_anchors

    store = MemoryStore(path=str(tmp_path / "store"))
    cue_vec = _unit_vec(30)
    _populate(store, n=10)
    anchor_rec = _make(text="investigate `alice_project_id_9001` staleness", vec=cue_vec)
    store.insert(anchor_rec)
    graph, assignment, rich_club = build_runtime_graph(store)

    version_before, _base_before = _prove_records_cache_warm_hit(
        store, graph, assignment, rich_club, cue_vec
    )

    result = backfill_entity_anchors(store, apply=True, store_path=tmp_path)
    assert result["records_written"] or result["aaak_refreshed"], (
        "the backfill must have actually written something for this test "
        "to be meaningful"
    )

    assert graph._pool_content_version == version_before, (
        "backfill_entity_anchors never receives a graph reference -- it "
        "structurally cannot bump _pool_content_version"
    )

    _call(store, graph, assignment, rich_club, cue_vec)
    version_after, _base_after = graph._records_view_cache
    assert version_after == version_before, "cache must still be a HIT"
