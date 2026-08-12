from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make

from iai_mcp.types import EMBED_DIM

@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    yield

def _norm_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()

def _make_store(tmp_path: Path):
    from iai_mcp.store import MemoryStore
    return MemoryStore(path=str(tmp_path / "store"))

def _build_overlay_snapshot(store, *, generation: int, rebuild_ts: str | None = None) -> bool:
    from iai_mcp.community import CommunityAssignment
    from iai_mcp import runtime_graph_cache as rgc

    with rgc._GEN_LOCK:
        rgc._current_generation = generation - 1
    if rebuild_ts is not None:
        with rgc._GEN_LOCK:
            rgc._rebuild_timestamp_override = rebuild_ts

    assignment = CommunityAssignment(
        node_to_community={},
        community_centroids={},
        modularity=0.5,
        backend="mosaic",
        top_communities=[],
        mid_regions={},
    )
    result = rgc.save_with_generation(store, assignment, [])
    if rebuild_ts is not None:
        pass
    return result

def test_cache_version_bumped():
    from iai_mcp import runtime_graph_cache as rgc
    assert rgc.CACHE_VERSION != "62-04-v4", (
        f"CACHE_VERSION must be bumped from 62-04-v4; got {rgc.CACHE_VERSION!r}"
    )
    assert rgc.CACHE_VERSION == "62-02-v8"

def test_overlay_hit_serves_cached_o1(tmp_path, monkeypatch):
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.community import CommunityAssignment

    store = _make_store(tmp_path)

    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()

    fresh_ts = datetime.now(timezone.utc).isoformat()
    node_id = uuid4()
    comm_id = uuid4()
    assignment = CommunityAssignment(
        node_to_community={node_id: comm_id},
        community_centroids={comm_id: _norm_vec(1)},
        modularity=0.5,
        backend="mosaic",
        top_communities=[comm_id],
        mid_regions={},
    )

    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = fresh_ts
    new_gen = rgc.advance_generation()
    rgc.save(store, assignment, [node_id], max_degree=5)
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = ""

    assert new_gen == 1

    detect_called = []
    richclub_called = []
    build_called = []

    import iai_mcp.community as _community_mod
    import iai_mcp.richclub as _richclub_mod
    import iai_mcp.retrieve as _retrieve_mod

    orig_detect = _community_mod.detect_communities
    orig_rc = _richclub_mod.rich_club_nodes
    orig_build = _retrieve_mod.build_runtime_graph

    def _mock_detect(*a, **kw):
        detect_called.append(True)
        return orig_detect(*a, **kw)

    def _mock_rc(*a, **kw):
        richclub_called.append(True)
        return orig_rc(*a, **kw)

    def _mock_build(*a, **kw):
        build_called.append(True)
        return orig_build(*a, **kw)

    monkeypatch.setattr(_community_mod, "detect_communities", _mock_detect)
    monkeypatch.setattr(_richclub_mod, "rich_club_nodes", _mock_rc)
    monkeypatch.setattr(_retrieve_mod, "build_runtime_graph", _mock_build)

    result = rgc.consult_overlay(store)

    assert not isinstance(result, rgc._OverlayBypass), (
        f"Expected overlay HIT, got bypass: {result!r}"
    )
    assert isinstance(result, tuple) and len(result) == 2
    assert not detect_called, "detect_communities was called — overlay should serve O(1)"
    assert not richclub_called, "rich_club_nodes was called — overlay should serve O(1)"
    assert not build_called, "build_runtime_graph was called — overlay should serve O(1)"

def test_epoch_mismatch_typed_bypass(tmp_path, monkeypatch):
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.community import CommunityAssignment

    store = _make_store(tmp_path)

    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()
    fresh_ts = datetime.now(timezone.utc).isoformat()
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = fresh_ts
    rgc.advance_generation()
    assignment = CommunityAssignment(
        node_to_community={}, community_centroids={}, modularity=0.5,
        backend="mosaic", top_communities=[], mid_regions={},
    )
    rgc.save(store, assignment, [])
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = ""

    rgc.advance_generation()
    assert rgc.get_current_generation() == 2

    import iai_mcp.community as _cm
    import iai_mcp.richclub as _rc_mod
    import iai_mcp.retrieve as _ret_mod

    monkeypatch.setattr(_cm, "detect_communities", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("detect_communities called on hot path")))
    monkeypatch.setattr(_rc_mod, "rich_club_nodes", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("rich_club_nodes called on hot path")))
    monkeypatch.setattr(_ret_mod, "build_runtime_graph", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("build_runtime_graph called on hot path")))

    result = rgc.consult_overlay(store)
    assert isinstance(result, rgc._OverlayBypass)
    assert result.reason == "epoch_mismatch"

    last_good = rgc.load_last_good_structural(store)
    assert last_good is not None, "load_last_good_structural must return the last-good snapshot on bypass"
    a2, rc2 = last_good
    assert a2 is not None

def test_invariant_failure_bypass(tmp_path, monkeypatch):
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.community import CommunityAssignment

    store = _make_store(tmp_path)

    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()
    fresh_ts = datetime.now(timezone.utc).isoformat()
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = fresh_ts
    rgc.advance_generation()
    node_id = uuid4()
    comm_id = uuid4()
    dangling_rc_id = uuid4()
    assignment = CommunityAssignment(
        node_to_community={node_id: comm_id},
        community_centroids={comm_id: _norm_vec(42)},
        modularity=0.5,
        backend="mosaic",
        top_communities=[comm_id],
        mid_regions={},
    )
    rgc.save(store, assignment, [dangling_rc_id])
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = ""

    import iai_mcp.community as _cm
    import iai_mcp.richclub as _rc_mod
    import iai_mcp.retrieve as _ret_mod

    monkeypatch.setattr(_cm, "detect_communities", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("detect_communities called on hot path")))
    monkeypatch.setattr(_rc_mod, "rich_club_nodes", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("rich_club_nodes called on hot path")))
    monkeypatch.setattr(_ret_mod, "build_runtime_graph", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("build_runtime_graph called on hot path")))

    result = rgc.consult_overlay(store)
    assert isinstance(result, rgc._OverlayBypass)
    assert result.reason == "invariant_failure"

def test_load_recall_structural_uses_overlay_hit(tmp_path, monkeypatch):
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.community import CommunityAssignment

    store = _make_store(tmp_path)

    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()
    fresh_ts = datetime.now(timezone.utc).isoformat()
    node_id = uuid4()
    comm_id = uuid4()
    assignment = CommunityAssignment(
        node_to_community={node_id: comm_id},
        community_centroids={comm_id: _norm_vec(5)},
        modularity=0.5,
        backend="mosaic",
        top_communities=[comm_id],
        mid_regions={},
    )
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = fresh_ts
    rgc.advance_generation()
    rgc.save(store, assignment, [node_id])
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = ""

    a, rc, md, source, _node_degrees = rgc.load_recall_structural(store)
    assert source == "overlay", (
        f"load_recall_structural must route via overlay HIT; got source={source!r}"
    )
    assert a is not None
    assert len(getattr(a, "node_to_community", {})) > 0

def test_load_recall_structural_epoch_mismatch_falls_to_last_good(tmp_path, monkeypatch):
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.community import CommunityAssignment
    import iai_mcp.community as _cm
    import iai_mcp.richclub as _rc_mod
    import iai_mcp.retrieve as _ret_mod

    store = _make_store(tmp_path)

    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()
    fresh_ts = datetime.now(timezone.utc).isoformat()
    node_id = uuid4()
    comm_id = uuid4()
    assignment = CommunityAssignment(
        node_to_community={node_id: comm_id},
        community_centroids={comm_id: _norm_vec(88)},
        modularity=0.5,
        backend="mosaic",
        top_communities=[comm_id],
        mid_regions={},
    )
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = fresh_ts
    rgc.advance_generation()
    rgc.save(store, assignment, [node_id])
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = ""

    rgc.advance_generation()

    monkeypatch.setattr(_cm, "detect_communities", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("detect_communities on hot path")))
    monkeypatch.setattr(_rc_mod, "rich_club_nodes", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("rich_club_nodes on hot path")))
    monkeypatch.setattr(_ret_mod, "build_runtime_graph", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("build_runtime_graph on hot path")))

    a, rc, md, source, _node_degrees = rgc.load_recall_structural(store)
    assert source in ("last_good", "normal"), (
        f"On overlay bypass, must return last_good or normal (Layer-1); got {source!r}"
    )
    ov_ntc = getattr(a, "node_to_community", {})
    assert len(ov_ntc) > 0, (
        "Overlay bypass must return non-empty structural data (last-good), not cold_degrade"
    )

def test_recall_path_boost_does_not_bump_epoch(tmp_path, monkeypatch):
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.community import CommunityAssignment

    store = _make_store(tmp_path)

    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()
    fresh_ts = datetime.now(timezone.utc).isoformat()
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = fresh_ts
    rgc.advance_generation()
    node_id_a = uuid4()
    node_id_b = uuid4()
    comm_id = uuid4()
    assignment = CommunityAssignment(
        node_to_community={node_id_a: comm_id, node_id_b: comm_id},
        community_centroids={comm_id: _norm_vec(7)},
        modularity=0.5,
        backend="mosaic",
        top_communities=[comm_id],
        mid_regions={},
    )
    rgc.save(store, assignment, [node_id_a, node_id_b])
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = ""
    gen_before = rgc.get_current_generation()
    dirty_before = rgc.get_dirty_counter()

    store.boost_edges([(node_id_a, node_id_b)], delta=0.1)

    assert rgc.get_current_generation() == gen_before, (
        "boost_edges must NOT advance the generation epoch"
    )
    assert rgc.get_dirty_counter() == dirty_before, (
        "boost_edges must NOT increment the dirty counter (RECORD-only hook)"
    )

    result = rgc.consult_overlay(store)
    assert not isinstance(result, rgc._OverlayBypass), (
        f"Overlay should still HIT after recall-path boost; got bypass {result!r}"
    )

def test_intra_day_insert_keeps_overlay_hit(tmp_path, monkeypatch):
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.community import CommunityAssignment
    from iai_mcp.retrieve import _make_graph_sync_hook
    from iai_mcp.graph import MemoryGraph

    store = _make_store(tmp_path)

    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()
    fresh_ts = datetime.now(timezone.utc).isoformat()
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = fresh_ts
    rgc.advance_generation()
    assignment = CommunityAssignment(
        node_to_community={}, community_centroids={}, modularity=0.5,
        backend="mosaic", top_communities=[], mid_regions={},
    )
    rgc.save(store, assignment, [])
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = ""
    dirty_before = rgc.get_dirty_counter()

    graph = MemoryGraph()
    store.register_graph_sync_hook(_make_graph_sync_hook(graph))

    rec = _make(text="User test record for dirty counter increment")
    store.insert(rec)

    dirty_after = rgc.get_dirty_counter()
    assert dirty_after == dirty_before + 1, (
        f"Expected dirty counter to increment by 1; was {dirty_before}, now {dirty_after}"
    )

    result = rgc.consult_overlay(store)
    assert not isinstance(result, rgc._OverlayBypass), (
        f"Overlay should still HIT after single insert; got bypass {result!r}"
    )

def test_freshness_fuse_trips_on_max_age(tmp_path, monkeypatch):
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.community import CommunityAssignment
    import iai_mcp.community as _cm
    import iai_mcp.richclub as _rc_mod
    import iai_mcp.retrieve as _ret_mod

    store = _make_store(tmp_path)

    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()

    old_ts = (datetime.now(timezone.utc) - timedelta(hours=27)).isoformat()
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = old_ts
    rgc.advance_generation()
    assignment = CommunityAssignment(
        node_to_community={}, community_centroids={}, modularity=0.5,
        backend="mosaic", top_communities=[], mid_regions={},
    )
    rgc.save(store, assignment, [])
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = ""

    orig_arc = getattr(store, "active_records_count", None)
    def _raise_arc(*a, **kw):
        raise AssertionError("active_records_count called on overlay consult hot path")
    monkeypatch.setattr(store, "active_records_count", _raise_arc)

    monkeypatch.setattr(_cm, "detect_communities", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("detect_communities on hot path")))
    monkeypatch.setattr(_rc_mod, "rich_club_nodes", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("rich_club_nodes on hot path")))
    monkeypatch.setattr(_ret_mod, "build_runtime_graph", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("build_runtime_graph on hot path")))

    result = rgc.consult_overlay(store)
    assert isinstance(result, rgc._OverlayBypass), (
        "Expected fuse-trip bypass due to old rebuild_timestamp"
    )
    assert result.reason == "fuse_tripped"
    assert result.age_ms > 0

    last_good = rgc.load_last_good_structural(store)
    assert last_good is not None

def test_freshness_fuse_trips_on_dirty_counter(tmp_path, monkeypatch):
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.community import CommunityAssignment
    import iai_mcp.community as _cm
    import iai_mcp.richclub as _rc_mod
    import iai_mcp.retrieve as _ret_mod

    store = _make_store(tmp_path)

    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()

    fresh_ts = datetime.now(timezone.utc).isoformat()
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = fresh_ts
    rgc.advance_generation()
    assignment = CommunityAssignment(
        node_to_community={}, community_centroids={}, modularity=0.5,
        backend="mosaic", top_communities=[], mid_regions={},
    )
    rgc.save(store, assignment, [])
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = ""

    from iai_mcp import runtime_graph_cache as _rgc
    with _rgc._DIRTY_COUNTER_LOCK:
        _rgc._dirty_counter = _rgc._FUSE_DIRTY_THRESHOLD + 1

    monkeypatch.setattr(_cm, "detect_communities", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("detect_communities on hot path")))
    monkeypatch.setattr(_rc_mod, "rich_club_nodes", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("rich_club_nodes on hot path")))
    monkeypatch.setattr(_ret_mod, "build_runtime_graph", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("build_runtime_graph on hot path")))

    def _raise_arc(*a, **kw):
        raise AssertionError("active_records_count called on hot path")
    monkeypatch.setattr(store, "active_records_count", _raise_arc)

    result = rgc.consult_overlay(store)
    assert isinstance(result, rgc._OverlayBypass)
    assert result.reason == "fuse_tripped"

def test_within_threshold_still_hits(tmp_path):
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.community import CommunityAssignment

    store = _make_store(tmp_path)
    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()
    fresh_ts = datetime.now(timezone.utc).isoformat()
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = fresh_ts
    rgc.advance_generation()
    assignment = CommunityAssignment(
        node_to_community={}, community_centroids={}, modularity=0.5,
        backend="mosaic", top_communities=[], mid_regions={},
    )
    rgc.save(store, assignment, [])
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = ""

    result = rgc.consult_overlay(store)
    assert not isinstance(result, rgc._OverlayBypass), (
        f"Should HIT; got bypass {result!r}"
    )

def test_dirty_counter_composed_hook_fires_on_record_insert(tmp_path):
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.retrieve import _make_graph_sync_hook
    from iai_mcp.graph import MemoryGraph

    store = _make_store(tmp_path)
    rgc.reset_dirty_counter()

    graph = MemoryGraph()
    mirror_fired = []

    store.register_graph_sync_hook(_make_graph_sync_hook(graph))
    dirty_before = rgc.get_dirty_counter()

    rec = _make(text="User hook test record", vec=_norm_vec(99))
    store.insert(rec)

    dirty_after = rgc.get_dirty_counter()
    assert dirty_after == dirty_before + 1, (
        f"Dirty counter should have incremented; was {dirty_before}, now {dirty_after}"
    )
    assert str(rec.id) in graph._node_payload, (
        "Node-mirror hook was clobbered — graph._node_payload is missing the inserted record"
    )

def test_boost_edges_does_not_increment_dirty_counter(tmp_path):
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.retrieve import _make_graph_sync_hook
    from iai_mcp.graph import MemoryGraph

    store = _make_store(tmp_path)
    rgc.reset_dirty_counter()

    graph = MemoryGraph()
    store.register_graph_sync_hook(_make_graph_sync_hook(graph))

    src_id = uuid4()
    dst_id = uuid4()
    dirty_before = rgc.get_dirty_counter()
    store.boost_edges([(src_id, dst_id)], delta=0.2)
    dirty_after = rgc.get_dirty_counter()
    assert dirty_after == dirty_before, (
        f"boost_edges must NOT increment dirty counter; was {dirty_before}, now {dirty_after}"
    )

def test_nightly_rebuild_resets_dirty_counter(tmp_path):
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.community import CommunityAssignment

    store = _make_store(tmp_path)

    with rgc._DIRTY_COUNTER_LOCK:
        rgc._dirty_counter = 30
    assert rgc.get_dirty_counter() == 30

    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    assignment = CommunityAssignment(
        node_to_community={}, community_centroids={}, modularity=0.5,
        backend="mosaic", top_communities=[], mid_regions={},
    )
    rgc.save_with_generation(store, assignment, [])

    assert rgc.get_dirty_counter() == 0, (
        "save_with_generation must reset the dirty counter to 0"
    )

def test_recall_index_rebuild_step_position():
    from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep, SleepPipeline

    step_order = SleepPipeline._STEP_ORDER
    assert SleepStep.RECALL_INDEX_REBUILD in step_order, (
        "RECALL_INDEX_REBUILD must be in _STEP_ORDER"
    )
    idx_rebuild = step_order.index(SleepStep.RECALL_INDEX_REBUILD)
    idx_cluster = step_order.index(SleepStep.CLUSTER_SUMMARY)
    assert idx_rebuild > idx_cluster, (
        f"RECALL_INDEX_REBUILD (index={idx_rebuild}) must be AFTER "
        f"CLUSTER_SUMMARY (index={idx_cluster}) in _STEP_ORDER"
    )
    # The rebuild must follow every RECORD-mutating step: the vector index
    # must contain the summaries minted this cycle. A trailing step may either
    # touch only edges (never the vector index) or feed the vector index
    # INCREMENTALLY per row (needing no full rebuild); new steps append at the
    # tail so WAL-recovery positions stay frozen.
    may_trail = {
        SleepStep.ENTITY_LINK,
        SleepStep.CURIOSITY_MINE,
        SleepStep.EMBEDDING_INTEGRITY,
    }
    trailing = set(step_order[idx_rebuild + 1:])
    assert trailing <= may_trail, (
        f"steps after RECALL_INDEX_REBUILD must be edges-only or "
        f"incremental-index feeders, got {trailing}"
    )

def test_recall_index_rebuild_step_stamps_fresh_generation(tmp_path, monkeypatch):
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline

    store = _make_store(tmp_path)
    store.insert(_make(text="User nightly rebuild test record 1", vec=_norm_vec(11)))
    store.insert(_make(text="User nightly rebuild test record 2", vec=_norm_vec(12)))

    with rgc._GEN_LOCK:
        rgc._current_generation = 5
    rgc.reset_dirty_counter()

    pipeline = SleepPipeline(store)
    done, payload = pipeline._step_recall_index_rebuild(None)

    assert done is True, "RECALL_INDEX_REBUILD step must return done=True"
    assert payload.get("rebuilt") is True, (
        f"Step must report rebuilt=True; got {payload}"
    )
    new_gen = payload.get("generation")
    assert new_gen == 6, (
        f"Generation must be incremented from 5 to 6; got {new_gen}"
    )
    assert rgc.get_dirty_counter() == 0, (
        "Nightly rebuild must reset dirty counter to 0"
    )
    snap = rgc.load_last_good_structural(store)
    assert snap is not None, "Snapshot must be readable after nightly rebuild"


def test_recall_index_rebuild_prewarms_read_model(tmp_path, monkeypatch):
    """The rebuild step pays the post-rewrite decode/build itself, so the
    first recall after a cycle is a pure memo hit instead of a cold decode +
    full-corpus graph stream + cold exact-authority lane."""
    from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline

    store = _make_store(tmp_path)
    store.insert(_make(text="User prewarm record one", vec=_norm_vec(21)))
    store.insert(_make(text="User prewarm record two", vec=_norm_vec(22)))

    # Force-cold everything the step must leave warm.
    store._decoded_structural_memo = None
    store._decoded_degrees_memo = None
    store._warm_graph_bundle = None
    store._exact_index.invalidate()

    pipeline = SleepPipeline(store)
    done, payload = pipeline._step_recall_index_rebuild(None)
    assert done is True

    for err_key in (
        "structural_prewarm_error",
        "bundle_prewarm_error",
        "exact_prewarm_error",
    ):
        assert err_key not in payload, payload[err_key]
    assert "structural_prewarm_elapsed_sec" in payload, payload
    assert "bundle_prewarm_elapsed_sec" in payload, payload
    assert "exact_prewarm_elapsed_sec" in payload, payload

    assert any(
        getattr(store, memo, None) is not None
        for memo in (
            "_decoded_structural_memo",
            "_decoded_overlay_pair_memo",
            "_decoded_degrees_memo",
        )
    ), "structural prewarm must leave a decoded memo behind"
    assert getattr(store, "_warm_graph_bundle", None) is not None, (
        "bundle prewarm must leave the warm graph bundle installed"
    )
    assert store._exact_index.is_warm, (
        "exact prewarm must rebuild the invalidated authority matrix"
    )


def test_recall_index_rebuild_prewarm_yields_to_foreground(tmp_path, monkeypatch):
    """Prewarm is pure acceleration: with a live read waiting, every phase
    must skip rather than hold the interpreter."""
    from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline

    store = _make_store(tmp_path)
    store.insert(_make(text="User prewarm yield record", vec=_norm_vec(31)))

    store._decoded_structural_memo = None
    store._warm_graph_bundle = None
    store._exact_index.invalidate()

    pipeline = SleepPipeline(store)
    # Interrupt fires only for chunk_idx > 0 style checks inside prewarm; the
    # step-entry check (chunk 0) must still let the rebuild itself run.
    calls = {"n": 0}

    def _foreground_after_step_entry() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    done, payload = pipeline._step_recall_index_rebuild(_foreground_after_step_entry)
    assert done is True
    assert payload.get("rebuilt") is True, payload
    for phase in ("structural", "bundle", "exact"):
        assert payload.get(f"{phase}_prewarm_skipped") == "foreground", (
            f"{phase} prewarm must yield to a waiting foreground read: {payload}"
        )
    assert store._warm_graph_bundle is None, (
        "a skipped bundle prewarm must not have built the bundle"
    )
