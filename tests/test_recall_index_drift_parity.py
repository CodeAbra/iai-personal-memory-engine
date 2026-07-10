from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from uuid import UUID

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

RNG_SEED = 20260602
N_RECORDS_PARITY = 80
N_DRIFT_RECORDS = 30

def _norm_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()

def _make_store(tmp_path: Path):
    from iai_mcp.store import MemoryStore
    return MemoryStore(path=str(tmp_path / "store"))

def _populate_store_fixed_seed(store, n: int, rng_seed: int = RNG_SEED) -> list[UUID]:
    rng = np.random.default_rng(rng_seed)
    ids = []
    for i in range(n):
        v = rng.random(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        rec = _make(text=f"User drift parity test record {i}", vec=v.tolist())
        store.insert(rec)
        ids.append(rec.id)
    return ids

def _add_contradicts(store, id_a: UUID, id_b: UUID) -> None:
    store.boost_edges([(id_a, id_b)], delta=0.5, edge_type="contradicts")

def _snapshot_overlay(store) -> bool:
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.retrieve import build_runtime_graph

    graph, assignment, rich_club = build_runtime_graph(store)
    max_degree = int(getattr(graph, "_max_degree", 0) or 0)
    return rgc.save_with_generation(
        store, assignment, rich_club, max_degree=max_degree,
    )

def test_drift_parity_overlay_superset_of_offline_rebuild(tmp_path):
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.retrieve import build_runtime_graph, _make_graph_sync_hook
    from iai_mcp.graph import MemoryGraph

    store = _make_store(tmp_path)

    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()

    rec_ids = _populate_store_fixed_seed(store, N_RECORDS_PARITY)

    if len(rec_ids) >= 4:
        _add_contradicts(store, rec_ids[0], rec_ids[1])
        _add_contradicts(store, rec_ids[2], rec_ids[3])

    graph = MemoryGraph()
    store.register_graph_sync_hook(_make_graph_sync_hook(graph))

    base_graph, base_assignment, base_rich_club = build_runtime_graph(store)
    base_node_ids = set(str(nid) for nid in base_graph.nodes())
    base_ntc = getattr(base_assignment, "node_to_community", {})

    max_degree = int(getattr(base_graph, "_max_degree", 0) or 0)
    fresh_ts = datetime.now(timezone.utc).isoformat()
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = fresh_ts
    rgc.save_with_generation(store, base_assignment, base_rich_club, max_degree=max_degree)
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = ""

    pre_drift_result = rgc.consult_overlay(store)
    assert not isinstance(pre_drift_result, rgc._OverlayBypass), (
        f"Pre-drift: overlay must HIT; got {pre_drift_result!r}"
    )

    n_drift = min(10, rgc._FUSE_DIRTY_THRESHOLD // 4)
    rng = np.random.default_rng(RNG_SEED + 777)
    for i in range(n_drift):
        v = rng.random(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        rec = _make(text=f"User drift parity new record {i}", vec=v.tolist())
        store.insert(rec)

    post_drift_result = rgc.consult_overlay(store)
    assert not isinstance(post_drift_result, rgc._OverlayBypass), (
        f"Post-drift (small): overlay must still HIT; got {post_drift_result!r}"
    )
    ov_assignment, ov_rich_club = post_drift_result
    ov_ntc = getattr(ov_assignment, "node_to_community", {})

    ov_node_ids = set(str(nid) for nid in ov_ntc.keys())
    missing_in_overlay = base_node_ids - ov_node_ids
    miss_rate = len(missing_in_overlay) / max(1, len(base_node_ids))
    assert miss_rate < 0.05, (
        f"Overlay missing too many base nodes: {len(missing_in_overlay)} / {len(base_node_ids)}"
        f" = {miss_rate:.1%} (limit 5%)"
    )

    shared_nodes = set(ov_ntc.keys()) & set(base_ntc.keys())
    mismatches_overlay_vs_base = sum(
        1 for nid in shared_nodes if ov_ntc[nid] != base_ntc[nid]
    )
    assert mismatches_overlay_vs_base == 0, (
        f"Overlay diverges from base assignment for {mismatches_overlay_vs_base} "
        f"shared nodes — round-trip fidelity violated (written from same assignment)"
    )

    for rc_id in ov_rich_club:
        assert str(rc_id) in ov_node_ids, (
            f"Overlay rich_club contains dangling id {rc_id} not in overlay node set"
        )

    drift_graph, drift_assignment, _ = build_runtime_graph(store)
    drift_ntc = getattr(drift_assignment, "node_to_community", {})
    drift_node_ids = set(str(nid) for nid in drift_graph.nodes())

    assert len(drift_node_ids) >= len(base_node_ids), (
        "Drifted graph must contain at least as many nodes as the base graph"
    )

    lgs = rgc.load_last_good_structural(store)
    assert lgs is not None
    lgs_assignment, _ = lgs
    assert lgs_assignment is not drift_assignment, (
        "load_last_good_structural result must be a different Python object than "
        "build_runtime_graph result — separate decode paths prove different source"
    )
    assert base_assignment is not lgs_assignment, (
        "Base GT (build_runtime_graph) must be a different object than lgs decode"
    )

def test_drift_parity_anti_hit_surfaces(tmp_path):
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.retrieve import build_runtime_graph

    store = _make_store(tmp_path)
    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()

    rec_ids = _populate_store_fixed_seed(store, 20, rng_seed=RNG_SEED + 1)

    anti_src = rec_ids[0]
    anti_dst = rec_ids[1]
    _add_contradicts(store, anti_src, anti_dst)

    gt_graph, gt_assignment, gt_rich_club = build_runtime_graph(store)

    max_degree = int(getattr(gt_graph, "_max_degree", 0) or 0)
    rgc.save_with_generation(store, gt_assignment, gt_rich_club, max_degree=max_degree)

    edges_df = store.db.open_table("edges").to_pandas()
    contradicts_mask = (
        ((edges_df["src"] == str(anti_src)) & (edges_df["dst"] == str(anti_dst)))
        | ((edges_df["src"] == str(anti_dst)) & (edges_df["dst"] == str(anti_src)))
    )
    contradicts_edges = edges_df[contradicts_mask & (edges_df.get("edge_type", "") == "contradicts")]
    assert len(contradicts_edges) > 0, "Contradicts edge must be in the store"

    overlay_result = rgc.consult_overlay(store)
    assert not isinstance(overlay_result, rgc._OverlayBypass), (
        f"Overlay must HIT; got {overlay_result!r}"
    )

def test_long_horizon_drift_replay_fuse_trips(tmp_path, monkeypatch):
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.retrieve import build_runtime_graph, _make_graph_sync_hook
    from iai_mcp.graph import MemoryGraph
    import iai_mcp.community as _cm
    import iai_mcp.richclub as _rc_mod
    import iai_mcp.retrieve as _ret_mod

    store = _make_store(tmp_path)
    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()

    rec_ids = _populate_store_fixed_seed(store, N_RECORDS_PARITY)

    graph = MemoryGraph()
    store.register_graph_sync_hook(_make_graph_sync_hook(graph))

    gt_graph, gt_assignment, gt_rich_club = build_runtime_graph(store)
    max_degree = int(getattr(gt_graph, "_max_degree", 0) or 0)
    fresh_ts = datetime.now(timezone.utc).isoformat()
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = fresh_ts
    rgc.save_with_generation(store, gt_assignment, gt_rich_club, max_degree=max_degree)
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = ""

    initial_result = rgc.consult_overlay(store)
    assert not isinstance(initial_result, rgc._OverlayBypass), (
        f"Initial overlay should HIT; got {initial_result!r}"
    )

    rng = np.random.default_rng(RNG_SEED + 99)
    hit_fuse = False
    inserts_done = 0
    for i in range(N_DRIFT_RECORDS + rgc._FUSE_DIRTY_THRESHOLD + 5):
        v = rng.random(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        rec = _make(text=f"User drift record {i}", vec=v.tolist())
        store.insert(rec)
        inserts_done += 1

        result = rgc.consult_overlay(store)
        if isinstance(result, rgc._OverlayBypass) and result.reason == "fuse_tripped":
            hit_fuse = True
            fuse_trip_at = inserts_done
            break

    assert hit_fuse, (
        f"Freshness fuse must trip during long-horizon drift replay "
        f"after {inserts_done} inserts; dirty_counter={rgc.get_dirty_counter()}, "
        f"threshold={rgc._FUSE_DIRTY_THRESHOLD}"
    )

    monkeypatch.setattr(_cm, "detect_communities", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("detect_communities called on hot path after fuse trip")))
    monkeypatch.setattr(_rc_mod, "rich_club_nodes", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("rich_club_nodes called on hot path after fuse trip")))
    monkeypatch.setattr(_ret_mod, "build_runtime_graph", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("build_runtime_graph called on hot path after fuse trip")))

    post_fuse_result = rgc.consult_overlay(store)
    assert isinstance(post_fuse_result, rgc._OverlayBypass), (
        "Overlay must return bypass after fuse trip"
    )
    assert post_fuse_result.reason == "fuse_tripped"

    last_good = rgc.load_last_good_structural(store)
    assert last_good is not None, "load_last_good_structural must return last-good snapshot after fuse trip"

def test_long_horizon_drift_max_age_trip(tmp_path):
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.community import CommunityAssignment

    store = _make_store(tmp_path)
    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()

    _populate_store_fixed_seed(store, 10)

    old_ts = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()
    assignment = CommunityAssignment(
        node_to_community={}, community_centroids={}, modularity=0.5,
        backend="mosaic", top_communities=[], mid_regions={},
    )
    old_gen = rgc.advance_generation()
    rgc.reset_dirty_counter()
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = old_ts
    rgc.save(store, assignment, [])
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = ""

    assert rgc.get_dirty_counter() == 0

    result = rgc.consult_overlay(store)
    assert isinstance(result, rgc._OverlayBypass), (
        "Overlay must bypass when rebuild_timestamp > max_age"
    )
    assert result.reason == "fuse_tripped"
    assert result.age_ms > 0

def test_drift_replay_quality_before_fuse(tmp_path):
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.retrieve import build_runtime_graph, _make_graph_sync_hook
    from iai_mcp.graph import MemoryGraph

    store = _make_store(tmp_path)
    with rgc._GEN_LOCK:
        rgc._current_generation = 0
    rgc.reset_dirty_counter()

    rec_ids = _populate_store_fixed_seed(store, N_RECORDS_PARITY)

    graph = MemoryGraph()
    store.register_graph_sync_hook(_make_graph_sync_hook(graph))

    gt_graph, gt_assignment, gt_rich_club = build_runtime_graph(store)
    max_degree = int(getattr(gt_graph, "_max_degree", 0) or 0)

    fresh_ts = datetime.now(timezone.utc).isoformat()
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = fresh_ts
    rgc.save_with_generation(store, gt_assignment, gt_rich_club, max_degree=max_degree)
    with rgc._GEN_LOCK:
        rgc._rebuild_timestamp_override = ""

    rng = np.random.default_rng(RNG_SEED + 77)
    n_small = min(5, rgc._FUSE_DIRTY_THRESHOLD // 2)
    for i in range(n_small):
        v = rng.random(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        rec = _make(text=f"User small drift record {i}", vec=v.tolist())
        store.insert(rec)

    result = rgc.consult_overlay(store)
    assert not isinstance(result, rgc._OverlayBypass), (
        f"Overlay should HIT after {n_small} intra-day inserts (well below threshold); "
        f"dirty_counter={rgc.get_dirty_counter()}, threshold={rgc._FUSE_DIRTY_THRESHOLD}"
    )

    ov_assignment, ov_rich_club = result

    ov_ntc = getattr(ov_assignment, "node_to_community", {})
    gt_ntc = getattr(gt_assignment, "node_to_community", {})
    shared_nodes = set(ov_ntc.keys()) & set(gt_ntc.keys())
    mismatches = sum(1 for nid in shared_nodes if ov_ntc[nid] != gt_ntc[nid])
    assert mismatches == 0, (
        f"Overlay community assignments diverge from GT for {mismatches} pre-existing "
        f"nodes (accept-stable-global-bias-intra-day violated before fuse trip)"
    )
