"""Parity + parent-region isolation for the community-detection child worker.

Two guarantees are pinned here:

  1. The community partition (which nodes share a community) returned by the
     spawn-context child equals the in-process `detect_communities` partition for
     the same graph, for both the seeded mode (topology snapshot) and the cold
     mode (crisis reclustering). Equality is frozenset-of-frozensets, NOT raw
     community-UUID equality: the flat fallback mints a fresh uuid4 per
     community, so two runs over the same graph yield the SAME grouping under
     DIFFERENT community UUIDs.

  2. Running the child helper repeatedly leaves the parent process's address
     space flat — the numba/llvmlite JIT arenas now live in the ephemeral child
     and the OS reclaims them on exit. Measured on macOS via the parent's
     VM_ALLOCATE region count; skipped on other platforms.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.community import detect_communities
from iai_mcp.graph import MemoryGraph
from iai_mcp.store import MemoryStore

_DIM = 384


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


def _partition(node_to_community: dict) -> frozenset:
    """Frozenset-of-frozensets grouping of node ids by their community.

    Invariant under community-UUID relabeling — this is the equality the
    migration must preserve, not raw UUID identity.
    """
    groups: dict = {}
    for node_id, community_id in node_to_community.items():
        groups.setdefault(community_id, set()).add(str(node_id))
    return frozenset(frozenset(members) for members in groups.values())


def _seed_planted_partition(
    store: MemoryStore,
    n_total: int,
    K: int,
    n_intra_per_block: int,
    n_inter: int,
    seed: int,
) -> list[UUID]:
    """Active-only fixture with a clear planted-partition structure.

    K dense blocks of `n_total // K` records each, with `n_intra_per_block`
    intra-block edges per block plus `n_inter` inter-block edges. The
    modularity is high enough that the mosaic Leiden path runs (the backend is
    `leiden-custom`, not the flat fallback), so the parity guard exercises the
    real community-detection path rather than the degenerate one.
    """
    rng = np.random.default_rng(seed)
    db = store.db
    ids: list[UUID] = []
    now_iso = datetime.now(timezone.utc).isoformat()
    block_size = n_total // K
    with db._conn_lock:
        cur = db._conn.cursor()
        rows = []
        for _ in range(n_total):
            rid = uuid4()
            ids.append(rid)
            emb = rng.standard_normal(_DIM).astype(np.float32).tobytes()
            rows.append((str(rid), "episodic", "", "", emb, now_iso))
        cur.executemany(
            "INSERT INTO records (id, tier, literal_surface, aaak_index,"
            " embedding, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        seen: set[tuple[str, str]] = set()
        e_rows: list = []
        for b in range(K):
            added = 0
            while added < n_intra_per_block:
                i = b * block_size + int(rng.integers(0, block_size))
                j = b * block_size + int(rng.integers(0, block_size))
                if i == j:
                    continue
                pair = (str(ids[i]), str(ids[j]))
                if pair in seen or (pair[1], pair[0]) in seen:
                    continue
                seen.add(pair)
                e_rows.append((pair[0], pair[1], "hebbian", 1.0, now_iso))
                added += 1
        added = 0
        while added < n_inter:
            i = int(rng.integers(0, n_total))
            j = int(rng.integers(0, n_total))
            if i == j:
                continue
            pair = (str(ids[i]), str(ids[j]))
            if pair in seen or (pair[1], pair[0]) in seen:
                continue
            seen.add(pair)
            e_rows.append((pair[0], pair[1], "hebbian", 1.0, now_iso))
            added += 1
        cur.executemany(
            "INSERT INTO edges (src, dst, edge_type, weight, updated_at)"
            " VALUES (?, ?, ?, ?, ?)",
            e_rows,
        )
    return ids


def _build_graph_from_store(store: MemoryStore) -> MemoryGraph:
    """Build the in-parent MemoryGraph the cold callers would build.

    Mirrors the unfiltered node loop + full-edges loop used by the crisis and
    topology paths so the in-process reference and the child both start from the
    identical graph.
    """
    g = MemoryGraph()
    for rec in store.all_records():
        g.add_node(rec.id, None, list(rec.embedding))
    edges_df = store.db.open_table("edges").to_pandas()
    if not edges_df.empty:
        for row in edges_df.itertuples():
            g.add_edge(
                UUID(str(row.src)),
                UUID(str(row.dst)),
                weight=float(getattr(row, "weight", 1.0) or 1.0),
            )
    return g


def test_child_partition_matches_in_process_seeded(store: MemoryStore) -> None:
    from iai_mcp.runtime_graph_cache import compute_assignment_in_child

    _seed_planted_partition(
        store, n_total=800, K=4, n_intra_per_block=5000, n_inter=200, seed=11,
    )
    graph = _build_graph_from_store(store)

    ref = detect_communities(graph, prior=None, prior_mode="seeded")
    assert ref.backend != "flat", (
        f"fixture regressed to flat backend ({ref.backend!r}); "
        f"parity guard is vacuous unless mosaic actually runs"
    )
    assert len(set(ref.node_to_community.values())) > 1, (
        "fixture must yield multiple communities to exercise the partition"
    )

    child = compute_assignment_in_child(graph, prior_mode="seeded")

    assert _partition(child.node_to_community) == _partition(ref.node_to_community)
    assert len(child.community_centroids) == len(ref.community_centroids)


def test_child_partition_matches_in_process_cold(store: MemoryStore) -> None:
    from iai_mcp.runtime_graph_cache import compute_assignment_in_child

    _seed_planted_partition(
        store, n_total=800, K=4, n_intra_per_block=5000, n_inter=200, seed=23,
    )
    graph = _build_graph_from_store(store)

    ref = detect_communities(graph, prior=None, prior_mode="cold")
    assert ref.backend != "flat", (
        f"fixture regressed to flat backend ({ref.backend!r}); "
        f"cold-mode parity guard is vacuous unless mosaic actually runs"
    )

    child = compute_assignment_in_child(graph, prior_mode="cold")

    assert _partition(child.node_to_community) == _partition(ref.node_to_community)


def test_topology_consumed_community_count_matches(store: MemoryStore) -> None:
    """The topology snapshot consumes len(community_centroids); it must be stable."""
    from iai_mcp.runtime_graph_cache import compute_assignment_in_child

    _seed_planted_partition(
        store, n_total=800, K=4, n_intra_per_block=5000, n_inter=200, seed=31,
    )
    graph = _build_graph_from_store(store)

    ref = detect_communities(graph, prior=None, prior_mode="seeded")
    child = compute_assignment_in_child(graph, prior_mode="seeded")

    assert int(len(child.community_centroids)) == int(len(ref.community_centroids))


def test_crisis_relabel_partition_matches(store: MemoryStore) -> None:
    """The crisis relabel loop maps nodes to int classes; the partition it would
    build from the child assignment must match the in-process one."""
    from iai_mcp.runtime_graph_cache import compute_assignment_in_child

    _seed_planted_partition(
        store, n_total=800, K=4, n_intra_per_block=5000, n_inter=200, seed=37,
    )
    graph = _build_graph_from_store(store)

    ref = detect_communities(graph, prior=None, prior_mode="cold")
    child = compute_assignment_in_child(graph, prior_mode="cold")

    assert _partition(child.node_to_community) == _partition(ref.node_to_community)


def test_synchronous_topology_snapshot_stays_in_process(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The request-synchronous topology snapshot must not spawn a child.

    `compute_topology_snapshot(graph)` serves the interactive health surface
    (`iai status` + the topology tool); it computes community detection
    in-process so the operator command stays near-instant.
    """
    import iai_mcp.runtime_graph_cache as rgc
    import iai_mcp.sigma as sigma

    spawned = {"n": 0}
    real = rgc.compute_assignment_in_child

    def _counting(graph, **kw):
        spawned["n"] += 1
        return real(graph, **kw)

    monkeypatch.setattr(rgc, "compute_assignment_in_child", _counting)

    _seed_planted_partition(
        store, n_total=300, K=4, n_intra_per_block=1500, n_inter=100, seed=53,
    )
    graph = _build_graph_from_store(store)

    snap = sigma.compute_topology_snapshot(graph)

    assert spawned["n"] == 0, (
        "synchronous topology snapshot spawned a child; the interactive path "
        "must compute community detection in-process"
    )
    assert snap["community_count"] >= 1


def test_background_topology_routes_to_child(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The background topology pass must route community detection to the child.

    `compute_and_emit` runs off the interactive path (hourly audit + offline
    sleep step); it computes community detection in a short-lived child so the
    daemon does not accumulate JIT arenas.
    """
    import iai_mcp.runtime_graph_cache as rgc
    import iai_mcp.sigma as sigma

    spawned = {"n": 0}
    real = rgc.compute_assignment_in_child

    def _counting(graph, **kw):
        spawned["n"] += 1
        return real(graph, **kw)

    monkeypatch.setattr(rgc, "compute_assignment_in_child", _counting)

    _seed_planted_partition(
        store, n_total=300, K=4, n_intra_per_block=1500, n_inter=100, seed=59,
    )

    snap = sigma.compute_and_emit(store)

    assert spawned["n"] == 1, (
        "background topology pass did not route community detection to the child"
    )
    assert snap["community_count"] >= 1


def _count_vm_allocate_regions(pid: int) -> int:
    out = subprocess.run(
        ["/usr/bin/vmmap", "-resident", str(pid)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return sum(
        1 for line in out.stdout.splitlines() if line.startswith("VM_ALLOCATE")
    )


@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="parent-region isolation is measured via macOS vmmap",
)
def test_parent_region_count_flat_across_repeated_child_calls(
    store: MemoryStore,
) -> None:
    """Repeated child calls must not grow the parent's VM_ALLOCATE region set.

    The numba JIT arenas live in the ephemeral child and are reclaimed on exit,
    so the parent (this pytest process) stays flat.
    """
    import os

    from iai_mcp.runtime_graph_cache import compute_assignment_in_child

    _seed_planted_partition(
        store, n_total=300, K=4, n_intra_per_block=1500, n_inter=100, seed=41,
    )
    graph = _build_graph_from_store(store)

    # Warm the first spawn (cold-start) before measuring steady state.
    compute_assignment_in_child(graph, prior_mode="seeded")

    before = _count_vm_allocate_regions(os.getpid())
    for _ in range(10):
        compute_assignment_in_child(graph, prior_mode="seeded")
    after = _count_vm_allocate_regions(os.getpid())

    delta = after - before
    assert delta < 50, (
        f"parent VM_ALLOCATE region count grew by {delta} across 10 child "
        f"calls (before={before}, after={after}); numba JIT is leaking into "
        f"the parent instead of the child"
    )


# ---------------------------------------------------------------------------
# WR-01: modularity parity — child must stream the real CPM modularity, not 0.0
# ---------------------------------------------------------------------------

def test_child_modularity_matches_in_process(store: MemoryStore) -> None:
    """The spawn path must return the real CPM modularity, not a hard-coded 0.0.

    Before the fix `compute_assignment_in_child` always returned modularity=0.0;
    the crisis-recluster telemetry therefore always emitted 0.0 regardless of
    graph structure. The child now streams the modularity it computed; this test
    pins that the round-tripped value matches the in-process reference.
    """
    from iai_mcp.runtime_graph_cache import compute_assignment_in_child

    _seed_planted_partition(
        store, n_total=800, K=4, n_intra_per_block=5000, n_inter=200, seed=71,
    )
    graph = _build_graph_from_store(store)

    ref = detect_communities(graph, prior=None, prior_mode="cold")
    assert ref.backend != "flat", (
        f"fixture regressed to flat backend ({ref.backend!r}); "
        f"modularity parity guard is vacuous unless mosaic actually runs"
    )
    assert ref.modularity > 0.0, (
        f"in-process modularity is {ref.modularity}; planted-partition fixture "
        f"must yield a positive CPM modularity to exercise the parity guard"
    )

    child = compute_assignment_in_child(graph, prior_mode="cold")

    assert abs(child.modularity - ref.modularity) < 1e-6, (
        f"child modularity {child.modularity} differs from in-process "
        f"{ref.modularity} by more than 1e-6; the spawn path is not streaming "
        f"the real modularity"
    )
    assert child.modularity != 0.0, (
        "child modularity is 0.0 — the spawn path is still returning the "
        "hard-coded sentinel instead of the real CPM value"
    )


# ---------------------------------------------------------------------------
# WR-02: sigma degrades to in-process on child crash (EOFError path)
# ---------------------------------------------------------------------------

def test_sigma_compute_and_emit_degrades_on_child_crash(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sigma.compute_and_emit must complete (degrade) when the child crashes.

    Before the fix an EOFError from a dead child escaped compute_assignment_in_child
    and was NOT caught by sigma's except tuple, so the entire topology pass was
    abandoned. The fix converts EOFError/BrokenPipeError/ConnectionError into
    WorkerCrashedError inside the helper; sigma's RuntimeError catch then
    degrades to assignment=None and compute_topology_snapshot runs in-process.
    This test injects a crash by making compute_assignment_in_child raise
    EOFError and asserts that compute_and_emit still returns a valid snapshot.
    """
    import iai_mcp.runtime_graph_cache as rgc
    import iai_mcp.sigma as sigma

    _seed_planted_partition(
        store, n_total=300, K=4, n_intra_per_block=1500, n_inter=100, seed=83,
    )

    def _crash(*_args, **_kwargs):
        raise EOFError("simulated pipe-EOF child crash")

    monkeypatch.setattr(rgc, "compute_assignment_in_child", _crash)

    # Must not raise — must degrade and return a valid snapshot.
    snap = sigma.compute_and_emit(store)

    assert snap is not None, "compute_and_emit returned None on child crash"
    assert "community_count" in snap, (
        f"degraded snapshot missing community_count: {snap!r}"
    )
    assert snap.get("community_count", 0) >= 0


# ---------------------------------------------------------------------------
# WR-03: with_centrality parity — child centrality matches in-process reference
# ---------------------------------------------------------------------------

def test_child_with_centrality_matches_in_process(store: MemoryStore) -> None:
    """The with_centrality spawn path must return a centrality map matching the
    in-process centrality_for_runtime reference.

    The child computes centrality via centrality_for_runtime on the same graph
    it built for community detection. This test locks that the round-tripped map
    matches an in-process call to the same function within a tight tolerance —
    pinning the 'same result' claim that previously rested only on a docstring.
    """
    from iai_mcp.centrality_approx import centrality_for_runtime
    from iai_mcp.runtime_graph_cache import compute_assignment_in_child

    _seed_planted_partition(
        store, n_total=300, K=4, n_intra_per_block=1500, n_inter=100, seed=97,
    )
    graph = _build_graph_from_store(store)

    # In-process reference — same function the child calls.
    ref_centrality = centrality_for_runtime(graph)
    assert ref_centrality, "fixture produced an empty centrality map"

    _child_assignment, child_centrality = compute_assignment_in_child(
        graph, prior_mode="seeded", with_centrality=True
    )

    assert set(child_centrality.keys()) == set(ref_centrality.keys()), (
        "child centrality map covers a different node set than in-process"
    )
    mismatches = []
    for node_id, ref_val in ref_centrality.items():
        child_val = child_centrality.get(node_id, float("nan"))
        if abs(child_val - ref_val) > 1e-9:
            mismatches.append((node_id, ref_val, child_val))
    assert not mismatches, (
        f"child centrality diverges from in-process for {len(mismatches)} nodes; "
        f"first 3: {mismatches[:3]}"
    )
