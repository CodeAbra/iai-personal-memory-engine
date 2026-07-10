"""Single-flight guard around the runtime-graph cache-miss rebuild.

The runtime-graph build has many call sites that fire concurrently at daemon
boot. When the cache is stale or absent, an unguarded build lets every caller
run the full rebuild at once — streaming the whole corpus and spawning a child
for community detection plus centrality on the same graph. The observed effect
was a fleet of redundant spawn children grinding identical work.

Two guarantees are pinned here:

  1. Under a stale/absent cache and concurrent callers, the heavy rebuild runs
     exactly once; the remaining callers reuse the freshly-saved cache. All
     callers still return a correct, equivalent graph and community assignment.

  2. A warm cache hit does not block on the single-flight lock: a caller that
     needs no rebuild proceeds while another thread holds the lock mid-rebuild.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp import retrieve, runtime_graph_cache
from iai_mcp.store import MemoryStore
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
        pinned=bool(seed % 7 == 0),
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


def _seed_store(store: MemoryStore, n: int, seed_base: int = 0) -> None:
    for i in range(n):
        store.insert(_make_rec(seed_base + i, store))


def _seed_store_connected(store: MemoryStore, n: int, seed_base: int = 0) -> None:
    """Seed n records and chain them into a path graph.

    A path topology gives the interior nodes non-zero betweenness centrality, so
    a build saves a centrality-bearing cache and a subsequent build takes the
    lock-free warm-cache branch. An edgeless corpus has all-zero betweenness and
    would re-trigger the centrality path on every build — not a warm cache.
    """
    recs = [_make_rec(seed_base + i, store) for i in range(n)]
    for rec in recs:
        store.insert(rec)
    ids = [rec.id for rec in recs]
    pairs = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
    store.boost_edges(pairs, delta=1.0, edge_type="hebbian")


def _node_ids(graph) -> set[str]:
    return {str(nid) for nid in graph.iter_nodes()}


def test_concurrent_cache_miss_rebuilds_exactly_once(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
):
    """Four threads call the build concurrently on an absent cache; the heavy
    community-detection rebuild runs exactly once and every caller returns an
    equivalent populated graph and assignment."""
    n_records = 18
    # Connect the corpus so the rebuild produces non-zero betweenness centrality
    # and saves a fully warm cache; peers that re-probe under the lock then take
    # the light path and spawn no child of their own. An edgeless corpus has
    # all-zero centrality and would never reach the warm state, so the herd could
    # not collapse — the connected topology is the realistic production case.
    _seed_store_connected(store, n=n_records, seed_base=700)

    # Ensure the cache is genuinely absent so the first caller takes the rebuild
    # path; later callers must reuse the cache the first caller saves.
    runtime_graph_cache.invalidate(store)

    rebuild_count = {"n": 0}
    count_lock = threading.Lock()

    real_detect = retrieve._detect_communities_isolated

    def _counting_detect(store_arg, graph, **kw):
        # Count every entry into the heavy detection path and run the real
        # in-process detection (no child spawn) so the test is hermetic. Honor
        # the with_centrality contract the build relies on.
        with count_lock:
            rebuild_count["n"] += 1
        import iai_mcp.community as _cm

        assignment = _cm.detect_communities(graph, prior=None, prior_mode="seeded")
        if kw.get("with_centrality"):
            return assignment, graph.centrality()
        return assignment

    monkeypatch.setattr(
        retrieve, "_detect_communities_isolated", _counting_detect
    )

    n_threads = 4
    barrier = threading.Barrier(n_threads)
    results: list = [None] * n_threads
    errors: list = [None] * n_threads

    def _worker(idx: int) -> None:
        try:
            # Release all threads into the build at the same instant so the herd
            # is real rather than serialized by thread-startup latency.
            barrier.wait(timeout=30)
            results[idx] = retrieve.build_runtime_graph(store)
        except Exception as exc:  # noqa: BLE001 -- surface in the assertion
            errors[idx] = exc

    threads = [
        threading.Thread(target=_worker, args=(i,)) for i in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert all(not t.is_alive() for t in threads), "a build worker hung"
    assert errors == [None] * n_threads, f"build workers raised: {errors}"

    # The core regression: the heavy rebuild ran exactly once across the herd.
    assert rebuild_count["n"] == 1, (
        f"expected exactly one cache-miss rebuild under {n_threads} concurrent "
        f"callers, got {rebuild_count['n']}"
    )

    # Every caller returned a correct, equivalent graph and assignment.
    node_sets = []
    assignments = []
    for idx, res in enumerate(results):
        assert res is not None, f"worker {idx} returned no result"
        graph, assignment, _rich_club = res
        assert graph.node_count() == n_records
        assert assignment is not None
        node_sets.append(_node_ids(graph))
        assignments.append(assignment)

    base_nodes = node_sets[0]
    for idx, nodes in enumerate(node_sets[1:], start=1):
        assert nodes == base_nodes, (
            f"worker {idx} built a different node set than worker 0"
        )

    # The partitions agree (callers compare partitions, not community ids): two
    # nodes share a community in one assignment iff they share one in another.
    base_assign = assignments[0]
    base_map = base_assign.node_to_community
    nodes_sorted = sorted(base_map)
    for idx, assign in enumerate(assignments[1:], start=1):
        other_map = assign.node_to_community
        assert set(other_map) == set(base_map), (
            f"worker {idx} assignment covers a different node set"
        )
        for a, b in zip(nodes_sorted, nodes_sorted[1:]):
            same_base = base_map[a] == base_map[b]
            same_other = other_map[a] == other_map[b]
            assert same_base == same_other, (
                f"worker {idx} partition disagrees with worker 0 on a pair"
            )


def test_warm_cache_hit_does_not_block_on_rebuild_lock(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
):
    """A warm cache-hit caller proceeds without acquiring the single-flight
    lock — proven by completing while another thread holds the lock and is
    parked mid-rebuild."""
    _seed_store_connected(store, n=12, seed_base=800)

    # Prime a complete, warm cache (payload + centrality) so a subsequent build
    # takes the lock-free cache-hit branch and spawns no child.
    retrieve.build_runtime_graph(store)
    assert not retrieve._runtime_graph_rebuild_needed(store), (
        "primed cache should require no rebuild"
    )

    # Park a holder thread inside the single-flight lock to simulate a peer
    # mid-rebuild, then confirm a warm cache-hit build still completes.
    holder_acquired = threading.Event()
    release_holder = threading.Event()

    def _hold_lock() -> None:
        with retrieve._RUNTIME_GRAPH_REBUILD_LOCK:
            holder_acquired.set()
            # Hold until the cache-hit build has had its chance to complete.
            release_holder.wait(timeout=30)

    holder = threading.Thread(target=_hold_lock)
    holder.start()
    assert holder_acquired.wait(timeout=10), "holder failed to take the lock"

    hit_done = threading.Event()
    hit_result: list = [None]
    hit_error: list = [None]

    def _cache_hit_build() -> None:
        try:
            hit_result[0] = retrieve.build_runtime_graph(store)
        except Exception as exc:  # noqa: BLE001 -- surface in the assertion
            hit_error[0] = exc
        finally:
            hit_done.set()

    hit_thread = threading.Thread(target=_cache_hit_build)
    hit_thread.start()

    # The cache-hit build must finish while the lock is still held by the holder.
    completed_while_locked = hit_done.wait(timeout=15)
    # Let the holder go regardless so the test never hangs on cleanup.
    release_holder.set()
    holder.join(timeout=10)
    hit_thread.join(timeout=10)

    assert completed_while_locked, (
        "warm cache-hit build blocked on the single-flight lock while a peer "
        "held it mid-rebuild"
    )
    assert hit_error[0] is None, f"cache-hit build raised: {hit_error[0]}"
    graph, assignment, _rc = hit_result[0]
    assert graph.node_count() == 12
    assert assignment is not None


def test_cache_hit_path_correctness_unchanged(store: MemoryStore):
    """The warm cache-hit build reproduces the same node set as the original
    rebuild — the lock-free fast path is correctness-preserving."""
    _seed_store_connected(store, n=15, seed_base=900)

    graph_a, assign_a, _ra = retrieve.build_runtime_graph(store)
    assert retrieve._runtime_graph_rebuild_needed(store) is False

    graph_b, assign_b, _rb = retrieve.build_runtime_graph(store)

    assert _node_ids(graph_a) == _node_ids(graph_b)
    assert graph_b.node_count() == 15
    assert assign_b is not None
    assert set(assign_a.node_to_community) == set(assign_b.node_to_community)


def test_child_crash_fallback_still_works_under_single_flight(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
):
    """The child-crash → in-parent fallback survives the single-flight wrapper:
    a crashing detection child still degrades to in-process detection exactly
    once and yields a correct graph."""
    _seed_store(store, n=14, seed_base=1000)
    runtime_graph_cache.invalidate(store)

    def _boom(graph, **kw):
        raise runtime_graph_cache.WorkerCrashedError("simulated worker crash")

    import iai_mcp.community as _cm

    real_in_process = _cm.detect_communities
    in_process_called = {"n": 0}

    def _counting_in_process(graph, **kw):
        in_process_called["n"] += 1
        return real_in_process(graph, **kw)

    monkeypatch.setattr(
        runtime_graph_cache, "compute_assignment_in_child", _boom
    )
    monkeypatch.setattr(_cm, "detect_communities", _counting_in_process)

    graph, assignment, _rich_club = retrieve.build_runtime_graph(store)

    assert in_process_called["n"] == 1, (
        "in-process detection should have run exactly once on child crash, "
        "even under the single-flight guard"
    )
    assert assignment is not None
    assert graph.node_count() == 14
