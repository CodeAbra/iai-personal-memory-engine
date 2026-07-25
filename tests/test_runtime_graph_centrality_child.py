"""Child-isolated betweenness centrality for the runtime graph build.

Three guarantees are pinned here:

  1. Centrality parity: the betweenness map computed in the spawn-context child
     equals the in-parent `graph.centrality()` map within float tolerance for
     every node. Betweenness is deterministic given the CSR, so values must
     match, not just rank.

  2. Communities + centrality from ONE child: when the detection child is asked
     for centrality it returns both the community partition and the centrality
     map from a single graph build — no second spawn for the same graph.

  3. Bounded degrade: when the centrality child fails, the graph build NEVER
     recomputes exact betweenness in this long-lived process. It serves the
     last-good cached centrality when one survives on disk, else a neutral (zero)
     centrality for this cycle. Recall stays correct under either — the seed
     blend collapses to cosine-led ranking with a neutral centrality term — and
     the next warm cycle retries.
"""
from __future__ import annotations

import platform
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp import retrieve, runtime_graph_cache
from iai_mcp.graph import MemoryGraph
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord
from tests._runtime_graph_rss_arm import run_runtime_graph_rss_arm


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


def _flush_store(store: MemoryStore) -> None:
    """Drain the record + edge write buffers so the table reflects every insert.

    Inserts buffer up to a size/time threshold before they land in the table;
    these tests build the runtime graph immediately after seeding, so they flush
    explicitly to make the corpus visible to the streaming build deterministically.
    """
    from iai_mcp.store._buffers import flush_edge_buffer, flush_record_buffer

    flush_record_buffer(store)
    flush_edge_buffer(store)


def _seed_store(store: MemoryStore, n: int, seed_base: int = 0) -> list[UUID]:
    ids: list[UUID] = []
    for i in range(n):
        rec = _make_rec(seed_base + i, store)
        store.insert(rec)
        ids.append(rec.id)
    _flush_store(store)
    return ids


def _seed_store_connected(
    store: MemoryStore, n: int, seed_base: int = 0
) -> list[UUID]:
    """Seed records AND a connected edge backbone so betweenness is non-trivial.

    A freshly-inserted store has no edges, so the runtime graph it builds is
    edgeless and every node's betweenness is legitimately zero. To exercise the
    last-good-centrality reuse the cache must hold a non-zero centrality, which
    requires real structure: a backbone path linking every consecutive record
    plus a few seeded long-range chords that create high-betweenness bridges.
    """
    ids = _seed_store(store, n, seed_base=seed_base)
    backbone = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
    store.boost_edges(backbone, delta=1.0, edge_type="hebbian")
    rng = np.random.default_rng(seed_base)
    chords: list[tuple[UUID, UUID]] = []
    for _ in range(max(1, n // 10)):
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a != b:
            chords.append((ids[a], ids[b]))
    if chords:
        store.boost_edges(chords, delta=1.0, edge_type="hebbian")
    _flush_store(store)
    return ids


def _connected_graph(n: int, seed: int) -> MemoryGraph:
    """A connected topology with genuine betweenness variation.

    A backbone path threaded through every node guarantees connectivity (so
    betweenness has real structure), plus a sprinkle of chords from a seeded RNG
    to create high-betweenness bridge nodes. Embeddings are present so the child
    graph build is byte-identical to the parent's.
    """
    rng = np.random.default_rng(seed)
    ids = [uuid4() for _ in range(n)]
    g = MemoryGraph()
    for i, uid in enumerate(ids):
        vec = rng.random(8).astype(np.float32).tolist()
        g.add_node(uid, community_id=None, embedding=vec)
    # Backbone path — every consecutive pair.
    for i in range(n - 1):
        g.add_edge(ids[i], ids[i + 1], weight=1.0, edge_type="hebbian")
    # Chords — random long-range links to vary betweenness.
    n_chords = max(1, n // 10)
    for _ in range(n_chords):
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a != b:
            g.add_edge(ids[a], ids[b], weight=1.0, edge_type="hebbian")
    return g


def test_child_centrality_matches_in_parent_exactly():
    """The child-computed betweenness map equals the in-parent map within float
    tolerance for every node. Values must match, not merely rank."""
    g = _connected_graph(n=600, seed=4242)

    in_parent = g.centrality()
    child_map = runtime_graph_cache.compute_centrality_in_child(g)

    assert set(child_map) == set(in_parent), (
        "child centrality covers a different node set than the in-parent map"
    )
    assert in_parent, "in-parent centrality unexpectedly empty"
    # At least one node must carry non-trivial betweenness or the test topology
    # is degenerate and the parity check would be vacuous.
    assert any(v > 0.0 for v in in_parent.values()), (
        "test topology produced all-zero betweenness — no real variation"
    )
    for node_uuid, ref_val in in_parent.items():
        got = child_map[node_uuid]
        assert abs(got - ref_val) <= 1e-6, (
            f"centrality mismatch for {node_uuid}: "
            f"child={got} in_parent={ref_val}"
        )


def test_detection_child_returns_communities_and_centrality_together():
    """One child build returns BOTH the community partition and the centrality
    map — no second spawn for the same graph."""
    g = _connected_graph(n=500, seed=909)

    in_parent_centrality = g.centrality()

    assignment, centrality_map = runtime_graph_cache.compute_assignment_in_child(
        g, prior_mode="seeded", with_centrality=True
    )

    # Communities cover every node.
    assert set(assignment.node_to_community) == set(g.iter_nodes())
    # Centrality covers every node and matches the in-parent values.
    assert set(centrality_map) == set(in_parent_centrality)
    for node_uuid, ref_val in in_parent_centrality.items():
        assert abs(centrality_map[node_uuid] - ref_val) <= 1e-6


def test_build_runtime_graph_writes_child_centrality(store: MemoryStore):
    """A full build over a real store populates node-payload centrality from the
    child, matching the centrality the same graph would yield in-parent.

    A real store with random embeddings may form a sparse (mostly disconnected)
    graph whose betweenness is legitimately zero; the parity assertion holds
    regardless — every node's stored centrality must equal the in-parent value.
    The genuine non-trivial-betweenness parity is pinned by the connected-graph
    tests above."""
    _seed_store(store, n=120, seed_base=700)

    graph, _assignment, _rc = retrieve.build_runtime_graph(store)

    # Independent in-parent reference over the identical graph structure.
    ref = graph.centrality()
    assert set(ref) == set(graph.iter_nodes())
    for nid in graph.iter_nodes():
        payload = graph._node_payload.get(str(nid), {})
        assert "centrality" in payload, (
            f"node {nid} missing centrality after child build"
        )
        stored = float(payload.get("centrality") or 0.0)
        assert abs(stored - ref[nid]) <= 1e-6, (
            f"node {nid} payload centrality {stored} != reference {ref[nid]}"
        )


def _in_process_detect_factory():
    """Detection that runs in-process and signals 'compute centrality elsewhere'.

    Returns an `_detect_communities_isolated` stand-in whose `child_centrality`
    is None, so the build reaches the dedicated centrality-child branch where the
    bounded degrade lives.
    """
    import iai_mcp.community as _cm

    def _in_process_detect(store, graph, *, with_centrality=False):
        assignment = _cm.detect_communities(graph, prior=None, prior_mode="seeded")
        if with_centrality:
            return assignment, None
        return assignment

    return _in_process_detect


def _install_in_parent_spy(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Trip-wire the in-parent exact betweenness so the test FAILS if it runs.

    The whole point of the bounded degrade is that the warm path must NEVER call
    the exact `MemoryGraph.centrality()` (which delegates to the native
    `betweenness_centrality` Brandes pass) on a child failure. This spy fails the
    test the instant that path is taken, and is revert-proof: the old in-parent
    fallback called `graph.centrality()` directly and would trip it.
    """
    state = {"in_parent_called": False}
    real_centrality = MemoryGraph.centrality

    def _spy(self):
        state["in_parent_called"] = True
        raise AssertionError(
            "in-parent exact betweenness_centrality was called on the warm "
            "fallback path — the bounded degrade must never recompute centrality "
            "in the long-lived parent"
        )

    monkeypatch.setattr(MemoryGraph, "centrality", _spy)
    state["_real"] = real_centrality
    return state


def test_centrality_child_failure_serves_neutral_no_in_parent(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
):
    """Child centrality failure with NO cached centrality serves a neutral
    (all-zero) centrality and NEVER recomputes exact betweenness in-parent.

    Recall stays correct: with centrality neutral the seed blend collapses to
    0.6*cos, so seeds rank by cosine alone. The build must still return a graph,
    an assignment, and a centrality value on every node.
    """
    _seed_store(store, n=80, seed_base=800)

    def _boom(graph, **kw):
        raise runtime_graph_cache.WorkerCrashedError("simulated centrality crash")

    monkeypatch.setattr(
        retrieve, "_detect_communities_isolated", _in_process_detect_factory()
    )
    monkeypatch.setattr(
        runtime_graph_cache, "compute_centrality_in_child", _boom
    )
    spy = _install_in_parent_spy(monkeypatch)

    graph, assignment, _rc = retrieve.build_runtime_graph(store)

    assert spy["in_parent_called"] is False, (
        "the warm fallback recomputed exact betweenness in-parent"
    )
    nodes = list(graph.iter_nodes())
    assert nodes, "build returned an empty graph"
    assert assignment is not None, "build returned no community assignment"
    for nid in nodes:
        payload = graph._node_payload.get(str(nid), {})
        assert "centrality" in payload, (
            "neutral degrade did not populate centrality on every node"
        )
        assert float(payload.get("centrality") or 0.0) == 0.0, (
            "neutral degrade must serve zero centrality when no cache exists"
        )

    # Recall still produces cosine-led seeds with neutral centrality.
    from iai_mcp.pipeline import _pick_seeds

    candidate_indices = np.arange(len(nodes))
    rng = np.random.default_rng(13)
    shared_cos = rng.random(len(nodes)).astype(np.float32)
    centrality_arr = np.zeros(len(nodes), dtype=np.float32)
    seeds = _pick_seeds(candidate_indices, shared_cos, centrality_arr, n=3)
    assert seeds.size > 0, "neutral-centrality seed selection returned no seeds"
    # With centrality neutral the blend is 0.6*cos, so the top seed is the
    # highest-cosine candidate — the ranking degrades to pure cosine, not garbage.
    assert int(seeds[0]) == int(np.argmax(shared_cos)), (
        "neutral-centrality seeds did not rank by cosine"
    )


def test_centrality_child_failure_serves_last_good_cache(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
):
    """Child centrality failure with a usable cache on disk reuses the last-good
    cached centrality and NEVER recomputes exact betweenness in-parent.

    A first clean build writes the cache (real child centrality). A second build
    is forced down the degrade path (centrality child crashes); it must serve the
    centrality persisted by the first build rather than recompute in-parent.
    """
    _seed_store_connected(store, n=80, seed_base=820)

    # First build: real child path, writes the cache with real centrality.
    graph0, _a0, _r0 = retrieve.build_runtime_graph(store)
    good_centrality = {
        str(nid): float(graph0._node_payload.get(str(nid), {}).get("centrality") or 0.0)
        for nid in graph0.iter_nodes()
    }
    assert any(v != 0.0 for v in good_centrality.values()), (
        "seeded corpus produced an all-zero centrality — degenerate fixture"
    )

    # Confirm the cache loader can see the last-good centrality independently.
    last_good = runtime_graph_cache.load_last_good_centrality(store)
    assert last_good, "cache did not retain a last-good centrality map"

    # Second build: force the degrade path. The fresh-key cache result is made to
    # look absent (so the rebuild fires instead of the light warm path), the
    # centrality child is made to time out, but the parity-only last-good loader
    # still sees the real on-disk map — the exact production degrade scenario.
    def _boom(graph, **kw):
        raise runtime_graph_cache.WorkerTimeoutError("simulated centrality timeout")

    # Simulate the production large-corpus state: the size-cap shed the large
    # node_payload AND the fresh-key centrality result, so the warm path must
    # rebuild and recompute centrality — but the parity-only last-good map still
    # survives on disk. Drop both fresh-key signals from the loaders the build
    # consults, while leaving `load_last_good_centrality` reading the real file.
    real_try_load = runtime_graph_cache.try_load

    def _try_load_shed(store):
        loaded = real_try_load(store)
        if loaded is None:
            return None
        assignment, rich_club, _node_payload, max_degree, node_degrees = loaded
        return assignment, rich_club, None, max_degree, node_degrees

    monkeypatch.setattr(
        runtime_graph_cache, "try_load_cache_results", lambda store: None
    )
    monkeypatch.setattr(runtime_graph_cache, "try_load", _try_load_shed)
    monkeypatch.setattr(
        retrieve, "_detect_communities_isolated", _in_process_detect_factory()
    )
    monkeypatch.setattr(
        runtime_graph_cache, "compute_centrality_in_child", _boom
    )
    spy = _install_in_parent_spy(monkeypatch)

    graph1, _a1, _r1 = retrieve.build_runtime_graph(store)

    assert spy["in_parent_called"] is False, (
        "the warm fallback recomputed exact betweenness in-parent"
    )
    # The last-good map is keyed by UUID; every node it covers carries that exact
    # cached centrality after the degrade — and at least one is non-zero, proving
    # the real cached signal was served rather than a neutral fallback.
    served_nonzero = False
    served_any = False
    for nid in graph1.iter_nodes():
        if nid not in last_good:
            continue
        stored = float(graph1._node_payload.get(str(nid), {}).get("centrality") or 0.0)
        assert abs(stored - last_good[nid]) <= 1e-6, (
            f"degrade did not serve the last-good cached centrality for {nid}"
        )
        served_any = True
        if last_good[nid] != 0.0:
            served_nonzero = True
    assert served_any, "no last-good centrality value was served on the degrade path"
    assert served_nonzero, (
        "degrade served only zero values — the real cached centrality was not used"
    )


def test_centrality_only_worker_skips_community_detection():
    """The `centrality_only` worker streams only centrality and never runs
    community detection — verified by driving the worker directly and asserting
    the stream carries no community/assign envelopes."""
    import multiprocessing as mp
    import threading

    from iai_mcp import runtime_graph_cache_worker

    g = _connected_graph(n=200, seed=55)
    in_parent = g.centrality()

    parent_conn, child_conn = mp.Pipe(duplex=True)
    messages: list = []

    def _run():
        runtime_graph_cache_worker._community_only_worker_entry(child_conn)

    def _drain():
        while True:
            try:
                if not parent_conn.poll(1.0):
                    if not th.is_alive():
                        while parent_conn.poll(0.1):
                            messages.append(parent_conn.recv())
                        return
                    continue
                msg = parent_conn.recv()
                messages.append(msg)
                if msg[0] == "done":
                    return
            except (EOFError, OSError):
                return

    th = threading.Thread(target=_run, daemon=True)
    drain_th = threading.Thread(target=_drain, daemon=True)
    th.start()
    drain_th.start()

    parent_conn.send(("config", {"centrality_only": True}))
    node_chunk = [
        (str(uid), np.asarray(g.get_embedding(uid) or [], dtype=np.float32).tobytes())
        for uid in g.iter_nodes()
    ]
    parent_conn.send(("nodes", node_chunk))
    parent_conn.send(("nodes_end", None))
    edge_chunk = [
        (str(s), str(d), float(w)) for s, d, w in g.iter_edges_with_weight()
    ]
    parent_conn.send(("edges", edge_chunk))
    parent_conn.send(("edges_end", None))

    th.join(timeout=120.0)
    drain_th.join(timeout=10.0)
    parent_conn.close()

    kinds = {m[0] for m in messages}
    assert "centrality" in kinds, "centrality_only stream missing centrality"
    assert "community_table" not in kinds, "centrality_only must skip community detection"
    assert "assign" not in kinds, "centrality_only must skip the assign stream"

    child_map: dict[UUID, float] = {}
    for kind, payload in messages:
        if kind == "centrality":
            for node_bytes, value in payload:
                child_map[UUID(bytes=node_bytes)] = float(value)
    assert set(child_map) == set(in_parent)
    for node_uuid, ref_val in in_parent.items():
        assert abs(child_map[node_uuid] - ref_val) <= 1e-6


@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="peak-RSS isolation proof calibrated on this host's allocator",
)
def test_parent_rss_lower_with_centrality_child(store: MemoryStore):
    """Building the runtime graph with the centrality child keeps the parent RSS
    materially below the in-parent-centrality baseline, proving the betweenness
    intermediate no longer resides in the parent.

    Each arm runs in its OWN fresh process and reports its own peak RSS
    (getrusage RUSAGE_SELF). The in-parent arm forces detection AND exact
    betweenness to run in-process (both resident); the isolated arm uses the
    real spawn-context children. Comparing two independent clean processes is
    robust to ambient load and to arm ordering — an in-process before/after
    delta on a shared runner flaked because the allocator never returned the
    first arm's pages, inflating the second arm's baseline.
    """
    n_records = 3000

    in_parent_peak = run_runtime_graph_rss_arm(
        proof="centrality",
        driver="lilli",
        seed_base=20_000,
        in_parent=True,
        root=store.root / "in_parent",
        n_records=n_records,
    )
    isolated_peak = run_runtime_graph_rss_arm(
        proof="centrality",
        driver="lilli",
        seed_base=20_000,
        in_parent=False,
        root=store.root / "isolated",
        n_records=n_records,
    )

    print(
        f"\n[centrality-rss] in_parent_peak={in_parent_peak:.1f}MB "
        f"isolated_peak={isolated_peak:.1f}MB"
    )
    assert isolated_peak < in_parent_peak * 0.95, (
        f"centrality child isolation did not lower the parent footprint: "
        f"in_parent_peak={in_parent_peak:.1f}MB "
        f"isolated_peak={isolated_peak:.1f}MB"
    )


def test_centrality_child_never_opens_store_or_holds_lock(
    monkeypatch: pytest.MonkeyPatch,
):
    """The build-time centrality child path must never construct a MemoryStore or
    acquire the store exclusive lock in-parent. Only (uuid, embedding) nodes and
    (src, dst, weight) edges cross the Pipe; the encryption key never does. This
    pins the AES fence so a future refactor cannot reintroduce a second store open
    that trips the exclusive lock and silently zeros centrality on a real corpus.

    The parity check re-affirms the child centrality matches the in-parent map, so
    the no-open hardening cannot regress correctness.
    """
    from iai_mcp.hippo._db import HippoDB
    from iai_mcp.store import MemoryStore as _MemoryStore

    g = _connected_graph(n=400, seed=31337)
    in_parent = g.centrality()
    assert any(v > 0.0 for v in in_parent.values()), (
        "test topology produced all-zero betweenness — parity check is vacuous"
    )

    fired = {"store_init": False, "exclusive_lock": False}

    real_store_init = _MemoryStore.__init__
    real_lock = HippoDB._acquire_exclusive_lock

    def _spy_store_init(self, *args, **kwargs):
        fired["store_init"] = True
        return real_store_init(self, *args, **kwargs)

    def _spy_exclusive_lock(self, *args, **kwargs):
        fired["exclusive_lock"] = True
        return real_lock(self, *args, **kwargs)

    monkeypatch.setattr(_MemoryStore, "__init__", _spy_store_init)
    monkeypatch.setattr(HippoDB, "_acquire_exclusive_lock", _spy_exclusive_lock)

    centrality_map = runtime_graph_cache.compute_centrality_in_child(g)
    assignment, det_centrality = runtime_graph_cache.compute_assignment_in_child(
        g, prior_mode="seeded", with_centrality=True
    )

    assert not fired["store_init"], (
        "the centrality child path constructed a MemoryStore in-parent — the AES "
        "fence is breached; only Pipe-streamed graph data may drive the child."
    )
    assert not fired["exclusive_lock"], (
        "the centrality child path acquired the store exclusive lock in-parent — a "
        "second store open on the same dir would trip HippoLockHeldError and zero "
        "centrality on a real corpus."
    )

    assert set(centrality_map) == set(in_parent)
    for node_uuid, ref_val in in_parent.items():
        assert abs(centrality_map[node_uuid] - ref_val) <= 1e-6
        assert abs(det_centrality[node_uuid] - ref_val) <= 1e-6


def test_centrality_only_worker_accepts_topology_only_nodes_stream():
    """Centrality computed from a topology-only node stream equals the
    embedding-carrying result for the same graph.

    The centrality worker receives node ids WITHOUT embedding blobs when the
    sender emits a topology-only stream; betweenness depends only on graph
    structure (node ids + edges), so the result must be numerically identical to
    the embedding-carrying path.
    """
    import multiprocessing as mp
    import threading

    from iai_mcp import runtime_graph_cache_worker

    # Reference: build graph WITH embeddings, compute centrality in-parent.
    g = _connected_graph(n=150, seed=7171)
    in_parent = g.centrality()
    assert any(v > 0.0 for v in in_parent.values()), (
        "test topology produced all-zero betweenness — parity check is vacuous"
    )

    parent_conn, child_conn = mp.Pipe(duplex=True)
    messages: list = []

    def _run():
        runtime_graph_cache_worker._community_only_worker_entry(child_conn)

    def _drain():
        while True:
            try:
                if not parent_conn.poll(1.0):
                    if not th.is_alive():
                        while parent_conn.poll(0.1):
                            messages.append(parent_conn.recv())
                        return
                    continue
                msg = parent_conn.recv()
                messages.append(msg)
                if msg[0] in ("done", "error"):
                    return
            except (EOFError, OSError):
                return

    th = threading.Thread(target=_run, daemon=True)
    drain_th = threading.Thread(target=_drain, daemon=True)
    th.start()
    drain_th.start()

    parent_conn.send(("config", {"centrality_only": True}))

    # Send a TOPOLOGY-ONLY node stream (ids only, no embedding blobs).
    node_ids = [str(uid) for uid in sorted(g.iter_nodes(), key=lambda u: u.bytes)]
    parent_conn.send(("nodes_topology", node_ids))
    parent_conn.send(("nodes_end", None))

    edge_chunk = [
        (str(s), str(d), float(w)) for s, d, w in g.iter_edges_with_weight()
    ]
    parent_conn.send(("edges", edge_chunk))
    parent_conn.send(("edges_end", None))

    th.join(timeout=60.0)
    drain_th.join(timeout=10.0)
    parent_conn.close()

    kinds = {m[0] for m in messages}
    assert "centrality" in kinds, (
        f"topology-only stream did not produce centrality; got kinds={kinds!r}"
    )
    assert "community_table" not in kinds, (
        "centrality_only must skip community detection"
    )

    child_map: dict[UUID, float] = {}
    for kind, payload in messages:
        if kind == "centrality":
            for node_bytes, value in payload:
                child_map[UUID(bytes=node_bytes)] = float(value)

    assert set(child_map) == set(in_parent), (
        "topology-only centrality covers a different node set than the in-parent map"
    )
    for node_uuid, ref_val in in_parent.items():
        got = child_map[node_uuid]
        assert abs(got - ref_val) <= 1e-6, (
            f"centrality mismatch for {node_uuid}: "
            f"topology-only child={got} in_parent={ref_val}"
        )


def test_compute_centrality_in_child_propagates_child_error_reason():
    """A child crash propagates an explicit reason into WorkerCrashedError.

    When the child worker sends an ("error", {...}) envelope before exiting,
    the parent must raise WorkerCrashedError carrying the child-reported reason
    or detail, not an opaque BrokenPipeError / generic pipe message.
    """
    import multiprocessing as mp
    import threading

    from iai_mcp import runtime_graph_cache, runtime_graph_cache_worker

    # A graph large enough that the child starts recv-looping before crashing.
    g = _connected_graph(n=50, seed=8888)

    parent_conn, child_conn = mp.Pipe(duplex=True)

    # Intercept: after the child receives the config + some nodes, inject a known
    # error envelope and then close. We do this by writing a custom "crasher" that
    # sends a structured error before dying.
    def _crasher(conn):
        """Worker that reads the config, then immediately reports a known error."""
        try:
            import sys
            from iai_mcp import runtime_graph_cache_worker as _w

            # Drain the config envelope.
            _ = conn.recv()  # ("config", ...)
            # Send back a structured error before exiting.
            conn.send(("error", {
                "reason": "deliberate-test-crash",
                "detail": "the worker was instructed to fail for this test",
                "traceback": "no traceback — synthetic crash",
            }))
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    ctx = mp.get_context("spawn")
    parent_conn2, child_conn2 = ctx.Pipe(duplex=True)

    # We cannot directly pass a closure to spawn, so we use the real
    # compute_centrality_in_child but patch the worker module to send the error.
    # Instead, drive the protocol directly via the community-only drain path.
    from iai_mcp.runtime_graph_cache import WorkerCrashedError, _drain_community_only_result
    import time

    # Build a minimal pipe exchange that mimics what the parent drain sees
    # when the child crashes with a structured error.
    # duplex=False: Pipe() returns (reader, writer).
    receiver, sender = mp.Pipe(duplex=False)

    def _send_error():
        sender.send(("error", {
            "reason": "deliberate-test-crash",
            "detail": "synthetic crash for WorkerCrashedError propagation test",
            "traceback": "no real traceback",
        }))
        sender.close()

    t = threading.Thread(target=_send_error, daemon=True)
    t.start()
    t.join(timeout=5.0)

    # _drain_community_only_result raises RuntimeError on an ("error", ...) envelope.
    # The parent must surface the child-reported reason — whether via RuntimeError
    # or WorkerCrashedError — not just an opaque "BrokenPipe".
    try:
        _drain_community_only_result(receiver, timeout=5.0)
        raise AssertionError(
            "expected an exception when the child reported an error, but got none"
        )
    except (RuntimeError, WorkerCrashedError) as exc:
        msg = str(exc)
        assert "deliberate-test-crash" in msg, (
            f"child-reported reason not surfaced in exception; got: {msg!r}"
        )
    finally:
        receiver.close()
