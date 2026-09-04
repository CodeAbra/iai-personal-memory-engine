"""Deferred profile_modulates edge boost through the shared reinforce queue.

Task 2 covers: end-to-end deferred-write-lands, the queue-absent synchronous
fallback, and the coactivation-still-defers-hebbian regression guard (the
refactor's most likely silent breakage). Task 3 adds the byte-identical
edge-state differential (plus a non-vacuity control), the answer-thread spy,
and the lock/FIFO guard in this same file.

A downstream-recall differential (a K-recall sequence read back through
ranking) is NOT included: ranking degree is read from the caller-supplied
in-memory graph object (pipeline.py's degree map), never from a live store
query, and nothing in the edge-write path mutates that object. The only path
by which a written edge could reach a later recall's rank is a fresh
runtime-graph rebuild, whose warm-cache/freshness-fuse behavior makes the
timing non-deterministic, and the profile_modulates degree delta cannot be
isolated from hebbian/reinforce edges landing in the same window. See the
phase SUMMARY for the full writeup.

Dual-driver: parametrizes LILLI_STORAGE_DRIVER, the same pattern as
test_lazy_decode_ranking_parity.py.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from iai_mcp import profile as profile_mod
from iai_mcp.pipeline import PROFILE_SENTINEL_UUID, recall_for_response
from iai_mcp.store import EDGES_TABLE, MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _select_driver(driver: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built -- lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _record(text: str = "n") -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.5,
        difficulty=0.3,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )


def _edge_weight(store, a: UUID, b: UUID, edge_type: str) -> "float | None":
    key = sorted([str(a), str(b)])
    df = store.db.open_table(EDGES_TABLE).to_pandas()
    if df.empty:
        return None
    mask = (
        (df["src"] == key[0])
        & (df["dst"] == key[1])
        & (df["edge_type"] == edge_type)
    )
    if not mask.any():
        return None
    return float(df.loc[mask, "weight"].iloc[0])


def _seeded_recs(store: MemoryStore, ids: "list[UUID]") -> "list[MemoryRecord]":
    """Insert records at fixed ids with deterministic one-hot embeddings, so
    two separately-built stores compare by the same (record_id, sentinel)
    edge pairs when seeded with the same id list."""
    now = datetime.now(timezone.utc)
    recs = []
    for i, rid in enumerate(ids):
        vec = [0.0] * EMBED_DIM
        vec[i % EMBED_DIM] = 1.0
        rec = MemoryRecord(
            id=rid,
            tier="episodic",
            literal_surface=f"rec{i}",
            aaak_index="",
            embedding=vec,
            community_id=None,
            centrality=0.0,
            detail_level=2,
            pinned=False,
            stability=0.5,
            difficulty=0.3,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[],
            created_at=now,
            updated_at=now,
            tags=[],
            language="en",
        )
        store.insert(rec)
        recs.append(rec)
    return recs


def _graph_for(recs: "list[MemoryRecord]"):
    from iai_mcp.graph import MemoryGraph

    graph = MemoryGraph()
    for r in recs:
        graph.add_node(r.id, community_id=None, embedding=list(r.embedding or []))
        graph.set_node_payload(r.id, {
            "embedding": list(r.embedding or []),
            "surface": r.literal_surface,
            "centrality": 0.0, "tier": r.tier, "tags": [], "language": "en",
        })
    return graph


def _all_profile_modulates_edges(store) -> dict:
    df = store.db.open_table(EDGES_TABLE).to_pandas()
    if df.empty:
        return {}
    mask = df["edge_type"] == "profile_modulates"
    return {
        (row["src"], row["dst"]): float(row["weight"])
        for _, row in df.loc[mask].iterrows()
    }


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_profile_modulate_edge_state_identical_legacy_vs_deferred(driver, tmp_path, monkeypatch):
    """Edge-state differential: the deferred path must write the SAME
    (pairs, weights) as the legacy synchronous path -- only later. n=10
    identically-seeded records under profile_state={"interest_boost": 0.5}
    give every record the same 1.5 gain, so the deferred arm's one grouped
    enqueue crosses boost_edges' >4 full-scan branch while the legacy
    chunked arm (<=4 pairs/call) stays on its cheap per-pair branch --
    exercising both internal boost_edges paths, not just timing."""
    _select_driver(driver, monkeypatch)
    from tests.test_recall_core_unit import _FakeEmbedder, _flat_assignment

    ids = [uuid4() for _ in range(10)]
    profile_state = {"interest_boost": 0.5}

    monkeypatch.setenv("IAI_MCP_DEFER_PROFILE_BOOST_OFF", "1")
    legacy_store = MemoryStore(path=tmp_path / "legacy")
    legacy_recs = _seeded_recs(legacy_store, ids)
    recall_for_response(
        store=legacy_store, graph=_graph_for(legacy_recs),
        assignment=_flat_assignment(legacy_recs), rich_club=[],
        embedder=_FakeEmbedder(), cue="rec0", session_id="s-diff-legacy",
        profile_state=profile_state,
    )
    legacy_edges = _all_profile_modulates_edges(legacy_store)

    monkeypatch.delenv("IAI_MCP_DEFER_PROFILE_BOOST_OFF", raising=False)
    deferred_store = MemoryStore(path=tmp_path / "deferred")
    deferred_recs = _seeded_recs(deferred_store, ids)
    deferred_store.enable_reinforce_queue(coalesce_ms=10)
    try:
        recall_for_response(
            store=deferred_store, graph=_graph_for(deferred_recs),
            assignment=_flat_assignment(deferred_recs), rich_club=[],
            embedder=_FakeEmbedder(), cue="rec0", session_id="s-diff-deferred",
            profile_state=profile_state,
        )
        deferred_store._reinforce_queue.flush(timeout=2.0)
    finally:
        deferred_store.disable_reinforce_queue()
    deferred_edges = _all_profile_modulates_edges(deferred_store)

    assert legacy_edges, f"[{driver}] legacy arm must have written profile_modulates edges"
    assert len(legacy_edges) >= 5, (
        f"[{driver}] need >=5 pairs to exercise boost_edges' >4 full-scan "
        f"branch on the deferred arm; got {len(legacy_edges)}"
    )
    assert deferred_edges == legacy_edges, (
        f"[{driver}] deferred and legacy profile_modulates edge state "
        f"diverged: legacy={legacy_edges!r} deferred={deferred_edges!r}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_profile_modulate_edge_state_differential_detects_broken_deferral(driver, tmp_path, monkeypatch):
    """Non-vacuity control for the edge-state differential above: if the
    deferral silently dropped the write, the comparator must be able to
    tell. Proven by forcing exactly that (stub queue_profile_modulate to a
    no-op) and asserting the resulting edge set is empty -- distinct from
    the real arms' non-empty set."""
    _select_driver(driver, monkeypatch)
    from tests.test_recall_core_unit import _FakeEmbedder, _flat_assignment

    ids = [uuid4() for _ in range(10)]
    profile_state = {"interest_boost": 0.5}

    store = MemoryStore(path=tmp_path / "broken")
    recs = _seeded_recs(store, ids)
    monkeypatch.setattr(store, "queue_profile_modulate", lambda *a, **kw: None)
    recall_for_response(
        store=store, graph=_graph_for(recs), assignment=_flat_assignment(recs),
        rich_club=[], embedder=_FakeEmbedder(), cue="rec0", session_id="s-broken",
        profile_state=profile_state,
    )
    broken_edges = _all_profile_modulates_edges(store)
    assert broken_edges == {}, (
        f"[{driver}] control arm (deferral stubbed to a no-op) must show no "
        f"profile_modulates edges -- proving the comparator would catch a "
        f"broken deferral rather than passing vacuously. Got {broken_edges!r}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_profile_modulate_edge_state_identical_legacy_vs_deferred_distinct_deltas(
    driver, tmp_path, monkeypatch,
):
    """Non-shared-delta companion to the byte-identical differential above:
    every hit gets a genuinely distinct total_gain (not one shared value),
    so queue_profile_modulate's live-queue branch issues one enqueue_pairs
    call PER DISTINCT DELTA -- the common multi-distinct-delta fan-out that
    a realistic per-hit profile_modulation_gain produces, which the
    byte-identical differential's single shared delta never exercises.
    profile_modulation_for_record is monkeypatched to a per-record-id gain
    so the distinctness is deterministic and independent of community-gate
    internals (already covered by tests/lilli/test_profile_modulates_edges.py).
    Asserts the deferred arm converges to the same final profile_modulates
    edge weights as the legacy arm, and (non-vacuity) that the fan-out
    actually produced more than one distinct enqueue_pairs delta."""
    _select_driver(driver, monkeypatch)
    from tests.test_recall_core_unit import _FakeEmbedder, _flat_assignment

    ids = [uuid4() for _ in range(10)]
    id_to_gain = {rid: 1.0 + 0.1 * (i + 1) for i, rid in enumerate(ids)}

    def _fake_gain(rec, profile_state, *, knobs_applied=None, community_id_override=None):
        return {"test_gain": id_to_gain[rec.id]}

    monkeypatch.setattr(profile_mod, "profile_modulation_for_record", _fake_gain)
    profile_state = {"interest_boost": 0.5}

    monkeypatch.setenv("IAI_MCP_DEFER_PROFILE_BOOST_OFF", "1")
    legacy_store = MemoryStore(path=tmp_path / "legacy")
    legacy_recs = _seeded_recs(legacy_store, ids)
    recall_for_response(
        store=legacy_store, graph=_graph_for(legacy_recs),
        assignment=_flat_assignment(legacy_recs), rich_club=[],
        embedder=_FakeEmbedder(), cue="rec0", session_id="s-distinct-legacy",
        profile_state=profile_state,
    )
    legacy_edges = _all_profile_modulates_edges(legacy_store)

    monkeypatch.delenv("IAI_MCP_DEFER_PROFILE_BOOST_OFF", raising=False)
    deferred_store = MemoryStore(path=tmp_path / "deferred")
    deferred_recs = _seeded_recs(deferred_store, ids)
    deferred_store.enable_reinforce_queue(coalesce_ms=10)
    enqueue_deltas: list[float] = []
    try:
        q = deferred_store._reinforce_queue
        orig_enqueue_pairs = q.enqueue_pairs

        def _spy_enqueue_pairs(pairs, delta, edge_type="hebbian"):
            if edge_type == "profile_modulates":
                enqueue_deltas.append(delta)
            return orig_enqueue_pairs(pairs, delta, edge_type=edge_type)

        monkeypatch.setattr(q, "enqueue_pairs", _spy_enqueue_pairs)
        recall_for_response(
            store=deferred_store, graph=_graph_for(deferred_recs),
            assignment=_flat_assignment(deferred_recs), rich_club=[],
            embedder=_FakeEmbedder(), cue="rec0", session_id="s-distinct-deferred",
            profile_state=profile_state,
        )
        deferred_store._reinforce_queue.flush(timeout=2.0)
    finally:
        deferred_store.disable_reinforce_queue()
    deferred_edges = _all_profile_modulates_edges(deferred_store)

    assert len(set(enqueue_deltas)) > 1, (
        f"[{driver}] non-vacuity: distinct per-hit gains must fan out into "
        f"more than one enqueue_pairs call (got deltas={enqueue_deltas!r}); "
        f"otherwise this test collapses back to the single shared-delta case "
        f"the byte-identical differential above already covers"
    )
    assert legacy_edges, f"[{driver}] legacy arm must have written profile_modulates edges"
    assert deferred_edges == legacy_edges, (
        f"[{driver}] deferred and legacy profile_modulates edge state "
        f"diverged under genuinely distinct per-hit deltas: "
        f"legacy={legacy_edges!r} deferred={deferred_edges!r}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_profile_modulate_boost_moves_off_answer_thread(driver, tmp_path, monkeypatch):
    """Non-vacuity spy: on the legacy arm, boost_edges(profile_modulates)
    fires synchronously on the answer (main) thread. On the deferred arm it
    must NOT fire on the main thread during the recall call, but must fire
    on a background thread once flushed, and the edge must land -- proving
    the write moved off-thread rather than vanishing."""
    _select_driver(driver, monkeypatch)
    from tests.test_recall_core_unit import _FakeEmbedder, _flat_assignment

    main_ident = threading.get_ident()
    profile_state = {"interest_boost": 0.5}

    def _spy_calls(store):
        calls: list[int] = []
        orig = store.boost_edges

        def _wrapped(pairs, **kw):
            if kw.get("edge_type") == "profile_modulates":
                calls.append(threading.get_ident())
            return orig(pairs, **kw)

        monkeypatch.setattr(store, "boost_edges", _wrapped)
        return calls

    ids_a = [uuid4() for _ in range(5)]
    monkeypatch.setenv("IAI_MCP_DEFER_PROFILE_BOOST_OFF", "1")
    legacy_store = MemoryStore(path=tmp_path / "legacy")
    legacy_recs = _seeded_recs(legacy_store, ids_a)
    legacy_calls = _spy_calls(legacy_store)
    recall_for_response(
        store=legacy_store, graph=_graph_for(legacy_recs),
        assignment=_flat_assignment(legacy_recs), rich_club=[],
        embedder=_FakeEmbedder(), cue="rec0", session_id="s-spy-legacy",
        profile_state=profile_state,
    )
    assert legacy_calls, f"[{driver}] legacy arm must call boost_edges(profile_modulates)"
    assert all(c == main_ident for c in legacy_calls), (
        f"[{driver}] legacy arm must boost profile_modulates synchronously "
        f"on the main/answer thread; got idents {legacy_calls!r} vs main {main_ident!r}"
    )

    monkeypatch.delenv("IAI_MCP_DEFER_PROFILE_BOOST_OFF", raising=False)
    ids_b = [uuid4() for _ in range(5)]
    deferred_store = MemoryStore(path=tmp_path / "deferred")
    deferred_recs = _seeded_recs(deferred_store, ids_b)
    deferred_store.enable_reinforce_queue(coalesce_ms=10)
    try:
        deferred_calls = _spy_calls(deferred_store)
        recall_for_response(
            store=deferred_store, graph=_graph_for(deferred_recs),
            assignment=_flat_assignment(deferred_recs), rich_club=[],
            embedder=_FakeEmbedder(), cue="rec0", session_id="s-spy-deferred",
            profile_state=profile_state,
        )
        assert not any(c == main_ident for c in deferred_calls), (
            f"[{driver}] deferred arm must NOT boost profile_modulates "
            f"synchronously on the answer thread; got {deferred_calls!r}"
        )
        deferred_store._reinforce_queue.flush(timeout=2.0)
        assert any(c != main_ident for c in deferred_calls), (
            f"[{driver}] deferred arm must boost profile_modulates on a "
            f"background thread after flush; got {deferred_calls!r}"
        )
        w = _edge_weight(deferred_store, ids_b[0], PROFILE_SENTINEL_UUID, "profile_modulates")
        assert w is not None, (
            f"[{driver}] deferred profile_modulates edge must land after "
            f"flush -- moved off-thread, not dropped"
        )
    finally:
        deferred_store.disable_reinforce_queue()


def test_profile_modulate_flush_pairs_routes_through_boost_edges_conn_lock():
    """Lock/FIFO guard (source form): the worker's edge-boost flush routes
    exclusively through boost_edges -- which acquires db._conn_lock itself
    on both its per-pair and full-scan branches -- rather than touching the
    connection directly, so lock order stays _hnsw_lock-outer /
    _conn_lock-inner with no new acquisition path. Same source-assertion
    style already used for this class of invariant
    (test_heavy_cycle_boost_edges_uses_hebbian_type)."""
    import inspect

    from iai_mcp import reinforce_queue as rq_mod

    flush_src = inspect.getsource(rq_mod.ReinforceWriteQueue._flush_pairs)
    assert "boost_edges" in flush_src, (
        "_flush_pairs must route profile_modulates (and hebbian) writes "
        "through boost_edges, not a direct connection access"
    )
    assert "self._store.db._conn" not in flush_src, (
        "_flush_pairs must not touch the connection directly -- boost_edges "
        "owns the _conn_lock acquisition"
    )
    assert "threading.Thread(" not in flush_src, (
        "_flush_pairs must not spawn a new worker thread for profile_modulates"
    )

    run_src = inspect.getsource(rq_mod.ReinforceWriteQueue._run)
    thread_spawns = run_src.count("threading.Thread(")
    assert thread_spawns == 0, (
        f"_run's drain loop must not spawn threads (found {thread_spawns}); "
        f"the queue keeps exactly one consumer thread, created once in "
        f"ReinforceWriteQueue.start()"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_profile_modulate_deferral_keeps_single_consumer_thread(driver, tmp_path, monkeypatch):
    """Runtime companion to the source guard above: enabling the queue and
    flushing a profile_modulates write does not add a second reinforce-queue
    thread -- strict FIFO per store is preserved."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    ids = [uuid4() for _ in range(2)]
    recs = _seeded_recs(store, ids)

    store.enable_reinforce_queue(coalesce_ms=10)
    try:
        store.queue_profile_modulate([(recs[0].id, PROFILE_SENTINEL_UUID)], [0.3])
        store._reinforce_queue.flush(timeout=2.0)
        reinforce_threads = [
            t for t in threading.enumerate()
            if t.name == "iai-mcp-reinforce-queue"
        ]
        assert len(reinforce_threads) == 1, (
            f"[{driver}] expected exactly one reinforce-queue consumer "
            f"thread, found {len(reinforce_threads)}"
        )
    finally:
        store.disable_reinforce_queue()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_profile_modulate_deferred_write_lands_after_flush(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    rec = _record()
    store.insert(rec)

    store.enable_reinforce_queue(coalesce_ms=10)
    try:
        store.queue_profile_modulate([(rec.id, PROFILE_SENTINEL_UUID)], [0.4])
        store._reinforce_queue.flush(timeout=2.0)

        w = _edge_weight(store, rec.id, PROFILE_SENTINEL_UUID, "profile_modulates")
        assert w == pytest.approx(0.4, abs=1e-3), (
            f"[{driver}] expected profile_modulates weight 0.4 after flush, got {w}"
        )
    finally:
        store.disable_reinforce_queue()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_profile_modulate_queue_absent_falls_back_to_sync(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    rec = _record()
    store.insert(rec)

    assert store._reinforce_queue is None, "precondition: queue must be absent"

    store.queue_profile_modulate([(rec.id, PROFILE_SENTINEL_UUID)], [0.4])

    w = _edge_weight(store, rec.id, PROFILE_SENTINEL_UUID, "profile_modulates")
    assert w == pytest.approx(0.4, abs=1e-3), (
        f"[{driver}] queue-absent fallback must write synchronously; got {w}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_coactivation_still_defers_hebbian_through_generalized_enqueue(driver, tmp_path, monkeypatch):
    """Regression guard: generalizing enqueue_pairs for profile_modulates must
    not break the already-working hebbian coactivation deferral -- the most
    likely silent breakage from this refactor."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    a = _record(text="a")
    b = _record(text="b")
    store.insert(a)
    store.insert(b)

    store.enable_reinforce_queue(coalesce_ms=10)
    try:
        store.queue_coactivation([(a.id, b.id)], 0.25)
        store._reinforce_queue.flush(timeout=2.0)

        w = _edge_weight(store, a.id, b.id, "hebbian")
        assert w == pytest.approx(0.25, abs=1e-3), (
            f"[{driver}] hebbian coactivation deferral broke: expected 0.25, got {w}"
        )
    finally:
        store.disable_reinforce_queue()
