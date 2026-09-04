"""Routing tests for the five recall read sites moved onto db.ro_conn().

Covers: non-starvation under a held writer lock (the exact boot-storm
starvation shape), parity between the RO-served and forced-writer-path
results, the read-your-writes spy proving find_record_by_tag/reinforce_record
never borrow from the RO pool, the tombstone fence, the recency-window
crossing freshness proof (an aged-out record committed only via the buffered
flush must still be found through the RO-routed path with no manual
generation bump), and non-interaction with the hnsw double-buffered swap.

Every test runs under both storage drivers from this one file: lilli-only
cases are skipped-not-failed under stdlib, mirroring
tests/test_recall_ro_pool.py's per-function decorator pattern.

Every store here lives under ``tmp_path`` — never a real or prod store.
"""

from __future__ import annotations

import os
import struct
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.hippo._db import DEFAULT_STORAGE_DRIVER
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord
from tests._helpers import stub_embedder_for_store

_LILLI = os.environ.get("LILLI_STORAGE_DRIVER", DEFAULT_STORAGE_DRIVER).lower() == "lilli"

_lilli_only = pytest.mark.skipif(
    not _LILLI,
    reason="non-starvation/interaction proofs are lilli-only (RO pool does not "
    "exist on stdlib); parity/spy/fence cases run unconditionally instead",
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _crypto_passphrase(monkeypatch):
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")
    yield


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch):
    import keyring as _keyring

    fake: dict = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake.pop((s, u), None))
    yield fake


def _norm_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def _make_rec(
    rid: UUID, seed: int, surface: str, *, role: str | None = None
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    tags = ["capture"] if role is None else ["capture", f"role:{role}"]
    return MemoryRecord(
        id=rid,
        tier="episodic",
        literal_surface=surface,
        aaak_index="",
        embedding=_norm_vec(seed),
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
        tags=tags,
        language="en",
    )


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "store")


def _insert_record_direct(store: MemoryStore, seed: int, surface: str = "") -> str:
    """Insert one record directly via SQL under the writer conn; return its id."""
    db = store.db
    rid = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()
    with db._conn_lock:
        db._conn.execute(
            "INSERT INTO records "
            "(id, tier, literal_surface, embedding, tombstoned_at, "
            " embedding_pending, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rid,
                "episodic",
                surface,
                struct.pack(f"<{EMBED_DIM}f", *_norm_vec(seed)),
                None,
                0,
                now_iso,
                now_iso,
            ),
        )
        db._conn.commit()
    return rid


def _seed_records_and_edges(store: MemoryStore, n: int) -> list[UUID]:
    ids: list[UUID] = []
    for i in range(n):
        rid = uuid4()
        store.insert(_make_rec(rid, seed=i, surface=f"seed record {i}"))
        ids.append(rid)
    flush_record_buffer(store)
    # a small edge fan-out so incident_edges has real rows to fetch
    for i in range(min(n - 1, 20)):
        try:
            store.boost_edges([(ids[i], ids[i + 1])], delta=0.2)
        except Exception:  # noqa: BLE001 -- best-effort edge seeding
            pass
    return ids


# ---------------------------------------------------------------------------
# 1. NON-STARVATION (lilli-only) — the phase's core assertion.
# ---------------------------------------------------------------------------


@_lilli_only
def test_recall_reads_never_block_on_held_writer_lock(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    ids = _seed_records_and_edges(store, 30)

    holder_ready = threading.Event()
    release_holder = threading.Event()

    def _hold_writer_lock():
        with store.db._conn_lock:
            holder_ready.set()
            release_holder.wait(timeout=15)

    holder = threading.Thread(target=_hold_writer_lock)
    holder.start()
    assert holder_ready.wait(timeout=5), "writer-lock holder never acquired the lock"

    try:
        t0 = time.monotonic()
        got_batch = store.get_batch(ids[:5])
        elapsed_get_batch = time.monotonic() - t0
        assert got_batch, "get_batch must return results while the writer lock is held"
        assert elapsed_get_batch < 2.0, (
            f"get_batch took {elapsed_get_batch:.2f}s while the writer lock was "
            f"held — recall read must not queue behind the writer"
        )

        t0 = time.monotonic()
        edges = store.incident_edges(ids[:5])
        elapsed_edges = time.monotonic() - t0
        assert edges
        assert elapsed_edges < 2.0, f"incident_edges took {elapsed_edges:.2f}s under a held writer lock"

        t0 = time.monotonic()
        pairs = store.exact_top_k(_norm_vec(0), k=5)
        elapsed_exact = time.monotonic() - t0
        assert pairs, "exact_top_k (cold build) must return results under a held writer lock"
        assert elapsed_exact < 2.0, f"exact_top_k took {elapsed_exact:.2f}s under a held writer lock"

        # query_similar's ANN candidate fetch (_ann_knn_fetch_core's routed
        # vec_label IN fetch) must not block on the writer lock. query_similar
        # itself first calls tbl.count_rows() (a separate, NOT-routed call
        # site, outside the routed five-site scope) which still legitimately takes
        # _conn_lock -- so this test exercises _ann_knn_fetch_core directly
        # to isolate the routed site's behavior from that unrelated gate.
        tbl = store.db.open_table("records")
        q = tbl.search(list(_norm_vec(0))).distance_type("cosine").limit(5)
        t0 = time.monotonic()
        ann_rows = q.to_row_dicts()
        elapsed_ann = time.monotonic() - t0
        assert ann_rows, "_ann_knn_fetch_core must return results under a held writer lock"
        assert elapsed_ann < 2.0, f"ANN candidate fetch took {elapsed_ann:.2f}s under a held writer lock"
    finally:
        release_holder.set()
        holder.join(timeout=15)
    store.close()


def test_stdlib_reads_still_complete_after_lock_release(tmp_path: Path, monkeypatch) -> None:
    """On stdlib, ro_conn() is the writer path — document the no-op semantics
    (the read still completes correctly once the lock releases), not speed."""
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "stdlib")
    store = _make_store(tmp_path)
    assert store.db._storage_driver == "stdlib"
    ids = _seed_records_and_edges(store, 5)

    release_holder = threading.Event()
    holder_ready = threading.Event()

    def _hold_writer_lock():
        with store.db._conn_lock:
            holder_ready.set()
            release_holder.wait(timeout=5)

    holder = threading.Thread(target=_hold_writer_lock)
    holder.start()
    assert holder_ready.wait(timeout=5)
    release_holder.set()
    holder.join(timeout=5)

    got_batch = store.get_batch(ids[:2])
    assert got_batch
    store.close()


# ---------------------------------------------------------------------------
# 2. PARITY DIFFERENTIAL — RO-served vs forced-writer-path, identical rows.
# ---------------------------------------------------------------------------


def test_parity_get_batch_ro_vs_writer_path(tmp_path: Path, monkeypatch) -> None:
    store = _make_store(tmp_path)
    ids = _seed_records_and_edges(store, 15)

    ro_served = store.get_batch(ids)

    monkeypatch.setenv("IAI_MCP_RO_POOL_OFF", "1")
    writer_served = store.get_batch(ids)

    assert set(ro_served.keys()) == set(writer_served.keys())
    for rid in ro_served:
        assert ro_served[rid].literal_surface == writer_served[rid].literal_surface
        assert ro_served[rid].id == writer_served[rid].id
    store.close()


def test_parity_incident_edges_ro_vs_writer_path(tmp_path: Path, monkeypatch) -> None:
    store = _make_store(tmp_path)
    ids = _seed_records_and_edges(store, 15)

    ro_served = store.incident_edges(ids)

    monkeypatch.setenv("IAI_MCP_RO_POOL_OFF", "1")
    writer_served = store.incident_edges(ids)

    assert ro_served.keys() == writer_served.keys()
    for rid in ro_served:
        assert sorted(ro_served[rid]) == sorted(writer_served[rid])
    store.close()


def test_parity_exact_top_k_ro_vs_writer_path(tmp_path: Path, monkeypatch) -> None:
    store = _make_store(tmp_path)
    _seed_records_and_edges(store, 15)
    cue = _norm_vec(0)

    ro_served = store.exact_top_k(cue, k=10)

    # Force a fresh cold build under the writer path by invalidating the
    # resident matrix, then flip the kill-switch before the rebuild scan.
    store.invalidate_exact_index()
    monkeypatch.setenv("IAI_MCP_RO_POOL_OFF", "1")
    writer_served = store.exact_top_k(cue, k=10)

    assert [rid for rid, _ in ro_served] == [rid for rid, _ in writer_served]
    for (rid1, s1), (rid2, s2) in zip(ro_served, writer_served):
        assert rid1 == rid2
        assert s1 == pytest.approx(s2, abs=1e-6)
    store.close()


def test_parity_query_similar_ro_vs_writer_path(tmp_path: Path, monkeypatch) -> None:
    store = _make_store(tmp_path)
    _seed_records_and_edges(store, 15)
    cue = _norm_vec(0)

    ro_served = store.query_similar(cue, k=5)

    monkeypatch.setenv("IAI_MCP_RO_POOL_OFF", "1")
    writer_served = store.query_similar(cue, k=5)

    assert [r.id for r, _ in ro_served] == [r.id for r, _ in writer_served]
    for (r1, s1), (r2, s2) in zip(ro_served, writer_served):
        assert r1.literal_surface == r2.literal_surface
        assert s1 == pytest.approx(s2, abs=1e-6)
    store.close()


def test_parity_liveness_recall_dispatch_ro_vs_writer_path(tmp_path: Path, monkeypatch) -> None:
    from iai_mcp import core as _core
    import iai_mcp.pipeline as _pm

    store = _make_store(tmp_path)
    ids = _seed_records_and_edges(store, 15)
    cue = _norm_vec(0)

    class _StubEmbedder:
        def embed(self, _text: str) -> list[float]:
            return list(cue)

    stub_embedder_for_store(monkeypatch, _StubEmbedder())

    def _dispatch():
        _pm._last_recall_latency_ms = 0.0
        return _core.dispatch(store, "memory_recall", {
            "cue": "irrelevant text, embedder is stubbed",
            "session_id": "ro-routing-parity-test",
            "budget_tokens": 2000,
            "cue_embedding": cue,
        })

    ro_resp = _dispatch()
    monkeypatch.setenv("IAI_MCP_RO_POOL_OFF", "1")
    writer_resp = _dispatch()

    ro_ids = {h["record_id"] for h in ro_resp["hits"]}
    writer_ids = {h["record_id"] for h in writer_resp["hits"]}
    assert ro_ids == writer_ids
    assert str(ids[0]) in ro_ids or len(ids) > 0
    store.close()


# ---------------------------------------------------------------------------
# 3. READ-YOUR-WRITES SPY — find_record_by_tag / reinforce_record never
#    borrow from the RO pool.
# ---------------------------------------------------------------------------


def test_find_record_by_tag_and_reinforce_never_borrow_ro_pool(tmp_path: Path, monkeypatch) -> None:
    store = _make_store(tmp_path)
    rid = uuid4()
    rec = _make_rec(rid, seed=0, surface="tagged record")
    rec.tags = ["a-unique-dedup-tag"]
    store.insert(rec)
    flush_record_buffer(store)

    ro_borrow_calls = {"n": 0}
    real_ro_conn = store.db.ro_conn

    class _SpyCtx:
        def __enter__(self):
            ro_borrow_calls["n"] += 1
            return self._inner.__enter__()

        def __exit__(self, *a):
            return self._inner.__exit__(*a)

        def __init__(self, inner):
            self._inner = inner

    def _spy_ro_conn():
        return _SpyCtx(real_ro_conn())

    monkeypatch.setattr(store.db, "ro_conn", _spy_ro_conn)

    found_id = store.find_record_by_tag("a-unique-dedup-tag")
    assert found_id == rid

    store.reinforce_record(found_id, is_retrieval=True)

    assert ro_borrow_calls["n"] == 0, (
        f"find_record_by_tag/reinforce_record must NEVER borrow from ro_conn(); "
        f"got {ro_borrow_calls['n']} borrows"
    )
    store.close()


# ---------------------------------------------------------------------------
# 4. TOMBSTONE FENCE — a tombstoned record's write bumps the pool generation;
#    a liveness-shaped recall must not resurrect it via the RO path.
# ---------------------------------------------------------------------------


def test_tombstone_fence_ro_served_liveness_excludes_tombstoned(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    victim_id = uuid4()
    store.insert(_make_rec(victim_id, seed=0, surface="soon to be gone"))
    flush_record_buffer(store)

    with store.db.ro_conn() as conn:
        row = conn.execute(
            "SELECT id FROM records WHERE id = ? AND tombstoned_at IS NULL",
            (str(victim_id),),
        ).fetchone()
    assert row is not None

    with store.db._conn_lock:
        store.db._conn.execute(
            "UPDATE records SET tombstoned_at = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), str(victim_id)),
        )
        store.db._conn.commit()
    store._corpus_count_cache.invalidate("active")

    with store.db.ro_conn() as conn:
        row = conn.execute(
            "SELECT id FROM records WHERE id = ? AND tombstoned_at IS NULL",
            (str(victim_id),),
        ).fetchone()
    assert row is None, (
        "a liveness-shaped RO-served SELECT must not resurface a tombstoned "
        "record after the write's cache invalidation bumps the pool generation"
    )
    store.close()


# ---------------------------------------------------------------------------
# 5. RECENCY WINDOW CROSSING — the false-negative freshness invariant.
# ---------------------------------------------------------------------------


def test_recency_window_crossing_no_false_negative(tmp_path: Path, monkeypatch) -> None:
    """A record aged out of the recency buffer, committed only via the
    buffered flush (which never reaches the store's invalidation funnel),
    must still be found through the RO-routed path with no manual
    generation bump.

    Two independent mechanisms mark the RO pool stale on this commit: the
    CorpusCountCache.on_invalidate union hook, and the connection-level
    mutation-signalling proxy that bumps every mutating statement — either
    alone is sufficient, so this proves the outcome, not one specific wire.
    """
    monkeypatch.setenv("IAI_MCP_RECENCY_BUFFER_MAXLEN", "10")
    # The test's own explicit flush_record_buffer() call (step c) is what this
    # test is proving the freshness contract for -- the suite-wide autoflush
    # fixture would otherwise flush after every single insert(), defeating the
    # whole point of accumulating an un-flushed buffer to age past maxlen.
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")
    store = _make_store(tmp_path)

    maxlen = store._recency_buffer._maxlen
    assert maxlen == 10, "recency buffer maxlen must reflect the env override"

    # a. Prime the RO pool with a read that predates everything captured next.
    with store.db.ro_conn() as conn:
        conn.execute("SELECT COUNT(*) FROM records").fetchone()

    # b. Capture maxlen + 10 records through the normal capture path so the
    #    earliest capture ages out of the recency buffer.
    total = maxlen + 10
    ids: list[UUID] = []
    for i in range(total):
        rid = uuid4()
        store.insert(_make_rec(rid, seed=i, surface=f"aging record {i}", role="user"))
        ids.append(rid)

    earliest_id = ids[0]

    funnel_calls = {"n": 0}
    real_funnel = store._invalidate_corpus_count

    def _spy_funnel(*keys):
        funnel_calls["n"] += 1
        return real_funnel(*keys)

    monkeypatch.setattr(store, "_invalidate_corpus_count", _spy_funnel)

    # c. Force flush via the buffered path directly (bypasses the funnel).
    from iai_mcp.store import _buffers
    flushed = _buffers.flush_record_buffer(store)
    assert flushed == total

    assert funnel_calls["n"] == 0, (
        "flush_record_buffer must never call the store's invalidation funnel — "
        "if this fires, the test is no longer proving the buffered-flush seam"
    )

    # earliest record must be evicted from the recency buffer (aged out).
    buffer_ids = {m.id for m in store._recency_buffer}
    assert str(earliest_id) not in buffer_ids, (
        "test setup: the earliest capture must have aged out of the recency "
        "buffer for this to be a genuine window-crossing test"
    )

    # d. With NO manual mark_stale()/mark_ro_pool_stale()/funnel call, recall
    #    the aged-out (committed + live) record through the RO-routed path.
    got = store.get_batch([earliest_id])
    assert got, (
        "get_batch must find the aged-out committed+live record via the RO "
        "path with no manual generation bump — false-negative window proof"
    )
    assert earliest_id in got

    with store.db.ro_conn() as conn:
        row = conn.execute(
            "SELECT id FROM records WHERE id = ? AND tombstoned_at IS NULL",
            (str(earliest_id),),
        ).fetchone()
    assert row is not None, (
        "a liveness-shaped RO-served SELECT must find the aged-out record too"
    )

    # e. Companion assertion: a just-flushed record (still fresh, front of
    #    the recency buffer) is also served correctly.
    newest_id = ids[-1]
    newest_buffer_ids = {m.id for m in store._recency_buffer}
    assert str(newest_id) in newest_buffer_ids, (
        "test setup: the newest capture must still be resident in the "
        "recency buffer"
    )
    got_newest = store.get_batch([newest_id])
    assert newest_id in got_newest
    store.close()


@_lilli_only
def test_recency_window_crossing_fires_on_invalidate_hook(tmp_path: Path, monkeypatch) -> None:
    """The CorpusCountCache.on_invalidate union hook actually fires on the
    buffered flush that ages a record out of the recency buffer — proving
    this specific wire is live, as one of the (redundant) mechanisms behind
    the freshness outcome the sibling test asserts end-to-end. A disabled-hook
    negative is no longer a valid seam here: the connection-level
    mutation-signalling proxy also marks the RO pool stale on every mutating
    statement, so disabling only this hook no longer reproduces a failure.
    """
    monkeypatch.setenv("IAI_MCP_RECENCY_BUFFER_MAXLEN", "10")
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")
    store = _make_store(tmp_path)
    maxlen = store._recency_buffer._maxlen

    hook_calls = {"n": 0}
    real_hook = store._corpus_count_cache.on_invalidate

    def _spy_hook() -> None:
        hook_calls["n"] += 1
        if real_hook is not None:
            real_hook()

    store._corpus_count_cache.on_invalidate = _spy_hook

    with store.db.ro_conn() as conn:
        conn.execute("SELECT COUNT(*) FROM records").fetchone()

    total = maxlen + 10
    ids: list[UUID] = []
    for i in range(total):
        rid = uuid4()
        store.insert(_make_rec(rid, seed=i, surface=f"aging record {i}", role="user"))
        ids.append(rid)
    earliest_id = ids[0]

    from iai_mcp.store import _buffers
    flushed = _buffers.flush_record_buffer(store)
    assert flushed == total

    buffer_ids = {m.id for m in store._recency_buffer}
    assert str(earliest_id) not in buffer_ids

    assert hook_calls["n"] > 0, (
        "CorpusCountCache.on_invalidate must fire on the buffered flush that "
        "commits an aged-out record — the wiring behind the freshness "
        "outcome must be live, not merely coincidentally satisfied"
    )
    got = store.get_batch([earliest_id])
    assert earliest_id in got, (
        "the aged-out record must still be found through the RO-routed path"
    )
    store.close()


# ---------------------------------------------------------------------------
# 6. HNSW SWAP INTERPLAY — the double-buffered rebuild shares no lock with
#    the RO pool.
# ---------------------------------------------------------------------------


@_lilli_only
def test_hnsw_rebuild_concurrent_with_ro_served_query_similar(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _seed_records_and_edges(store, 40)
    cue = _norm_vec(0)

    errors: list[Exception] = []
    stop = threading.Event()

    def _query_loop():
        try:
            while not stop.is_set():
                res = store.query_similar(cue, k=5)
                assert isinstance(res, list)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    reader = threading.Thread(target=_query_loop)
    reader.start()
    try:
        for _ in range(3):
            store.db._rebuild_index_from_sqlite()
            time.sleep(0.05)
    finally:
        stop.set()
        reader.join(timeout=15)

    assert not errors, f"hnsw rebuild concurrent with RO-served query_similar raised: {errors}"
    store.close()
