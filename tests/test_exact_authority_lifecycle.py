"""Per-write-path lifecycle contract for the exact-cosine authority index.

Contract surfaces these tests target:
- ``MemoryStore.exact_top_k(vec, k)`` — lazy-built resident exact-cosine
  authority over the active corpus.
- ``MemoryStore.invalidate_exact_index()`` — the no-raise bulk-invalidate seam
  the sleep-pipeline optimize/erasure steps and the dedupe migrator use.
- Every corpus-changing write path (sync buffered-insert flush, async flush,
  delete, reembed flip, sleep-pipeline tombstones, dedupe tombstones) keeps
  ``exact_top_k`` consistent with a brute-force reference scan of live SQL.

Reference helper reads live SQL directly with the same active-corpus
predicate the production build path uses:
    SELECT id, embedding FROM records
     WHERE tombstoned_at IS NULL AND COALESCE(embedding_pending, 0) = 0

No production source files are modified by this file.
"""
from __future__ import annotations

import gc
import json
import struct
import weakref
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord


# ---------------------------------------------------------------------------
# Fixtures (mirrors tests/test_corpus_count_cache.py style)
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


@pytest.fixture
def store(tmp_path: Path):
    s = MemoryStore(path=tmp_path / "store")
    try:
        yield s
    finally:
        # Some tests enable the async write queue and never disable it; close()
        # drains the queue, stops the background loop, and joins the daemon
        # thread so no async-writes thread or pending coalesce task leaks past
        # the test into the rest of the suite.
        s.close()


def _seeded_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _make_rec(seed: int, *, embedding: list[float] | None = None) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=f"surface text record {seed}",
        aaak_index="",
        embedding=embedding if embedding is not None else _seeded_vec(seed),
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
        tags=["capture"],
        language="en",
    )


def _insert_pending(store: MemoryStore, surface: str = "pending text") -> str:
    pid = str(uuid4())
    store.db.insert_pending_row(
        record_id=pid,
        tier="episodic",
        literal_surface=surface,
        tags_json=json.dumps(["capture"]),
        provenance_json=json.dumps([]),
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    return pid


class _StubEmbedder:
    """Deterministic stand-in embedder for reembed_pending_rows."""

    def __init__(self, vec: list[float]) -> None:
        self._vec = vec

    def embed(self, _text: str) -> list[float]:
        return self._vec


def _brute_force_reference(store: MemoryStore, cue: list[float], k: int) -> list[tuple[str, float]]:
    """Scan live SQL with the active-corpus predicate; exact cosine top-k in numpy.

    Mirrors the production build predicate exactly (plaintext id + embedding
    columns only, no decrypt, no record materialization).
    """
    with store.db._conn_lock:
        rows = store.db._conn.execute(
            "SELECT id, embedding FROM records"
            " WHERE tombstoned_at IS NULL AND COALESCE(embedding_pending, 0) = 0"
        ).fetchall()
    ids: list[str] = []
    vecs: list[np.ndarray] = []
    for row in rows:
        rid = row["id"]
        blob = row["embedding"]
        vec = np.frombuffer(blob, dtype=np.float32)
        ids.append(rid)
        vecs.append(vec)
    if not ids:
        return []
    matrix = np.stack(vecs).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit = matrix / norms
    cue_arr = np.asarray(cue, dtype=np.float32)
    cue_norm = float(np.linalg.norm(cue_arr))
    cue_unit = cue_arr / cue_norm if cue_norm != 0.0 else cue_arr
    sims = unit @ cue_unit
    order = np.argsort(-sims, kind="stable")[:k]
    return [(ids[i], float(sims[i])) for i in order]


def _assert_matches_reference(store: MemoryStore, cue: list[float], k: int = 10) -> None:
    reference = _brute_force_reference(store, cue, k)
    actual = store.exact_top_k(cue, k)
    actual_ids = [str(uid) for uid, _score in actual]
    reference_ids = [rid for rid, _score in reference]
    assert actual_ids == reference_ids, (
        f"exact_top_k diverged from brute-force reference: {actual_ids} != {reference_ids}"
    )
    for (_uid, actual_score), (_rid, ref_score) in zip(actual, reference):
        assert actual_score == pytest.approx(ref_score, abs=1e-5)


# ---------------------------------------------------------------------------
# S1 — lazy build, boot-lazy proof
# ---------------------------------------------------------------------------


def test_s1_construction_alone_does_not_build(store: MemoryStore) -> None:
    assert store._exact_index.is_warm is False


def test_s1_first_exact_top_k_builds_and_matches_reference(store: MemoryStore) -> None:
    recs = [_make_rec(i) for i in range(6)]
    for r in recs:
        store.insert(r)
    flush_record_buffer(store)
    assert store._exact_index.is_warm is False
    cue = recs[0].embedding
    _assert_matches_reference(store, cue)
    assert store._exact_index.is_warm is True


# ---------------------------------------------------------------------------
# S2 — sync insert + buffer flush seam
# ---------------------------------------------------------------------------


def test_s2_sync_insert_and_flush_seam_matches_reference(store: MemoryStore, monkeypatch) -> None:
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")
    recs = [_make_rec(100 + i) for i in range(4)]
    for r in recs:
        store.insert(r)
    # Index never AHEAD of SQL for ids it returns: warm-build before the flush
    # must not surface unflushed rows.
    cue = recs[0].embedding
    pre_flush = store.exact_top_k(cue, k=10)
    pre_flush_ids = {str(uid) for uid, _s in pre_flush}
    assert str(recs[0].id) not in pre_flush_ids

    flush_record_buffer(store)
    store.invalidate_exact_index()
    _assert_matches_reference(store, cue)
    post_ids = {str(uid) for uid, _s in store.exact_top_k(cue, k=10)}
    assert str(recs[0].id) in post_ids


def test_s2b_one_bad_item_in_flush_batch_does_not_drop_the_good_items(
    store: MemoryStore, monkeypatch
) -> None:
    """The sync-flush feed loop must isolate each item: one item whose feed
    call raises (a malformed key, a transient numpy failure) must skip only
    that item, never abort the rest of the flushed batch — mirroring the
    per-item discipline the reembed-flip feed already uses."""
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")
    recs = [_make_rec(150 + i) for i in range(4)]
    for r in recs:
        store.insert(r)

    # Warm the matrix first (upsert is a no-op while cold) so the feed calls
    # below are observable as real matrix mutations, not silent no-ops.
    flush_record_buffer(store)
    store.invalidate_exact_index()
    store.exact_top_k(recs[0].embedding, k=10)  # lazy-build warms the matrix
    assert store._exact_index.is_warm is True

    more_recs = [_make_rec(160 + i) for i in range(4)]
    for r in more_recs:
        store.insert(r)

    bad_id = str(more_recs[1].id)
    _orig_feed = store._feed_exact_index

    def _feed_one_bad(record_id, vec):
        if record_id == bad_id:
            raise RuntimeError("simulated malformed flush item")
        return _orig_feed(record_id, vec)

    monkeypatch.setattr(store, "_feed_exact_index", _feed_one_bad)

    flush_record_buffer(store)

    fed_ids = {str(uid) for uid, _s in store.exact_top_k(recs[0].embedding, k=100)}
    good_ids = {str(r.id) for r in more_recs if str(r.id) != bad_id}
    for gid in good_ids:
        assert gid in fed_ids, (
            f"good sibling id {gid} missing from the matrix after a "
            f"malformed item ({bad_id}) in the same flush batch — the "
            "per-item try must not let one bad item drop the rest"
        )


# ---------------------------------------------------------------------------
# S3 — async flush seam
# ---------------------------------------------------------------------------


def test_s3_async_flush_seam_matches_reference(store: MemoryStore) -> None:
    rec = _make_rec(200)

    async def _run_async():
        await store.enable_async_writes(coalesce_ms=10, max_batch=8, max_queue_size=64)
        store.insert(rec)

    import asyncio as _asyncio
    _asyncio.run(_run_async())

    cue = rec.embedding
    _assert_matches_reference(store, cue)
    post_ids = {str(uid) for uid, _s in store.exact_top_k(cue, k=10)}
    assert str(rec.id) in post_ids


# ---------------------------------------------------------------------------
# S4 — delete seam
# ---------------------------------------------------------------------------


def test_s4_delete_invalidates_and_rebuild_excludes_id(store: MemoryStore) -> None:
    recs = [_make_rec(300 + i) for i in range(4)]
    for r in recs:
        store.insert(r)
    flush_record_buffer(store)
    cue = recs[0].embedding
    _assert_matches_reference(store, cue)
    assert store._exact_index.is_warm is True

    store.delete(recs[0].id)
    assert store._exact_index.is_warm is False

    _assert_matches_reference(store, cue)
    post_ids = {str(uid) for uid, _s in store.exact_top_k(cue, k=10)}
    assert str(recs[0].id) not in post_ids


# ---------------------------------------------------------------------------
# S5 — reembed flip seam
# ---------------------------------------------------------------------------


def test_s5_reembed_flip_appears_after_flip_not_before(store: MemoryStore) -> None:
    pid = _insert_pending(store, "pending capture text")
    new_vec = _seeded_vec(400)

    cue = new_vec
    pre_flip = store.exact_top_k(cue, k=10)
    pre_flip_ids = {str(uid) for uid, _s in pre_flip}
    assert pid not in pre_flip_ids

    embedder = _StubEmbedder(new_vec)
    n = store.db.reembed_pending_rows(embedder)
    assert n == 1

    _assert_matches_reference(store, cue)
    post_ids = {str(uid) for uid, _s in store.exact_top_k(cue, k=10)}
    assert pid in post_ids


# ---------------------------------------------------------------------------
# S6 — bulk invalidate seam (the seam optimize/erasure/dedupe use)
# ---------------------------------------------------------------------------


def test_s6_invalidate_exact_index_forces_rebuild_and_matches(store: MemoryStore) -> None:
    recs = [_make_rec(500 + i) for i in range(5)]
    for r in recs:
        store.insert(r)
    flush_record_buffer(store)
    cue = recs[0].embedding
    _assert_matches_reference(store, cue)
    assert store._exact_index.is_warm is True

    store.invalidate_exact_index()
    assert store._exact_index.is_warm is False

    _assert_matches_reference(store, cue)
    assert store._exact_index.is_warm is True


# ---------------------------------------------------------------------------
# S6b — tombstone -> no-leak, end-to-end (the privacy guard)
# ---------------------------------------------------------------------------


def test_s6b_tombstone_via_erasure_path_no_leak_end_to_end(store: MemoryStore) -> None:
    recs = [_make_rec(600 + i) for i in range(5)]
    for r in recs:
        store.insert(r)
    flush_record_buffer(store)

    target = recs[0]
    cue = target.embedding

    # Warm-build the matrix with a query that RETURNS the soon-to-be-tombstoned id.
    warm_ids = {str(uid) for uid, _s in store.exact_top_k(cue, k=10)}
    assert str(target.id) in warm_ids
    assert store._exact_index.is_warm is True

    # Soft-tombstone via the same code path the erasure agent uses (direct SQL
    # UPDATE on tombstoned_at) WITH the invalidate hook fired, mirroring the
    # production seam discipline exactly.
    now = datetime.now(timezone.utc)
    with store.db._conn_lock:
        store.db._conn.execute(
            "UPDATE records SET tombstoned_at = ? WHERE id = ?",
            (now, str(target.id)),
        )
        store.db._conn.commit()
    store.invalidate_exact_index()

    post_ids = {str(uid) for uid, _s in store.exact_top_k(cue, k=10)}
    assert str(target.id) not in post_ids

    reference = _brute_force_reference(store, cue, k=10)
    reference_ids = {rid for rid, _s in reference}
    assert str(target.id) not in reference_ids
    _assert_matches_reference(store, cue)


# ---------------------------------------------------------------------------
# S7 — fail-safe: a broken index never raises to the caller
# ---------------------------------------------------------------------------


def test_s7_broken_index_returns_empty_list_never_raises(store: MemoryStore, monkeypatch) -> None:
    recs = [_make_rec(700 + i) for i in range(3)]
    for r in recs:
        store.insert(r)
    flush_record_buffer(store)

    def _boom(self, vec, k):
        raise RuntimeError("simulated exact-index fault")

    monkeypatch.setattr(type(store._exact_index), "top_k", _boom)

    result = store.exact_top_k(recs[0].embedding, k=10)
    assert result == []


# ---------------------------------------------------------------------------
# S8 — weakref: HippoDB never keeps MemoryStore alive
# ---------------------------------------------------------------------------


def test_s8_hippodb_feed_callback_does_not_keep_store_alive(tmp_path: Path) -> None:
    import iai_mcp.hippo._db as _db_mod

    store = MemoryStore(path=tmp_path / "weakref_store")
    db = store.db
    store_ref = weakref.ref(store)

    del store
    gc.collect()

    assert store_ref() is None, (
        "HippoDB-held exact-index feed callback kept the MemoryStore alive "
        "past its last external reference — the binding must be weakref-only"
    )
    db.close()


# ---------------------------------------------------------------------------
# S9 — empty corpus
# ---------------------------------------------------------------------------


def test_s9_empty_corpus_returns_empty_list(store: MemoryStore) -> None:
    cue = _seeded_vec(999)
    result = store.exact_top_k(cue, k=10)
    assert result == []


# ---------------------------------------------------------------------------
# S10 — one malformed embedding row: skip-don't-die + no re-scan every recall
# ---------------------------------------------------------------------------


def test_s10_malformed_row_still_builds_warm_over_good_rows(store: MemoryStore) -> None:
    """A single row whose embedding blob is a wrong byte length (real
    precedent: the same guard exists in ``_decode_raw_row``) must not abort
    the build — the matrix comes up warm over the remaining good rows."""
    recs = [_make_rec(800 + i) for i in range(5)]
    for r in recs:
        store.insert(r)
    flush_record_buffer(store)

    # Corrupt one row's embedding blob directly via SQL: wrong byte length,
    # simulating the real-world short/odd blob the review cites.
    victim = recs[0]
    with store.db._conn_lock:
        store.db._conn.execute(
            "UPDATE records SET embedding = ? WHERE id = ?",
            (b"\x00\x01\x02", str(victim.id)),
        )
        store.db._conn.commit()

    cue = recs[1].embedding
    result = store.exact_top_k(cue, k=10)

    assert store._exact_index.is_warm is True, (
        "one malformed row must not permanently prevent the matrix from "
        "becoming warm"
    )
    result_ids = {str(uid) for uid, _s in result}
    assert str(victim.id) not in result_ids, (
        "the malformed row must be excluded from the matrix, never surfaced"
    )
    for r in recs[1:]:
        assert str(r.id) in result_ids, (
            f"good row {r.id} missing from result after a sibling row was "
            f"malformed: {result_ids}"
        )


def test_s10_build_failure_backs_off_instead_of_rescanning_every_recall(
    store: MemoryStore, monkeypatch
) -> None:
    """When ``build()`` itself raises (a genuinely unbuildable state), a
    cooldown must prevent every subsequent ``exact_top_k`` call from
    re-running the full-corpus SELECT + build retry — one scan, not one
    scan per recall."""
    recs = [_make_rec(810 + i) for i in range(3)]
    for r in recs:
        store.insert(r)
    flush_record_buffer(store)

    select_calls = {"n": 0}
    real_conn = store.db._conn

    class _CountingProxy:
        def __init__(self, target):
            object.__setattr__(self, "_target", target)

        def execute(self, sql, *args, **kwargs):
            if "SELECT id, embedding FROM records" in sql:
                select_calls["n"] += 1
            return self._target.execute(sql, *args, **kwargs)

        def __getattr__(self, name: str):
            return getattr(self._target, name)

        def __setattr__(self, name: str, value) -> None:
            setattr(self._target, name, value)

    monkeypatch.setattr(store.db, "_conn", _CountingProxy(real_conn))

    # The whole-corpus build fetch may run on a dedicated snapshot reader
    # (see `_build_exact_index_sync`) instead of `store.db._conn` — wrap that
    # seam too so the count observes the SELECT regardless of which
    # connection actually serves it.
    import iai_mcp.hippo._raw_open as _raw_open_mod

    real_open_store_conn = _raw_open_mod.open_store_conn

    def _counting_open_store_conn(path, *, read_only=False):
        conn = real_open_store_conn(path, read_only=read_only)
        if conn is None:
            return None
        return _CountingProxy(conn)

    monkeypatch.setattr(_raw_open_mod, "open_store_conn", _counting_open_store_conn)

    def _boom(self, rows, expected_generation=None):
        raise RuntimeError("simulated unbuildable state")

    monkeypatch.setattr(type(store._exact_index), "build", _boom)

    cue = recs[0].embedding
    first = store.exact_top_k(cue, k=10)
    assert first == []
    assert select_calls["n"] == 1

    # Repeated calls during the cooldown window must NOT re-run the SELECT.
    for _ in range(5):
        result = store.exact_top_k(cue, k=10)
        assert result == []
    assert select_calls["n"] == 1, (
        "a persistently unbuildable state must cost one scan, not one scan "
        f"per recall; observed {select_calls['n']} SELECT calls"
    )

    # invalidate_exact_index() clears the cooldown for an immediate retry.
    # Restore a working build so the retry can actually succeed this time.
    import iai_mcp.store._exact_index as _ei_mod

    monkeypatch.setattr(
        type(store._exact_index),
        "build",
        _ei_mod.ExactCosineIndex.build,
    )
    store.invalidate_exact_index()
    store.exact_top_k(cue, k=10)
    assert select_calls["n"] == 2, (
        "invalidate_exact_index() must clear the cooldown so the next call "
        "retries immediately"
    )


# ---------------------------------------------------------------------------
# S11 — build/invalidate TOCTOU: a race-losing build must not install stale
# ---------------------------------------------------------------------------


def test_s11_build_refuses_stale_install_when_invalidate_races_the_select(
    store: MemoryStore, monkeypatch
) -> None:
    """Simulate the TOCTOU window: ``exact_top_k`` snapshots the generation,
    runs its SELECT, and — before ``build()`` installs — a concurrent
    destructive write calls ``invalidate_exact_index()``. The build must
    refuse to commit the stale pre-invalidate snapshot as warm."""
    recs = [_make_rec(820 + i) for i in range(4)]
    for r in recs:
        store.insert(r)
    flush_record_buffer(store)

    target = recs[0]
    cue = target.embedding

    real_conn = store.db._conn
    fired = {"invalidate": False}

    def _race_on_select(sql: str) -> None:
        # Simulate a concurrent delete/erasure landing between the SELECT and
        # build(): tombstone the target directly, then fire the same
        # invalidate seam the production write path fires. Always applied
        # through the real writer connection, regardless of which connection
        # served the triggering SELECT.
        if "SELECT id, embedding FROM records" in sql and not fired["invalidate"]:
            fired["invalidate"] = True
            with store.db._conn_lock:
                real_conn.execute(
                    "UPDATE records SET tombstoned_at = ? WHERE id = ?",
                    (datetime.now(timezone.utc).isoformat(), str(target.id)),
                )
                real_conn.commit()
            store.invalidate_exact_index()

    class _RacingProxy:
        def __init__(self, target):
            object.__setattr__(self, "_target", target)

        def execute(self, sql, *args, **kwargs):
            result = self._target.execute(sql, *args, **kwargs)
            _race_on_select(sql)
            return result

        def __getattr__(self, name: str):
            return getattr(self._target, name)

        def __setattr__(self, name: str, value) -> None:
            setattr(self._target, name, value)

    monkeypatch.setattr(store.db, "_conn", _RacingProxy(real_conn))

    # The whole-corpus build fetch may run on a dedicated snapshot reader
    # (see `_build_exact_index_sync`) instead of `store.db._conn` — wrap that
    # seam too so the race can land on the SELECT regardless of which
    # connection actually serves it.
    import iai_mcp.hippo._raw_open as _raw_open_mod

    real_open_store_conn = _raw_open_mod.open_store_conn

    def _racing_open_store_conn(path, *, read_only=False):
        conn = real_open_store_conn(path, read_only=read_only)
        if conn is None:
            return None
        return _RacingProxy(conn)

    monkeypatch.setattr(_raw_open_mod, "open_store_conn", _racing_open_store_conn)

    result = store.exact_top_k(cue, k=10)

    # The race-losing build refused to install (generation mismatch), so the
    # index is either cold (falls through to [] this call) or was rebuilt
    # fresh by a subsequent internal retry — either way the target id (now
    # tombstoned) must never surface, and no exception propagates.
    result_ids = {str(uid) for uid, _s in result}
    assert str(target.id) not in result_ids

    # A follow-up call rebuilds cleanly from the now-current (post-tombstone)
    # state and matches the live-SQL reference exactly.
    _assert_matches_reference(store, cue)
    post_ids = {str(uid) for uid, _s in store.exact_top_k(cue, k=10)}
    assert str(target.id) not in post_ids


# ---------------------------------------------------------------------------
# S12 — query_similar over-fetch: tombstoned ANN neighbors must not underfill k
# ---------------------------------------------------------------------------


def test_s12_query_similar_still_returns_k_live_results_after_tombstoning(
    store: MemoryStore,
) -> None:
    """Erasure/dedupe-style tombstoning is a plain SQL UPDATE with no
    ``hnsw.mark_deleted`` call — the vectors stay in the ANN index until the
    next sleep-cycle rebuild. ``query_similar(k)`` must still return k live
    results by over-fetching past the tombstoned neighbors, not silently
    underfill."""
    k = 5
    # An identical embedding across all rows makes every row an equally close
    # ANN neighbor of the cue, so knn_query(k) returning ANY k of them is
    # representative of the near-neighbor-tombstoned scenario the finding
    # describes.
    shared_vec = _seeded_vec(900)
    recs = [_make_rec(900 + i, embedding=shared_vec) for i in range(k + 3)]
    for r in recs:
        store.insert(r)
    flush_record_buffer(store)

    # Tombstone via a plain SQL UPDATE — mirrors the erasure/dedupe write
    # path exactly: NO hnsw.mark_deleted call, so the vectors remain live in
    # the ANN index until the next rebuild.
    to_tombstone = recs[:3]
    now = datetime.now(timezone.utc)
    with store.db._conn_lock:
        for r in to_tombstone:
            store.db._conn.execute(
                "UPDATE records SET tombstoned_at = ? WHERE id = ?",
                (now.isoformat(), str(r.id)),
            )
        store.db._conn.commit()

    results = store.query_similar(shared_vec, k=k)

    assert len(results) == k, (
        f"query_similar(k={k}) underfilled after tombstoning near neighbors "
        f"with no hnsw.mark_deleted call: got {len(results)} results"
    )
    result_ids = {str(rec.id) for rec, _score in results}
    tombstoned_ids = {str(r.id) for r in to_tombstone}
    assert not (result_ids & tombstoned_ids), (
        "query_similar must never return a tombstoned id even though it "
        f"remains in the raw ANN index: overlap={result_ids & tombstoned_ids}"
    )


# ---------------------------------------------------------------------------
# S13 — non-finite coercion telemetry (store emit sites)
# ---------------------------------------------------------------------------


def test_s13_coerced_cue_emits_telemetry_and_recall_survives(
    store: MemoryStore, monkeypatch
) -> None:
    """A NaN/inf cue at ``exact_top_k`` is coerced, the store emits exactly
    one ``TELEMETRY_EMBED_NONFINITE`` (source=cue, action=coerced,
    severity=warning), and recall still returns a list with no NaN scores."""
    recs = [_make_rec(1000 + i) for i in range(6)]
    for r in recs:
        store.insert(r)
    flush_record_buffer(store)

    # Warm the index with a healthy cue first — a cold index would build
    # inside this same call and could double-count the coercion.
    store.exact_top_k(recs[0].embedding, k=5)
    assert store._exact_index.is_warm is True

    import iai_mcp.events as _events
    from iai_mcp.events import TELEMETRY_EMBED_NONFINITE

    original_emit = _events.emit_best_effort
    captured: list[dict] = []

    def _spy(store_arg, kind, data, **kwargs):
        captured.append({"kind": kind, "data": data, "severity": kwargs.get("severity")})
        return original_emit(store_arg, kind, data, **kwargs)

    monkeypatch.setattr(_events, "emit_best_effort", _spy)

    nan_cue = np.full(EMBED_DIM, np.nan, dtype=np.float32).tolist()
    result = store.exact_top_k(nan_cue, k=5)
    assert isinstance(result, list)
    for _uid, score in result:
        assert score == score  # not NaN

    cue_events = [
        c for c in captured
        if c["kind"] == TELEMETRY_EMBED_NONFINITE and c["data"].get("action") == "coerced"
    ]
    assert len(cue_events) == 1
    assert cue_events[0]["data"]["source"] == "cue"
    assert cue_events[0]["severity"] == "warning"


def test_s13_coerced_row_at_cold_build_emits_telemetry_and_isolates_scores(
    store: MemoryStore, monkeypatch
) -> None:
    """A NaN row that reaches SQL WITHOUT the write-path guard (a stub
    embedder over ``reembed_pending_rows``, mirroring a legacy pre-guard
    row) is coerced at the first cold build; the store emits exactly one
    event (source=row, context=build) and the coerced row does not corrupt
    the other ids' scores."""
    recs = [_make_rec(1100 + i) for i in range(6)]
    for r in recs:
        store.insert(r)
    flush_record_buffer(store)
    assert store._exact_index.is_warm is False

    pid = _insert_pending(store, "legacy bad row text")
    embedder = _StubEmbedder([float("nan")] * EMBED_DIM)
    n = store.db.reembed_pending_rows(embedder)
    assert n == 1
    # The feed callback fires on the still-cold index — upsert is a no-op
    # while cold, so the bad row is picked up only by the first build below.
    assert store._exact_index.is_warm is False

    import iai_mcp.events as _events
    from iai_mcp.events import TELEMETRY_EMBED_NONFINITE

    original_write = _events.write_event
    captured: list[dict] = []

    def _spy(store_arg, kind, data, **kwargs):
        captured.append({"kind": kind, "data": data, "severity": kwargs.get("severity")})
        return original_write(store_arg, kind, data, **kwargs)

    monkeypatch.setattr(_events, "write_event", _spy)

    cue = recs[0].embedding
    got = store.exact_top_k(cue, k=10)
    assert store._exact_index.is_warm is True
    got_scores = {str(uid): score for uid, score in got}
    assert got_scores[pid] == 0.0

    cue_arr = np.asarray(cue, dtype=np.float32)
    cue_unit = cue_arr / np.linalg.norm(cue_arr)
    for r in recs:
        rec_vec = np.asarray(r.embedding, dtype=np.float32)
        rec_unit = rec_vec / np.linalg.norm(rec_vec)
        expected_score = float(rec_unit @ cue_unit)
        assert got_scores[str(r.id)] == pytest.approx(expected_score, abs=1e-5)

    row_events = [
        c for c in captured
        if c["kind"] == TELEMETRY_EMBED_NONFINITE and c["data"].get("action") == "coerced"
    ]
    assert len(row_events) == 1
    assert row_events[0]["data"]["source"] == "row"
    assert row_events[0]["data"]["context"] == "build"
    assert row_events[0]["severity"] == "warning"


def test_s13_coerced_row_via_feed_upsert_is_buffered_not_immediate(
    store: MemoryStore,
) -> None:
    """The feed/upsert coercion emit is buffered, never immediate: NOT
    queryable right after the feed call, queryable only after
    ``flush_event_buffer``. This is the proof of the hard "never immediate
    at the feed site" constraint."""
    recs = [_make_rec(1200 + i) for i in range(4)]
    for r in recs:
        store.insert(r)
    flush_record_buffer(store)

    store.exact_top_k(recs[0].embedding, k=5)  # warm the index first
    assert store._exact_index.is_warm is True

    from iai_mcp.events import (
        TELEMETRY_EMBED_NONFINITE,
        flush_event_buffer,
        query_events,
    )

    before_total = store._exact_index.coerced_row_total
    nan_vec = [float("nan")] * EMBED_DIM
    store._feed_exact_index("feed-nan-id", nan_vec)
    assert store._exact_index.coerced_row_total == before_total + 1

    def _coerced_upsert_events() -> list[dict]:
        events = query_events(store, kind=TELEMETRY_EMBED_NONFINITE)
        return [
            e for e in events
            if (e.get("data") or {}).get("action") == "coerced"
            and (e.get("data") or {}).get("context") == "upsert"
        ]

    assert not _coerced_upsert_events(), (
        "the feed-site emit must not be immediately queryable — it is "
        "buffered, not written through _conn_lock synchronously"
    )

    flush_event_buffer(store)

    after_flush = _coerced_upsert_events()
    assert len(after_flush) == 1
    assert after_flush[0]["data"]["source"] == "row"
    assert after_flush[0]["severity"] == "warning"


def test_s13_emit_failure_isolated_from_recall_build_and_feed(
    store: MemoryStore, monkeypatch
) -> None:
    """A raising write_event/emit_best_effort at any of the three sites must
    never break the caller: cue recall still returns a list, a build with a
    coerced row still installs and warms, feed upsert still returns."""
    recs = [_make_rec(1300 + i) for i in range(6)]
    for r in recs:
        store.insert(r)
    flush_record_buffer(store)

    import iai_mcp.events as _events

    def _boom(*_a, **_kw):
        raise RuntimeError("simulated event-store outage")

    monkeypatch.setattr(_events, "write_event", _boom)
    monkeypatch.setattr(_events, "emit_best_effort", _boom)

    nan_cue = np.full(EMBED_DIM, np.nan, dtype=np.float32).tolist()
    result = store.exact_top_k(nan_cue, k=5)
    assert isinstance(result, list)
    assert store._exact_index.is_warm is True

    _insert_pending(store, "outage-era pending text")
    embedder = _StubEmbedder([float("nan")] * EMBED_DIM)
    n = store.db.reembed_pending_rows(embedder)
    assert n == 1
    store.invalidate_exact_index()
    got = store.exact_top_k(recs[0].embedding, k=10)
    assert isinstance(got, list)
    assert store._exact_index.is_warm is True

    store._feed_exact_index("feed-nan-outage", [float("nan")] * EMBED_DIM)  # must not raise
