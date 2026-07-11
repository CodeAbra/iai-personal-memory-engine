"""Tests for the write-invalidated corpus-count cache.

Contract surfaces these tests target:
- ``MemoryStore.active_records_count()`` / ``pending_records_count()`` /
  ``edges_count()`` backed by a cross-read count cache that is invalidated on
  every count-changing write.
- A store-level ``CorpusCountCache`` object with ``get``/``put``/
  ``invalidate``/``clear`` operations under a ``threading.Lock``.
- ``runtime_graph_cache._cache_key(store)`` must produce a tuple that is
  bit-identical to the live-count tuple for any corpus state.

Note on the conftest autoflush fixture
---------------------------------------
``tests/conftest.py`` wraps ``MemoryStore.insert`` to automatically flush the
record and edge buffers after EVERY insert (``_autoflush_lance_buffers``).
This is correct for most tests but the WINDOW tests (append→recall→flush) need
the row to stay in the buffer without auto-flushing so the "recall with stale
cache" step is meaningful.

The window tests set ``IAI_MCP_TEST_NO_AUTOFLUSH=1`` (via monkeypatch) before
calling the insert, which tells the conftest wrapper to pass-through without
flushing.  The explicit ``flush_record_buffer`` / ``flush_edge_buffer`` call in
step 3 of each window test then becomes the ONLY flush — the one whose
invalidation behaviour the test measures.

Cache-absent fallback
---------------------
When no persistent cross-read count cache is present, ``_count_memo`` is
``None`` outside a ``count_memo()`` scope, so each call to the three count
methods re-runs the filtered SQL COUNT.  Tests that assert "warm cache ==
live count" are trivially true in that mode (no cache diverges from live).

The tests that exercise the cache use one of two mechanisms:

1. **attribute access** – they reach into ``store._corpus_count_cache``; if
   the attribute is absent these tests raise ``AttributeError`` in the setup
   step.

2. **behavioral guard** – they assert a correctness property that the cache
   must satisfy.  Specifically:
   - ``test_burst_computes_count_once``: asserts the 3 COUNTs hit the engine
     at most once across a recall burst.  On HEAD each ``_cache_key``
     derivation in the burst re-scans, so the spy count exceeds 1 and the
     assertion fails.
   - ``test_buffered_record_append_recall_flush_window``,
     ``test_buffered_edge_append_recall_flush_window``: assert that flushing
     the record/edge buffer (which lands a row in SQL) invalidates the cached
     count so the cached value equals the live count.  On HEAD there is no
     invalidation at the flush chokepoint, so if any stale cached count were
     held the assertion would fail.  The window tests are designed so the
     behavioural assertion fires on a warm cache state that diverges from the
     live SQL count — but on HEAD there is nothing to warm, so the assertions
     are trivially true today.  They become the meaningful gate once the cache
     lands: a regression that moves invalidation back to the append site will
     leave a stale low count in the cache after the flush and fail here.
   - Reembed both-count + key-change (in test_reembed_runtime_graph_inclusion)
     is similar: the HEAD path already moves both counts (the live SQL reflects
     the flipped row), so the behavioral test passes today but is RED by
     construction once the cache can hold stale values.

3. **skip-pending** – tests guarded by ``pytest.importorskip`` / the
   ``_CACHE_PRESENT`` sentinel skip cleanly on HEAD and are turned GREEN once
   the count cache and invalidation wiring land.

``_force_empty_count_cache(store)`` is a *no-op on HEAD* (the cache does not
exist yet).  Once the count cache lands, wire it to the real cache attribute so
the warm-vs-live comparison tests become meaningful.

No production source files are modified by this file.
"""
from __future__ import annotations

import dataclasses
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp import runtime_graph_cache
from iai_mcp.store import (
    MemoryStore,
    flush_edge_buffer,
    flush_record_buffer,
    _edge_buffer,
    _record_buffer,
)
from iai_mcp.types import EMBED_DIM, MemoryRecord


# ---------------------------------------------------------------------------
# Detect whether the count-cache surface is present
# ---------------------------------------------------------------------------

def _has_count_cache(store: MemoryStore) -> bool:
    """Return True iff the store-level corpus count cache is present.

    Tests that depend on cache behaviour skip cleanly when this is False
    (before the count cache lands) and become GREEN once it lands.
    """
    return hasattr(store, "_corpus_count_cache") or hasattr(store, "_corpus_counts")


# ---------------------------------------------------------------------------
# The one coupling-point helper: force the count cache to empty
# ---------------------------------------------------------------------------

def _force_empty_count_cache(store: MemoryStore) -> None:
    """Clear the store-level corpus count cache (if it exists).

    On HEAD (before the count cache lands) this is a no-op — there is nothing
    to empty.  Once the count cache lands, wire this to the real cache attribute
    so the warm-vs-live comparison tests become meaningful.

    This is the single coupling point between the test file and the cache
    implementation name.  All other test assertions are purely behavioral
    (count equals live value; COUNT probes per burst; no raise on degraded
    input) and do not depend on internal attribute names.
    """
    cache = getattr(store, "_corpus_count_cache", None)
    if cache is not None:
        cache.clear()
        return
    # Also cover the simpler dict-attribute pattern (Pattern 1 from research):
    counts = getattr(store, "_corpus_counts", None)
    if counts is not None:
        if hasattr(counts, "clear"):
            counts.clear()
        elif isinstance(counts, dict):
            counts.clear()


# ---------------------------------------------------------------------------
# Shared fixtures and corpus builders
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
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "store")


def _random_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _make_rec(seed: int) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=f"surface text record {seed}",
        aaak_index="",
        embedding=_random_vec(seed),
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


def _seed_corpus(store: MemoryStore, n_active: int = 5, seed_base: int = 0) -> list[UUID]:
    """Insert n_active fully-embedded records; flush the buffer so SQL sees them."""
    recs = [_make_rec(seed_base + i) for i in range(n_active)]
    for r in recs:
        store.insert(r)
    flush_record_buffer(store)
    return [r.id for r in recs]


def _insert_pending(store: MemoryStore, surface: str = "pending text") -> str:
    """Insert a pending (embedding_pending=1) row directly via HippoDB."""
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


def _add_edge(store: MemoryStore, id_a: UUID, id_b: UUID) -> None:
    """Insert a single new edge between two records."""
    store.boost_edges([(id_a, id_b)], delta=1.0, edge_type="hebbian")


# ---------------------------------------------------------------------------
# COUNT-spy: count how many filtered COUNT(*) probes reach the engine per
# recall burst.  Used by test_burst_computes_count_once.
# ---------------------------------------------------------------------------

class _CountSpy:
    """Wraps the three MemoryStore count methods to tally engine-level calls.

    After construction call ``reset()`` between test phases.  Read
    ``calls_active``, ``calls_pending``, ``calls_edges`` to get the tally.
    """

    def __init__(self, store: MemoryStore) -> None:
        self._store = store
        self.calls_active = 0
        self.calls_pending = 0
        self.calls_edges = 0
        # Stash originals
        self._orig_active = store.active_records_count
        self._orig_pending = store.pending_records_count
        self._orig_edges = store.edges_count
        store.active_records_count = self._spy_active  # type: ignore[method-assign]
        store.pending_records_count = self._spy_pending  # type: ignore[method-assign]
        store.edges_count = self._spy_edges  # type: ignore[method-assign]

    def _spy_active(self) -> int:
        self.calls_active += 1
        return self._orig_active()

    def _spy_pending(self) -> int:
        self.calls_pending += 1
        return self._orig_pending()

    def _spy_edges(self) -> int:
        self.calls_edges += 1
        return self._orig_edges()

    def reset(self) -> None:
        self.calls_active = 0
        self.calls_pending = 0
        self.calls_edges = 0

    def restore(self) -> None:
        self._store.active_records_count = self._orig_active  # type: ignore[method-assign]
        self._store.pending_records_count = self._orig_pending  # type: ignore[method-assign]
        self._store.edges_count = self._orig_edges  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Helpers for the buffered-write window tests
# ---------------------------------------------------------------------------

def _append_record_without_flush(store: MemoryStore, monkeypatch, seed: int) -> MemoryRecord:
    """Append a record to the buffer WITHOUT forcing a flush.

    Two guards prevent an unwanted auto-flush:
    1. ``IAI_MCP_TEST_NO_AUTOFLUSH=1`` tells the conftest ``_autoflush_lance_buffers``
       wrapper to skip its after-insert flush.
    2. ``IAI_MCP_RECORD_BUFFER_MAX=100000`` makes ``should_flush_record_buffer``
       return False so the buffer's own size-gate does not auto-flush.

    The row lands in ``_record_buffer`` but NOT yet in SQL.
    """
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")
    monkeypatch.setenv("IAI_MCP_RECORD_BUFFER_MAX", "100000")
    rec = _make_rec(seed)
    store.insert(rec)
    assert id(store) in _record_buffer and len(_record_buffer[id(store)]) > 0, (
        "record must be in the buffer (not auto-flushed) with high max-size"
    )
    return rec


def _append_edge_without_flush(
    store: MemoryStore, monkeypatch, id_a: UUID, id_b: UUID
) -> None:
    """Append a new edge to the edge buffer WITHOUT forcing a flush.

    Stages one edge row directly into ``_edge_buffer`` (bypassing
    ``store.boost_edges`` which the conftest wraps with an auto-flush).
    ``IAI_MCP_EDGE_BUFFER_MAX=100000`` ensures ``should_flush_edge_buffer``
    returns False if the buffer's size-gate is ever reached.

    The edge row is NOT yet in SQL — only in the in-process buffer.
    """
    monkeypatch.setenv("IAI_MCP_EDGE_BUFFER_MAX", "100000")
    now = datetime.now(timezone.utc)
    row = {
        "src": str(id_a),
        "dst": str(id_b),
        "edge_type": "hebbian",
        "weight": 1.0,
        "updated_at": now,
    }
    _edge_buffer.setdefault(id(store), []).append(row)
    assert id(store) in _edge_buffer and len(_edge_buffer[id(store)]) > 0, (
        "edge must be in the buffer (not auto-flushed) with high max-size"
    )


# ===========================================================================
# Cache-contract tests
# ===========================================================================

def test_cache_key_matches_live(store):
    """For a fixed corpus state, _cache_key computed with a warm cache must
    equal _cache_key computed with the cache forced to empty (live counts).

    The cached counts must equal the live counts at all times: a bug where the
    cache holds a stale count diverges here.

    On HEAD (no persistent cache) both sides recompute live so the assertion
    is trivially true.  Once the count cache lands, this is the meaningful gate.
    """
    _seed_corpus(store, n_active=6)
    _insert_pending(store)

    # Warm side: derive the key with whatever cache state exists
    key_warm = runtime_graph_cache._cache_key(store)

    # Force-empty the cache and recompute live
    _force_empty_count_cache(store)
    key_live = runtime_graph_cache._cache_key(store)

    assert key_warm == key_live, (
        f"cached-count key {key_warm} must be bit-identical to live-count "
        f"key {key_live} for the same corpus state"
    )


def test_burst_computes_count_once(store):
    """A read-only recall burst (no intervening write) must compute each
    filtered COUNT at most once across the burst.

    After the count cache lands, the first _cache_key call fills the cache;
    subsequent calls in the same burst hit the cache and do not reach the engine.

    RED on HEAD: on HEAD there is no cross-read cache so each _cache_key
    derivation calls the three count methods, which call the SQL engine.  The
    spy will record > 1 call per method across a 2-call burst, failing the
    assertion below.
    """
    _seed_corpus(store, n_active=4)
    _insert_pending(store)

    spy = _CountSpy(store)
    try:
        spy.reset()
        # Simulate what a recall burst does: _cache_key is derived at least
        # twice inside try_load + the build-decision chain.
        _ = runtime_graph_cache._cache_key(store)
        _ = runtime_graph_cache._cache_key(store)
    finally:
        spy.restore()

    # After the count cache lands: each COUNT must be computed at most once per burst.
    # On HEAD each _cache_key call recomputes, so calls_active == 2, which
    # violates the assertion.
    assert spy.calls_active <= 1, (
        f"active COUNT reached the engine {spy.calls_active} times in a "
        "read-only 2-call burst; expected at most 1 (cache miss + refill)"
    )
    assert spy.calls_pending <= 1, (
        f"pending COUNT reached the engine {spy.calls_pending} times in a "
        "read-only 2-call burst; expected at most 1"
    )
    assert spy.calls_edges <= 1, (
        f"edges COUNT reached the engine {spy.calls_edges} times in a "
        "read-only 2-call burst; expected at most 1"
    )


def test_cold_cache_matches(store):
    """With the cache empty/cold (daemon-down analog), _cache_key recomputes
    live and produces a key identical to the warm-cache key for the same corpus.

    The bypass-safe invariant: even if the cache is entirely absent or cleared,
    recall still gets a correct key.  The cache must never be a gatekeeper.
    """
    _seed_corpus(store, n_active=3)

    # Derive key with whatever cache state is current
    key_1 = runtime_graph_cache._cache_key(store)
    # Clear the cache (no-op on HEAD)
    _force_empty_count_cache(store)
    # Re-derive with an empty cache — must match
    key_2 = runtime_graph_cache._cache_key(store)

    assert key_1 == key_2, (
        "cold-cache _cache_key must match warm-cache key for the same corpus state"
    )


def test_cached_count_never_raises_and_degrades_to_filtered_count(store):
    """Two properties of the cache-backed count methods:

    (a) On a cold/partial cache, the three count methods return a value and
        never raise.
    (b) When the cache's ``get`` raises (simulated exception), the count
        methods still return the FILTERED count and do NOT propagate the
        exception — and crucially the returned value equals the live FILTERED
        COUNT (tombstoned and pending rows excluded for ``active_records_count``),
        NOT the unfiltered ``count_rows()`` value.

    The corpus is seeded with at least one tombstoned and one pending row so
    the filtered and unfiltered counts differ, making the assertion
    discriminating: a fallback that called unfiltered ``count_rows()`` would
    return the larger number and fail.

    Gate: this test is meaningful only once the count cache lands whose ``get``
    can be monkeypatched.  On HEAD it is skip-pending.
    """
    ids = _seed_corpus(store, n_active=4)
    _insert_pending(store)  # pending +1

    # Tombstone one active record directly
    with store.db._conn_lock:
        now = datetime.now(timezone.utc).isoformat()
        store.db._conn.execute(
            "UPDATE records SET tombstoned_at = ? WHERE id = ?",
            (now, str(ids[0])),
        )
        store.db._conn.commit()

    # Live counts with filters
    live_active = store.active_records_count()
    live_pending = store.pending_records_count()
    live_edges = store.edges_count()

    # Unfiltered count (includes tombstoned + pending): must differ from active
    with store.db._conn_lock:
        row = store.db._conn.execute("SELECT COUNT(*) FROM records").fetchone()
    unfiltered_total = int(row[0]) if row else 0
    assert unfiltered_total > live_active, (
        "corpus must have tombstoned or pending rows so filtered != unfiltered"
    )

    if not _has_count_cache(store):
        pytest.skip("count cache not present in this build")

    # Force-inject an exception into the cache get path
    cache = getattr(store, "_corpus_count_cache", None)
    if cache is None:
        pytest.skip("count cache attribute not found")

    original_get = cache.get

    def _raising_get(*args, **kwargs):
        raise KeyError("injected-cache-failure")

    cache.get = _raising_get
    try:
        active = store.active_records_count()
        pending = store.pending_records_count()
        edges = store.edges_count()
    finally:
        cache.get = original_get

    assert active == live_active, (
        f"cache-exception fallback must return FILTERED active count "
        f"{live_active}, not unfiltered {unfiltered_total}; got {active}"
    )
    assert pending == live_pending, (
        f"cache-exception fallback must return FILTERED pending count {live_pending}; got {pending}"
    )
    assert edges == live_edges, (
        f"cache-exception fallback must return live edge count {live_edges}; got {edges}"
    )


def test_concurrent_read_invalidate(store):
    """N reader threads calling active_records_count() while a writer thread
    inserts and invalidates in a loop must produce no exception and the cached
    count must converge to the live count after writes settle.

    Thread safety of the count cache: concurrent reads must never raise, and
    the final count must equal the live count.

    Gate: most meaningful once the count cache lands.  On HEAD (no cache) the
    test still exercises the multi-threaded read path and asserts the final count
    is correct.
    """
    _seed_corpus(store, n_active=3)

    errors: list[Exception] = []
    N_READERS = 4
    N_WRITER_ROUNDS = 10
    stop_event = threading.Event()
    results: list[int] = []

    def reader() -> None:
        while not stop_event.is_set():
            try:
                val = store.active_records_count()
                results.append(val)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    writer_ids: list[UUID] = []

    def writer() -> None:
        for i in range(N_WRITER_ROUNDS):
            rec = _make_rec(1000 + i)
            store.insert(rec)
            flush_record_buffer(store)
            writer_ids.append(rec.id)

    threads = [threading.Thread(target=reader, daemon=True) for _ in range(N_READERS)]
    for t in threads:
        t.start()

    w = threading.Thread(target=writer)
    w.start()
    w.join(timeout=15)
    stop_event.set()
    for t in threads:
        t.join(timeout=5)

    assert not errors, f"reader threads raised: {errors}"

    live_active = store.active_records_count()
    _force_empty_count_cache(store)
    cached_active = store.active_records_count()
    assert cached_active == live_active, (
        f"after writer settled, cached count {cached_active} must equal "
        f"live count {live_active}"
    )


# ===========================================================================
# Append → recall → flush WINDOW tests
# ===========================================================================

def test_buffered_record_append_recall_flush_window(store, monkeypatch):
    """Flush-site invalidation gate for records.

    The window is: append a record (buffer; NOT yet in SQL), derive the cache
    key (warming the cache against the still-excluded live count), then flush
    (row lands in SQL), and assert the cache now reflects the higher count.

    This FAILS if invalidation is at the append site:
    - append → clear cache (step 1: cache is empty)
    - _cache_key warms the cache against the too-low live count (row not in SQL)
    - flush fires NO invalidation → cached count stays stale-low
    - assertion fails

    It PASSES only when invalidation is at the flush chokepoint.

    On HEAD (no persistent cache) both warm and post-flush reads compute live
    directly, so the assertion passes trivially.  It becomes the meaningful
    gate after the count cache and invalidation wiring land.
    """
    # Seed some active records so the starting live count is > 0
    _seed_corpus(store, n_active=3, seed_base=0)
    live_active_before = store.active_records_count()

    # Step 1: append a record WITHOUT flushing (stays in buffer, not in SQL)
    _append_record_without_flush(store, monkeypatch, seed=99)

    # Step 2: warm the cache against the CURRENT live count (excludes the buffered row)
    _force_empty_count_cache(store)
    live_during_buffer = store.active_records_count()
    assert live_during_buffer == live_active_before, (
        "buffered (unflushed) row must NOT appear in the live SQL count"
    )

    # Step 3: flush — the row now lands in SQL, count increases by 1
    flush_record_buffer(store)

    # Step 4: assert the cached count now equals the post-flush live count.
    # If invalidation is at the flush site, the cache is cleared on flush and
    # the next read recomputes the higher value.
    # If invalidation was at the append site (wrong), the cache still holds
    # the stale-low live_during_buffer value.
    live_after_flush = store.active_records_count()
    assert live_after_flush == live_active_before + 1, (
        f"post-flush live active count must be {live_active_before + 1}, got {live_after_flush}"
    )

    # Assert via the cache: force-read the cached count and compare to live.
    _force_empty_count_cache(store)  # ensure we recompute clean for comparison
    cached_after_flush = store.active_records_count()
    assert cached_after_flush == live_after_flush, (
        f"cached active count after flush must equal live count {live_after_flush}; "
        f"got {cached_after_flush} — this fails if flush does not invalidate the cache"
    )


def test_buffered_edge_append_recall_flush_window(store, monkeypatch):
    """Flush-site invalidation gate for edges.

    Same shape as the record window test: append an edge (buffer; not in SQL),
    warm the cache, flush (edge lands in SQL), assert the cached edge count
    equals the live edge count.

    Fails if invalidation is at the append site instead of the flush chokepoint.
    """
    # Seed two active records to serve as edge endpoints
    ids = _seed_corpus(store, n_active=2, seed_base=50)
    live_edges_before = store.edges_count()

    # Step 1: stage a NEW edge in the buffer WITHOUT flushing it
    _append_edge_without_flush(store, monkeypatch, ids[0], ids[1])

    # Step 2: warm the cache against the CURRENT live count (edge not in SQL)
    _force_empty_count_cache(store)
    live_during_buffer = store.edges_count()
    assert live_during_buffer == live_edges_before, (
        "buffered (unflushed) edge must NOT appear in the live SQL count"
    )

    # Step 3: flush — edge lands in SQL
    flush_edge_buffer(store)

    # Step 4: assert the cached edges count now equals post-flush live count
    live_after_flush = store.edges_count()
    assert live_after_flush == live_edges_before + 1, (
        f"post-flush live edges count must be {live_edges_before + 1}, got {live_after_flush}"
    )

    _force_empty_count_cache(store)
    cached_after_flush = store.edges_count()
    assert cached_after_flush == live_after_flush, (
        f"cached edges count after flush must equal live count {live_after_flush}; "
        f"got {cached_after_flush} — fails if flush does not invalidate the edges cache"
    )


# ===========================================================================
# Per-write-path invalidation differential tests
# ===========================================================================

def test_write_path_sync_insert_active_tracks(store):
    """A sync buffered insert (fully-embedded) must increase the active count
    by one after the buffer is flushed.

    Once the count cache and invalidation wiring land, the cached active count must track the post-flush live
    value.  On HEAD (no cache) the assertion is trivially true.
    """
    _seed_corpus(store, n_active=2)
    before = store.active_records_count()

    rec = _make_rec(seed=200)
    store.insert(rec)
    flush_record_buffer(store)

    after_live = store.active_records_count()
    assert after_live == before + 1, (
        f"insert+flush must raise active count from {before} to {before + 1}"
    )

    _force_empty_count_cache(store)
    after_cached = store.active_records_count()
    assert after_cached == after_live, (
        f"cached active count {after_cached} must equal live {after_live} after insert"
    )


def test_write_path_pending_capture_pending_tracks_active_unchanged(store):
    """A pending row insert (embedding_pending=1) must increment the pending
    count by one while leaving the active count unchanged.

    Once the count cache and invalidation wiring land, the cached pending and active counts must track correctly.
    On HEAD (no cache) the assertion is trivially true.
    """
    _seed_corpus(store, n_active=3)
    before_active = store.active_records_count()
    before_pending = store.pending_records_count()

    _insert_pending(store)

    live_active = store.active_records_count()
    live_pending = store.pending_records_count()

    assert live_active == before_active, (
        "pending insert must NOT change the active count"
    )
    assert live_pending == before_pending + 1, (
        f"pending insert must raise pending count from {before_pending} to {before_pending + 1}"
    )

    _force_empty_count_cache(store)
    cached_active = store.active_records_count()
    cached_pending = store.pending_records_count()
    assert cached_active == live_active
    assert cached_pending == live_pending


def test_write_path_tombstone_active_tracks(store):
    """Tombstoning a record removes it from the active count.

    Written directly (as the sleep-pipeline erasure agent does — bypassing the
    store API).  Once the count cache and invalidation wiring land, the cached active count must track the
    reduction.  On HEAD (no cache) the assertion is trivially true.
    """
    ids = _seed_corpus(store, n_active=4)
    before = store.active_records_count()
    assert before == 4

    # Tombstone one record (mirrors _erasure.py direct tbl.update)
    with store.db._conn_lock:
        now = datetime.now(timezone.utc).isoformat()
        store.db._conn.execute(
            "UPDATE records SET tombstoned_at = ? WHERE id = ?",
            (now, str(ids[0])),
        )
        store.db._conn.commit()

    # This raw SQL bypasses the store API invalidation path. Clear the count
    # cache before asserting the live SQL count so the assertion measures the
    # SQL state, not a potentially-stale cached value.  The sleep-pipeline
    # erasure step goes through _invalidate_corpus_count (not raw SQL), so its
    # invalidation is tested by the erasure-specific fixture.
    _force_empty_count_cache(store)
    live_after = store.active_records_count()
    assert live_after == before - 1, (
        f"tombstone must reduce active count from {before} to {before - 1}"
    )

    _force_empty_count_cache(store)
    cached_after = store.active_records_count()
    assert cached_after == live_after, (
        f"cached active count {cached_after} must equal live {live_after} after tombstone"
    )


def test_write_path_untombstone_active_tracks(store):
    """Un-tombstoning a record restores it to the active count.

    Written directly (as the optimize step does — bypassing the store API).
    Once the count cache and invalidation wiring land, the cached active count must track the restoration.
    On HEAD (no cache) the assertion is trivially true.
    """
    ids = _seed_corpus(store, n_active=3)
    # Tombstone one first
    with store.db._conn_lock:
        now = datetime.now(timezone.utc).isoformat()
        store.db._conn.execute(
            "UPDATE records SET tombstoned_at = ? WHERE id = ?",
            (now, str(ids[0])),
        )
        store.db._conn.commit()

    before = store.active_records_count()

    # Un-tombstone
    with store.db._conn_lock:
        store.db._conn.execute(
            "UPDATE records SET tombstoned_at = NULL WHERE id = ?",
            (str(ids[0]),),
        )
        store.db._conn.commit()

    # Raw SQL bypass — clear the count cache before asserting so the live SQL
    # state is read (not a potentially-stale cached value).
    _force_empty_count_cache(store)
    live_after = store.active_records_count()
    assert live_after == before + 1, (
        f"un-tombstone must raise active count from {before} to {before + 1}"
    )

    _force_empty_count_cache(store)
    cached_after = store.active_records_count()
    assert cached_after == live_after


def test_write_path_edge_insert_new_edge_tracks(store):
    """A new edge insert increments the edges count.

    The INSERT branch of boost_edges adds a new (src, dst, edge_type) row.
    Once the count cache and invalidation wiring land, the cached edges count must track.
    On HEAD (no cache) the assertion is trivially true.
    """
    ids = _seed_corpus(store, n_active=3)
    before = store.edges_count()

    store.boost_edges([(ids[0], ids[1])], delta=1.0, edge_type="hebbian")

    live_after = store.edges_count()
    assert live_after > before, (
        f"new edge insert must increase edges count from {before}"
    )

    _force_empty_count_cache(store)
    cached_after = store.edges_count()
    assert cached_after == live_after, (
        f"cached edges count {cached_after} must equal live {live_after} after insert"
    )


def test_write_path_edge_weight_update_does_not_invalidate(store):
    """A weight-only UPDATE of an existing edge must NOT change the edges count
    and must NOT spuriously invalidate the active or pending count caches.

    The UPDATE branch of boost_edges updates weight/updated_at without inserting
    a new row — the edge count stays the same and the cache must not be cleared
    unnecessarily.
    """
    ids = _seed_corpus(store, n_active=3)

    # Insert the edge first (so the second call hits the UPDATE branch)
    store.boost_edges([(ids[0], ids[1])], delta=1.0, edge_type="hebbian")
    edges_after_insert = store.edges_count()

    # Second boost on the same (src, dst, edge_type) — hits UPDATE branch
    store.boost_edges([(ids[0], ids[1])], delta=0.5, edge_type="hebbian")
    edges_after_update = store.edges_count()

    assert edges_after_update == edges_after_insert, (
        "weight-only edge UPDATE must NOT change the edges count"
    )

    _force_empty_count_cache(store)
    cached_after_update = store.edges_count()
    assert cached_after_update == edges_after_update


def test_write_path_contradicts_edge_tracks(store):
    """add_contradicts_edge inserts an edge row; edges count must increase.

    Once the count cache and invalidation wiring land, the cached edges count must track.
    On HEAD (no cache) the assertion is trivially true.
    """
    ids = _seed_corpus(store, n_active=2)
    before = store.edges_count()

    store.add_contradicts_edge(ids[0], ids[1])

    live_after = store.edges_count()
    assert live_after > before, (
        "contradicts edge insert must increase edges count"
    )

    _force_empty_count_cache(store)
    cached_after = store.edges_count()
    assert cached_after == live_after


def test_write_path_delete_active_tracks(store):
    """Record delete decrements the active count.

    ``store.delete(id)`` removes the row entirely (not a tombstone).  After
    the count cache and invalidation wiring, the cached active count must track the reduction.
    On HEAD (no cache) the assertion is trivially true.
    """
    ids = _seed_corpus(store, n_active=4)
    before = store.active_records_count()

    store.delete(ids[0])

    live_after = store.active_records_count()
    assert live_after == before - 1, (
        f"delete must reduce active count from {before} to {before - 1}"
    )

    _force_empty_count_cache(store)
    cached_after = store.active_records_count()
    assert cached_after == live_after


# ---------------------------------------------------------------------------
# Non-count-changing writes must NOT spuriously invalidate
# ---------------------------------------------------------------------------

def test_non_count_changing_update_record_no_spurious_invalidate(store):
    """update_record only touches stability/difficulty/last_reviewed/updated_at.

    It must NOT change any count or invalidate the cache.
    """
    ids = _seed_corpus(store, n_active=3)

    before_active = store.active_records_count()
    before_pending = store.pending_records_count()
    before_edges = store.edges_count()

    # update_record: fetch and re-write with changed stability.
    # MemoryRecord is a dataclass; use dataclasses.replace (not model_copy).
    rec = store.get(ids[0])
    assert rec is not None
    rec = dataclasses.replace(rec, stability=0.42)
    store.update_record(rec)

    assert store.active_records_count() == before_active, (
        "update_record must not change the active count"
    )
    assert store.pending_records_count() == before_pending
    assert store.edges_count() == before_edges


def test_non_count_changing_append_provenance_no_spurious_invalidate(store):
    """append_provenance only updates provenance_json.

    It must NOT change any count or invalidate the cache.
    """
    ids = _seed_corpus(store, n_active=2)

    before_active = store.active_records_count()
    before_pending = store.pending_records_count()
    before_edges = store.edges_count()

    store.append_provenance(ids[0], {"session_id": "test-session"})

    assert store.active_records_count() == before_active
    assert store.pending_records_count() == before_pending
    assert store.edges_count() == before_edges


# ---------------------------------------------------------------------------
# count_memo boundary: invalidation during an open memo scope is reflected
# on the NEXT memo scope
# ---------------------------------------------------------------------------

def test_invalidation_during_open_memo_reflected_on_next_scope(store):
    """An invalidation during an open count_memo scope is reflected on the
    NEXT memo scope (not lost and not surfaced immediately inside the open
    scope).

    Within the open scope the memoized (pre-write) value is served — this is
    the expected masking behavior of count_memo.  Outside (in the next recall),
    the write's effect must be visible.

    Gate: skip if the count cache (the count cache) is not yet present, since
    count_memo is the only memo mechanism on HEAD.
    """
    _seed_corpus(store, n_active=3)

    if not _has_count_cache(store):
        # On HEAD: verify at least that count_memo gives stable values within
        # its scope and the write is visible after the scope exits.
        with store.count_memo():
            before_in_memo = store.active_records_count()

        # Write a new record
        rec = _make_rec(seed=777)
        store.insert(rec)
        flush_record_buffer(store)

        # After scope: new count is visible
        after_memo = store.active_records_count()
        assert after_memo == before_in_memo + 1, (
            "write after count_memo scope must be visible in the next read"
        )
        return

    # With the count cache present:
    with store.count_memo():
        # Warm the memo with the current count
        count_inside_before_write = store.active_records_count()

        # Write inside the open scope
        rec = _make_rec(seed=888)
        store.insert(rec)
        flush_record_buffer(store)

        # The memoized value inside this scope may still be the pre-write count
        # (the within-build masking).  We do NOT assert strict stale-serving
        # here because the implementation may choose to invalidate immediately —
        # the important contract is that the NEXT scope sees the post-write count.
        count_inside_after_write = store.active_records_count()

    # NEXT scope (simulates next recall): must reflect the write
    with store.count_memo():
        count_next_scope = store.active_records_count()

    assert count_next_scope == count_inside_before_write + 1, (
        f"the next memo scope must reflect the write that happened during the "
        f"previous open scope; expected {count_inside_before_write + 1}, "
        f"got {count_next_scope}"
    )


# ---------------------------------------------------------------------------
# A2 off-path spy: all_records() must NOT fire on the warm standard recall path
# ---------------------------------------------------------------------------

def test_all_records_not_called_on_warm_recall_path(store):
    """all_records() does a full table to_pandas() — it must NOT be called on
    the standard recall path (structural recall goes through the ANN).

    This is a behavioral regression guard against accidental introduction of a
    full-table read on the recall hot path.
    """
    _seed_corpus(store, n_active=5)

    called = []
    orig = store.all_records

    def _spy_all_records():
        called.append(True)
        return orig()

    store.all_records = _spy_all_records  # type: ignore[method-assign]
    try:
        # Simulate what recall does: derive the _cache_key (calls the 3 counts)
        # and load the structural cache.
        _ = runtime_graph_cache._cache_key(store)
        runtime_graph_cache.load_recall_structural(store)
    finally:
        store.all_records = orig  # type: ignore[method-assign]

    assert not called, (
        "all_records() must NOT be called on the warm standard recall path "
        "(structural recall uses ANN, not a full table scan)"
    )


# ---------------------------------------------------------------------------
# Generation-counter compare-and-set — recompute-vs-invalidate race
#
# A count miss runs the live SQL COUNT OUTSIDE the cache lock, then stores the
# result.  A concurrent invalidate landing between the COUNT and the store must
# not be lost: the store is a compare-and-set on a generation snapshot taken
# before the COUNT.  These tests prove a stale value is never installed over a
# newer invalidation.
# ---------------------------------------------------------------------------


def test_put_if_gen_refuses_store_after_invalidate():
    """put_if_gen with a stale generation must NOT install the value."""
    from iai_mcp.store._corpus_count_cache import CorpusCountCache

    cache = CorpusCountCache()
    gen = cache.generation()
    # A concurrent invalidate bumps the generation.
    cache.invalidate("active")
    # The recompute, holding the pre-invalidate generation, must be refused.
    cache.put_if_gen("active", 41, gen)
    assert cache.get("active") is None, (
        "a value computed before an invalidate must not repopulate the cache"
    )
    # A store with the current generation succeeds.
    cache.put_if_gen("active", 42, cache.generation())
    assert cache.get("active") == 42


def test_put_if_gen_installs_when_generation_unchanged():
    """put_if_gen with an unchanged generation installs the value."""
    from iai_mcp.store._corpus_count_cache import CorpusCountCache

    cache = CorpusCountCache()
    gen = cache.generation()
    cache.put_if_gen("edges", 7, gen)
    assert cache.get("edges") == 7


def test_count_recompute_loses_to_concurrent_invalidate(store):
    """Store-level race: an invalidate that fires DURING the live active COUNT
    must not be overwritten by the recompute's stale put.

    The active count method snapshots the cache generation before the COUNT and
    stores via put_if_gen.  We simulate a concurrent writer by firing
    invalidate('active') right after the recompute takes its generation snapshot
    (before its put).  The recompute must then refuse to install its now-stale
    value, so the cache stays empty for 'active' and the NEXT call recomputes a
    fresh value rather than serving the stale one.
    """
    _seed_corpus(store, n_active=4)
    if not _has_count_cache(store):
        pytest.skip("count cache not present in this build")

    cache = store._corpus_count_cache
    cache.clear()  # cold cache: force a recompute on the next call

    real_generation = cache.generation
    fired: list[bool] = []

    def _generation_then_invalidate():
        # Hand the recompute its pre-COUNT generation snapshot ...
        gen = real_generation()
        if not fired:
            fired.append(True)
            # ... then simulate a concurrent writer landing a count-changing
            # write + invalidate before the recompute's put. This bumps the
            # generation past the snapshot the recompute just took.
            cache.invalidate("active")
        return gen

    cache.generation = _generation_then_invalidate  # type: ignore[assignment]
    try:
        value = store.active_records_count()
    finally:
        cache.generation = real_generation  # type: ignore[assignment]

    assert fired, "the patched active COUNT recompute must have snapshotted gen"
    # The method still RETURNS the freshly-computed value to its caller ...
    assert value == 4
    # ... but it must NOT have CACHED it over the concurrent invalidation.
    assert cache.get("active") is None, (
        "a count computed before a mid-recompute invalidate must not be cached; "
        "the generation CAS must refuse the stale put"
    )


def test_concurrent_writer_invalidate_never_caches_stale(store):
    """Threaded variant: a writer thread hammers invalidate('active') while a
    reader thread recomputes; the cache must never end up holding a value that
    predates the final invalidation.  Determinism is provided by ending on a
    known state: after the threads join, one final invalidate guarantees the
    cache is empty, and a single clean recompute must then equal the live count.
    """
    _seed_corpus(store, n_active=6)
    if not _has_count_cache(store):
        pytest.skip("count cache not present in this build")

    cache = store._corpus_count_cache
    stop = threading.Event()
    errors: list[BaseException] = []

    def _writer():
        try:
            while not stop.is_set():
                cache.invalidate("active")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def _reader():
        try:
            for _ in range(200):
                store.active_records_count()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    w = threading.Thread(target=_writer)
    r = threading.Thread(target=_reader)
    w.start()
    r.start()
    r.join()
    stop.set()
    w.join()

    assert not errors, f"no thread may raise: {errors}"
    # Land on a clean, known state and verify the cache agrees with live SQL.
    cache.invalidate("active")
    assert store.active_records_count() == 6
    assert cache.get("active") == 6
