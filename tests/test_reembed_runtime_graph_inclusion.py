"""A re-embedded (pending->ready) row must land in the next warm community graph.

A deferred-capture row is written with no embedding (``embedding_pending = 1``),
excluded from the active corpus count and from the community graph. When its
embedding later lands (the wake sequence re-embeds the text, or a sidecar
embedding is ingested) the row becomes an active, index-findable record. Its +1
corpus delta can stay within the warm graph's drift tolerance and not flip the
staleness-window cache key, so without an explicit invalidation the next warm
build reuses the stale node set and the row is absent from community gating and
centrality until an unrelated over-tolerance rebuild.

These tests prove the data-operation boundary forces the row in:

* after ``pending_embeddings_wake_sequence`` completes a pending->ready
  transition, the next ``build_runtime_graph`` contains the row;
* a direct pending->ready UPDATE that bypasses that boundary leaves the row
  absent under warm reuse (the revert-proof: the fix, not an incidental
  rebuild, is what includes the row);
* an ordinary single capture within drift still reuses the warm graph and does
  not force a rebuild (the drift-tolerance optimization is intact -- the fix is
  scoped to pending->ready, not "invalidate on everything").

Hermetic: a tmp store path, an in-process deterministic embedder, no daemon, no
socket, no real home store.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp import retrieve, runtime_graph_cache
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord


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


@pytest.fixture(autouse=True)
def _tight_drift(monkeypatch):
    # An absolute drift floor of 3 and zero proportional band keeps a single +1
    # delta inside the tolerance window for these small corpora -- the exact
    # condition under which the warm graph would otherwise reuse a stale node set.
    monkeypatch.setenv("IAI_MCP_RGC_DRIFT_ABS", "3")
    monkeypatch.setenv("IAI_MCP_RGC_DRIFT_FRAC", "0.0")
    yield


class _DeterministicEmbedder:
    """Maps text to a stable unit vector by content (no model load)."""

    def embed(self, text: str) -> list[float]:
        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-12
        return v.tolist()


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    # Do NOT reassign ``s.root`` to a different path than the db root: in real
    # construction ``MemoryStore`` builds its ``HippoDB`` with ``self.root``, so
    # ``store.root`` and ``db._store_root`` are the same directory and the warm
    # graph cache lives where the storage layer invalidates it.
    return MemoryStore(path=tmp_path / "store")


def _make_rec(seed: int, store: MemoryStore) -> MemoryRecord:
    rng = np.random.default_rng(seed)
    vec = rng.random(store.embed_dim).astype(np.float32)
    vec = (vec / np.linalg.norm(vec)).tolist()
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(), tier="episodic",
        literal_surface=f"surface number {seed} carrying real text",
        aaak_index="", embedding=vec, community_id=None, centrality=0.0,
        detail_level=2, pinned=False, stability=0.0, difficulty=0.0,
        last_reviewed=None, never_decay=False, never_merge=False,
        provenance=[], created_at=now, updated_at=now, tags=[f"tag{seed % 3}"],
        language="en",
    )


def _seed_connected(store: MemoryStore, n: int, seed_base: int = 0) -> list:
    recs = [_make_rec(seed_base + i, store) for i in range(n)]
    for rec in recs:
        store.insert(rec)
    flush_record_buffer(store)
    ids = [rec.id for rec in recs]
    pairs = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
    store.boost_edges(pairs, delta=1.0, edge_type="hebbian")
    return recs


def _insert_pending(store: MemoryStore, surface: str) -> str:
    pid = str(uuid4())
    store.db.insert_pending_row(
        record_id=pid, tier="episodic",
        literal_surface=surface,
        tags_json=json.dumps(["tag0"]), provenance_json=json.dumps([]),
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    return pid


def _warm_a_graph(store: MemoryStore) -> int:
    """Build a centrality-bearing warm graph and assert it is warm; return node count."""
    runtime_graph_cache.invalidate(store)
    graph, _assign, _rc = retrieve.build_runtime_graph(store)
    assert not retrieve._runtime_graph_rebuild_needed(store), (
        "freshly-built centrality-bearing cache must be warm"
    )
    return graph.node_count()


def test_reembedded_row_included_in_warm_graph_via_wake_sequence(store):
    """The fix: a pending->ready transition through the wake sequence forces the
    re-embedded row into the next warm community graph (reembed_in_graph True)."""
    recs = _seed_connected(store, n=16, seed_base=200)
    n0 = _warm_a_graph(store)
    assert n0 == 16
    count0 = store.active_records_count()

    pid = _insert_pending(store, "pending surface carrying real text")
    assert store.active_records_count() == count0, "pending row must not count active"

    # Drive the REAL data-operation boundary: re-embed the pending row from its
    # text and rebuild the index. This is where the invalidation now fires -- no
    # daemon, no orchestration layer involved.
    result = store.db.pending_embeddings_wake_sequence(embedder=_DeterministicEmbedder())
    assert result.get("action") == "wake_sequence", result
    assert result.get("reembed_count") == 1, result

    count_after = store.active_records_count()
    assert count_after == count0 + 1, "re-embed must increment active count by 1"
    # The +1 delta is still within drift tolerance -- the bug's precondition. The
    # inclusion now comes from the invalidation, not from drift breaching.
    assert retrieve._within_drift_tolerance(count0, count_after), (
        "the +1 reembed delta must still be within drift tolerance; otherwise "
        "this test would pass for the wrong reason (an over-tolerance rebuild)"
    )

    graph2, _assign2, _rc2 = retrieve.build_runtime_graph(store)
    assert graph2.node_count() == count_after
    assert graph2.has_node(UUID(pid)), (
        "re-embedded row must be a node in the rebuilt warm graph after the "
        "wake-sequence invalidation"
    )


def test_direct_pending_to_ready_bypass_leaves_row_absent(store):
    """Revert-proof: a pending->ready UPDATE that bypasses the data-operation
    boundary (no invalidation) is reused stale -- the row stays absent under warm
    graph reuse. This is the exact behavior the fix removes from the real path."""
    recs = _seed_connected(store, n=16, seed_base=400)
    n0 = _warm_a_graph(store)
    assert n0 == 16
    count0 = store.active_records_count()

    pid = _insert_pending(store, "pending surface carrying real text")
    assert store.active_records_count() == count0

    # Flip embedding_pending 1->0 directly, NOT through the wake sequence -- no
    # edge added, no invalidation. Mirrors the latent-bug reproduction.
    with store.db._conn_lock:
        rng = np.random.default_rng(999)
        v = rng.random(store.embed_dim).astype(np.float32)
        v = v / np.linalg.norm(v)
        store.db._conn.execute(
            "UPDATE records SET embedding = ?, embedding_pending = 0 WHERE id = ?",
            (v.astype(np.float32).tobytes(), pid),
        )
        store.db._conn.commit()

    # The direct bypass deliberately skips the data-operation invalidation boundary.
    # The count cache may hold a stale value from before the bypass — clear it
    # before asserting the SQL count so this check measures the live SQL state,
    # not whatever the cache holds.  The graph-rebuild decision at line 199 still
    # returns False because the store cache is repopulated (17) and the drift
    # comparison (payload=16 vs live=17, delta=1, drift_abs=3) is within tolerance.
    _force_empty_count_cache_reembed_ext(store)
    count_after = store.active_records_count()
    assert count_after == count0 + 1
    assert retrieve._within_drift_tolerance(count0, count_after)
    assert not retrieve._runtime_graph_rebuild_needed(store), (
        "within-drift bypass must reuse the warm cache (no rebuild)"
    )

    graph2, _assign2, _rc2 = retrieve.build_runtime_graph(store)
    assert not graph2.has_node(UUID(pid)), (
        "without the data-operation invalidation the warm-reuse path excludes "
        "the row -- this is the latent bug the fix closes at the real path"
    )


def test_ordinary_capture_within_drift_does_not_force_rebuild(store):
    """Drift-tolerance optimization intact: an ordinary single capture within
    drift reuses the warm graph and does NOT force a rebuild. The fix is scoped
    to pending->ready, not 'invalidate on every write'."""
    recs = _seed_connected(store, n=16, seed_base=600)
    _warm_a_graph(store)
    count0 = store.active_records_count()

    # One ordinary capture: a fully-embedded insert, no pending lifecycle.
    rec = _make_rec(700, store)
    store.insert(rec)
    flush_record_buffer(store)

    count_after = store.active_records_count()
    assert count_after == count0 + 1, "ordinary capture increments active count"
    assert retrieve._within_drift_tolerance(count0, count_after)
    # The warm cache must still be reused -- no spurious invalidation on the
    # ordinary capture path.
    assert not retrieve._runtime_graph_rebuild_needed(store), (
        "an ordinary within-drift capture must NOT force a runtime-graph rebuild; "
        "the fix must not invalidate on every write"
    )


# ---------------------------------------------------------------------------
# Corpus-count cache — reembed both-count + cache-key change
#
# A re-embed flip (embedding_pending 1→0) must change BOTH the active and
# pending cached counts, and the resulting _cache_key tuple must change so
# the raw-pending-count rebuild trigger fires (runtime_graph_cache.py:626-630).
#
# RED mechanism on HEAD: the count methods today return live SQL values (no
# persistent cross-read cache), so the counts ARE correct after a reembed.
# However, the key-change assertion (assert key_before != key_after) is where
# this test can be RED: if the cached pending count does NOT drop by 1 (because
# a future implementation fails to invalidate the pending cache on reembed),
# the keys would remain equal and the assertion fails.
#
# On HEAD the test PASSES because the live SQL counts are always exact —
# there is no stale cache to hold the wrong pending count.  It becomes the
# meaningful regression gate once the count cache lands: a bug where reembed
# invalidates only one of {active, pending} would leave one count stale,
# making the key identical to the pre-reembed key and failing this test.
# ---------------------------------------------------------------------------

def _force_empty_count_cache_reembed_ext(store) -> None:
    """Clear the store-level corpus count cache if it is present."""
    cache = getattr(store, "_corpus_count_cache", None)
    if cache is not None and hasattr(cache, "clear"):
        cache.clear()
        return
    counts = getattr(store, "_corpus_counts", None)
    if counts is not None and hasattr(counts, "clear"):
        counts.clear()


def test_reembed_flip_changes_both_counts_and_cache_key(store):
    """After a re-embed flip (pending→active), BOTH the active and pending
    cached counts must change, and the _cache_key tuple must change.

    Both-count guard:
    - active must increase by the number of rows flipped
    - pending must decrease by the same number
    - _cache_key before-reembed != _cache_key after-reembed (raw pending
      count is folded in unwindowed so a single flip changes the key)

    The existing reembed-inclusion assertions above verify the newly-embedded
    row lands in the warm graph; this test verifies the count arithmetic and
    key-change property independently of graph structure.
    """
    recs = _seed_connected(store, n=8, seed_base=300)
    pid = _insert_pending(store, "both-count gate pending surface")

    # Snapshot counts before reembed
    active_before = store.active_records_count()
    pending_before = store.pending_records_count()
    key_before = runtime_graph_cache._cache_key(store)

    # Clear the count cache (no-op on HEAD, meaningful once the count cache lands)
    _force_empty_count_cache_reembed_ext(store)

    # Drive the reembed (pending 1→0 via the wake sequence)
    result = store.db.pending_embeddings_wake_sequence(embedder=_DeterministicEmbedder())
    assert result.get("action") == "wake_sequence", result
    assert result.get("reembed_count", 0) == 1, (
        "exactly one pending row must have been re-embedded"
    )

    # Counts after reembed
    active_after = store.active_records_count()
    pending_after = store.pending_records_count()
    key_after = runtime_graph_cache._cache_key(store)

    # (a) active must increase by 1
    assert active_after == active_before + 1, (
        f"reembed must increase active count from {active_before} to "
        f"{active_before + 1}; got {active_after}"
    )

    # (b) pending must decrease by 1
    assert pending_after == pending_before - 1, (
        f"reembed must decrease pending count from {pending_before} to "
        f"{pending_before - 1}; got {pending_after} — "
        "a cache that does not invalidate the pending count on reembed "
        "will hold the stale higher value and fail here"
    )

    # (c) _cache_key must change (raw pending is folded in unwindowed)
    assert key_after != key_before, (
        f"_cache_key must change after a reembed flip so the warm build "
        f"re-streams the newly-embedded row; key before={key_before}, "
        f"key after={key_after} — a cache that keeps the stale pending count "
        "leaves the key unchanged and the embedded row stays absent from recall"
    )
