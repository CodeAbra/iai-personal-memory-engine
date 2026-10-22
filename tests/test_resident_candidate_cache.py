"""Resident-candidate cache: capped, LRU, write-evicted, decrypt-skip
residency for the live recall path -- replaces the unbounded, never-evicted
per-store builder graph.
"""
from __future__ import annotations

import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp import retrieve
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.store._rank_index import _ResidentCandidates, _ResidentFields, rank_index_for
from iai_mcp.types import MemoryRecord

_DIM = 16  # small synthetic dim -- keeps this a single-file, native-embedder-free test


@pytest.fixture(autouse=True)
def _small_embed_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(_DIM))


@pytest.fixture(autouse=True)
def _crypto_passphrase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake.pop((s, u), None))
    yield fake


def _unit_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _make_rec(
    rid: UUID, surface: str, embedding: list[float],
    valence: float = 0.0, salience_level: str = "unflagged",
) -> MemoryRecord:
    ts = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid,
        tier="episodic",
        literal_surface=surface,
        aaak_index="",
        embedding=embedding,
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
        created_at=ts,
        updated_at=ts,
        tags=["t"],
        language="en",
        valence=valence,
        salience_level=salience_level,
    )


@pytest.fixture
def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "store")


# ---------------------------------------------------------------------------
# Unit-level: the LRU itself, independent of any store or Rust index.
# ---------------------------------------------------------------------------

def test_cache_disabled_at_cap_zero():
    cache = _ResidentCandidates(capacity=0)
    rec = _make_rec(uuid4(), "x", _unit_vec(1))
    cache.put(rec)
    assert cache.get(rec.id) is None
    assert len(cache) == 0


def test_cap_eviction_is_lru_order():
    cache = _ResidentCandidates(capacity=3)
    recs = [_make_rec(uuid4(), f"rec-{i}", _unit_vec(i)) for i in range(3)]
    for rec in recs:
        cache.put(rec)
    assert len(cache) == 3

    # Touch recs[0] so it becomes most-recently-used; recs[1] stays the
    # oldest untouched entry and must be evicted first on overflow.
    assert cache.get(recs[0].id) is not None
    cache.put(_make_rec(uuid4(), "newcomer", _unit_vec(9)))

    assert len(cache) == 3
    assert cache.get(recs[1].id) is None, "the least-recently-touched entry must be evicted first"
    assert cache.get(recs[0].id) is not None, "a get()-touched entry must survive the eviction"
    assert cache.get(recs[2].id) is not None


def test_evict_removes_entry():
    cache = _ResidentCandidates(capacity=10)
    rec = _make_rec(uuid4(), "x", _unit_vec(1))
    cache.put(rec)
    assert cache.get(rec.id) is not None
    cache.evict(rec.id)
    assert cache.get(rec.id) is None


def test_fields_carry_valence_and_salience_level():
    cache = _ResidentCandidates(capacity=10)
    rec = _make_rec(uuid4(), "x", _unit_vec(1), valence=0.7, salience_level="critical")
    cache.put(rec)
    fields = cache.get(rec.id)
    assert fields.valence == pytest.approx(0.7)
    assert fields.salience_level == "critical"


def test_concurrent_get_put_smoke():
    cache = _ResidentCandidates(capacity=64)
    recs = [_make_rec(uuid4(), f"rec-{i}", _unit_vec(i)) for i in range(200)]
    errors: list[Exception] = []

    def _writer() -> None:
        try:
            for rec in recs:
                cache.put(rec)
        except Exception as exc:  # noqa: BLE001 -- surfaced via `errors`, not a bare thread crash
            errors.append(exc)

    def _reader() -> None:
        try:
            for rec in recs:
                cache.get(rec.id)
        except Exception as exc:  # noqa: BLE001 -- surfaced via `errors`, not a bare thread crash
            errors.append(exc)

    threads = [threading.Thread(target=_writer) for _ in range(4)] + [
        threading.Thread(target=_reader) for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors, f"concurrent get/put raised: {errors}"
    assert len(cache) <= 64


def test_per_entry_bytes_stay_bounded_not_corpus_shaped():
    """Falsifies the O(cap) claim: an entry with a realistic surface length
    stays well under a pinned per-entry ceiling, so cap * ceiling bounds
    total residency independent of corpus size."""
    cache = _ResidentCandidates(capacity=1)
    surface = "a realistic captured memory surface sentence. " * 6  # ~288 chars
    rec = _make_rec(uuid4(), surface, _unit_vec(1))
    rec.aaak_index = "idx-marker"
    rec.tags = ["alpha", "beta", "gamma"]
    cache.put(rec)
    fields = cache.get(rec.id)

    entry_bytes = (
        sys.getsizeof(fields)
        + sys.getsizeof(fields.literal_surface)
        + sys.getsizeof(fields.aaak_index)
        + sys.getsizeof(fields.tags)
        + sum(sys.getsizeof(t) for t in fields.tags)
        + sys.getsizeof(fields.created_at)
        + sys.getsizeof(fields.tier)
        + sys.getsizeof(fields.language)
        + sys.getsizeof(fields.salience_level)
    )

    _PER_ENTRY_CEILING_BYTES = 2048
    assert entry_bytes < _PER_ENTRY_CEILING_BYTES, (
        f"per-entry resident bytes {entry_bytes} exceeds the pinned ceiling "
        f"{_PER_ENTRY_CEILING_BYTES} -- residency would no longer be O(cap)"
    )
    assert "structure_hv" not in _ResidentFields.__slots__, (
        "structure_hv must never be a resident field -- its ~10 KB payload "
        "would blow the per-entry budget ~10x"
    )
    assert "embedding" not in _ResidentFields.__slots__, (
        "embedding must never be cached in _ResidentFields -- it is sourced "
        "fresh from published_view() per hit"
    )


# ---------------------------------------------------------------------------
# Write-time eviction through the real hook.
# ---------------------------------------------------------------------------

def test_hook_evicts_on_update_and_delete(_store: MemoryStore):
    rec = _make_rec(uuid4(), "hook-eviction target", _unit_vec(1))
    _store.insert(rec)
    flush_record_buffer(_store)
    retrieve.build_runtime_graph(_store)

    handle = rank_index_for(_store)
    handle.resident.put(rec)
    assert handle.resident.get(rec.id) is not None

    updated = _make_rec(rec.id, "hook-eviction target updated", _unit_vec(1))
    _store.update(updated)
    assert handle.resident.get(rec.id) is None, "an update must evict the resident entry"

    handle.resident.put(rec)
    assert handle.resident.get(rec.id) is not None
    _store.delete(rec.id)
    assert handle.resident.get(rec.id) is None, "a delete must evict the resident entry"


# ---------------------------------------------------------------------------
# Embedding exactness: the resident hit's vector, sourced from
# published_view(), matches a fresh store decode exactly.
# ---------------------------------------------------------------------------

def test_resident_embedding_matches_store_decode_exactly(_store: MemoryStore):
    from iai_mcp.core import _resident_feed

    rec = _make_rec(uuid4(), "exact embedding proof", _unit_vec(2))
    _store.insert(rec)
    flush_record_buffer(_store)

    handle = rank_index_for(_store)
    fetched = _store.get_batch([rec.id], decode="rank")
    _resident_feed(handle, fetched)
    handle.snapshot()

    matrix, slot_map = handle.published_view()
    slot = slot_map[rec.id.int]
    store_decoded = _store.get_batch([rec.id], decode="rank")[rec.id].embedding
    expected = np.asarray(store_decoded, dtype=np.float32)

    assert np.array_equal(matrix[slot], expected), (
        "the resident hit's embedding row must be byte-identical to a fresh "
        "get_batch(decode='rank') decode, not merely close"
    )


# ---------------------------------------------------------------------------
# Miss on an id absent from the published slot_map -- never a fabricated
# zero vector.
# ---------------------------------------------------------------------------

def test_miss_on_id_absent_from_published_slot_map(_store: MemoryStore):
    """An id resident in the cache but fed THIS call (not yet drained by a
    real snapshot()) must resolve as a miss."""
    from iai_mcp.core import _resident_hit_views

    handle = rank_index_for(_store)
    handle.snapshot()  # builds against the (empty) corpus

    rec = _make_rec(uuid4(), "undrained resident probe", _unit_vec(3))
    handle.resident.put(rec)
    handle.feed("upsert", rec)  # queued into the overlay, never drained

    assert handle.resident.get(rec.id) is not None, (
        "fixture self-check: the id must be cache-resident"
    )

    views, miss = _resident_hit_views(handle, [rec.id])
    assert views == {}
    assert miss == [rec.id], (
        "a resident id absent from the published slot_map must resolve as "
        "a miss, never a fabricated zero-vector hit"
    )


def test_pending_resident_hit_excluded_never_served_zero_vector(_store: MemoryStore):
    """A record still `embedding_pending` (no real embedding computed yet)
    that reaches residency through hop-neighbor discovery must resolve as
    a miss on a later resident-cache hit -- never as a hit serving
    `_build()`'s fabricated zero-vector row from `published_view()`'s
    matrix."""
    from iai_mcp.core import _resident_feed, _resident_hit_views

    now = datetime.now(timezone.utc).isoformat()
    pending_id = uuid4()
    _store.insert_pending(
        record_id=str(pending_id),
        tier="episodic",
        literal_surface="a pending hop-neighbor row",
        tags_json="[]",
        provenance_json="[]",
        created_at=now,
        updated_at=now,
    )

    handle = rank_index_for(_store)
    fetched = _store.get_batch([pending_id], decode="rank")
    assert fetched, "fixture self-check: the pending row must be fetchable"
    _resident_feed(handle, fetched)  # caches into handle.resident; feed() is
    # a pre-build no-op, matching a real first-discovery miss.
    handle.snapshot()  # _build()'s pending second-source path publishes
    # this id into the index with a fabricated zero-vector row.

    assert handle.resident.get(pending_id) is not None, (
        "fixture self-check: the pending id must be cache-resident after "
        "discovery"
    )
    matrix, slot_map = handle.published_view()
    assert pending_id.int in slot_map, (
        "fixture self-check: the pending id must be a published-slot hit "
        "whose row is the fabricated zero vector for a pending record"
    )
    assert not np.any(matrix[slot_map[pending_id.int]]), (
        "fixture self-check: the published row must be the fabricated "
        "zero vector _build() fills for a pending row"
    )

    views, miss = _resident_hit_views(handle, [pending_id])
    assert views == {}, (
        "a resident-cached embedding_pending id must never be served as a "
        "hit with the published matrix's fabricated zero vector"
    )
    assert miss == [pending_id]


def test_hit_view_source_returns_aligned_triple_at_one_generation(_store: MemoryStore):
    """`hit_view_source()` must return a (matrix, slot_map, pending) triple
    whose three parts describe the same generation, and a pending resident
    id must route to miss through `_resident_hit_views` when resolved via
    that single accessor."""
    from iai_mcp.core import _resident_feed, _resident_hit_views

    rec = _make_rec(uuid4(), "aligned triple probe", _unit_vec(4))
    _store.insert(rec)
    flush_record_buffer(_store)

    now = datetime.now(timezone.utc).isoformat()
    pending_id = uuid4()
    _store.insert_pending(
        record_id=str(pending_id),
        tier="episodic",
        literal_surface="a pending row for the triple probe",
        tags_json="[]",
        provenance_json="[]",
        created_at=now,
        updated_at=now,
    )

    handle = rank_index_for(_store)
    fetched = _store.get_batch([rec.id, pending_id], decode="rank")
    _resident_feed(handle, fetched)
    handle.snapshot()

    matrix, slot_map, pending = handle.hit_view_source()
    assert rec.id.int in slot_map and pending_id.int in slot_map, (
        "fixture self-check: both ids must be published slots"
    )
    assert pending.get(pending_id.int) is True, (
        "the pending id's flag must be True at this generation"
    )
    assert pending.get(rec.id.int, False) is False, (
        "the non-pending id's flag must be False at this generation"
    )
    assert matrix.shape[0] == len(slot_map), (
        "matrix and slot_map must describe the same number of rows"
    )

    views, miss = _resident_hit_views(handle, [rec.id, pending_id])
    assert rec.id in views
    assert miss == [pending_id], (
        "a pending resident id must route to miss when resolved through "
        "the single hit_view_source() accessor"
    )


def test_no_resident_ids_never_calls_published_view(_store: MemoryStore, monkeypatch):
    """A miss-only batch must never pay a published_view() export."""
    from iai_mcp.core import _resident_hit_views

    handle = rank_index_for(_store)
    calls = {"n": 0}
    orig = handle.published_view

    def _spy():
        calls["n"] += 1
        return orig()

    monkeypatch.setattr(handle, "published_view", _spy)

    views, miss = _resident_hit_views(handle, [])
    assert views == {}
    assert miss == []
    assert calls["n"] == 0, "an empty resident-id batch must never call published_view()"
