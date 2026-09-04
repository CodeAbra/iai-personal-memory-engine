from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp import retrieve
from iai_mcp.store import MemoryStore
from iai_mcp.store._rank_index import rank_index_for
from iai_mcp.types import SALIENCE_LEVEL_RANK, MemoryRecord


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


def _make_record(
    store: MemoryStore,
    text: str = "hello",
    vec_seed: float = 0.1,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[vec_seed] * store.embed_dim,
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
        tags=["t"],
        language="en",
    )


# ---------------------------------------------------------------------------
# Store-identity-keyed adapter + cold-build salience backfill
# ---------------------------------------------------------------------------

def test_adapter_two_store_isolation(tmp_path: Path):
    store_a = MemoryStore(path=tmp_path / "a" / "lancedb")
    store_a.root = tmp_path / "a"
    store_b = MemoryStore(path=tmp_path / "b" / "lancedb")
    store_b.root = tmp_path / "b"

    rec_a = _make_record(store_a, "alpha-only", 0.11)
    store_a.insert(rec_a)
    rec_b = _make_record(store_b, "beta-only", 0.22)
    store_b.insert(rec_b)

    graph_a, _assignment_a, _rc_a = retrieve.build_runtime_graph(store_a)
    graph_b, _assignment_b, _rc_b = retrieve.build_runtime_graph(store_b)

    handle_a = rank_index_for(store_a, graph_a)
    handle_b = rank_index_for(store_b, graph_b)
    assert handle_a is not handle_b, "two stores must never share a rank-index handle"

    _gen_a, ids_a, *_rest_a = handle_a.snapshot(graph_a, [])
    _gen_b, ids_b, *_rest_b = handle_b.snapshot(graph_b, [])
    assert rec_a.id.int in ids_a
    assert rec_a.id.int not in ids_b
    assert rec_b.id.int in ids_b
    assert rec_b.id.int not in ids_a


def test_adapter_lazy_build_exposes_bulk_accessors_only(store: MemoryStore):
    rec = _make_record(store, "hello searchable world", 0.3)
    store.insert(rec)
    graph, _assignment, _rc = retrieve.build_runtime_graph(store)

    handle = rank_index_for(store, graph)
    generation, ids, vectors, degree_map, postings = handle.snapshot(graph, ["hello"])

    assert generation == graph._pool_content_version
    assert rec.id.int in ids
    assert vectors.shape == (len(ids), store.embed_dim)
    assert isinstance(degree_map, dict)
    assert "hello" in postings


def test_backfill_salience_from_plaintext_column_for_preexisting_corpus(
    store: MemoryStore,
):
    rec = _make_record(store, "critical-thing", 0.5)
    rec.salience_level = "critical"
    store.insert(rec)
    graph, _assignment, _rc = retrieve.build_runtime_graph(store)

    # Simulate the pre-existing-corpus gap this backfill exists to close: an
    # in-memory payload that never carried salience_level, regardless of
    # whether the write-time hook now sends it going forward.
    graph._node_payload[str(rec.id)].pop("salience_level", None)

    handle = rank_index_for(store, graph)
    handle.snapshot(graph, [])
    levels = handle.salience_levels()
    assert levels[rec.id.int] == SALIENCE_LEVEL_RANK["critical"]


# ---------------------------------------------------------------------------
# Hook wiring -- feed() on every write, salience_level in the payload
# ---------------------------------------------------------------------------

def test_hook_payload_carries_full_bucket_a_field_set(store: MemoryStore):
    # The graph must already be built (hook registered) before the write
    # under test, so this insert flows through the LIVE hook path -- not
    # the initial cold corpus stream, which sources salience_level from the
    # separate plaintext-column backfill covered by its own test above.
    seed = _make_record(store, "seed", 0.1)
    store.insert(seed)
    graph, _assignment, _rc = retrieve.build_runtime_graph(store)

    rec = _make_record(store, "field-audit", 0.7)
    rec.salience_level = "notable"
    store.insert(rec)
    payload = graph.get_payload(rec.id)
    for key in (
        "aaak_index", "created_at", "stability", "surface",
        "tier", "tags", "salience_level", "centrality",
    ):
        assert key in payload, f"Bucket-A field {key!r} missing from hook payload"


def test_hook_wires_rank_index_feed_with_salience_level_parity(store: MemoryStore):
    rec = _make_record(store, "wired-write", 0.4)
    rec.salience_level = "notable"
    store.insert(rec)
    graph, _assignment, _rc = retrieve.build_runtime_graph(store)

    handle = rank_index_for(store, graph)
    handle.snapshot(graph, [])
    assert handle.salience_levels().get(rec.id.int) == SALIENCE_LEVEL_RANK["notable"]

    # A record inserted AFTER the index already exists must be fed
    # incrementally through the shared hook, not require a fresh cold build.
    rec2 = _make_record(store, "second-write", 0.5)
    rec2.salience_level = "critical"
    store.insert(rec2)

    handle2 = rank_index_for(store, graph)
    assert handle2 is handle, "the handle must be reused for the same store+graph"
    gen2, ids2, *_rest2 = handle2.snapshot(graph, [])
    assert rec2.id.int in ids2
    assert handle2.salience_levels().get(rec2.id.int) == SALIENCE_LEVEL_RANK["critical"]


def test_hook_feed_delete_removes_from_rank_index(store: MemoryStore):
    rec = _make_record(store, "will-be-deleted", 0.2)
    store.insert(rec)
    graph, _assignment, _rc = retrieve.build_runtime_graph(store)
    handle = rank_index_for(store, graph)
    _gen1, ids1, *_rest1 = handle.snapshot(graph, [])
    assert rec.id.int in ids1

    store.delete(rec.id)
    _gen2, ids2, *_rest2 = handle.snapshot(graph, [])
    assert rec.id.int not in ids2


def test_snapshot_pure_read_when_generation_matches_no_extra_write(
    store: MemoryStore,
):
    rec = _make_record(store, "steady", 0.3)
    store.insert(rec)
    graph, _assignment, _rc = retrieve.build_runtime_graph(store)
    handle = rank_index_for(store, graph)
    gen1, ids1, *_rest1 = handle.snapshot(graph, [])
    gen1b, ids1b, *_rest1b = handle.snapshot(graph, [])
    assert gen1 == gen1b
    assert list(ids1) == list(ids1b)

    rec2 = _make_record(store, "adds-one", 0.6)
    store.insert(rec2)
    gen2, ids2, *_rest2 = handle.snapshot(graph, [])
    assert gen2 > gen1
    assert rec2.id.int in ids2


# ---------------------------------------------------------------------------
# Close the raise_salience_level_if_higher hook bypass
# ---------------------------------------------------------------------------

def test_salience_raise_bumps_generation_and_updates_resident_payload(
    store: MemoryStore,
):
    rec = _make_record(store, "raise-me", 0.3)
    store.insert(rec)
    graph, _assignment, _rc = retrieve.build_runtime_graph(store)
    handle = rank_index_for(store, graph)
    handle.snapshot(graph, [])

    gen_before = graph._pool_content_version
    changed = store.raise_salience_level_if_higher(rec.id, "critical")
    assert changed is True
    assert graph._pool_content_version > gen_before, (
        "a salience raise must bump graph._pool_content_version -- today it "
        "writes the column directly and never touches the graph"
    )
    assert (
        graph.get_payload(rec.id)["salience_level"]
        == SALIENCE_LEVEL_RANK["critical"]
    )

    handle.snapshot(graph, [])
    assert handle.salience_levels().get(rec.id.int) == SALIENCE_LEVEL_RANK["critical"]


def test_salience_raise_no_hook_registered_no_ops_sync_but_still_writes_column(
    tmp_path: Path,
):
    s = MemoryStore(path=tmp_path / "lancedb")
    s.root = tmp_path
    rec = _make_record(s, "no-hook", 0.2)
    s.insert(rec)
    # No graph built, no hook registered -- the sync must no-op, never raise.
    changed = s.raise_salience_level_if_higher(rec.id, "critical")
    assert changed is True
    reloaded = s.get(rec.id)
    assert reloaded.salience_level == "critical"


def test_salience_raise_monotone_lower_level_no_ops(store: MemoryStore):
    rec = _make_record(store, "already-critical", 0.4)
    rec.salience_level = "critical"
    store.insert(rec)
    graph, _assignment, _rc = retrieve.build_runtime_graph(store)
    gen_before = graph._pool_content_version

    changed = store.raise_salience_level_if_higher(rec.id, "notable")
    assert changed is False
    assert graph._pool_content_version == gen_before
    assert store.get(rec.id).salience_level == "critical"


# ---------------------------------------------------------------------------
# Delete-replay safety: a queued delete for an id the published buffer never
# actually held must never crash the drain on the next snapshot().
# ---------------------------------------------------------------------------

def test_hook_feed_delete_between_snapshots_is_benign(store: MemoryStore):
    seed = _make_record(store, "seed", 0.1)
    store.insert(seed)
    graph, _assignment, _rc = retrieve.build_runtime_graph(store)
    handle = rank_index_for(store, graph)
    handle.snapshot(graph, [])

    rec = _make_record(store, "insert-then-delete", 0.3)
    store.insert(rec)
    store.delete(rec.id)

    _gen, ids, *_rest = handle.snapshot(graph, [])
    assert rec.id.int not in ids


def test_hook_feed_delete_of_never_inserted_id_is_benign(store: MemoryStore):
    seed = _make_record(store, "seed2", 0.15)
    store.insert(seed)
    graph, _assignment, _rc = retrieve.build_runtime_graph(store)
    handle = rank_index_for(store, graph)
    handle.snapshot(graph, [])

    class _DeleteShim:
        def __init__(self, rid):
            self.id = rid

    phantom_id = uuid4()
    handle.feed("delete", _DeleteShim(phantom_id))

    # A matching-generation snapshot never drains pending, so advance the
    # generation with a real write before re-snapshotting -- otherwise this
    # test would never exercise the replay path it exists to cover.
    seed3 = _make_record(store, "advance-generation", 0.25)
    store.insert(seed3)

    _gen, ids, *_rest = handle.snapshot(graph, [])
    assert phantom_id.int not in ids
    assert seed3.id.int in ids
