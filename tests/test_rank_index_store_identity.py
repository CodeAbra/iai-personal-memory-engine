"""Two-store-in-process isolation for the rank-index adapter.

Observable proof of the store-identity-keyed handle (`rank_index_for`
stores the handle as an attribute of the `MemoryStore` instance, never a
module-level cache) -- mirrors `_topology_cache_key`'s multi-store
rationale. `test_rank_index_freshness.py` covers the initial-snapshot
isolation case as part of its own suite; this file goes further: it also
proves isolation survives an incremental write to one store after both
handles already exist.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp import retrieve
from iai_mcp.store import MemoryStore
from iai_mcp.store._rank_index import _HANDLE_ATTR, rank_index_for
from iai_mcp.types import MemoryRecord


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


def _make_record(
    store: MemoryStore,
    text: str,
    vec_seed: float,
    tags: "list[str] | None" = None,
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
        tags=tags or ["t"],
        language="en",
    )


def _two_stores(tmp_path: Path) -> tuple[MemoryStore, MemoryStore]:
    store_a = MemoryStore(path=tmp_path / "a" / "lancedb")
    store_a.root = tmp_path / "a"
    store_b = MemoryStore(path=tmp_path / "b" / "lancedb")
    store_b.root = tmp_path / "b"
    return store_a, store_b


def test_two_stores_get_two_distinct_handles(tmp_path: Path):
    store_a, store_b = _two_stores(tmp_path)
    rec_a = _make_record(store_a, "alpha-content", 0.11)
    store_a.insert(rec_a)
    rec_b = _make_record(store_b, "beta-content", 0.22)
    store_b.insert(rec_b)

    graph_a, _assignment_a, _rc_a = retrieve.build_runtime_graph(store_a)
    graph_b, _assignment_b, _rc_b = retrieve.build_runtime_graph(store_b)

    handle_a = rank_index_for(store_a, graph_a)
    handle_b = rank_index_for(store_b, graph_b)

    assert handle_a is not handle_b, "two stores must never share a rank-index handle"
    assert getattr(store_a, _HANDLE_ATTR) is handle_a, (
        "the handle must live as an attribute on the store instance itself, "
        "never a module-level cache keyed some other way"
    )
    assert getattr(store_b, _HANDLE_ATTR) is handle_b
    assert rank_index_for(store_a, graph_a) is handle_a, "repeat calls for the same store must reuse the handle"


def test_each_handle_serves_only_its_own_store_bulk_accessors(tmp_path: Path):
    store_a, store_b = _two_stores(tmp_path)
    rec_a = _make_record(store_a, "alpha-only-content", 0.11)
    rec_a.salience_level = "critical"
    store_a.insert(rec_a)
    rec_b = _make_record(store_b, "beta-only-content", 0.22)
    rec_b.salience_level = "notable"
    store_b.insert(rec_b)

    graph_a, _assignment_a, _rc_a = retrieve.build_runtime_graph(store_a)
    graph_b, _assignment_b, _rc_b = retrieve.build_runtime_graph(store_b)

    handle_a = rank_index_for(store_a, graph_a)
    handle_b = rank_index_for(store_b, graph_b)

    _gen_a, ids_a, _vec_a, _degree_a, _postings_a = handle_a.snapshot(graph_a, [])
    _gen_b, ids_b, _vec_b, _degree_b, _postings_b = handle_b.snapshot(graph_b, [])

    assert rec_a.id.int in ids_a and rec_a.id.int not in ids_b
    assert rec_b.id.int in ids_b and rec_b.id.int not in ids_a

    levels_a = handle_a.salience_levels()
    levels_b = handle_b.salience_levels()
    assert rec_b.id.int not in levels_a, "store B's salience data must never leak into store A's bulk accessor"
    assert rec_a.id.int not in levels_b, "store A's salience data must never leak into store B's bulk accessor"


def test_incremental_write_to_one_store_never_touches_the_other(tmp_path: Path):
    """The isolation must hold not only for the initial cold build but also
    for the per-write `feed()` path -- a shared underlying handle would leak
    a later write from one store into the other's already-built index."""
    store_a, store_b = _two_stores(tmp_path)
    rec_a = _make_record(store_a, "alpha-seed", 0.11)
    store_a.insert(rec_a)
    rec_b = _make_record(store_b, "beta-seed", 0.22)
    store_b.insert(rec_b)

    graph_a, _assignment_a, _rc_a = retrieve.build_runtime_graph(store_a)
    graph_b, _assignment_b, _rc_b = retrieve.build_runtime_graph(store_b)

    handle_a = rank_index_for(store_a, graph_a)
    handle_b = rank_index_for(store_b, graph_b)
    handle_a.snapshot(graph_a, [])
    handle_b.snapshot(graph_b, [])

    rec_b2 = _make_record(store_b, "beta-second-write", 0.33)
    store_b.insert(rec_b2)

    _gen_a, ids_a, *_rest_a = handle_a.snapshot(graph_a, [])
    _gen_b, ids_b, *_rest_b = handle_b.snapshot(graph_b, [])

    assert rec_b2.id.int in ids_b
    assert rec_b2.id.int not in ids_a, (
        "a write to store B after both handles exist must never appear in "
        "store A's index -- a process-global handle would leak it here"
    )
    assert rec_a.id.int in ids_a
