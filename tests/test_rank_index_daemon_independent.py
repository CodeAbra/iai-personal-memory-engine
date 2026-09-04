"""Daemon-independent cold start: index-to-index parity + recorded
cold-build time + cold-build salience_level backfill.

Every test in this file constructs its store and graph in-process with no
daemon involved (the CLI/bank-fallback shape) -- pytest never runs the
daemon, so that half of IDX-04 is automatically satisfied by construction.
The interesting half is index-to-index PARITY: `_RankIndexHandle` has
exactly two ways a record can enter the resident index -- the bulk
`_build()` cold path (reads an already-populated store in one shot, what
CLI/bank-fallback does on process start) and the incremental `feed()` path
(what a long-lived daemon does as writes stream in one at a time through
the live hook). This file builds the SAME corpus through both paths in two
separate stores and asserts the resulting index CONTENTS match, id-keyed --
proving a daemon-down cold start reproduces what a warm, continuously-fed
daemon index would hold. Recall-output parity would test the unchanged
Python recall path and prove nothing about the index; this is the honest
substitute given nothing reads the index on the recall path yet.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from uuid import UUID, uuid4

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


def _record_specs(n: int, salient_index: "int | None" = None):
    """Deterministic (id, text, vec_seed, tags, salience_level) specs shared
    verbatim by both stores -- id-keyed parity requires identical ids."""
    specs = []
    for i in range(n):
        salience = "critical" if i == salient_index else "unflagged"
        specs.append((uuid4(), f"corpus-record-{i}-marker{i % 5}", 0.05 + (i % 7) * 0.01, [f"tag{i % 4}"], salience))
    return specs


def _record_from_spec(store: MemoryStore, spec) -> MemoryRecord:
    rid, text, vec_seed, tags, salience = spec
    now = datetime.now(timezone.utc)
    rec = MemoryRecord(
        id=rid,
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
        tags=tags,
        language="en",
    )
    rec.salience_level = salience
    return rec


def _cold_store_and_handle(tmp_path: Path, name: str, specs):
    """The CLI/bank-fallback shape: every record already resident in the
    store BEFORE the graph or the rank index is ever constructed -- the
    handle's first snapshot() is a single bulk `_build()` over the whole
    corpus at once."""
    store = MemoryStore(path=tmp_path / name / "lancedb")
    store.root = tmp_path / name
    for spec in specs:
        store.insert(_record_from_spec(store, spec))
    graph, _assignment, _rc = retrieve.build_runtime_graph(store)
    handle = rank_index_for(store, graph)
    return store, graph, handle


def _warm_store_and_handle(tmp_path: Path, name: str, specs):
    """The long-lived-daemon shape: the graph and rank index exist FIRST
    (empty), then every record streams in one at a time through the live
    per-write feed() hook -- the incremental path a daemon actually
    exercises as writes arrive."""
    store = MemoryStore(path=tmp_path / name / "lancedb")
    store.root = tmp_path / name
    graph, _assignment, _rc = retrieve.build_runtime_graph(store)
    handle = rank_index_for(store, graph)
    handle.snapshot(graph, [])  # trivial empty build, establishes the handle
    for spec in specs:
        store.insert(_record_from_spec(store, spec))
    return store, graph, handle


def _vectors_by_id(ids, matrix) -> dict:
    return {rid: tuple(round(float(x), 6) for x in row) for rid, row in zip(ids, matrix)}


def test_cold_build_index_to_index_parity_with_warm_incremental_build(tmp_path: Path):
    n = 40
    salient_index = 7
    specs = _record_specs(n, salient_index=salient_index)
    salient_id = specs[salient_index][0]

    _store_cold, graph_cold, handle_cold = _cold_store_and_handle(tmp_path, "cold", specs)
    _store_warm, graph_warm, handle_warm = _warm_store_and_handle(tmp_path, "warm", specs)

    query_token = "marker3"
    gen_cold, ids_cold, matrix_cold, degree_cold, postings_cold = handle_cold.snapshot(graph_cold, [query_token])
    gen_warm, ids_warm, matrix_warm, degree_warm, postings_warm = handle_warm.snapshot(graph_warm, [query_token])

    assert set(ids_cold) == set(ids_warm), "cold and warm builds must resolve to the same id set"
    assert len(ids_cold) == n and len(ids_warm) == n

    vec_cold = _vectors_by_id(ids_cold, matrix_cold)
    vec_warm = _vectors_by_id(ids_warm, matrix_warm)
    assert vec_cold == vec_warm, "per-id vector rows must match between the bulk cold build and the incrementally fed warm build"

    assert degree_cold == degree_warm, "per-id degree map must match"
    assert handle_cold.adjacency_by_type() == handle_warm.adjacency_by_type(), "per-id per-edge-type map must match"

    assert postings_cold.get(query_token, {}) == postings_warm.get(query_token, {}), (
        "the requested token's posting list (doc id -> term frequency) must match"
    )

    levels_cold = handle_cold.salience_levels()
    levels_warm = handle_warm.salience_levels()
    assert levels_cold == levels_warm, "per-id salience_level rank must match"
    assert levels_cold[salient_id.int] == SALIENCE_LEVEL_RANK["critical"], (
        "the cold bulk build must resolve the pre-existing corpus's real salience_level "
        "via the plaintext-column read"
    )
    assert levels_warm[salient_id.int] == SALIENCE_LEVEL_RANK["critical"], (
        "the incremental feed path must resolve the same record's salience_level from "
        "the fed MemoryRecord attribute"
    )


def test_cold_start_build_time_is_measured_and_recorded(tmp_path: Path):
    n = 40
    repeats = 5
    samples_ms: list[float] = []
    for i in range(repeats):
        specs = _record_specs(n)
        store = MemoryStore(path=tmp_path / f"timing{i}" / "lancedb")
        store.root = tmp_path / f"timing{i}"
        for spec in specs:
            store.insert(_record_from_spec(store, spec))
        # build_runtime_graph is excluded from the timed window: it is
        # pre-existing infra outside 264's scope. Only the rank-index cold
        # build (_RankIndexHandle._build, triggered by the first snapshot())
        # is measured.
        graph, _assignment, _rc = retrieve.build_runtime_graph(store)
        handle = rank_index_for(store, graph)

        start = time.perf_counter()
        handle.snapshot(graph, [])
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        samples_ms.append(elapsed_ms)

    med = median(samples_ms)
    print(
        f"daemon_independent: cold-build (in-process, no daemon) n_records={n} "
        f"samples_ms={[round(s, 3) for s in samples_ms]} median_ms={med:.3f}"
    )

    assert med > 0.0, "cold-build time must be a real positive measurement"
    assert all(s > 0.0 for s in samples_ms)
