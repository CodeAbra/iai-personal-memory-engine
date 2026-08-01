from __future__ import annotations

from uuid import uuid4

import pytest

from iai_mcp.store import EDGES_TABLE, MemoryStore


def _versions(store: MemoryStore) -> int:
    tbl = store.db.open_table(EDGES_TABLE)
    return len(tbl.list_versions())


def test_boost_edges_emits_at_most_two_versions(tmp_path):
    store = MemoryStore(path=tmp_path)
    a, b, c, d, e, f, g = (uuid4() for _ in range(7))

    store.boost_edges([(a, b), (c, d), (e, f)], delta=0.1, edge_type="hebbian")

    versions_before = _versions(store)

    new = store.boost_edges(
        [(a, b), (c, d), (e, f), (a, c), (f, g)],
        delta=0.2,
        edge_type="hebbian",
    )

    versions_after = _versions(store)
    delta_versions = versions_after - versions_before

    assert delta_versions <= 2, (
        f"boost_edges emitted {delta_versions} versions "
        f"(expected <= 2 with batching)"
    )

    assert len(new) == 5
    for key, weight in new.items():
        if {key[0], key[1]} in ({str(a), str(b)}, {str(c), str(d)}, {str(e), str(f)}):
            assert abs(weight - 0.3) < 1e-5, f"{key} expected 0.3, got {weight}"
        else:
            assert abs(weight - 0.2) < 1e-5, f"{key} expected 0.2, got {weight}"


def test_boost_edges_scalar_delta_unchanged(tmp_path):
    store = MemoryStore(path=tmp_path)
    a, b, c, d = (uuid4() for _ in range(4))

    new = store.boost_edges([(a, b), (c, d)], delta=0.3, edge_type="hebbian")

    assert len(new) == 2
    for weight in new.values():
        assert abs(weight - 0.3) < 1e-5


def test_boost_edges_sequence_delta_per_pair(tmp_path):
    store = MemoryStore(path=tmp_path)
    a, b, c, d = (uuid4() for _ in range(4))

    new = store.boost_edges(
        [(a, b), (c, d)],
        delta=[0.5, 0.7],
        edge_type="hebbian",
    )

    assert len(new) == 2
    key_ab = tuple(sorted([str(a), str(b)]))
    key_cd = tuple(sorted([str(c), str(d)]))
    assert abs(new[key_ab] - 0.5) < 1e-5
    assert abs(new[key_cd] - 0.7) < 1e-5


def test_boost_edges_sequence_delta_length_mismatch_raises(tmp_path):
    store = MemoryStore(path=tmp_path)
    a, b, c, d = (uuid4() for _ in range(4))

    with pytest.raises(ValueError, match="deltas length"):
        store.boost_edges(
            [(a, b), (c, d)],
            delta=[0.5, 0.7, 0.9],
            edge_type="hebbian",
        )


def test_boost_edges_coalesces_duplicate_pairs(tmp_path):
    store = MemoryStore(path=tmp_path)
    a, b = uuid4(), uuid4()

    store.boost_edges([(a, b)], delta=0.1, edge_type="hebbian")

    new = store.boost_edges([(a, b), (a, b)], delta=0.1, edge_type="hebbian")

    assert len(new) == 1, "duplicate pair should collapse to ONE canonical key"
    canonical = tuple(sorted([str(a), str(b)]))
    assert abs(new[canonical] - 0.3) < 1e-5, (
        f"coalesced delta should be cur + 2*delta = 0.3, got {new[canonical]}"
    )


def test_boost_edges_coalesces_duplicate_pairs_first_call(tmp_path):
    store = MemoryStore(path=tmp_path)
    a, b = uuid4(), uuid4()

    new = store.boost_edges([(a, b), (a, b)], delta=0.1, edge_type="hebbian")
    canonical = tuple(sorted([str(a), str(b)]))
    assert abs(new[canonical] - 0.2) < 1e-5


def test_sleep_consolidated_from_batches_into_two_versions(tmp_path):
    from iai_mcp.sleep import _create_semantic_summary
    from tests.test_store import _make

    store = MemoryStore(path=tmp_path)

    cluster = [_make(text=f"source memory {i}") for i in range(5)]
    for r in cluster:
        store.insert(r)

    versions_before = _versions(store)
    summary_id, _folded = _create_semantic_summary(
        store,
        cluster,
        summary_text="cls summary of 5 source memories",
        language="en",
    )
    versions_after = _versions(store)

    delta_versions = versions_after - versions_before
    assert delta_versions <= 2, (
        f"sleep.cls boost emitted {delta_versions} versions for 5 sources "
        f"(expected <= 2 with batching)"
    )

    tbl = store.db.open_table(EDGES_TABLE)
    df = tbl.to_pandas()
    summary_str = str(summary_id)
    consolidated = df[
        (df["src"].isin([summary_str, *[str(r.id) for r in cluster]]))
        & (df["dst"].isin([summary_str, *[str(r.id) for r in cluster]]))
        & (df["edge_type"] == "consolidated_from")
    ]
    assert len(consolidated) == 5, (
        f"expected 5 consolidated_from edges, got {len(consolidated)}"
    )
    for w in consolidated["weight"]:
        assert abs(float(w) - 1.0) < 1e-5


def test_curiosity_bridge_batches_into_two_versions(tmp_path):
    from iai_mcp.curiosity import fire_curiosity
    from tests.test_store import _make

    store = MemoryStore(path=tmp_path)

    triggers = [_make(text=f"ambiguous memory {i}") for i in range(5)]
    for r in triggers:
        store.insert(r)

    class _Hit:
        def __init__(self, record_id):
            self.record_id = record_id
            self.score = 0.4

    hits = [_Hit(r.id) for r in triggers]

    versions_before = _versions(store)
    q = fire_curiosity(
        store,
        hits=hits,
        cue="what was that thing",
        entropy=1.5,
        session_id="sess-curiosity",
        turn=10,
    )
    versions_after = _versions(store)

    assert q is not None, "high-entropy curiosity call should fire"

    delta_versions = versions_after - versions_before
    assert delta_versions <= 2, (
        f"curiosity boost emitted {delta_versions} versions for 5 triggers "
        f"(expected <= 2 with batching)"
    )

    tbl = store.db.open_table(EDGES_TABLE)
    df = tbl.to_pandas()
    bridge = df[df["edge_type"] == "curiosity_bridge"]
    assert len(bridge) == 5, (
        f"expected 5 curiosity_bridge edges, got {len(bridge)}"
    )


def test_schema_bind_batches_into_two_versions(tmp_path):
    from iai_mcp.schema import SchemaCandidate, persist_schema
    from tests.test_store import _make

    store = MemoryStore(path=tmp_path)

    evidence = [_make(text=f"evidence {i}") for i in range(5)]
    for r in evidence:
        store.insert(r)

    candidate = SchemaCandidate(
        pattern="hebbian_test_pattern_unique",
        confidence=0.7,
        evidence_ids=[r.id for r in evidence],
        evidence_count=5,
        status="auto",
    )

    versions_before = _versions(store)
    schema_id = persist_schema(store, candidate)
    versions_after = _versions(store)

    assert schema_id is not None

    delta_versions = versions_after - versions_before
    assert delta_versions <= 2, (
        f"schema.bind boost emitted {delta_versions} versions for 5 evidence "
        f"(expected <= 2 with batching)"
    )

    tbl = store.db.open_table(EDGES_TABLE)
    df = tbl.to_pandas()
    instance_edges = df[df["edge_type"] == "schema_instance_of"]
    assert len(instance_edges) == 5, (
        f"expected 5 schema_instance_of edges, got {len(instance_edges)}"
    )


def test_pipeline_profile_modulates_batches_with_sequence_delta(tmp_path):
    from iai_mcp.pipeline import PROFILE_SENTINEL_UUID

    store = MemoryStore(path=tmp_path)

    record_ids = [uuid4() for _ in range(5)]
    gains_per_hit = [
        {"profile_match_strong": 0.4, "language_match": 0.1},
        {},
        {"profile_match_weak": 0.2},
        {"profile_match_neg": -0.5, "language_match": 0.1},
        {"profile_match_strong": 0.7},
    ]

    pairs: list[tuple] = []
    deltas: list[float] = []
    for rid, gains in zip(record_ids, gains_per_hit):
        if not gains:
            continue
        total_gain = float(sum(gains.values()))
        if total_gain <= 0:
            total_gain = 1.0
        pairs.append((rid, PROFILE_SENTINEL_UUID))
        deltas.append(total_gain)

    assert len(pairs) == 4, "4 hits should produce edges (1 skipped for empty gains)"
    assert len(deltas) == 4

    versions_before = _versions(store)
    new = store.boost_edges(
        pairs,
        delta=deltas,
        edge_type="profile_modulates",
    )
    versions_after = _versions(store)

    delta_versions = versions_after - versions_before
    assert delta_versions <= 2, (
        f"profile_modulates boost emitted {delta_versions} versions "
        f"(expected <= 2 with batching)"
    )

    assert len(new) == 4
    expected_per_pair = {
        tuple(sorted([str(record_ids[0]), str(PROFILE_SENTINEL_UUID)])): 0.5,
        tuple(sorted([str(record_ids[2]), str(PROFILE_SENTINEL_UUID)])): 0.2,
        tuple(sorted([str(record_ids[3]), str(PROFILE_SENTINEL_UUID)])): 1.0,
        tuple(sorted([str(record_ids[4]), str(PROFILE_SENTINEL_UUID)])): 0.7,
    }
    for key, exp in expected_per_pair.items():
        assert key in new, f"missing edge for {key}"
        assert abs(new[key] - exp) < 1e-5, (
            f"{key} expected {exp}, got {new[key]}"
        )
