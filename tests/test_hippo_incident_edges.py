from __future__ import annotations

from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord
from datetime import datetime, timezone


def _random_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _make_rec(tier: str = "episodic", text: str = "test", seed: int = 0) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
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
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    s = MemoryStore(str(tmp_path / "store"))
    yield s


def test_incident_edges_exists(store):
    assert hasattr(store, "incident_edges"), (
        "MemoryStore.incident_edges does not exist"
    )


def test_incident_edges_empty(store):
    r = _make_rec()
    store.insert(r)
    result = store.incident_edges([r.id])
    assert r.id in result
    assert result[r.id] == []


def test_incident_edges_both_orderings(store):
    a = _make_rec(seed=1)
    b = _make_rec(seed=2)
    store.insert(a)
    store.insert(b)
    store.boost_edges([(a.id, b.id)], delta=0.5, edge_type="hebbian")

    res_a = store.incident_edges([a.id])
    assert a.id in res_a, "incident_edges([A]) must return A's entry"
    neighbours_a = {t[0] for t in res_a[a.id]}
    assert b.id in neighbours_a, (
        "incident_edges([A]) must return B as A's neighbour"
    )

    res_b = store.incident_edges([b.id])
    assert b.id in res_b, "incident_edges([B]) must return B's entry"
    neighbours_b = {t[0] for t in res_b[b.id]}
    assert a.id in neighbours_b, (
        "incident_edges([B]) must return A as B's neighbour — "
        "tests that dst-stored seeds still spread"
    )


def test_incident_edges_batched_one_query(store):
    records = [_make_rec(seed=i) for i in range(3)]
    for r in records:
        store.insert(r)
    store.boost_edges([(records[0].id, records[1].id)], delta=0.3)

    ids = [r.id for r in records]
    result = store.incident_edges(ids)

    assert set(ids) == set(result.keys()), (
        f"Result must contain all input ids: {ids}"
    )
    r0_neighbours = {t[0] for t in result[records[0].id]}
    r1_neighbours = {t[0] for t in result[records[1].id]}
    assert records[1].id in r0_neighbours, "records[1] must be a neighbour of records[0]"
    assert records[0].id in r1_neighbours, "records[0] must be a neighbour of records[1]"
    assert result[records[2].id] == [], "records[2] with no edges must return empty list"


def test_incident_edges_parameterized_bind(store):
    import inspect
    src = inspect.getsource(store.incident_edges)
    assert '"?"' in src or "'?'" in src or "\"?\"\n" in src or "join(\"?\"" in src or 'join("?"' in src or "? " in src, (
        "incident_edges source must build parameterized placeholders with '?'"
    )

    a = _make_rec(seed=10)
    b = _make_rec(seed=11)
    store.insert(a)
    store.insert(b)
    store.boost_edges([(a.id, b.id)], delta=0.5)

    res_a = store.incident_edges([a.id])
    res_b = store.incident_edges([b.id])
    assert b.id in {t[0] for t in res_a.get(a.id, [])}, "B must be A's neighbour"
    assert a.id in {t[0] for t in res_b.get(b.id, [])}, "A must be B's neighbour"


def test_incident_edges_or_bind_two_placeholders(store):
    import inspect
    src = inspect.getsource(store.incident_edges)
    assert "src IN" in src or "src in" in src.lower(), (
        "incident_edges must use 'src IN (...)' in the OR-bind SQL"
    )
    assert "dst IN" in src or "dst in" in src.lower(), (
        "incident_edges must use 'dst IN (...)' in the OR-bind SQL"
    )

    a = _make_rec(seed=20)
    b = _make_rec(seed=21)
    store.insert(a)
    store.insert(b)
    store.boost_edges([(a.id, b.id)])

    result = store.incident_edges([a.id, b.id])
    assert b.id in {t[0] for t in result.get(a.id, [])}, "B must appear from A"
    assert a.id in {t[0] for t in result.get(b.id, [])}, "A must appear from B"


def test_incident_edges_top_k_cap(store):
    hub = _make_rec(seed=99)
    store.insert(hub)
    spokes = []
    for i in range(8):
        sp = _make_rec(seed=100 + i)
        store.insert(sp)
        spokes.append(sp)
        store.boost_edges([(hub.id, sp.id)], delta=float(i + 1) * 0.1)

    result = store.incident_edges([hub.id], top_k=5)
    assert hub.id in result
    assert len(result[hub.id]) <= 5, (
        f"Expected at most 5 neighbours (top_k=5), got {len(result[hub.id])}"
    )
    weights = [w for (_, _, w) in result[hub.id]]
    assert weights == sorted(weights, reverse=True), "Neighbours must be sorted by weight desc"


def test_incident_edges_uncapped_contradicts(store):
    hub = _make_rec(seed=200)
    store.insert(hub)

    for i in range(6):
        sp = _make_rec(seed=201 + i)
        store.insert(sp)
        store.boost_edges([(hub.id, sp.id)], delta=float(6 - i) * 0.2, edge_type="hebbian")

    contradicts_node = _make_rec(seed=210)
    store.insert(contradicts_node)
    store.boost_edges([(hub.id, contradicts_node.id)], delta=0.001, edge_type="contradicts")

    capped = store.incident_edges([hub.id], top_k=5)
    capped_neighbours = {t[0] for t in capped.get(hub.id, [])}

    uncapped = store.incident_edges([hub.id], top_k=None)
    uncapped_neighbours = {t[0] for t in uncapped.get(hub.id, [])}

    assert contradicts_node.id in uncapped_neighbours, (
        "top_k=None must return the low-weight contradicts edge"
    )

    assert len(capped_neighbours) <= 5


def test_incident_edges_edge_type_filter(store):
    a = _make_rec(seed=300)
    b = _make_rec(seed=301)
    c = _make_rec(seed=302)
    store.insert(a)
    store.insert(b)
    store.insert(c)
    store.boost_edges([(a.id, b.id)], delta=0.5, edge_type="hebbian")
    store.boost_edges([(a.id, c.id)], delta=0.5, edge_type="contradicts")

    result = store.incident_edges([a.id], edge_types=["contradicts"], top_k=None)
    assert a.id in result
    edge_types_returned = {et for (_, et, _) in result[a.id]}
    assert edge_types_returned == {"contradicts"}, (
        f"Expected only 'contradicts' edges, got {edge_types_returned}"
    )
    neighbour_ids = {t[0] for t in result[a.id]}
    assert c.id in neighbour_ids, "contradicts neighbour C must be present"
    assert b.id not in neighbour_ids, "hebbian neighbour B must be excluded"


def test_incident_edges_tuple_shape(store):
    a = _make_rec(seed=400)
    b = _make_rec(seed=401)
    store.insert(a)
    store.insert(b)
    store.boost_edges([(a.id, b.id)], delta=0.7, edge_type="hebbian")

    result = store.incident_edges([a.id])
    assert result[a.id], "Expected non-empty edge list"
    neighbour, edge_type, weight = result[a.id][0]
    assert isinstance(neighbour, UUID), f"neighbour must be UUID, got {type(neighbour)}"
    assert isinstance(edge_type, str), f"edge_type must be str, got {type(edge_type)}"
    assert isinstance(weight, float), f"weight must be float, got {type(weight)}"


def test_ef_raise_k200_no_error(store):
    n = 210
    for i in range(n):
        r = _make_rec(seed=500 + i)
        store.insert(r)

    cue = _random_vec(9999)
    results = store.query_similar(cue, k=200)
    assert len(results) >= 100, (
        f"Expected ~200 results from a {n}-record store at k=200, got {len(results)}"
    )
