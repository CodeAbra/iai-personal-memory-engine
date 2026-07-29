from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from iai_mcp.entity_link import ENTITY_EDGE_TYPE, mine_entity_edges
from iai_mcp.graph import MemoryGraph
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _vec(seed: int) -> list[float]:
    v = [0.0] * EMBED_DIM
    v[seed % EMBED_DIM] = 1.0
    return v


def _rec(text: str, seed: int) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(), tier="episodic", literal_surface=text, aaak_index="",
        embedding=_vec(seed), community_id=None, centrality=0.0,
        detail_level=2, pinned=False, stability=0.0, difficulty=0.0,
        last_reviewed=None, never_decay=False, never_merge=False,
        provenance=[], created_at=now, updated_at=now, tags=["t"],
        language="en",
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-entity-link")
    from iai_mcp.store import MemoryStore, flush_record_buffer
    s = MemoryStore(path=tmp_path / "lancedb")
    s._flush = lambda: flush_record_buffer(s)
    return s


def _edges_of_type(store, edge_type: str) -> "list[tuple[str, str]]":
    with store.db.ro_conn() as conn:
        rows = conn.execute(
            "SELECT src, dst FROM edges WHERE edge_type = ?", (edge_type,)
        ).fetchall()
    return [(str(r[0]), str(r[1])) for r in rows]


def test_shared_rare_token_mints_edge(store) -> None:
    a = _rec("the zephyrblatt rotor needs a caffeine-free lubricant", 1)
    b = _rec("I calibrated the zephyrblatt housing again yesterday", 2)
    c = _rec("completely unrelated grocery discussion about apples", 3)
    for r in (a, b, c):
        store.insert(r)
    store._flush()

    result = mine_entity_edges(store)
    assert result["edges_minted"] >= 1

    pairs = _edges_of_type(store, ENTITY_EDGE_TYPE)
    linked = {frozenset(p) for p in pairs}
    assert frozenset({str(a.id), str(b.id)}) in linked
    assert not any(str(c.id) in p for p in linked), (
        "record without the shared anchor must not be linked"
    )


def test_mining_is_idempotent(store) -> None:
    a = _rec("the quasarbolt fixture hums at night", 1)
    b = _rec("replaced the quasarbolt fixture mount", 2)
    for r in (a, b):
        store.insert(r)
    store._flush()

    first = mine_entity_edges(store)
    n_after_first = len(_edges_of_type(store, ENTITY_EDGE_TYPE))
    assert first["edges_minted"] >= 1
    mine_entity_edges(store)
    n_after_second = len(_edges_of_type(store, ENTITY_EDGE_TYPE))
    assert n_after_second == n_after_first


def test_df_band_and_grade_filters(store) -> None:
    # "cat" is below the length floor; a df=1 token has nobody to link;
    # a token above max_df is ambient vocabulary, not an entity.
    records = [_rec(f"ubiquitous flanged widget number {i} cat", i) for i in range(6)]
    solo = _rec("the xylocopter appears exactly once", 99)
    for r in records + [solo]:
        store.insert(r)
    store._flush()

    result = mine_entity_edges(store, max_df=4)
    pairs = _edges_of_type(store, ENTITY_EDGE_TYPE)
    assert not any(str(solo.id) in p for p in pairs), "df=1 token linked"
    # "ubiquitous"/"flanged"/"widget"/"number" all have df=6 > max_df=4.
    assert result["edges_minted"] == 0


def test_edge_cap_respected(store) -> None:
    records = [_rec(f"shared glimmerforge anchor variant {i}", i) for i in range(5)]
    for r in records:
        store.insert(r)
    store._flush()

    result = mine_entity_edges(store, max_edges_per_run=3)
    assert result["edges_minted"] <= 3


def test_entity_edges_excluded_from_ranking_degree_but_traversed() -> None:
    g = MemoryGraph()
    a, b, c = uuid4(), uuid4(), uuid4()
    for n in (a, b, c):
        g.add_node(n, community_id=None, embedding=[0.0] * EMBED_DIM)
    g.add_edge(a, b, weight=0.3, edge_type=ENTITY_EDGE_TYPE)
    g.add_edge(b, c, weight=1.0, edge_type="hebbian")

    earned = dict(g.degrees(exclude_types=g.RANKING_DEGREE_EXCLUDED))
    assert earned[a] == 0, "entity edge must not count toward ranking degree"
    assert earned[b] == 1

    reached = set(g.two_hop_neighborhood([a], top_k=5))
    assert b in reached and c in reached, (
        "spread must traverse entity edges even though rank ignores them"
    )


def test_sleep_step_registered_at_tail() -> None:
    from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline, SleepStep

    assert SleepStep.ENTITY_LINK.value == 14
    assert SleepPipeline._STEP_ORDER[-1] is SleepStep.ENTITY_LINK
    assert SleepPipeline._STEP_ORDER.index(SleepStep.RECALL_INDEX_REBUILD) == 12
    assert callable(SleepPipeline._step_entity_link)
