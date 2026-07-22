"""Semantic minting must work from the real connective graph.

Guards the three failure modes that froze the semantic tier:

1. Starvation — clustering over cross-record hebbian edges alone yields zero
   clusters on a store whose hebbian edges are canonical self-loops only; the
   builder must cluster over every connective edge kind (pattern-separation
   seeds included).
2. Re-mint loops — an unchanged cluster must NOT be re-summarized every cycle;
   a cluster is re-minted only after it regrows past the coverage ratio.
3. Silent wake path — co-recalled records must gain pairwise hebbian edges
   (bounded), otherwise the connective graph never reflects co-activation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from iai_mcp.types import EMBED_DIM, MemoryRecord


def _record(text: str, i: int) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    emb = [0.0] * EMBED_DIM
    emb[i % EMBED_DIM] = 1.0
    emb[(i * 31 + 7) % EMBED_DIM] = 0.5
    emb[(i * 53 + 13) % EMBED_DIM] = 0.25
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=emb,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.5,
        difficulty=0.3,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )


def _persisted_ids(store) -> list[UUID]:
    with store.db._conn_lock:
        rows = store.db._conn.execute(
            "SELECT id FROM records ORDER BY rowid"
        ).fetchall()
    return [UUID(str(r[0])) for r in rows]


def _seed(store, n: int) -> list[UUID]:
    from iai_mcp.store._buffers import flush_record_buffer

    for i in range(n):
        store.insert(_record(f"knowledge minting fixture row {i}", i))
    flush_record_buffer(store)
    return _persisted_ids(store)


def _psep_triangles(store, ids: list[UUID], n_triangles: int) -> None:
    from iai_mcp.store._buffers import flush_edge_buffer

    pairs = []
    for t in range(n_triangles):
        a, b, c = ids[3 * t], ids[3 * t + 1], ids[3 * t + 2]
        pairs += [(a, b), (b, c), (a, c)]
    store.boost_edges(pairs, delta=0.1, edge_type="pattern_separation_seed")
    flush_edge_buffer(store)


def _semantic_summaries(store) -> list[MemoryRecord]:
    return [
        r
        for r in store.all_records()
        if r.tier == "semantic" and "cls_summary" in (r.tags or [])
    ]


def test_psep_only_graph_mints_summaries(tmp_path):
    """A store with ZERO cross-record hebbian edges (self-loops only, the live
    failure mode) must still mint one summary per seed-linked cluster."""
    from iai_mcp.sleep import _process_cluster_summaries
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    ids = _seed(store, 9)
    _psep_triangles(store, ids, 3)

    created = _process_cluster_summaries(store)
    assert created == 3, f"expected 3 summaries from 3 seed triangles, got {created}"

    summaries = _semantic_summaries(store)
    assert len(summaries) == 3
    for s in summaries:
        assert s.never_decay is True
        assert "Cluster summary (3 records" in s.literal_surface


def test_community_id_stamped_on_members(tmp_path):
    from iai_mcp.sleep import _process_cluster_summaries
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    ids = _seed(store, 6)
    _psep_triangles(store, ids, 2)
    _process_cluster_summaries(store)

    clustered = ids[:6]
    stamped = {
        rid: rec.community_id
        for rid, rec in ((r.id, r) for r in store.all_records())
        if rid in clustered
    }
    assert all(v is not None and str(v) != "" for v in stamped.values()), (
        f"community_id missing after minting: {stamped}"
    )
    assert len({str(v) for v in stamped.values()}) == 2, (
        "two disjoint triangles must carry two distinct community ids"
    )


def test_unchanged_cluster_not_reminted(tmp_path):
    from iai_mcp.sleep import _process_cluster_summaries
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    ids = _seed(store, 6)
    _psep_triangles(store, ids, 2)

    first = _process_cluster_summaries(store)
    assert first == 2
    second = _process_cluster_summaries(store)
    assert second == 0, (
        f"unchanged clusters were re-minted: {second} new summaries on rerun"
    )
    assert len(_semantic_summaries(store)) == 2


def test_regrown_cluster_is_reminted(tmp_path):
    from iai_mcp.sleep import _process_cluster_summaries
    from iai_mcp.store import MemoryStore
    from iai_mcp.store._buffers import flush_edge_buffer, flush_record_buffer

    store = MemoryStore(path=tmp_path)
    ids = _seed(store, 3)
    _psep_triangles(store, ids, 1)
    assert _process_cluster_summaries(store) == 1

    for i in range(3, 8):
        store.insert(_record(f"knowledge minting fixture row {i}", i))
    flush_record_buffer(store)
    all_ids = _persisted_ids(store)
    grown = [rid for rid in all_ids if rid in set(ids)] + [
        rid for rid in all_ids if rid not in set(ids)
    ][:8]
    pairs = [(grown[i], grown[i + 1]) for i in range(len(grown) - 1)]
    store.boost_edges(pairs, delta=0.1, edge_type="pattern_separation_seed")
    flush_edge_buffer(store)

    reminted = _process_cluster_summaries(store)
    assert reminted == 1, (
        f"cluster regrown 3->8 members must re-mint exactly once, got {reminted}"
    )


def test_low_weight_edges_do_not_cluster(tmp_path):
    from iai_mcp.sleep import CLUSTER_EDGE_MIN_WEIGHT, _process_cluster_summaries
    from iai_mcp.store import MemoryStore
    from iai_mcp.store._buffers import flush_edge_buffer

    store = MemoryStore(path=tmp_path)
    ids = _seed(store, 3)
    weak = CLUSTER_EDGE_MIN_WEIGHT / 2.0
    store.boost_edges(
        [(ids[0], ids[1]), (ids[1], ids[2]), (ids[0], ids[2])],
        delta=weak,
        edge_type="pattern_separation_seed",
    )
    flush_edge_buffer(store)
    assert _process_cluster_summaries(store) == 0


def test_coactivation_wires_hebbian_pairs(tmp_path):
    from iai_mcp.retrieve import potentiate_coactivation
    from iai_mcp.store import EDGES_TABLE, MemoryStore
    from iai_mcp.store._buffers import flush_edge_buffer

    store = MemoryStore(path=tmp_path)
    ids = _seed(store, 3)

    wired = potentiate_coactivation(store, ids)
    assert wired == 3, f"3 co-recalled ids must wire 3 pairs, got {wired}"
    flush_edge_buffer(store)

    df = store.db.open_table(EDGES_TABLE).to_pandas()
    hebbian = df[(df["edge_type"] == "hebbian") & (df["src"] != df["dst"])]
    assert len(hebbian) == 3


def test_coactivation_kill_switch_and_bounds(tmp_path, monkeypatch):
    from iai_mcp.retrieve import (
        WAKE_COACTIVATION_MAX_HITS,
        potentiate_coactivation,
    )
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    ids = _seed(store, 8)

    monkeypatch.setenv("IAI_MCP_WAKE_COACTIVATION", "0")
    assert potentiate_coactivation(store, ids) == 0
    monkeypatch.delenv("IAI_MCP_WAKE_COACTIVATION")

    assert potentiate_coactivation(store, ids[:1]) == 0

    max_pairs = WAKE_COACTIVATION_MAX_HITS * (WAKE_COACTIVATION_MAX_HITS - 1) // 2
    assert potentiate_coactivation(store, ids) == max_pairs


def test_tombstoned_summary_does_not_block_reminting(tmp_path):
    """A summary that was later tombstoned must stop counting as coverage —
    otherwise its cluster is never re-minted after the summary dies."""
    from datetime import datetime, timezone

    from iai_mcp.sleep import _process_cluster_summaries
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    ids = _seed(store, 3)
    _psep_triangles(store, ids, 1)
    assert _process_cluster_summaries(store) == 1

    summary = _semantic_summaries(store)[0]
    store.db.open_table("records").update(
        where=f"id = '{summary.id}'",
        values={"tombstoned_at": datetime.now(timezone.utc).isoformat()},
    )

    assert _process_cluster_summaries(store) == 1, (
        "cluster must re-mint after its only summary was tombstoned"
    )


def test_coactivation_defers_through_reinforce_queue(tmp_path):
    """With the queue live, the recall path must not write edges inline —
    the pairs land only when the background worker drains."""
    from iai_mcp.retrieve import potentiate_coactivation
    from iai_mcp.store import EDGES_TABLE, MemoryStore
    from iai_mcp.store._buffers import flush_edge_buffer

    store = MemoryStore(path=tmp_path)
    ids = _seed(store, 3)
    store.enable_reinforce_queue(coalesce_ms=10)
    try:
        wired = potentiate_coactivation(store, ids)
        assert wired == 3
        store._reinforce_queue.flush(timeout=5.0)
        flush_edge_buffer(store)
        df = store.db.open_table(EDGES_TABLE).to_pandas()
        hebbian = df[(df["edge_type"] == "hebbian") & (df["src"] != df["dst"])]
        assert len(hebbian) == 3, (
            f"queued co-activation pairs must land after drain, got {len(hebbian)}"
        )
    finally:
        store.disable_reinforce_queue()


def test_oversize_cluster_stamped_but_not_summarized(tmp_path, monkeypatch):
    """A cluster past the summary-size cap keeps its community identity but
    must not become a mega-summary."""
    import iai_mcp.sleep as sleep_mod
    from iai_mcp.sleep import _process_cluster_summaries
    from iai_mcp.store import MemoryStore
    from iai_mcp.store._buffers import flush_edge_buffer

    store = MemoryStore(path=tmp_path)
    ids = _seed(store, 6)
    pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]
    store.boost_edges(pairs, delta=0.1, edge_type="pattern_separation_seed")
    flush_edge_buffer(store)

    monkeypatch.setattr(sleep_mod, "CLUSTER_MAX_SUMMARY_MEMBERS", 4)
    created = _process_cluster_summaries(store)
    assert created == 0, "oversize cluster must not be summarized"
    assert len(_semantic_summaries(store)) == 0

    stamped = [
        r.community_id
        for r in store.all_records()
        if r.tier == "episodic" and r.community_id
    ]
    assert len(stamped) == 6, "oversize cluster must still be stamped"
