"""Consolidation-edge backfill.

A summary's consolidated_from edges are its coverage ledger; historical
edge loss left old summaries anchored to a fraction of their cluster. The
backfill expands structurally only — an anchored summary claims the rest
of its community's live episodic members; edgeless summaries are counted,
never guessed at.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from iai_mcp.migrate import backfill_consolidated_edges
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _rec(text: str, i: int, *, tier: str = "episodic", community: str | None = None) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    emb = [0.0] * EMBED_DIM
    emb[i % EMBED_DIM] = 1.0
    emb[(i * 31 + 7) % EMBED_DIM] = 0.5
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=emb,
        community_id=community,
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


def _seed_world(store: MemoryStore):
    """4-member community with a one-edge summary, a stray community, and an
    edgeless summary."""
    comm_one, comm_two = str(uuid4()), str(uuid4())
    members = [_rec(f"community one member {i}", i, community=comm_one) for i in range(4)]
    strays = [_rec(f"stray community member {i}", 10 + i, community=comm_two) for i in range(2)]
    summary = _rec("Knowledge: community one condensed.", 20, tier="semantic")
    edgeless = _rec("Knowledge: an orphaned summary.", 21, tier="semantic")
    for r in members + strays + [summary, edgeless]:
        store.insert(r)
    flush_record_buffer(store)
    store.boost_edges(
        [(summary.id, members[0].id)], edge_type="consolidated_from", delta=1.0
    )
    return members, strays, summary, edgeless


def _cf_edge_count(store: MemoryStore) -> int:
    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT COUNT(*) FROM edges WHERE edge_type = 'consolidated_from'"
        ).fetchone()
    return int(row[0])


def test_dry_run_proposes_without_writing(tmp_path):
    store = MemoryStore(path=tmp_path)
    _seed_world(store)

    summary = backfill_consolidated_edges(store, apply=False)
    assert summary["mode"] == "dry-run"
    assert summary["summaries_live"] == 2
    assert summary["summaries_anchored"] == 1
    assert summary["summaries_edgeless"] == 1
    assert summary["links_existing"] == 1
    assert summary["links_proposed"] == 3, (
        "the anchored summary must claim the 3 remaining community members"
    )
    assert summary["links_written"] == 0
    assert summary["snapshot_dir"] is None
    assert _cf_edge_count(store) == 1, "dry-run must not write edges"


def test_apply_writes_snapshots_and_is_idempotent(tmp_path):
    store = MemoryStore(path=tmp_path)
    _seed_world(store)

    applied = backfill_consolidated_edges(store, apply=True, store_path=tmp_path)
    assert applied["links_written"] == 3
    assert applied["snapshot_dir"] is not None
    assert _cf_edge_count(store) == 4

    from iai_mcp.brainview import _knowledge_coverage
    with store.db._conn_lock:
        cov = _knowledge_coverage(store.db._conn)
    assert abs(cov - 4 / 6) < 1e-6, (
        f"4 of 6 live episodic records are now covered, got {cov}"
    )

    rerun = backfill_consolidated_edges(store, apply=False)
    assert rerun["links_proposed"] == 0, "backfill must be idempotent"


def test_oversize_community_is_never_claimed(tmp_path, monkeypatch):
    from iai_mcp.migrate import _cleanup

    monkeypatch.setattr(_cleanup, "CONSOLIDATION_BACKFILL_OVERSIZE", 3)
    store = MemoryStore(path=tmp_path)
    _seed_world(store)

    summary = backfill_consolidated_edges(store, apply=False)
    assert summary["links_proposed"] == 0
    assert summary["communities_oversize_skipped"] == 1


def test_tombstoned_records_are_invisible(tmp_path):
    store = MemoryStore(path=tmp_path)
    members, _strays, summary, _edgeless = _seed_world(store)

    now_iso = datetime.now(timezone.utc).isoformat()
    tbl = store.db.open_table("records")
    tbl.update(where=f"id = '{members[3].id}'", values={"tombstoned_at": now_iso})

    result = backfill_consolidated_edges(store, apply=False)
    assert result["links_proposed"] == 2, (
        "a tombstoned community member must never be proposed"
    )
