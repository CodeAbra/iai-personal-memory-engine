"""Label-map / active-buffer consistency under repeated rebuild + churn.

The recall ANN index is double-buffered: the rebuild builds into a reused
standby buffer (mark-deleted-all + add with slot replacement) and commits via
an atomic tuple-swap of the active / standby buffers under the recall lock,
together with a label-map repopulation from SQLite — so a concurrent reader
always sees a consistent (buffer, label_map) pair.

These tests pin the recall-visible consistency invariant across many
insert / tombstone / untombstone churn cycles:

  * ``len(_label_map)`` equals the SQLite non-tombstoned & non-pending count;
  * the label-map keys equal the SQLite live ``vec_label`` set exactly;
  * every live record resolves to its own vector through the active buffer at
    near-zero distance (no stale or missing entry);
  * the boot / fresh-allocation rebuild path carries no soft-deleted slots, so
    the raw element count equals the live count there.

Note: on the steady-state *reuse* path, ``_hnsw.get_current_count()`` is the
raw element count and legitimately includes soft-deleted slots when the corpus
shrinks; those slots are excluded from recall and from the label map. The
invariant is over the recall-visible state, not the raw element count.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp.store import MemoryStore, RECORDS_TABLE
from iai_mcp.store._buffers import flush_record_buffer
from iai_mcp.types import MemoryRecord

_DIM = 32


@pytest.fixture(autouse=True)
def _small_embed_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(_DIM))


def _unit_vec(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(_DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def _make_record(vec: np.ndarray) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="fixture record for alice",
        aaak_index="",
        embedding=vec.tolist(),
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
        tags=[],
        language="en",
    )


def _indexable_count(db) -> int:
    with db._conn_lock:
        row = db._conn.execute(
            "SELECT COUNT(*) FROM records"
            " WHERE tombstoned_at IS NULL"
            " AND COALESCE(embedding_pending, 0) = 0"
        ).fetchone()
    return int(row[0])


def _live_labels(db) -> set[int]:
    with db._conn_lock:
        rows = db._conn.execute(
            "SELECT vec_label FROM records"
            " WHERE tombstoned_at IS NULL"
            " AND COALESCE(embedding_pending, 0) = 0"
        ).fetchall()
    return {int(r["vec_label"]) for r in rows}


def _assert_recall_visible_consistent(db) -> None:
    """Every live row resolves to its own vector and the label map matches
    SQLite's live vec_label set exactly."""
    sqlite_labels = _live_labels(db)
    assert set(db._label_map.values()) == sqlite_labels, (
        "label map vec_labels diverged from SQLite live set"
    )
    assert len(db._label_map) == len(sqlite_labels) == _indexable_count(db)

    with db._conn_lock:
        rows = db._conn.execute(
            "SELECT vec_label, embedding FROM records"
            " WHERE tombstoned_at IS NULL"
            " AND COALESCE(embedding_pending, 0) = 0"
        ).fetchall()
    with db._hnsw_lock:
        for r in rows:
            emb = np.frombuffer(r["embedding"], dtype=np.float32).reshape(1, -1)
            found, dist = db._hnsw.knn_query(emb, k=1)
            assert int(found[0][0]) == int(r["vec_label"]), (
                "live record resolved to a different label (buffer desync)"
            )
            assert float(dist[0][0]) < 1e-4, (
                "live record vector absent from active buffer (buffer desync)"
            )


def test_rebuild_churn_keeps_label_map_consistent(tmp_path):
    rng = np.random.default_rng(99)
    store = MemoryStore(path=tmp_path)
    try:
        db = store.db
        tbl = db.open_table(RECORDS_TABLE)

        for _ in range(20):
            store.insert(_make_record(_unit_vec(rng)))
        flush_record_buffer(store)

        for cyc in range(40):
            random.seed(cyc)
            for _ in range(random.randint(0, 6)):
                store.insert(_make_record(_unit_vec(rng)))
            flush_record_buffer(store)

            live = list(_live_labels_ids(db))
            for victim in random.sample(live, random.randint(0, min(4, len(live)))):
                tbl.update(
                    where=f"id = '{victim}'",
                    values={"tombstoned_at": datetime.now(timezone.utc)},
                )

            tomb = _tombstoned_ids(db)
            for restore in random.sample(tomb, min(random.randint(0, 2), len(tomb))):
                tbl.update(
                    where=f"id = '{restore}'",
                    values={"tombstoned_at": None},
                )

            db._rebuild_index_from_sqlite()
            _assert_recall_visible_consistent(db)
    finally:
        store.close()


def test_boot_fresh_rebuild_has_no_soft_delete_inflation(tmp_path):
    """On the boot / fresh-allocation path (no standby reuse) the raw element
    count must equal the live count — no soft-deleted slots carry over."""
    rng = np.random.default_rng(7)
    store = MemoryStore(path=tmp_path)
    try:
        db = store.db
        for _ in range(60):
            store.insert(_make_record(_unit_vec(rng)))
        flush_record_buffer(store)

        # Force the fresh-allocation branch by clearing the standby buffer.
        db._hnsw_standby = None
        db._rebuild_index_from_sqlite()

        live = _indexable_count(db)
        with db._hnsw_lock:
            assert db._hnsw.get_current_count() == live
        assert len(db._label_map) == live
        _assert_recall_visible_consistent(db)
    finally:
        store.close()


def test_reuse_rebuild_keeps_every_live_label_resolvable(tmp_path):
    """Direct record-loss guard for the steady-state reuse-path rebuild.

    A reuse-path rebuild refills the standby with labels that OVERLAP the labels
    already present in the reused buffer. The slot-reuse refill can drop a live
    record on some graph-builder draws; this asserts the post-rebuild active
    buffer still resolves EVERY live SQLite label (count and membership), with no
    "Label not found", so a dropped record is repaired before the swap commits.
    """
    rng = np.random.default_rng(123)
    store = MemoryStore(path=tmp_path)
    try:
        db = store.db
        tbl = db.open_table(RECORDS_TABLE)

        for _ in range(40):
            store.insert(_make_record(_unit_vec(rng)))
        flush_record_buffer(store)
        db._rebuild_index_from_sqlite()

        # A few churn cycles that keep most labels (heavy overlap) so the
        # reuse-path refill repeatedly exercises same-label slot reuse.
        for cyc in range(12):
            random.seed(1000 + cyc)
            for _ in range(random.randint(0, 5)):
                store.insert(_make_record(_unit_vec(rng)))
            flush_record_buffer(store)

            live_ids = list(_live_labels_ids(db))
            for victim in random.sample(live_ids, random.randint(0, min(3, len(live_ids)))):
                tbl.update(
                    where=f"id = '{victim}'",
                    values={"tombstoned_at": datetime.now(timezone.utc)},
                )

            db._rebuild_index_from_sqlite()

            sqlite_labels = _live_labels(db)
            n = len(sqlite_labels)
            assert n > 0
            with db._conn_lock:
                rows = db._conn.execute(
                    "SELECT vec_label, embedding FROM records"
                    " WHERE tombstoned_at IS NULL"
                    " AND COALESCE(embedding_pending, 0) = 0"
                ).fetchall()
            # Every live vector resolves to its own label through the active
            # buffer — a dropped label would surface as a wrong nearest or
            # "Label not found".
            with db._hnsw_lock:
                resolved = set()
                for r in rows:
                    emb = np.frombuffer(r["embedding"], dtype=np.float32).reshape(1, -1)
                    found, dist = db._hnsw.knn_query(emb, k=1)
                    assert int(found[0][0]) == int(r["vec_label"])
                    assert float(dist[0][0]) < 1e-4
                    resolved.add(int(r["vec_label"]))
                # No live label missing from the active buffer.
                assert resolved == sqlite_labels
                db._hnsw.get_items(list(sqlite_labels))  # raises if any dropped
    finally:
        store.close()


def _live_labels_ids(db) -> set[str]:
    with db._conn_lock:
        rows = db._conn.execute(
            "SELECT id FROM records"
            " WHERE tombstoned_at IS NULL"
            " AND COALESCE(embedding_pending, 0) = 0"
        ).fetchall()
    return {str(r["id"]) for r in rows}


def _tombstoned_ids(db) -> list[str]:
    with db._conn_lock:
        rows = db._conn.execute(
            "SELECT id FROM records WHERE tombstoned_at IS NOT NULL"
        ).fetchall()
    return [str(r["id"]) for r in rows]
