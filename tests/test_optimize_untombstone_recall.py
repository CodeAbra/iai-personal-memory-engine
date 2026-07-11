"""Recall reconciliation for the storage-optimization untombstone path.

The storage-optimization step restores pinned / never-decay records that were
anomalously tombstoned (a defensive safety net — the erasure agent never
tombstones such records on the normal path). That restore is a plain SQLite
UPDATE that flips ``tombstoned_at`` back to NULL; it does not touch the ANN
recall index. The index rebuild that runs at the start of the same step
measures the corpus *before* the restore, so without an explicit
reconciliation the restored record is live in SQLite yet absent from the
recall index and its label map — semantically unrecallable.

This test drives the full storage-optimization step (not the rebuild in
isolation) and asserts a restored pinned record is both live in SQLite and
recallable through the ANN index afterwards.
"""

from __future__ import annotations

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


def _make_record(vec: np.ndarray, *, pinned: bool = False) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="fixture record for alice",
        aaak_index="",
        embedding=vec.tolist(),
        community_id=None,
        centrality=0.9,
        detail_level=2,
        pinned=pinned,
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


class _PipelineShim:
    """Minimal driver exposing only what the optimize step reads."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def _check_interrupt(self, *_args, **_kwargs) -> bool:
        return False

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)


def test_optimize_untombstone_restores_pinned_record_to_recall_index(tmp_path):
    """A pinned record tombstoned out-of-band and restored by the optimize
    step must be live in SQLite AND recallable through the ANN index after the
    full step runs."""
    from iai_mcp.lilli.cycle.sleep_pipeline._optimize import step_compact_hippo

    rng = np.random.default_rng(5)
    store = MemoryStore(path=tmp_path)
    try:
        db = store.db
        tbl = db.open_table(RECORDS_TABLE)

        seed_vec = _unit_vec(rng)
        seed = _make_record(seed_vec, pinned=True)
        store.insert(seed)
        for _ in range(20):
            store.insert(_make_record(_unit_vec(rng)))
        flush_record_buffer(store)

        seed_label = db._label_map.get(str(seed.id))
        assert seed_label is not None, "seed not indexed after flush"

        # Anomalous / defensive trigger: a pinned record gets tombstoned
        # out-of-band (the erasure agent never does this on the normal path).
        tbl.update(
            where=f"id = '{seed.id}'",
            values={"tombstoned_at": datetime.now(timezone.utc)},
        )

        done, info = step_compact_hippo(_PipelineShim(store), None)
        assert done is True
        assert info["count_untombstoned_by_pin_override"] >= 1

        # SQLite: the pinned record is restored (tombstoned_at cleared).
        with db._conn_lock:
            row = db._conn.execute(
                f"SELECT tombstoned_at FROM records WHERE id = '{seed.id}'"
            ).fetchone()
        assert row is not None
        assert row["tombstoned_at"] is None, "pinned record not restored in SQLite"

        # Recall index: the restored record is back in the label map and
        # resolves to its own vector through the ANN index.
        assert str(seed.id) in db._label_map, (
            "restored pinned record absent from recall label map"
        )
        with db._hnsw_lock:
            found, dist = db._hnsw.knn_query(
                seed_vec.reshape(1, -1).astype(np.float32), k=1
            )
        assert int(found[0][0]) == int(db._label_map[str(seed.id)])
        assert float(dist[0][0]) < 1e-4, (
            "restored pinned record not recallable via ANN index"
        )
    finally:
        store.close()
