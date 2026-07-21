"""Recall-parity guard for the double-buffered ANN index.

The recall ANN index is rebuilt by reusing a standby buffer across cycles
(mark_deleted-all + add_items(replace_deleted=True)) and committing via an
atomic tuple-swap under the recall lock, instead of allocating a fresh C++
index every cycle. These tests pin the two properties that make that reuse
safe:

  1. No fresh ``hnswlib.Index`` object is created on the steady-state reuse
     path (object-identity check on the two boot-allocated buffers).
  2. Recall quality is preserved across 1 / 10 / 100 reuse cycles
     (recall@10 >= 0.99 versus exact cosine ground truth, and within 0.01 of a
     fresh-allocation reference over the same vectors).
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord

_DIM = 32
_N_RECORDS = 250
_N_QUERIES = 30
_RECALL_FLOOR = 0.99


@pytest.fixture(autouse=True)
def _small_embed_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a small embedding dimension so the suite stays fast and never loads
    the real embedder (raw float32 vectors are written directly)."""
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(_DIM))


def _unit_vec(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(_DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def _make_record(vec: np.ndarray) -> MemoryRecord:
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
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )


def _seed_store(tmp_path, n: int, seed: int = 7) -> tuple[MemoryStore, np.random.Generator]:
    rng = np.random.default_rng(seed)
    store = MemoryStore(path=tmp_path)
    for _ in range(n):
        store.insert(_make_record(_unit_vec(rng)))
    return store, rng


def _label_vectors(store: MemoryStore) -> tuple[np.ndarray, np.ndarray]:
    """Read (vec_label, embedding) for the active corpus directly from sqlite.

    Returns parallel arrays: labels (int64) and normalized float32 vectors.
    """
    db = store.db
    with db._conn_lock:
        rows = db._conn.execute(
            "SELECT vec_label, embedding FROM records"
            " WHERE tombstoned_at IS NULL"
            " AND COALESCE(embedding_pending, 0) = 0"
            " ORDER BY vec_label"
        ).fetchall()
    labels = np.array([int(r["vec_label"]) for r in rows], dtype=np.int64)
    vecs = np.stack(
        [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
    )
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    vecs = vecs / norms
    return labels, vecs


def _exact_top10(query: np.ndarray, labels: np.ndarray, vecs: np.ndarray) -> set[int]:
    sims = vecs @ query
    top = np.argsort(-sims)[:10]
    return {int(labels[i]) for i in top}


def _knn_top10(store: MemoryStore, query: np.ndarray) -> set[int]:
    db = store.db
    with db._hnsw_lock:
        count = len(db._label_map)
        if count == 0:
            return set()
        k = min(10, count)
        found_labels, _dist = db._hnsw.knn_query(query, k=k)
    return {int(x) for x in found_labels[0].tolist()}


def _recall_at_10(predicted: set[int], truth: set[int]) -> float:
    denom = min(10, len(truth))
    if denom == 0:
        return 1.0
    return len(predicted & truth) / denom


def _mean_recall(store, queries, labels, vecs) -> float:
    scores = []
    for q in queries:
        truth = _exact_top10(q, labels, vecs)
        pred = _knn_top10(store, q)
        scores.append(_recall_at_10(pred, truth))
    return float(np.mean(scores))


def test_standby_allocated_at_boot(tmp_path):
    store = MemoryStore(path=tmp_path)
    try:
        db = store.db
        assert db._hnsw_standby is not None
        assert db._hnsw_standby is not db._hnsw
    finally:
        store.close()


def test_rebuild_reuses_standby_no_fresh_alloc(tmp_path):
    store, _rng = _seed_store(tmp_path, 40)
    try:
        db = store.db
        original_ids = {id(db._hnsw), id(db._hnsw_standby)}

        db._rebuild_index_from_sqlite()
        assert id(db._hnsw) in original_ids
        assert id(db._hnsw_standby) in original_ids
        assert id(db._hnsw) != id(db._hnsw_standby)

        db._rebuild_index_from_sqlite()
        assert id(db._hnsw) in original_ids
        assert id(db._hnsw_standby) in original_ids
        assert id(db._hnsw) != id(db._hnsw_standby)
    finally:
        store.close()


def test_rebuild_recall_parity_one_cycle(tmp_path):
    store, rng = _seed_store(tmp_path, _N_RECORDS)
    try:
        labels, vecs = _label_vectors(store)
        queries = [_unit_vec(rng) for _ in range(_N_QUERIES)]

        # Fresh-allocation reference over the SAME vectors.
        from iai_mcp.hippo import (
            HNSW_EF,
            HNSW_EF_CONSTRUCTION,
            HNSW_INITIAL_CAPACITY,
            HNSW_M,
            RECALL_INDEX_EF,
        )
        from iai_mcp.hippo import _vecindex as hnswlib

        cap = max(HNSW_INITIAL_CAPACITY, len(labels) * 2)
        ref = hnswlib.Index(space="cosine", dim=_DIM)
        ref.init_index(
            max_elements=cap,
            ef_construction=HNSW_EF_CONSTRUCTION,
            M=HNSW_M,
            allow_replace_deleted=True,
        )
        ref.set_ef(max(HNSW_EF, RECALL_INDEX_EF))
        ref.set_num_threads(1)
        ref.add_items(vecs, labels)

        ref_scores = []
        reuse_scores = []
        store.db._rebuild_index_from_sqlite()
        for q in queries:
            truth = _exact_top10(q, labels, vecs)
            ref_labels, _ = ref.knn_query(q, k=10)
            ref_pred = {int(x) for x in ref_labels[0].tolist()}
            ref_scores.append(_recall_at_10(ref_pred, truth))
            reuse_scores.append(_recall_at_10(_knn_top10(store, q), truth))

        reuse_recall = float(np.mean(reuse_scores))
        ref_recall = float(np.mean(ref_scores))
        assert reuse_recall >= _RECALL_FLOOR, reuse_recall
        assert abs(reuse_recall - ref_recall) <= 0.01, (reuse_recall, ref_recall)
    finally:
        store.close()


def test_rebuild_recall_parity_many_cycles(tmp_path):
    store, rng = _seed_store(tmp_path, _N_RECORDS)
    try:
        labels, vecs = _label_vectors(store)
        queries = [_unit_vec(rng) for _ in range(_N_QUERIES)]
        db = store.db

        for _ in range(10):
            db._rebuild_index_from_sqlite()
        assert _mean_recall(store, queries, labels, vecs) >= _RECALL_FLOOR

        for _ in range(90):  # 10 + 90 = 100 total reuse cycles
            db._rebuild_index_from_sqlite()
        assert _mean_recall(store, queries, labels, vecs) >= _RECALL_FLOOR
    finally:
        store.close()


def test_rebuild_empty_corpus_does_not_crash(tmp_path):
    store = MemoryStore(path=tmp_path)
    try:
        db = store.db
        # No records inserted; standby exists, the reuse path must no-op cleanly.
        result = db._rebuild_index_from_sqlite()
        assert result["action"] == "rebuild"
        assert result["rebuilt_count"] == 0
        assert len(db._label_map) == 0
    finally:
        store.close()


def test_boot_rebuild_before_standby_falls_back(tmp_path):
    store, rng = _seed_store(tmp_path, 60)
    try:
        db = store.db
        # Simulate the boot path: the standby is not yet allocated.
        db._hnsw_standby = None
        result = db._rebuild_index_from_sqlite()
        assert result["action"] == "rebuild"
        assert result["rebuilt_count"] == 60

        labels, vecs = _label_vectors(store)
        q = _unit_vec(rng)
        truth = _exact_top10(q, labels, vecs)
        assert _recall_at_10(_knn_top10(store, q), truth) >= _RECALL_FLOOR
    finally:
        store.close()
