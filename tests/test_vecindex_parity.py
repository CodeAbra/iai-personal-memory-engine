"""The native exact index must match a numpy brute-force reference exactly.

The adapter presents the ANN-shaped surface the storage layer calls; these
tests pin the contract pieces recall relies on: exact top-k against a
reference sweep, cosine-distance values, update-in-place on label re-add,
the mark/unmark raise semantics the refill path branches on, save/load
round-trip, and the [1, k] result shapes.
"""

from __future__ import annotations

import numpy as np
import pytest


def _reference_topk(matrix: np.ndarray, query: np.ndarray, k: int):
    def norm(m):
        n = np.linalg.norm(m, axis=-1, keepdims=True)
        n[n == 0] = 1.0
        return m / n

    sims = norm(matrix) @ norm(query.reshape(1, -1)).T
    dists = 1.0 - sims.ravel()
    order = np.argsort(dists, kind="stable")[:k]
    return order, dists[order]


def _make_index(vecs: np.ndarray, labels: list[int]):
    from iai_mcp.hippo._vecindex import Index

    idx = Index(space="cosine", dim=vecs.shape[1])
    idx.init_index(max_elements=len(labels) * 2)
    idx.add_items(vecs, np.array(labels, dtype=np.int64))
    return idx


def test_topk_matches_numpy_reference_exactly():
    rng = np.random.default_rng(20260719)
    vecs = rng.standard_normal((500, 64)).astype(np.float32)
    labels = list(range(1000, 1500))
    idx = _make_index(vecs, labels)

    for _ in range(20):
        q = rng.standard_normal(64).astype(np.float32)
        got_labels, got_dists = idx.knn_query(q, k=10)
        ref_order, ref_dists = _reference_topk(vecs, q, 10)
        assert got_labels.shape == (1, 10)
        assert got_dists.shape == (1, 10)
        assert [labels[i] for i in ref_order] == got_labels[0].tolist()
        np.testing.assert_allclose(got_dists[0], ref_dists, atol=1e-5)


def test_update_in_place_and_count_semantics():
    vecs = np.eye(8, dtype=np.float32)[:3]
    idx = _make_index(vecs, [10, 20, 30])
    assert idx.get_current_count() == 3

    moved = np.zeros((1, 8), dtype=np.float32)
    moved[0, 7] = 1.0
    idx.add_items(moved, np.array([10], dtype=np.int64))
    assert idx.get_current_count() == 3, "re-add must update, not append"

    q = np.zeros(8, dtype=np.float32)
    q[7] = 1.0
    got, _ = idx.knn_query(q, k=1)
    assert got[0][0] == 10


def test_mark_unmark_raise_semantics():
    vecs = np.eye(4, dtype=np.float32)[:2]
    idx = _make_index(vecs, [1, 2])

    idx.mark_deleted(1)
    with pytest.raises(RuntimeError):
        idx.mark_deleted(1)  # already deleted
    with pytest.raises(RuntimeError):
        idx.mark_deleted(999)  # unknown label

    got, _ = idx.knn_query(np.eye(4, dtype=np.float32)[0], k=2)
    assert 1 not in got[0].tolist()

    idx.unmark_deleted(1)
    with pytest.raises(RuntimeError):
        idx.unmark_deleted(1)  # not deleted
    got, _ = idx.knn_query(np.eye(4, dtype=np.float32)[0], k=1)
    assert got[0][0] == 1


def test_save_load_roundtrip(tmp_path):
    from iai_mcp.hippo._vecindex import Index

    rng = np.random.default_rng(7)
    vecs = rng.standard_normal((50, 16)).astype(np.float32)
    labels = list(range(50))
    idx = _make_index(vecs, labels)
    idx.mark_deleted(13)
    path = tmp_path / "exact.idx"
    idx.save_index(str(path))

    loaded = Index(space="cosine", dim=16)
    loaded.load_index(str(path), max_elements=100)
    assert loaded.get_current_count() == 50
    assert sorted(loaded.get_ids_list()) == labels

    q = vecs[13]
    got, _ = loaded.knn_query(q, k=50)
    assert 13 not in got[0].tolist(), "deleted flag must survive the roundtrip"

    got_a, dist_a = idx.knn_query(q, k=5)
    got_b, dist_b = loaded.knn_query(q, k=5)
    assert got_a[0].tolist() == got_b[0].tolist()
    np.testing.assert_allclose(dist_a[0], dist_b[0], atol=1e-7)


def test_load_rejects_legacy_ann_file(tmp_path):
    from iai_mcp.hippo._vecindex import Index

    legacy = tmp_path / "records.hnsw"
    legacy.write_bytes(b"\x00" * 256)
    idx = Index(space="cosine", dim=16)
    with pytest.raises(Exception):
        idx.load_index(str(legacy), max_elements=10)


def test_query_speed_at_corpus_scale():
    import time

    rng = np.random.default_rng(3)
    vecs = rng.standard_normal((20_000, 384)).astype(np.float32)
    labels = list(range(20_000))
    idx = _make_index(vecs, labels)

    q = rng.standard_normal(384).astype(np.float32)
    idx.knn_query(q, k=30)  # warm
    t0 = time.perf_counter()
    for _ in range(10):
        idx.knn_query(q, k=30)
    per_query_ms = (time.perf_counter() - t0) / 10 * 1000
    assert per_query_ms < 50, f"exact sweep too slow: {per_query_ms:.1f}ms"
