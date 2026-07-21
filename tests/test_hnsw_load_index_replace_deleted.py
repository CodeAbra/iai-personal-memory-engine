"""Pin the vector-index load/add contract around deleted labels.

The previous ANN library gated slot reuse behind a fragile flag pairing:
``load_index`` silently reset ``allow_replace_deleted`` unless the kwarg was
re-passed, and a later ``add_items(..., replace_deleted=True)`` then raised —
a whole class of load-site hazards this project had to pin test-by-test.

The exact index eliminates the class: adds never depend on load flags, an
existing label (deleted or not) updates its slot in place and revives it, and
a NEW label appends regardless of any flag. These tests pin THAT contract so
a future backend swap cannot silently reintroduce flag-gated adds.

Hermetic: uses ``tmp_path`` only, never touches the project store or the
live daemon.
"""

from __future__ import annotations

import numpy as np

from iai_mcp.hippo import _vecindex

_DIM = 8
_N_INITIAL = 5


def _build_and_save(tmp_path) -> str:
    idx = _vecindex.Index(space="cosine", dim=_DIM)
    idx.init_index(max_elements=32)
    vecs = np.eye(_DIM, dtype=np.float32)[:_N_INITIAL]
    idx.add_items(vecs, np.arange(_N_INITIAL, dtype=np.int64))
    idx.mark_deleted(2)
    path = str(tmp_path / "exact.idx")
    idx.save_index(path)
    return path


def test_bare_load_never_gates_adds(tmp_path):
    """A load WITHOUT any flag must leave the index fully writable — the
    old library's silent flag reset on bare loads is the hazard this pins
    out of existence."""
    path = _build_and_save(tmp_path)
    loaded = _vecindex.Index(space="cosine", dim=_DIM)
    loaded.load_index(path)

    fresh = np.ones((1, _DIM), dtype=np.float32)
    loaded.add_items(fresh, np.array([100], dtype=np.int64))
    assert 100 in loaded.get_ids_list()


def test_readd_of_deleted_label_revives_in_place(tmp_path):
    path = _build_and_save(tmp_path)
    loaded = _vecindex.Index(space="cosine", dim=_DIM)
    loaded.load_index(path, max_elements=32)

    replacement = np.zeros((1, _DIM), dtype=np.float32)
    replacement[0, 7] = 1.0
    loaded.add_items(replacement, np.array([2], dtype=np.int64))

    assert loaded.get_current_count() == _N_INITIAL, (
        "re-adding an existing label must update its slot, not append"
    )
    q = np.zeros(_DIM, dtype=np.float32)
    q[7] = 1.0
    labels, _ = loaded.knn_query(q, k=1)
    assert labels[0][0] == 2, "revived label must be queryable at its new vector"


def test_new_label_appends_regardless_of_replace_flag(tmp_path):
    path = _build_and_save(tmp_path)
    loaded = _vecindex.Index(space="cosine", dim=_DIM)
    loaded.load_index(path, max_elements=32)

    fresh = np.ones((1, _DIM), dtype=np.float32)
    loaded.add_items(fresh, np.array([200], dtype=np.int64), replace_deleted=True)
    assert loaded.get_current_count() == _N_INITIAL + 1
    assert 200 in loaded.get_ids_list()
    # The deleted label stays deleted — a new label must never cannibalize
    # a tombstoned slot's identity.
    q = np.eye(_DIM, dtype=np.float32)[2]
    labels, _ = loaded.knn_query(q, k=_N_INITIAL + 1)
    assert 2 not in labels[0].tolist()


def test_standby_pattern_load_then_refill(tmp_path):
    """The double-buffer standby loads a snapshot and is refilled by the
    rebuild: survivors update in place, stale labels mark deleted, new
    labels append — all without any load-time capability flag."""
    path = _build_and_save(tmp_path)
    standby = _vecindex.Index(space="cosine", dim=_DIM)
    standby.load_index(path, max_elements=64)

    survivors = np.eye(_DIM, dtype=np.float32)[:2]
    standby.add_items(survivors, np.arange(2, dtype=np.int64))
    standby.mark_deleted(4)
    fresh = np.full((1, _DIM), 0.5, dtype=np.float32)
    standby.add_items(fresh, np.array([300], dtype=np.int64), replace_deleted=True)

    assert standby.get_current_count() == _N_INITIAL + 1
    ids = set(standby.get_ids_list())
    assert {0, 1, 2, 3, 4, 300} == ids
