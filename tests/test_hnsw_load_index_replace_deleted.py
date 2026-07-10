"""Pin the hnswlib 0.8.0 ``load_index`` contract for ``allow_replace_deleted``.

hnswlib 0.8.0 silently resets ``allow_replace_deleted`` to ``False`` on every
``load_index`` call when the kwarg is omitted, even if the saved index was
built with ``allow_replace_deleted=True``. Any subsequent
``mark_deleted(label)`` followed by ``add_items(..., replace_deleted=True)``
raises ``RuntimeError: Replacement of deleted elements is disabled in constructor``.

Every ``load_index`` call site in this project that hands the loaded index to a
write path that may issue ``mark_deleted`` + ``add_items(replace_deleted=True)``
MUST pass ``allow_replace_deleted=True``. This file pins both arms of the
contract — the bare-load failure mode AND the with-flag success mode — so any
regression that re-introduces a bare-load on a write path fails here.

Hermetic: uses ``tmp_path`` only, never touches the project store or the live
daemon. Depends only on the ``hnswlib>=0.8.0`` runtime dep already declared in
``pyproject.toml``.
"""

from __future__ import annotations

import hnswlib
import numpy as np
import pytest


_DIM = 8
_N_INITIAL = 5
_EF = 50
_M = 8
_EF_CONSTRUCTION = 100


def _build_and_save(tmp_path) -> str:
    """Build a small hnswlib index with the flag set and save it to disk."""
    vecs = np.random.rand(_N_INITIAL, _DIM).astype(np.float32)
    labels = np.arange(_N_INITIAL, dtype=np.int64)

    h = hnswlib.Index(space="cosine", dim=_DIM)
    h.init_index(
        max_elements=_N_INITIAL * 2,
        ef_construction=_EF_CONSTRUCTION,
        M=_M,
        allow_replace_deleted=True,
    )
    h.set_ef(_EF)
    h.add_items(vecs, labels)

    path = str(tmp_path / "regression.hnsw")
    h.save_index(path)
    return path


class TestPreFixBehaviour:
    """Without the kwarg, the load_index call demotes the flag to False and
    the next mark_deleted + add_items(replace_deleted=True) raises."""

    def test_load_without_flag_then_replace_raises(self, tmp_path):
        path = _build_and_save(tmp_path)

        loaded = hnswlib.Index(space="cosine", dim=_DIM)
        loaded.load_index(path, max_elements=_N_INITIAL * 2)  # flag absent
        loaded.set_ef(_EF)

        loaded.mark_deleted(0)
        with pytest.raises(
            RuntimeError,
            match="Replacement of deleted elements is disabled",
        ):
            loaded.add_items(
                np.random.rand(1, _DIM).astype(np.float32),
                np.array([10], dtype=np.int64),
                replace_deleted=True,
            )

    def test_load_without_flag_standby_pattern_also_raises(self, tmp_path):
        """The standby-buffer pattern (build into a fresh Index instance,
        load from a sibling file) has the same failure mode."""
        path = _build_and_save(tmp_path)

        standby = hnswlib.Index(space="cosine", dim=_DIM)
        standby.load_index(path, max_elements=_N_INITIAL * 2)  # flag absent

        standby.mark_deleted(1)
        with pytest.raises(
            RuntimeError,
            match="Replacement of deleted elements is disabled",
        ):
            standby.add_items(
                np.random.rand(1, _DIM).astype(np.float32),
                np.array([20], dtype=np.int64),
                replace_deleted=True,
            )


class TestPostFixBehaviour:
    """With the kwarg, the load_index call preserves the flag and the
    mark_deleted + add_items(replace_deleted=True) sequence succeeds."""

    def test_load_with_flag_then_replace_succeeds(self, tmp_path):
        path = _build_and_save(tmp_path)

        loaded = hnswlib.Index(space="cosine", dim=_DIM)
        loaded.load_index(
            path, max_elements=_N_INITIAL * 2, allow_replace_deleted=True,
        )
        loaded.set_ef(_EF)

        loaded.mark_deleted(0)
        loaded.add_items(
            np.random.rand(1, _DIM).astype(np.float32),
            np.array([10], dtype=np.int64),
            replace_deleted=True,
        )
        assert loaded.get_current_count() == _N_INITIAL

    def test_load_with_flag_standby_pattern_succeeds(self, tmp_path):
        path = _build_and_save(tmp_path)

        standby = hnswlib.Index(space="cosine", dim=_DIM)
        standby.load_index(
            path, max_elements=_N_INITIAL * 2, allow_replace_deleted=True,
        )
        standby.set_ef(_EF)

        standby.mark_deleted(1)
        standby.add_items(
            np.random.rand(1, _DIM).astype(np.float32),
            np.array([20], dtype=np.int64),
            replace_deleted=True,
        )
        assert standby.get_current_count() == _N_INITIAL
