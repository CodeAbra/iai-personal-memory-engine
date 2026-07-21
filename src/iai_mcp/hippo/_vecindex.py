"""ANN-shaped adapter over the native exact cosine index.

Presents the constructor/init/load/query surface the storage layer was
written against, backed by ``iai_mcp_native.vec.ExactIndex`` — an exact
sweep over L2-normalized vectors. At this store's corpus scale the exact
sweep answers in single-digit milliseconds with perfect recall, so the
approximate graph parameters (``M``, ``ef``) are accepted and ignored.

Error parity is load-bearing: ``mark_deleted`` on an already-deleted label
and ``unmark_deleted`` on a live label both raise ``RuntimeError``, exactly
like the previous index — the refill path branches on those raises.
"""

from __future__ import annotations

import numpy as np

from iai_mcp_native.vec import ExactIndex


class Index:

    def __init__(self, space: str = "cosine", dim: int = 0) -> None:
        if space != "cosine":
            raise ValueError(f"unsupported space {space!r}: only cosine")
        self._dim = int(dim)
        self._impl: ExactIndex | None = None
        # Recorded search-width knob: exact search ignores it, but callers
        # raise-and-restore it around big-k queries and assert the restore.
        self.ef: int = 0

    def _require(self) -> ExactIndex:
        if self._impl is None:
            raise RuntimeError("index not initialized (call init_index or load_index)")
        return self._impl

    def init_index(
        self,
        max_elements: int,
        ef_construction: int = 0,
        M: int = 0,  # noqa: N803 -- mirrors the ANN library's parameter name
        allow_replace_deleted: bool = True,
    ) -> None:
        del ef_construction, M, allow_replace_deleted
        self._impl = ExactIndex(self._dim, int(max_elements))

    def load_index(
        self,
        path: str,
        max_elements: int = 0,
        allow_replace_deleted: bool = True,
    ) -> None:
        del allow_replace_deleted
        self._impl = ExactIndex.load(str(path), int(max_elements))

    def save_index(self, path: str) -> None:
        self._require().save_index(str(path))

    def set_ef(self, ef: int) -> None:
        self.ef = int(ef)

    def set_num_threads(self, n: int) -> None:
        del n

    def add_items(
        self,
        vecs: "np.ndarray",
        labels: "np.ndarray",
        replace_deleted: bool = False,
    ) -> None:
        del replace_deleted
        arr = np.ascontiguousarray(vecs, dtype=np.float32)
        label_list = [int(x) for x in np.asarray(labels).ravel()]
        self._require().add_items(arr, label_list)

    def knn_query(self, vec: "np.ndarray", k: int = 1):
        arr = np.ascontiguousarray(vec, dtype=np.float32)
        return self._require().knn_query(arr, int(k))

    def mark_deleted(self, label: int) -> None:
        self._require().mark_deleted(int(label))

    def unmark_deleted(self, label: int) -> None:
        self._require().unmark_deleted(int(label))

    def get_current_count(self) -> int:
        return self._require().get_current_count()

    def get_ids_list(self) -> list[int]:
        return self._require().get_ids_list()

    def get_items(self, labels) -> "np.ndarray":
        rows = self._require().get_items([int(x) for x in labels])
        return np.asarray(rows, dtype=np.float32)

    def get_max_elements(self) -> int:
        return self._require().get_max_elements()

    def resize_index(self, new_size: int) -> None:
        self._require().resize_index(int(new_size))
