"""Resident exact-cosine matrix — the lossless similarity authority.

Holds a normalized float32 matrix of active-corpus embeddings in process
memory and answers exact top-k cosine similarity directly, without an
approximate index. This is the exact-similarity authority over a lossy fast
index: the store remains the source of truth, a cold or stale index degrades
to a rebuild, never to data loss. Never serialized to disk.

Thread safety: all mutation and the top-k read are guarded by an internal
``threading.Lock``. ``top_k`` computes the full similarity scan under the
lock (a sub-millisecond hold at tens of thousands of rows) so a concurrent
upsert can never tear a row mid-read. ``build`` installs the full matrix,
id list, and id-to-row map in ONE lock hold (the replace-all discipline) so
a concurrent reader sees either the complete old state or the complete new
state, never a half-filled intermediate.

The module imports numpy and stdlib only — no hippo, sqlite, crypto, or
hnswlib dependency.
"""

from __future__ import annotations

import threading

import numpy as np

__all__ = ["ExactCosineIndex"]


def _normalize_row(vec: np.ndarray) -> tuple[np.ndarray, bool]:
    """L2-normalize a single vector; a zero-norm (or fully non-finite)
    vector stays all-zeros (cosine similarity 0 against any cue, never
    ranks above a positive match, never raises a divide-by-zero). Returns
    ``(unit, coerced)``: ``coerced`` is True when a non-finite component
    was zeroed before normalization."""
    v = np.asarray(vec, dtype=np.float32)
    coerced = False
    if not bool(np.isfinite(v).all()):
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)  # copy=True: vec may be a read-only frombuffer view
        coerced = True
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return np.zeros_like(v, dtype=np.float32), coerced
    return (v / norm).astype(np.float32), coerced


class ExactCosineIndex:
    """Resident normalized float32 matrix over active-corpus embeddings.

    Exact-cosine top-k authority: a cache/index over the lossless store —
    the store remains the source of truth; a cold or stale index degrades
    to a rebuild, never to data loss. Never serialized to disk.

    Internal storage is a preallocated float32 matrix with amortized
    capacity-doubling growth on upsert, a parallel ``_ids`` list, and an
    ``_id_to_row`` dict for O(1) existing-id lookup. No pickle,
    ``__reduce__``, or file I/O methods exist — by design.
    """

    def __init__(self, embed_dim: int = 384) -> None:
        self._embed_dim = embed_dim
        self._lock = threading.Lock()
        self._m: np.ndarray | None = None
        self._count = 0
        self._ids: list[str] = []
        self._id_to_row: dict[str, int] = {}
        self._warm = False
        self._generation = 0
        self._coerced_cue_total = 0
        self._coerced_row_total = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_warm(self) -> bool:
        """True after a successful ``build()``; false after ``invalidate()``
        or before the first build."""
        with self._lock:
            return self._warm

    @property
    def generation(self) -> int:
        """Monotonic counter incremented on every ``invalidate()``.

        Callers that snapshot rows for a future ``build()`` outside the
        lock can pass the generation observed before the snapshot to
        ``build(rows, expected_generation=...)`` so a destructive write
        that lands (and invalidates) during the snapshot window is
        detected and the stale install is refused.
        """
        with self._lock:
            return self._generation

    @property
    def coerced_cue_total(self) -> int:
        """Count of query cues coerced from non-finite to zero. Lock-free
        read; incremented under ``self._lock``."""
        return self._coerced_cue_total

    @property
    def coerced_row_total(self) -> int:
        """Count of stored rows (build or upsert) coerced from non-finite to
        zero. Lock-free read; incremented under ``self._lock``."""
        return self._coerced_row_total

    def build(
        self,
        rows: list[tuple[str, bytes]],
        expected_generation: int | None = None,
    ) -> bool:
        """Atomically install the full row set: (record_id, raw float32 LE blob).

        A row whose blob is falsy or whose byte length does not match
        ``embed_dim * 4`` is skipped — excluded from the matrix, never fatal —
        so one malformed row never aborts the whole build. Normalizes each
        surviving row once outside the lock, then installs the matrix, id
        list, and id map in ONE lock hold (the replace-all discipline — a
        concurrent reader sees old-complete or new-complete, never
        half-filled). Sets warm.

        When ``expected_generation`` is given, the install is refused (index
        stays in its current state, returns False) if ``invalidate()`` bumped
        the generation counter since the caller's row snapshot was taken —
        closing the build/invalidate TOCTOU window where a stale pre-delete
        snapshot could otherwise be committed as warm. Returns True when the
        install proceeded.
        """
        expected_len = self._embed_dim * 4
        capacity = max(len(rows), 16)
        matrix = np.zeros((capacity, self._embed_dim), dtype=np.float32)
        ids: list[str] = []
        id_to_row: dict[str, int] = {}

        row_idx = 0
        coerced_rows = 0
        for record_id, blob in rows:
            if not blob or len(blob) != expected_len:
                continue  # malformed row: excluded from the matrix, never fatal
            vec = np.frombuffer(blob, dtype=np.float32)
            # Per-row normalization MUST stay bit-identical to `upsert`'s
            # `_normalize_row` (the determinism contract: an upserted row and
            # a rebuilt row produce the same score) — a vectorized batch norm
            # accumulates differently in the low bits and breaks it.
            unit, coerced = _normalize_row(vec)
            matrix[row_idx] = unit
            if coerced:
                coerced_rows += 1
            ids.append(record_id)
            id_to_row[record_id] = row_idx
            row_idx += 1

        with self._lock:
            if (
                expected_generation is not None
                and self._generation != expected_generation
            ):
                return False
            self._m = matrix
            self._ids = ids
            self._id_to_row = id_to_row
            self._count = row_idx
            self._warm = True
            self._coerced_row_total += coerced_rows
            return True

    def top_k(
        self, vec: list[float] | np.ndarray, k: int
    ) -> list[tuple[str, float]] | None:
        """Exact top-k by cosine, descending. Returns None when cold (caller
        must rebuild via ``build()``). Empty list when warm but empty.
        Computed fully under the internal lock (sub-ms hold).
        """
        cue = np.asarray(vec, dtype=np.float32)
        cue_unit, cue_coerced = _normalize_row(cue)

        with self._lock:
            if cue_coerced:
                self._coerced_cue_total += 1
            if not self._warm or self._m is None:
                return None
            count = self._count
            if count == 0:
                return []

            active = self._m[:count]
            sims = active @ cue_unit

            k_eff = min(k, count)
            # A full stable argsort, not an argpartition fast path: at these
            # corpus sizes the matmul above dominates cost, and argpartition
            # gives no guarantee about WHICH boundary-tied elements land
            # inside the partition when the score at rank k_eff-1 ties with
            # scores at rank >= k_eff (duplicate-capture and zero-norm rows
            # both produce exact ties here). A full stable argsort is the
            # only path that deterministically keeps the lowest original
            # index among a tied group, matching the authority's determinism
            # contract exactly, including at the k-boundary.
            order = np.argsort(-sims, kind="stable")[:k_eff]

            ids = self._ids
            return [(ids[i], float(sims[i])) for i in order]

    def upsert(self, record_id: str, vec: list[float] | np.ndarray) -> None:
        """Insert-or-replace one row in a WARM index (normalize; overwrite in
        place when the id exists, append otherwise, amortized
        capacity-doubling growth). No-op when cold.
        """
        cue = np.asarray(vec, dtype=np.float32)
        unit, coerced = _normalize_row(cue)

        with self._lock:
            if not self._warm or self._m is None:
                return
            if coerced:
                self._coerced_row_total += 1

            existing_row = self._id_to_row.get(record_id)
            if existing_row is not None:
                self._m[existing_row] = unit
                return

            if self._count >= self._m.shape[0]:
                new_capacity = max(self._m.shape[0] * 2, 16)
                grown = np.zeros((new_capacity, self._embed_dim), dtype=np.float32)
                grown[: self._count] = self._m[: self._count]
                self._m = grown

            row_idx = self._count
            self._m[row_idx] = unit
            self._ids.append(record_id)
            self._id_to_row[record_id] = row_idx
            self._count += 1

    def invalidate(self) -> None:
        """Mark cold; drops warm state. Next top_k returns None.

        Bumps the generation counter so an in-flight ``build()`` snapshot
        taken before this call refuses to install its now-stale rows (see
        ``build``'s ``expected_generation`` parameter).
        """
        with self._lock:
            self._warm = False
            self._m = None
            self._ids = []
            self._id_to_row = {}
            self._count = 0
            self._generation += 1

    def __len__(self) -> int:
        with self._lock:
            return self._count
