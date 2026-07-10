"""Hermetic allocation regression guard for optimize_hippo_storage.

The OPTIMIZE_HIPPO sleep step calls db._rebuild_index_from_sqlite() inline
in the daemon process.  On a large corpus this materialises the full embedding
matrix in daemon heap simultaneously with the SQLite VACUUM transient, pushing
RSS past the watchdog ceiling.  The fix moves the rebuild into a subprocess so
the daemon's resident set stays flat.

This test confirms:
1. optimize_hippo_storage completes without error on a synthetic store.
2. The peak heap allocation observed in the *calling process* during
   optimize_hippo_storage stays below a per-record bound, verifying that the
   heavy numpy/hnswlib allocation happened elsewhere (child process) and was
   reclaimed by the OS.

tracemalloc captures Python + numpy C allocations on CPython 3.9+.  Pre-fix
the inline fetchall + np.stack() inside _rebuild_index_from_sqlite produces
roughly EMBED_DIM*4 bytes per record (1536 B for dim=384) in the calling
process.  Post-fix the rebuild runs in a child worker; the calling process sees
only IPC overhead — well under 1 KB/record.
"""

from __future__ import annotations

import tracemalloc
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from iai_mcp.hippo import HippoDB
from iai_mcp.types import EMBED_DIM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Record count large enough to produce a measurable allocation difference.
# 2000 records * 384 dims * 4 bytes = 3 MB of float32 matrix data.
_N_RECORDS = 2000

# Post-fix: rebuild is in a child worker; parent-side tracemalloc peak during
# the optimize call should be < 1 KB/record (only IPC + event writing overhead).
# Pre-fix: inline rebuild allocates ~1536 B/record (numpy matrix) plus
# hnswlib standby population — empirically measured at ~3.4 KB/record.
# Threshold of 1 KB/record distinguishes the two cases with a wide margin.
_PEAK_ALLOC_CAP_BYTES_PER_RECORD: int = 1024  # 1 KB/record


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vec(seed: int) -> list[float]:
    rng = np.random.RandomState(seed)  # noqa: NPY002 — deterministic seed
    v = rng.randn(EMBED_DIM).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-10
    return v.tolist()


def _record_row(seed: int) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": str(uuid.UUID(int=seed + 10_000_000)),
        "tier": "episodic",
        "literal_surface": f"rss test record {seed}",
        "embedding": _make_vec(seed),
        "created_at": now,
    }


def _populate_store(tmp_path: Path, n: int) -> HippoDB:
    db = HippoDB(tmp_path)
    tbl = db.open_table("records")
    for i in range(n):
        tbl.add([_record_row(i)])
    return db


class _MinimalStore:
    """Minimal store wrapper exposing .db attribute for optimize_hippo_storage."""

    def __init__(self, db: HippoDB, root: Path) -> None:
        self.db = db
        self.root = root
        self.user_id = "test"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_optimize_hippo_completes_without_error(tmp_path: Path) -> None:
    """optimize_hippo_storage must succeed on a synthetic store."""
    from iai_mcp.maintenance import optimize_hippo_storage

    db = _populate_store(tmp_path, _N_RECORDS)
    try:
        store = _MinimalStore(db, tmp_path)
        report = optimize_hippo_storage(store)
    finally:
        db.close()

    assert isinstance(report, dict), "optimize_hippo_storage must return a dict"
    for table_name, result in report.items():
        assert "error" not in result, (
            f"optimize_hippo_storage reported error on table {table_name!r}: "
            f"{result.get('error')}"
        )


def test_optimize_hippo_rebuild_alloc_bounded_in_caller(tmp_path: Path) -> None:
    """Peak Python-heap allocation in the calling process during optimize must
    stay below the per-record cap.

    Pre-fix: FAILS — _rebuild_index_from_sqlite() runs inline; the full
      numpy embedding matrix (~N * EMBED_DIM * 4 bytes) plus hnswlib standby
      population are allocated in this process.
    Post-fix: PASSES — the rebuild runs in a child worker; parent-side peak
      during optimize is only IPC protocol overhead (< 1 KB/record).
    """
    from iai_mcp.maintenance import optimize_hippo_storage

    db = _populate_store(tmp_path, _N_RECORDS)
    try:
        store = _MinimalStore(db, tmp_path)

        # Warm-up call: settle one-time allocations (first VACUUM page cache,
        # first index load) so the second call reflects steady-state cycle cost.
        optimize_hippo_storage(store)

        # Measure allocation peak during the second (steady-state) cycle.
        tracemalloc.start()
        try:
            optimize_hippo_storage(store)
            _current, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    finally:
        db.close()

    cap_bytes = _PEAK_ALLOC_CAP_BYTES_PER_RECORD * _N_RECORDS
    assert peak_bytes <= cap_bytes, (
        f"optimize_hippo_storage allocated {peak_bytes / 1024:.1f} KB peak in "
        f"the calling process for {_N_RECORDS} records "
        f"(cap={cap_bytes / 1024:.1f} KB = {_PEAK_ALLOC_CAP_BYTES_PER_RECORD} B/record).  "
        "The hnswlib index rebuild is running inline instead of in a child worker — "
        "it allocates the full numpy embedding matrix in the daemon process, "
        "causing RSS to spike above the watchdog ceiling on large corpora."
    )
