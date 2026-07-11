from __future__ import annotations

import gc
import os
import sqlite3
import subprocess
import tracemalloc
import uuid
from datetime import datetime, timezone
from pathlib import Path

from tests._store_raw import open_store_raw

import numpy as np

from iai_mcp.hippo import HippoDB
from iai_mcp.types import EMBED_DIM


def _make_row(seed: int) -> dict:
    rng = np.random.RandomState(seed)
    vec = rng.randn(EMBED_DIM).astype(np.float32).tolist()
    return {
        "id": str(uuid.uuid4()),
        "tier": "episodic",
        "literal_surface": f"footprint test seed {seed}",
        "aaak_index": "",
        "embedding": vec,
        "structure_hv": b"",
        "community_id": None,
        "centrality": 0.0,
        "detail_level": 1,
        "pinned": False,
        "stability": 0.0,
        "difficulty": 0.0,
        "last_reviewed": None,
        "never_decay": False,
        "never_merge": False,
        "tombstoned_at": None,
        "schema_bypass": None,
        "labile_until": None,
        "provenance_json": "[]",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "tags_json": "[]",
        "language": "en",
        "s5_trust_score": 0.5,
        "profile_modulation_gain_json": "{}",
        "schema_version": 4,
        "wing": None,
        "room": None,
        "drawer": None,
        "valence": None,
    }


def test_hippo_dir_size_bounded_per_record(tmp_path: Path) -> None:
    n = 20
    db = HippoDB(tmp_path)
    try:
        tbl = db.open_table("records")
        rows = [_make_row(seed=8000 + i) for i in range(n)]
        tbl.add(rows)
    finally:
        db.close()

    hippo_dir = tmp_path / "hippo"
    total_bytes = sum(f.stat().st_size for f in hippo_dir.rglob("*") if f.is_file())
    fixed_base_bytes = 256 * 1024
    per_record_cap = 8 * 1024
    max_expected = fixed_base_bytes + n * per_record_cap
    assert total_bytes < max_expected, (
        f"hippo/ occupies {total_bytes} bytes for {n} records; "
        f"expected < {max_expected} bytes "
        f"(256 KB base + {n} * 8 KB/record)"
    )


def test_vector_blob_encoding_is_float32_not_float64(tmp_path: Path) -> None:
    db = HippoDB(tmp_path)
    try:
        row = _make_row(seed=9001)
        tbl = db.open_table("records")
        tbl.add([row])
        row_id = row["id"]
    finally:
        db.close()

    db_path = tmp_path / "hippo" / "brain.sqlite3"
    conn = open_store_raw(db_path)
    conn.row_factory = sqlite3.Row
    result = conn.execute(
        "SELECT embedding FROM records WHERE id = ?", (row_id,)
    ).fetchone()
    conn.close()

    assert result is not None, "inserted row should be readable from SQLite"
    blob = bytes(result["embedding"])
    expected_len = EMBED_DIM * 4
    assert len(blob) == expected_len, (
        f"embedding BLOB is {len(blob)} bytes; expected {expected_len} "
        f"(float32 = {EMBED_DIM} * 4). Got {len(blob) // EMBED_DIM} bytes/element "
        f"(float64 would be {EMBED_DIM * 8} bytes)"
    )


def _count_open_fds() -> int:
    """Return the number of open file descriptors for this process."""
    pid = os.getpid()
    proc_fd = Path(f"/proc/{pid}/fd")
    if proc_fd.is_dir():
        return len(list(proc_fd.iterdir()))
    # Darwin / BSD: no /proc — fall back to lsof.
    try:
        out = subprocess.run(
            ["lsof", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return len(out.stdout.splitlines())
    except (OSError, subprocess.SubprocessError):
        return -1  # unavailable — fd check is skipped


def test_no_leak_on_repeated_open_close(tmp_path: Path) -> None:
    """Repeated open/close must reclaim heap and file descriptors.

    The signal here is reclamation, not a process high-water mark. An earlier
    version asserted on ``resource.getrusage().ru_maxrss`` — a peak RSS mark
    that monotonically rises and never falls. First-touch C arenas and lazy
    imports ratchet that mark up once even when every cycle's memory is fully
    reclaimed, so it crosses any fixed threshold on both storage backends and
    is unrelated to a real leak. The honest signals are:

    - the tracemalloc Python-heap delta after a settling ``gc.collect()`` — a
      true per-cycle leak would scale with the cycle count (megabytes over 100
      cycles); a clean teardown leaves it flat (sub-megabyte), and
    - the open file-descriptor delta — a leaked connection / lock / mmap would
      raise it; a clean teardown leaves it at zero.
    """
    cycles = 100

    # Warm one cycle so one-time imports and arena growth settle before the
    # measured window (those are not per-cycle and must not count as leak).
    db = HippoDB(tmp_path)
    try:
        db.open_table("records").add([_make_row(seed=6999)])
    finally:
        db.close()
    gc.collect()

    tracemalloc.start()
    heap_before, _ = tracemalloc.get_traced_memory()
    fd_before = _count_open_fds()

    for i in range(cycles):
        db = HippoDB(tmp_path)
        try:
            tbl = db.open_table("records")
            tbl.add([_make_row(seed=7000 + i)])
        finally:
            db.close()

    gc.collect()
    heap_after, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    fd_after = _count_open_fds()

    heap_growth_mb = (heap_after - heap_before) / (1024 * 1024)
    assert heap_growth_mb < 2.0, (
        f"Python heap grew by {heap_growth_mb:.2f} MB over {cycles} open/close "
        f"cycles; expected < 2 MB (a real per-cycle leak would scale to many MB)"
    )

    if fd_before >= 0 and fd_after >= 0:
        fd_growth = fd_after - fd_before
        assert fd_growth <= 1, (
            f"open file descriptors grew by {fd_growth} over {cycles} cycles; "
            f"expected no growth (possible connection / lock / mmap leak)"
        )
