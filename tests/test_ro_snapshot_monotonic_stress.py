"""Read-only snapshot monotonicity under sustained concurrent writes.

A pooled read-only reader samples ``count_rows()`` while a single writer only
ever ADDS rows. Any sample below the committed floor, or below an earlier
sample, means a reader was handed a torn or regressed snapshot — the class
where a raced WAL-overlay capture silently truncated a reader's view to an
older prefix and served committed rows as absent.

Slow lane: sustained-load stress, opt-in via --runslow.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp.store import MemoryStore

from test_hippo_conn_lock_guard import _make_record, _seed_store

_N_SEED = 150
_SAMPLES = 400


@pytest.mark.slow
def test_count_never_dips_below_floor_under_sustained_writes(tmp_path: Path) -> None:
    store_root = tmp_path / ".iai-mcp"
    store_root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_root)

    community_id = uuid4()
    _seed_store(store, _N_SEED, community_id)

    dim = store._embed_dim
    rng = np.random.default_rng(123)
    stop = threading.Event()
    writer_errors: list[BaseException] = []

    def _writer() -> None:
        idx = _N_SEED
        try:
            while not stop.is_set():
                vec = rng.standard_normal(dim).astype(np.float32)
                vec /= np.linalg.norm(vec) + 1e-9
                store.insert(_make_record(vec, community_id, idx))
                idx += 1
                time.sleep(0.001)
        except BaseException as exc:  # noqa: BLE001
            writer_errors.append(exc)
            stop.set()

    samples: list[int] = []
    writer = threading.Thread(target=_writer, daemon=True)
    try:
        writer.start()
        for _ in range(_SAMPLES):
            tbl = store.db.open_table("records")
            samples.append(tbl.count_rows())
            time.sleep(0.002)
    finally:
        stop.set()
        writer.join(timeout=5.0)
        store.close()

    assert not writer_errors, f"writer thread raised: {writer_errors!r}"
    assert samples and min(samples) >= _N_SEED, (
        f"count_rows dipped below the committed floor: min={min(samples)} "
        f"samples={samples!r}"
    )
    assert samples == sorted(samples), (
        f"count_rows went backwards under an add-only writer: {samples}"
    )
