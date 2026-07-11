"""Bounded-hold discipline for ``HippoQuery.to_batches``.

A full-corpus ``to_batches`` scan must never hold the shared connection lock
(``_conn_lock``) across the consumer's per-batch work. Holding it for the whole
iteration starves every concurrent writer / index rebuild for the entire scan
duration — the lock-held-across-consumer denial-of-service.

These guards pin BOTH halves of the invariant:

* the scan does not block on a writer that holds ``_conn_lock`` (it streams from
  a dedicated read-only snapshot connection that holds no shared lock);
* a second thread CAN acquire ``_conn_lock`` while the scan is suspended
  mid-stream — i.e. the lock is genuinely available between yielded batches.

The earlier repro asserted the inverse (the scan blocks on the held lock), which
is structurally unable to detect the bounded-hold regression and silently passed
against two refactors that broke it.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from uuid import uuid4

import numpy as np

from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord

_N_SEED = 200
_HOLD_SEC = 1.5
# The scan must finish well before the held-lock window elapses: if it streams
# from a snapshot connection it never waits on the shared lock at all.
_NONBLOCK_THRESHOLD = _HOLD_SEC * 0.5


def _make_record(vec, community_id, idx: int) -> MemoryRecord:
    import datetime

    now = datetime.datetime.now(datetime.timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=f"record {idx} content",
        aaak_index="",
        embedding=vec.tolist(),
        community_id=community_id,
        centrality=float(idx % 17),
        detail_level=3,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=["kind:note"],
        language="en",
        profile_modulation_gain={},
    )


def _seed_store(store: MemoryStore, n: int) -> None:
    dim = store._embed_dim
    cid = uuid4()
    rng = np.random.default_rng(11)
    for i in range(n):
        vec = rng.standard_normal(dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        store.insert(_make_record(vec, cid, i))


def _assert_hermetic(store: MemoryStore, tmp_path: Path) -> None:
    root = Path(store.root).resolve()
    assert str(root).startswith(str(tmp_path.resolve())), (
        f"store root {root} escaped tmp_path {tmp_path}"
    )
    real_home_store = (Path.home() / ".iai-mcp").resolve()
    assert real_home_store not in root.parents and root != real_home_store, (
        f"store root {root} resolved under the real ~/.iai-mcp"
    )


def _hold_conn_lock(store: MemoryStore, hold_sec: float,
                    started: threading.Event, done: threading.Event) -> None:
    with store.db._conn_lock:
        started.set()
        time.sleep(hold_sec)
    done.set()


def test_to_batches_does_not_block_on_held_conn_lock(tmp_path):
    """A held ``_conn_lock`` must NOT stall a ``to_batches`` scan.

    The scan streams from a read-only snapshot connection that holds no shared
    lock, so it completes even while another thread holds ``_conn_lock`` for a
    full window. Total rows + ordering must be identical to an unobstructed
    drain (no row skipped/duplicated across the snapshot boundary).
    """
    store_root = tmp_path / ".iai-mcp"
    store_root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_root)
    _assert_hermetic(store, tmp_path)
    _seed_store(store, _N_SEED)

    query = store.db.open_table("records").search()
    assert query._db is not None, "to_batches query lost its HippoDB reference"

    started = threading.Event()
    done = threading.Event()
    worker = threading.Thread(
        target=_hold_conn_lock,
        args=(store, _HOLD_SEC, started, done),
        daemon=True,
    )

    reader_elapsed = 0.0
    rows = 0
    try:
        worker.start()
        assert started.wait(timeout=10.0), "worker never acquired _conn_lock"
        t0 = time.monotonic()
        for batch in query.to_batches(batch_size=1):
            rows += batch.num_rows
        reader_elapsed = time.monotonic() - t0
        assert rows == _N_SEED, f"to_batches read {rows} rows, expected {_N_SEED}"
    finally:
        done.wait(timeout=_HOLD_SEC + 10.0)
        worker.join(timeout=5.0)
        store.close()

    assert reader_elapsed < _NONBLOCK_THRESHOLD, (
        "to_batches stalled on the held _conn_lock "
        f"(elapsed={reader_elapsed:.3f}s, hold={_HOLD_SEC}s) — the scan is "
        "holding the shared connection lock across the consumer iteration "
        "instead of streaming from an independent snapshot connection. This is "
        "the lock-held-across-consumer regression."
    )


def test_conn_lock_acquirable_while_to_batches_suspended_mid_stream(tmp_path):
    """A second thread MUST be able to acquire ``_conn_lock`` mid-scan.

    Consume one batch, then — with the generator suspended between yields —
    assert ``_conn_lock`` is acquirable (non-blocking) from this thread. A scan
    that held the lock across the consumer would make this acquire fail. Then
    finish the drain and assert the full row set, in order, with no loss or
    duplication.
    """
    store_root = tmp_path / ".iai-mcp"
    store_root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_root)
    _assert_hermetic(store, tmp_path)
    _seed_store(store, _N_SEED)

    try:
        # Reference order from a fully-drained scan (no suspension).
        ref_ids = [
            r["id"]
            for r in store.iter_record_columns(["id"], batch_size=1024)
        ]
        assert len(ref_ids) == _N_SEED

        query = store.db.open_table("records").search().select(["id"])
        gen = query.to_batches(batch_size=16)

        first = next(gen)
        assert first.num_rows == 16

        # The lock must be acquirable from another thread while the generator is
        # suspended between batches — i.e. it is NOT held across consumer time.
        acquired = {"ok": False}
        blocker = threading.Event()

        def _try_acquire() -> None:
            got = store.db._conn_lock.acquire(blocking=True, timeout=2.0)
            acquired["ok"] = got
            if got:
                store.db._conn_lock.release()
            blocker.set()

        t = threading.Thread(target=_try_acquire, daemon=True)
        t.start()
        assert blocker.wait(timeout=5.0), (
            "second thread never returned from _conn_lock.acquire — to_batches "
            "is holding the lock across the suspended consumer (DoS regression)"
        )
        t.join(timeout=5.0)
        assert acquired["ok"], (
            "a second thread could NOT acquire _conn_lock while to_batches was "
            "suspended mid-stream — the scan holds the lock across the consumer "
            "iteration (lock-held-across-consumer DoS)"
        )

        # Finish the drain; ordering + completeness must match the reference.
        out_ids = [r["id"] for r in first.to_pylist()]
        for batch in gen:
            out_ids.extend(r["id"] for r in batch.to_pylist())

        assert len(out_ids) == _N_SEED, (
            f"to_batches yielded {len(out_ids)} rows, expected {_N_SEED} "
            "(row skipped or duplicated across the snapshot boundary)"
        )
        assert len(set(out_ids)) == _N_SEED, "duplicate ids across batches"
        assert out_ids == ref_ids, (
            "to_batches row ordering diverged from the reference scan"
        )
    finally:
        store.close()
