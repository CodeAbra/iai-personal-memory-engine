"""Unit tests for the pooled read-only reader substrate (RoConnPool).

Covers: lazy pool construction, pooled (not open-per-call) borrow/release,
the CorpusCountCache.on_invalidate staleness bump (including the buffered-
flush path, which never reaches the store's invalidation funnel), the
kill-switch + fallback-on-failure paths, write-through-RO rejection, and the
nightly RECALL_INDEX_REBUILD recycle wiring.

Every test runs under both storage drivers from this one file: lilli-only
cases are skipped-not-failed under stdlib (mirroring
tests/test_col_index_persistence.py's per-function decorator pattern); the
stdlib-parity cases run unconditionally.

Every store here lives under ``tmp_path`` — never a real or prod store.
"""

from __future__ import annotations

import os
import struct
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
from iai_mcp.store import MemoryStore

_LILLI = os.environ.get("LILLI_STORAGE_DRIVER", "").lower() == "lilli"

_EMBED_DIM = 384

_lilli_only = pytest.mark.skipif(
    not _LILLI,
    reason="RoConnPool is lilli-only; skipped under stdlib "
    "(stdlib-parity cases run unconditionally instead)",
)


def _norm_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(_EMBED_DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def _embedding_blob(seed: int) -> bytes:
    vec = _norm_vec(seed)
    return struct.pack(f"<{_EMBED_DIM}f", *vec)


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path)


def _insert_record_direct(store: MemoryStore, seed: int) -> str:
    """Insert one record directly via SQL under the writer conn; return its id."""
    db = store.db
    rid = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()
    with db._conn_lock:
        db._conn.execute(
            "INSERT INTO records "
            "(id, tier, literal_surface, embedding, tombstoned_at, "
            " embedding_pending, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rid,
                "episodic",
                None,
                _embedding_blob(seed),
                None,
                0,
                now_iso,
                now_iso,
            ),
        )
        db._conn.commit()
    return rid


def _count_records_ro(store: MemoryStore) -> int:
    with store.db.ro_conn() as conn:
        row = conn.execute("SELECT COUNT(*) FROM records").fetchone()
    return int(row[0])


# ---------------------------------------------------------------------------
# Group 1 + 8: pool construction is lazy; stdlib never constructs a pool and
# ro_conn() is byte-equivalent to today's shared-conn-under-lock pattern.
# ---------------------------------------------------------------------------


@_lilli_only
def test_pool_construction_is_lazy_no_engine_open_at_init(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    pool = store.db._ro_pool
    assert pool is not None
    assert pool._filled == 0, "pool must not open any slot at HippoDB construction time"
    store.close()


def test_stdlib_driver_ro_conn_is_shared_conn_under_lock(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    store = _make_store(tmp_path)
    assert store.db._storage_driver == "stdlib"
    assert store.db._ro_pool is None, "pool must never be constructed on the stdlib driver"

    with store.db.ro_conn() as conn:
        assert conn is store.db._conn, (
            "on stdlib, ro_conn() must yield the SAME shared writer connection "
            "object as today's _conn_lock pattern"
        )
    store.close()


# ---------------------------------------------------------------------------
# Group 2: borrow/release round-trip; pooling proven via an open-call spy —
# M >> N borrows open at most N slots.
# ---------------------------------------------------------------------------


@_lilli_only
def test_pool_borrows_reuse_slots_not_open_per_call(tmp_path: Path, monkeypatch) -> None:
    store = _make_store(tmp_path)
    pool = store.db._ro_pool
    pool._size = 4  # shrink for a fast, deterministic test

    open_calls = {"n": 0}
    real_open_slot = pool._open_slot

    def _spy_open_slot():
        open_calls["n"] += 1
        return real_open_slot()

    monkeypatch.setattr(pool, "_open_slot", _spy_open_slot)

    for _ in range(50):
        with store.db.ro_conn() as conn:
            conn.execute("SELECT COUNT(*) FROM records").fetchone()

    assert open_calls["n"] <= pool._size, (
        f"expected at most {pool._size} opens across 50 borrows, "
        f"got {open_calls['n']} — pooling is not amortizing opens"
    )
    assert open_calls["n"] >= 1
    store.close()


@_lilli_only
def test_concurrent_borrows_across_threads_stay_bounded(tmp_path: Path, monkeypatch) -> None:
    """M=50 borrows across 4 threads open at most N slots."""
    store = _make_store(tmp_path)
    pool = store.db._ro_pool
    pool._size = 4

    open_calls = {"n": 0}
    lock = threading.Lock()
    real_open_slot = pool._open_slot

    def _spy_open_slot():
        with lock:
            open_calls["n"] += 1
        return real_open_slot()

    monkeypatch.setattr(pool, "_open_slot", _spy_open_slot)

    errors: list[Exception] = []

    def _worker():
        try:
            for _ in range(12):
                with store.db.ro_conn() as conn:
                    conn.execute("SELECT COUNT(*) FROM records").fetchone()
        except Exception as exc:  # noqa: BLE001 -- captured for assertion
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"concurrent borrow errors: {errors}"
    assert open_calls["n"] <= pool._size, (
        f"expected at most {pool._size} opens across concurrent borrows, "
        f"got {open_calls['n']}"
    )
    store.close()


# ---------------------------------------------------------------------------
# Group 3: N threads borrow + run recall-shaped SELECTs while a writer thread
# inserts under _conn_lock — zero errors, readers never block on the writer.
# ---------------------------------------------------------------------------


@_lilli_only
def test_readers_never_block_on_concurrent_writer(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    for i in range(20):
        _insert_record_direct(store, i)

    stop = threading.Event()
    write_errors: list[Exception] = []
    read_errors: list[Exception] = []
    reads_completed = {"n": 0}
    reads_lock = threading.Lock()

    def _writer():
        i = 100
        while not stop.is_set():
            try:
                _insert_record_direct(store, i)
                i += 1
            except Exception as exc:  # noqa: BLE001
                write_errors.append(exc)
                break

    def _reader():
        try:
            for _ in range(30):
                with store.db.ro_conn() as conn:
                    conn.execute(
                        "SELECT id FROM records WHERE tombstoned_at IS NULL LIMIT 5"
                    ).fetchall()
                with reads_lock:
                    reads_completed["n"] += 1
        except Exception as exc:  # noqa: BLE001
            read_errors.append(exc)

    writer_thread = threading.Thread(target=_writer)
    reader_threads = [threading.Thread(target=_reader) for _ in range(4)]

    writer_thread.start()
    for t in reader_threads:
        t.start()
    for t in reader_threads:
        t.join(timeout=30)
    stop.set()
    writer_thread.join(timeout=10)

    assert not read_errors, f"reader errors under concurrent writer: {read_errors}"
    assert not write_errors, f"writer errors: {write_errors}"
    assert reads_completed["n"] == 4 * 30
    store.close()


# ---------------------------------------------------------------------------
# Group 4 + 5: staleness bump proof — via the funnel-reached seams AND via
# the buffered flush (the seam the funnel never sees).
# ---------------------------------------------------------------------------


@_lilli_only
def test_staleness_via_delete_next_ro_read_excludes_row(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    rid = _insert_record_direct(store, 0)

    with store.db.ro_conn() as conn:
        row = conn.execute(
            "SELECT id FROM records WHERE id = ? AND tombstoned_at IS NULL", (rid,)
        ).fetchone()
    assert row is not None

    with store.db._conn_lock:
        store.db._conn.execute(
            "UPDATE records SET tombstoned_at = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), rid),
        )
        store.db._conn.commit()
    store._corpus_count_cache.invalidate("active")

    with store.db.ro_conn() as conn:
        row = conn.execute(
            "SELECT id FROM records WHERE id = ? AND tombstoned_at IS NULL", (rid,)
        ).fetchone()
    assert row is None, (
        "after a tombstone + cache invalidate, the next RO-served read must "
        "exclude the tombstoned row (stale pre-tombstone snapshot must not "
        "resurface it)"
    )
    store.close()


@_lilli_only
def test_staleness_via_insert_pending_row_db_fired(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    baseline = _count_records_ro(store)

    store.db.insert_pending_row(
        record_id=str(uuid.uuid4()),
        literal_surface="pending row content",
        tags_json="[]",
        provenance_json="{}",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        tier="episodic",
    )

    after = _count_records_ro(store)
    assert after == baseline + 1, (
        "insert_pending_row's direct mark_ro_pool_stale() fire site must make "
        "the new row visible to the very next RO-served read"
    )
    store.close()


@_lilli_only
def test_buffered_flush_bumps_generation_with_zero_funnel_calls(
    tmp_path: Path, monkeypatch
) -> None:
    """The no-false-negative freshness contract.

    flush_record_buffer invalidates the corpus-count cache DIRECTLY
    (_buffers.py:59-61) — it never calls store._invalidate_corpus_count (the
    funnel). This test proves the RO pool generation still bumps via the
    cache's on_invalidate hook, with the funnel spy asserting ZERO calls.
    """
    from iai_mcp.store import _buffers

    store = _make_store(tmp_path)
    baseline = _count_records_ro(store)

    funnel_calls = {"n": 0}
    real_funnel = store._invalidate_corpus_count

    def _spy_funnel(*keys):
        funnel_calls["n"] += 1
        return real_funnel(*keys)

    monkeypatch.setattr(store, "_invalidate_corpus_count", _spy_funnel)

    rid = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()
    _buffers._record_buffer[id(store)] = [
        {
            "id": rid,
            "tier": "episodic",
            "literal_surface": None,
            "embedding": _norm_vec(1),
            "tombstoned_at": None,
            "embedding_pending": 0,
            "created_at": now_iso,
            "updated_at": now_iso,
        }
    ]

    flushed = _buffers.flush_record_buffer(store)
    assert flushed == 1

    assert funnel_calls["n"] == 0, (
        "flush_record_buffer must NEVER call the store's invalidation funnel — "
        "if this fires, the test is no longer proving the buffered-flush seam"
    )

    after = _count_records_ro(store)
    assert after == baseline + 1, (
        "a buffered-flush commit must bump the RO pool generation via "
        "CorpusCountCache.on_invalidate (fired by the direct cache.invalidate "
        "call in _buffers.py) even though the funnel was never touched — this "
        "is the no-false-negative contract: a record landed via buffered "
        "flush must never be invisible to a RO-served read"
    )
    store.close()


@_lilli_only
def test_buffered_edge_flush_bumps_generation_with_zero_funnel_calls(
    tmp_path: Path, monkeypatch
) -> None:
    from iai_mcp.store import _buffers

    store = _make_store(tmp_path)

    funnel_calls = {"n": 0}
    real_funnel = store._invalidate_corpus_count

    def _spy_funnel(*keys):
        funnel_calls["n"] += 1
        return real_funnel(*keys)

    monkeypatch.setattr(store, "_invalidate_corpus_count", _spy_funnel)

    src, dst = str(uuid.uuid4()), str(uuid.uuid4())
    _buffers._edge_buffer[id(store)] = [
        {"src": src, "dst": dst, "edge_type": "hebbian", "weight": 0.7}
    ]

    gen_before = store.db._ro_pool._current_generation()
    flushed = _buffers.flush_edge_buffer(store)
    assert flushed == 1
    assert funnel_calls["n"] == 0

    gen_after = store.db._ro_pool._current_generation()
    assert gen_after > gen_before, (
        "flush_edge_buffer's direct cache.invalidate('edges') call must bump "
        "the RO pool generation via the on_invalidate hook, with zero funnel "
        "involvement"
    )
    store.close()


# ---------------------------------------------------------------------------
# Group 6: write-through-RO raises at the engine layer (defense in depth).
# ---------------------------------------------------------------------------


@_lilli_only
def test_write_through_ro_conn_raises(tmp_path: Path) -> None:
    from iai_mcp import errors

    store = _make_store(tmp_path)
    with pytest.raises(errors.Error):
        with store.db.ro_conn() as conn:
            conn.execute(
                "INSERT INTO records (id, tier) VALUES (?, ?)",
                (str(uuid.uuid4()), "episodic"),
            )
    store.close()


# ---------------------------------------------------------------------------
# Group 7: kill-switch — both drivers served via the writer path, no raise.
# ---------------------------------------------------------------------------


@_lilli_only
def test_kill_switch_serves_writer_path_no_raise(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("IAI_MCP_RO_POOL_OFF", "1")
    store = _make_store(tmp_path)
    rid = _insert_record_direct(store, 0)

    with store.db.ro_conn() as conn:
        assert conn is store.db._conn, (
            "kill-switch must serve ro_conn() via the writer conn under "
            "_conn_lock, identical to the stdlib fallback shape"
        )
        row = conn.execute("SELECT id FROM records WHERE id = ?", (rid,)).fetchone()
    assert row is not None
    store.close()


@_lilli_only
def test_injected_borrow_failure_falls_back_to_writer_path(tmp_path: Path, monkeypatch) -> None:
    store = _make_store(tmp_path)
    pool = store.db._ro_pool

    def _boom():
        raise RuntimeError("simulated pool borrow failure")

    monkeypatch.setattr(pool, "borrow", _boom)

    with store.db.ro_conn() as conn:
        assert conn is store.db._conn, (
            "a borrow() failure must fall back to the writer path without "
            "raising into the caller"
        )
        conn.execute("SELECT COUNT(*) FROM records").fetchone()
    store.close()


def test_no_ro_pool_attribute_falls_back_gracefully(tmp_path: Path, monkeypatch) -> None:
    """A db with no _ro_pool attribute at all (e.g. an older/mocked db) must
    still serve ro_conn() via the writer path without raising."""
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    store = _make_store(tmp_path)
    delattr(store.db, "_ro_pool") if hasattr(store.db, "_ro_pool") else None

    with store.db.ro_conn() as conn:
        assert conn is store.db._conn
    store.close()


# ---------------------------------------------------------------------------
# Group 9: recycle() reopens all idle slots, never raises on injected failure.
# ---------------------------------------------------------------------------


@_lilli_only
def test_recycle_reopens_idle_slots(tmp_path: Path, monkeypatch) -> None:
    store = _make_store(tmp_path)
    pool = store.db._ro_pool
    pool._size = 3

    # Warm the pool to full via sequential borrows (returns each slot to idle).
    for _ in range(pool._size):
        with pool.borrow():
            pass
    assert pool._filled == pool._size

    open_calls = {"n": 0}
    real_open_slot = pool._open_slot

    def _spy_open_slot():
        open_calls["n"] += 1
        return real_open_slot()

    monkeypatch.setattr(pool, "_open_slot", _spy_open_slot)

    pool.recycle()

    assert open_calls["n"] == pool._size, (
        f"recycle() must reopen every idle slot; expected {pool._size} opens, "
        f"got {open_calls['n']}"
    )
    store.close()


@_lilli_only
def test_recycle_never_raises_on_injected_open_failure(tmp_path: Path, monkeypatch) -> None:
    store = _make_store(tmp_path)
    pool = store.db._ro_pool
    pool._size = 2
    for _ in range(pool._size):
        with pool.borrow():
            pass

    def _boom():
        raise RuntimeError("simulated reopen failure")

    monkeypatch.setattr(pool, "_open_slot", _boom)

    # Must not raise.
    pool.recycle()
    store.close()


# ---------------------------------------------------------------------------
# Group 10: sleep-step wiring — RECALL_INDEX_REBUILD calls recycle(); an
# injected recycle failure leaves the step result intact.
# ---------------------------------------------------------------------------


def _seed_records(store: MemoryStore, n: int) -> None:
    for i in range(n):
        _insert_record_direct(store, i)


@_lilli_only
def test_sleep_step_calls_pool_recycle(tmp_path: Path, monkeypatch) -> None:
    store = _make_store(tmp_path)
    _seed_records(store, 5)

    recycle_calls = {"n": 0}
    real_recycle = store.db._ro_pool.recycle

    def _spy_recycle():
        recycle_calls["n"] += 1
        return real_recycle()

    monkeypatch.setattr(store.db._ro_pool, "recycle", _spy_recycle)

    pipeline = SleepPipeline(store)
    done, payload = pipeline._step_recall_index_rebuild(None)

    assert done is True
    assert recycle_calls["n"] == 1, (
        f"RECALL_INDEX_REBUILD must call pool.recycle() exactly once per "
        f"cycle; got {recycle_calls['n']} calls"
    )
    assert "ro_pool_recycle_elapsed_sec" in payload
    store.close()


@_lilli_only
def test_sleep_step_recycle_failure_never_fails_step(tmp_path: Path, monkeypatch) -> None:
    store = _make_store(tmp_path)
    _seed_records(store, 5)

    def _boom():
        raise RuntimeError("simulated recycle failure")

    monkeypatch.setattr(store.db._ro_pool, "recycle", _boom)

    pipeline = SleepPipeline(store)
    done, payload = pipeline._step_recall_index_rebuild(None)

    assert done is True, "a recycle failure must never fail the step"
    assert payload.get("ro_pool_recycle_error"), (
        f"the recycle failure must be reported via ro_pool_recycle_error; got {payload}"
    )
    assert payload.get("rebuilt") is not False or "error" not in payload, (
        f"a recycle failure must NOT discard the graph-cache/persist-index "
        f"outcome already computed earlier in the same step body; got {payload}"
    )
    store.close()


def test_sleep_step_stdlib_no_recycle_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    store = _make_store(tmp_path)
    assert store.db._storage_driver == "stdlib"
    _seed_records(store, 5)

    pipeline = SleepPipeline(store)
    done, payload = pipeline._step_recall_index_rebuild(None)

    assert done is True
    assert "ro_pool_recycle_elapsed_sec" not in payload
    assert "ro_pool_recycle_error" not in payload
    store.close()
