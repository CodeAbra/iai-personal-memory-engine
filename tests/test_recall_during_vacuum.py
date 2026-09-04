"""Regression tests: recall-path reads must not block on a held _conn_lock.

Reproduces the root cause: ``HippoTable.count_rows()`` and
``HippoQuery.to_pandas()``'s non-ANN branch acquire ``HippoDB._conn_lock``
directly instead of routing through ``HippoDB.ro_conn()`` — the same RLock
``optimize_hippo_storage``'s VACUUM step holds for its full wall-clock
duration. These tests simulate that hold with a background thread and prove
the recall-path calls either complete promptly (post-fix, GREEN) or block for
the full hold duration (pre-fix, RED).

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

from iai_mcp.hippo._db import DEFAULT_STORAGE_DRIVER
from iai_mcp.store import MemoryStore

_LILLI = os.environ.get("LILLI_STORAGE_DRIVER", DEFAULT_STORAGE_DRIVER).lower() == "lilli"

_EMBED_DIM = 384

_lilli_only = pytest.mark.skipif(
    not _LILLI,
    reason="This test targets the lilli ro_conn() fix path; skipped under stdlib",
)

_HOLD_SECONDS = 4.0
_BOUND_SECONDS = 2.0


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


def _hold_conn_lock_for(store: MemoryStore, seconds: float) -> threading.Thread:
    """Spawn + start a background thread that holds ``_conn_lock`` for ``seconds``,
    simulating VACUUM's hold duration. Returns the started thread (caller joins).
    """
    ready = threading.Event()

    def _holder() -> None:
        with store.db._conn_lock:
            ready.set()
            threading.Event().wait(seconds)

    t = threading.Thread(target=_holder, daemon=True)
    t.start()
    ready.wait(timeout=5)
    return t


def _run_with_timeout(fn, timeout: float):
    """Run ``fn`` in a worker thread; return (completed_in_time, result_or_None,
    exc_or_None). Does not kill the worker thread on timeout (Python threads
    cannot be forcibly killed) -- it is left to finish in the background as a
    daemon thread, which is fine for a short-lived test process.
    """
    result: dict = {}

    def _runner() -> None:
        try:
            result["value"] = fn()
        except Exception as exc:  # noqa: BLE001 -- captured for assertion
            result["error"] = exc

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=timeout)
    completed = not t.is_alive()
    return completed, result.get("value"), result.get("error")


# ---------------------------------------------------------------------------
# Test 1: count_rows() must not block during a held _conn_lock (VACUUM sim)
# ---------------------------------------------------------------------------


@_lilli_only
def test_count_rows_does_not_block_during_vacuum_hold(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    for i in range(20):
        _insert_record_direct(store, i)

    holder = _hold_conn_lock_for(store, _HOLD_SECONDS)

    completed, value, error = _run_with_timeout(
        lambda: store.db.open_table("records").count_rows(),
        timeout=_BOUND_SECONDS,
    )

    holder.join(timeout=_HOLD_SECONDS + 2)
    store.close()

    assert error is None, f"count_rows() raised: {error!r}"
    assert completed, (
        f"count_rows() did not complete within {_BOUND_SECONDS}s while "
        f"_conn_lock was held for {_HOLD_SECONDS}s -- it is blocking on the "
        f"writer lock instead of routing through ro_conn()"
    )
    assert value == 20


# ---------------------------------------------------------------------------
# Test 2: HippoQuery.to_pandas() non-ANN branch must not block either
# ---------------------------------------------------------------------------


@_lilli_only
def test_query_to_pandas_non_ann_does_not_block_during_vacuum_hold(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    for i in range(20):
        _insert_record_direct(store, i)

    holder = _hold_conn_lock_for(store, _HOLD_SECONDS)

    completed, value, error = _run_with_timeout(
        lambda: store.db.open_table("records").search().to_pandas(),
        timeout=_BOUND_SECONDS,
    )

    holder.join(timeout=_HOLD_SECONDS + 2)
    store.close()

    assert error is None, f"to_pandas() raised: {error!r}"
    assert completed, (
        f"to_pandas() (non-ANN branch) did not complete within "
        f"{_BOUND_SECONDS}s while _conn_lock was held for {_HOLD_SECONDS}s -- "
        f"it is blocking on the writer lock instead of routing through ro_conn()"
    )
    assert value is not None and len(value) == 20


# ---------------------------------------------------------------------------
# Test 3: same bug class on the edges table (used by anti-edge diagnostics)
# ---------------------------------------------------------------------------


@_lilli_only
def test_edges_count_rows_does_not_block_during_vacuum_hold(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    for i in range(20):
        _insert_record_direct(store, i)

    holder = _hold_conn_lock_for(store, _HOLD_SECONDS)

    completed, value, error = _run_with_timeout(
        lambda: store.db.open_table("edges").count_rows(),
        timeout=_BOUND_SECONDS,
    )

    holder.join(timeout=_HOLD_SECONDS + 2)
    store.close()

    assert error is None, f"edges.count_rows() raised: {error!r}"
    assert completed, (
        f"edges table count_rows() did not complete within {_BOUND_SECONDS}s "
        f"while _conn_lock was held for {_HOLD_SECONDS}s"
    )
    assert value == 0


# ---------------------------------------------------------------------------
# Test 4: stdlib baseline -- documents the pre-existing, separate stdlib gap
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="stdlib ro_conn() fallback yields the shared writer conn under "
    "_conn_lock -- count_rows() is expected to block during a held lock on "
    "stdlib today; tracked separately from the lilli fix in this phase",
    strict=False,
)
def test_stdlib_count_rows_blocks_during_vacuum_hold_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "stdlib")
    store = _make_store(tmp_path)
    assert store.db._storage_driver == "stdlib"
    for i in range(20):
        _insert_record_direct(store, i)

    holder = _hold_conn_lock_for(store, _HOLD_SECONDS)

    completed, value, error = _run_with_timeout(
        lambda: store.db.open_table("records").count_rows(),
        timeout=_BOUND_SECONDS,
    )

    holder.join(timeout=_HOLD_SECONDS + 2)
    store.close()

    assert error is None, f"count_rows() raised: {error!r}"
    assert completed, (
        f"count_rows() did not complete within {_BOUND_SECONDS}s on stdlib "
        f"-- expected (xfail) since stdlib's ro_conn() fallback shares the "
        f"writer connection under _conn_lock"
    )
    assert value == 20


# ---------------------------------------------------------------------------
# Test 5: count_rows() currently bypasses ro_conn() entirely (unit-level)
# ---------------------------------------------------------------------------


@_lilli_only
def test_count_rows_routes_through_ro_conn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _make_store(tmp_path)
    for i in range(5):
        _insert_record_direct(store, i)

    spy_calls = {"n": 0}
    real_ro_conn = store.db.ro_conn

    def _spy_ro_conn():
        spy_calls["n"] += 1
        return real_ro_conn()

    monkeypatch.setattr(store.db, "ro_conn", _spy_ro_conn)

    count = store.db.open_table("records").count_rows()

    store.close()

    assert count == 5
    assert spy_calls["n"] >= 1, (
        "count_rows() did not call db.ro_conn() at all -- it currently "
        "bypasses the RO-pool entirely and acquires _conn_lock directly"
    )
