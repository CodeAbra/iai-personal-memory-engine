from __future__ import annotations

import concurrent.futures
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from iai_mcp.store import (
    EDGES_TABLE,
    RECORDS_TABLE,
    MemoryStore,
    flush_record_buffer,
)
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _make_record(seed: int, text: str = "") -> MemoryRecord:
    rng = np.random.RandomState(seed)
    vec = rng.randn(EMBED_DIM).tolist()
    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=text or f"record seed {seed}",
        aaak_index="",
        embedding=vec,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        language="en",
    )


def _wait_for_count(
    table,
    target: int,
    *,
    timeout_sec: float = 5.0,
    poll: float = 0.05,
) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if table.count_rows() >= target:
            return
        time.sleep(poll)
    actual = table.count_rows()
    raise TimeoutError(
        f"_wait_for_count timed out: expected {target}, got {actual}"
    )


def test_e2e_capture_recall_via_memory_store(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, user_id="test")
    try:
        rec = _make_record(42, "hello world")
        store.insert(rec)
        flush_record_buffer(store)

        retrieved = store.get(rec.id)
        assert retrieved is not None, "record should be retrievable after flush"
        assert retrieved.id == rec.id
        assert "hello world" in retrieved.literal_surface
        assert len(retrieved.embedding) == EMBED_DIM
    finally:
        store.close()


def test_e2e_boost_edges_round_trip(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, user_id="test")
    try:
        rec_a = _make_record(10)
        rec_b = _make_record(11)
        store.insert(rec_a)
        store.insert(rec_b)
        flush_record_buffer(store)

        store.boost_edges([(rec_a.id, rec_b.id)], delta=1.0, edge_type="hebbian")

        edges_tbl = store.db.open_table(EDGES_TABLE)
        df = edges_tbl.to_pandas()
        assert len(df) > 0, "edges table should be non-empty after boost_edges"

        canonical_src, canonical_dst = sorted([str(rec_a.id), str(rec_b.id)])
        row = df[
            (df["src"] == canonical_src)
            & (df["dst"] == canonical_dst)
            & (df["edge_type"] == "hebbian")
        ]
        assert len(row) >= 1, "written edge should be present in edges table"
        assert float(row.iloc[0]["weight"]) > 0.0
    finally:
        store.close()


def test_e2e_pattern_separation_compatible(tmp_path: Path) -> None:
    n = 10
    store = MemoryStore(tmp_path, user_id="test")
    try:
        for i in range(n):
            store.insert(_make_record(seed=1000 + i))
        flush_record_buffer(store)

        records_tbl = store.db.open_table(RECORDS_TABLE)
        assert records_tbl.count_rows() == n, (
            f"all {n} distinct records should be stored (pattern sep should INSERT, not SKIP)"
        )
    finally:
        store.close()


def test_e2e_async_write_queue_drains(tmp_path: Path) -> None:
    import asyncio

    n = 50
    store = MemoryStore(tmp_path, user_id="test")
    try:
        try:
            asyncio.run(store.enable_async_writes(coalesce_ms=10, max_batch=128))

            for i in range(n):
                store.insert(_make_record(seed=2000 + i))
        finally:
            asyncio.run(store.disable_async_writes())

        records_tbl = store.db.open_table(RECORDS_TABLE)
        count = records_tbl.count_rows()
        assert count == n, (
            f"all {n} async-queued records should be visible after drain, got {count}"
        )
    finally:
        store.close()


def test_close_drains_async_writes_without_explicit_disable(tmp_path: Path) -> None:
    """Closing a store that has async writes enabled tears down the queue, the
    background loop, and the thread, leaving no pending coalesce task — even when
    the caller never calls ``disable_async_writes`` explicitly."""
    import asyncio

    store = MemoryStore(tmp_path, user_id="test")
    asyncio.run(store.enable_async_writes(coalesce_ms=10, max_batch=128))
    for i in range(10):
        store.insert(_make_record(seed=3000 + i))

    bg_thread = store._async_thread
    assert bg_thread is not None and bg_thread.is_alive()

    # Close WITHOUT an explicit disable — the drain must run inside close().
    store.close()

    assert store._write_queue is None
    assert store._async_loop is None
    assert store._async_thread is None
    assert store._async_conn is None
    bg_thread.join(timeout=5.0)
    assert not bg_thread.is_alive(), "async background thread must be joined on close"
    leaked = [
        t for t in threading.enumerate() if t.name == "iai-mcp-async-writes"
    ]
    assert not leaked, f"async write thread leaked past close: {leaked}"


def test_close_is_noop_for_async_when_never_enabled(tmp_path: Path) -> None:
    """The async drain on close is a strict no-op when async writes were never
    enabled — closing a plain sync store does not create or touch any async
    handles."""
    store = MemoryStore(tmp_path, user_id="test")
    store.insert(_make_record(seed=4000))
    assert store._write_queue is None
    store.close()
    assert store._write_queue is None
    assert store._async_loop is None
    assert store._async_thread is None


def test_concurrent_sync_and_async_writes(tmp_path: Path) -> None:
    from iai_mcp.hippo import HippoDB

    threads_n = 4
    per_thread = 25
    total = threads_n * per_thread
    errors: list[Exception] = []
    lock = threading.Lock()

    db = HippoDB(tmp_path)
    try:
        tbl = db.open_table(RECORDS_TABLE)

        def worker(thread_idx: int) -> None:
            try:
                rows = []
                for j in range(per_thread):
                    seed = 3000 + thread_idx * per_thread + j
                    rows.append({
                        "id": str(uuid.uuid4()),
                        "tier": "episodic",
                        "literal_surface": f"concurrent worker {thread_idx} j {j}",
                        "aaak_index": "",
                        "embedding": np.random.RandomState(seed).randn(EMBED_DIM).astype(np.float32).tolist(),
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
                        "provenance_json": "[]",
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "tags_json": "[]",
                        "language": "en",
                        "s5_trust_score": 0.5,
                        "profile_modulation_gain_json": "{}",
                        "schema_version": 4,
                    })
                tbl.add(rows)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads_n) as pool:
            futs = [pool.submit(worker, i) for i in range(threads_n)]
            for f in futs:
                f.result()

        assert not errors, f"concurrent writes raised exceptions: {errors}"

        _wait_for_count(tbl, total, timeout_sec=10)
        assert tbl.count_rows() == total
    finally:
        db.close()


def test_no_lancedb_concept_leaks_through_api(tmp_path: Path) -> None:
    for key in list(sys.modules):
        if key == "lancedb" or key.startswith("lancedb."):
            del sys.modules[key]

    store = MemoryStore(tmp_path, user_id="test")
    try:
        rec = _make_record(99)
        store.insert(rec)
        flush_record_buffer(store)
        _ = store.get(rec.id)
        _ = store.query_similar(rec.embedding, k=1)
    finally:
        store.close()

    lancedb_modules = [k for k in sys.modules if k == "lancedb" or k.startswith("lancedb.")]
    assert not lancedb_modules, (
        f"lancedb leaked into sys.modules: {lancedb_modules}"
    )
