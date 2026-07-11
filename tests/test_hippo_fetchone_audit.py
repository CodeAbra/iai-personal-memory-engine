from __future__ import annotations

import threading
import uuid
from pathlib import Path

import numpy as np

from iai_mcp.hippo import HippoDB


def _insert_events(db: HippoDB, n: int = 5) -> None:
    import json as _json
    for i in range(n):
        event_id = str(uuid.uuid4())
        ts = "2026-01-01T00:00:00+00:00"
        db._conn.execute("BEGIN")
        db._conn.execute(
            "INSERT INTO events (id, kind, severity, domain, ts, data_json, session_id, source_ids_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (event_id, "test_event", "info", "test", ts, _json.dumps({"i": i}), "sess", None),
        )
        db._conn.execute("COMMIT")


def _seed_records_direct(db: HippoDB, n: int = 10) -> None:
    embed_bytes = np.zeros(db._embed_dim, dtype=np.float32).tobytes()
    for i in range(n):
        rid = str(uuid.uuid4())
        ts = "2026-01-01T00:00:00+00:00"
        db._conn.execute(
            "INSERT INTO records (id, tier, embedding, created_at, hv_tier, structure_hv_payload)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (rid, "episodic", embed_bytes, ts, "bsc", b""),
        )


class TestConnLockType:

    def test_conn_lock_is_rlock(self, tmp_path: Path) -> None:
        db = HippoDB(tmp_path)
        lock = db._conn_lock
        assert lock.acquire(blocking=True), "Could not acquire _conn_lock first time"
        reentrant_ok = lock.acquire(blocking=False)
        lock.release()
        if reentrant_ok:
            lock.release()
        assert reentrant_ok, (
            "_conn_lock is not re-entrant — expected threading.RLock, "
            "got threading.Lock (same thread blocked on second acquire)"
        )
        db.close()

    def test_conn_lock_type_annotation(self, tmp_path: Path) -> None:
        db = HippoDB(tmp_path)
        lock = db._conn_lock
        lock.acquire()
        second_acquired = lock.acquire(blocking=False)
        lock.release()
        if second_acquired:
            lock.release()
        assert second_acquired, "_conn_lock must be RLock (re-entrant)"
        db.close()


class TestLockOrdering:

    def test_delete_and_count_rows_no_deadlock(self, tmp_path: Path) -> None:
        db = HippoDB(tmp_path)
        _seed_records_direct(db, 5)
        records_tbl = db.open_table("records")

        errors: list[str] = []
        stop = threading.Event()

        def count_loop() -> None:
            for _ in range(100):
                if stop.is_set():
                    break
                try:
                    records_tbl.count_rows()
                except Exception as exc:
                    errors.append(f"count_rows: {exc}")

        def writer_loop() -> None:
            while not stop.is_set():
                try:
                    _insert_events(db, 1)
                except Exception:
                    pass

        readers = [threading.Thread(target=count_loop) for _ in range(2)]
        writers = [threading.Thread(target=writer_loop) for _ in range(2)]
        for t in readers + writers:
            t.start()

        for t in readers:
            t.join(timeout=10)
        stop.set()
        for t in writers:
            t.join(timeout=5)

        db.close()
        assert not errors, f"Errors during lock-ordering test: {errors[:3]}"


class TestTableNamesConcurrentAccess:

    def test_table_names_stable_under_concurrent_writes(self, tmp_path: Path) -> None:
        db = HippoDB(tmp_path)
        _seed_records_direct(db, 5)

        wrong_results: list[str] = []
        errors: list[str] = []
        stop = threading.Event()

        EXPECTED_TABLES = {"_hippo_meta", "records", "edges", "events"}

        def reader_thread() -> None:
            for _ in range(200):
                if stop.is_set():
                    break
                try:
                    names = db.table_names()
                    missing = EXPECTED_TABLES - set(names)
                    if missing:
                        wrong_results.append(f"Missing tables: {missing}")
                except Exception as exc:
                    errors.append(str(exc))

        def writer_thread() -> None:
            while not stop.is_set():
                try:
                    _insert_events(db, 1)
                except Exception:
                    pass

        readers = [threading.Thread(target=reader_thread) for _ in range(3)]
        writers = [threading.Thread(target=writer_thread) for _ in range(3)]
        for t in readers + writers:
            t.start()

        for t in readers:
            t.join(timeout=15)
        stop.set()
        for t in writers:
            t.join(timeout=5)

        db.close()

        assert not errors, (
            f"table_names() raised exception under concurrent writes: {errors[:3]}"
        )
        assert not wrong_results, (
            f"table_names() returned truncated result under concurrent writes: "
            f"{wrong_results[:3]}"
        )


class TestSchemaConcurrentAccess:

    def test_schema_stable_under_concurrent_writes(self, tmp_path: Path) -> None:
        db = HippoDB(tmp_path)
        _seed_records_direct(db, 5)
        records_tbl = db.open_table("records")

        wrong_results: list[str] = []
        errors: list[str] = []
        stop = threading.Event()

        REQUIRED_COLS = {"id", "tier", "embedding", "created_at"}

        def reader_thread() -> None:
            for _ in range(200):
                if stop.is_set():
                    break
                try:
                    schema = records_tbl.schema
                    col_names = {f.name for f in schema}
                    missing = REQUIRED_COLS - col_names
                    if missing:
                        wrong_results.append(f"Missing columns in schema: {missing}")
                except Exception as exc:
                    errors.append(str(exc))

        def writer_thread() -> None:
            while not stop.is_set():
                try:
                    _insert_events(db, 1)
                except Exception:
                    pass

        readers = [threading.Thread(target=reader_thread) for _ in range(3)]
        writers = [threading.Thread(target=writer_thread) for _ in range(3)]
        for t in readers + writers:
            t.start()

        for t in readers:
            t.join(timeout=15)
        stop.set()
        for t in writers:
            t.join(timeout=5)

        db.close()

        assert not errors, (
            f"schema() raised exception under concurrent writes: {errors[:3]}"
        )
        assert not wrong_results, (
            f"schema() returned truncated result under concurrent writes: "
            f"{wrong_results[:3]}"
        )


class TestDeleteLockOrdering:

    def test_records_delete_no_error_single_thread(self, tmp_path: Path) -> None:
        db = HippoDB(tmp_path)
        records_tbl = db.open_table("records")
        _seed_records_direct(db, 10)

        all_ids = [row["id"] for row in db._conn.execute(
            "SELECT id FROM records WHERE tombstoned_at IS NULL"
        ).fetchall()]

        for rid in all_ids[:5]:
            records_tbl.delete(f"id = '{rid}'")

        remaining = records_tbl.count_rows()
        assert remaining == 5, f"Expected 5 remaining, got {remaining}"
        db.close()

    def test_delete_hnsw_conn_lock_ordering_no_deadlock(self, tmp_path: Path) -> None:
        db = HippoDB(tmp_path)
        _seed_records_direct(db, 5)

        errors: list[str] = []
        results: list[dict] = []

        def rebuild_thread() -> None:
            try:
                with db._hnsw_lock:
                    r = db._rebuild_index_from_sqlite()
                    results.append(r)
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=rebuild_thread) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        db.close()
        assert not errors, f"_hnsw_lock -> _conn_lock ordering caused errors: {errors}"
        assert not any(t.is_alive() for t in threads), "Thread deadlocked"
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"


class TestRebuildIndexRLock:

    def test_rebuild_inside_hnsw_lock_no_deadlock(self, tmp_path: Path) -> None:
        db = HippoDB(tmp_path)
        _seed_records_direct(db, 5)

        errors: list[str] = []

        def run_rebuild() -> None:
            try:
                with db._hnsw_lock:
                    db._rebuild_index_from_sqlite()
            except Exception as exc:
                errors.append(str(exc))

        t = threading.Thread(target=run_rebuild)
        t.start()
        t.join(timeout=10)

        db.close()
        assert not errors, f"_rebuild_index_from_sqlite inside _hnsw_lock raised: {errors}"
        assert not t.is_alive(), "_rebuild_index_from_sqlite deadlocked inside _hnsw_lock"

    def test_rebuild_result_correct_count(self, tmp_path: Path) -> None:
        db = HippoDB(tmp_path)
        _seed_records_direct(db, 8)
        with db._hnsw_lock:
            result = db._rebuild_index_from_sqlite()
        assert result["rebuilt_count"] == 8, (
            f"Expected 8 records rebuilt, got {result['rebuilt_count']}"
        )
        db.close()


class TestSchemaOpsConcurrentAccess:

    def test_add_columns_no_exception_under_concurrent_writes(self, tmp_path: Path) -> None:
        db = HippoDB(tmp_path)
        events_tbl = db.open_table("events")
        _seed_records_direct(db, 5)

        errors: list[str] = []
        stop = threading.Event()

        def schema_reader() -> None:
            for _ in range(100):
                if stop.is_set():
                    break
                try:
                    _ = events_tbl.schema
                except Exception as exc:
                    errors.append(str(exc))

        def writer_thread() -> None:
            while not stop.is_set():
                try:
                    _insert_events(db, 1)
                except Exception:
                    pass

        readers = [threading.Thread(target=schema_reader) for _ in range(3)]
        writers = [threading.Thread(target=writer_thread) for _ in range(3)]
        for t in readers + writers:
            t.start()

        for t in readers:
            t.join(timeout=15)
        stop.set()
        for t in writers:
            t.join(timeout=5)

        db.close()
        assert not errors, (
            f"schema()/PRAGMA fetchall raised under concurrent writes: {errors[:3]}"
        )
