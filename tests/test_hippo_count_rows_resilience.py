from __future__ import annotations

import asyncio
import threading
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

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


class TestCountRowsBaseline:
    def test_count_rows_returns_zero_on_fresh_table(self, tmp_path: Path) -> None:
        db = HippoDB(tmp_path)
        tbl = db.open_table("records")
        result = tbl.count_rows()
        assert result == 0, f"Expected 0, got {result!r}"
        db.close()

    def test_count_rows_returns_correct_int_after_inserts(self, tmp_path: Path) -> None:
        db = HippoDB(tmp_path)
        tbl = db.open_table("records")
        _seed_records_direct(db, 7)
        result = tbl.count_rows()
        assert result == 7, f"Expected 7, got {result!r}"
        db.close()

    def test_count_rows_never_returns_none_single_thread(self, tmp_path: Path) -> None:
        db = HippoDB(tmp_path)
        tbl = db.open_table("records")
        result = tbl.count_rows()
        assert result is not None, "count_rows returned None — TypeError would follow at int(row[0])"
        db.close()


@pytest.fixture(params=["stdlib", "lilli"])
def storage_driver(request, monkeypatch):
    """Select the storage driver before any HippoDB in the test is constructed.

    The driver is read from the environment at connection-open time, so setting
    it here ensures the concurrency bodies exercise BOTH the stdlib sqlite3 path
    (current) and the vendored engine (post-fix serialization).
    """
    driver = request.param
    if driver == "lilli":
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    return driver


@pytest.mark.usefixtures("storage_driver")
class TestCountRowsConcurrentAccess:

    def test_count_rows_never_none_under_concurrent_writes(self, tmp_path: Path) -> None:
        db = HippoDB(tmp_path)
        records_tbl = db.open_table("records")
        _seed_records_direct(db, 10)

        none_counts: list[str] = []
        type_errors: list[str] = []
        stop = threading.Event()

        def reader_thread() -> None:
            for _ in range(500):
                if stop.is_set():
                    break
                try:
                    result = records_tbl.count_rows()
                    if result is None:
                        none_counts.append("None from count_rows")
                except TypeError as exc:
                    type_errors.append(str(exc))

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

        assert not type_errors, (
            f"count_rows raised TypeError {len(type_errors)} time(s) under concurrent writes: "
            f"{type_errors[:3]}"
        )
        assert not none_counts, (
            f"count_rows returned None {len(none_counts)} time(s) under concurrent writes — "
            "pre-fix regression: execute()+fetchone() not protected by a threading lock. "
            "Fix: add threading.Lock to HippoDB and acquire it in count_rows."
        )


@pytest.mark.usefixtures("storage_driver")
class TestCountRowsAsyncioToThread:
    def test_count_rows_stable_under_asyncio_to_thread_concurrency(
        self, tmp_path: Path
    ) -> None:
        db = HippoDB(tmp_path)
        records_tbl = db.open_table("records")
        _seed_records_direct(db, 15)

        type_errors: list[str] = []
        none_results: list[str] = []

        def flush_events_sync() -> None:
            try:
                _insert_events(db, 5)
            except Exception:
                pass

        def build_graph_sim() -> int | None:
            try:
                return records_tbl.count_rows()
            except TypeError as exc:
                type_errors.append(str(exc))
                return None

        async def run_concurrent() -> None:
            for _ in range(30):
                results = await asyncio.gather(
                    asyncio.to_thread(build_graph_sim),
                    asyncio.to_thread(flush_events_sync),
                )
                count_result = results[0]
                if count_result is None and not type_errors:
                    none_results.append("None returned from count_rows (no TypeError raised)")

        asyncio.run(run_concurrent())
        db.close()

        assert not type_errors, (
            f"count_rows raised TypeError under asyncio.to_thread: {type_errors[:3]}"
        )
        assert not none_results, (
            f"count_rows returned None under asyncio.to_thread {len(none_results)} time(s) — "
            "pre-fix regression: daemon crash pattern reproduced"
        )


class TestCountRowsDefensiveRaise:

    def _make_tbl_with_none_fetchone(self, tmp_path: Path):
        from contextlib import contextmanager

        db = HippoDB(tmp_path)
        tbl = db.open_table("records")

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.execute.return_value = mock_cursor
        mock_conn.in_transaction = False

        # count_rows reads through the RO pool when a db is attached, and
        # through _conn otherwise — poison BOTH paths so the None-row check
        # is exercised regardless of which one the implementation takes.
        tbl._conn = mock_conn

        @contextmanager
        def _poisoned_ro_conn():
            yield mock_conn

        db.ro_conn = _poisoned_ro_conn
        return db, tbl

    def test_raises_hippo_integrity_error_not_type_error(self, tmp_path: Path) -> None:
        from iai_mcp.hippo import HippoIntegrityError

        db, tbl = self._make_tbl_with_none_fetchone(tmp_path)

        with pytest.raises(HippoIntegrityError) as exc_info:
            tbl.count_rows()

        err_msg = str(exc_info.value)
        assert any(kw in err_msg for kw in ("records", "COUNT", "None", "fetchone")), (
            f"HippoIntegrityError message lacks diagnostic context: {err_msg!r}"
        )
        db.close()

    def test_type_error_no_longer_escapes(self, tmp_path: Path) -> None:
        from iai_mcp.hippo import HippoIntegrityError

        db, tbl = self._make_tbl_with_none_fetchone(tmp_path)

        raised_type_error = False
        raised_integrity_error = False

        try:
            tbl.count_rows()
        except TypeError:
            raised_type_error = True
        except HippoIntegrityError:
            raised_integrity_error = True

        assert not raised_type_error, (
            "count_rows raised TypeError — pre-fix behavior. "
            "Must raise HippoIntegrityError instead."
        )
        assert raised_integrity_error, (
            "count_rows must raise HippoIntegrityError when fetchone returns None."
        )
        db.close()
