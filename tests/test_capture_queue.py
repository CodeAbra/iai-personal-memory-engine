from __future__ import annotations

import errno
from iai_mcp import _flock as fcntl
import json
import os
import threading
from pathlib import Path

import pytest

from iai_mcp.capture_queue import (
    SCHEMA_VERSION,
    CaptureQueue,
    CaptureQueueSchemaError,
    generate_ulid,
)


def _sample_record(i: int = 0, surface: str | None = None) -> dict:
    return {
        "surface": surface if surface is not None else f"sample text {i}",
        "cue": f"cue {i}",
        "tier": "episodic",
        "session_id": "test-session",
        "role": "user",
    }


def _write_envelope_directly(
    queue_dir: Path,
    ulid: str,
    record: dict,
    *,
    schema_version: int = SCHEMA_VERSION,
    appended_at: str = "2026-05-02T15:00:00+00:00",
) -> Path:
    path = queue_dir / f"pending-{ulid}.json"
    envelope = {
        "ulid": ulid,
        "appended_at": appended_at,
        "record": record,
        "schema_version": schema_version,
    }
    path.write_text(
        json.dumps(envelope, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def test_append_returns_ulid_and_creates_file(tmp_path):
    q = CaptureQueue(queue_dir=tmp_path)
    ulid = q.append(_sample_record(0))

    assert isinstance(ulid, str)
    assert len(ulid) == 26
    final = tmp_path / f"pending-{ulid}.json"
    assert final.exists()

    envelope = json.loads(final.read_text(encoding="utf-8"))
    assert envelope["ulid"] == ulid
    assert envelope["schema_version"] == SCHEMA_VERSION
    assert envelope["record"]["surface"] == "sample text 0"
    from datetime import datetime
    datetime.fromisoformat(envelope["appended_at"])

    assert q.pending_count() == 1


def test_append_atomic_under_crash_simulation(tmp_path, monkeypatch):
    q = CaptureQueue(queue_dir=tmp_path)

    real_replace = os.replace

    def boom(src, dst):
        raise OSError(errno.EIO, "simulated crash mid-rename")

    monkeypatch.setattr("iai_mcp.capture_queue.os.replace", boom)

    with pytest.raises(OSError):
        q.append(_sample_record(0))

    assert q.pending_count() == 0
    finals = list(tmp_path.glob("pending-*.json"))
    finals = [p for p in finals if not p.name.endswith(".tmp")]
    assert finals == []

    monkeypatch.setattr("iai_mcp.capture_queue.os.replace", real_replace)
    q.append(_sample_record(1))
    assert q.pending_count() == 1


def test_concurrent_append_50_threads(tmp_path):
    q = CaptureQueue(queue_dir=tmp_path)
    n_threads = 50
    n_per_thread = 10
    errors: list[BaseException] = []
    ulids: list[str] = []
    ulids_lock = threading.Lock()

    def worker(tid: int) -> None:
        try:
            local: list[str] = []
            for i in range(n_per_thread):
                ulid = q.append(_sample_record(i, f"thread-{tid}-record-{i}"))
                local.append(ulid)
            with ulids_lock:
                ulids.extend(local)
        except BaseException as exc:  # pragma: no cover - surfaced via assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive(), "worker thread hung"

    assert errors == [], f"workers raised: {errors!r}"
    assert len(ulids) == n_threads * n_per_thread
    assert len(set(ulids)) == len(ulids)
    pending = q.list_pending()
    assert len(pending) == n_threads * n_per_thread
    for p in pending:
        envelope = json.loads(p.read_text(encoding="utf-8"))
        assert envelope["schema_version"] == SCHEMA_VERSION
        assert envelope["record"]["surface"].startswith("thread-")


def test_idempotent_ingest_crash_mid_handler(tmp_path):
    q = CaptureQueue(queue_dir=tmp_path)
    ulid = q.append(_sample_record(42, surface="payload-42"))

    pending_path = tmp_path / f"pending-{ulid}.json"
    lock_path = tmp_path / f"pending-{ulid}.lock"

    def crashing_handler(_record: dict) -> None:
        raise RuntimeError("handler exploded")

    with pytest.raises(RuntimeError):
        q.ingest_pending(crashing_handler)

    assert pending_path.exists(), "pending file must remain after handler exception"
    assert lock_path.exists(), "lock file must remain to mark mid-flight crash"
    assert q.pending_count() == 1

    seen: list[dict] = []

    def good_handler(record: dict) -> None:
        seen.append(record)

    n = q.ingest_pending(good_handler)
    assert n == 1
    assert len(seen) == 1
    assert seen[0]["surface"] == "payload-42"
    assert not pending_path.exists()
    assert not lock_path.exists()
    assert q.pending_count() == 0


def test_idempotent_ingest_lock_skipped(tmp_path):
    q = CaptureQueue(queue_dir=tmp_path)
    ulid_a = q.append(_sample_record(1, surface="A"))
    ulid_b = q.append(_sample_record(2, surface="B"))
    ulid_c = q.append(_sample_record(3, surface="C"))

    lock_a = tmp_path / f"pending-{ulid_a}.lock"
    fd = os.open(str(lock_a), os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

        seen: list[str] = []

        def handler(record: dict) -> None:
            seen.append(record["surface"])

        n = q.ingest_pending(handler)
        assert n == 2
        assert sorted(seen) == ["B", "C"]
        assert (tmp_path / f"pending-{ulid_a}.json").exists()
        assert not (tmp_path / f"pending-{ulid_b}.json").exists()
        assert not (tmp_path / f"pending-{ulid_c}.json").exists()
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def test_overflow_prune_oldest(tmp_path):
    max_size = 100
    n_total = 110
    q = CaptureQueue(queue_dir=tmp_path, max_size=max_size)

    appended_ulids: list[str] = []
    for i in range(n_total):
        appended_ulids.append(q.append(_sample_record(i)))

    final_count = q.pending_count()
    assert final_count <= max_size

    audit_path = tmp_path / ".overflow-audit.log"
    assert audit_path.exists(), "audit log must exist after overflow"

    audit_lines = audit_path.read_text(encoding="utf-8").splitlines()
    audit_records = [json.loads(line) for line in audit_lines if line]

    dropped = n_total - final_count
    assert dropped > 0, "at least one record must have been dropped on overflow"
    assert len(audit_records) == dropped, (
        f"expected {dropped} audit entries, got {len(audit_records)}"
    )
    for rec in audit_records:
        assert rec["reason"] == "queue_overflow"
        assert rec["dropped_ulid"] in appended_ulids
        assert isinstance(rec["queue_size_before_prune"], int)


def test_verbatim_round_trip_unicode(tmp_path):
    q = CaptureQueue(queue_dir=tmp_path)
    payload = "こんにちは, world! 🧠 Δ ∑ — combining é vs é"

    q.append(_sample_record(0, surface=payload))
    seen: list[str] = []

    def handler(record: dict) -> None:
        seen.append(record["surface"])

    n = q.ingest_pending(handler)
    assert n == 1
    assert len(seen) == 1
    assert seen[0] == payload
    assert seen[0].encode("utf-8") == payload.encode("utf-8")


def test_list_pending_sort_order(tmp_path):
    q = CaptureQueue(queue_dir=tmp_path)
    ulids = [q.append(_sample_record(i)) for i in range(20)]
    listed = [q._ulid_from_path(p) for p in q.list_pending()]
    assert listed == ulids, "list_pending must be oldest-first"


def test_schema_version_mismatch_raises(tmp_path):
    q = CaptureQueue(queue_dir=tmp_path)
    _write_envelope_directly(
        tmp_path,
        ulid="01HZQTESTBADSCHEMA00000000",
        record=_sample_record(0),
        schema_version=99,
    )
    assert q.pending_count() == 1

    def handler(_record: dict) -> None:  # pragma: no cover -- never called
        pytest.fail("handler must not be called on schema mismatch")

    with pytest.raises(CaptureQueueSchemaError) as excinfo:
        q.ingest_pending(handler)
    assert "schema_version" in str(excinfo.value)
    assert "99" in str(excinfo.value)


def test_empty_queue_ingest_returns_zero(tmp_path):
    q = CaptureQueue(queue_dir=tmp_path)
    assert q.pending_count() == 0

    handler_called = [False]

    def handler(_record: dict) -> None:  # pragma: no cover -- never called
        handler_called[0] = True

    n = q.ingest_pending(handler)
    assert n == 0
    assert handler_called[0] is False


def test_ulid_lexicographic_sort_matches_time_order():
    n = 1000
    ulids = [generate_ulid() for _ in range(n)]
    assert len(set(ulids)) == n, "no ULID collisions allowed"
    assert sorted(ulids) == ulids, "lex sort must equal generation order"


def test_lock_file_cleanup_on_handler_success(tmp_path):
    q = CaptureQueue(queue_dir=tmp_path)
    ulid = q.append(_sample_record(0))
    lock_path = tmp_path / f"pending-{ulid}.lock"

    def handler(_record: dict) -> None:
        pass

    n = q.ingest_pending(handler)
    assert n == 1
    assert not lock_path.exists(), "lock file must be cleaned on success"
    assert not (tmp_path / f"pending-{ulid}.json").exists()


def test_lock_file_persists_on_handler_exception(tmp_path):
    q = CaptureQueue(queue_dir=tmp_path)
    ulid = q.append(_sample_record(0))
    pending_path = tmp_path / f"pending-{ulid}.json"
    lock_path = tmp_path / f"pending-{ulid}.lock"

    def handler(_record: dict) -> None:
        raise ValueError("simulated mid-handler crash")

    with pytest.raises(ValueError):
        q.ingest_pending(handler)

    assert pending_path.exists(), "pending must remain after handler exception"
    assert lock_path.exists(), "lock must remain to mark mid-flight crash"
