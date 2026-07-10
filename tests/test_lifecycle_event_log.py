from __future__ import annotations

import gzip
import json
import multiprocessing as mp
import os
from datetime import datetime, timedelta, timezone

import pytest

from iai_mcp.lifecycle_event_log import (
    KNOWN_EVENT_KINDS,
    LifecycleEventLog,
    _utc_date_string,
)


def test_append_writes_jsonl_line(tmp_path):
    log = LifecycleEventLog(log_dir=tmp_path)
    log.append({"event": "state_transition", "from": "WAKE", "to": "DROWSY",
                "trigger": "idle_5min"})

    path = log.current_file()
    assert path.exists()
    content = path.read_text()
    assert content.endswith("\n")
    record = json.loads(content.strip())
    assert record["event"] == "state_transition"
    assert record["from"] == "WAKE"
    assert record["to"] == "DROWSY"
    assert "ts" in record
    datetime.fromisoformat(record["ts"])


def test_append_preserves_caller_ts(tmp_path):
    log = LifecycleEventLog(log_dir=tmp_path)
    explicit_ts = "2026-05-02T15:00:00+00:00"
    log.append({"ts": explicit_ts, "event": "wrapper_event",
                "kind": "heartbeat_refresh", "wrapper_pid": 12345})

    records = log.read_all()
    assert len(records) == 1
    assert records[0]["ts"] == explicit_ts


def test_append_rejects_non_dict(tmp_path):
    log = LifecycleEventLog(log_dir=tmp_path)
    with pytest.raises(TypeError):
        log.append("not a dict")  # type: ignore[arg-type]


def test_append_rejects_missing_event_kind(tmp_path):
    log = LifecycleEventLog(log_dir=tmp_path)
    with pytest.raises(ValueError):
        log.append({"ts": "2026-05-02T00:00:00+00:00"})


def test_append_does_not_mutate_caller_dict(tmp_path):
    log = LifecycleEventLog(log_dir=tmp_path)
    payload = {"event": "wrapper_event", "kind": "heartbeat_refresh"}
    snapshot = dict(payload)
    log.append(payload)
    assert payload == snapshot, "append must not mutate caller's dict"


def test_append_creates_log_dir_if_missing(tmp_path):
    deep = tmp_path / "nested" / "a" / "b"
    log = LifecycleEventLog(log_dir=deep)
    log.append({"event": "wrapper_event", "kind": "heartbeat_refresh"})
    assert log.current_file().exists()


def test_append_accumulates_lines(tmp_path):
    log = LifecycleEventLog(log_dir=tmp_path)
    for i in range(10):
        log.append({"event": "wrapper_event", "kind": "heartbeat_refresh",
                    "wrapper_pid": 1000 + i})
    records = log.read_all()
    assert len(records) == 10
    assert [r["wrapper_pid"] for r in records] == list(range(1000, 1010))


def test_log_file_chmod_user_only(tmp_path):
    log = LifecycleEventLog(log_dir=tmp_path)
    log.append({"event": "wrapper_event", "kind": "heartbeat_refresh"})
    mode = os.stat(log.current_file()).st_mode & 0o777
    assert mode == 0o600


def test_rotation_writes_to_per_date_file(tmp_path):
    log = LifecycleEventLog(log_dir=tmp_path)
    day1 = datetime(2026, 5, 2, 23, 30, tzinfo=timezone.utc)
    day2 = datetime(2026, 5, 3, 0, 30, tzinfo=timezone.utc)

    log.append({"event": "wrapper_event", "kind": "heartbeat_refresh"}, now=day1)
    log.append({"event": "wrapper_event", "kind": "heartbeat_refresh"}, now=day2)

    f1 = tmp_path / "lifecycle-events-2026-05-02.jsonl"
    f2 = tmp_path / "lifecycle-events-2026-05-03.jsonl"
    assert f1.exists()
    assert f2.exists()
    assert len(f1.read_text().splitlines()) == 1
    assert len(f2.read_text().splitlines()) == 1


def test_rotation_uses_utc_not_local(tmp_path, monkeypatch):
    log = LifecycleEventLog(log_dir=tmp_path)
    moment = datetime(2026, 5, 2, 0, 0, 0, tzinfo=timezone.utc)
    log.append({"event": "wrapper_event", "kind": "heartbeat_refresh"}, now=moment)
    assert (tmp_path / "lifecycle-events-2026-05-02.jsonl").exists()


def test_rotate_old_files_gzips_files_past_retention(tmp_path):
    log = LifecycleEventLog(log_dir=tmp_path)
    today = datetime(2026, 5, 2, 12, tzinfo=timezone.utc)

    log.append({"event": "wrapper_event", "kind": "heartbeat_refresh"},
               now=today)
    old = today - timedelta(days=35)
    log.append({"event": "wrapper_event", "kind": "heartbeat_refresh"},
               now=old)

    f_today = tmp_path / "lifecycle-events-2026-05-02.jsonl"
    f_old_path = log.file_for_date(_utc_date_string(old))
    assert f_today.exists()
    assert f_old_path.exists()

    n = log.rotate_old_files(retention_days=30, now=today)
    assert n == 1
    assert not f_old_path.exists()
    assert f_old_path.with_suffix(".jsonl.gz").exists()
    assert f_today.exists()


def test_rotate_old_files_idempotent_on_already_compressed(tmp_path):
    log = LifecycleEventLog(log_dir=tmp_path)
    today = datetime(2026, 5, 2, 12, tzinfo=timezone.utc)
    old = today - timedelta(days=40)
    log.append({"event": "wrapper_event", "kind": "heartbeat_refresh"},
               now=old)

    n1 = log.rotate_old_files(retention_days=30, now=today)
    n2 = log.rotate_old_files(retention_days=30, now=today)
    assert n1 == 1
    assert n2 == 0


def test_rotate_old_files_gzip_content_matches(tmp_path):
    log = LifecycleEventLog(log_dir=tmp_path)
    today = datetime(2026, 5, 2, 12, tzinfo=timezone.utc)
    old = today - timedelta(days=35)

    log.append({"event": "state_transition", "from": "WAKE", "to": "DROWSY",
                "trigger": "idle_5min"}, now=old)
    src_path = log.file_for_date(_utc_date_string(old))
    src_text = src_path.read_text()

    log.rotate_old_files(retention_days=30, now=today)
    gz_path = src_path.with_suffix(".jsonl.gz")
    with gzip.open(gz_path, "rt") as f:
        assert f.read() == src_text


def test_rotate_old_files_skips_unrecognised_filenames(tmp_path):
    log = LifecycleEventLog(log_dir=tmp_path)
    today = datetime(2026, 5, 2, 12, tzinfo=timezone.utc)

    bogus = tmp_path / "lifecycle-events-not-a-date.jsonl"
    bogus.write_text('{"event": "wrapper_event"}\n')

    n = log.rotate_old_files(retention_days=30, now=today)
    assert n == 0
    assert bogus.exists()


def test_read_all_skips_truncated_trailing_line(tmp_path):
    log = LifecycleEventLog(log_dir=tmp_path)
    log.append({"event": "wrapper_event", "kind": "heartbeat_refresh", "i": 1})
    log.append({"event": "wrapper_event", "kind": "heartbeat_refresh", "i": 2})
    with log.current_file().open("a") as f:
        f.write('{"event": "wrapper_event", "kind": "heart')

    records = log.read_all()
    assert len(records) == 2
    assert [r["i"] for r in records] == [1, 2]


def test_read_all_returns_empty_when_no_file(tmp_path):
    log = LifecycleEventLog(log_dir=tmp_path)
    assert log.read_all() == []


def _writer_worker(log_dir_str: str, n: int, marker: str) -> None:
    from iai_mcp.lifecycle_event_log import LifecycleEventLog as _Log

    log = _Log(log_dir=__import__("pathlib").Path(log_dir_str))
    for i in range(n):
        log.append({"event": "wrapper_event", "kind": "heartbeat_refresh",
                    "marker": marker, "i": i})


@pytest.mark.skipif(
    os.name != "posix",
    reason="fcntl.flock concurrency invariant is POSIX-only",
)
def test_concurrent_writes_no_torn_lines(tmp_path):
    n_per_worker = 50
    procs = [
        mp.Process(target=_writer_worker, args=(str(tmp_path), n_per_worker, "A")),
        mp.Process(target=_writer_worker, args=(str(tmp_path), n_per_worker, "B")),
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0, f"worker {p.name} failed: {p.exitcode}"

    log = LifecycleEventLog(log_dir=tmp_path)
    records = log.read_all()
    assert len(records) == 2 * n_per_worker
    a_indices = [r["i"] for r in records if r.get("marker") == "A"]
    b_indices = [r["i"] for r in records if r.get("marker") == "B"]
    assert a_indices == list(range(n_per_worker))
    assert b_indices == list(range(n_per_worker))


def test_known_event_kinds_includes_spec(tmp_path):
    expected = {
        "state_transition",
        "wrapper_event",
        "shadow_run_warning",
        "sleep_step_started",
        "sleep_step_completed",
        "quarantine_entered",
        "quarantine_lifted",
    }
    assert expected.issubset(KNOWN_EVENT_KINDS)
