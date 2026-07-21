"""Bounded capture-queue drain: an arbitrary pending backlog must never own
the boot path, and bounded passes must converge to a fully drained queue."""

from __future__ import annotations

import time

import pytest

from iai_mcp.capture_queue import CaptureQueue


def _fill(queue: CaptureQueue, n: int) -> None:
    for i in range(n):
        queue.append({"cue": f"c{i}", "text": f"t{i}", "role": "user"})


@pytest.fixture()
def queue(tmp_path) -> CaptureQueue:
    return CaptureQueue(queue_dir=tmp_path / "pending")


def test_max_records_bounds_one_pass(queue: CaptureQueue) -> None:
    _fill(queue, 12)
    seen: list[dict] = []
    ingested = queue.ingest_pending(seen.append, max_records=5)
    assert ingested == 5
    assert len(seen) == 5
    assert queue.pending_count() == 7


def test_budget_sec_bounds_one_pass(queue: CaptureQueue) -> None:
    _fill(queue, 50)

    def slow_handler(record: dict) -> None:
        time.sleep(0.05)

    ingested = queue.ingest_pending(slow_handler, budget_sec=0.12)
    assert 1 <= ingested < 50
    assert queue.pending_count() == 50 - ingested


def test_bounded_passes_converge_to_empty(queue: CaptureQueue) -> None:
    _fill(queue, 17)
    total = 0
    passes = 0
    while queue.pending_count() > 0:
        total += queue.ingest_pending(lambda r: None, max_records=4)
        passes += 1
        assert passes <= 6, "bounded passes must make monotonic progress"
    assert total == 17


def test_unbounded_default_drains_everything(queue: CaptureQueue) -> None:
    _fill(queue, 9)
    assert queue.ingest_pending(lambda r: None) == 9
    assert queue.pending_count() == 0


def test_nonpositive_max_records_is_a_noop(queue: CaptureQueue) -> None:
    _fill(queue, 3)
    assert queue.ingest_pending(lambda r: None, max_records=0) == 0
    assert queue.pending_count() == 3


def test_daemon_bound_helpers_honor_env(monkeypatch) -> None:
    from iai_mcp.daemon import (
        _capture_drain_budget_sec,
        _capture_drain_max_records,
        _CAPTURE_DRAIN_BUDGET_SEC_DEFAULT,
        _CAPTURE_DRAIN_MAX_RECORDS_DEFAULT,
    )

    monkeypatch.delenv("IAI_MCP_CAPTURE_DRAIN_MAX_RECORDS", raising=False)
    monkeypatch.delenv("IAI_MCP_CAPTURE_DRAIN_BUDGET_SEC", raising=False)
    assert _capture_drain_max_records() == _CAPTURE_DRAIN_MAX_RECORDS_DEFAULT
    assert _capture_drain_budget_sec() == _CAPTURE_DRAIN_BUDGET_SEC_DEFAULT

    monkeypatch.setenv("IAI_MCP_CAPTURE_DRAIN_MAX_RECORDS", "50")
    monkeypatch.setenv("IAI_MCP_CAPTURE_DRAIN_BUDGET_SEC", "2.5")
    assert _capture_drain_max_records() == 50
    assert _capture_drain_budget_sec() == 2.5

    # Non-positive disables the bound (unbounded pass) rather than freezing it.
    monkeypatch.setenv("IAI_MCP_CAPTURE_DRAIN_MAX_RECORDS", "0")
    monkeypatch.setenv("IAI_MCP_CAPTURE_DRAIN_BUDGET_SEC", "-1")
    assert _capture_drain_max_records() is None
    assert _capture_drain_budget_sec() is None

    monkeypatch.setenv("IAI_MCP_CAPTURE_DRAIN_MAX_RECORDS", "junk")
    assert _capture_drain_max_records() == _CAPTURE_DRAIN_MAX_RECORDS_DEFAULT
