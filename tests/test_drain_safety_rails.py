from __future__ import annotations

import json
import platform
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX paths + atomic rename; deferred-drain is POSIX-only here",
)


@pytest.fixture
def drain_env(tmp_path, monkeypatch):
    """Hermetic drain environment.

    HOME is redirected so the drain reads from a throwaway deferred-captures
    directory. The per-event pending-row writer, write_event and the post-drain
    relief helper are stubbed so no embedder, store, or allocator call is
    exercised — the rails themselves are the unit under test.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("IAI_MCP_DISABLE_INDAEMON_DRAIN", raising=False)
    monkeypatch.delenv("IAI_MCP_DRAIN_RSS_SOFT_CAP_BYTES", raising=False)

    from iai_mcp import capture as capture_mod

    processed: list[str] = []

    def fake_write_pending(store, *, cue="", text="", tier="episodic",
                           session_id="-", role="user", ts=None, **_):
        processed.append(text)
        return {"status": "inserted", "record_id": "rec-x", "reason": ""}

    monkeypatch.setattr(capture_mod, "_drain_write_pending", fake_write_pending)

    # Stub the bank-recent append + prune so no real store/db is needed.
    import iai_mcp.memory_bank as bank_mod

    monkeypatch.setattr(
        bank_mod, "append_recent_record", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        bank_mod, "prune_recent_windows", lambda *a, **k: None, raising=False
    )

    # A fake store object: the pre-embed idem path calls store.find_record_by_tag;
    # returning None forces the "new record" branch so every event flows through
    # fake_capture_turn deterministically.
    store = SimpleNamespace(
        find_record_by_tag=lambda tag: None,
        reinforce_record=lambda rid: None,
        get=lambda rid: None,
    )

    yield SimpleNamespace(home=tmp_path, processed=processed, store=store)


def _deferred_dir(home: Path) -> Path:
    return home / ".iai-mcp" / ".deferred-captures"


def _write_file(deferred_dir: Path, session_id: str, n_events: int, ts_suffix: int) -> Path:
    deferred_dir.mkdir(parents=True, exist_ok=True)
    path = deferred_dir / f"{session_id}-{ts_suffix}.jsonl"
    header = {
        "version": 1,
        "deferred_at": "2026-06-21T00:00:00Z",
        "session_id": session_id,
        "cwd": "/tmp",
    }
    with path.open("w") as fh:
        fh.write(json.dumps(header) + "\n")
        for i in range(n_events):
            fh.write(json.dumps({
                "text": f"event {i} of {session_id} with sufficient text length",
                "cue": f"cue-{i}",
                "tier": "episodic",
                "role": "user",
                "ts": "2026-06-21T00:00:00Z",
            }) + "\n")
    return path


# --------------------------------------------------------------------------
# Rail 1: RSS soft cap stops the drain early, remaining files preserved.
# --------------------------------------------------------------------------


def test_rss_soft_cap_stops_drain_early(drain_env, monkeypatch):
    from iai_mcp import capture as capture_mod

    deferred = _deferred_dir(drain_env.home)
    f1 = _write_file(deferred, "sess-a", n_events=3, ts_suffix=1700000001)
    f2 = _write_file(deferred, "sess-b", n_events=3, ts_suffix=1700000002)
    f3 = _write_file(deferred, "sess-c", n_events=3, ts_suffix=1700000003)

    monkeypatch.setenv("IAI_MCP_DRAIN_RSS_SOFT_CAP_BYTES", "1000")
    monkeypatch.setenv("IAI_MCP_DRAIN_RSS_GROWTH_BUDGET_BYTES", "500")

    # The threshold is baseline-relative: baseline 2000 + budget 500 = 2500.
    # The first per-file sample reports 9000 — over threshold before any file
    # is claimed: the drain must stop immediately with zero files processed.
    samples = iter([2000])
    monkeypatch.setattr(
        capture_mod, "_drain_rss_bytes", lambda: next(samples, 9000)
    )

    relief_calls: list[str] = []
    import iai_mcp.lilli.cycle.sleep_pipeline._memory_relief as relief_mod
    monkeypatch.setattr(
        relief_mod, "_step_memory_relief",
        lambda label="": relief_calls.append(label) or {},
    )

    events: list[tuple] = []
    import iai_mcp.events as events_mod
    monkeypatch.setattr(
        events_mod, "write_event",
        lambda store, kind, data, **kw: events.append((kind, data)),
    )

    counts = capture_mod.drain_deferred_captures(drain_env.store)

    # No files were processed (the soft cap tripped before the first claim).
    assert counts["files_drained"] == 0, counts
    assert counts["events_inserted"] == 0, counts
    assert counts.get("rss_soft_cap_hit") == 1, counts
    assert drain_env.processed == [], "no event should have been embedded"

    # All three files are preserved on disk for the next cycle (deferred, not lost).
    remaining = sorted(p.name for p in deferred.glob("*.jsonl"))
    assert remaining == [f1.name, f2.name, f3.name], remaining
    assert list(deferred.glob("*.processing-*.jsonl")) == []

    # Telemetry fired; no relief on a zero-work run.
    kinds = [k for k, _ in events]
    assert "drain_rss_soft_cap" in kinds, events
    assert relief_calls == [], "relief is for runs that did real work"


def test_rss_soft_cap_partial_then_stop(drain_env, monkeypatch):
    from iai_mcp import capture as capture_mod

    deferred = _deferred_dir(drain_env.home)
    _write_file(deferred, "sess-a", n_events=2, ts_suffix=1700000001)
    _write_file(deferred, "sess-b", n_events=2, ts_suffix=1700000002)
    _write_file(deferred, "sess-c", n_events=2, ts_suffix=1700000003)

    monkeypatch.setenv("IAI_MCP_DRAIN_RSS_SOFT_CAP_BYTES", "1000")
    monkeypatch.setenv("IAI_MCP_DRAIN_RSS_GROWTH_BUDGET_BYTES", "500")

    # Baseline 500 + budget 500 = threshold 1000 (equals the floor). Under it
    # for the first file, far over afterwards: exactly one file drains, the
    # rest stay on disk.
    samples = iter([500, 800, 5000, 5000])
    monkeypatch.setattr(capture_mod, "_drain_rss_bytes", lambda: next(samples, 5000))

    import iai_mcp.lilli.cycle.sleep_pipeline._memory_relief as _relief_mod
    monkeypatch.setattr(_relief_mod, "_step_memory_relief", lambda label="": {})
    import iai_mcp.events as events_mod
    monkeypatch.setattr(events_mod, "write_event", lambda *a, **k: None)

    counts = capture_mod.drain_deferred_captures(drain_env.store)

    assert counts["files_drained"] == 1, counts
    assert counts.get("rss_soft_cap_hit") == 1, counts
    # Two files remain unclaimed and untouched.
    remaining = sorted(p.name for p in deferred.glob("*.jsonl"))
    assert len(remaining) == 2, remaining


def test_rss_soft_cap_zero_disables(drain_env, monkeypatch):
    from iai_mcp import capture as capture_mod

    deferred = _deferred_dir(drain_env.home)
    _write_file(deferred, "sess-a", n_events=2, ts_suffix=1700000001)

    # Soft cap disabled (0) — even a huge RSS reading must not stop the drain.
    monkeypatch.setenv("IAI_MCP_DRAIN_RSS_SOFT_CAP_BYTES", "0")
    monkeypatch.setattr(capture_mod, "_drain_rss_bytes", lambda: 10**12)
    import iai_mcp.lilli.cycle.sleep_pipeline._memory_relief as _relief_mod
    monkeypatch.setattr(_relief_mod, "_step_memory_relief", lambda label="": {})
    import iai_mcp.events as events_mod
    monkeypatch.setattr(events_mod, "write_event", lambda *a, **k: None)

    counts = capture_mod.drain_deferred_captures(drain_env.store)

    assert counts["files_drained"] == 1, counts
    assert counts.get("rss_soft_cap_hit") is None, counts


# --------------------------------------------------------------------------
# Rail 2: kill-switch makes the drain a no-op.
# --------------------------------------------------------------------------


def test_kill_switch_makes_drain_noop(drain_env, monkeypatch):
    from iai_mcp import capture as capture_mod

    deferred = _deferred_dir(drain_env.home)
    f1 = _write_file(deferred, "sess-a", n_events=5, ts_suffix=1700000001)
    f2 = _write_file(deferred, "sess-b", n_events=5, ts_suffix=1700000002)

    monkeypatch.setenv("IAI_MCP_DISABLE_INDAEMON_DRAIN", "1")

    # If the kill-switch fails to short-circuit, this RSS sampler would be hit.
    def boom():
        raise AssertionError("drain body ran despite kill-switch")

    monkeypatch.setattr(capture_mod, "_drain_rss_bytes", boom)

    counts = capture_mod.drain_deferred_captures(drain_env.store)

    assert counts.get("disabled") == 1, counts
    assert counts["files_drained"] == 0, counts
    assert counts["events_inserted"] == 0, counts
    assert drain_env.processed == [], "no event embedded when disabled"

    # Both files untouched, no processing marker left behind.
    remaining = sorted(p.name for p in deferred.glob("*.jsonl"))
    assert remaining == [f1.name, f2.name], remaining
    assert list(deferred.glob("*.processing-*.jsonl")) == []


@pytest.mark.parametrize("flag", ["1", "true", "TRUE", "yes", "on"])
def test_kill_switch_truthy_values(drain_env, monkeypatch, flag):
    from iai_mcp import capture as capture_mod

    _write_file(_deferred_dir(drain_env.home), "s", n_events=2, ts_suffix=1700000005)
    monkeypatch.setenv("IAI_MCP_DISABLE_INDAEMON_DRAIN", flag)
    counts = capture_mod.drain_deferred_captures(drain_env.store)
    assert counts.get("disabled") == 1, (flag, counts)


@pytest.mark.parametrize("flag", ["0", "false", "no", "off", ""])
def test_kill_switch_falsy_values_run_drain(drain_env, monkeypatch, flag):
    from iai_mcp import capture as capture_mod

    _write_file(_deferred_dir(drain_env.home), "s", n_events=2, ts_suffix=1700000006)
    monkeypatch.setenv("IAI_MCP_DISABLE_INDAEMON_DRAIN", flag)
    import iai_mcp.lilli.cycle.sleep_pipeline._memory_relief as _relief_mod
    monkeypatch.setattr(_relief_mod, "_step_memory_relief", lambda label="": {})
    import iai_mcp.events as events_mod
    monkeypatch.setattr(events_mod, "write_event", lambda *a, **k: None)

    counts = capture_mod.drain_deferred_captures(drain_env.store)
    assert counts.get("disabled") is None, (flag, counts)
    assert counts["files_drained"] == 1, (flag, counts)


# --------------------------------------------------------------------------
# Rail 3: single-flight — a second concurrent drain skips.
# --------------------------------------------------------------------------


def test_single_flight_second_caller_skips(drain_env, monkeypatch):
    from iai_mcp import capture as capture_mod

    _write_file(_deferred_dir(drain_env.home), "sess-a", n_events=2, ts_suffix=1700000001)

    # Hold the single-flight lock as if a first drain were in progress, then call
    # drain again on the same thread — the second call must skip immediately.
    acquired = capture_mod._DRAIN_SINGLE_FLIGHT_LOCK.acquire(blocking=False)
    assert acquired

    def boom():
        raise AssertionError("second drain ran while first held the lock")

    monkeypatch.setattr(capture_mod, "_drain_rss_bytes", boom)

    try:
        counts = capture_mod.drain_deferred_captures(drain_env.store)
    finally:
        capture_mod._DRAIN_SINGLE_FLIGHT_LOCK.release()

    assert counts.get("skipped_single_flight") == 1, counts
    assert counts["files_drained"] == 0, counts
    assert drain_env.processed == [], "skipped drain must not process events"


def test_single_flight_concurrent_no_double_processing(drain_env, monkeypatch):
    from iai_mcp import capture as capture_mod

    deferred = _deferred_dir(drain_env.home)
    _write_file(deferred, "sess-a", n_events=4, ts_suffix=1700000001)
    _write_file(deferred, "sess-b", n_events=4, ts_suffix=1700000002)

    import iai_mcp.lilli.cycle.sleep_pipeline._memory_relief as _relief_mod
    monkeypatch.setattr(_relief_mod, "_step_memory_relief", lambda label="": {})
    import iai_mcp.events as events_mod
    monkeypatch.setattr(events_mod, "write_event", lambda *a, **k: None)

    barrier = threading.Barrier(2, timeout=10)
    results: dict[str, dict] = {}

    def worker(name):
        barrier.wait()
        results[name] = capture_mod.drain_deferred_captures(drain_env.store)

    t1 = threading.Thread(target=worker, args=("t1",))
    t2 = threading.Thread(target=worker, args=("t2",))
    t1.start()
    t2.start()
    t1.join(timeout=15)
    t2.join(timeout=15)

    statuses = [r.get("skipped_single_flight") for r in results.values()]
    # At least one ran for real; if they happened to serialize, neither skipped
    # but no file is processed twice either. The invariant we assert is: total
    # events processed across both calls never exceeds the events on disk.
    total_inserted = sum(r.get("events_inserted", 0) for r in results.values())
    assert total_inserted <= 8, results
    # The lock is released after both calls return.
    assert capture_mod._DRAIN_SINGLE_FLIGHT_LOCK.acquire(blocking=False)
    capture_mod._DRAIN_SINGLE_FLIGHT_LOCK.release()
    assert len(statuses) == 2


# --------------------------------------------------------------------------
# Rail 4: post-drain relief is invoked after real work.
# --------------------------------------------------------------------------


def test_post_drain_relief_called_on_real_work(drain_env, monkeypatch):
    from iai_mcp import capture as capture_mod

    _write_file(_deferred_dir(drain_env.home), "sess-a", n_events=3, ts_suffix=1700000001)

    relief_calls: list[str] = []
    import iai_mcp.lilli.cycle.sleep_pipeline._memory_relief as relief_mod
    monkeypatch.setattr(
        relief_mod, "_step_memory_relief",
        lambda label="": relief_calls.append(label) or {},
    )
    import iai_mcp.events as events_mod
    monkeypatch.setattr(events_mod, "write_event", lambda *a, **k: None)

    counts = capture_mod.drain_deferred_captures(drain_env.store)

    assert counts["files_drained"] == 1, counts
    assert relief_calls == ["deferred_drain"], relief_calls


def test_post_drain_relief_skipped_on_empty(drain_env, monkeypatch):
    from iai_mcp import capture as capture_mod

    # No deferred files at all → zero work → no relief.
    _deferred_dir(drain_env.home).mkdir(parents=True, exist_ok=True)

    relief_calls: list[str] = []
    import iai_mcp.lilli.cycle.sleep_pipeline._memory_relief as relief_mod
    monkeypatch.setattr(
        relief_mod, "_step_memory_relief",
        lambda label="": relief_calls.append(label) or {},
    )

    counts = capture_mod.drain_deferred_captures(drain_env.store)

    assert counts["files_drained"] == 0, counts
    assert relief_calls == [], relief_calls
