from __future__ import annotations

import platform

import pytest


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="daemon module is POSIX-only on this project",
)


def _states():
    from iai_mcp.lifecycle_state import LifecycleState
    return LifecycleState


# --- _wake_valve_due -------------------------------------------------------


def test_wake_valve_not_due_when_not_wake():
    from iai_mcp.daemon import _wake_valve_due

    assert _wake_valve_due(
        now_mono=1000.0, last_probe_mono=0.0, floor_sec=600.0,
        inflight=False, is_wake=False,
    ) is False


def test_wake_valve_not_due_before_floor_elapses():
    from iai_mcp.daemon import _wake_valve_due

    assert _wake_valve_due(
        now_mono=100.0, last_probe_mono=0.0, floor_sec=600.0,
        inflight=False, is_wake=True,
    ) is False


def test_wake_valve_disabled_when_floor_non_positive():
    from iai_mcp.daemon import _wake_valve_due

    assert _wake_valve_due(
        now_mono=10_000.0, last_probe_mono=0.0, floor_sec=0.0,
        inflight=False, is_wake=True,
    ) is False
    assert _wake_valve_due(
        now_mono=10_000.0, last_probe_mono=0.0, floor_sec=-1.0,
        inflight=False, is_wake=True,
    ) is False


def test_wake_valve_not_due_while_inflight():
    from iai_mcp.daemon import _wake_valve_due

    assert _wake_valve_due(
        now_mono=1000.0, last_probe_mono=0.0, floor_sec=600.0,
        inflight=True, is_wake=True,
    ) is False


def test_wake_valve_due_when_wake_and_floor_elapsed():
    from iai_mcp.daemon import _wake_valve_due

    assert _wake_valve_due(
        now_mono=601.0, last_probe_mono=0.0, floor_sec=600.0,
        inflight=False, is_wake=True,
    ) is True


# --- _spool_backlog_over_threshold ------------------------------------------


def test_first_probe_calibrates_and_does_not_fire_on_bytes():
    from iai_mcp.daemon import _spool_backlog_over_threshold

    snapshot = {"finalized_files": 0, "live_bytes_total": 999_999}
    should_drain, new_baseline = _spool_backlog_over_threshold(snapshot, None, 65536)
    assert should_drain is False
    assert new_baseline == 999_999


def test_first_probe_fires_on_finalized_even_while_calibrating():
    from iai_mcp.daemon import _spool_backlog_over_threshold

    snapshot = {"finalized_files": 1, "live_bytes_total": 0}
    should_drain, new_baseline = _spool_backlog_over_threshold(snapshot, None, 65536)
    assert should_drain is True
    assert new_baseline == 0


def test_delta_under_min_bytes_does_not_fire():
    from iai_mcp.daemon import _spool_backlog_over_threshold

    snapshot = {"finalized_files": 0, "live_bytes_total": 100_100}
    should_drain, new_baseline = _spool_backlog_over_threshold(
        snapshot, 100_000, 65536,
    )
    assert should_drain is False
    # Growth below threshold: baseline must NOT advance, so a slow trickle
    # still accumulates toward the threshold across probes.
    assert new_baseline == 100_000


def test_delta_at_or_over_min_bytes_fires_and_advances_baseline():
    from iai_mcp.daemon import _spool_backlog_over_threshold

    snapshot = {"finalized_files": 0, "live_bytes_total": 165_536}
    should_drain, new_baseline = _spool_backlog_over_threshold(
        snapshot, 100_000, 65536,
    )
    assert should_drain is True
    assert new_baseline == 165_536


def test_finalized_files_fire_regardless_of_byte_delta():
    from iai_mcp.daemon import _spool_backlog_over_threshold

    snapshot = {"finalized_files": 1, "live_bytes_total": 100_000}
    should_drain, new_baseline = _spool_backlog_over_threshold(
        snapshot, 100_000, 65536,
    )
    assert should_drain is True
    assert new_baseline == 100_000


def test_shrunk_live_total_recalibrates_without_firing():
    from iai_mcp.daemon import _spool_backlog_over_threshold

    # Live file rotated/removed since the last probe: total dropped below
    # the old baseline. Must re-anchor, not treat this as negative growth.
    snapshot = {"finalized_files": 0, "live_bytes_total": 10_000}
    should_drain, new_baseline = _spool_backlog_over_threshold(
        snapshot, 100_000, 65536,
    )
    assert should_drain is False
    assert new_baseline == 10_000


def test_post_drain_baseline_prevents_immediate_refire():
    from iai_mcp.daemon import _spool_backlog_over_threshold

    # Probe 1: fires, baseline advances to the firing snapshot's total.
    snapshot1 = {"finalized_files": 0, "live_bytes_total": 165_536}
    should_drain1, baseline_after1 = _spool_backlog_over_threshold(
        snapshot1, 100_000, 65536,
    )
    assert should_drain1 is True
    assert baseline_after1 == 165_536

    # Probe 2: drain made no progress on disk (e.g. skipped/failed), total
    # unchanged -- must not immediately re-fire.
    snapshot2 = {"finalized_files": 0, "live_bytes_total": 165_536}
    should_drain2, baseline_after2 = _spool_backlog_over_threshold(
        snapshot2, baseline_after1, 65536,
    )
    assert should_drain2 is False
    assert baseline_after2 == 165_536


# --- spool_backlog_snapshot --------------------------------------------------


def test_snapshot_counts_finalized_files_and_live_bytes(tmp_path):
    from iai_mcp.capture import spool_backlog_snapshot

    deferred_dir = tmp_path / ".deferred-captures"
    deferred_dir.mkdir()

    (deferred_dir / "sess-a.crash-1.jsonl").write_text("x" * 10)
    (deferred_dir / "sess-b.processing-123.jsonl").write_text("y" * 10)

    live_path = deferred_dir / "sess-c.live.jsonl"
    live_path.write_text("h\n" + ("e" * 1000) + "\n")

    snapshot = spool_backlog_snapshot(deferred_dir)
    assert snapshot["finalized_files"] == 2
    assert snapshot["live_bytes_total"] == live_path.stat().st_size


def test_snapshot_sums_multiple_live_files(tmp_path):
    from iai_mcp.capture import spool_backlog_snapshot

    deferred_dir = tmp_path / ".deferred-captures"
    deferred_dir.mkdir()

    live1 = deferred_dir / "sess-d.live.jsonl"
    live1.write_text("h\n" + ("e" * 100) + "\n")
    live2 = deferred_dir / "sess-e.live.jsonl"
    live2.write_text("h\n" + ("f" * 200) + "\n")

    snapshot = spool_backlog_snapshot(deferred_dir)
    assert snapshot["finalized_files"] == 0
    assert snapshot["live_bytes_total"] == live1.stat().st_size + live2.stat().st_size


def test_snapshot_ignores_capture_state_dir(tmp_path):
    from iai_mcp.capture import spool_backlog_snapshot

    deferred_dir = tmp_path / ".deferred-captures"
    deferred_dir.mkdir()
    live_path = deferred_dir / "sess-f.live.jsonl"
    live_path.write_text("h\n" + ("z" * 200) + "\n")

    # No .drain-offset files anywhere -- snapshot must not need or read them.
    snapshot = spool_backlog_snapshot(deferred_dir)
    assert snapshot["live_bytes_total"] == live_path.stat().st_size


def test_snapshot_empty_dir_is_zeroed(tmp_path):
    from iai_mcp.capture import spool_backlog_snapshot

    deferred_dir = tmp_path / ".deferred-captures"
    deferred_dir.mkdir()

    snapshot = spool_backlog_snapshot(deferred_dir)
    assert snapshot == {"finalized_files": 0, "live_bytes_total": 0}


def test_snapshot_missing_dir_is_zeroed(tmp_path):
    from iai_mcp.capture import spool_backlog_snapshot

    snapshot = spool_backlog_snapshot(tmp_path / "does-not-exist")
    assert snapshot == {"finalized_files": 0, "live_bytes_total": 0}


# --- _run_drowsy_drain event-name generalization -----------------------------


def test_run_drowsy_drain_emits_wake_valve_event_name():
    from iai_mcp.daemon import _run_drowsy_drain

    events: list[tuple] = []

    def write_event(store, kind, data, severity="info"):
        events.append((kind, data, severity))

    def good_drain(store):
        return {"files_drained": 1, "files_failed": 0}

    _run_drowsy_drain(
        object(),
        drain_fn=good_drain,
        write_event_fn=write_event,
        event_name="deferred_drain_wake_valve",
        phase="wake_valve",
    )

    kinds = [e[0] for e in events]
    assert "deferred_drain_wake_valve" in kinds, events


def test_run_drowsy_drain_failure_tags_wake_valve_phase():
    from iai_mcp.daemon import _run_drowsy_drain

    events: list[tuple] = []

    def write_event(store, kind, data, severity="info"):
        events.append((kind, data, severity))

    def failing_drain(store):
        raise RuntimeError("boom")

    _run_drowsy_drain(
        object(),
        drain_fn=failing_drain,
        write_event_fn=write_event,
        event_name="deferred_drain_wake_valve",
        phase="wake_valve",
    )

    failed = next(e for e in events if e[0] == "deferred_drain_failed")
    assert failed[1].get("phase") == "wake_valve", failed
