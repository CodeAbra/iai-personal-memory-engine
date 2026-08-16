from __future__ import annotations

import argparse
import inspect
import json
import platform
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX paths + atomic rename semantics",
)


def _make_transcript_line(role: str, text: str) -> str:
    return json.dumps({"type": role, "message": {"role": role, "content": text}}) + "\n"


def _build_args(session_id: str, transcript_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        session_id=session_id,
        transcript_path=str(transcript_path),
        max_turns_per_call=200,
    )


def _isolate_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    return tmp_path / ".iai-mcp"


def test_real_turn_drives_m1_through_m6(tmp_path, monkeypatch):
    store_root = _isolate_store(tmp_path, monkeypatch)
    from iai_mcp.cli import cmd_capture_turn_deferred
    from iai_mcp.events import query_events
    from iai_mcp.store import MemoryStore

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(_make_transcript_line("user", "hello world enough chars"))

    rc = cmd_capture_turn_deferred(_build_args("stop-hook-1", transcript))
    assert rc == 0

    store = MemoryStore(path=store_root)
    events = query_events(store, kind="trajectory_metric", limit=100)
    by_metric_session = {
        (e["data"].get("metric"), e.get("session_id")) for e in events
    }
    for m in ("m1", "m3", "m5"):
        assert (m, "stop-hook-1") in by_metric_session, (
            f"{m} must be recorded under the real session_id; got {by_metric_session}"
        )
    for m in ("m2", "m4", "m6"):
        assert (m, "-") in by_metric_session, (
            f"{m} must be recorded under the sentinel session_id; got {by_metric_session}"
        )


def test_second_identical_call_does_not_duplicate_rows(tmp_path, monkeypatch):
    store_root = _isolate_store(tmp_path, monkeypatch)
    from iai_mcp.cli import cmd_capture_turn_deferred
    from iai_mcp.events import query_events
    from iai_mcp.store import MemoryStore

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(_make_transcript_line("user", "hello world enough chars"))

    rc1 = cmd_capture_turn_deferred(_build_args("stop-hook-2", transcript))
    assert rc1 == 0
    store = MemoryStore(path=store_root)
    first_count = len(query_events(store, kind="trajectory_metric", limit=200))
    assert first_count > 0

    rc2 = cmd_capture_turn_deferred(_build_args("stop-hook-2", transcript))
    assert rc2 == 0
    second_count = len(query_events(store, kind="trajectory_metric", limit=200))
    assert second_count == first_count, (
        f"unchanged inputs must not duplicate rows: {first_count} -> {second_count}"
    )


def test_driver_exception_is_swallowed_and_turn_capture_still_succeeds(tmp_path, monkeypatch):
    _isolate_store(tmp_path, monkeypatch)
    from iai_mcp.cli import _capture as capture_mod

    def _raise(*a, **kw):
        raise RuntimeError("simulated trajectory driver failure")

    monkeypatch.setattr(capture_mod, "_drive_trajectory_metrics", _raise)

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(_make_transcript_line("user", "hello world enough chars"))

    rc = capture_mod.cmd_capture_turn_deferred(_build_args("stop-hook-3", transcript))
    assert rc == 0

    live = tmp_path / ".iai-mcp" / ".deferred-captures" / "stop-hook-3.live.jsonl"
    assert live.exists(), "turn capture must still succeed when the trajectory driver raises"


def test_dedup_survives_a_flooded_trajectory_metric_stream(tmp_path, monkeypatch):
    # An unrelated writer pushing hundreds of trajectory_metric rows under
    # OTHER sessions must never evict this session's own dedup key from a
    # fixed-size lookback window -- the same horizon-eviction failure mode
    # the detector fix (plan 01) addresses at the metric-read layer.
    store_root = _isolate_store(tmp_path, monkeypatch)
    from iai_mcp.cli import cmd_capture_turn_deferred
    from iai_mcp.events import query_events, write_event
    from iai_mcp.store import MemoryStore

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(_make_transcript_line("user", "hello world enough chars"))

    rc1 = cmd_capture_turn_deferred(_build_args("stop-hook-horizon", transcript))
    assert rc1 == 0

    store = MemoryStore(path=store_root)
    first_count = len(query_events(store, kind="trajectory_metric", limit=2000))
    assert first_count > 0

    for i in range(600):
        write_event(
            store, kind="trajectory_metric",
            data={"metric": "m1", "value": float(i)},
            severity="info", session_id=f"noise-session-{i}",
        )
    noisy_count = len(query_events(store, kind="trajectory_metric", limit=2000))

    rc2 = cmd_capture_turn_deferred(_build_args("stop-hook-horizon", transcript))
    assert rc2 == 0
    second_count = len(query_events(store, kind="trajectory_metric", limit=2000))
    assert second_count == noisy_count, (
        "unchanged inputs must not duplicate rows even after 600 unrelated "
        f"rows land under other sessions: {noisy_count} -> {second_count}"
    )


@pytest.mark.perf
def test_driver_stays_well_under_the_stop_hook_timeout_at_scale(tmp_path, monkeypatch):
    # The Stop hook's `capture-turn-deferred` invocation carries a 30s
    # wall-clock timeout (iai-mcp-session-capture.sh); this benches the
    # trajectory driver alone at a corpus scale well beyond a single-user
    # store, so a regression here fails loudly instead of shipping unnoticed.
    # Seed count is bounded by per-event write cost on the slower storage
    # driver -- 3k rows already exercises the index-bounded query path the
    # driver relies on (see _trajectory_last_recorded); the per-call cost
    # this asserts on does not grow with corpus size beyond that point.
    import time

    store_root = _isolate_store(tmp_path, monkeypatch)
    from iai_mcp.cli._capture import _drive_trajectory_metrics
    from iai_mcp.events import write_event
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=store_root)
    kinds = (
        "curiosity_question", "session_started", "curiosity_silent_log",
        "retrieval_used", "profile_tuned", "trajectory_metric",
    )
    for i in range(3_000):
        kind = kinds[i % len(kinds)]
        data: dict = {"idx": i}
        if kind == "session_started":
            data["total_cached_tokens"] = 1000
            data["session_state_hash"] = f"h{i % 100}"
        write_event(store, kind=kind, data=data, severity="info", session_id=f"s{i % 50}")

    state_dir = tmp_path / ".iai-mcp" / ".capture-state"
    state_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    _drive_trajectory_metrics("perf-session", state_dir)
    elapsed = time.perf_counter() - t0
    print(f"\n_drive_trajectory_metrics @3k events: {elapsed * 1000:.1f} ms")
    assert elapsed < 5.0, (
        f"trajectory driver took {elapsed:.2f}s at 3k events, "
        "eating into the 30s Stop-hook timeout"
    )


def test_no_consolidation_reference_in_the_added_driver_code():
    from iai_mcp.cli._capture import _drive_trajectory_metrics, cmd_capture_turn_deferred

    combined_source = inspect.getsource(_drive_trajectory_metrics) + inspect.getsource(
        cmd_capture_turn_deferred
    )
    for forbidden in ("run_light_consolidation", "run_heavy_consolidation", "iai_mcp.sleep"):
        assert forbidden not in combined_source, (
            f"trajectory driver must never reference {forbidden}"
        )
