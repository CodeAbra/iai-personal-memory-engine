from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pytest

from iai_mcp.lifecycle_event_log import LifecycleEventLog, _LIVENESS_SPEC_VERSION


def _liveness_rows(event_log: LifecycleEventLog, promise: str) -> list[dict]:
    return [
        r for r in event_log.read_all()
        if r.get("event") == "promise_liveness" and r.get("promise") == promise
    ]


class TestKnownEventKind:
    def test_promise_liveness_is_a_known_kind(self) -> None:
        from iai_mcp.lifecycle_event_log import KNOWN_EVENT_KINDS

        assert "promise_liveness" in KNOWN_EVENT_KINDS


class TestIdentityAuditLivenessRow:
    def test_starved_stream_yields_zero_candidates_and_zero_processed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from iai_mcp import identity_audit

        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))

        identity_audit._stamp_identity_audit_liveness([], 0, 0)

        event_log = LifecycleEventLog(log_dir=tmp_path / "logs")
        rows = _liveness_rows(event_log, "identity_audit")
        assert len(rows) == 1
        row = rows[0]
        assert row["liveness_candidates"] == 0
        assert row["liveness_processed"] == 0
        assert row["liveness_spec_version"] == _LIVENESS_SPEC_VERSION
        assert row.get("ts")

    def test_filled_window_pairs_real_candidates_and_processed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from iai_mcp import identity_audit

        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))

        identity_audit._stamp_identity_audit_liveness(
            [{"kind": "s5_drift_alert"}], 5, 4,
        )

        event_log = LifecycleEventLog(log_dir=tmp_path / "logs")
        rows = _liveness_rows(event_log, "identity_audit")
        assert len(rows) == 1
        assert rows[0]["liveness_candidates"] == 5
        assert rows[0]["liveness_processed"] == 4


class TestIdentityAuditLivenessAgainstRealDetector:
    """Drives the real detect_drift_anomaly + the doctor's own classifier,
    so a passing unit test here cannot diverge from the live doctor verdict.
    """

    def test_starved_stream_classifies_as_warn(self, tmp_path: Path) -> None:
        from iai_mcp.doctor._lifecycle_checks import _classify_liveness_window
        from iai_mcp.s5 import detect_drift_anomaly
        from iai_mcp.store import MemoryStore

        store = MemoryStore(path=tmp_path)
        rows = []
        for _ in range(3):
            alerts, candidates, processed = detect_drift_anomaly(store, cycles=5)
            assert alerts == []
            rows.append({
                "liveness_candidates": candidates,
                "liveness_processed": processed,
                "liveness_spec_version": _LIVENESS_SPEC_VERSION,
            })
        assert _classify_liveness_window(rows, 3) == "warn"

    def test_filled_window_no_alert_classifies_as_pass(self, tmp_path: Path) -> None:
        from iai_mcp.doctor._lifecycle_checks import _classify_liveness_window
        from iai_mcp.events import write_event
        from iai_mcp.s5 import detect_drift_anomaly
        from iai_mcp.store import MemoryStore

        store = MemoryStore(path=tmp_path)
        for v in (3, 3, 3, 3, 3):
            write_event(store, kind="profile_tuned", data={"moved_count": v}, severity="info")

        rows = []
        for _ in range(3):
            alerts, candidates, processed = detect_drift_anomaly(store, cycles=5)
            assert alerts == []
            rows.append({
                "liveness_candidates": candidates,
                "liveness_processed": processed,
                "liveness_spec_version": _LIVENESS_SPEC_VERSION,
            })
        verdict = _classify_liveness_window(rows, 3)
        assert verdict not in ("fail", "warn")

    def test_filled_window_with_alert_classifies_as_pass(self, tmp_path: Path) -> None:
        from iai_mcp.doctor._lifecycle_checks import _classify_liveness_window
        from iai_mcp.events import write_event
        from iai_mcp.s5 import detect_drift_anomaly
        from iai_mcp.store import MemoryStore

        store = MemoryStore(path=tmp_path)
        for v in (1, 2, 3, 4, 5):
            write_event(store, kind="profile_tuned", data={"moved_count": v}, severity="info")

        rows = []
        for _ in range(3):
            alerts, candidates, processed = detect_drift_anomaly(store, cycles=5)
            assert len(alerts) == 1
            rows.append({
                "liveness_candidates": candidates,
                "liveness_processed": processed,
                "liveness_spec_version": _LIVENESS_SPEC_VERSION,
            })
        verdict = _classify_liveness_window(rows, 3)
        assert verdict not in ("fail", "warn")

    def test_no_reachable_return_yields_the_false_fail_shape(self, tmp_path: Path) -> None:
        # candidates > 0 with processed == 0 would misclassify a healthy
        # steady state as "fail" in _classify_liveness_window; the pairing
        # in s5.detect_drift_anomaly at the audit's real cycles=5 call site
        # must never produce that shape.
        from iai_mcp.events import write_event
        from iai_mcp.s5 import detect_drift_anomaly
        from iai_mcp.store import MemoryStore

        store = MemoryStore(path=tmp_path)
        for n in range(12):
            write_event(
                store, kind="profile_tuned",
                data={"moved_count": float(n % 4)}, severity="info",
            )
            _alerts, candidates, processed = detect_drift_anomaly(store, cycles=5)
            assert not (candidates > 0 and processed == 0), (n, candidates, processed)


class TestIdentityAuditLivePathWiring:
    def test_continuous_audit_stamps_liveness_on_its_hourly_loop(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from iai_mcp import identity_audit

        calls: list[tuple] = []

        monkeypatch.setattr(
            identity_audit, "detect_drift_anomaly",
            lambda store, n: (["alert"], 5, 4),
        )
        monkeypatch.setattr(identity_audit, "compute_and_emit", lambda store: None)
        monkeypatch.setattr(
            identity_audit, "_stamp_identity_audit_liveness",
            lambda alerts, candidates, processed: calls.append((alerts, candidates, processed)),
        )

        async def run() -> None:
            shutdown = asyncio.Event()
            task = asyncio.create_task(
                identity_audit.continuous_audit(
                    store=object(), shutdown=shutdown, interval_sec=100,
                )
            )
            await asyncio.sleep(0.05)
            shutdown.set()
            await asyncio.wait_for(task, timeout=5)

        asyncio.run(run())
        assert calls == [(["alert"], 5, 4)]

    def test_continuous_audit_skips_the_stamp_when_drift_detection_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from iai_mcp import identity_audit

        calls: list[tuple] = []

        def raising_drift(store, n):
            raise RuntimeError("s5 unavailable")

        monkeypatch.setattr(identity_audit, "detect_drift_anomaly", raising_drift)
        monkeypatch.setattr(identity_audit, "compute_and_emit", lambda store: None)
        monkeypatch.setattr(identity_audit, "write_event", lambda *a, **kw: None)
        monkeypatch.setattr(
            identity_audit, "_stamp_identity_audit_liveness",
            lambda alerts, candidates, processed: calls.append((alerts, candidates, processed)),
        )

        async def run() -> None:
            shutdown = asyncio.Event()
            task = asyncio.create_task(
                identity_audit.continuous_audit(
                    store=object(), shutdown=shutdown, interval_sec=100,
                )
            )
            await asyncio.sleep(0.05)
            shutdown.set()
            await asyncio.wait_for(task, timeout=5)

        asyncio.run(run())
        assert calls == []


class _FakeLexicalStore:
    def __init__(self, *, raise_on_search: bool = False) -> None:
        self._raise_on_search = raise_on_search

    def lexical_search(self, query: str, k: int = 1) -> list:
        if self._raise_on_search:
            raise RuntimeError("lexical index unavailable")
        return []


class TestWarmBM25NightlyLiveness:
    def test_successful_warmup_yields_one_of_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import iai_mcp.runtime_graph_cache as rgc
        from iai_mcp.lilli.cycle.sleep_pipeline import _recall_index

        monkeypatch.setattr(rgc, "_rebuild_and_save_rgc", lambda store: {})

        event_log = LifecycleEventLog(log_dir=tmp_path)

        class FakeSelf:
            _store = _FakeLexicalStore()
            _event_log = event_log

            def _check_interrupt(self, step, chunk_idx, interrupt_check) -> bool:
                return False

        done, result = _recall_index.step_recall_index_rebuild(FakeSelf(), lambda: True)

        assert done is True
        assert result.get("lexical_index_warm") is True
        rows = _liveness_rows(event_log, "warm_bm25_nightly")
        assert len(rows) == 1
        assert rows[0]["liveness_candidates"] == 1
        assert rows[0]["liveness_processed"] == 1
        assert rows[0]["liveness_spec_version"] == _LIVENESS_SPEC_VERSION

    def test_failed_warmup_yields_one_of_zero_not_a_silent_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import iai_mcp.runtime_graph_cache as rgc
        from iai_mcp.lilli.cycle.sleep_pipeline import _recall_index

        monkeypatch.setattr(rgc, "_rebuild_and_save_rgc", lambda store: {})

        event_log = LifecycleEventLog(log_dir=tmp_path)

        class FakeSelf:
            _store = _FakeLexicalStore(raise_on_search=True)
            _event_log = event_log

            def _check_interrupt(self, step, chunk_idx, interrupt_check) -> bool:
                return False

        done, result = _recall_index.step_recall_index_rebuild(FakeSelf(), lambda: True)

        assert done is True
        assert result.get("lexical_index_warm") is None
        rows = _liveness_rows(event_log, "warm_bm25_nightly")
        assert len(rows) == 1
        assert rows[0]["liveness_candidates"] == 1
        assert rows[0]["liveness_processed"] == 0


class TestCaptureBatchLiveness:
    def test_stamp_helper_maps_turns_and_records(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from iai_mcp.cli._capture import _stamp_capture_batch_liveness

        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
        _stamp_capture_batch_liveness(
            {"inserted": 2, "reinforced": 1, "skipped": 1, "errors": 0},
        )

        event_log = LifecycleEventLog(log_dir=tmp_path / "logs")
        rows = _liveness_rows(event_log, "capture_batch")
        assert len(rows) == 1
        assert rows[0]["liveness_candidates"] == 4
        assert rows[0]["liveness_processed"] == 3

    def test_empty_transcript_is_the_dead_case_not_a_none_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from iai_mcp.cli._capture import _stamp_capture_batch_liveness

        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
        _stamp_capture_batch_liveness(
            {"inserted": 0, "reinforced": 0, "skipped": 0, "errors": 0},
        )

        event_log = LifecycleEventLog(log_dir=tmp_path / "logs")
        rows = _liveness_rows(event_log, "capture_batch")
        assert len(rows) == 1
        assert rows[0]["liveness_candidates"] == 0
        assert rows[0]["liveness_processed"] == 0

    def test_missing_transcript_is_skipped_not_fabricated_as_a_candidate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The real capture_transcript return shape for a not-found path
        # (iai_mcp/capture.py) -- errors=1 with a "reason" key, not a
        # zero-processed batch over one genuine candidate.
        from iai_mcp.cli._capture import _stamp_capture_batch_liveness

        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
        _stamp_capture_batch_liveness(
            {"inserted": 0, "reinforced": 0, "skipped": 0, "errors": 1,
             "reason": "transcript not found: /nonexistent/path.jsonl"},
        )

        event_log = LifecycleEventLog(log_dir=tmp_path / "logs")
        assert _liveness_rows(event_log, "capture_batch") == []

    def test_real_capture_transcript_not_found_return_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Drives the actual capture_transcript function (no fakes) against a
        # nonexistent path, proving the skip matches its real return shape.
        from iai_mcp.capture import capture_transcript
        from iai_mcp.cli._capture import _stamp_capture_batch_liveness

        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
        counts = capture_transcript(object(), tmp_path / "does-not-exist.jsonl")
        assert counts.get("reason")

        _stamp_capture_batch_liveness(counts)

        event_log = LifecycleEventLog(log_dir=tmp_path / "logs")
        assert _liveness_rows(event_log, "capture_batch") == []

    def test_cmd_capture_transcript_stamps_on_its_live_default_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
    ) -> None:
        import iai_mcp.capture as capture_mod
        import iai_mcp.store as store_mod
        from iai_mcp.cli._capture import cmd_capture_transcript

        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))

        class FakeStore:
            pass

        def fake_capture_transcript(store, transcript_path, *, session_id, max_turns):
            return {"inserted": 3, "reinforced": 0, "skipped": 2, "errors": 1}

        monkeypatch.setattr(capture_mod, "capture_transcript", fake_capture_transcript)
        monkeypatch.setattr(store_mod, "MemoryStore", lambda *a, **kw: FakeStore())

        args = argparse.Namespace(
            transcript_path=str(tmp_path / "transcript.jsonl"),
            session_id="session-1",
            max_turns=100,
            no_spawn=False,
        )
        rc = cmd_capture_transcript(args)
        assert rc == 0

        event_log = LifecycleEventLog(log_dir=tmp_path / "logs")
        rows = _liveness_rows(event_log, "capture_batch")
        assert len(rows) == 1
        assert rows[0]["liveness_candidates"] == 5
        assert rows[0]["liveness_processed"] == 3

    def test_per_turn_deferred_hot_path_is_never_stamped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import inspect

        from iai_mcp.cli._capture import cmd_capture_turn_deferred

        source = inspect.getsource(cmd_capture_turn_deferred)
        assert "promise_liveness" not in source
        assert "_stamp_capture_batch_liveness" not in source
