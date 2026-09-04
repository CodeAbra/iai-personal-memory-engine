from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from iai_mcp.doctor._lifecycle_checks import (
    _BACKGROUND_LIVENESS_ALARM_ROWS,
    check_cc_background_liveness,
)
from iai_mcp.lifecycle_event_log import LifecycleEventLog

_NOW = datetime(2026, 8, 12, 15, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def lifecycle_log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    return tmp_path / "logs"


def _ts(day_offset: int, hour: int = 3) -> datetime:
    day = _NOW.date() - timedelta(days=day_offset)
    return datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)


def _seed_promise_row(
    log: LifecycleEventLog,
    promise: str,
    *,
    day_offset: int,
    candidates: int | None,
    processed: int | None,
    spec_version: int | None = 1,
    error: str | None = None,
) -> None:
    event: dict = {"event": "promise_liveness", "promise": promise}
    event["liveness_candidates"] = candidates
    event["liveness_processed"] = processed
    if spec_version is not None:
        event["liveness_spec_version"] = spec_version
    if error is not None:
        event["error"] = error
    log.append(event, now=_ts(day_offset))


def _seed_step_row(
    log: LifecycleEventLog,
    step: str,
    *,
    day_offset: int,
    candidates: int | None,
    processed: int | None,
    spec_version: int | None = 1,
    valence_saturated: int | None = None,
) -> None:
    event: dict = {
        "event": "sleep_step_completed",
        "step": step,
        "liveness_candidates": candidates,
        "liveness_processed": processed,
    }
    if spec_version is not None:
        event["liveness_spec_version"] = spec_version
    if valence_saturated is not None:
        event["valence_saturated"] = valence_saturated
    log.append(event, now=_ts(day_offset))


class TestNoHistoryYet:
    def test_no_log_dir_is_pass(self, lifecycle_log_dir: Path) -> None:
        result = check_cc_background_liveness(now=_NOW)
        assert result.status == "PASS"
        assert result.passed is True

    def test_empty_log_dir_is_pass(self, lifecycle_log_dir: Path) -> None:
        lifecycle_log_dir.mkdir(parents=True)
        result = check_cc_background_liveness(now=_NOW)
        assert result.status == "PASS"
        assert result.passed is True


class TestFailOnConsecutiveNoOp:
    def test_n_consecutive_zero_effect_rows_fails_and_names_identity(
        self, lifecycle_log_dir: Path,
    ) -> None:
        log = LifecycleEventLog(log_dir=lifecycle_log_dir)
        for offset in range(_BACKGROUND_LIVENESS_ALARM_ROWS):
            _seed_promise_row(
                log, "capture_batch",
                day_offset=offset, candidates=5, processed=0,
            )

        result = check_cc_background_liveness(now=_NOW)
        assert result.status == "FAIL"
        assert result.passed is False
        assert "capture_batch" in result.detail


class TestUnknownCandidatesNeverFail:
    def test_none_candidates_and_processed_yield_pass_never_fail(
        self, lifecycle_log_dir: Path,
    ) -> None:
        # A hypothetical future promise that has not been wired to real int
        # counts yet -- identity_audit itself now always writes real ints
        # (see TestIdentityAuditWiredReality), so this exercises the
        # classifier's generic unreadable-row tolerance, not identity_audit.
        log = LifecycleEventLog(log_dir=lifecycle_log_dir)
        for offset in range(_BACKGROUND_LIVENESS_ALARM_ROWS):
            _seed_promise_row(
                log, "hypothetical_future_promise",
                day_offset=offset, candidates=None, processed=None,
            )

        result = check_cc_background_liveness(now=_NOW)
        assert result.status != "FAIL"
        assert result.passed is True
        assert "hypothetical_future_promise" in result.detail
        assert "no wiring yet" not in result.detail


class TestUnknownCandidatesWithKnownEffect:
    def test_none_candidates_and_zero_processed_yields_pass_never_fail(
        self, lifecycle_log_dir: Path,
    ) -> None:
        log = LifecycleEventLog(log_dir=lifecycle_log_dir)
        for offset in range(_BACKGROUND_LIVENESS_ALARM_ROWS):
            _seed_promise_row(
                log, "hypothetical_future_promise",
                day_offset=offset, candidates=None, processed=0,
            )

        result = check_cc_background_liveness(now=_NOW)
        assert result.status == "PASS"
        assert result.passed is True
        assert "hypothetical_future_promise" in result.detail
        assert "no wiring yet" not in result.detail


class TestStarvedInputWarn:
    def test_n_consecutive_zero_candidate_rows_warns_and_names_identity(
        self, lifecycle_log_dir: Path,
    ) -> None:
        log = LifecycleEventLog(log_dir=lifecycle_log_dir)
        for offset in range(_BACKGROUND_LIVENESS_ALARM_ROWS):
            _seed_step_row(
                log, "CLUSTER_REPLAY",
                day_offset=offset, candidates=0, processed=0,
            )

        result = check_cc_background_liveness(now=_NOW)
        assert result.status == "WARN"
        assert result.passed is True
        assert "CLUSTER_REPLAY" in result.detail


class TestErrorRowExcludedFromCounter:
    def test_error_row_with_stale_noop_looking_values_does_not_count(
        self, lifecycle_log_dir: Path,
    ) -> None:
        log = LifecycleEventLog(log_dir=lifecycle_log_dir)
        # An error row carries stale liveness values that WOULD look like a
        # no-op if wrongly counted; excluding it correctly leaves only 2
        # usable rows in a 3-row window -- not enough to alarm.
        _seed_promise_row(
            log, "warm_bm25_nightly",
            day_offset=3, candidates=7, processed=0, error="rebuild crashed",
        )
        _seed_promise_row(
            log, "warm_bm25_nightly",
            day_offset=2, candidates=7, processed=0,
        )
        _seed_promise_row(
            log, "warm_bm25_nightly",
            day_offset=1, candidates=7, processed=0,
        )

        result = check_cc_background_liveness(now=_NOW)
        assert result.status != "FAIL"
        assert "warm_bm25_nightly" not in _fail_names(result)


class TestMissingSpecVersion:
    def test_missing_liveness_spec_version_is_unknown_not_a_crash(
        self, lifecycle_log_dir: Path,
    ) -> None:
        log = LifecycleEventLog(log_dir=lifecycle_log_dir)
        for offset in range(_BACKGROUND_LIVENESS_ALARM_ROWS):
            _seed_promise_row(
                log, "warm_bm25_nightly",
                day_offset=offset, candidates=1, processed=1, spec_version=None,
            )

        result = check_cc_background_liveness(now=_NOW)
        assert result.status != "FAIL"
        assert result.passed is True
        assert "warm_bm25_nightly" in result.detail


class TestKnownCandidateUnknownEffect:
    def test_candidates_positive_processed_none_is_listed_not_silent_pass(
        self, lifecycle_log_dir: Path,
    ) -> None:
        log = LifecycleEventLog(log_dir=lifecycle_log_dir)
        for offset in range(_BACKGROUND_LIVENESS_ALARM_ROWS):
            _seed_promise_row(
                log, "hypothetical_future_promise",
                day_offset=offset, candidates=5, processed=None,
            )

        result = check_cc_background_liveness(now=_NOW)
        assert result.status != "FAIL"
        assert result.passed is True
        assert "hypothetical_future_promise" in result.detail
        assert "no wiring yet" not in result.detail


class TestMalformedRowDoesNotCrashTheRun:
    def test_non_int_candidate_is_unknown_not_a_typeerror(
        self, lifecycle_log_dir: Path,
    ) -> None:
        log = LifecycleEventLog(log_dir=lifecycle_log_dir)
        # A corrupted or hand-edited row could carry a non-int value; the
        # ordering comparisons in the classifier must not raise on it.
        _seed_promise_row(
            log, "capture_batch",
            day_offset=0, candidates="5", processed=0,  # type: ignore[arg-type]
        )

        result = check_cc_background_liveness(now=_NOW)
        assert result.status != "FAIL"
        assert result.passed is True


class TestAbsentRotatedDayFiles:
    def test_only_one_of_seven_day_files_present_no_false_fail(
        self, lifecycle_log_dir: Path,
    ) -> None:
        log = LifecycleEventLog(log_dir=lifecycle_log_dir)
        _seed_promise_row(
            log, "capture_batch",
            day_offset=0, candidates=5, processed=3,
        )

        result = check_cc_background_liveness(now=_NOW)
        assert result.status != "FAIL"
        assert result.passed is True


class TestHealthyIdentity:
    def test_recent_confirmed_work_is_pass(self, lifecycle_log_dir: Path) -> None:
        log = LifecycleEventLog(log_dir=lifecycle_log_dir)
        _seed_promise_row(
            log, "capture_batch",
            day_offset=0, candidates=5, processed=3,
        )

        result = check_cc_background_liveness(now=_NOW)
        assert result.status == "PASS"
        assert result.passed is True

    def test_single_zero_candidate_row_below_alarm_threshold_is_pass(
        self, lifecycle_log_dir: Path,
    ) -> None:
        log = LifecycleEventLog(log_dir=lifecycle_log_dir)
        _seed_promise_row(
            log, "identity_audit",
            day_offset=0, candidates=0, processed=0,
        )

        result = check_cc_background_liveness(now=_NOW)
        assert result.status == "PASS"
        assert result.passed is True


class TestIdentityAuditWiredReality:
    def test_filled_steady_state_is_pass_not_unknown(
        self, lifecycle_log_dir: Path,
    ) -> None:
        log = LifecycleEventLog(log_dir=lifecycle_log_dir)
        for offset in range(_BACKGROUND_LIVENESS_ALARM_ROWS):
            _seed_promise_row(
                log, "identity_audit",
                day_offset=offset, candidates=5, processed=4,
            )

        result = check_cc_background_liveness(now=_NOW)
        assert result.status == "PASS"
        assert result.passed is True
        assert "no wiring yet" not in result.detail

    def test_starved_stream_is_warn_not_a_permanent_pass(
        self, lifecycle_log_dir: Path,
    ) -> None:
        log = LifecycleEventLog(log_dir=lifecycle_log_dir)
        for offset in range(_BACKGROUND_LIVENESS_ALARM_ROWS):
            _seed_promise_row(
                log, "identity_audit",
                day_offset=offset, candidates=0, processed=0,
            )

        result = check_cc_background_liveness(now=_NOW)
        assert result.status == "WARN"
        assert result.passed is True
        assert "identity_audit" in result.detail
        assert "no wiring yet" not in result.detail


class TestSaturatedOnlyNightsAreHealthy:
    def test_three_saturated_only_nights_is_not_fail(
        self, lifecycle_log_dir: Path,
    ) -> None:
        log = LifecycleEventLog(log_dir=lifecycle_log_dir)
        for offset in range(_BACKGROUND_LIVENESS_ALARM_ROWS):
            _seed_step_row(
                log, "RECONSOLIDATION_VALENCE",
                day_offset=offset, candidates=5, processed=0,
                valence_saturated=5,
            )

        result = check_cc_background_liveness(now=_NOW)
        assert result.status != "FAIL"
        assert result.passed is True
        assert "RECONSOLIDATION_VALENCE" not in _fail_names(result)


class TestGenuineStallStillFails:
    def test_partial_saturation_with_unexplained_remainder_still_fails(
        self, lifecycle_log_dir: Path,
    ) -> None:
        log = LifecycleEventLog(log_dir=lifecycle_log_dir)
        for offset in range(_BACKGROUND_LIVENESS_ALARM_ROWS):
            _seed_step_row(
                log, "RECONSOLIDATION_VALENCE",
                day_offset=offset, candidates=5, processed=0,
                valence_saturated=2,
            )

        result = check_cc_background_liveness(now=_NOW)
        assert result.status == "FAIL"
        assert "RECONSOLIDATION_VALENCE" in result.detail

    def test_no_saturation_signal_at_all_still_fails(
        self, lifecycle_log_dir: Path,
    ) -> None:
        log = LifecycleEventLog(log_dir=lifecycle_log_dir)
        for offset in range(_BACKGROUND_LIVENESS_ALARM_ROWS):
            _seed_step_row(
                log, "RECONSOLIDATION_VALENCE",
                day_offset=offset, candidates=5, processed=0,
            )

        result = check_cc_background_liveness(now=_NOW)
        assert result.status == "FAIL"
        assert "RECONSOLIDATION_VALENCE" in result.detail


class TestRegistration:
    def test_registered_in_run_diagnosis(self) -> None:
        import inspect

        from iai_mcp.doctor import run_diagnosis

        source = inspect.getsource(run_diagnosis)
        assert "check_cc_background_liveness()" in source

    def test_imported_at_package_level(self) -> None:
        from iai_mcp.doctor import check_cc_background_liveness as pkg_check

        assert pkg_check is check_cc_background_liveness

    def test_no_store_argument_accepted(self) -> None:
        import inspect

        sig = inspect.signature(check_cc_background_liveness)
        assert "store" not in sig.parameters


def _fail_names(result) -> list[str]:
    if result.status != "FAIL":
        return []
    return [part.strip() for part in result.detail.split(":", 1)[-1].split(",")]
