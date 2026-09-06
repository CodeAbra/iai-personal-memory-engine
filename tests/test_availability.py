"""Unit tests for the tri-state measurable-session accounting helper.

Mirrors the pass/underpowered/non-pass pattern in
bench/fhrr_verdict_recall_delta.py: an unavailable session must never
collapse into pass or fail, and an all-unavailable run must stay safe
and non-success-looking.
"""
from __future__ import annotations

import pytest

from iai_mcp.availability import (
    OUTCOME_EXCLUDED_UNAVAILABLE,
    OUTCOME_FAILED,
    OUTCOME_PASSED,
    MeasurableSummary,
    availability_from_source,
    classify_session_outcome,
    format_measurable_summary,
    summarize_measurable,
)


def test_unavailable_session_always_excluded_regardless_of_passed():
    assert classify_session_outcome(available=False, passed=True) == OUTCOME_EXCLUDED_UNAVAILABLE
    assert classify_session_outcome(available=False, passed=False) == OUTCOME_EXCLUDED_UNAVAILABLE
    assert classify_session_outcome(available=False, passed=None) == OUTCOME_EXCLUDED_UNAVAILABLE


def test_available_session_classifies_pass_or_fail():
    assert classify_session_outcome(available=True, passed=True) == OUTCOME_PASSED
    assert classify_session_outcome(available=True, passed=False) == OUTCOME_FAILED


def test_available_session_with_unknown_outcome_raises_instead_of_silent_fail():
    # available=True with passed=None means the caller never computed an
    # outcome -- this must never silently count as a measured FAIL, which
    # would distort the measurable-session accounting.
    with pytest.raises(ValueError):
        classify_session_outcome(available=True, passed=None)


def test_availability_from_source_marks_only_unavailable_signal_as_unavailable():
    assert availability_from_source(None) is True
    assert availability_from_source("unavailable") is False
    assert availability_from_source("UNAVAILABLE") is False
    # A degraded-but-responding source is still available -- degraded is
    # not the same state as unavailable.
    assert availability_from_source("cold-structural-degrade") is True
    assert availability_from_source("cortex-fallback") is True


def test_summarize_measurable_excludes_unavailable_from_denominator():
    outcomes = [
        OUTCOME_PASSED,
        OUTCOME_PASSED,
        OUTCOME_FAILED,
        OUTCOME_EXCLUDED_UNAVAILABLE,
    ]
    summary = summarize_measurable(outcomes)
    assert summary.passed == 2
    assert summary.failed == 1
    assert summary.excluded_unavailable == 1
    assert summary.measured == 3, "excluded must not be folded into the measured denominator"


def test_all_unavailable_summary_is_zero_measured_and_safe():
    outcomes = [OUTCOME_EXCLUDED_UNAVAILABLE, OUTCOME_EXCLUDED_UNAVAILABLE]
    summary = summarize_measurable(outcomes)
    assert summary.measured == 0
    assert summary.passed == 0
    assert summary.failed == 0
    assert summary.excluded_unavailable == 2

    rendered = format_measurable_summary(summary)  # must not raise ZeroDivisionError
    assert "0/0" in rendered
    assert "100%" not in rendered
    assert "excluded: unavailable" in rendered


def test_format_measurable_summary_exact_shape():
    summary = MeasurableSummary(measured=3, passed=2, failed=1, excluded_unavailable=1)
    assert format_measurable_summary(summary) == "2/3 measurable passed, 1 excluded: unavailable"


def test_measurable_summary_is_frozen():
    summary = MeasurableSummary(measured=1, passed=1, failed=0, excluded_unavailable=0)
    try:
        summary.measured = 5  # type: ignore[misc]
        raised = False
    except Exception:
        raised = True
    assert raised, "MeasurableSummary must be immutable"
