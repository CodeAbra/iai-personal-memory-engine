"""Tri-state measurable-session accounting.

An unavailable session is a distinct third state that never folds into
pass or fail, so any measurement built on top only reports a rate over
sessions that were actually measurable.

Pure stdlib, no heavy runtime imports (numba / embedder / store) --
independently importable and unit-testable on synthetic input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

OUTCOME_PASSED = "PASSED"
OUTCOME_FAILED = "FAILED"
OUTCOME_EXCLUDED_UNAVAILABLE = "EXCLUDED_UNAVAILABLE"

_UNAVAILABLE_SOURCE_VALUES = {"unavailable"}


def availability_from_source(source: "str | None") -> bool:
    """Canonical `_source`/reachability signal -> availability mapping.

    A degraded-but-responding source (e.g. cortex fallback, cold
    structural) is still available -- degraded is not unavailable. The
    absence of any signal maps to True (available); only a signal in the
    unavailable set maps to False.
    """
    if source is None:
        return True
    return source.strip().lower() not in _UNAVAILABLE_SOURCE_VALUES


def classify_session_outcome(available: bool, passed: "bool | None") -> str:
    """`available is False` returns EXCLUDED_UNAVAILABLE regardless of
    `passed` -- an unavailable session is never counted as a pass or a
    fail. `available is True` requires a real `passed` bool: `None` means
    the caller never computed an outcome, which must not silently count
    as a measured FAIL -- raises ValueError to catch that caller bug."""
    if not available:
        return OUTCOME_EXCLUDED_UNAVAILABLE
    if passed is None:
        raise ValueError(
            "classify_session_outcome: available=True requires a bool passed, not None"
        )
    return OUTCOME_PASSED if passed else OUTCOME_FAILED


@dataclass(frozen=True)
class MeasurableSummary:
    measured: int
    passed: int
    failed: int
    excluded_unavailable: int


def summarize_measurable(outcomes: Iterable[str]) -> MeasurableSummary:
    """`measured` (the denominator) is passed + failed only -- excluded
    sessions are never folded in, so M=0 when every session is
    unavailable."""
    passed = 0
    failed = 0
    excluded = 0
    for outcome in outcomes:
        if outcome == OUTCOME_PASSED:
            passed += 1
        elif outcome == OUTCOME_FAILED:
            failed += 1
        elif outcome == OUTCOME_EXCLUDED_UNAVAILABLE:
            excluded += 1
    return MeasurableSummary(
        measured=passed + failed,
        passed=passed,
        failed=failed,
        excluded_unavailable=excluded,
    )


def format_measurable_summary(summary: MeasurableSummary) -> str:
    """Renders "N/M measurable passed, K excluded: unavailable". Uses no
    division, so the all-unavailable M=0 case ("0/0 measurable passed, ...")
    is safe and reads as zero measured, never as success."""
    return (
        f"{summary.passed}/{summary.measured} measurable passed, "
        f"{summary.excluded_unavailable} excluded: unavailable"
    )
