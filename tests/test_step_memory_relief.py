"""Hermetic tests for the sleep-pipeline memory-relief postlude helper.

The helper runs a garbage collection and reports the RSS delta with the
current-RSS primitive (NOT peak RSS).  The telemetry dict includes both a
clamped ``rss_delta_mb`` (for dashboards) and an unclamped
``rss_signed_delta_mb`` (negative = RSS grew during relief, visible in
telemetry rather than masked to zero).  ``zone_reclaimed_mb`` is always 0.0
and is kept for schema stability.  The helper must never raise on any platform.
"""
from __future__ import annotations

import numpy as np
import pytest

from iai_mcp.lilli.cycle.sleep_pipeline._memory_relief import _step_memory_relief

_TELEMETRY_KEYS = (
    "rss_before_mb",
    "rss_after_mb",
    "rss_delta_mb",
    "rss_signed_delta_mb",
    "zone_reclaimed_mb",
    "elapsed_ms",
)


def test_relief_after_allocate_and_free_does_not_raise_and_reports_nonnegative_delta() -> None:
    # Allocate then free a sizeable transient so gc.collect() has cycles to
    # reclaim.  ~200 MB float64 array: 25_000_000 * 8 bytes.
    transient = np.ones(25_000_000, dtype=np.float64)
    # Touch it so the pages are resident, not just reserved.
    transient[::4096] = 2.0
    assert float(transient.sum()) > 0.0
    del transient

    result = _step_memory_relief(label="test")

    assert isinstance(result, dict)
    # psutil RSS-delta is flaky — assert >= 0 for the clamped field only;
    # the signed field may be negative if concurrent allocations raced in.
    assert result["rss_delta_mb"] >= 0.0
    # zone_reclaimed_mb is always 0.0 (mimalloc arenas are not reachable via
    # malloc_zone_pressure_relief; mi_collect not exported by bundled build).
    assert result["zone_reclaimed_mb"] == 0.0


def test_relief_returns_six_key_numeric_telemetry_dict() -> None:
    result = _step_memory_relief()

    assert isinstance(result, dict)
    for key in _TELEMETRY_KEYS:
        assert key in result, f"missing telemetry key: {key}"
        assert isinstance(result[key], (int, float)), (
            f"telemetry key {key} not numeric: {type(result[key])}"
        )
    # rss_delta_mb is clamped at 0 — never negative in this field.
    assert result["rss_delta_mb"] >= 0.0
    # rss_signed_delta_mb is unclamped.  When RSS did not grow it equals
    # rss_delta_mb (positive reclaim) or 0.0; when RSS grew it is negative.
    assert result["rss_signed_delta_mb"] <= result["rss_before_mb"] - result["rss_after_mb"] + 0.001
    # zone_reclaimed_mb is schema-stable placeholder.
    assert result["zone_reclaimed_mb"] == 0.0


def test_relief_runs_clean_on_any_platform() -> None:
    # No platform-specific branch exists any more — verify no exception raised
    # and all six keys are present regardless of platform.
    result = _step_memory_relief(label="any_platform")

    assert isinstance(result, dict)
    for key in _TELEMETRY_KEYS:
        assert key in result
    assert result["zone_reclaimed_mb"] == 0.0
    assert result["rss_delta_mb"] >= 0.0


def test_signed_delta_is_negative_when_rss_grows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When RSS grows during relief the signed delta is negative — not clamped to
    zero — so a regression is visible in telemetry rather than silently masked.

    Simulates a scenario where RSS increases from 500 MB to 600 MB during the
    relief window (e.g., a concurrent allocation).  Without the signed field,
    ``rss_delta_mb`` would clamp this to 0.0 and the regression would be
    invisible.  With ``rss_signed_delta_mb`` the true delta (−100 MB) is
    emitted alongside the dashboard-safe clamped value.
    """
    import iai_mcp.lilli.cycle.sleep_pipeline._memory_relief as _mod

    # Provide two successive RSS samples: rss_before=500 MB, rss_after=600 MB.
    rss_readings = iter([500_000_000, 600_000_000])
    monkeypatch.setattr(_mod, "_current_rss_bytes", lambda: next(rss_readings))

    result = _step_memory_relief(label="rss_grew")

    # Clamped field stays at 0 — safe for dashboards.
    assert result["rss_delta_mb"] == 0.0, (
        "rss_delta_mb must clamp to 0 when RSS grew (was "
        f"{result['rss_delta_mb']})"
    )
    # Signed field must be negative, showing the actual regression.
    assert result["rss_signed_delta_mb"] < 0.0, (
        "rss_signed_delta_mb must be negative when RSS grew "
        f"(got {result['rss_signed_delta_mb']})"
    )
    # Exact value: 500 MB - 600 MB = -100 MB.
    assert result["rss_signed_delta_mb"] == pytest.approx(-100.0, abs=0.01)
