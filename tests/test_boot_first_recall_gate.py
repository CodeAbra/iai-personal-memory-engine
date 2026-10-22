"""Unit + production-wired coverage for the phys_footprint harness's
bounded-marginal-cost gate.

Layer 1 (function-level) supplies EXPLICIT numeric ceilings and synthetic
three-scale result dicts -- green for the executor, no sweep required.

Layer 2 (production-wired) calls the gate with the module's OWN
`_MARGINAL_CEILINGS` against an ACTUAL sweep summary.json (a committed
default path, overridable via the IAI_MCP_FP_SUMMARY env var; never invented
in-test). If the committed default fixture is ever removed and no override
is set, this reports SKIPPED rather than raising.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.rss_rca.phys_footprint_harness import (
    _MARGINAL_CEILINGS,
    assert_bounded_marginal_cost,
)

_MIB = 1024.0 ** 2

_EXPLICIT_CEILINGS: dict[str, float | None] = {
    "first_recall_delta_mib_per_1k": 5.0,
    "first_recall_peak_mib_per_1k": 10.0,
    "boot_sequence_delta_mib_per_1k": 10.0,
    "first_recall_latency_ms": 500.0,
}


def _scale(
    n: int, first_recall_delta_mib: float, first_recall_peak_mib: float,
    boot_seq_delta_mib: float, wall_ms: float,
) -> dict:
    return {
        "ok": True,
        "n": n,
        "first_recall_delta_bytes": int(first_recall_delta_mib * _MIB),
        "first_recall_peak_total_rss": int(first_recall_peak_mib * _MIB),
        "boot_sequence_delta_bytes": int(boot_seq_delta_mib * _MIB),
        "first_recall_wall_ms": wall_ms,
    }


def _linear_scales(slope_mib_per_1k: float = 2.0) -> list[dict]:
    """3 scales, all metrics growing LINEARLY at ``slope_mib_per_1k``.

    Comfortably under ``_EXPLICIT_CEILINGS`` at the default slope.
    """
    return [
        _scale(
            3000, 10 + slope_mib_per_1k * 3, 20 + slope_mib_per_1k * 3,
            15 + slope_mib_per_1k * 3, 100.0,
        ),
        _scale(
            8000, 10 + slope_mib_per_1k * 8, 20 + slope_mib_per_1k * 8,
            15 + slope_mib_per_1k * 8, 150.0,
        ),
        _scale(
            15000, 10 + slope_mib_per_1k * 15, 20 + slope_mib_per_1k * 15,
            15 + slope_mib_per_1k * 15, 200.0,
        ),
    ]


# ---------------------------------------------------------------------------
# Layer 1 -- function-level, explicit ceilings.
# ---------------------------------------------------------------------------


def test_under_ceiling_passes() -> None:
    assert_bounded_marginal_cost(_linear_scales(slope_mib_per_1k=2.0), _EXPLICIT_CEILINGS)


def test_over_ceiling_on_first_recall_delta_fails_independently() -> None:
    scales = _linear_scales(slope_mib_per_1k=2.0)
    scales[-1]["first_recall_delta_bytes"] = int(500 * _MIB)
    with pytest.raises(AssertionError, match="first_recall_delta_bytes"):
        assert_bounded_marginal_cost(scales, _EXPLICIT_CEILINGS)


def test_over_ceiling_on_first_recall_peak_fails_independently() -> None:
    scales = _linear_scales(slope_mib_per_1k=2.0)
    scales[-1]["first_recall_peak_total_rss"] = int(500 * _MIB)
    with pytest.raises(AssertionError, match="first_recall_peak_total_rss"):
        assert_bounded_marginal_cost(scales, _EXPLICIT_CEILINGS)


def test_over_ceiling_on_boot_sequence_delta_fails_independently() -> None:
    scales = _linear_scales(slope_mib_per_1k=2.0)
    scales[-1]["boot_sequence_delta_bytes"] = int(500 * _MIB)
    with pytest.raises(AssertionError, match="boot_sequence_delta_bytes"):
        assert_bounded_marginal_cost(scales, _EXPLICIT_CEILINGS)


def test_over_ceiling_on_latency_fails_independently() -> None:
    scales = _linear_scales(slope_mib_per_1k=2.0)
    scales[-1]["first_recall_wall_ms"] = 9999.0
    with pytest.raises(AssertionError, match="first_recall_wall_ms"):
        assert_bounded_marginal_cost(scales, _EXPLICIT_CEILINGS)


def test_single_metric_regression_fails_even_with_others_near_zero_slope() -> None:
    # Proves independence, not averaging: three metrics near-zero slope
    # (would pass on their own) plus ONE outlier still fails the gate.
    scales = _linear_scales(slope_mib_per_1k=0.1)
    scales[-1]["boot_sequence_delta_bytes"] = int(1000 * _MIB)
    with pytest.raises(AssertionError, match="boot_sequence_delta_bytes"):
        assert_bounded_marginal_cost(scales, _EXPLICIT_CEILINGS)


@pytest.mark.parametrize("missing_key", list(_EXPLICIT_CEILINGS.keys()))
def test_unset_ceiling_raises_not_passes(missing_key: str) -> None:
    ceilings = dict(_EXPLICIT_CEILINGS)
    ceilings[missing_key] = None
    with pytest.raises(RuntimeError, match="UNPINNED"):
        assert_bounded_marginal_cost(_linear_scales(), ceilings)


def test_missing_ceiling_key_raises() -> None:
    ceilings = dict(_EXPLICIT_CEILINGS)
    del ceilings["first_recall_latency_ms"]
    with pytest.raises(RuntimeError, match="UNPINNED"):
        assert_bounded_marginal_cost(_linear_scales(), ceilings)


def test_fewer_than_two_scales_raises() -> None:
    with pytest.raises(RuntimeError, match="at least 2"):
        assert_bounded_marginal_cost([_linear_scales()[0]], _EXPLICIT_CEILINGS)


def test_missing_metric_in_a_result_is_reported_as_a_failure() -> None:
    scales = _linear_scales()
    del scales[-1]["first_recall_delta_bytes"]
    with pytest.raises(AssertionError, match="first_recall_delta_bytes missing"):
        assert_bounded_marginal_cost(scales, _EXPLICIT_CEILINGS)


def test_aborted_scale_is_excluded_not_counted() -> None:
    scales = _linear_scales()
    scales.append({"ok": False, "n": 30000, "aborted": True})
    # The aborted scale carries no metrics; excluding it must not raise
    # (unlike a genuinely missing metric on an "ok" scale).
    assert_bounded_marginal_cost(scales, _EXPLICIT_CEILINGS)


# ---------------------------------------------------------------------------
# Layer 2 -- production-wired: the module's OWN _MARGINAL_CEILINGS against a
# real-shaped sweep summary.json.
# ---------------------------------------------------------------------------


_REAL_SWEEP_SUMMARY_ENV = "IAI_MCP_FP_SUMMARY"

# Committed default so this test never depends on a per-invocation env var.
_DEFAULT_SUMMARY_PATH = (
    Path(__file__).parent / "fixtures" / "phys_footprint" / "summary.json"
)


def test_production_gate_against_real_sweep() -> None:
    """Assert the module's OWN ceilings against an ACTUAL sweep artifact.

    Reads the summary.json at the path named by IAI_MCP_FP_SUMMARY when set,
    else the committed _DEFAULT_SUMMARY_PATH -- a per-scale max-aggregate of
    a real sweep, never synthetic data invented in-test. Skips (never
    silently passes) when no artifact is available at either location.
    """
    summary_path_str = os.environ.get(_REAL_SWEEP_SUMMARY_ENV)
    summary_path = Path(summary_path_str) if summary_path_str else _DEFAULT_SUMMARY_PATH
    if not summary_path.is_file():
        pytest.skip(
            f"{summary_path} does not exist -- run the sweep and commit its "
            f"summary.json there, or set {_REAL_SWEEP_SUMMARY_ENV}, before "
            "this test can gate against production data"
        )
    results = json.loads(summary_path.read_text(encoding="utf-8"))
    assert_bounded_marginal_cost(results, _MARGINAL_CEILINGS)
