from __future__ import annotations

import sys
from pathlib import Path

import pytest


_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)


def _t(n: int) -> float:
    from bench.memory_footprint import _threshold_mb_for_n

    return _threshold_mb_for_n(n)


def test_threshold_matches_legacy_anchor_at_n_1000():
    assert _t(1000) == pytest.approx(1600.0)


def test_threshold_unchanged_at_or_below_anchor():
    assert _t(500) == pytest.approx(1600.0)
    assert _t(100) == pytest.approx(1600.0)
    assert _t(1) == pytest.approx(1600.0)


def test_threshold_grows_with_log_of_n():
    assert _t(10_000) == pytest.approx(2000.0, abs=0.5)
    assert _t(100_000) == pytest.approx(2400.0, abs=0.5)


def test_threshold_strictly_monotonic():
    sizes = [1, 100, 999, 1000, 1001, 5000, 10_000, 50_000, 100_000]
    thresholds = [_t(n) for n in sizes]
    for prev, cur in zip(thresholds, thresholds[1:]):
        assert cur >= prev, (
            f"threshold must be non-decreasing across N; got {prev} → {cur}"
        )


def test_threshold_covers_observed_n_10000_peak():
    observed_peak_n10k = 1862.0
    assert _t(10_000) >= observed_peak_n10k, (
        f"N-aware threshold at N=10000 ({_t(10_000)}) must cover the "
        f"observed peak ({observed_peak_n10k} MB)"
    )


def test_threshold_function_uses_log10():
    ratio = (_t(10_000) - 1600.0) / 1600.0
    assert ratio == pytest.approx(0.25, abs=0.001)


def test_bench_json_output_uses_n_aware_threshold(tmp_path, monkeypatch):
    pytest.importorskip("numpy")

    from bench.memory_footprint import (  # noqa: E402
        _threshold_mb_for_n,
        run_memory_footprint,
    )

    result = run_memory_footprint(n=50, store_path=tmp_path, skip_graph=True)
    assert result["threshold_mb"] == pytest.approx(_threshold_mb_for_n(50))
    assert result["n"] == 50
