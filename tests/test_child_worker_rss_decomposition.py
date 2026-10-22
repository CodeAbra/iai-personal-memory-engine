"""Bounded proof that `child_worker_rss_probe` runs and produces a parseable
decomposition. This asserts SHAPE, not a byte-count threshold -- the numbers
are diagnostic input, not a pass/fail gate. Single-file, well under
the 300s budget at this bounded N; the heavier N=3000/8000 anchor runs are
opt-in (`--runslow`), and the full N=15000/30000 sweep is a manual background
step (see `tests/rss_rca/child_worker_rss_probe.py`'s module docstring).
"""
from __future__ import annotations

import pytest

from tests.rss_rca.child_worker_rss_probe import run_all_arms

_BOUNDED_N = 1000


def test_probe_emits_full_decomposition_at_bounded_n() -> None:
    summary = run_all_arms(_BOUNDED_N, timeout_sec=280.0)

    staged = summary["staged"]
    assert staged.get("ok") is True, f"staged arm failed: {staged}"
    for key in (
        "after_build_mib", "after_community_mib",
        "after_centrality_mib", "after_rich_club_mib",
    ):
        assert key in staged, f"staged arm missing checkpoint {key!r}: {staged}"
    assert staged["staged_total_delta_mib"] is not None

    one_time = summary["one_time_vs_per_call"]
    assert one_time.get("ok") is True, f"one_time_vs_per_call arm failed: {one_time}"
    assert one_time.get("second_call_delta_mib") is not None, (
        f"per-spawn-vs-one-time number missing: {one_time}"
    )

    real_spawn = summary["real_spawn"]
    assert real_spawn.get("ok") is True, f"real_spawn arm failed: {real_spawn}"
    assert real_spawn.get("rebuilt") is True, (
        f"real_spawn arm did not complete a rebuild: {real_spawn}"
    )
    assert real_spawn.get("child_rss_delta_mib") is not None

    # Divergence must be surfaced (comparable) whenever both sides produced a
    # positive number -- absence here would mean the check silently no-oped.
    divergence = summary["divergence"]
    assert divergence.get("comparable") is True, (
        f"arm(i)/arm(iii) divergence check did not run: {divergence}"
    )


@pytest.mark.perf
def test_probe_at_n3000_and_n8000() -> None:
    """Heavier anchor points, opt-in (`--perf`), out of the default gate."""
    for n in (3000, 8000):
        summary = run_all_arms(n, timeout_sec=280.0)
        assert summary["staged"].get("ok") is True
        assert summary["real_spawn"].get("ok") is True
