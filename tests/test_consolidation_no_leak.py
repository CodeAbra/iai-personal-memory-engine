"""Sustained capture+consolidation no-leak gate on the watchdog's charged metric.

A thin wrapper over ``tests.rss_rca.harness.run(sleep_budget_sec=...)``: drives
three WAKE(drain)->SLEEP cycles against a synthetic corpus and asserts that the
per-cycle SETTLED ``phys_footprint`` (macOS's charged-memory metric -- the same
number the daemon watchdog SIGKILLs a process on, NOT resident-set size, which
counts reusable pages the allocator has freed but not yet returned) does not
grow beyond a calibrated bound from cycle 1 onward.

Opt-in (``@pytest.mark.slow``, ``--runslow``): each cycle drives a real,
deadline-bounded sleep pipeline pass plus a forced runtime-graph-cache
rebuild over a ~3000-record corpus, several minutes total.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tests.rss_rca import harness

_N_RECORDS = 3000
_CYCLES = 3
_SLEEP_BUDGET_SEC = 60.0
_DEFERRED_EVENTS_PER_CYCLE = 100
_SEED = 42

# ---------------------------------------------------------------------------
# Calibration (derived from one clean run of this exact configuration,
# including the per-cycle forced runtime-graph-cache rebuild, on macOS; see
# the Calibration Protocol -- never a hardcoded guess):
#
#   cycle 0 (warm-up, excluded from the bound -- corpus/embedder/index cold
#            caches materialize here and are not leak signal):
#     settled_phys = 552,830,272 bytes (~527.2 MiB)
#   cycle 1 (baseline):
#     settled_phys = 781,551,128 bytes (~745.4 MiB)
#   cycle 2 (final):
#     settled_phys = 837,666,424 bytes (~798.9 MiB)
#
#   observed delta (cycle 1 -> cycle 2) = 56,115,296 bytes (~53.5 MiB).
#
# This delta is expected corpus-driven growth, not a leak signature: each
# cycle also stages and drains _DEFERRED_EVENTS_PER_CYCLE fresh captures (the
# harness's own ambient-churn simulation), so the active corpus itself grows
# every cycle and a proportional settled-footprint increase is the CORRECT
# behaviour, not a defect. A genuine leak's signature is an ACCELERATING
# per-cycle delta (each cycle leaving behind more unreclaimed memory on top of
# the last); the calibration run's two consecutive per-cycle deltas
# (cycle0->1 ~218.1 MiB, cycle1->2 ~53.5 MiB) DECELERATE, consistent with a
# one-time warm-up cost amortizing away rather than compounding -- so no
# "strictly non-increasing delta" clause is asserted here: with only 3 cycles
# expected corpus growth alone can plausibly produce a rising sequence on a
# clean run, and gating on that shape would be gating on noise, not leak
# evidence. LEAK_BUDGET is set to roughly 2x the observed delta, comfortably
# above per-run noise and the expected corpus-growth contribution while still
# catching a genuine multi-hundred-MiB runaway.
# ---------------------------------------------------------------------------
_LEAK_BUDGET_BYTES = 150 * 1024 * 1024  # ~150 MiB


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_settled_phys_footprint_bounded_across_cycles(tmp_path: Path) -> None:
    """Settled phys_footprint growth from cycle 1 to the final cycle stays
    under the calibrated budget across 3 sustained capture+consolidation
    cycles.

    Reads the ``phys_footprint`` sidecar the harness writes per cycle (never
    resident-set size -- RSS on macOS counts reusable MADV_FREE pages the
    allocator freed but has not returned, so an RSS-only gate can stay green
    while the real process is over the watchdog's charged-memory cap). On a
    non-macOS host ``phys_footprint`` is unavailable and this test skips
    honestly rather than asserting an undercalibrated RSS proxy.
    """
    out = tmp_path / "out"
    out.mkdir()

    harness.run(
        n=_N_RECORDS,
        cycles=_CYCLES,
        out=out,
        seed=_SEED,
        deferred_events_per_cycle=_DEFERRED_EVENTS_PER_CYCLE,
        sleep_budget_sec=_SLEEP_BUDGET_SEC,
    )

    sidecar = out / "settled-footprint.json"
    assert sidecar.exists(), f"settled-footprint.json missing at {sidecar}"
    rows = json.loads(sidecar.read_text(encoding="utf-8"))
    assert len(rows) == _CYCLES, (
        f"expected {_CYCLES} settled-footprint rows, got {len(rows)}: {rows}"
    )

    # Provable coverage of the largest allocator. RECALL_INDEX_REBUILD
    # (rewrites the graph-cache snapshot, decodes the whole map, rebuilds the
    # exact-authority index) is NOT reliably exercised by the rolling
    # `force_run` pass alone: the deadline is checked only at each step's own
    # entry, so a slow earlier step can push the deadline past this step
    # before the rolling pass even reaches it -- and separately,
    # `_rebuild_and_save_rgc`'s own dirty-threshold fuse skips the heavy
    # rebuild body whenever the warm cache looks clean enough, independent of
    # the deadline. The harness therefore force-rebuilds once per cycle
    # (``force=True``, bypassing that fuse) before taking the settled
    # measurement; assert that forced call genuinely ran its body every
    # cycle, so the phys_footprint series below is never silently missing
    # this allocator's cost.
    bad_cycles = [
        int(r["cycle"]) for r in rows
        if r.get("recall_index_rebuilt") is not True
    ]
    assert not bad_cycles, (
        f"cycle(s) {bad_cycles} did not force-rebuild the runtime-graph cache "
        f"-- the settled-footprint measurement below would exclude the "
        f"pipeline's largest allocator for those cycles. Per-cycle "
        f"completed_steps / recall_index_rebuilt: "
        f"{[(int(r['cycle']), r.get('completed_steps'), r.get('recall_index_rebuilt')) for r in rows]}"
    )

    if sys.platform != "darwin" or any(r.get("settled_phys") is None for r in rows):
        pytest.skip(
            "phys_footprint is macOS-only (proc_pid_rusage); this host cannot "
            "supply the watchdog's charged-memory metric"
        )

    phys_by_cycle = [int(r["settled_phys"]) for r in rows]
    # Cycle 0 is a warm-up outlier (cold caches materializing) -- excluded
    # from the bound per the calibration above.
    baseline = phys_by_cycle[1]
    final = phys_by_cycle[-1]
    delta = final - baseline
    # Unconditional (not just on failure): every run self-documents its own
    # observed margin against the calibrated budget, so a future re-check
    # never needs a fresh multi-minute calibration pass to know how tight the
    # bound is.
    print(
        f"settled_phys series: {phys_by_cycle} bytes; cycle1->final delta "
        f"{delta} bytes ({delta / (1024 * 1024):.1f} MiB) against budget "
        f"{_LEAK_BUDGET_BYTES} bytes ({_LEAK_BUDGET_BYTES / (1024 * 1024):.0f} MiB)"
    )

    assert delta < _LEAK_BUDGET_BYTES, (
        f"settled phys_footprint grew {delta} bytes ({delta / (1024 * 1024):.1f} "
        f"MiB) from cycle 1 ({baseline} bytes) to the final cycle "
        f"({final} bytes), exceeding the calibrated budget of "
        f"{_LEAK_BUDGET_BYTES} bytes ({_LEAK_BUDGET_BYTES / (1024 * 1024):.0f} "
        f"MiB). Full per-cycle series: {phys_by_cycle}"
    )
