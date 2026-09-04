"""Multi-cycle WAKE->drain->SLEEP RSS soak for the GC-taming kill-switch.

Proves the ``gc.freeze()`` x sleep-relief ``gc.collect()`` interaction is
safe: the harness (``tests/rss_rca/harness.py``) drives repeated in-process
WAKE (deferred-capture drain, which fires ``_step_memory_relief`` from
``capture.py``'s ``deferred_drain`` site) -> SLEEP (the sleep pipeline's own
dispatch-loop relief call) cycles.

The settled per-cycle RSS naturally oscillates cycle to cycle (SQLite VACUUM
transients, cache-rebuild timing, a growing deferred-capture corpus) even
with GC taming off -- a single warm-to-warm delta against an absolute
ceiling cannot distinguish that natural oscillation from a genuine
frozen-generation leak; measured directly, a lone delta swings by tens of
MiB run to run with no taming involved at all. This gate instead runs a
taming-OFF baseline and a taming-ON (``gc_mode=both``) arm from the SAME
seed in the same test and compares each arm's MEAN warm-cycle delta
(averaged over multiple warm-to-warm transitions, not a single sample): if
``gc.freeze()`` were accumulating unbounded per-cycle growth, the ON arm's
mean would clearly and increasingly exceed the OFF arm's own natural
oscillation, not track it.

Opt-in behind @pytest.mark.slow / --runslow, matching the sibling gate
(test_rss_regression.py).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tests.test_rss_regression import (
    _REPO,
    _settled_rss_per_cycle,
    _shipped_mimalloc_env,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="the harness diagnostic dump (vmmap) is darwin-only",
)

# The comparative quantity this gate actually asserts on: how much MORE, on
# average, does a taming-ON warm cycle grow than a taming-OFF warm cycle
# measured in the same run. Measured natural OFF-arm oscillation is on the
# order of tens of MiB; a real frozen-generation leak would show up as a
# large, sustained excess, not noise of this size.
GC_TAMING_EXCESS_MEAN_DELTA_KIB_MAX = 60 * 1024  # 60 MiB
# Backstop: the ON arm's own mean warm delta must also stay bounded in
# absolute terms, so a scenario where the (untamed) baseline itself
# regressed cannot masquerade as "no excess" by comparison alone.
GC_TAMING_ON_MEAN_DELTA_KIB_MAX = 150 * 1024  # 150 MiB

_HARNESS_N = 1200
# Five cycles -> four warm-to-warm deltas (cycle0->1 excluded as the
# one-time cold-start cost) -- enough transitions to average out per-cycle
# oscillation into a stable mean instead of reading a single noisy sample.
_HARNESS_CYCLES = 5
_HARNESS_SEED = 1337
_HARNESS_TIMEOUT_SEC = 3600


def _run_harness(out_dir: Path, *, gc_mode: str) -> subprocess.CompletedProcess:
    child_env = dict(os.environ)
    child_env.update(_shipped_mimalloc_env())
    args = [
        sys.executable,
        "-m",
        "tests.rss_rca.harness",
        "--n",
        str(_HARNESS_N),
        "--cycles",
        str(_HARNESS_CYCLES),
        "--out",
        str(out_dir),
        "--seed",
        str(_HARNESS_SEED),
        "--gc-mode",
        gc_mode,
    ]
    return subprocess.run(
        args,
        cwd=str(_REPO),
        env=child_env,
        capture_output=True,
        text=True,
        timeout=_HARNESS_TIMEOUT_SEC,
    )


def _settled_series(out_dir: Path, *, label: str) -> dict[int, int]:
    rca_log = out_dir / "rca-log.jsonl"
    assert rca_log.exists(), f"[{label}] harness did not write {rca_log}"

    settled_rss = _settled_rss_per_cycle(rca_log)
    assert set(settled_rss) == set(range(_HARNESS_CYCLES)), (
        f"[{label}] expected one post-cycle-settled sample per cycle "
        f"{sorted(range(_HARNESS_CYCLES))}, saw {sorted(settled_rss)}"
    )
    return settled_rss


def _mean_warm_delta_bytes(settled_rss: dict[int, int]) -> float:
    """Average settled-RSS delta across every warm-to-warm cycle pair.

    Cycle 0->1 is excluded: it pays one-time JIT/index-build/cache-fill
    costs that never recur, exactly as test_rss_regression.py already
    excludes it from its own steady-state measurement.
    """
    deltas = [
        settled_rss[c] - settled_rss[c - 1]
        for c in range(2, _HARNESS_CYCLES)
    ]
    assert deltas, (
        "need at least one warm-to-warm cycle pair (cycles >= 3) to measure "
        "a steady-state delta"
    )
    return sum(deltas) / len(deltas)


def _format_series(settled_rss: dict[int, int]) -> str:
    return ", ".join(
        f"cycle{c}={settled_rss[c] / (1024 * 1024):.1f}MiB"
        for c in sorted(settled_rss)
    )


@pytest.mark.slow
def test_gc_taming_plateau_holds_across_wake_sleep_cycles() -> None:
    # Two fresh harness subprocesses -- taming is process-global
    # (gc.disable()/gc.freeze() applied inside the harness's own child
    # process), so OFF and ON never share interpreter state.
    off_dir = Path(tempfile.mkdtemp(prefix="iai-gc-taming-off-", dir="/tmp"))
    on_dir = Path(tempfile.mkdtemp(prefix="iai-gc-taming-on-", dir="/tmp"))
    try:
        off_proc = _run_harness(off_dir, gc_mode="off")
        assert off_proc.returncode == 0, (
            f"harness (taming OFF) exit {off_proc.returncode}\n"
            f"stderr (tail):\n{off_proc.stderr[-3000:]}"
        )
        off_settled = _settled_series(off_dir, label="taming-off-baseline")

        on_proc = _run_harness(on_dir, gc_mode="both")
        assert on_proc.returncode == 0, (
            f"harness (taming ON) exit {on_proc.returncode}\n"
            f"stderr (tail):\n{on_proc.stderr[-3000:]}"
        )
        on_settled = _settled_series(on_dir, label="taming-on")

        assert "gc_taming_applied" in (on_dir / "rca-log.jsonl").read_text(
            encoding="utf-8",
        ), (
            "harness (taming ON) never logged gc_taming_applied -- the flag "
            "was not actually exercised"
        )
        assert "gc_taming_applied" not in (off_dir / "rca-log.jsonl").read_text(
            encoding="utf-8",
        ), "harness (taming OFF) unexpectedly applied gc taming"

        off_mean = _mean_warm_delta_bytes(off_settled)
        on_mean = _mean_warm_delta_bytes(on_settled)
        excess = on_mean - off_mean

        evidence = (
            f"off_series=[{_format_series(off_settled)}] "
            f"off_mean_warm_delta={off_mean / (1024 * 1024):+.1f}MiB; "
            f"on_series=[{_format_series(on_settled)}] "
            f"on_mean_warm_delta={on_mean / (1024 * 1024):+.1f}MiB; "
            f"excess(on-off)={excess / (1024 * 1024):+.1f}MiB"
        )
        print(f"gc-taming soak: {evidence}")

        # The one assertion this plan exists to prove: gc.freeze() applied
        # right after the corpus (boot-resident set) is built does not make
        # the sleep-relief gc.collect() x WAKE-drain gc.collect() interaction
        # accumulate unbounded frozen-generation growth relative to the
        # untamed baseline's own natural per-cycle oscillation.
        assert excess < GC_TAMING_EXCESS_MEAN_DELTA_KIB_MAX * 1024, (
            f"taming-ON mean warm-cycle RSS growth exceeds the taming-OFF "
            f"baseline by more than the ceiling: {evidence}, "
            f"ceiling={GC_TAMING_EXCESS_MEAN_DELTA_KIB_MAX / 1024:.0f} MiB. "
            "A real frozen-generation leak has returned -- fix the leak, do "
            "not raise the ceiling."
        )
        assert on_mean < GC_TAMING_ON_MEAN_DELTA_KIB_MAX * 1024, (
            f"taming-ON mean warm-cycle RSS growth exceeds its own absolute "
            f"backstop: {evidence}, "
            f"ceiling={GC_TAMING_ON_MEAN_DELTA_KIB_MAX / 1024:.0f} MiB."
        )
    finally:
        shutil.rmtree(off_dir, ignore_errors=True)
        shutil.rmtree(on_dir, ignore_errors=True)
