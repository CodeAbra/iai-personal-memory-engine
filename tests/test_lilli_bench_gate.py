"""Bench gate: drive the Python bench scripts under the Rust storage driver and
assert each metric meets-or-beats the committed performance thresholds.

The canonical threshold values live in the Rust crate
(``crates/lilli-parity/src/thresholds.rs``) so they are exercised by
``cargo test`` and stored in exactly one place.  This module mirrors those
constants and a guard test asserts the mirror has not drifted -- a value changed
in either place is caught.

The bench runner itself is OPT-IN: it is heavy (it builds stores at scale and
runs many recall iterations) and runs only at phase verification by the
orchestrator in a single process, daemon-down -- never inside a parallel
worktree executor.  Set ``IAI_MCP_BENCH_GATE`` to enable it.  The helper +
threshold-mirror unit tests always run and need no store, wheel, or env.

Already-proven component wins (measured by ``cargo bench``, not re-measured
here): HDC kernels hot-path ~157x; the hand-rolled store insert is O(tree
height) at 30k; the engine's edges UPSERT dropped 439s -> 0.75s at 30k bulk.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Threshold mirror.  Canonical source: crates/lilli-parity/src/thresholds.rs.
# A guard test asserts these equal the committed Rust constants.
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "RECALL_P95_N100_MS": 77.0,
    "RECALL_P95_N1K_MS": 105.0,
    "RECALL_P95_N10K_MS": 368.0,
    "VERBATIM_RECALL": 1.0,
    "VERBATIM_FLOOR": 0.993,
    "RSS_N10K_MB": 589.0,
    "HDC_SPEEDUP_MIN": 10.0,
    "HDC_SPEEDUP_TARGET": 100.0,
    "SESSION_TOKEN_BUDGET": 3000.0,
}


def meets_or_beats(measured: float, threshold: float, lower_is_better: bool) -> bool:
    """Return True when ``measured`` meets or beats ``threshold``.

    For latency, resident memory, and token cost, lower is better. For speedup
    and verbatim recall, higher is better.  Mirrors the Rust helper.
    """
    if lower_is_better:
        return measured <= threshold
    return measured >= threshold


_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUST_THRESHOLDS = _REPO_ROOT / "crates" / "lilli-parity" / "src" / "thresholds.rs"


def _parse_rust_thresholds(source: str) -> dict[str, float]:
    """Extract ``pub const NAME: f64 = VALUE;`` pairs from the Rust source."""
    import re

    out: dict[str, float] = {}
    for name, value in re.findall(
        r"pub const (\w+):\s*f64\s*=\s*([0-9.]+);", source
    ):
        out[name] = float(value)
    return out


# ---------------------------------------------------------------------------
# Helper + mirror unit tests (always run; no store / wheel / env needed).
# ---------------------------------------------------------------------------
def test_meets_or_beats_lower_is_better():
    assert meets_or_beats(360.0, 368.0, lower_is_better=True) is True
    assert meets_or_beats(400.0, 368.0, lower_is_better=True) is False
    # Boundary: equal to the threshold meets it.
    assert meets_or_beats(368.0, 368.0, lower_is_better=True) is True


def test_meets_or_beats_higher_is_better():
    assert meets_or_beats(1.0, 0.993, lower_is_better=False) is True
    assert meets_or_beats(0.99, 0.993, lower_is_better=False) is False
    # Boundary: equal to the floor meets it.
    assert meets_or_beats(0.993, 0.993, lower_is_better=False) is True


def test_mirrored_thresholds_match_committed_rust_values():
    """The Python mirror must equal the constants committed in the Rust crate."""
    assert _RUST_THRESHOLDS.exists(), f"missing {_RUST_THRESHOLDS}"
    rust = _parse_rust_thresholds(_RUST_THRESHOLDS.read_text())
    for name, value in THRESHOLDS.items():
        assert name in rust, f"{name} not found in thresholds.rs"
        assert rust[name] == value, (
            f"threshold drift for {name}: python mirror={value} "
            f"rust={rust[name]} -- update the mirror or the Rust crate"
        )


# ---------------------------------------------------------------------------
# Bench runner (opt-in).
# ---------------------------------------------------------------------------
_BENCH_GATE_ENABLED = bool(os.environ.get("IAI_MCP_BENCH_GATE"))


def _run_bench(args: list[str]) -> list[dict]:
    """Run a bench module under the Rust driver; return parsed JSON line(s).

    The bench prints one JSON object per line (neural_map prints one per ``--n``);
    every JSON line on stdout is parsed and returned in order.
    """
    env = dict(os.environ)
    env["LILLI_STORAGE_DRIVER"] = "lilli"
    env["IAI_MCP_HD_BACKEND"] = "rust"
    proc = subprocess.run(
        [sys.executable, "-m", *args],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    objects: list[dict] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            objects.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not objects:
        raise AssertionError(
            f"bench {args} produced no JSON line "
            f"(rc={proc.returncode}); stderr tail: {proc.stderr[-500:]!r}"
        )
    return objects


@pytest.mark.skipif(
    not _BENCH_GATE_ENABLED,
    reason=(
        "bench gate is opt-in: set IAI_MCP_BENCH_GATE (heavy; runs only at "
        "phase verification in a single process, never in a worktree executor)"
    ),
)
def test_bench_match_or_beat_under_lilli_driver():
    measured: dict[str, float] = {}
    failures: list[str] = []

    def check(name: str, value: float, threshold: float, lower: bool) -> None:
        measured[name] = value
        if not meets_or_beats(value, threshold, lower):
            direction = "<=" if lower else ">="
            failures.append(
                f"{name}: measured {value} fails {direction} {threshold}"
            )

    # Recall p95 at three corpus sizes; the 10k bound is the hard gate (the
    # scale dimension the live cutover failed on).
    nm = _run_bench(
        [
            "bench.neural_map",
            "--n", "100",
            "--n", "1000",
            "--n", "10000",
            "--iterations", "30",
        ]
    )
    by_n = {int(o["n"]): o for o in nm if "n" in o and "latency_ms_p95" in o}
    assert {100, 1000, 10000} <= set(by_n), (
        f"neural_map did not report all three sizes; got {sorted(by_n)}"
    )
    check(
        "neural_map_p95_n100",
        float(by_n[100]["latency_ms_p95"]),
        THRESHOLDS["RECALL_P95_N100_MS"],
        lower=True,
    )
    check(
        "neural_map_p95_n1k",
        float(by_n[1000]["latency_ms_p95"]),
        THRESHOLDS["RECALL_P95_N1K_MS"],
        lower=True,
    )
    check(
        "neural_map_p95_n10k",
        float(by_n[10000]["latency_ms_p95"]),
        THRESHOLDS["RECALL_P95_N10K_MS"],
        lower=True,
    )

    # Verbatim recall accuracy (higher is better; floor 0.993, target 1.0).
    vb = _run_bench(["bench.verbatim"])[0]
    check(
        "verbatim_accuracy",
        float(vb["accuracy"]),
        THRESHOLDS["VERBATIM_FLOOR"],
        lower=False,
    )

    # Resident-set peak at 10k records (lower is better).
    mf = _run_bench(["bench.memory_footprint", "--n", "10000"])[0]
    check(
        "memory_footprint_rss_n10k",
        float(mf["rss_mb_peak"]),
        THRESHOLDS["RSS_N10K_MB"],
        lower=True,
    )

    # Session token cost.  Read the literal key and fail loud on a missing/zero
    # value BEFORE the budget check, so a renamed key or a measurement that did
    # not actually run cannot silently pass as a 0 that beats the budget.
    sc = _run_bench(["bench.total_session_cost"])[0]
    assert "total_tokens" in sc, (
        f"session-cost bench is missing the 'total_tokens' key; got keys "
        f"{sorted(sc)} -- the gate cannot measure session cost"
    )
    total_tokens = int(sc["total_tokens"])
    assert total_tokens > 0, (
        f"session-cost total_tokens is {total_tokens} (must be > 0); a zero "
        f"means the bench did not measure -- it must not pass as 0 <= 3000"
    )
    check(
        "session_total_tokens",
        float(total_tokens),
        THRESHOLDS["SESSION_TOKEN_BUDGET"],
        lower=True,
    )

    if failures:
        table = "\n".join(f"  {f}" for f in failures)
        full = "\n".join(f"  {k} = {v}" for k, v in sorted(measured.items()))
        raise AssertionError(
            "bench gate failed under the Rust driver:\n"
            + table
            + "\nfull measured table:\n"
            + full
        )

    print("\n[bench-gate] all metrics meet-or-beat the committed thresholds:")
    for k, v in sorted(measured.items()):
        print(f"[bench-gate]   {k} = {v}")


@pytest.mark.skip(
    reason=(
        "HDC kernel speedup is proven by 'cargo bench -p lilli-hd' (hot-path "
        "~157x, bundle ~12.8x), not re-measured in Python; the run+record step "
        "captures the cargo-bench number against HDC_SPEEDUP_MIN / _TARGET"
    )
)
def test_hdc_speedup_documented_via_cargo_bench():
    # Placeholder documenting where the HDC x10-100 win is measured.  The
    # speedup floor and target live in THRESHOLDS (HDC_SPEEDUP_MIN / _TARGET).
    assert THRESHOLDS["HDC_SPEEDUP_MIN"] == 10.0
    assert THRESHOLDS["HDC_SPEEDUP_TARGET"] == 100.0
