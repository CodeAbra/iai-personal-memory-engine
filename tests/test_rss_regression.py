"""Opt-in steady-state resident-set plateau gate.

Drives the multi-cycle attribution harness as a fresh subprocess — launched
with the same mimalloc page-decommit environment the daemon ships with — and
asserts that the *steady-state* retained footprint does not climb more than a
fixed ceiling from one warm cycle to the next.

Two properties make this gate measure the config that actually ships, rather
than an artefact of the test runner:

1. **mimalloc env via subprocess.** mimalloc reads its decommit knobs once, at
   process start. Under pytest the interpreter's allocator is already
   initialised, so setting the env in-process is a no-op — it measures a
   configuration that never ships. The daemon always launches via launchd with
   ``MIMALLOC_*`` set in ``EnvironmentVariables``. To measure that, the harness
   must run as a fresh child process with those vars in its ``env``. The values
   are read from the packaged LaunchAgent plist so this gate stays bound to the
   shipped config: if the plist changes, the gate follows.

2. **Deterministic settled RSS, not a random tick, not transient peak, not
   cold-start.** Three effects inflate a naive per-cycle comparison and none is
   a retained leak:

     - Reading the "last periodic sampler tick" of each cycle is
       non-deterministic. The sampler fires on a fixed wall-clock interval, so
       the last tick before the next cycle lands at a random moment — it can
       catch a VACUUM transient or a mid-rebuild spike. Two identical runs then
       disagree by ~100 MiB with opposite signs. Measuring the tick measures
       the jitter, not the footprint.
     - The ``OPTIMIZE_HIPPO`` step runs a SQLite VACUUM that briefly doubles the
       DB footprint, then frees it. That makes the per-cycle *peak* spike, but
       the memory_relief actions hand the pages back before the next cycle, so
       it is a bounded transient — counting it as retained is wrong.
     - The first warm cycle pays one-time costs (first numba JIT compile, first
       ANN index build, first cache fill). Counting the cold-start 0->1 jump as
       a per-cycle leak is wrong — it never recurs.

   The honest retained footprint is a *settled* sample taken at the cycle
   boundary: the harness forces every returnable page back to the OS (pyarrow
   pool release + gc.collect + macOS zone pressure relief) and then takes ONE
   RSS read, emitting it as a ``step == "post-cycle-settled"`` row. That "all
   returnable pages returned" footprint is reproducible. This gate reads only
   those tagged rows, computes the warm steady-state delta
   ``settled_rss[N] - settled_rss[N-1]`` for ``N >= 2`` (skipping the one-time
   cold-start cycle), and asserts it stays under the ceiling.

This test is opt-in behind @pytest.mark.slow / --runslow so it stays out of the
default fast suite. The harness diagnostic dump (vmmap) is darwin-only, so the
whole module is skipped off darwin.
"""

from __future__ import annotations

import json
import os
import platform
import plistlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="the harness diagnostic dump (vmmap) is darwin-only",
)

_REPO = Path(__file__).resolve().parent.parent
HARNESS = _REPO / "tests" / "rss_rca" / "harness.py"
PACKAGED_PLIST = (
    _REPO / "src" / "iai_mcp" / "_deploy" / "launchd" / "com.iai-mcp.daemon.plist"
)

# Steady-state retained RSS may not climb more than this between consecutive
# WARM cycles. A real >= 100 MiB/cycle steady-state regression must be FIXED,
# NOT papered over by raising this ceiling. The threshold is a signal that a
# leak has returned, not a knob to relax: if a future change trips it, the
# change re-introduced a retained per-cycle leak.
RSS_PLATEAU_DELTA_KIB_MAX = 100 * 1024  # 100 MiB

# Harness workload sizing for the gate. A warm corpus large enough to exercise
# the sleep pipeline's heavy steps; three cycles so there is one cold-start
# cycle (0) plus two warm cycles (1, 2) to take a steady-state delta across.
_HARNESS_N = 2000
_HARNESS_CYCLES = 3
_HARNESS_SEED = 42
# The harness spawns a fresh process, builds the corpus, and runs three full
# WAKE->drain->SLEEP cycles end to end; cap it well above the observed wall
# time so a genuine hang fails loudly instead of blocking the suite forever.
_HARNESS_TIMEOUT_SEC = 3600


def test_threshold_constant_is_documented() -> None:
    # Companion guard — runs in the default suite, not the slow gate.
    assert isinstance(RSS_PLATEAU_DELTA_KIB_MAX, int)
    assert RSS_PLATEAU_DELTA_KIB_MAX > 0


def _shipped_mimalloc_env() -> dict[str, str]:
    """Read the mimalloc decommit env the daemon ships with from the plist.

    Sourcing the values from the packaged LaunchAgent plist keeps this gate
    bound to the real shipped configuration: the daemon launches via launchd
    with exactly these ``EnvironmentVariables``, so the harness subprocess must
    carry the same ones for the measurement to reflect what runs in production.
    """
    rendered = PACKAGED_PLIST.read_text(encoding="utf-8").replace(
        "{USERNAME}", "fakeuser"
    )
    parsed = plistlib.loads(rendered.encode("utf-8"))
    env = parsed.get("EnvironmentVariables", {})
    mimalloc = {
        key: str(value)
        for key, value in env.items()
        if key.startswith("MIMALLOC_")
    }
    assert mimalloc, (
        "packaged plist carries no MIMALLOC_* EnvironmentVariables; the gate "
        "cannot measure the shipped decommit config without them"
    )
    return mimalloc


def _settled_rss_per_cycle(rca_log: Path) -> dict[int, int]:
    """Map cycle index -> the deterministic settled RSS of that cycle (bytes).

    The harness emits exactly one ``step == "post-cycle-settled"`` sampler row
    per cycle, at the cycle boundary, after forcing every returnable page back
    to the OS (pyarrow pool release + gc.collect + macOS zone pressure relief)
    and taking a single RSS read. That "all returnable pages returned"
    footprint is reproducible — unlike the periodic 5 s sampler tick, whose
    "last row per cycle" lands at a random moment and can catch a VACUUM
    transient or a mid-rebuild spike, making two identical runs disagree by
    ~100 MiB. Reading only the tagged settled row removes that jitter.

    The boot phase is tagged ``cycle_idx == -1``; only non-negative cycle
    indices are measured cycles. If a cycle somehow emits more than one settled
    row, the last one wins (rows are appended in monotonic order).
    """
    settled_rss: dict[int, int] = {}
    for raw in rca_log.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if row.get("kind") != "sample":
            continue
        if row.get("step") != "post-cycle-settled":
            continue
        cycle = row.get("cycle_idx")
        rss = row.get("rss_bytes")
        if not isinstance(cycle, int) or not isinstance(rss, int):
            continue
        if cycle < 0:
            continue
        settled_rss[cycle] = rss
    return settled_rss


@pytest.mark.slow
def test_rss_steady_state_plateau_bounded() -> None:
    # This bound is a settled post-GC/post-decommit RSS delta measured across
    # warm cycles -- load-independent by construction, not an ambient
    # wall-clock timing gate -- and already runs opt-in via --runslow.

    # The harness --out validator accepts /tmp, /private/tmp, /var/folders only;
    # a pytest tmp_path resolves under /private/var/folders which it rejects, so
    # use an explicit /tmp dir we clean up ourselves. This is also the harness
    # tmp output dir — the live ~/.iai-mcp/ is never touched (the harness is
    # hermetic: it redirects HOME/IAI_MCP_STORE to its own tmp root and asserts
    # the real store signature is unchanged).
    out_dir = Path(tempfile.mkdtemp(prefix="iai-rss-regression-", dir="/tmp"))

    # Launch the harness as a fresh child carrying the daemon's shipped mimalloc
    # decommit env. In-process env mutation cannot reach an already-initialised
    # allocator, so the subprocess boundary is what makes this measure the real
    # config (mimalloc reads its knobs at process start).
    child_env = dict(os.environ)
    child_env.update(_shipped_mimalloc_env())

    try:
        proc = subprocess.run(
            [
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
            ],
            cwd=str(_REPO),
            env=child_env,
            capture_output=True,
            text=True,
            timeout=_HARNESS_TIMEOUT_SEC,
        )
        assert proc.returncode == 0, (
            f"harness exit {proc.returncode}\nstderr (tail):\n{proc.stderr[-3000:]}"
        )

        rca_log = out_dir / "rca-log.jsonl"
        assert rca_log.exists(), f"harness did not write {rca_log}"

        settled_rss = _settled_rss_per_cycle(rca_log)
        assert set(settled_rss) == set(range(_HARNESS_CYCLES)), (
            f"expected one post-cycle-settled sample per cycle "
            f"{sorted(range(_HARNESS_CYCLES))}, saw {sorted(settled_rss)}"
        )

        # Steady-state delta across WARM cycles only (N >= 2). The cold-start
        # cycle (0 -> 1) pays one-time JIT + index-build + cache-fill costs and
        # is deliberately excluded — it is not a recurring leak.
        ceiling_bytes = RSS_PLATEAU_DELTA_KIB_MAX * 1024
        worst_delta_bytes = None
        worst_pair = None
        for cycle in range(2, _HARNESS_CYCLES):
            delta = settled_rss[cycle] - settled_rss[cycle - 1]
            if worst_delta_bytes is None or delta > worst_delta_bytes:
                worst_delta_bytes = delta
                worst_pair = (cycle - 1, cycle)

        assert worst_delta_bytes is not None, (
            "need at least one warm-to-warm cycle pair (cycles >= 3) to measure "
            "a steady-state delta"
        )

        per_cycle = ", ".join(
            f"cycle{c}_settled={settled_rss[c] / (1024 * 1024):.1f}MiB"
            for c in sorted(settled_rss)
        )
        prev_c, cur_c = worst_pair
        assert worst_delta_bytes < ceiling_bytes, (
            "steady-state per-cycle settled RSS growth exceeded the ceiling: "
            f"{per_cycle}; worst warm delta cycle{prev_c}->cycle{cur_c}="
            f"{worst_delta_bytes / (1024 * 1024):+.1f} MiB, "
            f"ceiling={ceiling_bytes / (1024 * 1024):.0f} MiB. "
            "A real per-cycle leak has returned — fix the leak, do not raise "
            "the ceiling."
        )
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
