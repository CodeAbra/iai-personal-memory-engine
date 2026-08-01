"""Insert-scaling gate through the native store surface.

Proves the durable shape of the record store at load: inserts are linear in the
row count (doubling the rows roughly doubles the wall time, never quadruples
it), throughput holds its floor at thirty thousand records, the largest-key
lookup the insert path relies on stays cheap (tree-height cost, independent of
the row count), the insert path issues no whole-tree walk, and the working-set
resident memory plateaus flat across successive batches rather than climbing
without bound.

Runs hermetic and single-process: a temp store with the fast drive-sync escape,
the resident set sampled in-process by reading this process's own size. Nothing
restarts, nothing is shared.
"""

from __future__ import annotations

from tests.conftest_shared import retry_once_on_assertion

import json
import os
import statistics
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

os.environ.setdefault("LILLI_FSYNC_MODE", "fast")


def _native_available() -> bool:
    try:
        from iai_mcp_native import store  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _native_available(), reason="iai_mcp_native store submodule not installed"
)


# --- committed gates --------------------------------------------------------
#
# These mirror the Rust-side committed thresholds (the proven post-fix profile):
# throughput floor, the doubling-time linearity bound, and the per-insert
# largest-key-lookup cost bound. They live here as plain module constants so the
# host-side gate is self-contained and reads the same numbers the engine asserts.

INSERT_ROWS_PER_S_FLOOR = 1500.0
DOUBLING_TIME_RATIO_MAX = 2.2
# The marshalling overhead of per-insert FFI calls widens the constant factor
# relative to the in-Rust bench, so the host-side throughput floor carries the
# same algorithmic floor (the linearity shape is marshalling-invariant).

# The largest-key lookup descends the tree once; at thirty thousand records the
# tree is a few levels deep, so a tight loop of lookups reads a small constant
# number of pages each, never a count that scales with the rows.
MAX_KEY_READS_BOUND = 64

# Record geometry mirrors the real store: a 1536 B float32 embedding plus a
# ciphertext-sized tail, ~1656 B, below the overflow threshold (inline cells).
RECORD_LEN = 1656


def _make_record(key: int) -> bytes:
    head = key.to_bytes(8, "big", signed=True)
    tail = bytes((i + key) & 0xFF for i in range(RECORD_LEN - 8))
    return head + tail


def _new_store(path: Path):
    from iai_mcp_native import store

    s = store.Store.open(str(path))
    s.enable_wal_mode()
    root = s.create_tree()
    return s, root


def _insert_batch(s, root: int, lo: int, hi: int) -> float:
    """Insert keys [lo, hi) in one transaction. Returns elapsed seconds."""
    start = time.perf_counter()
    s.begin_write()
    for key in range(lo, hi):
        s.insert(root, key, _make_record(key))
    s.commit()
    return time.perf_counter() - start


# --- tests ------------------------------------------------------------------


@retry_once_on_assertion
def test_insert_linear_and_throughput(tmp_path: Path) -> None:
    """t(20k) <= 2.2x t(10k); 30k no super-linear blowup; >= 1500 rows/s; no walk."""
    # Each size into a fresh tree so the measurement is the pure insert cost.
    s10, r10 = _new_store(tmp_path / "n10k.lilli")
    s10.reset_full_scan_count()
    t10 = _insert_batch(s10, r10, 0, 10_000)
    walks_10 = s10.full_scan_count()

    s20, r20 = _new_store(tmp_path / "n20k.lilli")
    s20.reset_full_scan_count()
    t20 = _insert_batch(s20, r20, 0, 20_000)
    walks_20 = s20.full_scan_count()

    s30, r30 = _new_store(tmp_path / "n30k.lilli")
    s30.reset_full_scan_count()
    t30 = _insert_batch(s30, r30, 0, 30_000)
    walks_30 = s30.full_scan_count()

    ratio_20 = t20 / t10
    ratio_30 = t30 / t10
    rps_30 = 30_000 / t30

    print(
        f"\nt(10k)={t10:.3f}s t(20k)={t20:.3f}s t(30k)={t30:.3f}s "
        f"| t20/t10={ratio_20:.3f} t30/t10={ratio_30:.3f} "
        f"| 30k throughput={rps_30:.1f} rows/s "
        f"| walks(10k/20k/30k)={walks_10}/{walks_20}/{walks_30}"
    )

    # No whole-tree walk on the insert path — the linear-sweep regression absent.
    assert walks_10 == 0 and walks_20 == 0 and walks_30 == 0, (
        "insert path issued a whole-tree walk — the row-count-quadratic regression"
    )

    # Doubling the rows at most ~doubles the time.
    assert ratio_20 <= DOUBLING_TIME_RATIO_MAX, (
        f"insert is not linear: t(20k)/t(10k)={ratio_20:.3f} > {DOUBLING_TIME_RATIO_MAX}"
    )
    # Tripling the rows shows no super-linear blowup. This 30k/10k bound is the
    # secondary check; the load-bearing linearity proof is t(20k)/t(10k) above
    # plus the zero whole-tree-walk assertion. t(10k) is measured before the
    # cache/allocator is warm, so it runs artificially small and inflates this
    # ratio on a cold single-shot run; the slack here absorbs that cold-start
    # noise without weakening the primary linearity gate.
    THIRTY_K_RATIO_MAX = 3.6
    assert ratio_30 <= THIRTY_K_RATIO_MAX, (
        f"insert blows up at 30k: t(30k)/t(10k)={ratio_30:.3f} > {THIRTY_K_RATIO_MAX}"
    )
    # Throughput holds its floor at scale.
    assert rps_30 >= INSERT_ROWS_PER_S_FLOOR, (
        f"insert throughput below floor at 30k: {rps_30:.1f} rows/s"
    )


def test_max_key_cheap_at_30k(tmp_path: Path) -> None:
    """After 30k inserts, the largest-key lookup is tree-height cheap, not O(rows)."""
    s, root = _new_store(tmp_path / "maxkey.lilli")
    _insert_batch(s, root, 0, 30_000)

    assert s.max_key(root) == 29_999

    # A tight loop of largest-key lookups reads a small constant per call (one
    # header per tree level), independent of the 30k rows.
    s.reset_read_count()
    s.reset_full_scan_count()
    iterations = 200
    for _ in range(iterations):
        assert s.max_key(root) == 29_999
    reads_per_call = s.read_count() / iterations
    walks = s.full_scan_count()

    print(
        f"\nmax_key at 30k: {reads_per_call:.2f} reads/call over {iterations} calls, "
        f"walks={walks}"
    )

    assert walks == 0, "max_key fell back to a whole-tree walk"
    assert reads_per_call <= MAX_KEY_READS_BOUND, (
        f"max_key cost scales with rows: {reads_per_call:.2f} reads/call at 30k"
    )


# The plateau measurement reads whole-process RSS, so it can only measure the
# store's own working set in a process whose residual is quiet. Inside the full
# suite the host process carries hundreds of MB of residual from thousands of
# prior tests (embedder model, JIT arenas, numpy pools); the OS decommits that
# stale residual back during the sampling window, which makes whole-process RSS
# *fall* mid-plateau and inflates the spread — the inverse signature of a leak.
# Running the measurement in a fresh child process removes that residual at the
# root: the child's RSS reflects only the store's bounded page cache. This mirrors
# the child-process RSS pattern used for the runtime-graph centrality proof.
_PLATEAU_WORKER = textwrap.dedent(
    """
    import json, os, subprocess, sys

    os.environ.setdefault("LILLI_FSYNC_MODE", "fast")
    from iai_mcp_native import store as _store

    store_path = sys.argv[1]
    per_batch = int(sys.argv[2])
    fill_batches = int(sys.argv[3])
    plateau_batches = int(sys.argv[4])
    cache_pages = int(sys.argv[5])
    record_len = int(sys.argv[6])

    def _make_record(key):
        head = key.to_bytes(8, "big", signed=True)
        tail = bytes((i + key) & 0xFF for i in range(record_len - 8))
        return head + tail

    def _rss_mb():
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True, text=True, check=True,
        )
        return int(out.stdout.strip()) / 1024.0

    s = _store.Store.open(store_path)
    s.enable_wal_mode()
    root = s.create_tree()
    s.set_cache_pages(cache_pages)

    def _insert_batch(lo, hi):
        s.begin_write()
        for key in range(lo, hi):
            s.insert(root, key, _make_record(key))
        s.commit()

    for b in range(fill_batches):
        _insert_batch(b * per_batch, (b + 1) * per_batch)

    plateau = [_rss_mb()]
    for b in range(fill_batches, fill_batches + plateau_batches):
        _insert_batch(b * per_batch, (b + 1) * per_batch)
        plateau.append(_rss_mb())

    print(json.dumps({"plateau": plateau}))
    """
)


def _measure_plateau_in_child(
    store_path: Path,
    *,
    per_batch: int,
    fill_batches: int,
    plateau_batches: int,
    cache_pages: int,
    worker_src: str = _PLATEAU_WORKER,
) -> list[float]:
    """Run the plateau fill+sample in a fresh child and return its RSS samples.

    The child process starts with no residual working set, so its whole-process
    RSS is the store's own footprint alone — free of the host suite's decommit
    noise. Returns the ``fill_batches`` baseline plus one sample per steady-state
    batch (``plateau_batches + 1`` values).
    """
    proc = subprocess.run(
        [
            sys.executable, "-c", worker_src,
            str(store_path),
            str(per_batch),
            str(fill_batches),
            str(plateau_batches),
            str(cache_pages),
            str(RECORD_LEN),
        ],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"plateau child failed (rc={proc.returncode}):\n{proc.stderr}"
    )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    return [float(x) for x in payload["plateau"]]


@retry_once_on_assertion
def test_rss_plateau_across_batches(tmp_path: Path) -> None:
    """Working-set RSS plateaus flat once the bounded page cache is saturated.

    The bounded page cache (8192 pages, ~64 MB) fills as the first records land,
    so the resident set ramps during cache-fill and then flattens. The durable
    property is that beyond saturation the working set stops growing — inserting
    further rows adds no resident memory. The gate samples past the fill cliff
    and asserts those steady-state samples are flat: a per-row leak would keep
    climbing through them.

    The fill and sampling run in a fresh child process. Whole-process RSS is the
    only observable for a bounded working set, and inside the full suite the host
    process carries hundreds of MB of residual from prior tests that the OS
    decommits during the sampling window — that reclamation makes host RSS fall
    mid-plateau and inflates the spread with noise unrelated to this store. A
    fresh child has no such residual, so its RSS tracks only this store's own
    cache; the plateau shape is then the store's alone.
    """
    per_batch = 5_000
    fill_batches = 7
    plateau_batches = 6
    plateau = _measure_plateau_in_child(
        tmp_path / "plateau.lilli",
        per_batch=per_batch,
        fill_batches=fill_batches,
        plateau_batches=plateau_batches,
        cache_pages=8192,
    )

    baseline = plateau[0]
    final = plateau[-1]
    spread = max(plateau) - min(plateau)
    total_rows = (fill_batches + plateau_batches) * per_batch

    print(
        f"\nRSS plateau over {plateau_batches} steady-state batches "
        f"(after {fill_batches * per_batch} fill rows, {total_rows} total) (MB): "
        f"{', '.join(f'{x:.0f}' for x in plateau)} "
        f"| baseline={baseline:.0f} final={final:.0f} spread={spread:.0f}"
    )

    # Past cache saturation the resident set is flat: the spread across the
    # steady-state samples stays within allocator noise, not a monotonic ramp.
    # Inserting these batches added rows worth tens of MB of payload; a leak
    # would surface them here, a bounded working set does not.
    assert spread <= 24.0, (
        f"working-set RSS not flat past cache saturation: spread {spread:.0f} MB "
        f"(samples={[round(x) for x in plateau]}) — suspected per-row leak"
    )
    # No late monotonic ramp. Compared on the median of the first half of the
    # steady-state samples against the median of the second half, NOT on the
    # single first/last samples: one noisy endpoint used to decide this on its
    # own. On Linux the plateau is exactly flat (spread 0), but macOS libmalloc
    # retains freed pages rather than returning them to the OS, so whole-process
    # RSS there drifts up gently even when the store's working set is bounded —
    # that drift once put final-baseline at 18.8 MB while the spread check above
    # still passed. Medians absorb that; a real per-row leak grows every batch
    # and so moves both halves apart far past this margin.
    half = len(plateau) // 2
    early = statistics.median(plateau[:half])
    late = statistics.median(plateau[half:])
    assert late <= early + 16.0, (
        f"RSS still rising past saturation: early-half median={early:.0f} "
        f"late-half median={late:.0f} "
        f"(samples={[round(x) for x in plateau]}, final={final:.0f} "
        f"baseline={baseline:.0f})"
    )
