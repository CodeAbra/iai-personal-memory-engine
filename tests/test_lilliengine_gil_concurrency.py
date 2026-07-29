"""Concurrency regression: a long engine scan must not hold the GIL.

When the engine holds the GIL during store I/O, thread A's scan starves all
other Python threads for its full Rust execution window. A counter thread that
increments a shared integer sees large gaps (near the scan duration) when the
GIL is held, because it cannot run at all until the Rust scan returns.

After the fix, the engine releases the GIL during the pure-Rust store I/O.
Thread A yields the GIL when it enters the Rust kernel; the counter thread
runs continuously, and gap durations stay near the normal OS scheduling jitter.

Proof: the maximum GIL-hold gap measured by the counter thread during a long
scan must be less than a bound. When the GIL is held, the max gap approaches
the scan-iteration duration (tens to hundreds of ms). When the GIL is released,
the max gap stays near the OS preemption window (< sys.getswitchinterval()).
"""
from __future__ import annotations

from tests.conftest_shared import retry_once_on_assertion

import os
import sys
import threading
import time

import pytest

os.environ.setdefault("LILLI_FSYNC_MODE", "fast")


# ---------------------------------------------------------------------------
# Skip guard — native module must be installed
# ---------------------------------------------------------------------------


def _native_available() -> bool:
    try:
        from iai_mcp_native import engine  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _native_available(),
    reason="iai_mcp_native.engine submodule not installed (build the native wheel)",
)

# ---------------------------------------------------------------------------
# Schema / SQL constants
# ---------------------------------------------------------------------------

_DDL = (
    "CREATE TABLE items ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "key TEXT NOT NULL, "
    "value TEXT NOT NULL)"
)
_INSERT = "INSERT INTO items (key, value) VALUES (?, ?)"
_SELECT_ALL = "SELECT id, key, value FROM items"

# Row count: enough for a single `run_execute` scan to take 50-150 ms.
# The counter thread's gap during that window is the GIL-hold signal.
_POPULATE_ROWS = 80_000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_engine(path: str):
    from iai_mcp_native import engine

    return engine.Connection.open(path, 384)


def _make_store(path: str, n: int = _POPULATE_ROWS) -> None:
    """Create and populate a lilli store, then close it."""
    conn = _open_engine(path)
    conn.execute(_DDL)
    batch = [(f"key_{i}", f"value_{i}") for i in range(n)]
    conn.executemany(_INSERT, batch)
    conn.close()


# ---------------------------------------------------------------------------
# GIL-gap probe via counter thread
# ---------------------------------------------------------------------------


def _max_gil_gap_during_scan(
    path: str,
    scan_wall_seconds: float = 1.0,
) -> tuple[float, float]:
    """Measure the maximum GIL gap seen by a counter thread while A scans.

    Thread A opens `path` and loops full-table SELECTs for `scan_wall_seconds`.
    A counter thread increments a shared counter as fast as it can, recording
    the wall time between each successful increment. The maximum gap between
    increments is the proxy for the longest GIL-hold during A's scan window.

    Returns (max_gap_seconds, baseline_max_gap_seconds) where baseline is
    measured in a quiet period before A starts.

    When the GIL is held during store I/O: A's full-table scan holds the GIL
    for its entire Rust execution window; the counter thread cannot run during
    that window and the max gap approaches the scan-iteration duration.

    When the GIL is released: A yields the GIL on entry to the Rust kernel;
    the counter thread runs continuously; the max gap stays near OS jitter.
    """
    counter: list[int] = [0]
    run_flag: list[bool] = [True]
    gaps: list[float] = []

    def counter_thread() -> None:
        last = time.perf_counter()
        while run_flag[0]:
            now = time.perf_counter()
            gap = now - last
            gaps.append(gap)
            counter[0] += 1
            last = now

    ct = threading.Thread(target=counter_thread, daemon=True)
    ct.start()

    # Baseline: collect gaps with no engine activity.
    time.sleep(0.2)
    baseline_gaps = list(gaps)
    baseline_max = max(baseline_gaps) if baseline_gaps else 0.0
    gaps.clear()

    # Scan phase.
    conn = _open_engine(path)
    deadline = time.perf_counter() + scan_wall_seconds
    while time.perf_counter() < deadline:
        conn.execute(_SELECT_ALL).fetchall()
    conn.close()

    run_flag[0] = False
    ct.join(timeout=2.0)

    scan_max = max(gaps) if gaps else 0.0
    return scan_max, baseline_max


# ---------------------------------------------------------------------------
# The regression test
# ---------------------------------------------------------------------------


@retry_once_on_assertion
def test_engine_scan_releases_gil(tmp_path):
    """The engine scan releases the GIL so other threads are not starved.

    Measured via a counter thread that records the maximum gap between
    increments. When the GIL is held during the Rust scan, the counter thread
    cannot increment for the full scan duration (tens to hundreds of ms per
    iteration on a large table). When the GIL is released, the max gap stays
    near the OS preemption interval (< 10 ms on a loaded box).

    The bound: max gap during scan < 30 ms.

    This bound is RED on a GIL-held engine where one `run_execute` iteration
    takes ~80-150 ms on 80 k rows — the counter thread is blocked for that
    full duration. It is GREEN on a GIL-releasing engine where the gap is
    bounded by the OS scheduler (typically 1-5 ms), with 30 ms absorbing
    worst-case scheduling jitter.

    The GIL switch-interval (`sys.getswitchinterval()` = 5 ms default) is
    the reference lower bound for the baseline; the scan gap must be clearly
    above it to be RED and clearly below 30 ms to be reliably GREEN.
    """
    path = str(tmp_path / "scan.lilli")
    _make_store(path)

    # Warm up the store so the first scan uses the page cache.
    warmup_conn = _open_engine(path)
    warmup_conn.execute(_SELECT_ALL).fetchall()
    warmup_conn.close()

    scan_max, baseline_max = _max_gil_gap_during_scan(path)

    # The bound encodes GIL-release: with the GIL held for ~100+ ms per scan
    # iteration, the counter thread sees max gaps >> 30 ms. With the GIL
    # released, the counter thread runs unimpeded and max gap << 30 ms.
    bound_seconds = 0.030  # 30 ms hard ceiling

    assert scan_max < bound_seconds, (
        f"counter thread saw a {scan_max * 1000:.1f} ms GIL gap during the engine "
        f"scan (baseline max gap: {baseline_max * 1000:.2f} ms, switch interval: "
        f"{sys.getswitchinterval() * 1000:.1f} ms). "
        f"Expected < {bound_seconds * 1000:.0f} ms — the GIL appears to be held "
        f"during store I/O, starving other Python threads."
    )
