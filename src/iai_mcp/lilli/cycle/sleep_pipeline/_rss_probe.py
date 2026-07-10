"""Best-effort resident-set snapshot for per-step memory attribution.

Samples the process's own RSS, a ``vmmap -summary`` region count + VM_ALLOCATE
size, and the pyarrow + numba allocator counters. Every path is best-effort:
``capture_step_snapshot`` wraps its whole body so it can never raise into the
caller, and each individual probe degrades to a sentinel (``None`` for the
darwin-only / parse-dependent vmmap fields, ``-1`` for the numeric allocator
fields) rather than propagating an exception. This lets the sleep-pipeline emit
hooks attach the snapshot to their event dicts without any risk of aborting a
sleep step.

The vmmap region count + VM_ALLOCATE size stand in as a page-reclaim proxy:
this build's allocator (mimalloc, statically linked into pyarrow) exports no
process-callable stats symbol, so its decommit behaviour is observed indirectly
through the mapped-region count and the VM_ALLOCATE virtual size, which both
drop together when the allocator returns pages to the OS. VM_ALLOCATE is read
from the VIRTUAL size column to match the diagnostic-harness baseline.

The vmmap subprocess is the only expensive sample (~tens of milliseconds), so
it is opt-in per call: the per-step hooks omit it (RSS + pool + NRT are
microsecond-cheap), and the continuous watchdog breadcrumb samples the region
count on its own throttled cadence.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Any


def _own_rss_kib() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss // 1024)
    except Exception:  # noqa: BLE001 -- psutil flakiness must never raise here
        return None


def _humansize_to_kib(token: str) -> int:
    """Parse a vmmap size token (``812.1M`` / ``1.2G`` / ``16K`` / bare bytes) to KiB.

    May raise on a malformed token; every caller guards the call.
    """
    token = token.strip()
    if not token:
        raise ValueError("empty size token")
    suffix = token[-1]
    if suffix == "G":
        return int(float(token[:-1]) * 1024 * 1024)
    if suffix == "M":
        return int(float(token[:-1]) * 1024)
    if suffix == "K":
        return int(float(token[:-1]))
    # Bare byte count.
    return int(float(token) // 1024)


def _parse_vmmap_summary(text: str) -> tuple[int | None, int | None]:
    """Read the region count + VM_ALLOCATE virtual size from ``vmmap -summary`` text.

    Region count is the last field of the all-caps ``TOTAL`` line. The
    mixed-case ``Total`` line carries no region count and is ignored. VM_ALLOCATE
    is read from its first (VIRTUAL) size column. Malformed tokens leave that
    field ``None``; this function never raises.
    """
    region_count: int | None = None
    vm_allocate_kib: int | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("TOTAL"):
            try:
                region_count = int(line.split()[-1])
            except (IndexError, ValueError):
                pass
        elif line.startswith("VM_ALLOCATE"):
            try:
                vm_allocate_kib = _humansize_to_kib(line.split()[1])
            except (IndexError, ValueError):
                pass
    return region_count, vm_allocate_kib


def _vmmap_summary_counts() -> tuple[int | None, int | None]:
    """Run ``/usr/bin/vmmap -summary`` for this process and parse the counts.

    Non-darwin or any subprocess failure returns ``(None, None)``.
    """
    if sys.platform != "darwin":
        return (None, None)
    try:
        proc = subprocess.run(
            ["/usr/bin/vmmap", "-summary", str(os.getpid())],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError):
        return (None, None)
    if proc.returncode != 0:
        return (None, None)
    return _parse_vmmap_summary(proc.stdout)


def _allocator_stats() -> dict[str, int]:
    """Best-effort pyarrow pool + numba NRT counters. Missing values are ``-1``."""
    out = {
        "pa_pool_bytes": -1,
        "pa_pool_max": -1,
        "numba_nrt_alloc_count": -1,
        "numba_nrt_free_count": -1,
    }
    try:
        import pyarrow as pa

        mp = pa.default_memory_pool()
        out["pa_pool_bytes"] = int(mp.bytes_allocated())
        out["pa_pool_max"] = int(mp.max_memory())
    except Exception:  # noqa: BLE001 -- pool stats are best-effort
        pass
    try:
        from numba.core.runtime import nrt

        stats = nrt.rtsys.get_allocation_stats()
        out["numba_nrt_alloc_count"] = int(stats.alloc)
        out["numba_nrt_free_count"] = int(stats.free)
    except Exception:  # noqa: BLE001 -- NRT stats may be disabled or cold; fall back to -1
        pass
    return out


def capture_step_snapshot(*, include_vmmap: bool = True) -> dict[str, Any]:
    """Return a best-effort resident-set + allocator snapshot. Never raises.

    ``include_vmmap`` gates the ``vmmap`` subprocess: the per-step emit hooks
    pass ``False`` (RSS + pool + NRT only, all microsecond-cheap), while the
    operator snapshot passes ``True`` for the full region/VM_ALLOCATE picture.
    The vmmap fields are present either way (``None`` when not sampled) so
    callers can read them uniformly.
    """
    try:
        region_count: int | None = None
        vm_allocate_kib: int | None = None
        if include_vmmap:
            region_count, vm_allocate_kib = _vmmap_summary_counts()
        snap: dict[str, Any] = {
            "rss_kib": _own_rss_kib(),
            "vmmap_region_count": region_count,
            "vm_allocate_kib": vm_allocate_kib,
            "t": time.monotonic(),
        }
        snap.update(_allocator_stats())
        return snap
    except Exception:  # noqa: BLE001 -- the snapshot must never raise into a caller
        return {
            "rss_kib": None,
            "vmmap_region_count": None,
            "vm_allocate_kib": None,
            "pa_pool_bytes": -1,
            "pa_pool_max": -1,
            "numba_nrt_alloc_count": -1,
            "numba_nrt_free_count": -1,
            "t": time.monotonic(),
        }
