"""Force the process allocator to return freed pages to the OS.

Called at the dispatch-loop tail after a heavy sleep step has finished and
released its store/index locks.  A single best-effort action is taken:

  1. ``gc.collect()`` — reclaim Python cycles holding large transients.

Resident-set size is read with the *current* RSS primitive
(``psutil.Process().memory_info().rss``), not peak RSS — peak is monotonic and
can never show a reclaim. The step is wrapped fail-soft: the helper must never
raise on any platform, because it runs after a successful step whose progress
is already persisted.

**Why the pyarrow/mimalloc arena is not targeted here:**
The dominant allocator for the heavy sleep steps is mimalloc, statically linked
into pyarrow.  Two primitives that were previously attempted both measure ~0
reclaim against those arenas:

- ``pa.default_memory_pool().release_unused()`` — the tracked arrow pool
  counter is decoupled from RSS in this build; measured +0.2 MB (noise).
- ``malloc_zone_pressure_relief(NULL, 0)`` (macOS only) — iterates the system
  libmalloc zones; mimalloc-in-pyarrow does NOT register itself as a libmalloc
  zone, so no mimalloc pages are reached.

``mi_collect(true)`` would be the correct primitive, but it is not exported as
a public symbol by pyarrow's bundled mimalloc build (verified: nm over all
pyarrow .so files finds no ``mi_collect``).

The actual RSS return after a heavy step comes from mimalloc decommitting pages
on free, governed by the ``MIMALLOC_ALLOW_DECOMMIT=1`` /
``MIMALLOC_DECOMMIT_DELAY=0`` / ``MIMALLOC_SEGMENT_DECOMMIT_DELAY=0``
environment variables set in the launchd plist.  Those env vars take effect at
process spawn and require no in-process call.

``zone_reclaimed_mb`` is kept at 0.0 in the telemetry dict for schema
stability (existing event readers expect the key).
"""
from __future__ import annotations

import gc
import logging
import time

logger = logging.getLogger(__name__)


def _current_rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001 -- psutil flakiness must not crash the cycle
        return 0


def _step_memory_relief(label: str = "") -> dict:
    """Release Python cycles and report the RSS reclaim.

    Returns a telemetry dict with six numeric keys: ``rss_before_mb``,
    ``rss_after_mb``, ``rss_delta_mb`` (clamped at 0, for dashboards),
    ``rss_signed_delta_mb`` (unclamped; negative means RSS grew during relief),
    ``zone_reclaimed_mb`` (always 0.0 — kept for schema stability),
    ``elapsed_ms``. Never raises.
    """
    t0 = time.monotonic()
    rss_before = _current_rss_bytes()

    try:
        gc.collect()
    except Exception as exc:  # noqa: BLE001 -- collection must not crash the cycle
        logger.debug("gc.collect failed for step %s: %s", label, exc)

    rss_after = _current_rss_bytes()
    elapsed_ms = (time.monotonic() - t0) * 1000.0

    rss_before_mb = rss_before / 1e6
    rss_after_mb = rss_after / 1e6
    rss_signed = rss_before_mb - rss_after_mb
    return {
        "rss_before_mb": round(rss_before_mb, 3),
        "rss_after_mb": round(rss_after_mb, 3),
        "rss_delta_mb": round(max(0.0, rss_signed), 3),
        "rss_signed_delta_mb": round(rss_signed, 3),
        "zone_reclaimed_mb": 0.0,
        "elapsed_ms": round(elapsed_ms, 3),
    }
