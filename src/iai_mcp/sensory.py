"""Sensory tier: named formalization of the pre-encode capture buffer.

Thin wrapper over the existing write/read/promote mechanisms in
``iai_mcp.capture`` — no reimplementation. The awake recall path must never
import this module.
"""

from __future__ import annotations

from typing import Any

from iai_mcp.capture import (
    _LIVE_SELECT_MAX_FILES,
    _TAIL_MAX_EVENT_LINES,
    _WRITE_SIDE_MAX_EVENT_LINES,
    drain_active_live_captures,
    read_pending_live_events,
    write_deferred_event,
)
from iai_mcp.store import MemoryStore

sensory_append = write_deferred_event
sensory_pending = read_pending_live_events
sensory_promote = drain_active_live_captures

SENSORY_TAIL_MAX_LINES = _TAIL_MAX_EVENT_LINES
SENSORY_SELECT_MAX_FILES = _LIVE_SELECT_MAX_FILES
SENSORY_WRITE_SIDE_MAX_LINES = _WRITE_SIDE_MAX_EVENT_LINES

__all__ = [
    "sensory_append",
    "sensory_pending",
    "sensory_promote",
    "SENSORY_TAIL_MAX_LINES",
    "SENSORY_SELECT_MAX_FILES",
    "SENSORY_WRITE_SIDE_MAX_LINES",
    "trace_promotion",
]


def trace_promotion(
    store: MemoryStore,
    session_id: str,
    *,
    exclude_session_id: str = "-",
) -> dict[str, Any]:
    """Observe a sensory entry's raw text before and after promotion.

    Reads the pending sensory events for ``session_id``, runs promotion via
    ``sensory_promote``, then returns both snapshots for a test to assert
    the raw text landed unchanged in a resulting Hippo record. Adds no new
    read/write/promotion mechanism — a synchronous wrapper around the two
    existing calls only.
    """
    pending_before = sensory_pending(session_id=session_id)
    counts = sensory_promote(store, exclude_session_id=exclude_session_id)
    pending_after = sensory_pending(session_id=session_id)
    return {
        "pending_before": pending_before,
        "counts": counts,
        "pending_after": pending_after,
    }
