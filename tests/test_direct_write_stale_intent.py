"""Daemon-free write must not be wedged by a crashed daemon's stale intent.

A SIGKILL'd daemon leaves a `.consolidation-pending` file on disk with no live
holder. The direct-write path must treat such an orphan as stale (the holder PID
is dead) and self-heal — a dead daemon must NEVER gate a write. A genuine live
in-progress consolidation (holder PID alive) must still correctly defer.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


def _intent_path(store_root: Path) -> Path:
    return store_root / "hippo" / ".consolidation-pending"


def _dead_pid() -> int:
    """Spawn a trivial process, wait for it to exit, return its now-dead PID."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_stale_intent_does_not_wedge_direct_write(hermetic_store: Path) -> None:
    from iai_mcp.direct_write import write_turn_direct

    # The hippo dir is created lazily on first open; create it so we can plant
    # the orphan intent BEFORE any HippoDB has run.
    intent = _intent_path(hermetic_store)
    intent.parent.mkdir(parents=True, exist_ok=True)

    # Orphan intent: dead holder PID + an old timestamp, no live flock.
    dead = _dead_pid()
    old_ts = time.time() - 10_000.0
    intent.write_text(f"{dead}\n{old_ts}\n")
    assert intent.exists(), "test setup: stale intent not planted"

    t0 = time.monotonic()
    result = write_turn_direct(
        store_root=hermetic_store,
        text="write under a crashed daemon's stale intent",
        session_id="stale-session",
        role="user",
        deferred_embedding=True,
    )
    elapsed = time.monotonic() - t0

    assert result["status"] == "inserted", (
        f"direct write was wedged by a stale intent instead of self-healing: {result}"
    )
    assert elapsed <= 1.5, f"stale-intent write took {elapsed:.3f} s (SLO ≤1.5 s)"
    assert not intent.exists(), (
        "orphaned consolidation intent not cleaned up after self-heal"
    )


def test_legacy_empty_intent_self_heals(hermetic_store: Path) -> None:
    """An empty (legacy, PID-less) intent with an old mtime is also reclaimed."""
    from iai_mcp.direct_write import write_turn_direct

    intent = _intent_path(hermetic_store)
    intent.parent.mkdir(parents=True, exist_ok=True)
    intent.touch()
    # Age the mtime far past the staleness ceiling.
    old = time.time() - 100_000.0
    os.utime(intent, (old, old))

    result = write_turn_direct(
        store_root=hermetic_store,
        text="write under a legacy empty stale intent",
        session_id="legacy-session",
        role="user",
        deferred_embedding=True,
    )
    assert result["status"] == "inserted", result
    assert not intent.exists(), "legacy empty stale intent not cleaned up"


def test_live_consolidation_intent_still_defers(hermetic_store: Path) -> None:
    """A live holder (PID alive, fresh ts) must still defer the direct write."""
    from iai_mcp.direct_write import write_turn_direct
    from iai_mcp.hippo import ConsolidationPendingError

    intent = _intent_path(hermetic_store)
    intent.parent.mkdir(parents=True, exist_ok=True)
    # Live holder: this process's own PID is alive, timestamp is fresh — exactly
    # what a real in-progress consolidation writes.
    intent.write_text(f"{os.getpid()}\n{time.time()}\n")

    raised = False
    try:
        write_turn_direct(
            store_root=hermetic_store,
            text="write attempt during a live consolidation",
            session_id="live-session",
            role="user",
            deferred_embedding=True,
        )
    except ConsolidationPendingError:
        raised = True

    assert raised, (
        "direct write did NOT defer under a LIVE consolidation intent — the "
        "real in-progress case must still wait, only orphans self-heal"
    )
    # A live intent is never removed by a reader.
    assert intent.exists(), "live consolidation intent was wrongly removed"
