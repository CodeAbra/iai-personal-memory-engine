"""Tests for the reinforce-defer measurement shim used in the latency probe and
accuracy harness.

Proves that:
  1. With the shim applied, a measured recall issues ZERO synchronous
     reinforce_record / boost_edges calls — mirroring the daemon's deferred
     reinforce behaviour, so the probe measures production-representative
     recall latency, not the sync-reinforce bench artifact.
  2. Without the shim (control assertion), the same recall DOES call
     reinforce_record at least once — proving the shim, not something else,
     is what removes the synchronous write.  This is the non-silent-no-op
     proof.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO / "src"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(text: str = "alice test") -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )


def _open_store_without_daemon(tmp_path: Path) -> MemoryStore:
    """Open a MemoryStore with no daemon — _reinforce_queue is None."""
    store = MemoryStore(path=tmp_path)
    # Ensure there is no background reinforce queue (non-daemon context).
    assert store._reinforce_queue is None, (
        "_reinforce_queue must be None (non-daemon context) for this test"
    )
    return store


def _run_dispatch_recall(store: MemoryStore) -> dict:
    """Run one memory_recall dispatch through the production path."""
    from iai_mcp import core
    return core.dispatch(
        store,
        "memory_recall",
        {"cue": "alice test", "session_id": "defer_shim_test"},
    )


# ---------------------------------------------------------------------------
# Task 1a: With the reinforce-defer shim applied, zero synchronous reinforce
#          writes occur during the measured recall.
# ---------------------------------------------------------------------------

def test_reinforce_defer_shim_prevents_sync_calls(tmp_path, monkeypatch):
    """A dispatch recall with the defer shim issues zero synchronous reinforce calls.

    Applies the same shim the latency probe installs — binding
    store.queue_reinforce to a no-op so the sync fallback (which fires when
    _reinforce_queue is None) never executes.  A spy counter on reinforce_record
    must record zero calls.
    """
    store = _open_store_without_daemon(tmp_path)

    # Insert a few records so the recall path has something to reinforce.
    for i in range(4):
        store.insert(_make_record(text=f"alice seeded record {i}"))

    reinforce_call_count: list[int] = [0]

    original_reinforce = store.__class__.reinforce_record

    def _counting_reinforce(self, record_id, anchor_id=None, edge_type="hebbian",
                            delta=0.1, *, is_retrieval=False):
        reinforce_call_count[0] += 1
        return original_reinforce(self, record_id, anchor_id=anchor_id,
                                  edge_type=edge_type, delta=delta,
                                  is_retrieval=is_retrieval)

    monkeypatch.setattr(store.__class__, "reinforce_record", _counting_reinforce)

    # Apply the reinforce-defer shim: neutralise queue_reinforce so the sync
    # fallback (queue is None → call reinforce_record per id) never fires.
    # This mirrors the daemon: the daemon has a live queue so queue_reinforce
    # returns immediately without touching reinforce_record on the recall thread.
    store.queue_reinforce = lambda *_a, **_kw: None  # type: ignore[method-assign]

    try:
        _run_dispatch_recall(store)
    finally:
        store.close()

    assert reinforce_call_count[0] == 0, (
        f"Expected 0 synchronous reinforce_record calls with the shim applied, "
        f"got {reinforce_call_count[0]}.  The shim is not preventing the sync fallback."
    )


# ---------------------------------------------------------------------------
# Task 1b: Control — WITHOUT the shim, the same recall DOES call reinforce_record
#          at least once.  Proves the shim is not a silent no-op.
# ---------------------------------------------------------------------------

def test_without_defer_shim_sync_reinforce_fires(tmp_path, monkeypatch):
    """Control: without the defer shim, a recall in a non-daemon context calls
    reinforce_record synchronously (the sync fallback path fires because
    _reinforce_queue is None).

    This confirms the shim in the companion test is what removes the sync
    write, not some other code path change.
    """
    store = _open_store_without_daemon(tmp_path)

    # Insert records so there are hits to reinforce.
    for i in range(4):
        store.insert(_make_record(text=f"alice control record {i}"))

    reinforce_call_count: list[int] = [0]
    original_reinforce = store.__class__.reinforce_record

    def _counting_reinforce(self, record_id, anchor_id=None, edge_type="hebbian",
                            delta=0.1, *, is_retrieval=False):
        reinforce_call_count[0] += 1
        return original_reinforce(self, record_id, anchor_id=anchor_id,
                                  edge_type=edge_type, delta=delta,
                                  is_retrieval=is_retrieval)

    monkeypatch.setattr(store.__class__, "reinforce_record", _counting_reinforce)

    # No shim installed — queue_reinforce uses the sync fallback because
    # _reinforce_queue is None.

    try:
        _run_dispatch_recall(store)
    finally:
        store.close()

    assert reinforce_call_count[0] >= 1, (
        "Control failed: expected at least one synchronous reinforce_record call "
        "WITHOUT the defer shim, but got zero.  Either the recall returned no hits "
        "or the sync fallback path changed — verify the test store has active records."
    )


# ---------------------------------------------------------------------------
# Task 1c: Shim is host-only — it does not affect recall correctness (hits
#          returned are not empty when the store has records).
# ---------------------------------------------------------------------------

def test_defer_shim_does_not_suppress_recall_results(tmp_path, monkeypatch):
    """Applying the defer shim does not prevent the recall from returning results.

    Ensures the shim is scoped to the reinforce write path and does not suppress
    the recall response itself.
    """
    store = _open_store_without_daemon(tmp_path)

    for i in range(4):
        store.insert(_make_record(text=f"alice result check {i}"))

    # Apply the defer shim.
    store.queue_reinforce = lambda *_a, **_kw: None  # type: ignore[method-assign]

    try:
        response = _run_dispatch_recall(store)
    finally:
        store.close()

    # The dispatch must still return a well-formed response dict with hits.
    assert isinstance(response, dict), "dispatch must return a dict"
    assert "hits" in response, "dispatch response must contain 'hits'"
    assert len(response["hits"]) > 0, (
        "dispatch returned no hits despite the store having active records — "
        "the shim may have suppressed more than just the reinforce path"
    )
