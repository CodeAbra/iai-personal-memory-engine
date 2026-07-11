from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path



def _make_user_turn(text: str = "generic user turn"):
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.0] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"session_id": "test-session", "role": "user"}],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=["role:user"],
        language="en",
    )


def test_recency_read_daemon_up_steady(hermetic_store: Path) -> None:
    from iai_mcp.store import MemoryStore, flush_record_buffer

    from iai_mcp.direct_recency import read_recent_user_turns_direct  # type: ignore[import]

    store = MemoryStore(hermetic_store)
    try:
        rec = _make_user_turn("direct recency probe text")
        store.insert(rec)
        flush_record_buffer(store)
    finally:
        store.close()

    t0 = time.monotonic()
    turns = read_recent_user_turns_direct(hermetic_store, n=5)
    elapsed = time.monotonic() - t0

    assert elapsed <= 1.5, f"direct recency read took {elapsed:.3f} s (SLO ≤1.5 s)"
    surfaces = [t.literal_surface for t in turns]
    assert any("direct recency probe text" in s for s in surfaces), (
        "stored turn not found in direct recency results"
    )


def test_recency_read_daemon_down_sigkill(hermetic_store: Path, tmp_path: Path) -> None:
    from iai_mcp.store import MemoryStore, flush_record_buffer
    from iai_mcp.direct_recency import read_recent_user_turns_direct  # type: ignore[import]

    store = MemoryStore(hermetic_store)
    try:
        rec = _make_user_turn("sigkill survival probe text")
        store.insert(rec)
        flush_record_buffer(store)
    finally:
        store.close()

    db_path = hermetic_store / "hippo" / "brain.sqlite3"
    shm_path = Path(str(db_path) + "-shm")
    if shm_path.exists():
        shm_path.unlink()

    t0 = time.monotonic()
    turns = read_recent_user_turns_direct(hermetic_store, n=5)
    elapsed = time.monotonic() - t0

    assert elapsed <= 1.5, f"daemon-down recency read took {elapsed:.3f} s (SLO ≤1.5 s)"
    surfaces = [t.literal_surface for t in turns]
    assert any("sigkill survival probe text" in s for s in surfaces), (
        "stored turn not found after simulated SIGKILL residue"
    )
