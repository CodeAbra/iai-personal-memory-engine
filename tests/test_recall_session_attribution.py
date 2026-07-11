from __future__ import annotations

from datetime import datetime


from iai_mcp.capture import capture_turn
from iai_mcp.core import dispatch
from tests._helpers import make_tmp_store

def test_recall_hit_carries_session_id_and_captured_at(tmp_path):
    store = make_tmp_store(tmp_path)

    result = capture_turn(
        store,
        cue="known user line mk59",
        text="known user line mk59 distinctive text",
        tier="episodic",
        session_id="sess-A1-mk59",
        role="user",
    )
    assert result["status"] == "inserted", f"capture failed: {result}"

    recall = dispatch(
        store,
        "memory_recall",
        {"cue": "known user line mk59"},
    )
    hits = recall.get("hits", [])
    assert hits, "expected at least one recall hit"

    top = hits[0]

    assert top.get("session_id") == "sess-A1-mk59", (
        f"hit session_id not surfaced (got {top.get('session_id')!r}); "
        "session_id missing from _hit_to_json / MemoryHit"
    )

    captured_at = top.get("captured_at")
    assert captured_at is not None, (
        "hit captured_at is null; it must be populated from record.created_at"
    )
    dt = datetime.fromisoformat(captured_at)
    assert dt.tzinfo is not None, f"captured_at must be timezone-aware, got {captured_at!r}"
