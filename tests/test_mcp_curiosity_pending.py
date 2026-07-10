from __future__ import annotations

from uuid import uuid4


from iai_mcp.core import dispatch
from iai_mcp.events import write_event
from iai_mcp.store import MemoryStore


def test_curiosity_pending_empty_store(tmp_path):
    store = MemoryStore(path=tmp_path)
    out = dispatch(store, "curiosity_pending", {})
    assert out == {"questions": [], "count": 0}


def test_curiosity_pending_returns_unresolved(tmp_path):
    store = MemoryStore(path=tmp_path)
    ids = [str(uuid4()) for _ in range(3)]
    for i, qid in enumerate(ids):
        write_event(
            store,
            kind="curiosity_question",
            data={
                "question_id": qid,
                "text": f"q{i}",
                "tier": "question",
                "entropy": 0.9,
                "turn": i,
                "triggered_by": [],
            },
            severity="info",
            session_id="s1",
        )
    out = dispatch(store, "curiosity_pending", {})
    assert out["count"] == 3
    assert len(out["questions"]) == 3
    for q in out["questions"]:
        assert "id" in q
        assert "text" in q
        assert "tier" in q
        assert "entropy" in q
        assert "triggered_by_record_ids" in q


def test_curiosity_pending_filters_session(tmp_path):
    store = MemoryStore(path=tmp_path)
    write_event(
        store,
        kind="curiosity_question",
        data={"question_id": str(uuid4()), "text": "a", "tier": "question",
              "entropy": 0.9, "turn": 1, "triggered_by": []},
        severity="info",
        session_id="s1",
    )
    write_event(
        store,
        kind="curiosity_question",
        data={"question_id": str(uuid4()), "text": "b", "tier": "question",
              "entropy": 0.9, "turn": 1, "triggered_by": []},
        severity="info",
        session_id="s2",
    )
    out = dispatch(store, "curiosity_pending", {"session_id": "s1"})
    assert out["count"] == 1
    assert out["questions"][0]["text"] == "a"


def test_curiosity_pending_excludes_resolved(tmp_path):
    store = MemoryStore(path=tmp_path)
    qid = str(uuid4())
    write_event(
        store,
        kind="curiosity_question",
        data={"question_id": qid, "text": "resolved-q", "tier": "question",
              "entropy": 0.9, "turn": 1, "triggered_by": []},
        severity="info",
        session_id="s1",
    )
    other_qid = str(uuid4())
    write_event(
        store,
        kind="curiosity_question",
        data={"question_id": other_qid, "text": "still-open", "tier": "question",
              "entropy": 0.9, "turn": 1, "triggered_by": []},
        severity="info",
        session_id="s1",
    )
    write_event(
        store,
        kind="curiosity_resolved",
        data={"question_id": qid},
        severity="info",
        session_id="s1",
    )
    out = dispatch(store, "curiosity_pending", {})
    assert out["count"] == 1
    assert out["questions"][0]["text"] == "still-open"
