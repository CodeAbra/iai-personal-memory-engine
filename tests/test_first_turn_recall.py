from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4


from iai_mcp import core
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _seed_one_record(store: MemoryStore, text: str = "reference content") -> None:
    now = datetime.now(timezone.utc)
    rec = MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.5,
        detail_level=3,
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
    store.insert(rec)


def test_first_turn_fires_exactly_once(tmp_path, monkeypatch):
    pending = {"s1": True}

    def _load_state():
        return {"first_turn_pending": dict(pending)}

    def _save_state(state):
        fresh = state.get("first_turn_pending", {})
        pending.clear()
        pending.update(fresh)

    monkeypatch.setattr("iai_mcp.daemon_state.load_state", _load_state)
    monkeypatch.setattr("iai_mcp.daemon_state.save_state", _save_state)

    store = MemoryStore(path=tmp_path)
    _seed_one_record(store, "session one reference content")

    params = {
        "cue": "reference content",
        "session_id": "s1",
        "cue_embedding": [0.1] * EMBED_DIM,
    }
    resp1 = core.dispatch(store, "memory_recall", params)
    resp2 = core.dispatch(store, "memory_recall", params)

    assert "first_turn_recall" in resp1, f"first dispatch missing hook: {resp1.keys()}"
    assert "first_turn_recall" not in resp2, (
        f"second dispatch should NOT have hook: {resp2.keys()}"
    )


def test_first_turn_budget_capped_at_400(tmp_path, monkeypatch):
    pending = {"s2": True}
    monkeypatch.setattr(
        "iai_mcp.daemon_state.load_state",
        lambda: {"first_turn_pending": dict(pending)},
    )
    monkeypatch.setattr(
        "iai_mcp.daemon_state.save_state",
        lambda s: pending.clear(),
    )

    store = MemoryStore(path=tmp_path)
    _seed_one_record(store)

    resp = core.dispatch(store, "memory_recall", {
        "cue": "X",
        "session_id": "s2",
        "cue_embedding": [0.1] * EMBED_DIM,
    })
    ftr = resp.get("first_turn_recall")
    assert ftr is not None, f"first_turn_recall missing: {resp.keys()}"
    assert ftr.get("budget_tokens", 0) <= 400, f"budget too high: {ftr}"


def test_daemon_unreachable_falls_back_silently(tmp_path, monkeypatch):
    def _boom():
        raise RuntimeError("synthetic daemon_state failure")

    monkeypatch.setattr("iai_mcp.daemon_state.load_state", _boom)

    store = MemoryStore(path=tmp_path)
    _seed_one_record(store)

    resp = core.dispatch(store, "memory_recall", {
        "cue": "X",
        "session_id": "s3",
        "cue_embedding": [0.1] * EMBED_DIM,
    })
    assert "hits" in resp
    assert "first_turn_recall" not in resp


def test_first_turn_emits_event(tmp_path, monkeypatch):
    from iai_mcp.events import query_events

    pending = {"s4": True}
    monkeypatch.setattr(
        "iai_mcp.daemon_state.load_state",
        lambda: {"first_turn_pending": dict(pending)},
    )
    monkeypatch.setattr(
        "iai_mcp.daemon_state.save_state",
        lambda s: pending.clear(),
    )

    store = MemoryStore(path=tmp_path)
    _seed_one_record(store)

    core.dispatch(store, "memory_recall", {
        "cue": "something",
        "session_id": "s4",
        "cue_embedding": [0.1] * EMBED_DIM,
    })

    events = query_events(store, kind="first_turn_recall", limit=10)
    assert len(events) >= 1, "first_turn_recall event should have been emitted"


def test_input_length_clamp_2000(tmp_path, monkeypatch):
    pending = {"s5": True}
    monkeypatch.setattr(
        "iai_mcp.daemon_state.load_state",
        lambda: {"first_turn_pending": dict(pending)},
    )
    monkeypatch.setattr(
        "iai_mcp.daemon_state.save_state",
        lambda s: pending.clear(),
    )

    store = MemoryStore(path=tmp_path)
    _seed_one_record(store)

    huge_cue = "X" * 5000

    seen_cues: list[str] = []
    from iai_mcp import retrieve as _retrieve
    orig = _retrieve.recall

    def _spy(*args, **kwargs):
        cue = kwargs.get("cue_text", "")
        if "first-turn" not in cue[:20]:
            seen_cues.append(cue)
        return orig(*args, **kwargs)

    monkeypatch.setattr("iai_mcp.retrieve.recall", _spy)

    core.dispatch(store, "memory_recall", {
        "cue": huge_cue,
        "session_id": "s5",
        "cue_embedding": [0.1] * EMBED_DIM,
    })

    assert any(len(c) <= 2000 for c in seen_cues), (
        f"no clamped cue observed; len spread: {[len(c) for c in seen_cues]}"
    )
