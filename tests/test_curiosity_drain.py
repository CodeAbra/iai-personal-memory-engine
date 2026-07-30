from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _rec(vec=None, tags=None):
    vec = vec or [1.0] + [0.0] * (EMBED_DIM - 1)
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="r",
        aaak_index="",
        embedding=vec,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=list(tags or []),
        language="en",
    )


def _write_deferred(store, rid, *, entropy, turn, session_id, cue="cue"):
    from iai_mcp.events import write_event

    data = {
        "hit_ids": [str(rid)],
        "cue": cue,
        "session_id": session_id,
    }
    if entropy is not None:
        data["entropy"] = float(entropy)
    if turn is not None:
        data["turn"] = int(turn)
    write_event(
        store,
        kind="deferred_curiosity_input",
        data=data,
        severity="info",
        session_id=session_id,
    )


# --- entropy normalization -------------------------------------------------

def test_compute_entropy_normalized_to_unit_range():
    from iai_mcp.curiosity import compute_entropy

    # A uniform distribution is maximal entropy at any arity: normalized to 1.0
    # regardless of how many outcomes. The pre-fix raw-bits form returned
    # log2(k) here (~1.585 for k=3), which spuriously exceeded ENTROPY_HIGH.
    assert abs(compute_entropy([0.5, 0.5, 0.5]) - 1.0) < 1e-6
    assert abs(compute_entropy([0.2, 0.2, 0.2, 0.2, 0.2]) - 1.0) < 1e-6

    # Every value stays within the calibrated [0, 1] band.
    for scores in ([0.9, 0.1], [0.6, 0.3, 0.1], [0.4, 0.3, 0.2, 0.1]):
        e = compute_entropy(scores)
        assert 0.0 <= e <= 1.0


def test_compute_entropy_peaked_many_hits_below_high_tier():
    from iai_mcp.curiosity import ENTROPY_HIGH, compute_entropy

    # A strongly peaked distribution over many outcomes must stay below the
    # direct-question tier so the inline/silent tiers remain reachable. Under
    # the old un-normalized form the extra outcomes inflated the sum over 0.9.
    e = compute_entropy([0.95, 0.01, 0.01, 0.01, 0.01, 0.01])
    assert e < ENTROPY_HIGH


# --- deferred-input drain --------------------------------------------------

def test_drain_fires_questions_and_is_idempotent(tmp_path):
    from iai_mcp.curiosity import drain_deferred_curiosity, pending_questions

    store = MemoryStore(path=tmp_path)
    r = _rec()
    store.insert(r)

    _write_deferred(store, r.id, entropy=0.95, turn=1, session_id="sA")
    _write_deferred(store, r.id, entropy=0.95, turn=1, session_id="sB")

    first = drain_deferred_curiosity(store)
    assert first == {"drained": 2, "fired": 2, "resolved": 0}
    assert len(pending_questions(store)) == 2

    # A second drain has advanced past every input already consumed: no
    # re-scan, no double-fire.
    second = drain_deferred_curiosity(store)
    assert second == {"drained": 0, "fired": 0, "resolved": 0}
    assert len(pending_questions(store)) == 2


def test_drain_skips_legacy_input_without_entropy(tmp_path):
    from iai_mcp.curiosity import drain_deferred_curiosity, pending_questions

    store = MemoryStore(path=tmp_path)
    r = _rec()
    store.insert(r)

    # Written before entropy/turn were recorded: consumed, never fired.
    _write_deferred(store, r.id, entropy=None, turn=None, session_id="sA")

    result = drain_deferred_curiosity(store)
    assert result["drained"] == 1
    assert result["fired"] == 0
    assert pending_questions(store) == []


def test_drain_below_threshold_input_not_fired(tmp_path):
    from iai_mcp.curiosity import drain_deferred_curiosity, pending_questions

    store = MemoryStore(path=tmp_path)
    r = _rec()
    store.insert(r)

    _write_deferred(store, r.id, entropy=0.1, turn=1, session_id="sA")

    result = drain_deferred_curiosity(store)
    assert result["drained"] == 1
    assert result["fired"] == 0
    assert pending_questions(store) == []


def test_drain_creates_curiosity_bridge_edges(tmp_path):
    # The knob-tuner keys off the curiosity_bridge edge ratio; before the drain
    # existed those edges were never produced, pinning the ratio at 0.
    from iai_mcp.curiosity import drain_deferred_curiosity

    store = MemoryStore(path=tmp_path)
    r = _rec()
    store.insert(r)
    _write_deferred(store, r.id, entropy=0.95, turn=1, session_id="sA")

    drain_deferred_curiosity(store)

    edges = store.db.open_table("edges")
    assert edges.count_rows("edge_type = 'curiosity_bridge'") >= 1


# --- session-scoped cooldown ----------------------------------------------

def test_confident_recall_resolves_the_question(tmp_path):
    # The engine's only path to closing a question: the same cue later comes
    # back unambiguous, so the recorded ambiguity no longer exists.
    from iai_mcp.curiosity import drain_deferred_curiosity, pending_questions

    store = MemoryStore(path=tmp_path)
    r = _rec()
    store.insert(r)

    _write_deferred(store, r.id, entropy=0.95, turn=1, session_id="sA", cue="which db")
    assert drain_deferred_curiosity(store)["fired"] == 1
    assert len(pending_questions(store)) == 1

    # Same cue, now confident.
    _write_deferred(store, r.id, entropy=0.05, turn=5, session_id="sA", cue="which db")
    result = drain_deferred_curiosity(store)
    assert result["resolved"] == 1
    assert pending_questions(store) == []


def test_resolution_matches_cue_normalized_and_scoped(tmp_path):
    from iai_mcp.curiosity import drain_deferred_curiosity, pending_questions

    store = MemoryStore(path=tmp_path)
    r = _rec()
    store.insert(r)

    _write_deferred(store, r.id, entropy=0.95, turn=1, session_id="sA", cue="Which  DB")
    drain_deferred_curiosity(store)
    assert len(pending_questions(store)) == 1

    # A confident recall on an unrelated cue must not close it.
    _write_deferred(store, r.id, entropy=0.05, turn=2, session_id="sA", cue="something else")
    assert drain_deferred_curiosity(store)["resolved"] == 0
    assert len(pending_questions(store)) == 1

    # Case/whitespace variation of the original cue does close it.
    _write_deferred(store, r.id, entropy=0.05, turn=3, session_id="sA", cue="which db")
    assert drain_deferred_curiosity(store)["resolved"] == 1
    assert pending_questions(store) == []


def test_resolution_does_not_cross_sessions(tmp_path):
    from iai_mcp.curiosity import drain_deferred_curiosity, pending_questions

    store = MemoryStore(path=tmp_path)
    r = _rec()
    store.insert(r)

    _write_deferred(store, r.id, entropy=0.95, turn=1, session_id="sA", cue="shared cue")
    drain_deferred_curiosity(store)
    assert len(pending_questions(store, session_id="sA")) == 1

    # Another session resolving the same cue text leaves sA's question open.
    _write_deferred(store, r.id, entropy=0.05, turn=1, session_id="sB", cue="shared cue")
    assert drain_deferred_curiosity(store)["resolved"] == 0
    assert len(pending_questions(store, session_id="sA")) == 1


def test_resolution_is_idempotent(tmp_path):
    from iai_mcp.curiosity import drain_deferred_curiosity, pending_questions
    from iai_mcp.events import query_events

    store = MemoryStore(path=tmp_path)
    r = _rec()
    store.insert(r)

    _write_deferred(store, r.id, entropy=0.95, turn=1, session_id="sA", cue="c")
    drain_deferred_curiosity(store)
    _write_deferred(store, r.id, entropy=0.05, turn=2, session_id="sA", cue="c")
    drain_deferred_curiosity(store)
    # A second confident recall must not write a duplicate resolution.
    _write_deferred(store, r.id, entropy=0.05, turn=3, session_id="sA", cue="c")
    assert drain_deferred_curiosity(store)["resolved"] == 0

    resolved = query_events(store, kind="curiosity_resolved", limit=50)
    assert len(resolved) == 1
    assert pending_questions(store) == []


# --- consumed-event prune --------------------------------------------------

def test_prune_drops_consumed_inputs_and_stale_cursors(tmp_path):
    from iai_mcp.curiosity import (
        drain_deferred_curiosity,
        prune_consumed_curiosity_events,
    )
    from iai_mcp.events import query_events

    store = MemoryStore(path=tmp_path)
    r = _rec()
    store.insert(r)

    for i in range(3):
        _write_deferred(store, r.id, entropy=0.95, turn=i, session_id=f"s{i}")
    # Two drains produce two cursor rows; only the newest is ever read.
    drain_deferred_curiosity(store)
    _write_deferred(store, r.id, entropy=0.95, turn=9, session_id="s9")
    drain_deferred_curiosity(store)

    assert len(query_events(store, kind="deferred_curiosity_input", limit=50)) == 4

    out = prune_consumed_curiosity_events(store)
    assert out["inputs_pruned"] == 4
    assert out["cursors_pruned"] >= 1
    assert query_events(store, kind="deferred_curiosity_input", limit=50) == []
    # Exactly one cursor survives, so the watermark still gates re-firing.
    assert len(query_events(store, kind="curiosity_drain_cursor", limit=50)) == 1


def test_prune_is_a_noop_before_anything_is_consumed(tmp_path):
    # With no cursor yet, deleting inputs would drop never-fired questions.
    from iai_mcp.curiosity import prune_consumed_curiosity_events
    from iai_mcp.events import query_events

    store = MemoryStore(path=tmp_path)
    r = _rec()
    store.insert(r)
    _write_deferred(store, r.id, entropy=0.95, turn=1, session_id="sA")

    out = prune_consumed_curiosity_events(store)
    assert out == {"inputs_pruned": 0, "cursors_pruned": 0}
    assert len(query_events(store, kind="deferred_curiosity_input", limit=50)) == 1


def test_prune_keeps_unconsumed_inputs_for_the_next_drain(tmp_path):
    from iai_mcp.curiosity import (
        drain_deferred_curiosity,
        prune_consumed_curiosity_events,
    )
    from iai_mcp.events import query_events

    store = MemoryStore(path=tmp_path)
    r = _rec()
    store.insert(r)

    _write_deferred(store, r.id, entropy=0.95, turn=1, session_id="sA")
    drain_deferred_curiosity(store)
    # Arrives after the watermark: must survive the prune and still fire.
    _write_deferred(store, r.id, entropy=0.95, turn=9, session_id="sB")

    prune_consumed_curiosity_events(store)
    survivors = query_events(store, kind="deferred_curiosity_input", limit=50)
    assert len(survivors) == 1
    assert drain_deferred_curiosity(store)["fired"] == 1


# --- bounded live MCP tool -------------------------------------------------

def test_curiosity_pending_tool_excludes_expired_questions(tmp_path):
    from datetime import timedelta

    from iai_mcp.core import dispatch
    from iai_mcp.curiosity import PENDING_MAX_AGE_DAYS
    from iai_mcp.events import write_event
    from iai_mcp.store import EVENTS_TABLE

    store = MemoryStore(path=tmp_path)
    write_event(
        store,
        kind="curiosity_question",
        data={"question_id": str(uuid4()), "text": "old", "tier": "question",
              "entropy": 0.95, "turn": 1, "triggered_by": []},
        severity="info",
        session_id="s1",
    )
    stale_ts = datetime.now(timezone.utc) - timedelta(
        days=PENDING_MAX_AGE_DAYS + 5,
    )
    store.db.open_table(EVENTS_TABLE).update(
        where="kind = 'curiosity_question'",
        values={"ts": stale_ts},
    )

    out = dispatch(store, "curiosity_pending", {})
    assert out == {"questions": [], "count": 0}


def test_last_curiosity_turn_is_session_scoped(tmp_path):
    from iai_mcp.curiosity import _last_curiosity_turn, fire_curiosity

    store = MemoryStore(path=tmp_path)
    r = _rec()
    store.insert(r)
    hits = [type("H", (), {"record_id": r.id, "score": 0.5})()]

    # Fire once in the target session, then bury it under >20 questions from
    # other sessions. The old 20-event global window would miss it and return
    # None (cooldown silently disabled); the session-scoped query still finds
    # turn 1.
    fire_curiosity(store, hits, "target", 0.95, "target-session", turn=1)
    for i in range(25):
        fire_curiosity(store, hits, f"c{i}", 0.95, f"other-{i}", turn=1)

    assert _last_curiosity_turn(store, "target-session") == 1
