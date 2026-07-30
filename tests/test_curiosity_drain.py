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
    assert first == {"drained": 2, "fired": 2}
    assert len(pending_questions(store)) == 2

    # A second drain has advanced past every input already consumed: no
    # re-scan, no double-fire.
    second = drain_deferred_curiosity(store)
    assert second == {"drained": 0, "fired": 0}
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
