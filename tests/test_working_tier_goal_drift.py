"""Drift-gated automatic goal re-anchor: sustained pivot fires, continuous
topic and single stray turns never do. All vectors here are SYNTHETIC and
pinned via a monkeypatched ``working_tier._goal_embedding`` seam -- the real
embedder is never touched (mirrors test_capture_dedup_never_merge.py's
synthetic-cosine discipline).
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from iai_mcp import working_tier as wt
from iai_mcp.types import EMBED_DIM, MemoryRecord

import pytest


def _pinned_vec(axis: int) -> list[float]:
    """Unit vector with all mass on one axis -- any two distinct axes are
    exactly orthogonal (cosine 0, distance 1.0), comfortably clear of the
    drift threshold on both sides."""
    vec = [0.0] * EMBED_DIM
    vec[axis] = 1.0
    return vec


TOPIC_A_TEXT = "alice discusses coffee brewing methods"
TOPIC_A_VEC = _pinned_vec(0)
TOPIC_B_TEXT = "alice debugs a rust borrow checker error"
TOPIC_B_VEC = _pinned_vec(1)
TOPIC_C_TEXT = "alice plans a hiking trip itinerary"
TOPIC_C_VEC = _pinned_vec(2)


def _vec_for_goal(_store: object, goal: str) -> list[float]:
    if TOPIC_C_TEXT in goal:
        return TOPIC_C_VEC
    if TOPIC_B_TEXT in goal:
        return TOPIC_B_VEC
    return TOPIC_A_VEC


def _make(
    text: str,
    session_id: str = "sess-a",
    *,
    embedding: list[float] | None = None,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=embedding if embedding is not None else [0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"session_id": session_id}],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )


class _FakeRealStore:
    """A store shim with a truthy .db so update_from_record's store-realness
    gate calls the (monkeypatched) _goal_embedding seam -- never a real
    Hippo/embedder."""

    def __init__(self, root: object = None):
        self.root = root
        self.db = object()


@pytest.fixture(autouse=True)
def _fresh_tier(tmp_path, monkeypatch):
    monkeypatch.delenv(wt.WORKING_TIER_CACHE_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    wt._reset()
    yield
    wt._reset()


@pytest.fixture(autouse=True)
def _pinned_goal_embedding(monkeypatch):
    monkeypatch.setattr(wt, "_goal_embedding", _vec_for_goal)


def _feed(text: str, vec: list[float], *, session_id: str = "sess-a", store=None):
    wt.update_from_record(_make(text, session_id, embedding=vec), store=store)


def test_continuous_topic_never_reanchors(tmp_path):
    store = _FakeRealStore(tmp_path)
    _feed(TOPIC_A_TEXT, TOPIC_A_VEC, store=store)
    original_goal = wt.read_task().goal
    assert original_goal == TOPIC_A_TEXT

    for i in range(10):
        _feed(f"{TOPIC_A_TEXT} continuing turn {i}", TOPIC_A_VEC, store=store)

    assert wt.read_task().goal == original_goal, (
        "continuous-topic turns must never change the goal"
    )


def test_sustained_cooled_pivot_reanchors(tmp_path):
    store = _FakeRealStore(tmp_path)
    _feed(TOPIC_A_TEXT, TOPIC_A_VEC, store=store)  # primes goal_embedding post-lock
    for _ in range(4):  # cooldown 2..5
        _feed(f"{TOPIC_A_TEXT} continuing", TOPIC_A_VEC, store=store)
    assert wt.read_task().goal == TOPIC_A_TEXT

    for _ in range(2):  # pivot turns 1..2 -- run below minimum, must not fire yet
        _feed(TOPIC_B_TEXT, TOPIC_B_VEC, store=store)
    assert wt.read_task().goal == TOPIC_A_TEXT, (
        "an under-minimum run must not fire even past cooldown"
    )

    _feed(TOPIC_B_TEXT, TOPIC_B_VEC, store=store)  # pivot turn 3 -- run + cooldown both satisfied
    entry = wt.read_task()
    assert entry.goal == TOPIC_B_TEXT, (
        "a sustained, cooled-down pivot must re-anchor to the pivot turn's own verbatim text"
    )
    assert entry.goal_embedding == TOPIC_B_VEC, (
        "goal_embedding must be recomputed against the NEW goal after a fire, "
        "never left pointing at the old goal's vector"
    )
    assert entry.goal_drift_run == 0
    assert entry.goal_drift_cooldown == 0


def test_explicit_goal_set_rebaselines_drift_gate(tmp_path):
    """An explicit update_task(goal=...) must reset the staleness fence just
    like an automatic fire -- otherwise the drift gate keeps comparing
    against the OLD goal's vector and can silently overwrite the just-set
    goal a few turns later."""
    store = _FakeRealStore(tmp_path)
    _feed(TOPIC_A_TEXT, TOPIC_A_VEC, store=store)  # primes goal_embedding on TOPIC_A_VEC

    wt.update_task(goal=TOPIC_B_TEXT, session_id="sess-a", store=store)
    entry = wt.read_task()
    assert entry.goal == TOPIC_B_TEXT
    assert entry.goal_embedding is None, (
        "an explicit goal-set must null the cached vector, not leave it "
        "pointed at the old goal"
    )
    assert entry.goal_drift_run == 0
    assert entry.goal_drift_cooldown == 0

    for _ in range(6):  # continuous topic on the NEW goal must never re-anchor
        _feed(f"{TOPIC_B_TEXT} continuing", TOPIC_B_VEC, store=store)
    assert wt.read_task().goal == TOPIC_B_TEXT, (
        "sustained continuous-topic turns on the newly-set goal must never "
        "re-anchor away from it"
    )
    assert wt.read_task().goal_embedding == TOPIC_B_VEC, (
        "the drift gate must re-prime against the NEW goal, not the stale one"
    )


def test_update_task_same_goal_text_does_not_reset_drift_counters(tmp_path):
    """Re-sending the identical goal text (no actual change) must leave the
    drift gate's counters and cached vector untouched -- otherwise a caller
    that echoes the current goal back every turn would silently disable the
    gate by resetting the cooldown/run counters on every single turn."""
    store = _FakeRealStore(tmp_path)
    _feed(TOPIC_A_TEXT, TOPIC_A_VEC, store=store)
    for _ in range(2):  # run 1..2, below the fire minimum -- must not fire
        _feed(TOPIC_B_TEXT, TOPIC_B_VEC, store=store)

    before = wt.read_task()
    before_run = before.goal_drift_run
    before_cooldown = before.goal_drift_cooldown
    before_vec = before.goal_embedding

    wt.update_task(goal=TOPIC_A_TEXT, session_id="sess-a", store=store)

    after = wt.read_task()
    assert after.goal_drift_run == before_run
    assert after.goal_drift_cooldown == before_cooldown
    assert after.goal_embedding == before_vec


def test_rendered_snapshot_never_leaks_drift_internals(tmp_path):
    store = _FakeRealStore(tmp_path)
    _feed(TOPIC_A_TEXT, TOPIC_A_VEC, store=store)
    entry = wt.read_task()
    snapshot = wt._render_snapshot(entry)
    assert "goal_embedding" not in snapshot
    assert "goal_drift_run" not in snapshot
    assert "goal_drift_cooldown" not in snapshot
    assert str(TOPIC_A_VEC) not in snapshot


# --- Anti-feature fence: fail-open and no-thrash gates ---


def test_single_stray_turn_never_reanchors(tmp_path):
    store = _FakeRealStore(tmp_path)
    _feed(TOPIC_A_TEXT, TOPIC_A_VEC, store=store)
    for _ in range(5):
        _feed(f"{TOPIC_A_TEXT} more", TOPIC_A_VEC, store=store)

    _feed(TOPIC_B_TEXT, TOPIC_B_VEC, store=store)  # ONE stray off-topic turn
    assert wt.read_task().goal == TOPIC_A_TEXT

    for _ in range(5):  # topic resumes -- run counter must have reset
        _feed(f"{TOPIC_A_TEXT} resumed", TOPIC_A_VEC, store=store)
    assert wt.read_task().goal == TOPIC_A_TEXT, (
        "a single stray off-topic turn amid a continuous topic must never re-anchor"
    )


def test_second_pivot_waits_out_cooldown(tmp_path):
    store = _FakeRealStore(tmp_path)
    _feed(TOPIC_A_TEXT, TOPIC_A_VEC, store=store)
    for _ in range(4):
        _feed(f"{TOPIC_A_TEXT} more", TOPIC_A_VEC, store=store)
    for _ in range(3):
        _feed(TOPIC_B_TEXT, TOPIC_B_VEC, store=store)
    entry = wt.read_task()
    assert entry.goal == TOPIC_B_TEXT, "first sustained pivot must fire"
    assert entry.goal_drift_cooldown == 0

    for _ in range(3):  # second pivot: run satisfied, cooldown NOT yet (0..3 < 5)
        _feed(TOPIC_C_TEXT, TOPIC_C_VEC, store=store)
    assert wt.read_task().goal == TOPIC_B_TEXT, (
        "a second genuine pivot must not fire before the cooldown minimum elapses"
    )

    for _ in range(2):  # cooldown now reaches the minimum; run stays sustained
        _feed(TOPIC_C_TEXT, TOPIC_C_VEC, store=store)
    assert wt.read_task().goal == TOPIC_C_TEXT, (
        "the second pivot must fire once the cooldown minimum has elapsed"
    )


def test_missing_embedding_is_fail_open(tmp_path):
    store = _FakeRealStore(tmp_path)
    _feed(TOPIC_A_TEXT, TOPIC_A_VEC, store=store)
    original_goal = wt.read_task().goal

    record = SimpleNamespace(
        literal_surface="a turn with no embedding attribute at all",
        provenance=[{"session_id": "sess-a"}],
        created_at=datetime.now(timezone.utc),
    )
    wt.update_from_record(record, store=store)  # must not raise

    assert wt.read_task().goal == original_goal


def test_store_none_is_fail_open():
    _feed(TOPIC_A_TEXT, TOPIC_A_VEC, store=None)
    for _ in range(4):
        _feed(f"{TOPIC_A_TEXT} more", TOPIC_A_VEC, store=None)
    for _ in range(3):
        _feed(TOPIC_B_TEXT, TOPIC_B_VEC, store=None)  # must not raise even without a store
    entry = wt.read_task()
    assert entry is not None
    assert entry.goal == TOPIC_A_TEXT, (
        "without a store, the drift gate can never prime a vector, so it "
        "must never fire regardless of turn count"
    )
    assert entry.goal_embedding is None


def test_open_task_then_drift_eligible_turn_does_not_crash(tmp_path):
    store = _FakeRealStore(tmp_path)
    wt.open_task(TOPIC_A_TEXT, session_id="sess-a")
    entry = wt.read_task()
    assert entry.goal_embedding is None

    _feed(f"{TOPIC_A_TEXT} first turn after open", TOPIC_A_VEC, store=store)  # must not raise
    assert wt.read_task() is not None
