"""The memory_capture RPC handler forwards next_action/focus to
working_tier.update_task AFTER capture_turn, session-routed and self-
persisting -- closing the gap where update_task had zero live callers.

Case 1 drives the REAL core.dispatch memory_capture handler end-to-end from
a cold start (no pre-seeded focal task): the
`session_id=params.get("session_id") or "-"` forwarding line is the thing
under test -- a typo'd param key or a wrong hardcoded session_id would leave
_select_entry_locked with no matching entry, next_action would never fold,
and the assertions below on the composed payload and per-session snapshot
would fail.

Case 2 proves the session-routing itself: after a task switch parks session
A and makes session B focal, a fold explicitly targeting A's own
session_id must land on A's entry (never B's) regardless of which session
is currently focal -- the concurrent-RPC-thread race the session param
closes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iai_mcp import working_tier as wt
from iai_mcp.community import CommunityAssignment
from iai_mcp.core import dispatch
from iai_mcp.session import _compose_session_start_payload, format_payload_as_markdown
from iai_mcp.store import MemoryStore


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(path=tmp_path / "lancedb")
    yield s
    s.close()


@pytest.fixture(autouse=True)
def _reset_working_tier_singleton():
    wt._reset()
    yield
    wt._reset()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_cold_start_capture_populates_live_state_without_manual_persist(
    driver, store, monkeypatch,
):
    _select_driver(driver, monkeypatch)
    assert wt.read_task(session_id="sess-cold") is None, "must start with no focal task"

    result = dispatch(
        store, "memory_capture",
        {
            "text": "investigate the recall latency regression on the dispatch path",
            "cue": "c",
            "session_id": "sess-cold",
            "role": "user",
            "next_action": "profile the dispatch hot path with the sampler",
            "focus": "recall latency regression",
        },
    )
    assert result["status"] == "inserted", result

    entry = wt.read_task(session_id="sess-cold")
    assert entry is not None
    assert entry.next_action == "profile the dispatch hot path with the sampler"
    assert entry.focus == "recall latency regression"

    payload = _compose_session_start_payload(
        store, CommunityAssignment(), [],
        session_id="sess-cold", profile_state={"wake_depth": "standard"},
    )
    rendered = format_payload_as_markdown(payload)
    assert "profile the dispatch hot path with the sampler" in rendered

    snapshot = wt._cache_path(store, "sess-cold").read_text(encoding="utf-8")
    assert "next action: profile the dispatch hot path with the sampler" in snapshot


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_cold_start_capture_without_next_action_or_focus_never_calls_update_task(
    driver, store, monkeypatch,
):
    """No forwarding call at all when neither field is present -- avoids a
    needless persist on every ordinary capture."""
    _select_driver(driver, monkeypatch)

    calls: list[dict] = []
    original = wt.update_task

    def _spy(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(wt, "update_task", _spy)

    result = dispatch(
        store, "memory_capture",
        {
            "text": "an ordinary capture with no live-state fields set",
            "cue": "c",
            "session_id": "sess-plain",
            "role": "user",
        },
    )
    assert result["status"] == "inserted", result
    assert calls == [], "update_task must not be called when next_action/focus are absent"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_cross_session_interleaving_fold_lands_only_on_own_session(
    driver, store, monkeypatch,
):
    _select_driver(driver, monkeypatch)

    wt.open_task("session A original goal", session_id="sess-a")
    # A second RPC thread's capture for session B arrives before session A's
    # fold runs -- the task switch parks A and makes B focal.
    wt.open_task("session B original goal", session_id="sess-b")
    assert wt.read_task() is not None
    assert wt.read_task().session_id == "sess-b"

    folded = wt.update_task(
        next_action="A's folded next action",
        session_id="sess-a",
        store=store,
    )
    assert folded is not None
    assert folded.session_id == "sess-a"
    assert folded.next_action == "A's folded next action"

    a_entry = wt.read_task(session_id="sess-a")
    assert a_entry is not None
    assert a_entry.next_action == "A's folded next action"

    b_entry = wt.read_task(session_id="sess-b")
    assert b_entry is not None
    assert b_entry.next_action == "", "the fold for A must never touch B's entry"

    a_snapshot = wt._cache_path(store, "sess-a").read_text(encoding="utf-8")
    assert "A's folded next action" in a_snapshot

    b_snapshot_path = wt._cache_path(store, "sess-b")
    if b_snapshot_path.exists():
        assert "A's folded next action" not in b_snapshot_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_omitted_session_id_forward_targets_the_dash_entry_capture_turn_opened(
    driver, store, monkeypatch,
):
    """When session_id is omitted entirely from the memory_capture params,
    capture_turn opens/refocuses the "-"-keyed entry (its own default); the
    update_task forward must route to that SAME entry even when a second
    RPC thread's capture for a DIFFERENT session interleaves between
    capture_turn's return and the forward call.

    A single sequential dispatch() call cannot reproduce this race on its
    own -- capture_turn's own store.insert() always refocuses "-" as the
    last thing it does, so the forward line would see "-" focal either way.
    The racing capture is injected via monkeypatch on capture.capture_turn
    (a bare module attribute core's local `from ... import` resolves at
    call time), mirroring test_cross_session_interleaving_fold_lands_only_
    on_own_session's ordering technique while still exercising the real
    forward line in core.dispatch, not a hand-written substitute."""
    _select_driver(driver, monkeypatch)

    from iai_mcp import capture as capture_mod

    real_capture_turn = capture_mod.capture_turn

    def _racing_capture_turn(*args, **kwargs):
        outcome = real_capture_turn(*args, **kwargs)
        # A second RPC thread's capture for a different session lands here,
        # between this call's capture_turn and its own forward line.
        wt.open_task("a concurrent session's task", session_id="sess-other")
        return outcome

    monkeypatch.setattr(capture_mod, "capture_turn", _racing_capture_turn)

    result = dispatch(
        store, "memory_capture",
        {
            "text": "a capture with no session_id param at all",
            "cue": "c",
            "role": "user",
            "next_action": "must fold onto the dash entry, not sess-other",
        },
    )
    assert result["status"] == "inserted", result

    dash_entry = wt.read_task(session_id="-")
    assert dash_entry is not None
    assert dash_entry.next_action == "must fold onto the dash entry, not sess-other"

    other_entry = wt.read_task(session_id="sess-other")
    assert other_entry is not None
    assert other_entry.next_action == "", (
        "the omitted-session_id fold must never land on whatever session "
        "happens to be globally focal after a concurrent capture interleaves"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_memory_contradict_never_forwards_to_update_task(driver, store, monkeypatch):
    """memory_contradict never forwards next_action/focus to update_task --
    the existing fence on contradict() stays untouched. A client sending
    next_action in a memory_contradict call gets it silently ignored, never
    routed to the working tier."""
    _select_driver(driver, monkeypatch)

    seed = dispatch(
        store, "memory_capture",
        {
            "text": "the deployment window closes at 5pm on release days",
            "cue": "c",
            "session_id": "sess-contradict",
            "role": "user",
        },
    )
    assert seed["status"] == "inserted", seed
    seeded_entry = wt.read_task(session_id="sess-contradict")
    assert seeded_entry is not None
    assert seeded_entry.next_action == "", (
        "an ordinary capture with no next_action/focus must never fold "
        "next_action onto the ambient task the insert itself opened"
    )

    calls: list[dict] = []
    original = wt.update_task

    def _spy(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(wt, "update_task", _spy)

    result = dispatch(
        store, "memory_contradict",
        {
            "id": seed["record_id"],
            "new_fact": "the deployment window closes at 6pm on release days",
            "next_action": "must never be forwarded",
        },
    )
    assert result["new_record_id"]
    assert calls == [], "memory_contradict must never forward to update_task"
    unaffected_entry = wt.read_task(session_id="sess-contradict")
    assert unaffected_entry is not None
    assert unaffected_entry.next_action == "", (
        "memory_contradict must never fold next_action onto the working tier"
    )
