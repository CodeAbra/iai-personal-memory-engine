"""Capture-side correction-candidate detector: queue, never write.

A live user turn that reads as a correction of a recently-surfaced belief
must QUEUE a confirmation candidate (a curiosity_question-shaped event) --
it must never auto-write a contradiction. The fence is structural: this
module asserts capture.py has no reachable import of contradict /
add_contradicts_edge / memory_contradict, so a passing behavioral test can
never mask a reachable write path.
"""
from __future__ import annotations

import ast
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CAPTURE_PY = _REPO_ROOT / "src" / "iai_mcp" / "capture.py"

_FORBIDDEN_IMPORT_SUBSTR = "contradict"
_FORBIDDEN_ATTR = "add_contradicts_edge"


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401, PLC0415
        except ImportError:
            pytest.skip("iai_mcp_native not built")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _record(text: str) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.01] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"ts": now.isoformat(), "cue": "seed", "session_id": "-", "role": "assistant"}],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )


def _mark_surfaced(store: MemoryStore, session_id: str, record_id, *, age_sec: float = 0.0) -> None:
    """Simulate: this session's next-turn pack served ``record_id``
    ``age_sec`` seconds ago (default: just now)."""
    from iai_mcp import foresight

    foresight._save_state(
        store,
        {"session_id": session_id, "served": {str(record_id): time.time() - age_sec}},
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_correction_turn_with_surfaced_belief_queues_one_candidate(
    tmp_path, monkeypatch, driver,
) -> None:
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    session = "sess-correction-1"

    surfaced = _record("The meeting is on Tuesday.")
    store.insert(surfaced)
    flush_record_buffer(store)
    _mark_surfaced(store, session, surfaced.id)

    from iai_mcp.capture import capture_turn
    from iai_mcp.curiosity import pending_questions

    result = capture_turn(
        store, cue="", text="нет, правильно — встреча в среду",
        tier="episodic", session_id=session, role="user", live_turn=True,
    )
    flush_record_buffer(store)
    assert result["status"] == "inserted"

    qs = pending_questions(store, session_id=session)
    assert len(qs) == 1
    assert str(surfaced.id) in [str(t) for t in qs[0].triggered_by_record_ids]

    # Never a write: no contradicts edge touches the surfaced record.
    edges = store.incident_edges([surfaced.id], edge_types=["contradicts"], top_k=None)
    assert not any(edges.values())


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_queued_candidate_cue_is_trigger_record_surface_not_raw_turn_text(
    tmp_path, monkeypatch, driver,
) -> None:
    """The queued event's `cue` must be the TRIGGERING RECORD's own surface
    -- NOT the raw per-turn text, and NOT a synthetic pattern label.
    foresight's tunnel-question path embeds every pending question's `cue`
    and injects the question into a session's next-turn pack when
    cos(turn_cue, cue) >= 0.60; a synthetic label measured cos=0.71 against
    an unrelated turn (well past the floor), which would have injected the
    candidate into an unrelated session's pack. The record's own topical
    text keeps that gate meaningful, and stays stable PER RECORD (not per
    turn) for the tunnel-question embed cache."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    session = "sess-cue-topical-1"

    surfaced = _record("The meeting is on Tuesday.")
    store.insert(surfaced)
    flush_record_buffer(store)
    _mark_surfaced(store, session, surfaced.id)

    from iai_mcp.capture import capture_turn
    from iai_mcp.curiosity import pending_questions

    capture_turn(
        store, cue="", text="на самом деле встреча перенесена",
        tier="episodic", session_id=session, role="user", live_turn=True,
    )
    flush_record_buffer(store)

    qs = pending_questions(store, session_id=session)
    assert len(qs) == 1
    assert qs[0].cue == "The meeting is on Tuesday."
    assert "correction-candidate" not in qs[0].cue


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_non_correction_turn_queues_nothing(tmp_path, monkeypatch, driver) -> None:
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    session = "sess-neutral-1"

    surfaced = _record("The meeting is on Tuesday.")
    store.insert(surfaced)
    flush_record_buffer(store)
    _mark_surfaced(store, session, surfaced.id)

    from iai_mcp.capture import capture_turn
    from iai_mcp.curiosity import pending_questions

    capture_turn(
        store, cue="", text="let's continue planning the rest of the week",
        tier="episodic", session_id=session, role="user", live_turn=True,
    )
    flush_record_buffer(store)

    assert pending_questions(store, session_id=session) == []


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_correction_pattern_with_nothing_to_correct_queues_nothing(
    tmp_path, monkeypatch, driver,
) -> None:
    """False-positive guard: a correction pattern fires, but this session's
    pack served nothing -- no candidate, no write."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    session = "sess-no-surface-1"

    from iai_mcp.capture import capture_turn
    from iai_mcp.curiosity import pending_questions

    capture_turn(
        store, cue="", text="нет, правильно — это было в мае",
        tier="episodic", session_id=session, role="user", live_turn=True,
    )
    flush_record_buffer(store)

    assert pending_questions(store, session_id=session) == []


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_lexical_false_positive_casual_aside_queues_nothing(
    tmp_path, monkeypatch, driver,
) -> None:
    """A casual aside that merely resembles a correction ('нет, не совсем')
    must not match the detector at all -- queues nothing, writes nothing --
    even when a belief was recently surfaced."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    session = "sess-casual-1"

    surfaced = _record("The meeting is on Tuesday.")
    store.insert(surfaced)
    flush_record_buffer(store)
    _mark_surfaced(store, session, surfaced.id)

    from iai_mcp.capture import capture_turn
    from iai_mcp.curiosity import pending_questions

    capture_turn(
        store, cue="", text="нет, не совсем, но давай продолжим",
        tier="episodic", session_id=session, role="user", live_turn=True,
    )
    flush_record_buffer(store)

    assert pending_questions(store, session_id=session) == []


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_stale_served_record_outside_recency_window_queues_nothing(
    tmp_path, monkeypatch, driver,
) -> None:
    """A record served long ago (outside CORRECTION_CANDIDATE_RECENCY_WINDOW_SEC)
    is not 'recently surfaced' -- the served-state dict accumulates for the
    whole session, so an unbounded read would treat 'ever served' as
    'just surfaced'."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    session = "sess-stale-1"

    from iai_mcp.capture import CORRECTION_CANDIDATE_RECENCY_WINDOW_SEC

    surfaced = _record("The meeting is on Tuesday.")
    store.insert(surfaced)
    flush_record_buffer(store)
    _mark_surfaced(
        store, session, surfaced.id,
        age_sec=CORRECTION_CANDIDATE_RECENCY_WINDOW_SEC + 60,
    )

    from iai_mcp.capture import capture_turn
    from iai_mcp.curiosity import pending_questions

    capture_turn(
        store, cue="", text="нет, правильно — встреча в среду",
        tier="episodic", session_id=session, role="user", live_turn=True,
    )
    flush_record_buffer(store)

    assert pending_questions(store, session_id=session) == []


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_pending_cap_blocks_a_third_correction_candidate(
    tmp_path, monkeypatch, driver,
) -> None:
    """CORRECTION_CANDIDATE_MAX_PENDING_PER_SESSION caps queueing so a
    chatty correction-shaped phrase cannot crowd out genuine curiosity
    questions on the fixed-size curiosity_pending surface."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    session = "sess-cap-1"

    from iai_mcp.capture import (
        CORRECTION_CANDIDATE_MAX_PENDING_PER_SESSION,
        capture_turn,
    )
    from iai_mcp.curiosity import pending_questions

    recs = [_record(f"Fact number {i}.") for i in range(4)]
    for r in recs:
        store.insert(r)
    flush_record_buffer(store)

    for i in range(CORRECTION_CANDIDATE_MAX_PENDING_PER_SESSION + 1):
        _mark_surfaced(store, session, recs[i].id)
        capture_turn(
            store, cue="", text="на самом деле это другое",
            tier="episodic", session_id=session, role="user", live_turn=True,
        )
        flush_record_buffer(store)

    assert len(pending_questions(store, session_id=session)) == (
        CORRECTION_CANDIDATE_MAX_PENDING_PER_SESSION
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_queued_candidate_preserves_absent_prior_curiosity_turn_signal(
    tmp_path, monkeypatch, driver,
) -> None:
    """When this session has no prior curiosity_question event,
    _last_curiosity_turn must keep returning None after the candidate is
    queued -- writing a fabricated turn (e.g. 0) would silently corrupt
    curiosity.fire_curiosity's (turn - last) < COOLDOWN_TURNS cooldown gate
    for every later real curiosity call in this session."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    session = "sess-turn-none-1"

    surfaced = _record("The meeting is on Tuesday.")
    store.insert(surfaced)
    flush_record_buffer(store)
    _mark_surfaced(store, session, surfaced.id)

    from iai_mcp.capture import capture_turn
    from iai_mcp.curiosity import _last_curiosity_turn

    assert _last_curiosity_turn(store, session) is None

    capture_turn(
        store, cue="", text="нет, правильно — встреча в среду",
        tier="episodic", session_id=session, role="user", live_turn=True,
    )
    flush_record_buffer(store)

    assert _last_curiosity_turn(store, session) is None


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_queued_candidate_preserves_existing_prior_curiosity_turn_signal(
    tmp_path, monkeypatch, driver,
) -> None:
    """When this session already has a real curiosity_question event at
    turn=7, queuing a correction candidate must not shift what
    _last_curiosity_turn reports -- the candidate's own `turn` value is
    carried through unchanged from the pre-write read."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    session = "sess-turn-preserve-1"

    surfaced = _record("The meeting is on Tuesday.")
    store.insert(surfaced)
    flush_record_buffer(store)
    _mark_surfaced(store, session, surfaced.id)

    from iai_mcp.events import write_event

    write_event(
        store,
        kind="curiosity_question",
        data={
            "question_id": str(uuid4()),
            "text": "an earlier, unrelated question",
            "tier": "question",
            "entropy": 0.95,
            "turn": 7,
            "cue": "unrelated-cue",
            "triggered_by": [],
        },
        severity="info",
        session_id=session,
    )
    flush_record_buffer(store)

    from iai_mcp.capture import capture_turn
    from iai_mcp.curiosity import _last_curiosity_turn

    assert _last_curiosity_turn(store, session) == 7

    capture_turn(
        store, cue="", text="нет, правильно — встреча в среду",
        tier="episodic", session_id=session, role="user", live_turn=True,
    )
    flush_record_buffer(store)

    assert _last_curiosity_turn(store, session) == 7


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_vanished_trigger_record_queues_nothing(
    tmp_path, monkeypatch, driver,
) -> None:
    """The served-state entry references a record id, but the record no
    longer exists in the store (e.g. purged between serving and this turn)
    -- nothing safe to reference or to build a topical cue from, so nothing
    queues."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    session = "sess-vanished-1"

    from uuid import uuid4 as _uuid4

    ghost_id = _uuid4()  # never inserted
    _mark_surfaced(store, session, ghost_id)

    from iai_mcp.capture import capture_turn
    from iai_mcp.curiosity import pending_questions

    capture_turn(
        store, cue="", text="нет, правильно — встреча в среду",
        tier="episodic", session_id=session, role="user", live_turn=True,
    )
    flush_record_buffer(store)

    assert pending_questions(store, session_id=session) == []


def test_capture_module_has_no_contradict_import_or_call() -> None:
    """Structural fence: capture.py must have no reachable import of
    contradict / memory_contradict, and never reference the
    add_contradicts_edge store method -- an import-graph assertion, not a
    behavioral one, so a passing behavioral test can never mask a reachable
    write path."""
    tree = ast.parse(_CAPTURE_PY.read_text(encoding="utf-8"), filename=str(_CAPTURE_PY))

    bad_imports: list[str] = []
    bad_attrs: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _FORBIDDEN_IMPORT_SUBSTR in alias.name.lower():
                    bad_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                if (
                    _FORBIDDEN_IMPORT_SUBSTR in mod.lower()
                    or _FORBIDDEN_IMPORT_SUBSTR in alias.name.lower()
                ):
                    bad_imports.append(f"{mod}.{alias.name}")
        elif isinstance(node, ast.Attribute):
            if node.attr == _FORBIDDEN_ATTR:
                bad_attrs.append(node.attr)

    assert not bad_imports, (
        f"capture.py must not import a contradiction-write symbol: {bad_imports}"
    )
    assert not bad_attrs, (
        f"capture.py must not reference {_FORBIDDEN_ATTR}: {bad_attrs}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_queued_candidate_surfaced_by_get_pending_questions_and_resolves(
    tmp_path, monkeypatch, driver,
) -> None:
    """Confirm-surface proof (Task 2): the queued candidate is readable via
    get_pending_questions -- the reader behind the curiosity_pending MCP
    tool -- with no reader or tool change, and drops out once a
    curiosity_resolved event marks it."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    session = "sess-confirm-1"

    surfaced = _record("The meeting is on Tuesday.")
    store.insert(surfaced)
    flush_record_buffer(store)
    _mark_surfaced(store, session, surfaced.id)

    from iai_mcp.capture import capture_turn
    from iai_mcp.curiosity import get_pending_questions, pending_questions

    capture_turn(
        store, cue="", text="no, actually the meeting moved to Wednesday",
        tier="episodic", session_id=session, role="user", live_turn=True,
    )
    flush_record_buffer(store)

    before = get_pending_questions(store, limit=10)
    assert len(before) == 1
    assert before[0]["text"]

    # Resolve it (the existing resolve path -- a human/model confirm action,
    # entirely outside the detector's own code path).
    qs = pending_questions(store, session_id=session)
    assert len(qs) == 1
    question_id = str(qs[0].id)

    from iai_mcp.events import write_event

    write_event(
        store,
        kind="curiosity_resolved",
        data={"question_id": question_id, "reason": "confirmed"},
        severity="info",
        session_id=session,
    )
    flush_record_buffer(store)

    after = get_pending_questions(store, limit=10)
    assert after == []
