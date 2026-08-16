"""The task_support display-path signal: one shared visibility predicate
drives both the decorator strip and the recorded ``retrieval_used`` flag,
and the buffered event carries the join keys the nightly tuner needs.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from iai_mcp import core
from iai_mcp.events import flush_event_buffer, query_events
from iai_mcp.graph import MemoryGraph
from iai_mcp.pipeline import _apply_post_rank_pipeline
from iai_mcp.response_decorator import apply_profile, suggestions_visible
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryHit


@pytest.fixture(autouse=True)
def _reset_probe_cache():
    saved = core._task_support_probe_active_until
    core.set_task_support_probe_active_until(None)
    yield
    core.set_task_support_probe_active_until(saved)


# ---------------------------------------------------------------------------
# 1. Predicate truth table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,probe_active,expected",
    [
        ("cued_recognition", False, True),
        ("cued_recognition", True, True),
        ("blank_recall", False, False),
        ("blank_recall", True, True),
    ],
)
def test_suggestions_visible_truth_table(mode, probe_active, expected) -> None:
    assert suggestions_visible({"task_support": mode}, probe_active) is expected


def test_suggestions_visible_defaults_to_cued_recognition_when_unset() -> None:
    assert suggestions_visible({}, False) is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_hit(store: MemoryStore, *, suggestions: list) -> MemoryHit:
    rid = uuid4()
    return MemoryHit(
        record_id=rid,
        score=0.9,
        reason="test",
        literal_surface="alice asked about the schema migration",
        adjacent_suggestions=suggestions,
        session_id="alice-session-1",
    )


def _run_pipeline(
    store: MemoryStore,
    *,
    profile_state: dict,
    session_id: str = "alice-session-1",
    suggestions: list | None = None,
) -> dict:
    hit = _seed_hit(store, suggestions=suggestions or [])
    _apply_post_rank_pipeline(
        [hit],
        store=store,
        graph=MemoryGraph(),
        records_cache={},
        cue="what did we decide about the schema",
        session_id=session_id,
        profile_state=profile_state,
        turn=1,
        mode="verbatim",
        budget_used=10,
        path_label="test_task_support_probe",
    )
    flush_event_buffer(store)
    events = query_events(store, kind="retrieval_used", limit=10)
    assert events, "expected a buffered retrieval_used event"
    return events[0]["data"]


# ---------------------------------------------------------------------------
# 2. cued_recognition: visible + non-empty suggestion_ids
# ---------------------------------------------------------------------------


def test_cued_recognition_event_records_visible_and_suggestion_ids(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    sugg_id = uuid4()

    data = _run_pipeline(
        store,
        profile_state={"task_support": "cued_recognition"},
        suggestions=[sugg_id],
    )

    assert data["suggestions_visible"] is True
    assert data["suggestion_ids"] == [str(sugg_id)]
    assert data["probe"] is False
    assert data["session_id"] == "alice-session-1"
    datetime.fromisoformat(data["timestamp"])


# ---------------------------------------------------------------------------
# 3. blank_recall, no probe: strip AND flag both say hidden
# ---------------------------------------------------------------------------


def test_blank_recall_no_probe_strips_and_flags_hidden(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    sugg_id = uuid4()
    profile_state = {"task_support": "blank_recall"}

    data = _run_pipeline(store, profile_state=profile_state, suggestions=[sugg_id])
    assert data["suggestions_visible"] is False
    assert data["probe"] is False

    response = {"hits": [{"adjacent_suggestions": [str(sugg_id)]}]}
    apply_profile(response, profile_state, probe_active=False)
    assert response["hits"][0]["adjacent_suggestions"] == []


# ---------------------------------------------------------------------------
# 4. blank_recall WITH probe: shown, not stripped
# ---------------------------------------------------------------------------


def test_blank_recall_with_probe_shows_and_flags_visible(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    sugg_id = uuid4()
    profile_state = {"task_support": "blank_recall"}
    core.set_task_support_probe_active_until(
        datetime.now(timezone.utc) + timedelta(hours=1)
    )

    data = _run_pipeline(store, profile_state=profile_state, suggestions=[sugg_id])
    assert data["suggestions_visible"] is True
    assert data["probe"] is True

    response = {"hits": [{"adjacent_suggestions": [str(sugg_id)]}]}
    apply_profile(response, profile_state, probe_active=True)
    assert response["hits"][0]["adjacent_suggestions"] == [str(sugg_id)]


# ---------------------------------------------------------------------------
# Non-vacuity: the strip decision and the recorded flag must be driven by
# the SAME predicate -- prove the test actually detects drift between them.
# ---------------------------------------------------------------------------


def test_non_vacuity_strip_and_flag_share_one_predicate(tmp_path) -> None:
    profile_state = {"task_support": "blank_recall"}
    sugg_id = uuid4()

    # A decorator that drifted from suggestions_visible() (e.g. hardcoded
    # False -> never strip) would fail this assertion.
    response = {"hits": [{"adjacent_suggestions": [str(sugg_id)]}]}
    apply_profile(response, profile_state, probe_active=False)
    assert response["hits"][0]["adjacent_suggestions"] == []
    assert suggestions_visible(profile_state, False) is False
