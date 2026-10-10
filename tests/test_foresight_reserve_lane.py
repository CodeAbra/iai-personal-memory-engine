"""Proves the salience/directive existence-based reserve lane: it admits a
record raw cosine excludes from the primary admission window, stays capped
to its own slot budget, degrades to empty on a forced discovery failure
without breaking the pack, respects the same-session exclusion every other
lane already honors, and is a no-op on a corpus with no signal at all.

Real embedder, real refresh_pack throughout -- store.query_similar is never
monkeypatched; only the lane's own discovery call is forced to fail in the
degrade-to-empty case.
"""
from __future__ import annotations

from uuid import UUID

import pytest

from iai_mcp import foresight
from iai_mcp.capture import capture_turn
from iai_mcp.embed import embedder_for_store
from iai_mcp.store import MemoryStore
from iai_mcp.store._buffers import flush_record_buffer
from tests.test_baseline_foresight_admission import (
    _ALL_DISTRACTORS,
    _QUERY_TEXT,
    _get_admission_window_probe,
)
from tests.test_baseline_knowledge_vs_chatter import (
    SESSION_CHATTER,
    SESSION_KNOWLEDGE,
    SESSION_QUERY,
    _select_driver,
)

_DISTRACTOR_COUNT = 24

_SIGNALLED_TEXTS = [
    "The greenhouse irrigation controller passed its final acceptance test "
    "and ships to the client site next week.",
    "The backup generator's transfer switch was replaced after last "
    "month's failed cutover drill.",
    "The warehouse conveyor motor was rebalanced and no longer trips the "
    "overload breaker.",
]


def _build_corpus(store: MemoryStore, *, session_id: str = SESSION_KNOWLEDGE) -> "list[UUID]":
    """Captures every _SIGNALLED_TEXTS entry as a real, salience-tagged
    record plus _DISTRACTOR_COUNT register-matched distractors, and returns
    the signalled records' ids. Each signalled text is deliberately far
    from the vague query's own vocabulary, mirroring the admission-window
    baseline -- raw cosine alone must not admit any of them."""
    ids = []
    for text in _SIGNALLED_TEXTS:
        result = capture_turn(
            store, cue="", text=text, tier="episodic",
            session_id=session_id, role="user", live_turn=False,
            salience_level="notable",
        )
        ids.append(UUID(result["record_id"]))
    flush_record_buffer(store)

    for text in _ALL_DISTRACTORS[:_DISTRACTOR_COUNT]:
        capture_turn(
            store, cue="", text=text, tier="episodic",
            session_id=SESSION_CHATTER, role="user", live_turn=False,
        )
    flush_record_buffer(store)
    return ids


def _assert_excluded_from_primary_window(store: MemoryStore, cue_embedding, ids) -> None:
    window_k = (
        foresight.FORESIGHT_MAX_ITEMS_DEFAULT
        + foresight.FORESIGHT_ASSISTANT_TAIL_RESERVE_DEFAULT
    ) * foresight._CANDIDATE_OVERFETCH
    primary = store.query_similar(cue_embedding, k=window_k)
    primary_ids = {rec.id for rec, _cos in primary}
    assert not (set(ids) & primary_ids), (
        "the signalled records must sit OUTSIDE the real production "
        "admission window for this to be a non-vacuous reserve-lane proof "
        "-- raw cosine alone would already admit them otherwise"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_reserve_lane_admits_excluded_record_and_stays_capped(driver, tmp_path):
    """Non-vacuous rescue + bounded-slot proof in one measurement: three
    salience-tagged records all sit outside the real admission window, but
    only foresight.FORESIGHT_RESERVE_LANE_CAP_DEFAULT of them are served --
    the lane admits by existence, not cosine, and never more than its own
    slot budget."""
    mp = pytest.MonkeyPatch()
    _select_driver(driver, mp)
    try:
        store = MemoryStore(path=tmp_path / f"reserve-lane-{driver}")
        try:
            signalled_ids = _build_corpus(store)

            embedder = embedder_for_store(store)
            cue_embedding = list(embedder.embed(_QUERY_TEXT))
            _assert_excluded_from_primary_window(store, cue_embedding, signalled_ids)

            report = foresight.refresh_pack(
                store, cue_text=_QUERY_TEXT, cue_embedding=cue_embedding,
                session_id=SESSION_QUERY,
            )
            packed_ids = set(report.get("packed_ids") or [])
            signalled_served = {str(i) for i in signalled_ids} & packed_ids

            assert signalled_served, (
                f"the reserve lane must admit at least one signalled record "
                f"the primary window excluded: report={report}"
            )
            assert len(signalled_served) == foresight.FORESIGHT_RESERVE_LANE_CAP_DEFAULT, (
                f"the reserve lane must never exceed its own slot cap "
                f"({foresight.FORESIGHT_RESERVE_LANE_CAP_DEFAULT}); served "
                f"{len(signalled_served)} of {len(signalled_ids)} eligible "
                f"signalled records: report={report}"
            )
        finally:
            store.close()
    finally:
        mp.undo()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_reserve_lane_degrades_to_empty_on_discovery_failure(driver, tmp_path, monkeypatch):
    """A forced failure in the lane's own discovery call must never break
    the pack -- refresh_pack still returns a well-formed report, exactly
    the degrade-to-empty discipline the other three reserve lanes already
    honor."""
    mp = pytest.MonkeyPatch()
    _select_driver(driver, mp)
    try:
        store = MemoryStore(path=tmp_path / f"reserve-lane-fail-{driver}")
        try:
            _build_corpus(store)

            embedder = embedder_for_store(store)
            cue_embedding = list(embedder.embed(_QUERY_TEXT))

            def _boom(*_args, **_kwargs):
                raise RuntimeError("forced reserve-lane discovery failure")

            monkeypatch.setattr(store, "iter_records", _boom)

            report = foresight.refresh_pack(
                store, cue_text=_QUERY_TEXT, cue_embedding=cue_embedding,
                session_id=SESSION_QUERY,
            )
            assert isinstance(report, dict) and "packed_ids" in report, (
                f"refresh_pack must still return a well-formed report when "
                f"the reserve lane's own discovery call raises: {report}"
            )
        finally:
            store.close()
    finally:
        mp.undo()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_reserve_lane_never_bleeds_across_sessions(driver, tmp_path):
    """A signalled record captured under session A must never surface in
    session A's own served pack (the same current-session exclusion every
    other lane already honors), yet remains discoverable from a different
    session -- proving the lane routes through the existing guard rather
    than adding a new, possibly divergent one."""
    mp = pytest.MonkeyPatch()
    _select_driver(driver, mp)
    try:
        store = MemoryStore(path=tmp_path / f"reserve-lane-session-{driver}")
        try:
            own_session = "reserve-lane-own-session"
            signalled_ids = _build_corpus(store, session_id=own_session)

            embedder = embedder_for_store(store)
            cue_embedding = list(embedder.embed(_QUERY_TEXT))
            _assert_excluded_from_primary_window(store, cue_embedding, signalled_ids)

            same_session_report = foresight.refresh_pack(
                store, cue_text=_QUERY_TEXT, cue_embedding=cue_embedding,
                session_id=own_session,
            )
            same_session_packed = set(same_session_report.get("packed_ids") or [])
            assert not ({str(i) for i in signalled_ids} & same_session_packed), (
                f"a session must never see its own records surfaced back to "
                f"it via the reserve lane: report={same_session_report}"
            )

            other_session_report = foresight.refresh_pack(
                store, cue_text=_QUERY_TEXT, cue_embedding=cue_embedding,
                session_id=SESSION_QUERY,
            )
            other_session_packed = set(other_session_report.get("packed_ids") or [])
            assert {str(i) for i in signalled_ids} & other_session_packed, (
                f"a DIFFERENT session must still be able to discover the "
                f"signalled record via the reserve lane -- otherwise this "
                f"proof cannot distinguish 'session-scoped' from 'broken': "
                f"report={other_session_report}"
            )
        finally:
            store.close()
    finally:
        mp.undo()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_reserve_lane_is_no_op_without_any_signal(driver):
    """On a corpus with no salience/directive signal at all, the reserve
    lane's indexed discovery finds nothing, so the served pack is exactly
    the pre-lane admission-window-narrow outcome: the excluded knowledge
    record stays excluded and a register-matched distractor is served in
    its place."""
    result = _get_admission_window_probe(driver)
    assert result["outcome"] == "constructed", (
        f"[{driver}] the unflagged construction this no-op proof depends "
        f"on did not reproduce: {result}"
    )
    assert not result["knowledge_served"], (
        f"[{driver}] the reserve lane must be a no-op when nothing in the "
        f"corpus carries a signal -- the unflagged record must stay "
        f"excluded exactly as it did before this lane existed: {result}"
    )
