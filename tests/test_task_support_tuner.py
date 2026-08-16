"""Pure-unit coverage of the task_support tuning spec: honest same-session
follow-through observe, hysteretic down move, and probe-verdict recovery
that only ever moves through the scheduler's marker.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from iai_mcp.lilli.cycle.sleep_pipeline._knob_tune_specs import (
    MIN_DISPLAYED_SESSIONS,
    MIN_PROBE_FOLLOW_SESSIONS,
    PROBE_WINDOW_SESSIONS,
    RECOVER_THRESHOLD,
    TUNING_SPECS,
    _apply_task_support_tuning,
    _observe_task_support,
    _task_support_session_votes,
)
from iai_mcp.lilli.profile.knobs import bayesian_update

SPEC = TUNING_SPECS["task_support"]


def _row(
    *,
    session_id: str,
    ts: datetime,
    hit_ids: list[str],
    suggestion_ids: list[str] | None = None,
    suggestions_visible: bool | None = None,
    probe: bool = False,
    qualifying: bool = True,
    path: str = "recall_for_response",
) -> dict:
    if not qualifying:
        return {
            "hit_ids": hit_ids,
            "query": "cue",
            "used": bool(hit_ids),
            "budget_used": 10,
            "path": path,
        }
    return {
        "hit_ids": hit_ids,
        "query": "cue",
        "used": bool(hit_ids),
        "budget_used": 10,
        "path": path,
        "session_id": session_id,
        "timestamp": ts.isoformat(),
        "suggestion_ids": suggestion_ids or [],
        "suggestions_visible": True if suggestions_visible is None else suggestions_visible,
        "probe": probe,
    }


def _displayed_session(
    prefix: str, i: int, base: datetime, *, followed: bool, probe: bool = False,
) -> list[dict]:
    """Two consecutive same-session rows: first shows [S], second either
    exact-id-hits S (follow) or hits something disjoint (non-follow)."""
    sid = f"{prefix}-{i}"
    t0 = base + timedelta(minutes=10 * i)
    t1 = t0 + timedelta(minutes=1)
    shown = f"sugg-{prefix}-{i}"
    first = _row(
        session_id=sid, ts=t0, hit_ids=[f"hit-{prefix}-{i}-a"],
        suggestion_ids=[shown], suggestions_visible=True, probe=probe,
    )
    second_hit = shown if followed else f"hit-{prefix}-{i}-unrelated"
    second = _row(
        session_id=sid, ts=t1, hit_ids=[second_hit],
        suggestion_ids=[], suggestions_visible=True, probe=probe,
    )
    return [second, first]  # rows arrive ts-DESC


def _desc(rows_asc: list[list[dict]]) -> list[dict]:
    """Flatten a list of per-session [second, first] pairs (already DESC
    within a session) into one overall ts-DESC list, newest session first."""
    out: list[dict] = []
    for pair in reversed(rows_asc):
        out.extend(pair)
    return out


# ---------------------------------------------------------------------------
# 1. Real follows: keeps cued_recognition (observe abstains, no down move)
# ---------------------------------------------------------------------------


def test_displayed_window_with_real_follows_observes_nothing() -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    sessions = [
        _displayed_session("follow", i, base, followed=True)
        for i in range(MIN_DISPLAYED_SESSIONS)
    ]
    rows = _desc(sessions)

    observed, n, signal = _observe_task_support(
        {"retrieval_used": rows}, current="cued_recognition",
    )

    assert observed is None
    assert n == 0
    assert signal == "implicit"


# ---------------------------------------------------------------------------
# 2. Zero follows over >= MIN_DISPLAYED_SESSIONS: observes blank_recall
# ---------------------------------------------------------------------------


def test_displayed_window_with_zero_follows_observes_blank_recall() -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    sessions = [
        _displayed_session("nofollow", i, base, followed=False)
        for i in range(MIN_DISPLAYED_SESSIONS)
    ]
    rows = _desc(sessions)

    observed, n, signal = _observe_task_support(
        {"retrieval_used": rows}, current="cued_recognition",
    )

    assert observed == "blank_recall"
    assert n == MIN_DISPLAYED_SESSIONS
    assert signal == "implicit"


# ---------------------------------------------------------------------------
# 3. Hidden-only window: (None, 0, ...)
# ---------------------------------------------------------------------------


def test_hidden_only_window_observes_nothing() -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(MIN_DISPLAYED_SESSIONS):
        sid = f"hidden-{i}"
        t0 = base + timedelta(minutes=10 * i)
        rows.append(_row(
            session_id=sid, ts=t0 + timedelta(minutes=1),
            hit_ids=[f"hit-{i}-b"], suggestions_visible=False,
        ))
        rows.append(_row(
            session_id=sid, ts=t0, hit_ids=[f"hit-{i}-a"],
            suggestion_ids=[f"sugg-{i}"], suggestions_visible=False,
        ))
    rows.reverse()  # newest first

    observed, n, signal = _observe_task_support(
        {"retrieval_used": rows}, current="blank_recall",
    )

    assert observed is None
    assert n == 0
    assert signal == "implicit"


# ---------------------------------------------------------------------------
# 4. Probe-close recovery: apply with the marker SETS cued_recognition
# ---------------------------------------------------------------------------


def test_probe_close_recovery_sets_cued_recognition_via_marker() -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    sessions = [
        _displayed_session("probe", i, base, followed=True, probe=True)
        for i in range(MIN_PROBE_FOLLOW_SESSIONS)
    ]
    rows = _desc(sessions)

    observed, n, signal = _observe_task_support(
        {"retrieval_used": rows}, current="blank_recall",
    )
    assert observed == "cued_recognition"
    assert n == MIN_PROBE_FOLLOW_SESSIONS
    assert signal == "implicit"

    marker_posterior = {"probe_active_until": "2026-01-02T00:00:00+00:00"}
    result = _apply_task_support_tuning("blank_recall", observed, marker_posterior)
    assert result == "cued_recognition"


# ---------------------------------------------------------------------------
# 5. Probe-close with too few evaluable sessions: empty probe, no move
# ---------------------------------------------------------------------------


def test_probe_close_too_few_sessions_is_empty_probe() -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    sessions = [
        _displayed_session("thinprobe", i, base, followed=True, probe=True)
        for i in range(MIN_PROBE_FOLLOW_SESSIONS - 1)
    ]
    rows = _desc(sessions)

    observed, n, signal = _observe_task_support(
        {"retrieval_used": rows}, current="blank_recall",
    )

    assert observed is None
    assert n == 0


# ---------------------------------------------------------------------------
# 6. apply WITHOUT the marker cannot jump blank_recall -> cued_recognition
# ---------------------------------------------------------------------------


def test_apply_without_marker_never_jumps_up_on_thin_signal() -> None:
    thin_posterior = {"alphas": {"cued_recognition": 5.0, "blank_recall": 0.3}}
    result = _apply_task_support_tuning("blank_recall", "cued_recognition", thin_posterior)
    assert result == "blank_recall"

    no_posterior_result = _apply_task_support_tuning("blank_recall", "cued_recognition", {})
    assert no_posterior_result == "blank_recall"


# ---------------------------------------------------------------------------
# 6b. The down branch itself: hysteresis band gates the blank_recall flip
# ---------------------------------------------------------------------------


def test_apply_down_move_respects_hysteresis_band() -> None:
    # Band not cleared (diff < 2.0) -- a narrow argmax win must not flip yet.
    held = _apply_task_support_tuning(
        "cued_recognition", "blank_recall",
        {"alphas": {"blank_recall": 3.1, "cued_recognition": 3.0}},
    )
    assert held == "cued_recognition"

    # Band cleared (diff >= 2.0) -- sustained down-votes finally flip it.
    moved = _apply_task_support_tuning(
        "cued_recognition", "blank_recall",
        {"alphas": {"blank_recall": 5.2, "cued_recognition": 3.0}},
    )
    assert moved == "blank_recall"


# ---------------------------------------------------------------------------
# 7. NO-CRASH: a non-qualifying row mixed into the window
# ---------------------------------------------------------------------------


def test_non_qualifying_row_does_not_crash_observe() -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    sessions = [
        _displayed_session("mix", i, base, followed=False)
        for i in range(MIN_DISPLAYED_SESSIONS)
    ]
    rows = _desc(sessions)
    rows.insert(1, _row(session_id="?", ts=base, hit_ids=["x"], qualifying=False))

    observed, n, signal = _observe_task_support(
        {"retrieval_used": rows}, current="cued_recognition",
    )
    assert signal == "implicit"


# ---------------------------------------------------------------------------
# 8. MARKER-SURVIVAL LINKAGE: causal chain through bayesian_update
# ---------------------------------------------------------------------------


def test_marker_survives_bayesian_update_and_wins_argmax_because_reset() -> None:
    state = {"task_support": "blank_recall"}
    reset_posterior = {"task_support": {"probe_active_until": "2026-01-02T00:00:00+00:00"}}

    new_raw, new_post = bayesian_update(
        "task_support", "implicit", "cued_recognition", state, reset_posterior,
    )

    assert new_post["task_support"]["probe_active_until"] == "2026-01-02T00:00:00+00:00"
    assert new_raw == "cued_recognition", (
        "the emptied alphas must let the single probe observation win the "
        "argmax against the current incumbent's epsilon mass"
    )
    verdict = _apply_task_support_tuning("blank_recall", new_raw, new_post["task_support"])
    assert verdict == "cued_recognition"

    # Contrast: an UNRESET posterior (incumbent still seeded) must NOT flip
    # on the same single observation -- proves the reset is what wins it,
    # not the observation alone.
    seeded_posterior = {"task_support": {"alphas": {"blank_recall": 3.0}}}
    locked_raw, _locked_post = bayesian_update(
        "task_support", "implicit", "cued_recognition",
        {"task_support": "blank_recall"}, seeded_posterior,
    )
    assert locked_raw == "blank_recall"


# ---------------------------------------------------------------------------
# 9. INTERLEAVE-ABSTAIN: a non-qualifying row between two survivors must
# never produce a false non-follow
# ---------------------------------------------------------------------------


def test_interleaved_non_qualifying_row_abstains_not_false_non_follow() -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t1 = _row(
        session_id="s1", ts=base, hit_ids=["hitA"],
        suggestion_ids=["A", "B"], suggestions_visible=True, probe=False,
    )
    t2 = _row(session_id="s1", ts=base + timedelta(minutes=1), hit_ids=["A"], qualifying=False)
    t3 = _row(
        session_id="s1", ts=base + timedelta(minutes=2), hit_ids=["hitC"],
        suggestion_ids=[], suggestions_visible=True, probe=False,
    )

    with_interposed = [t3, t2, t1]  # ts-DESC
    probe_votes, ordinary_votes = _task_support_session_votes(with_interposed)
    assert "s1" not in ordinary_votes, (
        "the interposing non-qualifying row must make (t1, t3) non-consecutive -- "
        "abstained, not scored as a false non-follow"
    )
    assert not probe_votes

    control = [t3, t1]  # same t1/t3, no interposing row
    _probe_votes_c, ordinary_votes_c = _task_support_session_votes(control)
    assert ordinary_votes_c.get("s1") is False, (
        "the control (no interposing row) proves the abstain above is the "
        "boundary's doing, not a dead assertion -- (t1, t3) directly adjacent "
        "IS scored, and correctly as a non-follow since hits are disjoint"
    )
