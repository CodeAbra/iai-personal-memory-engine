"""B-lane quality-gate MACHINERY -- A/A floor + non-vacuity plant-and-catch,
proven on SYNTHETIC data only.

Reuses bench.proc_corpus_census's per_pair_deltas / value_metric /
null_per_pair_deltas / two_sided_aa_floor unmodified -- the machinery is
generic over any directed-pair population (an established (A, B) pair, its
first-occurrence session, and its rank<=K recall observations); Signal B's
tool-bigram pairs are just another instance of that same shape, not a
reimplementation.

What this file proves: the machinery FIRES on a real planted improving
effect and STAYS WITHIN the null band on i.i.d. noise -- non-vacuity of the
measuring code. What it explicitly does NOT do: report a live B-lane
verdict. The real verdict needs actual tool_sequence chunks minted by live
sleep-pipeline runs plus real recall traffic to measure an A/A floor
against -- that is deferred, marked below, never silently skipped. No test
here claims to reproduce or validate a real bigram feasibility number;
these fixtures prove only that the measuring CODE is correct, never the
number.
"""
from __future__ import annotations

import random as _random
from datetime import datetime, timedelta, timezone

import pytest

from bench.proc_corpus_census import (
    RANK_K,
    null_per_pair_deltas,
    per_pair_deltas,
    two_sided_aa_floor,
    value_metric,
)

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _established_pair_row(session_id: str, pair: "tuple[str, str]", ts: datetime) -> dict:
    a, b = pair
    return {"data": {"reinforced_ids": [a, b]}, "session_id": session_id, "ts": ts}


def _recall_row(session_id: str, hit_ids: "list[str]", ts: datetime) -> dict:
    return {"data": {"hit_ids": hit_ids}, "session_id": session_id, "ts": ts}


def _planted_b_signal_corpus(
    n_bigrams: int = 20, n_repeat_sessions: int = 4,
) -> "tuple[list[dict], list[dict]]":
    """N established tool-bigram pairs, each with one HISTORY (miss)
    occurrence and several HELD-OUT (hit) occurrences -- a manufactured
    improving-rank trend, uniform across every bigram."""
    established: list[dict] = []
    recalls: list[dict] = []
    for i in range(n_bigrams):
        a, b = f"tool-bigram-a{i}", f"tool-bigram-b{i}"
        first_session = f"b-first-{i}"
        established.append(_established_pair_row(first_session, (a, b), _T0))
        recalls.append(
            _recall_row(
                first_session,
                [f"filler-{j}" for j in range(RANK_K + 1)] + [b],
                _T0 - timedelta(seconds=1),
            )
        )
        for r in range(n_repeat_sessions):
            session = f"b-repeat-{i}-{r}"
            ts = _T0 + timedelta(days=10, hours=r)
            established.append(_established_pair_row(session, (a, b), ts))
            recalls.append(_recall_row(session, [b, "filler"], ts - timedelta(seconds=1)))
    return established, recalls


def _iid_b_noise_corpus(
    n_bigrams: int = 20, n_repeat_sessions: int = 6, seed: int = 0,
) -> "tuple[list[dict], list[dict]]":
    """N established tool-bigram pairs where every recall outcome is an
    independent 50/50 coin flip, uncorrelated with FIRST-vs-REPEAT status --
    no rank/repetition relationship planted anywhere."""
    rng = _random.Random(seed)
    established: list[dict] = []
    recalls: list[dict] = []

    def _observation(session: str, ts: datetime, b: str) -> None:
        if rng.random() < 0.5:
            recalls.append(_recall_row(session, [b, "filler"], ts - timedelta(seconds=1)))
        else:
            recalls.append(
                _recall_row(
                    session,
                    [f"filler-{j}" for j in range(RANK_K + 1)] + [b],
                    ts - timedelta(seconds=1),
                )
            )

    for i in range(n_bigrams):
        a, b = f"noise-bigram-a{i}", f"noise-bigram-b{i}"
        first_session = f"b-noise-first-{i}"
        established.append(_established_pair_row(first_session, (a, b), _T0))
        _observation(first_session, _T0, b)
        for r in range(n_repeat_sessions):
            session = f"b-noise-repeat-{i}-{r}"
            ts = _T0 + timedelta(days=10, hours=r)
            established.append(_established_pair_row(session, (a, b), ts))
            _observation(session, ts, b)
    return established, recalls


def test_non_vacuity_planted_improving_signal_clears_the_floor():
    established, recalls = _planted_b_signal_corpus()
    t_cutoff = _T0 + timedelta(days=1)

    vm = value_metric(established, recalls, t_cutoff=t_cutoff, k=RANK_K)
    null_deltas = null_per_pair_deltas(established, recalls, t_cutoff=t_cutoff, k=RANK_K)
    low, high = two_sided_aa_floor(null_deltas, iters=300, seed=0)

    assert vm > high, "planted improving-rank B-lane signal did not clear its own A/A floor"


def test_non_vacuity_iid_noise_stays_within_the_floor():
    established, recalls = _iid_b_noise_corpus()
    t_cutoff = _T0 + timedelta(days=1)

    vm = value_metric(established, recalls, t_cutoff=t_cutoff, k=RANK_K)
    null_deltas = null_per_pair_deltas(established, recalls, t_cutoff=t_cutoff, k=RANK_K)
    low, high = two_sided_aa_floor(null_deltas, iters=500, seed=0)

    assert low <= vm <= high, (
        "i.i.d. B-lane noise with no planted relationship was flagged as a signal"
    )


def test_two_sided_aa_floor_brackets_zero_on_balanced_null_deltas():
    deltas = {(f"tool-a{i}", f"tool-b{i}"): 0.5 for i in range(5)}
    deltas.update({(f"tool-c{i}", f"tool-d{i}"): -0.5 for i in range(5)})
    low, high = two_sided_aa_floor(deltas, iters=300, seed=0)
    assert low <= 0.0
    assert high >= 0.0


def test_non_vacuity_machinery_consumes_per_pair_deltas_shape_directly():
    """The scalar value_metric and the deltas it is the mean of come from the
    SAME per_pair_deltas call -- no separately-maintained reimplementation of
    the formula."""
    established, recalls = _planted_b_signal_corpus()
    t_cutoff = _T0 + timedelta(days=1)

    deltas = per_pair_deltas(established, recalls, t_cutoff=t_cutoff, k=RANK_K)
    vm = value_metric(established, recalls, t_cutoff=t_cutoff, k=RANK_K)
    assert deltas
    assert vm == pytest.approx(sum(deltas.values()) / len(deltas))


@pytest.mark.skip(
    reason=(
        "live B-lane verdict awaits live sleep-pipeline accumulation and "
        "real recall traffic. The machinery above is proven non-vacuous on "
        "synthetic data; the real verdict is not reproducible from any "
        "fixture and is deferred here, never silently absent."
    )
)
def test_b_lane_live_aa_floor_verdict_deferred_pending_live_sleep_pipeline_accumulation():
    raise AssertionError(
        "the live B-lane verdict cannot run in this test suite -- it "
        "needs real tool_sequence chunks minted by live sleep-pipeline runs "
        "plus real recall traffic to measure an A/A floor against"
    )
