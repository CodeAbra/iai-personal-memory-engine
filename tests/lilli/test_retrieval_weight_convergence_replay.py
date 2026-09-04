"""Pure, store-free multi-night replay of the retrieval rank weight tuner:
proves the bounded-delta design (SC2) actually settles rather than
oscillates or walks off, in both directions, and never perturbs the sealed
10-knob registry.

Convergence contract: apply_retrieval_weight's per-night delta comes from
`(use_rate - 0.5) * LEARN_RATE` (tuner.py), which does NOT depend on the
current weight -- so a SUSTAINED one-sided use_rate does not settle at an
interior fixed point, it walks linearly to the production clamp edge and
holds there. "Converges" below therefore means "reaches and holds at the
clamp edge", not "approaches an interior value".
"""
from __future__ import annotations

import pytest

from iai_mcp.lilli.profile.knobs import PROFILE_KNOBS
from iai_mcp.lilli.profile.retrieval_tuning import (
    DEFAULT_W_COSINE,
    PROD_W_COSINE_MAX,
    PROD_W_COSINE_MIN,
    RETRIEVAL_MIN_SAMPLES,
    apply_retrieval_weight,
)

_NIGHTS: int = 150
_SETTLE_TAIL: int = 5
_SETTLE_TOLERANCE: float = 1e-9


def _replay(steady_state_use_rate: float, nights: int = _NIGHTS) -> list[float]:
    """Carry the weight forward across `nights` simulated nightly windows,
    each with a fixed steady-state use_rate and n == RETRIEVAL_MIN_SAMPLES.
    Returns the sequence of weights AFTER each night (index 0 = night 1)."""
    weight = DEFAULT_W_COSINE
    history: list[float] = []
    for _ in range(nights):
        weight = apply_retrieval_weight(weight, steady_state_use_rate, RETRIEVAL_MIN_SAMPLES)
        history.append(weight)
    return history


def _record_specs() -> tuple:
    return tuple(sorted(PROFILE_KNOBS.items(), key=lambda kv: kv[0]))


# ---------------------------------------------------------------------------
# Upward: sustained above-neutral use_rate
# ---------------------------------------------------------------------------


def test_converges_upward_to_clamp_edge_and_holds() -> None:
    history = _replay(steady_state_use_rate=0.8)

    for w in history:
        assert PROD_W_COSINE_MIN <= w <= PROD_W_COSINE_MAX, "no single night may overshoot the clamp"

    assert history[-1] == pytest.approx(PROD_W_COSINE_MAX)

    tail_deltas = [
        abs(history[i] - history[i - 1]) for i in range(len(history) - _SETTLE_TAIL, len(history))
    ]
    assert all(d <= _SETTLE_TOLERANCE for d in tail_deltas), (
        "the tail of the replay must hold at the clamp edge, not keep moving"
    )


def test_upward_replay_is_monotone_non_decreasing() -> None:
    history = _replay(steady_state_use_rate=0.8)

    for prev, cur in zip(history, history[1:]):
        assert cur >= prev - _SETTLE_TOLERANCE, "an above-neutral use_rate must never push the weight down"


# ---------------------------------------------------------------------------
# Downward: sustained below-neutral use_rate
# ---------------------------------------------------------------------------


def test_converges_downward_to_clamp_edge_without_dropping_below_it() -> None:
    history = _replay(steady_state_use_rate=0.2)

    for w in history:
        assert PROD_W_COSINE_MIN <= w <= PROD_W_COSINE_MAX, "no single night may undershoot the clamp"

    assert history[-1] == pytest.approx(PROD_W_COSINE_MIN)

    tail_deltas = [
        abs(history[i] - history[i - 1]) for i in range(len(history) - _SETTLE_TAIL, len(history))
    ]
    assert all(d <= _SETTLE_TOLERANCE for d in tail_deltas), (
        "the tail of the replay must hold at the clamp edge, not keep moving"
    )


def test_downward_replay_is_monotone_non_increasing() -> None:
    history = _replay(steady_state_use_rate=0.2)

    for prev, cur in zip(history, history[1:]):
        assert cur <= prev + _SETTLE_TOLERANCE, "a below-neutral use_rate must never push the weight up"


# ---------------------------------------------------------------------------
# No overshoot at any single night, across a range of use_rates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_rate", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
def test_no_overshoot_on_any_single_night_across_use_rates(use_rate: float) -> None:
    history = _replay(steady_state_use_rate=use_rate, nights=200)
    assert all(PROD_W_COSINE_MIN <= w <= PROD_W_COSINE_MAX for w in history)


# ---------------------------------------------------------------------------
# Sealed 10-knob registry untouched
# ---------------------------------------------------------------------------


def test_replay_never_touches_the_sealed_knob_registry() -> None:
    before = _record_specs()

    _replay(steady_state_use_rate=0.8)
    _replay(steady_state_use_rate=0.2)

    after = _record_specs()
    assert after == before, "the tuning math must never mutate PROFILE_KNOBS"


def test_replay_at_the_neutral_use_rate_stays_at_the_default() -> None:
    """use_rate == 0.5 is the fixed point at n == RETRIEVAL_MIN_SAMPLES --
    zero raw delta every night, so the weight never leaves the default,
    positive control for the two clamp-edge cases above."""
    history = _replay(steady_state_use_rate=0.5)

    assert all(w == pytest.approx(DEFAULT_W_COSINE) for w in history)
