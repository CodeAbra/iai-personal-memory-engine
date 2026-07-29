from __future__ import annotations

from bench.longitudinal_score import (
    poison_at_k,
    rank_of_substring,
    reciprocal_rank_of,
    rescue_at_k,
)

TEXTS = [
    "The team meets every Tuesday for sync.",
    "Correction: launch moved to 2026-09-01.",
    "The product launches on 2026-06-01.",
    "Monitoring dashboards live in the ops portal.",
]


def test_rank_of_substring_first_match_wins() -> None:
    assert rank_of_substring(TEXTS, "2026-09-01") == 2
    assert rank_of_substring(TEXTS, "2026-06-01") == 3
    assert rank_of_substring(TEXTS, "absent value") == -1
    assert rank_of_substring([], "anything") == -1
    assert rank_of_substring(TEXTS, "") == -1


def test_rescue_at_k_respects_cutoff() -> None:
    assert rescue_at_k(TEXTS, "2026-09-01", 1) == 0
    assert rescue_at_k(TEXTS, "2026-09-01", 2) == 1
    assert rescue_at_k(TEXTS, "2026-06-01", 2) == 0
    assert rescue_at_k(TEXTS, "2026-06-01", 5) == 1
    assert rescue_at_k(TEXTS, "absent value", 10) == 0


def test_poison_counterfact_above_expects() -> None:
    # counterfact at rank 2, expects at rank 3 -> poisoned at k>=3
    assert poison_at_k(TEXTS, "2026-06-01", "2026-09-01", 3) == 1
    # expects above counterfact -> clean
    assert poison_at_k(TEXTS, "2026-09-01", "2026-06-01", 5) == 0


def test_poison_when_expects_missing_from_topk() -> None:
    # counterfact present, expects absent from top-k -> poisoned
    assert poison_at_k(TEXTS, "2026-06-01", "2026-09-01", 2) == 1
    # neither present in top-k -> clean
    assert poison_at_k(TEXTS, "2026-06-01", "2026-09-01", 1) == 0
    # counterfact absent entirely -> clean regardless of expects
    assert poison_at_k(TEXTS, "2026-06-01", "absent value", 10) == 0


def test_reciprocal_rank_of() -> None:
    assert reciprocal_rank_of(TEXTS, "Tuesday") == 1.0
    assert reciprocal_rank_of(TEXTS, "2026-09-01") == 0.5
    assert reciprocal_rank_of(TEXTS, "absent value") == 0.0
