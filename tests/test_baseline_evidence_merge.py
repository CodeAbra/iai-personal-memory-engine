"""Regression guard for the baseline-evidence merge logic in
_baseline_evidence_merge.py -- pure dict manipulation, no embedder, no
bench flag required, runs in the default correctness lane."""
from __future__ import annotations

import pytest

from tests._baseline_evidence_merge import (
    _IncompleteEvidenceError,
    _merge_and_validate_evidence,
)


def test_merge_and_validate_evidence_completeness_guard():
    """Completeness is checked against THIS RUN's own per-driver
    contribution, not the merged result: an already-complete `existing`
    entry must never let a partial update for that same driver ride
    through unnoticed."""
    existing = {
        "stdlib": {
            "reproduced": True, "shape": "ranking-order-bury", "miss_rate": 0.5,
            "recall_served_rank": 2, "unrelated_kept_key": "x",
        },
    }
    complete_new = {
        "lilli": {
            "reproduced": True, "shape": "ranking-floor-drop", "miss_rate": 0.5,
            "recall_served_rank": 1,
        },
    }
    merged = _merge_and_validate_evidence(existing, complete_new)
    assert merged["stdlib"]["unrelated_kept_key"] == "x"
    assert merged["lilli"]["reproduced"] is True

    # Stale-forward regression: `stdlib` already has a complete entry in
    # `existing`, but this run's own contribution for `stdlib` is short of
    # the required fields. Checking the merged result would pass (the
    # missing fields are inherited from `existing`) -- checking this run's
    # own contribution must not.
    incomplete_for_already_complete_driver = {
        "stdlib": {
            "tree_sha": "def456", "tree_clean": True, "driver": "stdlib",
            "reproduced": False, "shape": "no-served-bury",
        },
    }
    with pytest.raises(_IncompleteEvidenceError):
        _merge_and_validate_evidence(existing, incomplete_for_already_complete_driver)

    # Brand-new driver, absent from `existing`, with an incomplete
    # contribution: nothing to inherit from, so this must also fail loud.
    incomplete_for_fresh_driver = {"gpt-astra": {"tree_sha": "abc123", "driver": "gpt-astra"}}
    with pytest.raises(_IncompleteEvidenceError):
        _merge_and_validate_evidence(existing, incomplete_for_fresh_driver)


def test_merge_and_validate_evidence_custom_required_fields():
    """A caller with a different evidence shape passes its own
    required-field tuple -- the default stays untouched for existing
    callers."""
    custom_fields = ("driver", "improved")
    complete = {"lilli": {"driver": "lilli", "improved": True}}
    merged = _merge_and_validate_evidence({}, complete, required_fields=custom_fields)
    assert merged["lilli"]["improved"] is True

    incomplete = {"lilli": {"driver": "lilli"}}
    with pytest.raises(_IncompleteEvidenceError):
        _merge_and_validate_evidence({}, incomplete, required_fields=custom_fields)
