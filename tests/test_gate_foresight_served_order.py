"""Sibling gate to test_gate_foresight_admission.py: proves not just that a
directive-signalled knowledge record is admitted onto the served pack where
an identical unflagged construction excludes it, but that it is served
FIRST -- ahead of every register-matched chatter distractor that remains
present at later positions. Imports the baseline module's fixture
construction UNMODIFIED and never mocks admission.

Shares `_PROBE_CACHE` (in test_baseline_foresight_admission.py) with
test_gate_foresight_admission.py, keyed by (driver, directive,
salience_level): whichever gate module pytest collects and runs first
performs the real construction, and the other reads that cached result.
Under a parallel run that only collects one of the two modules per worker,
each worker recomputes independently instead of sharing the cache.
"""
from __future__ import annotations

import json

import pytest

from iai_mcp import foresight
from tests.test_baseline_foresight_admission import _get_admission_window_probe


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_directive_signalled_knowledge_served_first_ahead_of_chatter(driver):
    """The directive-signalled knowledge record occupies packed_ids index 0
    while register-matched chatter distractors remain present at later
    positions, and the pack does not grow past the per-turn ceiling."""
    result = _get_admission_window_probe(driver, directive=True)
    print(json.dumps({"driver": driver, **result}, indent=2, default=str))

    if result["outcome"] == "inconclusive":
        pytest.fail(f"[{driver}] INCONCLUSIVE construction: {result}")

    assert result["knowledge_served"], (
        f"[{driver}] the directive-signalled knowledge record must be "
        f"served: {result}"
    )
    assert result["knowledge_index"] == 0, (
        f"[{driver}] the directive-signalled knowledge record must be "
        f"served FIRST (packed_ids index 0), ahead of every chatter "
        f"distractor: {result}"
    )
    assert result["distractor_served_count"] > 0, (
        f"[{driver}] register-matched chatter must remain present at later "
        f"positions -- knowledge precedes chatter, it does not displace it "
        f"off the pack: {result}"
    )
    ceiling = (
        foresight.FORESIGHT_MAX_ITEMS_DEFAULT
        + foresight.FORESIGHT_ASSISTANT_TAIL_RESERVE_DEFAULT
    )
    assert result["packed"] <= ceiling, (
        f"[{driver}] the served pack must not exceed today's per-turn "
        f"ceiling ({ceiling}): {result}"
    )
