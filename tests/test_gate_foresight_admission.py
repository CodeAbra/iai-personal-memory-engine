"""Sibling gate to test_baseline_foresight_admission.py: re-runs the exact
admission-window-narrow construction that reproduces the unflagged red
baseline, but with the knowledge record captured through capture_turn's real
directive channel (directive=True), which the shipped classify_salience path
promotes to a live, non-"unflagged" salience_level -- never a hand-set
salience_level. Imports the baseline module's fixture construction
UNMODIFIED and writes its own proof artifact, mirroring the
test_gate_knowledge_vs_chatter.py / test_baseline_knowledge_vs_chatter.py
sibling split.

Honest claim: the lane this gate proves is an existence-based admission
path inside refresh_pack, relocated to run before the cosine-ranked lanes --
an existence-admitted record is now served first, ahead of an
already-served distractor, by list position rather than by score. What
this module measures is admission itself: the identical construction that
excludes the unflagged record entirely now serves the signalled one,
alongside the still-present chatter. That excluded-to-served delta is what
this module measures; the served-order guarantee itself is proved by the
sibling served-order gate.

Shares `_PROBE_CACHE` (in test_baseline_foresight_admission.py) with
test_gate_foresight_served_order.py, keyed by (driver, directive,
salience_level): whichever gate module pytest collects and runs first
performs the real construction, and the other reads that cached result.
Under a parallel run that only collects one of the two modules per worker,
each worker recomputes independently instead of sharing the cache.
"""
from __future__ import annotations

import json

import pytest

from tests.test_baseline_foresight_admission import _get_admission_window_probe


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_directive_signalled_knowledge_admitted_where_unflagged_is_excluded(driver):
    """Re-runs the sibling red baseline's construction twice on the same
    driver: once unflagged (must reproduce the excluded/red shape) and once
    with the knowledge record captured with a real directive flag (must
    flip to admitted). Both a real classify_salience promotion (never a
    hand-set salience_level) and a non-vacuous chatter presence are
    required for the flip to count."""
    unflagged = _get_admission_window_probe(driver)
    signalled = _get_admission_window_probe(driver, directive=True)
    print(json.dumps(
        {"driver": driver, "unflagged": unflagged, "signalled": signalled},
        indent=2, default=str,
    ))

    for label, result in (("unflagged", unflagged), ("signalled", signalled)):
        if result["outcome"] == "inconclusive":
            pytest.fail(f"[{driver}/{label}] INCONCLUSIVE construction: {result}")

    assert not unflagged["knowledge_served"], (
        f"[{driver}] the unflagged control must reproduce the sibling red "
        f"baseline (knowledge excluded): {unflagged}"
    )
    assert unflagged["distractor_served_count"] > 0, (
        f"[{driver}] unflagged control must be non-vacuous (chatter present "
        f"on the served pack): {unflagged}"
    )

    assert signalled["knowledge_served"], (
        f"[{driver}] admission delta not reached -- a directive-signalled "
        f"knowledge record must be admitted onto the served pack where the "
        f"identical unflagged construction excludes it entirely: {signalled}"
    )
    assert signalled["distractor_served_count"] > 0, (
        f"[{driver}] register-matched chatter must remain present alongside "
        f"the admitted knowledge record -- both-served (not "
        f"knowledge-displacing-chatter) is the reserve lane's own designed "
        f"shape: {signalled}"
    )
