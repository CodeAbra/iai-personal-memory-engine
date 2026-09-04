"""Unit coverage for the pure lexical epistemic-status classifier."""

from __future__ import annotations

import pytest

from iai_mcp.epistemic_classify import (
    EPISTEMIC_TEXT_SCAN_CAP,
    classify_epistemic_status,
)
from iai_mcp.types import EPISTEMIC_STATUS_ENUM


@pytest.mark.parametrize(
    "text",
    [
        "the response latency was roughly 400ms on that run",
        "I think the release ships next week",
        "about ten people joined the call",
        "the fix should land around Friday",
        "it seems the cache is cold on first boot",
    ],
)
def test_hedge_phrasing_classified_as_estimate(text):
    assert classify_epistemic_status(text) == "estimate"


@pytest.mark.parametrize(
    "text",
    [
        "maybe the regression is in the embedder",
        "the bug might be a race condition",
        "I suspect the index is stale",
        "what if the cache never warms",
        "the fix could be a lock reorder",
    ],
)
def test_speculative_phrasing_classified_as_hypothesis(text):
    assert classify_epistemic_status(text) == "hypothesis"


@pytest.mark.parametrize(
    "text",
    [
        "I prefer the shorter migration path",
        "I like the new dashboard layout better",
        "this design is better than the old one",
        "I'd rather ship the smaller change first",
        "the old renderer was worse for large batches",
    ],
)
def test_preference_phrasing_classified_as_opinion(text):
    assert classify_epistemic_status(text) == "opinion"


@pytest.mark.parametrize(
    "text",
    [
        "the team confirmed the outage started at noon",
        "we measured the p95 latency at 230ms",
        "the fix was verified against the full suite",
        "the build definitely failed on the arm runner",
        "it turns out the config file was never read",
    ],
)
def test_asserted_measured_phrasing_classified_as_fact(text):
    assert classify_epistemic_status(text) == "fact"


@pytest.mark.parametrize(
    "text",
    [
        "the meeting is scheduled for next Monday",
        "the document lists several unrelated topics",
        "she mentioned the roadmap during the call",
        "the folder contains a handful of scripts",
    ],
)
def test_ambiguous_text_stays_unknown(text):
    assert classify_epistemic_status(text) == "unknown"


@pytest.mark.parametrize("text", ["", "   ", "\n\t  ", None, 123])
def test_empty_or_whitespace_input_stays_unknown(text):
    assert classify_epistemic_status(text) == "unknown"


def test_scan_cap_boundary_ignores_signal_past_the_cap():
    padding = "x " * (EPISTEMIC_TEXT_SCAN_CAP // 2 + 50)
    text = padding + "I confirmed the value is measured at 400ms"
    assert len(text) > EPISTEMIC_TEXT_SCAN_CAP
    assert classify_epistemic_status(text) == "unknown"


def test_signal_before_the_cap_boundary_is_still_detected():
    text = "confirmed: " + ("x " * 10) + "this is the finding"
    assert len(text) < EPISTEMIC_TEXT_SCAN_CAP
    assert classify_epistemic_status(text) == "fact"


def test_hedge_wins_over_bare_assertion_in_priority_order():
    text = "I think it is confirmed to be around 400ms"
    assert classify_epistemic_status(text) == "estimate"


@pytest.mark.parametrize(
    "text",
    [
        "roughly 400ms",
        "maybe a race condition",
        "I like it better",
        "we confirmed 230ms",
        "no signal here at all",
    ],
)
def test_return_value_always_in_enum(text):
    assert classify_epistemic_status(text) in EPISTEMIC_STATUS_ENUM
