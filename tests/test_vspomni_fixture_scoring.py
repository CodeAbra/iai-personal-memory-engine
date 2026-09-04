"""Gate-wired unit tests for the vspomni recall-reliability scoring logic
(bench/vspomni_fixture_eval.py).

Pure-logic tests only -- no store, no daemon, no embedder. The real
store-touching BEFORE run is a bench invocation recorded in the phase
SUMMARY, never a pytest verify (a daemon boot must never sit in a pytest
gate).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_PATH = str(Path(__file__).resolve().parent.parent)
if _REPO_PATH not in sys.path:
    sys.path.insert(0, _REPO_PATH)

from bench.vspomni_fixture_eval import (  # noqa: E402
    attribute_breakout,
    sample_path,
    score_case,
    strip_trigger_phrase,
)

_ANSWER = "code review runs on sonnet"
_CASE = {
    "id": "arbitration_case",
    "current_truth_answer": _ANSWER,
    "competing_stale_assertion": "pinned note: code review runs on opus (dated 2026-01-01)",
}


def test_dated_row_with_answer_passes():
    """current-truth-present-and-dated row PASSES: the answer is present
    in one hit AND a superseded sibling hit carries a valid_to date."""
    hits = [
        {"record_id": "r1", "literal_surface": _ANSWER, "valid_to": None},
        {
            "record_id": "r2",
            "literal_surface": "old routing note, code review runs on opus",
            "valid_to": "2026-01-01T00:00:00Z",
        },
    ]
    result = score_case(hits, _CASE)
    assert result["content_present"] is True
    assert result["staleness_present"] is True
    assert result["passed"] is True


def test_undated_row_against_competing_assertion_fails():
    """The arbitration case: the current-truth record IS present, but
    carries no staleness/supersession signal at all -- FAILS. Presence of
    the current-truth record alone must not be a pass."""
    hits = [{"record_id": "r1", "literal_surface": _ANSWER, "valid_to": None}]
    result = score_case(hits, _CASE)
    assert result["content_present"] is True
    assert result["staleness_present"] is False
    assert result["passed"] is False


def test_truncated_mid_table_row_fails():
    """A row whose answer content sits past the [:400] render boundary
    fails on content, even though a staleness signal is present."""
    padding = "x" * 450
    hits = [
        {
            "record_id": "r1",
            "literal_surface": f"{padding} {_ANSWER}",
            "valid_to": "2026-01-01T00:00:00Z",
        }
    ]
    result = score_case(hits, _CASE)
    assert result["content_present"] is False
    assert result["passed"] is False


def test_supersession_line_in_text_also_counts_as_staleness():
    """A ⚠ superseded / 'supersedes prior version dated ...' line embedded
    directly in the rendered text (the offline pack's own convention)
    counts as a staleness signal, independent of the valid_to field."""
    hits = [
        {
            "record_id": "r1",
            "literal_surface": f"{_ANSWER} — supersedes prior version dated 2026-08-15",
            "valid_to": None,
        }
    ]
    result = score_case(hits, _CASE)
    assert result["content_present"] is True
    assert result["staleness_present"] is True
    assert result["passed"] is True


def test_no_hits_never_passes():
    result = score_case([], _CASE)
    assert result["passed"] is False
    assert result["rendered_block"] == ""


def test_more_than_max_hits_ignored():
    """Only the first 3 hits are ever scored -- a staleness signal sitting
    in hit #4 cannot rescue the case."""
    hits = [
        {"record_id": "r1", "literal_surface": _ANSWER, "valid_to": None},
        {"record_id": "r2", "literal_surface": "unrelated", "valid_to": None},
        {"record_id": "r3", "literal_surface": "also unrelated", "valid_to": None},
        {"record_id": "r4", "literal_surface": "stale sibling", "valid_to": "2026-01-01T00:00:00Z"},
    ]
    result = score_case(hits, _CASE)
    assert result["staleness_present"] is False
    assert result["passed"] is False


def test_sample_fixture_loads_and_scores():
    """The committed synthetic sample loads and its arbitration case
    correctly FAILS at baseline (record present, no staleness signal)."""
    raw = json.loads(sample_path().read_text(encoding="utf-8"))
    assert raw["cases"], "sample fixture must carry at least one case"
    case = raw["cases"][0]
    hits = [{"record_id": "sample-1", "literal_surface": case["current_truth_answer"], "valid_to": None}]
    result = score_case(hits, case)
    assert result["content_present"] is True
    assert result["staleness_present"] is False
    assert result["passed"] is False


# ---------------------------------------------------------------------------
# Breakout attribution.
# ---------------------------------------------------------------------------


def test_breakout_attributes_lexical_only():
    result = attribute_breakout({"r1"}, lexical_ids={"r1"}, embedding_ids={"r9"})
    assert result["lexical_hit"] is True
    assert result["embedding_hit"] is False
    assert result["channel"] == "lexical"


def test_breakout_attributes_embedding_only():
    result = attribute_breakout({"r1"}, lexical_ids={"r9"}, embedding_ids={"r1"})
    assert result["lexical_hit"] is False
    assert result["embedding_hit"] is True
    assert result["channel"] == "embedding"


def test_breakout_attributes_both():
    result = attribute_breakout({"r1"}, lexical_ids={"r1"}, embedding_ids={"r1"})
    assert result["channel"] == "both"


def test_breakout_attributes_neither():
    result = attribute_breakout({"r1"}, lexical_ids={"r9"}, embedding_ids={"r8"})
    assert result["channel"] == "neither"
    assert result["lexical_hit"] is False
    assert result["embedding_hit"] is False


def test_strip_trigger_phrase_removes_ru_trigger():
    stripped = strip_trigger_phrase("вспомни правильный роутинг моделей")
    assert "вспомни" not in stripped.lower()
    assert "роутинг" in stripped

    stripped2 = strip_trigger_phrase("напомни, как мы пушаем")
    assert "напомни" not in stripped2.lower()
    assert "пушаем" in stripped2
