"""Guards the fused-score candidate-ranking term set against silent drift.

The candidate-scoring loop and its post-loop score-replacement step
accumulate a fixed set of additive/multiplicative terms. This module pins
that set by stable content anchors (function/constant names, gate
conditions, and the final sort-key shape) rather than by line number, so a
future edit that drops a term, renames its symbol, or removes a gate is
caught here instead of silently.
"""

from __future__ import annotations

import inspect

from iai_mcp import pipeline

TERM_ANCHORS: frozenset[str] = frozenset(
    {
        "W_COSINE",
        "_aaak_overlap",
        "effective_w_degree",
        "_age_penalty",
        "spread_contrib",
        "community_contrib",
        "structural_similarity",
        "profile_modulation_for_record",
        'getattr(rec, "stability"',
        'getattr(rec, "valence"',
        "_trigram_jaccard",
        "fts_hits",
        "_lex_fusion_w",
        "_tier_knowledge_boost",
        "SALIENCE_LEVEL_RANK",
        "parse_date_mentions",
        "corrector_base_score",
    },
)

GATE_CONDITIONS: tuple[str, ...] = (
    'mode != "verbatim"',
    'cue_intent != "historical_verbatim"',
)

TIE_BREAK_SHAPE = "(-x[0], str(x[1]))"


class TermContractViolation(AssertionError):
    """Raised when source text fails to carry the full frozen term contract."""


def scan_term_anchors(source_text: str) -> set[str]:
    """Return the subset of TERM_ANCHORS present verbatim in source_text."""
    return {anchor for anchor in TERM_ANCHORS if anchor in source_text}


def assert_term_contract(source_text: str) -> None:
    found = scan_term_anchors(source_text)
    missing = TERM_ANCHORS - found
    if missing:
        raise TermContractViolation(f"missing term anchors: {sorted(missing)}")
    for gate in GATE_CONDITIONS:
        if gate not in source_text:
            raise TermContractViolation(f"missing gate condition: {gate!r}")
    if TIE_BREAK_SHAPE not in source_text:
        raise TermContractViolation("missing tie-break shape")


def _scoring_source() -> str:
    return inspect.getsource(pipeline._recall_core)


def test_current_source_carries_the_full_frozen_term_contract():
    # Non-vacuous: assert the EXPLICIT frozen anchor set, never a count
    # derived from whatever the scanner happened to find.
    source = _scoring_source()
    found = scan_term_anchors(source)
    assert found == TERM_ANCHORS
    assert_term_contract(source)


def test_mutated_source_with_one_renamed_anchor_is_detected_as_drift():
    source = _scoring_source()
    mutated = source.replace("W_COSINE", "SIM_WEIGHT_PRIMARY")
    found = scan_term_anchors(mutated)
    assert "W_COSINE" not in found
    assert found == TERM_ANCHORS - {"W_COSINE"}
    try:
        assert_term_contract(mutated)
    except TermContractViolation as exc:
        assert "W_COSINE" in str(exc)
    else:
        raise AssertionError("expected TermContractViolation on a renamed anchor")
