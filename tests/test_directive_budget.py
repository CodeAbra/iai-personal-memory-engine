"""Shared directive budget module: constants and conservative token projector."""

from __future__ import annotations

from iai_mcp.directive_budget import (
    DIRECTIVE_BUDGET_TOKENS,
    DIRECTIVE_LINE_CHAR_CAP,
    DIRECTIVE_MAX_COUNT,
    _approx_tokens,
    projected_directive_tokens,
)


def test_budget_constants_are_defined():
    assert DIRECTIVE_MAX_COUNT == 10
    assert DIRECTIVE_BUDGET_TOKENS == 500
    assert isinstance(DIRECTIVE_LINE_CHAR_CAP, int)
    assert DIRECTIVE_LINE_CHAR_CAP > 0


def test_projector_empty_input_is_zero():
    assert projected_directive_tokens([]) == 0


def test_projector_is_monotonic_when_a_directive_is_added():
    base = projected_directive_tokens(["reply in English"])
    extended = projected_directive_tokens(["reply in English", "never use emojis"])
    assert extended >= base


def test_projector_caps_each_line_before_estimating():
    huge = "y" * 10_000
    projected = projected_directive_tokens([huge])
    uncapped_estimate = len(huge) // 4
    assert projected < uncapped_estimate


def test_projector_is_conservative_against_further_render_time_shortening():
    text = "x" * (DIRECTIVE_LINE_CHAR_CAP + 50)
    projected = projected_directive_tokens([text])

    capped = text[:DIRECTIVE_LINE_CHAR_CAP]
    # A render step that additionally marker-strips/collapses only shortens
    # the capped text -- simulate that with a strictly shorter substring.
    shortened = capped[: len(capped) // 2]
    assert projected >= _approx_tokens(shortened)
