"""Shared caps and conservative token projector for the directive tier.

Stdlib-only leaf: the capture-time gate and the render-time sizing both
import this module directly, so their caps can never drift apart.
"""

from __future__ import annotations

DIRECTIVE_MAX_COUNT = 10
DIRECTIVE_BUDGET_TOKENS = 500
DIRECTIVE_LINE_CHAR_CAP = 200

CONT_BUDGET_TOKENS = 500
"""Joint DIR+CONT budget reservation for the session-continuity block."""

AGENT_REGISTRY_BUDGET_TOKENS = 300
"""Own slice for the pending-agent registry block -- separate from
CONT_BUDGET_TOKENS, the fourth term in the joint session-start budget."""

AGENT_REGISTRY_LINE_CHAR_CAP = 100
AGENT_REGISTRY_MAX_RENDERED = 8

_BLOCK_OVERHEAD_TOKENS = 8


def _approx_tokens(text: str) -> int:
    # Must stay identical to session._approx_tokens -- this leaf module
    # cannot import iai_mcp.session, so the formula is duplicated, not
    # shared by reference.
    if not text:
        return 0
    return max(1, len(text) // 4)


def projected_directive_tokens(texts: list[str]) -> int:
    if not texts:
        return 0
    total = _BLOCK_OVERHEAD_TOKENS
    for text in texts:
        total += _approx_tokens(text[:DIRECTIVE_LINE_CHAR_CAP])
    return total
