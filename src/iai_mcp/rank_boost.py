"""Pure knowledge-tier and salience-level rank-boost arithmetic.

Shared by the full recall scorer (pipeline.py's `_recall_core`) and the
push-side foresight ranker (foresight.py's `refresh_pack`) so both paths let
informativeness -- not raw cosine alone -- decide which candidate wins a
slot. No store, graph, profile, or daemon dependency: callers own the
tier/tags/salience-level inputs already resolved on their own record view,
this module only computes the multiplier.
"""
from __future__ import annotations

from iai_mcp.types import SALIENCE_LEVEL_RANK

TIER_KNOWLEDGE_BOOST_DEFAULT = 1.05
"""Bounded soft multiplier for knowledge-grade sources at final rank -- a
nudge past equal-scored raw turns, never a filter; 1.0 disables. doc:*
chunks boost regardless of literal_preservation (they ARE literal curated
content); semantic summaries boost only when literal_preservation is not
"strong" -- that knob is precisely the raw-vs-summary preference and the
strong default must keep outranking condensations."""

SALIENCE_BOOST_STEP_DEFAULT = 0.05
"""Bounded additive-per-level rank step for a caller-declared salience
level -- never a filter. Applies to a flagged record regardless of tier,
because the flag is a general-purpose signal, not a knowledge-source
marker."""

SEMANTIC_VALUE_LIFT_DEFAULT = 0.5
"""Bounded additive lift for a semantic-tier record that also carries a
real salience mark -- distinct from and stacks with the
literal_preservation-gated multiplicative term below; never fires for an
unflagged record, so it cannot perturb a salience-free fixture."""


def has_doc_tag(tags: "object") -> bool:
    return any(isinstance(t, str) and t.startswith("doc:") for t in (tags or ()))


def tier_multiplier(
    *,
    tier: "str | None",
    tags: "object",
    tier_boost: float,
    literal_preservation_strong: bool = True,
    salience_level: "str | None" = None,
    semantic_lift: float = SEMANTIC_VALUE_LIFT_DEFAULT,
) -> float:
    """Multiplier for a doc-tagged or semantic-tier record; 1.0 = no boost.

    The salience-gated additive term stacks with (never replaces) the
    literal_preservation-gated multiplicative term above -- a record that
    hits both branches gets tier_boost + semantic_lift, not max() of the
    two, so the two signals compound by design."""
    result = 1.0
    if has_doc_tag(tags) or (tier == "semantic" and not literal_preservation_strong):
        result = tier_boost
    if semantic_lift and tier == "semantic" and salience_level not in (None, "unflagged"):
        result += semantic_lift
    return result


def salience_multiplier(*, salience_level: "str | None", salience_step: float) -> float:
    """Multiplier for a caller-declared salience level; 1.0 = unflagged."""
    rank = SALIENCE_LEVEL_RANK.get(salience_level or "unflagged", 0)
    if rank <= 0 or salience_step == 0.0:
        return 1.0
    return 1.0 + rank * salience_step


def boosted_score(
    cos: float,
    *,
    tier: "str | None",
    tags: "object",
    salience_level: "str | None",
    tier_boost: float = TIER_KNOWLEDGE_BOOST_DEFAULT,
    salience_step: float = SALIENCE_BOOST_STEP_DEFAULT,
    literal_preservation_strong: bool = True,
    semantic_lift: float = SEMANTIC_VALUE_LIFT_DEFAULT,
) -> float:
    """`cos` scaled by the knowledge-tier and salience multipliers -- the
    combined informativeness vote both pipeline.py and foresight.py apply
    at final rank. Callers that only need ranking order (not a displayed
    score) may pass this as a sort key; it is never a substitute for the
    raw cosine used at the confidence floor or in a rendered hint."""
    score = cos
    score *= tier_multiplier(
        tier=tier, tags=tags, tier_boost=tier_boost,
        literal_preservation_strong=literal_preservation_strong,
        salience_level=salience_level, semantic_lift=semantic_lift,
    )
    score *= salience_multiplier(salience_level=salience_level, salience_step=salience_step)
    return score
