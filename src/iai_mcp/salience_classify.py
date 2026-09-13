"""Pure weighted-OR salience composite over already-computed capture signals.

No I/O, no text scanning, no import of capture/store/embed modules. Mirrors
epistemic_classify.py's shape: module-level constants, keyword-only entry
point, fail-safe by construction.
"""
from __future__ import annotations

from iai_mcp.types import SALIENCE_LEVEL_ENUM

# Weights are plain constants, not a tunable knob. A lone non-empty
# entity_tags signal (0.5) must stay BELOW _NOTABLE_THRESHOLD (1.0) on its
# own -- raising this weight or lowering the threshold makes entity_tags
# independently sufficient and flips the pinned coercion-to-unflagged
# regression. Ceiling is "notable"; this composite never emits "critical".
# With only these three signals, _WEIGHT_ENTITY_TAGS never independently
# changes an outcome today -- it exists for a future signal below threshold.
_WEIGHT_DIRECTIVE = 1.0
_WEIGHT_FACT = 1.0
_WEIGHT_ENTITY_TAGS = 0.5
_NOTABLE_THRESHOLD = 1.0


def classify_salience(
    *, directive: bool, entity_tags: "list[str]", epistemic_status: str,
) -> str:
    """Score three free signals and return a SALIENCE_LEVEL_ENUM member.

    Weighted-OR: directive or epistemic_status == "fact" alone reaches the
    threshold; non-empty entity_tags alone does not. epistemic_status ==
    "unknown" contributes nothing (never a penalty). Never returns
    "critical" -- that value is reserved for explicit caller intent.
    """
    score = 0.0
    if directive:
        score += _WEIGHT_DIRECTIVE
    if epistemic_status == "fact":
        score += _WEIGHT_FACT
    if entity_tags:
        score += _WEIGHT_ENTITY_TAGS
    level = "notable" if score >= _NOTABLE_THRESHOLD else "unflagged"
    return level if level in SALIENCE_LEVEL_ENUM else "unflagged"
