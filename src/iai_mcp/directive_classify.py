"""Pure lexical standing-order suggestion signal for capture-time text.

Non-storing: this module never sets the stored directive flag. It is a
soft signal only -- callers may use its output for surfacing or review, but
no capture path may treat a True return as authorization to write a
standing order. The only paths permitted to mint a directive are an
explicit caller-supplied value or the anchored typed-marker recognizer
(directive_marker.py), gated on a fail-closed opt-in.

Precision-over-recall: patterns stay narrow and the scoped always/never form
requires an adjacent bare-infinitive policy verb rather than firing on any
bare "always"/"never". Pure string processing, no I/O, no import of
capture/store/embed modules.
"""
from __future__ import annotations

import re

DIRECTIVE_TEXT_SCAN_CAP = 2000

# Every pattern anchored with \b, IGNORECASE, no nested quantifiers
# (house ReDoS rule, mirrors epistemic_classify.py).

_FROM_NOW_ON = re.compile(r"\bfrom now on\b", re.IGNORECASE)
_ACROSS_ALL_SESSIONS = re.compile(r"\bacross all sessions\b", re.IGNORECASE)
_SAVE_FOR_ALL_SESSIONS = re.compile(r"\bsave this for all sessions\b", re.IGNORECASE)

# Bare-infinitive command verbs. The exact-token match (word-boundary on
# both sides) is what separates an imperative ("never use markdown") from a
# conjugated descriptive form ("she never uses the staging branch") -- the
# latter fails the trailing \b because "uses" continues past "use" with no
# word-boundary in between.
_DIRECTIVE_VERBS = (
    "reply|answer|respond|use|call|address|treat|assume|include|write|speak|say"
    "|remember|capture|ask|check|confirm|avoid|follow|keep|start|stay|run|apply"
    "|prefer|choose|route|store|save|skip|delete|share|translate|format|ignore"
    "|trust|verify|validate|retry|notify|mention|hide|encrypt|decrypt|backup"
    "|escalate|forget|retain|discard|set|make|default|refer|log|block|allow"
    "|deny|grant|revoke|disclose|reveal|redact|expose|wait|merge"
)
_ALWAYS_NEVER_POLICY = re.compile(
    rf"\b(?:always|never)\s+(?:{_DIRECTIVE_VERBS})\b", re.IGNORECASE
)

_PATTERNS: "list[re.Pattern]" = [
    _FROM_NOW_ON,
    _ACROSS_ALL_SESSIONS,
    _SAVE_FOR_ALL_SESSIONS,
    _ALWAYS_NEVER_POLICY,
]


def classify_is_directive(text: str) -> bool:
    """True when capture text carries a standing-order signal phrasing.

    Fail-safe by construction: never raises. "" / non-str / whitespace-only
    -> False.
    """
    if not isinstance(text, str):
        return False
    scanned = text[:DIRECTIVE_TEXT_SCAN_CAP]
    if not scanned.strip():
        return False
    return any(pattern.search(scanned) for pattern in _PATTERNS)
