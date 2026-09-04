"""Pure lexical epistemic-status classification for capture-time text.

Precision-over-recall on the ``fact`` label: a wrong ``fact`` tag misleads
every future recall of the record, so ``fact`` patterns stay deliberately
narrow and are checked LAST, after hedge/speculation/opinion signals. Pure
string processing, no I/O, no import of capture/store/embed modules.
"""
from __future__ import annotations

import re

from iai_mcp.types import EPISTEMIC_STATUS_ENUM

EPISTEMIC_TEXT_SCAN_CAP = 2000

# Every pattern anchored with \b, IGNORECASE, no nested quantifiers
# (house ReDoS rule, mirrors entity_anchors.py / capture.py's
# _CORRECTION_PATTERNS shape).

_ESTIMATE_PATTERNS: "list[tuple[str, re.Pattern]]" = [
    ("approximately", re.compile(r"\bapproximately\b", re.IGNORECASE)),
    ("roughly", re.compile(r"\broughly\b", re.IGNORECASE)),
    ("about", re.compile(r"\babout\b", re.IGNORECASE)),
    ("around", re.compile(r"\baround\b", re.IGNORECASE)),
    ("tilde-numeric", re.compile(r"~\s?\d")),
    ("probably", re.compile(r"\bprobably\b", re.IGNORECASE)),
    ("i-think", re.compile(r"\bi think\b", re.IGNORECASE)),
    ("seems", re.compile(r"\bseems?\b", re.IGNORECASE)),
    ("i-guess", re.compile(r"\bi(?:'d)? guess\b", re.IGNORECASE)),
    ("more-or-less", re.compile(r"\bmore or less\b", re.IGNORECASE)),
]

_HYPOTHESIS_PATTERNS: "list[tuple[str, re.Pattern]]" = [
    ("maybe", re.compile(r"\bmaybe\b", re.IGNORECASE)),
    ("might-be", re.compile(r"\bmight be\b", re.IGNORECASE)),
    ("could-be-that", re.compile(r"\bcould be that\b", re.IGNORECASE)),
    ("i-suspect", re.compile(r"\bi suspect\b", re.IGNORECASE)),
    ("what-if", re.compile(r"\bwhat if\b", re.IGNORECASE)),
    ("could-be", re.compile(r"\bcould be\b", re.IGNORECASE)),
    ("possibly", re.compile(r"\bpossibly\b", re.IGNORECASE)),
    ("perhaps", re.compile(r"\bperhaps\b", re.IGNORECASE)),
]

_OPINION_PATTERNS: "list[tuple[str, re.Pattern]]" = [
    ("i-prefer", re.compile(r"\bi prefer\b", re.IGNORECASE)),
    ("i-like", re.compile(r"\bi like\b", re.IGNORECASE)),
    ("better", re.compile(r"\bbetter\b", re.IGNORECASE)),
    ("worse", re.compile(r"\bworse\b", re.IGNORECASE)),
    ("id-rather", re.compile(r"\bi'?d rather\b", re.IGNORECASE)),
    ("i-love", re.compile(r"\bi love\b", re.IGNORECASE)),
    ("my-opinion", re.compile(r"\bin my opinion\b", re.IGNORECASE)),
]

# fact requires an explicit assertion/measurement marker word -- no bare
# copula or numeric-only trigger (kills fact precision).
_FACT_PATTERNS: "list[tuple[str, re.Pattern]]" = [
    ("confirmed", re.compile(r"\bconfirmed\b", re.IGNORECASE)),
    ("measured", re.compile(r"\bmeasured\b", re.IGNORECASE)),
    ("verified", re.compile(r"\bverified\b", re.IGNORECASE)),
    ("definitely", re.compile(r"\bdefinitely\b", re.IGNORECASE)),
    ("turns-out", re.compile(r"\bturns out\b", re.IGNORECASE)),
    ("in-fact", re.compile(r"\bin fact\b", re.IGNORECASE)),
    ("the-fact-is", re.compile(r"\bthe fact is\b", re.IGNORECASE)),
]

# Priority order protects fact precision: a hedge/speculation/opinion
# signal always wins over a bare assertion when both are present in the
# same text ("I think it is confirmed" is a hedge, not a fact).
_GROUPS: "list[tuple[str, list[tuple[str, re.Pattern]]]]" = [
    ("estimate", _ESTIMATE_PATTERNS),
    ("hypothesis", _HYPOTHESIS_PATTERNS),
    ("opinion", _OPINION_PATTERNS),
    ("fact", _FACT_PATTERNS),
]


def classify_epistemic_status(text: str) -> str:
    """Classify capture text into one of EPISTEMIC_STATUS_ENUM.

    Fail-safe by construction: never raises, always returns an enum
    member. "" / None / non-str / whitespace-only -> "unknown".
    """
    if not isinstance(text, str):
        return "unknown"
    scanned = text[:EPISTEMIC_TEXT_SCAN_CAP]
    if not scanned.strip():
        return "unknown"
    for label, patterns in _GROUPS:
        for _, pattern in patterns:
            if pattern.search(scanned):
                if label not in EPISTEMIC_STATUS_ENUM:
                    return "unknown"
                return label
    return "unknown"
