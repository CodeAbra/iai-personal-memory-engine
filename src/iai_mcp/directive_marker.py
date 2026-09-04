"""Pure-lexical typed-marker recognizer for capture-time standing orders.

Distinct from directive_classify.py's fuzzy phrasing signal: this module
recognizes ONE deterministic anchored prefix, "standing directive:", as the
sole evidence a human explicitly typed a standing order. Pure string
processing, no I/O, no import of capture/store/embed modules.

Every pattern anchored at the start (after leading whitespace), IGNORECASE,
no nested quantifiers (house ReDoS rule, mirrors directive_classify.py).
"""
from __future__ import annotations

import re

_MARKER_PREFIX = re.compile(r"^\s*standing directive\s*:", re.IGNORECASE)


def is_directive_marker(text: "str | None") -> bool:
    """True when text begins (after leading whitespace) with the anchored
    "standing directive:" prefix, case-insensitive.

    Fail-safe by construction: never raises. "" / non-str / whitespace-only
    -> False. A mid-sentence occurrence (not an anchored prefix) -> False.
    The recognizer decides the flag only -- it never rewrites text.
    """
    if not isinstance(text, str):
        return False
    if not text.strip():
        return False
    return _MARKER_PREFIX.match(text) is not None
