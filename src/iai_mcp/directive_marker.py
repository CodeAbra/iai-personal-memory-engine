"""Pure-lexical typed-marker recognizers for capture-time standing orders.

Distinct from directive_classify.py's fuzzy phrasing signal: this module
recognizes two deterministic anchored prefixes -- "standing directive:" as
the sole evidence a human explicitly typed a standing order, and
"remove directive:" as the sole evidence a human wants one retired. Pure
string processing, no I/O, no import of capture/store/embed modules.

Every pattern anchored at the start (after leading whitespace), IGNORECASE,
no nested quantifiers (house ReDoS rule, mirrors directive_classify.py).
"""
from __future__ import annotations

import re

_MARKER_PREFIX = re.compile(r"^\s*standing directive\s*:", re.IGNORECASE)
_REMOVE_MARKER_PREFIX = re.compile(r"^\s*remove directive\s*:\s*(\S+)", re.IGNORECASE)


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


def parse_directive_remove_marker(text: "str | None") -> "str | None":
    """Returns the id token when text begins (after leading whitespace)
    with the anchored "remove directive:" prefix, case-insensitive, else
    None.

    Fail-safe by construction: never raises. "" / non-str / whitespace-only
    -> None. A mid-sentence occurrence (not an anchored prefix) -> None.
    The recognizer decides which id to resolve -- it never mutates state.
    """
    if not isinstance(text, str):
        return None
    if not text.strip():
        return None
    match = _REMOVE_MARKER_PREFIX.match(text)
    if match is None:
        return None
    return match.group(1)
