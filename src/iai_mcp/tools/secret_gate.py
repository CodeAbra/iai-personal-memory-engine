"""Secret-shaped-content gate for the command-output store boundary.

Stdlib-only: imports nothing from iai_mcp. `gate_paths_for_command_output`
is the SINGLE gate implementation -- both the store-boundary branch in
`iai_mcp.tools.index_paths`/`index_all` and the `index_command_output`
convenience constructor call it, so the invariant cannot drift into two
divergent checks.
"""

from __future__ import annotations

import re
from pathlib import Path

# Two independent layers, either one refusing is enough (belt AND
# suspenders): a common-English trigger-word substring match, plus a
# value-SHAPE match against real credential formats. The shape layer
# exists because the trigger words alone miss the highest-value leaks --
# a PEM/OpenSSH private-key header reads "PRIVATE KEY" with a SPACE, never
# the `private_key` underscore variant, and provider tokens/JWTs carry no
# trigger word at all.
_TRIGGER_PATTERN = re.compile(
    r"(?i:token|secret|password|api_key|private[ _]key|bearer)"
    r"|AKIA[0-9A-Z]{16}"
    r"|ghp_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|sntryu_[A-Za-z0-9]{20,}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|sk_live_[A-Za-z0-9]{16,}"
    r"|sk_test_[A-Za-z0-9]{16,}"
    r"|AIza[0-9A-Za-z_\-]{35}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|eyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"
)


def contains_secret_shaped_content(text: str) -> bool:
    """True when `text` matches the credential trigger alphabet."""
    return _TRIGGER_PATTERN.search(text) is not None


# The fixed-prefix value-shape alternatives from `_TRIGGER_PATTERN`, each
# rewritten with a no-minimum body quantifier and anchored to the end of
# the search window: an external byte cap can cut a real credential short
# of the body length `_TRIGGER_PATTERN` itself requires, so a caller that
# bounds output needs to catch a fragment `_TRIGGER_PATTERN` would miss.
# The plain-English trigger words (token/secret/password/...) are excluded
# on purpose -- a cut prefix of an ordinary word is not distinguishable
# from unrelated text. Kept as ONE alphabet with `_TRIGGER_PATTERN` in this
# module so neither can drift from the other.
_TAIL_FRAGMENT_PATTERN = re.compile(
    r"AKIA[0-9A-Z]{0,16}\Z"
    r"|ghp_[A-Za-z0-9]*\Z"
    r"|github_pat_[A-Za-z0-9_]*\Z"
    r"|sntryu_[A-Za-z0-9]*\Z"
    r"|xox[baprs]-[A-Za-z0-9-]*\Z"
    r"|sk_live_[A-Za-z0-9]*\Z"
    r"|sk_test_[A-Za-z0-9]*\Z"
    r"|AIza[0-9A-Za-z_\-]{0,35}\Z"
    r"|-----BEGIN [A-Z ]*\Z"
    r"|eyJ[A-Za-z0-9_\-]*(?:\.[A-Za-z0-9_\-]*){0,2}\Z"
)

# How far back from the end of `text` a fragment start may be found. Covers
# the longest fixed prefix plus a full body for every shape above; a
# complete secret longer than this already matches `_TRIGGER_PATTERN` in
# full and never needs this path. Bounding the search keeps a coincidental
# mid-text prefix from wiping everything after it.
_TAIL_FRAGMENT_WINDOW = 256


def secret_shaped_tail_start(text: str) -> "int | None":
    """Index in `text` where a possibly-truncated secret-shaped fragment
    begins and runs to the end of `text`, or None if the tail holds
    nothing shaped like a cut credential."""
    if not text:
        return None
    window_start = max(0, len(text) - _TAIL_FRAGMENT_WINDOW)
    match = _TAIL_FRAGMENT_PATTERN.search(text, window_start)
    return match.start() if match else None


def file_has_secret_shaped_content(path: "str | Path") -> bool:
    """Reads `path` as UTF-8 and checks it for secret-shaped content.

    An unreadable or non-UTF-8 file cannot be verified -- fail-closed
    (refuse) rather than silently admit it to the searchable store.
    """
    try:
        text = Path(path).read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return True
    return contains_secret_shaped_content(text)


def gate_paths_for_command_output(paths: "list[str]") -> "tuple[list[str], list[str]]":
    """Splits `paths` into (passing, refused) by secret-shaped content."""
    passing: "list[str]" = []
    refused: "list[str]" = []
    for p in paths:
        if file_has_secret_shaped_content(p):
            refused.append(p)
        else:
            passing.append(p)
    return passing, refused
