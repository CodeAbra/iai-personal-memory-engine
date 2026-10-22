"""Shared helper for hygiene-guard ratchets.

Both `test_no_dev_process_markers.py` and `test_no_foreign_tool_refs.py`
scan the tracked tree for phrases they forbid. A small, explicitly named
set of OTHER files are themselves detectors for a different concern and
legitimately need their own `re.compile(...)` definition, plus its
banner comment, exempted from the scan so a detector's own pattern
literals never count as a hit. Kept in one place so the two ratchets
cannot drift apart.
"""
from __future__ import annotations

import ast


def own_detector_ranges(text: str) -> set[int]:
    """Lines that define a `re.compile(...)` detector, plus its banner.

    Recognized structurally by AST shape (any `re.compile(...)`
    assignment, plus a contiguous comment banner directly above it,
    tolerating one blank-line gap). Callers MUST restrict which files
    this is applied to -- applying it tree-wide would silently exempt
    any unrelated regex whose banner happens to match the shape.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    lines = text.splitlines()
    ranges: set[int] = set()
    for node in ast.walk(tree):
        value = getattr(node, "value", None)
        if not isinstance(value, ast.Call):
            continue
        func = value.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "compile"
            and isinstance(func.value, ast.Name)
            and func.value.id == "re"
        ):
            continue
        start, end = node.lineno, getattr(node, "end_lineno", node.lineno)
        cursor = start - 1
        if cursor >= 1 and not lines[cursor - 1].strip():
            cursor -= 1
        while cursor >= 1 and lines[cursor - 1].lstrip().startswith("#"):
            cursor -= 1
        ranges.update(range(cursor + 1, end + 1))
    return ranges
