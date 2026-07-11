from __future__ import annotations

import re
from pathlib import Path


TOOLS_TS = Path(__file__).resolve().parent.parent / "mcp-wrapper" / "src" / "tools.ts"

MARKERS = frozenset({
    "use when",
    "use for",
    "returns",
    "read-only",
    "mutates",
    "idempotent",
    "prefer",
    "detect",
})

CAM_KEYWORDS = frozenset({
    "formality",
    "register",
    "behavioral",
    "pattern",
    "anomaly",
})


def _tok(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return max(1, len(text) // 4) if text else 0


def _extract_top_level_descriptions() -> list[tuple[str, str]]:
    text = TOOLS_TS.read_text()
    name_re = re.compile(r'name:\s*"([^"]+)"', re.MULTILINE)
    out: list[tuple[str, str]] = []
    positions = [(m.group(1), m.end()) for m in name_re.finditer(text)]
    for tool_name, pos in positions:
        region = text[pos:]
        next_name = name_re.search(region)
        end = next_name.start() if next_name else len(region)
        region = region[:end]
        concat_re = re.compile(
            r'description:\s*('
            r'"(?:[^"\\]|\\.)*"'
            r'(?:\s*\+\s*"(?:[^"\\]|\\.)*")*'
            r')',
            re.MULTILINE,
        )
        m = concat_re.search(region)
        if not m:
            continue
        literal = m.group(1)
        parts = re.findall(r'"((?:[^"\\]|\\.)*)"', literal)
        desc = "".join(parts)
        desc = desc.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")
        out.append((tool_name, desc))
    return out


_TOOL_NAME_LINE = re.compile(
    r"^  (?P<name>[a-zA-Z_][a-zA-Z0-9_]*):\s*\{",
    re.MULTILINE,
)


def _balance_braces(text: str, start_idx: int) -> int:
    assert text[start_idx] == "{", f"expected '{{' at {start_idx}"
    depth = 0
    i = start_idx
    in_str: str | None = None
    while i < len(text):
        ch = text[i]
        if in_str is None:
            if ch == '"' or ch == "'":
                in_str = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
        else:
            if ch == "\\" and i + 1 < len(text):
                i += 2
                continue
            if ch == in_str:
                in_str = None
        i += 1
    raise AssertionError(f"unbalanced braces starting at {start_idx}")


def _enumerate_tool_blocks() -> list[tuple[str, int, int]]:
    text = TOOLS_TS.read_text()
    blocks: list[tuple[str, int, int]] = []
    for m in _TOOL_NAME_LINE.finditer(text):
        open_idx = m.end() - 1
        close_idx = _balance_braces(text, open_idx)
        blocks.append((m.group("name"), open_idx, close_idx))
    return blocks


def test_each_description_has_usage_or_behavior_marker() -> None:
    descs = _extract_top_level_descriptions()
    offenders: list[tuple[str, str]] = []
    for name, desc in descs:
        lower = desc.lower()
        if not any(marker in lower for marker in MARKERS):
            offenders.append((name, desc[:80]))
    assert not offenders, (
        "Quality-floor violation: descriptions missing a Usage-or-Behavior "
        f"marker (one of {sorted(MARKERS)}):\n"
        + "\n".join(f"  {n}: {d!r}" for n, d in offenders)
    )


def test_camouflaging_status_defines_what_is_detected() -> None:
    descs = dict(_extract_top_level_descriptions())
    assert "camouflaging_status" in descs, (
        "camouflaging_status tool not found in toolSchemas"
    )
    desc = descs["camouflaging_status"].lower()
    assert "detect" in desc, (
        "camouflaging_status desc must contain 'detect' to lift the "
        f"1.7/5 Glama ambiguity score; got: {descs['camouflaging_status']!r}"
    )
    assert any(kw in desc for kw in CAM_KEYWORDS), (
        f"camouflaging_status desc must contain at least one of "
        f"{sorted(CAM_KEYWORDS)} (defines WHAT is detected); "
        f"got: {descs['camouflaging_status']!r}"
    )


def test_every_tool_has_annotations_block() -> None:
    text = TOOLS_TS.read_text()
    blocks = _enumerate_tool_blocks()
    assert len(blocks) == 15, (
        f"expected 15 tool entries, found {len(blocks)}: {[b[0] for b in blocks]}"
    )
    annotations_re = re.compile(r"^    annotations:\s*\{", re.MULTILINE)
    total = len(annotations_re.findall(text))
    missing: list[str] = []
    for name, open_idx, close_idx in blocks:
        span = text[open_idx:close_idx]
        if not annotations_re.search(span):
            missing.append(name)
    assert not missing and total == 15, (
        f"Quality-floor violation: tools missing `annotations: {{` block at "
        f"column 4 (found {total}, expected 15); missing tools: "
        f"{sorted(missing)}"
    )


def test_every_tool_has_output_schema_block() -> None:
    text = TOOLS_TS.read_text()
    blocks = _enumerate_tool_blocks()
    assert len(blocks) == 15, (
        f"expected 15 tool entries, found {len(blocks)}: {[b[0] for b in blocks]}"
    )
    output_re = re.compile(r"^    outputSchema:\s*\{", re.MULTILINE)
    total = len(output_re.findall(text))
    missing: list[str] = []
    for name, open_idx, close_idx in blocks:
        span = text[open_idx:close_idx]
        if not output_re.search(span):
            missing.append(name)
    assert not missing and total == 15, (
        f"Quality-floor violation: tools missing `outputSchema: {{` block at "
        f"column 4 (found {total}, expected 15); missing tools: "
        f"{sorted(missing)}"
    )


def test_no_top_level_description_exceeds_30_cl100k_tokens() -> None:
    descs = _extract_top_level_descriptions()
    offenders: list[tuple[str, int, str]] = []
    for name, desc in descs:
        n = _tok(desc)
        if n > 30:
            offenders.append((name, n, desc[:80]))
    assert not offenders, (
        "Quality-floor 30-tok ceiling violation (redundant with "
        "test_tool_description_budget.py; this message is the friendly "
        "per-tool variant):\n"
        + "\n".join(f"  {n}: {t} tok -- {d!r}" for n, t, d in offenders)
    )
