"""Description-quality and token-budget floor for the iai-tools MCP server.

Mirrors tests/test_mcp_tools_description_quality.py and
tests/test_tool_description_budget.py, applied to
mcp-wrapper/src/tools-index.ts's tool descriptions instead of the
memory server's fifteen.
"""

from __future__ import annotations

import re
from pathlib import Path

TOOLS_TS = Path(__file__).resolve().parent.parent / "mcp-wrapper" / "src" / "tools-index.ts"

EXPECTED_TOOL_COUNT = 7
# PER_TOOL_TOKEN_CEILING is the next multiple of 5 above the largest measured
# top-level description; TOTAL_TOKEN_BUDGET is the next multiple of 10 above
# the measured sum across all top-level descriptions. Recompute both from
# _extract_top_level_descriptions() + _tok() when a description changes.
TOTAL_TOKEN_BUDGET = 240
PER_TOOL_TOKEN_CEILING = 55

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
    r"^const [a-zA-Z][a-zA-Z0-9]*Schema = \{",
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


def _enumerate_tool_blocks() -> list[tuple[int, int]]:
    text = TOOLS_TS.read_text()
    blocks: list[tuple[int, int]] = []
    for m in _TOOL_NAME_LINE.finditer(text):
        open_idx = m.end() - 1
        close_idx = _balance_braces(text, open_idx)
        blocks.append((open_idx, close_idx))
    return blocks


def test_tool_count_pinned() -> None:
    descs = _extract_top_level_descriptions()
    assert len(descs) == EXPECTED_TOOL_COUNT, (
        f"expected {EXPECTED_TOOL_COUNT} tool descriptions, found {len(descs)}: "
        f"{[n for n, _ in descs]}"
    )


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


def test_every_tool_has_annotations_block() -> None:
    text = TOOLS_TS.read_text()
    blocks = _enumerate_tool_blocks()
    assert len(blocks) == EXPECTED_TOOL_COUNT, (
        f"expected {EXPECTED_TOOL_COUNT} tool schema blocks, found {len(blocks)}"
    )
    annotations_re = re.compile(r"^  annotations:\s*\{", re.MULTILINE)
    total = len(annotations_re.findall(text))
    missing = 0
    for open_idx, close_idx in blocks:
        span = text[open_idx:close_idx]
        if not annotations_re.search(span):
            missing += 1
    assert missing == 0 and total == EXPECTED_TOOL_COUNT, (
        f"Quality-floor violation: tools missing `annotations: {{` block "
        f"(found {total}, expected {EXPECTED_TOOL_COUNT}, {missing} blocks lacked one)"
    )


def test_every_tool_has_output_schema_block() -> None:
    text = TOOLS_TS.read_text()
    blocks = _enumerate_tool_blocks()
    assert len(blocks) == EXPECTED_TOOL_COUNT, (
        f"expected {EXPECTED_TOOL_COUNT} tool schema blocks, found {len(blocks)}"
    )
    output_re = re.compile(r"^  outputSchema:\s*\{", re.MULTILINE)
    total = len(output_re.findall(text))
    missing = 0
    for open_idx, close_idx in blocks:
        span = text[open_idx:close_idx]
        if not output_re.search(span):
            missing += 1
    assert missing == 0 and total == EXPECTED_TOOL_COUNT, (
        f"Quality-floor violation: tools missing `outputSchema: {{` block "
        f"(found {total}, expected {EXPECTED_TOOL_COUNT}, {missing} blocks lacked one)"
    )


def test_each_tool_description_within_per_tool_ceiling() -> None:
    descs = _extract_top_level_descriptions()
    offenders: list[tuple[str, int, str]] = []
    for name, desc in descs:
        n = _tok(desc)
        if n > PER_TOOL_TOKEN_CEILING:
            offenders.append((name, n, desc[:80]))
    assert not offenders, (
        f"tool description exceeds the {PER_TOOL_TOKEN_CEILING}-token budget:\n"
        + "\n".join(f"  {n}: {t} tok -- {d!r}" for n, t, d in offenders)
    )


def test_total_tool_descriptions_within_budget() -> None:
    descs = _extract_top_level_descriptions()
    total = sum(_tok(d) for _, d in descs)
    assert total <= TOTAL_TOKEN_BUDGET, (
        f"total description budget {total} tok > {TOTAL_TOKEN_BUDGET}\n"
        + "\n".join(f"  {n}: {_tok(d)} tok" for n, d in descs)
    )
