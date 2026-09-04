from __future__ import annotations

import re
from pathlib import Path

TOOLS_TS = Path(__file__).resolve().parent.parent / "mcp-wrapper" / "src" / "tools.ts"

def _tok(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return max(1, len(text) // 4) if text else 0

_DESC_BLOCK_RE = re.compile(
    r"description:\s*"
    r'"((?:[^"\\]|\\.)*)"',
    re.MULTILINE,
)

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

def test_tool_count_pinned():
    descs = _extract_top_level_descriptions()
    assert len(descs) == 15, (
        f"expected 15 tool descriptions, found {len(descs)}: {[n for n, _ in descs]}"
    )

def test_each_tool_description_le_30_tokens():
    descs = _extract_top_level_descriptions()
    offenders: list[tuple[str, int, str]] = []
    for name, desc in descs:
        n = _tok(desc)
        if n > 30:
            offenders.append((name, n, desc[:80]))
    assert not offenders, (
        "tool description exceeds the 30-token budget:\n"
        + "\n".join(f"  {n}: {t} tok -- {d!r}" for n, t, d in offenders)
    )

def test_total_tool_descriptions_within_budget():
    # Total budget = N_tools * avg target (15 * 25 tok/tool); per-tool ceiling
    # stays at 30 (enforced by the sibling test). Adjust both when N_tools changes.
    descs = _extract_top_level_descriptions()
    total = sum(_tok(d) for _, d in descs)
    assert total <= 375, (
        f"total description budget {total} tok > 375\n"
        + "\n".join(f"  {n}: {_tok(d)} tok" for n, d in descs)
    )
