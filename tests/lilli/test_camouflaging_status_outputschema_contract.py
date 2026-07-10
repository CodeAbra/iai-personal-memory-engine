from __future__ import annotations

import re
from pathlib import Path

TOOLS_TS = Path(__file__).resolve().parent.parent.parent / "mcp-wrapper" / "src" / "tools.ts"

EXPECTED_RESPONSE_KEYS = frozenset({
    "detected",
    "trajectory_slope",
    "current_mean",
    "sample_count",
    "camouflaging_relaxation",
})

FORBIDDEN_KEYS = frozenset({
    "formality_trend",
    "anomaly_score",
})

def _extract_camouflaging_status_outputschema(text: str) -> str:
    entry_marker = "camouflaging_status: {"
    entry_start = text.find(entry_marker)
    assert entry_start != -1, (
        "camouflaging_status entry not found in tools.ts — file "
        "structure must have changed; update this test."
    )

    schema_marker = "outputSchema: {"
    schema_start = text.find(schema_marker, entry_start)
    assert schema_start != -1, (
        "outputSchema block not found inside camouflaging_status entry "
        "in tools.ts."
    )

    open_brace = schema_start + len(schema_marker) - 1
    depth = 0
    for i in range(open_brace, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1:i]
    raise AssertionError(
        "outputSchema block in camouflaging_status is not brace-balanced."
    )

def _declared_property_keys(schema_body: str) -> set[str]:
    props_marker = "properties: {"
    props_start = schema_body.find(props_marker)
    assert props_start != -1, (
        "outputSchema for camouflaging_status has no properties block."
    )

    open_brace = props_start + len(props_marker) - 1
    depth = 0
    end = None
    for i in range(open_brace, len(schema_body)):
        ch = schema_body[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    assert end is not None, "properties block in outputSchema not brace-balanced."

    body = schema_body[open_brace + 1:end]
    keys: set[str] = set()
    depth = 0
    i = 0
    n = len(body)
    while i < n:
        ch = body[i]
        if ch == "{":
            depth += 1
            i += 1
            continue
        if ch == "}":
            depth -= 1
            i += 1
            continue
        if depth == 0 and ch.isalpha():
            m = re.match(r"([A-Za-z_][A-Za-z_0-9]*)\s*:", body[i:])
            if m:
                keys.add(m.group(1))
                i += m.end()
                continue
        i += 1
    return keys

def test_tools_ts_exists() -> None:
    assert TOOLS_TS.is_file(), f"missing {TOOLS_TS}"

def test_camouflaging_status_outputschema_matches_actual_response() -> None:
    text = TOOLS_TS.read_text(encoding="utf-8")
    schema_body = _extract_camouflaging_status_outputschema(text)
    declared = _declared_property_keys(schema_body)

    missing_from_schema = EXPECTED_RESPONSE_KEYS - declared
    extras_in_schema = declared - EXPECTED_RESPONSE_KEYS

    assert not missing_from_schema, (
        f"camouflaging_status outputSchema is missing these keys that "
        f"camouflaging.detect_camouflaging() + the camouflaging_status dispatch actually returns: "
        f"{sorted(missing_from_schema)}. "
        f"Source of truth: src/iai_mcp/camouflaging.py:87-92 and "
        f"src/iai_mcp/core/__init__.py (camouflaging_status dispatch)."
    )
    assert not extras_in_schema, (
        f"camouflaging_status outputSchema declares these keys that are "
        f"NEVER in the response: {sorted(extras_in_schema)}. "
        f"Source of truth: src/iai_mcp/camouflaging.py:87-92 and "
        f"src/iai_mcp/core/__init__.py (camouflaging_status dispatch)."
    )

def test_camouflaging_status_outputschema_has_no_legacy_names() -> None:
    text = TOOLS_TS.read_text(encoding="utf-8")
    schema_body = _extract_camouflaging_status_outputschema(text)
    declared = _declared_property_keys(schema_body)

    leaks = declared & FORBIDDEN_KEYS
    assert not leaks, (
        f"camouflaging_status outputSchema still declares legacy field "
        f"names that no Python code path returns: {sorted(leaks)}. "
        f"These came from commit 8229af1 and must be removed."
    )
