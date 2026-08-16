from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CORE_PY = REPO / "src" / "iai_mcp" / "core" / "__init__.py"
TOOLS_TS = REPO / "mcp-wrapper" / "src" / "tools.ts"

TOOL_NAMES: list[str] = [
    "memory_recall",
    "memory_recall_structural",
    "memory_reinforce",
    "memory_contradict",
    "memory_capture",
    "memory_consolidate",
    "profile_get_set",
    "curiosity_pending",
    "schema_list",
    "events_query",
    "topology",
]

PROFILE_DISPATCH_BRANCHES: dict[str, list[str]] = {
    "profile_get_set": ["profile_get", "profile_set"],
}

def _extract_python_keys(module_ast: ast.Module, dispatch_method: str) -> set[str]:
    keys: set[str] = set()
    dispatch_fn = next(
        (
            n for n in module_ast.body
            if isinstance(n, ast.FunctionDef) and n.name == "dispatch"
        ),
        None,
    )
    assert dispatch_fn is not None, "core.dispatch function not found"

    for node in ast.walk(dispatch_fn):
        if not isinstance(node, ast.If):
            continue
        t = node.test
        if not (
            isinstance(t, ast.Compare)
            and isinstance(t.left, ast.Name)
            and t.left.id == "method"
            and len(t.ops) == 1
            and isinstance(t.ops[0], ast.Eq)
            and len(t.comparators) == 1
            and isinstance(t.comparators[0], ast.Constant)
            and isinstance(t.comparators[0].value, str)
        ):
            continue
        if t.comparators[0].value != dispatch_method:
            continue

        for sub in node.body:
            for n in ast.walk(sub):
                if (
                    isinstance(n, ast.Subscript)
                    and isinstance(n.value, ast.Name)
                    and n.value.id == "params"
                ):
                    slc = n.slice
                    if isinstance(slc, ast.Constant) and isinstance(slc.value, str):
                        keys.add(slc.value)
                if (
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "get"
                    and isinstance(n.func.value, ast.Name)
                    and n.func.value.id == "params"
                    and n.args
                    and isinstance(n.args[0], ast.Constant)
                    and isinstance(n.args[0].value, str)
                ):
                    keys.add(n.args[0].value)
    return keys

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

def _find_block(text: str, key: str, search_from: int, search_to: int) -> tuple[int, int]:
    pattern = re.compile(rf"\b{re.escape(key)}\s*:\s*\{{")
    m = pattern.search(text, search_from, search_to)
    if not m:
        return -1, -1
    open_idx = m.end() - 1
    close_idx = _balance_braces(text, open_idx)
    return open_idx, close_idx

_PROP_KEY_LINE = re.compile(
    r"^\s*(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*\{",
    re.MULTILINE,
)

def _extract_property_keys(properties_block: str) -> set[str]:
    keys: set[str] = set()
    i = 0
    depth = 0
    in_str: str | None = None
    while i < len(properties_block):
        ch = properties_block[i]
        if in_str is None:
            if ch == '"' or ch == "'":
                in_str = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            elif depth == 0 and (ch.isalpha() or ch == "_"):
                m = re.match(
                    r"([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*",
                    properties_block[i:],
                )
                if m:
                    keys.add(m.group(1))
                    i += m.end()
                    continue
        else:
            if ch == "\\" and i + 1 < len(properties_block):
                i += 2
                continue
            if ch == in_str:
                in_str = None
        i += 1
    return keys

def _extract_ts_keys(ts_text: str, tool_name: str) -> set[str]:
    for m in _TOOL_NAME_LINE.finditer(ts_text):
        if m.group("name") != tool_name:
            continue
        tool_open = m.end() - 1
        tool_close = _balance_braces(ts_text, tool_open)
        is_open, is_close = _find_block(
            ts_text, "inputSchema", tool_open + 1, tool_close,
        )
        if is_open == -1:
            raise AssertionError(
                f"tool {tool_name!r}: inputSchema block not found"
            )
        props_open, props_close = _find_block(
            ts_text, "properties", is_open + 1, is_close,
        )
        if props_open == -1:
            raise AssertionError(
                f"tool {tool_name!r}: properties block not found"
            )
        props_block = ts_text[props_open + 1 : props_close - 1]
        return _extract_property_keys(props_block)
    raise AssertionError(
        f"tool {tool_name!r} not found in {TOOLS_TS}; update TOOL_NAMES "
        f"mirror in tests/test_tool_schema_python_parity.py"
    )

def test_ts_extractor_finds_known_tool() -> None:
    keys = _extract_ts_keys(TOOLS_TS.read_text(), "memory_capture")
    assert keys, "memory_capture schema parsed as empty — regex broken"
    assert "text" in keys, keys
    assert "cue" in keys, keys
    assert "tier" in keys, keys
    assert "role" in keys, keys

def test_ts_extractor_handles_empty_properties() -> None:
    keys = _extract_ts_keys(TOOLS_TS.read_text(), "topology")
    assert keys == set(), keys

def test_python_extractor_finds_known_method() -> None:
    core_ast = ast.parse(CORE_PY.read_text())
    keys = _extract_python_keys(core_ast, "memory_recall")
    assert "cue" in keys, keys
    assert "cue_embedding" in keys, keys
    assert "session_id" in keys, keys
    assert "budget_tokens" in keys, keys
    assert "language" in keys, keys

def test_ts_schema_advertises_every_python_param_key() -> None:
    core_ast = ast.parse(CORE_PY.read_text())
    ts_text = TOOLS_TS.read_text()

    offenders: list[str] = []
    for tool_name in TOOL_NAMES:
        ts_keys = _extract_ts_keys(ts_text, tool_name)
        dispatch_methods = PROFILE_DISPATCH_BRANCHES.get(
            tool_name, [tool_name],
        )
        py_keys: set[str] = set()
        for m in dispatch_methods:
            py_keys |= _extract_python_keys(core_ast, m)
        missing = py_keys - ts_keys
        if missing:
            offenders.append(
                f"  {tool_name}: TS schema missing keys "
                f"{sorted(missing)} (Python dispatch reads them; "
                f"advertise as optional inputSchema.properties entries)"
            )

    assert not offenders, (
        "V3-02 schema/dispatch parity broken:\n"
        + "\n".join(offenders)
        + f"\n\nFiles to fix:\n  TS schema: {TOOLS_TS}\n  Python dispatch: {CORE_PY}"
    )
