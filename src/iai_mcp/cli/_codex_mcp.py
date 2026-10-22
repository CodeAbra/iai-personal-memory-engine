"""Codex MCP-server registration for the iai-tools server.

Surgical upsert of only the `[mcp_servers.iai-tools]` table (and its
`.env` sub-table) in the Codex config.toml. Never re-serializes the
whole file and never writes a backup copy -- the live file holds
unrelated secrets in other tables, and those bytes must survive
untouched. The read side uses the stdlib TOML reader for the
idempotency check; the write side is a bounded text edit of only the
table's own byte range.
"""
from __future__ import annotations

import sys
import tomllib
from importlib import resources as _res
from pathlib import Path

from iai_mcp.cli._atomic_write import _atomic_write_text
from iai_mcp.cli._codex_hooks import _codex_home

_TABLE_PATH = "mcp_servers.iai-tools"
_TABLE_HEADER = f"[{_TABLE_PATH}]"
_ENV_TABLE_HEADER = f"[{_TABLE_PATH}.env]"


def _codex_config_path() -> Path:
    return _codex_home() / "config.toml"


def _resolve_tools_wrapper_path() -> Path:
    import os

    env_val = os.environ.get("IAI_MCP_TOOLS_WRAPPER_PATH")
    if env_val:
        p = Path(env_val)
        if p.exists():
            return p
        raise FileNotFoundError(
            f"IAI_MCP_TOOLS_WRAPPER_PATH={env_val!r} is set but the file does not exist."
        )

    import iai_mcp as _pkg

    try:
        pkg_p = Path(str(_res.files("iai_mcp") / "_wrapper" / "tools-index.js"))
        if pkg_p.exists():
            return pkg_p
    except (TypeError, FileNotFoundError):
        pass

    src_file = Path(_pkg.__file__).resolve()
    repo_root = src_file.parent.parent.parent
    editable_path = repo_root / "mcp-wrapper" / "dist" / "tools-index.js"
    if editable_path.exists():
        return editable_path

    raise FileNotFoundError(
        "MCP tools wrapper (tools-index.js) not found. Checked locations:\n"
        "  1. IAI_MCP_TOOLS_WRAPPER_PATH env var (not set)\n"
        f"  2. Package data: {str(_res.files('iai_mcp') / '_wrapper' / 'tools-index.js')}\n"
        f"  3. Editable source: {editable_path}\n"
        "To build: cd mcp-wrapper && npm run build"
    )


def build_iai_tools_server_entry() -> dict:
    wrapper = _resolve_tools_wrapper_path()
    return {
        "command": "node",
        "args": [str(wrapper)],
        "env": {"IAI_MCP_PYTHON": sys.executable},
    }


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _render_table(entry: dict) -> str:
    lines = [f"{_TABLE_HEADER}\n"]
    lines.append(f'command = {_toml_string(entry["command"])}\n')
    args_repr = ", ".join(_toml_string(a) for a in entry["args"])
    lines.append(f"args = [{args_repr}]\n")
    lines.append("\n")
    lines.append(f"{_ENV_TABLE_HEADER}\n")
    for key, val in entry["env"].items():
        lines.append(f"{key} = {_toml_string(val)}\n")
    return "".join(lines)


def _split_table_path(inner: str) -> "list[str]":
    # Splits on '.' outside quotes and strips one layer of quoting per
    # segment, so `mcp_servers."iai-tools"` normalizes the same as
    # `mcp_servers.iai-tools` -- tomllib treats both identically and the
    # text scanner must agree, or install/uninstall corrupt the file.
    parts: "list[str]" = []
    current: "list[str]" = []
    quote: "str | None" = None
    for ch in inner:
        if quote:
            if ch == quote:
                quote = None
            else:
                current.append(ch)
        elif ch in ("'", '"'):
            quote = ch
        elif ch == ".":
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    parts.append("".join(current).strip())
    return parts


def _normalize_table_path(inner: str) -> str:
    return ".".join(_split_table_path(inner))


def _header_kind_and_path(line: str) -> "tuple[str, str | None] | None":
    # Returns (kind, path) for any line that is a TOML table header --
    # 'table' for `[a.b]`, 'array' for `[[a.b]]` (path unused, an
    # array-of-tables is never a match for our fixed single-table
    # entry). Returns None only for lines that are NOT a header at all,
    # so a trailing `# comment` after `]` never hides a header from the
    # boundary scan.
    s = line.strip()
    if not s.startswith("["):
        return None
    if s.startswith("[["):
        return ("array", None)
    end = s.find("]")
    if end == -1:
        return None
    inner = s[1:end]
    if not inner.strip():
        return None
    return ("table", _normalize_table_path(inner))


def _belongs_to_table(path: str, table_path: str) -> bool:
    return path == table_path or path.startswith(table_path + ".")


def _find_table_range(text: str, table_path: str) -> "tuple[int, int] | None":
    lines = text.splitlines(keepends=True)
    start = None
    for i, line in enumerate(lines):
        kind = _header_kind_and_path(line)
        if kind is not None and kind[0] == "table" and kind[1] == table_path:
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        kind = _header_kind_and_path(lines[j])
        if kind is None:
            continue
        if kind[0] == "array" or not _belongs_to_table(kind[1], table_path):
            end = j
            break
    start_offset = sum(len(l) for l in lines[:start])
    end_offset = sum(len(l) for l in lines[:end])
    return start_offset, end_offset


def _load_config_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _existing_entry(text: str) -> "dict | None":
    if not text.strip():
        return None
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return None
    servers = data.get("mcp_servers")
    if not isinstance(servers, dict):
        return None
    entry = servers.get("iai-tools")
    return entry if isinstance(entry, dict) else None


def _toml_parses(text: str) -> bool:
    if not text.strip():
        return True
    try:
        tomllib.loads(text)
        return True
    except tomllib.TOMLDecodeError:
        return False


def _report_ambiguity_refusal(config_path: Path, text: str, verb: str) -> None:
    # `existing is None` from _existing_entry collapses two distinct causes
    # (a TOML syntax error anywhere in the file, or a genuinely absent
    # table) into one value -- disambiguate here so the refusal message
    # points at the real cause instead of sending the user hunting for a
    # header problem that isn't there.
    if not _toml_parses(text):
        print(f"NOT {verb}: {config_path} does not parse as TOML — fix it, then rerun.")
        return
    print(
        f"NOT {verb}: {config_path} — tomllib and the text scanner "
        f"disagree about whether {_TABLE_HEADER} exists — refusing to "
        "touch the file. Normalize the header by hand and rerun."
    )


def _placeholder_entry() -> dict:
    return {
        "command": "node",
        "args": ["<run: cd mcp-wrapper && npm run build>"],
        "env": {"IAI_MCP_PYTHON": sys.executable},
    }


def install_codex_mcp() -> int:
    config_path = _codex_config_path()
    try:
        entry = build_iai_tools_server_entry()
    except FileNotFoundError as exc:
        entry = _placeholder_entry()
        print(
            f"WARN: MCP tools wrapper not found — Codex config.toml entry "
            f"written with placeholder args. Build it first: cd mcp-wrapper "
            f"&& npm run build. ({exc})"
        )

    text = _load_config_text(config_path)
    existing = _existing_entry(text)
    rng = _find_table_range(text, _TABLE_PATH)

    if (existing is not None) != (rng is not None):
        _report_ambiguity_refusal(config_path, text, "patched")
        return 1

    if existing == entry:
        print(f"Codex: {config_path} already has {_TABLE_HEADER} — no change")
        return 0

    block = _render_table(entry)
    if rng is None:
        if text and not text.endswith("\n"):
            text += "\n"
        new_text = text + block
    else:
        start, end = rng
        new_text = text[:start] + block + text[end:]

    config_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(config_path, new_text)
    print(f"Codex: patched {config_path} ({_TABLE_HEADER} registered)")
    return 0


def uninstall_codex_mcp() -> int:
    config_path = _codex_config_path()
    text = _load_config_text(config_path)
    if not text:
        print(f"Codex: {config_path} absent — skipped")
        return 0

    existing = _existing_entry(text)
    rng = _find_table_range(text, _TABLE_PATH)

    if (existing is not None) != (rng is not None):
        _report_ambiguity_refusal(config_path, text, "removed")
        return 1

    if rng is None:
        print(f"Codex: {_TABLE_HEADER} not in {config_path} — no change")
        return 0

    start, end = rng
    new_text = text[:start] + text[end:]
    _atomic_write_text(config_path, new_text)
    print(f"Codex: removed {_TABLE_HEADER} from {config_path}")
    return 0


def codex_mcp_registered() -> bool:
    text = _load_config_text(_codex_config_path())
    return _existing_entry(text) is not None or _find_table_range(text, _TABLE_PATH) is not None


def status_codex_mcp() -> int:
    config_path = _codex_config_path()
    text = _load_config_text(config_path)
    entry = _existing_entry(text)
    if entry is None:
        print(f"Codex {_TABLE_HEADER}: {config_path} NOT WIRED")
        print("\nstatus: INACTIVE — run: iai-mcp capture-hooks install --target codex --with-tools")
        return 1

    try:
        expected = build_iai_tools_server_entry()
    except FileNotFoundError as exc:
        print(f"Codex {_TABLE_HEADER}: WIRED but wrapper unresolved ({exc})")
        return 1

    if entry == expected:
        print(f"Codex {_TABLE_HEADER}: {config_path} WIRED")
        print("\nstatus: ACTIVE — Codex MCP server registered")
        return 0
    print(
        f"Codex {_TABLE_HEADER}: {config_path} WIRED but STALE "
        "(differs from expected — rerun install)"
    )
    return 1
