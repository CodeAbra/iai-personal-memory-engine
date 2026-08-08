"""OpenClaw wiring: MCP server registration only.

OpenClaw exposes no shell hooks — context injection needs a TypeScript
plugin, which this package does not ship yet. What works today is its
native MCP support: registering the bundled wrapper in
`~/.openclaw/openclaw.json` gives memory_recall / memory_capture as
callable tools. Tools-on-request, not ambient — the installer says so.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

_SERVER_KEY = "iai-pme"


def _openclaw_home() -> Path:
    env = os.environ.get("IAI_MCP_OPENCLAW_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".openclaw"


def _openclaw_config() -> Path:
    return _openclaw_home() / "openclaw.json"


def _load_config(path: Path) -> "dict | None":
    """None means the file exists but is unreadable/unparseable — callers
    must refuse to rewrite it, or foreign servers would be silently lost."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def install_openclaw_mcp() -> int:
    from iai_mcp.cli._capture import _build_iai_mcp_server_entry

    config = _openclaw_config()
    try:
        entry = _build_iai_mcp_server_entry()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    if not shutil.which("node"):
        print("ERROR: node not found in PATH — the MCP wrapper needs Node.js")
        return 1

    data = _load_config(config)
    if data is None:
        print(f"ERROR: cannot parse {config} — fix or remove it, then re-run")
        return 1
    mcp = data.setdefault("mcp", {})
    if not isinstance(mcp, dict):
        print(f"ERROR: {config} has an unexpected 'mcp' shape — not touching it")
        return 1
    servers = mcp.setdefault("servers", {})
    if not isinstance(servers, dict):
        print(f"ERROR: {config} has an unexpected 'mcp.servers' shape — not touching it")
        return 1
    if servers.get(_SERVER_KEY) == entry:
        print(f"openclaw.json already registers {_SERVER_KEY} — no change")
    else:
        servers[_SERVER_KEY] = entry
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"patched: {config} (mcp.servers.{_SERVER_KEY})")

    print(
        "\nNote: OpenClaw gets memory tools on request (memory_recall / "
        "memory_capture over MCP). Ambient capture and automatic recall "
        "injection are not available on OpenClaw — it has no shell hooks."
    )
    return 0


def uninstall_openclaw_mcp() -> int:
    config = _openclaw_config()
    if not config.exists():
        print(f"(not present) {config}")
        return 0
    data = _load_config(config)
    if data is None:
        print(f"NOT patched: cannot parse {config} — remove the {_SERVER_KEY} server by hand")
        return 1
    servers = (data.get("mcp") or {}).get("servers")
    if isinstance(servers, dict) and _SERVER_KEY in servers:
        servers.pop(_SERVER_KEY, None)
        config.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"patched: {config} (mcp.servers.{_SERVER_KEY} removed)")
    else:
        print(f"(no {_SERVER_KEY} server to remove) {config}")
    return 0


def status_openclaw_mcp() -> int:
    config = _openclaw_config()
    data = _load_config(config)
    if data is None:
        print(f"WARNING: cannot parse {config}")
        data = {}
    servers = (data.get("mcp") or {}).get("servers")
    wired = isinstance(servers, dict) and _SERVER_KEY in servers
    print(f"OpenClaw openclaw.json (mcp.servers.{_SERVER_KEY}): {'WIRED' if wired else 'NOT WIRED'}")
    if wired:
        print(
            "\nstatus: ACTIVE — MCP tools registered (tools-on-request; "
            "ambient memory is not available on OpenClaw)"
        )
        return 0
    print("\nstatus: INACTIVE — run: iai-mcp capture-hooks install --target openclaw")
    return 1
