"""`iai-mcp-server` console script: launch the stdio MCP server via node.

Resolves the wrapper path and env through `_build_iai_mcp_server_entry`,
then replaces this process with node so the host talks the MCP protocol
directly over inherited stdio.
"""

from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> int:
    # argv kept for signature parity with sibling entry points; this
    # launcher takes no options.
    from iai_mcp.cli._capture import _build_iai_mcp_server_entry

    try:
        entry = _build_iai_mcp_server_entry()
    except FileNotFoundError as exc:
        print(f"iai-mcp-server: {exc}", file=sys.stderr)
        return 1

    # Merge, never replace -- PATH must survive so node resolves.
    env = {**os.environ, **entry["env"]}
    os.execvpe(entry["command"], [entry["command"], *entry["args"]], env)
