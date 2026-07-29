"""The wheel may only ship a SELF-CONTAINED wrapper bundle. The tsc tree
keeps bare imports (@modelcontextprotocol/sdk, zod) that no wheel install
can resolve — staging it shipped a wrapper that died with
ERR_MODULE_NOT_FOUND on every wheel user's MCP registration.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BUNDLE = (
    Path(__file__).resolve().parents[1]
    / "mcp-wrapper" / "dist-bundle" / "index.js"
)

# Top-level statements only: the SDK's own JSDoc examples carry the same
# import text inside comment blocks, where it is inert.
_BARE_IMPORT_RE = re.compile(
    r"""^\s*(?:import|export)[^;\n]{0,200}?from\s*["']@modelcontextprotocol/""",
    re.MULTILINE,
)


def test_bundle_has_no_bare_sdk_imports() -> None:
    if not _BUNDLE.exists():
        pytest.skip("dist-bundle not built in this checkout (npm run bundle)")
    text = _BUNDLE.read_text(encoding="utf-8", errors="replace")
    assert not _BARE_IMPORT_RE.search(text), (
        "the bundle still imports @modelcontextprotocol/* by bare specifier — "
        "it is not self-contained and will not run from a wheel install"
    )
    # A real bundle inlines the SDK: it must be far larger than the bare
    # tsc entrypoint ever is.
    assert _BUNDLE.stat().st_size > 100_000, (
        f"bundle suspiciously small ({_BUNDLE.stat().st_size} bytes) — "
        "esbuild likely failed to inline dependencies"
    )
