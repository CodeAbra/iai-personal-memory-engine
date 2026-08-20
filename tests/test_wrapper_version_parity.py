"""The bundled MCP wrapper hardcodes its version in the TypeScript source,
separate from the Python distribution metadata. A release that bumps
``mcp-wrapper/package.json`` but not the source strings ships a wrapper that
announces the wrong version over the protocol (the 3.0.3 release did exactly
that). These guards fail the build the moment the two drift.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WRAPPER_SRC = _REPO_ROOT / "mcp-wrapper" / "src"
_PACKAGE_JSON = _REPO_ROOT / "mcp-wrapper" / "package.json"

# The wrapper source is not shipped inside the installed wheel; only guard when
# running from a checkout that carries it.
pytestmark = pytest.mark.skipif(
    not _WRAPPER_SRC.is_dir(), reason="mcp-wrapper source not present (wheel context)"
)


def _package_version() -> str:
    return json.loads(_PACKAGE_JSON.read_text(encoding="utf-8"))["version"]


def test_lifecycle_wrapper_version_matches_package_json():
    text = (_WRAPPER_SRC / "lifecycle.ts").read_text(encoding="utf-8")
    m = re.search(r'WRAPPER_VERSION\s*=\s*"([^"]+)"', text)
    assert m is not None, "WRAPPER_VERSION constant not found in lifecycle.ts"
    assert m.group(1) == _package_version(), (
        "lifecycle.ts WRAPPER_VERSION is out of step with mcp-wrapper/package.json — "
        "bump both on every release"
    )


def test_index_server_version_matches_package_json():
    text = (_WRAPPER_SRC / "index.ts").read_text(encoding="utf-8")
    m = re.search(r'version:\s*"([^"]+)"', text)
    assert m is not None, "server version literal not found in index.ts"
    assert m.group(1) == _package_version(), (
        "index.ts server version is out of step with mcp-wrapper/package.json — "
        "bump both on every release"
    )
