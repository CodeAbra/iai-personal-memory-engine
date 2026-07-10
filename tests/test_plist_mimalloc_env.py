"""Both LaunchAgent surfaces must carry the mimalloc page-decommit env vars.

There are two render targets — the editable-install template and the packaged
operator plist. Editing only one ships a silent regression for the other
install path, so this asserts the three keys land in BOTH after rendering.
"""
from __future__ import annotations

import platform
import plistlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="LaunchAgent plists are macOS-only",
)

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "scripts" / "com.iai-mcp.daemon.plist.template"
PACKAGED = REPO / "src" / "iai_mcp" / "_deploy" / "launchd" / "com.iai-mcp.daemon.plist"

_EXPECTED = {
    "MIMALLOC_DECOMMIT_DELAY": "0",
    "MIMALLOC_SEGMENT_DECOMMIT_DELAY": "0",
    "MIMALLOC_ALLOW_DECOMMIT": "1",
}


def _assert_mimalloc_env(rendered: str, surface: str) -> None:
    parsed = plistlib.loads(rendered.encode("utf-8"))
    env = parsed["EnvironmentVariables"]
    for key, value in _EXPECTED.items():
        assert env.get(key) == value, (
            f"{surface}: EnvironmentVariables[{key!r}] == {env.get(key)!r}, "
            f"expected {value!r}"
        )


def test_template_surface_carries_mimalloc_decommit_env() -> None:
    rendered = (
        TEMPLATE.read_text()
        .replace("{PYTHON_PATH}", "/usr/bin/python3")
        .replace("{HOME}", "/tmp/iai-fake-home")
    )
    _assert_mimalloc_env(rendered, "template")


def test_packaged_surface_carries_mimalloc_decommit_env() -> None:
    rendered = PACKAGED.read_text().replace("{USERNAME}", "fakeuser")
    _assert_mimalloc_env(rendered, "packaged")
