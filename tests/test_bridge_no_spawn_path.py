from __future__ import annotations

import platform
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WRAPPER = REPO / "mcp-wrapper"

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="bash + npm tooling assumed POSIX (mcp-wrapper build path)",
)


@pytest.fixture(scope="module")
def built_bridge_js() -> Path:
    if not (WRAPPER / "node_modules").exists():
        subprocess.run(["npm", "install"], cwd=WRAPPER, check=True)
    subprocess.run(["npm", "run", "build"], cwd=WRAPPER, check=True)
    dist = WRAPPER / "dist" / "bridge.js"
    assert dist.exists(), (
        "npm run build should have produced dist/bridge.js — actual: "
        f"{list((WRAPPER / 'dist').glob('*.js')) if (WRAPPER / 'dist').exists() else 'no dist dir'}"
    )
    return dist


def test_dist_bridge_js_has_no_child_process_spawn(built_bridge_js):
    text = built_bridge_js.read_text(encoding="utf-8")

    forbidden_substrings = [
        'child_process.spawn',
        'from "node:child_process"',
        "from 'node:child_process'",
        'require("node:child_process")',
        "require('node:child_process')",
        'require("child_process")',
        "require('child_process')",
    ]

    found = [s for s in forbidden_substrings if s in text]
    assert not found, (
        "REGRESSION: dist/bridge.js contains spawn-related substring(s) "
        f"that the pure-connector refactor removed: {found}. "
        "Someone has re-introduced the TOCTOU spawn race that the "
        "pure-connector refactor eliminated."
    )


def test_dist_bridge_js_has_DaemonUnreachableError(built_bridge_js):
    text = built_bridge_js.read_text(encoding="utf-8")

    count = text.count("DaemonUnreachableError")
    assert count >= 2, (
        f"REGRESSION: dist/bridge.js contains DaemonUnreachableError "
        f"only {count} times (expected >=2: class definition + at least "
        f"one throw-site). The fail-loud error path may have been "
        f"removed or renamed."
    )


def test_dist_bridge_js_has_5000_socket_timeout(built_bridge_js):
    text = built_bridge_js.read_text(encoding="utf-8")

    assert "SOCKET_CONNECT_TIMEOUT_MS = 5000" in text, (
        "REGRESSION: dist/bridge.js does not contain "
        "'SOCKET_CONNECT_TIMEOUT_MS = 5000'. Either the constant was "
        "renamed, the value was changed, or tsc minification was "
        "enabled (which would also break the source-level grep done "
        "criteria). The bridge requires 5000ms to cover the "
        "launchd socket-activation cold-start window."
    )
