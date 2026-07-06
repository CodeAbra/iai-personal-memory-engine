from __future__ import annotations

import platform
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="plutil is macOS-only",
)

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "scripts" / "com.iai-mcp.daemon.plist.template"

def test_template_renders_to_valid_plist(tmp_path: Path) -> None:
    template_text = TEMPLATE.read_text()
    rendered = template_text.replace(
        "{PYTHON_PATH}", "/usr/bin/python3"
    ).replace("{HOME}", "/tmp/iai-fake-home")
    rendered_path = tmp_path / "com.iai-mcp.daemon.plist"
    rendered_path.write_text(rendered)

    result = subprocess.run(
        ["plutil", "-lint", str(rendered_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"plutil -lint FAILED on rendered template:\n"
        f"--- STDOUT ---\n{result.stdout}\n"
        f"--- STDERR ---\n{result.stderr}\n"
    )
    assert "OK" in result.stdout, result.stdout

def test_template_has_required_keys() -> None:
    text = TEMPLATE.read_text()
    required_markers = [
        "<key>RunAtLoad</key>",
        "<true/>",
        "<key>KeepAlive</key>",
        "<key>Crashed</key>",
        "<key>SoftResourceLimits</key>",
        "IAI_MCP_LAUNCHD_MANAGED",
    ]
    missing = [m for m in required_markers if m not in text]
    assert not missing, f"template missing required markers: {missing}"

def test_template_does_not_run_under_background_policy() -> None:
    text = TEMPLATE.read_text()
    assert "<key>ProcessType</key>" not in text
    assert "<string>Background</string>" not in text

def test_template_has_RunAtLoad_true() -> None:
    text = TEMPLATE.read_text()
    match = re.search(r"<key>RunAtLoad</key>\s*<true/>", text)
    assert match is not None, (
        "template must contain <key>RunAtLoad</key>...<true/> for always-on "
        "activation (daemon starts at login and restarts on crash)"
    )
