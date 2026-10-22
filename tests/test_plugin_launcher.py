"""The plugin's MCP launcher must resolve the installed engine, never a
same-named package in the directory Claude Code was started from."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "plugin" / "scripts" / "iai-pme-mcp"


def _plant_impostor_package(project: Path, decoy: Path) -> None:
    pkg = project / "iai_mcp" / "cli"
    pkg.mkdir(parents=True)
    (project / "iai_mcp" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "_capture.py").write_text(
        f"def _resolve_wrapper_path():\n    return {str(decoy)!r}\n",
        encoding="utf-8",
    )


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")
def test_launcher_ignores_a_project_folder_named_like_the_package(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    decoy = tmp_path / "decoy-wrapper.js"
    decoy.write_text("// not the engine\n", encoding="utf-8")
    _plant_impostor_package(project, decoy)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    node_log = tmp_path / "node-args.txt"
    fake_node = fake_bin / "node"
    fake_node.write_text(f'#!/bin/sh\nprintf "%s" "$1" > "{node_log}"\n', encoding="utf-8")
    fake_node.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
    env["IAI_MCP_PYTHON"] = sys.executable
    env.pop("PYTHONPATH", None)

    subprocess.run(
        ["sh", str(LAUNCHER)], cwd=project, env=env,
        capture_output=True, text=True, timeout=60,
    )

    launched = node_log.read_text(encoding="utf-8") if node_log.exists() else ""
    assert launched != str(decoy), "the launcher ran a wrapper path from the project folder"

    from iai_mcp.cli._capture import _resolve_wrapper_path

    try:
        expected = str(_resolve_wrapper_path())
    except Exception:
        expected = None
    if expected is not None and Path(expected).is_file():
        assert launched == expected
