"""End-to-end tracer for `iai-tools index`/`search` as real subprocesses.

Confirms A1 (PyO3 binding reachable from Python) and A2 (a second
setuptools-rust RustExtension is sufficient to ship a second wheel module)
against the live `iai-tools` console script, and that a `search` dispatch
never loads a brain module.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _iai_tools_bin() -> str:
    candidate = Path(sys.executable).parent / "iai-tools"
    return str(candidate) if candidate.exists() else "iai-tools"


def _run_iai_tools(args: list[str], *, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_iai_tools_bin(), *args],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_tools_index_then_search_ranks_owning_file_first(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "alpha.py").write_text("def unrelated_helper():\n    return 1\n")
    (corpus / "beta.py").write_text(
        "def zebracorn_finder():\n    return 'zebracorn habitat lookup'\n"
    )
    (corpus / "gamma.py").write_text("def other_thing():\n    return None\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["IAI_MCP_ROOT"] = str(tmp_path / "iai-mcp-root")
    env["HOME"] = str(tmp_path)  # default_store_root() resolves under $HOME/.iai-mcp

    index_proc = _run_iai_tools(["index", str(corpus)], env=env)
    assert index_proc.returncode == 0, index_proc.stderr
    assert "indexed 3 files" in index_proc.stdout

    latencies = []
    for _ in range(5):
        start = time.monotonic()
        search_proc = _run_iai_tools(["search", "zebracorn"], env=env)
        latencies.append(time.monotonic() - start)
        assert search_proc.returncode == 0, search_proc.stderr

    lines = [ln for ln in search_proc.stdout.splitlines() if ln.strip()]
    assert lines, "expected at least one ranked hit"
    assert "beta.py" in lines[0], f"expected beta.py ranked first, got: {lines[0]}"

    latencies.sort()
    p95 = latencies[int(len(latencies) * 0.95) - 1] if len(latencies) > 1 else latencies[0]
    print(
        f"iai-tools search wall-clock: max={max(latencies) * 1000:.1f}ms "
        f"p95={p95 * 1000:.1f}ms n={len(latencies)}"
    )


def test_tools_search_dispatch_loads_no_brain_module(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "only.py").write_text("def marker_token_xyz():\n    pass\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["HOME"] = str(tmp_path)

    index_proc = _run_iai_tools(["index", str(corpus)], env=env)
    assert index_proc.returncode == 0, index_proc.stderr

    brain_modules = (
        "iai_mcp.hippo",
        "iai_mcp.lilli",
        "iai_mcp.store",
        "iai_mcp_native",
        "pandas",
    )
    probe = (
        "import sys, json, iai_mcp.tools_cli as tools_cli;"
        "rc = tools_cli.main(['search', 'marker_token_xyz']);"
        f"brain = [m for m in {brain_modules!r} if m in sys.modules];"
        "print(json.dumps({'rc': rc, 'brain_modules': brain}))"
    )
    probe_proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert probe_proc.returncode == 0, probe_proc.stderr
    result = json.loads(probe_proc.stdout.strip().splitlines()[-1])
    assert result["rc"] == 0
    assert result["brain_modules"] == [], (
        f"tools search must load no brain module, found: {result['brain_modules']}"
    )
