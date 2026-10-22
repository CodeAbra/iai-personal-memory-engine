"""End-to-end tracer for the xref lane: a real cross-file reference (one
file importing a symbol defined in another) resolves through
`search_xref`, with the referenced definition file appended after the
base ranked hits -- never evicting one of them.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from iai_mcp.tools import index_paths, index_xref, search_xref  # noqa: E402


def test_search_xref_appends_referenced_definition_file_after_base_hits_without_eviction(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    def_file = corpus / "def_file.py"
    def_file.write_text("def zebracorn_helper():\n    return 1\n")

    importer_file = corpus / "importer_file.py"
    importer_file.write_text(
        "from def_file import zebracorn_helper\n"
        "result = zebracorn_helper()\n"
        "print('lighthouse beacon signal pattern seven')\n"
    )

    store_root = tmp_path / "tools"
    xref_store_root = store_root / "xref"
    files = [str(def_file), str(importer_file)]

    indexed = index_paths(files, store_root=store_root)
    assert indexed == 2

    xref_indexed = index_xref(files, store_root=xref_store_root)
    assert xref_indexed == 2

    # A query that only overlaps importer_file.py's content: without the
    # xref injection, def_file.py shares zero tokens with this query and
    # would not appear in the ranked output at all.
    hits = search_xref(
        "lighthouse beacon signal pattern seven",
        k=10,
        store_root=store_root,
        xref_store_root=xref_store_root,
    )
    paths = [p for p, _ in hits]

    importer_idx = next(
        (i for i, p in enumerate(paths) if p.endswith("importer_file.py")), None
    )
    def_idx = next((i for i, p in enumerate(paths) if p.endswith("def_file.py")), None)

    assert importer_idx is not None, f"expected importer_file.py in hits, got: {paths}"
    assert def_idx is not None, f"expected def_file.py injected into hits, got: {paths}"
    assert def_idx == importer_idx + 1, (
        "def_file.py must be appended after the base hits without evicting any of them; "
        f"with a single base hit here that lands directly after importer_file.py, got: {paths}"
    )
    assert def_idx < 10, f"the injected definition file must land inside the top K, got: {paths}"


def test_iai_tools_index_builds_three_lanes_and_search_xref_resolves_end_to_end(
    tmp_path: Path,
) -> None:
    import os
    import subprocess

    corpus = tmp_path / "corpus"
    corpus.mkdir()

    def_file = corpus / "def_file.py"
    def_file.write_text("def zebracorn_helper():\n    return 1\n")

    importer_file = corpus / "importer_file.py"
    importer_file.write_text(
        "from def_file import zebracorn_helper\n"
        "result = zebracorn_helper()\n"
        "print('nautilus copper harbor signal marker nine')\n"
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["HOME"] = str(tmp_path)

    index_proc = subprocess.run(
        [sys.executable, "-m", "iai_mcp.tools_cli", "index", str(corpus)],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert index_proc.returncode == 0, index_proc.stderr

    store_root = tmp_path / ".iai-mcp" / "tools"
    assert (store_root / "manifest.lcman").exists(), "whole-file lane manifest missing"
    assert (store_root / "symbols" / "manifest.lsman").exists(), "symbol lane manifest missing"
    assert (store_root / "xref" / "manifest.lxman").exists(), "xref lane manifest missing"

    search_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "iai_mcp.tools_cli",
            "search-xref",
            "nautilus copper harbor signal marker nine",
            "--k",
            "10",
        ],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert search_proc.returncode == 0, search_proc.stderr
    lines = [line for line in search_proc.stdout.strip().splitlines() if line.strip()]
    paths = [line.split("\t", 1)[1] for line in lines if "\t" in line]

    importer_idx = next((i for i, p in enumerate(paths) if p.endswith("importer_file.py")), None)
    def_idx = next((i for i, p in enumerate(paths) if p.endswith("def_file.py")), None)
    assert importer_idx is not None, f"expected importer_file.py in hits, got: {paths}"
    assert def_idx is not None, f"expected def_file.py injected into hits, got: {paths}"
    assert def_idx == importer_idx + 1, (
        f"def_file.py must rank immediately after importer_file.py, got: {paths}"
    )


def test_search_xref_dispatch_loads_no_brain_module(tmp_path: Path) -> None:
    import json
    import os
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["HOME"] = str(tmp_path)

    brain_modules = (
        "iai_mcp.hippo",
        "iai_mcp.lilli",
        "iai_mcp.store",
        "iai_mcp_native",
        "pandas",
    )
    probe = (
        "import sys, json, iai_mcp.tools_cli as tools_cli;"
        "rc = tools_cli.main(['search-xref', 'marker_symbol_xyz']);"
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
        f"search-xref must load no brain module, found: {result['brain_modules']}"
    )
