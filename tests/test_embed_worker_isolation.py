"""Static (no-import) guard: iai_mcp_native is imported by embed_worker.py
only, never by any other file in the tools subsystem or its two
console-script entrypoints.

Source text is parsed with `ast` -- these modules are never imported to
inspect them, since importing embed_worker would load the brain's native
extension into THIS test process.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "iai_mcp"

BRAIN_EMBEDDER_MODULE = "iai_mcp_native"
SANCTIONED_FILE = "embed_worker.py"


def _scanned_files() -> "list[Path]":
    files = sorted((SRC / "tools").rglob("*.py"))
    files.append(SRC / "tools_cli.py")
    files.append(SRC / "cli" / "_tools.py")
    return files


def _imports_brain_embedder(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == BRAIN_EMBEDDER_MODULE or alias.name.startswith(
                    f"{BRAIN_EMBEDDER_MODULE}."
                ):
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and (
                node.module == BRAIN_EMBEDDER_MODULE
                or node.module.startswith(f"{BRAIN_EMBEDDER_MODULE}.")
            ):
                return True
    return False


def test_isolation_only_embed_worker_imports_brain_embedder() -> None:
    offenders: "list[str]" = []
    sanctioned_imports = False
    for path in _scanned_files():
        if not path.exists():
            continue
        imports_brain = _imports_brain_embedder(path)
        if path.name == SANCTIONED_FILE:
            sanctioned_imports = imports_brain
            continue
        if imports_brain:
            offenders.append(str(path.relative_to(REPO)))

    assert sanctioned_imports, f"{SANCTIONED_FILE} must import {BRAIN_EMBEDDER_MODULE}"
    assert offenders == [], (
        f"only {SANCTIONED_FILE} may import {BRAIN_EMBEDDER_MODULE}, found in: {offenders}"
    )


@pytest.mark.slow
def test_embed_then_search_ranks_the_matching_file_first(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "alpha.py").write_text("def unrelated_marker():\n    pass\n")
    (corpus / "beta.py").write_text("def another_unrelated_marker():\n    pass\n")
    (corpus / "gamma.py").write_text(
        "def compute_prime_factorization(n):\n"
        '    """Return the prime factorization of a positive integer."""\n'
        "    factors = []\n"
        "    divisor = 2\n"
        "    while n > 1:\n"
        "        while n % divisor == 0:\n"
        "            factors.append(divisor)\n"
        "            n //= divisor\n"
        "        divisor += 1\n"
        "    return factors\n"
    )

    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")

    store_root = tmp_path / ".iai-mcp" / "tools" / "vectors"
    files = [str(corpus / "alpha.py"), str(corpus / "beta.py"), str(corpus / "gamma.py")]

    index_proc = subprocess.run(
        [sys.executable, "-m", "iai_mcp.tools.embed_worker", "index", str(store_root), *files],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert index_proc.returncode == 0, index_proc.stderr

    search_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "iai_mcp.tools_cli",
            "search-vector",
            "prime factorization algorithm",
            "--k",
            "3",
        ],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert search_proc.returncode == 0, search_proc.stderr
    lines = [line for line in search_proc.stdout.strip().splitlines() if line]
    assert lines, "expected at least one hit"
    top_path = lines[0].split("\t", 1)[1]
    assert top_path.endswith("gamma.py"), f"expected gamma.py at rank 1, got: {lines}"
