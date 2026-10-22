"""Structural regression: the lexical/symbol/xref index hot path never
reaches the embed worker or the brain embedder, whether or not the
semantic lane's segment exists on disk.

Non-goal, stated explicitly: there is deliberately NO `index_all`-plus-
vectors lockstep publish. The vector lane is background-only and
additive, populated ONLY through `embed-index`'s detached executor run
-- never a fourth lane in `index_all`'s synchronous publish sequence.
This module encodes that decision so a future edit does not "helpfully"
wire embedding into the hot path to keep the lanes in lockstep.

Every assertion here is a STATIC scan (`ast`, source text only) -- the
index entrypoints are never imported to inspect them, since importing
`iai_mcp.tools` would be cheap but importing `embed_worker` itself would
load the brain's native extension into this test process.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "iai_mcp"

FORBIDDEN_MODULES = ("iai_mcp_native", "iai_mcp.tools.embed_worker")

# The lexical/symbol/xref index entrypoints: the two library functions
# that do the actual indexing, plus the two CLI call sites that invoke
# them from `iai-tools index` / `iai-mcp tools index`.
ENTRYPOINTS = [
    (SRC / "tools" / "__init__.py", "index_paths"),
    (SRC / "tools" / "__init__.py", "index_symbols"),
    (SRC / "tools" / "__init__.py", "index_xref"),
    (SRC / "tools" / "__init__.py", "index_both"),
    (SRC / "tools" / "__init__.py", "index_all"),
    (SRC / "tools_cli.py", "_cmd_index"),
    (SRC / "cli" / "_tools.py", "cmd_tools_index"),
]


def _module_file(dotted: str) -> "Path | None":
    """Resolves a dotted `iai_mcp.*` module name to its source file under
    `src/`. Anything outside the `iai_mcp` top-level package (stdlib,
    the native extensions `iai_mcp_tools_core`/`iai_mcp_native`) resolves
    to None and is treated as an opaque leaf -- the forbidden-name check
    still catches `iai_mcp_native` by name, it just does not recurse
    into it (there is no first-party source file to recurse into).
    """
    if dotted.split(".")[0] != "iai_mcp":
        return None
    rel = Path(*dotted.split("."))
    candidate_pkg = REPO / "src" / rel / "__init__.py"
    if candidate_pkg.exists():
        return candidate_pkg
    candidate_mod = (REPO / "src" / rel).with_suffix(".py")
    if candidate_mod.exists():
        return candidate_mod
    return None


def _imported_module_names(tree: ast.AST) -> "set[str]":
    """ALL imports anywhere under `tree` (any nesting depth) -- the correct
    scan for a specific FUNCTION's own body, since every statement inside
    it runs when that function is called, function-scoped imports included.
    """
    names: "set[str]" = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
    return names


def _module_level_imports(tree: ast.Module) -> "set[str]":
    """Only imports that execute merely by IMPORTING the module: top-level
    statements, descending into `if`/`try` control flow but NEVER into a
    `def`/`class` body. A function-scoped `import X` inside some unrelated
    function of this module does not run just because the module itself
    was imported -- it runs only when that function is later called, a
    separate question this BFS does not need to answer for the index
    entrypoints, since their own direct imports are captured separately
    (see `_imported_module_names` on the entrypoint function itself).
    """
    names: "set[str]" = set()

    def walk_body(body: "list[ast.stmt]") -> None:
        for node in body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    names.add(node.module)
            elif isinstance(node, (ast.If, ast.Try)):
                for attr in ("body", "orelse", "finalbody"):
                    sub = getattr(node, attr, None)
                    if sub:
                        walk_body(sub)
                if isinstance(node, ast.Try):
                    for handler in node.handlers:
                        walk_body(handler.body)

    walk_body(tree.body)
    return names


def _find_function(tree: ast.Module, name: str) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def _is_forbidden(name: str, forbidden: str) -> bool:
    return name == forbidden or name.startswith(f"{forbidden}.")


def _reachable_modules(start_names: "set[str]") -> "set[str]":
    """BFS over the first-party `iai_mcp.*` import graph, modeling actual
    Python import semantics: importing a module runs only that module's
    OWN top-level code (plus its parent packages' top-level code) --
    never an unrelated function's lazy, function-scoped import inside it.
    `start_names` is the entrypoint function's own direct imports (every
    nesting level, since the function body itself always runs); each
    further hop is module-level-only, matching what actually loads.
    """
    visited: "set[str]" = set()
    queue = list(start_names)
    while queue:
        name = queue.pop()
        if name in visited:
            continue
        visited.add(name)
        mod_file = _module_file(name)
        if mod_file is None:
            continue
        tree = ast.parse(mod_file.read_text(encoding="utf-8"), filename=str(mod_file))
        for imported in _module_level_imports(tree):
            if imported not in visited:
                queue.append(imported)
    return visited


def test_index_hot_path_never_reaches_embed_worker_or_brain_embedder() -> None:
    for path, func_name in ENTRYPOINTS:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        func_node = _find_function(tree, func_name)
        direct_imports = _imported_module_names(func_node)

        for forbidden in FORBIDDEN_MODULES:
            hit = {n for n in direct_imports if _is_forbidden(n, forbidden)}
            assert not hit, (
                f"{path.relative_to(REPO)}::{func_name} directly imports {forbidden}: {hit}"
            )

        reachable = _reachable_modules(direct_imports)
        for forbidden in FORBIDDEN_MODULES:
            hit = {n for n in reachable if _is_forbidden(n, forbidden)}
            assert not hit, (
                f"{path.relative_to(REPO)}::{func_name} transitively reaches {forbidden} "
                f"via: {sorted(reachable)}"
            )


def test_index_all_does_not_publish_the_vector_lane() -> None:
    """Non-goal, asserted structurally: `index_all`'s lockstep publish
    stays whole_file -> symbol -> xref. The vector lane is never a
    fourth lane here -- it is populated only via `embed-index`'s
    detached executor run, never inline with the O(file) index path.
    """
    path = SRC / "tools" / "__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    func_node = _find_function(tree, "index_all")

    call_attrs = {
        node.func.attr
        for node in ast.walk(func_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    forbidden_calls = {"index_vectors", "search_vector"}
    hit = call_attrs & forbidden_calls
    assert not hit, f"index_all must not call vector-lane natives, found: {hit}"


# ---------------------------------------------------------------------------
# Warm embed-latency floor: sampling structure, exercised via an injected
# stub encode callable (no brain load) -- see embed_worker.warm_latency.
# ---------------------------------------------------------------------------


def test_warm_latency_sampling_emits_median_and_p95(tmp_path: Path) -> None:
    from iai_mcp.tools.embed_worker import _percentiles, warm_latency

    # 1) Pure percentile helper, known sample list, nearest-rank method:
    # n=5, p50 idx=round(0.5*4)=2 -> ordered[2]=30; p95 idx=round(0.95*4)=4 -> ordered[4]=50.
    known = [50.0, 10.0, 30.0, 40.0, 20.0]
    pct = _percentiles(known)
    assert pct["median"] == 30.0
    assert pct["p95"] == 50.0

    # 2) Sampling STRUCTURE via an injected fixed-vector stub encode callable
    # -- no brain load. Warm-up discard (1 call) + one timed call per file.
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    n_files = 5
    paths = []
    for i in range(n_files):
        p = corpus / f"file_{i}.py"
        p.write_text(f"def f_{i}(): return {i}\n")
        paths.append(str(p))

    call_count = {"n": 0}

    def stub_encode(text: str) -> "list[float]":
        call_count["n"] += 1
        return [0.0, 1.0, 0.0]

    result = warm_latency(paths, encode=stub_encode)

    assert call_count["n"] == n_files + 1, (
        "expected one discarded cold call plus one timed call per sampled file"
    )
    assert result["n"] == n_files
    assert result["floor"] is True
    assert "median_ms" in result and "p95_ms" in result
    assert isinstance(result["median_ms"], float)
    assert isinstance(result["p95_ms"], float)
    assert result["median_ms"] >= 0.0
    assert result["p95_ms"] >= result["median_ms"] - 1e-9


@pytest.mark.perf
def test_warm_latency_real_corpus_floor() -> None:
    """The real corpus-representative floor: a bounded sample of real
    source files, the real embedder, WARM repeats only. It runs only on
    request -- `--perf` opt-in, never part of the default correctness
    gate. A direct run of this test must stay a bounded
    sample tail (well under 300s), never a full-repo embed.
    """
    from iai_mcp.tools.embed_worker import warm_latency

    sample = sorted(SRC.rglob("*.py"))[:50]
    result = warm_latency([str(p) for p in sample])
    assert result["floor"] is True
    assert result["n"] == len(sample)
