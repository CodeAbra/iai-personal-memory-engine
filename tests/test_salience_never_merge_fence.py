"""Structural proof that salience_level never gates a merge, drop, or
pattern-separation decision, and never drives NT-style rewriting or
decay-resistance coupling. AST-based rather than a text grep -- line numbers
and surrounding prose drift, an AST walk does not.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / "src" / "iai_mcp"

_FIELD_NAME = "salience_level"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def _references_field(node: ast.AST, field: str = _FIELD_NAME) -> list[ast.AST]:
    hits: list[ast.AST] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id == field:
            hits.append(sub)
        elif isinstance(sub, ast.Attribute) and sub.attr == field:
            hits.append(sub)
        elif isinstance(sub, ast.Constant) and sub.value == field:
            hits.append(sub)
    return hits


def test_salience_level_absent_from_exact_near_dup_target():
    tree = _parse(_SRC_ROOT / "store" / "_store.py")
    func = _find_function(tree, "_exact_near_dup_target")
    hits = _references_field(func)
    assert hits == [], (
        f"salience_level must never appear inside _exact_near_dup_target -- "
        f"this function decides which record to SKIP as a near-dup; found "
        f"{len(hits)} reference(s) at line(s) {[h.lineno for h in hits]}"
    )


def test_salience_level_absent_from_pattern_separation_gate():
    tree = _parse(_SRC_ROOT / "store" / "_store.py")
    func = _find_function(tree, "_pattern_separation_gate_with_hits")
    hits = _references_field(func)
    assert hits == [], (
        f"salience_level must never appear inside "
        f"_pattern_separation_gate_with_hits -- this function makes the "
        f"merge/drop gate decision; found {len(hits)} reference(s) at "
        f"line(s) {[h.lineno for h in hits]}"
    )


def _find_near_dup_dedup_loop(tree: ast.Module) -> ast.For:
    candidates = []
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            src = ast.unparse(node)
            if "never_merge" in src and "continue" in src:
                candidates.append(node)
    assert len(candidates) == 1, (
        f"expected exactly one never_merge-guarded dedup for-loop in "
        f"capture.py; found {len(candidates)}"
    )
    return candidates[0]


def test_salience_level_never_gates_the_near_dup_dedup_loop():
    tree = _parse(_SRC_ROOT / "capture.py")
    loop = _find_near_dup_dedup_loop(tree)

    # salience_level (via the raise_salience_level_if_higher call) MUST
    # appear in the loop -- proving the monotone-raise wiring is present,
    # not just structurally absent by accident.
    all_hits = _references_field(loop)
    assert all_hits, (
        "expected salience_level to appear in the near-dup dedup loop as a "
        "monotone-raise call argument; found none"
    )

    condition_hits: list[ast.AST] = []
    for sub in ast.walk(loop):
        if isinstance(sub, ast.If):
            condition_hits.extend(_references_field(sub.test))
        if isinstance(sub, ast.Compare):
            condition_hits.extend(_references_field(sub))

    assert condition_hits == [], (
        f"salience_level must never appear inside a branch condition of the "
        f"near-dup dedup loop -- it may only appear as a plain call "
        f"argument; found {len(condition_hits)} reference(s) at line(s) "
        f"{[h.lineno for h in condition_hits]}"
    )


def _iter_source_files(root: Path):
    yield from root.rglob("*.py")


def test_salience_level_absent_from_sleep_pipeline():
    sleep_pipeline_root = _SRC_ROOT / "lilli" / "cycle" / "sleep_pipeline"
    assert sleep_pipeline_root.is_dir(), (
        f"expected sleep pipeline package at {sleep_pipeline_root}"
    )
    offenders = [
        str(path)
        for path in _iter_source_files(sleep_pipeline_root)
        if _FIELD_NAME in path.read_text()
    ]
    assert offenders == [], (
        f"salience_level must never appear under the sleep-pipeline package "
        f"(decay-resistance coupling was explicitly rejected); found in: "
        f"{offenders}"
    )


def test_salience_level_absent_from_paraphrase_smooth_empathy_functions():
    offenders = []
    for path in _iter_source_files(_SRC_ROOT):
        try:
            tree = _parse(path)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lowered = node.name.lower()
                if any(term in lowered for term in ("paraphrase", "smooth", "empathy")):
                    if _references_field(node):
                        offenders.append(f"{path}:{node.name}")
    assert offenders == [], (
        f"salience_level must never appear inside a paraphrase/smooth/empathy "
        f"function -- the mark must never drive NT-style rewriting; found: "
        f"{offenders}"
    )


def test_salience_level_absent_from_merge_dedup_collapse_separation_functions():
    """Repo-wide fence: any function whose name suggests it decides which
    record to merge/dedup/collapse/separate must never reference
    salience_level. Named-function tests above cover the two functions
    already fenced explicitly -- this sweep catches a future merge path
    added anywhere else in the tree."""
    terms = ("merge", "dedup", "collapse", "separat")
    offenders = []
    walked_matches = 0
    for path in _iter_source_files(_SRC_ROOT):
        try:
            tree = _parse(path)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lowered = node.name.lower()
                if any(term in lowered for term in terms):
                    walked_matches += 1
                    if _references_field(node):
                        offenders.append(f"{path}:{node.name}")
    assert walked_matches > 0, (
        "expected at least one merge/dedup/collapse/separat-named function "
        "in src/iai_mcp -- zero matches means this sweep is vacuous"
    )
    assert offenders == [], (
        f"salience_level must never appear inside a merge/dedup/collapse/"
        f"separation function -- salience is a rank-boost signal only, never "
        f"a merge/drop decision input; found: {offenders}"
    )
