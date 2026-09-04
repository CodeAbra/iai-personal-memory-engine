"""Structural proof that the epistemic classifier never touches stored
verbatim content. AST-based rather than a text grep -- line numbers and
surrounding prose drift, an AST walk does not.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / "src" / "iai_mcp"
_CLASSIFY_MODULE = _SRC_ROOT / "epistemic_classify.py"
_CAPTURE_MODULE = _SRC_ROOT / "capture.py"

_CLASSIFY_FUNC_NAME = "classify_epistemic_status"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def _references_literal_surface(tree: ast.AST) -> list[ast.AST]:
    hits: list[ast.AST] = []
    for sub in ast.walk(tree):
        if isinstance(sub, ast.Name) and sub.id == "literal_surface":
            hits.append(sub)
        elif isinstance(sub, ast.Attribute) and sub.attr == "literal_surface":
            hits.append(sub)
        elif isinstance(sub, ast.Constant) and sub.value == "literal_surface":
            hits.append(sub)
    return hits


def test_classifier_module_never_references_literal_surface():
    tree = _parse(_CLASSIFY_MODULE)
    hits = _references_literal_surface(tree)
    assert hits == [], (
        f"epistemic_classify.py must never reference literal_surface -- "
        f"found {len(hits)} hit(s) at line(s) {[h.lineno for h in hits]}"
    )


def test_classifier_module_imports_nothing_from_capture_store_embed():
    tree = _parse(_CLASSIFY_MODULE)
    forbidden_modules = {"capture", "store", "embed"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[-1]
                assert top not in forbidden_modules, (
                    f"forbidden import of {alias.name!r} at line {node.lineno}"
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            top = module.split(".")[-1]
            assert top not in forbidden_modules, (
                f"forbidden import from {module!r} at line {node.lineno}"
            )


def test_classifier_module_never_writes_an_attribute():
    """Pure metadata computation: no object mutation anywhere in the
    module (no `x.attr = value` or `x.attr += value` assignment target)."""
    tree = _parse(_CLASSIFY_MODULE)
    hits = [
        node for node in ast.walk(tree)
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Attribute) for t in node.targets)
        )
        or (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Attribute)
        )
    ]
    assert hits == [], (
        f"epistemic_classify.py must never assign to an attribute -- "
        f"found assignment(s) at line(s) {[h.lineno for h in hits]}"
    )


def _classify_call_sites(node: ast.AST) -> list[ast.Call]:
    calls: list[ast.Call] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == _CLASSIFY_FUNC_NAME:
                calls.append(sub)
    return calls


def test_capture_turn_binds_classify_result_only_to_epistemic_status():
    """The classifier return value is only ever bound to the local
    `epistemic_status` name, never interpolated into or assigned to
    `text` -- this discharges the verbatim-untouched invariant
    structurally rather than by convention. Covers both plain and
    augmented (`+=`) assignment shapes."""
    tree = _parse(_CAPTURE_MODULE)
    func = _find_function(tree, "capture_turn")

    assert _classify_call_sites(func), (
        "capture_turn must call classify_epistemic_status at least once"
    )

    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            if not _classify_call_sites(node.value):
                continue
            target_names = {
                t.id for t in node.targets if isinstance(t, ast.Name)
            }
            non_name_targets = [t for t in node.targets if not isinstance(t, ast.Name)]
            assert not non_name_targets, (
                f"classify_epistemic_status result assigned to a non-Name "
                f"target at line {node.lineno}: {ast.dump(node.targets[0])}"
            )
            assert target_names == {"epistemic_status"}, (
                f"classify_epistemic_status result must bind only to "
                f"'epistemic_status', found {target_names} at line {node.lineno}"
            )
        elif isinstance(node, ast.AugAssign):
            if not _classify_call_sites(node.value):
                continue
            assert (
                isinstance(node.target, ast.Name)
                and node.target.id == "epistemic_status"
            ), (
                f"classify_epistemic_status result must bind only to "
                f"'epistemic_status', found augmented assignment to "
                f"{ast.dump(node.target)} at line {node.lineno}"
            )

    text_writes_from_classify = [
        node for node in ast.walk(func)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "text" for t in node.targets)
        and _classify_call_sites(node.value)
    ] + [
        node for node in ast.walk(func)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "text"
        and _classify_call_sites(node.value)
    ]
    assert text_writes_from_classify == [], (
        "classify_epistemic_status result must never be assigned to `text`"
    )
