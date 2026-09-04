"""Dormancy contract for the parked FHRR entity-bind design: the mint-side
encoder and the entity-bind tag constant are retained substrate, but neither
may be referenced from any production module.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "iai_mcp"

_DORMANT_SYMBOLS = ("from_embedding_fhrr", "FHRR_ENTITY_BIND_TAG")

_ALLOWED_MODULES: set[str] = set()  # relative to SRC_ROOT -- empty: no production module may reference either dormant symbol


def _references_symbol(path: Path, symbol: str) -> bool:
    # AST-only scope: catches import/attribute/name references, not a
    # dynamic getattr()/importlib lookup built from a string literal.
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.alias) and node.name == symbol:
            return True
        if isinstance(node, ast.Name) and node.id == symbol:
            return True
        if isinstance(node, ast.Attribute) and node.attr == symbol:
            return True
    return False


def test_dormant_fhrr_symbols_referenced_by_no_production_module() -> None:
    assert SRC_ROOT.exists(), f"scan root missing: {SRC_ROOT}"
    files = list(SRC_ROOT.rglob("*.py"))
    assert files, f"scan root yielded zero .py files: {SRC_ROOT}"

    for symbol in _DORMANT_SYMBOLS:
        referencing: set[str] = set()
        for path in files:
            if _references_symbol(path, symbol):
                referencing.add(str(path.relative_to(SRC_ROOT)))
        assert referencing == _ALLOWED_MODULES, (
            f"{symbol} referenced by unexpected modules: "
            f"{referencing - _ALLOWED_MODULES}; expected exactly {_ALLOWED_MODULES}"
        )
