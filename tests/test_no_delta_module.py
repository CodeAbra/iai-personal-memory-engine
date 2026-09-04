from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOTS = ("src", "bench", "tests")
TARGET_DOTTED = "iai_mcp.delta"


def _iter_py_files(root: Path):
    yield from root.rglob("*.py")


def _containing_package(path: Path, root: Path) -> str:
    # __package__ is the dotted parent directory for every module, including
    # __init__.py (whose own dotted name equals its parent directory's).
    parts = list(path.relative_to(root).parts)[:-1]
    return ".".join(parts)


def _resolve_import_from(node: ast.ImportFrom, containing_package: str) -> str | None:
    if node.level == 0:
        return node.module
    # importlib._bootstrap._resolve_name semantics: a relative import resolves
    # against the importing file's own package, not the file's module name.
    bits = containing_package.rsplit(".", node.level - 1)
    if len(bits) < node.level or not bits[0]:
        return None
    return f"{bits[0]}.{node.module}" if node.module else bits[0]


def _find_offenses(path: Path, root: Path) -> list[str]:
    offenses: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = _containing_package(path, root)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == TARGET_DOTTED or alias.name.startswith(TARGET_DOTTED + "."):
                    offenses.append(f"{path}:{node.lineno}")
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_import_from(node, package)
            if resolved == TARGET_DOTTED:
                offenses.append(f"{path}:{node.lineno}")
            elif resolved == "iai_mcp":
                for alias in node.names:
                    if alias.name == "delta":
                        offenses.append(f"{path}:{node.lineno}")
    return offenses


def test_iai_mcp_delta_module_absent():
    import iai_mcp  # noqa: F401

    assert importlib.util.find_spec("iai_mcp.delta") is None


def test_no_dotted_path_import_of_iai_mcp_delta():
    offenses: list[str] = []
    for root_name in SCAN_ROOTS:
        root = REPO_ROOT / root_name
        assert root.exists(), f"scan root missing: {root}"
        files = list(_iter_py_files(root))
        assert files, f"scan root yielded zero .py files: {root}"
        for path in files:
            offenses.extend(_find_offenses(path, root))

    assert offenses == [], f"iai_mcp.delta importers found: {offenses}"
