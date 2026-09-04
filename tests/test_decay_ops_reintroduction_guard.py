from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOTS = ("src", "bench", "tests")
SYMBOL_SCAN_ROOTS = ("src/iai_mcp/lilli/ops",)
TARGET_MODULE = "iai_mcp.lilli.ops.decay"
TARGET_PACKAGE = "iai_mcp.lilli.ops"
TARGET_ALIAS = "decay"
FORBIDDEN_SYMBOLS = {"temporal_decay", "decay_structure_edge"}


def _iter_py_files(root: Path):
    yield from root.rglob("*.py")


def _containing_package(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root).parts)[:-1]
    return ".".join(parts)


def _resolve_import_from(node: ast.ImportFrom, containing_package: str) -> str | None:
    if node.level == 0:
        return node.module
    bits = containing_package.rsplit(".", node.level - 1)
    if len(bits) < node.level or not bits[0]:
        return None
    return f"{bits[0]}.{node.module}" if node.module else bits[0]


def _find_import_offenses(path: Path, root: Path) -> list[str]:
    offenses: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = _containing_package(path, root)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == TARGET_MODULE or alias.name.startswith(TARGET_MODULE + "."):
                    offenses.append(f"{path}:{node.lineno}")
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_import_from(node, package)
            if resolved == TARGET_MODULE:
                offenses.append(f"{path}:{node.lineno}")
            elif resolved == TARGET_PACKAGE:
                for alias in node.names:
                    if alias.name == TARGET_ALIAS:
                        offenses.append(f"{path}:{node.lineno}")
    return offenses


def _find_symbol_offenses(path: Path) -> list[str]:
    offenses: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_SYMBOLS:
            offenses.append(f"{path}:{node.lineno}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_SYMBOLS:
            offenses.append(f"{path}:{node.lineno}")
    return offenses


def test_iai_mcp_lilli_ops_decay_module_absent():
    import iai_mcp  # noqa: F401

    assert importlib.util.find_spec(TARGET_MODULE) is None


def test_no_dotted_path_import_of_ops_decay():
    offenses: list[str] = []
    for root_name in SCAN_ROOTS:
        root = REPO_ROOT / root_name
        assert root.exists(), f"scan root missing: {root}"
        files = list(_iter_py_files(root))
        assert files, f"scan root yielded zero .py files: {root}"
        for path in files:
            offenses.extend(_find_import_offenses(path, root))

    assert offenses == [], f"{TARGET_MODULE} importers found: {offenses}"


def test_no_decay_symbol_references_in_ops_package():
    offenses: list[str] = []
    for root_name in SYMBOL_SCAN_ROOTS:
        root = REPO_ROOT / root_name
        assert root.exists(), f"scan root missing: {root}"
        files = list(_iter_py_files(root))
        assert files, f"scan root yielded zero .py files: {root}"
        for path in files:
            offenses.extend(_find_symbol_offenses(path))

    assert offenses == [], f"orphaned decay symbol references found: {offenses}"
