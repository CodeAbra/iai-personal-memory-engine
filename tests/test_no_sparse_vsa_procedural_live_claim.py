from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOTS = ("src",)

# Only a field literally named "tier"/"hv_tier" bound to its paired literal is
# an offense. ast.Compare (dispatch arms), Set/List/tuple elements (enum
# literals), dict keys whose value is not the paired literal, bare string
# Constants (e.g. SQL fragments), and `for x in (...)` loop targets are NOT
# scanned -- they are not a record-tier assignment.
TIER_FIELDS = {"tier", "hv_tier"}
PAIRED_LITERAL = {"hv_tier": "sparse_vsa", "tier": "procedural"}

SANCTIONED_CHUNK_MODULE = "src/iai_mcp/lilli/cycle/chunk.py"
SANCTIONED_MINT_FN = "persist_proc_chunk"


def _iter_py_files(root: Path):
    yield from root.rglob("*.py")


def _target_field_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name) and node.id in TIER_FIELDS:
        return node.id
    if isinstance(node, ast.Attribute) and node.attr in TIER_FIELDS:
        return node.attr
    return None


def _is_paired_literal(field: str, value: ast.expr) -> bool:
    return (
        isinstance(value, ast.Constant)
        and isinstance(value.value, str)
        and value.value == PAIRED_LITERAL.get(field)
    )


def _sanctioned_fn_ranges(tree: ast.Module) -> list[tuple[int, int]]:
    # Only a direct module-level def is sanctioned scope -- a same-named
    # nested closure or class method must still trip the guard.
    return [
        (node.lineno, node.end_lineno)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == SANCTIONED_MINT_FN
    ]


def _in_sanctioned_scope(lineno: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= lineno <= end for start, end in ranges)


def _find_offenses(source: str, rel_path: str) -> list[str]:
    offenses: list[str] = []
    tree = ast.parse(source, filename=rel_path)
    # hv_tier is never allow-listed; only tier, and only inside
    # persist_proc_chunk at the locked path.
    sanctioned_ranges = (
        _sanctioned_fn_ranges(tree) if rel_path == SANCTIONED_CHUNK_MODULE else []
    )

    def _record(field: str, lineno: int) -> None:
        if field == "tier" and _in_sanctioned_scope(lineno, sanctioned_ranges):
            return
        offenses.append(f"{rel_path}:{lineno}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                field = _target_field_name(target)
                if field and _is_paired_literal(field, node.value):
                    _record(field, node.lineno)
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None:
                field = _target_field_name(node.target)
                if field and _is_paired_literal(field, node.value):
                    _record(field, node.lineno)
        elif isinstance(node, ast.keyword):
            if node.arg in TIER_FIELDS and _is_paired_literal(node.arg, node.value):
                _record(node.arg, node.lineno)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value in TIER_FIELDS:
                    if _is_paired_literal(key.value, value):
                        _record(key.value, node.lineno)
    return offenses


def test_no_sparse_vsa_producer():
    # src/ never assigns the procedural store tier to a real record.
    offenses: list[str] = []
    for root_name in SCAN_ROOTS:
        root = REPO_ROOT / root_name
        assert root.exists(), f"scan root missing: {root}"
        files = list(_iter_py_files(root))
        assert files, f"scan root yielded zero .py files: {root}"
        for path in files:
            rel_path = path.relative_to(REPO_ROOT).as_posix()
            offenses.extend(_find_offenses(path.read_text(encoding="utf-8"), rel_path))

    assert offenses == [], f"tier/hv_tier bound to a live-record literal: {offenses}"


def test_sparse_vsa_substrate_present():
    assert importlib.util.find_spec("iai_mcp.lilli.tiers.sparse_vsa") is not None

    from iai_mcp.lilli.tiers import sparse_vsa

    for name in ("bind", "unbind", "bundle", "permute", "similarity"):
        assert callable(getattr(sparse_vsa, name))


def test_schema_enum_entries_retained():
    from iai_mcp.types import HV_TIER_ENUM, TIER_ENUM

    assert "sparse_vsa" in HV_TIER_ENUM
    assert "procedural" in TIER_ENUM


_SANCTIONED_MINT_SOURCE = """
def persist_proc_chunk(record):
    chunk_rec = MemoryRecord(
        id=record.id,
        tier="procedural",
    )
    return chunk_rec
"""

_TIER_OUTSIDE_MINT_SOURCE = """
def persist_proc_chunk(record):
    return record


chunk_rec = MemoryRecord(
    id="x",
    tier="procedural",
)
"""

_HV_TIER_INSIDE_MINT_SOURCE = """
def persist_proc_chunk(record):
    chunk_rec = MemoryRecord(
        id=record.id,
        hv_tier="sparse_vsa",
    )
    return chunk_rec
"""


def test_procedural_tier_allowed_in_sanctioned_minter():
    offenses = _find_offenses(_SANCTIONED_MINT_SOURCE, SANCTIONED_CHUNK_MODULE)
    assert offenses == []


def test_procedural_tier_denied_outside_sanctioned_minter():
    offenses = _find_offenses(_TIER_OUTSIDE_MINT_SOURCE, SANCTIONED_CHUNK_MODULE)
    assert offenses != []


def test_procedural_tier_denied_in_non_sanctioned_module():
    other_module = "src/iai_mcp/lilli/cycle/other.py"
    offenses = _find_offenses(_SANCTIONED_MINT_SOURCE, other_module)
    assert offenses != []


def test_sparse_vsa_hv_tier_denied_even_in_sanctioned_minter():
    offenses = _find_offenses(_HV_TIER_INSIDE_MINT_SOURCE, SANCTIONED_CHUNK_MODULE)
    assert offenses != []


_NESTED_MINT_SOURCE = """
def outer_helper():
    def persist_proc_chunk(record):
        chunk_rec = MemoryRecord(id=record.id, tier="procedural")
        return chunk_rec
"""

_METHOD_MINT_SOURCE = """
class Unrelated:
    def persist_proc_chunk(self, record):
        chunk_rec = MemoryRecord(id=record.id, tier="procedural")
        return chunk_rec
"""


def test_procedural_tier_denied_when_minter_nested_in_function():
    offenses = _find_offenses(_NESTED_MINT_SOURCE, SANCTIONED_CHUNK_MODULE)
    assert offenses == [f"{SANCTIONED_CHUNK_MODULE}:4"]


def test_procedural_tier_denied_when_minter_is_class_method():
    offenses = _find_offenses(_METHOD_MINT_SOURCE, SANCTIONED_CHUNK_MODULE)
    assert offenses == [f"{SANCTIONED_CHUNK_MODULE}:4"]


_ASYNC_SANCTIONED_MINT_SOURCE = """
async def persist_proc_chunk(record):
    chunk_rec = MemoryRecord(
        id=record.id,
        tier="procedural",
    )
    return chunk_rec
"""

_ASYNC_NESTED_MINT_SOURCE = """
def outer_helper():
    async def persist_proc_chunk(record):
        chunk_rec = MemoryRecord(id=record.id, tier="procedural")
        return chunk_rec
"""


def test_procedural_tier_allowed_in_async_sanctioned_minter():
    offenses = _find_offenses(_ASYNC_SANCTIONED_MINT_SOURCE, SANCTIONED_CHUNK_MODULE)
    assert offenses == []


def test_procedural_tier_denied_when_async_minter_nested_in_function():
    offenses = _find_offenses(_ASYNC_NESTED_MINT_SOURCE, SANCTIONED_CHUNK_MODULE)
    assert offenses == [f"{SANCTIONED_CHUNK_MODULE}:4"]


def test_sanctioned_module_parent_dir_exists():
    assert (REPO_ROOT / SANCTIONED_CHUNK_MODULE).parent.is_dir()
