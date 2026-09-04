"""Structural proof that the procedural chunk minter never reads another
record's literal_surface. Payload is ids/roles/hashes only -- an AST walk,
not a text grep, so line numbers and surrounding prose can drift freely."""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CHUNK_MODULE = _REPO_ROOT / "src" / "iai_mcp" / "lilli" / "cycle" / "chunk.py"

_MINT_FN_NAME = "persist_proc_chunk"


def _literal_surface_reads(node: ast.AST) -> list[ast.Attribute]:
    return [
        n for n in ast.walk(node)
        if isinstance(n, ast.Attribute) and n.attr == "literal_surface"
    ]


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def test_persist_proc_chunk_never_reads_literal_surface_off_another_record():
    source = _CHUNK_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_CHUNK_MODULE))
    func = _find_function(tree, _MINT_FN_NAME)

    # ast.keyword (MemoryRecord(literal_surface=label)) is the permitted
    # own-field SET -- it is not an ast.Attribute node and is never flagged.
    hits = _literal_surface_reads(func)
    assert hits == [], (
        f"persist_proc_chunk must never READ another record's "
        f"literal_surface; found {len(hits)} reference(s) at line(s) "
        f"{[h.lineno for h in hits]}"
    )


def test_chunk_module_never_reads_literal_surface_outside_own_field_set():
    # Repo-wide-for-this-module sweep: catches a future helper the mint
    # function delegates to, not just the one named FunctionDef.
    source = _CHUNK_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_CHUNK_MODULE))
    hits = _literal_surface_reads(tree)
    assert hits == [], (
        f"src/iai_mcp/lilli/cycle/chunk.py must never READ a record's "
        f"literal_surface anywhere in the module; found {len(hits)} "
        f"reference(s) at line(s) {[h.lineno for h in hits]}"
    )


_DECOY_SOURCE = """
def persist_proc_chunk(store, candidate):
    src_rec = store.get(candidate.pair[0])
    return src_rec.literal_surface
"""


def test_guard_fires_on_decoy():
    tree = ast.parse(_DECOY_SOURCE, filename="<decoy>")
    func = _find_function(tree, _MINT_FN_NAME)
    hits = _literal_surface_reads(func)
    assert hits != [], (
        "the guard must fire on a decoy persist_proc_chunk that reads "
        "src_rec.literal_surface off another record -- an empty hit list "
        "here means the scan has no teeth"
    )
