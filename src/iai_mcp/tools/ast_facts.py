"""Python AST fact extraction for the AST code-index lane.

The sole `ast.parse` call site feeding the AST lane's facts across the
PyO3 seam. Rust never parses Python source -- every function here is a
pure transform over already-read source text, never touching disk.
"""

from __future__ import annotations

import ast


def unreadable_fact(path: str) -> dict:
    """The same zero-fact, `parse_ok=False` shape `extract_facts` returns
    for a `SyntaxError`, for a file that could not even be read (deleted,
    permission-denied, truncated mid-write). Lets a disk-read failure
    degrade like a parse failure instead of a distinct error shape.
    """
    return {
        "path": path,
        "parse_ok": False,
        "defs": [],
        "refs": [],
        "calls": [],
        "imports": [],
    }


def extract_facts(path: str, source: str) -> dict:
    """Parses `source` (already read from `path`) into a plain dict of
    `{path, parse_ok, defs, refs, calls, imports}`.

    `defs` are function/class definitions (`{name, line, kind}`, kind is
    `"function"` or `"class"`). `calls` are call sites whose callee is a
    bare name (`{name, line}`) -- the primary go-to-def/find-refs
    resolution candidate. `refs` are every other bare-name load, kept
    distinct from `calls` (excluded from `refs` to avoid double-counting
    the same source position). `imports` records every imported name
    (`{name, module, line, wildcard}`); a `from X import *` records
    `wildcard=True` with `name=None`, since a wildcard names no specific
    symbol at parse time.

    A file `ast.parse` cannot parse (invalid syntax, a stray null byte)
    is never propagated as an exception -- it is recorded as a zero-fact
    entry with `parse_ok=False` instead, so one broken file never aborts
    an `index_all` batch that also touches every other file.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return unreadable_fact(path)

    defs: list[dict] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.append({"name": node.name, "line": node.lineno, "kind": "function"})
        elif isinstance(node, ast.ClassDef):
            defs.append({"name": node.name, "line": node.lineno, "kind": "class"})

    calls: list[dict] = []
    call_func_ids: "set[int]" = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.append({"name": node.func.id, "line": node.lineno})
            call_func_ids.add(id(node.func))

    refs: list[dict] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and id(node) not in call_func_ids
        ):
            refs.append({"name": node.id, "line": node.lineno})

    imports: list[dict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    {
                        "name": alias.asname or alias.name,
                        "module": None,
                        "line": node.lineno,
                        "wildcard": False,
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    imports.append(
                        {"name": None, "module": node.module, "line": node.lineno, "wildcard": True}
                    )
                else:
                    imports.append(
                        {
                            "name": alias.asname or alias.name,
                            "module": node.module,
                            "line": node.lineno,
                            "wildcard": False,
                        }
                    )

    return {
        "path": path,
        "parse_ok": True,
        "defs": defs,
        "refs": refs,
        "calls": calls,
        "imports": imports,
    }
