from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "iai_mcp"

DAEMON_REACHABLE: tuple[str, ...] = (
    "daemon/__init__.py",
    "daemon/_watchdog.py",
    "dream.py",
    "identity_audit.py",
    "hippea_cascade.py",
    "socket_server.py",
    "concurrency.py",
    "insight.py",
    "maintenance.py",
)

BLOCKING_NAMES: frozenset[str] = frozenset({
    "build_runtime_graph",
    "optimize_lance_storage",
    "load_state",
    "save_state",
})

ALLOWLIST: frozenset[tuple[str, str, str]] = frozenset()

def _callable_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None

class BareBlockingCallFinder(ast.NodeVisitor):

    def __init__(self, file_basename: str) -> None:
        self.file = file_basename
        self.violations: list[tuple[str, str, str, int]] = []
        self._in_async_depth = 0
        self._async_fn_stack: list[str] = []
        self._in_to_thread_args = False

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._in_async_depth += 1
        self._async_fn_stack.append(node.name)
        self.generic_visit(node)
        self._async_fn_stack.pop()
        self._in_async_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prev = self._in_async_depth
        self._in_async_depth = 0
        self.generic_visit(node)
        self._in_async_depth = prev

    def visit_Await(self, node: ast.Await) -> None:
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "to_thread"
        ):
            prev = self._in_to_thread_args
            self._in_to_thread_args = True
            for arg in node.value.args:
                self.visit(arg)
            for kw in node.value.keywords:
                self.visit(kw)
            self._in_to_thread_args = prev
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _callable_name(node.func)
        current_fn = (
            self._async_fn_stack[-1] if self._async_fn_stack else "?"
        )
        if (
            self._in_async_depth > 0
            and not self._in_to_thread_args
            and name is not None
            and name in BLOCKING_NAMES
            and (self.file, current_fn, name) not in ALLOWLIST
        ):
            self.violations.append((
                self.file,
                current_fn,
                name,
                node.lineno,
            ))
        self.generic_visit(node)

def test_no_bare_blocking_call_in_async_def() -> None:
    all_violations: list[tuple[str, str, str, int]] = []
    for filename in DAEMON_REACHABLE:
        path = SRC / filename
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(), filename=filename)
        finder = BareBlockingCallFinder(filename)
        finder.visit(tree)
        all_violations.extend(finder.violations)
    assert not all_violations, (
        "R4 regression fence: bare sync calls to blocking functions "
        "inside `async def`. Each violation must be either (a) wrapped "
        "in `await asyncio.to_thread(...)` or (b) added to ALLOWLIST "
        "with a row in 07.2-BLOCKING-CALLS-AUDIT.md classifying it as "
        "`safe-fast` with measurement evidence. Violations:\n"
        + "\n".join(
            f"  {f}:{ln} async def {fn} -> {callee}()"
            for f, fn, callee, ln in all_violations
        )
    )

def test_daemon_file_is_actually_scanned() -> None:
    """The async-blocking-call fence skips any listed file whose path does not
    exist (`if not path.exists(): continue`), so it FAILS OPEN: a bad repath
    drops daemon coverage with no error. Pin the daemon package file so the
    scan is proven to visit it."""
    daemon_path = SRC / "daemon" / "__init__.py"
    assert daemon_path.exists(), "daemon package source not on disk"
    reachable = [SRC / n for n in DAEMON_REACHABLE if (SRC / n).exists()]
    assert reachable, "async-blocking-call scan resolved to an empty file list"
    assert daemon_path in reachable, (
        "daemon package source is not in the scanned reachable list — "
        "the bare-sync-in-async fence would silently skip it"
    )


def test_blocking_names_set_is_non_empty() -> None:
    assert "build_runtime_graph" in BLOCKING_NAMES, (
        "BLOCKING_NAMES must contain 'build_runtime_graph' (the "
        "smoking-gun call that drove a 71-min CPU "
        "saturation). Fence is useless without this; it catches future "
        "re-introduction."
    )

def test_ast_walker_correctly_identifies_to_thread_exemption() -> None:
    snippet = '''
import asyncio
from iai_mcp import retrieve

async def good_path(store):
    g, a, r = await asyncio.to_thread(retrieve.build_runtime_graph, store)
    return g

async def bad_path(store):
    g, a, r = retrieve.build_runtime_graph(store)
    return g
'''
    tree = ast.parse(snippet)
    finder = BareBlockingCallFinder("synthetic_snippet.py")
    finder.visit(tree)
    violations = list(finder.violations)
    assert len(violations) == 1, (
        f"Expected exactly 1 violation (in bad_path); got "
        f"{len(violations)}: {violations}"
    )
    f, fn, callee, ln = violations[0]
    assert fn == "bad_path", f"Expected violation in bad_path; got {fn}"
    assert callee == "build_runtime_graph", (
        f"Expected violation on build_runtime_graph; got {callee}"
    )
