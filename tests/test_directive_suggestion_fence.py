"""Structural fence: the retired classify_is_directive can never again
become a storage trigger, and the typed-marker recognizer's fail-closed
opt-in is wired ONLY on the human-typed transcript-drain path, never on the
assistant-invoked memory_capture RPC.

These are import-graph / call-site assertions, not behavioral ones -- a
passing behavioral test can never mask a reachable auto-classify write path
or a mis-wired opt-in.
"""
from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CAPTURE_PY = _REPO_ROOT / "src" / "iai_mcp" / "capture.py"
_CORE_INIT_PY = _REPO_ROOT / "src" / "iai_mcp" / "core" / "__init__.py"
_DRAIN_WORKER_PY = _REPO_ROOT / "src" / "iai_mcp" / "deferred_drain_worker.py"

_FORBIDDEN_MODULE_SUBSTR = "directive_classify"
_FORBIDDEN_CALL_NAME = "classify_is_directive"
_ALLOWED_DIRECTIVE_MODULE_SUBSTR = "directive_marker"
_ALLOWED_DIRECTIVE_SYMBOL = "is_directive_marker"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_capture_module_never_imports_or_calls_the_retired_classifier() -> None:
    """capture.py must have no reachable import of directive_classify /
    classify_is_directive, and the only symbol it imports to auto-set
    directive is is_directive_marker from directive_marker."""
    tree = _parse(_CAPTURE_PY)

    bad_imports: list[str] = []
    bad_calls: list[str] = []
    directive_marker_imports: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                full = f"{mod}.{alias.name}"
                if _FORBIDDEN_MODULE_SUBSTR in mod.lower() or (
                    _FORBIDDEN_MODULE_SUBSTR in alias.name.lower()
                ):
                    bad_imports.append(full)
                if _ALLOWED_DIRECTIVE_MODULE_SUBSTR in mod.lower():
                    directive_marker_imports.append(full)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _FORBIDDEN_MODULE_SUBSTR in alias.name.lower():
                    bad_imports.append(alias.name)
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else (
                func.attr if isinstance(func, ast.Attribute) else None
            )
            if name == _FORBIDDEN_CALL_NAME:
                bad_calls.append(name)

    assert not bad_imports, (
        f"capture.py must not import the retired classifier: {bad_imports}"
    )
    assert not bad_calls, (
        f"capture.py must not call {_FORBIDDEN_CALL_NAME}: {bad_calls}"
    )
    assert directive_marker_imports, (
        "capture.py must import is_directive_marker from directive_marker "
        "to auto-set directive"
    )
    assert all(
        imp.endswith(_ALLOWED_DIRECTIVE_SYMBOL) for imp in directive_marker_imports
    ), directive_marker_imports


def _capture_turn_calls(tree: ast.Module) -> list[ast.Call]:
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else (
            func.attr if isinstance(func, ast.Attribute) else None
        )
        if name == "capture_turn":
            calls.append(node)
    return calls


def _keyword_bool_value(call: ast.Call, keyword: str) -> "bool | None":
    for kw in call.keywords:
        if kw.arg == keyword and isinstance(kw.value, ast.Constant):
            return bool(kw.value.value)
    return None


def _keyword_value_node(call: ast.Call, keyword: str) -> "ast.expr | None":
    for kw in call.keywords:
        if kw.arg == keyword:
            return kw.value
    return None


def _call_never_enables_marker(call: ast.Call) -> bool:
    """True iff `call` either omits directive_marker_allowed, or passes it
    as the literal constant False. ANY other expression -- a Name, a Call,
    a BinOp, or a non-False constant such as True or 1 -- returns False,
    because a computed expression could evaluate to True at runtime and the
    fence must not be fooled by it. A `**kwargs`/`**opts` splat (ast
    represents it as a keyword with `arg is None`) also returns False -- the
    splat's contents are opaque to static analysis and could carry the flag."""
    if any(kw.arg is None for kw in call.keywords):
        return False
    value = _keyword_value_node(call, "directive_marker_allowed")
    if value is None:
        return True
    return isinstance(value, ast.Constant) and value.value is False


def test_rpc_handler_never_enables_the_marker() -> None:
    """The memory_capture RPC handler's capture_turn call must NOT pass
    directive_marker_allowed at all, or must pass only the literal False --
    the assistant-invoked path must never let the typed marker mint a
    directive, via any expression (Name, Call, BinOp, or any non-False
    constant), not just a literal True, and not via a `**kwargs` splat."""
    tree = _parse(_CORE_INIT_PY)
    calls = _capture_turn_calls(tree)
    assert calls, "core/__init__.py must call capture_turn for memory_capture"

    for call in calls:
        value = _keyword_value_node(call, "directive_marker_allowed")
        has_splat = any(kw.arg is None for kw in call.keywords)
        assert _call_never_enables_marker(call), (
            "memory_capture RPC handler must not pass directive_marker_allowed "
            "as anything other than omitted or the literal False; "
            + (
                "a **kwargs splat is present and could carry the flag"
                if has_splat
                else f"got {ast.dump(value)}"
            )
        )


def test_fence_rejects_computed_directive_marker_allowed_expression() -> None:
    """Self-test of the fence logic itself: a literal
    directive_marker_allowed=True is caught (the pre-existing behavior), and
    so is a COMPUTED expression -- a bare Name, a function Call, a BinOp, and
    a `**kwargs` splat -- none of which are the ast.Constant(False) the fence
    requires. This proves the fence cannot be defeated by hiding True behind
    an expression or smuggling it through a splat."""
    computed_sources = [
        "capture_turn(store, directive_marker_allowed=some_flag)",
        "capture_turn(store, directive_marker_allowed=compute_flag())",
        "capture_turn(store, directive_marker_allowed=(role == 'user'))",
        "capture_turn(store, directive_marker_allowed=True)",
        "capture_turn(store, directive_marker_allowed=1)",
        "capture_turn(store, **opts)",
    ]
    for src in computed_sources:
        tree = ast.parse(src)
        [call] = _capture_turn_calls(tree)
        assert not _call_never_enables_marker(call), (
            f"fence must reject computed/true expression: {src!r}"
        )

    compliant_sources = [
        "capture_turn(store)",
        "capture_turn(store, directive_marker_allowed=False)",
    ]
    for src in compliant_sources:
        tree = ast.parse(src)
        [call] = _capture_turn_calls(tree)
        assert _call_never_enables_marker(call), (
            f"fence must accept omitted/literal-False expression: {src!r}"
        )


def _call_never_forces_directive(call: ast.Call) -> bool:
    """True iff `call` either omits `directive`, or passes it as the literal
    constant `None` or `False`. ANY other expression -- a Name, a Call, a
    BinOp/IfExp, or a non-False/non-None constant such as `True` or `1` --
    returns False, because a computed expression (including one derived
    from RPC params) could evaluate to True at runtime and the fence must
    not be fooled by it. A `**kwargs`/`**opts` splat also returns False --
    its contents are opaque to static analysis and could carry the flag."""
    if any(kw.arg is None for kw in call.keywords):
        return False
    value = _keyword_value_node(call, "directive")
    if value is None:
        return True
    return isinstance(value, ast.Constant) and (
        value.value is False or value.value is None
    )


def test_rpc_handler_never_forces_a_directive() -> None:
    """The memory_capture RPC handler's capture_turn call must not pass
    `directive` as anything other than omitted or a literal `False`/`None`
    -- an RPC caller must never be able to mint a directive by any
    expression, including one derived from params.get('directive')."""
    tree = _parse(_CORE_INIT_PY)
    calls = _capture_turn_calls(tree)
    assert calls, "core/__init__.py must call capture_turn for memory_capture"

    for call in calls:
        value = _keyword_value_node(call, "directive")
        has_splat = any(kw.arg is None for kw in call.keywords)
        assert _call_never_forces_directive(call), (
            "memory_capture RPC handler must not pass directive as anything "
            "other than omitted or a literal False/None; "
            + (
                "a **kwargs splat is present and could carry the flag"
                if has_splat
                else f"got {ast.dump(value)}"
            )
        )


def test_directive_fence_rejects_computed_directive_expression() -> None:
    """Self-test of the fence logic itself: a literal directive=True is
    caught (the vulnerability this fence closes), and so is a COMPUTED
    expression -- a bare Name, a function Call, a ternary, a BinOp, and a
    `**kwargs` splat -- none of which are the ast.Constant(False/None) the
    fence requires."""
    computed_sources = [
        "capture_turn(store, directive=some_flag)",
        "capture_turn(store, directive=compute_flag())",
        "capture_turn(store, directive=(x if y else None))",
        "capture_turn(store, directive=True)",
        "capture_turn(store, directive=1)",
        "capture_turn(store, **opts)",
    ]
    for src in computed_sources:
        tree = ast.parse(src)
        [call] = _capture_turn_calls(tree)
        assert not _call_never_forces_directive(call), (
            f"fence must reject computed/true expression: {src!r}"
        )

    compliant_sources = [
        "capture_turn(store)",
        "capture_turn(store, directive=False)",
        "capture_turn(store, directive=None)",
    ]
    for src in compliant_sources:
        tree = ast.parse(src)
        [call] = _capture_turn_calls(tree)
        assert _call_never_forces_directive(call), (
            f"fence must accept omitted/literal-False/None expression: {src!r}"
        )


def test_transcript_drain_worker_enables_the_marker() -> None:
    """The ambient transcript-batch drain worker's capture_turn call must
    pass directive_marker_allowed=True -- the drained role/text are a
    verbatim record of the human's own turn."""
    tree = _parse(_DRAIN_WORKER_PY)
    calls = _capture_turn_calls(tree)
    assert calls, "deferred_drain_worker.py must call capture_turn"

    assert any(
        _keyword_bool_value(call, "directive_marker_allowed") is True
        for call in calls
    ), "deferred_drain_worker.py must enable directive_marker_allowed=True"
