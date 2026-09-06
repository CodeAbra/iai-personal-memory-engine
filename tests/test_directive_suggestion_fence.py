"""Structural fence: the retired classify_is_directive can never again
become a storage trigger, and the typed-marker recognizer's fail-closed
opt-in is wired ONLY on the human-typed transcript-drain path, never on the
assistant-invoked memory_capture RPC. Also fences the symmetric write
guarantee: the isatty guard lexically dominates and bails before both the
mint and CLI-remove mutations, no self-suppliable override flag exists in
either guarded function, and every directive=False writer in the source
tree is one of the three sanctioned retire paths.

These are import-graph / call-site assertions, not behavioral ones -- a
passing behavioral test can never mask a reachable auto-classify write path
or a mis-wired opt-in.
"""
from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / "src" / "iai_mcp"
_CAPTURE_PY = _SRC_ROOT / "capture.py"
_CORE_INIT_PY = _SRC_ROOT / "core" / "__init__.py"
_DRAIN_WORKER_PY = _SRC_ROOT / "deferred_drain_worker.py"
_IAI_CLI_PY = _SRC_ROOT / "iai_cli.py"

# The only sanctioned writers of directive=False via Table.update -- a new,
# ungated, model-reachable writer must widen this set explicitly, not slip
# in silently.
_ALLOWED_DIRECTIVE_FALSE_WRITERS = frozenset({
    "retrieve.py",
    "migrate/_directive_sweep.py",
    "directive_ops.py",
})

_OVERRIDE_FLAG_NAMES = frozenset({"yes", "force", "no_confirm", "confirm"})

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


# --- REMOVE-side fence: memory_capture has no directive param --------------


def test_memory_capture_handler_exposes_no_directive_param() -> None:
    """core/__init__.py's memory_capture RPC handler must never read a
    "directive" param key or use a `directive=` keyword anywhere in the
    file -- the assistant-facing RPC surface has no such param to smuggle
    a value through. Distinct from test_rpc_handler_never_forces_a_directive
    above, which checks the capture_turn call's keyword VALUE; this checks
    that the param key itself is never referenced at all."""
    tree = _parse(_CORE_INIT_PY)

    bad_keywords = [
        kw.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "directive"
    ]
    bad_string_literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value == "directive"
    ]
    assert not bad_keywords, (
        f"core/__init__.py must not use a directive= keyword: {bad_keywords}"
    )
    assert not bad_string_literals, (
        "core/__init__.py must not reference a 'directive' param key anywhere"
    )


# --- isatty-dominance fence: mint and CLI-remove ----------------------------


def _find_function(tree: ast.Module, name: str) -> "ast.FunctionDef | None":
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _contains_isatty(expr: ast.expr) -> bool:
    """True iff `expr` contains an Attribute node named "isatty" -- an
    inline `.isatty()` call (or a bare attribute reference to it). A bare
    Name (e.g. a variable that merely HOLDS a prior isatty() result) does
    NOT count -- the fence only accepts a guard whose own test directly
    names the check, not one that trusts an untraceable earlier
    computation."""
    return any(
        isinstance(node, ast.Attribute) and node.attr == "isatty"
        for node in ast.walk(expr)
    )


def _guard_bails(if_node: ast.If) -> bool:
    return any(isinstance(n, (ast.Return, ast.Raise)) for n in if_node.body)


def _calls_named(scope: ast.AST, names: "set[str]") -> list[ast.Call]:
    out: list[ast.Call] = []
    for node in ast.walk(scope):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else (
            func.attr if isinstance(func, ast.Attribute) else None
        )
        if name in names:
            out.append(node)
    return out


def _first_bailing_isatty_if(scope: ast.AST) -> "ast.If | None":
    """Earliest `ast.If` in `scope` whose test contains `.isatty()` and
    whose body bails via Return/Raise, or None if no such guard exists."""
    isatty_ifs = [
        node for node in ast.walk(scope)
        if isinstance(node, ast.If) and _contains_isatty(node.test)
    ]
    bailing_ifs = [n for n in isatty_ifs if _guard_bails(n)]
    if not bailing_ifs:
        return None
    return min(bailing_ifs, key=lambda n: n.lineno)


def _isatty_dominates_and_bails(scope: ast.AST, mutating_names: "set[str]") -> bool:
    """True iff a bailing isatty-guard `ast.If` (test contains `.isatty()`,
    body contains Return/Raise) appears LEXICALLY BEFORE every call to a
    name in `mutating_names` within `scope`. False when no mutating call or
    no bailing guard is found at all -- callers must assert those exist
    separately (an empty search space must never silently pass here)."""
    guard = _first_bailing_isatty_if(scope)
    if guard is None:
        return False
    mutating_calls = _calls_named(scope, mutating_names)
    if not mutating_calls:
        return False
    return all(call.lineno > guard.lineno for call in mutating_calls)


def test_cmd_capture_isatty_guard_dominates_mint() -> None:
    tree = _parse(_IAI_CLI_PY)
    func_node = _find_function(tree, "cmd_capture")
    assert func_node is not None, "iai_cli.py must define cmd_capture"

    mutating_calls = _calls_named(func_node, {"capture_turn"})
    assert mutating_calls, "cmd_capture must call capture_turn"

    assert _isatty_dominates_and_bails(func_node, {"capture_turn"}), (
        "cmd_capture's isatty guard must lexically precede and bail before "
        "the capture_turn mint call"
    )


def test_cmd_directive_remove_isatty_guard_dominates_retire() -> None:
    tree = _parse(_IAI_CLI_PY)
    func_node = _find_function(tree, "cmd_directive_remove")
    assert func_node is not None, "iai_cli.py must define cmd_directive_remove"

    mutating_calls = _calls_named(func_node, {"retire_directive"})
    assert mutating_calls, "cmd_directive_remove must call retire_directive"

    assert _isatty_dominates_and_bails(func_node, {"retire_directive"}), (
        "cmd_directive_remove's isatty guard must lexically precede and "
        "bail before the retire_directive call"
    )


def test_isatty_dominance_fence_rejects_guard_after_mutation() -> None:
    """Self-test: a guard placed AFTER the mutating call must fail."""
    src = (
        "def cmd_x():\n"
        "    retire_directive(store, rid)\n"
        "    if not sys.stdin.isatty():\n"
        "        return 2\n"
    )
    tree = ast.parse(src)
    func_node = _find_function(tree, "cmd_x")
    assert not _isatty_dominates_and_bails(func_node, {"retire_directive"})


def test_isatty_dominance_fence_rejects_computed_isatty_expression() -> None:
    """Self-test: testing a variable that merely HOLDS a prior isatty()
    result (not an inline `.isatty()` call in the guard's own test) must
    fail -- the fence cannot verify a dataflow indirection statically."""
    src = (
        "def cmd_x():\n"
        "    ok = sys.stdin.isatty()\n"
        "    if not ok:\n"
        "        return 2\n"
        "    retire_directive(store, rid)\n"
    )
    tree = ast.parse(src)
    func_node = _find_function(tree, "cmd_x")
    assert not _isatty_dominates_and_bails(func_node, {"retire_directive"})


def test_isatty_dominance_fence_rejects_non_bailing_guard() -> None:
    """Self-test: a guard whose body has no Return/Raise (a no-op branch)
    must fail -- guard-and-bail, not guard-and-continue."""
    src = (
        "def cmd_x():\n"
        "    if not sys.stdin.isatty():\n"
        "        log.debug('non-tty')\n"
        "    retire_directive(store, rid)\n"
    )
    tree = ast.parse(src)
    func_node = _find_function(tree, "cmd_x")
    assert not _isatty_dominates_and_bails(func_node, {"retire_directive"})


def test_isatty_dominance_fence_accepts_compliant_guard() -> None:
    """Self-test: the compliant shape (inline isatty guard, bails, precedes
    the mutating call) must pass."""
    src = (
        "def cmd_x():\n"
        "    if not sys.stdin.isatty():\n"
        "        return 2\n"
        "    retire_directive(store, rid)\n"
    )
    tree = ast.parse(src)
    func_node = _find_function(tree, "cmd_x")
    assert _isatty_dominates_and_bails(func_node, {"retire_directive"})


# --- no-bypass-flag fence: mint and CLI-remove ------------------------------


def _guard_block_has_no_bypass_flag(if_node: ast.If) -> bool:
    """True iff the isatty guard's own `ast.If` (test + body) contains no
    os.environ.get/os.getenv call and no getattr(obj, <override-name>, ...)
    call for a bypass-shaped name (yes/force/no_confirm/confirm) -- scoped
    to the guard construct itself, not the whole function, so an unrelated
    env-var read elsewhere in the same function (e.g. IAI_MCP_STORE) is not
    mistaken for a bypass. A future edit cannot quietly reintroduce a
    self-suppliable override inside the guard's own condition or body."""
    for node in ast.walk(if_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "getenv":
            return False
        if isinstance(func, ast.Name) and func.id == "getenv":
            return False
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "get"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "environ"
        ):
            return False
        if isinstance(func, ast.Name) and func.id == "getattr":
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                name = node.args[1].value
                if isinstance(name, str) and name in _OVERRIDE_FLAG_NAMES:
                    return False
    return True


def test_cmd_capture_mint_guard_has_no_bypass_flag() -> None:
    tree = _parse(_IAI_CLI_PY)
    func_node = _find_function(tree, "cmd_capture")
    assert func_node is not None, "iai_cli.py must define cmd_capture"
    guard = _first_bailing_isatty_if(func_node)
    assert guard is not None, "cmd_capture must have a bailing isatty guard"
    assert _guard_block_has_no_bypass_flag(guard), (
        "cmd_capture's isatty guard must not read an override-shaped env "
        "var or getattr flag"
    )


def test_cmd_directive_remove_guard_has_no_bypass_flag() -> None:
    tree = _parse(_IAI_CLI_PY)
    func_node = _find_function(tree, "cmd_directive_remove")
    assert func_node is not None, "iai_cli.py must define cmd_directive_remove"
    guard = _first_bailing_isatty_if(func_node)
    assert guard is not None, "cmd_directive_remove must have a bailing isatty guard"
    assert _guard_block_has_no_bypass_flag(guard), (
        "cmd_directive_remove's isatty guard must not read an override-shaped "
        "env var or getattr flag"
    )


def test_bypass_flag_fence_rejects_env_override() -> None:
    src = (
        "def cmd_x():\n"
        "    if not sys.stdin.isatty() and not os.environ.get('FORCE'):\n"
        "        return 2\n"
    )
    tree = ast.parse(src)
    func_node = _find_function(tree, "cmd_x")
    guard = _first_bailing_isatty_if(func_node)
    assert guard is not None
    assert not _guard_block_has_no_bypass_flag(guard)


def test_bypass_flag_fence_rejects_getattr_override() -> None:
    src = (
        "def cmd_x(args):\n"
        "    if not sys.stdin.isatty() and not getattr(args, 'yes', False):\n"
        "        return 2\n"
    )
    tree = ast.parse(src)
    func_node = _find_function(tree, "cmd_x")
    guard = _first_bailing_isatty_if(func_node)
    assert guard is not None
    assert not _guard_block_has_no_bypass_flag(guard)


def test_bypass_flag_fence_accepts_clean_block() -> None:
    src = (
        "def cmd_x(args):\n"
        "    if not sys.stdin.isatty():\n"
        "        return 2\n"
        "    text = getattr(args, 'text', None)\n"
    )
    tree = ast.parse(src)
    func_node = _find_function(tree, "cmd_x")
    guard = _first_bailing_isatty_if(func_node)
    assert guard is not None
    # The unrelated getattr(args, 'text', ...) sits OUTSIDE the guard's own
    # If construct -- scoping to the guard proves it doesn't false-positive
    # on legitimate arg reads elsewhere in the function.
    assert _guard_block_has_no_bypass_flag(guard)


# --- shared retire seam: allowlist + positive routing -----------------------


def _module_paths_with_directive_false_update() -> "set[str]":
    hits: set[str] = set()
    for path in _SRC_ROOT.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else (
                func.id if isinstance(func, ast.Name) else None
            )
            if name != "update":
                continue
            for kw in node.keywords:
                if kw.arg != "values" or not isinstance(kw.value, ast.Dict):
                    continue
                for key_node, val_node in zip(kw.value.keys, kw.value.values):
                    if (
                        isinstance(key_node, ast.Constant)
                        and key_node.value == "directive"
                        and isinstance(val_node, ast.Constant)
                        and val_node.value is False
                    ):
                        hits.add(str(path.relative_to(_SRC_ROOT)))
    return hits


def test_directive_false_writers_match_allowlist_exactly() -> None:
    """No NEW, ungated, model-reachable `directive=False` writer may be
    introduced -- every `Table.update(values={"directive": False})` call in
    the source tree must be one of the three sanctioned retire paths."""
    found = _module_paths_with_directive_false_update()
    assert found == set(_ALLOWED_DIRECTIVE_FALSE_WRITERS), (
        f"unexpected writer(s): {found - _ALLOWED_DIRECTIVE_FALSE_WRITERS}; "
        f"missing expected writer(s): {_ALLOWED_DIRECTIVE_FALSE_WRITERS - found}"
    )


def test_drain_worker_remove_path_uses_shared_retire_seam() -> None:
    tree = _parse(_DRAIN_WORKER_PY)
    calls = _calls_named(tree, {"retire_directive"})
    assert calls, (
        "deferred_drain_worker.py must retire through "
        "directive_ops.retire_directive, not a raw Table.update"
    )


def test_cli_remove_handler_uses_shared_retire_seam() -> None:
    tree = _parse(_IAI_CLI_PY)
    func_node = _find_function(tree, "cmd_directive_remove")
    assert func_node is not None, "iai_cli.py must define cmd_directive_remove"
    calls = _calls_named(func_node, {"retire_directive"})
    assert calls, (
        "cmd_directive_remove must retire through directive_ops.retire_directive, "
        "not a raw Table.update"
    )
