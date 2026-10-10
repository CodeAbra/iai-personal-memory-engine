"""Brain-independent whole-file code-index glue.

Thin Python wrapper around the compiled iai_mcp_tools_core Rust extension.
This module imports only the standard library and the native extension at
call time -- it must never pull in Hippo, lilli, or any other brain module,
since the search path exists to answer without loading the memory system.
"""

from __future__ import annotations

from pathlib import Path

_DEFAULT_IAI_ROOT = Path.home() / ".iai-mcp"

COMMAND_OUTPUT_STORE_DIRNAME = "tools-output"


def default_store_root() -> Path:
    """Sibling of the Hippo store under the same root: ~/.iai-mcp/tools/."""
    return _DEFAULT_IAI_ROOT / "tools"


def default_command_output_store_root() -> Path:
    """Sibling of the source-code store: ~/.iai-mcp/tools-output/.

    Deliberately a SEPARATE root from `default_store_root()` -- command
    output must never contaminate source-code BM25 term statistics.
    """
    return _DEFAULT_IAI_ROOT / COMMAND_OUTPUT_STORE_DIRNAME


def is_command_output_store_root(root: Path) -> bool:
    """Name-based predicate, NOT a path-equality check against
    `default_command_output_store_root()`. `_DEFAULT_IAI_ROOT` is bound
    to `Path.home()` at import time, so an equality check would never
    fire under a hermetic tmp store_root and would force in-process
    tests to touch the user's real store. The name predicate fires for
    ANY store dir named `tools-output` -- fails toward EXTRA gating
    (safe), never toward NO gating (unsafe).
    """
    return root.name == COMMAND_OUTPUT_STORE_DIRNAME


DEPENDENCY_STORE_DIRNAME = "tools-deps"


def default_dependency_store_root() -> Path:
    """Sibling of the source-code and command-output stores:
    ~/.iai-mcp/tools-deps/. Never contaminates source-code BM25 stats."""
    return _DEFAULT_IAI_ROOT / DEPENDENCY_STORE_DIRNAME


class DependencySegmentError(ValueError):
    """Raised by `validate_dependency_segment` for a malformed ecosystem or
    package name (empty, leading-dash, multi-segment, path-escape). A
    `ValueError` subclass -- `pytest.raises(ValueError)` still matches --
    but distinct from a bare `ValueError` so a caller (e.g.
    `index_dependency`) can catch exactly this malformed-input shape as a
    clean refusal without also absorbing an unrelated `ValueError` (such
    as `resolve_package_location`'s "unsupported ecosystem" signal for a
    value that IS one safe path segment but names no supported
    ecosystem) into the same not-found message.
    """


def validate_dependency_segment(value: str, *, label: str) -> None:
    """The single choke point for a dependency path segment (ecosystem or
    package name): rejects empty, leading-dash (subprocess argv
    injection), and any shape that is not one single safe path component
    -- BEFORE it reaches a path-join or a subprocess argv, whether the
    caller is the store-root builder below or the package-location
    resolver in `dependency.py`.
    """
    if not value:
        raise DependencySegmentError(f"dependency {label} must not be empty")
    if value.startswith("-"):
        raise DependencySegmentError(f"dependency {label} must not start with '-': {value!r}")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts or len(candidate.parts) != 1:
        raise DependencySegmentError(
            f"dependency {label} is not a single safe path segment: {value!r}"
        )


def dependency_store_root_for(
    ecosystem: str, package: str, *, root: "Path | None" = None
) -> Path:
    """One nested store per (ecosystem, package) under the dependency
    root -- package-scoped isolation comes free from the existing generic
    `store_root` parameter every search/index function already accepts.
    A package literally named "tools-output" nested here would trip
    `is_command_output_store_root`'s name predicate on its own segment --
    that only adds secret-gating, never removes it, so it is left
    unhandled.
    """
    validate_dependency_segment(ecosystem, label="ecosystem")
    validate_dependency_segment(package, label="package")
    base = root or default_dependency_store_root()
    return base / ecosystem / package


def index_paths(paths: list[str], *, store_root: "Path | None" = None) -> int:
    """Index every path decodable as UTF-8 text; skip the rest.

    The native segment reader hard-aborts the whole build on the first
    non-UTF-8 byte it hits -- one binary file (a stray .pdf/.docx admitted
    by the shared study-file suffix allowlist) would otherwise fail every
    other file in the same call. Returns the count actually indexed.

    Every stored path is resolved to absolute before it reaches the
    index, regardless of whether `paths` carried an absolute or a
    relative entry -- `_normalize_hit_path`/`attach_snippets` assume
    every hit is either already absolute or was rewritten repo-relative
    from an absolute path, and never fall back to the process cwd.

    STRUCTURAL secret gate: when the resolved store is the command-output
    store, every path is gated by `secret_gate.gate_paths_for_command_output`
    first -- this is the load-bearing boundary, not `index_command_output`,
    so no caller (a direct call included) can bypass it.
    """
    import iai_mcp_tools_core as _native

    root = store_root or default_store_root()
    root.mkdir(parents=True, exist_ok=True)
    if is_command_output_store_root(root):
        from iai_mcp.tools import secret_gate

        paths, _refused = secret_gate.gate_paths_for_command_output(paths)
    decodable: list[str] = []
    for p in paths:
        path = Path(p).resolve()
        try:
            path.read_bytes().decode("utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        decodable.append(str(path))
    _native.index_paths(str(root), decodable)
    return len(decodable)


def search(
    query: str, *, k: int = 10, store_root: "Path | None" = None
) -> list[tuple[str, float]]:
    import iai_mcp_tools_core as _native

    root = store_root or default_store_root()
    return _native.search(str(root), query, k)


def manifest_scope(store_root: "Path | None" = None) -> dict:
    """Live `{"n_files": int, "generation": int}` for the whole-file
    manifest at `store_root` (or the default store when `None`),
    server-computed and query-independent -- never derived from a caller's
    search string. Callers MUST pass the SAME `store_root` their search
    already resolved (e.g. `search_store_root(args)`'s result), never a
    fresh default, or the reported scope silently describes the wrong
    store.

    A caller that must pair this scope with a specific search's own hits
    (never a second, independently-timed manifest read) uses
    `search_with_scope`/`search_with_snippets_and_scope` instead -- those
    read the manifest exactly once for both. This function itself is one
    independent read; the xref/fused CLI lanes call it right after their
    own (different) search, not atomically paired with it.
    """
    import iai_mcp_tools_core as _native

    root = store_root or default_store_root()
    n_files, generation = _native.manifest_stats(str(root))
    return {"n_files": n_files, "generation": generation}


def search_with_scope(
    query: str, *, k: int = 10, store_root: "Path | None" = None
) -> "tuple[list[tuple[str, float]], dict]":
    """`search()` plus its `{"n_files", "generation"}` scope, read off the
    SAME manifest load the hits came from -- never a second, independently
    -timed `manifest_scope` call. Use this (or
    `search_with_snippets_and_scope`) whenever a response must pair a
    scope claim with these exact hits.
    """
    import iai_mcp_tools_core as _native

    root = store_root or default_store_root()
    hits, n_files, generation = _native.search_with_scope(str(root), query, k)
    return hits, {"n_files": n_files, "generation": generation}


SNIPPET_LINE_CAP = 10
SNIPPET_TOTAL_BYTE_BUDGET = 24 * 1024


def _resolve_snippet_read_path(
    hit_path: str, abs_paths: "dict[str, str] | None" = None
) -> str:
    """Resolves a hit path to the file `extract_snippet` should re-read.

    An absolute hit path (whole/symbol/xref lanes) is used as-is. A
    repo-relative hit path (fused/vector, via `_normalize_hit_path`) is
    resolved through `abs_paths` -- the original absolute path recorded
    at normalization time, from the SAME indexed corpus the hit actually
    came from. This never re-derives a root from an ambient, spawn-
    dependent value such as the process cwd: the MCP wrapper spawns this
    CLI with an inherited cwd that is not guaranteed to be, or even be
    inside, the indexed repo, so cwd-based root discovery could silently
    resolve a same-named file in a DIFFERENT git repo. A relative hit
    path with no matching `abs_paths` entry raises rather than guessing.
    """
    p = Path(hit_path)
    if p.is_absolute():
        return str(p)
    if abs_paths is not None:
        resolved = abs_paths.get(hit_path)
        if resolved is not None:
            return resolved
    raise RuntimeError(f"no absolute path recorded for normalized hit path {hit_path!r}")


def attach_snippets(
    hits: list[tuple[str, float]],
    query: str,
    *,
    line_cap: int = SNIPPET_LINE_CAP,
    total_byte_budget: int = SNIPPET_TOTAL_BYTE_BUDGET,
    abs_paths: "dict[str, str] | None" = None,
) -> list[dict]:
    """Attaches a bounded snippet to every hit, in rank order, never
    reordering or dropping a row. Per-hit content is capped at `line_cap`
    lines by `extract_snippet` itself; once the running total of attached
    snippet bytes would exceed `total_byte_budget`, every REMAINING hit
    still gets its row (path, score, line range) but an empty snippet and
    a `budget_elided_lines` count -- the budget caps snippet CONTENT
    only, never the hit list itself (the xref lane is contractually
    allowed to exceed `k`; truncating rows would silently evict its
    injected files).

    `elided_lines` and `budget_elided_lines` are two DISTINCT counts, never
    conflated: `elided_lines` is `extract_snippet`'s own count of query-
    matching lines that exist in the file but fall outside the returned
    window, always present regardless of budget state. `budget_elided_lines`
    is 0 unless this hit's snippet was withheld once the total byte budget
    was exhausted, in which case it holds the line count the withheld
    window would have shown. `snippet == "" and budget_elided_lines > 0`
    means "withheld for budget"; it does not mean "no other matches exist"
    -- check `elided_lines` for that.

    `abs_paths`, when a hit's path is repo-relative (fused/vector lanes),
    supplies the original absolute path recorded by `_normalize_hit_path`
    at fusion time -- pass through whatever `search_vector` or
    `search_fused_detailed` produced so a relative hit resolves against
    the corpus that was actually indexed, never the calling process's cwd.
    """
    import iai_mcp_tools_core as _native

    out: list[dict] = []
    used_bytes = 0
    budget_exhausted = False
    for path, score in hits:
        try:
            read_path = _resolve_snippet_read_path(path, abs_paths=abs_paths)
            line_start, line_end, snippet, elided_lines = _native.extract_snippet(
                read_path, query, line_cap
            )
        except RuntimeError:
            line_start, line_end, snippet, elided_lines = 0, 0, "", 0

        window_lines = (line_end - line_start + 1) if line_end > 0 else 0
        if not budget_exhausted:
            snippet_bytes = len(snippet.encode("utf-8"))
            if used_bytes + snippet_bytes > total_byte_budget:
                budget_exhausted = True
            else:
                used_bytes += snippet_bytes
        budget_elided_lines = 0
        if budget_exhausted:
            snippet = ""
            budget_elided_lines = window_lines

        out.append(
            {
                "path": path,
                "score": score,
                "line_start": line_start,
                "line_end": line_end,
                "snippet": snippet,
                "elided_lines": elided_lines,
                "budget_elided_lines": budget_elided_lines,
            }
        )
    return out


def search_with_snippets(
    query: str,
    *,
    k: int = 10,
    line_cap: int = SNIPPET_LINE_CAP,
    store_root: "Path | None" = None,
) -> list[dict]:
    """`search()` plus a bounded, query-time-extracted snippet per hit --
    the token-firewall shape: a where/what question is answerable from
    this response alone, with no follow-up file read. Base ranking and
    hit order are identical to `search()`'s for the same query; snippets
    attach onto that order, never reorder it.
    """
    hits = search(query, k=k, store_root=store_root)
    return attach_snippets(hits, query, line_cap=line_cap)


def search_with_snippets_and_scope(
    query: str,
    *,
    k: int = 10,
    line_cap: int = SNIPPET_LINE_CAP,
    store_root: "Path | None" = None,
) -> "tuple[list[dict], dict]":
    """`search_with_snippets()` plus scope read off that same search's own
    manifest load -- the whole-file lane's `--json` entry point, so
    `scope.generation` always matches the generation the hits were served
    from, even under a concurrent index write.
    """
    hits, scope = search_with_scope(query, k=k, store_root=store_root)
    return attach_snippets(hits, query, line_cap=line_cap), scope


def search_literal(literal: str, *, store_root: "Path | None" = None) -> tuple:
    """Exact byte-for-byte substring scan over the whole-file lane's
    currently-indexed file set -- one native call, one manifest load, no
    re-derivation in Python. Returns `(hits, files_scanned,
    files_unreadable, files_truncated, n_files, generation)`; `hits` is a
    list of `(path, line)` pairs. Unlike every tokenizing lane, an empty
    `hits` here is a PROOF the literal is absent from every file counted
    in `files_scanned` -- see `exact_scan::scan`'s doc for the truncation/
    unreadable exclusions that keep that proof honest.
    """
    import iai_mcp_tools_core as _native

    root = store_root or default_store_root()
    return _native.scan_literal(str(root), literal)


def default_symbol_store_root() -> Path:
    """Sibling subdir of the whole-file store: ~/.iai-mcp/tools/symbols/."""
    return default_store_root() / "symbols"


def index_symbols(paths: list[str], *, store_root: "Path | None" = None) -> int:
    """Scan every path decodable as UTF-8 text for symbol definitions/mentions.

    The Python-side filter is a defensive pre-check; the native symbol
    reader already skips an individual unreadable/non-UTF-8 file rather
    than aborting the batch, so this filter only avoids a redundant
    native-side read-and-reject round trip, not a batch-wide failure.
    Returns the count actually indexed. Same absolute-path resolution
    guarantee as `index_paths`.
    """
    import iai_mcp_tools_core as _native

    root = store_root or default_symbol_store_root()
    root.mkdir(parents=True, exist_ok=True)
    decodable: list[str] = []
    for p in paths:
        path = Path(p).resolve()
        try:
            path.read_bytes().decode("utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        decodable.append(str(path))
    _native.index_symbols(str(root), decodable)
    return len(decodable)


def search_symbol(
    query: str, *, k: int = 10, store_root: "Path | None" = None
) -> list[tuple[str, float]]:
    import iai_mcp_tools_core as _native

    root = store_root or default_symbol_store_root()
    return _native.search_symbol(str(root), query, k)


def symbol_manifest_scope(store_root: "Path | None" = None) -> dict:
    """Live `{"n_files": int, "generation": int}` for the symbol lane's OWN
    manifest at `store_root` (or the default symbol store when `None`) --
    independent of the whole-file lane's manifest."""
    import iai_mcp_tools_core as _native

    root = store_root or default_symbol_store_root()
    n_files, generation = _native.symbol_manifest_stats(str(root))
    return {"n_files": n_files, "generation": generation}


def search_symbol_with_scope(
    query: str, *, k: int = 10, store_root: "Path | None" = None
) -> "tuple[list[tuple[str, float]], dict]":
    """`search_symbol()` plus its own-manifest scope, read off the SAME
    manifest load the hits came from -- symbol-lane counterpart of
    `search_with_scope`."""
    import iai_mcp_tools_core as _native

    root = store_root or default_symbol_store_root()
    hits, n_files, generation = _native.search_symbol_with_scope(str(root), query, k)
    return hits, {"n_files": n_files, "generation": generation}


def default_xref_store_root() -> Path:
    """Sibling subdir of the whole-file store: ~/.iai-mcp/tools/xref/."""
    return default_store_root() / "xref"


def index_xref(paths: list[str], *, store_root: "Path | None" = None) -> int:
    """Scan every path decodable as UTF-8 text for the one recognized cross-file
    import shape and publish one new xref edge segment.

    Returns the count actually indexed (files decodable as UTF-8 text).
    Same absolute-path resolution guarantee as `index_paths`.
    """
    import iai_mcp_tools_core as _native

    root = store_root or default_xref_store_root()
    root.mkdir(parents=True, exist_ok=True)
    decodable: list[str] = []
    for p in paths:
        path = Path(p).resolve()
        try:
            path.read_bytes().decode("utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        decodable.append(str(path))
    _native.index_xref(str(root), decodable)
    return len(decodable)


def search_xref(
    query: str,
    *,
    k: int = 10,
    store_root: "Path | None" = None,
    xref_store_root: "Path | None" = None,
) -> list[tuple[str, float]]:
    """Whole-file BM25 search with referenced-definition files injected
    after every base hit. A base hit's presence and rank relative to
    other base hits are never altered by injection; the returned list
    holds `k` rows when nothing was injected, and may hold more than
    `k` when one or more files were injected. `store_root` is the
    whole-file lane's store; `xref_store_root` is the xref lane's own
    store.
    """
    import iai_mcp_tools_core as _native

    root = store_root or default_store_root()
    xref_root = xref_store_root or default_xref_store_root()
    return _native.search_xref(str(root), str(xref_root), query, k)


def importers_for_symbol(
    name: str,
    *,
    def_path: "str | None" = None,
    k: int = 200,
    xref_store_root: "Path | None" = None,
) -> "tuple[list[tuple[str, str]], list[str], bool, bool]":
    """Reverse xref lookup: every currently-indexed file with an edge
    (import or call, resolved or not) referencing `name` -- the inverse
    of `search_xref`'s forward direction. Rows are partitioned into
    resolved importers and unresolved references FIRST, then EACH half
    is independently bounded to `k` -- never bounding the combined row
    set before partitioning, which could hide a real importer behind an
    unresolved flood.

    `def_path`, when given, filters ONLY the resolved half to rows whose
    target is that file; it is matched after resolving both sides to an
    absolute path (the same normalization `index_paths`/`index_all`
    apply before a path ever reaches the native index), so a caller may
    pass a relative, `./`-prefixed, or otherwise differently-spelled but
    equivalent path and still match the indexed form. An unresolved
    reference never resolved to any file, so it is always reported at
    the whole-symbol level regardless of `def_path`.

    `k` must be a positive row-count bound (>=1); the caller-facing
    surfaces (CLI `--k`, MCP schema) reject `k<1` before this is ever
    called, but this function also fails loud on a non-positive `k`
    rather than silently applying Python's negative-index slice
    semantics to a row-count bound.

    Returns `(hits, unresolved, truncated, has_resolved)`: `hits` is a
    list of `(path, kind)` pairs (`kind` is `"import"` or `"call"`),
    `unresolved` a list of paths, `truncated` true when either half was
    cut by `k`, and `has_resolved` true iff at least one resolved
    importer exists BEFORE the `k` bound is applied. Callers MUST derive
    "no importers" status from `has_resolved`, never from `hits` being
    empty -- `hits` reflects the `k`-bounded page, `has_resolved`
    reflects ground truth.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    import iai_mcp_tools_core as _native

    xref_root = xref_store_root or default_xref_store_root()
    rows = _native.definitions_for_reference(str(xref_root), name)

    def_path_normalized = str(Path(def_path).resolve()) if def_path is not None else None
    resolved_all = [
        (from_path, kind)
        for (from_path, to_path, kind) in rows
        if to_path is not None and (def_path_normalized is None or to_path == def_path_normalized)
    ]
    unresolved_all = [from_path for (from_path, to_path, _kind) in rows if to_path is None]

    hits = resolved_all[:k]
    unresolved = unresolved_all[:k]
    truncated = len(resolved_all) > k or len(unresolved_all) > k
    return hits, unresolved, truncated, bool(resolved_all)


def default_ast_store_root() -> Path:
    """Sibling subdir of the whole-file store: ~/.iai-mcp/tools/ast/."""
    return default_store_root() / "ast"


_PYTHON_SUFFIX = ".py"


def goto_definition(
    name: str,
    *,
    from_path: str,
    ast_store_root: "Path | None" = None,
) -> dict:
    """Go-to-definition over the AST lane's own facts, read fresh from
    disk. Returns `{query, lane, from_path, hits, status}`; `hits` holds
    0 or 1 `{path, line}` entries. `status` is one of `resolved` (`hits`
    has one entry), `unresolved` (indexed and parsed but no definite
    target found for `name`), `unsupported_language` (`from_path` is not
    a `.py` file), or `parse_error` (`from_path` recorded as
    parse-failed). Never a silent empty `hits` with no explanation.

    `unsupported_language` is checked here, before the native query runs.
    """
    if not Path(from_path).suffix == _PYTHON_SUFFIX:
        return {
            "query": name,
            "lane": "goto-def",
            "from_path": from_path,
            "hits": [],
            "status": "unsupported_language",
        }

    import iai_mcp_tools_core as _native

    root = ast_store_root or default_ast_store_root()
    from_path_resolved = str(Path(from_path).resolve())
    status, def_path, line = _native.goto_definition(str(root), name, from_path_resolved)

    hits = [{"path": def_path, "line": line}] if status == "resolved" else []
    return {
        "query": name,
        "lane": "goto-def",
        "from_path": from_path,
        "hits": hits,
        "status": status,
    }


def find_references(
    name: str,
    *,
    def_path: str,
    k: int = 200,
    ast_store_root: "Path | None" = None,
) -> dict:
    """Find-references over the AST lane's own edges -- the reverse of
    `goto_definition`. Returns `{query, lane, def_path, hits, status}`;
    `hits` holds `{path, status}` entries, each `status` one of
    `resolved`/`unresolved`. The response-level `status` is one of
    `hits` (at least one row), `no_references` (indexed and parsed but
    zero rows), `unsupported_language` (`def_path` is not a `.py` file),
    or `parse_error` (`def_path` recorded as parse-failed). Never a
    silent empty `hits` with no explanation.

    Rows are partitioned by row status first, then each half is
    independently bounded to `k`, so an unresolved flood never hides a
    real resolved reference behind the row cap.

    `unsupported_language` is checked here, before the native query runs.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    if not Path(def_path).suffix == _PYTHON_SUFFIX:
        return {
            "query": name,
            "lane": "references",
            "def_path": def_path,
            "hits": [],
            "status": "unsupported_language",
        }

    import iai_mcp_tools_core as _native

    root = ast_store_root or default_ast_store_root()
    def_path_resolved = str(Path(def_path).resolve())
    status, rows = _native.find_references(str(root), name, def_path_resolved)

    if status != "hits":
        return {
            "query": name,
            "lane": "references",
            "def_path": def_path,
            "hits": [],
            "status": status,
        }

    resolved_all = [(path, row_status) for (path, row_status) in rows if row_status == "resolved"]
    unresolved_all = [(path, row_status) for (path, row_status) in rows if row_status == "unresolved"]
    truncated = len(resolved_all) > k or len(unresolved_all) > k

    hits = [
        {"path": path, "status": row_status}
        for (path, row_status) in resolved_all[:k] + unresolved_all[:k]
    ]

    payload = {
        "query": name,
        "lane": "references",
        "def_path": def_path,
        "hits": hits,
        "status": "hits",
    }
    if truncated:
        payload["truncated"] = True
    return payload


def index_both(
    paths: list[str],
    *,
    store_root: "Path | None" = None,
    symbol_store_root: "Path | None" = None,
) -> int:
    """Index `paths` into both lanes in the fixed lockstep write order.

    The whole-file lane publishes first, then the symbol lane -- each its
    own atomic rename, enforced inside the native extension rather than
    relying on caller call order. A crash between the two publishes
    leaves the symbol lane at most one batch behind the whole-file lane;
    a subsequent call converges both. Returns the count actually indexed
    (files decodable as UTF-8 text). Same absolute-path resolution
    guarantee as `index_paths`.
    """
    import iai_mcp_tools_core as _native

    root = store_root or default_store_root()
    symbol_root = symbol_store_root or default_symbol_store_root()
    root.mkdir(parents=True, exist_ok=True)
    symbol_root.mkdir(parents=True, exist_ok=True)
    decodable: list[str] = []
    for p in paths:
        path = Path(p).resolve()
        try:
            path.read_bytes().decode("utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        decodable.append(str(path))
    _native.index_both(str(root), str(symbol_root), decodable)
    return len(decodable)


def default_vector_store_root() -> Path:
    """Sibling subdir of the whole-file store: ~/.iai-mcp/tools/vectors/."""
    return default_store_root() / "vectors"


def _find_repo_root(start: Path) -> "Path | None":
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _normalize_hit_path(path: str, *, abs_paths: "dict[str, str] | None" = None) -> str:
    """Canonical repo-relative form for any lane's hit path -- the fusion
    join key later lanes reuse. A relative path is returned normalized
    as-is; an absolute path under a discoverable repo root (walking up
    for `.git`) is rewritten relative to that root, otherwise left as-is.

    When `abs_paths` is given, every absolute-to-relative rewrite records
    its original absolute form there, keyed by the returned relative
    string -- the sole channel `attach_snippets` uses to re-read a
    normalized hit's file without re-deriving a root from process cwd.
    """
    p = Path(path)
    if not p.is_absolute():
        return str(p)
    root = _find_repo_root(p)
    if root is None:
        return str(p)
    try:
        rel = str(p.relative_to(root))
    except ValueError:
        return str(p)
    if abs_paths is not None:
        abs_paths[rel] = str(p)
    return rel


def search_vector(
    query_vector: list[float],
    k: int = 10,
    *,
    store_root: "Path | None" = None,
    abs_paths: "dict[str, str] | None" = None,
) -> list[tuple[str, float]]:
    """Cosine-ranked hits over the tools-owned exact-cosine vector store.
    `query_vector` must arrive already embedded -- this function never
    imports the brain embedder. A missing store returns an empty result;
    an existing store built at a different embedding dimension raises
    `RuntimeError` instead of collapsing to the same empty result.

    `abs_paths`, when given, is populated with each normalized hit's
    original absolute path -- pass it on to `attach_snippets` so a
    repo-relative hit resolves against the corpus that was actually
    indexed, not the calling process's cwd.
    """
    import iai_mcp_tools_core as _native

    root = store_root or default_vector_store_root()
    hits = _native.search_vector(str(root), query_vector, k)
    return [(_normalize_hit_path(path, abs_paths=abs_paths), score) for path, score in hits]


def _dedup_preserve_order(paths: list[str]) -> list[str]:
    seen: "set[str]" = set()
    out: "list[str]" = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _xref_derived_lane(base_lane: "list[str]", xref_all: "list[str]") -> "list[str]":
    """The xref lane's rank list is the referenced-definition rows
    `search_xref` adds BEYOND the plain whole-file result -- the set
    difference against `base_lane`, preserving injected order. Feeding
    `search_xref`'s full output verbatim would make this lane nearly
    identical to the whole-file lane and double-weight the lexical signal.
    """
    base_set = set(base_lane)
    seen: "set[str]" = set()
    derived: "list[str]" = []
    for p in xref_all:
        if p in base_set or p in seen:
            continue
        seen.add(p)
        derived.append(p)
    return derived


EMBED_WORKER_QUERY_TIMEOUT_SECONDS = 120


class EmbedWorkerError(RuntimeError):
    """Raised when the detached `embed_worker` query subprocess fails,
    times out, or produces output this process cannot parse as the
    expected JSON query vector -- the one error type every caller of
    `_embed_query_via_subprocess` can catch and report cleanly."""


def _embed_query_via_subprocess(text: str) -> "list[float]":
    """Embeds `text` by spawning `embed_worker` as a subprocess -- the
    gated semantic lane's only path to the brain embedder. Never imports
    `iai_mcp_native` in this process. The sole shared spawn-and-parse
    helper for every `search-vector`/`search-fused` call site; raises
    `EmbedWorkerError` on a timeout, non-zero exit, or unparseable output
    instead of letting each caller hand-roll its own fragile parsing.
    """
    import json
    import subprocess
    import sys

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "iai_mcp.tools.embed_worker", "query", text],
            capture_output=True,
            text=True,
            timeout=EMBED_WORKER_QUERY_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as e:
        raise EmbedWorkerError(
            f"embed worker query timed out after {EMBED_WORKER_QUERY_TIMEOUT_SECONDS}s"
        ) from e

    if proc.returncode != 0:
        raise EmbedWorkerError(proc.stderr.strip() or "embed worker failed")

    lines = [line for line in proc.stdout.strip().splitlines() if line.strip()]
    if not lines:
        raise EmbedWorkerError("embed worker produced no output")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as e:
        raise EmbedWorkerError(f"embed worker output is not valid JSON: {e}") from e


def search_fused_detailed(
    query: str,
    k: int = 10,
    weak_threshold: "float | None" = None,
    *,
    store_root: "Path | None" = None,
    symbol_store_root: "Path | None" = None,
    xref_store_root: "Path | None" = None,
    vector_store_root: "Path | None" = None,
) -> dict:
    """RRF-fuses (K=60) three base lanes -- whole-file, symbol, and the
    xref-derived lane -- through the `_normalize_hit_path` choke point,
    then adds the gated semantic lane ONLY when the base fused top
    result is weak. `weak_threshold` overrides the gate's default; the
    embed subprocess never runs when the gate does not fire.

    Returns `{"hits": list[tuple[str, float]], "vector_lane": ..., "reason":
    ..., "abs_paths": dict[str, str]}`. `vector_lane` is one of three
    DISTINCT states a boolean cannot tell apart: `"not_attempted"` (the
    gate never fired -- the vector lane was never tried, by design),
    `"ok"` (the gate fired and the vector lane returned), or `"degraded"`
    (the gate fired but the vector lane faulted), with `reason` set to
    `"index_dimension_mismatch"` (a stale vector store after an embedder
    change) or `"embed_unavailable"` (the query-embed subprocess itself
    failed). A degraded vector lane NEVER takes down the whole fused
    query -- the gate fires precisely when base results are weakest, so
    this always falls back to base-lanes-only hits instead of
    propagating. `abs_paths` maps every repo-relative hit produced by
    this call to its original absolute path -- pass it to
    `attach_snippets` so a caller resolves snippet content against the
    corpus actually indexed, not the calling process's cwd.
    """
    import iai_mcp_tools_core as _native

    root = store_root or default_store_root()
    symbol_root = symbol_store_root or default_symbol_store_root()
    xref_root = xref_store_root or default_xref_store_root()
    vector_root = vector_store_root or default_vector_store_root()

    whole_hits = _native.search(str(root), query, k)
    symbol_hits = _native.search_symbol(str(symbol_root), query, k)
    xref_hits = _native.search_xref(str(root), str(xref_root), query, k)

    abs_paths: "dict[str, str]" = {}
    whole_lane = _dedup_preserve_order(
        [_normalize_hit_path(p, abs_paths=abs_paths) for p, _ in whole_hits]
    )
    symbol_lane = _dedup_preserve_order(
        [_normalize_hit_path(p, abs_paths=abs_paths) for p, _ in symbol_hits]
    )
    xref_all = [_normalize_hit_path(p, abs_paths=abs_paths) for p, _ in xref_hits]
    xref_lane = _xref_derived_lane(whole_lane, xref_all)

    base_lanes = [whole_lane, symbol_lane, xref_lane]
    base_fused = _native.rrf_merge(base_lanes)

    if not _native.gate_is_weak(base_fused, weak_threshold):
        return {
            "hits": base_fused[:k],
            "vector_lane": "not_attempted",
            "reason": None,
            "abs_paths": abs_paths,
        }

    vector_lane_state = "ok"
    reason: "str | None" = None
    vector_hits: "list[tuple[str, float]]" = []
    try:
        query_vector = _embed_query_via_subprocess(query)
        vector_hits = search_vector(query_vector, k=k, store_root=vector_root, abs_paths=abs_paths)
    except RuntimeError as e:
        import sys

        vector_lane_state = "degraded"
        reason = "embed_unavailable" if isinstance(e, EmbedWorkerError) else "index_dimension_mismatch"
        print(
            f"search_fused: vector lane unavailable, continuing without it: {e}",
            file=sys.stderr,
        )
        vector_hits = []

    vector_lane = _dedup_preserve_order([p for p, _ in vector_hits])
    fused = _native.rrf_merge([*base_lanes, vector_lane])
    return {
        "hits": fused[:k],
        "vector_lane": vector_lane_state,
        "reason": reason,
        "abs_paths": abs_paths,
    }


def search_fused(
    query: str,
    k: int = 10,
    weak_threshold: "float | None" = None,
    *,
    store_root: "Path | None" = None,
    symbol_store_root: "Path | None" = None,
    xref_store_root: "Path | None" = None,
    vector_store_root: "Path | None" = None,
) -> list[tuple[str, float]]:
    """RRF-fuses (K=60) three base lanes -- whole-file, symbol, and the
    xref-derived lane -- then adds the gated semantic lane ONLY when the
    base fused top result is weak. This is an explicit, additional search
    surface -- not wired as any command's default. Delegates to
    `search_fused_detailed` and returns just its hit list, unchanged
    contract for every existing caller; see that function for the
    vector-lane degradation states this call discards.
    """
    result = search_fused_detailed(
        query,
        k,
        weak_threshold,
        store_root=store_root,
        symbol_store_root=symbol_store_root,
        xref_store_root=xref_store_root,
        vector_store_root=vector_store_root,
    )
    return result["hits"][:k]


def index_all(
    paths: list[str],
    *,
    store_root: "Path | None" = None,
    symbol_store_root: "Path | None" = None,
    xref_store_root: "Path | None" = None,
    ast_store_root: "Path | None" = None,
) -> int:
    """Index `paths` into all three Rust lanes in the fixed lockstep write
    order, then run the AST lane as a 4th step.

    Extends `index_both`'s posture by one more Rust publish: whole-file,
    then symbol, then xref -- each its own atomic rename, enforced inside
    the native extension. A crash between publishes leaves a later lane
    at most one batch behind; a subsequent call converges all three.

    The AST step runs AFTER the Rust `index_all` call returns, since it
    needs Python to run `ast.parse` first -- it cannot be folded into
    that same Rust call. It is scoped to `.py` files among `decodable`
    and is SKIPPED entirely when the resolved store is the command-output
    store, the same structural boundary already applied to the three
    Rust lanes above -- command output must never be AST-parsed into the
    source AST store. Before parsing, the `.py` subset is narrowed to
    `_native.stale_paths`' result -- a file whose content hash already
    matches the AST manifest's own `file_index` never reaches `ast.parse`
    at all, so a re-index after one file changes pays the parse cost for
    that one file only; a fresh/cold AST store has no prior entries, so
    every `.py` file comes back stale and the first index parses all of
    them. A single file whose `ast.parse` fails, or whose re-read fails
    (deleted/permission-denied/truncated between the earlier decodability
    check and here), degrades to a zero-fact, `parse_ok=False` entry and
    never aborts the batch; a native publish failure is likewise caught
    and skipped for this run rather than propagated.

    Returns the count actually indexed (files decodable as UTF-8 text).
    Same absolute-path resolution guarantee as `index_paths`.

    STRUCTURAL secret gate: same store-boundary check as `index_paths`,
    applied to the whole-file root before any lane is written.
    """
    import json
    import sys

    import iai_mcp_tools_core as _native

    root = store_root or default_store_root()
    symbol_root = symbol_store_root or default_symbol_store_root()
    xref_root = xref_store_root or default_xref_store_root()
    ast_root = ast_store_root or default_ast_store_root()
    root.mkdir(parents=True, exist_ok=True)
    symbol_root.mkdir(parents=True, exist_ok=True)
    xref_root.mkdir(parents=True, exist_ok=True)
    if is_command_output_store_root(root):
        from iai_mcp.tools import secret_gate

        paths, _refused = secret_gate.gate_paths_for_command_output(paths)
    decodable: list[str] = []
    for p in paths:
        path = Path(p).resolve()
        try:
            path.read_bytes().decode("utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        decodable.append(str(path))
    _native.index_all(str(root), str(symbol_root), str(xref_root), decodable)

    if not is_command_output_store_root(root):
        from iai_mcp.tools import ast_facts

        py_files = [p for p in decodable if p.endswith(".py")]
        if py_files:
            ast_root.mkdir(parents=True, exist_ok=True)
            # A cold/fresh AST store has an empty manifest file_index, so
            # every path comes back stale here -- the cold path always
            # parses everything, matching a from-scratch index. On any
            # later call, only a path whose content hash actually changed
            # ever reaches `ast.parse` below.
            stale_py_files = _native.stale_paths(str(ast_root), py_files)
            if stale_py_files:
                files_facts: list[dict] = []
                for p in stale_py_files:
                    try:
                        source = Path(p).read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError) as err:
                        print(
                            f"iai_mcp: could not re-read {p} for AST facts ({err}); "
                            "recording it as parse-failed for this batch",
                            file=sys.stderr,
                        )
                        files_facts.append(ast_facts.unreadable_fact(p))
                        continue
                    files_facts.append(ast_facts.extract_facts(p, source))
                try:
                    _native.index_ast_facts(str(ast_root), json.dumps({"files": files_facts}))
                except Exception as err:
                    print(
                        f"iai_mcp: AST lane publish failed for this batch ({err}); "
                        "skipped -- a later index_all call converges",
                        file=sys.stderr,
                    )

    return len(decodable)
