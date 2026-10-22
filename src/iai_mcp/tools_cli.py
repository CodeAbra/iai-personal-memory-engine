"""`iai-tools` console script: the fast, brain-independent code-index CLI.

Must never import iai_mcp.cli, iai_mcp.hippo, iai_mcp.store, iai_mcp.capture,
iai_mcp.lilli, or iai_mcp_native -- the whole point of this script is a
search path that answers without loading the memory system. Only stdlib,
iai_mcp.study (file discovery, used by `index`/`embed-index`), iai_mcp.tools
(which itself only reaches iai_mcp_tools_core), iai_mcp.tools_exec_client
(a thin socket client to the executor daemon, used by `embed-index`), and
iai_mcp.tools.exec_cli (the shared `run`/`fetch`/`batch-run` command body,
also used by the `iai-mcp tools` operator CLI so the two surfaces cannot
drift) are allowed here. `search-vector` and
`search-fused` SPAWN `iai_mcp.tools.embed_worker` as a subprocess to embed
query text -- neither imports that module in-process, so iai_mcp_native
never loads into this script's own process; `search-fused` only spawns it
when its gate reports the base lanes' top result is weak. `embed-index`
submits the SAME worker module as a detached background run through the executor daemon --
this script hands the daemon a command line, it never imports
iai_mcp_native or embed_worker itself.
"""

from __future__ import annotations

import argparse
import json
import sys


def _cmd_index(args: argparse.Namespace) -> int:
    from iai_mcp.tools.dependency import (
        DependencyRequestError,
        describe_index_outcome,
        index_dependency_gated,
        validate_index_target,
    )

    ecosystem = getattr(args, "ecosystem", None)
    package = getattr(args, "package", None)
    project_dir = getattr(args, "project_dir", None)

    try:
        validate_index_target(
            path=args.path, ecosystem=ecosystem, package=package, project_dir=project_dir
        )
    except DependencyRequestError as e:
        print(str(e), file=sys.stderr)
        return 1

    if args.path is not None:
        from iai_mcp.study import TOOLS_INDEX_DOT_DIR_ALLOWLIST, iter_study_files
        from iai_mcp.tools import index_all

        files = [str(f) for f in iter_study_files(
            args.path, max_files=sys.maxsize, dot_dir_allowlist=TOOLS_INDEX_DOT_DIR_ALLOWLIST,
        )]
        indexed = index_all(files)
        print(f"indexed {indexed} files under {args.path}")
        return 0

    report = index_dependency_gated(ecosystem, package, project_dir)
    message, code = describe_index_outcome(ecosystem, package, report)
    print(message, file=sys.stderr if code else sys.stdout)
    return code


def _cmd_search(args: argparse.Namespace) -> int:
    from iai_mcp.tools.exec_cli import search_store_root

    try:
        store_root = search_store_root(args)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    if getattr(args, "store", "code") == "dependency":
        from iai_mcp.tools.dependency import is_dependency_store_indexed

        if not is_dependency_store_indexed(store_root):
            if getattr(args, "json", False):
                print(
                    json.dumps(
                        {"query": args.query, "lane": "whole", "hits": [], "status": "not_indexed"}
                    )
                )
            else:
                print("not indexed")
            return 0

    if getattr(args, "json", False):
        from iai_mcp.tools import search_with_snippets_and_scope

        # scope is read off the SAME manifest load the hits came from
        # (search_with_snippets_and_scope), never a second, independently
        # -timed read.
        hits, scope = search_with_snippets_and_scope(args.query, k=args.k, store_root=store_root)
        print(
            json.dumps(
                {
                    "query": args.query,
                    "lane": "whole",
                    "hits": hits,
                    "scope": scope,
                    "status": "unknown" if not hits else "hits",
                }
            )
        )
        return 0

    from iai_mcp.tools import search

    hits = search(args.query, k=args.k, store_root=store_root)
    if not hits:
        print("no hits")
        return 0
    for path, score in hits:
        print(f"{score:.4f}\t{path}")
    return 0


def _cmd_search_symbol(args: argparse.Namespace) -> int:
    if getattr(args, "json", False):
        from iai_mcp.tools import attach_snippets, search_symbol_with_scope

        # scope is read off the SAME manifest load the hits came from,
        # same single-load discipline as the whole-file lane above.
        hits, scope = search_symbol_with_scope(args.query, k=args.k)
        hits_out = attach_snippets(hits, args.query)
        print(
            json.dumps(
                {
                    "query": args.query,
                    "lane": "symbol",
                    "hits": hits_out,
                    "scope": scope,
                    "status": "unknown" if not hits_out else "hits",
                }
            )
        )
        return 0

    from iai_mcp.tools import search_symbol

    hits = search_symbol(args.query, k=args.k)
    if not hits:
        print("no hits")
        return 0
    for path, score in hits:
        print(f"{score:.4f}\t{path}")
    return 0


def _cmd_search_xref(args: argparse.Namespace) -> int:
    from iai_mcp.tools import search_xref

    hits = search_xref(args.query, k=args.k)
    if getattr(args, "json", False):
        from iai_mcp.tools import attach_snippets, manifest_scope

        hits_out = attach_snippets(hits, args.query)
        print(
            json.dumps(
                {
                    "query": args.query,
                    "lane": "xref",
                    "hits": hits_out,
                    # The xref lane has no generation of its own -- it always
                    # rides on a whole-file search, so it inherits the
                    # whole-file lane's N/G rather than inventing a new field.
                    # This is a separate manifest read from search_xref's own,
                    # not guaranteed atomic with it under a concurrent write.
                    "scope": manifest_scope(None),
                    "status": "unknown" if not hits_out else "hits",
                }
            )
        )
        return 0
    if not hits:
        print("no hits")
        return 0
    for path, score in hits:
        print(f"{score:.4f}\t{path}")
    return 0


def _importers_absence_claim_text(n_unresolved: int, n_files: int, generation: int) -> str:
    clause = f" ({n_unresolved} unresolved)" if n_unresolved > 0 else ""
    return f"0 resolved importers{clause} over {n_files} xref-indexed files as of generation {generation}"


def _cmd_search_importers(args: argparse.Namespace) -> int:
    from iai_mcp.tools import importers_for_symbol, manifest_scope

    if not args.query:
        # An empty symbol name has no meaningful importer/caller answer --
        # reject before the lookup rather than return a vacuous zero-row
        # response (mirrors _cmd_search_literal's empty-needle rejection).
        print("search-importers: symbol must not be empty", file=sys.stderr)
        return 1

    if args.k < 1:
        # A non-positive row-count bound has no sane "top k" reading --
        # reject before the lookup rather than let a k=0 (or Python's
        # negative-index slice semantics for k<0) silently empty `hits`
        # for a symbol that genuinely has resolved importers.
        print("search-importers: --k must be >= 1", file=sys.stderr)
        return 1

    hits, unresolved, truncated, has_resolved = importers_for_symbol(args.query, def_path=args.def_path, k=args.k)

    if getattr(args, "json", False):
        payload = {
            "query": args.query,
            "lane": "importers",
            "hits": [{"path": path, "kind": kind} for path, kind in hits],
            # The reverse lookup has no generation of its own -- it
            # inherits the whole-file lane's N/G, same as the xref lane.
            "scope": manifest_scope(None),
            # Derived from has_resolved (the PRE-k-bound partition), never
            # from `hits` -- a small --k must never turn a real importer
            # into a false "no_importers" claim.
            "status": "hits" if has_resolved else "no_importers",
        }
        if args.def_path is not None:
            payload["def_path"] = args.def_path
        if unresolved:
            payload["unresolved"] = [{"path": path} for path in unresolved]
        if truncated:
            payload["truncated"] = True
        print(json.dumps(payload))
        return 0

    if not hits:
        scope = manifest_scope(None)
        print(_importers_absence_claim_text(len(unresolved), scope["n_files"], scope["generation"]))
        return 0
    for path, kind in hits:
        print(f"{kind}\t{path}")
    return 0


def _cmd_search_goto_def(args: argparse.Namespace) -> int:
    from iai_mcp.tools import goto_definition

    if not args.query:
        # An empty symbol name has no meaningful definition to find --
        # reject before the lookup rather than return a vacuous
        # unresolved response (mirrors _cmd_search_importers's empty-
        # symbol rejection).
        print("search-goto-def: symbol must not be empty", file=sys.stderr)
        return 1

    payload = goto_definition(args.query, from_path=args.from_path)

    if getattr(args, "json", False):
        print(json.dumps(payload))
        return 0

    if not payload["hits"]:
        print(f"0 hits (status: {payload['status']})")
        return 0
    for hit in payload["hits"]:
        print(f"{hit['line']}\t{hit['path']}")
    return 0


def _cmd_search_references(args: argparse.Namespace) -> int:
    from iai_mcp.tools import find_references

    if not args.query:
        # An empty symbol name has no meaningful reference set to find --
        # reject before the lookup rather than return a vacuous
        # no_references response (mirrors _cmd_search_importers's empty-
        # symbol rejection).
        print("search-references: symbol must not be empty", file=sys.stderr)
        return 1

    if args.k < 1:
        print("search-references: --k must be >= 1", file=sys.stderr)
        return 1

    payload = find_references(args.query, def_path=args.def_path, k=args.k)

    if getattr(args, "json", False):
        print(json.dumps(payload))
        return 0

    if not payload["hits"]:
        print(f"0 hits (status: {payload['status']})")
        return 0
    for hit in payload["hits"]:
        print(f"{hit['status']}\t{hit['path']}")
    return 0


def _absence_claim_text(files_scanned: int, n_files: int, files_unreadable: int, files_truncated: int, generation: int) -> str:
    clauses = [f"of {n_files} indexed"]
    if files_unreadable > 0:
        clauses.append(f"{files_unreadable} unreadable")
    if files_truncated > 0:
        clauses.append(f"{files_truncated} truncated")
    return f"0 of {files_scanned} scanned ({'; '.join(clauses)}) as of generation {generation}"


def _cmd_search_literal(args: argparse.Namespace) -> int:
    from iai_mcp.tools import search_literal
    from iai_mcp.tools.exec_cli import search_store_root

    if not args.query:
        # An empty needle proves nothing -- reject before the scan loop
        # rather than return a zero-bucket response indistinguishable from
        # "scanned every file, found nothing" (mirrors the MCP surface's
        # existing empty-query rejection).
        print("search-literal: literal must not be empty", file=sys.stderr)
        return 1

    store_root = search_store_root(args)
    hits, files_scanned, files_unreadable, files_truncated, n_files, generation = search_literal(
        args.query, store_root=store_root
    )

    if getattr(args, "json", False):
        payload = {
            "query": args.query,
            "lane": "literal",
            "hits": [{"path": path, "line": line} for path, line in hits],
            "scope": {"n_files": n_files, "generation": generation},
            "files_scanned": files_scanned,
            "status": "hits" if hits else "absent",
        }
        if files_unreadable > 0:
            payload["files_unreadable"] = files_unreadable
        if files_truncated > 0:
            payload["files_truncated"] = files_truncated
        print(json.dumps(payload))
        return 0

    if not hits:
        print(_absence_claim_text(files_scanned, n_files, files_unreadable, files_truncated, generation))
        return 0
    for path, line in hits:
        print(f"{line}\t{path}")
    return 0


def _cmd_search_vector(args: argparse.Namespace) -> int:
    from iai_mcp.tools import EmbedWorkerError, _embed_query_via_subprocess, search_vector

    try:
        query_vector = _embed_query_via_subprocess(args.query)
    except EmbedWorkerError as e:
        print(str(e), file=sys.stderr)
        return 1

    if getattr(args, "json", False):
        from iai_mcp.tools import attach_snippets

        abs_paths: dict[str, str] = {}
        hits = search_vector(query_vector, k=args.k, abs_paths=abs_paths)
        hits_out = attach_snippets(hits, args.query, abs_paths=abs_paths)
        print(
            json.dumps(
                {
                    "query": args.query,
                    "lane": "vector",
                    "hits": hits_out,
                    # No scope: the vector store has no manifest/generation.
                    # status is still required -- an empty cosine result
                    # cannot prove byte-absence any more than a BM25 zero can.
                    "status": "unknown" if not hits_out else "hits",
                }
            )
        )
        return 0

    hits = search_vector(query_vector, k=args.k)
    if not hits:
        print("no hits")
        return 0
    for path, score in hits:
        print(f"{score:.4f}\t{path}")
    return 0


def _cmd_search_fused(args: argparse.Namespace) -> int:
    from iai_mcp.tools import search_fused, search_fused_detailed

    if getattr(args, "json", False):
        from iai_mcp.tools import attach_snippets, manifest_scope

        detailed = search_fused_detailed(args.query, k=args.k, weak_threshold=args.weak_threshold)
        hits_out = attach_snippets(detailed["hits"], args.query, abs_paths=detailed["abs_paths"])
        print(
            json.dumps(
                {
                    "query": args.query,
                    "lane": "fused",
                    "hits": hits_out,
                    # The whole-file lane is the only byte-bearing corpus
                    # among fused's inputs (symbol/xref/vector are derived),
                    # so its N/G is the meaningful scanned scope here -- no
                    # combined cross-lane generation is invented. Separate
                    # manifest read from search_fused_detailed's own, not
                    # guaranteed atomic with it under a concurrent write.
                    "scope": manifest_scope(None),
                    "status": "unknown" if not hits_out else "hits",
                    "vector_lane": detailed["vector_lane"],
                    "reason": detailed["reason"],
                }
            )
        )
        return 0

    hits = search_fused(args.query, k=args.k, weak_threshold=args.weak_threshold)
    if not hits:
        print("no hits")
        return 0
    for path, score in hits:
        print(f"{score:.4f}\t{path}")
    return 0


def _cmd_embed_index(args: argparse.Namespace) -> int:
    from iai_mcp import tools_exec_client
    from iai_mcp.study import iter_study_files
    from iai_mcp.tools import default_vector_store_root

    files = [str(f) for f in iter_study_files(args.path, max_files=sys.maxsize)]
    store_root = str(default_vector_store_root())
    worker_args = ["-m", "iai_mcp.tools.embed_worker", "index", store_root, *files]

    result = tools_exec_client.run(sys.executable, worker_args, background=not args.foreground)
    tail = result["bounded_tail"]
    sys.stdout.write(tail.get("stdout", ""))
    sys.stderr.write(tail.get("stderr", ""))
    print(f"run_id: {result['run_id']} state: {result['state']}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from iai_mcp.tools.exec_cli import cmd_run

    return cmd_run(args)


def _cmd_fetch(args: argparse.Namespace) -> int:
    from iai_mcp.tools.exec_cli import cmd_fetch

    return cmd_fetch(args)


def _cmd_batch_run(args: argparse.Namespace) -> int:
    from iai_mcp.tools.exec_cli import cmd_batch_run

    return cmd_batch_run(args)


_CONTENT_TOOLS = ("TodoWrite", "Task", "Skill")


def _translate_hook_payload(host: dict, source_hook: "str | None") -> dict:
    """Translates a raw host hook payload (nested `tool_input`/`tool_response`,
    or a flat prompt-submit shape) into the classifier's flat schema. Total
    and defensive: a missing or oddly-typed field maps to an unset flat
    field, never an exception.

    `is_error` is fed ONLY from the host's structural `tool_response.interrupted`
    flag -- never from a substring match on response text, which a clean
    command's own output can trip incidentally.
    """

    def _text(value: object) -> "str | None":
        return value if isinstance(value, str) and value else None

    tool_input = host.get("tool_input")
    if not isinstance(tool_input, dict):
        tool_input = {}
    tool_response = host.get("tool_response")
    if not isinstance(tool_response, dict):
        tool_response = {}

    flat: dict = {}
    if source_hook:
        flat["source_hook"] = source_hook

    session_id = _text(host.get("session_id"))
    if session_id:
        flat["session_id"] = session_id

    prompt = _text(host.get("prompt"))
    if prompt:
        flat["prompt"] = prompt

    if tool_response.get("interrupted") is True:
        flat["is_error"] = True

    tool_name = _text(host.get("tool_name"))
    if tool_name:
        flat["tool_name"] = tool_name

    file_path = _text(tool_input.get("file_path"))
    if file_path:
        flat["file_path"] = file_path

    command = _text(tool_input.get("command"))
    if command:
        flat["command"] = command

    if tool_name in _CONTENT_TOOLS and tool_input:
        try:
            flat["content"] = json.dumps(tool_input, sort_keys=True)
        except (TypeError, ValueError):
            pass

    return flat


def _cmd_capture(args: argparse.Namespace) -> int:
    from iai_mcp.tools.session import capture_event

    try:
        raw = sys.stdin.read()
    except Exception:
        return 0
    if not raw or not raw.strip():
        return 0
    try:
        host = json.loads(raw)
    except (ValueError, TypeError):
        return 0
    if not isinstance(host, dict):
        return 0

    try:
        flat = _translate_hook_payload(host, getattr(args, "source_hook", None))
        capture_event(flat)
    except Exception:
        return 0
    return 0


def _cmd_session_search(args: argparse.Namespace) -> int:
    also_sessions = getattr(args, "also_session", None) or []
    if also_sessions:
        from iai_mcp.tools.session import query_events_multi

        events = query_events_multi(
            [args.session_id, *also_sessions],
            substring=args.substring,
            category=args.category,
            since=args.since,
            until=args.until,
        )
    else:
        from iai_mcp.tools.session import query_events

        events = query_events(
            args.session_id,
            substring=args.substring,
            category=args.category,
            since=args.since,
            until=args.until,
        )
    if not events:
        print("no events")
        return 0
    for event in events:
        print(json.dumps(event))
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    from iai_mcp.tools.stats import gather_stats

    print(json.dumps(gather_stats()))
    return 0


def _cmd_command_insight(args: argparse.Namespace) -> int:
    from iai_mcp.tools.stats import gather_command_insight

    print(json.dumps(gather_command_insight()))
    return 0


def _cmd_git_log(args: argparse.Namespace) -> int:
    from iai_mcp.tools.git_query import cmd_git_log

    return cmd_git_log(args)


def _cmd_git_blame(args: argparse.Namespace) -> int:
    from iai_mcp.tools.git_query import cmd_git_blame

    return cmd_git_blame(args)


def _cmd_git_diff(args: argparse.Namespace) -> int:
    from iai_mcp.tools.git_query import cmd_git_diff

    return cmd_git_diff(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="iai-tools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser(
        "index",
        help="walk a directory and build one code-index segment from it, or index one "
        "resolved dependency package with --ecosystem/--package/--project-dir",
    )
    p_index.add_argument(
        "path",
        nargs="?",
        default=None,
        help="directory to index (mutually exclusive with --ecosystem/--package/--project-dir)",
    )
    p_index.add_argument(
        "--ecosystem",
        default=None,
        help="dependency ecosystem, e.g. python (requires --package and --project-dir)",
    )
    p_index.add_argument(
        "--package",
        default=None,
        help="dependency package name (requires --ecosystem and --project-dir)",
    )
    p_index.add_argument(
        "--project-dir",
        default=None,
        dest="project_dir",
        help="project directory whose environment resolves the package "
        "(requires --ecosystem and --package)",
    )
    p_index.set_defaults(func=_cmd_index)

    p_search = sub.add_parser(
        "search", help="query the code-index store, ranked by whole-file BM25"
    )
    p_search.add_argument("query", help="search query")
    p_search.add_argument(
        "--k", type=int, default=10, help="max ranked hits (default 10)"
    )
    p_search.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON blob with per-hit line range + bounded snippet text, instead of plain text",
    )
    p_search.add_argument(
        "--store",
        choices=["code", "output", "dependency"],
        default="code",
        help="store to query: 'code' (default, source-code index), 'output' (indexed "
        "command output), or 'dependency' (one indexed package, requires "
        "--ecosystem/--package)",
    )
    p_search.add_argument(
        "--ecosystem",
        default=None,
        help="dependency ecosystem for --store dependency, e.g. python",
    )
    p_search.add_argument(
        "--package",
        default=None,
        help="dependency package name for --store dependency",
    )
    p_search.set_defaults(func=_cmd_search)

    p_search_symbol = sub.add_parser(
        "search-symbol",
        help="query the symbol-index store by exact identifier name, ranked Def above Unknown",
    )
    p_search_symbol.add_argument("query", help="identifier to search for")
    p_search_symbol.add_argument(
        "--k", type=int, default=10, help="max ranked hits (default 10)"
    )
    p_search_symbol.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON blob with per-hit line range + bounded snippet text, instead of plain text",
    )
    p_search_symbol.set_defaults(func=_cmd_search_symbol)

    p_search_xref = sub.add_parser(
        "search-xref",
        help="query the cross-file reference index, injecting referenced-definition files after the base ranking without evicting any base hit",
    )
    p_search_xref.add_argument("query", help="search query")
    p_search_xref.add_argument(
        "--k",
        type=int,
        default=10,
        help="ranked base hits (default 10); more rows are returned when files are injected",
    )
    p_search_xref.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON blob with per-hit line range + bounded snippet text, instead of plain text",
    )
    p_search_xref.set_defaults(func=_cmd_search_xref)

    p_search_importers = sub.add_parser(
        "search-importers",
        help="reverse xref query: every currently-indexed file with a resolved or unresolved import/call edge referencing this symbol",
    )
    p_search_importers.add_argument("query", help="symbol name to find importers/callers of")
    p_search_importers.add_argument(
        "--def-path",
        default=None,
        dest="def_path",
        help="scope resolved hits to importers/callers whose target is this file, matched after resolving to an absolute path (unresolved rows stay reported at the symbol level)",
    )
    p_search_importers.add_argument(
        "--k",
        type=int,
        default=200,
        help="max rows per bucket (hits, unresolved), independently bounded (default 200, must be >= 1)",
    )
    p_search_importers.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON blob with hits, unresolved, scope, and status, instead of plain text",
    )
    p_search_importers.set_defaults(func=_cmd_search_importers)

    p_search_goto_def = sub.add_parser(
        "search-goto-def",
        help="go to a symbol's definition via the Python AST index lane; a result is tagged resolved, or an honestly-scoped nothing-found status when no definite target exists",
    )
    p_search_goto_def.add_argument("query", help="symbol name to find the definition of")
    p_search_goto_def.add_argument(
        "--from-path",
        required=True,
        dest="from_path",
        help="the file the symbol is referenced from",
    )
    p_search_goto_def.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON blob with hits, from_path, and status, instead of plain text",
    )
    p_search_goto_def.set_defaults(func=_cmd_search_goto_def)

    p_search_references = sub.add_parser(
        "search-references",
        help="find every reference to a symbol's definition via the Python AST index lane; each row is tagged resolved or unresolved, never a silent best-guess",
    )
    p_search_references.add_argument("query", help="symbol name to find references to")
    p_search_references.add_argument(
        "--def-path",
        required=True,
        dest="def_path",
        help="the file the symbol is defined in",
    )
    p_search_references.add_argument(
        "--k",
        type=int,
        default=200,
        help="max rows per bucket (resolved, unresolved), independently bounded before merging (default 200, must be >= 1)",
    )
    p_search_references.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON blob with hits, def_path, and status, instead of plain text",
    )
    p_search_references.set_defaults(func=_cmd_search_references)

    p_search_literal = sub.add_parser(
        "search-literal",
        help="exact byte-for-byte substring scan over every currently-indexed file; an empty result is a proof of absence over files_scanned, not an unknown",
    )
    p_search_literal.add_argument("query", help="exact literal to search for")
    p_search_literal.add_argument(
        "--k",
        type=int,
        default=10,
        help="accepted and ignored: the scan is exhaustive over the indexed set, never top-k",
    )
    p_search_literal.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON blob with hits, scope, files_scanned, and status, instead of plain text",
    )
    p_search_literal.add_argument(
        "--store",
        choices=["code", "output"],
        default="code",
        help="store to query: 'code' (default, source-code index) or 'output' (indexed command output)",
    )
    p_search_literal.set_defaults(func=_cmd_search_literal)

    p_search_vector = sub.add_parser(
        "search-vector",
        help="embed the query via a detached worker subprocess, then query the exact-cosine vector store",
    )
    p_search_vector.add_argument("query", help="search query")
    p_search_vector.add_argument(
        "--k", type=int, default=10, help="max ranked hits (default 10)"
    )
    p_search_vector.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON blob with per-hit line range + bounded snippet text, instead of plain text",
    )
    p_search_vector.set_defaults(func=_cmd_search_vector)

    p_search_fused = sub.add_parser(
        "search-fused",
        help="RRF-fuse (K=60) the whole-file/symbol/xref-derived lanes, adding a gated semantic lane only when the base top result is weak",
    )
    p_search_fused.add_argument("query", help="search query")
    p_search_fused.add_argument(
        "--k", type=int, default=10, help="max ranked hits (default 10)"
    )
    p_search_fused.add_argument(
        "--weak-threshold",
        type=float,
        default=None,
        dest="weak_threshold",
        help="override the gate's default weak-base-signal threshold",
    )
    p_search_fused.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON blob with per-hit line range + bounded snippet text plus the vector-lane degradation state, instead of plain text",
    )
    p_search_fused.set_defaults(func=_cmd_search_fused)

    p_embed_index = sub.add_parser(
        "embed-index",
        help=(
            "discover files under a directory and submit the embed worker to the "
            "executor daemon as a detached background run, returning a run_id "
            "immediately; --foreground runs it synchronously instead"
        ),
    )
    p_embed_index.add_argument("path", help="directory to embed-index")
    p_embed_index.add_argument(
        "--foreground",
        action="store_true",
        help="run the embed worker synchronously via the executor daemon instead of detaching it",
    )
    p_embed_index.set_defaults(func=_cmd_embed_index)

    p_run = sub.add_parser(
        "run", help="run a command via the executor daemon (spawn+capture, lazy-spawns the daemon)"
    )
    p_run.add_argument(
        "--timeout", type=float, default=None, help="seconds before the daemon times out the run"
    )
    p_run.add_argument(
        "--background",
        action="store_true",
        help="return immediately; the daemon keeps running the command and owns its state",
    )
    p_run.add_argument(
        "--auto-index",
        action="store_true",
        dest="auto_index",
        help="opt-in: index this run's captured output for later search (never default-on); "
        "secret-shaped output is refused, not indexed; skipped (index_error set) when "
        "--background is also set",
    )
    p_run.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON object (run_id/state/stdout/stderr, plus indexed/refused when "
        "--auto-index is set) instead of the plain-text trailing-marker format",
    )
    p_run.add_argument("command", help="program to run")
    p_run.add_argument("args", nargs=argparse.REMAINDER, help="arguments to the program")
    p_run.set_defaults(func=_cmd_run)

    p_fetch = sub.add_parser(
        "fetch", help="re-serve a prior run's next bounded output window without re-running it"
    )
    p_fetch.add_argument("run_id", help="run id returned by a prior `run`")
    p_fetch.add_argument(
        "--bound", type=int, default=None, help="max bytes to re-serve in this window (default 64KiB)"
    )
    p_fetch.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON object instead of the plain-text trailing-marker format",
    )
    p_fetch.set_defaults(func=_cmd_fetch)

    p_batch = sub.add_parser(
        "batch-run",
        help="run a list of shell-quoted commands under the daemon's batch concurrency semantics",
    )
    p_batch.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="1 = one shared timeout budget run in sequence; 2-8 = each command gets its own timeout, run in parallel",
    )
    p_batch.add_argument(
        "--timeout-ms", type=int, default=None, dest="timeout_ms",
        help="total shared budget at concurrency=1, per-command timeout at concurrency 2-8",
    )
    p_batch.add_argument(
        "commands", nargs="*", default=[],
        help="each argument is one shell-quoted command, e.g. 'echo hi'; omit when using --commands-json",
    )
    p_batch.add_argument(
        "--commands-json",
        default=None,
        help="JSON array of {command, args} objects, run without shell tokenization; "
        "mutually exclusive with positional commands",
    )
    p_batch.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON array aligned 1:1 with the input commands, instead of plain text",
    )
    p_batch.set_defaults(func=_cmd_batch_run)

    p_git_log = sub.add_parser(
        "git-log",
        help="bounded git-log query scoped by a date window, revision range, or path filter",
    )
    p_git_log.add_argument("--since", default=None, help="lower date bound (git's --since syntax)")
    p_git_log.add_argument("--until", default=None, help="upper date bound (git's --until syntax)")
    p_git_log.add_argument("--author", default=None, help="author filter (git's --author syntax)")
    p_git_log.add_argument("--path", default=None, help="limit to commits touching this path")
    p_git_log.add_argument(
        "--repo",
        default=None,
        help="path to the git repository (default: the executor daemon's "
        "working directory, which may not match your shell -- pass --repo "
        "for a reliable target)",
    )
    p_git_log.add_argument(
        "rev_range", nargs="?", default=None, help="revision or revision range"
    )
    p_git_log.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON object (run_id/state/stdout/stderr/truncated/elided_bytes) "
        "instead of the plain-text trailing-marker format",
    )
    p_git_log.set_defaults(func=_cmd_git_log)

    p_git_blame = sub.add_parser(
        "git-blame",
        help="bounded git-blame query for a file, optionally scoped to a line range",
    )
    p_git_blame.add_argument(
        "--start-line", type=int, default=None, dest="start_line",
        help="1-based inclusive start line of an optional line range",
    )
    p_git_blame.add_argument(
        "--end-line", type=int, default=None, dest="end_line",
        help="1-based inclusive end line of an optional line range",
    )
    p_git_blame.add_argument(
        "--repo",
        default=None,
        help="path to the git repository (default: the executor daemon's "
        "working directory, which may not match your shell -- pass --repo "
        "for a reliable target)",
    )
    p_git_blame.add_argument("file", nargs="?", default=None, help="file to blame")
    p_git_blame.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON object (run_id/state/stdout/stderr/truncated/elided_bytes) "
        "instead of the plain-text trailing-marker format",
    )
    p_git_blame.set_defaults(func=_cmd_git_blame)

    p_git_diff = sub.add_parser(
        "git-diff",
        help="bounded git-diff query between two revisions, or scoped by a path filter",
    )
    p_git_diff.add_argument("--path", default=None, help="limit the diff to this path")
    p_git_diff.add_argument(
        "--repo",
        default=None,
        help="path to the git repository (default: the executor daemon's "
        "working directory, which may not match your shell -- pass --repo "
        "for a reliable target)",
    )
    p_git_diff.add_argument(
        "rev_a", nargs="?", default=None, help="a revision range, or the first of two revisions"
    )
    p_git_diff.add_argument("rev_b", nargs="?", default=None, help="the second of two revisions")
    p_git_diff.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON object (run_id/state/stdout/stderr/truncated/elided_bytes) "
        "instead of the plain-text trailing-marker format",
    )
    p_git_diff.set_defaults(func=_cmd_git_diff)

    p_capture = sub.add_parser(
        "capture",
        help="translate a raw hook payload on stdin into the classifier's flat schema and capture the resulting event, if any",
    )
    p_capture.add_argument(
        "--source-hook",
        default=None,
        dest="source_hook",
        help="lifecycle label of the firing hook (e.g. PostToolUse, UserPromptSubmit)",
    )
    p_capture.set_defaults(func=_cmd_capture)

    p_session_search = sub.add_parser(
        "session-search",
        help="query one session's captured events, scoped by the session-id validation choke point",
    )
    p_session_search.add_argument("session_id", help="session id to query")
    p_session_search.add_argument(
        "--also-session",
        action="append",
        default=None,
        dest="also_session",
        help="additional session id to include (repeatable); union of all named sessions' events",
    )
    p_session_search.add_argument(
        "--substring", default=None, help="case-insensitive substring filter on event data"
    )
    p_session_search.add_argument(
        "--category", default=None, help="exact category filter"
    )
    p_session_search.add_argument(
        "--since", default=None, help="RFC3339 lower bound (inclusive) on event timestamp"
    )
    p_session_search.add_argument(
        "--until", default=None, help="RFC3339 upper bound (inclusive) on event timestamp"
    )
    p_session_search.set_defaults(func=_cmd_session_search)

    p_stats = sub.add_parser(
        "stats",
        help="aggregate code-index segment, session, and executor run counts from the on-disk stores",
    )
    p_stats.set_defaults(func=_cmd_stats)

    p_command_insight = sub.add_parser(
        "command-insight",
        help="actionable report over accumulated run history: failing commands and the most recent failure",
    )
    p_command_insight.set_defaults(func=_cmd_command_insight)

    return parser


def main(argv: "list[str] | None" = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
