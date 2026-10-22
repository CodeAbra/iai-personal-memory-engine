"""`iai-mcp tools` operator CLI: index/search the brain-independent code
index, session store, and executor.

Asymmetry, load-bearing: `index` and `embed-index` MAY import iai_mcp.study
for file discovery. Every other subcommand (`search`, `search-symbol`,
`search-xref`, `search-vector`, `search-fused`, `session-search`, `stats`,
`run`, `fetch`, `batch-run`) MUST NOT import iai_mcp.study, iai_mcp.hippo,
iai_mcp.lilli, or iai_mcp_native -- only iai_mcp.tools (reaches only
iai_mcp_tools_core), iai_mcp.tools.session, iai_mcp.tools.stats,
iai_mcp.tools_exec_client (a thin socket client to the executor daemon), or
iai_mcp.tools.exec_cli (the shared `run`/`fetch`/`batch-run` command body,
also used by the `iai-tools` console script so the two surfaces cannot
drift).
`search-vector` and `search-fused` SPAWN `iai_mcp.tools.embed_worker` as a
subprocess to embed query text -- `search-fused` only when its gate
reports the base lanes' top result is weak; `embed-index` submits that
same worker module as a detached background run through the executor
daemon -- none of these import iai_mcp_native or embed_worker in this
process.

This module's own discipline is necessary but NOT sufficient for a
brain-free `iai-mcp tools` invocation: `iai_mcp.cli.__init__` imports
`._analytics`/`._maintenance`/`._capture` at module load time, and those
transitively reach `iai_mcp.hippo`/`iai_mcp.store` before argparse ever
dispatches into this file -- so `iai-mcp tools search/run/fetch/batch-run`
still pays that import cost. The zero-brain-import guarantee this module
follows is actually reached end-to-end only via the standalone `iai-tools`
console script (`tools_cli.py`), whose own module-level imports are
stdlib-only.
"""

from __future__ import annotations

import argparse
import sys


def cmd_tools_index(args: argparse.Namespace) -> int:
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
        from iai_mcp.tools import index_paths

        files = iter_study_files(
            args.path, max_files=sys.maxsize, dot_dir_allowlist=TOOLS_INDEX_DOT_DIR_ALLOWLIST,
        )
        indexed = index_paths([str(f) for f in files])
        print(f"indexed {indexed} files under {args.path}")
        return 0

    report = index_dependency_gated(ecosystem, package, project_dir)
    message, code = describe_index_outcome(ecosystem, package, report)
    print(message, file=sys.stderr if code else sys.stdout)
    return code


def cmd_tools_search(args: argparse.Namespace) -> int:
    from iai_mcp.tools import search
    from iai_mcp.tools.exec_cli import search_store_root

    try:
        store_root = search_store_root(args)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    if getattr(args, "store", "code") == "dependency":
        from iai_mcp.tools.dependency import is_dependency_store_indexed

        if not is_dependency_store_indexed(store_root):
            print("not indexed")
            return 0

    hits = search(args.query, k=args.k, store_root=store_root)
    if not hits:
        print("no hits")
        return 0
    for path, score in hits:
        print(f"{score:.4f}\t{path}")
    return 0


def cmd_tools_search_symbol(args: argparse.Namespace) -> int:
    from iai_mcp.tools import search_symbol

    hits = search_symbol(args.query, k=args.k)
    if not hits:
        print("no hits")
        return 0
    for path, score in hits:
        print(f"{score:.4f}\t{path}")
    return 0


def cmd_tools_search_xref(args: argparse.Namespace) -> int:
    from iai_mcp.tools import search_xref

    hits = search_xref(args.query, k=args.k)
    if not hits:
        print("no hits")
        return 0
    for path, score in hits:
        print(f"{score:.4f}\t{path}")
    return 0


def cmd_tools_search_vector(args: argparse.Namespace) -> int:
    from iai_mcp.tools import EmbedWorkerError, _embed_query_via_subprocess, search_vector

    try:
        query_vector = _embed_query_via_subprocess(args.query)
    except EmbedWorkerError as e:
        print(str(e), file=sys.stderr)
        return 1

    hits = search_vector(query_vector, k=args.k)
    if not hits:
        print("no hits")
        return 0
    for path, score in hits:
        print(f"{score:.4f}\t{path}")
    return 0


def cmd_tools_search_fused(args: argparse.Namespace) -> int:
    from iai_mcp.tools import search_fused

    hits = search_fused(args.query, k=args.k, weak_threshold=args.weak_threshold)
    if not hits:
        print("no hits")
        return 0
    for path, score in hits:
        print(f"{score:.4f}\t{path}")
    return 0


def cmd_tools_embed_index(args: argparse.Namespace) -> int:
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


def cmd_tools_session_search(args: argparse.Namespace) -> int:
    import json

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


def cmd_tools_stats(args: argparse.Namespace) -> int:
    import json

    from iai_mcp.tools.stats import gather_stats

    print(json.dumps(gather_stats()))
    return 0


def cmd_tools_command_insight(args: argparse.Namespace) -> int:
    import json

    from iai_mcp.tools.stats import gather_command_insight

    print(json.dumps(gather_command_insight()))
    return 0


def cmd_tools_run(args: argparse.Namespace) -> int:
    from iai_mcp.tools.exec_cli import cmd_run

    return cmd_run(args)


def cmd_tools_fetch(args: argparse.Namespace) -> int:
    from iai_mcp.tools.exec_cli import cmd_fetch

    return cmd_fetch(args)


def cmd_tools_batch_run(args: argparse.Namespace) -> int:
    from iai_mcp.tools.exec_cli import cmd_batch_run

    return cmd_batch_run(args)


def cmd_tools_git_log(args: argparse.Namespace) -> int:
    from iai_mcp.tools.git_query import cmd_git_log

    return cmd_git_log(args)


def cmd_tools_git_blame(args: argparse.Namespace) -> int:
    from iai_mcp.tools.git_query import cmd_git_blame

    return cmd_git_blame(args)


def cmd_tools_git_diff(args: argparse.Namespace) -> int:
    from iai_mcp.tools.git_query import cmd_git_diff

    return cmd_git_diff(args)
