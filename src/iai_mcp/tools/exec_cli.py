"""Shared render layer for `run`/`fetch`/`batch-run`, used by both the
`iai-tools` console script and the `iai-mcp tools` operator CLI so the
two entry points cannot drift.

Module-level imports are stdlib only. `iai_mcp.tools_exec_client` is
imported at call time; `iai_mcp.tools.output_index` is imported only
inside the `auto_index` branch, so a plain run stays native-extension-
free and brain-free.

`cmd_run`/`cmd_fetch`/`cmd_batch_run` are the ONE shared command body
each surface's argparse dispatcher delegates to -- `tools_cli.py`'s
`_cmd_run`/`_cmd_fetch`/`_cmd_batch_run` and `cli/_tools.py`'s
`cmd_tools_run`/`cmd_tools_fetch`/`cmd_tools_batch_run` call these
directly, so the two CLIs cannot drift. `search_store_root` is the same
kind of shared body for the `search` command's `--store` selector.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Mirrors MAX_BATCH_COMMANDS in mcp-wrapper/src/tools-index.ts -- keep both in sync.
MAX_BATCH_COMMANDS = 64


def run_and_render(
    command: str,
    args: "list[str] | None" = None,
    *,
    timeout: "float | None" = None,
    background: bool = False,
    auto_index: bool = False,
    output_store_root: "Path | None" = None,
) -> dict[str, Any]:
    """Runs `command` via the executor daemon; optionally auto-indexes
    its captured output afterward.

    Auto-index is opt-in only (`auto_index=False` by default) -- a run
    without the flag never creates the command-output store and returns
    the same shape as `tools_exec_client.run` (run_id/state/bounded_tail),
    byte-unchanged. When `auto_index` is set and the run has a final
    (non-background) result, the post-run index step is wrapped so an
    index-time error NEVER turns a successful run into a failed call --
    it folds into `index_error` instead of propagating.

    `auto_index` combined with `background=True` is explicitly NOT
    supported -- a backgrounded run's output is not yet final when this
    call returns, and `fetch` never runs the index step either -- so
    indexing is skipped and `index_error` is set to say so, rather than
    silently dropping the caller's request with no signal at all.
    """
    from iai_mcp import tools_exec_client

    result = tools_exec_client.run(
        command, args, timeout=timeout, background=background
    )

    if not auto_index:
        return result

    if background:
        result["index_error"] = (
            "auto-index is not applied to background runs; fetch the run "
            "once it completes, then index its output directly"
        )
        return result

    run_id = result.get("run_id")
    if not run_id:
        return result

    try:
        from iai_mcp.tools import output_index
        from iai_mcp.tools_exec_client import _tools_root

        run_dir = _tools_root() / "artifacts" / run_id
        report = output_index.index_command_output(run_dir, store_root=output_store_root)
        result["indexed"] = report["indexed"]
        result["refused"] = report["refused"]
    except Exception as e:  # noqa: BLE001 -- index step must never fail a successful run
        result["index_error"] = str(e)

    return result


def search_store_root(args: Any) -> "Path | None":
    """`--store output`/`--store dependency` select an alternate store
    instead of the default source-code store -- a store SELECTOR on the
    same whole-file engine, not a new search lane. Shared by both
    `tools_cli.py`'s `_cmd_search` and `cli/_tools.py`'s
    `cmd_tools_search` so the selector cannot drift between the two
    surfaces. Raises `DependencyRequestError` (a `ValueError`) for
    `--store dependency` missing `--ecosystem`/`--package` -- callers must
    catch and render that as a normal CLI error, not let it propagate.
    """
    store = getattr(args, "store", "code")
    if store == "output":
        from iai_mcp.tools import default_command_output_store_root

        return default_command_output_store_root()
    if store == "dependency":
        from iai_mcp.tools import dependency_store_root_for
        from iai_mcp.tools.dependency import DependencyRequestError

        ecosystem = getattr(args, "ecosystem", None)
        package = getattr(args, "package", None)
        if not ecosystem or not package:
            raise DependencyRequestError(
                "search --store dependency requires --ecosystem and --package"
            )
        return dependency_store_root_for(ecosystem, package)
    return None


def fetch_and_render(run_id: str, *, bound: "int | None" = None) -> dict[str, Any]:
    """Re-serves a prior run's next bounded output window. Thin pass-through
    to `tools_exec_client.fetch` -- no auto-index concept applies to fetch."""
    from iai_mcp import tools_exec_client

    return tools_exec_client.fetch(run_id, bound=bound)


def batch_run_and_render(
    commands: "list[dict[str, Any]]",
    *,
    concurrency: int = 1,
    timeout_ms: "int | None" = None,
) -> "list[dict[str, Any]]":
    """Runs a list of commands via the executor daemon. Thin pass-through
    to `tools_exec_client.batch_run` -- no auto-index concept applies."""
    from iai_mcp import tools_exec_client

    return tools_exec_client.batch_run(commands, concurrency=concurrency, timeout_ms=timeout_ms)


def _flatten_run_result(result: "dict[str, Any]") -> "dict[str, Any]":
    """Flattens `bounded_tail` into top-level `stdout`/`stderr` for the
    JSON transport -- the shape the MCP wrapper's structured output and
    the CLI's `--json` mode both carry: run_id, state, stdout, stderr,
    plus indexed/refused/index_error when auto-index ran.
    """
    tail = result.get("bounded_tail") or {}
    flat: "dict[str, Any]" = {
        "run_id": result.get("run_id"),
        "state": result.get("state"),
        "stdout": tail.get("stdout", ""),
        "stderr": tail.get("stderr", ""),
    }
    for key in ("indexed", "refused", "index_error"):
        if key in result:
            flat[key] = result[key]
    return flat


def cmd_run(args: Any) -> int:
    """Shared `run` command body for both CLI surfaces."""
    result = run_and_render(
        args.command,
        args.args,
        timeout=args.timeout,
        background=args.background,
        auto_index=getattr(args, "auto_index", False),
    )
    if getattr(args, "json", False):
        print(json.dumps(_flatten_run_result(result)))
        return 0

    tail = result["bounded_tail"]
    sys.stdout.write(tail.get("stdout", ""))
    sys.stderr.write(tail.get("stderr", ""))
    print(f"run_id: {result['run_id']} state: {result['state']}")
    return 0


def cmd_fetch(args: Any) -> int:
    """Shared `fetch` command body for both CLI surfaces."""
    result = fetch_and_render(args.run_id, bound=args.bound)
    if getattr(args, "json", False):
        print(json.dumps(_flatten_run_result(result)))
        return 0

    tail = result["bounded_tail"]
    sys.stdout.write(tail.get("stdout", ""))
    sys.stderr.write(tail.get("stderr", ""))
    print(f"run_id: {result['run_id']} state: {result['state']}")
    return 0


def cmd_batch_run(args: Any) -> int:
    """Shared `batch-run` command body for both CLI surfaces."""
    import shlex

    commands_json = getattr(args, "commands_json", None)
    positional = getattr(args, "commands", None) or []

    if commands_json and positional:
        print(
            "batch-run: pass either --commands-json or positional commands, not both",
            file=sys.stderr,
        )
        return 1
    if not commands_json and not positional:
        print(
            "batch-run: supply --commands-json or at least one positional command",
            file=sys.stderr,
        )
        return 1

    if commands_json:
        try:
            parsed = json.loads(commands_json)
        except json.JSONDecodeError as e:
            print(f"batch-run: --commands-json is not valid JSON: {e}", file=sys.stderr)
            return 1
        if (
            not isinstance(parsed, list)
            or not parsed
            or not all(
                isinstance(item, dict) and isinstance(item.get("command"), str) for item in parsed
            )
        ):
            print(
                "batch-run: --commands-json must be a non-empty JSON array of objects each with "
                "a string 'command'",
                file=sys.stderr,
            )
            return 1
        commands = parsed
    else:
        commands = []
        for raw in positional:
            parts = shlex.split(raw)
            if not parts:
                print(f"empty command in batch-run argument: {raw!r}", file=sys.stderr)
                return 1
            commands.append({"command": parts[0], "args": parts[1:]})

    if len(commands) > MAX_BATCH_COMMANDS:
        print(
            f"batch-run: at most {MAX_BATCH_COMMANDS} commands per batch, got {len(commands)}",
            file=sys.stderr,
        )
        return 1

    results = batch_run_and_render(
        commands, concurrency=args.concurrency, timeout_ms=args.timeout_ms
    )

    if getattr(args, "json", False):
        print(json.dumps([_flatten_run_result(r) for r in results]))
        return 0

    for i, element in enumerate(results):
        tail = element.get("bounded_tail") or {}
        sys.stdout.write(tail.get("stdout", ""))
        sys.stderr.write(tail.get("stderr", ""))
        print(f"[{i}] run_id: {element.get('run_id')} state: {element.get('state')}")
    return 0
