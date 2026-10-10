"""On-disk aggregation glue backing `iai-tools stats`.

No native aggregate function exists for this -- every count here is derived
by walking the same on-disk stores the rest of the tools subsystem already
writes (code-index segment files, the session JSONL store, the executor
artifacts directory). Imports only the standard library and the tools/
session glue at call time -- must never pull in a brain module.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Retention bound on the redirect savings log: past this many well-formed
# lines, the reader-side trim rewrites the file to its last N lines. Pinned
# to the read side, not the hook -- the hook only ever appends.
_SAVINGS_LOG_MAX_LINES = 10000


def _lane_stats(root: "Path | None", suffix: str) -> dict:
    """Segment count, total bytes, and latest mtime for one code-index lane.

    Matches by the lane's own segment-file suffix (`.lcidx`/`.lsidx`/`.lxidx`,
    see `rust/iai_mcp_tools_core/src/index.rs` and its symbol/xref siblings)
    rather than excluding the manifest by name -- a positive allowlist, so a
    stray non-segment file dropped into the store root (e.g. a filesystem
    marker file) is never miscounted as a segment.
    """
    if root is None or not root.is_dir():
        return {"segment_count": 0, "total_bytes": 0, "latest_modified": None}

    segment_files = [p for p in root.glob(f"*{suffix}") if p.is_file()]
    if not segment_files:
        return {"segment_count": 0, "total_bytes": 0, "latest_modified": None}

    sizes = [p.stat().st_size for p in segment_files]
    latest_mtime = max(p.stat().st_mtime for p in segment_files)
    latest_modified = datetime.fromtimestamp(latest_mtime, tz=timezone.utc).isoformat()
    return {
        "segment_count": len(segment_files),
        "total_bytes": sum(sizes),
        "latest_modified": latest_modified,
    }


def _session_stats(sessions_root: "Path | None") -> dict:
    """Session count, total event count, and per-category event totals.

    Session ids are discovered from the store's own `<id>.jsonl` file
    names (there is no native "list sessions" call); each session's
    events are then read through `query_events`, the same choke point
    every other session read goes through -- never a hand-parsed JSONL
    read here.
    """
    if sessions_root is None or not sessions_root.is_dir():
        return {"session_count": 0, "total_events": 0, "events_by_category": {}}

    from iai_mcp.tools.session import query_events

    session_ids = sorted(p.stem for p in sessions_root.glob("*.jsonl") if p.is_file())
    total_events = 0
    events_by_category: dict[str, int] = {}
    for session_id in session_ids:
        events = query_events(session_id, store_root=sessions_root)
        total_events += len(events)
        for event in events:
            category = event.get("category", "unknown")
            events_by_category[category] = events_by_category.get(category, 0) + 1

    return {
        "session_count": len(session_ids),
        "total_events": total_events,
        "events_by_category": events_by_category,
    }


def _executor_stats(artifacts_root: "Path | None") -> dict:
    """Run count from the executor's `artifacts/<run_id>/` subdirectories."""
    if artifacts_root is None or not artifacts_root.is_dir():
        return {"run_count": 0}
    run_count = sum(1 for p in artifacts_root.iterdir() if p.is_dir())
    return {"run_count": run_count}


def _trim_savings_log(savings_log: Path, retained_lines: "list[str]") -> None:
    """Rewrites the savings log to exactly `retained_lines`, via a tmp file
    plus atomic `os.replace` -- a single Python writer, never a bash
    read-modify-write. Not atomic against a concurrent hook append: a line
    appended between the read above and this replace lands in the
    replaced-away copy and is undercounted, never corrupted or torn. Rare
    and best-effort -- fires only once the log exceeds the retention cap.
    """
    try:
        tmp = savings_log.with_name(f"{savings_log.name}.trim-{os.getpid()}.tmp")
        tmp.write_text("\n".join(retained_lines) + "\n", encoding="utf-8")
        os.replace(tmp, savings_log)
    except OSError:
        pass


def _redirect_savings(savings_log: "Path | None") -> dict:
    """Counts hook-fired redirect events from the append-only savings log
    the enforce hook writes. Absent log -> zero, never raises. Malformed or
    blank lines are skipped, never counted. Past the retention cap, trims
    the on-disk log to its last N lines and counts from those."""
    if savings_log is None or not savings_log.is_file():
        return {"redirect_count": 0}
    try:
        text = savings_log.read_text(encoding="utf-8")
    except OSError:
        return {"redirect_count": 0}

    valid_lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
        except (ValueError, TypeError):
            continue
        valid_lines.append(line)

    if len(valid_lines) > _SAVINGS_LOG_MAX_LINES:
        valid_lines = valid_lines[-_SAVINGS_LOG_MAX_LINES:]
        _trim_savings_log(savings_log, valid_lines)

    return {"redirect_count": len(valid_lines)}


def _bytes_captured(artifacts_root: "Path | None") -> dict:
    """Sums stdout_offset+stderr_offset across every run's meta.json -- the
    exact bytes the executor kept in its sandbox rather than returning to
    the model. Reuses `_iter_run_metas`. Honest secondary measurement,
    never conflated with redirect_count under a single 'savings' label."""
    metas = _iter_run_metas(artifacts_root)
    total = 0
    for meta in metas:
        try:
            total += int(meta.get("stdout_offset", 0) or 0) + int(meta.get("stderr_offset", 0) or 0)
        except (TypeError, ValueError):
            continue
    return {"bytes_captured": total}


def gather_stats(
    *,
    store_root: "Path | None" = None,
    symbol_store_root: "Path | None" = None,
    xref_store_root: "Path | None" = None,
    session_store_root: "Path | None" = None,
    artifacts_root: "Path | None" = None,
    savings_log: "Path | None" = None,
) -> dict:
    """Aggregates counts across every on-disk store the tools subsystem
    already writes. Every argument defaults to the canonical default-root
    helper already defined for that store; an absent store reports zeros
    and never raises.
    """
    from iai_mcp.tools import (
        default_store_root,
        default_symbol_store_root,
        default_xref_store_root,
    )
    from iai_mcp.tools.session import default_session_store_root

    whole_file_root = store_root or default_store_root()
    symbol_root = symbol_store_root or default_symbol_store_root()
    xref_root = xref_store_root or default_xref_store_root()
    sessions_root = session_store_root or default_session_store_root()
    artifacts = artifacts_root or (default_session_store_root().parent / "artifacts")
    savings_root = savings_log or (default_session_store_root().parent / "tools-savings.jsonl")

    executor_stats = _executor_stats(artifacts)
    executor_stats.update(_redirect_savings(savings_root))
    executor_stats.update(_bytes_captured(artifacts))

    return {
        "code_index": {
            "whole_file": _lane_stats(whole_file_root, ".lcidx"),
            "symbol": _lane_stats(symbol_root, ".lsidx"),
            "xref": _lane_stats(xref_root, ".lxidx"),
        },
        "sessions": _session_stats(sessions_root),
        "executor": executor_stats,
    }


def _iter_run_metas(artifacts_root: "Path | None") -> list[dict]:
    """Reads every completed run's `meta.json` under `artifacts_root`.

    A missing/corrupt/unreadable `meta.json`, or one that does not parse
    to a JSON object, is skipped rather than raised -- mirrors
    `_executor_stats`' zeros-on-absent contract.
    """
    if artifacts_root is None or not artifacts_root.is_dir():
        return []

    metas: list[dict] = []
    for run_dir in artifacts_root.iterdir():
        if not run_dir.is_dir():
            continue
        try:
            meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(meta, dict):
            metas.append(meta)
    return metas


def gather_command_insight(*, artifacts_root: "Path | None" = None) -> dict:
    """Actionable aggregate over the executor's own per-run metadata.

    Ships ONLY the fields a recorded measurement found actionable --
    naming a failing command, never a bare count of runs. `failing_runs`
    lists every non-zero-exit run's cmd/args/exit_code/run_id; a
    background run whose `exit_code` is still null is excluded (not yet
    finished, not a failure). `most_recent_failure` is the newest entry
    of that list by `started_at`, or None when there are no failures.
    An absent/empty `artifacts_root` returns a well-formed empty report
    and never raises.
    """
    from iai_mcp.tools.session import default_session_store_root

    root = artifacts_root or (default_session_store_root().parent / "artifacts")
    metas = _iter_run_metas(root)

    failing_runs: list[dict] = []
    for meta in metas:
        exit_code = meta.get("exit_code")
        if exit_code is None or exit_code == 0:
            continue
        failing_runs.append(
            {
                "run_id": meta.get("run_id"),
                "cmd": meta.get("cmd"),
                "args": meta.get("args", []),
                "exit_code": exit_code,
                "started_at": meta.get("started_at"),
            }
        )

    most_recent_failure = None
    if failing_runs:
        most_recent_failure = max(failing_runs, key=lambda r: r["started_at"] or 0)

    return {"failing_runs": failing_runs, "most_recent_failure": most_recent_failure}
