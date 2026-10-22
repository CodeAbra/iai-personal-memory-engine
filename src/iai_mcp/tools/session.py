"""Brain-independent session-event capture, query, and snapshot glue.

Thin Python wrapper around the compiled iai_mcp_tools_core Rust extension.
This module imports only the standard library and the native extension at
call time -- it must never pull in iai_mcp.capture, iai_mcp.hippo,
iai_mcp.lilli, iai_mcp_native, or any other brain module, since the session
store exists to capture and answer without loading the memory system.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_ENV_ROOT = "IAI_TOOLS_ROOT"
_DEFAULT_ROOT_NAME = ".iai-tools"


def default_session_store_root() -> Path:
    """`$IAI_TOOLS_ROOT/sessions` if set and non-empty, else `~/.iai-tools/sessions`."""
    override = os.environ.get(_ENV_ROOT, "")
    root = Path(override) if override else Path.home() / _DEFAULT_ROOT_NAME
    return root / "sessions"


def capture_event(hook_payload: dict, *, store_root: "Path | None" = None) -> None:
    """Classifies `hook_payload` and appends the resulting event, if any."""
    import iai_mcp_tools_core as _native

    root = (store_root or default_session_store_root()).parent
    _native.session_capture(str(root), json.dumps(hook_payload))


def query_events(
    session_id: str,
    *,
    substring: "str | None" = None,
    category: "str | None" = None,
    since: "str | None" = None,
    until: "str | None" = None,
    store_root: "Path | None" = None,
) -> list[dict]:
    """Substring/category/time-range-filtered events for one session, in
    timeline (file append) order. `since`/`until` are RFC3339 timestamp
    strings compared against each event's `ts`.
    """
    import iai_mcp_tools_core as _native

    root = (store_root or default_session_store_root()).parent
    lines = _native.session_query(str(root), session_id, substring, category, since, until)
    return [json.loads(line) for line in lines]


def query_events_multi(
    session_ids: "list[str]",
    *,
    substring: "str | None" = None,
    category: "str | None" = None,
    since: "str | None" = None,
    until: "str | None" = None,
    store_root: "Path | None" = None,
) -> list[dict]:
    """Concatenates `query_events` per id, in the given order. The single
    home of cross-session list semantics -- loops the existing per-session
    choke point, never touches the Rust store's single-session scoping.
    """
    events: list[dict] = []
    for session_id in session_ids:
        events.extend(
            query_events(
                session_id,
                substring=substring,
                category=category,
                since=since,
                until=until,
                store_root=store_root,
            )
        )
    return events


def build_snapshot(session_id: str, *, store_root: "Path | None" = None) -> str:
    """Post-compaction snapshot text for one session: loads the session's
    events (the only I/O) then renders them through the pure snapshot
    builder.
    """
    import iai_mcp_tools_core as _native

    root = (store_root or default_session_store_root()).parent
    return _native.session_snapshot(str(root), session_id)


def build_snapshot_from_events(events: list[dict]) -> str:
    """Renders `events` (the same shape `query_events` returns) directly
    through the pure snapshot builder -- no store, no I/O. Lets callers
    (and tests) exercise the builder against an in-memory fixture.
    """
    import iai_mcp_tools_core as _native

    return _native.session_snapshot_from_events(json.dumps(events))
