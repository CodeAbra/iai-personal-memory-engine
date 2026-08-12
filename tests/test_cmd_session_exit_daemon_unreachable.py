from __future__ import annotations

import argparse
import uuid


def test_daemon_unreachable_exit_zero(tmp_path, monkeypatch):
    """cmd_session_exit is the CLI backend for the Stop-hook's session_exit
    notification (run_light_consolidation's FSRS decay/reinforcement tick).
    Mirrors test_cmd_session_start_daemon_unreachable.py: an unreachable
    daemon must never fail the hook that calls this."""
    from iai_mcp.cli._capture import cmd_session_exit

    bad_sock = tmp_path / f"iai-mcp-does-not-exist-{uuid.uuid4().hex}.sock"
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(bad_sock))

    rc = cmd_session_exit(argparse.Namespace(session_id="s-exit-test"))
    assert rc == 0


def test_registered_as_cli_subcommand():
    """session-exit must be reachable as `iai-mcp session-exit --session-id
    ...` — the Stop hook shells out to the installed CLI, not to Python."""
    from iai_mcp.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["session-exit", "--session-id", "s-1"])
    assert args.session_id == "s-1"
    assert args.func.__name__ == "cmd_session_exit"


def test_session_exit_dedupes_within_cooldown_window(tmp_path, monkeypatch):
    """Stop fires after every assistant response, not at session end, so a
    long session would otherwise call run_light_consolidation -- and its
    unconditional per-record FSRS stability boost -- on every single turn.
    The RPC handler now dedupes in time per session: a second session_exit
    within the cooldown window must be a no-op, not a second FSRS pass."""
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.core import dispatch
    from iai_mcp.events import query_events
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    session_id = "s-dedup-test"

    result1 = dispatch(store, "session_exit", {"session_id": session_id})
    assert result1.get("deduped") is not True

    result2 = dispatch(store, "session_exit", {"session_id": session_id})
    assert result2.get("deduped") is True
    assert result2["cooldown_remaining_sec"] > 0

    events = query_events(store, kind="cls_consolidation_run", limit=10)
    light_events = [
        e for e in events
        if e["data"].get("mode") == "light" and e.get("session_id") == session_id
    ]
    assert len(light_events) == 1, (
        "a session_exit within the cooldown window must not run a second "
        "light consolidation pass for the same session"
    )


def test_session_exit_dedup_is_per_session(tmp_path, monkeypatch):
    """The cooldown must not bleed across sessions -- a different session_id
    must not be blocked by another session's recent light pass."""
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.core import dispatch
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)

    dispatch(store, "session_exit", {"session_id": "s-a"})
    result_b = dispatch(store, "session_exit", {"session_id": "s-b"})
    assert result_b.get("deduped") is not True
