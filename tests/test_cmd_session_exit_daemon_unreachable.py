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
