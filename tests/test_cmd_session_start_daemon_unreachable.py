from __future__ import annotations

import argparse
import uuid


def test_daemon_unreachable_renders_unavailable_marker_exit_zero(tmp_path, monkeypatch, capsys):
    # This assertion was deliberately flipped: the daemon-unreachable branch
    # used to write nothing to stdout (silent blank), which made an
    # unavailable session indistinguishable from a healthy one with no
    # payload. This phase retires that silent-blank behavior in favor of a
    # visible UNAVAILABLE marker so the caller can tell the two apart.
    from iai_mcp import cli as cli_mod

    bad_sock = tmp_path / f"iai-mcp-does-not-exist-{uuid.uuid4().hex}.sock"
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(bad_sock))

    rc = cli_mod.cmd_session_start(argparse.Namespace(session_id="-"))

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out != ""
    assert "UNAVAILABLE" in captured.out


def test_daemon_unreachable_session_classifies_as_excluded_unavailable(tmp_path, monkeypatch, capsys):
    """A genuinely unreachable session must resolve to excluded, not a
    silent pass -- proven end-to-end from the real CLI marker, never a
    synthesized bool."""
    from iai_mcp import cli as cli_mod
    from iai_mcp.availability import classify_session_outcome, OUTCOME_EXCLUDED_UNAVAILABLE

    bad_sock = tmp_path / f"iai-mcp-does-not-exist-{uuid.uuid4().hex}.sock"
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(bad_sock))

    rc = cli_mod.cmd_session_start(argparse.Namespace(session_id="-"))
    captured = capsys.readouterr()
    assert rc == 0

    available = "UNAVAILABLE" not in captured.out

    outcome = classify_session_outcome(available=available, passed=None)
    assert outcome == OUTCOME_EXCLUDED_UNAVAILABLE
