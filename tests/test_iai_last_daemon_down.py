from __future__ import annotations

import argparse
import datetime
import io
import json
import sys
from pathlib import Path



SID = "last-dd-test-abc"


def _write_live_file(home: Path, session_id: str, events: list[dict]) -> Path:
    deferred = home / ".iai-mcp" / ".deferred-captures"
    deferred.mkdir(parents=True, exist_ok=True)
    ts_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    live = deferred / f"{session_id}.live.jsonl"
    with live.open("w") as fh:
        fh.write(json.dumps({"version": 1, "session_id": session_id}) + "\n")
        for ev in events:
            ev_full = {
                "ts": ts_now,
                "tier": "episodic",
                "cue": f"session {session_id} turn",
            }
            ev_full.update(ev)
            fh.write(json.dumps(ev_full) + "\n")
    return live


def _run_cmd_last(n: int, session: str | None, monkeypatch, tmp_path, capture_stdout=True):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "nonexistent.sock"))
    monkeypatch.setenv("NO_COLOR", "1")

    args = argparse.Namespace(n=n, session=session)

    from iai_mcp.iai_cli import cmd_last

    if capture_stdout:
        buf = io.StringIO()
        monkeypatch.setattr(sys, "stdout", buf)
        rc = cmd_last(args)
        return rc, buf.getvalue()
    else:
        rc = cmd_last(args)
        return rc, ""


def test_cmd_last_daemon_down_returns_live_user_turn(tmp_path, monkeypatch):
    _write_live_file(tmp_path, SID, [
        {"text": "hello live fallback world", "role": "user"},
    ])
    rc, out = _run_cmd_last(n=5, session=SID, monkeypatch=monkeypatch, tmp_path=tmp_path)
    assert rc == 0
    assert "hello live fallback world" in out


def test_cmd_last_daemon_down_role_filter(tmp_path, monkeypatch):
    _write_live_file(tmp_path, SID, [
        {"text": "user text here", "role": "user"},
        {"text": "assistant reply here", "role": "assistant"},
    ])
    rc, out = _run_cmd_last(n=10, session=SID, monkeypatch=monkeypatch, tmp_path=tmp_path)
    assert rc == 0
    assert "user text here" in out
    assert "assistant reply here" not in out


def test_cmd_last_daemon_down_empty_live_shows_no_turns_found(tmp_path, monkeypatch):
    rc, out = _run_cmd_last(n=5, session=SID, monkeypatch=monkeypatch, tmp_path=tmp_path)
    assert rc == 0
    assert "(no user turns found)" in out


def test_cmd_last_daemon_down_n_zero_shows_empty(tmp_path, monkeypatch):
    _write_live_file(tmp_path, SID, [
        {"text": "some user turn", "role": "user"},
    ])
    rc, out = _run_cmd_last(n=0, session=SID, monkeypatch=monkeypatch, tmp_path=tmp_path)
    assert rc == 0
    assert "(no user turns found)" in out
    assert "some user turn" not in out


def test_cmd_last_daemon_down_format_matches_daemon_path(tmp_path, monkeypatch):
    ts_iso = "2026-05-31T12:34:56+00:00"
    _write_live_file(tmp_path, SID, [
        {"text": "format check text", "role": "user", "ts": ts_iso},
    ])
    rc, out = _run_cmd_last(n=5, session=SID, monkeypatch=monkeypatch, tmp_path=tmp_path)
    assert rc == 0
    expected_ts = ts_iso[:19]
    expected_sid = SID[:8]
    assert expected_ts in out
    assert expected_sid in out
    assert "format check text" in out
    assert f"[{expected_ts}] {expected_sid}:" in out
