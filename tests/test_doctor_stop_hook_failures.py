"""(kk) stop-hook failure marker doctor check: PASS when the marker is
absent or holds only failures outside the recent window; WARN/FAIL on
recent failures; a malformed line is skipped without crashing the check."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    yield tmp_path


def _marker_path(home: Path) -> Path:
    return home / ".iai-mcp" / ".capture-state" / ".stop-hook-failures.jsonl"


def _write_marker(home: Path, lines: list[str]) -> Path:
    path = _marker_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def _entry(*, ts: datetime, session_id: str = "sess-1", rc: str = "1", channel: str = "settings") -> str:
    return json.dumps({
        "ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "session_id": session_id,
        "rc": rc,
        "channel": channel,
    })


def test_no_marker_file_passes(iai_home):
    from iai_mcp.doctor._storage_checks import check_kk_stop_hook_failures

    result = check_kk_stop_hook_failures()
    assert result.passed is True
    assert result.status == "PASS"


def test_only_old_failures_pass(iai_home):
    from iai_mcp.doctor._storage_checks import check_kk_stop_hook_failures

    old_ts = datetime.now(timezone.utc) - timedelta(hours=3)
    _write_marker(iai_home, [_entry(ts=old_ts), _entry(ts=old_ts, session_id="sess-2")])

    result = check_kk_stop_hook_failures()
    assert result.passed is True
    assert result.status == "PASS"


def test_single_recent_failure_warns(iai_home):
    from iai_mcp.doctor._storage_checks import check_kk_stop_hook_failures

    now = datetime.now(timezone.utc)
    _write_marker(iai_home, [_entry(ts=now)])

    result = check_kk_stop_hook_failures()
    assert result.status == "WARN"
    assert "1" in result.detail


def test_multiple_recent_failures_fail_with_count(iai_home):
    from iai_mcp.doctor._storage_checks import check_kk_stop_hook_failures

    now = datetime.now(timezone.utc)
    _write_marker(iai_home, [
        _entry(ts=now, session_id="sess-a"),
        _entry(ts=now, session_id="sess-b", rc="cli-not-found"),
    ])

    result = check_kk_stop_hook_failures()
    assert result.passed is False
    assert result.status == "FAIL"
    assert "2" in result.detail


def test_malformed_line_is_skipped_without_crashing(iai_home):
    from iai_mcp.doctor._storage_checks import check_kk_stop_hook_failures

    now = datetime.now(timezone.utc)
    _write_marker(iai_home, [
        "{not valid json",
        _entry(ts=now, session_id="sess-a"),
        _entry(ts=now, session_id="sess-b"),
        "",
    ])

    result = check_kk_stop_hook_failures()
    assert result.status == "FAIL"
    assert "2" in result.detail


def test_entry_missing_ts_field_is_skipped(iai_home):
    from iai_mcp.doctor._storage_checks import check_kk_stop_hook_failures

    _write_marker(iai_home, [json.dumps({"session_id": "sess-1", "rc": "1"})])

    result = check_kk_stop_hook_failures()
    assert result.status == "PASS"
