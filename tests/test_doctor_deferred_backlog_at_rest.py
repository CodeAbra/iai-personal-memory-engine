from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest


@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    yield tmp_path


def _write_live_file(deferred_dir: Path, session_id: str, events: list[dict]) -> Path:
    deferred_dir.mkdir(parents=True, exist_ok=True)
    path = deferred_dir / f"{session_id}.live.jsonl"
    header = {
        "version": 1,
        "deferred_at": "2026-05-31T04:45:00.000000+00:00",
        "session_id": session_id,
        "cwd": "/tmp/test",
    }
    lines = [json.dumps(header, ensure_ascii=False)]
    for ev in events:
        lines.append(json.dumps(ev, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_rotated_file(deferred_dir: Path, name: str, n_events: int = 2) -> Path:
    deferred_dir.mkdir(parents=True, exist_ok=True)
    path = deferred_dir / name
    header = {
        "version": 1,
        "deferred_at": "2026-05-31T04:45:00.000000+00:00",
        "session_id": "rotated-session",
        "cwd": "/tmp/test",
    }
    lines = [json.dumps(header, ensure_ascii=False)]
    for i in range(n_events):
        lines.append(
            json.dumps(
                {
                    "text": f"rotated backlog event {i}",
                    "cue": "rotated backlog turn",
                    "tier": "episodic",
                    "role": "user",
                    "ts": "2026-05-31T04:45:43.000000+00:00",
                },
                ensure_ascii=False,
            )
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def test_absent_spool_dir_passes(iai_home):
    from iai_mcp.doctor._storage_checks import check_ee_deferred_capture_backlog_at_rest

    result = check_ee_deferred_capture_backlog_at_rest()
    assert result.passed is True


def test_empty_spool_passes(iai_home):
    from iai_mcp.capture import deferred_captures_dir
    from iai_mcp.doctor._storage_checks import check_ee_deferred_capture_backlog_at_rest

    deferred_captures_dir().mkdir(parents=True, exist_ok=True)
    result = check_ee_deferred_capture_backlog_at_rest()
    assert result.passed is True


def test_rotated_backlog_file_fails(iai_home):
    from iai_mcp.capture import deferred_captures_dir
    from iai_mcp.doctor._storage_checks import check_ee_deferred_capture_backlog_at_rest

    deferred_dir = deferred_captures_dir()
    _write_rotated_file(deferred_dir, "rotated-backlog.jsonl", n_events=2)

    result = check_ee_deferred_capture_backlog_at_rest()
    assert result.passed is False
    assert result.status == "FAIL"
    assert "2" in result.detail


def test_live_spool_within_cadence_window_does_not_false_positive(iai_home):
    from iai_mcp.capture import deferred_captures_dir
    from iai_mcp.doctor._storage_checks import check_ee_deferred_capture_backlog_at_rest

    deferred_dir = deferred_captures_dir()
    _write_live_file(
        deferred_dir,
        "sess-fresh-active",
        [
            {
                "text": "fresh active wake-sweep lag event one",
                "cue": "active session turn",
                "tier": "episodic",
                "role": "user",
                "ts": "2026-05-31T04:45:43.000000+00:00",
            },
            {
                "text": "fresh active wake-sweep lag event two",
                "cue": "active session turn",
                "tier": "episodic",
                "role": "user",
                "ts": "2026-05-31T04:45:44.000000+00:00",
            },
        ],
    )

    result = check_ee_deferred_capture_backlog_at_rest()
    assert result.passed is True


def test_live_spool_stale_and_pending_fails(iai_home):
    from iai_mcp.capture import deferred_captures_dir
    from iai_mcp.doctor._storage_checks import check_ee_deferred_capture_backlog_at_rest

    deferred_dir = deferred_captures_dir()
    path = _write_live_file(
        deferred_dir,
        "sess-stale-pending",
        [
            {
                "text": "stranded live spool event one",
                "cue": "stranded session turn",
                "tier": "episodic",
                "role": "user",
                "ts": "2026-05-31T04:45:43.000000+00:00",
            },
            {
                "text": "stranded live spool event two",
                "cue": "stranded session turn",
                "tier": "episodic",
                "role": "user",
                "ts": "2026-05-31T04:45:44.000000+00:00",
            },
        ],
    )
    old_ts = time.time() - 3600
    os.utime(path, (old_ts, old_ts))

    result = check_ee_deferred_capture_backlog_at_rest()
    assert result.passed is False
    assert result.status == "FAIL"
    assert "2" in result.detail


def test_live_spool_torn_trailing_line_not_double_counted(iai_home):
    """A torn (no trailing newline) partial final line must not be counted as
    a pending event: it mirrors the drain writer's own completeness rule, so
    doctor's count matches what the drain would actually consider drainable.
    """
    from iai_mcp.capture import deferred_captures_dir
    from iai_mcp.doctor._storage_checks import check_ee_deferred_capture_backlog_at_rest

    deferred_dir = deferred_captures_dir()
    deferred_dir.mkdir(parents=True, exist_ok=True)
    path = deferred_dir / "sess-torn-tail.live.jsonl"
    header = {
        "version": 1,
        "deferred_at": "2026-05-31T04:45:00.000000+00:00",
        "session_id": "sess-torn-tail",
        "cwd": "/tmp/test",
    }
    complete_event = {
        "text": "one complete stranded event",
        "cue": "stranded session turn",
        "tier": "episodic",
        "role": "user",
        "ts": "2026-05-31T04:45:43.000000+00:00",
    }
    torn_event = {
        "text": "a partial write that never finished",
        "cue": "stranded session turn",
        "tier": "episodic",
        "role": "user",
        "ts": "2026-05-31T04:45:44.000000+00:00",
    }
    import json as _json

    content = (
        _json.dumps(header, ensure_ascii=False)
        + "\n"
        + _json.dumps(complete_event, ensure_ascii=False)
        + "\n"
        + _json.dumps(torn_event, ensure_ascii=False)  # deliberately no trailing "\n"
    )
    path.write_text(content)
    old_ts = time.time() - 3600
    os.utime(path, (old_ts, old_ts))

    result = check_ee_deferred_capture_backlog_at_rest()
    assert result.passed is False
    assert result.status == "FAIL"
    assert "1 undrained capture event" in result.detail, result.detail


def test_live_spool_stale_but_fully_drained_passes(iai_home):
    from iai_mcp.capture import capture_state_dir, deferred_captures_dir
    from iai_mcp.doctor._storage_checks import check_ee_deferred_capture_backlog_at_rest

    deferred_dir = deferred_captures_dir()
    path = _write_live_file(
        deferred_dir,
        "sess-stale-drained",
        [
            {
                "text": "fully drained live spool event one",
                "cue": "drained session turn",
                "tier": "episodic",
                "role": "user",
                "ts": "2026-05-31T04:45:43.000000+00:00",
            },
            {
                "text": "fully drained live spool event two",
                "cue": "drained session turn",
                "tier": "episodic",
                "role": "user",
                "ts": "2026-05-31T04:45:44.000000+00:00",
            },
        ],
    )
    old_ts = time.time() - 3600
    os.utime(path, (old_ts, old_ts))

    state_dir = capture_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "sess-stale-drained.drain-offset").write_text("2")

    result = check_ee_deferred_capture_backlog_at_rest()
    assert result.passed is True


def test_unreadable_backlog_dir_warns_not_crashes(iai_home, monkeypatch):
    from iai_mcp import deferred_drain
    from iai_mcp.capture import deferred_captures_dir
    from iai_mcp.doctor._storage_checks import check_ee_deferred_capture_backlog_at_rest

    deferred_captures_dir().mkdir(parents=True, exist_ok=True)

    def _raise_enumerate(_deferred_dir):
        raise OSError("permission denied")

    monkeypatch.setattr(deferred_drain, "enumerate_backlog", _raise_enumerate)

    result = check_ee_deferred_capture_backlog_at_rest()
    assert result.passed is True
    assert result.status == "WARN"


def test_check_ee_survives_headless_downgrade(iai_home):
    import iai_mcp.doctor as doctor_pkg
    from iai_mcp.capture import deferred_captures_dir
    from iai_mcp.doctor._storage_checks import check_ee_deferred_capture_backlog_at_rest

    deferred_dir = deferred_captures_dir()
    _write_rotated_file(deferred_dir, "rotated-backlog-headless.jsonl", n_events=1)

    result = check_ee_deferred_capture_backlog_at_rest()
    assert result.passed is False

    downgraded = doctor_pkg._apply_headless_downgrade([result], headless=True)
    assert downgraded[0].passed is False
    assert downgraded[0].status == "FAIL"
