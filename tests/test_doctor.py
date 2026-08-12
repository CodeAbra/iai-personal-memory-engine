from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from iai_mcp.idle_detector import IdleStatus


@pytest.fixture
def wrappers_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    wdir = tmp_path / "wrappers"
    wdir.mkdir(parents=True)
    return wdir


def _write_fresh_heartbeat(wrappers_dir: Path, pid: int, uuid: str) -> Path:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    path = wrappers_dir / f"heartbeat-{pid}-{uuid}.json"
    path.write_text(
        json.dumps(
            {
                "pid": pid,
                "uuid": uuid,
                "started_at": now,
                "last_refresh": now,
                "wrapper_version": "1.0.0",
                "schema_version": 1,
            }
        )
    )
    return path


def test_doctor_row_m_heartbeat_scanner_with_fresh_wrappers(
    wrappers_dir: Path,
) -> None:
    own_pid = os.getpid()
    _write_fresh_heartbeat(wrappers_dir, own_pid, "uuid-aaa")
    _write_fresh_heartbeat(wrappers_dir, own_pid, "uuid-bbb")

    from iai_mcp.doctor import check_m_heartbeat_scanner

    result = check_m_heartbeat_scanner()
    assert result.status == "PASS"
    assert result.passed is True
    assert "n=2 fresh" in result.detail
    assert "0 stale" in result.detail
    assert "0 orphan" in result.detail


def test_doctor_row_m_heartbeat_scanner_empty(wrappers_dir: Path) -> None:
    from iai_mcp.doctor import check_m_heartbeat_scanner

    result = check_m_heartbeat_scanner()
    assert result.status == "PASS"
    assert result.passed is True
    assert "n=0 fresh" in result.detail


def test_doctor_row_m_heartbeat_scanner_dir_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.doctor import check_m_heartbeat_scanner

    result = check_m_heartbeat_scanner()
    assert result.status == "PASS"
    assert result.passed is True
    assert "not present yet" in result.detail


def test_doctor_row_n_hid_idle_source_macos() -> None:
    fake_status = IdleStatus(
        hid_idle_sec=612,
        pmset_recent_sleep=False,
        available_signals=["HIDIdleTime", "pmset"],
    )

    with patch(
        "iai_mcp.idle_detector.IdleDetector.status",
        return_value=fake_status,
    ):
        from iai_mcp.doctor import check_n_hid_idle_source

        result = check_n_hid_idle_source()

    assert result.status == "PASS"
    assert result.passed is True
    assert "HIDIdleTime: 612s" in result.detail
    assert "pmset: clean" in result.detail
    assert "HIDIdleTime" in result.detail


def test_doctor_row_n_hid_idle_source_linux_logind() -> None:
    fake_status = IdleStatus(
        hid_idle_sec=None,
        pmset_recent_sleep=False,
        available_signals=["logind"],
    )

    with patch(
        "iai_mcp.idle_detector.IdleDetector.status",
        return_value=fake_status,
    ):
        from iai_mcp.doctor import check_n_hid_idle_source

        result = check_n_hid_idle_source()

    assert result.status == "PASS"
    assert result.passed is True
    assert "logind IdleHint: not idle" in result.detail
    assert "available: logind" in result.detail


def test_doctor_row_n_hid_idle_source_missing() -> None:
    fake_status = IdleStatus(
        hid_idle_sec=None,
        pmset_recent_sleep=False,
        available_signals=[],
    )

    with patch(
        "iai_mcp.idle_detector.IdleDetector.status",
        return_value=fake_status,
    ):
        from iai_mcp.doctor import check_n_hid_idle_source

        result = check_n_hid_idle_source()

    assert result.status == "WARN"
    assert result.passed is True
    assert "HIDIdleTime: unavailable" in result.detail
    assert "available: none" in result.detail
    assert "fall back to heartbeat-idle only" in result.detail


def test_run_diagnosis_includes_rows_m_and_n(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.doctor import run_diagnosis

    results = run_diagnosis()
    names = [r.name for r in results]

    m_rows = [r for r in results if "(m)" in r.name]
    n_rows = [r for r in results if "(n)" in r.name]
    assert len(m_rows) == 1, f"expected exactly one (m) row, got {names}"
    assert len(n_rows) == 1, f"expected exactly one (n) row, got {names}"
    assert names.index(m_rows[0].name) < names.index(n_rows[0].name)


@pytest.fixture
def lifecycle_state_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Path:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    return tmp_path


def test_doctor_row_j_lifecycle_state_default_when_absent(
    lifecycle_state_root: Path,
) -> None:
    from iai_mcp.doctor import check_j_lifecycle_current_state

    result = check_j_lifecycle_current_state()
    assert result.status == "PASS"
    assert result.passed is True
    assert "WAKE" in result.detail
    assert "shadow_run=" in result.detail


def test_doctor_row_j_lifecycle_state_reports_drowsy(
    lifecycle_state_root: Path,
) -> None:
    from iai_mcp.lifecycle_state import save_state

    record = {
        "current_state": "DROWSY",
        "since_ts": "2026-05-02T15:00:00+00:00",
        "last_activity_ts": "2026-05-02T15:00:00+00:00",
        "wrapper_event_seq": 7,
        "sleep_cycle_progress": None,
        "quarantine": None,
        "shadow_run": False,
    }
    save_state(record, lifecycle_state_root / "lifecycle_state.json")

    from iai_mcp.doctor import check_j_lifecycle_current_state

    result = check_j_lifecycle_current_state()
    assert result.status == "PASS"
    assert "DROWSY" in result.detail
    assert "shadow_run=false" in result.detail


def test_doctor_row_k_lifecycle_history_24h_no_log(
    lifecycle_state_root: Path,
) -> None:
    from iai_mcp.doctor import check_k_lifecycle_history_24h

    result = check_k_lifecycle_history_24h()
    assert result.status == "PASS"
    assert "no event log" in result.detail


def test_doctor_row_k_lifecycle_history_24h_zero_transitions(
    lifecycle_state_root: Path,
) -> None:
    (lifecycle_state_root / "logs").mkdir()
    from iai_mcp.doctor import check_k_lifecycle_history_24h

    result = check_k_lifecycle_history_24h()
    assert result.status == "PASS"
    assert "0 transitions" in result.detail


def test_doctor_row_k_lifecycle_history_24h_counts_transitions(
    lifecycle_state_root: Path,
) -> None:
    from iai_mcp.lifecycle_event_log import LifecycleEventLog

    log = LifecycleEventLog(log_dir=lifecycle_state_root / "logs")
    log.append(
        {"event": "state_transition", "from": "WAKE", "to": "DROWSY",
         "trigger": "idle_5min"}
    )
    log.append(
        {"event": "state_transition", "from": "DROWSY", "to": "WAKE",
         "trigger": "heartbeat_refresh"}
    )
    log.append(
        {"event": "state_transition", "from": "DROWSY", "to": "SLEEP",
         "trigger": "idle_30min"}
    )
    log.append({"event": "wrapper_event", "kind": "boot"})

    from iai_mcp.doctor import check_k_lifecycle_history_24h

    result = check_k_lifecycle_history_24h()
    assert result.status == "PASS"
    assert "3 transitions" in result.detail
    assert "DROWSY=" in result.detail
    assert "WAKE=" in result.detail
    assert "SLEEP=" in result.detail


def test_doctor_row_l_quarantine_none_passes(
    lifecycle_state_root: Path,
) -> None:
    from iai_mcp.doctor import check_l_sleep_cycle_status

    result = check_l_sleep_cycle_status()
    assert result.status == "PASS"
    assert "no quarantine" in result.detail


def test_doctor_row_l_quarantine_active_short_warns(
    lifecycle_state_root: Path,
) -> None:
    from datetime import datetime as _dt
    from datetime import timedelta as _td
    from datetime import timezone as _tz

    from iai_mcp.lifecycle_state import save_state

    now = _dt.now(_tz.utc)
    since = (now - _td(hours=2)).isoformat()
    until = (now + _td(hours=22)).isoformat()
    record = {
        "current_state": "WAKE",
        "since_ts": now.isoformat(),
        "last_activity_ts": now.isoformat(),
        "wrapper_event_seq": 0,
        "sleep_cycle_progress": None,
        "quarantine": {
            "since_ts": since,
            "until_ts": until,
            "reason": "sleep step 3 (DREAM_DECAY) failed 3x",
        },
        "shadow_run": False,
    }
    save_state(record, lifecycle_state_root / "lifecycle_state.json")

    from iai_mcp.doctor import check_l_sleep_cycle_status

    result = check_l_sleep_cycle_status()
    assert result.status == "WARN"
    assert result.passed is True
    assert "quarantined" in result.detail
    assert "DREAM_DECAY" in result.detail


def test_doctor_row_l_quarantine_active_long_fails(
    lifecycle_state_root: Path,
) -> None:
    from datetime import datetime as _dt
    from datetime import timedelta as _td
    from datetime import timezone as _tz

    from iai_mcp.lifecycle_state import save_state

    now = _dt.now(_tz.utc)
    since = (now - _td(hours=14)).isoformat()
    until = (now + _td(hours=10)).isoformat()
    record = {
        "current_state": "WAKE",
        "since_ts": now.isoformat(),
        "last_activity_ts": now.isoformat(),
        "wrapper_event_seq": 0,
        "sleep_cycle_progress": None,
        "quarantine": {
            "since_ts": since,
            "until_ts": until,
            "reason": "sleep step 4 (OPTIMIZE_HIPPO) failed 3x",
        },
        "shadow_run": False,
    }
    save_state(record, lifecycle_state_root / "lifecycle_state.json")

    from iai_mcp.doctor import check_l_sleep_cycle_status

    result = check_l_sleep_cycle_status()
    assert result.status == "FAIL"
    assert result.passed is False
    assert "reset-quarantine" in result.detail


def test_doctor_row_l_quarantine_expired_passes(
    lifecycle_state_root: Path,
) -> None:
    from datetime import datetime as _dt
    from datetime import timedelta as _td
    from datetime import timezone as _tz

    from iai_mcp.lifecycle_state import save_state

    now = _dt.now(_tz.utc)
    since = (now - _td(hours=25)).isoformat()
    until = (now - _td(hours=1)).isoformat()
    record = {
        "current_state": "WAKE",
        "since_ts": now.isoformat(),
        "last_activity_ts": now.isoformat(),
        "wrapper_event_seq": 0,
        "sleep_cycle_progress": None,
        "quarantine": {
            "since_ts": since,
            "until_ts": until,
            "reason": "sleep step 5 (HIPPO_CLEANUP) failed 3x",
        },
        "shadow_run": False,
    }
    save_state(record, lifecycle_state_root / "lifecycle_state.json")

    from iai_mcp.doctor import check_l_sleep_cycle_status

    result = check_l_sleep_cycle_status()
    assert result.status == "PASS"
    assert "expired" in result.detail


def test_run_diagnosis_includes_rows_j_k_l_in_order(
    lifecycle_state_root: Path,
) -> None:
    from iai_mcp.doctor import run_diagnosis

    results = run_diagnosis()
    names = [r.name for r in results]

    assert len(results) == 30, f"expected 30 rows, got {len(results)}: {names}"
    assert any(r.name.startswith("(u)") for r in results), f"missing (u) recall centrality regression row: {names}"

    j_idx = next(i for i, r in enumerate(results) if "(j)" in r.name)
    k_idx = next(i for i, r in enumerate(results) if "(k)" in r.name)
    l_idx = next(i for i, r in enumerate(results) if "(l)" in r.name)
    m_idx = next(i for i, r in enumerate(results) if "(m)" in r.name)

    assert j_idx < k_idx < l_idx < m_idx, (
        f"row order broken: j={j_idx} k={k_idx} l={l_idx} m={m_idx}"
    )
