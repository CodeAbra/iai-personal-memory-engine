from __future__ import annotations

import platform
from pathlib import Path

import pytest

from iai_mcp import cli as cli_mod
from iai_mcp import daemon as daemon_mod
from iai_mcp import daemon_state as state_mod


def test_clear_sentinel_no_disk_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_path = tmp_path / ".daemon-state.json"
    monkeypatch.setattr(state_mod, "STATE_PATH", state_path, raising=True)

    state: dict = {"fsm_state": "WAKE", "daemon_pid": 12345}
    snapshot = dict(state)
    daemon_mod._clear_user_shutdown_sentinel(state)
    assert state == snapshot


def test_clear_sentinel_true_on_disk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_path = tmp_path / ".daemon-state.json"
    monkeypatch.setattr(state_mod, "STATE_PATH", state_path, raising=True)
    state_mod.save_state(
        {"user_requested_shutdown": True, "fsm_state": "WAKE"}
    )

    daemon_in_memory: dict = {
        "fsm_state": "DREAMING",
        "daemon_pid": 999,
    }
    daemon_mod._clear_user_shutdown_sentinel(daemon_in_memory)

    on_disk = state_mod.load_state()
    assert "user_requested_shutdown" not in on_disk
    assert "user_requested_shutdown" not in daemon_in_memory


def test_clear_sentinel_preserves_unrelated_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_path = tmp_path / ".daemon-state.json"
    monkeypatch.setattr(state_mod, "STATE_PATH", state_path, raising=True)
    state_mod.save_state({"user_requested_shutdown": True, "fsm_state": "WAKE"})

    snapshot = {
        "fsm_state": "DREAMING",
        "daemon_pid": 42,
        "pending_digest": {"rem_cycles_completed": 79},
        "user_requested_shutdown": True,
        "fsm_transition_at": "2026-05-01T10:17:54+00:00",
    }
    state = dict(snapshot)
    daemon_mod._clear_user_shutdown_sentinel(state)
    expected = {
        k: v for k, v in snapshot.items() if k != "user_requested_shutdown"
    }
    assert state == expected


def test_clear_sentinel_disk_read_failure_is_fail_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    def boom() -> dict:
        raise OSError("simulated transient read error")

    monkeypatch.setattr(daemon_mod, "load_state", boom)

    state: dict = {"fsm_state": "WAKE", "user_requested_shutdown": True}
    daemon_mod._clear_user_shutdown_sentinel(state)
    assert "user_requested_shutdown" not in state


def test_e_cmd_daemon_stop_writes_sentinel_before_launchctl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")

    state_path = tmp_path / ".daemon-state.json"
    monkeypatch.setattr(state_mod, "STATE_PATH", state_path, raising=True)

    call_log: list[str] = []

    real_update_state = state_mod.update_state

    def tracking_update_state(mutator):
        result = real_update_state(mutator)
        call_log.append(f"save_state:{result.get('user_requested_shutdown')}")
        return result

    monkeypatch.setattr(state_mod, "update_state", tracking_update_state)

    def fake_run(argv, **_kwargs):
        call_log.append(f"subprocess.run:{argv[0]}:{argv[1]}")
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)

    import iai_mcp.lifecycle_lock as lifecycle_lock

    monkeypatch.setattr(lifecycle_lock.LifecycleLock, "read", lambda self: None)

    rc = cli_mod.main(["daemon", "stop"])
    assert rc == 0

    import json as json_mod
    persisted = json_mod.loads(state_path.read_text())
    assert persisted.get("user_requested_shutdown") is True

    assert call_log[0].startswith("save_state:True"), call_log
    assert any(
        entry.startswith("subprocess.run:launchctl") for entry in call_log
    ), call_log
    save_idx = next(
        i for i, e in enumerate(call_log) if e.startswith("save_state:")
    )
    launchctl_idx = next(
        i for i, e in enumerate(call_log)
        if e.startswith("subprocess.run:launchctl")
    )
    assert save_idx < launchctl_idx, call_log


def test_f_cmd_daemon_stop_writes_sentinel_before_systemctl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")

    state_path = tmp_path / ".daemon-state.json"
    monkeypatch.setattr(state_mod, "STATE_PATH", state_path, raising=True)

    call_log: list[str] = []

    real_update_state = state_mod.update_state

    def tracking_update_state(mutator):
        result = real_update_state(mutator)
        call_log.append(f"save_state:{result.get('user_requested_shutdown')}")
        return result

    monkeypatch.setattr(state_mod, "update_state", tracking_update_state)

    def fake_run(argv, **_kwargs):
        call_log.append(f"subprocess.run:{argv[0]}")
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)

    rc = cli_mod.main(["daemon", "stop"])
    assert rc == 0

    import json as json_mod
    persisted = json_mod.loads(state_path.read_text())
    assert persisted.get("user_requested_shutdown") is True

    save_idx = next(
        i for i, e in enumerate(call_log) if e.startswith("save_state:")
    )
    systemctl_idx = next(
        i for i, e in enumerate(call_log)
        if e.startswith("subprocess.run:systemctl")
    )
    assert save_idx < systemctl_idx, call_log


def test_g_user_shutdown_flag_constant_is_stable() -> None:
    assert daemon_mod._USER_SHUTDOWN_FLAG == "user_requested_shutdown"
