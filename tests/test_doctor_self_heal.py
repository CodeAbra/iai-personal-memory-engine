from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from iai_mcp.doctor import (
    CheckResult,
    DOCTOR_AUTO_COOLDOWN_SEC,
    RepairAction,
    _auto_recently_ran,
    _cleanup_stale_heartbeats,
    _plan_repair_actions,
    _quarantine_rename,
    _stamp_auto_run,
    _write_wake_signal,
    check_j_lifecycle_current_state,
)


def _result(name: str, passed: bool, detail: str = "", status: str = "") -> CheckResult:
    return CheckResult(name=name, passed=passed, detail=detail, status=status)


class TestCheckJParkedEngine:
    @pytest.fixture(autouse=True)
    def _isolated_store(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
        return tmp_path

    def _write_lifecycle(self, root: Path, state: str) -> None:
        from iai_mcp.lifecycle_state import default_state, save_state

        record = default_state()
        record["current_state"] = state
        record["shadow_run"] = False
        save_state(record, root / "lifecycle_state.json")

    def _write_daemon_state(self, root: Path, pid: int | None) -> None:
        payload: dict = {"fsm_state": "WAKE"}
        if pid is not None:
            payload["daemon_pid"] = pid
        (root / ".daemon-state.json").write_text(
            json.dumps(payload), encoding="utf-8",
        )

    def test_hibernation_with_dead_daemon_fails(self, _isolated_store: Path) -> None:
        self._write_lifecycle(_isolated_store, "HIBERNATION")
        self._write_daemon_state(_isolated_store, pid=None)
        result = check_j_lifecycle_current_state()
        assert result.passed is False
        assert result.status == "FAIL"
        assert "parked" in result.detail

    def test_sleep_with_dead_daemon_fails(self, _isolated_store: Path) -> None:
        self._write_lifecycle(_isolated_store, "SLEEP")
        self._write_daemon_state(_isolated_store, pid=None)
        result = check_j_lifecycle_current_state()
        assert result.passed is False

    def test_hibernation_with_live_daemon_passes(
        self, _isolated_store: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import os

        self._write_lifecycle(_isolated_store, "HIBERNATION")
        self._write_daemon_state(_isolated_store, pid=os.getpid())
        # The real probe demands an iai_mcp.daemon cmdline; the pytest pid
        # can never satisfy it, so pin the liveness answer.
        monkeypatch.setattr(
            "iai_mcp.lifecycle_lock._is_pid_alive", lambda pid: True,
        )
        result = check_j_lifecycle_current_state()
        assert result.passed is True

    def test_wake_with_dead_daemon_passes_here(self, _isolated_store: Path) -> None:
        # A dead daemon in WAKE is check (a)'s finding; (j) stays green so
        # the respawn heal (not the signal-first wake) is planned.
        self._write_lifecycle(_isolated_store, "WAKE")
        self._write_daemon_state(_isolated_store, pid=None)
        result = check_j_lifecycle_current_state()
        assert result.passed is True


class TestHealPrimitives:
    def test_quarantine_rename_moves_file_aside(self, tmp_path: Path) -> None:
        victim = tmp_path / "records.hnsw"
        victim.write_bytes(b"corrupt")
        ok, msg, _ = _quarantine_rename(victim)
        assert ok is True
        assert not victim.exists()
        survivors = list(tmp_path.glob("records.hnsw.corrupt-*"))
        assert len(survivors) == 1
        assert survivors[0].read_bytes() == b"corrupt"

    def test_quarantine_rename_absent_file_is_ok(self, tmp_path: Path) -> None:
        ok, msg, _ = _quarantine_rename(tmp_path / "nothing")
        assert ok is True
        assert "absent" in msg

    def test_write_wake_signal_creates_json_payload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
        ok, msg, _ = _write_wake_signal()
        assert ok is True
        payload = json.loads((tmp_path / "wake.signal").read_text())
        assert payload["source"] == "doctor"
        assert "requested_at" in payload

    def test_cleanup_stale_heartbeats_removes_dead_pid_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
        wrappers = tmp_path / "wrappers"
        wrappers.mkdir()
        stale = wrappers / "heartbeat-999999-dead.json"
        stale.write_text(json.dumps({
            "pid": 999999,
            "uuid": "dead",
            "last_refresh": "2020-01-01T00:00:00+00:00",
        }), encoding="utf-8")
        ok, msg, _ = _cleanup_stale_heartbeats()
        assert ok is True
        assert not stale.exists()


class TestAutoDamper:
    def test_fresh_stamp_blocks_next_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
        assert _auto_recently_ran() is False
        _stamp_auto_run()
        assert _auto_recently_ran() is True

    def test_stale_stamp_allows_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
        _stamp_auto_run()
        future = time.time() + DOCTOR_AUTO_COOLDOWN_SEC + 1.0
        assert _auto_recently_ran(now=future) is False


class TestRepairPlanning:
    def test_parked_engine_plans_signal_first_wake_not_respawn(self) -> None:
        results = [
            _result("(a) daemon process alive", False, "ABSENT"),
            _result(
                "(j) lifecycle current state", False,
                "HIBERNATION since 5 d; engine parked",
            ),
        ]
        actions = _plan_repair_actions(results)
        labels = [a.label for a in actions]
        assert "wake_parked_engine" in labels
        assert "respawn_daemon" not in labels, (
            "respawn without the signal would boot back into the parked state"
        )
        wake = next(a for a in actions if a.label == "wake_parked_engine")
        assert wake.auto_safe is True

    def test_dead_daemon_in_wake_plans_plain_respawn(self) -> None:
        results = [_result("(a) daemon process alive", False, "ABSENT")]
        labels = [a.label for a in _plan_repair_actions(results)]
        assert "respawn_daemon" in labels
        assert "wake_parked_engine" not in labels

    def test_corrupt_state_file_plans_quarantine(self) -> None:
        results = [_result(
            "(e) daemon state file valid", False, "unreadable: ValueError: bad json",
        )]
        labels = [a.label for a in _plan_repair_actions(results)]
        assert "quarantine_state_file" in labels

    def test_unknown_fsm_state_does_not_plan_quarantine(self) -> None:
        results = [_result(
            "(e) daemon state file valid", False, "fsm_state='LIMBO' not in [...]",
        )]
        labels = [a.label for a in _plan_repair_actions(results)]
        assert "quarantine_state_file" not in labels

    def test_corrupt_vec_index_plans_quarantine(self) -> None:
        results = [_result(
            "(r) hippo hnsw index", False, "records.hnsw is zero bytes (corrupt)",
        )]
        labels = [a.label for a in _plan_repair_actions(results)]
        assert "quarantine_vec_index" in labels

    def test_stale_heartbeats_plan_cleanup(self) -> None:
        results = [_result(
            "(m) heartbeat scanner", True, "n=1 fresh, 3 stale, 0 orphan",
            status="PASS",
        )]
        labels = [a.label for a in _plan_repair_actions(results)]
        assert "cleanup_stale_heartbeats" in labels

    def test_clean_heartbeats_plan_nothing(self) -> None:
        results = [_result(
            "(m) heartbeat scanner", True, "n=2 fresh, 0 stale, 0 orphan",
            status="PASS",
        )]
        labels = [a.label for a in _plan_repair_actions(results)]
        assert "cleanup_stale_heartbeats" not in labels

    def test_stuck_quarantine_plans_reset(self) -> None:
        results = [_result(
            "(l) sleep cycle quarantine", False, "quarantined for 14.0h",
        )]
        labels = [a.label for a in _plan_repair_actions(results)]
        assert "reset_sleep_quarantine" in labels

    def test_store_mutating_actions_are_never_auto_safe(self) -> None:
        results = [
            _result(
                "(w) no permanent-failed captures", True,
                "3 permanent-failed capture file(s)", status="WARN",
            ),
            _result(
                "(x) no collapsed-timestamp groups", True,
                "2 group(s) with >= 5 records at one timestamp", status="WARN",
            ),
            _result("(i) hippo db size", False, "2500.0 MB — run compaction"),
        ]
        actions = _plan_repair_actions(results)
        assert {a.label for a in actions} == {
            "drain_permanent_failed", "rederive_timestamps", "compact_hippo",
        }
        assert all(a.auto_safe is False for a in actions)

    def test_process_kills_are_never_auto_safe(self) -> None:
        results = [
            _result("(g) no dup binders", False, "2 processes bound"),
            _result("(d) no orphan iai_mcp.core procs", False, "1 found"),
        ]
        actions = _plan_repair_actions(results)
        assert len(actions) == 2
        assert all(a.auto_safe is False for a in actions)
        assert all(a.destructive is True for a in actions)


class TestRepairActionShape:
    def test_auto_safe_defaults_false(self) -> None:
        action = RepairAction(
            label="x", description="y", destructive=False,
            execute=lambda: (True, "", 0),
        )
        assert action.auto_safe is False


class TestAutoSafeAllowlist:
    """The unattended subset is a closed list. A new action becoming
    auto_safe must be a deliberate decision that updates this test."""

    def test_full_board_auto_safe_set_is_exactly_the_allowlist(self) -> None:
        results = [
            _result("(a) daemon process alive", False, "ABSENT"),
            _result("(b) socket file fresh", False, "unreachable"),
            _result("(d) no orphan iai_mcp.core procs", False, "1 found"),
            _result("(e) daemon state file valid", False, "unreadable: bad"),
            _result("(g) no dup binders", False, "2 bound"),
            _result("(i) hippo db size", False, "2500.0 MB"),
            _result(
                "(j) lifecycle current state", False, "HIBERNATION; parked",
            ),
            _result("(l) sleep cycle quarantine", False, "quarantined 14h"),
            _result(
                "(m) heartbeat scanner", True, "n=0 fresh, 2 stale, 1 orphan",
                status="PASS",
            ),
            _result("(r) hippo hnsw index", False, "zero bytes (corrupt)"),
            _result(
                "(w) no permanent-failed captures", True,
                "2 permanent-failed capture file(s)", status="WARN",
            ),
            _result(
                "(x) no collapsed-timestamp groups", True,
                "3 group(s) with >= 5 records", status="WARN",
            ),
        ]
        actions = _plan_repair_actions(results)
        auto = {a.label for a in actions if a.auto_safe}
        assert auto == {
            "unlink_stale_socket",  # only because (a) failed too
            "wake_parked_engine",
            "quarantine_state_file",
            "quarantine_vec_index",
            "cleanup_stale_heartbeats",
            "reset_sleep_quarantine",
        }

    def test_socket_unlink_not_auto_when_daemon_alive(self) -> None:
        # (b) alone fails on a 1s connect timeout — a busy daemon. The
        # unattended reflex must never sever a live daemon's socket.
        results = [_result("(b) socket file fresh", False, "unreachable")]
        actions = _plan_repair_actions(results)
        unlink = next(a for a in actions if a.label == "unlink_stale_socket")
        assert unlink.auto_safe is False


class TestCmdDoctorAuto:
    @pytest.fixture(autouse=True)
    def _isolated_store(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
        return tmp_path

    def _patch_board(
        self,
        monkeypatch: pytest.MonkeyPatch,
        results: list[CheckResult],
        actions: list[RepairAction],
    ) -> None:
        import iai_mcp.doctor as doc

        monkeypatch.setattr(doc, "run_diagnosis", lambda **kw: list(results))
        monkeypatch.setattr(doc, "_plan_repair_actions", lambda _r: list(actions))

    def test_cooldown_short_circuits_before_diagnosis(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import iai_mcp.doctor as doc

        _stamp_auto_run()
        calls = {"n": 0}

        def _boom(**kw) -> list[CheckResult]:
            calls["n"] += 1
            return []

        monkeypatch.setattr(doc, "run_diagnosis", _boom)
        assert doc.cmd_doctor_auto() == 0
        assert calls["n"] == 0

    def test_stamps_even_when_no_auto_heal_exists(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import iai_mcp.doctor as doc

        self._patch_board(
            monkeypatch,
            results=[_result("(z) AVX2 CPU support", False, "no AVX2")],
            actions=[],
        )
        assert doc.cmd_doctor_auto() == 1
        assert _auto_recently_ran() is True

    def test_runs_only_auto_safe_actions(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import iai_mcp.doctor as doc

        ran: list[str] = []

        def _mk(label: str, auto: bool) -> RepairAction:
            return RepairAction(
                label=label, description=label, destructive=False,
                execute=lambda: (ran.append(label) or True, "ok", 1),
                auto_safe=auto,
            )

        self._patch_board(
            monkeypatch,
            results=[_result("(a) daemon process alive", False, "ABSENT")],
            actions=[_mk("safe_one", True), _mk("manual_one", False)],
        )
        monkeypatch.setattr(doc, "_execute_action", lambda a: a.execute())
        doc.cmd_doctor_auto()
        assert ran == ["safe_one"]

    def test_exit_zero_when_rediagnosis_green(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import iai_mcp.doctor as doc

        boards = [
            [_result("(a) daemon process alive", False, "ABSENT")],
            [_result("(a) daemon process alive", True, "PID 1")],
        ]
        monkeypatch.setattr(doc, "run_diagnosis", lambda **kw: list(boards.pop(0)))
        heal = RepairAction(
            label="h", description="h", destructive=False,
            execute=lambda: (True, "ok", 1), auto_safe=True,
        )
        monkeypatch.setattr(doc, "_plan_repair_actions", lambda _r: [heal])
        monkeypatch.setattr(doc, "_execute_action", lambda a: a.execute())
        assert doc.cmd_doctor_auto() == 0

    def test_exit_one_when_fail_persists(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import iai_mcp.doctor as doc

        self._patch_board(
            monkeypatch,
            results=[_result("(a) daemon process alive", False, "ABSENT")],
            actions=[RepairAction(
                label="h", description="h", destructive=False,
                execute=lambda: (False, "no", 1), auto_safe=True,
            )],
        )
        monkeypatch.setattr(doc, "_execute_action", lambda a: a.execute())
        assert doc.cmd_doctor_auto() == 1
