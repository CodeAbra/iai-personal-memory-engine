from __future__ import annotations

import json
import sys

if sys.platform != "win32":
    import resource
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from iai_mcp.daemon import _raise_fd_limit
from iai_mcp.fsm_reconcile import _CANONICAL_TO_LEGACY, reconcile_fsm_state
from iai_mcp.s2_coordinator import S2Coordinator


@pytest.mark.skipif(sys.platform == "win32", reason="resource module not available on Windows")
class TestRaiseFdLimitClampsToHard:

    def test_raises_low_soft_to_floor(self):
        fake_soft = 128
        fake_hard = 65536
        calls = []

        def fake_setrlimit(res, limits):
            calls.append((res, limits))

        with (
            patch("resource.getrlimit", return_value=(fake_soft, fake_hard)),
            patch("resource.setrlimit", side_effect=fake_setrlimit),
        ):
            _raise_fd_limit()

        assert len(calls) == 1
        _res, (new_soft, new_hard) = calls[0]
        assert new_soft >= 8192
        assert new_soft <= fake_hard
        assert new_hard == fake_hard

    def test_does_not_lower_already_high_soft(self):
        fake_soft = 32768
        fake_hard = 65536
        calls = []

        def fake_setrlimit(res, limits):
            calls.append((res, limits))

        with (
            patch("resource.getrlimit", return_value=(fake_soft, fake_hard)),
            patch("resource.setrlimit", side_effect=fake_setrlimit),
        ):
            _raise_fd_limit()

        for _res, (new_soft, _) in calls:
            assert new_soft >= fake_soft

    def test_clamped_by_hard_limit(self):
        fake_soft = 64
        fake_hard = 256
        calls = []

        def fake_setrlimit(res, limits):
            calls.append((res, limits))

        with (
            patch("resource.getrlimit", return_value=(fake_soft, fake_hard)),
            patch("resource.setrlimit", side_effect=fake_setrlimit),
        ):
            _raise_fd_limit()

        assert len(calls) == 1
        _res, (new_soft, new_hard) = calls[0]
        assert new_soft <= fake_hard

    def test_infinity_hard_does_not_request_huge_value(self):
        fake_soft = 128
        fake_hard = resource.RLIM_INFINITY
        calls = []

        def fake_setrlimit(res, limits):
            calls.append((res, limits))

        with (
            patch("resource.getrlimit", return_value=(fake_soft, fake_hard)),
            patch("resource.setrlimit", side_effect=fake_setrlimit),
        ):
            _raise_fd_limit()

        assert len(calls) == 1
        _res, (new_soft, new_hard) = calls[0]
        assert new_soft != resource.RLIM_INFINITY
        assert new_soft >= 8192
        assert new_hard == fake_hard

    def test_setrlimit_failure_is_swallowed(self):
        with (
            patch("resource.getrlimit", return_value=(64, 65536)),
            patch("resource.setrlimit", side_effect=OSError("permission denied")),
        ):
            _raise_fd_limit()

    def test_setrlimit_value_error_is_swallowed(self):
        with (
            patch("resource.getrlimit", return_value=(64, 65536)),
            patch("resource.setrlimit", side_effect=ValueError("bad value")),
        ):
            _raise_fd_limit()

    def test_env_tunable_floor(self, monkeypatch):
        monkeypatch.setenv("IAI_MCP_DAEMON_NOFILE_FLOOR", "16384")
        fake_soft = 64
        fake_hard = 65536
        calls = []

        def fake_setrlimit(res, limits):
            calls.append((res, limits))

        with (
            patch("resource.getrlimit", return_value=(fake_soft, fake_hard)),
            patch("resource.setrlimit", side_effect=fake_setrlimit),
        ):
            _raise_fd_limit()

        assert len(calls) == 1
        _res, (new_soft, _new_hard) = calls[0]
        assert new_soft >= 16384


class TestPlistRendersFdFloor:

    def test_plist_template_contains_fd_key(self):
        from iai_mcp.cli import _launchd_template

        text = _launchd_template().read_text()
        assert "SoftResourceLimits" in text
        assert "NumberOfFiles" in text

    def test_rendered_plist_contains_fd_floor(self, tmp_path, monkeypatch):
        import importlib
        import os

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USER", "testuser")

        from iai_mcp.cli import _render_launchd_plist

        rendered = _render_launchd_plist()
        assert "SoftResourceLimits" in rendered
        assert "NumberOfFiles" in rendered

        try:
            import defusedxml.ElementTree as ET
        except ModuleNotFoundError:
            import xml.etree.ElementTree as ET

        root = ET.fromstring(rendered)
        top_dict = root.find("dict")
        assert top_dict is not None

        keys = [el.text for el in top_dict.findall("key")]
        assert "SoftResourceLimits" in keys

        idx = list(top_dict).index(
            next(el for el in top_dict if el.tag == "key" and el.text == "SoftResourceLimits")
        )
        sub_dict = list(top_dict)[idx + 1]
        assert sub_dict.tag == "dict"

        num_el = sub_dict.find("integer")
        assert num_el is not None
        assert int(num_el.text) >= 8192

    def test_rendered_plist_preserves_python_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USER", "testuser")

        from iai_mcp.cli import _render_launchd_plist

        rendered = _render_launchd_plist()
        assert sys.executable in rendered
        assert "/usr/local/bin/python3" not in rendered

    def test_rendered_plist_preserves_watchdog_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USER", "testuser")

        from iai_mcp.cli import _render_launchd_plist

        rendered = _render_launchd_plist()
        assert "IAI_MCP_WATCHDOG_LIVENESS_POLL_SEC" in rendered


def _write_canonical(path: Path, state: str) -> None:
    path.write_text(
        json.dumps(
            {
                "current_state": state,
                "since_ts": "2026-06-01T00:00:00+00:00",
                "last_activity_ts": "2026-06-01T00:00:00+00:00",
                "wrapper_event_seq": 0,
                "sleep_cycle_progress": None,
                "quarantine": None,
                "shadow_run": False,
                "crisis_mode": False,
            }
        )
    )


def _write_legacy(path: Path, fsm_state: str) -> None:
    path.write_text(json.dumps({"fsm_state": fsm_state}))


class TestNoFsmDriftAfterCanonicalTransitions:

    def test_canonical_wake_legacy_mirrors_wake(self, tmp_path):
        canonical_path = tmp_path / "lifecycle_state.json"
        legacy_path = tmp_path / ".daemon-state.json"

        _write_canonical(canonical_path, "WAKE")
        _write_legacy(legacy_path, "WAKE")

        report = reconcile_fsm_state(canonical_path, legacy_path)
        assert report["drift"] is False

    def test_canonical_drowsy_legacy_mirrors_transitioning(self, tmp_path):
        canonical_path = tmp_path / "lifecycle_state.json"
        legacy_path = tmp_path / ".daemon-state.json"

        _write_canonical(canonical_path, "DROWSY")
        _write_legacy(legacy_path, "TRANSITIONING")

        report = reconcile_fsm_state(canonical_path, legacy_path)
        assert report["drift"] is False

    def test_canonical_sleep_legacy_mirrors_sleep(self, tmp_path):
        canonical_path = tmp_path / "lifecycle_state.json"
        legacy_path = tmp_path / ".daemon-state.json"

        _write_canonical(canonical_path, "SLEEP")
        _write_legacy(legacy_path, "SLEEP")

        report = reconcile_fsm_state(canonical_path, legacy_path)
        assert report["drift"] is False

    def test_canonical_to_legacy_mapping_is_complete(self):
        from iai_mcp.lifecycle_state import LifecycleState

        for state in LifecycleState:
            assert state.value in _CANONICAL_TO_LEGACY, (
                f"Missing mapping for canonical state {state.value}"
            )

    def test_s2_coordinator_writes_legacy_mirror_on_transition(self, tmp_path):
        import asyncio
        from iai_mcp.lifecycle_state import LifecycleState, default_state, save_state as ls_save

        canonical_path = tmp_path / "lifecycle_state.json"
        legacy_path = tmp_path / ".daemon-state.json"

        initial = default_state()
        ls_save(initial, canonical_path)
        _write_legacy(legacy_path, "WAKE")

        store_mock = MagicMock()
        store_mock.root = tmp_path

        coord = S2Coordinator(
            store=store_mock,
            state_path=canonical_path,
            legacy_path=legacy_path,
        )

        asyncio.run(
            coord.transition(
                LifecycleState.WAKE,
                LifecycleState.DROWSY,
                "test_idle_5min",
            )
        )

        report = reconcile_fsm_state(canonical_path, legacy_path)
        assert report["drift"] is False, (
            f"Expected no drift after WAKE→DROWSY transition, got: {report}"
        )
        assert report["legacy"] == "TRANSITIONING"


class TestReconcileStillFlagsRealDrift:

    def test_canonical_wake_legacy_sleep_is_drift(self, tmp_path):
        canonical_path = tmp_path / "lifecycle_state.json"
        legacy_path = tmp_path / ".daemon-state.json"

        _write_canonical(canonical_path, "WAKE")
        _write_legacy(legacy_path, "SLEEP")

        report = reconcile_fsm_state(canonical_path, legacy_path)
        assert report["drift"] is True

    def test_canonical_sleep_legacy_wake_is_drift(self, tmp_path):
        canonical_path = tmp_path / "lifecycle_state.json"
        legacy_path = tmp_path / ".daemon-state.json"

        _write_canonical(canonical_path, "SLEEP")
        _write_legacy(legacy_path, "WAKE")

        report = reconcile_fsm_state(canonical_path, legacy_path)
        assert report["drift"] is True

    def test_canonical_drowsy_legacy_dreaming_is_drift(self, tmp_path):
        canonical_path = tmp_path / "lifecycle_state.json"
        legacy_path = tmp_path / ".daemon-state.json"

        _write_canonical(canonical_path, "DROWSY")
        _write_legacy(legacy_path, "DREAMING")

        report = reconcile_fsm_state(canonical_path, legacy_path)
        assert report["drift"] is True
