from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pytest

from iai_mcp import version_check as vc
from iai_mcp.cli._daemon import cmd_self_update


class TestVersionParseCompare:
    def test_newer_patch(self) -> None:
        assert vc.is_newer("2.7.4", "2.7.3") is True

    def test_newer_minor_beats_patch(self) -> None:
        assert vc.is_newer("2.8.0", "2.7.9") is True

    def test_equal_is_not_newer(self) -> None:
        assert vc.is_newer("2.7.3", "2.7.3") is False

    def test_older_is_not_newer(self) -> None:
        assert vc.is_newer("2.7.2", "2.7.3") is False

    def test_longer_tuple_wins(self) -> None:
        assert vc.is_newer("2.7.3.1", "2.7.3") is True

    def test_unparseable_never_nags(self) -> None:
        assert vc.is_newer("garbage", "2.7.3") is False
        assert vc.is_newer("2.8.0", "garbage") is False

    def test_prerelease_latest_never_nags(self) -> None:
        # The upgrade pins ==latest and an explicit pin installs
        # prereleases — a non-clean latest must never register as newer.
        assert vc.is_newer("2.8.0rc1", "2.7.3") is False
        assert vc.is_newer("2.7.3rc1", "2.7.3") is False

    def test_dev_suffixed_current_still_gets_clean_upgrades(self) -> None:
        assert vc.is_newer("2.8.0", "2.7.3.dev0") is True


class TestCacheAndKillSwitch:
    @pytest.fixture(autouse=True)
    def _isolated(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
        return tmp_path

    def test_kill_switch_blocks_fetch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(vc.ENV_TOGGLE, "0")
        called = {"n": 0}

        def _boom(*a, **k):
            called["n"] += 1

        monkeypatch.setattr("urllib.request.urlopen", _boom)
        assert vc.fetch_latest_version() is None
        assert called["n"] == 0

    def test_refresh_respects_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fetches = {"n": 0}

        def _fetch(timeout: float = 0) -> str:
            fetches["n"] += 1
            return "9.9.9"

        monkeypatch.setattr(vc, "fetch_latest_version", _fetch)
        first = vc.refresh_cache()
        second = vc.refresh_cache()
        assert fetches["n"] == 1
        assert first is not None and second is not None
        assert second["latest"] == "9.9.9"

    def test_offline_refresh_keeps_previous_latest_but_stamps(
        self, monkeypatch: pytest.MonkeyPatch, _isolated: Path,
    ) -> None:
        (_isolated / vc.CACHE_FILENAME).write_text(json.dumps({
            "checked_at": time.time() - vc.CHECK_TTL_SEC - 10,
            "latest": "9.9.9",
        }))
        monkeypatch.setattr(vc, "fetch_latest_version", lambda timeout=0: None)
        cache = vc.refresh_cache()
        assert cache is not None
        assert cache["latest"] == "9.9.9"
        assert time.time() - cache["checked_at"] < 60

    def test_pending_update_is_cache_only(
        self, monkeypatch: pytest.MonkeyPatch, _isolated: Path,
    ) -> None:
        def _boom(*a, **k):
            raise AssertionError("pending_update must never touch the network")

        monkeypatch.setenv(vc.ENV_TOGGLE, "1")
        monkeypatch.setattr(vc, "fetch_latest_version", _boom)
        monkeypatch.setattr(vc, "installed_version", lambda: "2.7.3")
        monkeypatch.setattr(vc, "is_editable_install", lambda: False)
        (_isolated / vc.CACHE_FILENAME).write_text(json.dumps({
            "checked_at": time.time(), "latest": "9.9.9",
        }))
        assert vc.pending_update() == ("2.7.3", "9.9.9")

    def test_kill_switch_mutes_pending_update_entirely(
        self, monkeypatch: pytest.MonkeyPatch, _isolated: Path,
    ) -> None:
        monkeypatch.setenv(vc.ENV_TOGGLE, "0")
        monkeypatch.setattr(vc, "installed_version", lambda: "2.7.3")
        monkeypatch.setattr(vc, "is_editable_install", lambda: False)
        (_isolated / vc.CACHE_FILENAME).write_text(json.dumps({
            "checked_at": time.time(), "latest": "9.9.9",
        }))
        assert vc.pending_update() is None

    def test_editable_install_never_nagged(
        self, monkeypatch: pytest.MonkeyPatch, _isolated: Path,
    ) -> None:
        monkeypatch.setenv(vc.ENV_TOGGLE, "1")
        monkeypatch.setattr(vc, "installed_version", lambda: "2.4.1")
        monkeypatch.setattr(vc, "is_editable_install", lambda: True)
        (_isolated / vc.CACHE_FILENAME).write_text(json.dumps({
            "checked_at": time.time(), "latest": "9.9.9",
        }))
        assert vc.pending_update() is None

    def test_this_repo_tree_reads_as_editable(self) -> None:
        # The guard's ground truth: a module living inside a repo tree with
        # pyproject.toml is a source checkout even without direct_url.json
        # (legacy egg-info editable installs carry none).
        assert vc.is_editable_install() is True

    def test_no_cache_means_no_notice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(vc.ENV_TOGGLE, "1")
        monkeypatch.setattr(vc, "installed_version", lambda: "2.7.3")
        monkeypatch.setattr(vc, "is_editable_install", lambda: False)
        assert vc.pending_update() is None
        assert vc.pending_update_line() is None


class TestSessionStartNotice:
    def _arm(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
        monkeypatch.setenv(vc.ENV_TOGGLE, "1")
        monkeypatch.setattr(vc, "installed_version", lambda: "2.7.3")
        monkeypatch.setattr(vc, "is_editable_install", lambda: False)
        (tmp_path / vc.CACHE_FILENAME).write_text(json.dumps({
            "checked_at": time.time(), "latest": "9.9.9",
        }))

    def test_markdown_carries_update_line_from_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from iai_mcp.session import format_payload_as_markdown

        self._arm(tmp_path, monkeypatch)
        out = format_payload_as_markdown({"l0": "identity"})
        assert "9.9.9" in out
        assert "self-update" in out

    def test_kill_switch_suppresses_the_notice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from iai_mcp.session import format_payload_as_markdown

        self._arm(tmp_path, monkeypatch)
        monkeypatch.setenv(vc.ENV_TOGGLE, "0")
        out = format_payload_as_markdown({"l0": "identity"})
        assert "self-update" not in out

    def test_notice_never_travels_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from iai_mcp.session import format_payload_as_markdown

        self._arm(tmp_path, monkeypatch)
        assert format_payload_as_markdown({}) == ""

    def test_markdown_silent_without_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from iai_mcp.session import format_payload_as_markdown

        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
        monkeypatch.setenv(vc.ENV_TOGGLE, "1")
        out = format_payload_as_markdown({"l0": "identity"})
        assert "self-update" not in out


class TestDoctorUpdateRow:
    @pytest.fixture(autouse=True)
    def _isolated(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
        return tmp_path

    def test_disabled_switch_is_a_quiet_pass(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from iai_mcp.doctor import check_plus_update_available

        monkeypatch.setenv(vc.ENV_TOGGLE, "0")
        row = check_plus_update_available()
        assert row.passed is True
        assert "disabled" in row.detail

    def test_editable_checkout_is_a_quiet_pass(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from iai_mcp.doctor import check_plus_update_available

        monkeypatch.setenv(vc.ENV_TOGGLE, "1")
        row = check_plus_update_available()
        assert row.passed is True
        assert "source checkout" in row.detail

    def test_pending_update_warns_and_names_the_command(
        self, monkeypatch: pytest.MonkeyPatch, _isolated: Path,
    ) -> None:
        from iai_mcp.doctor import check_plus_update_available

        monkeypatch.setenv(vc.ENV_TOGGLE, "1")
        monkeypatch.setattr(vc, "installed_version", lambda: "2.7.3")
        monkeypatch.setattr(vc, "is_editable_install", lambda: False)
        (_isolated / vc.CACHE_FILENAME).write_text(json.dumps({
            "checked_at": time.time(), "latest": "9.9.9",
        }))
        row = check_plus_update_available(fetch=False)
        assert row.passed is True
        assert row.status == "WARN"
        assert "self-update" in row.detail

    def test_fetch_false_never_touches_the_network(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from iai_mcp.doctor import check_plus_update_available

        monkeypatch.setenv(vc.ENV_TOGGLE, "1")
        monkeypatch.setattr(vc, "is_editable_install", lambda: False)

        def _boom(*a, **k):
            raise AssertionError("fetch=False must not refresh the cache")

        monkeypatch.setattr(vc, "refresh_cache", _boom)
        row = check_plus_update_available(fetch=False)
        assert row.passed is True


def _args(**kw) -> argparse.Namespace:
    ns = argparse.Namespace()
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


class TestCmdSelfUpdate:
    @pytest.fixture(autouse=True)
    def _isolated(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
        return tmp_path

    def test_refuses_editable_install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(vc, "installed_version", lambda: "2.7.3")
        monkeypatch.setattr(vc, "is_editable_install", lambda: True)
        assert cmd_self_update(_args(check=False, yes=True)) == 1

    def test_offline_is_an_error_not_an_upgrade(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(vc, "installed_version", lambda: "2.7.3")
        monkeypatch.setattr(vc, "is_editable_install", lambda: False)
        monkeypatch.setattr(vc, "fetch_latest_version", lambda timeout=0: None)
        assert cmd_self_update(_args(check=False, yes=True)) == 1

    def test_already_latest_exits_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(vc, "installed_version", lambda: "2.7.3")
        monkeypatch.setattr(vc, "is_editable_install", lambda: False)
        monkeypatch.setattr(vc, "fetch_latest_version", lambda timeout=0: "2.7.3")
        assert cmd_self_update(_args(check=False, yes=True)) == 0

    def test_check_only_changes_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(vc, "installed_version", lambda: "2.7.3")
        monkeypatch.setattr(vc, "is_editable_install", lambda: False)
        monkeypatch.setattr(vc, "fetch_latest_version", lambda timeout=0: "9.9.9")

        def _no_pip(*a, **k):
            raise AssertionError("--check must not run pip")

        monkeypatch.setattr("subprocess.run", _no_pip)
        assert cmd_self_update(_args(check=True, yes=False)) == 0

    def test_pip_failure_does_not_restart_daemon(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import subprocess

        monkeypatch.setattr(vc, "installed_version", lambda: "2.7.3")
        monkeypatch.setattr(vc, "is_editable_install", lambda: False)
        monkeypatch.setattr(vc, "fetch_latest_version", lambda timeout=0: "9.9.9")

        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "boom"),
        )
        restarted = {"n": 0}
        monkeypatch.setattr(
            "iai_mcp.cli._daemon.cmd_daemon_restart",
            lambda a: restarted.__setitem__("n", restarted["n"] + 1) or 0,
        )
        assert cmd_self_update(_args(check=False, yes=True)) == 1
        assert restarted["n"] == 0

    def _arm_upgrade(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[list[list[str]], dict]:
        import subprocess

        monkeypatch.setattr(vc, "installed_version", lambda: "2.7.3")
        monkeypatch.setattr(vc, "is_editable_install", lambda: False)
        monkeypatch.setattr(vc, "fetch_latest_version", lambda timeout=0: "9.9.9")
        pip_argv: list[list[str]] = []

        def _pip(argv, **k):
            pip_argv.append(list(argv))
            return subprocess.CompletedProcess(argv, 0, "ok", "")

        monkeypatch.setattr("subprocess.run", _pip)
        restarted = {"n": 0}
        monkeypatch.setattr(
            "iai_mcp.cli._daemon.cmd_daemon_restart",
            lambda a: restarted.__setitem__("n", restarted["n"] + 1) or 0,
        )
        return pip_argv, restarted

    def test_success_requires_status_roundtrip_with_new_version(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pip_argv, restarted = self._arm_upgrade(monkeypatch)

        async def _probe(path, timeout=0):
            return {"ok": True, "version": "9.9.9"}

        monkeypatch.setattr("iai_mcp.doctor._socket_status_probe", _probe)
        assert cmd_self_update(_args(check=False, yes=True)) == 0
        assert restarted["n"] == 1
        assert any("iai-pme==9.9.9" in " ".join(a) for a in pip_argv)

    def test_stale_socket_never_reads_as_live(
        self, monkeypatch: pytest.MonkeyPatch, _isolated: Path,
    ) -> None:
        # A SIGKILLed daemon leaves its socket file on disk — the success
        # claim must come from a status round-trip, not a stat().
        self._arm_upgrade(monkeypatch)
        sock = _isolated / ".daemon.sock"
        monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(sock))
        sock.write_text("")

        async def _dead(path, timeout=0):
            return None

        monkeypatch.setattr("iai_mcp.doctor._socket_status_probe", _dead)
        monkeypatch.setattr(
            "iai_mcp.cli._daemon._SELF_UPDATE_SOCKET_WAIT_SEC", 1.5,
        )
        assert cmd_self_update(_args(check=False, yes=True)) == 1

    def test_old_version_still_serving_is_reported_not_celebrated(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._arm_upgrade(monkeypatch)

        async def _old(path, timeout=0):
            return {"ok": True, "version": "2.7.3"}

        monkeypatch.setattr("iai_mcp.doctor._socket_status_probe", _old)
        monkeypatch.setattr(
            "iai_mcp.cli._daemon._SELF_UPDATE_SOCKET_WAIT_SEC", 1.5,
        )
        assert cmd_self_update(_args(check=False, yes=True)) == 1
