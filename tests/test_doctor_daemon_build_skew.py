"""``doctor`` daemon-build-skew check: (hh) daemon build matches installed package.

Catches a still-running, supervised daemon left on an old build after an
in-place package upgrade -- a class the sleep-path source-digest check
(ff) cannot see, since its stamp covers only the sleep-pipeline roots and an
upgrade of the storage layer leaves those untouched. Every case is driven
from a fixture status payload; no test reaches a real socket, store or
daemon.
"""

from __future__ import annotations

import pytest


def _patch_probe(monkeypatch: pytest.MonkeyPatch, result=None, *, raises: bool = False):
    async def _fake_probe(socket_path, timeout):
        if raises:
            raise ConnectionRefusedError("simulated: no daemon listening")
        return result

    monkeypatch.setattr("iai_mcp.doctor._socket_status_probe", _fake_probe)


def _patch_installed_version(monkeypatch: pytest.MonkeyPatch, version) -> None:
    if version is None:
        monkeypatch.delattr("iai_mcp.__version__", raising=False)
    else:
        monkeypatch.setattr("iai_mcp.__version__", version, raising=False)


def test_versions_differ_is_advisory_naming_both_and_the_remedy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iai_mcp.doctor import check_hh_daemon_build_skew

    _patch_probe(monkeypatch, {"version": "3.0.1"})
    _patch_installed_version(monkeypatch, "3.0.2")

    result = check_hh_daemon_build_skew()
    assert result.passed is True
    assert result.status == "WARN"
    assert "3.0.1" in result.detail
    assert "3.0.2" in result.detail
    assert "daemon stop" in result.detail
    assert "daemon start" in result.detail


def test_versions_equal_passes_and_names_the_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iai_mcp.doctor import check_hh_daemon_build_skew

    _patch_probe(monkeypatch, {"version": "3.0.2"})
    _patch_installed_version(monkeypatch, "3.0.2")

    result = check_hh_daemon_build_skew()
    assert result.passed is True
    assert result.status == "PASS"
    assert "3.0.2" in result.detail


def test_no_reachable_daemon_passes_never_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iai_mcp.doctor import check_hh_daemon_build_skew

    _patch_probe(monkeypatch, None)
    _patch_installed_version(monkeypatch, "3.0.2")

    result = check_hh_daemon_build_skew()
    assert result.passed is True
    assert result.status == "PASS"
    assert "no daemon" in result.detail.lower()


def test_status_reply_without_version_field_passes_and_names_the_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iai_mcp.doctor import check_hh_daemon_build_skew

    _patch_probe(monkeypatch, {"daemon_pid": 12345})
    _patch_installed_version(monkeypatch, "3.0.2")

    result = check_hh_daemon_build_skew()
    assert result.passed is True
    assert result.status == "PASS"
    assert "version" in result.detail.lower()
    assert "daemon stop" in result.detail
    assert "daemon start" in result.detail


def test_versions_agree_but_digest_diverges_stays_passing_and_mentions_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The non-duplication contract with (ff): one skew, one warning."""
    from iai_mcp.doctor import check_hh_daemon_build_skew

    _patch_probe(
        monkeypatch,
        {"version": "3.0.2", "code_stamp": {"roots": ["x"], "digest": "stale"}},
    )
    _patch_installed_version(monkeypatch, "3.0.2")
    monkeypatch.setattr(
        "iai_mcp.code_stamp.stamp_divergence", lambda boot_stamp: "digest"
    )

    result = check_hh_daemon_build_skew()
    assert result.passed is True
    assert result.status == "PASS"
    assert "stamp" in result.detail.lower()


def test_probe_raises_is_swallowed_and_reads_as_no_daemon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iai_mcp.doctor import check_hh_daemon_build_skew

    _patch_probe(monkeypatch, raises=True)
    _patch_installed_version(monkeypatch, "3.0.2")

    result = check_hh_daemon_build_skew()
    assert result.passed is True
    assert result.status == "PASS"
    assert "no daemon" in result.detail.lower()


def test_installed_version_unknown_passes_with_daemon_version_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iai_mcp.doctor import check_hh_daemon_build_skew

    _patch_probe(monkeypatch, {"version": "3.0.1"})
    _patch_installed_version(monkeypatch, None)

    result = check_hh_daemon_build_skew()
    assert result.passed is True
    assert result.status == "PASS"
    assert "3.0.1" in result.detail


def test_no_case_ever_produces_a_failing_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iai_mcp.doctor import check_hh_daemon_build_skew

    scenarios = [
        (None, "3.0.2", False),
        ({"version": "3.0.1"}, "3.0.2", False),
        ({"version": "3.0.2"}, "3.0.2", False),
        ({"daemon_pid": 1}, "3.0.2", False),
        ({"version": "3.0.1"}, None, False),
        (None, "3.0.2", True),
    ]
    for result_payload, installed, raises in scenarios:
        _patch_probe(monkeypatch, result_payload, raises=raises)
        _patch_installed_version(monkeypatch, installed)
        result = check_hh_daemon_build_skew()
        assert result.passed is True, (result_payload, installed, raises, result)


def test_check_body_has_no_store_open_file_write_or_process_control() -> None:
    import inspect

    from iai_mcp.doctor._lifecycle_checks import check_hh_daemon_build_skew

    src = inspect.getsource(check_hh_daemon_build_skew)
    assert "open_store_conn" not in src
    assert "subprocess" not in src
    assert "os.kill" not in src
    assert "SIGTERM" not in src
    assert "launchctl" not in src
    assert "systemctl" not in src


def test_no_repair_action_registered_for_build_skew_check() -> None:
    import inspect

    from iai_mcp.doctor import _plan_repair_actions

    src = inspect.getsource(_plan_repair_actions)
    assert "check_hh_daemon_build_skew" not in src
    assert "(hh) daemon build" not in src
