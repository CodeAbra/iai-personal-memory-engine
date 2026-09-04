"""Pin the Desktop-layout fixture to the research-verified shapes: the three
on-disk layouts, both session-record field sets, and both hook-receipt
kinds. Exists so a later change cannot silently drift the fixture out of
agreement with what a real Desktop build and a real hook script produce."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from iai_mcp.cli._cowork import _looks_like_cowork_home

from tests._cowork_desktop_fixture import (
    ACCOUNT_ID,
    USER_ID,
    build_desktop_tree,
    write_agent_session,
    write_code_session,
    write_hook_receipt,
)


def test_current_layout_has_rpm_manifest_and_no_marketplaces(tmp_path):
    paths = build_desktop_tree(tmp_path, layout="current")
    assert paths["cowork_settings"].exists()
    assert paths["cowork_plugins"].is_dir()
    assert paths["rpm_manifest"].exists()
    assert paths["agent_config_dir"].is_dir()
    assert not (paths["agent_config_dir"] / "settings.json").exists()
    assert paths["marketplaces_dir"] is None
    assert paths["known_marketplaces"] is None
    assert paths["code_session_home"].is_dir()
    assert _looks_like_cowork_home(paths["device_home"]) is True


def test_legacy_140_layout_has_marketplaces_and_no_rpm(tmp_path):
    paths = build_desktop_tree(tmp_path, layout="legacy_140")
    assert paths["marketplaces_dir"].is_dir()
    assert paths["known_marketplaces"].exists()
    assert paths["rpm_manifest"] is None
    assert _looks_like_cowork_home(paths["device_home"]) is True


def test_windows_layout_has_no_markers(tmp_path):
    paths = build_desktop_tree(tmp_path, layout="windows")
    assert paths["cowork_settings"] is None
    assert paths["cowork_plugins"] is None
    assert paths["rpm_manifest"] is None
    assert paths["marketplaces_dir"] is None
    assert paths["device_home"].is_dir()
    assert paths["code_session_home"].is_dir()
    assert _looks_like_cowork_home(paths["device_home"]) is False


def test_unknown_layout_rejected(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        build_desktop_tree(tmp_path, layout="bogus")


def test_code_session_record_has_no_session_type_key(tmp_path):
    paths = build_desktop_tree(tmp_path, layout="current")
    dst = write_code_session(
        paths,
        session_id="sess-code-1",
        cwd="/tmp/proj",
        last_activity=datetime(2026, 9, 2, tzinfo=timezone.utc),
    )
    record = json.loads(dst.read_text())
    assert "sessionType" not in record
    assert record["sessionId"] == "sess-code-1"
    assert record["cwd"] == "/tmp/proj"


def test_agent_session_record_has_agent_session_type(tmp_path):
    paths = build_desktop_tree(tmp_path, layout="current")
    dst = write_agent_session(
        paths,
        session_id="sess-agent-1",
        cli_session_id="cli-agent-1",
        cwd="/tmp/agent-out",
        last_activity=datetime(2026, 9, 2, tzinfo=timezone.utc),
        plugin_install_paths=["/tmp/plugins/example"],
    )
    record = json.loads(dst.read_text())
    assert record["sessionType"] == "agent"
    assert record["cliSessionId"] == "cli-agent-1"
    assert record["pluginInstallPaths"] == ["/tmp/plugins/example"]


def test_write_hook_receipt_recall_kind(tmp_path):
    iai_home = tmp_path / "iai-home"
    when = datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc)
    log_path = write_hook_receipt(
        iai_home, kind="recall", session_id="sess-r1", channel="plugin", when=when,
    )
    assert log_path.name == "recall-2026-09-02.log"
    text = log_path.read_text()
    assert "session=sess-r1" in text
    assert "channel=plugin" in text


def test_write_hook_receipt_capture_kind(tmp_path):
    iai_home = tmp_path / "iai-home"
    when = datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc)
    log_path = write_hook_receipt(
        iai_home, kind="capture", session_id="sess-c1", channel="settings", when=when,
    )
    assert log_path.name == "turn-capture-2026-09-02.log"
    text = log_path.read_text()
    assert "session=sess-c1" in text
    assert "rc=0" in text
    assert "channel=settings" in text


def test_write_hook_receipt_unknown_kind_rejected(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        write_hook_receipt(
            tmp_path / "iai-home",
            kind="bogus",
            session_id="sess-x",
            channel="plugin",
            when=datetime.now(timezone.utc),
        )


def test_module_identity_constants_are_uuid_shaped(tmp_path):
    import re

    uuid_re = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
    assert uuid_re.match(ACCOUNT_ID)
    assert uuid_re.match(USER_ID)
