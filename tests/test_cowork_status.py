"""`iai-mcp cowork status` must ground every tier in evidence this tool did
not write itself: Desktop's own session records and hook-written receipt log
lines. This file pins the two read-only readers, the pure tier engine, and
the rewritten status command end to end."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests._cowork_desktop_fixture import (
    build_desktop_tree,
    write_agent_session,
    write_code_session,
    write_hook_receipt,
)


def test_read_desktop_sessions_classifies_both_shapes(tmp_path):
    from iai_mcp.cli._cowork import _read_desktop_sessions

    paths = build_desktop_tree(tmp_path, layout="current")
    when = datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc)
    write_code_session(paths, session_id="sess-code-1", cwd="/tmp/proj", last_activity=when)
    write_agent_session(
        paths,
        session_id="sess-agent-1",
        cli_session_id="cli-agent-1",
        cwd="/tmp/agent-out",
        last_activity=when,
        plugin_install_paths=["/tmp/plugins/example"],
    )

    sessions = _read_desktop_sessions([paths["base"]])
    assert len(sessions) == 2
    code = next(s for s in sessions if s.kind == "code")
    agent = next(s for s in sessions if s.kind == "agent")
    assert code.id == "sess-code-1"
    assert code.last_activity == when
    assert agent.id == "cli-agent-1"
    assert agent.last_activity == when


def test_read_desktop_sessions_prefers_cli_session_id_on_code_shape(tmp_path):
    """Real Desktop data (verified live on this machine) shows a
    code-shaped record's own "sessionId" field carries a "local_" prefix
    that never matches the transcript-sweep session id (the ".jsonl"
    filename stem) -- but every record's "cliSessionId" field, when
    present, does. This must be preferred regardless of "code" vs "agent"
    classification, or a real local session can never cross-reference."""
    from iai_mcp.cli._cowork import _read_desktop_sessions

    paths = build_desktop_tree(tmp_path, layout="current")
    when = datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc)
    code_home = paths["code_session_home"]
    code_home.mkdir(parents=True, exist_ok=True)
    record = {
        "sessionId": "local_9a0c2866-789f-4e98-b79e-6c41fa492c73",
        "cliSessionId": "9a0c2866-789f-4e98-b79e-6c41fa492c73",
        "cwd": "/tmp/proj",
        "createdAt": when.isoformat(),
        "lastActivityAt": when.isoformat(),
    }
    (code_home / "local_9a0c2866-789f-4e98-b79e-6c41fa492c73.json").write_text(
        json.dumps(record), encoding="utf-8"
    )

    sessions = _read_desktop_sessions([paths["base"]])
    assert len(sessions) == 1
    assert sessions[0].kind == "code"
    assert sessions[0].id == "9a0c2866-789f-4e98-b79e-6c41fa492c73"


def test_read_desktop_sessions_skips_invalid_ids(tmp_path):
    from iai_mcp.cli._cowork import _read_desktop_sessions

    paths = build_desktop_tree(tmp_path, layout="current")
    code_home = paths["code_session_home"]
    (code_home / "local_missing.json").write_text(json.dumps({"cwd": "/x"}))
    (code_home / "local_nonstring.json").write_text(json.dumps({"sessionId": 12345}))
    (code_home / "local_unsafe.json").write_text(json.dumps({"sessionId": "not a safe id!"}))

    sessions = _read_desktop_sessions([paths["base"]])
    assert sessions == []


def test_read_desktop_sessions_skips_malformed_files(tmp_path):
    from iai_mcp.cli._cowork import _read_desktop_sessions

    paths = build_desktop_tree(tmp_path, layout="current")
    code_home = paths["code_session_home"]
    (code_home / "local_badjson.json").write_text("{not json")
    (code_home / "local_adirectory.json").mkdir()

    sessions = _read_desktop_sessions([paths["base"]])
    assert sessions == []


def test_read_desktop_sessions_empty_for_missing_root(tmp_path):
    from iai_mcp.cli._cowork import _read_desktop_sessions

    sessions = _read_desktop_sessions([tmp_path / "does-not-exist"])
    assert sessions == []


def test_read_hook_receipts_returns_all_channels(tmp_path):
    from iai_mcp.cli._cowork import _read_hook_receipts

    iai_home = tmp_path / "iai-home"
    when = datetime.now(timezone.utc)
    write_hook_receipt(iai_home, kind="recall", session_id="sess-r1", channel="plugin", when=when)
    write_hook_receipt(iai_home, kind="capture", session_id="sess-c1", channel="settings", when=when)

    receipts = _read_hook_receipts(iai_home, days=7)
    assert len(receipts) == 2
    assert {r.channel for r in receipts} == {"plugin", "settings"}
    assert {r.kind for r in receipts} == {"recall", "capture"}


def test_read_hook_receipts_accepts_transcript_sweep_log_kind(tmp_path):
    """Mirrors the exact line grammar iai_mcp.transcript_sweep._append_receipt
    writes -- reusing the shared field regexes unchanged, only the log-name
    and receipt-kind registries are extended."""
    from iai_mcp.cli._cowork import _read_hook_receipts

    iai_home = tmp_path / "iai-home"
    logs_dir = iai_home / "logs"
    logs_dir.mkdir(parents=True)
    when = datetime.now(timezone.utc)
    log_path = logs_dir / f"transcript-sweep-{when:%Y-%m-%d}.log"
    log_path.write_text(
        f"{when:%Y-%m-%dT%H:%M:%SZ} session=cli-agent-1 lines=3 channel=transcript-sweep\n",
        encoding="utf-8",
    )

    receipts = _read_hook_receipts(iai_home, days=7)
    assert len(receipts) == 1
    assert receipts[0].session_id == "cli-agent-1"
    assert receipts[0].channel == "transcript-sweep"
    assert receipts[0].kind == "transcript-sweep"


def test_read_hook_receipts_skips_invalid_or_truncated_lines(tmp_path):
    from iai_mcp.cli._cowork import _read_hook_receipts

    iai_home = tmp_path / "iai-home"
    logs_dir = iai_home / "logs"
    logs_dir.mkdir(parents=True)
    log_path = logs_dir / f"recall-{datetime.now(timezone.utc):%Y-%m-%d}.log"
    ts = f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{ts} session=not!safe channel=plugin\n")
        f.write(f"{ts} session=sess-ok source=startup chan\n")
        f.write(f"{ts} session=sess-good source=startup channel=plugin\n")

    receipts = _read_hook_receipts(iai_home, days=7)
    assert len(receipts) == 1
    assert receipts[0].session_id == "sess-good"


def test_read_hook_receipts_empty_for_missing_root(tmp_path):
    from iai_mcp.cli._cowork import _read_hook_receipts

    receipts = _read_hook_receipts(tmp_path / "does-not-exist", days=7)
    assert receipts == []


def test_status_reader_is_read_only(tmp_path):
    from iai_mcp.cli._cowork import _read_desktop_sessions, _read_hook_receipts

    paths = build_desktop_tree(tmp_path, layout="current")
    when = datetime.now(timezone.utc)
    write_code_session(paths, session_id="sess-code-1", cwd="/tmp/proj", last_activity=when)
    write_agent_session(
        paths,
        session_id="sess-agent-1",
        cli_session_id="cli-agent-1",
        cwd="/tmp/agent-out",
        last_activity=when,
        plugin_install_paths=["/tmp/plugins/example"],
    )
    iai_home = tmp_path / "iai-home"
    logs_dir = iai_home / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / f"transcript-sweep-{when:%Y-%m-%d}.log").write_text(
        f"{when:%Y-%m-%dT%H:%M:%SZ} session=cli-agent-1 lines=3 channel=transcript-sweep\n",
        encoding="utf-8",
    )

    def snapshot():
        return {
            str(p.relative_to(tmp_path)): p.stat().st_mtime_ns
            for p in sorted(tmp_path.rglob("*"))
        }

    before = snapshot()
    _read_desktop_sessions([paths["base"]])
    _read_hook_receipts(iai_home, days=7)
    after = snapshot()
    assert before == after


def test_agent_channel_state_staged_without_receipts():
    import iai_mcp.cli._cowork as cowork

    state = cowork._agent_channel_state(
        staged=True, build="1.44121.1", receipts=[], sessions=[]
    )
    assert state.tier == cowork.TIER_STAGED


def test_agent_channel_state_not_installed_without_flag_or_receipts():
    import iai_mcp.cli._cowork as cowork

    state = cowork._agent_channel_state(
        staged=False, build="1.44121.1", receipts=[], sessions=[]
    )
    assert state.tier == cowork.TIER_NOT_INSTALLED
    assert state.tier != cowork.TIER_STAGED


def test_agent_channel_state_active_regardless_of_unconfirmed_build():
    """The build-version disqualifier is dropped for this channel: the sweep
    reads transcript files present on every Desktop build. An unconfirmed
    or unrecognized build string must not block ACTIVE, and is carried
    through only as informational output."""
    import iai_mcp.cli._cowork as cowork
    from iai_mcp.cli._cowork import DesktopSession, HookReceipt

    now = datetime.now(timezone.utc)
    session = DesktopSession(id="sess-1", kind="agent", cwd="/tmp", last_activity=now)
    receipt = HookReceipt(
        session_id="sess-1",
        channel=cowork._AGENT_RECEIPT_CHANNEL,
        timestamp=now,
        kind="transcript-sweep",
        log_name="transcript-sweep-x.log",
    )
    state = cowork._agent_channel_state(
        staged=True, build="9.9.9-unconfirmed", receipts=[receipt], sessions=[session]
    )
    assert state.tier == cowork.TIER_ACTIVE
    assert state.build == "9.9.9-unconfirmed"


def test_agent_channel_state_active_regardless_of_unknown_build():
    import iai_mcp.cli._cowork as cowork
    from iai_mcp.cli._cowork import DesktopSession, HookReceipt

    now = datetime.now(timezone.utc)
    session = DesktopSession(id="sess-1", kind="agent", cwd="/tmp", last_activity=now)
    receipt = HookReceipt(
        session_id="sess-1",
        channel=cowork._AGENT_RECEIPT_CHANNEL,
        timestamp=now,
        kind="transcript-sweep",
        log_name="transcript-sweep-x.log",
    )
    state = cowork._agent_channel_state(
        staged=True, build=None, receipts=[receipt], sessions=[session]
    )
    assert state.tier == cowork.TIER_ACTIVE
    assert state.build is None


def test_agent_channel_state_active_requires_transcript_sweep_channel():
    import iai_mcp.cli._cowork as cowork
    from iai_mcp.cli._cowork import DesktopSession, HookReceipt

    now = datetime.now(timezone.utc)
    session = DesktopSession(id="sess-1", kind="agent", cwd="/tmp", last_activity=now)
    receipt = HookReceipt(
        session_id="sess-1", channel="settings", timestamp=now, kind="recall", log_name="recall-x.log"
    )
    state = cowork._agent_channel_state(
        staged=True, build="1.44121.1", receipts=[receipt], sessions=[session]
    )
    assert state.tier != cowork.TIER_ACTIVE


def test_agent_channel_state_active_requires_known_session_id():
    import iai_mcp.cli._cowork as cowork
    from iai_mcp.cli._cowork import DesktopSession, HookReceipt

    now = datetime.now(timezone.utc)
    session = DesktopSession(id="sess-1", kind="agent", cwd="/tmp", last_activity=now)
    receipt = HookReceipt(
        session_id="sess-unknown",
        channel=cowork._AGENT_RECEIPT_CHANNEL,
        timestamp=now,
        kind="transcript-sweep",
        log_name="transcript-sweep-x.log",
    )
    state = cowork._agent_channel_state(
        staged=True, build="1.44121.1", receipts=[receipt], sessions=[session]
    )
    assert state.tier != cowork.TIER_ACTIVE


def test_agent_channel_state_active_requires_activity_window():
    """Also pins the backstop-only capture case: a receipt written hours
    after lastActivityAt (as a nightly sleep-pipeline backstop pass would
    write it, rather than the awake courier) falls outside the tolerance
    and must not read ACTIVE."""
    import iai_mcp.cli._cowork as cowork
    from iai_mcp.cli._cowork import DesktopSession, HookReceipt

    now = datetime.now(timezone.utc)
    session = DesktopSession(
        id="sess-1", kind="agent", cwd="/tmp", last_activity=now - timedelta(hours=2)
    )
    receipt = HookReceipt(
        session_id="sess-1",
        channel=cowork._AGENT_RECEIPT_CHANNEL,
        timestamp=now,
        kind="transcript-sweep",
        log_name="transcript-sweep-x.log",
    )
    state = cowork._agent_channel_state(
        staged=True, build="1.44121.1", receipts=[receipt], sessions=[session]
    )
    assert state.tier != cowork.TIER_ACTIVE


def test_agent_channel_state_active_when_all_three_conditions_met():
    import iai_mcp.cli._cowork as cowork
    from iai_mcp.cli._cowork import DesktopSession, HookReceipt

    now = datetime.now(timezone.utc)
    session = DesktopSession(id="sess-1", kind="agent", cwd="/tmp", last_activity=now)
    receipt = HookReceipt(
        session_id="sess-1",
        channel=cowork._AGENT_RECEIPT_CHANNEL,
        timestamp=now,
        kind="transcript-sweep",
        log_name="transcript-sweep-x.log",
    )
    state = cowork._agent_channel_state(
        staged=True, build="1.44121.1", receipts=[receipt], sessions=[session]
    )
    assert state.tier == cowork.TIER_ACTIVE
    assert state.receipt is receipt


def test_agent_channel_state_is_pure(monkeypatch):
    import pathlib

    import iai_mcp.cli._cowork as cowork

    def _raise_on_open(*args, **kwargs):
        raise AssertionError("pure function must never touch the filesystem")

    monkeypatch.setattr(pathlib.Path, "open", _raise_on_open)
    state = cowork._agent_channel_state(
        staged=True, build="1.44121.1", receipts=[], sessions=[]
    )
    assert state.tier == cowork.TIER_STAGED


@pytest.fixture()
def cowork_status_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(
        "iai_mcp.cli._capture._patch_claude_desktop_config",
        lambda action: "Claude Desktop: stubbed",
    )
    monkeypatch.setattr(
        "iai_mcp.cli._cowork._issue_sweeper_load", lambda paths: None,
    )
    base = home / "Library" / "Application Support" / "Claude"
    paths = build_desktop_tree(base, layout="current")
    return {"home": home, "paths": paths}


def _write_transcript_sweep_receipt(iai_home: Path, *, session_id: str, when: datetime) -> None:
    """Writes the exact line grammar iai_mcp.transcript_sweep._append_receipt
    produces, without importing the daemon-facing spool machinery that
    function depends on."""
    logs_dir = iai_home / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"transcript-sweep-{when:%Y-%m-%d}.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"{when:%Y-%m-%dT%H:%M:%SZ} session={session_id} lines=3 "
            "channel=transcript-sweep\n"
        )


def test_status_not_installed_without_flag(cowork_status_env, monkeypatch, capsys):
    import iai_mcp.cli._cowork as cowork

    monkeypatch.setattr(cowork, "_desktop_build_version", lambda: "1.44121.1")

    rc = cowork.cmd_cowork_status(argparse.Namespace())
    out = capsys.readouterr().out

    assert rc != 0
    assert cowork.TIER_NOT_INSTALLED in out
    assert f"tier:                 {cowork.TIER_ACTIVE}" not in out


def test_status_staged_after_install_without_receipt(cowork_status_env, monkeypatch, capsys):
    import iai_mcp.cli._cowork as cowork

    monkeypatch.setattr(cowork, "_desktop_build_version", lambda: "1.44121.1")

    assert cowork.cmd_cowork_install(argparse.Namespace()) == 0
    rc = cowork.cmd_cowork_status(argparse.Namespace())
    out = capsys.readouterr().out

    assert rc != 0
    assert cowork.TIER_STAGED in out
    assert f"tier:                 {cowork.TIER_ACTIVE}" not in out


def test_status_active_only_from_cross_referenced_receipt(cowork_status_env, monkeypatch, capsys):
    import iai_mcp.cli._cowork as cowork

    monkeypatch.setattr(cowork, "_desktop_build_version", lambda: "1.44121.1")
    assert cowork.cmd_cowork_install(argparse.Namespace()) == 0

    when = datetime.now(timezone.utc)
    write_agent_session(
        cowork_status_env["paths"],
        session_id="sess-agent-1",
        cli_session_id="cli-agent-1",
        cwd="/tmp/agent-out",
        last_activity=when,
        plugin_install_paths=["/tmp/plugins/example"],
    )
    _write_transcript_sweep_receipt(
        cowork_status_env["home"] / ".iai-mcp", session_id="cli-agent-1", when=when
    )

    rc = cowork.cmd_cowork_status(argparse.Namespace())
    out = capsys.readouterr().out

    assert rc == 0
    assert cowork.TIER_ACTIVE in out


def test_status_active_regardless_of_unconfirmed_build(cowork_status_env, monkeypatch, capsys):
    import iai_mcp.cli._cowork as cowork

    monkeypatch.setattr(cowork, "_desktop_build_version", lambda: "9.9.9-unconfirmed")
    assert cowork.cmd_cowork_install(argparse.Namespace()) == 0

    when = datetime.now(timezone.utc)
    write_agent_session(
        cowork_status_env["paths"],
        session_id="sess-agent-1",
        cli_session_id="cli-agent-1",
        cwd="/tmp/agent-out",
        last_activity=when,
        plugin_install_paths=["/tmp/plugins/example"],
    )
    _write_transcript_sweep_receipt(
        cowork_status_env["home"] / ".iai-mcp", session_id="cli-agent-1", when=when
    )

    rc = cowork.cmd_cowork_status(argparse.Namespace())
    out = capsys.readouterr().out

    assert rc == 0
    assert cowork.TIER_ACTIVE in out
    assert "9.9.9-unconfirmed" in out


def test_status_never_active_from_installer_write(cowork_status_env, monkeypatch, capsys):
    import iai_mcp.cli._cowork as cowork

    monkeypatch.setattr(cowork, "_desktop_build_version", lambda: "1.44121.1")
    assert cowork.cmd_cowork_install(argparse.Namespace()) == 0

    rc = cowork.cmd_cowork_status(argparse.Namespace())
    out = capsys.readouterr().out

    assert rc != 0
    assert f"tier:                 {cowork.TIER_ACTIVE}" not in out


def test_status_names_channel_per_session_shape(cowork_status_env, monkeypatch, capsys):
    import iai_mcp.cli._cowork as cowork

    monkeypatch.setattr(cowork, "_desktop_build_version", lambda: "1.44121.1")

    cowork.cmd_cowork_status(argparse.Namespace())
    out = capsys.readouterr().out

    assert "capture-hooks" in out
    assert cowork._AGENT_CHANNEL_LABEL in out
