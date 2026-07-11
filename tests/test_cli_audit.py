from __future__ import annotations



from iai_mcp.cli import main as cli_main
from iai_mcp.events import write_event
from iai_mcp.store import MemoryStore


def test_cli_audit_empty(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    code = cli_main(["audit"])
    assert code == 0
    out = capsys.readouterr().out
    assert (
        "no identity events" in out.lower()
        or "no events" in out.lower()
    )


def test_cli_audit_renders_events(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    store = MemoryStore(path=tmp_path)
    write_event(
        store, kind="s5_invariant_update",
        data={"anchor_id": "abc", "new_record_id": "def"},
        severity="info", session_id="s1",
    )
    store.close()
    code = cli_main(["audit"])
    assert code == 0
    out = capsys.readouterr().out
    assert "s5_invariant_update" in out
    assert "info" in out


def test_cli_audit_since_weeks(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    store = MemoryStore(path=tmp_path)
    write_event(
        store, kind="s5_invariant_update",
        data={"anchor_id": "abc"},
        severity="info", session_id="s1",
    )
    store.close()
    code = cli_main(["audit", "--since=2"])
    assert code == 0


def test_cli_audit_severity_filter_warning_only(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    store = MemoryStore(path=tmp_path)
    write_event(
        store, kind="s5_invariant_update",
        data={"anchor_id": "abc"},
        severity="info", session_id="s1",
    )
    write_event(
        store, kind="s5_drift_alert",
        data={"first_value": 0.1, "last_value": 0.5},
        severity="warning", session_id="s2",
    )
    store.close()
    code = cli_main(["audit", "--severity=warning"])
    assert code == 0
    out = capsys.readouterr().out
    assert "s5_drift_alert" in out
    assert "s5_invariant_update" not in out


def test_cli_audit_shows_shield_rejections_redacted(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    store = MemoryStore(path=tmp_path)
    write_event(
        store, kind="shield_rejection",
        data={
            "tier": "hard_block",
            "matched": ["forget", "you are now", "override"],
            "record_id": "aabbcc",
            "action": "reject",
        },
        severity="critical", session_id="s1",
    )
    store.close()
    code = cli_main(["audit"])
    assert code == 0
    out = capsys.readouterr().out
    assert "shield_rejection" in out
    assert "3" in out or "matched_count=3" in out.replace(" ", "")
    assert "forget" not in out
    assert "you are now" not in out


def test_cli_audit_shield_subcommand(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    store = MemoryStore(path=tmp_path)
    write_event(
        store, kind="shield_rejection",
        data={"tier": "hard_block", "matched": ["forget"], "action": "reject"},
        severity="critical", session_id="s1",
    )
    store.close()
    code = cli_main(["audit", "shield", "--since=7"])
    assert code == 0


def test_cli_audit_drift_subcommand(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    store = MemoryStore(path=tmp_path)
    for i, v in enumerate([0.1, 0.2, 0.3, 0.4, 0.5]):
        write_event(
            store, kind="trajectory_metric",
            data={"metric": "m4", "value": v},
            severity="info", session_id=f"s{i}",
        )
    store.close()
    code = cli_main(["audit", "drift"])
    assert code == 0
    out = capsys.readouterr().out
    assert "drift" in out.lower()


def test_cli_audit_identity_subcommand(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    store = MemoryStore(path=tmp_path)
    write_event(
        store, kind="s5_invariant_update",
        data={"anchor_id": "abc"},
        severity="info", session_id="s1",
    )
    write_event(
        store, kind="shield_rejection",
        data={"tier": "hard_block", "matched": ["forget"], "action": "reject"},
        severity="critical", session_id="s2",
    )
    store.close()
    code = cli_main(["audit", "identity"])
    assert code == 0
    out = capsys.readouterr().out
    assert "s5_invariant_update" in out
    assert "shield_rejection" not in out
