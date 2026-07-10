from __future__ import annotations

from datetime import datetime, timedelta, timezone


from iai_mcp.events import write_event

def _seed_m4(store, values: list[float], session_prefix: str = "s") -> None:
    for i, v in enumerate(values):
        write_event(
            store,
            kind="trajectory_metric",
            data={"metric": "m4", "value": float(v)},
            severity="info",
            session_id=f"{session_prefix}{i}",
        )

def test_detect_drift_no_events_returns_empty(tmp_path):
    from iai_mcp.s5 import detect_drift_anomaly
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    alerts = detect_drift_anomaly(store)
    assert alerts == []

def test_detect_drift_single_session_no_alert(tmp_path):
    from iai_mcp.s5 import detect_drift_anomaly
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    _seed_m4(store, [0.5])
    alerts = detect_drift_anomaly(store, window_sessions=5)
    assert alerts == []

def test_detect_drift_stable_variance_no_alert(tmp_path):
    from iai_mcp.s5 import detect_drift_anomaly
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    _seed_m4(store, [0.3, 0.3, 0.3, 0.3, 0.3])
    alerts = detect_drift_anomaly(store, window_sessions=5)
    assert alerts == []

def test_detect_drift_decreasing_variance_no_alert(tmp_path):
    from iai_mcp.s5 import detect_drift_anomaly
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    _seed_m4(store, [0.9, 0.8, 0.7, 0.6, 0.5])
    alerts = detect_drift_anomaly(store, window_sessions=5)
    assert alerts == []

def test_detect_drift_increasing_variance_triggers_alert(tmp_path):
    from iai_mcp.s5 import detect_drift_anomaly
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    _seed_m4(store, [0.2, 0.3, 0.4, 0.5, 0.6])
    alerts = detect_drift_anomaly(store, window_sessions=5)
    assert len(alerts) == 1
    assert alerts[0]["kind"] == "s5_drift_alert"
    assert alerts[0]["severity"] == "warning"

def test_detect_drift_emits_event_on_alert(tmp_path):
    from iai_mcp.events import query_events
    from iai_mcp.s5 import detect_drift_anomaly
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    _seed_m4(store, [0.1, 0.2, 0.3, 0.4, 0.5])
    detect_drift_anomaly(store, window_sessions=5)
    alert_events = query_events(store, kind="s5_drift_alert", limit=5)
    assert len(alert_events) >= 1
    assert alert_events[0]["severity"] == "warning"
    assert "first_value" in alert_events[0]["data"]
    assert "last_value" in alert_events[0]["data"]

def test_detect_drift_respects_window_sessions(tmp_path):
    from iai_mcp.s5 import detect_drift_anomaly
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    _seed_m4(store, [0.1, 0.2, 0.3])
    alerts_short = detect_drift_anomaly(store, window_sessions=3)
    assert len(alerts_short) == 1

def test_detect_drift_insufficient_window_larger_than_data(tmp_path):
    from iai_mcp.s5 import detect_drift_anomaly
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    _seed_m4(store, [0.1, 0.2])
    alerts = detect_drift_anomaly(store, window_sessions=10)
    assert alerts == []

def test_audit_identity_events_empty(tmp_path):
    from iai_mcp.s5 import audit_identity_events
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    out = audit_identity_events(store)
    assert out == []

def test_audit_identity_events_chronological(tmp_path):
    from iai_mcp.s5 import audit_identity_events
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    write_event(store, kind="s5_invariant_update", data={"anchor_id": "x"}, severity="info")
    write_event(store, kind="s5_cooldown_block", data={"anchor_id": "x"}, severity="warning")
    write_event(store, kind="shield_rejection", data={"tier": "hard_block"}, severity="critical")
    write_event(store, kind="shield_flag", data={"tier": "flag"}, severity="warning")
    write_event(store, kind="s5_drift_alert", data={"first_value": 0.1, "last_value": 0.5}, severity="warning")

    out = audit_identity_events(store)
    assert len(out) == 5
    for i in range(1, len(out)):
        assert out[i]["ts"] <= out[i - 1]["ts"]

def test_audit_identity_events_since_filter(tmp_path):
    from iai_mcp.s5 import audit_identity_events
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    write_event(store, kind="s5_invariant_update", data={"anchor_id": "x"}, severity="info")

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=7)
    out = audit_identity_events(store, since=since)
    assert len(out) == 1

def test_audit_identity_events_excludes_non_identity_kinds(tmp_path):
    from iai_mcp.s5 import audit_identity_events
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    write_event(store, kind="llm_health", data={"status": "ok"}, severity="info")
    write_event(store, kind="s5_invariant_update", data={"anchor_id": "x"}, severity="info")

    out = audit_identity_events(store)
    assert len(out) == 1
    assert out[0]["kind"] == "s5_invariant_update"
