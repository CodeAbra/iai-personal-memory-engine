from __future__ import annotations

from datetime import datetime, timedelta, timezone


from iai_mcp.events import write_event

def _seed_profile_tuned(store, moved_counts: list[float]) -> None:
    for i, v in enumerate(moved_counts):
        write_event(
            store,
            kind="profile_tuned",
            data={"moved_count": v},
            severity="info",
            session_id=f"nightly{i}",
        )

def test_detect_drift_no_events_returns_empty(tmp_path):
    from iai_mcp.s5 import detect_drift_anomaly
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    alerts, candidates, processed = detect_drift_anomaly(store)
    assert alerts == []
    assert candidates == 0
    assert processed == 0

def test_detect_drift_single_cycle_no_alert(tmp_path):
    from iai_mcp.s5 import detect_drift_anomaly
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    _seed_profile_tuned(store, [0.5])
    alerts, candidates, processed = detect_drift_anomaly(store, cycles=5)
    assert alerts == []
    assert candidates == 0
    assert processed == 0

def test_detect_drift_stable_movement_no_alert(tmp_path):
    from iai_mcp.s5 import detect_drift_anomaly
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    _seed_profile_tuned(store, [3, 3, 3, 3, 3])
    alerts, candidates, processed = detect_drift_anomaly(store, cycles=5)
    assert alerts == []
    assert candidates == 5
    assert processed == 4

def test_detect_drift_decreasing_movement_no_alert(tmp_path):
    from iai_mcp.s5 import detect_drift_anomaly
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    _seed_profile_tuned(store, [9, 8, 7, 6, 5])
    alerts, candidates, processed = detect_drift_anomaly(store, cycles=5)
    assert alerts == []
    assert candidates == 5
    assert processed == 4

def test_detect_drift_increasing_movement_triggers_alert(tmp_path):
    from iai_mcp.s5 import detect_drift_anomaly
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    _seed_profile_tuned(store, [2, 3, 4, 5, 6])
    alerts, candidates, processed = detect_drift_anomaly(store, cycles=5)
    assert len(alerts) == 1
    assert alerts[0]["kind"] == "s5_drift_alert"
    assert alerts[0]["severity"] == "warning"
    assert candidates == 5
    assert processed == 4

def test_detect_drift_emits_event_on_alert(tmp_path):
    from iai_mcp.events import query_events
    from iai_mcp.s5 import detect_drift_anomaly
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    _seed_profile_tuned(store, [1, 2, 3, 4, 5])
    detect_drift_anomaly(store, cycles=5)
    alert_events = query_events(store, kind="s5_drift_alert", limit=5)
    assert len(alert_events) >= 1
    assert alert_events[0]["severity"] == "warning"
    assert "first_value" in alert_events[0]["data"]
    assert "last_value" in alert_events[0]["data"]

def test_detect_drift_dedups_persisted_alert_across_cycles(tmp_path):
    from iai_mcp.events import query_events
    from iai_mcp.s5 import detect_drift_anomaly
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    _seed_profile_tuned(store, [2, 3, 4, 5, 6])

    first = detect_drift_anomaly(store, cycles=5)
    second = detect_drift_anomaly(store, cycles=5)

    # Detection unchanged across both passes: same live alert and counts.
    assert len(first[0]) == 1
    assert len(second[0]) == 1
    a1, a2 = first[0][0], second[0][0]
    for field in ("kind", "severity", "cycles", "increases", "first_value", "last_value"):
        assert a2[field] == a1[field]
    assert (second[1], second[2]) == (5, 4)

    # Persistence deduped: one drift episode writes exactly one row.
    rows = query_events(store, kind="s5_drift_alert", limit=10)
    assert len(rows) == 1

def test_detect_drift_persists_new_row_for_changed_episode(tmp_path):
    from iai_mcp.events import query_events
    from iai_mcp.s5 import detect_drift_anomaly
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    _seed_profile_tuned(store, [2, 3, 4, 5, 6])
    detect_drift_anomaly(store, cycles=5)
    detect_drift_anomaly(store, cycles=5)
    assert len(query_events(store, kind="s5_drift_alert", limit=10)) == 1

    # A distinct rising episode is a genuinely new alert -> a second row.
    _seed_profile_tuned(store, [10, 20, 30, 40, 50])
    changed = detect_drift_anomaly(store, cycles=5)
    assert len(changed[0]) == 1
    assert len(query_events(store, kind="s5_drift_alert", limit=10)) == 2

def test_detect_drift_respects_cycles(tmp_path):
    from iai_mcp.s5 import detect_drift_anomaly
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    _seed_profile_tuned(store, [1, 2, 3])
    alerts, candidates, processed = detect_drift_anomaly(store, cycles=3)
    assert len(alerts) == 1
    assert candidates == 3
    assert processed == 2

def test_detect_drift_insufficient_cycles_larger_than_data(tmp_path):
    from iai_mcp.s5 import detect_drift_anomaly
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    _seed_profile_tuned(store, [1, 2])
    alerts, candidates, processed = detect_drift_anomaly(store, cycles=10)
    assert alerts == []
    assert candidates == 0
    assert processed == 0

def test_detect_drift_ignores_trajectory_metric_stream(tmp_path):
    from iai_mcp.s5 import detect_drift_anomaly
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    for i in range(10):
        write_event(
            store,
            kind="trajectory_metric",
            data={"metric": "m4", "value": float(i)},
            severity="info",
            session_id=f"s{i}",
        )
    alerts, candidates, processed = detect_drift_anomaly(store, cycles=5)
    assert alerts == []
    assert candidates == 0
    assert processed == 0

def test_detect_drift_skips_non_numeric_moved_count(tmp_path):
    from iai_mcp.s5 import detect_drift_anomaly
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    write_event(store, kind="profile_tuned", data={"moved_count": True}, severity="info")
    write_event(store, kind="profile_tuned", data={"moved_count": "nope"}, severity="info")
    write_event(store, kind="profile_tuned", data={}, severity="info")
    _seed_profile_tuned(store, [1, 2, 3])
    alerts, candidates, processed = detect_drift_anomaly(store, cycles=3)
    assert candidates == 3
    assert processed == 2

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
