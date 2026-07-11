from __future__ import annotations

from datetime import datetime, timedelta, timezone


from iai_mcp.events import query_events, write_event
from iai_mcp.store import MemoryStore

def _seed_weekly_scores(store, values: list[float]) -> None:
    base = datetime.now(timezone.utc) - timedelta(days=7 * len(values))
    for i, v in enumerate(values):
        write_event(
            store,
            kind="formality_score_weekly",
            data={
                "score": float(v),
                "lang": "en",
                "week_iso": (base + timedelta(days=7 * i)).isoformat(),
                "samples": 10,
            },
            severity="info",
        )

def test_detect_camouflaging_rising_trajectory(tmp_path):
    from iai_mcp.camouflaging import detect_camouflaging

    store = MemoryStore(path=tmp_path)
    _seed_weekly_scores(store, [0.4, 0.55, 0.65, 0.75, 0.85])
    result = detect_camouflaging(store)
    assert result["detected"] is True
    assert result["trajectory_slope"] > 0.05
    assert result["current_mean"] > 0.6

def test_detect_camouflaging_flat_trajectory(tmp_path):
    from iai_mcp.camouflaging import detect_camouflaging

    store = MemoryStore(path=tmp_path)
    _seed_weekly_scores(store, [0.5, 0.5, 0.5, 0.5, 0.5])
    result = detect_camouflaging(store)
    assert result["detected"] is False

def test_detect_camouflaging_insufficient_samples(tmp_path):
    from iai_mcp.camouflaging import detect_camouflaging

    store = MemoryStore(path=tmp_path)
    _seed_weekly_scores(store, [0.3, 0.5])
    result = detect_camouflaging(store)
    assert result["detected"] is False
    assert result["sample_count"] == 2

def test_detect_camouflaging_high_mean_but_flat_no_detect(tmp_path):
    from iai_mcp.camouflaging import detect_camouflaging

    store = MemoryStore(path=tmp_path)
    _seed_weekly_scores(store, [0.7, 0.7, 0.7, 0.7, 0.7])
    result = detect_camouflaging(store)
    assert result["detected"] is False

def test_detect_camouflaging_rising_but_low_mean_no_detect(tmp_path):
    from iai_mcp.camouflaging import detect_camouflaging

    store = MemoryStore(path=tmp_path)
    _seed_weekly_scores(store, [0.1, 0.15, 0.2, 0.3, 0.4])
    result = detect_camouflaging(store)
    assert result["detected"] is False

def test_run_weekly_pass_emits_events_and_bumps_knob(tmp_path):
    from iai_mcp.camouflaging import run_weekly_pass

    import iai_mcp.core as core
    core._profile_state["camouflaging_relaxation"] = 0.0

    store = MemoryStore(path=tmp_path)
    _seed_weekly_scores(store, [0.4, 0.55, 0.65, 0.75, 0.85])
    run_weekly_pass(store)

    detected = query_events(store, kind="camouflaging_detected", limit=5)
    relaxed = query_events(store, kind="register_relaxed", limit=5)
    assert len(detected) >= 1
    assert len(relaxed) >= 1

    value = core._profile_state["camouflaging_relaxation"]
    assert value > 0.0

def test_run_weekly_pass_flat_no_events(tmp_path):
    from iai_mcp.camouflaging import run_weekly_pass

    import iai_mcp.core as core
    core._profile_state["camouflaging_relaxation"] = 0.0

    store = MemoryStore(path=tmp_path)
    _seed_weekly_scores(store, [0.5, 0.5, 0.5, 0.5, 0.5])
    run_weekly_pass(store)

    detected = query_events(store, kind="camouflaging_detected", limit=5)
    relaxed = query_events(store, kind="register_relaxed", limit=5)
    assert detected == []
    assert relaxed == []
    assert core._profile_state["camouflaging_relaxation"] == 0.0

def test_record_user_formality_writes_weekly_event(tmp_path):
    from iai_mcp.camouflaging import record_user_formality

    store = MemoryStore(path=tmp_path)
    record_user_formality(
        store,
        "The proposal is, therefore, accepted.",
        "en",
    )
    events = query_events(store, kind="formality_score_weekly", limit=5)
    assert len(events) == 1
    assert "score" in events[0]["data"]
    assert 0.0 <= events[0]["data"]["score"] <= 1.0

def test_relax_register_bumps_and_emits(tmp_path):
    from iai_mcp.camouflaging import relax_register

    import iai_mcp.core as core
    core._profile_state["camouflaging_relaxation"] = 0.0

    store = MemoryStore(path=tmp_path)
    relax_register(store, delta=0.25)
    assert abs(core._profile_state["camouflaging_relaxation"] - 0.25) < 1e-9

    events = query_events(store, kind="register_relaxed", limit=5)
    assert len(events) == 1
    assert abs(events[0]["data"]["delta"] - 0.25) < 1e-9
    assert abs(events[0]["data"]["from"] - 0.0) < 1e-9
    assert abs(events[0]["data"]["to"] - 0.25) < 1e-9

def test_relax_register_caps_at_one(tmp_path):
    from iai_mcp.camouflaging import relax_register

    import iai_mcp.core as core
    core._profile_state["camouflaging_relaxation"] = 0.95

    store = MemoryStore(path=tmp_path)
    relax_register(store, delta=0.5)
    assert core._profile_state["camouflaging_relaxation"] == 1.0
