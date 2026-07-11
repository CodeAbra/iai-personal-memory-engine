from __future__ import annotations


from iai_mcp.profile import (
    PHASE_1_LIVE,
    PHASE_2_DEFERRED,
    PHASE_3_DEFERRED,
    PROFILE_KNOBS,
    default_state,
    profile_get,
    profile_set,
)

def test_phase_1_live_is_14():
    assert len(PHASE_1_LIVE) == 11
    assert "camouflaging_relaxation" in PHASE_1_LIVE

def test_phase_3_deferred_is_empty():
    assert len(PHASE_3_DEFERRED) == 0
    assert "camouflaging_relaxation" not in PHASE_3_DEFERRED

def test_phase_2_deferred_is_empty():
    assert len(PHASE_2_DEFERRED) == 0

def test_knob_spec_phase_is_1():
    spec = PROFILE_KNOBS["camouflaging_relaxation"]
    assert spec.phase == 1
    assert spec.requirement_id == "AUTIST-13"
    assert "Phase 3" not in spec.description

def test_core_import_succeeds_with_deferred_knobs_zero():
    import iai_mcp.core as core
    assert len(core.DEFERRED_KNOBS) == 0

def test_profile_get_returns_14():
    state = default_state()
    r = profile_get(None, state)
    assert r["total_knobs"] == 11
    assert len(r["live"]) == 11
    assert len(r["deferred"]) == 0

def test_profile_get_camouflaging_returns_live_value():
    state = default_state()
    r = profile_get("camouflaging_relaxation", state)
    assert r["knob"] == "camouflaging_relaxation"
    assert r["value"] == 0.0

def test_profile_set_camouflaging_accepts_in_range():
    state = default_state()
    r = profile_set("camouflaging_relaxation", 0.3, state)
    assert r["status"] == "ok"
    assert state["camouflaging_relaxation"] == 0.3

def test_profile_set_camouflaging_rejects_out_of_range():
    state = default_state()
    r = profile_set("camouflaging_relaxation", 1.5, state)
    assert r["status"] == "error"

def test_profile_set_camouflaging_rejects_negative():
    state = default_state()
    r = profile_set("camouflaging_relaxation", -0.1, state)
    assert r["status"] == "error"

def test_default_state_includes_camouflaging_relaxation():
    state = default_state()
    assert "camouflaging_relaxation" in state
    assert state["camouflaging_relaxation"] == 0.0
