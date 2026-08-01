from __future__ import annotations


from iai_mcp.profile import (
    LIVE_KNOB_NAMES,
    DEFERRED_KNOB_NAMES,
    PROFILE_KNOBS,
    default_state,
    profile_get,
    profile_set,
)

def test_camouflaging_relaxation_is_live():
    assert len(LIVE_KNOB_NAMES) == 11
    assert "camouflaging_relaxation" in LIVE_KNOB_NAMES

def test_deferred_knob_names_empty():
    assert len(DEFERRED_KNOB_NAMES) == 0
    assert "camouflaging_relaxation" not in DEFERRED_KNOB_NAMES

def test_knob_spec_is_live():
    spec = PROFILE_KNOBS["camouflaging_relaxation"]
    assert spec.phase == 1
    assert spec.requirement_id == "AUTIST-13"

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
