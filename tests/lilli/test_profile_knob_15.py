from __future__ import annotations

from iai_mcp.profile import (
    LIVE_KNOB_NAMES,
    PROFILE_KNOBS,
    default_state,
    profile_get,
    profile_set,
)

def test_registry_has_11_knobs():
    assert len(PROFILE_KNOBS) == 11

def test_wake_depth_knob_exists():
    assert "wake_depth" in PROFILE_KNOBS

def test_wake_depth_knob_shape():
    spec = PROFILE_KNOBS["wake_depth"]
    assert spec.value_schema == "enum:minimal|standard|deep", spec.value_schema
    assert spec.default == "minimal"
    assert spec.phase == 1
    assert spec.requirement_id == "MCP-12"

def test_wake_depth_is_live():
    assert "wake_depth" in LIVE_KNOB_NAMES

def test_wake_depth_default_applies():
    state = default_state()
    assert state.get("wake_depth") == "minimal"

def test_wake_depth_set_valid():
    state = default_state()
    r = profile_set("wake_depth", "standard", state)
    assert r["status"] == "ok"
    assert state["wake_depth"] == "standard"
    r2 = profile_set("wake_depth", "deep", state)
    assert r2["status"] == "ok"

def test_wake_depth_set_invalid_rejected():
    state = default_state()
    r = profile_set("wake_depth", "weird", state)
    assert r["status"] == "error"

def test_profile_get_wake_depth_returns_value():
    state = default_state()
    r = profile_get("wake_depth", state)
    assert r["knob"] == "wake_depth"
    assert r["value"] == "minimal"
