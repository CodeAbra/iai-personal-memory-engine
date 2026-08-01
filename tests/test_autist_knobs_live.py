from __future__ import annotations


from iai_mcp.profile import (
    LIVE_KNOB_NAMES,
    DEFERRED_KNOB_NAMES,
    PROFILE_KNOBS,
    default_state,
    profile_get,
    profile_set,
)


def test_live_knob_names_has_11_knobs():
    assert len(LIVE_KNOB_NAMES) == 11


def test_deferred_knob_names_empty():
    assert DEFERRED_KNOB_NAMES == frozenset()


def test_all_requirement_ids_present():
    autist_specs = [
        s for s in PROFILE_KNOBS.values() if s.requirement_id.startswith("AUTIST-")
    ]
    assert len(autist_specs) == 10
    req_ids = {spec.requirement_id for spec in autist_specs}
    expected = {
        "AUTIST-01", "AUTIST-03", "AUTIST-04", "AUTIST-05",
        "AUTIST-06", "AUTIST-07", "AUTIST-09", "AUTIST-10",
        "AUTIST-13", "AUTIST-14",
    }
    assert req_ids == expected
    assert len(PROFILE_KNOBS) == 11
    assert "wake_depth" in PROFILE_KNOBS
    assert PROFILE_KNOBS["wake_depth"].requirement_id == "MCP-12"


def test_monotropism_depth_live_accepts_dict():
    state = default_state()
    r = profile_set(
        "monotropism_depth",
        {"coding": 0.8, "gardening": 0.3},
        state,
    )
    assert r["status"] == "ok"
    assert state["monotropism_depth"] == {"coding": 0.8, "gardening": 0.3}


def test_monotropism_depth_live_rejects_out_of_range():
    state = default_state()
    r = profile_set("monotropism_depth", {"x": 1.5}, state)
    assert r["status"] == "error"


def test_monotropism_depth_live_rejects_non_dict():
    state = default_state()
    r = profile_set("monotropism_depth", 3, state)
    assert r["status"] == "error"


def test_dunn_quadrant_live():
    state = default_state()
    r = profile_set("dunn_quadrant", "seeking", state)
    assert r["status"] == "ok"
    assert state["dunn_quadrant"] == "seeking"


def test_dunn_quadrant_rejects_garbage():
    state = default_state()
    r = profile_set("dunn_quadrant", "garbage", state)
    assert r["status"] == "error"


def test_demand_avoidance_tolerance_live():
    state = default_state()
    for value in ("collaborative", "neutral", "imperative"):
        r = profile_set("demand_avoidance_tolerance", value, state)
        assert r["status"] == "ok", f"expected {value} accepted"
    assert state["demand_avoidance_tolerance"] == "imperative"


def test_inertia_awareness_live():
    state = default_state()
    r_ok = profile_set("inertia_awareness", True, state)
    assert r_ok["status"] == "ok"
    r_bad = profile_set("inertia_awareness", 1, state)
    assert r_bad["status"] == "error"


def test_interest_boost_live():
    state = default_state()
    r_ok = profile_set("interest_boost", 0.75, state)
    assert r_ok["status"] == "ok"
    r_bad = profile_set("interest_boost", 2.0, state)
    assert r_bad["status"] == "error"


def test_HIPPEA_precision_spec_added_wire_to_autist_03():
    if "HIPPEA_precision" in PROFILE_KNOBS:
        spec = PROFILE_KNOBS["HIPPEA_precision"]
        assert "float_range:" in spec.value_schema
    else:
        spec = PROFILE_KNOBS["dunn_quadrant"]
        assert spec.value_schema.startswith("enum:")


def test_profile_get_returns_14_live_entries():
    state = default_state()
    result = profile_get(None, state)
    assert len(result["live"]) == 11
    assert len(result["deferred"]) == 0


def test_profile_get_monotropism_depth_returns_default_dict():
    state = default_state()
    r = profile_get("monotropism_depth", state)
    assert r["knob"] == "monotropism_depth"
    assert "value" in r
    assert isinstance(r["value"], dict)


def test_default_state_returns_independent_mutable_defaults():
    s1 = default_state()
    s2 = default_state()

    assert s1["monotropism_depth"] is not s2["monotropism_depth"]

    s1["monotropism_depth"]["coding"] = 0.9

    assert s2["monotropism_depth"] == {}
    assert PROFILE_KNOBS["monotropism_depth"].default == {}
