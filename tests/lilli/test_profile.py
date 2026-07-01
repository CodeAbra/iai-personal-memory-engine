from __future__ import annotations

from iai_mcp.profile import (
    PHASE_1_LIVE,
    PHASE_2_DEFERRED,
    PHASE_3_DEFERRED,
    PROFILE_KNOBS,
    coerce_json_stringified,
    default_state,
    profile_get,
    profile_set,
)

def test_profile_has_exactly_14_knobs():
    assert len(PROFILE_KNOBS) == 11

def test_phase_1_live_has_exactly_fourteen():
    assert len(PHASE_1_LIVE) == 11
    assert "literal_preservation" in PHASE_1_LIVE
    assert "masking_off" in PHASE_1_LIVE
    assert "task_support" in PHASE_1_LIVE
    assert "scene_construction_scaffold" in PHASE_1_LIVE
    assert "monotropism_depth" in PHASE_1_LIVE
    assert "dunn_quadrant" in PHASE_1_LIVE
    assert "camouflaging_relaxation" in PHASE_1_LIVE
    assert "wake_depth" in PHASE_1_LIVE

def test_phase_2_deferred_is_empty():
    assert PHASE_2_DEFERRED == frozenset()
    assert len(PHASE_2_DEFERRED) == 0

def test_phase_3_deferred_is_empty_after_autist13_flip():
    assert PHASE_3_DEFERRED == frozenset()
    assert len(PHASE_3_DEFERRED) == 0

def test_every_knob_has_autist_requirement_id():
    for name, spec in PROFILE_KNOBS.items():
        if name == "wake_depth":
            assert spec.requirement_id == "MCP-12", (
                f"wake_depth must carry MCP-12 requirement_id, got {spec.requirement_id}"
            )
            continue
        assert spec.requirement_id.startswith("AUTIST-"), (
            f"knob {name} missing AUTIST-* requirement_id"
        )

def test_live_knob_defaults_match_d11():
    state = default_state()
    assert state["literal_preservation"] == "strong"
    assert state["masking_off"] is True
    assert state["task_support"] == "cued_recognition"
    assert state["scene_construction_scaffold"] is True

def test_default_state_excludes_deferred_knobs():
    state = default_state()
    assert set(state.keys()) == PHASE_1_LIVE
    assert len(state) == 11

def test_profile_get_none_returns_total_14():
    state = default_state()
    result = profile_get(None, state)
    assert result["total_knobs"] == 11
    assert len(result["live"]) == 11
    assert len(result["deferred"]) == 0

def test_profile_get_none_live_values_match_d11():
    state = default_state()
    result = profile_get(None, state)
    assert result["live"]["literal_preservation"] == "strong"
    assert result["live"]["masking_off"] is True
    assert result["live"]["task_support"] == "cued_recognition"
    assert result["live"]["scene_construction_scaffold"] is True

def test_profile_get_none_deferred_entries_have_requirement_id():
    state = default_state()
    result = profile_get(None, state)
    for name, entry in result["deferred"].items():
        assert entry["status"] == "not-yet-implemented"
        assert entry["phase"] in (2, 3)
        assert entry["requirement_id"].startswith("AUTIST-")
        assert "description" in entry

def test_profile_get_live_specific_knob():
    state = default_state()
    r = profile_get("literal_preservation", state)
    assert r == {"knob": "literal_preservation", "value": "strong"}

def test_profile_get_monotropism_depth_now_live():
    state = default_state()
    r = profile_get("monotropism_depth", state)
    assert r["knob"] == "monotropism_depth"
    assert "value" in r
    assert r["value"] == {}

def test_profile_get_camouflaging_now_live_after_autist13_flip():
    state = default_state()
    r = profile_get("camouflaging_relaxation", state)
    assert r["knob"] == "camouflaging_relaxation"
    assert "value" in r
    assert r["value"] == 0.0

def test_profile_get_unknown_knob():
    state = default_state()
    r = profile_get("does_not_exist", state)
    assert r == {"knob": "does_not_exist", "status": "unknown"}

def test_profile_set_live_enum_success():
    state = default_state()
    r = profile_set("literal_preservation", "loose", state)
    assert r["status"] == "ok"
    assert r["value"] == "loose"
    assert profile_get("literal_preservation", state)["value"] == "loose"

def test_profile_set_live_enum_rejects_bogus_value():
    state = default_state()
    r = profile_set("literal_preservation", "bogus", state)
    assert r["status"] == "error"
    assert state["literal_preservation"] == "strong"

def test_profile_set_live_bool_rejects_non_bool():
    state = default_state()
    r = profile_set("masking_off", 1, state)
    assert r["status"] == "error"
    assert state["masking_off"] is True

def test_profile_set_live_bool_accepts_true():
    state = default_state()
    r = profile_set("masking_off", False, state)
    assert r["status"] == "ok"
    assert state["masking_off"] is False

def test_profile_set_monotropism_depth_rejects_non_dict():
    state = default_state()
    r = profile_set("monotropism_depth", 3, state)
    assert r["status"] == "error"
    assert "dict" in r["reason"].lower()

def test_profile_set_camouflaging_now_accepts_value_after_autist13_flip():
    state = default_state()
    r = profile_set("camouflaging_relaxation", 0.5, state)
    assert r["status"] == "ok"
    assert state["camouflaging_relaxation"] == 0.5

def test_profile_set_camouflaging_rejects_out_of_range():
    state = default_state()
    r = profile_set("camouflaging_relaxation", 1.5, state)
    assert r["status"] == "error"

def test_profile_set_unknown_knob_returns_unknown_reason():
    state = default_state()
    r = profile_set("does_not_exist", 1, state)
    assert r["status"] == "error"
    assert r["reason"] == "unknown knob"

def test_profile_set_task_support_enum_accepts_blank_recall():
    state = default_state()
    r = profile_set("task_support", "blank_recall", state)
    assert r["status"] == "ok"
    assert state["task_support"] == "blank_recall"

def test_profile_set_scene_construction_scaffold_rejects_string():
    state = default_state()
    r = profile_set("scene_construction_scaffold", "yes", state)
    assert r["status"] == "error"


# --- coerce_json_stringified: undo JSON-type loss at the untyped-client edge ---

def test_coerce_bool_true_false_strings():
    assert coerce_json_stringified("bool", "true") is True
    assert coerce_json_stringified("bool", "false") is False
    assert coerce_json_stringified("bool", "True") is True
    assert coerce_json_stringified("bool", "FALSE") is False


def test_coerce_bool_passes_through_real_bools():
    assert coerce_json_stringified("bool", True) is True
    assert coerce_json_stringified("bool", False) is False


def test_coerce_bool_does_not_widen_strict_contract():
    # The loose spellings profile_set rejects on purpose must NOT be coerced:
    # int 1 and truthy words stay as-is so the strict validator still rejects them.
    assert coerce_json_stringified("bool", 1) == 1
    assert coerce_json_stringified("bool", "yes") == "yes"
    assert coerce_json_stringified("bool", "1") == "1"
    assert coerce_json_stringified("bool", "on") == "on"


def test_coerce_numeric_strings():
    assert coerce_json_stringified("float_range:0.0..1.0", "0.5") == 0.5
    assert coerce_json_stringified("int_range:0..10", "3") == 3
    # non-numeric strings are left for the validator to reject
    assert coerce_json_stringified("float_range:0.0..1.0", "abc") == "abc"


def test_coerce_dict_recurses_on_values():
    out = coerce_json_stringified("dict:str:float_range:0.0..1.0", {"music": "0.9"})
    assert out == {"music": 0.9}


def test_coerce_enum_left_untouched():
    assert coerce_json_stringified("enum:neutral|seeking", "seeking") == "seeking"


def test_coerce_then_set_bool_end_to_end():
    # Mirrors the dispatch edge: coerce a stringified bool, then the strict
    # profile_set accepts it. This is the path that was previously unsettable
    # for bool knobs via the MCP tool.
    state = default_state()
    spec = PROFILE_KNOBS["inertia_awareness"]
    value = coerce_json_stringified(spec.value_schema, "true")
    r = profile_set("inertia_awareness", value, state)
    assert r["status"] == "ok"
    assert state["inertia_awareness"] is True
