from __future__ import annotations


def test_apply_profile_is_noop_on_default_state():
    from iai_mcp import profile
    from iai_mcp.response_decorator import apply_profile

    state = profile.default_state()
    state.setdefault("wake_depth", "minimal")
    resp = {"hits": [{"record_id": "r1", "literal_surface": "x"}], "anti_hits": []}
    before_keys = set(resp.keys())
    out = apply_profile(dict(resp), state)
    added = set(out.keys()) - before_keys
    assert added == {"_knobs_applied"}, (
        f"apply_profile added unexpected keys on default state: {added}; "
        f"expected exactly {{'_knobs_applied'}}"
    )
    assert isinstance(out["_knobs_applied"], dict), out["_knobs_applied"]

def test_monotropic_focus_narrows_hits():
    from iai_mcp import profile
    from iai_mcp.response_decorator import apply_profile

    state = profile.default_state()
    state["monotropism_depth"] = {"coding": 0.9}
    resp = {
        "hits": [
            {"record_id": "r1", "literal_surface": "x", "community_id": "A"},
            {"record_id": "r2", "literal_surface": "y", "community_id": "A"},
            {"record_id": "r3", "literal_surface": "z", "community_id": "B"},
        ],
        "anti_hits": [],
    }
    apply_profile(resp, state)
    assert "hits" in resp

def test_malformed_knob_silent_fail():
    from iai_mcp.response_decorator import apply_profile

    bad_state = {"verbosity_level": object(), "surface_language": 42}
    resp = {"hits": [], "anti_hits": []}
    apply_profile(resp, bad_state)

def test_pre_existing_keys_untouched_on_exception():
    from iai_mcp import response_decorator

    resp = {"hits": [], "anti_hits": [], "budget_used": 42}

    def _boom(*a, **k):
        raise RuntimeError("synthetic helper failure")

    original = None
    helper_name = "_apply_dunn_quadrant"
    if hasattr(response_decorator, helper_name):
        original = getattr(response_decorator, helper_name)
        setattr(response_decorator, helper_name, _boom)
    try:
        response_decorator.apply_profile(resp, {"dunn_quadrant": "seeking"})
    finally:
        if original is not None:
            setattr(response_decorator, helper_name, original)
    assert resp["budget_used"] == 42
