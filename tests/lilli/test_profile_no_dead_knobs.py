
import inspect

from iai_mcp import response_decorator
from iai_mcp.profile import PROFILE_KNOBS, default_state, profile_set

def test_registry_has_10_knobs() -> None:
    assert len(PROFILE_KNOBS) == 10, (
        f"Expected 10 knobs (9 AUTIST + wake_depth), "
        f"got {len(PROFILE_KNOBS)}: {sorted(PROFILE_KNOBS.keys())}"
    )
    autist_specs = [
        s for s in PROFILE_KNOBS.values() if s.requirement_id.startswith("AUTIST-")
    ]
    assert len(autist_specs) == 9
    assert "wake_depth" in PROFILE_KNOBS
    assert "sensory_channel_weights" not in PROFILE_KNOBS
    assert "event_vs_time_cue" not in PROFILE_KNOBS
    assert "alexithymia_accommodation" not in PROFILE_KNOBS
    assert "double_empathy" not in PROFILE_KNOBS
    assert "camouflaging_relaxation" not in PROFILE_KNOBS

def test_profile_set_rejects_sensory_channel_weights() -> None:
    state = default_state()
    result = profile_set("sensory_channel_weights", {"vision": 0.5}, state)
    assert result["status"] == "error", result
    assert result["reason"] == "unknown knob", result

def test_profile_set_rejects_event_vs_time_cue() -> None:
    state = default_state()
    result = profile_set("event_vs_time_cue", "time", state)
    assert result["status"] == "error", result
    assert result["reason"] == "unknown knob", result

def test_profile_set_rejects_alexithymia_accommodation() -> None:
    state = default_state()
    result = profile_set("alexithymia_accommodation", "labeled", state)
    assert result["status"] == "error", result
    assert result["reason"] == "unknown knob", result

def test_profile_set_rejects_double_empathy() -> None:
    state = default_state()
    result = profile_set("double_empathy", False, state)
    assert result["status"] == "error", result
    assert result["reason"] == "unknown knob", result

def test_profile_set_rejects_camouflaging_relaxation() -> None:
    state = default_state()
    result = profile_set("camouflaging_relaxation", 0.8, state)
    assert result["status"] == "error", result
    assert result["reason"] == "unknown knob", result

def test_orphan_helpers_absent_from_dispatch_tuple() -> None:
    assert not hasattr(response_decorator, "_apply_verbosity_level"), (
        "_apply_verbosity_level should be deleted — orphan"
    )
    assert not hasattr(response_decorator, "_apply_surface_language"), (
        "_apply_surface_language should be deleted — orphan"
    )
    assert not hasattr(response_decorator, "_apply_formality_relaxation"), (
        "_apply_formality_relaxation should be deleted — orphan"
    )
    src = inspect.getsource(response_decorator.apply_profile)
    assert "_apply_verbosity_level" not in src, src
    assert "_apply_surface_language" not in src, src
    assert "_apply_formality_relaxation" not in src, src
