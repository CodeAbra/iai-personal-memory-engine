from __future__ import annotations

import json

import pytest

from iai_mcp.profile import (
    default_state,
    load_profile_overrides,
    save_profile_overrides,
)


# --- pure persistence: save / load / re-validation -------------------------

def test_save_load_roundtrip(tmp_path):
    path = tmp_path / ".profile-knobs.json"
    save_profile_overrides(
        {"inertia_awareness": True, "dunn_quadrant": "low-registration"}, path=path
    )
    assert path.exists()
    assert load_profile_overrides(path=path) == {
        "inertia_awareness": True,
        "dunn_quadrant": "low-registration",
    }


def test_load_missing_file_returns_empty(tmp_path):
    assert load_profile_overrides(path=tmp_path / "nope.json") == {}


def test_load_ignores_non_dict_json(tmp_path):
    path = tmp_path / ".profile-knobs.json"
    path.write_text(json.dumps(["not", "a", "dict"]))
    assert load_profile_overrides(path=path) == {}


def test_load_drops_unknown_invalid_and_out_of_range(tmp_path):
    path = tmp_path / ".profile-knobs.json"
    path.write_text(
        json.dumps(
            {
                "inertia_awareness": True,          # valid -> kept
                "dunn_quadrant": "not-a-quadrant",  # invalid enum -> dropped
                "not_a_real_knob": 1,               # unknown -> dropped
                "camouflaging_relaxation": 5.0,     # out of 0..1 range -> dropped
            }
        )
    )
    assert load_profile_overrides(path=path) == {"inertia_awareness": True}


# --- core integration: rehydrate, pin, persist, bayesian skip --------------

@pytest.fixture
def fresh_core():
    import iai_mcp.core as core

    saved_state = dict(core._profile_state)
    saved_pins = set(core._pinned_knobs)
    core._profile_state.clear()
    core._profile_state.update(default_state())
    core._pinned_knobs.clear()
    try:
        yield core
    finally:
        core._profile_state.clear()
        core._profile_state.update(saved_state)
        core._pinned_knobs.clear()
        core._pinned_knobs.update(saved_pins)


def test_rehydrate_applies_overrides_and_pins(fresh_core, tmp_path):
    path = tmp_path / ".profile-knobs.json"
    save_profile_overrides(
        {"inertia_awareness": True, "dunn_quadrant": "seeking"}, path=path
    )

    restored = fresh_core.rehydrate_profile_overrides(path=path)

    assert restored == {"inertia_awareness": True, "dunn_quadrant": "seeking"}
    assert fresh_core._profile_state["inertia_awareness"] is True
    assert fresh_core._profile_state["dunn_quadrant"] == "seeking"
    assert {"inertia_awareness", "dunn_quadrant"} <= fresh_core._pinned_knobs


def test_rehydrate_missing_file_is_noop(fresh_core, tmp_path):
    assert fresh_core.rehydrate_profile_overrides(path=tmp_path / "absent.json") == {}
    # defaults intact, nothing pinned
    assert fresh_core._profile_state["inertia_awareness"] is False
    assert fresh_core._pinned_knobs == set()


def test_profile_set_dispatch_persists_and_pins(fresh_core, tmp_path, monkeypatch):
    import iai_mcp.lilli.profile.knobs as knobs

    path = tmp_path / ".profile-knobs.json"
    monkeypatch.setattr(knobs, "PROFILE_OVERRIDES_PATH", path, raising=False)

    r = fresh_core.dispatch(
        None, "profile_set", {"knob": "inertia_awareness", "value": True}
    )

    assert r["status"] == "ok"
    assert "inertia_awareness" in fresh_core._pinned_knobs
    assert load_profile_overrides(path=path) == {"inertia_awareness": True}


def test_rejected_set_is_not_pinned_or_persisted(fresh_core, tmp_path, monkeypatch):
    import iai_mcp.lilli.profile.knobs as knobs

    path = tmp_path / ".profile-knobs.json"
    monkeypatch.setattr(knobs, "PROFILE_OVERRIDES_PATH", path, raising=False)

    r = fresh_core.dispatch(
        None, "profile_set", {"knob": "dunn_quadrant", "value": "bogus"}
    )

    assert r["status"] == "error"
    assert "dunn_quadrant" not in fresh_core._pinned_knobs
    assert not path.exists()


def test_pinned_knob_survives_contrary_bayesian_signal(fresh_core):
    fresh_core._profile_state["inertia_awareness"] = True
    fresh_core._pinned_knobs.add("inertia_awareness")

    r = fresh_core.dispatch(
        None,
        "profile_update_from_signal",
        {"knob": "inertia_awareness", "signal": "correction", "observed": False},
    )

    assert r.get("pinned") is True
    assert fresh_core._profile_state["inertia_awareness"] is True
