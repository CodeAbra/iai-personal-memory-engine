from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from iai_mcp.profile import default_state, profile_modulation_for_record
from iai_mcp.response_decorator import HELPER_TO_KNOB_ID, apply_profile
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _hit(literal: str = "h", suggestions: list[str] | None = None) -> dict:
    return {
        "record_id": "00000000-0000-0000-0000-000000000001",
        "score": 0.5,
        "reason": "test",
        "literal_surface": literal,
        "adjacent_suggestions": suggestions or [],
    }


def _resp(hits: list[dict], **extra) -> dict:
    base: dict = {"hits": hits}
    base.update(extra)
    return base


def test_knobs_applied_present_after_apply_profile() -> None:
    response = _resp([_hit()])
    profile = default_state()
    apply_profile(response, profile)
    assert "_knobs_applied" in response, response
    assert isinstance(response["_knobs_applied"], dict), response["_knobs_applied"]


def test_knobs_applied_provenance_shape() -> None:
    response = _resp([_hit()])
    apply_profile(response, default_state())
    assert response["_knobs_applied"], "expected at least one helper entry"
    for knob_id, provenance in response["_knobs_applied"].items():
        assert isinstance(provenance, str), (knob_id, provenance)
        assert provenance, (knob_id, provenance)
        parts = provenance.split(":")
        assert len(parts) >= 2, (knob_id, provenance)
        assert parts[0].endswith(".py"), (knob_id, provenance)


def test_knobs_applied_deterministic() -> None:
    response_1 = _resp([_hit()])
    response_2 = _resp([_hit()])
    profile = default_state()
    apply_profile(response_1, profile)
    apply_profile(response_2, profile)
    assert response_1["_knobs_applied"] == response_2["_knobs_applied"]


def test_knobs_applied_preserves_upstream_seeded_entries() -> None:
    response = _resp(
        [_hit()],
        _knobs_applied={
            "AUTIST-03": "profile.py:profile_modulation_for_record:dunn_quadrant=seeking",
            "AUTIST-09": "profile.py:profile_modulation_for_record:interest_boost",
            "MCP-12": "session.py:assemble_session_start:wake_depth=minimal",
        },
    )
    profile = default_state()
    apply_profile(response, profile)
    ka = response["_knobs_applied"]
    assert "AUTIST-03" in ka
    assert "profile.py" in ka["AUTIST-03"]
    assert "AUTIST-09" in ka
    assert "profile.py" in ka["AUTIST-09"]
    assert "MCP-12" in ka
    assert "session.py" in ka["MCP-12"]


def test_knobs_applied_no_op_markers_for_pda_neutral() -> None:
    response = _resp([_hit()])
    profile = default_state()
    profile["demand_avoidance_tolerance"] = "neutral"
    apply_profile(response, profile)
    ka = response["_knobs_applied"]
    assert "AUTIST-05" in ka
    assert "no-op" in ka["AUTIST-05"], ka["AUTIST-05"]
    assert "neutral" in ka["AUTIST-05"], ka["AUTIST-05"]


def test_knobs_applied_no_op_markers_for_inertia_off() -> None:
    response = _resp([_hit()])
    profile = default_state()
    apply_profile(response, profile)
    ka = response["_knobs_applied"]
    assert "AUTIST-10" in ka
    assert "no-op" in ka["AUTIST-10"], ka["AUTIST-10"]


def test_knobs_applied_no_op_marker_for_scene_construction_off() -> None:
    response = _resp([_hit()])
    profile = default_state()
    profile["scene_construction_scaffold"] = False
    apply_profile(response, profile)
    ka = response["_knobs_applied"]
    assert "AUTIST-14" in ka
    assert "no-op" in ka["AUTIST-14"], ka["AUTIST-14"]


def test_helper_to_knob_id_has_10_verified_entries() -> None:
    assert len(HELPER_TO_KNOB_ID) == 10, (
        f"HELPER_TO_KNOB_ID must have exactly 10 verified entries "
        f"(7 helper + 2 upstream-gains + 1 wake_depth seed), "
        f"got {len(HELPER_TO_KNOB_ID)}: {HELPER_TO_KNOB_ID}"
    )
    knob_ids = set(HELPER_TO_KNOB_ID.values())
    assert len(knob_ids) == 10, knob_ids
    for removed in ("AUTIST-02", "AUTIST-08", "AUTIST-11", "AUTIST-12", "AUTIST-13"):
        assert removed not in knob_ids, (
            f"{removed} was removed; do not re-add"
        )
    expected_autist = {f"AUTIST-{i:02d}" for i in (1, 3, 4, 5, 6, 7, 9, 10, 14)}
    assert expected_autist.issubset(knob_ids), (expected_autist - knob_ids)
    assert "MCP-12" in knob_ids


def test_profile_modulation_records_into_accumulator() -> None:
    from iai_mcp import core

    now = datetime.now(timezone.utc)
    cid = uuid4()
    rec = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="x",
        aaak_index="",
        embedding=[0.0] * EMBED_DIM,
        community_id=cid,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        language="en",
        tags=[],
    )
    state = default_state()
    state["monotropism_depth"] = {"coding": 0.5}
    state["interest_boost"] = 0.3
    state["dunn_quadrant"] = "seeking"

    saved_names = dict(core._community_names_cache)
    core.set_community_names({str(cid): "coding"})
    try:
        accumulator: dict[str, str] = {}
        gains = profile_modulation_for_record(rec, state, knobs_applied=accumulator)
    finally:
        core.set_community_names(saved_names)
    assert "monotropism_depth" in gains
    assert "AUTIST-01" in accumulator, accumulator
    assert "AUTIST-09" in accumulator, accumulator
    assert "AUTIST-03" in accumulator, accumulator
    assert "profile.py" in accumulator["AUTIST-01"], accumulator["AUTIST-01"]
    assert "profile.py" in accumulator["AUTIST-03"], accumulator["AUTIST-03"]
    assert "profile.py" in accumulator["AUTIST-09"], accumulator["AUTIST-09"]


def test_profile_modulation_back_compat_without_kwarg() -> None:
    now = datetime.now(timezone.utc)
    rec = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="x",
        aaak_index="",
        embedding=[0.0] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        language="en",
        tags=["domain:coding"],
    )
    state = default_state()
    state["interest_boost"] = 0.3
    gains = profile_modulation_for_record(rec, state)
    assert "interest_boost" in gains


def _seed_one_record(store, text: str = "reference content", *, community_id=None) -> None:
    now = datetime.now(timezone.utc)
    rec = MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=community_id,
        centrality=0.5,
        detail_level=3,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        language="en",
        tags=[],
    )
    store.insert(rec)


def _call_production_dispatch_path(tmp_path, monkeypatch) -> dict:
    from iai_mcp import core
    from iai_mcp.store import MemoryStore

    saved_profile = dict(core._profile_state)
    pending = {"sknobs": True}

    def _load_state():
        return {"first_turn_pending": dict(pending)}

    def _save_state(state):
        fresh = state.get("first_turn_pending", {})
        pending.clear()
        pending.update(fresh)

    monkeypatch.setattr("iai_mcp.daemon_state.load_state", _load_state)
    monkeypatch.setattr("iai_mcp.daemon_state.save_state", _save_state)

    store = MemoryStore(path=tmp_path)
    cid = uuid4()
    _seed_one_record(store, "reference content for knobs telemetry test", community_id=cid)

    saved_names = dict(core._community_names_cache)
    core.set_community_names({str(cid): "coding"})
    try:
        core._profile_state["dunn_quadrant"] = "seeking"
        core._profile_state["interest_boost"] = 0.5
        core._profile_state["monotropism_depth"] = {"coding": 0.5}

        params = {
            "cue": "reference content for knobs telemetry test",
            "session_id": "sknobs",
            "cue_embedding": [0.1] * EMBED_DIM,
        }
        response = core.dispatch(store, "memory_recall", params)
    finally:
        core._profile_state.clear()
        core._profile_state.update(saved_profile)
        core.set_community_names(saved_names)
    return response


def test_knobs_applied_via_production_dispatch_path(tmp_path, monkeypatch) -> None:
    response = _call_production_dispatch_path(tmp_path, monkeypatch)

    assert "_knobs_applied" in response, sorted(response.keys())
    ka = response["_knobs_applied"]
    assert isinstance(ka, dict), ka

    assert len(ka) == 10, ka

    for required in ("AUTIST-03", "AUTIST-09", "MCP-12"):
        assert required in ka, (required, sorted(ka.keys()))
    assert "profile.py" in ka["AUTIST-03"], ka["AUTIST-03"]
    assert "profile.py" in ka["AUTIST-09"], ka["AUTIST-09"]
    assert "session.py" in ka["MCP-12"], ka["MCP-12"]

    for removed in ("AUTIST-02", "AUTIST-08", "AUTIST-11", "AUTIST-12", "AUTIST-13"):
        assert removed not in ka, (removed, ka)

    for autist in (
        "AUTIST-01", "AUTIST-03", "AUTIST-04", "AUTIST-05",
        "AUTIST-06", "AUTIST-07", "AUTIST-09", "AUTIST-10",
        "AUTIST-14",
    ):
        assert autist in ka, (autist, sorted(ka.keys()))
