from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.events import write_event
from iai_mcp.profile import SIGNAL_WEIGHT, bayesian_update
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord

def test_signal_weights_d20():
    assert SIGNAL_WEIGHT["implicit"] == 0.3
    assert SIGNAL_WEIGHT["inferred"] == 0.5
    assert SIGNAL_WEIGHT["explicit"] == 1.0

def test_bayesian_update_bool_implicit():
    state = {"masking_off": True}
    posterior = {}
    new_val, new_post = bayesian_update(
        "masking_off", "implicit", False, state, posterior,
    )
    assert "masking_off" in new_post
    assert new_post["masking_off"]["beta"] > 1.0

def test_bayesian_update_bool_explicit_flips():
    state = {"masking_off": True}
    posterior = {}
    new_val, new_post = bayesian_update(
        "masking_off", "explicit", False, state, posterior,
    )
    assert new_val is False

def test_bayesian_update_enum_dominant_vote():
    state = {"literal_preservation": "strong"}
    posterior = {}
    for _ in range(3):
        _, posterior = bayesian_update(
            "literal_preservation", "explicit", "medium",
            state, posterior,
        )
    assert state["literal_preservation"] == "medium"

def test_bayesian_update_float_converges():
    state = {"interest_boost": 0.0}
    posterior = {}
    for _ in range(10):
        _, posterior = bayesian_update(
            "interest_boost", "implicit", 0.6, state, posterior,
        )
    assert abs(state["interest_boost"] - 0.6) < 0.05

def test_bayesian_update_respects_signal_weight():
    state = {"masking_off": True}
    posterior = {}
    _, posterior = bayesian_update(
        "masking_off", "explicit", False, state, posterior,
    )
    for _ in range(3):
        _, posterior = bayesian_update(
            "masking_off", "implicit", True, state, posterior,
        )
    assert state["masking_off"] is False

def test_bayesian_update_unknown_knob_noop():
    state = {}
    posterior = {}
    val, post = bayesian_update("does_not_exist", "explicit", True, state, posterior)
    assert val is None
    assert "does_not_exist" not in post

def test_bayesian_update_dict_per_key():
    state = {"monotropism_depth": {}}
    posterior = {}
    _, posterior = bayesian_update(
        "monotropism_depth", "explicit",
        {"coding": 0.8, "gardening": 0.3}, state, posterior,
    )
    assert "coding" in state["monotropism_depth"]
    assert "gardening" in state["monotropism_depth"]
    assert abs(state["monotropism_depth"]["coding"] - 0.8) < 0.01
    assert abs(state["monotropism_depth"]["gardening"] - 0.3) < 0.01

def test_bayesian_update_int_range():
    pytest.skip("no int_range knob in the registry (all knobs are float/dict/enum/bool)")

def test_trajectory_m4_computed():
    state = {"interest_boost": 0.0}
    posterior = {}
    for _ in range(10):
        _, posterior = bayesian_update(
            "interest_boost", "explicit", 0.5, state, posterior,
        )
    early_weight = posterior["interest_boost"]["total_weight"]
    for _ in range(20):
        _, posterior = bayesian_update(
            "interest_boost", "explicit", 0.5, state, posterior,
        )
    late_weight = posterior["interest_boost"]["total_weight"]
    assert late_weight > early_weight

def _record(vec, tier="semantic", s5_trust_score=0.5, language="en", tags=None):
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface="x",
        aaak_index="",
        embedding=vec,
        community_id=None,
        centrality=0.0,
        detail_level=3,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=list(tags or []),
        language=language,
        s5_trust_score=s5_trust_score,
    )

def test_identity_refinement_increases_s5_trust(tmp_path):
    from iai_mcp.learn import refine_s5_trust_score

    store = MemoryStore(path=tmp_path)
    anchor_id = uuid4()
    for _ in range(3):
        write_event(
            store, kind="s5_invariant_update",
            data={"anchor_id": str(anchor_id), "agree_count": 3},
        )
    new_score = refine_s5_trust_score(store, anchor_id, current=0.5)
    assert new_score > 0.5

def test_identity_refinement_decreases_on_rejected(tmp_path):
    from iai_mcp.learn import refine_s5_trust_score

    store = MemoryStore(path=tmp_path)
    anchor_id = uuid4()
    for _ in range(5):
        write_event(
            store, kind="s5_invariant_proposal",
            data={"anchor_id": str(anchor_id), "passes_vigilance": False},
        )
    new_score = refine_s5_trust_score(store, anchor_id, current=0.5)
    assert new_score < 0.5

def test_identity_refinement_clamps_0_1(tmp_path):
    from iai_mcp.learn import refine_s5_trust_score

    store = MemoryStore(path=tmp_path)
    anchor_id = uuid4()
    for _ in range(100):
        write_event(
            store, kind="s5_invariant_update",
            data={"anchor_id": str(anchor_id), "agree_count": 3},
        )
    score = refine_s5_trust_score(store, anchor_id, current=0.5)
    assert 0.0 <= score <= 1.0
