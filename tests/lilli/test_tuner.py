from __future__ import annotations

from uuid import uuid4


from iai_mcp.events import write_event
from iai_mcp.store import MemoryStore

def test_constants_present():
    from iai_mcp.lilli.profile.tuner import (
        LEARN_RATE,
        MAX_WEIGHT,
        MIN_WEIGHT,
        TRUST_INCREMENT_PER_COMMIT,
        TRUST_DECREMENT_PER_REJECT,
    )
    assert LEARN_RATE == 0.05
    assert MAX_WEIGHT == 5.0
    assert MIN_WEIGHT == 0.0
    assert TRUST_INCREMENT_PER_COMMIT > 0
    assert TRUST_DECREMENT_PER_REJECT > 0

def test_retrieval_feedback_dataclass():
    from iai_mcp.lilli.profile.tuner import RetrievalFeedback

    fb = RetrievalFeedback(
        query_type="fact_lookup",
        hit_ids=[uuid4(), uuid4()],
        used_ids=[],
        corrected=False,
        re_asked=False,
    )
    assert fb.query_type == "fact_lookup"
    assert len(fb.hit_ids) == 2

def test_retrieval_feedback_used_boosts_weights():
    from iai_mcp.lilli.profile.tuner import RetrievalFeedback, update_retrieval_weights

    ids = [uuid4() for _ in range(3)]
    fb = RetrievalFeedback(
        query_type="lookup",
        hit_ids=ids,
        used_ids=ids[:3],
        corrected=False,
        re_asked=False,
    )
    before = {"W_COSINE": 1.0, "W_AAAK": 0.3, "W_DEGREE": 0.1, "W_AGE": 0.05}
    after = update_retrieval_weights(fb, before)
    assert after["W_COSINE"] > before["W_COSINE"]

def test_retrieval_feedback_corrected_reduces_weights():
    from iai_mcp.lilli.profile.tuner import RetrievalFeedback, update_retrieval_weights

    ids = [uuid4() for _ in range(3)]
    fb = RetrievalFeedback(
        query_type="lookup",
        hit_ids=ids,
        used_ids=[],
        corrected=True,
        re_asked=False,
    )
    before = {"W_COSINE": 1.0, "W_AAAK": 0.3, "W_DEGREE": 0.1, "W_AGE": 0.05}
    after = update_retrieval_weights(fb, before)
    assert after["W_COSINE"] < before["W_COSINE"]

def test_retrieval_feedback_re_asked_reduces_weights():
    from iai_mcp.lilli.profile.tuner import RetrievalFeedback, update_retrieval_weights

    ids = [uuid4() for _ in range(3)]
    fb = RetrievalFeedback(
        query_type="lookup",
        hit_ids=ids,
        used_ids=[],
        corrected=False,
        re_asked=True,
    )
    before = {"W_COSINE": 1.0, "W_AAAK": 0.3, "W_DEGREE": 0.1, "W_AGE": 0.05}
    after = update_retrieval_weights(fb, before)
    assert after["W_COSINE"] < before["W_COSINE"]

def test_retrieval_weights_bounded():
    from iai_mcp.lilli.profile.tuner import (
        MAX_WEIGHT, MIN_WEIGHT, RetrievalFeedback, update_retrieval_weights,
    )

    ids = [uuid4() for _ in range(3)]
    weights = {"W_COSINE": 1.0, "W_AAAK": 0.3, "W_DEGREE": 0.1, "W_AGE": 0.05}
    for _ in range(1000):
        fb = RetrievalFeedback(
            query_type="x", hit_ids=ids, used_ids=ids,
            corrected=False, re_asked=False,
        )
        weights = update_retrieval_weights(fb, weights)
    assert weights["W_COSINE"] <= MAX_WEIGHT
    assert weights["W_COSINE"] >= MIN_WEIGHT

def test_does_not_mutate_input():
    from iai_mcp.lilli.profile.tuner import RetrievalFeedback, update_retrieval_weights

    ids = [uuid4()]
    before = {"W_COSINE": 1.0}
    before_copy = dict(before)
    fb = RetrievalFeedback("x", ids, ids, False, False)
    update_retrieval_weights(fb, before)
    assert before == before_copy

def test_identity_refinement_increases_s5_trust(tmp_path):
    from iai_mcp.lilli.profile.tuner import refine_s5_trust_score

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
    from iai_mcp.lilli.profile.tuner import refine_s5_trust_score

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
    from iai_mcp.lilli.profile.tuner import refine_s5_trust_score

    store = MemoryStore(path=tmp_path)
    anchor_id = uuid4()
    for _ in range(100):
        write_event(
            store, kind="s5_invariant_update",
            data={"anchor_id": str(anchor_id), "agree_count": 3},
        )
    score = refine_s5_trust_score(store, anchor_id, current=0.5)
    assert 0.0 <= score <= 1.0

def test_trust_unrelated_anchor_unaffected(tmp_path):
    from iai_mcp.lilli.profile.tuner import refine_s5_trust_score

    store = MemoryStore(path=tmp_path)
    anchor_id = uuid4()
    other_id = uuid4()
    for _ in range(50):
        write_event(
            store, kind="s5_invariant_update",
            data={"anchor_id": str(other_id), "agree_count": 5},
        )
    score = refine_s5_trust_score(store, anchor_id, current=0.5)
    assert score == 0.5
