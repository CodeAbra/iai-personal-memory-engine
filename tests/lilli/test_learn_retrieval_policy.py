from __future__ import annotations

from uuid import uuid4


def test_retrieval_feedback_dataclass():
    from iai_mcp.learn import RetrievalFeedback

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
    from iai_mcp.learn import RetrievalFeedback, update_retrieval_weights

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
    from iai_mcp.learn import RetrievalFeedback, update_retrieval_weights

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
    from iai_mcp.learn import RetrievalFeedback, update_retrieval_weights

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
    from iai_mcp.learn import MAX_WEIGHT, MIN_WEIGHT, RetrievalFeedback, update_retrieval_weights

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

def test_retrieval_policy_per_query_type():
    from iai_mcp.learn import RetrievalFeedback, update_retrieval_weights

    ids = [uuid4()]
    w1 = {"W_COSINE": 1.0, "W_AAAK": 0.3, "W_DEGREE": 0.1, "W_AGE": 0.05}
    w2 = {"W_COSINE": 1.0, "W_AAAK": 0.3, "W_DEGREE": 0.1, "W_AGE": 0.05}
    fb_a = RetrievalFeedback("A", ids, ids, False, False)
    w1 = update_retrieval_weights(fb_a, w1)
    fb_b = RetrievalFeedback("B", ids, [], True, False)
    w2 = update_retrieval_weights(fb_b, w2)
    assert w1["W_COSINE"] > w2["W_COSINE"]
