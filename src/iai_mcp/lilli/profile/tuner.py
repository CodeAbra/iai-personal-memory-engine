from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from iai_mcp.events import query_events
from iai_mcp.store import MemoryStore


LEARN_RATE: float = 0.05
MAX_WEIGHT: float = 5.0
MIN_WEIGHT: float = 0.0


@dataclass
class RetrievalFeedback:

    query_type: str
    hit_ids: list[UUID]
    used_ids: list[UUID] = field(default_factory=list)
    corrected: bool = False
    re_asked: bool = False


def update_retrieval_weights(
    feedback: RetrievalFeedback,
    current_weights: dict[str, float],
) -> dict[str, float]:
    w = dict(current_weights)
    delta = 0.0
    if feedback.hit_ids:
        hits_set = set(feedback.hit_ids)
        used_set = set(feedback.used_ids)
        use_rate = len(hits_set & used_set) / len(feedback.hit_ids)
        delta = (use_rate - 0.5) * LEARN_RATE
    if feedback.corrected:
        delta -= LEARN_RATE
    if feedback.re_asked:
        delta -= LEARN_RATE * 0.5

    w_cos = w.get("W_COSINE", 1.0)
    w["W_COSINE"] = max(MIN_WEIGHT, min(MAX_WEIGHT, w_cos + delta))

    for k in ("W_AAAK", "W_DEGREE", "W_AGE"):
        if k in w:
            w[k] = max(MIN_WEIGHT, min(MAX_WEIGHT, w[k]))
    return w


TRUST_INCREMENT_PER_COMMIT: float = 0.02
TRUST_DECREMENT_PER_REJECT: float = 0.01


def refine_s5_trust_score(
    store: MemoryStore,
    record_id: UUID,
    current: float,
) -> float:
    updates = query_events(store, kind="s5_invariant_update", limit=200)
    commits = sum(
        1 for e in updates
        if e["data"].get("anchor_id") == str(record_id)
        and int(e["data"].get("agree_count", 0)) >= 3
    )
    rejects_events = query_events(store, kind="s5_invariant_proposal", limit=500)
    rejects = sum(
        1 for e in rejects_events
        if e["data"].get("anchor_id") == str(record_id)
        and not e["data"].get("passes_vigilance", True)
    )
    new_score = (
        current
        + TRUST_INCREMENT_PER_COMMIT * commits
        - TRUST_DECREMENT_PER_REJECT * rejects
    )
    return max(0.0, min(1.0, new_score))
