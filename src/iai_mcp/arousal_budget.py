from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

AROUSAL_DECAY_RATE = 0.95
AROUSAL_MAX = 1.0
AROUSAL_MIN = 0.0
STRESS_THRESHOLD_HIGH = 0.7
STRESS_THRESHOLD_LOW = 0.3

BUDGET_MIN_TOKENS = 800
BUDGET_MAX_TOKENS = 3000
HOPS_HIGH_STRESS = 1
HOPS_LOW_STRESS = 2
RANK_THRESHOLD_HIGH = 0.6
RANK_THRESHOLD_LOW = 0.3

BASTA_WRITE_CAP_PER_MINUTE = 10
BASTA_CAPACITY_RATIO = 0.8


@dataclass
class ArousalState:
    level: float = 0.5
    last_updated: float = field(default_factory=time.time)
    error_count: int = 0
    success_count: int = 0
    queries_last_minute: int = 0


@dataclass
class RetrievalParams:
    budget_tokens: int
    max_hops: int
    rank_threshold: float
    mode: str


def update_arousal(state: ArousalState, event: str) -> ArousalState:
    now = time.time()
    elapsed = now - state.last_updated
    state.level *= AROUSAL_DECAY_RATE ** elapsed
    state.last_updated = now

    if event == "recall_failed":
        state.level = min(AROUSAL_MAX, state.level + 0.15)
        state.error_count += 1
    elif event == "error":
        state.level = min(AROUSAL_MAX, state.level + 0.2)
        state.error_count += 1
    elif event == "rapid_query":
        state.level = min(AROUSAL_MAX, state.level + 0.1)
        state.queries_last_minute += 1
    elif event == "recall_success":
        state.level = max(AROUSAL_MIN, state.level - 0.05)
        state.success_count += 1
    elif event == "idle":
        state.level = max(AROUSAL_MIN, state.level - 0.1)
    elif event == "sleep_complete":
        state.level = max(AROUSAL_MIN, state.level - 0.2)

    state.level = max(AROUSAL_MIN, min(AROUSAL_MAX, state.level))
    return state


def compute_retrieval_params(arousal: ArousalState) -> RetrievalParams:
    level = arousal.level

    if level >= STRESS_THRESHOLD_HIGH:
        return RetrievalParams(
            budget_tokens=BUDGET_MIN_TOKENS,
            max_hops=HOPS_HIGH_STRESS,
            rank_threshold=RANK_THRESHOLD_HIGH,
            mode="monotropic_tunnel",
        )
    elif level <= STRESS_THRESHOLD_LOW:
        return RetrievalParams(
            budget_tokens=BUDGET_MAX_TOKENS,
            max_hops=HOPS_LOW_STRESS,
            rank_threshold=RANK_THRESHOLD_LOW,
            mode="associative_dream",
        )
    else:
        progress = (level - STRESS_THRESHOLD_LOW) / (STRESS_THRESHOLD_HIGH - STRESS_THRESHOLD_LOW)
        budget = int(BUDGET_MAX_TOKENS - progress * (BUDGET_MAX_TOKENS - BUDGET_MIN_TOKENS))
        rank = RANK_THRESHOLD_LOW + progress * (RANK_THRESHOLD_HIGH - RANK_THRESHOLD_LOW)
        return RetrievalParams(
            budget_tokens=budget,
            max_hops=HOPS_LOW_STRESS,
            rank_threshold=rank,
            mode="balanced",
        )
