"""Labelled distinct-vs-duplicate eval set for the write-time dedup gate.

Ground truth is synthetic exact cosine, never the real embedder — the same
approach as ``tests/test_pattern_separation_never_merge_bypass.py``, so every
pair's similarity is an exact, reproducible number, not an emergent embedder
output.

Verdicts are relative to the write-time gate's ``near_dup_threshold`` (default
0.92, see ``daemon_config.py``): a pair at or above it is labelled
``"duplicate"`` (expected SKIP-merge); below it, ``"distinct"`` (expected
INSERT, its own row).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Each pair gets its own 2-d rotation plane (axis_index, axis_index + 1) inside
# a shared embedding space, so anchors/candidates from DIFFERENT pairs are
# mutually orthogonal (cosine 0) and never cross-contaminate when every pair
# is inserted sequentially into the SAME store (as the eval-set tests do).
# EMBED_DIM must cover the highest axis_index used below, plus one.
EMBED_DIM = 18
NEAR_DUP_THRESHOLD_DEFAULT: float = 0.92
OLD_CAPTURE_THRESHOLD: float = 0.95


def _make_embedding_at_cosine(
    cos_target: float, embed_dim: int = EMBED_DIM, axis_index: int = 0,
) -> list[float]:
    """A unit vector at exact cosine ``cos_target`` against the unit vector
    on ``axis_index``, confined to the (axis_index, axis_index + 1) plane —
    zero everywhere else, so vectors on disjoint axis pairs are orthogonal."""
    if not (-1.0 <= cos_target <= 1.0):
        raise ValueError(f"cos_target must be in [-1.0, 1.0], got {cos_target}")
    if embed_dim < axis_index + 2:
        raise ValueError(
            f"embed_dim must be >= axis_index + 2, got embed_dim={embed_dim}, "
            f"axis_index={axis_index}"
        )
    residual = math.sqrt(max(0.0, 1.0 - cos_target * cos_target))
    vec = [0.0] * embed_dim
    vec[axis_index] = cos_target
    vec[axis_index + 1] = residual
    return vec


def _reference_embedding(axis_index: int = 0, embed_dim: int = EMBED_DIM) -> list[float]:
    vec = [0.0] * embed_dim
    vec[axis_index] = 1.0
    return vec


@dataclass(frozen=True)
class DedupEvalPair:
    """One labelled (anchor, candidate) pair with known exact cosine."""

    name: str
    cosine: float
    verdict: str  # "duplicate" or "distinct"
    anchor_surface: str
    candidate_surface: str
    axis_index: int
    strictly_stronger_than_95: bool = False
    band: str = ""
    notes: str = ""

    @property
    def anchor_embedding(self) -> list[float]:
        return _reference_embedding(self.axis_index)

    @property
    def candidate_embedding(self) -> list[float]:
        return _make_embedding_at_cosine(self.cosine, axis_index=self.axis_index)


# Cosine ladder straddling both the write-time near_dup_threshold (0.92) and
# the old capture-time DEDUP_COS_THRESHOLD (0.95). Covers >= 7 distinct
# cosine values as required by the plan's acceptance criteria.
DEDUP_EVAL_PAIRS: list[DedupEvalPair] = [
    DedupEvalPair(
        name="verbatim_identical_high_cos",
        cosine=0.99,
        verdict="duplicate",
        anchor_surface="alice prefers tea over coffee",
        candidate_surface="alice prefers tea over coffee",
        axis_index=0,
        band="above_0.95",
        notes="verbatim-identical text at near-1.0 cosine; the easy case both old and new gates catch",
    ),
    DedupEvalPair(
        name="whitespace_case_variant",
        cosine=0.97,
        verdict="duplicate",
        anchor_surface="Alice prefers tea over coffee.",
        candidate_surface="alice   prefers tea over coffee",
        axis_index=2,
        band="above_0.95",
        notes="whitespace/case variant; caught by both the old 0.95 capture threshold and the 0.92 write-time gate",
    ),
    DedupEvalPair(
        name="strengthened_band_0.96",
        cosine=0.96,
        verdict="duplicate",
        anchor_surface="the daemon restarts nightly at 3am for consolidation",
        candidate_surface="the daemon restarts nightly at 3am for consolidation, roughly",
        axis_index=4,
        band="above_0.95",
        notes="above old 0.95 threshold; regression guard that strengthening never weakens the easy case",
    ),
    DedupEvalPair(
        name="strengthened_band_0.93",
        cosine=0.93,
        verdict="duplicate",
        anchor_surface="the recall path never touches the working tier",
        candidate_surface="the recall path never touches the working tier at all",
        axis_index=6,
        band="near_dup_to_0.95",
        strictly_stronger_than_95=True,
        notes="in the (near_dup_threshold=0.92, 0.95) band: the old 0.95 capture threshold would have "
              "inserted this as a new row; the strengthened write-time gate must SKIP-merge it — "
              "the evidence that the gate is now strictly stronger than the old 0.95 threshold",
    ),
    DedupEvalPair(
        name="threshold_boundary_0.90",
        cosine=0.90,
        verdict="distinct",
        anchor_surface="the hnswlib index is approximate by construction",
        candidate_surface="quantum chromodynamics describes the strong nuclear force",
        axis_index=8,
        band="below_near_dup",
        notes="below near_dup_threshold (0.92); must remain a distinct row — zero false-merge",
    ),
    DedupEvalPair(
        name="genuinely_distinct_0.85",
        cosine=0.85,
        verdict="distinct",
        anchor_surface="the sensory tier formalizes the existing .live.jsonl mechanism",
        candidate_surface="bge-small-en-v1.5 produces 384-dimensional embeddings",
        axis_index=10,
        band="below_near_dup",
        notes="clearly below threshold; zero-false-merge regression guard",
    ),
    DedupEvalPair(
        name="genuinely_distinct_0.70",
        cosine=0.70,
        verdict="distinct",
        anchor_surface="the write-time gate is the universal choke point",
        candidate_surface="autistic cognition treats prediction errors as high-precision signal",
        axis_index=12,
        band="below_near_dup",
        notes="moderate cosine, still below near_dup_threshold; zero-false-merge regression guard",
    ),
    DedupEvalPair(
        name="unrelated_topic_0.40",
        cosine=0.40,
        verdict="distinct",
        anchor_surface="the exact-cosine authority is a resident float32 matrix",
        candidate_surface="never touch time machine on this machine",
        axis_index=14,
        band="unrelated",
        notes="genuinely-different-topic text; the clean-negative control",
    ),
]

_USED_AXES = [p.axis_index for p in DEDUP_EVAL_PAIRS]
assert len(_USED_AXES) == len(set(_USED_AXES)), (
    "every pair must use a distinct axis_index so pairs never cross-contaminate "
    "cosine similarity when inserted sequentially into the same store"
)
assert max(_USED_AXES) + 2 <= EMBED_DIM, (
    f"EMBED_DIM={EMBED_DIM} too small for the highest axis_index used "
    f"({max(_USED_AXES)}); each pair needs 2 dedicated dimensions"
)


DUPLICATE_PAIRS: list[DedupEvalPair] = [
    p for p in DEDUP_EVAL_PAIRS if p.verdict == "duplicate"
]
DISTINCT_PAIRS: list[DedupEvalPair] = [
    p for p in DEDUP_EVAL_PAIRS if p.verdict == "distinct"
]
STRICTLY_STRONGER_THAN_95_PAIRS: list[DedupEvalPair] = [
    p for p in DEDUP_EVAL_PAIRS if p.strictly_stronger_than_95
]

assert len(DEDUP_EVAL_PAIRS) >= 7, "eval set must cover >= 7 distinct cosine values"
assert STRICTLY_STRONGER_THAN_95_PAIRS, (
    "eval set must include at least one (near_dup_threshold, 0.95)-band pair "
    "to prove the gate is strictly stronger than the old 0.95 threshold"
)
