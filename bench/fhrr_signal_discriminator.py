"""Decorrelation acceptance gate for the FHRR entity-bind mint design.

Three states: real orthogonal structure (low correlation against every flat
baseline, real correct/wrong separation); monotone re-encoding (correlates
with a flat baseline -- the design collapsed to something cosine already
computes); noise collapse (decorrelated but carries no discriminative
signal). Imports the shared compose helper -- never re-derives the bind
formula. The noise floor is measured at runtime, never a literal constant.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy.stats import spearmanr

_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
_REPO_PATH = str(Path(__file__).resolve().parent.parent)
if _REPO_PATH not in sys.path:
    sys.path.insert(0, _REPO_PATH)

from bench.fhrr_multiplicity_calibration import _compose_fhrr_entity_bind_payload  # noqa: E402
from iai_mcp.entity_anchors import MAX_ENTITIES  # noqa: E402
from iai_mcp.lilli.core.projection import EMBED_DIM  # noqa: E402
from iai_mcp.lilli.crossmodal.embed_to_hv import from_embedding_fhrr  # noqa: E402
from iai_mcp.lilli.tiers import fhrr as fhrr_ops  # noqa: E402
from iai_mcp.types import MemoryRecord  # noqa: E402

RHO_GATE: float = 0.3
COHEN_D_GATE: float = 2.0


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0.0 else v


def _random_unit_vec(rng: np.random.Generator, dim: int = EMBED_DIM) -> np.ndarray:
    v = rng.standard_normal(dim)
    return _normalize(v)


def _measure_noise_floor(rng: np.random.Generator, n_pairs: int = 300) -> float:
    """SD of fhrr.similarity on unrelated random D=10000 HVs -- the finite-D
    Monte-Carlo noise floor, measured fresh every run."""
    sims = []
    for _ in range(n_pairs):
        seed_a = int(rng.integers(0, 2**31 - 1))
        seed_b = int(rng.integers(0, 2**31 - 1))
        a = fhrr_ops.random_hv(seed_a)
        b = fhrr_ops.random_hv(seed_b)
        sims.append(fhrr_ops.similarity(a, b))
    return float(np.std(sims))


def _make_synthetic_record(vec: list[float], entities: list[str]) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="synthetic member",
        aaak_index="",
        embedding=vec,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=True,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[f"entity:{e}" for e in entities],
        language="en",
    )


def _build_synthetic_clusters(
    rng: np.random.Generator, n_clusters: int, cluster_size: int,
) -> list[list[MemoryRecord]]:
    """Each member gets ONE entity, cluster-uniquely namespaced -- no entity
    string is ever reused across two different synthetic clusters, which
    forecloses the cross-cluster distractor collision A4 flagged."""
    clusters: list[list[MemoryRecord]] = []
    for ci in range(n_clusters):
        members = []
        for mi in range(cluster_size):
            vec = _random_unit_vec(rng)
            entity = f"c{ci}_e{mi}"
            members.append(_make_synthetic_record(vec.tolist(), [entity]))
        clusters.append(members)
    return clusters


def _member_entity_pairs(cluster_recs: list[MemoryRecord]) -> list[tuple[int, str]]:
    pairs: list[tuple[int, str]] = []
    for i, rec in enumerate(cluster_recs):
        for tag in rec.tags:
            if tag.startswith("entity:"):
                pairs.append((i, tag.removeprefix("entity:")))
    return pairs


def _max_agg_score(payload: bytes, cue_entities: list[str], cue_hv: bytes) -> float:
    best = 0.0
    for e in cue_entities:
        role = fhrr_ops.role_hv(f"entity:{e}")
        retrieved = fhrr_ops.unbind(payload, role)
        best = max(best, fhrr_ops.similarity(retrieved, cue_hv))
    return best


def measure(
    clusters: list[list[MemoryRecord]],
    *,
    rng: np.random.Generator,
    cues_per_cluster: int,
    n_distractors: int,
    max_entities: int = MAX_ENTITIES,
) -> dict:
    """Score every (cluster, cue) pair: the consumer's max-aggregated
    correct-entity score, the max-aggregated wrong-entity-only score, and
    the three flat baselines (summary/first-5 proxy, full centroid,
    per-entity sharing-group centroid)."""
    per_cluster: list[dict] = []
    global_entity_pool: list[str] = []
    for cluster_recs in clusters:
        global_entity_pool.extend(e for _, e in _member_entity_pairs(cluster_recs))

    consumer_scores: list[float] = []
    wrong_only_scores: list[float] = []
    baseline_full: list[float] = []
    baseline_first5: list[float] = []
    baseline_entity: list[float] = []

    for cluster_recs in clusters:
        payload, tags = _compose_fhrr_entity_bind_payload(cluster_recs)
        pairs = _member_entity_pairs(cluster_recs)
        if not payload or not pairs:
            per_cluster.append({"eligible": False, "size": len(cluster_recs)})
            continue

        vecs = np.array([r.embedding for r in cluster_recs], dtype=np.float64)
        full_centroid = _normalize(vecs.mean(axis=0))
        head = min(5, len(vecs))
        first5_centroid = _normalize(vecs[:head].mean(axis=0))

        entity_counts = Counter(e for _, e in pairs)
        entity_to_members: dict[str, list[int]] = {}
        for idx, e in pairs:
            entity_to_members.setdefault(e, []).append(idx)
        own_entities = set(entity_counts)
        distractor_pool = [e for e in global_entity_pool if e not in own_entities]

        n_cues = min(cues_per_cluster, len(pairs)) if cues_per_cluster else 0
        for _ in range(n_cues):
            member_idx, target_entity = pairs[int(rng.integers(0, len(pairs)))]
            target_vec = vecs[member_idx]
            noise = _random_unit_vec(rng)
            cue_vec = _normalize(0.85 * target_vec + 0.15 * noise)
            cue_hv = from_embedding_fhrr(cue_vec.tolist())

            distractors: list[str] = []
            if distractor_pool:
                idx = rng.choice(
                    len(distractor_pool),
                    size=min(n_distractors, len(distractor_pool)),
                    replace=False,
                )
                distractors = [distractor_pool[i] for i in idx]

            cue_entities = ([target_entity] + distractors)[:max_entities]
            consumer_scores.append(_max_agg_score(payload, cue_entities, cue_hv))
            if distractors:
                wrong_only_scores.append(
                    _max_agg_score(payload, distractors[:max_entities], cue_hv)
                )

            group_members = entity_to_members[target_entity]
            entity_centroid = _normalize(vecs[group_members].mean(axis=0))

            baseline_full.append(float(np.dot(cue_vec, full_centroid)))
            baseline_first5.append(float(np.dot(cue_vec, first5_centroid)))
            baseline_entity.append(float(np.dot(cue_vec, entity_centroid)))

        per_cluster.append({"eligible": True, "size": len(cluster_recs), "cues": n_cues})

    noise_floor_sd = _measure_noise_floor(rng)

    def _rho(baseline: list[float]) -> float:
        if len(baseline) < 2 or len(set(consumer_scores)) < 2:
            return 0.0
        r, _p = spearmanr(consumer_scores, baseline)
        return float(r) if r == r else 0.0  # r==r filters NaN

    rho_full = _rho(baseline_full)
    rho_first5 = _rho(baseline_first5)
    rho_entity = _rho(baseline_entity)

    if consumer_scores and wrong_only_scores:
        mean_correct = float(np.mean(consumer_scores))
        mean_wrong = float(np.mean(wrong_only_scores))
        pooled_sd = float(
            np.sqrt((np.var(consumer_scores) + np.var(wrong_only_scores)) / 2.0)
        )
        cohend = (mean_correct - mean_wrong) / pooled_sd if pooled_sd > 0.0 else 0.0
    else:
        mean_correct = float(np.mean(consumer_scores)) if consumer_scores else 0.0
        mean_wrong = float(np.mean(wrong_only_scores)) if wrong_only_scores else 0.0
        cohend = 0.0

    # The spread that must clear the noise floor is the correct/wrong
    # SEPARATION, not the correct-score population's own variance -- a tight
    # but well-separated correct distribution is real signal, not collapse.
    separation = mean_correct - mean_wrong
    spread = float(np.std(consumer_scores)) if consumer_scores else 0.0

    monotone = any(abs(r) >= RHO_GATE for r in (rho_full, rho_first5, rho_entity))
    if monotone:
        state = "monotone_re-encoding"
    elif cohend < COHEN_D_GATE or separation <= noise_floor_sd:
        state = "noise_collapse"
    else:
        state = "real_structure"

    return {
        "pairs_scored": len(consumer_scores),
        "clusters_eligible": sum(1 for c in per_cluster if c.get("eligible")),
        "clusters_total": len(clusters),
        "rho_vs_summary_first5_centroid": rho_first5,
        "rho_vs_full_centroid": rho_full,
        "rho_vs_per_entity_centroid": rho_entity,
        "cohens_d_max_aggregation": cohend,
        "mean_correct_score": mean_correct,
        "mean_wrong_only_score": mean_wrong,
        "correct_minus_wrong_separation": separation,
        "consumer_score_spread_sd": spread,
        "noise_floor_sd": noise_floor_sd,
        "gate_state": state,
    }


def run_synthetic_sanity(
    *, n_clusters: int, cluster_size: int, cues_per_cluster: int,
    n_distractors: int, seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    clusters = _build_synthetic_clusters(rng, n_clusters, cluster_size)
    return measure(
        clusters, rng=rng, cues_per_cluster=cues_per_cluster, n_distractors=n_distractors,
    )


def run_real_corpus(
    *, cues_per_cluster: int, n_distractors: int, seed: int, driver: str | None,
) -> dict:
    from bench.fhrr_multiplicity_calibration import real_clusters
    from bench.recall_accuracy_real import open_eval_copy_store

    rng = np.random.default_rng(seed)
    with open_eval_copy_store(driver=driver) as store:
        clusters = real_clusters(store)
        result = measure(
            clusters, rng=rng, cues_per_cluster=cues_per_cluster, n_distractors=n_distractors,
        )
    result["driver"] = driver or "stdlib"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decorrelation acceptance gate for the FHRR entity-bind mint design."
    )
    parser.add_argument("--synthetic-sanity", action="store_true")
    parser.add_argument("--real-corpus", action="store_true")
    parser.add_argument("--driver", choices=["stdlib", "lilli"], default=None)
    parser.add_argument("--n-clusters", type=int, default=30)
    parser.add_argument("--cluster-size", type=int, default=20)
    parser.add_argument("--cues-per-cluster", type=int, default=5)
    parser.add_argument("--distractors", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.synthetic_sanity and not args.real_corpus:
        parser.error("one of --synthetic-sanity or --real-corpus is required")

    if args.synthetic_sanity:
        report = run_synthetic_sanity(
            n_clusters=args.n_clusters,
            cluster_size=args.cluster_size,
            cues_per_cluster=args.cues_per_cluster,
            n_distractors=args.distractors,
            seed=args.seed,
        )
        print(json.dumps(report, indent=2, default=str))
        low_multiplicity_reproduced = (
            abs(report["rho_vs_full_centroid"]) < 0.1
            and abs(report["rho_vs_summary_first5_centroid"]) < 0.1
        )
        if not low_multiplicity_reproduced:
            print(
                "FAIL: synthetic sanity did not reproduce the research's "
                "low-multiplicity regime (|rho| < 0.1 vs full/first-5 centroid)",
                file=sys.stderr,
            )
            sys.exit(1)
        print("PASS: synthetic sanity reproduces the low-multiplicity regime")

    if args.real_corpus:
        report = run_real_corpus(
            cues_per_cluster=args.cues_per_cluster,
            n_distractors=args.distractors,
            seed=args.seed,
            driver=args.driver,
        )
        print(json.dumps(report, indent=2, default=str))
