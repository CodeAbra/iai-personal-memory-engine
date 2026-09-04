"""Real-corpus calibration of MULTIPLICITY_CAP: eligible-bind-term rate by
cluster size, a cap sweep, and the per-cluster entity-sharing distribution.

Reconstructs already-minted clusters from the real store's ``consolidated_from``
edges (the exact membership the sleep pipeline itself minted against) via an
isolated read-only copy (``bench.recall_accuracy_real.open_eval_copy_store`),
never the live store directly.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from uuid import UUID

_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
_REPO_PATH = str(Path(__file__).resolve().parent.parent)
if _REPO_PATH not in sys.path:
    sys.path.insert(0, _REPO_PATH)

from iai_mcp.lilli.crossmodal.embed_to_hv import from_embedding_fhrr  # noqa: E402
from iai_mcp.lilli.tiers import fhrr as fhrr_ops  # noqa: E402
from iai_mcp.sleep import _existing_summary_members  # noqa: E402
from iai_mcp.store import EDGES_TABLE  # noqa: E402
from iai_mcp.types import MemoryRecord  # noqa: E402

LARGE_CLUSTER_THRESHOLD: int = 5
"""Clusters at or below this member count are the "small" bucket; above it,
the "large" bucket -- the tail-member value case this consumer targets."""

SMALL_SAMPLE_WARNING_N: int = 20
"""Below this many large clusters, the large-bucket eligible-bind-term rate
is too small a sample to inform a cap decision."""

#: Per-cluster cap on how many members may share an entity string before
#: that entity is excluded from the FHRR entity-bind term -- recovery
#: fidelity toward the shared-entity centroid rises with multiplicity, so
#: an uncapped bind re-collapses to a flat-cosine-computable signal.
MULTIPLICITY_CAP: int = 2

FHRR_ENTITY_BIND_TAG: str = "fhrr_entity_bind_v2"
"""Tag marking a structure_hv_payload as a multiplicity-capped entity
role-filler bind (not the flat re-projected embedding)."""


def _compose_fhrr_entity_bind_payload(
    cluster_recs: list[MemoryRecord],
    cap: int = MULTIPLICITY_CAP,
) -> tuple[bytes, list[str]]:
    """Bind each (member, entity) pair whose entity's cluster-local
    multiplicity is at or below ``cap`` and bundle the terms into one
    payload. Both bench scripts import this function rather than
    re-deriving the bind formula."""
    member_entity_pairs = [
        (rec, e)
        for rec in cluster_recs
        for e in (t.removeprefix("entity:") for t in rec.tags if t.startswith("entity:"))
    ]
    if not member_entity_pairs:
        return b"", []
    entity_counts = Counter(e for _, e in member_entity_pairs)
    bind_terms = [
        fhrr_ops.bind(fhrr_ops.role_hv(f"entity:{e}"), from_embedding_fhrr(rec.embedding))
        for rec, e in member_entity_pairs
        if entity_counts[e] <= cap
    ]
    if not bind_terms:
        return b"", []
    return fhrr_ops.bundle(bind_terms), [FHRR_ENTITY_BIND_TAG]


def real_clusters(store) -> list[list[MemoryRecord]]:
    """Reconstruct clusters already minted on the real corpus from
    ``consolidated_from`` edges -- real membership, not a fresh re-cluster."""
    edges_df = store.db.open_table(EDGES_TABLE).to_pandas()
    semantic_ids: set[UUID] = set()
    for row in store.iter_record_columns(["id", "tier"], batch_size=2048):
        if row.get("tier") == "semantic":
            try:
                semantic_ids.add(UUID(str(row["id"])))
            except (ValueError, TypeError):
                continue
    members_by_summary = _existing_summary_members(edges_df, semantic_ids)
    clusters: list[list[MemoryRecord]] = []
    for member_ids in members_by_summary.values():
        if not member_ids:
            continue
        batch = store.get_batch(list(member_ids))
        cluster_recs = list(batch.values())
        if cluster_recs:
            clusters.append(cluster_recs)
    return clusters


def _entity_multiplicity(cluster: list[MemoryRecord]) -> Counter:
    counts: Counter = Counter()
    for rec in cluster:
        for tag in rec.tags:
            if tag.startswith("entity:"):
                counts[tag.removeprefix("entity:")] += 1
    return counts


def cap_sweep(clusters: list[list[MemoryRecord]], caps: list[int]) -> dict:
    small = [c for c in clusters if len(c) <= LARGE_CLUSTER_THRESHOLD]
    large = [c for c in clusters if len(c) > LARGE_CLUSTER_THRESHOLD]

    sharing_counts: Counter = Counter()
    for cluster in clusters:
        for count in _entity_multiplicity(cluster).values():
            sharing_counts[count] += 1

    report: dict = {
        "cluster_count": len(clusters),
        "small_bucket_n": len(small),
        "large_bucket_n": len(large),
        "entity_sharing_distribution": dict(sorted(sharing_counts.items())),
        "caps": {},
    }
    if len(large) < SMALL_SAMPLE_WARNING_N:
        report["large_bucket_warning"] = (
            f"large-cluster (>{LARGE_CLUSTER_THRESHOLD}-member) bucket n={len(large)} "
            f"is below {SMALL_SAMPLE_WARNING_N} -- too small to be informative"
        )

    for cap in caps:
        eligible_small = sum(
            1 for c in small if _compose_fhrr_entity_bind_payload(c, cap=cap)[0]
        )
        eligible_large = sum(
            1 for c in large if _compose_fhrr_entity_bind_payload(c, cap=cap)[0]
        )
        eligible_all = eligible_small + eligible_large
        report["caps"][cap] = {
            "small_bucket_eligible_rate": (eligible_small / len(small)) if small else None,
            "large_bucket_eligible_rate": (eligible_large / len(large)) if large else None,
            "large_bucket_n": len(large),
            "overall_fallback_rate": (
                (len(clusters) - eligible_all) / len(clusters) if clusters else None
            ),
        }
    return report


def run(caps: list[int], driver: str | None) -> dict:
    from bench.recall_accuracy_real import open_eval_copy_store

    with open_eval_copy_store(driver=driver) as store:
        clusters = real_clusters(store)
        report = cap_sweep(clusters, caps)
        report["driver"] = driver or "stdlib"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate MULTIPLICITY_CAP against real-corpus cluster entity-sharing."
    )
    parser.add_argument("--caps", type=str, default="1,2,3")
    parser.add_argument("--driver", choices=["stdlib", "lilli"], default=None)
    args = parser.parse_args()
    parsed_caps = [int(c) for c in args.caps.split(",")]
    result = run(parsed_caps, args.driver)
    print(json.dumps(result, indent=2, default=str))
