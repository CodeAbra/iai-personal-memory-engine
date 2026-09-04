"""The Delta conditional Recall@K verdict metric engine.

Measures whether a measurement-only, throwaway entity-anchor ceiling
expansion (an exact ``entity -> member-id`` posting-list lookup over the
corpus, derived from the cue's own text alone) could rescue queries the
production dispatch path (``core.dispatch``) misses -- the ceiling a
future entity-anchor lane could recover, without building that lane.

``conditional_recall_delta`` is arm-agnostic: the same function computes
the correct-anchor arm's delta and the size-matched shuffled-anchor null
arm's delta, over paired per-query outcomes restricted to the
eligible-and-missed subpopulation (baseline pinned to the miss indicator
for every row). ``cluster_stratified_bootstrap_ci`` resamples CLUSTERS,
not rows, and returns a lower-95%-only bound; a two-sided interval is
derived by callers as ``[ci(signed), -ci(-signed)]`` without changing
this function's signature. ``verdict_predicate`` gates PASS on a positive
lower bound, a distinct-rescue floor, no regression, and a pre-registered
power floor on the eligible-and-missed denominator -- UNDERPOWERED is a
distinct outcome from a clean NON-PASS, checked before any other gate.

The entity-anchor ceiling expansion is measurement-only: it is never
imported by ``src/`` and never wired into production recall.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np

_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
_REPO_PATH = str(Path(__file__).resolve().parent.parent)
if _REPO_PATH not in sys.path:
    sys.path.insert(0, _REPO_PATH)

N_MIN: int = 15
"""Pre-registered floor on the eligible-and-missed denominator. Set below
the labelled fixture's initial 30-50 query size so a faithful baseline's
own low miss rate cannot force a structural false-UNDERPOWERED outcome;
large enough to support the >=10-distinct-rescue gate plus a
non-degenerate cluster-stratified bootstrap; reachable by relabelling a
larger held-out set given the corpus's large-cluster headroom well above
this floor."""

VERDICT_PASS = "PASS"
VERDICT_UNDERPOWERED = "UNDERPOWERED"
VERDICT_NON_PASS = "NON-PASS"

MIN_DISTINCT_RESCUES = 10

DEFAULT_CEILING_BOUND = 50
"""Size bound on one cue's ceiling expansion -- matches the production
rich-club top-K cap so an unbounded posting-list dump cannot manufacture
a rescue by pool size alone."""

_WORD_RE = re.compile(r"[a-z0-9._-]+")
_MIN_TOKEN_LEN = 3


class CeilingArmUnmeasurable(RuntimeError):
    """Raised when neither an entity-tag posting list nor a
    literal_surface posting list can be built over the corpus copy -- the
    ceiling arm cannot be measured on this corpus and needs owner
    escalation, not a silent empty expansion."""


# ---------------------------------------------------------------------------
# Pure statistics core -- no store dependency, importable and
# unit-testable on synthetic input.
# ---------------------------------------------------------------------------


def conditional_recall_delta(paired_outcomes: Any) -> float:
    """Recall of (baseline UNION arm) minus recall of baseline, as a
    paired within-query mean difference.

    ``paired_outcomes`` is a sequence of ``(baseline_served, arm_served)``
    pairs restricted to the eligible-and-missed subpopulation -- every
    row's ``baseline_served`` is the miss indicator (0/False). Under that
    restriction the paired difference reduces to ``mean(arm_served)``,
    identical to the boolean recall-of-union formulation; the arithmetic
    form generalizes cleanly to a signed per-query value (not only 0/1),
    which is what the two-sided A/A floor's ``ci_high = -ci(-signed)``
    derivation needs from this same function.

    Arm-agnostic: the identical code path computes the correct-anchor
    arm's delta and the shuffled-anchor null arm's delta -- callers pass
    a different ``arm_served`` column, never a different code path.
    """
    pairs = list(paired_outcomes)
    if not pairs:
        return 0.0
    baseline = np.fromiter((float(b) for b, _a in pairs), dtype=float, count=len(pairs))
    arm = np.fromiter((float(a) for _b, a in pairs), dtype=float, count=len(pairs))
    return float(np.mean(arm) - np.mean(baseline))


def cluster_stratified_bootstrap_ci(
    paired_outcomes: Any,
    cluster_ids: Any,
    iters: int = 2000,
    alpha: float = 0.05,
    *,
    seed: int = 0,
) -> float:
    """Lower ``1 - alpha`` one-sided bootstrap bound on
    ``conditional_recall_delta``, resampling CLUSTERS (not rows).

    Returns a LOWER bound only -- never a two-sided interval. A two-sided
    interval is derived by a caller as ``[ci_low, -ci(-signed_outcomes,
    ...)]`` without changing this function.
    """
    pairs = list(paired_outcomes)
    clusters = list(cluster_ids)
    if len(pairs) != len(clusters):
        raise ValueError(
            f"paired_outcomes length {len(pairs)} != cluster_ids length {len(clusters)}"
        )
    if not pairs:
        return 0.0

    by_cluster: dict[Any, list] = {}
    for pair, cid in zip(pairs, clusters):
        by_cluster.setdefault(cid, []).append(pair)
    cluster_keys = list(by_cluster.keys())
    n_clusters = len(cluster_keys)

    rng = np.random.default_rng(seed)
    boot_deltas = np.empty(iters, dtype=float)
    for i in range(iters):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        resampled: list = []
        for idx in chosen:
            resampled.extend(by_cluster[cluster_keys[idx]])
        boot_deltas[i] = conditional_recall_delta(resampled)

    return float(np.quantile(boot_deltas, alpha))


def verdict_predicate(
    delta: float,
    ci_low: float,
    distinct_rescues: int,
    regression_flags: Any,
    eligible_missed_n: int,
    n_min: int = N_MIN,
) -> str:
    """Three-way PASS / UNDERPOWERED / NON-PASS outcome.

    UNDERPOWERED is checked FIRST and is returned regardless of the
    point-delta or CI whenever the eligible-and-missed denominator falls
    below the pre-registered power floor -- distinct from a clean
    NON-PASS, which means "measured, and did not clear the bar."
    """
    if eligible_missed_n < n_min:
        return VERDICT_UNDERPOWERED

    if isinstance(regression_flags, dict):
        any_regression = any(regression_flags.values())
    else:
        any_regression = any(regression_flags)

    if ci_low > 0 and distinct_rescues >= MIN_DISTINCT_RESCUES and not any_regression:
        return VERDICT_PASS

    return VERDICT_NON_PASS


# ---------------------------------------------------------------------------
# Measurement-only entity-anchor ceiling expansion -- never imported by
# src/, never a shipped lane.
# ---------------------------------------------------------------------------


def _entity_posting_list(store: Any) -> "dict[str, set]":
    """Corpus-wide ``entity -> {member-id}`` posting list built from the
    plaintext ``tags_json`` column -- the SAME ``entity:X`` tags
    ``iai_mcp.entity_anchors.entity_tags`` writes at capture time. No
    decrypt needed: tags are stored in plaintext by design."""
    posting: "dict[str, set]" = {}
    for row in store.iter_record_columns(["id", "tags_json"], batch_size=2048):
        raw_tags = row.get("tags_json")
        if not raw_tags:
            continue
        try:
            tags = json.loads(raw_tags) if isinstance(raw_tags, str) else list(raw_tags)
        except (TypeError, ValueError):
            continue
        if not tags:
            continue
        try:
            rid = UUID(str(row["id"]))
        except (TypeError, ValueError):
            continue
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("entity:"):
                posting.setdefault(tag.removeprefix("entity:"), set()).add(rid)
    return posting


def _literal_surface_posting_list(store: Any) -> "dict[str, set]":
    """Fallback corpus-wide ``token -> {member-id}`` posting list over
    ``literal_surface`` -- used only when the corpus carries no
    ``entity:X`` tags at all. Decrypts via ``get_batch`` (the existing
    high-level decrypt path), never a hand-rolled row decrypt."""
    ids: "list[UUID]" = []
    for row in store.iter_record_columns(["id"], batch_size=2048):
        try:
            ids.append(UUID(str(row["id"])))
        except (TypeError, ValueError):
            continue
    if not ids:
        return {}

    posting: "dict[str, set]" = {}
    batch = store.get_batch(ids)
    for rid, rec in batch.items():
        surface = (getattr(rec, "literal_surface", "") or "").lower()
        for token in _WORD_RE.findall(surface):
            if len(token) >= _MIN_TOKEN_LEN:
                posting.setdefault(token, set()).add(rid)
    return posting


def _resolve_posting_list(store: Any) -> "tuple[dict[str, set], str]":
    """Prefer the entity-tag posting list; fall back to literal_surface;
    escalate (raise) rather than silently returning an empty expansion
    when neither is buildable over this corpus copy."""
    entity_posting = _entity_posting_list(store)
    if entity_posting:
        return entity_posting, "entity_tag"
    literal_posting = _literal_surface_posting_list(store)
    if literal_posting:
        return literal_posting, "literal_surface"
    raise CeilingArmUnmeasurable(
        "neither an entity-tag posting list nor a literal_surface posting "
        "list could be built over this corpus copy -- the measurement-only "
        "ceiling arm is unmeasurable on this store; escalate to the "
        "operator rather than reporting a silent empty expansion"
    )


def entity_anchor_ceiling_expansion(
    store: Any,
    cue_text: str,
    k: int = DEFAULT_CEILING_BOUND,
    *,
    anchors: "list[str] | None" = None,
    posting_list: "tuple[dict[str, set], str] | None" = None,
) -> set:
    """Size-bounded exact entity-anchor posting-list expansion over the
    corpus, derived from the CUE'S OWN TEXT ALONE.

    Never reads a gold record: the anchor set defaults to
    ``iai_mcp.entity_anchors.extract_entities(cue_text)`` (the identical
    extraction ``capture.py`` runs at capture time) and there is no
    parameter through which a gold record or its tags could reach this
    function -- a gold-peeking variant would trivially return the gold
    and is structurally impossible here, not merely disallowed by
    convention.
    """
    from iai_mcp.entity_anchors import extract_entities

    used_anchors = anchors if anchors is not None else extract_entities(cue_text)
    posting, _provenance = (
        posting_list if posting_list is not None else _resolve_posting_list(store)
    )

    ids: set = set()
    for anchor in used_anchors:
        ids |= posting.get(anchor, set())

    bounded = sorted(ids, key=str)[:k]
    return set(bounded)


def shuffled_anchor_expansion(
    store: Any,
    match_size: int,
    *,
    wrong_anchors: "list[str]",
    posting_list: "tuple[dict[str, set], str] | None" = None,
) -> set:
    """Size-matched null-arm expansion from WRONG/shuffled anchors -- the
    SAME id-count cap as the real expansion it is compared against, so a
    large pool cannot 'rescue' gold by sheer size alone."""
    posting, _provenance = (
        posting_list if posting_list is not None else _resolve_posting_list(store)
    )

    ids: set = set()
    for anchor in wrong_anchors:
        ids |= posting.get(anchor, set())

    bounded = sorted(ids, key=str)[:match_size]
    return set(bounded)


# ---------------------------------------------------------------------------
# spread_edge_missing signal: a real edge-adjacency check over the
# corpus, not a caller-supplied constant.
# ---------------------------------------------------------------------------


def build_edge_adjacency(store: Any) -> "dict[UUID, set]":
    """Corpus-wide, direction-less edge adjacency: ``id -> {neighbor
    ids}`` over every edge row, regardless of ``edge_type`` or which
    column (``src``/``dst``) carries which semantic end -- edge writes in
    this store canonicalize pair order, so both orientations must be
    checked."""
    from iai_mcp.store import EDGES_TABLE

    adjacency: "dict[UUID, set]" = {}
    edges_df = store.db.open_table(EDGES_TABLE).to_pandas()
    if edges_df.empty:
        return adjacency
    for _, row in edges_df.iterrows():
        try:
            a, b = UUID(str(row["src"])), UUID(str(row["dst"]))
        except (TypeError, ValueError):
            continue
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    return adjacency


def spread_edge_missing_signal(
    gold_id: Any, post_cap_ids: "set", adjacency: "dict[UUID, set]",
) -> bool:
    """True when the gold id has NO edge to anything in the post-cap
    pool -- the structurally-possible-2-hop-path check
    ``classify_miss_cause``'s ``spread_did_not_reach`` category needs.
    Computed from the real edge table, not left as an always-False
    constant: without this, that miss cause is unreachable from real
    measurement output."""
    try:
        gold_uuid = gold_id if isinstance(gold_id, UUID) else UUID(str(gold_id))
    except (TypeError, ValueError):
        return True
    neighbors = adjacency.get(gold_uuid, set())
    pool = {p if isinstance(p, UUID) else UUID(str(p)) for p in post_cap_ids}
    return not bool(neighbors & pool)


FIXTURE_REQUIRED_KEYS = ("cue_id", "cue_text", "gold_record_id", "cluster_id")
"""The holdout tail-gold fixture's own key contract
(``scripts/label_holdout_tail_cues.py``'s ``_write_fixture`` output) --
NOT the ``relevant_record_ids`` shape ``bench.recall_accuracy_real``'s
real-thread fixture uses; the two fixture families are validated
separately because they carry different required keys."""

_TOKEN_BUDGET_PER_QUERY = 2000
"""Matches ``bench.recall_accuracy_real._BUDGET_TOKENS`` -- the per-query
budget the production dispatch path packs candidates against."""

_TOKEN_REGRESSION_RATIO = 0.5
"""Flag a token regression when the mean incremental token cost of one
missed cue's ceiling-expansion candidates exceeds this fraction of the
per-query budget -- a large expansion would itself threaten the budget a
future entity-anchor lane would have to respect."""

_EXPANSION_LATENCY_BUDGET_MS = 200.0
"""Pre-registered per-cue wall-clock budget for computing the ceiling
expansion; the operator run can override this via ``run_verdict``'s
``latency_budget_ms`` parameter."""


def _validate_verdict_fixture_dict(raw: dict) -> list[dict]:
    if "cues" not in raw or not isinstance(raw["cues"], list):
        raise ValueError("fixture missing 'cues' array")
    cues = raw["cues"]
    for entry in cues:
        for required_key in FIXTURE_REQUIRED_KEYS:
            if required_key not in entry:
                raise ValueError(f"fixture cue missing required key {required_key!r}: {entry}")
    return cues


def _rank_of(gold_id: str, returned_ids: "list[str]", k: int) -> "int | None":
    window = returned_ids[:k]
    try:
        return window.index(gold_id)
    except ValueError:
        return None


def _ndcg_at_rank(rank: "int | None") -> float:
    """Binary-relevance nDCG for a single relevant document: IDCG is 1
    (the ideal ranking places it first), so this already IS the
    normalized score."""
    if rank is None:
        return 0.0
    return 1.0 / math.log2(rank + 2)


def _estimate_tokens(literal_surface: str) -> int:
    """The SAME char/4 heuristic the production budget-packer uses
    (``core/__init__.py``), reused rather than a bench-side reinvention."""
    return max(1, len(literal_surface or "") // 4)


def _baseline_precision_at_k(baseline_rows: "list[dict]", k: int) -> float:
    """Precision@k with exactly one relevant id per cue: 1/k when served,
    0 when missed -- NOT equal to recall@k except at k=1."""
    if not baseline_rows:
        return 0.0
    return sum(1.0 / k for r in baseline_rows if r["served"]) / len(baseline_rows)


def _posting_list_provenance(posting_list: "tuple[dict[str, set], str]") -> str:
    """The provenance half of ``_resolve_posting_list``'s return -- a
    corpus-level constant for the whole run, never a per-query
    measurement."""
    return posting_list[1]


def _select_null_arm_cue_id(
    row_index: int,
    row_cluster_id: Any,
    missed_cue_ids: "list[str]",
    cluster_id_by_cue_id: "dict[str, Any]",
) -> "str | None":
    """Pick a DIFFERENT-cluster missed cue's id for the shuffled-anchor
    null arm. The fixture is cluster-stratified, so a same-cluster draw
    can share entity anchors with the real cue and inflate the null.
    Returns None when every other missed cue shares this row's cluster
    (including the single-missed-cue case) -- the null arm is
    unmeasurable for this row, never a self-referential draw."""
    candidates = [
        cid for cid in missed_cue_ids
        if cid != missed_cue_ids[row_index] and cluster_id_by_cue_id[cid] != row_cluster_id
    ]
    if not candidates:
        return None
    return candidates[row_index % len(candidates)]


# ---------------------------------------------------------------------------
# Operator-run wrapper -- NOT a pytest target (needs the LOCAL labelled
# fixture); produces the checkpoint evidence for the verdict fork.
# ---------------------------------------------------------------------------


def run_verdict(
    fixture_path: "str | Path",
    *,
    driver: "str | None" = None,
    k: int = 10,
    ceiling_k: int = DEFAULT_CEILING_BOUND,
    n_min: int = N_MIN,
    iters: int = 2000,
    seed: int = 0,
    token_budget_per_query: int = _TOKEN_BUDGET_PER_QUERY,
    token_regression_ratio: float = _TOKEN_REGRESSION_RATIO,
    latency_budget_ms: float = _EXPANSION_LATENCY_BUDGET_MS,
) -> dict:
    """Drive a single warmed baseline pass over the LOCAL labelled
    fixture, compute the ceiling-expansion outcome for every
    baseline-MISSED eligible cue plus a shuffled-anchor null-arm outcome
    wherever a different-cluster partner cue exists, and report the
    verdict metric plus unconditional Recall@K/nDCG, a baseline
    precision@k, a token/latency regression veto, the per-miss cause
    distribution, and posting-list provenance -- aggregate numbers only,
    never raw record content.

    Precision is reported as the baseline's OWN precision@k (an
    informational aggregate): the ceiling expansion is never merged into
    an actual response, so there is no second, post-merge response to
    compare a precision delta against -- inventing one would be a merge
    policy this measurement-only arm does not get to define. Token and
    latency ARE genuine within-run comparators (incremental candidate
    cost against the query's own budget; wall-clock cost of computing
    the expansion) and do gate the verdict.
    """
    from bench.fhrr_verdict_membership import capture_discovery_membership, classify_miss_cause
    from bench.recall_accuracy_real import (
        _SESSION_ID_HARNESS,
        _dispatch_real_cue,
        assert_graphcache_generation_parity,
        open_eval_copy_store,
        warm_eval_copy_store,
    )
    from iai_mcp.entity_anchors import extract_entities

    path = Path(fixture_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"labelled fixture not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    cues = _validate_verdict_fixture_dict(raw)
    cue_text_by_id = {c["cue_id"]: c["cue_text"] for c in cues}

    baseline_rows: list[dict] = []

    with open_eval_copy_store(driver=driver) as store:
        warm_eval_copy_store(store)
        assert_graphcache_generation_parity(store)

        for cue in cues:
            returned_ids = _dispatch_real_cue(store, cue["cue_text"], 0.0)
            gold_id = str(cue["gold_record_id"])
            rank = _rank_of(gold_id, returned_ids, k)
            baseline_rows.append({
                "cue_id": cue["cue_id"],
                "cluster_id": cue["cluster_id"],
                "gold_id": gold_id,
                "served": rank is not None,
                "rank": rank,
            })

        missed_rows = [r for r in baseline_rows if not r["served"]]

        posting_list = _resolve_posting_list(store)
        adjacency = build_edge_adjacency(store)

        pairs: list[tuple[int, int]] = []
        null_pairs: list[tuple[int, int]] = []
        cluster_ids: list[Any] = []
        cause_counts: "dict[str, int]" = {}
        distinct_rescue_ids: set = set()
        expansion_latencies_ms: list[float] = []
        expansion_token_costs: list[int] = []

        missed_cue_ids = [r["cue_id"] for r in missed_rows]
        cluster_id_by_cue_id = {r["cue_id"]: r["cluster_id"] for r in missed_rows}
        for i, row in enumerate(missed_rows):
            cue_text = cue_text_by_id[row["cue_id"]]
            gold_uuid = UUID(row["gold_id"])

            start = time.perf_counter()
            expansion = entity_anchor_ceiling_expansion(
                store, cue_text, ceiling_k, posting_list=posting_list,
            )
            expansion_latencies_ms.append((time.perf_counter() - start) * 1000.0)

            if expansion:
                batch = store.get_batch(list(expansion))
                expansion_token_costs.append(
                    sum(_estimate_tokens(rec.literal_surface) for rec in batch.values())
                )
            else:
                expansion_token_costs.append(0)

            rescued = gold_uuid in expansion
            pairs.append((0, 1 if rescued else 0))
            cluster_ids.append(row["cluster_id"])
            if rescued:
                distinct_rescue_ids.add(gold_uuid)

            other_cue_id = _select_null_arm_cue_id(
                i, row["cluster_id"], missed_cue_ids, cluster_id_by_cue_id,
            )
            if other_cue_id is not None:
                wrong_anchors = extract_entities(cue_text_by_id[other_cue_id])
                null_expansion = shuffled_anchor_expansion(
                    store, len(expansion), wrong_anchors=wrong_anchors, posting_list=posting_list,
                )
                null_pairs.append((0, 1 if gold_uuid in null_expansion else 0))

            membership = capture_discovery_membership(store, {
                "cue": cue_text,
                "session_id": _SESSION_ID_HARNESS,
                "budget_tokens": token_budget_per_query,
            })
            missing_signal = spread_edge_missing_signal(gold_uuid, membership.post_cap_ids, adjacency)
            gm = membership.gold_membership(gold_uuid, spread_edge_missing=missing_signal)
            cause = classify_miss_cause(gm, {"in_budget_topk": False}, {})
            cause_counts[cause] = cause_counts.get(cause, 0) + 1

    delta = conditional_recall_delta(pairs)
    ci_low = cluster_stratified_bootstrap_ci(pairs, cluster_ids, iters=iters, seed=seed)
    null_delta = conditional_recall_delta(null_pairs) if null_pairs else None
    eligible_missed_n = len(pairs)

    unconditional_recall_at_k = (
        sum(1 for r in baseline_rows if r["served"]) / len(baseline_rows) if baseline_rows else 0.0
    )
    ndcg_at_k = (
        sum(_ndcg_at_rank(r["rank"]) for r in baseline_rows) / len(baseline_rows) if baseline_rows else 0.0
    )
    baseline_precision_at_10 = _baseline_precision_at_k(baseline_rows, k)

    mean_expansion_latency_ms = (
        sum(expansion_latencies_ms) / len(expansion_latencies_ms) if expansion_latencies_ms else 0.0
    )
    mean_expansion_token_cost = (
        sum(expansion_token_costs) / len(expansion_token_costs) if expansion_token_costs else 0.0
    )
    regression_flags = {
        "token": mean_expansion_token_cost > token_regression_ratio * token_budget_per_query,
        "latency": mean_expansion_latency_ms > latency_budget_ms,
    }

    verdict = verdict_predicate(
        delta, ci_low, len(distinct_rescue_ids), regression_flags, eligible_missed_n, n_min,
    )

    result = {
        "cue_count": len(cues),
        "eligible_missed_n": eligible_missed_n,
        "n_min": n_min,
        "delta": delta,
        "ci_low": ci_low,
        "distinct_rescues": len(distinct_rescue_ids),
        "verdict": verdict,
        "null_arm_delta": null_delta,
        "null_arm_n": len(null_pairs),
        "unconditional_recall_at_k": unconditional_recall_at_k,
        "ndcg_at_k": ndcg_at_k,
        "baseline_precision_at_10": baseline_precision_at_10,
        "mean_expansion_latency_ms": mean_expansion_latency_ms,
        "mean_expansion_token_cost": mean_expansion_token_cost,
        "regression_flags": regression_flags,
        "cause_distribution": cause_counts,
        "posting_list_provenance": _posting_list_provenance(posting_list),
    }
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure the Delta conditional Recall@K verdict metric on the LOCAL labelled fixture."
    )
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--driver", choices=["stdlib", "lilli"], default="stdlib")
    parser.add_argument("--n-min", type=int, default=N_MIN)
    args = parser.parse_args()
    run_verdict(args.fixture, driver=args.driver, n_min=args.n_min)
