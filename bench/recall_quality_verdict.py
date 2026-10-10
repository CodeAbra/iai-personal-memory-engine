"""Dev/holdout-aware recall-quality baseline runner.

Dispatches every cue in a ``recall_quality_fixture`` corpus through the
real production entry point (``core.dispatch``, read-only), classifies
every miss via the reused four-cause taxonomy
(``bench.fhrr_verdict_membership``), and reports recall/nDCG plus the
per-cause distribution. The statistics core
(``bench.fhrr_verdict_recall_delta``) is reused unmodified.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
_REPO_PATH = str(Path(__file__).resolve().parent.parent)
if _REPO_PATH not in sys.path:
    sys.path.insert(0, _REPO_PATH)

from bench.recall_accuracy import _dispatch_recall_ids, _inject_embedder_into_dispatch  # noqa: E402
from bench.fhrr_verdict_membership import (  # noqa: E402
    ALL_CAUSES,
    capture_discovery_membership,
    classify_miss_cause,
)
from bench.fhrr_verdict_recall_delta import (  # noqa: E402
    N_MIN,
    build_edge_adjacency,
    cluster_stratified_bootstrap_ci,
    conditional_recall_delta,
    spread_edge_missing_signal,
    verdict_predicate,
)

_SMOKE_WALL_CLOCK_BUDGET_SEC = 180.0
"""Target wall-clock ceiling for one full baseline pass in CI -- the bar
the smoke run's recommended full N is sized against. This is a bench
(opt-in) pipeline, never the default fast lane, so the budget is a few
minutes rather than seconds."""

_RESPONSE_K = 10
"""Final response top-K checked for the served/rank and
in_budget_topk determination -- independent of the ANN candidate
window (``K_CANDIDATES``)."""


def _warm_synthetic_store(store: Any, embedder: Any) -> None:
    """Force both in-RAM recall indexes warm on a freshly built synthetic
    store -- a cold index degrades to empty rather than rebuilding on the
    recall path, so an unwarmed store would manufacture misses production
    never serves."""
    store.lexical_search("warm", k=1)
    warm_vec = embedder.embed("warm")
    store.exact_top_k(list(warm_vec), k=1, build_if_cold=True)


def _compute_lexical_context(store: Any, cue_text: str, gold_id: UUID) -> dict:
    """Derive ``lexical_only_rescue``/``lexical_contributed`` from OBSERVED
    dispatch-adjacent state -- never a fixture-declared intent field.

    ``lexical_only_rescue`` mirrors the exact production gate
    (``pipeline._cue_identifier_grade``): the lexical lane only ever fires
    for an identifier-shaped cue, so a miss's lexical rescue path is only
    "structurally possible" when the cue itself clears that gate.
    ``lexical_contributed`` is a direct, real store query: whether the
    gold id appears in the same warm-BM25 lookup production's fusion lane
    calls, at the same k/min_idf production uses.
    """
    from iai_mcp.pipeline import LEX_FUSION_K, LEX_FUSION_MIN_IDF, _cue_identifier_grade

    lexical_only_rescue = bool(_cue_identifier_grade(cue_text))
    hits = store.lexical_query_warm(cue_text, k=LEX_FUSION_K, min_idf=LEX_FUSION_MIN_IDF)
    served_ids = set()
    for rid, _score in hits:
        try:
            served_ids.add(UUID(str(rid)))
        except (TypeError, ValueError):
            continue
    lexical_contributed = gold_id in served_ids
    return {
        "lexical_only_rescue": lexical_only_rescue,
        "lexical_contributed": lexical_contributed,
    }


def _rank_of(gold_id: str, returned_ids: "list[UUID]", k: int) -> "int | None":
    window = [str(rid) for rid in returned_ids[:k]]
    try:
        return window.index(gold_id)
    except ValueError:
        return None


def _ndcg_at_rank(rank: "int | None") -> float:
    if rank is None:
        return 0.0
    return 1.0 / math.log2(rank + 2)


def run_baseline_pass(
    store: Any,
    shim: Any,
    cues: "list[dict]",
    *,
    k: int = _RESPONSE_K,
    session_id: str = "recall_quality_harness",
) -> "list[dict]":
    """Dispatch every cue once through the production path and classify
    every miss. Returns one row per cue: served/rank plus, for a miss,
    the classified cause."""
    adjacency = build_edge_adjacency(store)
    rows: "list[dict]" = []
    with _inject_embedder_into_dispatch(shim):
        for cue in cues:
            cue_text = cue["cue_text"]
            gold_id = str(cue["gold_record_id"])
            returned = _dispatch_recall_ids(store, cue_text, session_id)
            rank = _rank_of(gold_id, returned, k)
            row: "dict[str, Any]" = {
                "cue_id": cue["cue_id"],
                "cluster_id": cue["cluster_id"],
                "gold_id": gold_id,
                "served": rank is not None,
                "rank": rank,
            }
            if rank is None:
                gold_uuid = UUID(gold_id)
                membership = capture_discovery_membership(store, {
                    "cue": cue_text,
                    "session_id": session_id,
                })
                missing_signal = spread_edge_missing_signal(
                    gold_uuid, membership.post_cap_ids, adjacency,
                )
                gm = membership.gold_membership(gold_uuid, spread_edge_missing=missing_signal)
                lexical_context = _compute_lexical_context(store, cue_text, gold_uuid)
                cause = classify_miss_cause(gm, {"in_budget_topk": False}, lexical_context)
                row["cause"] = cause
                row["_diagnostic"] = {
                    "sources": sorted(membership.sources_for(gold_uuid)),
                    "capped_out": gm.capped_out,
                    "spread_edge_missing": gm.spread_edge_missing,
                    "lexical_only_rescue": lexical_context["lexical_only_rescue"],
                    "lexical_contributed": lexical_context["lexical_contributed"],
                }
            rows.append(row)
    return rows


def _cause_distribution(rows: "list[dict]") -> "dict[str, int]":
    dist: "dict[str, int]" = {c: 0 for c in ALL_CAUSES}
    for row in rows:
        cause = row.get("cause")
        if cause is not None:
            dist[cause] = dist.get(cause, 0) + 1
    return dist


def _run_smoke() -> int:
    from bench.recall_quality_fixture import build_minimal_fixture_corpus

    corpus = build_minimal_fixture_corpus(seed=0)
    try:
        _warm_synthetic_store(corpus.store, corpus.embedder)

        per_cue_ms: "list[float]" = []
        rows: "list[dict]" = []
        adjacency = build_edge_adjacency(corpus.store)
        with _inject_embedder_into_dispatch(corpus.shim):
            for cue in corpus.cues:
                start = time.perf_counter()
                cue_text = cue["cue_text"]
                gold_id = str(cue["gold_record_id"])
                returned = _dispatch_recall_ids(corpus.store, cue_text, "smoke")
                rank = _rank_of(gold_id, returned, _RESPONSE_K)
                row: "dict[str, Any]" = {
                    "cue_id": cue["cue_id"],
                    "served": rank is not None,
                    "rank": rank,
                }
                if rank is None:
                    gold_uuid = UUID(gold_id)
                    membership = capture_discovery_membership(corpus.store, {
                        "cue": cue_text,
                        "session_id": "smoke",
                    })
                    missing_signal = spread_edge_missing_signal(
                        gold_uuid, membership.post_cap_ids, adjacency,
                    )
                    gm = membership.gold_membership(gold_uuid, spread_edge_missing=missing_signal)
                    lexical_context = _compute_lexical_context(corpus.store, cue_text, gold_uuid)
                    cause = classify_miss_cause(gm, {"in_budget_topk": False}, lexical_context)
                    row["cause"] = cause
                    row["diagnostic"] = {
                        "sources": sorted(membership.sources_for(gold_uuid)),
                        "capped_out": gm.capped_out,
                        "spread_edge_missing": gm.spread_edge_missing,
                        **lexical_context,
                    }
                per_cue_ms.append((time.perf_counter() - start) * 1000.0)
                rows.append(row)

        dist = _cause_distribution(rows)
        mean_ms = sum(per_cue_ms) / len(per_cue_ms) if per_cue_ms else 0.0
        total_ms = sum(per_cue_ms)

        # Each fixture cue costs ~2 dispatches once missed (baseline +
        # membership capture); a full baseline pass plus one A/A replay
        # pass costs roughly 2x this per-cue average per cue.
        per_cue_budgeted_cost_ms = mean_ms * 2.0
        recommended_n = int(
            (_SMOKE_WALL_CLOCK_BUDGET_SEC * 1000.0) / max(per_cue_budgeted_cost_ms, 1.0)
        )
        recommended_n = max(15, min(recommended_n, 100))

        print(json.dumps({
            "rows": rows,
            "cause_distribution": dist,
            "per_cue_mean_ms": mean_ms,
            "per_cue_total_ms": total_ms,
            "wall_clock_budget_sec": _SMOKE_WALL_CLOCK_BUDGET_SEC,
            "recommended_full_n": recommended_n,
        }, indent=2, default=str))

        missing_causes = [c for c in ALL_CAUSES if dist.get(c, 0) < 1]
        if missing_causes:
            print(f"SMOKE FAILED: causes never provoked: {missing_causes}", file=sys.stderr)
            return 1
        return 0
    finally:
        corpus.close()


_NULL_ARM_FILLER_SEED_OFFSET_A = 9001
_NULL_ARM_FILLER_SEED_OFFSET_B = 9002
_NULL_ARM_NOISE_DECOYS = 6
"""Nuisance-only noise decoys inserted at the served cues' own cosine
range in EACH A/A replica (different per replica via filler_seed) -- the
source of genuine seed-to-seed rank variance the null arm measures.
See build_full_fixture_corpus's filler_seed/n_filler_noise_decoys."""


def _git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _load_fixture(path: "Path | str") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _split_block(rows: "list[dict]") -> dict:
    served_n = sum(1 for r in rows if r["served"])
    missed = [r for r in rows if not r["served"]]
    return {
        "n": len(rows),
        "recall_at_k": (served_n / len(rows)) if rows else 0.0,
        "ndcg_at_k": (
            sum(_ndcg_at_rank(r["rank"]) for r in rows) / len(rows) if rows else 0.0
        ),
        "eligible_missed_n": len(missed),
        "cause_distribution": _cause_distribution(missed),
    }


def _compute_null_arm(seed: int, cues: "list[dict]") -> dict:
    """Two intact replays -- identical gold/cue set (``seed``), a
    different nuisance filler seed each -- paired into a signed per-cue
    delta. NEVER a self-pairing identity: each replica is dispatched
    independently through the real production path."""
    from bench.recall_quality_fixture import build_full_fixture_corpus

    def _dispatch_replica(filler_seed: int) -> "list[dict]":
        corpus, _rationale = build_full_fixture_corpus(
            seed=seed, filler_seed=filler_seed, n_filler_noise_decoys=_NULL_ARM_NOISE_DECOYS,
        )
        try:
            _warm_synthetic_store(corpus.store, corpus.embedder)
            return run_baseline_pass(corpus.store, corpus.shim, cues)
        finally:
            corpus.close()

    rows_a = _dispatch_replica(seed + _NULL_ARM_FILLER_SEED_OFFSET_A)
    rows_b = _dispatch_replica(seed + _NULL_ARM_FILLER_SEED_OFFSET_B)

    paired: "list[tuple[float, float]]" = []
    cluster_ids: "list[str]" = []
    discordant_n = 0
    for row_a, row_b, cue in zip(rows_a, rows_b, cues):
        if (row_a["served"], row_a["rank"]) != (row_b["served"], row_b["rank"]):
            discordant_n += 1
        paired.append((float(row_a["served"]), float(row_b["served"])))
        cluster_ids.append(cue["cluster_id"])

    delta = conditional_recall_delta(paired)
    ci_low = cluster_stratified_bootstrap_ci(paired, cluster_ids)
    negated = [(-b, -a) for (b, a) in paired]
    ci_high = -cluster_stratified_bootstrap_ci(negated, cluster_ids)
    n = len(paired)
    verdict = verdict_predicate(delta, ci_low, 0, {}, n, n_min=N_MIN)

    return {
        "null_arm_delta": delta,
        "null_arm_ci_floor_band": [ci_low, ci_high],
        "null_arm_n": n,
        "null_arm_discordant_n": discordant_n,
        "null_arm_verdict": verdict,
    }


def run_full_fixture(fixture_path: "Path | str", out_path: "Path | str") -> dict:
    """Load the committed fixture, rebuild the synthetic store
    deterministically from its seed, dispatch every cue, and report
    dev/holdout recall SEPARATELY plus a genuine two-intact-replay A/A
    null. Writes the JSON to ``out_path`` before returning (and before
    raising on a vacuous null, so the diagnostic is never lost)."""
    from bench.recall_quality_fixture import build_full_fixture_corpus

    payload = _load_fixture(fixture_path)
    seed = payload["seed"]
    cues = payload["cues"]

    corpus, _rationale = build_full_fixture_corpus(seed=seed)
    try:
        rebuilt_map = {c["cue_id"]: c["gold_record_id"] for c in corpus.cues}
        loaded_map = {c["cue_id"]: c["gold_record_id"] for c in cues}
        if rebuilt_map != loaded_map:
            raise RuntimeError(
                "rebuilding the synthetic store from the fixture's own seed "
                "produced a different cue-to-gold-record mapping than the "
                "committed fixture -- regenerate "
                "bench/results/v28/recall_quality_fixture.json"
            )
        _warm_synthetic_store(corpus.store, corpus.embedder)
        rows = run_baseline_pass(corpus.store, corpus.shim, cues)
    finally:
        corpus.close()

    dev_rows = [row for row, cue in zip(rows, cues) if cue["split"] == "dev"]
    holdout_rows = [row for row, cue in zip(rows, cues) if cue["split"] == "holdout"]

    result: "dict[str, Any]" = {
        "fixture_path": str(fixture_path),
        "fixture_seed": seed,
        "git_sha": _git_sha(Path(_REPO_PATH)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_min": N_MIN,
        "dev": _split_block(dev_rows),
        "holdout": _split_block(holdout_rows),
    }
    result.update(_compute_null_arm(seed, cues))
    result["null_arm_status"] = "OK" if result["null_arm_discordant_n"] > 0 else "VACUOUS"
    result["dev"]["clears_n_min"] = result["dev"]["eligible_missed_n"] >= N_MIN
    result["holdout"]["clears_n_min"] = result["holdout"]["eligible_missed_n"] >= N_MIN

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")

    if result["null_arm_discordant_n"] == 0:
        raise RuntimeError(
            "null_arm_discordant_n == 0 -- the two-intact-replay A/A null "
            "observed zero seed-to-seed variance, which cannot be reported "
            "as a resolved verdict (it would false-green on a second store). "
            f"Written to {out_path} with null_arm_status=VACUOUS; the fixture "
            "needs more marginal cues."
        )

    return result


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    if args.smoke:
        return _run_smoke()

    if args.fixture:
        if not args.out:
            parser.error("--fixture requires --out")
        try:
            result = run_full_fixture(args.fixture, args.out)
        except RuntimeError as exc:
            print(f"BASELINE RUN HALTED: {exc}", file=sys.stderr)
            return 1

        print(json.dumps({
            "out": str(args.out),
            "dev_recall_at_k": result["dev"]["recall_at_k"],
            "holdout_recall_at_k": result["holdout"]["recall_at_k"],
            "null_arm_verdict": result["null_arm_verdict"],
            "null_arm_discordant_n": result["null_arm_discordant_n"],
        }, indent=2))
        return 0

    parser.error("no mode selected -- pass --smoke or --fixture/--out")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
