"""Diagnostic: attempts to construct a genuine pre-fix loss for knowledge
against register-matched chatter on the `memory_recall` / served-order
surface, under real embeddings, with salience left at whatever the real
capture path actually produces -- never hand-set.

Two legs, both measurement-first (no fabricated tie, no forced assertion on
an outcome that was not observed):

  (a) per-pair reporting -- iterates every calibrated pair from the shipped
      knowledge-vs-chatter harness and reports each pair's served-recall
      state individually, not just the single locked pair the shipped
      fixture reports.
  (b) constructed probe -- attempts to find a near-tied-cosine knowledge/
      chatter pair (real embeddings, real capture path) and records the
      served-order outcome, or records honestly that no near-tie could be
      constructed within the candidate budget.

A run that collects zero pairs or constructs nothing is INCONCLUSIVE, not a
refutation -- this repo's rule is to suspect the fixture before declaring a
symptom absent.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pytest

from iai_mcp import core, foresight
from iai_mcp.capture import capture_turn
from iai_mcp.embed import embedder_for_store
from iai_mcp.store import MemoryStore
from iai_mcp.store._buffers import flush_record_buffer
from tests.test_baseline_knowledge_vs_chatter import (
    _effective_admission_window,
    _effective_min_cos,
    _get_calibration,
    _select_driver,
)

pytestmark = pytest.mark.bench

_DIAGNOSTIC_EVIDENCE: "dict[str, dict]" = {}


# --- Leg (a): per-pair reporting over the shipped calibration -------------

_NOT_MEASURED_SHAPES = ("admission-window", "no-register-tension")


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_per_pair_recall_surface_state_before_any_change(driver, capsys):
    """Reports EVERY calibrated pair's served-recall outcome, not just the
    single locked[0] pair the shipped fixture reports -- a full per-pair
    picture of the current recall/served-order surface. A pair whose shape
    is admission-window or no-register-tension never reached the recall
    dispatch call in the shared harness, so its recall state is reported as
    "not measured", never conflated with a genuine miss."""
    calib = _get_calibration(driver)
    assert calib.attempts, (
        f"[{driver}] diagnostic collected zero pairs -- this is INCONCLUSIVE, "
        "never record it as 'no red-before found'"
    )

    locked_ids = sorted(a.pair_id for a in calib.locked)
    rows = []
    for attempt in calib.attempts:
        recall_measured = attempt.shape not in _NOT_MEASURED_SHAPES
        recall_miss = (attempt.recall_served_rank is None) if recall_measured else None
        rows.append({
            "pair_id": attempt.pair_id,
            "shape": attempt.shape,
            "recall_measured": recall_measured,
            "recall_served_rank": attempt.recall_served_rank,
            "recall_miss": recall_miss,
        })

    _DIAGNOSTIC_EVIDENCE.setdefault(driver, {})
    _DIAGNOSTIC_EVIDENCE[driver]["attempts_collected"] = len(calib.attempts)
    _DIAGNOSTIC_EVIDENCE[driver]["locked_pair_ids"] = locked_ids
    _DIAGNOSTIC_EVIDENCE[driver]["per_pair"] = rows

    with capsys.disabled():
        print()
        print(f"[{driver}] attempts collected: {len(calib.attempts)}")
        print(f"[{driver}] locked pair_ids: {locked_ids}")
        for row in rows:
            print(
                f"[{driver}] pair={row['pair_id']} shape={row['shape']} "
                f"recall_measured={row['recall_measured']} "
                f"recall_served_rank={row['recall_served_rank']} "
                f"recall_miss={row['recall_miss']}"
            )


# --- Leg (b): constructed near-tied-cosine probe, real capture path -------

# A near tie, not an exact one -- an exact cosine tie is not constructible
# under real embeddings without monkeypatching the vectors, which this
# diagnostic must never do.
_NEAR_TIE_EPSILON = 0.02

_SESSION_KNOWLEDGE = "value-signal-diagnostic-knowledge"
_SESSION_CHATTER = "value-signal-diagnostic-chatter"
_SESSION_QUERY = "value-signal-diagnostic-query"

# Register-matched, entity-repeating chatter (unlike the shipped harness's
# small-talk chatter, which never mentions the entity) -- deliberately
# topically closer to the knowledge text, to raise the odds of a near-tied
# cosine rather than the floor-drop shape the shipped small-talk pairs hit.
_NEAR_TIE_CANDIDATES = [
    {
        "pair_id": "value-diag-01-crestline-batch",
        "knowledge_text": (
            "The Crestline batch job now finishes in under ten minutes after "
            "alice's index change landed this week."
        ),
        "chatter_text": (
            "hey, did the Crestline batch job speed up after that change "
            "landed this week, curious how it's going"
        ),
        "query": "hey, how's the Crestline batch job doing lately, any speed improvements this week",
    },
    {
        "pair_id": "value-diag-02-harrow-search",
        "knowledge_text": (
            "The Harrow search index started returning fresher results after "
            "alice tuned the refresh interval yesterday."
        ),
        "chatter_text": (
            "hey, is the Harrow search index still showing stale results, or "
            "did that get fixed yesterday"
        ),
        "query": "hey, how's the Harrow search index doing, still seeing stale results or is it better now",
    },
    {
        "pair_id": "value-diag-03-quillfeed-latency",
        "knowledge_text": (
            "Quillfeed's median latency dropped by half after alice moved the "
            "cache warm-up earlier in the boot sequence."
        ),
        "chatter_text": (
            "hey, has Quillfeed's latency gotten any better since that cache "
            "change, or still about the same"
        ),
        "query": "hey, what's the latest on Quillfeed's latency, any improvement since the cache change",
    },
    {
        "pair_id": "value-diag-04-mossvale-sync",
        "knowledge_text": (
            "The Mossvale sync job stopped dropping records after alice fixed "
            "the retry backoff logic this morning."
        ),
        "chatter_text": (
            "hey, is the Mossvale sync job still dropping records, or did "
            "that retry fix land this morning"
        ),
        "query": "hey, how's the Mossvale sync job doing, still dropping records or is that sorted now",
    },
    {
        "pair_id": "value-diag-05-driftpine-alerts",
        "knowledge_text": (
            "Driftpine's alerting pipeline now suppresses duplicate pages "
            "after alice added a dedup window last night."
        ),
        "chatter_text": (
            "hey, are we still getting duplicate pages from Driftpine, or did "
            "that dedup fix go in last night"
        ),
        "query": "hey, quick check, still getting duplicate pages from Driftpine or did that get fixed",
    },
    {
        "pair_id": "value-diag-06-thornwick-billing",
        "knowledge_text": (
            "Thornwick's billing reconciler now matches invoices correctly "
            "after alice corrected the rounding rule this afternoon."
        ),
        "chatter_text": (
            "hey, does Thornwick's billing reconciler still mismatch "
            "invoices, or did that rounding fix land this afternoon"
        ),
        "query": "hey, any update on Thornwick's billing mismatches, did that rounding fix land",
    },
    {
        "pair_id": "value-diag-07-galewood-auth",
        "knowledge_text": (
            "Galewood's auth gateway stopped rejecting valid tokens after "
            "alice patched the clock-skew check overnight."
        ),
        "chatter_text": (
            "hey, is Galewood's auth gateway still rejecting valid tokens, or "
            "did that clock fix land overnight"
        ),
        "query": "hey, quick status on Galewood's auth gateway, still rejecting tokens or fixed now",
    },
    {
        "pair_id": "value-diag-08-sablecreek-export",
        "knowledge_text": (
            "Sablecreek's export job now completes without truncating rows "
            "after alice widened the buffer size this week."
        ),
        "chatter_text": (
            "hey, is Sablecreek's export job still truncating rows, or did "
            "that buffer fix land this week"
        ),
        "query": "hey, how's Sablecreek's export job doing, still truncating rows or fixed now",
    },
]


@dataclass
class _NearTieAttempt:
    pair_id: str
    knowledge_id: "UUID | None" = None
    chatter_id: "UUID | None" = None
    knowledge_raw_cos: "float | None" = None
    chatter_raw_cos: "float | None" = None
    effective_min_cos: "float | None" = None


def _run_near_tie_candidate(store: MemoryStore, candidate: dict) -> _NearTieAttempt:
    know_result = capture_turn(
        store, cue="", text=candidate["knowledge_text"],
        tier="episodic", session_id=_SESSION_KNOWLEDGE, role="user", live_turn=False,
    )
    flush_record_buffer(store)
    knowledge_id = UUID(know_result["record_id"])

    chatter_result = capture_turn(
        store, cue="", text=candidate["chatter_text"],
        tier="episodic", session_id=_SESSION_CHATTER, role="user", live_turn=False,
    )
    flush_record_buffer(store)
    chatter_id = UUID(chatter_result["record_id"])

    knowledge_rec = store.get(knowledge_id)
    chatter_rec = store.get(chatter_id)
    assert knowledge_rec.salience_level == "unflagged", (
        f"[{candidate['pair_id']}] knowledge must start unflagged (real capture "
        "path, nothing hand-set) for this to be a genuine pre-backfill probe"
    )
    assert chatter_rec.salience_level == "unflagged", (
        f"[{candidate['pair_id']}] chatter must start unflagged"
    )

    embedder = embedder_for_store(store)
    cue_embedding = list(embedder.embed(candidate["query"]))
    raw_order = store.query_similar(cue_embedding, k=_effective_admission_window())
    raw_cos_by_id = {rec.id: cos for rec, cos in raw_order}

    return _NearTieAttempt(
        pair_id=candidate["pair_id"], knowledge_id=knowledge_id, chatter_id=chatter_id,
        knowledge_raw_cos=raw_cos_by_id.get(knowledge_id),
        chatter_raw_cos=raw_cos_by_id.get(chatter_id),
        effective_min_cos=_effective_min_cos(),
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_constructed_near_tie_probe_records_pre_backfill_outcome(driver, capsys):
    """Attempts to construct a near-tied-cosine knowledge/chatter pair under
    real embeddings, salience left at its real (unflagged) pre-backfill
    state -- never hand-set. Measurement only: records whether knowledge
    loses to chatter in served recall order when a near-tie is found; if no
    near-tie constructs within the candidate budget, records that outcome
    honestly rather than asserting a result never observed."""
    mp = pytest.MonkeyPatch()
    try:
        _select_driver(driver, mp)
        mp.setenv(foresight.FORESIGHT_MAX_ITEMS_ENV, "8")

        base = Path(tempfile.mkdtemp(prefix=f"iai-neartie-probe-{driver}-"))
        attempts_log: "list[_NearTieAttempt]" = []
        winner: "_NearTieAttempt | None" = None
        winner_store: "MemoryStore | None" = None
        winner_candidate: "dict | None" = None

        for candidate in _NEAR_TIE_CANDIDATES:
            store = MemoryStore(path=base / candidate["pair_id"])
            attempt = _run_near_tie_candidate(store, candidate)
            attempts_log.append(attempt)

            both_present = (
                attempt.knowledge_raw_cos is not None and attempt.chatter_raw_cos is not None
            )
            both_above_floor = (
                both_present
                and attempt.knowledge_raw_cos >= attempt.effective_min_cos
                and attempt.chatter_raw_cos >= attempt.effective_min_cos
            )
            near_tied = (
                both_above_floor
                and abs(attempt.knowledge_raw_cos - attempt.chatter_raw_cos) < _NEAR_TIE_EPSILON
            )
            if near_tied:
                winner = attempt
                winner_store = store
                winner_candidate = candidate
                break
            store.close()

        if winner is None:
            _DIAGNOSTIC_EVIDENCE.setdefault(driver, {})
            _DIAGNOSTIC_EVIDENCE[driver]["near_tie_probe"] = {
                "outcome": "could-not-construct-near-tie",
                "candidates_tried": len(attempts_log),
                "closest_margin": min(
                    (
                        abs(a.knowledge_raw_cos - a.chatter_raw_cos)
                        for a in attempts_log
                        if a.knowledge_raw_cos is not None and a.chatter_raw_cos is not None
                    ),
                    default=None,
                ),
            }
            with capsys.disabled():
                print()
                print(
                    f"[{driver}] near-tie probe: could not construct within "
                    f"{len(_NEAR_TIE_CANDIDATES)} candidates -- INCONCLUSIVE for this leg"
                )
            return

        assert winner_store is not None and winner_candidate is not None
        cue_embedding = list(embedder_for_store(winner_store).embed(winner_candidate["query"]))
        try:
            recall_resp = core.dispatch(winner_store, "memory_recall", {
                "cue": winner_candidate["query"],
                "session_id": _SESSION_QUERY,
                "budget_tokens": 2000,
                "cue_embedding": cue_embedding,
            })
        finally:
            winner_store.close()

        assert "error" not in recall_resp, f"recall dispatch errored: {recall_resp.get('error')}"
        hit_ids = [h["record_id"] for h in recall_resp.get("hits", [])]
        knowledge_served_rank = (
            hit_ids.index(str(winner.knowledge_id)) if str(winner.knowledge_id) in hit_ids else None
        )
        chatter_served_rank = (
            hit_ids.index(str(winner.chatter_id)) if str(winner.chatter_id) in hit_ids else None
        )
        pre_backfill_loss = chatter_served_rank is not None and (
            knowledge_served_rank is None or knowledge_served_rank > chatter_served_rank
        )

        _DIAGNOSTIC_EVIDENCE.setdefault(driver, {})
        _DIAGNOSTIC_EVIDENCE[driver]["near_tie_probe"] = {
            "outcome": "constructed",
            "pair_id": winner.pair_id,
            "knowledge_raw_cos": winner.knowledge_raw_cos,
            "chatter_raw_cos": winner.chatter_raw_cos,
            "knowledge_served_rank": knowledge_served_rank,
            "chatter_served_rank": chatter_served_rank,
            "pre_backfill_loss": pre_backfill_loss,
        }
        with capsys.disabled():
            print()
            print(f"[{driver}] near-tie probe constructed: pair={winner.pair_id}")
            print(
                f"[{driver}]   knowledge_raw_cos={winner.knowledge_raw_cos} "
                f"chatter_raw_cos={winner.chatter_raw_cos}"
            )
            print(
                f"[{driver}]   knowledge_served_rank={knowledge_served_rank} "
                f"chatter_served_rank={chatter_served_rank} "
                f"pre_backfill_loss={pre_backfill_loss}"
            )
    finally:
        mp.undo()
