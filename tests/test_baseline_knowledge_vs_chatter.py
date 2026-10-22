"""Pre-fix baseline: does a named-entity knowledge record, minted through
the real capture+consolidation path with nothing hand-set on
salience_level/tier, survive register-matched chatter on the served
foresight surface? Production defaults throughout (min_cos, tier boost,
goal blend, assistant-tail lane) -- this instrument changes no ranking code,
it only measures the current path.

Session design: knowledge sources live under SESSION_KNOWLEDGE, chatter
under SESSION_CHATTER, and the querying turn under a THIRD session
(SESSION_QUERY) -- foresight excludes same-session records from its own
served pack (foresight.py:_pack_candidates), so chatter and the query MUST
NOT share a session or chatter could never reach packed_ids regardless of
ranking.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import pytest

from iai_mcp import core, foresight
from iai_mcp.capture import capture_turn
from iai_mcp.embed import embedder_for_store
from iai_mcp.sleep import _create_semantic_summary
from iai_mcp.store import MemoryStore
from iai_mcp.store._buffers import flush_record_buffer
from tests._baseline_evidence_merge import (
    _IncompleteEvidenceError,
    _merge_and_validate_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git_head_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def _git_tree_clean() -> bool:
    out = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return out.stdout.strip() == ""


pytestmark = pytest.mark.bench

EVIDENCE_PATH = REPO_ROOT / "bench" / "results" / "baseline_knowledge_vs_chatter.json"

# Three distinct sessions -- foresight's same-session exclusion means chatter
# and the query session must differ, or chatter can never reach packed_ids.
SESSION_KNOWLEDGE = "past-knowledge"
SESSION_CHATTER = "past-chatter"
SESSION_QUERY = "today-knowledge-vs-chatter"

CANDIDATE_PAIRS = [
    {
        "pair_id": "gpt-astra-release",
        "entity": "GPT-6 Astra",
        "knowledge_parts": [
            "GPT-6 Astra release notes, part 1: the coding-focused planning variant ships with extended context windows for large repositories.",
            "GPT-6 Astra release notes, part 2: the planning variant adds a dedicated tool-call budget and stricter function-schema validation.",
            "GPT-6 Astra release notes, part 3: the planning variant defaults to a longer reasoning trace before emitting the final tool call.",
        ],
        "chatter": "yeah so anyway, nothing much new on my end, just been keeping busy with the usual stuff honestly",
        "query": "so hey, anything new on your end lately, just checking in on how things are going",
    },
    {
        "pair_id": "meridian-relay-brief",
        "entity": "Meridian-9 relay satellite",
        "knowledge_parts": [
            "Meridian-9 relay satellite briefing, part 1: the new relay satellite uses a phased-array antenna for higher bandwidth links.",
            "Meridian-9 relay satellite briefing, part 2: the relay satellite's onboard battery was redesigned for a longer eclipse duration.",
            "Meridian-9 relay satellite briefing, part 3: ground stations will need a firmware update before the relay satellite launch window.",
        ],
        "chatter": "honestly not much going on over here, just kind of coasting through the week so far",
        "query": "hey what's been going on with you lately, anything worth mentioning this week",
    },
    {
        "pair_id": "harborview-transit-plan",
        "entity": "Harborview transit expansion",
        "knowledge_parts": [
            "Harborview transit expansion notes, part 1: the transit expansion adds a new light-rail line connecting downtown to the waterfront district.",
            "Harborview transit expansion notes, part 2: the transit expansion budget was approved with additional funding for accessible station platforms.",
            "Harborview transit expansion notes, part 3: construction for the transit expansion is expected to begin next spring after permitting.",
        ],
        "chatter": "not really, same old same old around here, nothing too exciting to report on my side",
        "query": "hey, how's it going, anything exciting happening with you these days",
    },
    {
        "pair_id": "riverstone-acquisition-memo",
        "entity": "Riverstone Analytics acquisition",
        "knowledge_parts": [
            "Riverstone Analytics acquisition memo, part 1: the acquisition closes next quarter pending final regulatory approval.",
            "Riverstone Analytics acquisition memo, part 2: the acquisition brings a new data-pipeline team under the existing engineering org.",
            "Riverstone Analytics acquisition memo, part 3: customer contracts from the acquisition will be migrated over a six-month window.",
        ],
        "chatter": "eh, nothing new really, just been chilling and taking it easy the past couple days",
        "query": "so what's new with you, anything happening worth talking about recently",
    },
    {
        "pair_id": "lumen-cell-briefing",
        "entity": "Lumen solid-state cell",
        "knowledge_parts": [
            "Lumen solid-state cell briefing, part 1: the new solid-state cell doubles energy density over the prior lithium-ion generation.",
            "Lumen solid-state cell briefing, part 2: the solid-state cell tolerates a wider operating temperature range without capacity loss.",
            "Lumen solid-state cell briefing, part 3: manufacturing partners will begin pilot production of the solid-state cell next quarter.",
        ],
        "chatter": "ah nothing much, honestly just been hanging around, nothing new to really talk about today",
        "query": "so what's up with you today, anything at all new or interesting going on",
    },
    {
        "pair_id": "kestrel-reef-survey",
        "entity": "Kestrel Bay reef survey",
        "knowledge_parts": [
            "Kestrel Bay reef survey findings, part 1: the reef survey recorded a rebound in coral coverage along the northern shelf.",
            "Kestrel Bay reef survey findings, part 2: the reef survey team documented three previously unrecorded fish species nearby.",
            "Kestrel Bay reef survey findings, part 3: water temperature readings from the reef survey stayed within the safe threshold this season.",
        ],
        "chatter": "meh, not a whole lot honestly, just kicking back and doing nothing in particular lately",
        "query": "hey what's going on with you these days, anything new or interesting happening",
    },
]

CALIBRATION_ATTEMPT_CAP = len(CANDIDATE_PAIRS)
assert len(CANDIDATE_PAIRS) == CALIBRATION_ATTEMPT_CAP, (
    "cap must equal the authored pair count -- a mismatch silently drops "
    "authored pairs or turns the cap into a no-op"
)

# Separate lane, deliberately NOT part of CANDIDATE_PAIRS: those six pairs
# all use a small-talk query where knowledge's raw cosine sits below the
# admission floor by construction (floor-drop). This pair uses a WORK query
# semantically near the knowledge, so knowledge clears the floor -- the
# failure mode under test here is knowledge admitted but buried in SERVED
# order, not dropped at admission.
WORK_CUE_PAIR = {
    "pair_id": "nightfall-migration-status",
    "entity": "Nightfall API migration",
    "knowledge_parts": [
        "Nightfall API migration status update, part 1: the migration to the new authentication endpoint completed for all internal services last week.",
        "Nightfall API migration status update, part 2: the remaining external partner integrations are scheduled to cut over by the end of the month.",
        "Nightfall API migration status update, part 3: the old authentication endpoint will be decommissioned two weeks after the last partner cuts over.",
    ],
    "chatter": "still working through it, nothing solid to report yet, should have a clearer picture by end of week",
    "query": "hey, what's the current status of the API migration, has everything cut over yet",
}


# Adversarial construction, deliberately NOT organic like WORK_CUE_PAIR:
# chatter reuses the query's surface phrasing near-verbatim (high lexical
# overlap, zero substance) while knowledge states a real fact on-topic but
# in different words. The goal is to reproduce the order-bury failure mode
# -- phrasing-echoing chatter out-cosining substantive knowledge while both
# clear the admission floor -- under a synthetic maximum, to learn whether
# it reproduces at all. Iterated in order until construction succeeds or
# the cap is exhausted; see _construct_order_bury_probe.
_ORDER_BURY_CANDIDATE_CAP = 8
_ORDER_BURY_CANDIDATES = [
    {
        "pair_id": "order-bury-01-cascade-ledger",
        "knowledge_parts": [
            "Cascade Ledger sync service, note 1: the reconciliation job now completes inside the four-hour maintenance window after the index rebuild landed.",
            "Cascade Ledger sync service, note 2: duplicate-transaction detection was tightened to catch replay events from the retry queue.",
            "Cascade Ledger sync service, note 3: the nightly checksum comparison against the mirror region now runs automatically.",
        ],
        "chatter": "hey, no update on the ledger sync issue yet, still waiting to hear something myself",
        "query": "hey, any update on the ledger sync issue yet, still waiting to hear something",
    },
    {
        "pair_id": "order-bury-02-fenwick-queue",
        "knowledge_parts": [
            "Fenwick order queue, note 1: the consumer pool was scaled from four to twelve workers to drain the pending backlog faster.",
            "Fenwick order queue, note 2: a dead-letter routing rule now isolates malformed orders instead of blocking the main lane.",
            "Fenwick order queue, note 3: average processing latency dropped from six minutes to forty seconds after the scaling change.",
        ],
        "chatter": "hey, no update on the order queue backlog yet, still waiting to hear something myself",
        "query": "hey, any update on the order queue backlog yet, still waiting to hear something",
    },
    {
        "pair_id": "order-bury-03-thistle-notify",
        "knowledge_parts": [
            "Thistle notification service, note 1: the delivery worker now batches outbound pushes in groups of fifty to cut round trips.",
            "Thistle notification service, note 2: a regional failover route was added for the push gateway outage last month.",
            "Thistle notification service, note 3: median delivery time fell from ninety seconds to twelve seconds after the batching change.",
        ],
        "chatter": "hey, no update on the notification delay yet, still waiting to hear something myself",
        "query": "hey, any update on the notification delay yet, still waiting to hear something",
    },
    {
        "pair_id": "order-bury-04-bramblewood-export",
        "knowledge_parts": [
            "Bramblewood export job, note 1: the schema validator now rejects malformed rows before they reach the writer stage.",
            "Bramblewood export job, note 2: retry backoff was extended from thirty seconds to five minutes to avoid hammering the destination.",
            "Bramblewood export job, note 3: failure rate dropped from eleven percent to under one percent after the validator change.",
        ],
        "chatter": "hey, no update on the export job failures yet, still waiting to hear something myself",
        "query": "hey, any update on the export job failures yet, still waiting to hear something",
    },
    {
        "pair_id": "order-bury-05-anchorpoint-auth",
        "knowledge_parts": [
            "Anchor Point auth gateway, note 1: the token cache now falls back to the read replica when the primary is unreachable.",
            "Anchor Point auth gateway, note 2: connection pool limits were raised to handle the morning traffic spike.",
            "Anchor Point auth gateway, note 3: the certificate rotation job now runs a week ahead of expiry instead of the day of.",
        ],
        "chatter": "hey, no update on the auth gateway outage yet, still waiting to hear something myself",
        "query": "hey, any update on the auth gateway outage yet, still waiting to hear something",
    },
    {
        "pair_id": "order-bury-06-driftwood-cache",
        "knowledge_parts": [
            "Driftwood cache layer, note 1: the eviction policy switched from strict LRU to a frequency-weighted variant.",
            "Driftwood cache layer, note 2: stale entries past their TTL are now purged by a background sweep instead of on read.",
            "Driftwood cache layer, note 3: hit rate improved from sixty-two percent to eighty-one percent after the policy change.",
        ],
        "chatter": "hey, no update on the cache eviction bug yet, still waiting to hear something myself",
        "query": "hey, any update on the cache eviction bug yet, still waiting to hear something",
    },
    {
        "pair_id": "order-bury-07-palisade-billing",
        "knowledge_parts": [
            "Palisade billing reconciler, note 1: rounding differences between the ledger and the invoice generator were traced to a currency-conversion step.",
            "Palisade billing reconciler, note 2: the reconciler now recomputes totals in a fixed-point type instead of floating point.",
            "Palisade billing reconciler, note 3: the monthly mismatch count fell from several hundred to single digits after the fix.",
        ],
        "chatter": "hey, no update on the billing reconciliation mismatch yet, still waiting to hear something myself",
        "query": "hey, any update on the billing reconciliation mismatch yet, still waiting to hear something",
    },
    {
        "pair_id": "order-bury-08-somerset-deploy",
        "knowledge_parts": [
            "Somerset deploy pipeline, note 1: the flaky integration test was isolated to a race in the container startup probe.",
            "Somerset deploy pipeline, note 2: a readiness check now waits for a health endpoint instead of a fixed sleep.",
            "Somerset deploy pipeline, note 3: pipeline success rate rose from seventy percent to ninety-eight percent after the probe fix.",
        ],
        "chatter": "hey, no update on the deploy pipeline flakiness yet, still waiting to hear something myself",
        "query": "hey, any update on the deploy pipeline flakiness yet, still waiting to hear something",
    },
]
assert len(_ORDER_BURY_CANDIDATES) == _ORDER_BURY_CANDIDATE_CAP, (
    "cap must equal the authored candidate count -- a mismatch silently "
    "drops authored candidates or turns the cap into a no-op"
)


def _select_driver(driver: str, mp: pytest.MonkeyPatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        mp.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        mp.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _tokens(text: str) -> "set[str]":
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _effective_min_cos() -> float:
    """Read the min_cos floor the same way refresh_pack resolves it
    (foresight.py's own env-or-default read) -- never hardcode the default,
    an env override changes what a served/dropped candidate must clear."""
    return foresight._f(foresight.FORESIGHT_MIN_COS_ENV, foresight.FORESIGHT_MIN_COS_DEFAULT)


def _effective_admission_window() -> int:
    """Mirror refresh_pack's own candidate-window derivation: effective_max
    items = max_items + assistant_tail_reserve, window = effective_max_items
    * the overfetch factor -- a fixture-local env override (e.g. this file's
    own FORESIGHT_MAX_ITEMS_ENV override) must widen this window the same
    way it widens the real one, or admission classification and the real
    pack disagree on what is even a candidate."""
    max_items = int(
        foresight._f(foresight.FORESIGHT_MAX_ITEMS_ENV, foresight.FORESIGHT_MAX_ITEMS_DEFAULT)
    )
    tail_off = os.environ.get(foresight.FORESIGHT_ASSISTANT_TAIL_OFF_ENV) == "1"
    tail_reserve = 0 if tail_off else max(
        0,
        int(foresight._f(
            foresight.FORESIGHT_ASSISTANT_TAIL_RESERVE_ENV,
            foresight.FORESIGHT_ASSISTANT_TAIL_RESERVE_DEFAULT,
        )),
    )
    return (max_items + tail_reserve) * foresight._CANDIDATE_OVERFETCH


_RANKING_SHAPES = frozenset({"ranking-floor-drop", "ranking-order-bury"})


@dataclass
class _AttemptResult:
    pair_id: str
    shape: str
    knowledge_id: "str | None" = None
    chatter_id: "str | None" = None
    cosine_rank: "int | None" = None
    chatter_cosine_rank: "int | None" = None
    foresight_served_rank: "int | None" = None
    chatter_served_rank: "int | None" = None
    recall_served_rank: "int | None" = None
    packed_len: "int | None" = None
    skip_counters: "dict | None" = None
    knowledge_created_at: "str | None" = None
    chatter_created_at: "str | None" = None
    knowledge_raw_cos: "float | None" = None
    chatter_raw_cos: "float | None" = None
    effective_min_cos: "float | None" = None
    knowledge_skip_reason: "str | None" = None
    work_cue_shape: "str | None" = None


@dataclass
class _OrderBuryAttempt:
    pair_id: str
    knowledge_id: UUID
    chatter_id: UUID
    knowledge_raw_cos: "float | None"
    chatter_raw_cos: "float | None"
    effective_min_cos: float
    cue_embedding: list


@dataclass
class _CalibrationResult:
    driver: str
    attempts: "list[_AttemptResult]" = field(default_factory=list)
    locked: "list[_AttemptResult]" = field(default_factory=list)
    exhausted_shape: "str | None" = None


def _run_one_pair(store: MemoryStore, pair: dict) -> _AttemptResult:
    # Real capture, defaults only -- no salience_level/tier override is the
    # whole point: the pattern this fixture must NOT reproduce is a
    # hand-set salience_level.
    recs = []
    for part in pair["knowledge_parts"]:
        result = capture_turn(
            store, cue="", text=part,
            tier="episodic", session_id=SESSION_KNOWLEDGE, role="user",
            live_turn=False,
        )
        flush_record_buffer(store)
        recs.append(store.get(UUID(result["record_id"])))

    # Mirror sleep.py's _process_cluster_summaries snippet-concat exactly
    # (newest-first, first-80-chars, first 5 records) -- fidelity to the
    # real pre-fix consolidation output shape, not an idealized summary.
    recs_sorted = sorted(recs, key=lambda r: r.updated_at or r.created_at, reverse=True)
    langs = [r.language for r in recs_sorted if r.language]
    dom_lang = max(set(langs), key=langs.count) if langs else "en"
    summary_text = (
        f"Cluster summary ({len(recs_sorted)} records, lang={dom_lang}): "
        + "; ".join(r.literal_surface[:80] for r in recs_sorted[:5])
    )
    summary_id, folded = _create_semantic_summary(store, recs_sorted, summary_text, dom_lang)
    assert not folded, (
        f"[{pair['pair_id']}] summary folded into an existing record — "
        "no fresh semantic record was minted"
    )
    knowledge_id = summary_id
    knowledge_rec = store.get(knowledge_id)
    knowledge_created_at = (
        knowledge_rec.created_at.isoformat() if knowledge_rec.created_at else None
    )

    chatter_result = capture_turn(
        store, cue="", text=pair["chatter"],
        tier="episodic", session_id=SESSION_CHATTER, role="user",
        live_turn=False,
    )
    flush_record_buffer(store)
    chatter_id = UUID(chatter_result["record_id"])
    chatter_rec = store.get(chatter_id)
    chatter_created_at = chatter_rec.created_at.isoformat() if chatter_rec.created_at else None

    entity_tokens = _tokens(pair["entity"])
    query_tokens = _tokens(pair["query"])
    assert entity_tokens.isdisjoint(query_tokens), (
        f"[{pair['pair_id']}] query shares tokens with the entity name: "
        f"{entity_tokens & query_tokens}"
    )

    embedder = embedder_for_store(store)
    cue_embedding = list(embedder.embed(pair["query"]))

    # Admission classify: a ranking fix cannot help a record that never
    # reaches the raw-cosine ANN window in the first place.
    raw_order = store.query_similar(cue_embedding, k=_effective_admission_window())
    raw_ids = [rec.id for rec, _ in raw_order]
    raw_cos_by_id = {rec.id: cos for rec, cos in raw_order}
    effective_min_cos = _effective_min_cos()
    if knowledge_id not in raw_ids:
        return _AttemptResult(
            pair_id=pair["pair_id"], shape="admission-window",
            knowledge_id=str(knowledge_id), chatter_id=str(chatter_id),
            knowledge_created_at=knowledge_created_at, chatter_created_at=chatter_created_at,
            knowledge_raw_cos=raw_cos_by_id.get(knowledge_id),
            chatter_raw_cos=raw_cos_by_id.get(chatter_id),
            effective_min_cos=effective_min_cos,
        )
    cosine_rank = raw_ids.index(knowledge_id)
    chatter_cosine_rank = raw_ids.index(chatter_id) if chatter_id in raw_ids else None
    knowledge_raw_cos = raw_cos_by_id.get(knowledge_id)
    chatter_raw_cos = raw_cos_by_id.get(chatter_id)

    register_tension = chatter_cosine_rank is not None and chatter_cosine_rank < cosine_rank
    if not register_tension:
        return _AttemptResult(
            pair_id=pair["pair_id"], shape="no-register-tension",
            knowledge_id=str(knowledge_id), chatter_id=str(chatter_id),
            cosine_rank=cosine_rank, chatter_cosine_rank=chatter_cosine_rank,
            knowledge_created_at=knowledge_created_at, chatter_created_at=chatter_created_at,
            knowledge_raw_cos=knowledge_raw_cos, chatter_raw_cos=chatter_raw_cos,
            effective_min_cos=effective_min_cos,
        )

    report = foresight.refresh_pack(
        store, cue_text=pair["query"], cue_embedding=cue_embedding, session_id=SESSION_QUERY,
    )
    packed_ids_raw = report.get("packed_ids") or []
    # q:-prefixed entries are tunnel-question slots, not record ids -- an
    # ordinal rank over the raw list is unstable run-to-run.
    packed_ids = [pid for pid in packed_ids_raw if not pid.startswith("q:")]
    foresight_served_rank = (
        packed_ids.index(str(knowledge_id)) if str(knowledge_id) in packed_ids else None
    )
    chatter_served_rank = (
        packed_ids.index(str(chatter_id)) if str(chatter_id) in packed_ids else None
    )

    if chatter_served_rank is None and foresight_served_rank is None:
        shape = "degenerate-both-dropped"
    elif chatter_served_rank is not None and foresight_served_rank is None:
        # Measured, not inferred: only a raw cosine actually below the
        # floor in force this run is a floor-drop -- an order-only boost
        # (foresight.py's _boost_sort) can never rescue that, so the two
        # sub-shapes point a downstream fix at different mechanisms.
        shape = (
            "ranking-floor-drop"
            if knowledge_raw_cos is not None and knowledge_raw_cos < effective_min_cos
            else "ranking-order-bury"
        )
    elif chatter_served_rank is not None and foresight_served_rank > chatter_served_rank:
        shape = "ranking-order-bury"
    else:
        shape = "no-served-bury"

    if foresight_served_rank is not None:
        knowledge_skip_reason = None
    elif knowledge_raw_cos is not None and knowledge_raw_cos < effective_min_cos:
        knowledge_skip_reason = "cos-below-floor"
    else:
        # refresh_pack's report exposes only aggregate skipped_* counters,
        # never per-record attribution -- honest label for "not the floor,
        # exact cause not observable from the public pack result; see the
        # skip_counters recorded alongside this field."
        knowledge_skip_reason = "unattributed-non-floor-see-skip-counters"

    skip_counters = {
        k: report.get(k)
        for k in (
            "skipped_low_cos", "skipped_same_session",
            "skipped_already_injected", "skipped_duplicate_content",
            "skipped_unconfirmed",
        )
    }

    # Secondary surface, reported separately -- the audit isolates the live
    # defect to the foresight push path specifically, so recall may already
    # pass pre-fix while foresight does not; never blended into one number.
    recall_resp = core.dispatch(store, "memory_recall", {
        "cue": pair["query"], "session_id": SESSION_QUERY, "cue_embedding": list(cue_embedding),
    })
    recall_hit_ids = [h["record_id"] for h in recall_resp.get("hits", [])]
    recall_served_rank = (
        recall_hit_ids.index(str(knowledge_id)) if str(knowledge_id) in recall_hit_ids else None
    )

    return _AttemptResult(
        pair_id=pair["pair_id"], shape=shape,
        knowledge_id=str(knowledge_id), chatter_id=str(chatter_id),
        cosine_rank=cosine_rank, chatter_cosine_rank=chatter_cosine_rank,
        foresight_served_rank=foresight_served_rank, chatter_served_rank=chatter_served_rank,
        recall_served_rank=recall_served_rank,
        packed_len=len(packed_ids_raw), skip_counters=skip_counters,
        knowledge_created_at=knowledge_created_at, chatter_created_at=chatter_created_at,
        knowledge_raw_cos=knowledge_raw_cos, chatter_raw_cos=chatter_raw_cos,
        effective_min_cos=effective_min_cos, knowledge_skip_reason=knowledge_skip_reason,
    )


def _run_work_cue_pair(store: MemoryStore, pair: dict) -> _AttemptResult:
    """Work-cue lane. Unlike _run_one_pair, does not gate on raw-cosine
    register tension -- the failure mode under test is knowledge admitted
    (raw cosine above the floor) but buried in SERVED order, which can occur
    even when knowledge's raw cosine already leads chatter's."""
    recs = []
    for part in pair["knowledge_parts"]:
        result = capture_turn(
            store, cue="", text=part,
            tier="episodic", session_id=SESSION_KNOWLEDGE, role="user",
            live_turn=False,
        )
        flush_record_buffer(store)
        recs.append(store.get(UUID(result["record_id"])))

    recs_sorted = sorted(recs, key=lambda r: r.updated_at or r.created_at, reverse=True)
    langs = [r.language for r in recs_sorted if r.language]
    dom_lang = max(set(langs), key=langs.count) if langs else "en"
    summary_text = (
        f"Cluster summary ({len(recs_sorted)} records, lang={dom_lang}): "
        + "; ".join(r.literal_surface[:80] for r in recs_sorted[:5])
    )
    summary_id, folded = _create_semantic_summary(store, recs_sorted, summary_text, dom_lang)
    assert not folded, (
        f"[{pair['pair_id']}] summary folded into an existing record — "
        "no fresh semantic record was minted"
    )
    knowledge_id = summary_id

    chatter_result = capture_turn(
        store, cue="", text=pair["chatter"],
        tier="episodic", session_id=SESSION_CHATTER, role="user",
        live_turn=False,
    )
    flush_record_buffer(store)
    chatter_id = UUID(chatter_result["record_id"])

    embedder = embedder_for_store(store)
    cue_embedding = list(embedder.embed(pair["query"]))

    raw_order = store.query_similar(cue_embedding, k=_effective_admission_window())
    raw_ids = [rec.id for rec, _ in raw_order]
    raw_cos_by_id = {rec.id: cos for rec, cos in raw_order}
    effective_min_cos = _effective_min_cos()
    knowledge_raw_cos = raw_cos_by_id.get(knowledge_id)
    chatter_raw_cos = raw_cos_by_id.get(chatter_id)
    cosine_rank = raw_ids.index(knowledge_id) if knowledge_id in raw_ids else None
    chatter_cosine_rank = raw_ids.index(chatter_id) if chatter_id in raw_ids else None

    report = foresight.refresh_pack(
        store, cue_text=pair["query"], cue_embedding=cue_embedding, session_id=SESSION_QUERY,
    )
    packed_ids_raw = report.get("packed_ids") or []
    packed_ids = [pid for pid in packed_ids_raw if not pid.startswith("q:")]
    foresight_served_rank = (
        packed_ids.index(str(knowledge_id)) if str(knowledge_id) in packed_ids else None
    )
    chatter_served_rank = (
        packed_ids.index(str(chatter_id)) if str(chatter_id) in packed_ids else None
    )

    knowledge_above_floor = (
        knowledge_raw_cos is not None and knowledge_raw_cos >= effective_min_cos
    )
    chatter_above_floor = chatter_raw_cos is not None and chatter_raw_cos >= effective_min_cos

    if knowledge_raw_cos is None:
        # Knowledge never reached the raw-cosine admission window at all --
        # distinct from clearing the window but sitting below the floor.
        shape = "admission-window"
    elif not knowledge_above_floor:
        shape = "both-below-floor-precondition-unmet" if not chatter_above_floor else "floor-drop"
    elif foresight_served_rank is not None and (
        chatter_served_rank is None or foresight_served_rank <= chatter_served_rank
    ):
        shape = "served-above-chatter"
    elif foresight_served_rank is not None and chatter_served_rank is not None:
        shape = "ranking-order-bury"
    else:
        # Knowledge clears the floor but is absent from the served pack --
        # order- or admission-side either way; refresh_pack's public report
        # never attributes a per-record cause, so label the ambiguity
        # honestly rather than guessing which mechanism dropped it.
        shape = "served-bury-or-nonfloor-drop"

    if foresight_served_rank is not None:
        knowledge_skip_reason = None
    elif knowledge_raw_cos is None:
        knowledge_skip_reason = "not-in-admission-window"
    elif not knowledge_above_floor:
        knowledge_skip_reason = "cos-below-floor"
    else:
        knowledge_skip_reason = "unattributed-non-floor-see-skip-counters"

    skip_counters = {
        k: report.get(k)
        for k in (
            "skipped_low_cos", "skipped_same_session",
            "skipped_already_injected", "skipped_duplicate_content",
            "skipped_unconfirmed",
        )
    }

    return _AttemptResult(
        pair_id=pair["pair_id"], shape=shape,
        knowledge_id=str(knowledge_id), chatter_id=str(chatter_id),
        cosine_rank=cosine_rank, chatter_cosine_rank=chatter_cosine_rank,
        foresight_served_rank=foresight_served_rank, chatter_served_rank=chatter_served_rank,
        packed_len=len(packed_ids_raw), skip_counters=skip_counters,
        knowledge_raw_cos=knowledge_raw_cos, chatter_raw_cos=chatter_raw_cos,
        effective_min_cos=effective_min_cos, knowledge_skip_reason=knowledge_skip_reason,
        work_cue_shape=shape,
    )


def _run_order_bury_candidate(store: MemoryStore, candidate: dict) -> _OrderBuryAttempt:
    """Raw-cosine measurement only, no refresh_pack call -- construction
    tunes on admission-floor cosines alone (cheap relative to serving), and
    refresh_pack is called exactly once, on the winning candidate, so the
    recorded served shape is never itself iterated toward."""
    recs = []
    for part in candidate["knowledge_parts"]:
        result = capture_turn(
            store, cue="", text=part,
            tier="episodic", session_id=SESSION_KNOWLEDGE, role="user",
            live_turn=False,
        )
        flush_record_buffer(store)
        recs.append(store.get(UUID(result["record_id"])))

    recs_sorted = sorted(recs, key=lambda r: r.updated_at or r.created_at, reverse=True)
    langs = [r.language for r in recs_sorted if r.language]
    dom_lang = max(set(langs), key=langs.count) if langs else "en"
    summary_text = (
        f"Cluster summary ({len(recs_sorted)} records, lang={dom_lang}): "
        + "; ".join(r.literal_surface[:80] for r in recs_sorted[:5])
    )
    summary_id, folded = _create_semantic_summary(store, recs_sorted, summary_text, dom_lang)
    assert not folded, (
        f"[{candidate['pair_id']}] summary folded into an existing record — "
        "no fresh semantic record was minted"
    )
    knowledge_id = summary_id

    chatter_result = capture_turn(
        store, cue="", text=candidate["chatter"],
        tier="episodic", session_id=SESSION_CHATTER, role="user",
        live_turn=False,
    )
    flush_record_buffer(store)
    chatter_id = UUID(chatter_result["record_id"])

    embedder = embedder_for_store(store)
    cue_embedding = list(embedder.embed(candidate["query"]))

    raw_order = store.query_similar(cue_embedding, k=_effective_admission_window())
    raw_cos_by_id = {rec.id: cos for rec, cos in raw_order}

    return _OrderBuryAttempt(
        pair_id=candidate["pair_id"],
        knowledge_id=knowledge_id, chatter_id=chatter_id,
        knowledge_raw_cos=raw_cos_by_id.get(knowledge_id),
        chatter_raw_cos=raw_cos_by_id.get(chatter_id),
        effective_min_cos=_effective_min_cos(), cue_embedding=cue_embedding,
    )


def _construct_order_bury_probe(driver: str) -> dict:
    """Iterate _ORDER_BURY_CANDIDATES in order, at most _ORDER_BURY_CANDIDATE_CAP
    times, until BOTH knowledge_raw_cos > effective_min_cos AND
    chatter_raw_cos > knowledge_raw_cos hold on the same candidate. Once hit,
    STOP tuning -- call refresh_pack exactly once on the winning candidate and
    record the served shape as-measured, no further tuning toward a specific
    served outcome. If the cap is exhausted, record the closest cosines
    reached as a first-class outcome, not a fabricated one."""
    mp = pytest.MonkeyPatch()
    try:
        _select_driver(driver, mp)
        mp.setenv(foresight.FORESIGHT_MAX_ITEMS_ENV, "8")

        base = Path(tempfile.mkdtemp(prefix=f"iai-329-orderbury-{driver}-"))
        attempts_log: "list[dict]" = []
        winner: "_OrderBuryAttempt | None" = None
        winner_store: "MemoryStore | None" = None
        winner_candidate: "dict | None" = None

        for candidate in _ORDER_BURY_CANDIDATES[:_ORDER_BURY_CANDIDATE_CAP]:
            store = MemoryStore(path=base / candidate["pair_id"])
            attempt = _run_order_bury_candidate(store, candidate)
            attempts_log.append({
                "pair_id": attempt.pair_id,
                "knowledge_raw_cos": attempt.knowledge_raw_cos,
                "chatter_raw_cos": attempt.chatter_raw_cos,
            })
            constructed = (
                attempt.knowledge_raw_cos is not None
                and attempt.knowledge_raw_cos > attempt.effective_min_cos
                and attempt.chatter_raw_cos is not None
                and attempt.chatter_raw_cos > attempt.knowledge_raw_cos
            )
            if constructed:
                winner = attempt
                winner_store = store
                winner_candidate = candidate
                break
            store.close()

        iterations_to_construct = len(attempts_log)

        if winner is None:
            return {
                "order_bury_shape": "could-not-construct-chatter-above-knowledge",
                "iterations_to_construct": iterations_to_construct,
                "knowledge_raw_cos": attempts_log[-1]["knowledge_raw_cos"] if attempts_log else None,
                "chatter_raw_cos": attempts_log[-1]["chatter_raw_cos"] if attempts_log else None,
                "effective_min_cos": _effective_min_cos(),
                "foresight_served_rank": None,
                "chatter_served_rank": None,
                "knowledge_skip_reason": None,
                "skip_counters": None,
            }

        try:
            report = foresight.refresh_pack(
                winner_store, cue_text=winner_candidate["query"],
                cue_embedding=winner.cue_embedding, session_id=SESSION_QUERY,
            )
        finally:
            winner_store.close()

        packed_ids_raw = report.get("packed_ids") or []
        packed_ids = [pid for pid in packed_ids_raw if not pid.startswith("q:")]
        foresight_served_rank = (
            packed_ids.index(str(winner.knowledge_id)) if str(winner.knowledge_id) in packed_ids else None
        )
        chatter_served_rank = (
            packed_ids.index(str(winner.chatter_id)) if str(winner.chatter_id) in packed_ids else None
        )

        if foresight_served_rank is None and chatter_served_rank is None:
            pytest.fail(
                f"[{driver}] order-bury probe degenerate: both knowledge and "
                f"chatter absent from the served pack despite chatter_raw_cos "
                f"({winner.chatter_raw_cos}) > knowledge_raw_cos "
                f"({winner.knowledge_raw_cos}) > effective_min_cos "
                f"({winner.effective_min_cos}) -- report={report}"
            )

        if (
            foresight_served_rank is not None and chatter_served_rank is not None
            and foresight_served_rank > chatter_served_rank
        ):
            order_bury_shape = "order-bury-reproduced"
        elif chatter_served_rank is not None and foresight_served_rank is None:
            order_bury_shape = "chatter-served-knowledge-dropped-despite-floor"
        else:
            order_bury_shape = "knowledge-still-leads"

        if foresight_served_rank is not None:
            knowledge_skip_reason = None
        elif winner.knowledge_raw_cos is not None and winner.knowledge_raw_cos < winner.effective_min_cos:
            knowledge_skip_reason = "cos-below-floor"
        else:
            knowledge_skip_reason = "unattributed-non-floor-see-skip-counters"

        skip_counters = {
            k: report.get(k)
            for k in (
                "skipped_low_cos", "skipped_same_session",
                "skipped_already_injected", "skipped_duplicate_content",
                "skipped_unconfirmed",
            )
        }

        return {
            "order_bury_shape": order_bury_shape,
            "iterations_to_construct": iterations_to_construct,
            "pair_id": winner.pair_id,
            "knowledge_raw_cos": winner.knowledge_raw_cos,
            "chatter_raw_cos": winner.chatter_raw_cos,
            "effective_min_cos": winner.effective_min_cos,
            "foresight_served_rank": foresight_served_rank,
            "chatter_served_rank": chatter_served_rank,
            "knowledge_skip_reason": knowledge_skip_reason,
            "skip_counters": skip_counters,
        }
    finally:
        mp.undo()


_ORDER_BURY_CACHE: "dict[str, dict]" = {}


def _get_order_bury_probe(driver: str) -> dict:
    if driver not in _ORDER_BURY_CACHE:
        _ORDER_BURY_CACHE[driver] = _construct_order_bury_probe(driver)
    return _ORDER_BURY_CACHE[driver]


_CALIBRATION_CACHE: "dict[str, _CalibrationResult]" = {}
_WORK_CUE_CACHE: "dict[str, _AttemptResult]" = {}
_EVIDENCE: "dict[str, dict]" = {}


def _exhaustion_shape(attempts: "list[_AttemptResult]") -> str:
    admitted = [a for a in attempts if a.shape != "admission-window"]
    if not admitted:
        return "admission-window"
    if any(a.shape in ("no-served-bury", "no-register-tension") for a in admitted):
        return "no-served-bury"
    return "degenerate-both-dropped"


def _get_calibration(driver: str) -> _CalibrationResult:
    if driver in _CALIBRATION_CACHE:
        return _CALIBRATION_CACHE[driver]

    mp = pytest.MonkeyPatch()
    try:
        _select_driver(driver, mp)
        mp.setenv(foresight.FORESIGHT_MAX_ITEMS_ENV, "8")

        base = Path(tempfile.mkdtemp(prefix=f"iai-329-baseline-{driver}-"))
        attempts: "list[_AttemptResult]" = []
        for pair in CANDIDATE_PAIRS[:CALIBRATION_ATTEMPT_CAP]:
            store = MemoryStore(path=base / pair["pair_id"])
            try:
                attempts.append(_run_one_pair(store, pair))
            finally:
                store.close()
    finally:
        mp.undo()

    locked = [a for a in attempts if a.shape in _RANKING_SHAPES]
    result = _CalibrationResult(
        driver=driver, attempts=attempts, locked=locked,
        exhausted_shape=None if locked else _exhaustion_shape(attempts),
    )
    _CALIBRATION_CACHE[driver] = result
    return result


def _get_work_cue_attempt(driver: str) -> _AttemptResult:
    if driver in _WORK_CUE_CACHE:
        return _WORK_CUE_CACHE[driver]

    mp = pytest.MonkeyPatch()
    try:
        _select_driver(driver, mp)
        mp.setenv(foresight.FORESIGHT_MAX_ITEMS_ENV, "8")

        base = Path(tempfile.mkdtemp(prefix=f"iai-329-workcue-{driver}-"))
        store = MemoryStore(path=base / WORK_CUE_PAIR["pair_id"])
        try:
            attempt = _run_work_cue_pair(store, WORK_CUE_PAIR)
        finally:
            store.close()
    finally:
        mp.undo()

    _WORK_CUE_CACHE[driver] = attempt
    return attempt


@pytest.fixture(scope="session", autouse=True)
def _write_baseline_evidence_at_session_end():
    yield
    if not _EVIDENCE:
        return
    existing: dict = {}
    if EVIDENCE_PATH.exists():
        try:
            existing = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            pytest.fail(f"committed evidence file is corrupt JSON at {EVIDENCE_PATH}: {exc}")
            return
    try:
        merged = _merge_and_validate_evidence(existing, _EVIDENCE)
    except _IncompleteEvidenceError as exc:
        pytest.fail(
            f"refusing to write baseline evidence: {exc} -- re-run the full "
            "file (not a filtered subset) so every writer test populates its "
            "fields before this fixture writes"
        )
        return
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_served_surface_reproduces_pre_fix_miss(driver):
    """Load-bearing guard, stable name: this small-talk floor-drop lane is
    expected to remain unchanged post-fix -- its raw-cosine admission
    floor-drop is structurally immune to a ranking fix. The measured
    reorder improvement is proved separately, against the order_bury_probe
    lane, in the sibling gate module. Reproduction is NON-VACUOUS only when
    chatter is PRESENT on the pack AND knowledge is absent-or-lower —
    both-absent is degenerate and proves nothing."""
    calib = _get_calibration(driver)

    if not calib.locked:
        _EVIDENCE.setdefault(driver, {})
        _EVIDENCE[driver].update({
            "tree_sha": _git_head_sha(),
            "tree_clean": _git_tree_clean(),
            "driver": driver,
            "reproduced": False,
            "shape": calib.exhausted_shape,
            "selected_pair_id": None,
            "attempts": [a.pair_id for a in calib.attempts],
            "attempt_shapes": {a.pair_id: a.shape for a in calib.attempts},
        })
        pytest.fail(
            f"[{driver}] pre-fix baseline not reproduced within "
            f"{CALIBRATION_ATTEMPT_CAP} attempts; shape={calib.exhausted_shape}"
        )

    attempt = calib.locked[0]
    assert attempt.chatter_served_rank is not None, (
        f"[{driver}/{attempt.pair_id}] chatter must be PRESENT on the served "
        f"pack for a non-vacuous reproduction: {attempt}"
    )
    assert attempt.foresight_served_rank is None or (
        attempt.foresight_served_rank > attempt.chatter_served_rank
    ), (
        f"[{driver}/{attempt.pair_id}] knowledge must be absent-or-lower than "
        f"chatter on the served pack: {attempt}"
    )

    foresight_miss = attempt.foresight_served_rank is None or (
        attempt.foresight_served_rank > attempt.chatter_served_rank
    )
    foresight_rank_delta = (
        attempt.foresight_served_rank - attempt.cosine_rank
        if attempt.foresight_served_rank is not None else None
    )
    _EVIDENCE.setdefault(driver, {})
    _EVIDENCE[driver].update({
        "tree_sha": _git_head_sha(),
        "tree_clean": _git_tree_clean(),
        "driver": driver,
        "reproduced": True,
        "shape": attempt.shape,
        "selected_pair_id": attempt.pair_id,
        "cosine_rank": attempt.cosine_rank,
        "foresight_served_rank": attempt.foresight_served_rank,
        "chatter_served_rank": attempt.chatter_served_rank,
        "foresight_rank_delta": foresight_rank_delta,
        "packed_len": attempt.packed_len,
        "foresight_miss": foresight_miss,
        **(attempt.skip_counters or {}),
        "knowledge_created_at": attempt.knowledge_created_at,
        "chatter_created_at": attempt.chatter_created_at,
        "knowledge_raw_cos": attempt.knowledge_raw_cos,
        "chatter_raw_cos": attempt.chatter_raw_cos,
        "effective_min_cos": attempt.effective_min_cos,
        "knowledge_skip_reason": attempt.knowledge_skip_reason,
    })


def test_both_drivers_select_identical_locked_pair():
    """Driver choice changes neither the embedder nor cosine values, so a
    divergent selection would make a later gate compare one pair's pre-fix
    number against a DIFFERENT pair's post-fix number -- fail loud instead."""
    stdlib_calib = _get_calibration("stdlib")
    lilli_calib = _get_calibration("lilli")  # skips the whole test if not built

    assert stdlib_calib.locked and lilli_calib.locked, (
        f"both drivers must reproduce: stdlib_shape={stdlib_calib.exhausted_shape} "
        f"lilli_shape={lilli_calib.exhausted_shape}"
    )
    assert stdlib_calib.locked[0].pair_id == lilli_calib.locked[0].pair_id, (
        f"driver selection diverged: stdlib={stdlib_calib.locked[0].pair_id} "
        f"lilli={lilli_calib.locked[0].pair_id}"
    )


_ALL_PAIR_IDS = [p["pair_id"] for p in CANDIDATE_PAIRS]


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
@pytest.mark.parametrize("pair_id", _ALL_PAIR_IDS)
def test_calibrated_pair_reproduces_register_tension(pair_id, driver):
    """Non-vacuity check: every LOCKED pair must individually reproduce
    admission + register-tension under the real embedder, so a later
    un-calibrated text edit fails loudly instead of silently going vacuous.
    A pair not classified into a ranking shape is skipped here, not asserted
    false — the ranking miss-rate test below is where non-ranking shapes
    count."""
    calib = _get_calibration(driver)
    attempt = next(a for a in calib.attempts if a.pair_id == pair_id)
    if attempt.shape not in _RANKING_SHAPES:
        pytest.skip(
            f"[{driver}/{pair_id}] not calibrated to a ranking shape (shape={attempt.shape})"
        )

    assert attempt.cosine_rank is not None, (
        f"[{driver}/{pair_id}] knowledge must be a real reachable ANN "
        f"candidate (non-vacuity check): {attempt}"
    )
    assert (
        attempt.chatter_cosine_rank is not None
        and attempt.chatter_cosine_rank < attempt.cosine_rank
    ), (
        f"[{driver}/{pair_id}] chatter must outrank knowledge by raw cosine "
        f"(register tension, non-vacuity check): {attempt}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_foresight_missrate_over_calibrated_set(driver):
    """Miss-rate denominator is the ADMISSION-ELIGIBLE set only (pairs whose
    knowledge id reached the raw-cosine top-24) -- an admission-shaped pair
    is a ranking fix could never touch and is reported separately via
    admission_shaped_count, never folded into the ranking miss-rate."""
    calib = _get_calibration(driver)
    admission_shaped = [a for a in calib.attempts if a.shape == "admission-window"]
    eligible = [a for a in calib.attempts if a.shape != "admission-window"]

    if not eligible:
        _EVIDENCE.setdefault(driver, {})
        _EVIDENCE[driver].update({
            "tree_sha": _git_head_sha(),
            "tree_clean": _git_tree_clean(),
            "driver": driver,
            "reproduced": False,
            "shape": "admission-window",
            "selected_pair_id": None,
            "admission_shaped_count": len(admission_shaped),
            "attempt_shapes": {a.pair_id: a.shape for a in calib.attempts},
        })
        pytest.fail(
            f"[{driver}] zero pairs passed admission -- this IS the "
            "admission-window escalation, not a ranking-fixable failure"
        )

    def _is_miss(a: _AttemptResult) -> bool:
        return a.chatter_served_rank is not None and (
            a.foresight_served_rank is None or a.foresight_served_rank > a.chatter_served_rank
        )

    misses = [a for a in eligible if _is_miss(a)]
    miss_rate = len(misses) / len(eligible)
    assert miss_rate > 0, (
        f"[{driver}] miss-rate is 0 over {len(eligible)} admission-eligible "
        "pairs -- the pre-fix failure this fixture measures is absent, not reproduced"
    )

    calib_locked_ids = {a.pair_id for a in calib.locked}
    _EVIDENCE.setdefault(driver, {})
    _EVIDENCE[driver].update({
        "tree_sha": _git_head_sha(),
        "tree_clean": _git_tree_clean(),
        "driver": driver,
        "miss_rate": round(miss_rate, 4),
        "admission_shaped_count": len(admission_shaped),
        "admission_eligible_count": len(eligible),
        "locked_pair_ids": sorted(calib_locked_ids),
        "attempt_shapes": {a.pair_id: a.shape for a in calib.attempts},
    })


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_recall_secondary_surface_reported_separately(driver):
    """The awake memory_recall surface, reported under its OWN key -- never
    blended with the foresight number. The audit isolates the live defect to
    the foresight push path specifically; recall may already pass pre-fix
    while foresight does not, and a blended figure would make a later
    gate's delta uninterpretable."""
    calib = _get_calibration(driver)
    if not calib.locked:
        pytest.skip(f"[{driver}] no locked pair to report the recall surface for")

    attempt = calib.locked[0]
    recall_miss = attempt.recall_served_rank is None
    recall_rank_delta = (
        attempt.recall_served_rank - attempt.cosine_rank
        if attempt.recall_served_rank is not None else None
    )
    _EVIDENCE.setdefault(driver, {})
    _EVIDENCE[driver].update({
        "tree_sha": _git_head_sha(),
        "tree_clean": _git_tree_clean(),
        "driver": driver,
        "recall_served_rank": attempt.recall_served_rank,
        "recall_miss": recall_miss,
        "recall_rank_delta": recall_rank_delta,
    })


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_work_cue_order_bury_lane(driver):
    """Separate lane, does not disturb the small-talk pairs above: measures
    whether a WORK-phrased query -- where knowledge's raw cosine already
    clears the admission floor, unlike the small-talk pairs -- still buries
    knowledge below chatter in SERVED order. Measurement only, no assertion
    on the outcome; the shape itself is the evidence."""
    attempt = _get_work_cue_attempt(driver)
    _EVIDENCE.setdefault(driver, {})
    _EVIDENCE[driver]["work_cue"] = {
        "pair_id": attempt.pair_id,
        "knowledge_raw_cos": attempt.knowledge_raw_cos,
        "chatter_raw_cos": attempt.chatter_raw_cos,
        "effective_min_cos": attempt.effective_min_cos,
        "foresight_served_rank": attempt.foresight_served_rank,
        "chatter_served_rank": attempt.chatter_served_rank,
        "knowledge_skip_reason": attempt.knowledge_skip_reason,
        "skip_counters": attempt.skip_counters,
        "work_cue_shape": attempt.work_cue_shape,
    }


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_order_bury_probe_lane(driver):
    """Adversarial construction, deliberately NOT organic: builds
    phrasing-echoing chatter that out-cosines a topically-relevant knowledge
    record while both clear the admission floor, to check whether the
    order-bury failure mode reproduces at all under a synthetic maximum.
    Separate evidence key from the small-talk pairs and the work_cue lane.
    Measurement only -- no assertion on which order_bury_shape results; the
    shape itself is the finding."""
    result = _get_order_bury_probe(driver)
    _EVIDENCE.setdefault(driver, {})
    _EVIDENCE[driver]["order_bury_probe"] = result
