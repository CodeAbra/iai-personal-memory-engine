from __future__ import annotations

import hashlib
import logging
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from math import log
from uuid import UUID

import numpy as np

from iai_mcp.community import CommunityAssignment
from iai_mcp.embed import Embedder, _valid_cue_vec, embed_query
from iai_mcp.events import TELEMETRY_EMBED_NATIVE_FAILURE, write_event
from iai_mcp.recall_suppression import recall_suppressed
from iai_mcp.exceptions import (
    NativeError,
)
from iai_mcp.graph import MemoryGraph
from iai_mcp.store import MemoryStore
from iai_mcp.store._store import BOOST_EDGES_SMALL_BATCH
from iai_mcp.types import EMBED_DIM, SALIENCE_LEVEL_RANK, MemoryHit, RecallResponse

logger = logging.getLogger(__name__)


@dataclass
class SimpleRecordView:

    id: UUID
    # Graph-sourced views alias the graph's own float32 buffer; store-sourced
    # records still carry a Python list, and a node stored without a vector
    # carries None. Read it with `is not None`, never a bare truth test.
    embedding: "np.ndarray | list[float] | None"
    literal_surface: str
    centrality: float
    tier: str
    aaak_index: str = ""
    created_at: "datetime | None" = None
    stability: float = 0.5
    profile_modulation_gain: dict = field(default_factory=dict)
    structure_hv: bytes = b""
    provenance: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    language: str = "en"
    community_id: "UUID | None" = None
    valence: float = 0.0


def _payload_created_at(raw: object) -> "datetime | None":
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if isinstance(raw, str) and raw:
        try:
            parsed = datetime.fromisoformat(raw)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def _read_record_payload(graph, rid: UUID, store: MemoryStore):
    if rid is None:
        node = None
    elif hasattr(graph, "get_payload"):
        node = graph.get_payload(rid) or None
    else:
        node = graph.nodes.get(str(rid)) if hasattr(graph, "nodes") else None
        node = dict(node) if node else None
    if node is not None and "embedding" in node and "surface" in node:
        surface = node.get("surface")
        if surface in (None, "") or node.get("_decrypt_failed"):
            pass
        else:
            return SimpleRecordView(
                id=rid,
                embedding=node["embedding"],
                literal_surface=str(surface),
                centrality=float(node.get("centrality", 0.0) or 0.0),
                tier=str(node.get("tier", "episodic")),
                tags=list(node.get("tags") or []),
                language=str(node.get("language", "en") or "en"),
                aaak_index=str(node.get("aaak_index", "") or ""),
                created_at=_payload_created_at(node.get("created_at")),
                stability=float(node.get("stability", 0.5) or 0.5),
                valence=float(node.get("valence") or 0.0),
            )
    try:
        return store.get(rid)
    except Exception as exc:  # noqa: BLE001 -- retrieval hot-path fail-safe
        logger.debug("read_record_payload_store_fallback_failed rid=%s: %s", rid, exc)
        return None

W_COSINE = 1.0
W_AAAK = 0.3
W_DEGREE = 0.1
W_AGE = 0.05
COS_SPREAD_MIN = 0.02
"""Below this cosine spread across the candidate head, similarity carries no
decision signal and the degree term must not inherit it (dampened
proportionally; the response gets a flat_cosine hint)."""


def _flat_cosine_damp(head_spread: float, threshold: float) -> float:
    """Degree-weight damp factor for a cosine-flat candidate head.

    Proportional ramp: 0.0 at a fully flat head, 1.0 at or above the
    threshold — a fixed cutoff would make the degree term snap back to full
    strength one ulp above it.
    """
    if threshold <= 0.0:
        return 1.0
    return min(1.0, max(0.0, head_spread / threshold))

W_SPREAD_ACT = 0.0
"""Activation transfer: a graph-reached candidate inherits a decayed
fraction of its originating seed's cue-cosine. MUST stay 0.0 in prod:
cross-bench gates fail at every measured weight — an all-edges transfer
floods turn-granularity corpora with seed near-neighbours, and an
entity-gated transfer promotes shared-token distractors over evidence.
Do not enable (IAI_MCP_W_SPREAD_ACT) without fresh cross-bench proof."""

SPREAD_ACT_DECAY = 0.6
"""Per-hop attenuation of the transferred activation."""

TEMPORAL_MATCH_BOOST = 1.15
"""Rank multiplier for records whose creation date matches an explicit
date mention in the cue. Fires ONLY when the cue names a date (ISO or an
unambiguous month form) — non-temporal recall is byte-identical. This is
where temporal binding lives: encoding the date into the stored vector
inflates same-day unrelated-pair cosine past the similarity floors."""


def _env_weight(name: str, default: float) -> float:
    # Bench-only override tier, same convention as the mechanism
    # kill-switches: lets a measurement isolate one ranking term without a
    # rebuild. Absent env means the shipped constant.
    raw = os.environ.get(name, "")
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return default

TIER_KNOWLEDGE_BOOST_DEFAULT = 1.05
"""Bounded soft multiplier for knowledge-grade sources at final rank — a
nudge past equal-scored raw turns, never a filter; 1.0 disables. Both
classes stand down for verbatim mode/intent. doc:* teach chunks boost
regardless of profile (they ARE literal curated content); semantic
summaries boost only when literal_preservation is not "strong" — that knob
is precisely the raw-vs-summary preference and the strong default must
keep outranking condensations."""

SALIENCE_BOOST_STEP_DEFAULT = 0.05
"""Bounded additive-per-level rank step for a caller-declared salience
level -- never a filter. Stands down for verbatim mode/intent, the same as
the tier_boost family. Independent of any tier/doc-tag gate: it applies to
a flagged record regardless of tier, because the flag is a general-purpose
signal, not a knowledge-source marker. `IAI_MCP_SALIENCE_BOOST` overrides."""

PROC_PRIME_SEED_CAP: int = 2
"""Own cap for the priming widening block -- never MULTI_SEED_CAP."""

PROC_PRIME_BOOST_DEFAULT: float = 1.05
"""Bounded nudge multiplier for a primed candidate at final rank;
`IAI_MCP_PROC_PRIME_BOOST` overrides. Stands down for verbatim mode/intent,
mirroring tier_boost."""

_PROC_PRIME_CLAMP_EPS: float = 1e-4


def _salience_boost_step() -> float:
    return _env_weight("IAI_MCP_SALIENCE_BOOST", SALIENCE_BOOST_STEP_DEFAULT)


def _crossing_consolidation_off() -> bool:
    """Kill-switch: force the legacy pre-consolidation recall path."""
    return os.environ.get("IAI_MCP_CROSSING_CONSOLIDATION_OFF") == "1"


def _defer_profile_boost_off() -> bool:
    """Kill-switch: force the legacy synchronous profile_modulates boost."""
    return os.environ.get("IAI_MCP_DEFER_PROFILE_BOOST_OFF") == "1"


def _generational_cache_off() -> bool:
    """Kill-switch: force the unconditional records_cache sweep every recall."""
    return os.environ.get("IAI_MCP_GENERATIONAL_CACHE_OFF") == "1"


def _resolve_use_rust_scorer(explicit: "bool | None", structural_weight: float) -> bool:
    """Kill-switch resolution for the hybrid Rust scorer. `explicit` is the
    caller's own request: the live recall dispatch passes `True`, while
    every other caller (direct pipeline callers, unit tests, CLI/fallback
    entry points that never touch the live dispatch) leaves it unset, so
    the function-level default (`explicit=None` -> the pre-Rust Python
    reference) stays byte-for-byte backward compatible with every existing
    caller. `IAI_MCP_RECALL_RUST_SCORER_OFF` forces the reference path even
    where a caller requested Rust -- the emergency/differential seam.
    `structural_weight > 0.0` also forces the reference path regardless of
    the flag: the structural-blend term has no production knob writer, so a
    caller that deliberately overrides it gets the slower, correct term
    instead of one silently computed against an always-absent vector."""
    if explicit is None or not explicit:
        return False
    if os.environ.get("IAI_MCP_RECALL_RUST_SCORER_OFF") == "1":
        return False
    return structural_weight <= 0.0


def _reinsert_rust_winner_gain(
    partial_score: float, pre_gain_base: float, term_multiplier: float, gain_product: float,
) -> float:
    """Reinserts a per-call multiplicative gain (T8 profile modulation) at
    the exact arithmetic point today's Python formula applies it -- BEFORE
    the stability lift and the trigram/FTS/lex terms already folded into
    `partial_score` -- without a second Rust scoring pass. `partial_score`
    is a Rust `WinnerRow`'s ungained value:
    `partial_score == (pre_gain_base + stability_lift) * term_multiplier + lex_add`.
    Multiplying `gain_product` onto the whole `partial_score` would
    incorrectly scale `stability_lift`/`lex_add` too, which today's formula
    never does; this reconstruction is algebraically equivalent to
    recomputing with `pre_gain_base * gain_product` in place of
    `pre_gain_base`, without needing `stability_lift`/`lex_add` separately."""
    return partial_score + pre_gain_base * term_multiplier * (gain_product - 1.0)


LEX_FUSION_W = 0.35
"""Additive rank-fusion weight for the warm lexical lane: the top BM25 hit
gains this much, decaying by 1/(1+rank). Comparable to W_AAAK — a signal,
never a takeover. `IAI_MCP_LEX_FUSION_W` overrides."""


def _lex_fusion_w() -> float:
    raw = os.environ.get("IAI_MCP_LEX_FUSION_W", "")
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return LEX_FUSION_W

LEX_FUSION_K = 32
LEX_FUSION_MIN_IDF = 4.0
"""Fusion fires only when the cue carries at least one genuinely rare
in-corpus token — common-word cues have no lexical signal worth fusing."""

_IDENTIFIER_CUE_RE = re.compile(
    r'"[^"]{4,}"'                              # quoted exact phrase
    r"|\b(?=\w*[A-Za-z])(?=\w*\d)\w+\b"        # letter+digit mix
    r"|\b\w+_\w+\b"                            # snake_case
    r"|\b[a-z]+[A-Z]\w*\b"                     # camelCase
    r"|\b[A-Z]{5,}\b"                          # long ALL-CAPS name
    r"|\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b"    # dotted.path (letter-led)
    r"|\b[A-Za-z_]\w*/[A-Za-z_]\w\S*"          # path-like
)

# Mid-cue capitalized token (not sentence-initial): a proper-name signal.
_PROPER_NAME_CUE_RE = re.compile(r"(?<=[a-z0-9,;] )[A-Z][a-z]{2,}\b")


def _cue_identifier_grade(cue: str) -> bool:
    """Whether the cue carries an identifier-shaped token (code name, env
    var, path, quoted phrase). The lexical lane is trustworthy exactly
    there; on natural prose, literal-token evidence is anti-correlated
    with the right answer on paraphrase-style cues (measured -1pp), so
    the lane must stay silent for prose no matter how rare the words."""
    if _IDENTIFIER_CUE_RE.search(cue):
        return True
    if os.environ.get("IAI_MCP_LEX_PROPER") == "true":
        return bool(_PROPER_NAME_CUE_RE.search(cue))
    return False

AGE_HALF_LIFE_DAYS = 30.0

LITERAL_PRESERVATION_W_DEGREE_SCALE: dict[str, float] = {
    "strong": 0.3,
    "medium": 1.0,
    "loose":  1.5,
}

K_CANDIDATES: int = 200

COMMUNITY_BIAS_VERBATIM: float = 0.0
COMMUNITY_BIAS_CONCEPT: float = 0.1

_POST_RANK_MAX_HITS: int = 50

#: Over-fetch margin above _POST_RANK_MAX_HITS the hybrid Rust scorer keeps
#: so a per-call-state (Bucket-B) promotion can still land inside the served
#: window after the rank-only Rust pass: the max observed promotion distance
#: across a 57-cue production-shaped sweep of real Bucket-B events (T8/T14-
#: T17), 44 events, p95=14, p99=32, max=32.
RUST_SCORER_K_MARGIN: int = 32

#: Pending-recency markers may claim at most this share of the recall token
#: budget when reclaiming space from ranked hits; past it a marker is dropped,
#: never another ranked hit. Freshness must bias the response, not starve the
#: associative lane (dozens of same-day markers would otherwise evict every
#: ranked hit).
_MARKER_BUDGET_SHARE: float = 0.25


def _build_contradicts_dst_set(
    contradicts_outgoing: dict[str, list[str]] | None,
) -> set[str]:
    if not contradicts_outgoing:
        return set()
    dst_set: set[str] = set()
    for dsts in contradicts_outgoing.values():
        if dsts:
            dst_set.update(str(d) for d in dsts)
    return dst_set


def _gate_bias_for_mode(mode: str) -> float:
    if mode == "concept":
        return _env_weight("IAI_MCP_COMMUNITY_BIAS", COMMUNITY_BIAS_CONCEPT)
    return COMMUNITY_BIAS_VERBATIM


@dataclass
class _RecallCoreResult:

    scored_hits: list[MemoryHit] = field(default_factory=list)
    activation_trace: list[UUID] = field(default_factory=list)
    anti_hits: list[MemoryHit] = field(default_factory=list)
    hints: list[dict] = field(default_factory=list)
    patterns_observed: list[dict] = field(default_factory=list)
    cue_mode: str = "concept"
    budget_used: int = 0
    _records_cache: dict = field(default_factory=dict)
    # Call-local record-id -> profile_modulation_gain, threaded to
    # _apply_post_rank_pipeline instead of a rec field write -- the cached
    # SimpleRecordView objects in graph._records_view_cache must never carry
    # a prior call's gain.
    _profile_gains: dict = field(default_factory=dict)
    # Pre-rank community-gate top-1 + K + backend, carried out for the
    # monotropism_depth signal -- never the post-rank top hit.
    cue_community_id: "str | None" = None
    community_k: "int | None" = None
    community_backend: "str | None" = None
    # Per-call; threaded to RecallResponse.stage_timings -- never sourced
    # from the _last_stage_timings_ms module global.
    stage_timings: dict = field(default_factory=dict)


PROFILE_SENTINEL_UUID = UUID("00000000-0000-0000-0000-0000000000f1")


def _sanitize_vec(v: "np.ndarray") -> "np.ndarray":
    """Return a finite float32 view of v (NaN/inf replaced with 0.0).

    This is a no-op on already-finite vectors (nan_to_num is cheap and
    returns the same values unchanged), so calling it on the common/healthy
    path incurs no correctness cost.
    """
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def _trigram_jaccard(a: str, b: str) -> float:
    if len(a) < 3 or len(b) < 3:
        return 0.0
    set_a = {a[i:i + 3] for i in range(len(a) - 2)}
    set_b = {b[i:i + 3] for i in range(len(b) - 2)}
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return intersection / union


def _cosine(a: list[float], b: list[float]) -> float:
    av = _sanitize_vec(np.asarray(a, dtype=np.float32))
    bv = _sanitize_vec(np.asarray(b, dtype=np.float32))
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


# Tag keys whose VALUES carry content a natural-language cue can name.
# Bookkeeping tags (capture, role:*, idem:*, shield:*, raw:*) must never
# match a cue — a common word like "user" would otherwise bias every
# record of one role.
_AAAK_CONTENT_TAG_KEYS = frozenset({"doc"})


def _aaak_overlap(cue_text: str, aaak_index: str) -> float:
    # Match the cue against the content of the index — entity anchors and
    # doc names — never against machine tokens (field keys, wing letters,
    # hex room ids) or bookkeeping tag values. Entity anchors stay dormant
    # until the capture path writes entity: tags; until then this term is
    # honestly near-zero rather than falsely confident.
    if not aaak_index:
        return 0.0
    from iai_mcp.aaak import parse_aaak_index

    parsed = parse_aaak_index(aaak_index)
    meaningful: set[str] = set()
    for ent in parsed["entities"]:
        meaningful.update(ent.lower().split())
    for tag in parsed["tags"]:
        key, sep, value = tag.partition(":")
        if sep and key.lower() in _AAAK_CONTENT_TAG_KEYS and value.strip():
            meaningful.add(value.lower().strip())
    # Cue tokens get the same edge-punctuation strip as anchors do at
    # extraction — "zephyrbot?" must match the anchor "zephyrbot".
    cue_set = {
        tok
        for raw in cue_text.lower().replace("/", " ").split()
        if (tok := raw.strip(" \t\r\n.,;:!?()[]{}\"'«»"))
    }
    if not cue_set or not meaningful:
        return 0.0
    # Cue-normalized containment: a record with many anchors must not be
    # penalized for having them — only the cue's coverage matters. A cue
    # token also matches an anchor when one is a prefix of the other with
    # at most 3 trailing characters of difference and a stem of at least
    # 5 — inflected forms of the same name (Russian case endings, English
    # plurals) must not defeat an exact-name lane.
    matched = 0
    for tok in cue_set:
        if tok in meaningful:
            matched += 1
            continue
        for anchor in meaningful:
            short, long_ = (tok, anchor) if len(tok) <= len(anchor) else (anchor, tok)
            if len(short) >= 5 and len(long_) - len(short) <= 3 and long_.startswith(short):
                matched += 1
                break
    return matched / len(cue_set)


def _has_doc_tag(rec) -> bool:
    return any(
        isinstance(t, str) and t.startswith("doc:")
        for t in (getattr(rec, "tags", None) or ())
    )


def _tier_knowledge_boost() -> float:
    try:
        return float(
            os.environ.get("IAI_MCP_TIER_BOOST", "") or TIER_KNOWLEDGE_BOOST_DEFAULT
        )
    except ValueError:
        return TIER_KNOWLEDGE_BOOST_DEFAULT


def _age_penalty(created_at: "datetime | None") -> float:
    if created_at is None:
        return 0.0
    now = datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    days = (now - created_at).total_seconds() / 86400.0
    if days < 0:
        return 0.0
    return min(1.0, days / AGE_HALF_LIFE_DAYS)


def _community_gate(
    cue_emb: list[float],
    assignment: CommunityAssignment,
    top_n: int = 3,
    member_embeddings: dict[UUID, list[float]] | None = None,
) -> list[UUID]:
    gated, _community_scores = _community_gate_scored(
        cue_emb, assignment, top_n, member_embeddings,
    )
    return gated


def _community_gate_scored(
    cue_emb: list[float],
    assignment: CommunityAssignment,
    top_n: int = 3,
    member_embeddings: dict[UUID, list[float]] | None = None,
) -> tuple[list[UUID], dict[UUID, float]]:
    """Same ranking as ``_community_gate``, also returning the continuous
    per-community cue-centroid score for the graded soft-gate bonus."""
    cue_vec = _sanitize_vec(np.asarray(cue_emb, dtype=np.float32))
    cue_norm = float(np.linalg.norm(cue_vec))
    if cue_norm > 0.0:
        cue_vec = cue_vec / cue_norm

    if member_embeddings is not None:
        return _community_gate_max_node_scored(
            cue_vec, assignment, top_n, member_embeddings,
        )

    centroids = assignment.community_centroids
    if not centroids:
        return [], {}
    cids = list(centroids.keys())
    mat = np.asarray(
        [centroids[c] for c in cids], dtype=np.float32
    )
    mat = _sanitize_vec(mat)
    norms = np.linalg.norm(mat, axis=1)
    norms[norms == 0.0] = 1.0
    mat = mat / norms[:, None]
    scores = mat @ cue_vec
    order = np.argsort(-scores, kind="stable")
    community_scores = {cids[int(i)]: float(scores[int(i)]) for i in range(len(cids))}
    return [cids[int(i)] for i in order[:top_n]], community_scores


def _community_gate_max_node(
    cue_vec: np.ndarray,
    assignment: CommunityAssignment,
    top_n: int,
    member_embeddings: dict[UUID, list[float] | np.ndarray],
) -> list[UUID]:
    gated, _community_scores = _community_gate_max_node_scored(
        cue_vec, assignment, top_n, member_embeddings,
    )
    return gated


def _community_gate_max_node_scored(
    cue_vec: np.ndarray,
    assignment: CommunityAssignment,
    top_n: int,
    member_embeddings: dict[UUID, list[float] | np.ndarray],
) -> tuple[list[UUID], dict[UUID, float]]:
    """Same ranking as ``_community_gate_max_node``, also returning the
    continuous per-community max-member score (``comm_max``) for the graded
    soft-gate bonus. No new corpus pass — reuses the scores already computed
    for the top-n slice."""
    mid_regions = assignment.mid_regions
    if not mid_regions:
        return _community_gate_scored(
            cue_vec.tolist(), assignment, top_n, member_embeddings=None,
        )

    cids: list[UUID] = []
    rows: list[np.ndarray] = []
    breaks: list[int] = []
    total = 0
    for cid, members in mid_regions.items():
        valid: list[np.ndarray] = []
        for m in members:
            emb = member_embeddings.get(m)
            if emb is None:
                continue
            if not isinstance(emb, np.ndarray):
                emb = np.asarray(emb, dtype=np.float32)
            valid.append(emb)
        if not valid:
            continue
        cids.append(cid)
        breaks.append(total)
        total += len(valid)
        rows.extend(valid)

    if not rows:
        return [], {}

    mat = np.stack(rows).astype(np.float32, copy=False)
    # The gate ranks members by cosine against the cue. On the recall path the
    # member vectors are already the once-sanitized + L2-normalized pool rows, so
    # the full nan_to_num + per-row renorm is repeated waste. Keep only a cheap
    # finite-check: sanitize + renorm once when a non-finite value sneaks in,
    # otherwise use the already-normalized rows directly. Cosine scores are
    # unchanged because the rows are unit vectors.
    if not np.isfinite(mat).all():
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(mat, axis=1)
        norms[norms == 0.0] = 1.0
        mat = mat / norms[:, None]
    member_scores = mat @ cue_vec

    comm_max = np.maximum.reduceat(member_scores, breaks)

    str_order = sorted(range(len(cids)), key=lambda i: str(cids[i]))
    lex_sorted_cids = [cids[i] for i in str_order]
    lex_sorted_scores = comm_max[str_order]
    score_order = np.argsort(-lex_sorted_scores, kind="stable")
    community_scores = {
        lex_sorted_cids[i]: float(lex_sorted_scores[i]) for i in range(len(lex_sorted_cids))
    }
    return [lex_sorted_cids[int(i)] for i in score_order[:top_n]], community_scores


def _pick_seeds(
    candidate_indices: np.ndarray,
    shared_cos: np.ndarray,
    centrality_arr: np.ndarray,
    n: int = 3,
) -> np.ndarray:
    if candidate_indices.size == 0:
        return np.empty(0, dtype=candidate_indices.dtype)
    # Per-cycle percentile normalization of the centrality term over the
    # candidate slice. The centrality map may be an exact betweenness or a
    # bounded approximation whose raw magnitude drifts with the pivot count and
    # corpus size; the 0.6/0.4 blend is scale-sensitive, so feeding the rank
    # rather than the raw value keeps the seed boundary set by topology, not by
    # the magnitude the centrality variant happened to produce. The relative
    # ordering -- all the seed blend needs from centrality -- is preserved.
    from iai_mcp.centrality_approx import _percentile_normalize

    cand_centrality = centrality_arr[candidate_indices].astype(np.float64)
    cand_centrality_normed = _percentile_normalize(cand_centrality)
    blended = (
        0.6 * shared_cos[candidate_indices]
        + 0.4 * cand_centrality_normed
    )
    top_local = np.argsort(-blended, kind="stable")[:n]
    return candidate_indices[top_local]


def _collect_graph_pool(
    graph: MemoryGraph,
    records_cache: dict[UUID, "object"] | None,
    store: MemoryStore,
) -> tuple[list[UUID], np.ndarray]:
    # The pool (id sequence + embedding matrix) is recomputed every call from
    # the graph's own node embeddings -- a vectorized numpy build, not a
    # per-record Python object construction, so re-running it every recall
    # carries no material cost. No cross-call memoization on the graph.
    pool_ids: list[UUID] = []
    pool_embs_rows: list["np.ndarray | list[float]"] = []

    # The graph may be the process-wide warm bundle, mutated live by the
    # store's graph_sync_hook on a concurrent write; a dict resize mid-iteration
    # raises RuntimeError. One retry re-snapshots — the second pass sees the
    # post-mutation dict and the bumped _pool_content_version keys the cache.
    try:
        node_ids = list(graph.iter_nodes())
    except RuntimeError:
        node_ids = list(graph.iter_nodes())

    # Resolve which nodes lack an embedding on the graph node and in the cache;
    # those (and only those) fall through to the store, batched in one fetch.
    def _node_emb(rid: UUID) -> "np.ndarray | list[float] | None":
        node_emb = graph.get_embedding(rid)
        if node_emb is not None:
            return node_emb
        if records_cache is not None and rid in records_cache:
            cached_emb = getattr(records_cache[rid], "embedding", None)
            if cached_emb is not None and len(cached_emb):
                return cached_emb
        return None

    _fallback_ids = [rid for rid in node_ids if _node_emb(rid) is None]
    _fallback_batch: dict = {}
    if _fallback_ids:
        try:
            _fallback_batch = store.get_batch(_fallback_ids)
        except Exception as exc:  # noqa: BLE001 -- retrieval hot-path fail-safe
            logger.debug("collect_graph_pool_store_fallback_failed: %s", exc)
            _fallback_batch = {}

    for rid in node_ids:
        emb = _node_emb(rid)
        rec = None
        if emb is None or not len(emb):
            emb = None
            rec = _fallback_batch.get(rid)
            rec_emb = getattr(rec, "embedding", None) if rec is not None else None
            if rec_emb is not None and len(rec_emb):
                emb = rec_emb
        if emb is not None:
            pool_ids.append(rid)
            pool_embs_rows.append(emb)
            # The store fallback above already paid for a full record decode
            # -- stash it into records_cache so a pool member the graph
            # payload build skipped (embedding_pending there) still resolves
            # at scoring time instead of hitting the `rec is None: continue`
            # drop. Cheap-path only; the caller re-checks coverage for any
            # node this cannot reach (no embedding here at all, or no store
            # fallback attempted because the embedding already resolved from
            # the graph/cache).
            if rec is not None and records_cache is not None and rid not in records_cache:
                records_cache[rid] = rec
    if not pool_ids:
        return [], np.zeros((0, store.embed_dim), dtype=np.float32)
    pool_embs = np.asarray(pool_embs_rows, dtype=np.float32)
    return pool_ids, pool_embs


def _normalize_pool(
    graph: MemoryGraph,
    pool_ids: list[UUID],
    pool_embs: np.ndarray,
) -> np.ndarray:
    """Return the sanitized + L2-normalized pool matrix.

    Recomputed every call -- a vectorized numpy nan_to_num + norm + divide
    over the whole N x D matrix, cheap enough to re-run per recall. No
    cross-call memoization on the graph.
    """
    del graph, pool_ids
    if not np.isfinite(pool_embs).all():
        pool_embs = np.nan_to_num(pool_embs, nan=0.0, posinf=0.0, neginf=0.0)
    pool_norms = np.linalg.norm(pool_embs, axis=1)
    pool_norms[pool_norms == 0.0] = 1.0
    normalized = pool_embs / pool_norms[:, None]
    return normalized


def _t11_t12_flags(
    pool_ids: list[UUID],
    reachable_indices: "np.ndarray",
    records_cache: dict[UUID, "object"],
    fts_hits: "set[UUID]",
    cue: str,
) -> tuple[np.ndarray, np.ndarray]:
    """T11 (trigram-jaccard>0.3, x2.0) and T12 (fts substring, x3.0) as
    boolean arrays parallel to `pool_ids`, computed from `records_cache` --
    the same surfaces v16 reads at `fts_hits` construction (T12) and the
    per-candidate trigram gate (T11). T11 is batched into one call to the
    Rust `trigram_t11_flags` helper (call-scoped, never resident -- see its
    docstring) over every present-and-nonempty-surface candidate; this
    process still holds `_trigram_jaccard` as the byte-identical reference
    for tests and the non-Rust-scorer path. A pool_id absent from
    `records_cache` gets `False` for both: v16's own scoring loop drops any
    candidate absent from records_cache entirely (`rec =
    records_cache.get(cid); if rec is None: continue`), so False here is
    the byte-identical read, not a gap -- see
    `test_reachable_covered_by_records_cache`/`..._under_store_fallback`
    for the (rare, escalated) divergence this can never silently widen
    past. `reachable_indices` must be the pre-verbatim-filter union: Rust's
    own `reachable` narrows under `verbatim_filter` by its own resident tier
    column, a source independent from `records_cache`'s tier, so passing the
    unfiltered union is the only way to guarantee every position Rust's
    filter can retain has a populated flag. Call-local arrays, never
    written back onto `records_cache`.
    """
    n_pool = len(pool_ids)
    t11 = np.zeros(n_pool, dtype=bool)
    t12 = np.zeros(n_pool, dtype=bool)
    cue_nonempty = bool(cue)
    cue_lower = cue.lower() if cue_nonempty else ""
    present_positions: list[int] = []
    present_surfaces_lower: list[str] = []
    for idx in reachable_indices:
        i = int(idx)
        cid = pool_ids[i]
        rec = records_cache.get(cid)
        if rec is None:
            continue
        # fts_hits also drives seed widening above -- scoring must read the
        # exact same set, never an independently recomputed one.
        t12[i] = cid in fts_hits
        if cue_nonempty:
            surface = getattr(rec, "literal_surface", "") or ""
            if surface:
                present_positions.append(i)
                present_surfaces_lower.append(surface.lower())
    if present_positions:
        from iai_mcp_native import rank as _rank_native

        flags = _rank_native.trigram_t11_flags(cue_lower, present_surfaces_lower)
        for pos, flag in zip(present_positions, flags):
            t11[pos] = flag
    return t11, t12


def _log_malformed_anti_edges(store: MemoryStore, hit_ids: "list[UUID]") -> None:
    try:
        str_ids = [str(i) for i in hit_ids]
        ph = ", ".join("?" for _ in str_ids)
        sql = (  # nosemgrep: sql-injection
            f"SELECT src, dst FROM edges"  # noqa: S608
            f" WHERE (src IN ({ph}) OR dst IN ({ph}))"
            f" AND edge_type = 'contradicts'"
        )
        params: list = str_ids + str_ids
        with store.db.ro_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        for row in rows:
            src_s = str(row[0])
            dst_s = str(row[1])
            for val, label in ((src_s, "src"), (dst_s, "dst")):
                try:
                    UUID(val)
                except (ValueError, AttributeError):
                    logger.warning(
                        "anti_hits_skip_malformed_edge %s=%s",
                        label, val,
                    )
    except Exception:  # noqa: BLE001 -- observability is best-effort
        pass


def _contradicts_edges_with_malformed_warning(
    store: MemoryStore, hit_ids: "list[UUID]",
) -> "dict[UUID, list[tuple[UUID, str, float]]]":
    """Mirrors `incident_edges(hit_ids, edge_types=["contradicts"], top_k=None)`
    (store/_store.py:3063-3156) exactly so the two never diverge silently --
    NOT a general `incident_edges` replacement, scoped to this call shape.
    """
    str_ids = [str(i) for i in hit_ids]
    id_set = set(str_ids)
    ph = ", ".join("?" for _ in str_ids)
    sql = (  # nosemgrep: sql-injection
        f"SELECT src, dst, edge_type, weight FROM edges"  # noqa: S608
        f" WHERE (src IN ({ph}) OR dst IN ({ph}))"
        f" AND edge_type = 'contradicts'"
    )
    params: list = str_ids + str_ids
    with store.db.ro_conn() as conn:
        rows = conn.execute(sql, params).fetchall()

    id_to_uuid: dict[str, UUID] = {str(i): i for i in hit_ids}
    result: dict = {i: [] for i in hit_ids}
    for row in rows:
        src_s = str(row[0] if hasattr(row, "__getitem__") else row["src"])
        dst_s = str(row[1] if hasattr(row, "__getitem__") else row["dst"])
        et = str(row[2] if hasattr(row, "__getitem__") else row["edge_type"])
        wt = float(row[3] if hasattr(row, "__getitem__") else row["weight"])

        for val, label in ((src_s, "src"), (dst_s, "dst")):
            try:
                UUID(val)
            except (ValueError, AttributeError):
                logger.warning(
                    "anti_hits_skip_malformed_edge %s=%s",
                    label, val,
                )

        if src_s in id_set:
            qid = id_to_uuid[src_s]
            try:
                neighbour = UUID(dst_s)
            except (ValueError, AttributeError):
                continue
            result[qid].append((neighbour, et, wt))

        if dst_s in id_set and dst_s != src_s:
            qid = id_to_uuid[dst_s]
            try:
                neighbour = UUID(src_s)
            except (ValueError, AttributeError):
                continue
            result[qid].append((neighbour, et, wt))

    return result


def _find_anti_hits(
    hits: list[MemoryHit],
    store: MemoryStore,
    graph: MemoryGraph,
    k: int = 3,
    records_cache: dict[UUID, "object"] | None = None,
) -> list[MemoryHit]:
    seen: set[UUID] = {h.record_id for h in hits}
    anti_ids: list[UUID] = []

    hit_ids = [h.record_id for h in hits]
    if not hit_ids:
        return []

    if _crossing_consolidation_off():
        _log_malformed_anti_edges(store, hit_ids)
        try:
            _contr_map = store.incident_edges(
                hit_ids, edge_types=["contradicts"], top_k=None,
            )
        except Exception as exc:  # noqa: BLE001 -- anti-hits is enrichment; degrade to []
            logger.debug("_find_anti_hits incident_edges failed: %s", exc)
            return []
    else:
        # Default path: one query derives the contradicts neighbour map AND
        # the malformed-endpoint warning from the same fetched rows, instead
        # of a second ro_conn() acquisition for _log_malformed_anti_edges.
        try:
            _contr_map = _contradicts_edges_with_malformed_warning(store, hit_ids)
        except Exception as exc:  # noqa: BLE001 -- anti-hits is enrichment; degrade to []
            logger.debug("_find_anti_hits contradicts_fetch_failed: %s", exc)
            return []

    for h in hits:
        for (_nbr, _et, _wt) in _contr_map.get(h.record_id, []):
            if _nbr in seen:
                continue
            anti_ids.append(_nbr)
            seen.add(_nbr)
            if len(anti_ids) >= k:
                break
        if len(anti_ids) >= k:
            break

    out: list[MemoryHit] = []
    _anti_slice = anti_ids[:k]
    _missing_anti = [
        aid for aid in _anti_slice
        if not (records_cache is not None and aid in records_cache)
    ]
    _anti_batch: dict = store.get_batch(_missing_anti) if _missing_anti else {}
    for aid in _anti_slice:
        rec = records_cache.get(aid) if records_cache is not None else None
        if rec is None:
            rec = _anti_batch.get(aid)
        if rec is None:
            continue
        _prov = (rec.provenance or [{}])[0]
        out.append(
            MemoryHit(
                record_id=aid,
                score=0.0,
                reason="contradicts-edge neighbour",
                literal_surface=rec.literal_surface,
                adjacent_suggestions=[],
                session_id=_prov.get("session_id"),
                captured_at=rec.created_at.isoformat() if rec.created_at else None,
                epistemic_status=getattr(rec, "epistemic_status", None),
                salience_level=getattr(rec, "salience_level", None),
            )
        )
    return out


def _backfill_hit_metadata(
    hits: list[MemoryHit],
    anti_hits: list[MemoryHit],
    store: MemoryStore,
) -> None:
    """Fill epistemic_status, salience_level, session_id, and captured_at
    on any hit or anti-hit still carrying None for one of those fields.
    Every recall entry point MUST call this before returning.
    """
    _all = [*hits, *anti_hits]
    _missing_ids = list({
        h.record_id for h in _all
        if h.epistemic_status is None or h.salience_level is None
        or h.session_id is None or h.captured_at is None
    })
    if not _missing_ids:
        return
    try:
        _full_batch = store.get_batch(_missing_ids)
    except Exception as exc:  # noqa: BLE001 -- additive enrichment, never crash recall
        logger.debug("epistemic_status_backfill_failed: %s", exc)
        return
    for _h in _all:
        _full = _full_batch.get(_h.record_id)
        if _full is None:
            continue
        if _h.epistemic_status is None:
            _h.epistemic_status = getattr(_full, "epistemic_status", None)
        if _h.salience_level is None:
            _h.salience_level = getattr(_full, "salience_level", None)
        if _h.session_id is None:
            _full_prov = (getattr(_full, "provenance", None) or [{}])[0]
            _h.session_id = _full_prov.get("session_id")
        if _h.captured_at is None:
            _full_created = getattr(_full, "created_at", None)
            _h.captured_at = _full_created.isoformat() if _full_created else None


_last_recall_latency_ms: float = 0.0
_last_stage_timings_ms: dict[str, float] = {}
"""Opt-in per-stage recall timings, populated only when IAI_MCP_STAGE_PROFILE=1.

Mirrors the _last_recall_latency_ms pattern: written once at _recall_core exit,
read by callers after the recall_for_response() call returns.
"""


MULTI_SEED_CAP: int = 6
"""Hard ceiling on the total seed count (base + fts_hits union) after the
unconditional multi-seed widen. Seed count directly multiplies 2-hop spread
cost, so this cap must never be relaxed without re-measuring spread latency.
"""


_VERBATIM_FILTER_DEBUG: dict | None = None


def _recall_core(
    store: MemoryStore,
    graph: MemoryGraph,
    assignment: CommunityAssignment,
    rich_club: list[UUID],
    embedder: Embedder,
    cue: str,
    session_id: str,
    profile_state: dict | None = None,
    turn: int = 0,
    mode: str = "concept",
    *,
    knobs_applied: dict | None = None,
    k_communities: int = 3,
    spread_hops: int = 2,
    cue_intent: str | None = None,
    contradicts_outgoing: dict[str, list[str]] | None = None,
    trace_mark: Callable[[str], None] | None = None,
    cue_embedding: "list[float] | None" = None,
    hydrate_stage_timings: dict | None = None,
    use_rust_scorer: bool | None = None,
    retrieval_weights: dict[str, float] | None = None,
) -> _RecallCoreResult:
    profile_state = profile_state or {}
    _stage_profile_on = os.environ.get("IAI_MCP_STAGE_PROFILE") == "1"
    _stage_timings: dict[str, float] = {}

    try:
        from iai_mcp import gate as _gate_mod
        _skip_fn = _gate_mod.should_skip_retrieval
        skip_flag, skip_reason = _skip_fn(cue)
    except Exception as exc:  # noqa: BLE001 -- retrieval hot-path fail-safe
        logger.debug("active_inference_gate_failed: %s", exc)
        skip_flag, skip_reason = False, ""
    if skip_flag:
        l0_uuid = UUID("00000000-0000-0000-0000-000000000001")
        l0_rec = store.get(l0_uuid)
        if l0_rec is not None:
            budget_used_l0 = len(l0_rec.literal_surface) // 4
            _l0_prov = (l0_rec.provenance or [{}])[0]
            l0_hit = MemoryHit(
                record_id=l0_rec.id,
                score=1.0,
                reason="L0 identity (always skipped)",
                literal_surface=l0_rec.literal_surface,
                adjacent_suggestions=[],
                session_id=_l0_prov.get("session_id"),
                captured_at=l0_rec.created_at.isoformat() if l0_rec.created_at else None,
                community_id=getattr(l0_rec, "community_id", None),
                epistemic_status=l0_rec.epistemic_status,
                salience_level=l0_rec.salience_level,
            )
            try:
                # _l0_prov above already read from l0_rec.provenance before
                # this append -- the returned hit is fixed either way, so
                # suppressing this write changes zero bytes of the response.
                if not recall_suppressed.get():
                    store.append_provenance(
                        l0_rec.id,
                        {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "cue": cue,
                            "session_id": session_id,
                        },
                    )
            except Exception as exc:  # noqa: BLE001 -- retrieval hot-path fail-safe
                logger.debug("l0_provenance_append_failed: %s", exc)
            try:
                write_event(
                    store,
                    kind="retrieval_used",
                    data={
                        "hit_ids": [str(l0_rec.id)],
                        "query": cue,
                        "used": True,
                        "budget_used": budget_used_l0,
                        "path": "recall_core_l0_fastpath",
                    },
                    severity="info",
                    session_id=session_id,
                )
            except Exception as exc:  # noqa: BLE001 -- retrieval hot-path fail-safe
                logger.debug("l0_retrieval_used_event_failed: %s", exc)
            return _RecallCoreResult(
                scored_hits=[l0_hit],
                activation_trace=[l0_rec.id],
                anti_hits=[],
                hints=[{
                    "kind": "retrieval_skipped",
                    "severity": "info",
                    "source_ids": [],
                    "text": skip_reason,
                }],
                patterns_observed=[],
                cue_mode=mode,
                budget_used=budget_used_l0,
            )

    # Tool-schema contract: the cue is embedded server-side UNLESS the caller
    # supplied a usable cue_embedding (validated: finite, per-store dim,
    # nonzero norm -- see _valid_cue_vec).
    _embed_t0 = time.perf_counter() if _stage_profile_on else 0.0
    cue_emb = _valid_cue_vec(
        cue_embedding, getattr(store, "embed_dim", None) or EMBED_DIM,
    )
    if cue_emb is None:
        if cue_embedding is not None:
            logger.debug("cue_embedding_rejected: server-side embed used")
        try:
            cue_emb = embed_query(embedder, cue)
        except Exception as exc:
            write_event(
                store,
                TELEMETRY_EMBED_NATIVE_FAILURE,
                {
                    "op_type": "recall_cue",
                    "backend": "rust",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise NativeError(f"recall cue encode failed: {exc}") from exc
    if _stage_profile_on:
        _stage_timings["embed"] = (time.perf_counter() - _embed_t0) * 1000.0

    _pool_build_t0 = time.perf_counter() if _stage_profile_on else 0.0
    # Generational cache: keyed ONLY on graph._pool_content_version (no new
    # counter), mirroring _collect_graph_pool. Captured ONCE here and never
    # re-read at commit time below -- a mutation landing mid-sweep must key
    # the torn view under the OLD stamp so the next live-version comparison
    # misses and rebuilds; the torn view is never served.
    _records_view_version = getattr(graph, "_pool_content_version", None)
    _cached_records_view = getattr(graph, "_records_view_cache", None)
    if (
        not _generational_cache_off()
        and _records_view_version is not None
        and _cached_records_view is not None
        and _cached_records_view[0] == _records_view_version
    ):
        _records_base: dict[UUID, "object"] = _cached_records_view[1]
    else:
        _records_base = {}
        try:
            try:
                node_ids = list(graph.iter_nodes())
            except RuntimeError:
                # The warm bundle can be mutated concurrently by the store's
                # graph_sync_hook; a dict resize mid-iteration raises. One
                # retry re-snapshots -- the second pass sees the
                # post-mutation node set under the already-bumped version.
                node_ids = list(graph.iter_nodes())
            for rid in node_ids:
                node = graph.get_payload(rid)
                if "embedding" not in node or "surface" not in node:
                    continue
                _records_base[rid] = SimpleRecordView(
                    id=rid,
                    embedding=node["embedding"],
                    literal_surface=str(node.get("surface", "")),
                    centrality=float(node.get("centrality", 0.0) or 0.0),
                    tier=str(node.get("tier", "episodic")),
                    tags=list(node.get("tags") or []),
                    language=str(node.get("language", "en") or "en"),
                    aaak_index=str(node.get("aaak_index", "") or ""),
                    created_at=_payload_created_at(node.get("created_at")),
                    stability=float(node.get("stability", 0.5) or 0.5),
                    valence=float(node.get("valence") or 0.0),
                )
        except Exception as exc:  # noqa: BLE001 -- retrieval hot-path fail-safe
            logger.debug("records_cache_graph_build_failed: %s", exc)
            _records_base = {}
        # Cache only when fully resolved from the graph (never the store
        # fallback below), mirroring _collect_graph_pool's discipline. The
        # cached object is the pristine just-built base -- it is committed
        # BEFORE the copy-on-serve step below, so no per-call write further
        # down can ever reach it, even within this same call.
        if (
            _records_base
            and _records_view_version is not None
            and not _generational_cache_off()
        ):
            try:
                graph._records_view_cache = (_records_view_version, _records_base)
            except AttributeError:
                pass
    if _stage_profile_on:
        _stage_timings["pool"] = (time.perf_counter() - _pool_build_t0) * 1000.0
    if not _records_base:
        # Bounded fallback: an empty candidate graph must not trigger a
        # full-corpus materialization on the recall path. 1024 recent live
        # records give the ranking a working pool; the primary path never
        # lands here. Never cached (store-sourced, not graph-resolved).
        import itertools as _it

        records_cache = {
            r.id: r
            for r in _it.islice(
                store.iter_records(where="tombstoned_at IS NULL"), 1024,
            )
        }
    else:
        # Copy-on-serve, unconditionally -- whether this call built the base
        # fresh or hit the cache from a prior call. The pool-gap backfill
        # further below (records_cache[rid] = rec) writes into this per-call
        # shallow copy; the object stored in graph._records_view_cache is
        # never mutated by any call. This is a dict-key copy only --
        # SimpleRecordView is a mutable dataclass, so per-call values
        # (community_id, profile_modulation_gain) are tracked in call-local
        # dicts below and never written onto the shared record objects.
        records_cache: dict[UUID, "object"] = dict(_records_base)

    _pool_t0 = time.perf_counter()
    pool_ids, pool_embs = _collect_graph_pool(graph, records_cache, store)
    _recall_pool_collection_ms = (time.perf_counter() - _pool_t0) * 1000.0
    if _stage_profile_on:
        # Reuses the pre-existing unconditional _pool_t0 timer above (already
        # runs regardless of the flag for the recall_timing telemetry sample)
        # -- adds a dict write only, zero incremental perf_counter() calls.
        _stage_timings["pool_collection"] = _recall_pool_collection_ms

    # Guarantee the invariant _collect_graph_pool's own cheap-path stash
    # cannot always reach on its own (e.g. a node with a graph-resolvable
    # embedding but no "surface" payload key never enters its fallback
    # branch): resolve any remaining pool member records_cache still lacks,
    # in one batched fetch, before episodic_ids/fts_hits read records_cache
    # below -- so both see the backfilled surface too. A pool member the
    # store itself cannot resolve (never inserted) stays absent from
    # records_cache; the winners loop already tolerates that gracefully.
    #
    # Capped at the same ceiling as the empty-pool fallback above: the pool
    # itself is uncapped by design, so an unbounded gap here could turn into
    # a full-corpus decrypt if a caller-supplied graph has surface-less
    # nodes throughout. decode="full" (not "rank") is deliberate, matching
    # _collect_graph_pool's own store-fallback: RankCandidateView still lacks
    # `epistemic_status` (served on the hit), so a backfilled record must
    # resolve through the same decode tier as every other records_cache
    # entry these consumers share.
    _pool_records_gap_ids = [
        rid for rid in pool_ids if rid not in records_cache
    ][:1024]
    if _pool_records_gap_ids:
        try:
            _pool_records_gap_batch = store.get_batch(_pool_records_gap_ids)
        except Exception as exc:  # noqa: BLE001 -- best-effort backfill, never blocks recall
            logger.debug("pool_records_cache_gap_backfill_failed: %s", exc)
            _pool_records_gap_batch = {}
        for rid, rec in _pool_records_gap_batch.items():
            if rid not in records_cache:
                records_cache[rid] = rec

    # The age term needs an authoritative created_at; a graph payload
    # written without that key leaves the cached view's created_at None.
    # Recover it from the store in one batched fetch -- same cap/never-block
    # discipline as the gap backfill above. Never mutate the shared cached
    # object: a fresh SimpleRecordView replaces only this per-call dict
    # entry, so graph._records_view_cache stays pristine for the next call.
    _missing_created_at_ids = [
        rid for rid in pool_ids
        if rid in records_cache and getattr(records_cache[rid], "created_at", None) is None
    ][:1024]
    if _missing_created_at_ids:
        try:
            _created_at_batch = store.get_batch(_missing_created_at_ids)
        except Exception as exc:  # noqa: BLE001 -- best-effort backfill, never blocks recall
            logger.debug("records_cache_created_at_backfill_failed: %s", exc)
            _created_at_batch = {}
        for rid, _full in _created_at_batch.items():
            _full_created_at = getattr(_full, "created_at", None)
            if _full_created_at is None:
                continue
            try:
                records_cache[rid] = replace(
                    records_cache[rid], created_at=_full_created_at
                )
            except TypeError as exc:  # noqa: BLE001 -- best-effort backfill, never blocks recall
                logger.debug("records_cache_created_at_replace_failed rid=%s: %s", rid, exc)

    episodic_ids: set | None = None
    if mode == "verbatim":
        episodic_ids = {
            cid for cid, rec in records_cache.items()
            if getattr(rec, "tier", "episodic") == "episodic"
        }

    # Computed here so it is available to the ranking boost later in the
    # function.
    fts_hits: set[UUID] = set()
    if cue and len(cue) >= 4:
        cue_lower = cue.lower()
        for rid, rec in records_cache.items():
            if rec.literal_surface and cue_lower in rec.literal_surface.lower():
                fts_hits.add(rid)

    cue_vec = _sanitize_vec(np.asarray(cue_emb, dtype=np.float32))
    cnorm = float(np.linalg.norm(cue_vec))
    if cnorm > 0.0:
        cue_vec = cue_vec / cnorm
    if pool_embs.size:
        pool_embs = _normalize_pool(graph, pool_ids, pool_embs)
        shared_cos = np.matmul(pool_embs, cue_vec).astype(np.float32)
    else:
        shared_cos = np.empty(0, dtype=np.float32)
    if shared_cos.size:
        shared_order = np.argsort(-shared_cos, kind="stable")
        cosine_top_indices = shared_order
    else:
        shared_order = np.empty(0, dtype=np.int64)
        cosine_top_indices = np.empty(0, dtype=np.int64)

    # Warm BM25 lane: rank map for the fusion bonus + candidate inclusion.
    # Fires only for identifier-grade cues — the lane's designed competence.
    # Answers only from an already-built, generation-current index — a cold
    # index yields nothing here and warms in the background instead.
    lex_rank: dict[UUID, int] = {}
    if (
        cue
        and os.environ.get("IAI_MCP_LEX_FUSION_OFF") != "true"
        and _cue_identifier_grade(cue)
    ):
        try:
            _min_idf = float(
                os.environ.get("IAI_MCP_LEX_MIN_IDF", "") or LEX_FUSION_MIN_IDF
            )
        except ValueError:
            _min_idf = LEX_FUSION_MIN_IDF
        try:
            for _lrank, (_lrid, _lscore) in enumerate(
                store.lexical_query_warm(cue, k=LEX_FUSION_K, min_idf=_min_idf)
            ):
                try:
                    lex_rank[UUID(str(_lrid))] = _lrank
                except (TypeError, ValueError):
                    continue
        except Exception as exc:  # noqa: BLE001 -- the lexical lane is a bonus, never a dependency
            logger.debug("lexical fusion lane skipped: %s", exc)
    _arousal_cue_hash_bytes = hashlib.md5(str(cue).encode("utf-8")).digest()
    _arousal_cue_hash_hex = _arousal_cue_hash_bytes[:4].hex()
    if os.environ.get("IAI_MCP_AROUSAL_USE_SHADOW") == "1":
        _arousal_route = "arousal_shadow"
    else:
        _arousal_route = "arousal_real" if (_arousal_cue_hash_bytes[0] & 1) else "arousal_shadow"

    _arousal_level_for_telemetry: float = 0.5
    _arousal_mode_for_telemetry: str | None = None
    _arousal_max_hops_used: int = spread_hops
    _arousal_rank_threshold_used: float = 0.0
    _arousal_mode_bias_adjust: float = 0.0
    _arousal_budget_for_telemetry: int = 1500

    if _arousal_route == "arousal_real":
        try:
            from iai_mcp.arousal_budget import (
                ArousalState as _ArousalState,
                compute_retrieval_params as _compute_retrieval_params,
            )
            _arousal_state_local = _ArousalState()
            _arousal_params = _compute_retrieval_params(_arousal_state_local)
            _arousal_level_for_telemetry = float(_arousal_state_local.level)
            _arousal_mode_for_telemetry = _arousal_params.mode
            _arousal_budget_for_telemetry = int(_arousal_params.budget_tokens)
            _arousal_rank_threshold_used = float(_arousal_params.rank_threshold)
            _arousal_max_hops_used = int(min(int(_arousal_params.max_hops), spread_hops))
            spread_hops = _arousal_max_hops_used
            _amode = _arousal_params.mode
            if _amode == "monotropic_tunnel":
                _arousal_mode_bias_adjust = -0.05
            elif _amode == "associative_dream":
                _arousal_mode_bias_adjust = +0.05
            else:
                _arousal_mode_bias_adjust = 0.0
        except Exception as exc:  # noqa: BLE001 -- arousal hot-path fail-safe
            logger.debug("arousal_budget_real_route_failed: %s", exc)
            _arousal_route = "arousal_skip"
            _arousal_rank_threshold_used = 0.0
            _arousal_max_hops_used = spread_hops
            _arousal_mode_bias_adjust = 0.0

    id_to_idx = {rid: i for i, rid in enumerate(pool_ids)}

    gate_member_embeddings: dict[UUID, np.ndarray] = {
        pool_ids[i]: pool_embs[i]
        for i in range(len(pool_ids))
    }
    _gate_t0 = time.perf_counter() if _stage_profile_on else 0.0
    _gated_top_n, community_scores = _community_gate_scored(
        cue_emb, assignment, top_n=k_communities,
        member_embeddings=gate_member_embeddings,
    )
    if _stage_profile_on:
        _stage_timings["gate"] = (time.perf_counter() - _gate_t0) * 1000.0
    max_community_score = max(community_scores.values()) if community_scores else 0.0
    community_id_by_member: dict[UUID, UUID] = {}
    for gc in community_scores:
        for rid in assignment.mid_regions.get(gc, []):
            community_id_by_member[rid] = gc

    _centrality_t0 = time.perf_counter()
    centrality_arr = np.zeros(len(pool_ids), dtype=np.float32)
    for i, rid in enumerate(pool_ids):
        centrality_arr[i] = float(graph.get_centrality(rid))
    if not getattr(graph, "_centrality_resolved", False) and pool_ids:
        try:
            from iai_mcp.centrality_approx import centrality_for_runtime

            # Bounded recompute on the recall path: exact below the node-count
            # cutoff, deterministic k-source sampled betweenness above it. The
            # long-lived recall process must never run an unbounded O(V*E)
            # Brandes pass when the warm graph is large.
            cen_dict = centrality_for_runtime(graph)
            for i, rid in enumerate(pool_ids):
                centrality_arr[i] = float(cen_dict.get(rid, 0.0))
        except Exception as exc:  # noqa: BLE001 -- emit diagnostic then re-raise as NativeError
            write_event(
                store,
                "recall_centrality_failed",
                {"error_type": type(exc).__name__, "error": str(exc)},
            )
            raise NativeError(f"centrality recompute failed: {exc}") from exc
    _recall_centrality_ms = (time.perf_counter() - _centrality_t0) * 1000.0
    if _stage_profile_on:
        # Reuses the pre-existing unconditional _centrality_t0 timer above
        # (already runs regardless of the flag for the recall_timing
        # telemetry sample) -- adds a dict write only, mirroring the
        # pool_collection bucket's zero-incremental-timer-call discipline.
        _stage_timings["centrality"] = _recall_centrality_ms

    _seeds_t0 = time.perf_counter() if _stage_profile_on else 0.0
    seed_indices = _pick_seeds(
        cosine_top_indices, shared_cos, centrality_arr, n=3,
    )
    seed_ids = [pool_ids[int(i)] for i in seed_indices]
    if _stage_profile_on:
        _stage_timings["seeds"] = (time.perf_counter() - _seeds_t0) * 1000.0

    # Multi-seed widening: unconditional (capped union of fts_hits ids into
    # the seed set, so 2-hop spread also fans out from an exact-substring
    # match the cosine+centrality blend alone would miss).
    if os.environ.get("IAI_MCP_MULTI_SEED_OFF") != "true":
        if fts_hits:
            _fts_seed_ids = [
                rid
                for rid in fts_hits
                if rid in id_to_idx and rid not in seed_ids
            ]
            remaining = max(0, MULTI_SEED_CAP - len(seed_ids))
            seed_ids = list(dict.fromkeys(seed_ids + _fts_seed_ids[:remaining]))
        if trace_mark is not None:
            trace_mark("multi_seed")

    primed_ids: set[UUID] = set()
    if os.environ.get("IAI_MCP_PROC_PRIME") == "1":
        from iai_mcp import prime_cache

        if trace_mark is not None:
            trace_mark("proc_prime")
        _cache = prime_cache.load(store)
        _seed_to_chunks = _cache.get("seed_to_chunks", {})
        _chunk_members = _cache.get("chunk_members", {})
        if _seed_to_chunks:
            _proc_prime_candidates: list[UUID] = []
            for _sid in seed_ids:
                for _chunk_id in _seed_to_chunks.get(str(_sid), []):
                    _members = _chunk_members.get(_chunk_id)
                    if not _members or len(_members) < 2:
                        continue
                    try:
                        _next = UUID(_members[1])
                    except (ValueError, AttributeError, TypeError):
                        continue
                    if _next in id_to_idx and _next not in seed_ids:
                        _proc_prime_candidates.append(_next)
            _proc_prime_added = list(
                dict.fromkeys(_proc_prime_candidates)
            )[:PROC_PRIME_SEED_CAP]
            if _proc_prime_added:
                seed_ids = list(dict.fromkeys(seed_ids + _proc_prime_added))
                primed_ids.update(_proc_prime_added)

    _spread_t0 = time.perf_counter() if _stage_profile_on else 0.0
    spread_provenance: "dict[UUID, tuple[UUID, int, bool]]" = (
        graph.two_hop_neighborhood_with_provenance(seed_ids, top_k=5)
        if spread_hops > 0
        else {}
    )
    if _stage_profile_on:
        _stage_timings["spread"] = (time.perf_counter() - _spread_t0) * 1000.0
    spread_ids = sorted(spread_provenance, key=str)
    spread_indices = np.array(
        [id_to_idx[r] for r in spread_ids if r in id_to_idx],
        dtype=np.int64,
    )
    rich_indices = np.array(
        [id_to_idx[r] for r in (rich_club or []) if r in id_to_idx],
        dtype=np.int64,
    )
    if _arousal_rank_threshold_used > 0.0 and shared_cos.size:
        if spread_indices.size:
            spread_indices = spread_indices[
                shared_cos[spread_indices] >= _arousal_rank_threshold_used
            ]
        if rich_indices.size:
            rich_indices = rich_indices[
                shared_cos[rich_indices] >= _arousal_rank_threshold_used
            ]
    # Warm-lexical hits join as scored CANDIDATES only — never as spread
    # seeds: seeding them dilutes the shared two-hop quota and starves the
    # dense seeds' structural neighborhoods (measured -1pp on natural
    # questions). Inclusion, not steering — mirror of the exact-authority
    # contract.
    lex_indices = (
        np.array(
            [id_to_idx[r] for r in lex_rank if r in id_to_idx],
            dtype=np.int64,
        )
        if lex_rank
        else np.empty(0, dtype=np.int64)
    )
    if (
        cosine_top_indices.size
        or spread_indices.size
        or rich_indices.size
        or lex_indices.size
    ):
        reachable_indices = np.union1d(
            np.union1d(
                np.union1d(cosine_top_indices, spread_indices),
                rich_indices,
            ),
            lex_indices,
        ).astype(np.int64)
    else:
        reachable_indices = np.empty(0, dtype=np.int64)

    pre_filter_reachable_ids = [pool_ids[int(i)] for i in reachable_indices]
    # Captured before the verbatim/episodic narrowing below: Rust's own
    # `reachable` union (lib.rs `union_set`) is built from the same four
    # index arrays with no additional filtering at this stage, then narrowed
    # under `verbatim_filter` by its own resident tier column -- a source
    # independent from `episodic_ids` below. Flags must be populated over
    # this pre-filter union so a position Rust's own filter retains can
    # never read an unset (default-False) flag.
    flag_reachable_indices = reachable_indices
    if mode == "verbatim" and episodic_ids is not None:
        reachable_indices = np.array(
            [int(i) for i in reachable_indices if pool_ids[int(i)] in episodic_ids],
            dtype=np.int64,
        )
    post_filter_reachable_ids = [pool_ids[int(i)] for i in reachable_indices]

    if _VERBATIM_FILTER_DEBUG is not None:
        _VERBATIM_FILTER_DEBUG["pre_filter_reachable_ids"] = list(
            pre_filter_reachable_ids,
        )
        _VERBATIM_FILTER_DEBUG["post_filter_reachable_ids"] = list(
            post_filter_reachable_ids,
        )

    from iai_mcp.profile import profile_modulation_for_record

    structural_weight: float = 0.0
    cue_structure_hv: bytes | None = None
    if profile_state:
        try:
            structural_weight = float(profile_state.get("structural_weight", 0.0) or 0.0)
        except (TypeError, ValueError):
            structural_weight = 0.0
        structural_weight = max(0.0, min(1.0, structural_weight))

    lp_value = "medium"
    if profile_state:
        try:
            raw_lp = profile_state.get("literal_preservation", "medium")
            if isinstance(raw_lp, str) and raw_lp in LITERAL_PRESERVATION_W_DEGREE_SCALE:
                lp_value = raw_lp
        except (TypeError, ValueError, AttributeError) as exc:
            logger.debug("literal_preservation_parse_failed: %s", exc)
            lp_value = "medium"
    lp_scale = LITERAL_PRESERVATION_W_DEGREE_SCALE[lp_value]
    effective_w_degree = _env_weight("IAI_MCP_W_DEGREE", W_DEGREE) * lp_scale
    if mode == "verbatim":
        effective_w_degree = 0.0
    effective_w_cosine = (retrieval_weights or {}).get("W_COSINE", W_COSINE)

    if structural_weight > 0.0:
        from iai_mcp import tem
        cue_structure_hv = tem.pack_pairs([("TOPIC", tem.filler_hv(cue))])

    mode_bias = _gate_bias_for_mode(mode)
    mode_bias = mode_bias + _arousal_mode_bias_adjust

    contradicts_dst_set: set[str] = set()
    if cue_intent == "historical_verbatim":
        contradicts_dst_set = _build_contradicts_dst_set(contradicts_outgoing)

    corrector_base_score: dict[str, float] = {}

    # Each entry carries the served reason string, assembled DURING scoring
    # so every additive term and multiplier that touched the score is in it
    # — the printed arithmetic must reconcile with the served number.
    scored: list[tuple[float, UUID, str]] = []
    # Call-local: the per-call community_id / profile_modulation_gain a
    # candidate resolves to this call, keyed by record id. Read at response
    # construction and threaded to _apply_post_rank_pipeline -- the shared
    # SimpleRecordView objects in graph._records_view_cache are never
    # field-mutated with these values (they would otherwise leak across
    # calls that reuse the same cached object on a HIT).
    _local_community_id: dict[UUID, UUID] = {}
    _local_profile_gain: dict[UUID, dict] = {}
    tier_boost = _tier_knowledge_boost()
    salience_step = _salience_boost_step()
    proc_prime_boost = _env_weight("IAI_MCP_PROC_PRIME_BOOST", PROC_PRIME_BOOST_DEFAULT)
    w_spread_act = _env_weight("IAI_MCP_W_SPREAD_ACT", W_SPREAD_ACT)
    spread_act_decay = _env_weight("IAI_MCP_SPREAD_ACT_DECAY", SPREAD_ACT_DECAY)
    temporal_boost = _env_weight("IAI_MCP_TEMPORAL_BOOST", TEMPORAL_MATCH_BOOST)
    date_mentions: list = []
    if cue and temporal_boost != 1.0:
        from iai_mcp.temporal_cue import parse_date_mentions
        date_mentions = parse_date_mentions(cue)

    flat_cosine_pool = False
    _use_rust_scorer = _resolve_use_rust_scorer(use_rust_scorer, structural_weight)

    if _use_rust_scorer:
        from iai_mcp.store._rank_index import rank_index_for

        if trace_mark is not None:
            trace_mark("cleanup_attractor")

        _lex_lane_enabled = bool(
            cue
            and os.environ.get("IAI_MCP_LEX_FUSION_OFF") != "true"
            and _cue_identifier_grade(cue)
        )
        try:
            _lex_min_idf = float(
                os.environ.get("IAI_MCP_LEX_MIN_IDF", "") or LEX_FUSION_MIN_IDF
            )
        except ValueError:
            _lex_min_idf = LEX_FUSION_MIN_IDF

        _t11_t12_t0 = time.perf_counter() if _stage_profile_on else 0.0
        _t11_flags, _t12_flags = _t11_t12_flags(
            pool_ids, flag_reachable_indices, records_cache, fts_hits, cue,
        )
        if _stage_profile_on:
            _stage_timings["t11_t12"] = (time.perf_counter() - _t11_t12_t0) * 1000.0

        _rank_t0 = time.perf_counter() if _stage_profile_on else 0.0
        # Widened so a primed candidate survives winners.truncate(k+k_margin)
        # (lib.rs) -- byte-identical to the shipped constant when unprimed.
        _prime_k_margin = (
            RUST_SCORER_K_MARGIN if not primed_ids
            else max(RUST_SCORER_K_MARGIN, int(flag_reachable_indices.size))
        )
        winners, coverage, result_damp = rank_index_for(store, graph).score(
            graph,
            pool_ids,
            shared_cos,
            cosine_top_indices,
            spread_indices,
            rich_indices,
            lex_indices,
            _t11_flags,
            _t12_flags,
            mode == "verbatim",
            cue,
            int(time.time()),
            effective_w_degree,
            effective_w_cosine,
            graph.RANKING_DEGREE_EXCLUDED,
            spread_provenance,
            w_spread_act,
            spread_act_decay,
            community_id_by_member,
            community_scores,
            max_community_score,
            mode_bias,
            _env_weight("IAI_MCP_COS_SPREAD_MIN", COS_SPREAD_MIN),
            structural_weight,
            cue_structure_hv,
            _lex_lane_enabled,
            _lex_min_idf,
            _lex_fusion_w(),
            _POST_RANK_MAX_HITS,
            _prime_k_margin,
        )
        if _stage_profile_on:
            _stage_timings["reachable_count"] = float(coverage[0])
            _stage_timings["rank"] = (time.perf_counter() - _rank_t0) * 1000.0

        if result_damp < 1.0:
            # Mirrors the Python-path damp: degree/community/knowledge
            # boosts already carry it (applied inside the Rust call), tier
            # and salience are Bucket-B and carry it here, at the same
            # insertion point the Python path uses.
            flat_cosine_pool = True
            tier_boost = 1.0 + (tier_boost - 1.0) * result_damp
            salience_step = salience_step * result_damp

        _reason_w_degree = effective_w_degree * result_damp
        _reason_w_cosine = effective_w_cosine
        for (
            _wid, _partial_score, _pre_gain_base, _term_multiplier,
            _w_created_at, _w_salience_level, _w_tier, _w_tags, _terms,
        ) in winners:
            cid = UUID(int=_wid)
            rec = records_cache.get(cid)
            if rec is None:
                continue
            s = _partial_score
            (
                _t_cos, _t_aaak, _t_deg_norm, _t_age,
                _t_spread, _t_community, _t_structural,
            ) = _terms
            reason = (
                f"cos {_t_cos:.3f}*{_reason_w_cosine:g} + aaak {_t_aaak:.2f}*{W_AAAK:g} "
                f"+ deg_norm {_t_deg_norm:.3f}*{_reason_w_degree:.3g} "
                f"- age {_t_age:.2f}*{W_AGE:g}"
            )
            if _t_spread:
                reason += f" + spread {_t_spread:.3f}"
            if _t_community:
                reason += f" + community {_t_community:.3f}"
            if structural_weight > 0.0:
                reason += (
                    f" | structural {_t_structural:.3f} "
                    f"(w={structural_weight:.2f})"
                )
            if _term_multiplier >= 6.0:
                reason += " | x2.0 trigram | x3.0 fts"
            elif _term_multiplier >= 3.0:
                reason += " | x3.0 fts"
            elif _term_multiplier >= 2.0:
                reason += " | x2.0 trigram"
            cand_community = community_id_by_member.get(cid)
            if cand_community is not None:
                _local_community_id[cid] = cand_community
            if profile_state:
                if cand_community is not None or isinstance(rec, SimpleRecordView):
                    gains = profile_modulation_for_record(
                        rec, profile_state, knobs_applied=knobs_applied,
                        community_id_override=cand_community,
                    )
                else:
                    gains = profile_modulation_for_record(
                        rec, profile_state, knobs_applied=knobs_applied,
                    )
                if gains:
                    _local_profile_gain[cid] = dict(gains)
                    gain_product = 1.0
                    for gv in gains.values():
                        try:
                            gain_product *= float(gv)
                        except (TypeError, ValueError):
                            continue
                    if gain_product != 1.0:
                        s = _reinsert_rust_winner_gain(
                            _partial_score, _pre_gain_base, _term_multiplier, gain_product,
                        )
                        reason += f" | xgain {gain_product:.3f}"
            if (
                tier_boost != 1.0
                and mode != "verbatim"
                and cue_intent != "historical_verbatim"
                and (
                    _has_doc_tag(rec)
                    or (rec.tier == "semantic" and lp_value != "strong")
                )
            ):
                s *= tier_boost
                reason += f" | xtier {tier_boost:g}"
            _salience_rank = SALIENCE_LEVEL_RANK.get(
                getattr(rec, "salience_level", "unflagged"), 0,
            )
            if (
                _salience_rank > 0
                and mode != "verbatim"
                and cue_intent != "historical_verbatim"
            ):
                salience_multiplier = 1.0 + _salience_rank * salience_step
                if salience_multiplier != 1.0:
                    s *= salience_multiplier
                    reason += f" | xsalience {salience_multiplier:g}"
            if date_mentions:
                from iai_mcp.temporal_cue import matches_mentions
                if matches_mentions(rec.created_at, date_mentions):
                    s *= temporal_boost
                    reason += f" | xtemp {temporal_boost:g}"
            if (
                cid in primed_ids
                and mode != "verbatim"
                and cue_intent != "historical_verbatim"
            ):
                s *= proc_prime_boost
                reason += f" | xproc_prime {proc_prime_boost:g}"
            if cue_intent == "historical_verbatim" and contradicts_dst_set:
                if str(cid) in contradicts_dst_set:
                    corrector_base_score[str(cid)] = s
            scored.append((s, cid, reason))
    else:
        _degree_t0 = time.perf_counter() if _stage_profile_on else 0.0
        _global_deg_override: "dict[str, int] | None" = getattr(graph, "_global_degree", None)
        if _global_deg_override:
            degree = _global_deg_override
            max_deg = float(getattr(graph, "_max_degree", 0) or 0)
        else:
            # Recomputed every call -- no cross-call memoization on the graph.
            # Ranking degree counts EARNED edges only: similarity links inferred
            # at insert and entity anchors minted at sleep must not inflate hub
            # rank — on look-alike corpora the inferred clique otherwise
            # outranks the true target on degree alone. Every other edge type
            # (hebbian, contradicts, schema, temporal) is earned by use or by
            # consolidation and keeps its degree weight.
            degree = {
                str(nid): deg
                for nid, deg in graph.degrees(
                    exclude_types=graph.RANKING_DEGREE_EXCLUDED
                )
            }
            max_deg = float(max(degree.values(), default=0))
        if _stage_profile_on:
            _stage_timings["degree"] = (time.perf_counter() - _degree_t0) * 1000.0
        log_max_deg = log(1.0 + max_deg) if max_deg > 0 else 0.0

        # Cleanup-attractor: a bounded shortlist of the
        # current candidates' structural HVs, used to snap a noisy structural HV
        # to its nearest codebook entry ONLY within a rejection threshold, before
        # the structural-similarity score is computed. Bounded to <=200 entries
        # regardless of how far the confidence widen grew reachable_indices.
        _cleanup_shortlist_hvs: list[bytes] = [
            records_cache[pool_ids[int(i)]].structure_hv
            for i in reachable_indices[:200]
            if pool_ids[int(i)] in records_cache
            and getattr(records_cache[pool_ids[int(i)]], "structure_hv", None)
        ]
        if trace_mark is not None:
            trace_mark("cleanup_attractor")

        if reachable_indices.size >= 3:
            # Spread is measured over the competing HEAD of the pool: a single
            # distant graph- or lexical-reached candidate must not mask that the
            # slot-winning candidates are cosine-indistinguishable.
            _pool_cos = np.sort(shared_cos[reachable_indices])[::-1]
            _head = _pool_cos[: min(10, _pool_cos.size)]
            _cos_spread = float(_head[0] - _head[-1])
            _spread_min = _env_weight("IAI_MCP_COS_SPREAD_MIN", COS_SPREAD_MIN)
            _damp = _flat_cosine_damp(_cos_spread, _spread_min)
            if _damp < 1.0:
                # Degree, community bias and the knowledge boost are dampened
                # together: with the head flat, any of them would decide the
                # ranking exactly the way degree used to. The age penalty
                # survives deliberately — recency is the honest tie-breaker
                # when similarity carries no signal.
                effective_w_degree *= _damp
                mode_bias *= _damp
                flat_cosine_pool = True
        if flat_cosine_pool:
            tier_boost = 1.0 + (tier_boost - 1.0) * _damp
            salience_step = salience_step * _damp

        if _stage_profile_on:
            _stage_timings["reachable_count"] = float(reachable_indices.size)
        _rank_t0 = time.perf_counter() if _stage_profile_on else 0.0
        if reachable_indices.size:
            from iai_mcp.hebbian_structure import structural_similarity
            from iai_mcp.lilli.ops.cleanup import _cleanup_if_confident
            for idx in reachable_indices:
                i = int(idx)
                cid = pool_ids[i]
                rec = records_cache.get(cid)
                if rec is None:
                    continue
                cos = float(shared_cos[i])
                aaak = _aaak_overlap(cue, rec.aaak_index)
                deg = float(degree.get(str(cid), 0))
                age = _age_penalty(rec.created_at)
                if log_max_deg > 0.0:
                    deg_norm = log(1.0 + deg) / log_max_deg
                else:
                    deg_norm = 0.0
                base_s = (
                    effective_w_cosine * cos
                    + W_AAAK * aaak
                    + effective_w_degree * deg_norm
                    - W_AGE * age
                )
                spread_contrib = 0.0
                if w_spread_act > 0.0:
                    _prov = spread_provenance.get(cid)
                    # Transfer rides ONLY a fully transfer-carrying path (entity
                    # anchors); the similarity/hebbian mesh must not carry it.
                    if _prov is not None and _prov[2]:
                        _seed_idx = id_to_idx.get(_prov[0])
                        if _seed_idx is not None:
                            spread_contrib = (
                                w_spread_act
                                * float(shared_cos[int(_seed_idx)])
                                * (spread_act_decay ** _prov[1])
                            )
                            base_s += spread_contrib
                community_contrib = 0.0
                cand_community = community_id_by_member.get(cid)
                if cand_community is not None:
                    _local_community_id[cid] = cand_community
                if cand_community is not None and max_community_score > 0.0:
                    graded_weight = max(
                        0.0, community_scores.get(cand_community, 0.0) / max_community_score,
                    )
                    community_contrib = mode_bias * cos * graded_weight
                    base_s += community_contrib
                structural_score = 0.0
                if (
                    structural_weight > 0.0
                    and cue_structure_hv is not None
                    and rec.structure_hv
                ):
                    _cleaned_structure_hv = _cleanup_if_confident(
                        rec.structure_hv, _cleanup_shortlist_hvs, max_hamming_frac=0.15,
                    )
                    structural_score = structural_similarity(
                        cue_structure_hv, _cleaned_structure_hv,
                    )
                reason = (
                    f"cos {cos:.3f}*{effective_w_cosine:g} + aaak {aaak:.2f}*{W_AAAK:g} "
                    f"+ deg_norm {deg_norm:.3f}*{effective_w_degree:.3g} "
                    f"- age {age:.2f}*{W_AGE:g}"
                )
                if spread_contrib:
                    reason += f" + spread {spread_contrib:.3f}"
                if community_contrib:
                    reason += f" + community {community_contrib:.3f}"
                if structural_weight > 0.0:
                    base_s = (
                        (1.0 - structural_weight) * base_s
                        + structural_weight * structural_score
                    )
                    reason += (
                        f" | structural {structural_score:.3f} "
                        f"(w={structural_weight:.2f})"
                    )
                if profile_state:
                    if cand_community is not None or isinstance(rec, SimpleRecordView):
                        # SimpleRecordView is always call-local (never trust a
                        # residual attribute -- cache-HIT reuse risk); other
                        # records_cache value types are per-call-fresh, so when
                        # this call did not gate them into a community, fall
                        # back to their own persisted community_id below,
                        # exactly as before this fix.
                        gains = profile_modulation_for_record(
                            rec, profile_state, knobs_applied=knobs_applied,
                            community_id_override=cand_community,
                        )
                    else:
                        gains = profile_modulation_for_record(
                            rec, profile_state, knobs_applied=knobs_applied,
                        )
                    if gains:
                        _local_profile_gain[cid] = dict(gains)
                        gain_product = 1.0
                        for gv in gains.values():
                            try:
                                gain_product *= float(gv)
                            except (TypeError, ValueError):
                                continue
                        s = base_s * gain_product
                        if gain_product != 1.0:
                            reason += f" | xgain {gain_product:.3f}"
                    else:
                        s = base_s
                else:
                    s = base_s
                try:
                    _stability = getattr(rec, "stability", 0.5) or 0.5
                    _ig = (1.0 - min(float(_stability), 1.0)) * 0.1
                    s += _ig
                    if _ig:
                        reason += f" + stab {_ig:.3f}"
                except (TypeError, ValueError, AttributeError) as exc:
                    logger.debug("stability_lift_failed: %s", exc)
                _valence = getattr(rec, "valence", None) or 0.0
                if _valence > 0.0:
                    s *= (1.0 + _valence)
                    reason += f" | xval {1.0 + _valence:.2f}"
                if cue and rec.literal_surface and _trigram_jaccard(cue.lower(), rec.literal_surface.lower()) > 0.3:
                    s *= 2.0
                    reason += " | x2.0 trigram"
                if fts_hits and cid in fts_hits:
                    s *= 3.0
                    reason += " | x3.0 fts"
                if lex_rank and cid in lex_rank:
                    _lex_add = _lex_fusion_w() / (1.0 + lex_rank[cid])
                    s += _lex_add
                    reason += f" + lex {_lex_add:.3f}"
                if (
                    tier_boost != 1.0
                    and mode != "verbatim"
                    and cue_intent != "historical_verbatim"
                    and (
                        _has_doc_tag(rec)
                        or (rec.tier == "semantic" and lp_value != "strong")
                    )
                ):
                    s *= tier_boost
                    reason += f" | xtier {tier_boost:g}"
                _salience_rank = SALIENCE_LEVEL_RANK.get(
                    getattr(rec, "salience_level", "unflagged"), 0,
                )
                if (
                    _salience_rank > 0
                    and mode != "verbatim"
                    and cue_intent != "historical_verbatim"
                ):
                    salience_multiplier = 1.0 + _salience_rank * salience_step
                    if salience_multiplier != 1.0:
                        s *= salience_multiplier
                        reason += f" | xsalience {salience_multiplier:g}"
                if date_mentions:
                    from iai_mcp.temporal_cue import matches_mentions
                    if matches_mentions(rec.created_at, date_mentions):
                        s *= temporal_boost
                        reason += f" | xtemp {temporal_boost:g}"
                if (
                    cid in primed_ids
                    and mode != "verbatim"
                    and cue_intent != "historical_verbatim"
                ):
                    s *= proc_prime_boost
                    reason += f" | xproc_prime {proc_prime_boost:g}"
                if cue_intent == "historical_verbatim" and contradicts_dst_set:
                    if str(cid) in contradicts_dst_set:
                        corrector_base_score[str(cid)] = s
                scored.append((s, cid, reason))
        if _stage_profile_on:
            _stage_timings["rank"] = (time.perf_counter() - _rank_t0) * 1000.0

    if (
        cue_intent == "historical_verbatim"
        and contradicts_outgoing
        and corrector_base_score
        and scored
    ):
        _ANCHOR_EPSILON = 1e-4
        anchor_target: dict[str, float] = {}
        for src_s, dsts in contradicts_outgoing.items():
            best: float | None = None
            for d in dsts or []:
                cs = corrector_base_score.get(str(d))
                if cs is not None and (best is None or cs > best):
                    best = cs
            if best is not None:
                anchor_target[str(src_s)] = best - _ANCHOR_EPSILON
        if anchor_target:
            for j, row in enumerate(scored):
                tgt = anchor_target.get(str(row[1]))
                if tgt is not None and row[0] < tgt:
                    # The served score is replaced wholesale — the reason
                    # must say so instead of describing the old arithmetic.
                    scored[j] = (tgt, row[1], row[2] + " | anchored-below-corrector")

    if primed_ids:
        # A primed candidate can never eclipse the top genuine fused score.
        _proc_prime_unprimed = [s for s, cid, _ in scored if cid not in primed_ids]
        if _proc_prime_unprimed:
            _proc_prime_ceiling = max(_proc_prime_unprimed) - _PROC_PRIME_CLAMP_EPS
            scored = [
                (_proc_prime_ceiling, cid, r + " | proc_prime-clamped")
                if cid in primed_ids and s > _proc_prime_ceiling
                else (s, cid, r)
                for s, cid, r in scored
            ]

    scored.sort(key=lambda x: (-x[0], str(x[1])))
    if trace_mark is not None:
        trace_mark("soft_gate")

    _hit_assembly_t0 = time.perf_counter() if _stage_profile_on else 0.0
    scored_hits: list[MemoryHit] = []
    budget_used = 0
    for s, cid, reason in scored:
        rec = records_cache.get(cid)
        if rec is None:
            continue
        tokens = len(rec.literal_surface) // 4
        suggestions = graph.two_hop_neighborhood([cid], top_k=3)[:3]
        _prov = (rec.provenance or [{}])[0]
        # SimpleRecordView instances can be the SAME object served again on
        # a cache HIT, which would leak a stale attribute across cues if
        # trusted -- this local dict exists to prevent that, so their
        # attribute is never trusted, even as a fallback. Other
        # records_cache value types (MemoryRecord from the store-fallback
        # branch) are always fetched fresh this call and never cache-reused
        # across calls -- their own persisted field is an unchanged, safe
        # fallback when this call did not gate the record into a community.
        _served_community_id = _local_community_id.get(
            cid,
            None if isinstance(rec, SimpleRecordView) else getattr(rec, "community_id", None),
        )
        scored_hits.append(
            MemoryHit(
                record_id=cid,
                score=float(s),
                reason=reason,
                literal_surface=rec.literal_surface,
                adjacent_suggestions=suggestions,
                session_id=_prov.get("session_id"),
                captured_at=rec.created_at.isoformat() if rec.created_at else None,
                community_id=_served_community_id,
                epistemic_status=getattr(rec, "epistemic_status", None),
                salience_level=getattr(rec, "salience_level", None),
            ),
        )
        budget_used += tokens

    if _stage_profile_on:
        _stage_timings["hit_assembly"] = (
            (time.perf_counter() - _hit_assembly_t0) * 1000.0
        )
        _stage_timings["scored_count"] = float(len(scored))

    activation_trace = list({*seed_ids, *spread_ids})

    try:
        _top_hit_id_for_telemetry: str | None = None
        if scored_hits:
            _top_hit_id_for_telemetry = str(scored_hits[0].record_id)
        write_event(
            store,
            kind="retrieval_arousal_ab",
            data={
                "cue_hash": _arousal_cue_hash_hex,
                "route": _arousal_route,
                "n_hits": len(scored_hits),
                "budget_tokens_used": _arousal_budget_for_telemetry,
                "max_hops_used": _arousal_max_hops_used,
                "rank_threshold_used": _arousal_rank_threshold_used,
                "arousal_level": _arousal_level_for_telemetry,
                "arousal_mode": _arousal_mode_for_telemetry,
                "top_hit_id": _top_hit_id_for_telemetry,
            },
            severity="info",
            session_id=session_id,
            buffered=True,
        )
    except Exception as exc:  # noqa: BLE001 -- telemetry must never crash recall
        logger.debug("retrieval_arousal_ab_emit_failed: %s", exc)

    try:
        _sample_rate = float(os.environ.get("IAI_MCP_RECALL_SAMPLE_RATE", "0.1"))
    except (TypeError, ValueError):
        _sample_rate = 0.1
    if random.random() < _sample_rate:
        try:
            write_event(
                store,
                kind="recall_timing",
                data={
                    "centrality_ms": float(_recall_centrality_ms),
                    "sigma_ms": 0.0,
                    "pool_collection_ms": float(_recall_pool_collection_ms),
                    "n_nodes": int(len(pool_ids)),
                },
                severity="info",
                session_id=session_id,
                buffered=True,
            )
        except Exception as exc:  # noqa: BLE001 -- telemetry MUST NOT break recall
            logger.debug("recall_timing_emit_failed: %s", exc)

    core_hints: list[dict] = []
    if flat_cosine_pool:
        core_hints.append({
            "kind": "flat_cosine",
            "severity": "info",
            "source_ids": [],
            "text": (
                "candidate cosines are near-identical for this cue — the "
                "ranking carries no similarity signal; degree, community "
                "bias and the knowledge boost were dampened and this "
                "ordering should be treated as low confidence"
            ),
        })
    _gate_top1 = str(_gated_top_n[0]) if _gated_top_n else None
    if _stage_profile_on:
        if hydrate_stage_timings:
            _hydrate_ann = float(hydrate_stage_timings.get("hydrate_ann", 0.0) or 0.0)
            _hydrate_getbatch = float(hydrate_stage_timings.get("hydrate_getbatch", 0.0) or 0.0)
            _stage_timings["hydrate_ann"] = _hydrate_ann
            _stage_timings["hydrate_getbatch"] = _hydrate_getbatch
            _stage_timings["hydrate"] = _hydrate_ann + _hydrate_getbatch
            _overlap = hydrate_stage_timings.get("candidate_overlap_fraction")
            if _overlap is not None:
                _stage_timings["candidate_overlap_fraction"] = float(_overlap)
            for _key in (
                "structural", "authority_scan", "hop1_edges", "hop2_edges",
                "ann_scan", "ann_inlist", "ann_decode", "ann_rows_fetched",
                "ann_rows_served", "ge_populate", "ge_incident", "ge_split",
                "ge_contr_fetch", "hops_snapshot",
            ):
                if _key in hydrate_stage_timings:
                    _stage_timings[_key] = float(hydrate_stage_timings[_key] or 0.0)
        _last_stage_timings_ms.clear()
        _last_stage_timings_ms.update(_stage_timings)
    # The corpus-stable community count, NOT len(community_scores) -- the
    # max-node gate scores only communities with a member in this query's
    # candidate pool, which shrinks and grows per query. mid_regions is the
    # full corpus grouping the gate was built from.
    return _RecallCoreResult(
        scored_hits=scored_hits,
        activation_trace=activation_trace,
        anti_hits=[],
        hints=core_hints,
        patterns_observed=[],
        cue_mode=mode,
        budget_used=budget_used,
        _records_cache=records_cache,
        _profile_gains=_local_profile_gain,
        stage_timings=_stage_timings,
        cue_community_id=_gate_top1,
        community_k=len(assignment.mid_regions),
        community_backend=assignment.backend,
    )


def _apply_post_rank_pipeline(
    hits: list[MemoryHit],
    *,
    store: MemoryStore,
    graph: MemoryGraph,
    records_cache: dict[UUID, "object"],
    cue: str,
    session_id: str,
    profile_state: dict | None,
    turn: int,
    mode: str,
    budget_used: int,
    path_label: str,
    knobs_applied: dict | None = None,
    contradicts_outgoing: dict[str, list[str]] | None = None,
    cue_community_id: "str | None" = None,
    community_k: "int | None" = None,
    community_backend: "str | None" = None,
    profile_gains: "dict[UUID, dict] | None" = None,
) -> tuple[list[MemoryHit], list[MemoryHit], list[dict], list[dict]]:
    s4_scope_hits = hits[:_POST_RANK_MAX_HITS]

    if hits:
        try:
            from iai_mcp.provenance_buffer import defer_provenance
            # Read-only input (hits), no return value read by this call --
            # suppressing it changes zero bytes of the response.
            if not recall_suppressed.get():
                defer_provenance(
                    store,
                    [(h.record_id, cue, session_id) for h in hits],
                )
        except Exception as exc:  # noqa: BLE001 -- retrieval hot-path fail-safe
            logger.debug("provenance_defer_failed: %s", exc)

    anti_hits = _find_anti_hits(
        s4_scope_hits, store, graph, k=3, records_cache=records_cache,
    )

    if mode == "verbatim":
        hints: list[dict] = []
    else:
        try:
            from iai_mcp.s4 import on_read_check_batch
            hints = on_read_check_batch(
                store, s4_scope_hits, session_id=session_id,
                records_cache=records_cache,
                contradicts_outgoing=contradicts_outgoing,
            )
        except Exception as exc:  # noqa: BLE001 -- retrieval hot-path fail-safe
            logger.debug("s4_on_read_check_batch_failed: %s", exc)
            hints = []

    if profile_state:
        modulate_pairs: list[tuple] = []
        modulate_deltas: list[float] = []
        for h in hits:
            try:
                rec = records_cache.get(h.record_id)
                if rec is None:
                    continue
                # Same rationale as the community_id read above: SimpleRecordView
                # instances are never trusted for a residual field value (cache-HIT
                # reuse risk); other records_cache value types are per-call-fresh
                # and their own persisted field is a safe, unchanged fallback.
                gains = (profile_gains or {}).get(h.record_id)
                if gains is None:
                    gains = (
                        {} if isinstance(rec, SimpleRecordView)
                        else (getattr(rec, "profile_modulation_gain", None) or {})
                    )
                if not gains:
                    continue
                total_gain = float(sum(gains.values()))
                if total_gain <= 0:
                    total_gain = 1.0
                modulate_pairs.append((h.record_id, PROFILE_SENTINEL_UUID))
                modulate_deltas.append(total_gain)
            except (TypeError, ValueError, AttributeError) as exc:
                logger.debug("profile_modulate_per_hit_failed rid=%s: %s", h.record_id, exc)
                continue
        if modulate_pairs and not recall_suppressed.get():
            if _defer_profile_boost_off():
                try:
                    for _chunk_start in range(0, len(modulate_pairs), BOOST_EDGES_SMALL_BATCH):
                        _chunk_pairs = modulate_pairs[_chunk_start:_chunk_start + BOOST_EDGES_SMALL_BATCH]
                        _chunk_deltas = modulate_deltas[_chunk_start:_chunk_start + BOOST_EDGES_SMALL_BATCH]
                        try:
                            store.boost_edges(
                                _chunk_pairs,
                                edge_type="profile_modulates",
                                delta=_chunk_deltas,
                            )
                        except Exception as _chunk_exc:  # noqa: BLE001 — per-chunk degrade
                            logger.debug("boost_edges_chunk_failed: %s", _chunk_exc)
                except Exception as exc:  # noqa: BLE001 -- retrieval hot-path fail-safe
                    logger.debug("boost_edges_profile_modulates_failed: %s", exc)
            else:
                try:
                    store.queue_profile_modulate(modulate_pairs, modulate_deltas)
                except Exception as exc:  # noqa: BLE001 -- retrieval hot-path fail-safe
                    logger.debug("queue_profile_modulate_failed: %s", exc)

    # Freshness markers carry score 0.0 by design — they are recency signal,
    # not ranker output, and would collapse the nightly entropy to zero.
    _curio_hits = [
        h for h in s4_scope_hits if h.reason != "pending-recency"
    ][:10]
    if mode != "verbatim" and _curio_hits:
        try:
            write_event(
                store,
                kind="deferred_curiosity_input",
                data={
                    "hit_ids": [str(h.record_id) for h in _curio_hits],
                    # Entropy is computed at night from these scores; the
                    # ranker state cannot be reconstructed after the fact.
                    "scores": [float(h.score) for h in _curio_hits],
                    "cue": cue[:200],
                    "session_id": session_id,
                    "turn": int(turn),
                },
                severity="info",
                session_id=session_id,
                buffered=True,
            )
        except Exception as exc:  # noqa: BLE001 -- retrieval hot-path fail-safe
            logger.debug("deferred_curiosity_input_event_failed: %s", exc)

    patterns_observed: list[dict] = []
    if mode == "concept":
        kept_hits: list[MemoryHit] = []
        for h in hits:
            rec = records_cache.get(h.record_id)
            if rec is None:
                kept_hits.append(h)
                continue
            tier = getattr(rec, "tier", "episodic")
            tags = list(getattr(rec, "tags", []) or [])
            is_schema = (
                tier == "semantic"
                and any(t.startswith("pattern:") for t in tags)
            )
            # procedural chunks are dropped silently here, never summarized —
            # a structured priming surface belongs to the retrieval-priming
            # consumer, not this filter
            is_procedural = tier == "procedural"
            if is_procedural:
                continue
            if is_schema:
                if len(patterns_observed) < 3:
                    pattern_str = ""
                    for t in tags:
                        if t.startswith("pattern:"):
                            pattern_str = t.split(":", 1)[1] if ":" in t else ""
                            break
                    evidence_count = 0
                    try:
                        _schema_edges = store.incident_edges(
                            [h.record_id],
                            edge_types=["schema_instance_of"],
                            top_k=None,
                        )
                        evidence_count = sum(
                            len(v) for v in _schema_edges.values()
                        )
                    except Exception as exc:  # noqa: BLE001 — degradable evidence count
                        logger.debug("evidence_count_incident_edges_failed: %s", exc)
                        evidence_count = 0
                    patterns_observed.append({
                        "pattern": pattern_str,
                        "evidence_count": evidence_count,
                        "schema_id": str(h.record_id),
                    })
            else:
                kept_hits.append(h)
        hits = kept_hits

    try:
        from iai_mcp.response_decorator import suggestions_visible
        try:
            from iai_mcp.core import task_support_probe_active
            _probe_active = task_support_probe_active()
        except Exception as exc:  # noqa: BLE001 -- retrieval hot-path fail-safe
            logger.debug("task_support_probe_active_lookup_failed: %s", exc)
            _probe_active = False
        _suggestions_visible_now = suggestions_visible(profile_state or {}, _probe_active)
        # Gated to null at emit (never inside the pure spec): a flat backend
        # or K below the floor carries no honest concentration signal.
        from iai_mcp.lilli.cycle.sleep_pipeline._knob_tune_specs import K_MIN
        _gate_signal_valid = (
            community_backend != "flat"
            and community_k is not None
            and community_k >= K_MIN
        )
        write_event(
            store,
            kind="retrieval_used",
            data={
                "hit_ids": [str(h.record_id) for h in hits],
                "query": cue,
                "used": len(hits) > 0,
                "budget_used": budget_used,
                "path": path_label,
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "suggestion_ids": [
                    str(s) for h in hits for s in (h.adjacent_suggestions or [])
                ],
                "suggestions_visible": _suggestions_visible_now,
                "probe": _probe_active,
                "cue_community_id": cue_community_id if _gate_signal_valid else None,
                "community_k": community_k if _gate_signal_valid else None,
            },
            severity="info",
            session_id=session_id,
            buffered=True,
        )
    except Exception as exc:  # noqa: BLE001 -- retrieval hot-path fail-safe
        logger.debug("retrieval_used_event_failed: %s", exc)

    return hits, anti_hits, hints, patterns_observed


def recall_for_response(
    store: MemoryStore,
    graph: MemoryGraph,
    assignment: CommunityAssignment,
    rich_club: list[UUID],
    embedder: Embedder,
    cue: str,
    session_id: str,
    budget_tokens: int = 1500,
    profile_state: dict | None = None,
    turn: int = 0,
    mode: str = "concept",
    *,
    knobs_applied: dict | None = None,
    arousal_state: dict | None = None,
    tv_maps: "tuple[dict, dict] | None" = None,
    trace_mark: Callable[[str], None] | None = None,
    cue_embedding: "list[float] | None" = None,
    hydrate_stage_timings: dict | None = None,
    use_rust_scorer: bool | None = None,
    retrieval_weights: dict[str, float] | None = None,
) -> RecallResponse:
    import time as _time
    global _last_recall_latency_ms
    _rfr_t0 = _time.perf_counter()

    if arousal_state:
        logger.debug(
            "arousal_recall: level=%.2f mode=%s budget=%d",
            arousal_state.get("level", 0.5),
            arousal_state.get("mode", "unknown"),
            budget_tokens,
        )

    # Bounded full-quality spread — the default fast-inner-loop parameters.
    # Wall-clock latency is never a control input; it is written at the end
    # of this function for telemetry/probe use only.
    _k_com = 3
    _s_hops = 2

    from iai_mcp.cue_router import _classify_cue
    from iai_mcp.retrieve import (
        apply_stale_downweight,
        apply_supersede_cap,
        build_temporal_validity_maps,
        derive_temporal_validity,
        sort_served_hits,
    )
    _cue_mode_unused, _cue_intent, _cue_label_unused = _classify_cue(cue)
    if tv_maps is not None:
        _tv_outgoing, _tv_ts = tv_maps
    else:
        _tv_maps_built = build_temporal_validity_maps(store)
        _tv_outgoing, _tv_ts = (_tv_maps_built if _tv_maps_built is not None else ({}, {}))

    core = _recall_core(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=cue, session_id=session_id,
        profile_state=profile_state, turn=turn, mode=mode,
        knobs_applied=knobs_applied,
        k_communities=_k_com,
        spread_hops=_s_hops,
        cue_intent=_cue_intent,
        contradicts_outgoing=_tv_outgoing,
        trace_mark=trace_mark,
        cue_embedding=cue_embedding,
        hydrate_stage_timings=hydrate_stage_timings,
        use_rust_scorer=use_rust_scorer,
        retrieval_weights=retrieval_weights,
    )

    # store is passed for the bounded hit-timestamp fill: the maps carry only
    # contradiction-involved timestamps, and each hit's own valid_from comes
    # from one <=k-id lookup.
    derive_temporal_validity(
        store, core.scored_hits, outgoing=_tv_outgoing, ts_by_id=_tv_ts,
    )
    derive_temporal_validity(
        store, core.anti_hits, outgoing=_tv_outgoing, ts_by_id=_tv_ts,
    )
    apply_stale_downweight(core.scored_hits, cue_intent=_cue_intent)
    apply_stale_downweight(core.anti_hits, cue_intent=_cue_intent)
    apply_supersede_cap(core.scored_hits, _tv_outgoing, cue_intent=_cue_intent)
    sort_served_hits(core.scored_hits)

    if (
        len(core.scored_hits) == 1
        and any(h.get("kind") == "retrieval_skipped" for h in core.hints)
    ):
        return RecallResponse(
            hits=core.scored_hits,
            anti_hits=core.anti_hits,
            activation_trace=core.activation_trace,
            budget_used=core.budget_used,
            hints=core.hints,
            cue_mode=core.cue_mode,
            patterns_observed=core.patterns_observed,
            stage_timings=core.stage_timings,
        )

    hits: list[MemoryHit] = []
    budget_used = 0
    for hit in core.scored_hits:
        if len(hits) >= _POST_RANK_MAX_HITS:
            break
        tokens = len(hit.literal_surface) // 4
        if budget_used + tokens > budget_tokens and len(hits) >= 1:
            break
        hits.append(hit)
        budget_used += tokens

    # Fold the pending-recency markers THROUGH the same token budget that
    # capped the regular hits above, instead of appending them past the cap.
    # The freshness markers (score=0.0) are the lowest-priority surfaces; a
    # marker that would push past budget may trim the lowest-priority *regular*
    # hit to make room, but the markers' total token cost is capped at
    # _MARKER_BUDGET_SHARE of the budget — past that cap the marker is dropped,
    # never another ranked hit.
    try:
        _pending_n = max(10, len(hits))
        _pending_markers = store.recent_pending_markers(n=_pending_n)
        _ranked_ids: set = {h.record_id for h in hits}
        _marker_budget = int(budget_tokens * _MARKER_BUDGET_SHARE)
        _marker_used = 0
        # The share cap only protects ranked hits; with none packed, markers
        # may fill the whole budget (degraded freshness-only serving).
        _has_ranked = bool(hits)
        for _pm in _pending_markers:
            if _pm.id in _ranked_ids:
                continue
            _pm_tokens = len(_pm.literal_surface or "") // 4
            if _has_ranked and _marker_used + _pm_tokens > _marker_budget:
                continue
            # Reclaim budget from the lowest-priority regular (non-marker) hit
            # when this marker would overflow. Regular hits keep score order
            # (highest first), so the last regular hit is the cheapest to drop.
            while budget_used + _pm_tokens > budget_tokens and hits:
                _regular_idx = next(
                    (
                        _i
                        for _i in range(len(hits) - 1, -1, -1)
                        if hits[_i].reason != "pending-recency"
                    ),
                    None,
                )
                if _regular_idx is None:
                    break
                _dropped = hits.pop(_regular_idx)
                budget_used -= len(_dropped.literal_surface) // 4
            if budget_used + _pm_tokens > budget_tokens and hits:
                # No regular hit left to reclaim and the marker still overflows;
                # skip it rather than exceed budget.
                continue
            _ranked_ids.add(_pm.id)
            hits.append(MemoryHit(
                record_id=_pm.id,
                score=0.0,
                reason="pending-recency",
                literal_surface=_pm.literal_surface or "",
                adjacent_suggestions=[],
                session_id=(_pm.provenance[0].get("session_id") if _pm.provenance else None),
                captured_at=(
                    _pm.created_at.isoformat() if _pm.created_at else None
                ),
                community_id=getattr(_pm, "community_id", None),
                epistemic_status=_pm.epistemic_status,
                salience_level=_pm.salience_level,
            ))
            budget_used += _pm_tokens
            _marker_used += _pm_tokens
    except Exception as _pm_exc:  # noqa: BLE001 -- recency union is additive; never crash recall
        logger.debug("pending_markers_union_failed: %s", _pm_exc)

    # Legacy-only pre-budget-pack enrichment. The sole post-budget-pack
    # _backfill_hit_metadata() call below covers hits+anti_hits with the
    # identical 4 fields; running this here too wastes decode on any hit
    # later dropped before the response is built.
    if _crossing_consolidation_off():
        _enrich_ids = [_h.record_id for _h in hits if _h.session_id is None]
        _enrich_batch: dict = {}
        if _enrich_ids:
            try:
                _enrich_batch = store.get_batch(_enrich_ids)
            except Exception as _exc:  # noqa: BLE001 -- additive enrichment, never crash recall
                logger.debug("hit_provenance_enrich_batch_failed: %s", _exc)
                _enrich_batch = {}
        for _h in hits:
            if _h.session_id is None:
                _full_rec = _enrich_batch.get(_h.record_id)
                if _full_rec is not None:
                    _h_prov = (_full_rec.provenance or [{}])[0]
                    _h.session_id = _h_prov.get("session_id")
                    _h.captured_at = (
                        _full_rec.created_at.isoformat()
                        if _full_rec.created_at else None
                    )
                    if _h.epistemic_status is None:
                        _h.epistemic_status = getattr(_full_rec, "epistemic_status", None)
                    if _h.salience_level is None:
                        _h.salience_level = getattr(_full_rec, "salience_level", None)

    hits, anti_hits, hints, patterns_observed = _apply_post_rank_pipeline(
        hits,
        store=store, graph=graph, records_cache=core._records_cache,
        cue=cue, session_id=session_id,
        profile_state=profile_state, turn=turn, mode=mode,
        budget_used=budget_used, path_label="recall_for_response",
        knobs_applied=knobs_applied,
        contradicts_outgoing=_tv_outgoing,
        cue_community_id=core.cue_community_id,
        community_k=core.community_k,
        community_backend=core.community_backend,
        profile_gains=core._profile_gains,
    )

    if hits:
        # Final budget pack after the post-rank pipeline may have reordered the
        # list. Pack regular hits in their post-rank order up to budget, then
        # fold the recency markers back in within whatever budget remains,
        # trimming the lowest-priority regular hit when a marker needs room —
        # but only within the markers' _MARKER_BUDGET_SHARE slice. Freshness
        # markers survive the cap without starving the ranked hits; total
        # emitted token cost stays at or under budget_tokens.
        _regular = [h for h in hits if h.reason != "pending-recency"]
        _markers = [h for h in hits if h.reason == "pending-recency"]

        _final_hits: list[MemoryHit] = []
        _final_budget = 0
        for _fh in _regular:
            _fh_tokens = len(_fh.literal_surface) // 4
            if _final_budget + _fh_tokens > budget_tokens and _final_hits:
                break
            _final_hits.append(_fh)
            _final_budget += _fh_tokens

        # Same share cap as the union fold above: markers reclaim ranked
        # budget only within _MARKER_BUDGET_SHARE; with no ranked hits packed
        # they may fill the whole budget.
        _mk_budget = int(budget_tokens * _MARKER_BUDGET_SHARE)
        _mk_used = 0
        _has_ranked_final = bool(_final_hits)
        for _mk in _markers:
            _mk_tokens = len(_mk.literal_surface) // 4
            if _has_ranked_final and _mk_used + _mk_tokens > _mk_budget:
                continue
            while _final_budget + _mk_tokens > budget_tokens and _final_hits:
                _regular_idx = next(
                    (
                        _i
                        for _i in range(len(_final_hits) - 1, -1, -1)
                        if _final_hits[_i].reason != "pending-recency"
                    ),
                    None,
                )
                if _regular_idx is None:
                    break
                _dropped = _final_hits.pop(_regular_idx)
                _final_budget -= len(_dropped.literal_surface) // 4
            if _final_budget + _mk_tokens > budget_tokens and _final_hits:
                continue
            _final_hits.append(_mk)
            _final_budget += _mk_tokens
            _mk_used += _mk_tokens

        hits = _final_hits
        budget_used = _final_budget

    derive_temporal_validity(
        None, anti_hits, outgoing=_tv_outgoing, ts_by_id=_tv_ts,
    )
    apply_stale_downweight(anti_hits)
    _backfill_hit_metadata(hits, anti_hits, store)

    _last_recall_latency_ms = (_time.perf_counter() - _rfr_t0) * 1000

    return RecallResponse(
        hits=hits,
        anti_hits=anti_hits,
        activation_trace=core.activation_trace,
        budget_used=budget_used,
        hints=[*core.hints, *hints],
        cue_mode=core.cue_mode,
        patterns_observed=patterns_observed,
        stage_timings=core.stage_timings,
    )


def merge_authority_hits(
    pipeline_hits: list[MemoryHit],
    authority_hits: list[MemoryHit],
    budget_tokens: int,
    max_hits: int = _POST_RANK_MAX_HITS,
) -> tuple[list[MemoryHit], int]:
    """Union exact-similarity authority hits (head) with the pipeline's
    associative hits (tail), re-packed to the token budget.

    The head is exact-similarity hits in the given order, deduped against the
    tail by record id. The tail is the pipeline hits not already in the head,
    in their existing order — this preserves graph rank order and any
    pending-recency markers exactly where the pipeline placed them. The union
    is packed head-then-tail with the same token-budget semantics as the
    pipeline's own pack loop: token cost is ``len(literal_surface) // 4``,
    packing stops at ``max_hits``, and stops on overflow once at least one hit
    is packed. Because the head packs first, no tail hit can ever consume
    budget before a head hit — ranking may only reorder the tail.

    The authority guarantee is INCLUSION in the packed response (no false
    negatives), not a frozen head order. This function fixes membership and
    budget only; a caller applying a correctness-driven re-rank afterward
    (e.g. temporal-validity downweight for stale/contradicted records) is
    expected to reorder the returned list freely as long as it never drops an
    authority hit that was packed here.

    This plain pack loop does not run the pipeline's separate marker-reclaim
    fold, so a pending-recency marker (score 0.0, placed at the tail end) is
    the first casualty of head budget consumption here rather than getting
    the reclaim protection it has on the pipeline-only path. This is an
    accepted trade-off of authority-first ranking, not a bug: authority
    (guaranteed-correct exact matches) outranks freshness markers under
    budget pressure.

    An empty authority list is an identity fast path: the pipeline hits and
    their caller-supplied budget accounting are returned unchanged, so
    disabling or omitting the authority produces zero behavioral drift.

    When a record appears in both head and tail, the displaced tail hit's
    ``adjacent_suggestions`` (graph-derived, absent on a fresh authority hit)
    are copied onto the surviving head hit before the tail hit is dropped, so
    the association data is not silently lost to the authority promotion.
    The displaced hit's SCORE survives the same way when it is higher: the
    graph rank carries correctness signals the raw cosine cannot (knowledge
    boost, lexical fusion, community bonus), and the authority contributes
    inclusion, never a score wipe — otherwise the caller's post-merge
    re-sort would silently reduce every authority-head candidate back to
    pure cosine order.
    """
    if not authority_hits:
        return pipeline_hits, sum(len(h.literal_surface) // 4 for h in pipeline_hits)

    pipeline_by_id = {h.record_id: h for h in pipeline_hits}
    for h in authority_hits:
        displaced = pipeline_by_id.get(h.record_id)
        if displaced is not None:
            if displaced.adjacent_suggestions and not h.adjacent_suggestions:
                h.adjacent_suggestions = list(displaced.adjacent_suggestions)
            if displaced.score > h.score:
                # The authority hit keeps its identity but serves the higher
                # pipeline score — the reason must carry both provenances,
                # and the downweight state must travel WITH the score or a
                # later stale pass halves an already-halved number.
                h.score = displaced.score
                h.reason += f" | score from pipeline rank: {displaced.reason}"
                if getattr(displaced, "_stale_downweighted", False):
                    h._stale_downweighted = True
    head_ids = {h.record_id for h in authority_hits}
    tail = [h for h in pipeline_hits if h.record_id not in head_ids]
    union = list(authority_hits) + tail

    packed: list[MemoryHit] = []
    budget_used = 0
    for hit in union:
        if len(packed) >= max_hits:
            break
        tokens = len(hit.literal_surface) // 4
        if budget_used + tokens > budget_tokens and len(packed) >= 1:
            break
        packed.append(hit)
        budget_used += tokens

    return packed, budget_used


def recall_for_benchmark(
    store: MemoryStore,
    graph: MemoryGraph,
    assignment: CommunityAssignment,
    rich_club: list[UUID],
    embedder: Embedder,
    cue: str,
    session_id: str,
    k_hits: int = 10,
    profile_state: dict | None = None,
    turn: int = 0,
    mode: str = "concept",
    *,
    knobs_applied: dict | None = None,
    cue_embedding: "list[float] | None" = None,
) -> RecallResponse:
    from iai_mcp.cue_router import _classify_cue
    from iai_mcp.retrieve import (
        apply_stale_downweight,
        apply_supersede_cap,
        build_temporal_validity_maps,
        derive_temporal_validity,
        sort_served_hits,
    )
    _cue_mode_unused, _cue_intent, _cue_label_unused = _classify_cue(cue)
    _tv_maps = build_temporal_validity_maps(store)
    _tv_outgoing, _tv_ts = (_tv_maps if _tv_maps is not None else ({}, {}))

    core = _recall_core(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=cue, session_id=session_id,
        profile_state=profile_state, turn=turn, mode=mode,
        knobs_applied=knobs_applied,
        cue_intent=_cue_intent,
        contradicts_outgoing=_tv_outgoing,
        cue_embedding=cue_embedding,
    )
    if (
        len(core.scored_hits) == 1
        and any(h.get("kind") == "retrieval_skipped" for h in core.hints)
    ):
        return RecallResponse(
            hits=core.scored_hits,
            anti_hits=core.anti_hits,
            activation_trace=core.activation_trace,
            budget_used=core.budget_used,
            hints=core.hints,
            cue_mode=core.cue_mode,
            patterns_observed=core.patterns_observed,
        )

    # Parity with the production recall path: superseded hits carry a past
    # valid_to and are downweighted BEFORE truncation, so the benchmark
    # measures the ordering production actually serves.
    derive_temporal_validity(
        store, core.scored_hits, outgoing=_tv_outgoing, ts_by_id=_tv_ts,
    )
    derive_temporal_validity(
        store, core.anti_hits, outgoing=_tv_outgoing, ts_by_id=_tv_ts,
    )
    apply_stale_downweight(core.scored_hits, cue_intent=_cue_intent)
    apply_stale_downweight(core.anti_hits, cue_intent=_cue_intent)
    apply_supersede_cap(core.scored_hits, _tv_outgoing, cue_intent=_cue_intent)
    sort_served_hits(core.scored_hits)

    hits = core.scored_hits[:k_hits]
    budget_used = sum(len(h.literal_surface) // 4 for h in hits)

    hits, anti_hits, hints, patterns_observed = _apply_post_rank_pipeline(
        hits,
        store=store, graph=graph, records_cache=core._records_cache,
        cue=cue, session_id=session_id,
        profile_state=profile_state, turn=turn, mode=mode,
        budget_used=budget_used, path_label="recall_for_benchmark",
        knobs_applied=knobs_applied,
        contradicts_outgoing=_tv_outgoing,
        cue_community_id=core.cue_community_id,
        community_k=core.community_k,
        community_backend=core.community_backend,
        profile_gains=core._profile_gains,
    )
    _backfill_hit_metadata(hits, anti_hits, store)

    return RecallResponse(
        hits=hits,
        anti_hits=anti_hits,
        activation_trace=core.activation_trace,
        budget_used=budget_used,
        hints=[*core.hints, *hints],
        cue_mode=core.cue_mode,
        patterns_observed=patterns_observed,
    )
