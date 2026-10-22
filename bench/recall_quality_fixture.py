"""Synthetic, quotation-backed recall-quality fixture builder.

Builds an isolated, seeded synthetic corpus from pre-computed vectors --
never a real embed, never the real store -- and deliberately provokes
each of the four production discovery-miss causes
(``bench.fhrr_verdict_membership.ALL_CAUSES``) via geometry and edge
weight, not by hand-labelling. Every cue carries a ``quote`` that is a
verbatim substring of its gold record's ``literal_surface``: the target
is unambiguous, never a pinned-cosine guess.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass, field
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

from bench.recall_accuracy import (  # noqa: E402
    _VectorInjectionEmbedder,
    _insert_raw,
    _make_eval_record,
    _unit_vec,
)

REQUIRED_CUE_KEYS = ("cue_id", "cue_text", "gold_record_id", "cluster_id", "quote", "split")

_STRONG_EDGE_DELTA = 6.0
"""Weight given to the decoy edges that must outrank the engineered miss
edge in the top-5-per-node spread traversal (``core._incident_edges_warm``,
sorted descending by weight)."""

_WEAK_EDGE_DELTA = 0.02
"""Weight of the engineered gold->pool edge -- must lose every weight tie
against the decoy edges above so 2-hop spread never traverses it, while
the edge still exists in the raw edges table (``build_edge_adjacency``
sees it; the top-5 cap is what makes it structurally unreachable)."""


def _validate_quotation_backed(cue: dict, gold_literal_surface: str) -> None:
    """Fail loud when a cue's ``quote`` is not a verbatim substring of its
    gold record's ``literal_surface`` -- a pinned-cosine or hand-tagged
    target with no quotation anchor is never an acceptable substitute."""
    quote = cue["quote"]
    if quote not in gold_literal_surface:
        raise ValueError(
            f"cue {cue['cue_id']!r} quote {quote!r} is not a verbatim "
            f"substring of gold record {cue['gold_record_id']!r} "
            f"literal_surface {gold_literal_surface!r}"
        )


def validate_cue_schema(cue: dict) -> None:
    for key in REQUIRED_CUE_KEYS:
        if key not in cue:
            raise ValueError(f"fixture cue missing required key {key!r}: {cue}")


@dataclass
class FixtureCorpus:
    """A built synthetic store plus its labelled cue set.

    ``gold_literal_surface`` maps every gold record id to its stored
    ``literal_surface`` so quotation-verbatim checks never need a store
    read -- callers that only need schema/quotation integrity can work
    from this dict alone.
    """

    store: Any
    shim: "_VectorInjectionEmbedder"
    embedder: Any
    cues: "list[dict]"
    gold_literal_surface: "dict[str, str]"
    tmpdir: Any = field(default=None, repr=False)

    def close(self) -> None:
        self.store.close()
        if self.tmpdir is not None:
            self.tmpdir.cleanup()


def _weighted_edge(store: Any, a: UUID, b: UUID, weight: float) -> None:
    store.boost_edges([(a, b)], delta=weight, edge_type="hebbian")


def _build_local_pool(
    rng: "np.random.Generator",
    store: Any,
    cue_vec: np.ndarray,
    n_local: int,
    text_prefix: str,
    extra_tokens: "list[str] | None" = None,
) -> "list[UUID]":
    """Insert ``n_local`` records with cosine 0.3-0.9 to ``cue_vec`` so
    they anchor a small, cue-specific ANN pool. ``extra_tokens`` are
    appended verbatim to each filler's ``literal_surface`` so a shared
    identifier token's in-corpus frequency (hence IDF) can be tuned."""
    ids: "list[UUID]" = []
    dim = cue_vec.shape[0]
    for i in range(n_local):
        alpha = 0.3 + 0.6 * float(rng.random())
        perp = _unit_vec(rng, dim)
        perp = perp - float(np.dot(perp, cue_vec)) * cue_vec
        pn = float(np.linalg.norm(perp))
        if pn > 0.0:
            perp = perp / pn
        beta = (1.0 - alpha ** 2) ** 0.5
        vec = alpha * cue_vec + beta * perp
        vn = float(np.linalg.norm(vec))
        if vn > 0.0:
            vec = vec / vn
        text = f"{text_prefix} local filler record {i}"
        if extra_tokens:
            text = f"{text} {' '.join(extra_tokens)}"
        rec = _make_eval_record(vec.tolist(), text, rng)
        _insert_raw(store, rec)
        ids.append(rec.id)
    return ids


def _cosine_vec(rng: "np.random.Generator", ref_vec: np.ndarray, alpha: float, dim: int) -> np.ndarray:
    """Unit vector at exact cosine ``alpha`` to ``ref_vec`` (a fresh random
    direction in the orthogonal complement otherwise)."""
    perp = _unit_vec(rng, dim)
    perp = perp - float(np.dot(perp, ref_vec)) * ref_vec
    pn = float(np.linalg.norm(perp))
    if pn > 0.0:
        perp = perp / pn
    beta = max(0.0, 1.0 - alpha ** 2) ** 0.5
    vec = alpha * ref_vec + beta * perp
    vn = float(np.linalg.norm(vec))
    return vec / vn if vn > 0.0 else vec


def build_minimal_fixture_corpus(
    seed: int = 0,
    store_path: "Path | str | None" = None,
    n_hub_filler: int = 230,
) -> FixtureCorpus:
    """Build the smallest synthetic corpus that provokes all four
    ``ALL_CAUSES`` misses, plus one high-confidence served cue.

    Geometry per cause (verified empirically against the real production
    dispatch path, not derived analytically):

    - ``rank_budget_dropped``: gold shares the hub cue direction at a
      cosine comfortably inside the ANN candidate window (survives every
      per-source cap, becomes a pool node) but ranks behind >=20 stronger
      decoys sharing the same direction -- past any reasonable final
      top-K response cutoff.
    - ``spread_did_not_reach``: gold is near-orthogonal to the hub cue
      direction (ranks far beyond the ANN candidate window, like every
      other source) and carries no edge at all into the post-cap pool --
      the ``EDGE_LESS_ISLAND`` geometry from ``build_eval_set``, adapted
      to the four-cause taxonomy.
    - ``discovery_excluded``: gold is near-orthogonal to a cue-local
      direction (no source returns it) but carries one genuine edge into
      a pool node -- deliberately given LESS weight than that pool
      node's five other edges, so the top-5-per-node 2-hop traversal
      never surfaces it even though the edge is real (confirmed via the
      raw edges table). The cue is ordinary prose (no identifier-shaped
      token), so the lexical rescue path never applies either.
    - ``lexical_cold_or_idf_gated``: gold is near-orthogonal to a
      cue-local direction (no source returns it) with the same
      below-top-5-weight edge into a pool node, but its cue carries a
      single identifier-shaped token that also appears in enough local
      filler records to be gated below the production minimum IDF.
    """
    from iai_mcp.store import MemoryStore
    from bench.neural_map import _BenchEmbedder

    import os as _os
    dim = int(_os.environ.get("IAI_MCP_EMBED_DIM", "384"))

    rng = np.random.default_rng(seed)
    wrapped_embedder = _BenchEmbedder(base_seed=seed, dim=dim)
    shim = _VectorInjectionEmbedder(wrapped_embedder)

    tmpdir = None
    if store_path is None:
        tmpdir = tempfile.TemporaryDirectory(prefix="iai-recall-quality-")
        store_path = Path(tmpdir.name)
    store = MemoryStore(path=Path(store_path))

    cues: "list[dict]" = []
    gold_literal_surface: "dict[str, str]" = {}

    # ------------------------------------------------------------------
    # Hub direction: shared by the rank-budget and spread cues, mirroring
    # build_eval_set's cue-reuse pattern.
    # ------------------------------------------------------------------
    hub_vec = _unit_vec(rng, dim)
    hub_sentinel = shim.register("hub_cue", hub_vec.tolist())

    # 25 decoys ranked strictly above the rank-budget gold.
    for i in range(25):
        alpha = 0.97 + 0.02 * float(rng.random())
        vec = _cosine_vec(rng, hub_vec, alpha, dim)
        _insert_raw(store, _make_eval_record(vec.tolist(), f"hub decoy record {i}", rng))

    # rank_budget_dropped gold: moderate cosine, ranked behind the 25 decoys.
    rank_vec = _cosine_vec(rng, hub_vec, 0.80, dim)
    rank_gold_text = "quarterly budget review notes for the infrastructure rollout"
    rank_gold = _make_eval_record(rank_vec.tolist(), rank_gold_text, rng)
    _insert_raw(store, rank_gold)
    gold_literal_surface[str(rank_gold.id)] = rank_gold_text

    # served cue: highest-cosine record of all, always ranks first -- the
    # A/A discordance anchor (rank moves under filler-seed variation).
    served_vec = _cosine_vec(rng, hub_vec, 0.995, dim)
    served_gold_text = "onboarding checklist for a new infrastructure teammate"
    served_gold = _make_eval_record(served_vec.tolist(), served_gold_text, rng)
    _insert_raw(store, served_gold)
    gold_literal_surface[str(served_gold.id)] = served_gold_text

    # Remaining generic hub filler mass -- pushes the corpus past
    # K_CANDIDATES=200 so an orthogonal target ranks beyond the ANN window.
    n_remaining = max(0, n_hub_filler - 25)
    for i in range(n_remaining):
        alpha = 0.3 + 0.45 * float(rng.random())
        vec = _cosine_vec(rng, hub_vec, alpha, dim)
        _insert_raw(store, _make_eval_record(vec.tolist(), f"hub generic filler {i}", rng))

    # spread_did_not_reach gold: near-orthogonal to the hub direction, no edges.
    spread_raw = _unit_vec(rng, dim)
    spread_raw = spread_raw - float(np.dot(spread_raw, hub_vec)) * hub_vec
    srn = float(np.linalg.norm(spread_raw))
    if srn > 0.0:
        spread_raw = spread_raw / srn
    spread_gold_text = "isolated archival record with no relational context at all"
    spread_gold = _make_eval_record(spread_raw.tolist(), spread_gold_text, rng)
    _insert_raw(store, spread_gold)
    gold_literal_surface[str(spread_gold.id)] = spread_gold_text

    # ------------------------------------------------------------------
    # discovery_excluded: its own cue-local pool + an engineered
    # below-top-5-weight edge from a pool node to the gold.
    # ------------------------------------------------------------------
    excluded_cue_text = "how does the archive protocol handle stored records"
    excluded_cue_vec = np.array(wrapped_embedder.embed(excluded_cue_text), dtype=np.float32)
    ecn = float(np.linalg.norm(excluded_cue_vec))
    if ecn > 0.0:
        excluded_cue_vec = excluded_cue_vec / ecn
    excluded_pool_ids = _build_local_pool(
        rng, store, excluded_cue_vec, n_local=8, text_prefix="archive protocol",
    )

    excluded_raw = _unit_vec(rng, dim)
    excluded_raw = excluded_raw - float(np.dot(excluded_raw, excluded_cue_vec)) * excluded_cue_vec
    exn = float(np.linalg.norm(excluded_raw))
    if exn > 0.0:
        excluded_raw = excluded_raw / exn
    excluded_gold_text = "vault decommission log entry sealed for the winter migration"
    excluded_gold = _make_eval_record(excluded_raw.tolist(), excluded_gold_text, rng)
    _insert_raw(store, excluded_gold)
    gold_literal_surface[str(excluded_gold.id)] = excluded_gold_text

    # ------------------------------------------------------------------
    # lexical_cold_or_idf_gated: its own cue-local pool, an identifier
    # token shared with enough fillers to be IDF-gated, and the same
    # below-top-5-weight edge trick.
    # ------------------------------------------------------------------
    lexical_token = "gateway_auth_token_zzqpx"
    # Every non-identifier word here is drawn from the junk-filler template
    # below so its own corpus document-frequency is high (~half the
    # corpus) -- the IDF gate reads the QUERY'S OWN max idf across every
    # token, so a rare companion word (e.g. "status", "right now") would
    # keep the gate open regardless of how common the identifier token
    # itself is.
    lexical_cue_text = f"unrelated background note mentioning {lexical_token} in passing"
    lexical_cue_vec = np.array(wrapped_embedder.embed(lexical_cue_text), dtype=np.float32)
    lcn = float(np.linalg.norm(lexical_cue_vec))
    if lcn > 0.0:
        lexical_cue_vec = lexical_cue_vec / lcn
    lexical_pool_ids = _build_local_pool(
        rng, store, lexical_cue_vec, n_local=8, text_prefix="gateway status",
        extra_tokens=[lexical_token],
    )
    # Extra IDF-suppression fillers: push the token's corpus frequency up
    # without perturbing the ANN geometry above (scattered on a separate
    # random direction so they never compete for cosine rank).
    for i in range(260):
        junk_vec = _unit_vec(rng, dim)
        _insert_raw(
            store,
            _make_eval_record(
                junk_vec.tolist(),
                f"unrelated background note {i} mentioning {lexical_token} in passing",
                rng,
            ),
        )

    lexical_raw = _unit_vec(rng, dim)
    lexical_raw = lexical_raw - float(np.dot(lexical_raw, lexical_cue_vec)) * lexical_cue_vec
    lxn = float(np.linalg.norm(lexical_raw))
    if lxn > 0.0:
        lexical_raw = lexical_raw / lxn
    lexical_gold_text = f"the {lexical_token} rotated last night in the token store"
    lexical_gold = _make_eval_record(lexical_raw.tolist(), lexical_gold_text, rng)
    _insert_raw(store, lexical_gold)
    gold_literal_surface[str(lexical_gold.id)] = lexical_gold_text

    from iai_mcp.store import flush_edge_buffer, flush_record_buffer
    flush_record_buffer(store)
    flush_edge_buffer(store)

    # Engineer the below-top-5-weight edges now that every record id exists.
    excluded_pool_node = excluded_pool_ids[0]
    for other in excluded_pool_ids[1:6]:
        _weighted_edge(store, excluded_pool_node, other, _STRONG_EDGE_DELTA)
    _weighted_edge(store, excluded_pool_node, excluded_gold.id, _WEAK_EDGE_DELTA)

    lexical_pool_node = lexical_pool_ids[0]
    for other in lexical_pool_ids[1:6]:
        _weighted_edge(store, lexical_pool_node, other, _STRONG_EDGE_DELTA)
    _weighted_edge(store, lexical_pool_node, lexical_gold.id, _WEAK_EDGE_DELTA)

    flush_edge_buffer(store)

    from iai_mcp.retrieve import build_runtime_graph
    build_runtime_graph(store)

    cues = [
        {
            "cue_id": "rank_budget_dropped_1",
            "cue_text": hub_sentinel,
            "gold_record_id": str(rank_gold.id),
            "cluster_id": "hub",
            "quote": "quarterly budget review notes",
            "split": "dev",
        },
        {
            "cue_id": "served_1",
            "cue_text": hub_sentinel,
            "gold_record_id": str(served_gold.id),
            "cluster_id": "hub",
            "quote": "onboarding checklist for a new infrastructure teammate",
            "split": "dev",
        },
        {
            "cue_id": "spread_did_not_reach_1",
            "cue_text": hub_sentinel,
            "gold_record_id": str(spread_gold.id),
            "cluster_id": "hub",
            "quote": "isolated archival record",
            "split": "dev",
        },
        {
            "cue_id": "discovery_excluded_1",
            "cue_text": excluded_cue_text,
            "gold_record_id": str(excluded_gold.id),
            "cluster_id": "archive",
            "quote": "vault decommission log entry",
            "split": "dev",
        },
        {
            "cue_id": "lexical_cold_or_idf_gated_1",
            "cue_text": lexical_cue_text,
            "gold_record_id": str(lexical_gold.id),
            "cluster_id": "gateway",
            "quote": lexical_token,
            "split": "dev",
        },
    ]

    for cue in cues:
        validate_cue_schema(cue)
        _validate_quotation_backed(cue, gold_literal_surface[cue["gold_record_id"]])

    return FixtureCorpus(
        store=store,
        shim=shim,
        embedder=wrapped_embedder,
        cues=cues,
        gold_literal_surface=gold_literal_surface,
        tmpdir=tmpdir,
    )


# ---------------------------------------------------------------------------
# Full sized fixture -- reuses the same cause geometry as
# build_minimal_fixture_corpus, replicated across many independent
# instances, sized from a measured per-cue wall-clock.
# ---------------------------------------------------------------------------


def _force_lexical_gate_all(
    store: Any,
    rng: "np.random.Generator",
    dim: int,
    specs: "list[tuple[dict, UUID, str]]",
    *,
    max_rounds: int = 20,
    batch: int = 60,
) -> None:
    """Insert copies of each cue's own text as unrelated filler content,
    on fresh random directions, until the production IDF gate
    (``pipeline.LEX_FUSION_MIN_IDF``) excludes every listed gold id from
    the warm BM25 lane's own results -- SIMULTANEOUSLY, in lockstep
    rounds across every lexical cue in ``specs``.

    Gating cues ONE AT A TIME (each fully converged before moving to the
    next) is unsound: a later cue's own IDF-suppression insertions grow
    the corpus denominator, which can silently push an EARLIER cue's
    already-gated query back open (that cue's own token count did not
    grow, but the corpus it is measured against did). Converging every
    cue together against a single, monotonically growing corpus is the
    only construction that is still gated once the corpus stops growing.

    Fails loud if any cue's gate cannot be forced within the round
    budget: a silent partial-gate would mislabel the cause.
    """
    from iai_mcp.pipeline import LEX_FUSION_K, LEX_FUSION_MIN_IDF
    from iai_mcp.store import flush_record_buffer

    pending = list(specs)
    for _ in range(max_rounds):
        store.lexical_search("warm", k=1)
        still_open: "list[tuple[dict, UUID, str]]" = []
        for cue, gold_id, cue_text in pending:
            hits = store.lexical_query_warm(cue_text, k=LEX_FUSION_K, min_idf=LEX_FUSION_MIN_IDF)
            served_ids = set()
            for rid, _score in hits:
                try:
                    served_ids.add(UUID(str(rid)))
                except (TypeError, ValueError):
                    continue
            if gold_id in served_ids:
                still_open.append((cue, gold_id, cue_text))
        if not still_open:
            return
        for _cue, _gold_id, cue_text in still_open:
            for _i in range(batch):
                junk_vec = _unit_vec(rng, dim)
                _insert_raw(store, _make_eval_record(junk_vec.tolist(), cue_text, rng))
        flush_record_buffer(store)
        pending = still_open

    raise RuntimeError(
        "could not force the production IDF gate to exclude gold ids "
        f"{[str(g) for _c, g, _t in pending]} within {max_rounds} rounds -- "
        "the lexical_cold_or_idf_gated cause cannot be honestly provoked "
        "at this corpus size"
    )


def _build_edge_engineered_cue(
    rng: "np.random.Generator",
    store: Any,
    dim: int,
    wrapped_embedder: Any,
    *,
    cue_id: str,
    cue_text: str,
    gold_text: str,
    quote: str,
    cluster_id: str,
    pool_size: int,
    text_prefix: str,
    idf_gate_gold_id: "UUID | None" = None,
) -> "tuple[dict, str, UUID]":
    """Shared construction for ``discovery_excluded`` and
    ``lexical_cold_or_idf_gated`` cues: a small cue-local ANN pool, a gold
    record near-orthogonal to the cue (no source discovers it), and one
    genuine but structurally-unreachable edge from a pool node into gold
    (weight below the top-5-per-node spread cap, but present in the raw
    edges table).

    Returns ``(cue_dict, gold_literal_surface, gold_id)`` -- the caller
    inserts the cue dict into the running cue list and gold_literal_surface
    map, and (for the lexical cause) may follow up with
    ``_force_lexical_gate``.
    """
    cue_vec = np.array(wrapped_embedder.embed(cue_text), dtype=np.float32)
    cn = float(np.linalg.norm(cue_vec))
    if cn > 0.0:
        cue_vec = cue_vec / cn

    pool_ids = _build_local_pool(rng, store, cue_vec, n_local=pool_size, text_prefix=text_prefix)

    gold_vec = _cosine_vec(rng, cue_vec, 0.0, dim)
    gold_rec = _make_eval_record(gold_vec.tolist(), gold_text, rng)
    _insert_raw(store, gold_rec)

    pool_node = pool_ids[0]
    for other in pool_ids[1:6]:
        _weighted_edge(store, pool_node, other, _STRONG_EDGE_DELTA)
    _weighted_edge(store, pool_node, gold_rec.id, _WEAK_EDGE_DELTA)

    cue = {
        "cue_id": cue_id,
        "cue_text": cue_text,
        "gold_record_id": str(gold_rec.id),
        "cluster_id": cluster_id,
        "quote": quote,
        "split": "dev",
    }
    return cue, gold_text, gold_rec.id


def build_full_fixture_corpus(
    seed: int = 0,
    store_path: "Path | str | None" = None,
    *,
    n_rank_cues: int = 12,
    n_served_cues: int = 8,
    n_spread_cues: int = 6,
    n_excluded_cues: int = 9,
    n_lexical_cues: int = 9,
    cluster_group_size: int = 3,
    n_hub_filler: int = 260,
    filler_seed: "int | None" = None,
    n_filler_noise_decoys: int = 0,
) -> "tuple[FixtureCorpus, str]":
    """Build the full sized fixture, replicating
    ``build_minimal_fixture_corpus``'s validated geometry across many
    independent cue instances. Returns ``(corpus, n_rationale)``.

    Cluster assignment: hub-sharing cues (rank_budget_dropped, served,
    spread) are grouped ``cluster_group_size`` at a time since they
    genuinely share corpus context (the same hub filler mass); every
    edge-engineered cue (discovery_excluded, lexical_cold_or_idf_gated)
    gets its own cluster since each is built from its own dedicated local
    pool -- clusters reflect real shared construction context, never an
    arbitrary label.

    ``filler_seed`` (default: same as ``seed``) seeds ONLY the
    ``n_filler_noise_decoys`` records inserted at the same cosine range as
    the served cues -- every gold vector, cue definition, and edge is
    drawn from ``seed`` regardless of ``filler_seed``, so two builds with
    the same ``seed`` and DIFFERENT ``filler_seed`` share an identical
    gold/cue set and differ ONLY in this nuisance content. Those noise
    decoys interleave in rank with the served cues, so a different
    ``filler_seed`` genuinely moves served cues' exact rank position (or
    occasionally serves/misses a boundary one) without touching any
    engineered miss cue's cause.
    """
    from iai_mcp.store import MemoryStore, flush_edge_buffer, flush_record_buffer
    from iai_mcp.retrieve import build_runtime_graph
    from bench.neural_map import _BenchEmbedder

    import os as _os
    dim = int(_os.environ.get("IAI_MCP_EMBED_DIM", "384"))

    rng = np.random.default_rng(seed)
    filler_rng = np.random.default_rng(filler_seed if filler_seed is not None else seed)
    wrapped_embedder = _BenchEmbedder(base_seed=seed, dim=dim)
    shim = _VectorInjectionEmbedder(wrapped_embedder)

    tmpdir = None
    if store_path is None:
        tmpdir = tempfile.TemporaryDirectory(prefix="iai-recall-quality-full-")
        store_path = Path(tmpdir.name)
    store = MemoryStore(path=Path(store_path))

    cues: "list[dict]" = []
    gold_literal_surface: "dict[str, str]" = {}

    # ------------------------------------------------------------------
    # Shared hub mass: rank_budget_dropped, served, and spread cues all
    # draw on this one filler pool, replicated with independently varied
    # gold vectors per cue.
    # ------------------------------------------------------------------
    hub_vec = _unit_vec(rng, dim)
    hub_sentinel = shim.register("hub_cue", hub_vec.tolist())

    n_decoy = 25
    for i in range(n_decoy):
        alpha = 0.97 + 0.02 * float(rng.random())
        vec = _cosine_vec(rng, hub_vec, alpha, dim)
        _insert_raw(store, _make_eval_record(vec.tolist(), f"hub decoy record {i}", rng))

    for i in range(n_rank_cues):
        alpha = 0.75 + 0.10 * float(rng.random())
        vec = _cosine_vec(rng, hub_vec, alpha, dim)
        text = f"quarterly budget review notes for rollout cluster {i}"
        rec = _make_eval_record(vec.tolist(), text, rng)
        _insert_raw(store, rec)
        gold_literal_surface[str(rec.id)] = text
        cues.append({
            "cue_id": f"rank_budget_dropped_{i}",
            "cue_text": hub_sentinel,
            "gold_record_id": str(rec.id),
            "cluster_id": f"hub_rank_{i // cluster_group_size}",
            "quote": "quarterly budget review notes",
            "split": "dev",
        })

    for i in range(n_served_cues):
        alpha = 0.990 + 0.008 * float(rng.random())
        vec = _cosine_vec(rng, hub_vec, alpha, dim)
        text = f"onboarding checklist entry {i} for a new infrastructure teammate"
        rec = _make_eval_record(vec.tolist(), text, rng)
        _insert_raw(store, rec)
        gold_literal_surface[str(rec.id)] = text
        cues.append({
            "cue_id": f"served_{i}",
            "cue_text": hub_sentinel,
            "gold_record_id": str(rec.id),
            "cluster_id": f"hub_served_{i // cluster_group_size}",
            "quote": "onboarding checklist entry",
            "split": "dev",
        })

    # Nuisance-only noise decoys: same cosine range as the served cues, so
    # they interleave directly with served-cue ranking. Drawn from
    # ``filler_rng`` (NOT ``rng``) -- the only insertion in this whole
    # builder whose content depends on ``filler_seed`` rather than
    # ``seed``, so an A/A replay pair shares every gold vector and cue
    # definition and differs ONLY here.
    for i in range(n_filler_noise_decoys):
        alpha = 0.990 + 0.008 * float(filler_rng.random())
        vec = _cosine_vec(filler_rng, hub_vec, alpha, dim)
        _insert_raw(
            store,
            _make_eval_record(vec.tolist(), f"hub filler-noise decoy {i}", filler_rng),
        )

    n_remaining = max(0, n_hub_filler - n_decoy)
    for i in range(n_remaining):
        alpha = 0.3 + 0.45 * float(rng.random())
        vec = _cosine_vec(rng, hub_vec, alpha, dim)
        _insert_raw(store, _make_eval_record(vec.tolist(), f"hub generic filler {i}", rng))

    for i in range(n_spread_cues):
        vec = _cosine_vec(rng, hub_vec, 0.0, dim)
        text = f"isolated archival record {i} with no relational context at all"
        rec = _make_eval_record(vec.tolist(), text, rng)
        _insert_raw(store, rec)
        gold_literal_surface[str(rec.id)] = text
        cues.append({
            "cue_id": f"spread_did_not_reach_{i}",
            "cue_text": hub_sentinel,
            "gold_record_id": str(rec.id),
            "cluster_id": f"hub_spread_{i // cluster_group_size}",
            "quote": "isolated archival record",
            "split": "dev",
        })

    flush_record_buffer(store)
    flush_edge_buffer(store)

    # ------------------------------------------------------------------
    # discovery_excluded: each instance is its own dedicated cluster.
    # ------------------------------------------------------------------
    for i in range(n_excluded_cues):
        cue_text = f"how does the archive protocol handle stored records in bucket {i}"
        gold_text = f"vault decommission log entry {i} sealed for the winter migration"
        cue, _, gold_id = _build_edge_engineered_cue(
            rng, store, dim, wrapped_embedder,
            cue_id=f"discovery_excluded_{i}",
            cue_text=cue_text,
            gold_text=gold_text,
            quote="vault decommission log entry",
            cluster_id=f"archive_{i}",
            pool_size=8,
            text_prefix=f"archive protocol bucket {i}",
        )
        cues.append(cue)
        gold_literal_surface[str(gold_id)] = gold_text
        flush_edge_buffer(store)

    # ------------------------------------------------------------------
    # lexical_cold_or_idf_gated: each instance is its own dedicated
    # cluster with its own identifier token. Every instance's base
    # geometry is built FIRST, then ALL of them are IDF-gated TOGETHER in
    # lockstep rounds against the FINAL corpus size -- gating one cue at a
    # time while later cues keep growing the corpus would silently
    # re-open an earlier cue's gate (the query's own max-idf rises as the
    # corpus denominator grows without that cue's specific token growing
    # with it).
    # ------------------------------------------------------------------
    lexical_specs: "list[tuple[dict, UUID, str]]" = []
    for i in range(n_lexical_cues):
        token = f"gateway_auth_token_zzq{i:03d}"
        cue_text = f"unrelated background note mentioning {token} in passing"
        gold_text = f"the {token} rotated last night in the token store"
        cue, _, gold_id = _build_edge_engineered_cue(
            rng, store, dim, wrapped_embedder,
            cue_id=f"lexical_cold_or_idf_gated_{i}",
            cue_text=cue_text,
            gold_text=gold_text,
            quote=token,
            cluster_id=f"gateway_{i}",
            pool_size=8,
            text_prefix=f"gateway status bucket {i}",
        )
        cues.append(cue)
        gold_literal_surface[str(gold_id)] = gold_text
        lexical_specs.append((cue, gold_id, cue_text))

    flush_record_buffer(store)
    flush_edge_buffer(store)
    _force_lexical_gate_all(store, rng, dim, lexical_specs)

    flush_record_buffer(store)
    flush_edge_buffer(store)
    build_runtime_graph(store)

    # ------------------------------------------------------------------
    # Verify against REAL production dispatch, never assume: graph
    # structure (betweenness-driven rich-club membership in particular)
    # can occasionally sweep an engineered miss into a different outcome
    # than the geometry alone predicts. Every kept cue's cue_id prefix is
    # checked against its ACTUAL classified outcome; a cue whose real
    # behavior disagrees is dropped rather than shipped mislabeled.
    # ------------------------------------------------------------------
    from bench.recall_quality_verdict import _warm_synthetic_store, run_baseline_pass

    _warm_synthetic_store(store, wrapped_embedder)
    verified_rows = run_baseline_pass(store, shim, cues)
    verified_cues: "list[dict]" = []
    dropped: "list[str]" = []
    for cue, row in zip(cues, verified_rows):
        expected = cue["cue_id"].rsplit("_", 1)[0]
        if expected == "served":
            ok = row["served"]
        else:
            ok = (not row["served"]) and row.get("cause") == expected
        if ok:
            verified_cues.append(cue)
        else:
            dropped.append(cue["cue_id"])
    cues = verified_cues

    for cause in (
        "discovery_excluded", "spread_did_not_reach",
        "rank_budget_dropped", "lexical_cold_or_idf_gated",
    ):
        if not any(c["cue_id"].rsplit("_", 1)[0] == cause for c in cues):
            raise RuntimeError(
                f"cause {cause!r} was not verified in any surviving cue -- "
                f"dropped cues this build: {dropped}"
            )
    if not any(c["cue_id"].rsplit("_", 1)[0] == "served" for c in cues):
        raise RuntimeError(
            f"no 'served' cue survived verification -- dropped cues this build: {dropped}"
        )

    for cue in cues:
        validate_cue_schema(cue)
        _validate_quotation_backed(cue, gold_literal_surface[cue["gold_record_id"]])

    corpus = FixtureCorpus(
        store=store,
        shim=shim,
        embedder=wrapped_embedder,
        cues=cues,
        gold_literal_surface=gold_literal_surface,
        tmpdir=tmpdir,
    )
    n_rationale = (
        f"{len(cues)} cues verified against real dispatch (built "
        f"{n_rank_cues + n_served_cues + n_spread_cues + n_excluded_cues + n_lexical_cues}, "
        f"{len(dropped)} dropped for disagreeing with their engineered cause: {dropped}) -- "
        "target sized from a per-cue wall-clock measurement's recommended full N "
        "(bench.recall_quality_verdict --smoke), split across "
        f"{n_rank_cues} rank_budget_dropped, {n_served_cues} served, "
        f"{n_spread_cues} spread_did_not_reach, {n_excluded_cues} discovery_excluded, "
        f"and {n_lexical_cues} lexical_cold_or_idf_gated instances."
    )
    return corpus, n_rationale


def _assign_cluster_splits(
    cues: "list[dict]", seed: int, dev_fraction: float = 0.7,
) -> None:
    """Assign ``split`` in place at CLUSTER granularity: every cue sharing
    a ``cluster_id`` gets the SAME split, so a cluster never straddles
    dev/holdout."""
    cluster_ids = sorted({cue["cluster_id"] for cue in cues})
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(len(cluster_ids))
    shuffled = [cluster_ids[i] for i in order]
    n_dev = max(1, int(round(len(shuffled) * dev_fraction)))
    n_dev = min(n_dev, len(shuffled) - 1) if len(shuffled) > 1 else len(shuffled)
    dev_clusters = set(shuffled[:n_dev])
    for cue in cues:
        cue["split"] = "dev" if cue["cluster_id"] in dev_clusters else "holdout"


def write_fixture(path: "Path | str", seed: int = 0, **build_kwargs: Any) -> dict:
    """Build the full fixture, assign the cluster-level dev/holdout split,
    and write the self-contained committed JSON artifact (gold
    ``literal_surface`` included per cue so integrity tests never need a
    store or an embedder)."""
    corpus, n_rationale = build_full_fixture_corpus(seed=seed, **build_kwargs)
    try:
        _assign_cluster_splits(corpus.cues, seed)
        cue_rows = []
        for cue in corpus.cues:
            row = dict(cue)
            row["gold_literal_surface"] = corpus.gold_literal_surface[cue["gold_record_id"]]
            cue_rows.append(row)

        dev_count = sum(1 for c in cue_rows if c["split"] == "dev")
        holdout_count = len(cue_rows) - dev_count
        payload = {
            "seed": seed,
            "cue_count": len(cue_rows),
            "dev_count": dev_count,
            "holdout_count": holdout_count,
            "n_rationale": n_rationale,
            "cues": cue_rows,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload
    finally:
        corpus.close()
