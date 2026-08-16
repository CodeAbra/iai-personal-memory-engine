"""Exhaustive-cosine oracle, recall accuracy metrics, and reproducible eval set.

This module provides:

1. oracle_top_k — the exhaustive-cosine ground truth.

   is_semantic_hit / semantic_equivalence_class / class_cosine_rank — the
   dedup-safe hit primitives run_baseline scores with: a recall is a hit when a
   returned record IS the target memory (exact id, exact-embedding equivalence
   class, or within SEMANTIC_HIT_EPS cosine), which is scoreable even when many
   records share one embedding.

   recall_at_k / is_false_negative / mean_recall_at_k / false_negative_rate —
   oracle-overlap / oracle-rank-1 DIAGNOSTICS, not the baseline metric; kept for
   frontier-overlap inspection only.

2. Archetype / EvalCase / build_eval_set — a reproducible labelled cue→target
   eval set that deliberately seeds the three dominant false-negative archetypes
   so recall@k is less than 1.0 and each miss is attributable.

3. run_accuracy — end-to-end graph-vs-oracle runner: calls recall_for_response
   for each cue, scores against the oracle, and attributes each miss using a
   documented degrade→island→beyond-cutoff precedence.

4. run_baseline — offline baseline runner against a supplied isolated store copy.
   Measures recall through the PRODUCTION dispatch entry point
   (``core.dispatch(store, "memory_recall", ...)``) so latency and ranking are
   representative, not the ~10x-slower hand-assembled exhaustive-cosine path.
   Dedup-aware sampling of n_samples records, a perturbed-vector cue per target
   injected through the production embedder seam (independent of the record's
   literal_surface so the exact-term/trigram boosts do not fire), a warm-up cue
   plus a per-cue degrade reset so latency reflects non-degraded recall,
   semantic-hit scoring, and a JSON summary.  Entry point for the
   isolated-snapshot baseline measurement (``python -m bench.recall_accuracy``).

The oracle and runner are measurement-only.  They set existing inputs (the
module latency global, cue text, corpus edges) to provoke each archetype; they
never edit K_CANDIDATES, the degrade rule, or any src/iai_mcp/ recall logic.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np

# ---------------------------------------------------------------------------
# Standalone-script sys.path preamble (no-op when imported as a package member)
# ---------------------------------------------------------------------------
_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)

if TYPE_CHECKING:
    from iai_mcp.store import MemoryStore
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.community import CommunityAssignment
    from iai_mcp.embed import Embedder


# ---------------------------------------------------------------------------
# Active-corpus SQL predicate — identical to the one the production graph pool
# uses and to `_label_vectors` in the HNSW parity test.
# ---------------------------------------------------------------------------
_ACTIVE_CORPUS_SQL = (
    "SELECT id, embedding FROM records"
    " WHERE tombstoned_at IS NULL"
    " AND COALESCE(embedding_pending, 0) = 0"
)


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------


def oracle_top_k(store: "MemoryStore", cue_embedding: np.ndarray, k: int) -> list[UUID]:
    """Return the top-k record ids by cosine similarity over the full active corpus.

    This is the exhaustive cosine ground truth: every active record is scored,
    so no record in the active corpus can be missed (unlike the graph path whose
    candidacy is bounded to a fixed frontier).  Run offline only — this is
    O(N·D) and intentionally absent from the production recall path.

    Read-only: opens no writer, calls no mutator, and does not change any
    index, count, or store state.

    Args:
        store: an open MemoryStore instance.
        cue_embedding: 1-D float32 array of the cue vector (any norm; the
            oracle normalises it internally before scoring).
        k: number of top records to return.  If k >= N (corpus size) all N
            active records are returned ordered by cosine.

    Returns:
        Ordered list of UUID values (MemoryRecord.id), most-similar first.
        Returns an empty list when the corpus has no active records.
    """
    db = store.db

    # --- Read active corpus from SQLite under the shared connection lock ----
    with db._conn_lock:
        rows = db._conn.execute(_ACTIVE_CORPUS_SQL).fetchall()

    if not rows:
        return []

    # --- Build matrix of L2-normalised float32 embeddings ------------------
    # Embedding dimension is inferred from the buffer so both the real 384-d
    # model and test monkeypatches (small dim) work without hard-coding.
    ids: list[UUID] = []
    vec_list: list[np.ndarray] = []
    for row in rows:
        ids.append(UUID(row["id"]))
        v = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        norm = float(np.linalg.norm(v))
        if norm > 0.0:
            v = v / norm
        vec_list.append(v)

    vecs = np.stack(vec_list)  # shape (N, D)

    # --- L2-normalise the cue ----------------------------------------------
    cue = cue_embedding.astype(np.float32)
    cue_norm = float(np.linalg.norm(cue))
    if cue_norm > 0.0:
        cue = cue / cue_norm

    # --- Exhaustive cosine scores and top-k sort ---------------------------
    sims = vecs @ cue  # shape (N,)
    effective_k = min(k, len(ids))
    top_indices = np.argsort(-sims)[:effective_k]
    return [ids[i] for i in top_indices]


# ---------------------------------------------------------------------------
# Recall metrics (pure host-side, no store, no recall internals)
# ---------------------------------------------------------------------------


# Cosine floor for "same memory": a returned record within this cosine of the
# target counts as a hit even if its id differs.  Chosen below the ~0.989 cue
# perturbation floor and above the [0.70, 0.92) Hebbian link band, so it does
# not collapse into "any near neighbour" — it captures only records that are
# effectively the same memory as the target.  Reported as a run condition in the
# baseline JSON and the baseline summary.
SEMANTIC_HIT_EPS = 0.98


def semantic_equivalence_class(
    target_id: UUID,
    id_to_vec: "dict[UUID, np.ndarray]",
) -> "set[UUID]":
    """Return the set of ids whose stored embedding is byte-identical to the target's.

    Exact-duplicate embedding clusters (repeated capture / session-summary text)
    make the exact-``target_id`` recall metric structurally unscoreable: when
    >= k records share the target's embedding, the tie-mates sort ahead of the
    specific id, so neither the oracle nor a perfect recall can surface that one
    id in the top-k.  The equivalence class captures every id that is the same
    memory as the target at cosine 1.0, so a hit on any of them is a correct
    recall of "that memory".

    Args:
        target_id: the sampled target's id.
        id_to_vec: map from id to its unit-normalised float32 embedding.

    Returns:
        Set of ids (including ``target_id`` itself) whose embedding bytes equal
        the target's.  Bytes are compared on the raw float32 buffer so numerically
        identical vectors group together without floating-point tolerance.
    """
    target_vec = id_to_vec.get(target_id)
    if target_vec is None:
        return {target_id}
    target_bytes = target_vec.astype(np.float32).tobytes()
    cls: set[UUID] = set()
    for rid, v in id_to_vec.items():
        if v.astype(np.float32).tobytes() == target_bytes:
            cls.add(rid)
    cls.add(target_id)
    return cls


def is_semantic_hit(
    returned_ids: list[UUID],
    target_id: UUID,
    target_vec: "np.ndarray",
    id_to_vec: "dict[UUID, np.ndarray]",
    target_class: "set[UUID]",
    k: int,
    eps: float = SEMANTIC_HIT_EPS,
) -> bool:
    """True iff any of the first ``k`` returned ids is the target memory.

    A returned id counts as recalling the target when it is the exact target id,
    a member of the target's exact-embedding equivalence class, or a record whose
    stored embedding is within cosine ``eps`` of the target.  This is the honest
    hit definition under duplicate-embedding pollution: it asks "did recall return
    THIS memory" rather than "did recall return THIS specific id", which is
    unscoreable when many ids share one embedding.

    Args:
        returned_ids: ids from the recall path (only the first ``k`` are examined).
        target_id: the sampled target's id.
        target_vec: the target's unit-normalised float32 embedding.
        id_to_vec: map from id to unit-normalised float32 embedding for the corpus.
        target_class: the target's exact-embedding equivalence class
            (from ``semantic_equivalence_class``).
        k: rank cutoff applied to ``returned_ids``.
        eps: cosine floor for the near-duplicate branch.

    Returns:
        True when a member of the top-k satisfies the semantic-match predicate.
    """
    for rid in returned_ids[:k]:
        if rid == target_id or rid in target_class:
            return True
        v = id_to_vec.get(rid)
        if v is not None and float(np.dot(v, target_vec)) >= eps:
            return True
    return False


def class_cosine_rank(
    oracle_ids: list[UUID],
    target_class: "set[UUID]",
    not_found_rank: int,
) -> int:
    """Rank of the BEST-placed member of the target's equivalence class in the oracle list.

    Under duplicate pollution the specific ``target_id`` may sit deep in the
    oracle list behind its own twins, yielding a misleadingly poor rank.  The
    honest rank of the memory is the rank of whichever class member the oracle
    placed highest (0-based; 0 = oracle top-1).

    Args:
        oracle_ids: exhaustive-cosine oracle list, most-similar first.
        target_class: the target's exact-embedding equivalence class.
        not_found_rank: sentinel returned when no class member appears in
            ``oracle_ids`` (the caller excludes this from the rank distribution).

    Returns:
        The 0-based rank of the first class member found, or ``not_found_rank``.
    """
    for rank, rid in enumerate(oracle_ids):
        if rid in target_class:
            return rank
    return not_found_rank


def recall_at_k(
    returned_ids: list[UUID],
    oracle_ids: list[UUID],
    k: int,
) -> float:
    """Oracle-overlap diagnostic — NOT the baseline recall metric.

    Measures the fraction of the oracle top-k the returned set reproduces.  The
    baseline scores hits via ``is_semantic_hit`` (target-memory recall under
    duplicate-embedding pollution), NOT this oracle-overlap quantity.  Kept as a
    diagnostic for oracle-vs-graph frontier overlap.

    Args:
        returned_ids: ids returned by the recall path (may be longer than k;
            only the first k elements are considered).
        oracle_ids: ids from the exhaustive oracle (most-similar first).
        k: the rank cutoff applied to both lists.

    Returns:
        |returned[:k] ∩ oracle[:k]| / min(k, len(oracle_ids)).
        Returns 1.0 when the oracle is empty (nothing to miss).
    """
    denom = min(k, len(oracle_ids))
    if denom == 0:
        return 1.0
    returned_set = set(returned_ids[:k])
    oracle_set = set(oracle_ids[:k])
    return len(returned_set & oracle_set) / denom


def is_false_negative(
    returned_ids: list[UUID],
    oracle_ids: list[UUID],
    k: int,
) -> bool:
    """Oracle-rank-1 diagnostic — NOT the baseline false-negative metric.

    True iff the oracle's rank-1 result (oracle_ids[0]) is absent from the
    returned set.  The baseline derives false negatives from ``is_semantic_hit``
    (the negation of a target-memory hit), NOT from this oracle-rank-1 check,
    which keys on a specific id and is unscoreable under duplicate-embedding
    clusters.  Kept as a diagnostic only.

    Args:
        returned_ids: ids from the recall path (only first k are examined).
        oracle_ids: ids from the exhaustive oracle (most-similar first).
        k: rank cutoff applied to returned_ids.

    Returns:
        True when oracle_ids is non-empty and oracle_ids[0] is not in
        returned_ids[:k].  False when the oracle is empty (nothing can be missed).
    """
    if not oracle_ids:
        return False
    return oracle_ids[0] not in returned_ids[:k]


def mean_recall_at_k(
    pairs: list[tuple[list[UUID], list[UUID]]],
    k: int,
) -> float:
    """Mean recall@k over a collection of (returned, oracle) pairs.

    Args:
        pairs: list of (returned_ids, oracle_ids) tuples.
        k: rank cutoff passed to recall_at_k.

    Returns:
        Arithmetic mean of recall_at_k for each pair.
        Returns 1.0 for an empty pairs list.
    """
    if not pairs:
        return 1.0
    scores = [recall_at_k(ret, oracle, k) for ret, oracle in pairs]
    return float(np.mean(scores))


def false_negative_rate(
    pairs: list[tuple[list[UUID], list[UUID]]],
    k: int,
) -> float:
    """Fraction of (returned, oracle) pairs where the oracle top-1 is missed.

    Args:
        pairs: list of (returned_ids, oracle_ids) tuples.
        k: rank cutoff passed to is_false_negative.

    Returns:
        Fraction of pairs that are false negatives.
        Returns 0.0 for an empty pairs list.
    """
    if not pairs:
        return 0.0
    fn_count = sum(is_false_negative(ret, oracle, k) for ret, oracle in pairs)
    return fn_count / len(pairs)


# ---------------------------------------------------------------------------
# Archetype identifiers (functional names — no planning codes)
# ---------------------------------------------------------------------------

class Archetype(str, Enum):
    """The three dominant false-negative archetypes measurable in the graph path.

    Membership in each archetype is defined by an observable predicate on
    the corpus, not by expectation:

    COSINE_RANK_BEYOND_CUTOFF — the target's exhaustive-cosine rank over the
        whole pool exceeds the fixed candidate frontier (K_CANDIDATES=200), so
        it is only reachable via spread from a seed or via the rich-club.
        Provoked by inserting enough filler records whose vectors are closer
        to the cue than the target's vector.

    SPREAD_DISABLED_BY_PRIOR_LATENCY — the self-reinforcing latency degrade:
        after any recall above the threshold (~2 s), the next recall performs
        zero hops and scans only one community.  Provoked by pre-setting the
        module latency global above the threshold before the probe call.

    EDGE_LESS_ISLAND — the target has only a self-loop in the runtime graph
        because write-time pattern-separation found no neighbor within the link
        band [0.70, 0.92).  2-hop spread cannot reach or leave it.
        Provoked by inserting a target whose embedding is orthogonal (or
        near-orthogonal) to every other record so no Hebbian edge is seeded.
    """

    COSINE_RANK_BEYOND_CUTOFF = "COSINE_RANK_BEYOND_CUTOFF"
    SPREAD_DISABLED_BY_PRIOR_LATENCY = "SPREAD_DISABLED_BY_PRIOR_LATENCY"
    EDGE_LESS_ISLAND = "EDGE_LESS_ISLAND"


@dataclass
class EvalCase:
    """One labelled (cue_text, target_id, archetype) entry in the eval set.

    cue_text: the paraphrased / perturbed query string fed to recall_for_response.
        It is deliberately phrased away from the target's verbatim literal_surface
        so exact-term and trigram boosts (pipeline.py:897-899) do not trivially
        surface the target.

    target_id: the UUID of the record that the oracle considers most relevant
        (oracle rank 1 among all active records for the embedded cue_text).
        The runner checks whether this id is absent from recall_for_response's
        hit list to declare a false negative.

    archetype: which failure mode this case exercises.
    """

    cue_text: str
    target_id: UUID
    archetype: Archetype


# ---------------------------------------------------------------------------
# Internal helpers for eval-set construction
# ---------------------------------------------------------------------------

def _unit_vec(rng: np.random.Generator, dim: int) -> np.ndarray:
    """Generate a unit-normalised float32 vector via rng (no module-level random)."""
    v = rng.standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0.0 else v


def _make_eval_record(
    vec: list[float],
    text: str,
    rng: np.random.Generator,
) -> "object":  # MemoryRecord
    """Build a MemoryRecord with a pre-computed embedding — no embedder call.

    The record UUID is derived from rng so that two calls with the same
    seeded generator produce the same UUID sequence (fully reproducible).
    """
    from iai_mcp.types import MemoryRecord

    # Derive a deterministic UUID from rng (4 random uint32 values → 16 bytes)
    raw = rng.integers(0, 2**32, size=4, dtype=np.uint64).tobytes()[:16]
    rec_id = UUID(bytes=raw)

    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rec_id,
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=list(vec),
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )


def _insert_raw(store: "MemoryStore", rec: "object") -> None:
    """Insert a MemoryRecord into the store via the normal insert path.

    The insert path runs pattern-separation (ANN probe + edge seeding) — see
    store/_store.py.  Edges seeded here will be persisted to the EDGES_TABLE
    and later loaded by build_runtime_graph.
    """
    store.insert(rec)


def _flush(store: "MemoryStore") -> None:
    """Flush buffered records and edges to SQLite.

    Flush failures are NOT swallowed: a silent flush failure builds the runtime
    graph from an incomplete corpus, so the archetype guarantees (beyond-cutoff,
    island) no longer hold while the run still reports success.  Exceptions
    propagate so the eval-set build fails loudly instead of producing
    false-green supporting evidence.
    """
    from iai_mcp.store import flush_record_buffer, flush_edge_buffer

    flush_record_buffer(store)
    flush_edge_buffer(store)


def _seed_beyond_edge(
    store: "MemoryStore",
    graph: "MemoryGraph",
    beyond_id: UUID,
    island_id: UUID,
    spread_id: UUID,
) -> None:
    """Inject one Hebbian edge from the BEYOND target to its closest non-archetype
    neighbour in the corpus so the BEYOND target is structurally connected
    (pred_island=False) while the ISLAND target remains structurally isolated
    (pred_island=True).

    Without this injection, both BEYOND and ISLAND targets would have only a
    self-loop, making them identical under the _is_island predicate and preventing
    disjoint attribution.

    The edge is written directly to the SQLite edges table because the BEYOND
    target was inserted before the ANN index was populated — pattern-separation
    found no neighbours at write time and seeded no Hebbian edges.

    Excludes archetype IDs from the candidate neighbours so we don't accidentally
    link two archetype targets together.
    """
    from datetime import datetime, timezone

    db = store.db
    exclude = {str(beyond_id), str(island_id), str(spread_id)}

    # Find the neighbour with highest cosine to the BEYOND target that is not
    # another archetype target.  Read the BEYOND target's embedding.
    with db._conn_lock:
        row = db._conn.execute(
            "SELECT embedding FROM records WHERE id=?", (str(beyond_id),)
        ).fetchone()
        if row is None:
            # A missing beyond row means the corpus was not built as expected;
            # leaving BEYOND and ISLAND indistinguishable would silently break
            # disjoint attribution.  Fail loudly instead of returning silently.
            raise RuntimeError(
                f"beyond target row absent from store: {beyond_id}"
            )
        beyond_vec = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        bn = float(np.linalg.norm(beyond_vec))
        if bn > 0.0:
            beyond_vec = beyond_vec / bn

        # Fetch all non-archetype active records
        rows = db._conn.execute(
            "SELECT id, embedding FROM records"
            " WHERE tombstoned_at IS NULL AND COALESCE(embedding_pending, 0) = 0"
        ).fetchall()

    best_id: "str | None" = None
    best_cos = -1.0
    for r in rows:
        if r["id"] in exclude:
            continue
        v = np.frombuffer(r["embedding"], dtype=np.float32).copy()
        vn = float(np.linalg.norm(v))
        if vn > 0.0:
            v = v / vn
        cos = float(np.dot(beyond_vec, v))
        if cos > best_cos:
            best_cos = cos
            best_id = r["id"]

    if best_id is None:
        # No non-archetype neighbour to link to — the BEYOND target would remain
        # an island, collapsing into the ISLAND archetype.  Fail loudly.
        raise RuntimeError(
            "no non-archetype neighbour available to seed the beyond edge"
        )

    now_str = datetime.now(timezone.utc).isoformat()
    # Use IGNORE to avoid IntegrityError if the edge already exists
    with db._conn_lock:
        db._conn.execute(
            "INSERT OR IGNORE INTO edges (src, dst, edge_type, weight, updated_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (str(beyond_id), best_id, "hebbian", best_cos, now_str),
        )


# ---------------------------------------------------------------------------
# Reproducible eval-set builder
# ---------------------------------------------------------------------------

def build_eval_set(
    seed: int = 0,
    n_filler: int = 220,
    store_path: "Path | str | None" = None,
) -> "tuple[MemoryStore, MemoryGraph, CommunityAssignment, list[UUID], Embedder, list[EvalCase]]":
    """Build a reproducible labelled eval set that seeds the three miss archetypes.

    All randomness is routed through a single seeded np.random.default_rng(seed)
    instance so the same seed always produces the same (cue_text, target_id,
    archetype) list in the same order.

    The function inserts records into an isolated temporary store, builds the
    runtime graph (which loads all Hebbian/pattern-separation edges), then
    assembles and verifies three EvalCase instances — one per archetype.

    Parameters
    ----------
    seed:
        Master RNG seed.  Same seed → identical ordered cases across
        independent calls and across process restarts.
    n_filler:
        Number of filler records to insert.  Must be > K_CANDIDATES so the
        COSINE_RANK_BEYOND_CUTOFF target lands past the cosine frontier.
        Default 220 exceeds K_CANDIDATES=200 by a safe margin.
    store_path:
        Directory for the isolated store.  If None a temporary directory is
        created; the caller is responsible for calling store.close() but NOT
        for cleaning up the tmpdir (it is kept for the store's lifetime via
        the tmpdir reference returned implicitly through the store).

    Returns
    -------
    (store, graph, assignment, rich_club, embedder, cases)
        All inputs needed by run_accuracy.

    Notes
    -----
    K_CANDIDATES is imported read-only from iai_mcp.pipeline — it is never
    modified.  The function constructs vectors relative to the live constant
    so the COSINE_RANK_BEYOND_CUTOFF case is always defined correctly even if
    K_CANDIDATES is changed in a future phase.
    """
    from iai_mcp.pipeline import K_CANDIDATES
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.store import MemoryStore

    # ---- Single seeded RNG instance — all randomness goes through here ------
    rng = np.random.default_rng(seed)

    # ---- Determine embedding dimension from environment ---------------------
    import os as _os
    dim = int(_os.environ.get("IAI_MCP_EMBED_DIM", "384"))

    # ---- Bench embedder (hash-based, deterministic by text + base_seed) -----
    # Used to embed cue texts so recall_for_response sees the right vector.
    # The base_seed matches `seed` so the cue embedding is also reproducible.
    from bench.neural_map import _BenchEmbedder
    embedder = _BenchEmbedder(base_seed=seed, dim=dim)

    # ---- Open isolated store ------------------------------------------------
    _tmpdir = None
    if store_path is None:
        _tmpdir = tempfile.TemporaryDirectory(prefix="iai-eval-")
        store_path = Path(_tmpdir.name)
    store = MemoryStore(path=Path(store_path))

    # =========================================================================
    # Case A: COSINE_RANK_BEYOND_CUTOFF
    #
    # Strategy:
    #   - Pick a cue text.  Embed it with the bench embedder → cue_vec.
    #   - Build a target vector nearly orthogonal to cue_vec (cosine ~ 0).
    #   - Build n_filler vectors with cosine to cue_vec spread uniformly from
    #     0.3 to 0.99.  Since n_filler > K_CANDIDATES the target will rank
    #     beyond position K_CANDIDATES.
    #   - Cue text is paraphrased from the target surface to avoid trigram/
    #     exact-term boosts (pipeline.py:897-899).
    # =========================================================================

    # Cue text — paraphrased away from the target literal surface
    cue_text_beyond = "how does the gateway authenticate connections for alice"
    # Target literal surface — distinct tokens from cue text
    target_text_beyond = "session credential refresh cycle in the token store"

    # Embed the cue to get cue_vec
    cue_vec_beyond = np.array(embedder.embed(cue_text_beyond), dtype=np.float32)
    cue_norm = float(np.linalg.norm(cue_vec_beyond))
    if cue_norm > 0.0:
        cue_vec_beyond = cue_vec_beyond / cue_norm

    # Target vector: make it nearly orthogonal to the cue.
    # Build a random vector, then subtract the projection onto cue_vec_beyond.
    raw_target = _unit_vec(rng, dim)
    proj = float(np.dot(raw_target, cue_vec_beyond))
    target_vec_beyond = raw_target - proj * cue_vec_beyond
    tn = float(np.linalg.norm(target_vec_beyond))
    if tn > 0.0:
        target_vec_beyond = target_vec_beyond / tn
    # Insert the beyond-cutoff target FIRST (before fillers) so it gets inserted
    # into an empty ANN index — no pattern-separation edges from it to fillers
    # (the ANN probe is over records already in the index at insert time).
    beyond_rec = _make_eval_record(target_vec_beyond.tolist(), target_text_beyond, rng)
    beyond_target_id = beyond_rec.id
    _insert_raw(store, beyond_rec)

    # Insert n_filler records with vectors close to cue_vec_beyond
    for i in range(n_filler):
        # Cosine to cue ranges from 0.3 to 0.99 — all higher than the target.
        # Construct: v = alpha*cue + beta*perp; alpha = cosine value.
        alpha = 0.3 + 0.69 * float(rng.random())
        perp = _unit_vec(rng, dim)
        perp = perp - float(np.dot(perp, cue_vec_beyond)) * cue_vec_beyond
        pn = float(np.linalg.norm(perp))
        if pn > 0.0:
            perp = perp / pn
        beta = (1.0 - alpha ** 2) ** 0.5
        filler_vec = alpha * cue_vec_beyond + beta * perp
        fn_ = float(np.linalg.norm(filler_vec))
        if fn_ > 0.0:
            filler_vec = filler_vec / fn_
        filler_rec = _make_eval_record(
            filler_vec.tolist(),
            f"filler record for the index position {i}",
            rng,
        )
        _insert_raw(store, filler_rec)

    # =========================================================================
    # Case B: EDGE_LESS_ISLAND
    #
    # Strategy:
    #   - Build a target vector orthogonal to everything in the store.
    #     With high probability, cosine < 0.70 to all existing records so no
    #     pattern-separation edge is seeded at write time.
    #   - After build_runtime_graph the target's adjacency row contains only
    #     its self-loop (or is empty if no self-loop was inserted, but the
    #     store.insert path always seeds a self-loop via boost_edges).
    #   - Cue: a paraphrased query that embedder maps close to the island target.
    #     Since the island has almost no cosine similarity to the cue (the cue
    #     we use points AWAY from the island), it may not even be in oracle top-k.
    #     The archetype is STRUCTURAL (no edges), not about oracle rank.
    #
    # Note: the "miss" for this archetype comes from the graph structure (spread
    # cannot reach it), so we just need to verify the structural condition holds.
    # The run_accuracy probe for this case exercises a cue that points toward the
    # hub cluster, so the island target is unreachable via spread.
    # =========================================================================

    # Reuse the beyond-cutoff cue for the island case.  The island vector is
    # near-orthogonal to cue_vec_beyond (we projected it out), so the island
    # target ranks at the bottom of the pool for cue_text_beyond — well outside
    # cosine top-200.  Without edges and without cosine candidacy, the graph
    # path cannot reach the island target.
    cue_text_island = cue_text_beyond
    target_text_island = "isolated quantum protocol negotiation record"

    # Build a vector near-orthogonal to cue_vec_beyond so the island target
    # has near-zero cosine to the island cue, placing it well outside the
    # cosine top-200 candidates.  This ensures the only path to the island is
    # via graph edges — which don't exist because it was written far from every
    # stored record.
    raw_island = _unit_vec(rng, dim)
    # Project out the cue direction to make the island near-orthogonal to cue_vec_beyond
    raw_island = raw_island - float(np.dot(raw_island, cue_vec_beyond)) * cue_vec_beyond
    iln = float(np.linalg.norm(raw_island))
    if iln > 0.0:
        raw_island = raw_island / iln
    island_vec = raw_island

    island_rec = _make_eval_record(island_vec.tolist(), target_text_island, rng)
    island_target_id = island_rec.id
    _insert_raw(store, island_rec)

    # =========================================================================
    # Case C: SPREAD_DISABLED_BY_PRIOR_LATENCY
    #
    # Strategy:
    #   - Reuse cue_text_beyond (same cue vector as Cases A and B).  The 220
    #     fillers already ensure that any near-orthogonal target ranks well
    #     beyond K_CANDIDATES for that cue.
    #   - Build a target vector near-orthogonal to cue_vec_beyond (independent
    #     random direction, not the same as the beyond/island targets) so it
    #     sits outside cosine top-K_CANDIDATES.
    #   - In run_accuracy, probe this case with pipeline._last_recall_latency_ms
    #     pre-set to 2001.0 to prove the sentinel does NOT gate spread.  The
    #     spread-depth controller is driven by a confidence signal; a high prior
    #     latency value cannot force degrade.  Misses for this probe are expected
    #     to fall into the island or beyond-cutoff buckets (not degrade) as the
    #     fix lands — the degrade bucket should trend to zero.
    #   - The cue text is the same as the beyond/island cases; the target text
    #     is distinct so no exact-term boost fires for this specific record.
    # =========================================================================

    cue_text_spread = cue_text_beyond  # shared cue; 220 fillers already in place
    target_text_spread = "degrade probe latency threshold boundary record"

    # Spread target: near-orthogonal to cue_vec_beyond — independent random
    # direction, distinct from the beyond and island target directions.
    raw_spread = _unit_vec(rng, dim)
    raw_spread = raw_spread - float(np.dot(raw_spread, cue_vec_beyond)) * cue_vec_beyond
    srn = float(np.linalg.norm(raw_spread))
    if srn > 0.0:
        raw_spread = raw_spread / srn
    spread_target_vec = raw_spread

    spread_rec = _make_eval_record(spread_target_vec.tolist(), target_text_spread, rng)
    spread_target_id = spread_rec.id
    _insert_raw(store, spread_rec)

    # ---- Flush and build the runtime graph ----------------------------------
    _flush(store)
    graph, assignment, rich_club = build_runtime_graph(store)

    # ---- Seed Hebbian edge for the BEYOND target ---------------------------
    # Without this, the BEYOND target would also be an island (no edges),
    # making it indistinguishable from the ISLAND target by the _is_island
    # predicate.  We insert one directional Hebbian edge from the BEYOND target
    # to its closest neighbour in the corpus so that:
    #   BEYOND target → pred_island=False → attributed to COSINE_RANK_BEYOND_CUTOFF
    #   ISLAND target → pred_island=True  → attributed to EDGE_LESS_ISLAND
    # The edge is injected directly into SQLite because pattern-separation did
    # not seed it at write time (the BEYOND target was inserted into an empty
    # ANN before any fillers existed).
    _seed_beyond_edge(store, graph, beyond_target_id, island_target_id, spread_target_id)
    # Rebuild graph to load the newly injected edge. The injection is an
    # out-of-band SQL write: it bumps no dirty counter and rarely moves the
    # windowed cache key, so without an explicit invalidation the build layer
    # legitimately serves its warm bundle and the edge never appears —
    # misclassifying the BEYOND target as an edge-less island.
    from iai_mcp import runtime_graph_cache as _rgc
    _rgc.invalidate(store)
    graph, assignment, rich_club = build_runtime_graph(store)

    # ---- Verify COSINE_RANK_BEYOND_CUTOFF: target must be outside top-K ----
    # If it is still within K_CANDIDATES, add more filler in the same direction.
    cue_emb_beyond = np.array(embedder.embed(cue_text_beyond), dtype=np.float32)
    oracle_beyond = oracle_top_k(store, cue_emb_beyond, k=K_CANDIDATES + 50)
    top_k_set = set(oracle_beyond[:K_CANDIDATES])

    if beyond_target_id in top_k_set:
        # Boost filler: insert more records with cosine > target cosine to cue
        _extra = 0
        while beyond_target_id in top_k_set and _extra < 100:
            alpha = float(rng.uniform(0.5, 0.99))
            perp = _unit_vec(rng, dim)
            perp = perp - float(np.dot(perp, cue_vec_beyond)) * cue_vec_beyond
            pn = float(np.linalg.norm(perp))
            if pn > 0.0:
                perp = perp / pn
            beta = (1.0 - alpha ** 2) ** 0.5
            fv = alpha * cue_vec_beyond + beta * perp
            fn = float(np.linalg.norm(fv))
            if fn > 0.0:
                fv = fv / fn
            _insert_raw(store, _make_eval_record(fv.tolist(), f"extra filler {_extra}", rng))
            _extra += 1
        _flush(store)
        _rgc.invalidate(store)
        graph, assignment, rich_club = build_runtime_graph(store)
        oracle_beyond = oracle_top_k(store, cue_emb_beyond, k=K_CANDIDATES + 50)
        top_k_set = set(oracle_beyond[:K_CANDIDATES])

    # ---- Verify EDGE_LESS_ISLAND: target has no non-self neighbours ---------
    island_neighbours = graph._adj.get(str(island_target_id), {})
    non_self = {n for n in island_neighbours if n != str(island_target_id)}
    # If island somehow got edges from fillers, we need to ensure the target
    # was genuinely isolated.  If it picked up edges during insert (cosine of
    # the ANN probe was >= 0.70), note this in the result; the structural check
    # is what matters for attribution.

    # ---- Assemble eval cases ------------------------------------------------
    cases: list[EvalCase] = [
        EvalCase(
            cue_text=cue_text_beyond,
            target_id=beyond_target_id,
            archetype=Archetype.COSINE_RANK_BEYOND_CUTOFF,
        ),
        EvalCase(
            cue_text=cue_text_island,
            target_id=island_target_id,
            archetype=Archetype.EDGE_LESS_ISLAND,
        ),
        EvalCase(
            cue_text=cue_text_spread,
            target_id=spread_target_id,
            archetype=Archetype.SPREAD_DISABLED_BY_PRIOR_LATENCY,
        ),
    ]

    return store, graph, assignment, rich_club, embedder, cases


# ---------------------------------------------------------------------------
# End-to-end graph-vs-oracle runner
# ---------------------------------------------------------------------------

def run_accuracy(
    store: "MemoryStore",
    graph: "MemoryGraph",
    assignment: "CommunityAssignment",
    rich_club: "list[UUID]",
    embedder: "Embedder",
    cases: "list[EvalCase]",
    k: int = 10,
) -> dict:
    """Run the graph recall path for each cue and score against the oracle.

    For each EvalCase:
      - Embeds the cue text.
      - Computes the oracle top-k (exhaustive cosine) to check target presence.
      - Calls recall_for_response with the real graph path.
      - For the SPREAD_DISABLED_BY_PRIOR_LATENCY case, pre-sets
        pipeline._last_recall_latency_ms = 2001.0 to verify that a high
        prior-latency sentinel does NOT disable spread.  The spread-depth
        controller is driven by a confidence signal, not by wall-clock latency;
        the sentinel proves the archetype failure mode is fixed.
        The try/finally resets the latency global for in-run hygiene.
        The end-of-run _last_recall_latency_ms = 0.0 reset is the actual
        cross-test pollution guard.
      - Attributes each miss to exactly one bucket using the documented
        precedence (degrade-active → island → beyond-cutoff) and separately
        counts multi-label overlaps (misses that matched more than one predicate).

    Attribution precedence (disjoint by construction on this crafted set,
    but the predicates are NOT mutually exclusive on a real corpus):
      1. If degrade was active on that call → SPREAD_DISABLED_BY_PRIOR_LATENCY
         (degrade is no longer possible; this bucket should remain at zero).
      2. Else if the target's graph neighbour set is self-only → EDGE_LESS_ISLAND
      3. Else if oracle rank > K_CANDIDATES → COSINE_RANK_BEYOND_CUTOFF
      4. Else → unattributed (signals incomplete attribution on the crafted set)

    Rationale for island before beyond-cutoff: the island predicate is a
    structural graph property (no non-self neighbours), while the beyond-cutoff
    predicate is a cosine-distance property.  A record can be both an island AND
    beyond the cosine cutoff; in that case, the structural isolation is the more
    fundamental explanation — the record cannot be reached even if the cosine
    frontier were widened.  Reserving COSINE_RANK_BEYOND_CUTOFF for records that
    have edges (are structurally reachable via spread) but sit outside the cosine
    frontier clarifies which mechanism is the proximate cause.

    At end-of-run, resets pipeline._last_recall_latency_ms = 0.0 so no
    >2000 sentinel leaks to any subsequent test that imports pipeline.

    Returns
    -------
    dict with keys:
      recall_at_k:              float in [0, 1]
      false_negative_rate:      float in [0, 1]
      n_cues:                   int
      miss_counts_by_archetype: dict[str, int]
      multi_label_overlap_count: int
      unattributed:             int
    """
    import iai_mcp.pipeline as _pl
    from iai_mcp.pipeline import K_CANDIDATES

    recall_scores: list[float] = []
    fn_flags: list[bool] = []
    miss_counts: dict[str, int] = {
        Archetype.COSINE_RANK_BEYOND_CUTOFF: 0,
        Archetype.SPREAD_DISABLED_BY_PRIOR_LATENCY: 0,
        Archetype.EDGE_LESS_ISLAND: 0,
    }
    multi_label_overlap_count = 0
    unattributed = 0

    # Margin for oracle lookup: use K_CANDIDATES + buffer so targets just
    # outside the cutoff are still found by the oracle.
    oracle_k = max(k, K_CANDIDATES + 50)

    for case in cases:
        is_spread_probe = (case.archetype == Archetype.SPREAD_DISABLED_BY_PRIOR_LATENCY)

        # --- Embed the cue -----------------------------------------------
        cue_emb = np.array(embedder.embed(case.cue_text), dtype=np.float32)

        # --- Oracle top-k (exhaustive cosine ground truth) ----------------
        oracle_ids = oracle_top_k(store, cue_emb, k=oracle_k)

        # --- Latency sentinel for the spread-probe case -------------------
        # Pre-set _last_recall_latency_ms = 2001.0 to prove that a high prior
        # latency does NOT disable spread (the wall-clock switch is removed).
        # degrade_was_active is always False — the adaptive-depth controller
        # ignores wall-clock; the bucket should trend to zero misses.
        degrade_was_active = False
        if is_spread_probe:
            _prev_latency = _pl._last_recall_latency_ms
            _pl._last_recall_latency_ms = 2001.0

        try:
            resp = _pl.recall_for_response(
                store=store,
                graph=graph,
                assignment=assignment,
                rich_club=rich_club,
                embedder=embedder,
                cue=case.cue_text,
                session_id="accuracy_probe",
                budget_tokens=1500,
            )
        finally:
            if is_spread_probe:
                # In-run hygiene restore: prevents the sentinel leaking to the
                # NEXT case's call within this loop.
                _pl._last_recall_latency_ms = _prev_latency

        returned_ids = [h.record_id for h in resp.hits]

        # --- Score: target-based per-case recall -----------------------------
        # Whether the labeled target id was returned within the top-k positions.
        # This is the signal that matters: the archetype target is the ground
        # truth for the crafted eval set, and the oracle is used only for
        # attribution predicates below (not for scoring).
        target_missed = case.target_id not in set(returned_ids[:k])
        case_recall = 0.0 if target_missed else 1.0

        recall_scores.append(case_recall)
        fn_flags.append(target_missed)

        # --- Attribute each miss -------------------------------------------
        if not target_missed:
            continue  # hit: no attribution needed

        # Check all three predicates for multi-label overlap counting
        pred_degrade = degrade_was_active  # always False (wall-clock switch removed)
        pred_island = _is_island(graph, case.target_id)
        pred_beyond = _oracle_rank_exceeds_cutoff(
            oracle_ids, case.target_id, K_CANDIDATES
        )

        n_predicates_matched = sum([pred_degrade, pred_island, pred_beyond])
        if n_predicates_matched > 1:
            multi_label_overlap_count += 1

        # Assign to exactly one bucket by documented precedence:
        # degrade-active → island → beyond-cutoff → unattributed
        if pred_degrade:
            miss_counts[Archetype.SPREAD_DISABLED_BY_PRIOR_LATENCY] += 1
        elif pred_island:
            miss_counts[Archetype.EDGE_LESS_ISLAND] += 1
        elif pred_beyond:
            miss_counts[Archetype.COSINE_RANK_BEYOND_CUTOFF] += 1
        else:
            unattributed += 1

    # --- Cross-test pollution guard: reset the latency global to 0.0 --------
    # Resetting to 0.0 here ensures no >2000 sentinel is left in the module
    # for any subsequent test.
    _pl._last_recall_latency_ms = 0.0

    mean_recall = float(np.mean(recall_scores)) if recall_scores else 1.0
    fn_rate = sum(fn_flags) / len(fn_flags) if fn_flags else 0.0

    return {
        "recall_at_k": mean_recall,
        "false_negative_rate": fn_rate,
        "n_cues": len(cases),
        "miss_counts_by_archetype": dict(miss_counts),
        "multi_label_overlap_count": multi_label_overlap_count,
        "unattributed": unattributed,
    }


# ---------------------------------------------------------------------------
# Attribution predicates (read-only, no side effects)
# ---------------------------------------------------------------------------

def _is_island(graph: "MemoryGraph", target_id: UUID) -> bool:
    """True iff the target has no non-self neighbours in the runtime graph."""
    neighbours = graph._adj.get(str(target_id), {})
    non_self = {n for n in neighbours if n != str(target_id)}
    return len(non_self) == 0


def _oracle_rank_exceeds_cutoff(
    oracle_ids: list[UUID],
    target_id: UUID,
    cutoff: int,
) -> bool:
    """True iff target_id is absent from the oracle top-cutoff slice."""
    if len(oracle_ids) <= cutoff:
        # oracle_ids shorter than cutoff means we asked for fewer than cutoff;
        # if target is absent from whatever we fetched, treat as beyond-cutoff.
        return target_id not in set(oracle_ids)
    return target_id not in set(oracle_ids[:cutoff])


# ---------------------------------------------------------------------------
# Baseline measurement helpers
# ---------------------------------------------------------------------------

# Sentinel prefix used by _VectorInjectionEmbedder to intercept pre-computed
# cue vectors.  The sentinel is NOT a real memory surface — exact-term and
# trigram boosts (pipeline.py:897-899) cannot fire on it.
_INJECTION_SENTINEL_PREFIX = "__baseline_cue_vec__:"


class _VectorInjectionEmbedder:
    """Embedder shim that returns a pre-computed vector for sentinel cue strings.

    When ``embed()`` is called with a string of the form
    ``"__baseline_cue_vec__:<uuid>"`` the shim returns the vector registered
    for that UUID via ``register``.  Any other string is forwarded to the
    wrapped embedder.

    This lets run_baseline inject a perturbed-vector cue without going through
    text→embedding so that the cue is provably independent of the target's
    ``literal_surface``.  Exact-term and trigram boosts (pipeline.py:897-899)
    cannot fire because the sentinel string shares no tokens with any stored
    record.

    Implements the same ``embed(text) -> list[float]`` and dimension attributes
    the production pipeline expects.
    """

    def __init__(self, wrapped: "Embedder") -> None:
        self._wrapped = wrapped
        self._registry: dict[str, list[float]] = {}
        # Mirror dimension attributes so pipeline code that reads these works.
        self.DIM = getattr(wrapped, "DIM", getattr(wrapped, "DEFAULT_DIM", 384))
        self.DEFAULT_DIM = self.DIM
        self.DEFAULT_MODEL_KEY = getattr(wrapped, "DEFAULT_MODEL_KEY", "bench")

    def register(self, key: str, vec: list[float]) -> str:
        """Store ``vec`` under ``key`` and return the sentinel cue string."""
        sentinel = f"{_INJECTION_SENTINEL_PREFIX}{key}"
        self._registry[sentinel] = vec
        return sentinel

    def embed(self, text: str) -> list[float]:
        if text in self._registry:
            return self._registry[text]
        if self._wrapped is None:
            raise RuntimeError(
                "synthetic-dim baseline attempted a real text embed; every "
                "cue must go through the sentinel registry"
            )
        return self._wrapped.embed(text)


class _inject_embedder_into_dispatch:
    """Route ``core.dispatch``'s internally-built embedder through a shim.

    ``dispatch(store, "memory_recall", ...)`` constructs its own embedder via
    ``iai_mcp.embed.embedder_for_store(store)`` and encodes the cue TEXT with it
    (the warm ANN path reads ``params["cue"]``, not ``params["cue_embedding"]``).
    To inject a pre-computed perturbed cue vector through the real production
    recall path, the harness registers the vector in ``shim`` under a sentinel
    string, passes that sentinel as the cue, and — for the duration of the
    ``with`` block — makes ``embedder_for_store`` return the shim so dispatch
    encodes the sentinel through it and gets the pre-computed vector back.

    The original ``embedder_for_store`` is restored on exit, so the patch is
    scoped to the measurement loop and never leaks into other callers.
    """

    # shared with tests/_helpers.RECALL_STUB_ACTIVE_ENV -- bench cannot
    # import the test tree, so both sides hardcode this string literal;
    # the test-suite degrade guard reads it to see this non-monkeypatch stub
    _RECALL_STUB_ACTIVE_ENV = "IAI_MCP_TEST_RECALL_STUB_ACTIVE"

    def __init__(self, shim: "_VectorInjectionEmbedder") -> None:
        self._shim = shim
        self._orig = None

    def __enter__(self) -> "_inject_embedder_into_dispatch":
        import os

        import iai_mcp.embed as _emb
        self._emb_mod = _emb
        self._orig = _emb.embedder_for_store
        # **_kwargs absorbs allow_identity_mismatch/build_timeout, which the
        # recall dispatch always passes -- a one-arg shim raises TypeError,
        # caught by the pipeline as a silent degrade instead of using it
        _emb.embedder_for_store = lambda _store, **_kwargs: self._shim
        os.environ[self._RECALL_STUB_ACTIVE_ENV] = "1"
        return self

    def __exit__(self, *_exc) -> None:
        import os

        if self._orig is not None:
            self._emb_mod.embedder_for_store = self._orig
            self._orig = None
        os.environ.pop(self._RECALL_STUB_ACTIVE_ENV, None)


def _dispatch_recall_ids(
    store: "MemoryStore",
    sentinel_cue: str,
    session_id: str,
) -> list[UUID]:
    """Run the production ``memory_recall`` dispatch path and return hit ids.

    This is the EXACT entry point the daemon serves recall through: it builds the
    bounded ANN-seeded candidate graph, spreads over the store's own edges, and
    ranks via the production pipeline — so latency and ranking match production
    rather than the ~10x-slower exhaustive-cosine hand-assembly.  The perturbed
    cue vector is injected via the embedder shim installed by the caller's
    ``_inject_embedder_into_dispatch`` context.

    Hit ids are extracted from the dispatch result exactly as production returns
    them (``response["hits"][i]["record_id"]``), as UUIDs for the semantic-hit
    scorer.
    """
    from iai_mcp import core

    response = core.dispatch(
        store,
        "memory_recall",
        {"cue": sentinel_cue, "session_id": session_id},
    )
    return [UUID(h["record_id"]) for h in response.get("hits", [])]


def _perturb_vec(
    vec: np.ndarray,
    rng: "np.random.Generator",
    noise_fraction: float = 0.15,
) -> np.ndarray:
    """Return a unit-normalised vector near ``vec`` with additive orthogonal noise.

    Builds a random perturbation direction orthogonal to ``vec`` and blends
    it in at ``noise_fraction`` magnitude, then re-normalises.  The result is
    close to but distinct from ``vec`` in cosine space (cosine ≈ 1 − ε²/2),
    so the oracle still ranks the original record near the top while the cue
    is not a verbatim copy of the stored embedding.

    The perturbation is seeded from ``rng`` (the caller's single seeded
    instance) — no module-level random calls.

    Args:
        vec:            Unit-normalised float32 source vector (the stored record
                        embedding).
        rng:            Caller's seeded ``np.random.Generator`` instance.
        noise_fraction: Fraction of the perturbation component relative to
                        the source vector.  Default 0.15 → cosine ≈ 0.989.

    Returns:
        Unit-normalised float32 perturbed vector.
    """
    dim = len(vec)
    # Draw an orthogonal-noise direction; re-draw (bounded) if the orthogonalised
    # noise collapses to zero (vec collinear with the draw, or dim == 1), which
    # would otherwise return the cue verbatim (cosine == 1.0) and defeat the
    # "not a verbatim copy" guarantee.
    noise = np.zeros(dim, dtype=np.float32)
    for _ in range(8):
        cand = rng.standard_normal(dim).astype(np.float32)
        cand = cand - float(np.dot(cand, vec)) * vec
        nn = float(np.linalg.norm(cand))
        if nn > 0.0:
            noise = cand / nn
            break
    else:
        # Degenerate corpus (e.g. dim == 1): fall back to a fixed orthogonal
        # basis vector when one exists.
        if dim > 1:
            basis = np.zeros(dim, dtype=np.float32)
            axis = int(np.argmin(np.abs(vec)))
            basis[axis] = 1.0
            basis = basis - float(np.dot(basis, vec)) * vec
            bn = float(np.linalg.norm(basis))
            if bn > 0.0:
                noise = basis / bn

    perturbed = vec + noise_fraction * noise
    pn = float(np.linalg.norm(perturbed))
    result = (perturbed / pn).astype(np.float32) if pn > 0.0 else vec.astype(np.float32)
    # Guarantee the cue is not a verbatim copy when a transverse direction exists.
    if float(np.linalg.norm(noise)) > 0.0:
        assert float(np.dot(result, vec)) < 0.999, "perturbed cue collapsed onto source"
    return result


def _scan_active_corpus(
    store: "MemoryStore",
) -> "tuple[dict[UUID, np.ndarray], dict[UUID, int]]":
    """Scan the active corpus once, returning the id→embedding map and class sizes.

    Reads ``(id, embedding)`` for all active records in a single SQL query
    under ``_conn_lock`` (same discipline as ``oracle_top_k``).  Embeddings are
    unit-normalised.  Exact-embedding equivalence-class sizes are computed by
    grouping on the raw float32 bytes so duplicate-embedding clusters are
    detectable for dedup-aware sampling and for the semantic metric.

    Read-only: no writer, no mutator, no index modification.

    Returns:
        (id_to_vec, class_size) where ``id_to_vec`` maps id → unit-normalised
        float32 embedding and ``class_size`` maps id → the count of active records
        sharing that id's exact embedding bytes.
    """
    db = store.db
    with db._conn_lock:
        rows = db._conn.execute(_ACTIVE_CORPUS_SQL).fetchall()

    id_to_vec: dict[UUID, np.ndarray] = {}
    bytes_count: dict[bytes, int] = {}
    id_bytes: dict[UUID, bytes] = {}
    for row in rows:
        uid = UUID(row["id"])
        v = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        vn = float(np.linalg.norm(v))
        if vn > 0.0:
            v = v / vn
        id_to_vec[uid] = v
        b = v.astype(np.float32).tobytes()
        id_bytes[uid] = b
        bytes_count[b] = bytes_count.get(b, 0) + 1

    class_size = {uid: bytes_count[b] for uid, b in id_bytes.items()}
    return id_to_vec, class_size


def _sample_active_record_embeddings(
    store: "MemoryStore",
    n_samples: int,
    rng: "np.random.Generator",
    id_to_vec: "dict[UUID, np.ndarray] | None" = None,
    class_size: "dict[UUID, int] | None" = None,
    max_class_size: int = 1,
) -> "tuple[list[tuple[UUID, np.ndarray]], int]":
    """Dedup-aware sample of up to ``n_samples`` active records.

    Prefers targets whose exact-embedding equivalence class is small
    (``class_size <= max_class_size``) so the headline recall number is measured
    on records that recall can actually surface as themselves.  Targets living in
    large duplicate clusters are excluded from the preferred pool; the number
    excluded is returned so the pollution level is visible in the baseline output.
    If fewer than ``n_samples`` unique targets exist, the remaining slots are
    filled from the excluded (duplicate-cluster) targets so the sample count is
    preserved — those are still scoreable under the semantic metric.

    Read-only: no writer, no mutator, no index modification.

    Args:
        store: open MemoryStore (used only if ``id_to_vec`` is not supplied).
        n_samples: number of targets to draw.
        rng: the caller's seeded ``np.random.Generator`` (deterministic order).
        id_to_vec: precomputed corpus map; scanned from ``store`` if None.
        class_size: precomputed equivalence-class sizes; scanned if None.
        max_class_size: inclusive upper bound on class size for the preferred
            (unique) pool.  Default 1 → strictly-unique targets only.

    Returns:
        (samples, excluded_count) where ``samples`` is a list of
        (id, unit-normalised embedding) tuples of length min(n_samples, N) and
        ``excluded_count`` is the number of active records that fell in
        duplicate clusters (class_size > max_class_size).
    """
    if id_to_vec is None or class_size is None:
        id_to_vec, class_size = _scan_active_corpus(store)

    if not id_to_vec:
        return [], 0

    all_ids = list(id_to_vec.keys())
    unique_ids = [uid for uid in all_ids if class_size.get(uid, 1) <= max_class_size]
    duplicate_ids = [uid for uid in all_ids if class_size.get(uid, 1) > max_class_size]
    excluded_count = len(duplicate_ids)

    # Shuffle each pool deterministically via the seeded rng.
    uniq_perm = list(range(len(unique_ids)))
    rng.shuffle(uniq_perm)
    ordered = [unique_ids[i] for i in uniq_perm]

    if len(ordered) < n_samples and duplicate_ids:
        dup_perm = list(range(len(duplicate_ids)))
        rng.shuffle(dup_perm)
        ordered.extend(duplicate_ids[i] for i in dup_perm)

    selected = ordered[: min(n_samples, len(ordered))]
    samples = [(uid, id_to_vec[uid]) for uid in selected]
    return samples, excluded_count


def _count_active_records(store: "MemoryStore") -> int:
    """Count active records in the store (read-only)."""
    db = store.db
    with db._conn_lock:
        row = db._conn.execute(
            "SELECT COUNT(*) AS n FROM records"
            " WHERE tombstoned_at IS NULL"
            " AND COALESCE(embedding_pending, 0) = 0"
        ).fetchone()
    return int(row["n"]) if row else 0


def run_baseline(
    store_path: "Path | str",
    k: int = 10,
    n_samples: int = 300,
    seed: int = 42,
    out: "Path | str | None" = None,
) -> dict:
    """Measure current PRODUCTION recall@k on a supplied isolated store copy.

    Recall is measured through the exact production entry point the daemon
    serves — ``core.dispatch(store, "memory_recall", ...)`` — so both the
    latency and the ranking are representative.  Dispatch builds its own bounded
    ANN-seeded candidate graph (via ``store.query_similar`` + edge spread), which
    is why the store is opened so the HNSW ANN index is available (see below).
    All randomness routes through a single seeded ``np.random.default_rng(seed)``
    instance.

    Store open (isolated copy, corpus never mutated):
        The isolated snapshot copy is opened in SHARED (non-read-only) mode so
        the HNSW ANN index is built at boot; the production recall path requires
        it.  Only the derived ``records.hnsw`` index file is written — the corpus
        (record rows) is never mutated, so the copy's active-record count is
        invariant across the run.  Run against an isolated snapshot copy, never
        the live production store.

    Cue derivation (``vector_perturbed_from_stored_embedding`` method):
        Each cue is a perturbed version of the target's stored embedding,
        obtained by adding small orthogonal noise via ``_perturb_vec``.  The
        vector is injected into the production dispatch path via
        ``_VectorInjectionEmbedder``: the perturbed vector is registered under a
        sentinel string, that string is passed as the dispatch cue, and — for
        the measurement window — ``embedder_for_store`` is routed through the
        shim so dispatch encodes the sentinel into the pre-computed vector.  The
        sentinel string shares NO tokens with any stored record, so the
        exact-term and trigram boosts (pipeline.py:897-899) cannot fire on the
        target — the measurement is an honest, not optimistically-biased,
        recall@k.

    Hit definition (semantic, dedup-safe):
        A recall counts as a HIT when any returned record is the target memory:
        the exact target id, a member of the target's exact-embedding
        equivalence class, or a record within cosine ``SEMANTIC_HIT_EPS`` of the
        target.  Exact-``target_id`` equality alone is unscoreable under
        duplicate-embedding clusters (the k tie-mates crowd the specific id out
        of the top-k), so the semantic hit is the honest headline metric.  The
        exact-id number is also reported as ``recall_at_k_exact`` for reference.

    Sampling (dedup-aware):
        Targets are drawn preferentially from records whose exact-embedding
        equivalence class is size 1 (unique).  ``excluded_duplicate_targets``
        reports how many active records fell in duplicate clusters, exposing the
        pollution level.

    Latency measurement (non-degraded, warmed):
        The production dispatch path is warmed once before the timed loop
        (through the same dispatch call measured below) so first-cue cold-build
        costs (ANN warm-up, structural-cache load, count cache, centrality
        resolve) do not land on cue 1.  The degrade latency global is reset to
        0.0 before EACH cue so a slow cue cannot self-sustain the >2 s degrade
        cascade — the degrade then fires only for cues genuinely slow in
        isolation.  The degrade decision the pipeline actually used is captured
        from the START-of-call latency value (what pipeline.py:1219 read for
        THIS call), not the post-call value.

    Misses are attributed to exactly one archetype by the documented
    degrade→island→beyond-cutoff precedence, with a ``multi_label_overlap_count``
    for misses matching more than one predicate.

    Args:
        store_path: Path to the isolated store copy to measure.
        k:          Rank cutoff for recall@k (default 10).
        n_samples:  Number of records to sample as evaluation targets.
        seed:       RNG seed (pins the sample and perturbation for
                    reproducibility).
        out:        If given, writes the JSON summary to this path in addition
                    to returning it.  None → stdout only.

    Returns:
        dict with keys: n_cues, recall_at_k, recall_at_k_exact,
        false_negative_rate, miss_counts_by_archetype, multi_label_overlap_count,
        unattributed, cue_derivation_method, semantic_hit_eps,
        excluded_duplicate_targets, cosine_rank_percentiles, cosine_rank_not_found,
        degrade_fire_rate, island_rate, p50_ms, p95_ms, corpus_size, driver, k, seed.

    Raises:
        FileNotFoundError: if ``store_path`` does not exist.
        RuntimeError:      if the store has no active records.
    """
    from iai_mcp.embed import embedder_for_store
    from iai_mcp.hippo import AccessMode
    from iai_mcp.pipeline import K_CANDIDATES
    import iai_mcp.pipeline as _pl
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.store import MemoryStore

    store_path = Path(store_path)
    if not store_path.exists():
        raise FileNotFoundError(f"store_path does not exist: {store_path}")

    # Open the isolated snapshot copy in SHARED (non-read-only) mode so the
    # HNSW ANN index is built at boot.  The production recall path
    # (``dispatch`` → ``store.query_similar``) requires the ANN index; a
    # read-only open leaves it unbuilt.  Only the derived ``records.hnsw`` index
    # file is written — the corpus (record rows) is never mutated, so the copy's
    # active-record count is invariant across the run.
    store = MemoryStore(
        path=store_path,
        access_mode=AccessMode.SHARED,
    )

    # ---- Reinforce-defer shim — mirrors the daemon (deferred reinforce) ------
    # In this non-daemon context store._reinforce_queue is None, so without the
    # shim queue_reinforce falls back to per-id synchronous reinforce_record
    # calls on the recall thread — a non-production code path.  The daemon has a
    # live ReinforceWriteQueue so the call returns immediately (deferred).
    # Binding queue_reinforce to a no-op here mirrors the daemon: the timed
    # recall measures production-representative latency (reinforce deferred),
    # not the sync-fallback artifact.
    store.queue_reinforce = lambda *_a, **_kw: None  # type: ignore[method-assign]

    # ---- Single seeded RNG instance — all randomness through here -----------
    rng = np.random.default_rng(seed)

    corpus_size = _count_active_records(store)
    if corpus_size == 0:
        store.close()
        raise RuntimeError("Store has no active records — nothing to measure.")

    # ---- Full runtime graph — used ONLY for the structural island predicate --
    # (``_is_island``) in miss attribution.  It is NOT the recall path: recall
    # goes through the production ``dispatch`` entry point below, which builds
    # its own bounded ANN-seeded candidate graph exactly like the daemon.
    graph, assignment, rich_club = build_runtime_graph(store)

    # ---- Embedder shim — injects the perturbed cue vector into the production
    # dispatch path (see ``_inject_embedder_into_dispatch``). --------------------
    try:
        real_embedder = embedder_for_store(store)
    except ValueError:
        # A synthetic-dim store has no matching real embedder — and needs
        # none: every baseline cue goes through the sentinel registry, and
        # the shim fails loudly if a real text embed is ever attempted.
        real_embedder = None
    embedder = _VectorInjectionEmbedder(real_embedder)

    # ---- Scan the active corpus once (id→vec + exact-embedding class sizes) --
    # The map feeds both the semantic hit metric and dedup-aware sampling; the
    # class sizes drive the equivalence-class logic.
    id_to_vec, class_size = _scan_active_corpus(store)

    # ---- Sample evaluation targets (dedup-aware: prefer unique-embedding) ----
    samples, excluded_duplicate_targets = _sample_active_record_embeddings(
        store, n_samples, rng, id_to_vec=id_to_vec, class_size=class_size
    )
    if not samples:
        store.close()
        raise RuntimeError("No active records could be sampled.")

    # ---- Precompute oracle lookup depth (K_CANDIDATES + buffer for rank dist) -
    oracle_depth = max(k, K_CANDIDATES + 100)

    # ---- Install the embedder injection for the whole measurement window ------
    # For the duration of this block, ``dispatch`` encodes the sentinel cue
    # through the shim and receives the pre-computed perturbed vector, while
    # otherwise following the exact production recall path.
    recall_scores: list[float] = []       # semantic hit (headline)
    exact_recall_scores: list[float] = []  # exact-id hit (diagnostic)
    fn_flags: list[bool] = []
    cosine_ranks: list[int] = []          # class-best rank; excludes not-found
    not_found_count = 0
    latency_ms_list: list[float] = []
    degrade_fire_count = 0
    island_count = 0
    miss_counts: dict[str, int] = {
        Archetype.COSINE_RANK_BEYOND_CUTOFF: 0,
        Archetype.SPREAD_DISABLED_BY_PRIOR_LATENCY: 0,
        Archetype.EDGE_LESS_ISLAND: 0,
    }
    multi_label_overlap_count = 0
    unattributed = 0

    # Sentinel rank used for targets whose class is absent from the oracle slice;
    # excluded from the percentile distribution rather than folded in as a value.
    _NOT_FOUND_RANK = oracle_depth

    # For the whole measurement window, route ``dispatch``'s internally-built
    # embedder through the shim so each production recall encodes the sentinel
    # cue into the pre-computed perturbed vector.  The context manager restores
    # the real ``embedder_for_store`` on exit — including on any error — so the
    # global patch never leaks past this block.
    with _inject_embedder_into_dispatch(embedder):
        # ---- Warm the recall path once (throwaway cue) so first-cue cold-build
        # costs (ANN warm-up, structural-cache load, count cache, recency buffer,
        # centrality resolve) do NOT land on cue 1 of the timed loop.  The warm
        # cue uses the first sample's perturbed vector but its latency and result
        # are discarded.  It runs through the SAME production dispatch path
        # measured below, so dispatch's own lazy caches are warmed the way it
        # warms them.
        _pl._last_recall_latency_ms = 0.0
        _warm_id, _warm_vec = samples[0]
        _warm_cue = _perturb_vec(_warm_vec, rng, noise_fraction=0.15)
        _warm_sentinel = embedder.register(
            f"__warmup__{_warm_id}", _warm_cue.tolist()
        )
        _dispatch_recall_ids(store, _warm_sentinel, session_id="baseline_warmup")

        # ---- Per-cue measurement loop ---------------------------------------
        for target_id, target_vec in samples:
            # Reset the degrade global BEFORE each cue so a slow cue cannot
            # poison the next (no self-sustained >2 s cascade).  The degrade
            # fires only for cues genuinely slow in isolation — the realistic
            # single-user condition.
            _pl._last_recall_latency_ms = 0.0

            # The wall-clock degrade switch is removed; spread depth is driven
            # by a confidence signal, not by prior latency.  pred_degrade is
            # always False — captured here for the attribution loop below which
            # still checks the predicate for the degrade bucket.
            pred_degrade = False  # wall-clock switch removed; degrade cannot fire

            # Derive cue: perturb the stored embedding by small orthogonal
            # noise.  ``noise_fraction=0.15`` → cosine(cue, target) ≈ 0.989, so
            # the oracle still ranks the target near the top but the cue is not
            # verbatim.
            cue_vec = _perturb_vec(target_vec, rng, noise_fraction=0.15)

            # Register the perturbed vector in the injection shim; get a
            # sentinel cue string dispatch will route through the shim.
            sentinel = embedder.register(str(target_id), cue_vec.tolist())

            # Oracle top-k: exhaustive cosine ground truth for this cue.
            oracle_ids = oracle_top_k(store, cue_vec, k=oracle_depth)

            # Equivalence class of the target (ids sharing its embedding bytes).
            target_class = semantic_equivalence_class(target_id, id_to_vec)

            # Honest cosine rank = rank of the BEST-placed class member.
            # Targets whose class does not appear in the oracle slice are
            # censored (counted separately, excluded from the percentiles).
            target_rank = class_cosine_rank(
                oracle_ids, target_class, _NOT_FOUND_RANK
            )
            if target_rank == _NOT_FOUND_RANK:
                not_found_count += 1
            else:
                cosine_ranks.append(target_rank)

            # Island check (structural, read-only, no side effects).
            is_island_target = _is_island(graph, target_id)
            if is_island_target:
                island_count += 1

            # Production recall path — the exact ``memory_recall`` dispatch
            # entry the daemon serves.  Measure wall-clock latency of the
            # representative path.
            t0 = time.perf_counter()
            returned_ids = _dispatch_recall_ids(
                store, sentinel, session_id="baseline_probe"
            )
            call_ms = (time.perf_counter() - t0) * 1000.0
            latency_ms_list.append(call_ms)

            # Diagnostic: count cues whose measured latency exceeded 2000ms.
            # The wall-clock switch is removed so this never gates spread, but
            # it is kept for cross-phase latency tracking (degrade_rate in the
            # output will trend toward 0 as recall latency improves).
            if _pl._last_recall_latency_ms > 2000.0:
                degrade_fire_count += 1

            # ``returned_ids`` came from the production dispatch result above.
            # Headline: semantic hit (target memory recalled), dedup-safe.
            semantic_hit = is_semantic_hit(
                returned_ids, target_id, target_vec, id_to_vec, target_class, k
            )
            target_missed = not semantic_hit
            recall_scores.append(1.0 if semantic_hit else 0.0)
            # Diagnostic: exact-id hit (unscoreable under dups — kept for ref).
            exact_hit = target_id in set(returned_ids[:k])
            exact_recall_scores.append(1.0 if exact_hit else 0.0)
            fn_flags.append(target_missed)

            if not target_missed:
                continue

            # --- Attribute the miss ---
            # pred_degrade was captured at START-of-call (the value
            # pipeline.py:1219 actually used); pred_beyond uses the class rank
            # so a duplicate target is not spuriously flagged beyond-cutoff.
            pred_island = is_island_target
            pred_beyond = (
                target_rank == _NOT_FOUND_RANK or target_rank >= K_CANDIDATES
            )

            n_matched = sum([pred_degrade, pred_island, pred_beyond])
            if n_matched > 1:
                multi_label_overlap_count += 1

            if pred_degrade:
                miss_counts[Archetype.SPREAD_DISABLED_BY_PRIOR_LATENCY] += 1
            elif pred_island:
                miss_counts[Archetype.EDGE_LESS_ISLAND] += 1
            elif pred_beyond:
                miss_counts[Archetype.COSINE_RANK_BEYOND_CUTOFF] += 1
            else:
                unattributed += 1

    # ---- Cross-run hygiene: reset latency global ----------------------------
    _pl._last_recall_latency_ms = 0.0

    store.close()

    # ---- Aggregate metrics --------------------------------------------------
    n = len(recall_scores)
    mean_recall = float(np.mean(recall_scores)) if recall_scores else 1.0
    mean_recall_exact = float(np.mean(exact_recall_scores)) if exact_recall_scores else 1.0
    fn_rate = float(sum(fn_flags) / n) if n else 0.0
    degrade_rate = float(degrade_fire_count / n) if n else 0.0
    island_rate = float(island_count / n) if n else 0.0

    from bench.neural_map import _percentile as _pct
    p50_ms = _pct(latency_ms_list, 0.50)
    p95_ms = _pct(latency_ms_list, 0.95)

    # Percentiles over the non-censored ranks only (censored targets counted in
    # not_found_count).  Empty distribution → -1 sentinel (all targets censored).
    if cosine_ranks:
        rank_pct = {
            "p50": int(_pct(cosine_ranks, 0.50)),
            "p90": int(_pct(cosine_ranks, 0.90)),
            "p95": int(_pct(cosine_ranks, 0.95)),
            "p99": int(_pct(cosine_ranks, 0.99)),
        }
    else:
        rank_pct = {"p50": -1, "p90": -1, "p95": -1, "p99": -1}

    # driver: read from env (same convention as neural_map / the pipeline)
    import os as _os
    driver = _os.environ.get("LILLI_STORAGE_DRIVER", "hippo")

    result: dict = {
        "n_cues": n,
        "recall_at_k": round(mean_recall, 6),
        "recall_at_k_exact": round(mean_recall_exact, 6),
        "false_negative_rate": round(fn_rate, 6),
        "k": k,
        "seed": seed,
        "corpus_size": corpus_size,
        "driver": driver,
        "cue_derivation_method": "vector_perturbed_from_stored_embedding",
        "semantic_hit_eps": SEMANTIC_HIT_EPS,
        "excluded_duplicate_targets": excluded_duplicate_targets,
        "miss_counts_by_archetype": {
            Archetype.COSINE_RANK_BEYOND_CUTOFF: miss_counts[Archetype.COSINE_RANK_BEYOND_CUTOFF],
            Archetype.SPREAD_DISABLED_BY_PRIOR_LATENCY: miss_counts[Archetype.SPREAD_DISABLED_BY_PRIOR_LATENCY],
            Archetype.EDGE_LESS_ISLAND: miss_counts[Archetype.EDGE_LESS_ISLAND],
        },
        "multi_label_overlap_count": multi_label_overlap_count,
        "unattributed": unattributed,
        "cosine_rank_percentiles": rank_pct,
        "cosine_rank_not_found": not_found_count,
        "degrade_fire_rate": round(degrade_rate, 6),
        "island_rate": round(island_rate, 6),
        "p50_ms": round(p50_ms, 2),
        "p95_ms": round(p95_ms, 2),
    }

    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_baseline_args(argv: "list[str] | None" = None) -> "argparse.Namespace":
    parser = argparse.ArgumentParser(
        prog="bench.recall_accuracy",
        description=(
            "Measure current production recall@k on an isolated store copy via "
            "the memory_recall dispatch path. Builds only the derived ANN index; "
            "never mutates the corpus. "
            "Run against an isolated snapshot copy, never the live production store."
        ),
    )
    parser.add_argument(
        "--store-path",
        dest="store_path",
        required=True,
        help="Path to the isolated store copy to measure.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Rank cutoff for recall@k (default: 10).",
    )
    parser.add_argument(
        "--n-samples",
        dest="n_samples",
        type=int,
        default=300,
        help="Number of records to sample as evaluation targets (default: 300).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for deterministic sampling and perturbation (default: 42).",
    )
    parser.add_argument(
        "--out",
        dest="out",
        default=None,
        help="Optional path to write the JSON summary (also printed to stdout).",
    )
    return parser.parse_args(argv)


def _install_baseline_noop_keyring() -> None:
    """Install a no-op keyring backend to suppress prompts in bench context."""
    try:
        import keyring
        from keyring.backend import KeyringBackend

        if getattr(keyring.get_keyring(), "_iai_bench_noop", False):
            return

        class _BenchNoOpKeyring(KeyringBackend):
            priority = 99
            _iai_bench_noop = True
            _kv: dict[tuple[str, str], str] = {}

            def get_password(self, s: str, u: str) -> "str | None":
                return self._kv.get((s, u))

            def set_password(self, s: str, u: str, p: str) -> None:
                self._kv[(s, u)] = p

            def delete_password(self, s: str, u: str) -> None:
                self._kv.pop((s, u), None)

        keyring.set_keyring(_BenchNoOpKeyring())
    except Exception:
        pass


if __name__ == "__main__":
    _install_baseline_noop_keyring()
    _args = _parse_baseline_args()
    _result = run_baseline(
        store_path=_args.store_path,
        k=_args.k,
        n_samples=_args.n_samples,
        seed=_args.seed,
        out=_args.out,
    )
    print(json.dumps(_result, indent=2))
