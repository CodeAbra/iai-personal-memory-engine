from __future__ import annotations

import hashlib
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import log
from uuid import UUID

import numpy as np

from iai_mcp.community import CommunityAssignment
from iai_mcp.embed import Embedder
from iai_mcp.events import TELEMETRY_EMBED_NATIVE_FAILURE, write_event
from iai_mcp.exceptions import (
    NativeError,
)
from iai_mcp.graph import MemoryGraph
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryHit, RecallResponse

logger = logging.getLogger(__name__)


@dataclass
class SimpleRecordView:

    id: UUID
    embedding: list[float]
    literal_surface: str
    centrality: float
    tier: str
    aaak_index: str = ""
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    profile_modulation_gain: dict = field(default_factory=dict)
    structure_hv: bytes = b""
    provenance: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    language: str = "en"


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
                embedding=list(node["embedding"]),
                literal_surface=str(surface),
                centrality=float(node.get("centrality", 0.0) or 0.0),
                tier=str(node.get("tier", "episodic")),
                tags=list(node.get("tags") or []),
                language=str(node.get("language", "en") or "en"),
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


import os as _os_phase24  # noqa: E402 -- local alias to avoid os import collision
HISTORICAL_VERBATIM_DOWNWEIGHT: float = float(
    _os_phase24.environ.get("IAI_MCP_HISTORICAL_VERBATIM_DOWNWEIGHT", "0.25"),
)


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
    return COMMUNITY_BIAS_CONCEPT if mode == "concept" else COMMUNITY_BIAS_VERBATIM


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


PROFILE_SENTINEL_UUID = UUID("00000000-0000-0000-0000-0000000000f1")


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
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def _aaak_overlap(cue_text: str, aaak_index: str) -> float:
    if not aaak_index:
        return 0.0
    cue_set = set(cue_text.lower().replace("/", " ").split())
    idx_set = set(aaak_index.lower().replace("/", " ").split())
    if not cue_set or not idx_set:
        return 0.0
    return len(cue_set & idx_set) / len(cue_set | idx_set)


def _age_penalty(created_at: datetime, now: datetime) -> float:
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    # Round to whole seconds: AGE_HALF_LIFE_DAYS is a days-scale constant, so
    # sub-second creation-time jitter (e.g. between records seeded in a tight
    # loop) carries no real signal here but is fine-grained enough to break
    # float-equality ties between otherwise-identical records, non-reproducibly,
    # since it depends on the exact instant `now` happens to be read.
    seconds = round((now - created_at).total_seconds())
    if seconds < 0:
        return 0.0
    days = seconds / 86400.0
    return min(1.0, days / AGE_HALF_LIFE_DAYS)


def _community_gate(
    cue_emb: list[float],
    assignment: CommunityAssignment,
    top_n: int = 3,
    member_embeddings: dict[UUID, list[float]] | None = None,
) -> list[UUID]:
    cue_vec = np.asarray(cue_emb, dtype=np.float32)
    cue_norm = float(np.linalg.norm(cue_vec))
    if cue_norm > 0.0:
        cue_vec = cue_vec / cue_norm

    if member_embeddings is not None:
        return _community_gate_max_node(
            cue_vec, assignment, top_n, member_embeddings,
        )

    centroids = assignment.community_centroids
    if not centroids:
        return []
    cids = list(centroids.keys())
    mat = np.asarray(
        [centroids[c] for c in cids], dtype=np.float32
    )
    norms = np.linalg.norm(mat, axis=1)
    norms[norms == 0.0] = 1.0
    mat = mat / norms[:, None]
    scores = mat @ cue_vec
    order = np.argsort(-scores, kind="stable")
    return [cids[int(i)] for i in order[:top_n]]


def _community_gate_max_node(
    cue_vec: np.ndarray,
    assignment: CommunityAssignment,
    top_n: int,
    member_embeddings: dict[UUID, list[float] | np.ndarray],
) -> list[UUID]:
    mid_regions = assignment.mid_regions
    if not mid_regions:
        return _community_gate(
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
        return []

    mat = np.stack(rows).astype(np.float32, copy=False)
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
    return [lex_sorted_cids[int(i)] for i in score_order[:top_n]]


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
    pool_ids: list[UUID] = []
    pool_embs_rows: list[list[float]] = []
    for rid in graph.iter_nodes():
        emb: list[float] | None = None
        node_emb = graph.get_embedding(rid)
        if node_emb:
            emb = list(node_emb)
        if not emb and records_cache is not None and rid in records_cache:
            rec = records_cache[rid]
            cached_emb = getattr(rec, "embedding", None)
            if cached_emb:
                emb = list(cached_emb)
        if not emb:
            try:
                rec = store.get(rid)
                if rec is not None and rec.embedding:
                    emb = list(rec.embedding)
            except Exception as exc:  # noqa: BLE001 -- retrieval hot-path fail-safe
                logger.debug("collect_graph_pool_store_fallback_failed rid=%s: %s", rid, exc)
                emb = None
        if emb:
            pool_ids.append(rid)
            pool_embs_rows.append(emb)
    if not pool_ids:
        return [], np.zeros((0, store.embed_dim), dtype=np.float32)
    return pool_ids, np.asarray(pool_embs_rows, dtype=np.float32)


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
        with store.db._conn_lock:
            rows = store.db._conn.execute(sql, params).fetchall()
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

    _log_malformed_anti_edges(store, hit_ids)

    try:
        _contr_map = store.incident_edges(
            hit_ids, edge_types=["contradicts"], top_k=None,
        )
    except Exception as exc:  # noqa: BLE001 -- anti-hits is enrichment; degrade to []
        logger.debug("_find_anti_hits incident_edges failed: %s", exc)
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
    for aid in anti_ids[:k]:
        rec = records_cache.get(aid) if records_cache is not None else None
        if rec is None:
            rec = store.get(aid)
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
            )
        )
    return out


_last_recall_latency_ms: float = 0.0


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
) -> _RecallCoreResult:
    profile_state = profile_state or {}

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
            )
            try:
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

    try:
        cue_emb = embedder.embed(cue)
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

    records_cache: dict[UUID, "object"] = {}
    try:
        for rid in graph.iter_nodes():
            node = graph.get_payload(rid)
            if "embedding" not in node or "surface" not in node:
                continue
            records_cache[rid] = SimpleRecordView(
                id=rid,
                embedding=list(node["embedding"]),
                literal_surface=str(node.get("surface", "")),
                centrality=float(node.get("centrality", 0.0) or 0.0),
                tier=str(node.get("tier", "episodic")),
                tags=list(node.get("tags") or []),
                language=str(node.get("language", "en") or "en"),
            )
    except Exception as exc:  # noqa: BLE001 -- retrieval hot-path fail-safe
        logger.debug("records_cache_graph_build_failed: %s", exc)
        records_cache = {}
    if not records_cache:
        records_cache = {r.id: r for r in store.all_records()}

    episodic_ids: set | None = None
    if mode == "verbatim":
        episodic_ids = {
            cid for cid, rec in records_cache.items()
            if getattr(rec, "tier", "episodic") == "episodic"
        }

    _pool_t0 = time.perf_counter()
    pool_ids, pool_embs = _collect_graph_pool(graph, records_cache, store)
    _recall_pool_collection_ms = (time.perf_counter() - _pool_t0) * 1000.0
    cue_vec = np.asarray(cue_emb, dtype=np.float32)
    cnorm = float(np.linalg.norm(cue_vec))
    if cnorm > 0.0:
        cue_vec = cue_vec / cnorm
    if pool_embs.size:
        pool_embs = np.nan_to_num(pool_embs, nan=0.0, posinf=0.0, neginf=0.0)
        pool_norms = np.linalg.norm(pool_embs, axis=1)
        pool_norms[pool_norms == 0.0] = 1.0
        pool_embs = pool_embs / pool_norms[:, None]
        shared_cos = np.matmul(pool_embs, cue_vec).astype(np.float32)
    else:
        shared_cos = np.empty(0, dtype=np.float32)
    if shared_cos.size:
        shared_order = np.argsort(-shared_cos, kind="stable")
        cosine_top_indices = shared_order[:K_CANDIDATES]
    else:
        shared_order = np.empty(0, dtype=np.int64)
        cosine_top_indices = np.empty(0, dtype=np.int64)

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
    gated = _community_gate(
        cue_emb, assignment, top_n=k_communities,
        member_embeddings=gate_member_embeddings,
    )
    gated_set: set[UUID] = set()
    for gc in gated:
        for rid in assignment.mid_regions.get(gc, []):
            gated_set.add(rid)

    _centrality_t0 = time.perf_counter()
    centrality_arr = np.zeros(len(pool_ids), dtype=np.float32)
    for i, rid in enumerate(pool_ids):
        centrality_arr[i] = float(graph.get_centrality(rid))
    if not np.any(centrality_arr) and pool_ids:
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

    seed_indices = _pick_seeds(
        cosine_top_indices, shared_cos, centrality_arr, n=3,
    )
    seed_ids = [pool_ids[int(i)] for i in seed_indices]

    spread_ids = graph.two_hop_neighborhood(seed_ids, top_k=5) if spread_hops > 0 else []
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
    if cosine_top_indices.size or spread_indices.size or rich_indices.size:
        reachable_indices = np.union1d(
            np.union1d(cosine_top_indices, spread_indices),
            rich_indices,
        ).astype(np.int64)
    else:
        reachable_indices = np.empty(0, dtype=np.int64)

    pre_filter_reachable_ids = [pool_ids[int(i)] for i in reachable_indices]
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
    effective_w_degree = W_DEGREE * lp_scale
    if mode == "verbatim":
        effective_w_degree = 0.0

    if structural_weight > 0.0:
        from iai_mcp import tem
        cue_structure_hv = tem.pack_pairs([("TOPIC", tem.filler_hv(cue))])

    max_deg = float(getattr(graph, "_max_degree", 0) or 0)
    log_max_deg = log(1.0 + max_deg) if max_deg > 0 else 0.0
    _global_deg_override: "dict[str, int] | None" = getattr(graph, "_global_degree", None)
    if _global_deg_override:
        degree = _global_deg_override
    else:
        degree = {str(nid): deg for nid, deg in graph.degrees()}

    mode_bias = _gate_bias_for_mode(mode)
    mode_bias = mode_bias + _arousal_mode_bias_adjust

    fts_hits: set[UUID] = set()
    if cue and len(cue) >= 4:
        cue_lower = cue.lower()
        for rid, rec in records_cache.items():
            if rec.literal_surface and cue_lower in rec.literal_surface.lower():
                fts_hits.add(rid)

    contradicts_dst_set: set[str] = set()
    if cue_intent == "historical_verbatim":
        contradicts_dst_set = _build_contradicts_dst_set(contradicts_outgoing)

    corrector_base_score: dict[str, float] = {}

    _scoring_now = datetime.now(timezone.utc)

    scored: list[tuple[float, UUID, float, float, float, float, float, float]] = []
    if reachable_indices.size:
        from iai_mcp.hebbian_structure import structural_similarity
        for idx in reachable_indices:
            i = int(idx)
            cid = pool_ids[i]
            rec = records_cache.get(cid)
            if rec is None:
                continue
            cos = float(shared_cos[i])
            aaak = _aaak_overlap(cue, rec.aaak_index)
            deg = float(degree.get(str(cid), 0))
            age = _age_penalty(rec.created_at, _scoring_now)
            if log_max_deg > 0.0:
                deg_norm = log(1.0 + deg) / log_max_deg
            else:
                deg_norm = 0.0
            base_s = (
                W_COSINE * cos
                + W_AAAK * aaak
                + effective_w_degree * deg_norm
                - W_AGE * age
            )
            if cid in gated_set:
                base_s += mode_bias * cos
            structural_score = 0.0
            if (
                structural_weight > 0.0
                and cue_structure_hv is not None
                and rec.structure_hv
            ):
                structural_score = structural_similarity(
                    cue_structure_hv, rec.structure_hv,
                )
            if structural_weight > 0.0:
                base_s = (
                    (1.0 - structural_weight) * base_s
                    + structural_weight * structural_score
                )
            if profile_state:
                gains = profile_modulation_for_record(
                    rec, profile_state, knobs_applied=knobs_applied,
                )
                if gains:
                    rec.profile_modulation_gain = dict(gains)
                    gain_product = 1.0
                    for gv in gains.values():
                        try:
                            gain_product *= float(gv)
                        except (TypeError, ValueError):
                            continue
                    s = base_s * gain_product
                else:
                    s = base_s
            else:
                s = base_s
            try:
                _stability = getattr(rec, "stability", 0.5) or 0.5
                _ig = (1.0 - min(float(_stability), 1.0)) * 0.1
                s += _ig
            except (TypeError, ValueError, AttributeError) as exc:
                logger.debug("stability_lift_failed: %s", exc)
            _valence = getattr(rec, "valence", None) or 0.0
            if _valence > 0.0:
                s *= (1.0 + _valence)
            if cue and rec.literal_surface and _trigram_jaccard(cue.lower(), rec.literal_surface.lower()) > 0.3:
                s *= 2.0
            if fts_hits and cid in fts_hits:
                s *= 3.0
            if cue_intent == "historical_verbatim" and contradicts_dst_set:
                if str(cid) in contradicts_dst_set:
                    corrector_base_score[str(cid)] = s
            scored.append(
                (s, cid, cos, aaak, deg, deg_norm, age, structural_score),
            )

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
                    scored[j] = (tgt,) + row[1:]

    scored.sort(key=lambda x: (-x[0], str(x[1])))

    scored_hits: list[MemoryHit] = []
    budget_used = 0
    for s, cid, cos, aaak, deg, deg_norm, age, structural_score in scored:
        rec = records_cache.get(cid)
        if rec is None:
            continue
        tokens = len(rec.literal_surface) // 4
        suggestions = graph.two_hop_neighborhood([cid], top_k=3)[:3]
        if structural_weight > 0.0:
            reason = (
                f"cos {cos:.3f} + aaak {aaak:.2f} "
                f"+ deg_norm {deg_norm:.3f} "
                f"- age {age:.2f} | structural {structural_score:.3f} "
                f"(w={structural_weight:.2f})"
            )
        else:
            reason = (
                f"cos {cos:.3f} + aaak {aaak:.2f} "
                f"+ deg_norm {deg_norm:.3f} "
                f"- age {age:.2f}"
            )
        _prov = (rec.provenance or [{}])[0]
        scored_hits.append(
            MemoryHit(
                record_id=cid,
                score=float(s),
                # Keep the raw, unclamped score as the internal ranking key so
                # post-rank re-sorts preserve engine order even when the
                # displayed score is later clamped to [0,1] at serialization.
                sort_score=float(s),
                reason=reason,
                literal_surface=rec.literal_surface,
                adjacent_suggestions=suggestions,
                session_id=_prov.get("session_id"),
                captured_at=rec.created_at.isoformat() if rec.created_at else None,
            ),
        )
        budget_used += tokens

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
            )
        except Exception as exc:  # noqa: BLE001 -- telemetry MUST NOT break recall
            logger.debug("recall_timing_emit_failed: %s", exc)

    return _RecallCoreResult(
        scored_hits=scored_hits,
        activation_trace=activation_trace,
        anti_hits=[],
        hints=[],
        patterns_observed=[],
        cue_mode=mode,
        budget_used=budget_used,
        _records_cache=records_cache,
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
) -> tuple[list[MemoryHit], list[MemoryHit], list[dict], list[dict]]:
    s4_scope_hits = hits[:_POST_RANK_MAX_HITS]

    if hits:
        try:
            from iai_mcp.provenance_buffer import defer_provenance
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

    _BOOST_SMALL_BATCH: int = 4
    if profile_state:
        modulate_pairs: list[tuple] = []
        modulate_deltas: list[float] = []
        for h in hits:
            try:
                rec = records_cache.get(h.record_id)
                if rec is None:
                    continue
                gains = getattr(rec, "profile_modulation_gain", None) or {}
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
        if modulate_pairs:
            try:
                for _chunk_start in range(0, len(modulate_pairs), _BOOST_SMALL_BATCH):
                    _chunk_pairs = modulate_pairs[_chunk_start:_chunk_start + _BOOST_SMALL_BATCH]
                    _chunk_deltas = modulate_deltas[_chunk_start:_chunk_start + _BOOST_SMALL_BATCH]
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

    if mode != "verbatim" and s4_scope_hits:
        try:
            write_event(
                store,
                kind="deferred_curiosity_input",
                data={
                    "hit_ids": [str(h.record_id) for h in s4_scope_hits[:10]],
                    "cue": cue[:200],
                    "session_id": session_id,
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
        write_event(
            store,
            kind="retrieval_used",
            data={
                "hit_ids": [str(h.record_id) for h in hits],
                "query": cue,
                "used": len(hits) > 0,
                "budget_used": budget_used,
                "path": path_label,
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

    _k_com = 1 if _last_recall_latency_ms > 2000 else 3
    _s_hops = 0 if _last_recall_latency_ms > 2000 else 2

    from iai_mcp.cue_router import _classify_cue
    from iai_mcp.retrieve import (
        apply_stale_downweight,
        build_temporal_validity_maps,
        derive_temporal_validity,
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
    )

    derive_temporal_validity(
        None, core.scored_hits, outgoing=_tv_outgoing, ts_by_id=_tv_ts,
    )
    derive_temporal_validity(
        None, core.anti_hits, outgoing=_tv_outgoing, ts_by_id=_tv_ts,
    )
    apply_stale_downweight(core.scored_hits, cue_intent=_cue_intent)
    apply_stale_downweight(core.anti_hits, cue_intent=_cue_intent)
    # Rank on the internal unclamped key (falls back to score when a hit was
    # built without sort_score), so ordering is preserved across the display
    # clamp applied at serialization. Tie-break on record_id so equal-scoring
    # hits resolve deterministically -- two code paths that produce the same
    # score via different summation orders (e.g. empty profile_state falling
    # back to the medium scale) must yield byte-identical orderings.
    core.scored_hits.sort(
        key=lambda h: (
            -(h.sort_score if h.sort_score is not None else h.score),
            str(h.record_id),
        ),
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

    try:
        _pending_n = max(10, len(hits))
        _pending_markers = store.recent_pending_markers(n=_pending_n)
        _ranked_ids: set = {h.record_id for h in hits}
        for _pm in _pending_markers:
            if _pm.id not in _ranked_ids:
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
                ))
    except Exception as _pm_exc:  # noqa: BLE001 -- recency union is additive; never crash recall
        logger.debug("pending_markers_union_failed: %s", _pm_exc)

    for _h in hits:
        if _h.session_id is None:
            try:
                _full_rec = store.get(_h.record_id)
                if _full_rec is not None:
                    _h_prov = (_full_rec.provenance or [{}])[0]
                    _h.session_id = _h_prov.get("session_id")
                    _h.captured_at = (
                        _full_rec.created_at.isoformat()
                        if _full_rec.created_at else None
                    )
            except Exception as _exc:  # noqa: BLE001 -- additive enrichment, never crash recall
                logger.debug("hit_provenance_enrich_failed rid=%s: %s", _h.record_id, _exc)

    hits, anti_hits, hints, patterns_observed = _apply_post_rank_pipeline(
        hits,
        store=store, graph=graph, records_cache=core._records_cache,
        cue=cue, session_id=session_id,
        profile_state=profile_state, turn=turn, mode=mode,
        budget_used=budget_used, path_label="recall_for_response",
        knobs_applied=knobs_applied,
        contradicts_outgoing=_tv_outgoing,
    )

    if hits:
        _final_hits: list[MemoryHit] = []
        _final_budget = 0
        for _fh in hits:
            _fh_tokens = len(_fh.literal_surface) // 4
            if _final_budget + _fh_tokens > budget_tokens and _final_hits:
                break
            _final_hits.append(_fh)
            _final_budget += _fh_tokens
        hits = _final_hits
        budget_used = _final_budget

    derive_temporal_validity(
        None, anti_hits, outgoing=_tv_outgoing, ts_by_id=_tv_ts,
    )
    apply_stale_downweight(anti_hits)

    _last_recall_latency_ms = (_time.perf_counter() - _rfr_t0) * 1000

    return RecallResponse(
        hits=hits,
        anti_hits=anti_hits,
        activation_trace=core.activation_trace,
        budget_used=budget_used,
        hints=hints,
        cue_mode=core.cue_mode,
        patterns_observed=patterns_observed,
    )


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
) -> RecallResponse:
    from iai_mcp.cue_router import _classify_cue
    from iai_mcp.retrieve import build_temporal_validity_maps
    _cue_mode_unused, _cue_intent, _cue_label_unused = _classify_cue(cue)
    _tv_maps = build_temporal_validity_maps(store)
    _tv_outgoing, _tv_ts_unused = (_tv_maps if _tv_maps is not None else ({}, {}))

    core = _recall_core(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=cue, session_id=session_id,
        profile_state=profile_state, turn=turn, mode=mode,
        knobs_applied=knobs_applied,
        cue_intent=_cue_intent,
        contradicts_outgoing=_tv_outgoing,
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
    )

    return RecallResponse(
        hits=hits,
        anti_hits=anti_hits,
        activation_trace=core.activation_trace,
        budget_used=budget_used,
        hints=hints,
        cue_mode=core.cue_mode,
        patterns_observed=patterns_observed,
    )
