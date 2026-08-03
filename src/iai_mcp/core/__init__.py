from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time as _time
import traceback
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from iai_mcp.exceptions import IAIMCPError, RetrievalError, EmbeddingError, StoreError, NativeError

from iai_mcp import profile, retrieve

logger = logging.getLogger(__name__)
from iai_mcp.aaak import enforce_english_raw, generate_aaak_index
from iai_mcp.concurrency import SOCKET_PATH
from iai_mcp.daemon_state import get_pending_digest, load_state
from iai_mcp.native_guard import _require_native
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


class UnknownMethodError(Exception):
    pass


def _passes_mode_filter(record: MemoryRecord, cue_mode: str) -> bool:
    """True when *record* is eligible to surface under *cue_mode*.

    Verbatim recall excludes non-episodic records (schema/pattern surfaces) —
    the same predicate `retrieve.recall` applies to its baseline candidates.
    Any tier that guarantees a record's presence (e.g. the exact-similarity
    authority) must still respect this exclusion: presence of a real memory is
    never license to resurrect a record class the active mode deliberately
    filters out. Concept mode applies no tier filter.
    """
    if cue_mode != "verbatim":
        return True
    return record.tier == "episodic" and not any(
        t.startswith("pattern:") for t in (record.tags or [])
    )


FORCE_WAKE_TIMEOUT_SEC: int = 15 * 60


from cachetools import TTLCache as _CoreTTLCache

_CORE_WARM_LRU: _CoreTTLCache = _CoreTTLCache(maxsize=50, ttl=300)
_CORE_CASCADE_FIRED_PER_SESSION: set[str] = set()

# Crisis-state is cached for 1 s so the recall hot path does at most one
# filesystem read per TTL window, not one per call. A 1 s TTL still reflects
# any crisis flip (S2 tick cadence is 30 s) well within the safety margin.
_CRISIS_STATE_CACHE: _CoreTTLCache = _CoreTTLCache(maxsize=1, ttl=1)


_profile_state: dict[str, Any] = profile.default_state()

_posterior_state: dict[str, Any] = {}

_arousal_state: object | None = None

_last_injection_embedding: list[float] | None = None
_last_injection_ids: list[str] = []

_profile_lock: threading.RLock = threading.RLock()

#: Snapshot cache for the deep topology surface. The deep compute walks the
#: full graph (clustering + APSL + sigma baselines + community detection);
#: serving it per-call let a status poller stack unbounded multi-minute
#: dispatches and starve the daemon for hours. At most ONE deep compute runs
#: at a time (non-blocking acquire = single-flight); concurrent and TTL-fresh
#: callers are served from the cache.
_TOPOLOGY_SNAPSHOT_TTL_S: float = 900.0
_topology_cache: dict[str, Any] | None = None
_topology_cache_at: float = 0.0
_topology_state_lock: threading.Lock = threading.Lock()
_topology_inflight: threading.Lock = threading.Lock()

LIVE_KNOBS: dict[str, Any] = _profile_state
DEFERRED_KNOBS: frozenset[str] = frozenset(profile.DEFERRED_KNOB_NAMES)
assert len(DEFERRED_KNOBS) == 0, "all 10 autistic-kernel knobs live"


def _crisis_degraded_last_resort() -> dict:
    # Fresh dict with a fresh ``hits`` list per call, so no two responses ever
    # alias the same list object.
    return {
        "hits": [],
        "_degraded": True,
        "_reason": "daemon_consolidation_stuck",
    }


def _crisis_degraded_recall(store: MemoryStore, params: dict) -> dict:
    """Serve recall through the bypass-safe tier while consolidation is
    stuck in crisis_mode.

    Consolidation health must never zero recall: this tier reads only
    consolidation-independent structures (raw ANN similarity plus the
    exact-cosine authority), never runtime_graph_cache, community
    assignments, or any other consolidation-maintained structure. Hits are
    empty ONLY when the data genuinely has no match or this tier itself
    fails -- never as an unconditional policy choice.
    """
    try:
        from iai_mcp.cue_router import _classify_cue
        from iai_mcp.embed import embed_query, embedder_for_store
        from iai_mcp.pipeline import K_CANDIDATES
        from iai_mcp.core._serializers import _hit_to_json
        from iai_mcp.types import MemoryHit

        cue_mode, _cue_intent, _triggered_pattern = _classify_cue(params.get("cue", ""))

        embedder = embedder_for_store(store)
        _cue_vec = embed_query(embedder, params["cue"])

        _ann_pairs = store.query_similar(_cue_vec, k=K_CANDIDATES)
        _candidate_recs: dict = {_r.id: _r for _r, _s in _ann_pairs}
        _ann_scores: dict = {_r.id: float(_s) for _r, _s in _ann_pairs}

        _authority_pairs: list = []
        _live_auth_ids: set = set()
        if os.environ.get("IAI_MCP_EXACT_AUTHORITY_OFF") != "1":
            try:
                # build_if_cold=False: a cold matrix kicks a background build
                # and this recall proceeds authority-degraded — index
                # maintenance never sits on the awake recall path.
                _authority_pairs = store.exact_top_k(
                    _cue_vec, k=10, build_if_cold=False,
                )
                if _authority_pairs:
                    _auth_id_strs = [str(_rid) for _rid, _s in _authority_pairs]
                    with store.db.ro_conn() as _conn:
                        _live_rows = _conn.execute(
                            "SELECT id FROM records WHERE id IN (%s)"
                            " AND tombstoned_at IS NULL"
                            % ",".join("?" * len(_auth_id_strs)),
                            _auth_id_strs,
                        ).fetchall()
                    _live_auth_ids = {str(_row[0]) for _row in _live_rows}
                    _authority_pairs = [
                        (_rid, _s)
                        for _rid, _s in _authority_pairs
                        if str(_rid) in _live_auth_ids
                    ]
                _auth_new = [
                    _rid for _rid, _s in _authority_pairs
                    if _rid not in _candidate_recs
                ]
                if _auth_new:
                    for _arid, _arec in store.get_batch(_auth_new).items():
                        _candidate_recs[_arid] = _arec
            except Exception as _ea_exc:  # noqa: BLE001 -- a broken authority
                # must never break the degraded tier.
                logger.debug("crisis_degraded_authority_failed: %s", _ea_exc)
                _authority_pairs = []
                _live_auth_ids = set()

        # Merge: authority hits head-ranked first (deduped by id, in
        # authority score order), then ANN pairs by score descending. The
        # mode filter applies to every candidate -- authority presence is
        # never license to resurrect a mode-excluded record class.
        _ordered_ids: list = []
        _seen_ids: set = set()
        for _rid, _score in _authority_pairs:
            if _rid in _seen_ids or _rid not in _candidate_recs:
                continue
            _seen_ids.add(_rid)
            _ordered_ids.append((_rid, float(_score)))
        _ann_sorted = sorted(_ann_pairs, key=lambda _pair: _pair[1], reverse=True)
        for _rec, _score in _ann_sorted:
            if _rec.id in _seen_ids:
                continue
            _seen_ids.add(_rec.id)
            _ordered_ids.append((_rec.id, float(_score)))

        _hits_json: list = []
        for _rid, _score in _ordered_ids:
            if len(_hits_json) >= 10:
                break
            _rec = _candidate_recs.get(_rid)
            if _rec is None:
                continue
            if not _passes_mode_filter(_rec, cue_mode):
                continue
            # Per-hit build + serialize: one hit that fails to assemble or
            # serialize is skipped, never collapsing the whole response to
            # the last-resort empty.
            try:
                _session_id = None
                if _rec.provenance:
                    _session_id = _rec.provenance[0].get("session_id")
                _captured_at = _rec.created_at.isoformat() if _rec.created_at else None
                _hit = MemoryHit(
                    record_id=_rid,
                    score=_score,
                    reason="crisis_degraded_direct",
                    literal_surface=_rec.literal_surface or "",
                    adjacent_suggestions=[],
                    session_id=_session_id,
                    captured_at=_captured_at,
                )
                _hits_json.append(_hit_to_json(_hit))
            except Exception as _hit_exc:  # noqa: BLE001 -- skip the bad hit,
                # keep the rest of the degraded response.
                logger.debug("crisis_degraded_hit_skipped: %s", _hit_exc)
                continue

        return {
            "hits": _hits_json,
            "anti_hits": [],
            "activation_trace": [],
            "budget_used": 0,
            "cue_mode": cue_mode,
            "_degraded": True,
            "_reason": "daemon_consolidation_stuck",
        }
    except Exception as exc:  # noqa: BLE001 -- a broken degraded tier must
        # never escape as an exception; last-resort empty response only.
        logger.warning("crisis_degraded_recall_failed: %s", exc)
        return _crisis_degraded_last_resort()


def _incident_edges_warm(
    store: "MemoryStore", ids: list, top_k: "int | None" = 5,
) -> dict:
    """Incident edges served from the in-process warm graph's RAM adjacency.

    The store implementation runs `src IN (...) OR dst IN (...)` over the
    whole edges table per recall hop — an engine scan the warm graph already
    holds in memory. Edge weights lag until the next bundle refresh, the same
    bounded staleness the bundle's centrality carries. Falls back to the
    store when no warm bundle exists (cold boot, tests without a build).
    Ordering and shape mirror store.incident_edges exactly.
    """
    memo = getattr(store, "_warm_graph_bundle", None)
    graph = memo[0][0] if memo is not None else None
    adj = getattr(graph, "_adj", None) if graph is not None else None
    if not adj:
        return store.incident_edges(ids, top_k=top_k)
    result: dict = {}
    for rid in ids:
        nbrs = adj.get(str(rid))
        if not nbrs:
            result[rid] = []
            continue
        try:
            items = list(nbrs.items())
        except RuntimeError:  # concurrent sync-hook mutation — one re-snapshot
            items = list(nbrs.items())
        rows = []
        for nbr_label, attrs in items:
            try:
                nbr = UUID(nbr_label)
            except (ValueError, AttributeError):
                continue
            rows.append((
                nbr,
                str(attrs.get("edge_type", "hebbian")),
                float(attrs.get("weight", 1.0)),
            ))
        rows.sort(key=lambda t: (-t[2], str(t[0]), t[1]))
        result[rid] = rows[:top_k] if top_k is not None else rows
    return result


def dispatch(store: MemoryStore, method: str, params: dict) -> dict:
    global _last_injection_embedding, _last_injection_ids, _arousal_state
    global _topology_cache, _topology_cache_at
    if method == "memory_recall":
        _recall_t0 = _time.perf_counter()
        # Stamp the foreground-activity beacon so polite background loops
        # (the deferred-capture drain and friends) yield the shared writer
        # connection and the GIL to this live read.
        try:
            from iai_mcp.concurrency import foreground_touch as _fg_touch
            _fg_touch()
        except Exception:  # noqa: BLE001 -- the beacon is advisory
            pass
        # Opt-in phase trace (IAI_MCP_RECALL_TRACE=1): cumulative
        # milliseconds-since-dispatch at each recall phase boundary, returned
        # as `_recall_trace_ms` so an operator can attribute a slow recall to
        # its exact phase without a profiler attached.
        _trace_spans: "list | None" = (
            [] if os.environ.get("IAI_MCP_RECALL_TRACE") == "1" else None
        )

        def _trace_mark(_name: str) -> None:
            if _trace_spans is not None:
                _trace_spans.append(
                    (_name, round((_time.perf_counter() - _recall_t0) * 1000.0, 1))
                )

        # crisis_mode honest-degrade: when consolidation is stuck (the
        # scheduler is looping a deferred step and cannot advance), the warm
        # recall path serves stale schema-dominated results. Honour the
        # always-available invariant by serving recall through the
        # bypass-safe tier (ANN + exact-cosine authority, both
        # consolidation-independent) rather than an unconditional empty
        # response -- consolidation health must never zero recall. The
        # client is still told via `_degraded` to prefer bank-recall for a
        # non-degraded answer.
        try:
            _crisis_state = _CRISIS_STATE_CACHE.get("crisis")
            if _crisis_state is None:
                from iai_mcp.lifecycle_state import load_state as _ls_load_cm
                _crisis_state = _ls_load_cm()
                _CRISIS_STATE_CACHE["crisis"] = _crisis_state
            if bool(_crisis_state.get("crisis_mode", False)):
                logger.warning(
                    "memory_recall served degraded under crisis_mode; "
                    "client should fall back to bank-recall"
                )
                return _crisis_degraded_recall(store, params)
        except Exception as exc:  # noqa: BLE001 -- never let the guard crash recall
            logger.debug("crisis_mode load_state failed; serving warm path: %s", exc)
        from iai_mcp.cue_router import _classify_cue
        cue_mode, _cue_intent, _triggered_pattern = _classify_cue(params.get("cue", ""))

        knobs_applied: dict[str, str] = {}
        _wake_depth_value = (_profile_state or {}).get("wake_depth", "minimal")
        if _wake_depth_value not in ("minimal", "standard", "deep"):
            _wake_depth_value = "minimal"
        knobs_applied["MCP-12"] = (
            f"session.py:assemble_session_start:wake_depth={_wake_depth_value}"
        )

        _arousal_budget_tokens: int = 1500
        _arousal_retrieval_params = None
        _arousal_diag: dict | None = None
        try:
            from iai_mcp.arousal_budget import (
                ArousalState as _ArousalState,
                compute_retrieval_params as _compute_retrieval_params,
                update_arousal as _update_arousal,
            )
            global _arousal_state
            if _arousal_state is None:
                _arousal_state = _ArousalState()
            _arousal_retrieval_params = _compute_retrieval_params(_arousal_state)
            _arousal_budget_tokens = _arousal_retrieval_params.budget_tokens
            _arousal_diag = {
                "level": _arousal_state.level,
                "mode": _arousal_retrieval_params.mode,
            }
        except Exception as exc:  # noqa: BLE001 -- graceful degradation
            logger.debug("arousal_budget_init_failed: %s", exc)
            _arousal_budget_tokens = 1500
            _arousal_diag = None

        _cortex_fallback = False
        _structural_source: str = ""
        _authority_pairs: list = []
        # Set True only when merge_authority_hits actually folds at least one
        # authority hit into the response -- never merely because seeding
        # found candidates (the merge can still no-op on an empty
        # _auth_hits, throw, or never run at all on the cortex-fallback
        # path).
        _exact_authority_used = False
        # Emptiness gate via the corpus-count cache (O(1) warm): a raw
        # count_rows on the lilli engine re-scans every leaf page, and this
        # gate sits on a per-call read path.
        records_count = store.active_records_count()
        if records_count == 0:
            cue_embedding = params.get("cue_embedding") or [0.0] * EMBED_DIM
            resp = retrieve.recall(
                store=store,
                cue_embedding=cue_embedding,
                cue_text=params["cue"],
                session_id=params.get("session_id", "unknown"),
                budget_tokens=params.get("budget_tokens") or _arousal_budget_tokens,
                mode=cue_mode,
            )
        else:
            from iai_mcp.embed import embed_query, embedder_for_store
            from iai_mcp.pipeline import recall_for_response
            try:
                from iai_mcp.daemon_state import load_state as _ds_load
                _ds = _ds_load()
                if _ds.get("current_state", "WAKE") in ("SLEEP", "DREAMING"):
                    cue_embedding = params.get("cue_embedding") or [0.0] * EMBED_DIM
                    resp = retrieve.recall(
                        store=store,
                        cue_embedding=cue_embedding,
                        cue_text=params["cue"],
                        session_id=params.get("session_id", "unknown"),
                        budget_tokens=params.get("budget_tokens") or _arousal_budget_tokens,
                        mode=cue_mode,
                    )
                    _cortex_fallback = True
            except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
                logger.debug("cqrs_sleep_detection_failed: %s", exc)
            if not _cortex_fallback:
                try:
                    from iai_mcp import runtime_graph_cache as _rgc
                    from iai_mcp.graph import MemoryGraph
                    from iai_mcp.pipeline import K_CANDIDATES

                    embedder = embedder_for_store(store)

                    assignment, rc, _cached_max_degree, _structural_source, _cached_node_degrees = _rgc.load_recall_structural(store)
                    _trace_mark("structural")

                    _encode_ms: "float | None" = None
                    _encode_t0 = _time.perf_counter()
                    try:
                        _cue_vec = embed_query(embedder, params["cue"])
                        _encode_ms = (_time.perf_counter() - _encode_t0) * 1000.0
                        _trace_mark("encode")
                    except Exception as _emb_exc:
                        try:
                            from iai_mcp.events import write_event, TELEMETRY_EMBED_NATIVE_FAILURE
                            write_event(
                                store,
                                TELEMETRY_EMBED_NATIVE_FAILURE,
                                {"op_type": "recall_cue", "error": str(_emb_exc)},
                                severity="critical",
                                buffered=True,
                            )
                        except Exception:  # noqa: BLE001
                            pass
                        raise NativeError(f"recall cue encode failed: {_emb_exc}") from _emb_exc

                    _ann_pairs = store.query_similar(_cue_vec, k=K_CANDIDATES)
                    _candidate_recs: dict = {_r.id: _r for _r, _s in _ann_pairs}
                    _trace_mark("ann")

                    # Exact-similarity authority: a bounded exact-cosine scan
                    # that backstops the fast (approximate) index above. The
                    # scan is ~0.5 ms measured, always-on, and its ids are
                    # seeded into the candidate set now so hop-1/hop-2 spread
                    # can follow associative threads from guaranteed-correct
                    # memories. The merge that makes the authority hits
                    # unconditionally head-ranked happens later, after the
                    # graph pipeline returns.
                    _authority_pairs: list = []
                    _live_auth_ids: set = set()
                    if os.environ.get("IAI_MCP_EXACT_AUTHORITY_OFF") != "1":
                        try:
                            # build_if_cold=False: a cold matrix kicks a
                            # background build and this recall proceeds
                            # authority-degraded — index maintenance never
                            # sits on the awake recall path.
                            _authority_pairs = store.exact_top_k(
                                _cue_vec, k=10, build_if_cold=False,
                            )
                            _trace_mark("auth_topk")
                            if _authority_pairs:
                                # Liveness re-check against live SQL. Neither
                                # by-id resolution nor the resolved record
                                # object can prove liveness on their own: the
                                # store's get/get_batch apply no tombstone
                                # filter, and the record type carries no
                                # tombstone field to inspect. Only a plaintext
                                # id-set SQL query can prove which of these ids
                                # are still live, so a stale matrix (or an
                                # ever-missed invalidate seam) can never
                                # surface an erased record here.
                                _auth_id_strs = [str(_rid) for _rid, _s in _authority_pairs]
                                with store.db.ro_conn() as _conn:
                                    _live_rows = _conn.execute(
                                        "SELECT id FROM records WHERE id IN (%s)"
                                        " AND tombstoned_at IS NULL"
                                        % ",".join("?" * len(_auth_id_strs)),
                                        _auth_id_strs,
                                    ).fetchall()
                                _live_auth_ids = {str(_row[0]) for _row in _live_rows}
                                _authority_pairs = [
                                    (_rid, _s)
                                    for _rid, _s in _authority_pairs
                                    if str(_rid) in _live_auth_ids
                                ]
                            _trace_mark("auth_liveness")
                            _auth_new = [
                                _rid for _rid, _s in _authority_pairs
                                if _rid not in _candidate_recs
                            ]
                            if _auth_new:
                                for _arid, _arec in store.get_batch(_auth_new).items():
                                    _candidate_recs[_arid] = _arec
                        except Exception as _ea_exc:  # noqa: BLE001 -- a broken
                            # authority must never break recall.
                            logger.debug("exact_authority_seed_failed: %s", _ea_exc)
                            _authority_pairs = []
                            _live_auth_ids = set()

                    _trace_mark("authority")
                    _hop1_edges = _incident_edges_warm(store, list(_candidate_recs.keys()), top_k=5)
                    _trace_mark("hop1_edges")
                    _hop1_new_ids = list({
                        _nbr
                        for _nbr_list in _hop1_edges.values()
                        for (_nbr, _et, _wt) in _nbr_list
                        if _nbr not in _candidate_recs
                    })
                    if _hop1_new_ids:
                        _candidate_recs.update(store.get_batch(_hop1_new_ids))
                    _trace_mark("hop1_fetch")

                    _hop2_edges = _incident_edges_warm(store, _hop1_new_ids, top_k=5) if _hop1_new_ids else {}
                    _trace_mark("hop2_edges")
                    _hop2_new_ids = list({
                        _nbr
                        for _nbr_list in _hop2_edges.values()
                        for (_nbr, _et, _wt) in _nbr_list
                        if _nbr not in _candidate_recs
                    })
                    if _hop2_new_ids:
                        _candidate_recs.update(store.get_batch(_hop2_new_ids))
                    _trace_mark("hop2_fetch")

                    _RC_CAP = 50
                    _rc_cap = (rc or [])[:_RC_CAP]
                    _rc_new_ids = [_rid for _rid in _rc_cap if _rid not in _candidate_recs]
                    if _rc_new_ids:
                        _candidate_recs.update(store.get_batch(_rc_new_ids))

                    _trace_mark("hops")
                    graph = MemoryGraph()
                    for _rec in _candidate_recs.values():
                        graph.add_node(
                            _rec.id,
                            community_id=getattr(_rec, "community_id", None),
                            embedding=list(_rec.embedding or []),
                        )
                        graph.set_node_payload(_rec.id, {
                            "embedding": list(_rec.embedding or []),
                            "surface": _rec.literal_surface or "",
                            "centrality": float(getattr(_rec, "centrality", 0.0) or 0.0),
                            "tier": _rec.tier or "episodic",
                            "tags": list(_rec.tags or []),
                            "language": _rec.language or "en",
                        })
                    for _qid, _nbr_list in _hop1_edges.items():
                        for (_nbr, _et, _wt) in _nbr_list:
                            if _nbr in _candidate_recs:
                                try:
                                    graph.add_edge(_qid, _nbr, weight=_wt, edge_type=_et)
                                except Exception:  # noqa: BLE001 — edge add fail-safe
                                    pass
                    for _qid2, _nbr_list2 in _hop2_edges.items():
                        for (_nbr2, _et2, _wt2) in _nbr_list2:
                            if _nbr2 in _candidate_recs:
                                try:
                                    graph.add_edge(_qid2, _nbr2, weight=_wt2, edge_type=_et2)
                                except Exception:  # noqa: BLE001 — edge add fail-safe
                                    pass

                    # The hebbian degree-map traversal and the contradicts
                    # traversal both run over the identical final candidate
                    # set, differing only in which edge type they keep. Fetch
                    # both types in one call and split the result in Python —
                    # halves the edge-materialization volume on this pass
                    # without changing either consumer's edge set.
                    _HEBB_TRAVERSAL_CAP = 50
                    _all_cand_ids = list(_candidate_recs.keys())
                    _paired_edges: "dict | None" = None
                    _paired_fetch_failed = False
                    try:
                        _paired_edges = store.incident_edges(
                            _all_cand_ids,
                            edge_types=["hebbian", "contradicts"],
                            top_k=None,
                            neighbor_keys_as_str=True,
                        )
                    except Exception as _pf_exc:  # noqa: BLE001 — degrade gracefully
                        logger.debug("layer1_paired_edge_fetch_failed: %s", _pf_exc)
                        _paired_fetch_failed = True

                    try:
                        if _paired_fetch_failed:
                            raise RuntimeError("paired edge fetch unavailable")
                        _hebb_split = {
                            _q: [_t for _t in _lst if _t[1] == "hebbian"]
                            for _q, _lst in _paired_edges.items()
                        }
                        # Per-node edge budget for the hebbian traversal — bounds
                        # the fan-out for latency without clamping the ranking degree.
                        # A hub with true degree > _HEBB_TRAVERSAL_CAP returns a
                        # capped edge list here but its REAL degree still feeds
                        # deg_norm (read from the cached per-node map below).
                        #
                        # Warm path: bound the traversal (cheap) and read the true
                        # per-node degree from the cached map so an over-cap hub
                        # keeps its real degree (deg_norm numerator unchanged vs
                        # the unbounded pass).
                        #
                        # Cold path (empty cache — e.g. immediately after a cache
                        # version bump, before the daemon rebuilds): the map is
                        # unavailable, so len(traversal) IS the ranking degree.
                        # A bounded traversal would flatten every hub with true
                        # degree >= the cap to the same value and corrupt hub
                        # ranking on the first recall. Keep the degree-count
                        # split UNBOUNDED on the cold path so len() is the true
                        # degree; the bound is a warm-path-only latency
                        # optimization and pays off only when the map exists.
                        if _cached_node_degrees:
                            _global_edges_hebb = {
                                _q: sorted(_lst, key=lambda _t: (-_t[2], str(_t[0]), _t[1]))[:_HEBB_TRAVERSAL_CAP]
                                for _q, _lst in _hebb_split.items()
                            }
                            graph._global_degree = {
                                str(_cid): _cached_node_degrees.get(_cid, len(_nbrs))
                                for _cid, _nbrs in _global_edges_hebb.items()
                            }
                        else:
                            _global_edges_hebb = _hebb_split  # cold path: true unbounded degree count
                            # Hebbian-only ON PURPOSE: a node whose only edge
                            # is contradicts must not gain rank from it — on a
                            # small candidate set that degree alone lifts an
                            # irrelevant contradicted record into hits, which
                            # both pollutes recall and starves the anti-hit
                            # channel (it only flags non-hits).
                            graph._global_degree = {
                                str(_cid): len(_nbrs)
                                for _cid, _nbrs in _global_edges_hebb.items()
                            }
                        if _cached_max_degree > 0:
                            graph._max_degree = int(_cached_max_degree)
                        else:
                            _local_max = max(graph._global_degree.values(), default=0)
                            if _local_max > 0:
                                graph._max_degree = _local_max
                    except Exception as _gd_exc:  # noqa: BLE001 — degrade gracefully
                        logger.debug("layer1_global_degree_failed: %s", _gd_exc)

                    _tv_outgoing_l1: dict[str, list[str]] = {}
                    _tv_ts_l1: dict = {}
                    try:
                        if _paired_fetch_failed:
                            raise RuntimeError("paired edge fetch unavailable")
                        _contr_edges = {
                            _q: [_t for _t in _lst if _t[1] == "contradicts"]
                            for _q, _lst in _paired_edges.items()
                        }
                        for _rec in _candidate_recs.values():
                            _ca = getattr(_rec, "created_at", None)
                            if _ca is not None:
                                _tv_ts_l1[str(_rec.id)] = _ca
                        # String-neighbour mode: _tv_outgoing_l1 needs str keys
                        # and str values; the candidate-presence check is done
                        # against the string form of candidate ids so no UUID
                        # is constructed from the neighbour string.
                        _cand_str_set = {str(_k) for _k in _candidate_recs}
                        _contr_dst_ids = []
                        for _src_id, _edges in _contr_edges.items():
                            for (_dst_s, _et, _wt) in _edges:
                                _src_s = str(_src_id)
                                _tv_outgoing_l1.setdefault(_src_s, []).append(_dst_s)
                                if _dst_s not in _cand_str_set:
                                    try:
                                        _contr_dst_ids.append(UUID(_dst_s))
                                    except (ValueError, AttributeError):
                                        pass
                        if _contr_dst_ids:
                            _contr_recs = store.get_batch(_contr_dst_ids)
                            for _cr in _contr_recs.values():
                                _ca = getattr(_cr, "created_at", None)
                                if _ca is not None:
                                    _tv_ts_l1[str(_cr.id)] = _ca
                    except Exception as _tv_exc:  # noqa: BLE001 — degrade gracefully
                        logger.debug("layer1_tv_build_failed: %s", _tv_exc)
                        _tv_outgoing_l1, _tv_ts_l1 = {}, {}

                    _trace_mark("graph_edges")
                    resp = recall_for_response(
                        store=store,
                        graph=graph,
                        assignment=assignment,
                        rich_club=rc,
                        embedder=embedder,
                        cue=params["cue"],
                        session_id=params.get("session_id", "unknown"),
                        budget_tokens=params.get("budget_tokens") or _arousal_budget_tokens,
                        profile_state=_profile_state,
                        mode=cue_mode,
                        knobs_applied=knobs_applied,
                        arousal_state=_arousal_diag,
                        tv_maps=(_tv_outgoing_l1, _tv_ts_l1) if _tv_ts_l1 else None,
                        trace_mark=_trace_mark,
                    )
                    resp.ann_path_used = True
                    _trace_mark("pipeline")

                    # Authority merge: union the exact-similarity hits found
                    # above (head, exact-cos order) with the graph pipeline's
                    # associative hits (tail, existing rank order) and re-pack
                    # the token budget over the union. Authority hits ARE
                    # final hits, so each one gets its real decrypted surface
                    # here rather than any deferred/candidate-only surface.
                    if _authority_pairs:
                        try:
                            from iai_mcp.pipeline import merge_authority_hits
                            from iai_mcp.types import MemoryHit

                            _auth_new_ids = [
                                _rid for _rid, _s in _authority_pairs
                                if str(_rid) in _live_auth_ids
                            ]
                            _auth_full_recs = (
                                store.get_batch(_auth_new_ids) if _auth_new_ids else {}
                            )
                            _auth_hits = []
                            for _rid, _acos in _authority_pairs:
                                if str(_rid) not in _live_auth_ids:
                                    # not proven live by the id-set liveness
                                    # query -- must never surface as a hit
                                    continue
                                _arec = _auth_full_recs.get(_rid)
                                if _arec is None:
                                    # unresolvable (bounded index staleness);
                                    # never fabricate a hit for it
                                    continue
                                if not _passes_mode_filter(_arec, cue_mode):
                                    # The authority guarantees presence of real
                                    # memories, never resurrection of a record
                                    # class the active recall mode deliberately
                                    # excludes (e.g. schema/meta records under a
                                    # verbatim cue).
                                    continue
                                _auth_hits.append(MemoryHit(
                                    record_id=_rid,
                                    score=float(_acos),
                                    reason="exact-cosine",
                                    literal_surface=_arec.literal_surface or "",
                                    adjacent_suggestions=[],
                                    session_id=(_arec.provenance or [{}])[0].get("session_id"),
                                    captured_at=(
                                        _arec.created_at.isoformat()
                                        if _arec.created_at else None
                                    ),
                                ))
                            if _auth_hits:
                                # This authority merge is unconditional by design: exact_top_k
                                # (k=10, build_if_cold=False) is measured ~0.5ms, so there is no
                                # latency case for gating it. The confidence signal drives ONLY
                                # the pre-rank widen in pipeline.py (see the conf_escalate
                                # branch in _recall_core), never this merge.
                                _auth_budget = params.get("budget_tokens") or _arousal_budget_tokens
                                resp.hits, resp.budget_used = merge_authority_hits(
                                    resp.hits, _auth_hits, _auth_budget,
                                )
                                _exact_authority_used = True

                                # The authority guarantees INCLUSION in the
                                # response (no false negatives), never a frozen
                                # ORDER: temporal-validity downweight (fresh
                                # outranks stale/contradicted) is a correctness
                                # signal and must still apply across the merged
                                # union, including the authority head. Apply it
                                # here -- the freshly-built authority hits never
                                # went through it -- then re-sort by score.
                                # Sorting only ever reorders; merge_authority_hits
                                # already fixed which hits are present and the
                                # token budget, so no hit can be dropped here.
                                from iai_mcp.retrieve import (
                                    apply_stale_downweight as _apply_stale_downweight,
                                    derive_temporal_validity as _derive_temporal_validity,
                                )
                                _derive_temporal_validity(
                                    None, resp.hits,
                                    outgoing=_tv_outgoing_l1, ts_by_id=_tv_ts_l1,
                                )
                                _apply_stale_downweight(
                                    resp.hits, cue_intent=_cue_intent,
                                )
                                from iai_mcp.retrieve import (
                                    sort_served_hits as _sort_served_hits,
                                )
                                _sort_served_hits(resp.hits)
                        except Exception as _em_exc:  # noqa: BLE001 -- a broken
                            # merge must never break recall; degrade to the
                            # pipeline-only hits already computed above.
                            logger.debug("exact_authority_merge_failed: %s", _em_exc)

                    _trace_mark("merge")
                    try:
                        from iai_mcp.events import emit_best_effort, TELEMETRY_RECALL_SOURCE
                        _du_data = {"source": "daemon"}
                        if _encode_ms is not None:
                            _du_data["encode_ms"] = round(_encode_ms, 2)
                        emit_best_effort(
                            store,
                            TELEMETRY_RECALL_SOURCE,
                            _du_data,
                            severity="info",
                            session_id=params.get("session_id", "unknown"),
                        )
                    except Exception:  # noqa: BLE001 -- telemetry must never break recall
                        pass
                except NativeError:
                    raise
                except Exception as exc:  # noqa: BLE001 -- soft availability fallback
                    logger.warning("recall_pipeline_fallback: %s", exc)
                    try:
                        _update_arousal(_arousal_state, "error")
                    except Exception:  # noqa: BLE001 -- arousal update fail-safe
                        pass
                    cue_embedding = params.get("cue_embedding") or [0.0] * EMBED_DIM
                    resp = retrieve.recall(
                        store=store,
                        cue_embedding=cue_embedding,
                        cue_text=params["cue"],
                        session_id=params.get("session_id", "unknown"),
                        budget_tokens=params.get("budget_tokens") or _arousal_budget_tokens,
                        mode=cue_mode,
                    )
        try:
            _arousal_event = "recall_success" if resp.hits else "recall_failed"
            _update_arousal(_arousal_state, _arousal_event)
        except Exception:  # noqa: BLE001 -- arousal update fail-safe
            pass

        response = {
            "hits": [_hit_to_json(h) for h in resp.hits],
            "anti_hits": [_hit_to_json(h) for h in resp.anti_hits],
            "activation_trace": [str(x) for x in resp.activation_trace],
            "budget_used": resp.budget_used,
            "cue_mode": resp.cue_mode,
            "patterns_observed": list(resp.patterns_observed or []),
            "hints": list(resp.hints or []),
            "_knobs_applied": knobs_applied,
            "ann_path_used": getattr(resp, "ann_path_used", False),
            "exact_authority_used": _exact_authority_used,
        }
        if _cortex_fallback:
            response["_source"] = "cortex-fallback"
        if not _cortex_fallback and _structural_source == "cold_degrade":
            response["_source"] = "cold-structural-degrade"
        if _trace_spans is not None:
            _trace_mark("respond")
            response["_recall_trace_ms"] = list(_trace_spans)
        try:
            _recall_ms = (_time.perf_counter() - _recall_t0) * 1000
            response["_recall_latency_ms"] = round(_recall_ms, 1)
        except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
            logger.debug("recall_latency_measure_failed: %s", exc)
        # Every agent-initiated search leaves a countable trace with its
        # payload weight — the denominator of the anticipation economy.
        try:
            from iai_mcp.events import write_event as _we
            _hits_list = response.get("hits") or []
            _resp_chars = sum(
                len(str(h.get("literal_surface") or "")) for h in _hits_list
            ) + 200 * max(len(_hits_list), 1)
            _we(
                store,
                "recall_dispatched",
                {"hits": len(_hits_list), "resp_chars": int(_resp_chars)},
                severity="info",
                session_id=params.get("session_id", "-"),
                buffered=True,
            )
        except Exception as exc:  # noqa: BLE001 -- telemetry MUST NOT break recall
            logger.debug("recall_dispatched_emit_failed: %s", exc)
        try:
            from iai_mcp.curiosity import get_pending_questions_cached
            _qs = get_pending_questions_cached(store, limit=2)
            if _qs:
                response["curiosity_signals"] = _qs
        except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
            logger.debug("curiosity_signals_failed: %s", exc)
        try:
            _reinforce_ids = [hit.record_id for hit in resp.hits]
            store.queue_reinforce(_reinforce_ids)
        except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
            logger.debug("labile_write_failed: %s", exc)
        try:
            from iai_mcp.retrieve import (
                WAKE_COACTIVATION_MIN_SCORE,
                potentiate_coactivation,
            )
            _wire_ids = [
                hit.record_id
                for hit in resp.hits
                if float(getattr(hit, "score", 0.0) or 0.0)
                >= WAKE_COACTIVATION_MIN_SCORE
            ]
            potentiate_coactivation(store, _wire_ids)
        except Exception as exc:  # noqa: BLE001 -- plasticity MUST NOT break recall
            logger.debug("coactivation_potentiate_failed: %s", exc)
        _inject_sleep_suggestion(
            response,
            cue=params.get("cue", ""),
            language=params.get("language", "en"),
        )
        _inject_overnight_digest(response, store=store)
        _first_turn_recall_hook(response, params=params, store=store)
        try:
            from iai_mcp.response_decorator import apply_profile
            apply_profile(response, _profile_state)
        except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
            logger.debug("apply_profile_failed: %s", exc)
        try:
            from iai_mcp.daemon_config import _load_pask_config
            from iai_mcp.events import write_event
            from iai_mcp.pask_teachback import verify_hit_set
            pask_cfg = _load_pask_config()
            if pask_cfg.enabled:
                hit_ids = [
                    h.record_id if hasattr(h, "record_id") else h.get("record_id")
                    for h in resp.hits
                ]
                hit_ids = [h for h in hit_ids if h is not None]
                teachback = verify_hit_set(store, hit_ids)
                response["pask_teachback"] = teachback
                try:
                    write_event(
                        store,
                        "pask_teachback_pass",
                        {
                            "hit_count": teachback["hit_count"],
                            "has_contradictions": teachback["has_contradictions"],
                            "contradiction_count": len(teachback["contradiction_pairs"]),
                            "dry_run_mode": pask_cfg.dry_run,
                        },
                        severity="info",
                    )
                except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
                    logger.debug("pask_teachback_event_failed: %s", exc)
        except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
            logger.debug("pask_teachback_failed: %s", exc)
        try:
            if resp.hits:
                import numpy as _np
                embeddings = [h.embedding for h in resp.hits if hasattr(h, "embedding") and h.embedding]
                if not embeddings:
                    _emb_cache = {}
                    _top_hits = resp.hits[:5]
                    _emb_batch = store.get_batch([h.record_id for h in _top_hits])
                    for h in _top_hits:
                        rec = _emb_batch.get(h.record_id)
                        if rec and rec.embedding:
                            _emb_cache[h.record_id] = rec.embedding
                    embeddings = list(_emb_cache.values())
                if embeddings:
                    _last_injection_embedding = _np.mean(embeddings, axis=0).tolist()
                    _last_injection_ids = [str(h.record_id) for h in resp.hits[:5]]
                else:
                    _last_injection_embedding = None
                    _last_injection_ids = []
        except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
            logger.debug("trajectory_coupling_store_failed: %s", exc)
            _last_injection_embedding = None
            _last_injection_ids = []
        return response

    if method == "brain_view":
        # Server side of the dashboard relay: while the daemon is alive it is
        # the single writer, so the view's verbs run HERE against the live
        # store and the HTTP process stays a thin client.
        from iai_mcp.brainview import BRAIN_VIEW_VERBS, view_for_store

        verb = str(params.get("verb") or "")
        if verb not in BRAIN_VIEW_VERBS:
            return {"status": "error", "reason": f"unknown brain verb {verb!r}"}
        kwargs = params.get("kwargs")
        if not isinstance(kwargs, dict):
            kwargs = {}
        return getattr(view_for_store(store), verb)(**kwargs)

    if method == "memory_search":
        # Scoped hybrid search: a lexical (identifier-exact) lane beside the
        # semantic one. This is the agent-facing replacement for an external
        # code-index tool — framed as hints, never as ground truth.
        query = str(params.get("query") or "").strip()
        if not query:
            return {"hits": [], "frame": "empty query"}
        try:
            k = max(1, min(int(params.get("k") or 8), 24))
        except (TypeError, ValueError):
            # A raw socket client can send any JSON type; a bad k must not
            # raise out of dispatch.
            k = 8

        # Reciprocal-rank fusion: BM25 and cosine live on incomparable
        # scales, so any score-first sort degrades hybrid to lexical-first
        # (k lexical matches evict EVERY semantic hit however strong).
        # 1/(60+rank) per lane sums naturally, so a record both lanes agree
        # on still rises to the top.
        merged: dict = {}
        try:
            for lex_rank, (rec, score) in enumerate(store.lexical_search(query, k=k)):
                merged[str(rec.id)] = {
                    "record_id": str(rec.id),
                    "surface": (rec.literal_surface or "")[:500],
                    "tier": rec.tier,
                    "lane": "lexical",
                    "score": float(score),
                    "_rrf": 1.0 / (60.0 + lex_rank),
                }
        except Exception as exc:  # noqa: BLE001 -- one lane failing must not blank the other
            logger.debug("memory_search lexical lane failed: %s", exc)
        try:
            from iai_mcp.embed import embed_query, embedder_for_store

            vec = embed_query(embedder_for_store(store), query[:512])
            for sem_rank, (rec, cos) in enumerate(store.query_similar(list(vec), k=k)):
                rid = str(rec.id)
                if rid in merged:
                    merged[rid]["lane"] = "both"
                    merged[rid]["cos"] = round(float(cos), 3)
                    merged[rid]["_rrf"] += 1.0 / (60.0 + sem_rank)
                else:
                    merged[rid] = {
                        "record_id": rid,
                        "surface": (rec.literal_surface or "")[:500],
                        "tier": rec.tier,
                        "lane": "semantic",
                        "cos": round(float(cos), 3),
                        "_rrf": 1.0 / (60.0 + sem_rank),
                    }
        except Exception as exc:  # noqa: BLE001
            logger.debug("memory_search semantic lane failed: %s", exc)

        hits = sorted(
            merged.values(),
            key=lambda h: (
                -(h.get("_rrf") or 0.0),
                -(h.get("score") or 0.0),
                -(h.get("cos") or 0.0),
                h["record_id"],
            ),
        )[:k]
        for h in hits:
            h.pop("_rrf", None)
        return {
            "hits": hits,
            "frame": (
                "hints, not ground truth — verify against sources; "
                "truncated surfaces recoverable via memory_recall"
            ),
        }

    if method == "memory_recall_structural":
        from iai_mcp import tem
        from iai_mcp.hebbian_structure import structural_similarity
        from iai_mcp.types import STRUCTURE_HV_BYTES

        structure_query: dict = params.get("structure_query") or {}
        budget_tokens = int(params.get("budget_tokens", 2000))
        max_records = int(params.get("max_records", 5000))
        if max_records < 1:
            max_records = 5000
        if max_records > 50_000:
            max_records = 50_000

        if structure_query:
            query_pairs = [
                (str(role), tem.filler_hv(str(value)))
                for role, value in structure_query.items()
            ]
            query_hv = tem.pack_pairs(query_pairs)
        else:
            query_hv = bytes(STRUCTURE_HV_BYTES)

        # The cap is pushed into the ITERATION, never applied after a
        # full-table materialization — all_records() would hold the whole
        # corpus in memory just to slice it.
        import itertools as _it

        records = list(_it.islice(
            store.iter_records(where="tombstoned_at IS NULL"), max_records,
        ))
        scored: list[tuple[float, "object"]] = []
        for rec in records:
            if not rec.structure_hv:
                continue
            sim = structural_similarity(query_hv, rec.structure_hv)
            scored.append((sim, rec))
        scored.sort(key=lambda x: x[0], reverse=True)

        hits_out: list[dict] = []
        budget_used = 0
        for sim, rec in scored:
            tokens = max(1, len(rec.literal_surface) // 4)
            if budget_used + tokens > budget_tokens and hits_out:
                break
            hits_out.append({
                "record_id": str(rec.id),
                "score": float(sim),
                "reason": f"structural similarity {sim:.3f} (D=10000 BSC Hamming)",
                "literal_surface": rec.literal_surface,
                "adjacent_suggestions": [],
            })
            budget_used += tokens

        return {
            "hits": hits_out,
            "anti_hits": [],
            "activation_trace": [],
            "budget_used": budget_used,
            "structural_query_size": len(structure_query),
        }

    if method == "memory_reinforce":
        ids = [UUID(x) for x in params["ids"]]
        upd = retrieve.reinforce_edges(store, ids)
        return {
            "edges_boosted": upd.edges_boosted,
            "new_weights": upd.new_weights,
        }

    if method == "memory_contradict":
        cue_embedding = params.get("cue_embedding") or [0.0] * EMBED_DIM
        rec = retrieve.contradict(
            store, UUID(params["id"]), params["new_fact"], cue_embedding
        )
        return {
            "original_id": str(rec.original_id),
            "new_record_id": str(rec.new_record_id),
            "edge_type": rec.edge_type,
            "ts": rec.ts.isoformat(),
        }

    if method == "memory_capture":
        from iai_mcp.capture import capture_turn
        if _last_injection_embedding:
            try:
                import numpy as _np
                from iai_mcp.embed import embedder_for_store
                from iai_mcp.events import write_event
                _emb = embedder_for_store(store)
                _cap_vec = _emb.embed(params["text"])
                _inj_vec = _np.asarray(_last_injection_embedding, dtype=_np.float32)
                _cap_arr = _np.asarray(_cap_vec, dtype=_np.float32)
                _n1 = float(_np.linalg.norm(_inj_vec))
                _n2 = float(_np.linalg.norm(_cap_arr))
                _coupling = float(_np.dot(_inj_vec, _cap_arr) / (_n1 * _n2)) if _n1 > 0 and _n2 > 0 else 0.0
                write_event(
                    store,
                    kind="trajectory_coupling",
                    data={
                        "coupling_score": round(_coupling, 4),
                        "injected_ids": _last_injection_ids[:5],
                        "direction": "toward" if _coupling > 0.3 else "neutral",
                    },
                    severity="info",
                    session_id=params.get("session_id", "-"),
                )
            except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
                logger.debug("trajectory_coupling_measure_failed: %s", exc)
            _last_injection_embedding = None
            _last_injection_ids = []
        result = capture_turn(
            store,
            cue=params.get("cue", ""),
            text=params["text"],
            tier=params.get("tier", "episodic"),
            session_id=params.get("session_id", "-"),
            role=params.get("role", "user"),
            live_turn=True,
        )
        try:
            from iai_mcp.store import flush_record_buffer
            flush_record_buffer(store)
        except Exception:  # noqa: BLE001
            pass
        return result

    if method == "memory_consolidate":
        from iai_mcp.guard import BudgetLedger, RateLimitLedger
        from iai_mcp.sleep import SleepConfig, run_heavy_consolidation

        cfg = SleepConfig()
        budget = BudgetLedger(store)
        rate = RateLimitLedger(store)
        result = run_heavy_consolidation(
            store,
            session_id=params.get("session_id", "-"),
            config=cfg,
            budget=budget,
            rate=rate,
            has_api_key=False,
        )
        return {
            "mode": result["mode"],
            "tier": result["tier"],
            "summaries_created": int(result["summaries_created"]),
            "decay_result": dict(result["decay_result"]),
            "schema_candidates": list(result["schema_candidates"]),
        }

    if method == "session_exit":
        from iai_mcp.sleep import run_light_consolidation
        from iai_mcp.trajectory import (
            compute_session_metrics_snapshot,
            record_session_metrics,
        )

        sid = params.get("session_id", "-")
        result = run_light_consolidation(store, session_id=sid)
        snapshot = compute_session_metrics_snapshot(store, sid)
        record_session_metrics(store, session_id=sid, metrics=snapshot)
        result["trajectory_metrics_emitted"] = len(snapshot)
        return result

    if method == "s5_propose":
        from iai_mcp.s5 import propose_invariant_update

        verdict, pid = propose_invariant_update(
            store,
            UUID(params["anchor_id"]),
            params["new_fact"],
            params.get("session_id", "-"),
        )
        return {
            "verdict": verdict,
            "proposal_id": str(pid) if pid is not None else None,
        }

    if method == "profile_update_from_signal":
        from iai_mcp.profile import bayesian_update

        global _posterior_state
        knob = params["knob"]
        signal = params["signal"]
        observed = params["observed"]
        with _profile_lock:
            new_val, new_post = bayesian_update(
                knob, signal, observed, _profile_state, _posterior_state,
            )
            _posterior_state = new_post
        return {"new_value": new_val, "knob": knob, "signal": signal}

    if method == "schema_induce":
        from iai_mcp.guard import BudgetLedger, RateLimitLedger
        from iai_mcp.schema import induce_schemas_tier1

        budget = BudgetLedger(store)
        rate = RateLimitLedger(store)
        candidates = induce_schemas_tier1(
            store, budget=budget, rate=rate, llm_enabled=False,
        )
        return {
            "candidates": [
                {
                    "pattern": c.pattern,
                    "confidence": c.confidence,
                    "evidence_count": c.evidence_count,
                    "status": c.status,
                }
                for c in candidates
            ],
            "count": len(candidates),
        }

    if method == "curiosity_pending":
        from iai_mcp.curiosity import pending_questions

        qs = pending_questions(store, params.get("session_id"))
        return {
            "questions": [
                {
                    "id": str(q.id),
                    "text": q.text,
                    "tier": q.tier,
                    "entropy": q.entropy,
                    "triggered_by_record_ids": [str(t) for t in q.triggered_by_record_ids],
                }
                for q in qs
            ],
            "count": len(qs),
        }

    if method == "trajectory_record":
        from iai_mcp.trajectory import record_session_metrics

        metrics = params.get("metrics", {})
        record_session_metrics(
            store, session_id=params.get("session_id", "-"), metrics=metrics,
        )
        return {"recorded": len(metrics), "session_id": params.get("session_id", "-")}

    if method == "schema_list":
        return _schema_list_dispatch(store, params)

    if method == "events_query":
        return _events_query_dispatch(store, params)

    if method == "memory_temporal_recall":
        from iai_mcp.events import flush_event_buffer, query_events
        from iai_mcp.embed import embed_query, embedder_for_store
        from iai_mcp.store._store import _normalize_ts_for_compare

        try:
            flush_event_buffer(store)
        except Exception as exc:  # noqa: BLE001 -- best-effort flush; never abort the read
            logger.debug("temporal_recall_flush_skipped err=%s", str(exc)[:80])

        cue = params.get("cue") or ""
        as_of_raw = params.get("as_of")
        changed_since_raw = params.get("changed_since")
        limit = int(params.get("limit", 10) or 10)

        as_of_norm: str | None = None
        if as_of_raw is not None and as_of_raw != "":
            try:
                as_of_norm = _normalize_ts_for_compare(as_of_raw)
            except ValueError as exc:
                return {"error": f"as_of must be ISO-8601, got {as_of_raw!r}: {exc}"}

        changed_since_norm: str | None = None
        if changed_since_raw is not None and changed_since_raw != "":
            try:
                changed_since_norm = _normalize_ts_for_compare(changed_since_raw)
            except ValueError as exc:
                return {
                    "error": (
                        f"changed_since must be ISO-8601, got {changed_since_raw!r}: {exc}"
                    )
                }

        record_hits: list[tuple[Any, float]] = []
        if as_of_norm is not None or cue:
            cue_vec: list[float] | None = None
            if cue:
                embedder = embedder_for_store(store)
                cue_vec = embed_query(embedder, cue)
            record_hits = store.query_similar_temporal(
                vec=cue_vec, as_of=as_of_norm, k=limit,
            )

        hits_out: list[dict] = []
        for record, score in record_hits:
            hits_out.append({
                "id": str(record.id) if record.id is not None else None,
                "literal_surface": record.literal_surface or "",
                "tier": record.tier,
                "score": float(score),
                "created_at": (
                    record.created_at.isoformat() if record.created_at else None
                ),
                "updated_at": (
                    record.updated_at.isoformat() if record.updated_at else None
                ),
                "tags": list(record.tags or []),
                "language": record.language or "en",
            })

        events_out: list[dict] = []
        if changed_since_norm is not None:
            ledger_events = query_events(
                store,
                kind=None,
                since=changed_since_norm,
                since_exclusive=True,
                limit=limit,
            )
            for ev in ledger_events:
                ts = ev.get("ts")
                ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                events_out.append({
                    "id": str(ev.get("id")),
                    "kind": ev.get("kind"),
                    "severity": ev.get("severity"),
                    "domain": ev.get("domain"),
                    "ts": ts_str,
                    "data": ev.get("data", {}),
                    "session_id": ev.get("session_id"),
                    "source_ids": ev.get("source_ids", []),
                })

        result: dict[str, Any] = {
            "hits": hits_out,
            "changed_since_events": events_out,
        }
        # Unscoped queries omit the field entirely: the MCP output schema
        # types _scope as string, and an explicit null fails validation.
        if as_of_norm is not None or changed_since_norm is not None:
            result["_scope"] = "committed_by_time_t"
        return result

    if method == "audit_query":
        from iai_mcp.s5 import AUDIT_EVENT_KINDS, audit_identity_events

        since_raw = params.get("since")
        since_dt = None
        if since_raw:
            try:
                since_dt = datetime.fromisoformat(
                    str(since_raw).replace("Z", "+00:00"),
                )
                if since_dt.tzinfo is None:
                    since_dt = since_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                return {"error": f"since must be ISO-8601, got {since_raw!r}"}

        kinds_param = params.get("kinds")
        kinds = (
            tuple(kinds_param) if isinstance(kinds_param, (list, tuple))
            else AUDIT_EVENT_KINDS
        )
        events = audit_identity_events(store, since=since_dt, kinds=kinds)
        out_events: list[dict] = []
        for e in events:
            ts = e.get("ts")
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            out_events.append({
                "id": str(e.get("id")),
                "kind": e.get("kind"),
                "severity": e.get("severity"),
                "ts": ts_str,
                "data": e.get("data", {}),
                "session_id": e.get("session_id"),
            })
        return {"events": out_events, "count": len(out_events)}

    if method == "detect_drift":
        from iai_mcp.s5 import detect_drift_anomaly

        window = int(params.get("window_sessions", 5) or 5)
        alerts = detect_drift_anomaly(store, window_sessions=window)
        return {"alerts": alerts, "count": len(alerts)}

    if method == "shield_check":
        from iai_mcp.shield import ShieldTier, evaluate_injection_risk

        text = params.get("text", "") or ""
        tier_name = str(params.get("tier", "hard_block")).lower()
        try:
            tier = ShieldTier(tier_name)
        except ValueError:
            return {"error": f"unknown shield tier {tier_name!r}"}
        verdict = evaluate_injection_risk(
            text, tier, target_language=params.get("language"),
        )
        return {
            "tier": verdict.tier.value,
            "detected": verdict.detected,
            "matched_patterns": list(verdict.matched_patterns),
            "severity": verdict.severity,
            "action": verdict.action,
            "reason": verdict.reason,
            "confidence": verdict.confidence,
            "language": verdict.language,
        }

    if method == "status_light":
        # Liveness + headline numbers with ZERO graph work: the awake path
        # must answer instantly, so this surface reads only the O(1) corpus
        # count and whatever topology snapshot the cache already holds.
        records_count = store.active_records_count()
        with _topology_state_lock:
            cached_snapshot = _topology_cache
        return {
            "N": records_count,
            "regime": (cached_snapshot or {}).get("regime", "unknown"),
            "sigma": (cached_snapshot or {}).get("sigma"),
            "topology_cached": cached_snapshot is not None,
        }

    if method == "topology":
        from iai_mcp import sigma as sigma_mod
        from iai_mcp.events import write_event

        # Emptiness gate via the corpus-count cache (O(1) warm): a raw
        # count_rows on the lilli engine re-scans every leaf page, and this
        # gate sits on a per-call read path.
        records_count = store.active_records_count()
        if records_count == 0:
            return {
                "N": 0, "C": 0.0, "L": 0.0, "sigma": None,
                "community_count": 0, "rich_club_ratio": 0.0,
                "regime": "insufficient_data",
            }
        now = _time.monotonic()
        with _topology_state_lock:
            cached_snapshot = _topology_cache
            cache_fresh = (
                cached_snapshot is not None
                and (now - _topology_cache_at) < _TOPOLOGY_SNAPSHOT_TTL_S
            )
        if cache_fresh:
            return dict(cached_snapshot)
        if not _topology_inflight.acquire(blocking=False):
            # A deep compute is already running; never stack another one.
            # Serve the stale snapshot when there is one; otherwise answer
            # with an honest shape-stable placeholder.
            if cached_snapshot is not None:
                return dict(cached_snapshot)
            return {
                "N": records_count, "C": 0.0, "L": 0.0, "sigma": None,
                "community_count": 0, "rich_club_ratio": 0.0,
                "regime": "computing",
            }
        try:
            graph_bundle = retrieve.build_runtime_graph(store)
            if isinstance(graph_bundle, tuple):
                graph = graph_bundle[0]
                # The bundle already carries the community assignment; passing
                # it through skips a full in-process re-detection per snapshot.
                # A degraded bundle assignment (no centroids) would report a
                # zero community count — fall back to detection instead.
                bundle_assignment = (
                    graph_bundle[1] if len(graph_bundle) > 1 else None
                )
                if not getattr(bundle_assignment, "community_centroids", None):
                    bundle_assignment = None
            else:
                graph, bundle_assignment = graph_bundle, None
            snapshot = sigma_mod.compute_topology_snapshot(
                graph, assignment=bundle_assignment
            )
            with _topology_state_lock:
                _topology_cache = dict(snapshot)
                _topology_cache_at = _time.monotonic()
            return snapshot
        except Exception as exc:
            write_event(
                store,
                "topology_native_failed",
                {"error_type": type(exc).__name__, "error": str(exc)},
            )
            raise
        finally:
            _topology_inflight.release()

    if method == "camouflaging_status":
        from iai_mcp import camouflaging

        window = int(params.get("window_size", 5) or 5)
        result = camouflaging.detect_camouflaging(store, window_size=window)
        result["camouflaging_relaxation"] = float(
            _profile_state.get("camouflaging_relaxation", 0.0),
        )
        return result

    if method == "initiate_sleep_mode":
        return asyncio.run(handle_initiate_sleep_mode(params))

    if method == "force_wake":
        return asyncio.run(handle_force_wake(params))

    if method == "profile_get":
        return profile.profile_get(params.get("knob"), _profile_state)

    if method == "profile_set":
        with _profile_lock:
            return profile.profile_set(
                params["knob"], params["value"], _profile_state, store=store,
            )

    if method == "session_start_payload":
        from iai_mcp.session import assemble_session_start, SessionStartPayload
        sid = params.get("session_id", "-")
        # Emptiness gate via the corpus-count cache (O(1) warm): a raw
        # count_rows on the lilli engine re-scans every leaf page, and this
        # gate sits on a per-call read path.
        records_count = store.active_records_count()
        if records_count == 0:
            empty = SessionStartPayload(
                l0="",
                l1="",
                l2=[],
                rich_club="",
                total_cached_tokens=0,
                total_dynamic_tokens=1000,
            )
            return _payload_to_json(empty)
        _graph, assignment, rc = retrieve.build_runtime_graph(store)
        payload = assemble_session_start(
            store, assignment, rc,
            session_id=sid,
            profile_state=_profile_state,
        )

        try:
            from iai_mcp.user_model import (
                UserModelPrefetcher,
                load as _user_model_load,
            )
            from iai_mcp.daemon_config import _load_user_model_config
            _user_model_cfg = _load_user_model_config()
            _user_model = _user_model_load()
            _prefetched_ids = UserModelPrefetcher().prefetch(
                store, _user_model, top_k=_user_model_cfg.prefetch_top_k,
            )
            if _prefetched_ids:
                _existing = set(payload.l2)
                _new = [
                    rid for rid in _prefetched_ids if rid not in _existing
                ]
                payload.l2 = _new + list(payload.l2)
                _cap = len(_existing) + _user_model_cfg.prefetch_top_k
                if len(payload.l2) > _cap:
                    payload.l2 = payload.l2[:_cap]
        except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
            import logging
            logging.getLogger(__name__).warning(
                "user_model_prefetch_failed",
                extra={
                    "err_type": type(exc).__name__,
                    "err": str(exc)[:120],
                },
            )

        return _payload_to_json(payload)

    if method == "session_refresh_if_stale":
        from iai_mcp.capture import drain_active_live_captures, drain_deferred_captures
        from iai_mcp.session import (
            SESSION_START_CACHE_MAX_CHARS,
            _compose_session_start_payload,
            format_payload_as_markdown,
            max_record_created_at,
        )

        caller_watermark = params.get("watermark") or ""
        refreshing_session_id = params.get("session_id", "-")

        try:
            drain_deferred_captures(store)
        except Exception as _drain_exc:  # noqa: BLE001
            logger.warning(
                "session_refresh_drain_failed",
                extra={"err": str(_drain_exc)[:120]},
            )
            return {"rendered": "", "new_max_ts": ""}

        try:
            drain_active_live_captures(store, exclude_session_id=refreshing_session_id)
        except Exception as _live_drain_exc:  # noqa: BLE001
            logger.warning(
                "session_refresh_live_drain_failed",
                extra={"err": str(_live_drain_exc)[:120]},
            )

        try:
            from iai_mcp.store import flush_record_buffer
            flush_record_buffer(store)
        except Exception as _flush_exc:  # noqa: BLE001
            logger.warning(
                "session_refresh_flush_failed",
                extra={"err": str(_flush_exc)[:120]},
            )

        new_max_ts = max_record_created_at(store)

        def _norm(ts: str) -> str:
            try:
                from datetime import datetime, timezone as _tz
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00").replace(" ", "T"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_tz.utc)
                return dt.astimezone(_tz.utc).isoformat()
            except (TypeError, ValueError):
                return ts

        _new_max_norm = _norm(new_max_ts) if new_max_ts else ""
        _wm_norm = _norm(caller_watermark) if caller_watermark else ""

        if not new_max_ts or (caller_watermark and _new_max_norm <= _wm_norm):
            return {"rendered": "", "new_max_ts": new_max_ts or ""}

        _graph, assignment, rc = retrieve.build_runtime_graph(store)
        payload = _compose_session_start_payload(
            store,
            assignment,
            rc,
            session_id=params.get("session_id", "-"),
            profile_state={"wake_depth": "standard"},
        )
        rendered = format_payload_as_markdown(payload)
        if len(rendered) > SESSION_START_CACHE_MAX_CHARS:
            rendered = rendered[:SESSION_START_CACHE_MAX_CHARS]
        return {"rendered": rendered, "new_max_ts": new_max_ts}

    if method == "episodes_recent":
        from iai_mcp.capture import read_pending_live_events
        n = max(0, min(int(params.get("n", 10)), 1000))
        session_id = params.get("session_id")
        pending = read_pending_live_events(session_id=session_id)
        records = store.recent_user_turns(n, session_id=session_id, pending_live_events=pending)
        turns = []
        for r in records:
            if r.id is None:
                su = getattr(r, "_pending_source_uuid", None)
                idem = getattr(r, "_pending_idem_tag", "")
                if su:
                    rid = f"pending:{su}"
                else:
                    idem_hex = idem[5:] if idem.startswith("idem:") else idem
                    rid = f"pending:{idem_hex}" if idem_hex else f"pending:unknown"
            else:
                rid = str(r.id)
            turns.append({
                "record_id": rid,
                "literal_surface": r.literal_surface,
                "session_id": (r.provenance or [{}])[0].get("session_id"),
                "captured_at": (
                    r.created_at.isoformat() if r.created_at else None
                ),
            })
        return {"turns": turns, "count": len(turns)}

    if method == "drain_permanent_failed":
        from iai_mcp.capture import drain_permanent_failed_files
        from pathlib import Path as _Path

        dry_run = bool(params.get("dry_run", False))
        try:
            deferred_dir = _Path(store.root) / ".deferred-captures"
        except Exception:  # noqa: BLE001 -- deferred_dir=None triggers default resolution
            deferred_dir = None
        result = drain_permanent_failed_files(store, deferred_dir=deferred_dir, dry_run=dry_run)
        return result

    if method == "rss_stats":
        return _rss_stats_snapshot()

    raise UnknownMethodError(method)


def _rss_stats_snapshot() -> dict:
    """Return the daemon's process-local resident-set + allocator snapshot."""
    try:
        from iai_mcp.lilli.cycle.sleep_pipeline._rss_probe import (
            capture_step_snapshot,
        )

        return capture_step_snapshot(include_vmmap=True)
    except Exception:  # noqa: BLE001 -- the read method must always return a dict
        return {
            "rss_kib": None,
            "vmmap_region_count": None,
            "vm_allocate_kib": None,
            "numba_nrt_alloc_count": -1,
            "numba_nrt_free_count": -1,
        }


async def _send_to_daemon(
    message: dict,
    *,
    timeout: float = 30.0,
    socket_path=None,
) -> dict:
    path_used = socket_path if socket_path is not None else SOCKET_PATH
    try:
        from iai_mcp._ipc import open_ipc_connection
        reader, writer = await open_ipc_connection(str(path_used))
    except (FileNotFoundError, ConnectionRefusedError) as exc:
        return {"ok": False, "reason": "daemon_not_running", "error": str(exc)}

    try:
        writer.write((json.dumps(message) + "\n").encode("utf-8"))
        await writer.drain()
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        except asyncio.TimeoutError:
            return {"ok": False, "reason": "timeout"}
        if not line:
            return {"ok": False, "reason": "empty_response"}
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            return {"ok": False, "reason": "invalid_json", "error": str(exc)}
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
            logger.debug("socket_writer_close_failed: %s", exc)


async def handle_initiate_sleep_mode(params: dict) -> dict:
    if not isinstance(params, dict):
        raise ValueError("initiate_sleep_mode params must be an object")
    if "consent" not in params:
        raise ValueError("initiate_sleep_mode requires 'consent' (bool)")
    if "reason" not in params:
        raise ValueError("initiate_sleep_mode requires 'reason' (str)")
    if not isinstance(params["consent"], bool):
        raise ValueError("'consent' must be bool")
    if not isinstance(params["reason"], str):
        raise ValueError("'reason' must be str")

    if params["consent"] is not True:
        return {"ok": False, "reason": "consent_declined"}

    reason = params["reason"][:500]
    return await _send_to_daemon({
        "type": "user_initiated_sleep",
        "reason": reason,
        "ts": datetime.now(timezone.utc).isoformat(),
    })


async def handle_force_wake(params: dict) -> dict:
    return await _send_to_daemon(
        {
            "type": "force_wake",
            "ts": datetime.now(timezone.utc).isoformat(),
        },
        timeout=float(FORCE_WAKE_TIMEOUT_SEC),
    )


def _inject_sleep_suggestion(
    response: dict,
    *,
    cue: str,
    language: str,
) -> None:
    try:
        from iai_mcp.bedtime import detect_wind_down
        from iai_mcp.daemon_state import load_state
        from iai_mcp.tz import load_user_tz

        state = load_state()
        now = datetime.now(timezone.utc)
        tz = load_user_tz()
        suggestion = detect_wind_down(cue, language, state, now, tz)
        if suggestion:
            response["sleep_suggestion"] = suggestion
    except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
        logger.debug("sleep_suggestion_failed: %s", exc)


_EMPTY_OVERNIGHT_DIGEST: dict = {
    "rem_cycles_completed": 0,
    "episodes_processed": 0,
    "schemas_induced_tier0": 0,
    "claude_call_used": False,
    "quota_used_pct": 0.0,
    "main_insight_text": None,
    "sigma_observed": None,
    "s5_drift_alerts": [],
    "daemon_uptime_hours": 0,
    "timed_out_cycles": 0,
}


def _inject_overnight_digest(response: dict, store: MemoryStore | None = None) -> None:
    try:
        from iai_mcp.daemon_state import load_state as _load_state
        from iai_mcp.daemon_state import get_pending_digest as _get_pending_digest
        state = _load_state()
        now = datetime.now(timezone.utc)
        digest = _get_pending_digest(state, now)
        if not digest:
            response["overnight_digest"] = dict(_EMPTY_OVERNIGHT_DIGEST)
            return
        response["overnight_digest"] = {
            "rem_cycles_completed": digest.get("rem_cycles_completed", 0),
            "episodes_processed": digest.get("episodes_processed", 0),
            "schemas_induced_tier0": digest.get("schemas_induced_tier0", 0),
            "claude_call_used": digest.get("claude_call_used", False),
            "quota_used_pct": digest.get("quota_used_pct", 0.0),
            "main_insight_text": digest.get("main_insight_text"),
            "sigma_observed": digest.get("sigma_observed"),
            "s5_drift_alerts": digest.get("s5_drift_alerts", []),
            "daemon_uptime_hours": digest.get("daemon_uptime_hours", 0),
            "timed_out_cycles": digest.get("timed_out_cycles", 0),
        }
    except Exception as exc:  # noqa: BLE001 -- hot path must never break
        response["overnight_digest"] = dict(_EMPTY_OVERNIGHT_DIGEST)
        if store is not None:
            try:
                from iai_mcp.events import write_event
                write_event(
                    store,
                    "digest_inject_error",
                    {"error": str(exc)[:500]},
                    severity="warning",
                )
            except Exception as exc2:  # noqa: BLE001 -- MCP boundary fail-safe
                logger.debug("digest_inject_error_event_failed: %s", exc2)


def _first_turn_recall_hook(
    response: dict,
    *,
    params: dict,
    store: MemoryStore,
) -> None:
    try:
        from iai_mcp.daemon_state import consume_first_turn, load_state
        state = load_state()
        session_id = params.get("session_id", "unknown")
        if not consume_first_turn(state, session_id):
            return
        raw_cue = params.get("cue", "")
        cue = str(raw_cue)[:2000] if raw_cue is not None else ""
        if not cue:
            return
        warm_hit_ids: list = []
        try:
            from iai_mcp.hippea_cascade import snapshot_warm_ids
            warm_hit_ids = snapshot_warm_ids()
        except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
            logger.debug("snapshot_warm_ids_failed: %s", exc)
            warm_hit_ids = []

        warm_lru_source = "daemon" if warm_hit_ids else "none"
        from iai_mcp.runtime_graph_cache import preload_ready as _preload_ready
        if (
            not warm_hit_ids
            and str(session_id) not in _CORE_CASCADE_FIRED_PER_SESSION
            and _preload_ready.is_set()
        ):
            try:
                from iai_mcp.hippea_cascade import compute_core_side_warm_snapshot
                from iai_mcp import retrieve as _retrieve
                _graph, assignment, _rc = _retrieve.build_runtime_graph(store)
                warm_ids = compute_core_side_warm_snapshot(
                    store, assignment, top_k=3, max_records=50,
                )
                try:
                    _warm_batch = store.get_batch(list(warm_ids))
                except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
                    logger.debug("warm_lru_store_get_batch_failed: %s", exc)
                    _warm_batch = {}
                for rid in warm_ids:
                    rec = _warm_batch.get(rid)
                    if rec is not None:
                        _CORE_WARM_LRU[rid] = rec
                _CORE_CASCADE_FIRED_PER_SESSION.add(str(session_id))
                if _CORE_WARM_LRU:
                    warm_hit_ids = list(_CORE_WARM_LRU.keys())
                    warm_lru_source = "core_fallback"
            except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
                logger.debug("core_cascade_failed: %s", exc)

        cue_embedding = params.get("cue_embedding") or [0.0] * EMBED_DIM
        result = retrieve.recall(
            store=store,
            cue_embedding=cue_embedding,
            cue_text=cue,
            session_id=str(session_id),
            budget_tokens=400,
            k_hits=5,
            k_anti=2,
            mode="concept",
        )
        response["first_turn_recall"] = {
            "hits": [_hit_to_json(h) for h in result.hits],
            "budget_tokens": 400,
            "budget_used": result.budget_used,
            "warm_lru_size": len(warm_hit_ids),
            "warm_lru_source": warm_lru_source,
        }
        try:
            from iai_mcp.events import write_event
            write_event(
                store,
                "first_turn_recall",
                {"session_id": str(session_id), "cue_len": len(cue)},
                severity="info",
            )
        except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
            logger.debug("first_turn_recall_event_failed: %s", exc)
    except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
        logger.debug("first_turn_recall_hook_failed: %s", exc)


def main() -> None:
    _require_native()

    store = MemoryStore()
    _seed_l0_identity(store)

    try:
        from iai_mcp.tz import load_user_tz
        tz = load_user_tz()
        sys.stderr.write(f"iai-mcp: timezone={tz.key}\n")
        sys.stderr.flush()
    except Exception as e:  # noqa: BLE001 pragma: no cover -- boot diagnostics must not break
        sys.stderr.write(f"iai-mcp: timezone detection failed: {e}\n")
        sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id: Any = None
        try:
            req = json.loads(line)
            req_id = req.get("id") if isinstance(req, dict) else None
            method = req.get("method")
            params = req.get("params") or {}
            if not method:
                raise ValueError("missing method")
            result = dispatch(store, method, params)
            sys.stdout.write(
                json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n"
            )
        except Exception as e:  # noqa: BLE001 -- MCP boundary fail-safe
            err = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32000,
                    "message": str(e),
                    "trace": traceback.format_exc() if sys.flags.dev_mode else None,
                },
            }
            sys.stdout.write(json.dumps(err) + "\n")
        sys.stdout.flush()


from iai_mcp.core._serializers import _hit_to_json, _payload_to_json  # noqa: E402
from iai_mcp.core._query_dispatch import (  # noqa: E402
    _schema_list_dispatch,
    _events_query_dispatch,
    EVENTS_QUERY_WHITELIST,
)
from iai_mcp.core._identity import (  # noqa: E402
    _load_l0_identity_seed,
    _seed_l0_identity,
    L0_ID,
    _DEFAULT_L0_SEED,
)

__all__ = [
    "dispatch",
    "main",
    "UnknownMethodError",
    "_profile_state",
    "LIVE_KNOBS",
    "DEFERRED_KNOBS",
    "FORCE_WAKE_TIMEOUT_SEC",
    "SOCKET_PATH",
    "get_pending_digest",
    "load_state",
    "L0_ID",
    "_seed_l0_identity",
    "_load_l0_identity_seed",
    "EVENTS_QUERY_WHITELIST",
    "_inject_overnight_digest",
    "_inject_sleep_suggestion",
    "_first_turn_recall_hook",
    "_send_to_daemon",
    "handle_initiate_sleep_mode",
    "handle_force_wake",
    "_hit_to_json",
    "_payload_to_json",
    "_schema_list_dispatch",
    "_events_query_dispatch",
    "_DEFAULT_L0_SEED",
]


if __name__ == "__main__":
    main()
