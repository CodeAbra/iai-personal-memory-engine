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


FORCE_WAKE_TIMEOUT_SEC: int = 15 * 60


from cachetools import TTLCache as _CoreTTLCache

_CORE_WARM_LRU: _CoreTTLCache = _CoreTTLCache(maxsize=50, ttl=300)
_CORE_CASCADE_FIRED_PER_SESSION: set[str] = set()


_profile_state: dict[str, Any] = profile.default_state()

_posterior_state: dict[str, Any] = {}

_arousal_state: object | None = None

_last_injection_embedding: list[float] | None = None
_last_injection_ids: list[str] = []

_profile_lock: threading.RLock = threading.RLock()

LIVE_KNOBS: dict[str, Any] = _profile_state
DEFERRED_KNOBS: frozenset[str] = frozenset(
    profile.PHASE_2_DEFERRED | profile.PHASE_3_DEFERRED
)
assert len(DEFERRED_KNOBS) == 0, "all 10 autistic-kernel knobs live"


def dispatch(store: MemoryStore, method: str, params: dict) -> dict:
    global _last_injection_embedding, _last_injection_ids, _arousal_state
    if method == "memory_recall":
        _recall_t0 = _time.perf_counter()
        # crisis_mode honest-degrade: when consolidation is stuck (the
        # scheduler is looping a deferred step and cannot advance), the warm
        # recall path serves stale schema-dominated results. Honour the
        # always-available invariant by returning an explicitly degraded
        # response so the wrapper falls back to bank-recall via its existing
        # socket-unreachable code path.
        try:
            from iai_mcp.lifecycle_state import load_state as _ls_load_cm
            _crisis_state = _ls_load_cm()
            if bool(_crisis_state.get("crisis_mode", False)):
                logger.warning(
                    "memory_recall served degraded under crisis_mode; "
                    "client should fall back to bank-recall"
                )
                return {
                    "hits": [],
                    "_degraded": True,
                    "_reason": "daemon_consolidation_stuck",
                }
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
        records_count = store.db.open_table("records").count_rows()
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
            from iai_mcp.embed import embedder_for_store
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

                    assignment, rc, _cached_max_degree, _structural_source = _rgc.load_recall_structural(store)

                    _encode_ms: "float | None" = None
                    _encode_t0 = _time.perf_counter()
                    try:
                        _cue_vec = embedder.embed(params["cue"])
                        _encode_ms = (_time.perf_counter() - _encode_t0) * 1000.0
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

                    _hop1_edges = store.incident_edges(list(_candidate_recs.keys()), top_k=5)
                    _hop1_new_ids = list({
                        _nbr
                        for _nbr_list in _hop1_edges.values()
                        for (_nbr, _et, _wt) in _nbr_list
                        if _nbr not in _candidate_recs
                    })
                    if _hop1_new_ids:
                        _candidate_recs.update(store.get_batch(_hop1_new_ids))

                    _hop2_edges = store.incident_edges(_hop1_new_ids, top_k=5) if _hop1_new_ids else {}
                    _hop2_new_ids = list({
                        _nbr
                        for _nbr_list in _hop2_edges.values()
                        for (_nbr, _et, _wt) in _nbr_list
                        if _nbr not in _candidate_recs
                    })
                    if _hop2_new_ids:
                        _candidate_recs.update(store.get_batch(_hop2_new_ids))

                    _RC_CAP = 50
                    _rc_cap = (rc or [])[:_RC_CAP]
                    _rc_new_ids = [_rid for _rid in _rc_cap if _rid not in _candidate_recs]
                    if _rc_new_ids:
                        _candidate_recs.update(store.get_batch(_rc_new_ids))

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

                    try:
                        _all_cand_ids = list(_candidate_recs.keys())
                        _global_edges_hebb = store.incident_edges(
                            _all_cand_ids,
                            edge_types=["hebbian"],
                            top_k=None,
                        )
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
                        _all_candidate_ids = list(_candidate_recs.keys())
                        for _rec in _candidate_recs.values():
                            _ca = getattr(_rec, "created_at", None)
                            if _ca is not None:
                                _tv_ts_l1[str(_rec.id)] = _ca
                        _contr_edges = store.incident_edges(
                            _all_candidate_ids,
                            edge_types=["contradicts"],
                            top_k=None,
                        )
                        _contr_dst_ids = []
                        for _src_id, _edges in _contr_edges.items():
                            for (_dst, _et, _wt) in _edges:
                                _src_s = str(_src_id)
                                _dst_s = str(_dst)
                                _tv_outgoing_l1.setdefault(_src_s, []).append(_dst_s)
                                if _dst not in _candidate_recs:
                                    _contr_dst_ids.append(_dst)
                        if _contr_dst_ids:
                            _contr_recs = store.get_batch(_contr_dst_ids)
                            for _cr in _contr_recs.values():
                                _ca = getattr(_cr, "created_at", None)
                                if _ca is not None:
                                    _tv_ts_l1[str(_cr.id)] = _ca
                    except Exception as _tv_exc:  # noqa: BLE001 — degrade gracefully
                        logger.debug("layer1_tv_build_failed: %s", _tv_exc)
                        _tv_outgoing_l1, _tv_ts_l1 = {}, {}

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
                    )
                    resp.ann_path_used = True
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
            "_knobs_applied": knobs_applied,
            "ann_path_used": getattr(resp, "ann_path_used", False),
        }
        if _cortex_fallback:
            response["_source"] = "cortex-fallback"
        if not _cortex_fallback and _structural_source == "cold_degrade":
            response["_source"] = "cold-structural-degrade"
        try:
            _recall_ms = (_time.perf_counter() - _recall_t0) * 1000
            response["_recall_latency_ms"] = round(_recall_ms, 1)
        except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
            logger.debug("recall_latency_measure_failed: %s", exc)
        try:
            from iai_mcp.curiosity import get_pending_questions
            _qs = get_pending_questions(store, limit=2)
            if _qs:
                response["curiosity_signals"] = _qs
        except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
            logger.debug("curiosity_signals_failed: %s", exc)
        try:
            for hit in resp.hits:
                store.reinforce_record(hit.record_id, is_retrieval=True)
        except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
            logger.debug("labile_write_failed: %s", exc)
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
                    for h in resp.hits[:5]:
                        rec = store.get(h.record_id)
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

        records = store.all_records()
        if len(records) > max_records:
            records = records[:max_records]
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

    if method == "topology":
        from iai_mcp import sigma as sigma_mod
        from iai_mcp.events import write_event

        records_count = store.db.open_table("records").count_rows()
        if records_count == 0:
            return {
                "N": 0, "C": 0.0, "L": 0.0, "sigma": None,
                "community_count": 0, "rich_club_ratio": 0.0,
                "regime": "insufficient_data",
            }
        try:
            graph_bundle = retrieve.build_runtime_graph(store)
            graph = graph_bundle[0] if isinstance(graph_bundle, tuple) else graph_bundle
            return sigma_mod.compute_topology_snapshot(graph)
        except Exception as exc:
            write_event(
                store,
                "topology_native_failed",
                {"error_type": type(exc).__name__, "error": str(exc)},
            )
            raise

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
            _knob = params["knob"]
            _value = params["value"]
            # Untyped MCP / JSON-RPC clients can lose the JSON type of a value
            # (a bool `true` arrives as the string "true"), which the strict
            # profile_set validator would reject. Undo only that exact
            # stringification here, at the client edge; profile_set itself
            # stays strict for every other case.
            _spec = profile.PROFILE_KNOBS.get(_knob)
            if _spec is not None:
                _value = profile.coerce_json_stringified(_spec.value_schema, _value)
            return profile.profile_set(
                _knob, _value, _profile_state, store=store,
            )

    if method == "session_start_payload":
        from iai_mcp.session import assemble_session_start, SessionStartPayload
        sid = params.get("session_id", "-")
        records_count = store.db.open_table("records").count_rows()
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

    raise UnknownMethodError(method)


async def _send_to_daemon(
    message: dict,
    *,
    timeout: float = 30.0,
    socket_path=None,
) -> dict:
    from iai_mcp._ipc import open_ipc_connection
    path_used = socket_path if socket_path is not None else SOCKET_PATH
    try:
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
        if not warm_hit_ids and str(session_id) not in _CORE_CASCADE_FIRED_PER_SESSION:
            try:
                from iai_mcp.hippea_cascade import compute_core_side_warm_snapshot
                from iai_mcp import retrieve as _retrieve
                _graph, assignment, _rc = _retrieve.build_runtime_graph(store)
                warm_ids = compute_core_side_warm_snapshot(
                    store, assignment, top_k=3, max_records=50,
                )
                for rid in warm_ids:
                    try:
                        rec = store.get(rid)
                        if rec is not None:
                            _CORE_WARM_LRU[rid] = rec
                    except Exception as exc:  # noqa: BLE001 -- MCP boundary fail-safe
                        logger.debug("warm_lru_store_get_failed rid=%s: %s", rid, exc)
                        continue
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
