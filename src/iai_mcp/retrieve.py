from __future__ import annotations

import logging
import math
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from itertools import combinations
from uuid import UUID, uuid4

from iai_mcp.aaak import enforce_english_raw, generate_aaak_index
from iai_mcp.events import query_events, write_event
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import (
    EMBED_DIM,
    EdgeUpdate,
    MemoryHit,
    MemoryRecord,
    RecallResponse,
    ReconsolidationReceipt,
)


log = logging.getLogger(__name__)

_GRAPH_DECRYPT_WARN_LAST: dict[str, float] = {}
_GRAPH_DECRYPT_WARN_INTERVAL_SEC = 300.0


TEMPORAL_NEXT_WINDOW = timedelta(minutes=5)


STALE_DOWNWEIGHT_FACTOR: float = 0.5

_STALE_REASON_SUFFIX: str = " · stale"


def _has_tombstone_marker(value) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip()
    if not text:
        return False
    return text.lower() not in {"none", "nan", "nat", "<na>"}


# The community graph is an enhancement layer over the index recall path: a
# record is findable by cosine/ANN as soon as it lands in the index, before it
# is ever folded into the community graph. So the community graph may lag the
# corpus by a bounded number of records without losing recall correctness — the
# lagging records are still returned by the index. This tolerance defines that
# bound. While the corpus stays within tolerance of the cached node set, the
# build reuses the cached graph and cached centrality and skips the heavy
# betweenness recompute; only an over-tolerance drift triggers a full rebuild
# (which then folds in every accumulated record at once). Without this bound a
# single new record changes the corpus count and forces a full betweenness pass
# on every write.
_DRIFT_DEFAULT_ABS: int = 500
_DRIFT_DEFAULT_FRAC: float = 0.05


def _drift_tolerance(cached_count: int) -> int:
    """Largest corpus/cache count delta the cached graph may absorb without a
    full rebuild.

    `max(abs_floor, ceil(frac * cached_count))` — an absolute floor for small
    corpora plus a proportional band for large ones. Both bounds are
    operator-overridable: `IAI_MCP_RGC_DRIFT_ABS` (int ≥ 0) and
    `IAI_MCP_RGC_DRIFT_FRAC` (float ≥ 0). A malformed or negative override falls
    back to the default rather than failing recall.
    """
    abs_floor = _DRIFT_DEFAULT_ABS
    raw_abs = os.environ.get("IAI_MCP_RGC_DRIFT_ABS")
    if raw_abs is not None:
        try:
            parsed_abs = int(raw_abs)
            if parsed_abs >= 0:
                abs_floor = parsed_abs
        except (TypeError, ValueError):
            pass

    frac = _DRIFT_DEFAULT_FRAC
    raw_frac = os.environ.get("IAI_MCP_RGC_DRIFT_FRAC")
    if raw_frac is not None:
        try:
            parsed_frac = float(raw_frac)
            if parsed_frac >= 0.0:
                frac = parsed_frac
        except (TypeError, ValueError):
            pass

    proportional = math.ceil(frac * max(0, int(cached_count)))
    return max(abs_floor, proportional)


def _within_drift_tolerance(cached_count: int, records_count: int) -> bool:
    """True when the live corpus count is close enough to the cached node set
    that the cached community graph + centrality remain serviceable.

    Single source of truth for the drift decision so the lock-free
    `_runtime_graph_rebuild_needed` probe and the in-build `use_cached_payload`
    gate cannot diverge.
    """
    return abs(int(records_count) - int(cached_count)) <= _drift_tolerance(
        int(cached_count)
    )


def recall(
    store: MemoryStore,
    cue_embedding: list[float],
    cue_text: str,
    session_id: str,
    budget_tokens: int = 1500,
    k_hits: int = 5,
    k_anti: int = 3,
    mode: str = "verbatim",
) -> RecallResponse:
    raw = store.query_similar(cue_embedding, k=k_hits + k_anti)

    if mode == "verbatim":
        raw = [
            (rec, score) for rec, score in raw
            if rec.tier == "episodic"
            and not any(t.startswith("pattern:") for t in (rec.tags or []))
        ]

    hits: list[MemoryHit] = []
    provenance_pending: list[tuple[UUID, dict]] = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for record, score in raw[:k_hits]:
        _prov = (record.provenance or [{}])[0]
        hits.append(
            MemoryHit(
                record_id=record.id,
                score=float(score),
                reason=f"cosine {score:.3f}",
                literal_surface=record.literal_surface,
                adjacent_suggestions=[],
                session_id=_prov.get("session_id"),
                captured_at=record.created_at.isoformat() if record.created_at else None,
            )
        )
        provenance_pending.append((
            record.id,
            {
                "ts": now_iso,
                "cue": cue_text,
                "session_id": session_id,
            },
        ))

    if provenance_pending:
        try:
            store.queue_provenance_batch(provenance_pending)
        except (OSError, ValueError, RuntimeError) as exc:
            log.warning("provenance_batch write failed: %s", exc)

    anti_hits: list[MemoryHit] = []
    tail = raw[-k_anti:] if len(raw) >= k_anti else []
    for record, score in reversed(tail):
        anti_hits.append(
            MemoryHit(
                record_id=record.id,
                score=float(score),
                reason="low-similarity baseline anti-hit",
                literal_surface=record.literal_surface,
                adjacent_suggestions=[],
            )
        )

    derive_temporal_validity(store, hits)
    derive_temporal_validity(store, anti_hits)
    apply_stale_downweight(hits)
    apply_stale_downweight(anti_hits)
    # Rank on the internal unclamped key (falls back to score), so ordering
    # survives the display clamp applied at serialization.
    hits.sort(
        key=lambda h: (h.sort_score if h.sort_score is not None else h.score),
        reverse=True,
    )

    try:
        from iai_mcp.s4 import on_read_check
        s4_hints = on_read_check(store, hits, session_id=session_id)
    except (OSError, ValueError, RuntimeError) as exc:
        log.warning("s4 on_read_check failed: %s", exc)
        s4_hints = []

    response = RecallResponse(
        hits=hits,
        anti_hits=anti_hits,
        activation_trace=[h.record_id for h in hits],
        budget_used=sum(len(h.literal_surface) for h in hits) // 4,
        hints=s4_hints,
        cue_mode=mode,
        patterns_observed=[],
    )

    try:
        write_event(
            store,
            kind="retrieval_used",
            data={
                "hit_ids": [str(h.record_id) for h in hits],
                "query": cue_text,
                "used": len(hits) > 0,
                "budget_used": response.budget_used,
                "path": "baseline_recall",
            },
            severity="info",
            session_id=session_id,
            buffered=True,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        log.warning("retrieval_used event write failed: %s", exc)

    return response


def reinforce_edges(
    store: MemoryStore, ids: list[UUID], delta: float = 0.1
) -> EdgeUpdate:
    pairs: list[tuple[UUID, UUID]] = list(combinations(ids, 2))
    new_weights = store.boost_edges(pairs, delta=delta)
    new_weights_str = {f"{a}|{b}": float(w) for (a, b), w in new_weights.items()}
    return EdgeUpdate(
        edges_boosted=len(pairs),
        pairs=pairs,
        new_weights=new_weights_str,
    )


def contradict(
    store: MemoryStore,
    original_id: UUID,
    new_fact: str,
    new_embedding: list[float],
) -> ReconsolidationReceipt:
    flush_record_buffer(store)
    original = store.get(original_id)
    if original is None:
        raise ValueError(f"unknown record {original_id}")
    target_dim = store.embed_dim
    if len(new_embedding) != target_dim:
        raise ValueError(
            f"new_embedding must be {target_dim}d, got {len(new_embedding)}"
        )
    now = datetime.now(timezone.utc)
    new_rec = MemoryRecord(
        id=uuid4(),
        tier=original.tier,
        literal_surface=new_fact,
        aaak_index="",
        embedding=list(new_embedding),
        community_id=original.community_id,
        centrality=0.0,
        detail_level=original.detail_level,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=(original.detail_level >= 3),
        never_merge=False,
        provenance=[{"ts": now.isoformat(), "cue": "contradict", "session_id": "-"}],
        created_at=now,
        updated_at=now,
        tags=["contradict"],
        language=getattr(original, "language", "en") or "en",
    )
    enforce_english_raw(new_rec)
    new_rec.aaak_index = generate_aaak_index(new_rec)
    store.insert(new_rec)
    store.add_contradicts_edge(original_id, new_rec.id)
    invalidate_temporal_validity_cache(store)

    try:
        from iai_mcp.s4 import monotropic_proactive_check
        monotropic_proactive_check(store, new_rec, {}, session_id="-")
    except (OSError, ValueError, RuntimeError) as exc:
        log.warning("monotropic_proactive_check failed: %s", exc)

    return ReconsolidationReceipt(
        original_id=original_id,
        new_record_id=new_rec.id,
        edge_type="contradicts",
        ts=now,
    )


_tv_cache: dict[int, tuple[dict[str, list[str]], dict[str, datetime]]] = {}
_tv_cache_dirty: dict[int, bool] = {}


def invalidate_temporal_validity_cache(store: "MemoryStore") -> None:
    _tv_cache_dirty[id(store)] = True


def build_temporal_validity_maps(
    store: MemoryStore,
) -> tuple[dict[str, list[str]], dict[str, datetime]] | None:
    store_id = id(store)
    if not _tv_cache_dirty.get(store_id, True) and store_id in _tv_cache:
        return _tv_cache[store_id]

    edges_tbl = store.db.open_table("edges")
    try:
        edges_count = int(edges_tbl.count_rows())
        if edges_count > 0:
            edges_df = (
                edges_tbl.search()
                .select(["src", "dst", "edge_type"])
                .limit(edges_count)
                .to_pandas()
            )
        else:
            edges_df = None
    except (OSError, ValueError, RuntimeError) as exc:
        log.warning("build_temporal_validity_maps edges read failed: %s", exc)
        return None

    outgoing: dict[str, list[str]] = {}
    if edges_df is not None and not edges_df.empty:
        try:
            ctr = edges_df[edges_df["edge_type"] == "contradicts"]
        except (KeyError, ValueError, RuntimeError) as exc:
            log.warning("build_temporal_validity_maps filter failed: %s", exc)
            return None
        if not ctr.empty:
            try:
                for src_s, dst_s in zip(
                    ctr["src"].tolist(), ctr["dst"].tolist(), strict=False
                ):
                    outgoing.setdefault(str(src_s), []).append(str(dst_s))
            except (KeyError, ValueError, RuntimeError) as exc:
                log.warning("build_temporal_validity_maps zip failed: %s", exc)
                return None


    try:
        records_tbl = store.db.open_table("records")
        records_count = int(records_tbl.count_rows())
        if records_count > 0:
            records_df = (
                records_tbl.search()
                .select(["id", "created_at"])
                .limit(records_count)
                .to_pandas()
            )
            def _parse_ts(v: object) -> datetime:
                if isinstance(v, datetime):
                    return v if v.tzinfo is not None else v.replace(tzinfo=timezone.utc)
                s = str(v)
                dt = datetime.fromisoformat(s)
                return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

            ts_by_id: dict[str, datetime] = {
                str(k): _parse_ts(v)
                for k, v in zip(
                    records_df["id"].tolist(),
                    records_df["created_at"].tolist(),
                    strict=False,
                )
            }
        else:
            ts_by_id = {}
    except (OSError, ValueError, RuntimeError) as exc:
        log.warning("build_temporal_validity_maps records read failed: %s", exc)
        return None
    _result_full: tuple[dict[str, list[str]], dict[str, datetime]] = (outgoing, ts_by_id)
    _tv_cache[store_id] = _result_full
    _tv_cache_dirty[store_id] = False
    return _result_full


def derive_temporal_validity(
    store: MemoryStore | None,
    hits: list[MemoryHit],
    records_cache: dict[UUID, MemoryRecord] | None = None,
    *,
    outgoing: dict[str, list[str]] | None = None,
    ts_by_id: dict[str, datetime] | None = None,
) -> list[MemoryHit]:
    if not hits:
        return hits

    if outgoing is None or ts_by_id is None:
        if store is None:
            return hits
        built = build_temporal_validity_maps(store)
        if built is None:
            return hits
        outgoing, ts_by_id = built

    def _created_at(rid: UUID) -> datetime | None:
        return ts_by_id.get(str(rid))

    for hit in hits:
        src_ts = _created_at(hit.record_id)
        if src_ts is None:
            continue
        hit.valid_from = src_ts
        candidates = outgoing.get(str(hit.record_id), [])
        if not candidates:
            continue
        oldest_newer: datetime | None = None
        for dst_str in candidates:
            try:
                dst_id = UUID(dst_str)
            except (TypeError, ValueError):
                continue
            dst_ts = _created_at(dst_id)
            if dst_ts is None:
                continue
            if dst_ts <= src_ts:
                continue
            if oldest_newer is None or dst_ts < oldest_newer:
                oldest_newer = dst_ts
        if oldest_newer is not None:
            hit.valid_to = oldest_newer
    return hits


def apply_stale_downweight(
    hits: list[MemoryHit],
    now: datetime | None = None,
    *,
    cue_intent: str | None = None,
) -> list[MemoryHit]:
    if cue_intent == "historical_verbatim":
        return hits
    now_value = now or datetime.now(timezone.utc)
    for hit in hits:
        if hit.valid_to is None or hit.valid_to >= now_value:
            continue
        if not getattr(hit, "_stale_downweighted", False):
            hit.score *= STALE_DOWNWEIGHT_FACTOR
            # Keep the internal ranking key in lock-step with the displayed
            # score so stale hits demote in the actual ordering too.
            if getattr(hit, "sort_score", None) is not None:
                hit.sort_score *= STALE_DOWNWEIGHT_FACTOR
            hit._stale_downweighted = True
        if not hit.reason.endswith(_STALE_REASON_SUFFIX):
            hit.reason = f"{hit.reason}{_STALE_REASON_SUFFIX}"
    return hits


def link_temporal_next(
    store: MemoryStore,
    new_record: MemoryRecord,
    session_id: str,
) -> UUID | None:
    now = datetime.now(timezone.utc)
    prior_events = query_events(
        store, kind="record_inserted",
        since=now - TEMPORAL_NEXT_WINDOW, limit=20,
    )
    previous_id: UUID | None = None
    for ev in prior_events:
        if ev.get("session_id") != session_id:
            continue
        raw = ev["data"].get("record_id")
        if not raw:
            continue
        try:
            candidate = UUID(raw)
        except (TypeError, ValueError):
            continue
        if candidate == new_record.id:
            continue
        previous_id = candidate
        break

    if previous_id is not None:
        try:
            store.boost_edges(
                [(previous_id, new_record.id)],
                edge_type="temporal_next",
                delta=1.0,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            log.warning("temporal_next edge creation failed: %s", exc)

    write_event(
        store,
        kind="record_inserted",
        data={
            "record_id": str(new_record.id),
            "tier": new_record.tier,
        },
        severity="info",
        session_id=session_id,
        source_ids=[new_record.id],
    )
    return previous_id


def _make_graph_sync_hook(graph):
    def _hook(op: str, record) -> None:
        nid = record.id
        nid_str = str(nid)
        if op in ("insert", "update"):
            payload = {
                "embedding": list(record.embedding),
                "surface": record.literal_surface,
                "centrality": float(record.centrality),
                "tier": record.tier,
                "pinned": bool(record.pinned),
                "tags": list(getattr(record, "tags", []) or []),
                "language": str(getattr(record, "language", "en") or "en"),
            }
            if nid_str not in graph._node_payload:
                graph.add_node(
                    nid,
                    community_id=None,
                    embedding=payload["embedding"],
                )
            graph.set_node_payload(nid, payload)
            try:
                from iai_mcp import runtime_graph_cache as _rgc
                _rgc.increment_dirty_counter()
            except Exception:  # noqa: BLE001 -- never break a record write
                pass
        elif op == "delete":
            graph.remove_node(nid)
            try:
                from iai_mcp import runtime_graph_cache as _rgc
                _rgc.increment_dirty_counter()
            except Exception:  # noqa: BLE001 -- never break a record delete
                pass
    return _hook


def _detect_communities_isolated(store: MemoryStore, graph, *, with_centrality: bool = False):
    """Run community detection without retaining the kernel arenas in-parent.

    The detection kernel's JIT compilation reserves large allocator arenas that
    the long-lived process never hands back. Running it in a short-lived
    spawn-context child confines those arenas to that child, which the OS
    reclaims on exit, keeping the parent footprint flat.

    The child receives only node ids, float32 embeddings, and edges — never the
    storage handle or the encryption key. The returned partition (which nodes
    share a community) is identical to the in-process call; only the community
    identifiers may differ, and callers compare partitions, not identifiers.

    When `with_centrality` is True the same child also computes the full
    betweenness centrality and the function returns `(assignment,
    centrality_map)`; on the in-process fallback the centrality map is returned
    as `None` so the caller computes centrality on its own path. When False the
    function returns just the assignment.

    If the child path fails for any reason, detection falls back to running
    in-process so recall is never blocked.
    """
    from iai_mcp import runtime_graph_cache

    try:
        result = runtime_graph_cache.compute_assignment_in_child(
            graph, prior_mode="seeded", with_centrality=with_centrality
        )
        if with_centrality:
            assignment, centrality_map = result
            return assignment, centrality_map
        return result
    except (
        runtime_graph_cache.WorkerCrashedError,
        runtime_graph_cache.WorkerTimeoutError,
        BrokenPipeError,
        EOFError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        log.warning(
            "community detection child failed; "
            "falling back to in-process detection: %s",
            exc,
        )
        from iai_mcp.community import detect_communities

        assignment = detect_communities(graph, prior=None, prior_mode="seeded")
        if with_centrality:
            # Signal the caller to compute centrality on its own path (the
            # in-parent fallback) — the child never produced it.
            return assignment, None
        return assignment


# Serializes the cache-MISS rebuild of the runtime graph across concurrent
# callers. The rebuild streams the whole corpus and spawns a child for community
# detection plus centrality; under a stale or absent cache several callers can
# fire at once and each run the full rebuild, spawning a redundant fleet of
# children grinding the same graph. A single-flight guard around the rebuild
# section collapses that fleet to one: the first caller rebuilds and saves the
# cache, the rest re-check the freshly-saved cache and take the light path. The
# daemon's callers run in `asyncio.to_thread` worker threads, so a threading
# lock serializes them. The cheap cache-hit probe runs OUTSIDE this lock, so the
# common warm path stays lock-free.
_RUNTIME_GRAPH_REBUILD_LOCK = threading.Lock()


def _runtime_graph_rebuild_needed(store: MemoryStore) -> bool:
    """Cheap probe: does a full runtime-graph rebuild need to run?

    Returns False only when the cache holds the expensive results — a non-empty
    community assignment AND a cached centrality map — for a corpus whose size is
    within drift tolerance of the live count. That is the exact condition under
    which `_build_runtime_graph_impl` reconstructs the graph by streaming the
    cheap node_payload from the store and applies the cached centrality, spawning
    no detection or centrality child. Any other state (no cache, size drift, or
    no cached centrality) means a child-spawning rebuild is required.

    Crucially this gates on the compact `payload_record_count` + cached
    `centrality` map, NOT on the large `node_payload`. The node_payload is shed
    when the cache exceeds its size cap (at production-scale corpora it always
    is), so gating on its presence would force a betweenness recompute on every
    warm. The expensive centrality survives the cap and is what the warm path
    reuses.

    Mirrors the in-function `cache_results_fresh` logic so the single-flight
    decision and the rebuild decision cannot diverge. Performs only a disk read
    (`try_load_cache_results`) and a COUNT(*) (`active_records_count`) — no
    rebuild, no child spawn — so it is safe to call lock-free and again under the
    lock for the double check.
    """
    from iai_mcp import runtime_graph_cache

    results = runtime_graph_cache.try_load_cache_results(store)
    if results is None:
        return True
    cached_centrality, payload_record_count = results
    # An empty centrality map means the cache carries no expensive result to
    # reuse — rebuild so the betweenness pass actually runs once.
    if not cached_centrality:
        return True
    # A payload built from an all-zero centrality set is not a usable warm result
    # (e.g. a single isolated node). Treat it as needing a rebuild.
    if not any(value != 0.0 for value in cached_centrality.values()):
        return True
    # Drift tolerance: a corpus that has grown/shrunk by a bounded number of
    # records since the cache was built still reuses the cached graph — the
    # lagging records remain index-findable, so no rebuild is forced for small
    # drift. Only an over-tolerance delta requires a full child-spawning rebuild.
    if not _within_drift_tolerance(
        payload_record_count, store.active_records_count()
    ):
        return True
    return False


def build_runtime_graph(store: MemoryStore):
    # Common path: a warm cache that needs no rebuild reconstructs the graph
    # without spawning any child — run it lock-free so warm recall never blocks
    # on a peer's rebuild.
    if not _runtime_graph_rebuild_needed(store):
        return _build_runtime_graph_impl(store)

    # Cache miss: single-flight the rebuild so concurrent callers don't each
    # spawn a redundant child fleet on the same graph. Double-checked — a peer
    # that held the lock may have rebuilt and saved a fresh cache while this
    # caller waited, in which case the re-probe passes and the impl takes the
    # light cache-hit branch, spawning no child of its own. Otherwise this caller
    # is the single-flight winner and performs the one rebuild.
    with _RUNTIME_GRAPH_REBUILD_LOCK:
        return _build_runtime_graph_impl(store)


def _build_runtime_graph_impl(store: MemoryStore):
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.richclub import rich_club_nodes
    from iai_mcp import runtime_graph_cache

    graph = MemoryGraph()

    cached = runtime_graph_cache.try_load(store)
    assignment = None
    rich_club = None
    cached_node_payload: dict[str, dict] | None = None
    cached_max_degree: int = 0
    if cached is not None:
        assignment, rich_club, cached_node_payload, cached_max_degree = cached

    records_count_for_shrink = store.active_records_count()
    if (
        cached_node_payload is not None
        and len(cached_node_payload) > records_count_for_shrink
    ):
        # The cache holds MORE nodes than are live: records were tombstoned
        # (dedup/erasure) since it was built, so the cached assignment / rich_club
        # were computed over now-dead nodes. Drop them and recompute on the fresh
        # live graph (rebuilt from the records table below, which already excludes
        # tombstoned rows). Pure GROWTH (cache has FEWER nodes than live) is left
        # to the drift-tolerance gate, which reuses the assignment so a single
        # insert is absorbed without an O(n^2) recompute. Dropping the cached
        # payload here also forces the streaming rebuild so no dead node survives
        # via a verbatim payload reuse, and a stale rich_club can never leak a
        # tombstoned node downstream.
        assignment = None
        rich_club = None
        cached_node_payload = None

    # The compact results (community assignment + centrality) survive the size
    # cap even when the large node_payload is shed; at production-scale corpora
    # the payload is always shed, so this is the only signal that the expensive
    # betweenness was already computed. Decoupling it from node_payload presence
    # is what keeps the betweenness recompute off the warm path.
    cache_results = runtime_graph_cache.try_load_cache_results(store)
    cached_centrality: dict | None = None
    cached_payload_record_count = 0
    if cache_results is not None:
        cached_centrality, cached_payload_record_count = cache_results

    records_count = store.active_records_count()
    # The expensive results are fresh — a non-empty assignment plus a cached
    # centrality map for a corpus within drift of the live count. When fresh, the
    # graph is rebuilt cheaply (streaming the node_payload) and the cached
    # centrality is applied directly: neither community detection nor the
    # betweenness child fires. Records added since the cache was built are absent
    # from the community graph until the next over-tolerance rebuild, but stay
    # findable via the index recall path, so recall correctness holds.
    cache_results_fresh = (
        assignment is not None
        and cached_centrality is not None
        and len(cached_centrality) > 0
        and _within_drift_tolerance(cached_payload_record_count, records_count)
    )

    # Fast path: the full node_payload is still present (small-corpus cache that
    # was not shed) and within drift — reuse it verbatim, no re-streaming. When
    # the payload was shed (large corpus) this is False and the graph is rebuilt
    # by streaming below, but the cached centrality is still applied so the warm
    # path stays betweenness-free.
    use_cached_payload = (
        cached_node_payload is not None
        and len(cached_node_payload) > 0
        and _within_drift_tolerance(len(cached_node_payload), records_count)
    )

    if use_cached_payload:
        for nid, payload in cached_node_payload.items():
            graph.add_node(
                UUID(nid),
                community_id=None,
                embedding=list(payload.get("embedding") or []),
            )
            graph.set_node_payload(nid, {
                "embedding": list(payload.get("embedding") or []),
                "surface": payload.get("surface", ""),
                "centrality": float(payload.get("centrality") or 0.0),
                "tier": payload.get("tier", "episodic"),
                "pinned": bool(payload.get("pinned", False)),
                "tags": list(payload.get("tags") or []),
                "language": str(payload.get("language", "en") or "en"),
            })
        node_payload_for_cache = cached_node_payload
    else:
        node_payload_for_cache = {}
        decrypt_fail_events = 0
        decrypt_fail_unique: set[str] = set()
        # Stream the corpus column-by-column in bounded batches instead of
        # materializing the whole records table into one DataFrame. Only the
        # columns the graph needs are projected; the embedding blob is decoded
        # to a float list by the streaming reader, matching the prior decode.
        stream_cols = [
            "id",
            "embedding",
            "community_id",
            "embedding_pending",
            "literal_surface",
            "tier",
            "centrality",
            "pinned",
            "tags_json",
            "language",
            # Projected so the streamed dict carries the tombstone marker for the
            # in-loop defensive guard below (the primary filter is the SQL WHERE).
            "tombstoned_at",
        ]
        # Exclude tombstoned (soft-deleted / deduped / erased) records from the
        # runtime graph at the SQL layer: they must not pollute communities,
        # centrality, rich_club or the sigma topology audit, and including them
        # keeps the node count out of sync with active_records_count() (the
        # cache-validity anchor), permanently invalidating the cache -> a full
        # rebuild every wake. Matches active_records_count(): tombstoned_at IS NULL.
        for row in store.iter_record_columns(
            stream_cols, batch_size=1024, where="tombstoned_at IS NULL"
        ):
            if int(row.get("embedding_pending") or 0) != 0:
                continue
            # Defensive in-loop guard mirroring the SQL filter: only a real,
            # non-empty timestamp string marks a tombstone. Keep pandas out of
            # this daemon rebuild path; it showed up in live latency profiles
            # while recall was waiting on the daemon.
            _tomb = row.get("tombstoned_at")
            if _has_tombstone_marker(_tomb):
                continue
            rid = UUID(row["id"])
            _comm_raw = row.get("community_id")
            community_id = UUID(_comm_raw) if _comm_raw else None
            _emb_raw = row.get("embedding")
            embedding = (
                list(_emb_raw)
                if _emb_raw is not None
                else [0.0] * EMBED_DIM
            )
            literal_raw = row.get("literal_surface") or ""
            try:
                from iai_mcp.crypto import is_encrypted
                if is_encrypted(literal_raw):
                    literal_raw = store._decrypt_for_record(rid, literal_raw)
            except Exception:  # noqa: BLE001 -- InvalidTag / OSError / ValueError / RuntimeError
                rid_s = str(rid)
                decrypt_fail_events += 1
                decrypt_fail_unique.add(rid_s)
                now_m = time.monotonic()
                last_m = _GRAPH_DECRYPT_WARN_LAST.get(rid_s, 0.0)
                if now_m - last_m >= _GRAPH_DECRYPT_WARN_INTERVAL_SEC:
                    _GRAPH_DECRYPT_WARN_LAST[rid_s] = now_m
                    log.warning(
                        "graph_build_decrypt_failed",
                        extra={"record_id": rid_s},
                    )
                continue

            tier = row.get("tier") or "episodic"
            centrality = float(row.get("centrality") or 0.0)
            pinned = bool(row.get("pinned") or False)
            tags_raw = row.get("tags_json") or "[]"
            try:
                import json as _json
                tags_list = _json.loads(tags_raw) if isinstance(tags_raw, str) else list(tags_raw)
                if not isinstance(tags_list, list):
                    tags_list = []
            except (ValueError, TypeError):
                tags_list = []
            language = str(row.get("language") or "en")

            graph.add_node(
                rid,
                community_id=community_id,
                embedding=embedding,
            )
            graph.set_node_payload(rid, {
                "embedding": list(embedding),
                "surface": str(literal_raw),
                "centrality": centrality,
                "tier": str(tier),
                "pinned": pinned,
                "tags": list(tags_list),
                "language": language,
            })
            node_payload_for_cache[str(rid)] = {
                "embedding": list(embedding),
                "surface": str(literal_raw),
                "centrality": centrality,
                "tier": str(tier),
                "pinned": pinned,
                "tags": list(tags_list),
                "language": language,
            }

        if decrypt_fail_events > 0:
            log.warning(
                "graph_build_decrypt_failed_summary",
                extra={
                    "unique_records": len(decrypt_fail_unique),
                    "total_skip_events": decrypt_fail_events,
                },
            )

    with store.db._conn_lock:
        edge_rows = store.db._conn.execute(
            "SELECT src, dst, edge_type, weight FROM edges"
        ).fetchall()
    for row in edge_rows:
        # Skip edges whose endpoints are not live nodes: graph.add_edge() does
        # setdefault() on both endpoints, so an edge referencing a tombstoned
        # record would re-create it as a payload-less node and undo the tombstone
        # filter above (re-bloating the graph and the sigma audit).
        src_s, dst_s = row["src"], row["dst"]
        if not graph.has_node(src_s) or not graph.has_node(dst_s):
            continue
        graph.add_edge(
            UUID(src_s),
            UUID(dst_s),
            weight=float(row["weight"]),
            edge_type=row["edge_type"],
        )

    try:
        deg_values = [d for _, d in graph.degrees()]
        max_degree = max(deg_values) if deg_values else 0
    except (ValueError, RuntimeError, AttributeError):
        max_degree = cached_max_degree
    if max_degree == 0 and cached_max_degree > 0:
        max_degree = cached_max_degree
    graph._max_degree = int(max_degree)

    def _apply_centrality_map(centrality_map) -> None:
        """Write a node->value centrality map into the graph and the cache
        payload, identically to the in-parent path."""
        for rid, cval in centrality_map.items():
            nid_str = str(rid)
            if nid_str in graph._node_payload:
                graph.set_node_centrality(rid, float(cval))
                if (
                    node_payload_for_cache is not None
                    and nid_str in node_payload_for_cache
                ):
                    node_payload_for_cache[nid_str]["centrality"] = float(cval)

    # `child_centrality` carries the centrality map when the detection child
    # computed it on the same graph build (cache-miss path) — avoiding both a
    # second child spawn and the in-parent betweenness intermediate. The rich
    # club is deferred until the centrality is resolved (child / cached / neutral)
    # so it can rank from that map rather than triggering its own in-parent
    # betweenness pass on the long-lived process.
    child_centrality = None
    recompute_rich_club = False
    if assignment is None:
        assignment, child_centrality = _detect_communities_isolated(
            store, graph, with_centrality=True
        )
        recompute_rich_club = True

    # Warm path: the expensive results (community partition + centrality) were
    # already computed and survived the cache size cap. Apply the cached
    # centrality to the freshly-streamed (or cache-reused) graph and skip the
    # betweenness child entirely. Nodes absent from the cached map — the bounded
    # drift delta — keep the centrality their row carried (or 0.0 by default),
    # which is the same bounded staleness the community graph already tolerates.
    # The centrality map this cycle resolved to (child / cached / neutral), in
    # the same node->value shape `graph.centrality()` would return. The rich club
    # ranks from it so the parent never runs a second exact betweenness pass.
    resolved_centrality: dict = {}
    if cache_results_fresh and cached_centrality is not None:
        _apply_centrality_map(cached_centrality)
        resolved_centrality = dict(cached_centrality)
        needs_centrality = False
    else:
        needs_centrality = True
        if use_cached_payload and cached_node_payload is not None:
            any_nonzero = any(
                float(p.get("centrality") or 0.0) != 0.0
                for p in cached_node_payload.values()
            )
            needs_centrality = not any_nonzero
            if not needs_centrality:
                resolved_centrality = {
                    UUID(nid): float(p.get("centrality") or 0.0)
                    for nid, p in cached_node_payload.items()
                }
    # Set when the centrality for this cycle is a bounded degrade (last-good
    # cached map, or a neutral all-zero map) rather than a freshly-computed
    # result. A degraded result is never persisted under the current key — that
    # would mask the retry and let a stale signal masquerade as fresh — so the
    # prior good cache stays on disk and the next warm cycle recomputes.
    centrality_degraded = False
    if needs_centrality:
        if child_centrality is not None:
            # The detection child already produced centrality on this graph.
            _apply_centrality_map(child_centrality)
            resolved_centrality = dict(child_centrality)
        else:
            # Either the cache-hit path (no fresh detection) or the in-process
            # detection fallback (child crashed). Compute centrality in a child
            # so the betweenness intermediate stays out of the parent.
            try:
                centrality_map = runtime_graph_cache.compute_centrality_in_child(
                    graph
                )
                _apply_centrality_map(centrality_map)
                resolved_centrality = dict(centrality_map)
            except (
                runtime_graph_cache.WorkerCrashedError,
                runtime_graph_cache.WorkerTimeoutError,
                BrokenPipeError,
                EOFError,
                OSError,
                RuntimeError,
                ValueError,
            ) as exc:
                # Bounded degrade. The child centrality timed out or failed.
                # Computing exact betweenness in this long-lived process is an
                # unbounded O(V*E) compute that, at scale, spikes the resident
                # set toward the watchdog cap, never completes, never caches, and
                # is retried every cycle — the over-cap kill loop. So the warm
                # path NEVER recomputes centrality in-parent here. It serves the
                # last-good cached centrality when one survives on disk, else a
                # neutral (zero) centrality for this cycle. Recall stays correct
                # under either: seeds rank by 0.6*cos + 0.4*centrality, so a
                # stale or neutral centrality term degrades to cosine-led seeds,
                # never a crash or an empty recall.
                last_good = runtime_graph_cache.load_last_good_centrality(store)
                if last_good:
                    log.warning(
                        "centrality child failed; serving last-good cached "
                        "centrality (%d nodes), will retry next cycle: %s",
                        len(last_good),
                        exc,
                    )
                    _apply_centrality_map(last_good)
                    resolved_centrality = dict(last_good)
                else:
                    log.warning(
                        "centrality child failed and no cached centrality is "
                        "available; serving neutral centrality (cosine-led "
                        "seeds), will retry next cycle: %s",
                        exc,
                    )
                    resolved_centrality = {}
                    for nid in graph.iter_nodes():
                        graph.set_node_centrality(nid, 0.0)
                        resolved_centrality[nid] = 0.0
                        nid_str = str(nid)
                        if (
                            node_payload_for_cache is not None
                            and nid_str in node_payload_for_cache
                        ):
                            node_payload_for_cache[nid_str]["centrality"] = 0.0
                centrality_degraded = True

    # Rich club from the resolved centrality, never a fresh in-parent betweenness
    # pass. Only the cache-miss path (where detection ran in the child) needs it
    # recomputed; the cache-hit path already carries the cached rich club.
    if recompute_rich_club:
        rich_club = rich_club_nodes(
            graph, percent=0.10, centrality=resolved_centrality
        )

    # A bounded degrade is never persisted: leaving the prior good cache intact
    # both preserves the last-good signal for the next cycle's degrade and forces
    # the retry (the freshly-recomputed result will overwrite it once a child
    # succeeds).
    if not centrality_degraded and (cached_node_payload is None or needs_centrality):
        runtime_graph_cache.save(
            store, assignment, rich_club,
            node_payload=node_payload_for_cache,
            max_degree=int(getattr(graph, "_max_degree", 0) or 0),
        )

    try:
        store.register_graph_sync_hook(_make_graph_sync_hook(graph))
    except (AttributeError, TypeError, RuntimeError) as exc:
        log.warning("graph_sync_hook registration failed: %s", exc)

    if not hasattr(graph, "_max_degree"):
        graph._max_degree = 0

    return graph, assignment, rich_club
