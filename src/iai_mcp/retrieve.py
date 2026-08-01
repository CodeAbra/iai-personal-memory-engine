from __future__ import annotations

import logging
import math
import os
from iai_mcp import errors
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from itertools import combinations
from typing import Any
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


# Boot-rebuild back-pressure: the streaming loops run in a worker thread, but a
# long uninterrupted pass can still starve a sibling foreground-recall thread
# past a sub-second SLA. `time.sleep(0)` forces a real OS-scheduler yield;
# yielding every `_GRAPH_STREAM_CHUNK_ROWS` rows bounds how long any single
# slice runs. Operator-overridable; a non-positive override uses the default.
_GRAPH_STREAM_CHUNK_ROWS_DEFAULT: int = 256


def _graph_stream_chunk_rows() -> int:
    raw = os.environ.get("IAI_MCP_RGC_STREAM_CHUNK_ROWS")
    if raw is None:
        return _GRAPH_STREAM_CHUNK_ROWS_DEFAULT
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return _GRAPH_STREAM_CHUNK_ROWS_DEFAULT
    return parsed if parsed > 0 else _GRAPH_STREAM_CHUNK_ROWS_DEFAULT


def _yield_to_event_loop() -> None:
    """Force an actual OS-level thread-scheduling yield at a streaming-loop
    chunk boundary so a foreground recall in a sibling thread gets to run.

    Also honours the foreground-activity beacon: while a live recall is in
    flight the rebuild pauses here (bounded, fail-open) — background rebuild
    work must never compete with an awake read.
    """
    try:
        from iai_mcp.concurrency import foreground_backoff

        foreground_backoff(max_wait_s=0.5)
    except Exception:  # noqa: BLE001 -- politeness is advisory
        pass
    time.sleep(0)


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
    # A missing or all-zero cue vector must be embedded HERE, not searched:
    # the SLEEP/exception fallback dispatchers pad an absent cue_embedding
    # with zeros, and ranking by distance to the zero vector returns
    # cue-IRRELEVANT memories — the hippocampus must answer correctly on
    # every path, not just the primary one. Embed failure keeps the caller's
    # vector (degraded, never a crash into recall).
    if cue_text and (not cue_embedding or not any(cue_embedding)):
        try:
            from iai_mcp.embed import embed_query, embedder_for_store
            # Full cue, same as the primary path — the encoder truncates at
            # its own token limit; a char slice here would rank differently.
            cue_embedding = list(embed_query(embedder_for_store(store), cue_text))
        except Exception as exc:  # noqa: BLE001 -- degraded beats dead
            log.warning("recall cue re-embed failed, using caller vector: %s", exc)

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
    # The anti window must never overlap the hit window: with fewer than
    # k_hits+k_anti survivors (small corpus, verbatim-filtered) the head and
    # tail slices intersect, and a TOP memory would be served as a
    # low-similarity anti-hit — anti_hits carry the same contractual weight
    # as hits.
    hit_ids = {h.record_id for h in hits}
    tail = raw[-k_anti:] if len(raw) >= k_anti else []
    for record, score in reversed(tail):
        if record.id in hit_ids:
            continue
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
    hits.sort(key=lambda h: h.score, reverse=True)

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


WAKE_COACTIVATION_DELTA: float = 0.1
WAKE_COACTIVATION_MAX_HITS: int = 5
WAKE_COACTIVATION_MIN_SCORE: float = 0.5


def potentiate_coactivation(
    store: MemoryStore,
    ids: "list[UUID]",
    delta: float = WAKE_COACTIVATION_DELTA,
) -> int:
    """Awake Hebbian plasticity: records recalled together get a bounded
    pairwise potentiation, so the connective graph consolidation clusters
    over reflects actual co-activation — without it the hebbian graph never
    gains a cross-record edge and the semantic minter starves. The write is
    DEFERRED through the reinforce queue — plasticity never runs on the
    synchronous recall path. Kill-switch: IAI_MCP_WAKE_COACTIVATION=0."""
    import os

    if os.environ.get("IAI_MCP_WAKE_COACTIVATION", "1") == "0":
        return 0
    uniq: list[UUID] = []
    seen: set[UUID] = set()
    for rid in ids:
        if rid not in seen:
            seen.add(rid)
            uniq.append(rid)
    uniq = uniq[:WAKE_COACTIVATION_MAX_HITS]
    if len(uniq) < 2:
        return 0
    pairs = [(a, b) for a, b in combinations(uniq, 2) if a != b]
    if not pairs:
        return 0
    store.queue_coactivation(pairs, delta)
    return len(pairs)


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
    # An absent or all-zero corrector vector must be embedded HERE (the
    # dispatcher pads omissions with zeros and no MCP client sends one):
    # a zero-embedded corrector is semantically invisible while the
    # SUPERSEDED belief keeps surfacing — the inverse of what contradiction
    # exists for. Same contract as recall's cue re-embed.
    if new_fact and (not new_embedding or not any(new_embedding)):
        from iai_mcp.embed import embedder_for_store
        new_embedding = list(embedder_for_store(store).embed(new_fact))
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
    # The insert dedup gate may fold a near-identical corrector into an
    # existing record (rewriting new_rec.id to the winner). Folding into the
    # CONTRADICTED record itself would wire a self-contradiction loop — the
    # correction cannot be the belief it corrects.
    if new_rec.id == original_id:
        raise ValueError(
            "corrector text deduplicated into the contradicted record itself; "
            "rephrase the correction so it is distinguishable"
        )
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


def _parse_created_ts(v: object) -> datetime:
    if isinstance(v, datetime):
        return v if v.tzinfo is not None else v.replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(str(v))
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


# Weak-keyed on the store OBJECT, not id(store): an id can be reused by a
# new store after the old one is collected (serving a dead store's maps),
# and id-keyed entries never evict. Weak keys evict with the store.
import weakref as _weakref

_tv_cache: "_weakref.WeakKeyDictionary" = _weakref.WeakKeyDictionary()
_tv_cache_dirty: "_weakref.WeakKeyDictionary" = _weakref.WeakKeyDictionary()


def invalidate_temporal_validity_cache(store: "MemoryStore") -> None:
    try:
        _tv_cache_dirty[store] = True
    except TypeError:
        # Non-weakrefable store stub (tests): no cache, nothing to dirty.
        pass


def build_temporal_validity_maps(
    store: MemoryStore,
) -> tuple[dict[str, list[str]], dict[str, datetime]] | None:
    try:
        if not _tv_cache_dirty.get(store, True) and store in _tv_cache:
            return _tv_cache[store]
    except TypeError:
        pass

    # Bounded by the number of CONTRADICTIONS, never the corpus: this runs on
    # the awake fallback recall lane and is invalidated by every contradict —
    # including sleep-time reconsolidation — so a full-table materialization
    # here couples an awake recall's latency to corpus size. Only contradicts
    # edges and the timestamps of the records they touch are needed;
    # derive_temporal_validity falls back to the hit's own captured_at for
    # valid_from.
    outgoing: dict[str, list[str]] = {}
    involved: set[str] = set()
    try:
        with store.db.ro_conn() as conn:
            rows = conn.execute(
                "SELECT src, dst FROM edges WHERE edge_type = 'contradicts'"
            ).fetchall()
        for row in rows:
            src_s, dst_s = str(row[0]), str(row[1])
            outgoing.setdefault(src_s, []).append(dst_s)
            involved.add(src_s)
            involved.add(dst_s)
    except (OSError, ValueError, RuntimeError, errors.Error) as exc:
        log.warning("build_temporal_validity_maps edges read failed: %s", exc)
        return None

    ts_by_id: dict[str, datetime] = {}
    try:
        ids = sorted(involved)
        with store.db.ro_conn() as conn:
            for start in range(0, len(ids), 400):
                window = ids[start:start + 400]
                placeholders = ", ".join("?" for _ in window)
                for row in conn.execute(
                    "SELECT id, created_at FROM records"  # nosemgrep: sql-injection
                    f" WHERE id IN ({placeholders})",
                    tuple(window),
                ).fetchall():
                    try:
                        ts_by_id[str(row[0])] = _parse_created_ts(row[1])
                    except (TypeError, ValueError):
                        continue
    except (OSError, ValueError, RuntimeError, errors.Error) as exc:
        log.warning("build_temporal_validity_maps records read failed: %s", exc)
        return None
    _result_full: tuple[dict[str, list[str]], dict[str, datetime]] = (outgoing, ts_by_id)
    try:
        _tv_cache[store] = _result_full
        _tv_cache_dirty[store] = False
    except TypeError:
        pass
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

    # Bounded maps carry timestamps only for contradiction-involved records;
    # the hits' own created_at is fetched here — at most k ids, one query.
    missing = [
        str(h.record_id) for h in hits if str(h.record_id) not in ts_by_id
    ]
    if missing and store is not None:
        try:
            filled = dict(ts_by_id)
            with store.db.ro_conn() as conn:
                placeholders = ", ".join("?" for _ in missing)
                for row in conn.execute(
                    "SELECT id, created_at FROM records"  # nosemgrep: sql-injection
                    f" WHERE id IN ({placeholders})",
                    tuple(missing),
                ).fetchall():
                    try:
                        filled[str(row[0])] = _parse_created_ts(row[1])
                    except (TypeError, ValueError):
                        continue
            ts_by_id = filled
        except Exception as exc:  # noqa: BLE001 -- validity derivation is best-effort
            log.debug("temporal validity hit-timestamp fill failed: %s", exc)

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
            hit._stale_downweighted = True
        if not hit.reason.endswith(_STALE_REASON_SUFFIX):
            hit.reason = f"{hit.reason}{_STALE_REASON_SUFFIX}"
    return hits


_SUPERSEDE_CAP_EPSILON = 1e-4
SUPERSEDE_CAP_WINDOW = 10


def apply_supersede_cap(
    hits: list[MemoryHit],
    outgoing: dict[str, list[str]] | None,
    now: datetime | None = None,
    *,
    cue_intent: str | None = None,
    window: int = SUPERSEDE_CAP_WINDOW,
) -> list[MemoryHit]:
    """Ordering guarantee for current-intent cues: a superseded hit (past
    valid_to) never outranks its best retrieved corrector.

    Mirror image of the historical-verbatim anchor, which lifts the original
    to just below its corrector; here the stale end is capped to just below
    its best corrector. Multiplicative lexical boosts on the stale phrasing
    cannot outbid the cap. Fixpoint loop covers correction chains (A
    corrected by B, B corrected by C).

    A corrector qualifies only while it ranks inside the top-`window` of the
    current scores — the guarantee concerns the SERVED ordering. A corrector
    that is irrelevant to the cue (ranked far below everything) must not drag
    a legitimately matching hit under the noise floor; the dual-route
    anti-hit channel still flags that contradiction.
    """
    if cue_intent == "historical_verbatim" or not outgoing or not hits:
        return hits
    now_value = now or datetime.now(timezone.utc)
    by_id = {str(h.record_id): h for h in hits}
    for _ in range(len(hits)):
        top_ids = {
            str(h.record_id)
            for h in sorted(hits, key=lambda h: h.score, reverse=True)[:window]
        }
        changed = False
        for src_s, dsts in outgoing.items():
            src_hit = by_id.get(src_s)
            if src_hit is None:
                continue
            if src_hit.valid_to is None or src_hit.valid_to >= now_value:
                continue
            best: float | None = None
            for dst_s in dsts or []:
                if str(dst_s) not in top_ids:
                    continue
                dst_hit = by_id.get(str(dst_s))
                if dst_hit is not None and (best is None or dst_hit.score > best):
                    best = dst_hit.score
            if best is not None and src_hit.score >= best:
                src_hit.score = best - _SUPERSEDE_CAP_EPSILON
                changed = True
        if not changed:
            break
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


# Bounds worst-case boot-time lock contention across the whole background
# fleet (graph rebuild, sigma, foraging, deferred-capture drain), not just the
# graph-rebuild callers the lock above already dedups. On a freshly migrated
# large corpus every one of these tasks can independently hold the storage
# engine's writer connection for seconds at a time; letting several of them run
# concurrently at boot serializes them anyway (they all fight over the same
# connection lock) while ALSO fragmenting the executor's thread pool and
# starving the event loop's own scheduling turns (including the socket
# server's bind and the daemon's own liveness-probe roundtrip). A single
# low-priority semaphore around each task's heavy body bounds concurrent
# lock-holders to 1, freeing the executor sooner without changing what any
# task does. Recall never acquires this gate — reader isolation (the RO pool)
# already keeps recall off the writer path entirely, so recall always
# preempts the background fleet by construction, not by priority inversion.
#
# Bounded-wait + retry-with-backoff, never indefinite deferral: a task that
# cannot acquire the gate after every retry runs ANYWAY (fail-open) rather
# than being silently skipped or starved forever — every boot task must still
# complete. A gate bug (acquire always raising) degrades the same way:
# the task still runs, just without the serialization benefit.
_BACKGROUND_STORE_WORK_GATE = threading.BoundedSemaphore(1)

_BACKGROUND_GATE_MAX_WAIT_SEC_DEFAULT = 5.0
_BACKGROUND_GATE_MAX_RETRIES_DEFAULT = 3
_BACKGROUND_GATE_BACKOFF_BASE_SEC_DEFAULT = 0.5


@contextmanager
def background_store_work(
    name: str,
    *,
    max_wait_s: float = _BACKGROUND_GATE_MAX_WAIT_SEC_DEFAULT,
    max_retries: int = _BACKGROUND_GATE_MAX_RETRIES_DEFAULT,
    backoff_base_s: float = _BACKGROUND_GATE_BACKOFF_BASE_SEC_DEFAULT,
):
    """Low-priority single-flight gate around one boot-fleet task's heavy body.

    Acquires ``_BACKGROUND_STORE_WORK_GATE`` with a bounded wait; on a timeout,
    sleeps with linear backoff and retries up to ``max_retries`` times. Yields
    True when the gate is held (or when acquisition itself raised — a gate bug
    must never block a boot task). Yields False when the gate stayed busy: the
    caller must SKIP its heavy body and retry on its next cadence. Running
    anyway was tried and disproven — every gate-timeout task piling on at once
    summed resident sets past the watchdog's hard cap and the SIGKILL lost all
    of them; a deferred task completes later, a killed daemon completes nothing.

    Intended to wrap only the synchronous, heavy store-work body of a boot
    task inside its own ``asyncio.to_thread`` call — never the async
    scheduling wrapper itself, and never any part of the recall/dispatch path
    (recall is reader-isolated and must never acquire this gate).
    """
    acquired = False
    gate_broken = False
    waited_total = 0.0
    try:
        for attempt in range(max_retries + 1):
            try:
                acquired = _BACKGROUND_STORE_WORK_GATE.acquire(timeout=max_wait_s)
            except Exception as exc:  # noqa: BLE001 -- a broken gate must never block boot
                log.warning(
                    "background_store_work gate acquire raised for %s "
                    "(attempt %d): %s -- running without the gate",
                    name, attempt, exc,
                )
                acquired = False
                gate_broken = True
                break
            if acquired:
                break
            if attempt < max_retries:
                backoff_s = backoff_base_s * (attempt + 1)
                log.debug(
                    "background_store_work gate busy for %s (attempt %d), "
                    "backing off %.2fs", name, attempt, backoff_s,
                )
                time.sleep(backoff_s)
                waited_total += max_wait_s + backoff_s
        if not acquired and not gate_broken:
            log.warning(
                "background_store_work gate not acquired for %s after %d "
                "retries (~%.1fs waited) -- deferring to the next cycle",
                name, max_retries, waited_total,
            )
        elif acquired:
            log.debug("background_store_work gate acquired for %s", name)
        yield acquired or gate_broken
    finally:
        if acquired:
            _BACKGROUND_STORE_WORK_GATE.release()


# Serializes the cache-MISS rebuild of the runtime graph across concurrent
# callers. The rebuild streams the whole corpus and spawns a child for community
# detection plus centrality; under a stale or absent cache several callers can
# fire at once and each run the full rebuild, spawning redundant children.
# A single-flight guard around the rebuild
# section collapses them to one: the first caller rebuilds and saves the
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


def _rgc_disk_mtime_ns(store: MemoryStore) -> "int | None":
    from iai_mcp import runtime_graph_cache

    try:
        return runtime_graph_cache._cache_path(store).stat().st_mtime_ns
    except (OSError, AttributeError):
        return None


def _warm_graph_bundle_if_valid(store: MemoryStore):
    """Return the last-built (graph, assignment, rich_club) when still valid.

    Every build registers a graph_sync_hook, so record writes land on the
    memoized graph LIVE — the bundle stays node-fresh between builds. Validity
    reuses exactly the staleness signals the design already blesses: the
    on-disk cache file is unchanged (every invalidate unlinks it, every rebuild
    rewrites it), the corpus is within the persisted-cache drift tolerance, and
    the freshness fuse has not tripped (dirty-counter delta and age within the
    same bounds the snapshot overlay enforces). Without this memo every recall
    re-streams the whole corpus: the shed node_payload makes the "warm" path a
    full scan.
    """
    from iai_mcp import runtime_graph_cache

    memo = getattr(store, "_warm_graph_bundle", None)
    if memo is None:
        return None
    bundle, cache_key, mtime_ns, dirty_at_build, built_mono = memo
    if mtime_ns != _rgc_disk_mtime_ns(store):
        return None
    # Same windowed key the persisted cache uses: a window-crossing write set
    # invalidates the memo exactly when it invalidates the disk cache.
    if cache_key != runtime_graph_cache._cache_key(store):
        return None
    dirty_delta = runtime_graph_cache.get_dirty_counter() - dirty_at_build
    if dirty_delta > runtime_graph_cache._fuse_dirty_threshold(store):
        return None
    if (time.monotonic() - built_mono) > runtime_graph_cache._FUSE_MAX_AGE_SECONDS:
        return None
    return bundle


def _memoize_graph_bundle(store: MemoryStore, bundle) -> None:
    from iai_mcp import runtime_graph_cache

    try:
        store._warm_graph_bundle = (
            bundle,
            runtime_graph_cache._cache_key(store),
            _rgc_disk_mtime_ns(store),
            runtime_graph_cache.get_dirty_counter(),
            time.monotonic(),
        )
    except (AttributeError, OSError, RuntimeError) as exc:
        log.warning("warm graph bundle memoization failed: %s", exc)


_REFRESH_COOLDOWN_SECONDS: float = 120.0


def _refresh_cooldown_s() -> float:
    import os

    raw = os.environ.get("IAI_MCP_GRAPH_REFRESH_COOLDOWN_S")
    if raw is None:
        return _REFRESH_COOLDOWN_SECONDS
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _REFRESH_COOLDOWN_SECONDS
    return val if val >= 0 else _REFRESH_COOLDOWN_SECONDS


def _refresh_graph_bundle_async(store: MemoryStore) -> "threading.Thread | None":
    """Single-flight background rebuild + re-memoization of the warm bundle.

    Rate-limited: a full-corpus refresh costs seconds of CPU, and ambient
    write churn re-keys the memo far more often than the graph meaningfully
    changes. Between refreshes the stale-while-revalidate serve stays correct
    (the sync hook keeps nodes live; recall reads the index). An EXPLICIT
    freshness demand is never throttled — invalidation unlinks the disk cache,
    which routes the next build INLINE, bypassing this refresher entirely.

    Returns the refresher thread (for deterministic joins in tests), or None
    when a refresh is already in flight or inside the cooldown window.
    """
    now_mono = time.monotonic()
    last = getattr(store, "_graph_refresh_last_mono", 0.0)
    if now_mono - last < _refresh_cooldown_s():
        return None
    latch = getattr(store, "_graph_refresh_inflight", None)
    if latch is None:
        latch = threading.Lock()
        store._graph_refresh_inflight = latch
    if not latch.acquire(blocking=False):
        return None
    store._graph_refresh_last_mono = now_mono

    def _refresh() -> None:
        try:
            with store.count_memo(), _RUNTIME_GRAPH_REBUILD_LOCK:
                if _warm_graph_bundle_if_valid(store) is None:
                    _memoize_graph_bundle(store, _build_runtime_graph_impl(store))
        except Exception:  # noqa: BLE001 -- refresh must never kill its host
            log.warning("background graph refresh failed", exc_info=True)
        finally:
            latch.release()

    t = threading.Thread(target=_refresh, name="graph-bundle-refresh", daemon=True)
    store._graph_refresh_thread = t
    t.start()
    return t


def _flush_buffered_edges_for_build(store: MemoryStore) -> None:
    """Flush-before-read barrier for graph builds: edge INSERTs buffer
    in-process (flushed by size or the daemon's periodic tick), so a table
    stream in a non-daemon process can otherwise miss edges this same process
    just wrote — the graph would silently diverge from the corpus."""
    try:
        from iai_mcp.store import flush_edge_buffer
        flush_edge_buffer(store)
    except Exception as exc:  # noqa: BLE001 -- a failed flush degrades to the pre-barrier lag, never a failed build
        log.debug("edge-buffer flush before graph build failed: %s", exc)


def _replay_runtime_edge(graph, row: dict) -> None:
    """Replay an edge only when both endpoints are active graph nodes."""
    src = UUID(row["src"])
    dst = UUID(row["dst"])
    if not graph.has_node(src) or not graph.has_node(dst):
        return
    graph.add_edge(
        src,
        dst,
        weight=float(row["weight"]),
        edge_type=row["edge_type"],
    )


def build_runtime_graph(store: MemoryStore):
    # Memoize the corpus COUNT(*) probes for the duration of this one build. The
    # cache-key derivation, the drift gate, and the impl each ask for the active
    # / pending / edges counts; on the lilli engine each filtered COUNT re-scans
    # every leaf page, so without the memo a single warm build pays a dozen
    # full-corpus scans. The scope is operation-local — torn down on return — so
    # no count is ever cached across a write.
    with store.count_memo():
        warm = _warm_graph_bundle_if_valid(store)
        if warm is not None:
            return warm

        # Stale-while-revalidate: background churn (consolidation cycles saving
        # the disk cache, window-crossing write bursts) constantly re-keys the
        # memo, and rebuilding INLINE would put a full corpus stream on the
        # awake read path every time — the exact cost this cache exists to
        # remove. A bundle inside the freshness fuse is served immediately
        # (its sync-hook has kept nodes live; new records stay index-findable
        # — the same bounded lag the persisted-cache drift gate blesses) while
        # a single-flight refresher rebuilds behind it. Two states refuse the
        # stale serve and force the inline rebuild below: a bundle past the
        # fuse age, and a MISSING disk cache — churn REWRITES the cache file,
        # only an explicit invalidation (invalidate / invalidate_at_root, e.g.
        # the pending-embedding wake sequence) UNLINKS it, and that caller has
        # demanded that the next build see the new rows.
        memo = getattr(store, "_warm_graph_bundle", None)
        if memo is not None and _rgc_disk_mtime_ns(store) is not None:
            from iai_mcp import runtime_graph_cache
            if (time.monotonic() - memo[4]) <= runtime_graph_cache._FUSE_MAX_AGE_SECONDS:
                _refresh_graph_bundle_async(store)
                return memo[0]

        # Common path: a warm cache that needs no rebuild reconstructs the graph
        # without spawning any child — run it lock-free so warm recall never
        # blocks on a peer's rebuild.
        if not _runtime_graph_rebuild_needed(store):
            bundle = _build_runtime_graph_impl(store)
            _memoize_graph_bundle(store, bundle)
            return bundle

        # Cache miss: single-flight the rebuild so concurrent callers don't each
        # spawn a redundant child fleet on the same graph. Double-checked — a
        # peer that held the lock may have rebuilt and saved a fresh cache while
        # this caller waited, in which case the re-probe passes and the impl
        # takes the light cache-hit branch, spawning no child of its own.
        # Otherwise this caller is the single-flight winner and performs the one
        # rebuild.
        with _RUNTIME_GRAPH_REBUILD_LOCK:
            warm = _warm_graph_bundle_if_valid(store)
            if warm is not None:
                return warm
            bundle = _build_runtime_graph_impl(store)
            _memoize_graph_bundle(store, bundle)
            return bundle


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
        assignment, rich_club, cached_node_payload, cached_max_degree, _ = cached

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
        ]
        # Stream only ACTIVE records — tombstoned (deleted) records are not graph
        # nodes. This matches the active-records predicate used everywhere else
        # (active_records_count, the read-only graph export, and recall), so the
        # node set built here equals active_records_count exactly. Aligning the
        # two is what lets the drift gate recognise a freshly-built cache as fresh
        # (otherwise payload_record_count is inflated by the tombstone count and
        # the cache is rejected as stale on every boot, forcing a cold rebuild).
        active_where = (
            "tombstoned_at IS NULL AND COALESCE(embedding_pending, 0) = 0"
        )
        _chunk_rows = _graph_stream_chunk_rows()
        _rows_since_yield = 0
        for row in store.iter_record_columns(
            stream_cols, batch_size=1024, where=active_where
        ):
            _rows_since_yield += 1
            if _rows_since_yield >= _chunk_rows:
                _yield_to_event_loop()
                _rows_since_yield = 0
            if int(row.get("embedding_pending") or 0) != 0:
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

    # Stream the edges table as batched row dicts over a read-only snapshot
    # connection instead of materializing it into one DataFrame — avoids a
    # full-table pandas round-trip under the shared connection lock. Row order
    # is irrelevant to the resulting graph (the edge set is order-insensitive).
    _flush_buffered_edges_for_build(store)
    edges_query = (
        store.db.open_table("edges")
        .search()
        .select(["src", "dst", "weight", "edge_type"])
    )
    _edge_chunk_rows = _graph_stream_chunk_rows()
    _edge_rows_since_yield = 0
    for batch in edges_query.to_batches(batch_size=2048):
        for row in batch.to_pylist():
            _edge_rows_since_yield += 1
            if _edge_rows_since_yield >= _edge_chunk_rows:
                _yield_to_event_loop()
                _edge_rows_since_yield = 0
            _replay_runtime_edge(graph, row)

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

    # Centrality has been resolved through exactly one of the branches above
    # (cached-fresh, cached-payload-nonzero, child map, child recompute,
    # last-good degrade, or neutral all-zero degrade). All converge here, so the
    # map carried by the graph is final for this build. Stamp it resolved once so
    # the recall path reads this map instead of recomputing betweenness per
    # recall. A legitimately all-zero (edgeless) map and a neutral-degrade map are
    # both RESOLVED — those are precisely the states a value-based guard would
    # mistake for "needs recompute" and loop on every recall.
    graph._centrality_resolved = True

    return graph, assignment, rich_club


def _parse_record_timestamp(raw: Any) -> "datetime | None":
    """Parse a records/edges `updated_at` cell (the store's own
    `str(datetime.now(timezone.utc))` or a driver-native datetime) into a
    tz-aware ``datetime``. Returns ``None`` on anything unparseable rather than
    raising -- a delta build treats an unparseable timestamp as "changed"
    (falls into the delta set) so it is never silently dropped.
    """
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo is not None else raw.replace(tzinfo=timezone.utc)
    if hasattr(raw, "to_pydatetime"):
        dt = raw.to_pydatetime()
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def build_runtime_graph_incremental(store: MemoryStore):
    """Delta-only runtime-graph rebuild: update just the records/edges changed
    since the last saved cache, instead of re-streaming the whole corpus.

    This is the chunked/incremental path the boot rebuild uses so a small
    drift never pays a whole-corpus rebuild even off the loop. It is additive
    to (never a replacement for) `build_runtime_graph`'s existing warm-cache
    behaviour: the drift-tolerance gate's "reuse the cached graph verbatim,
    tolerate bounded staleness" contract is untouched (recall stays correct via
    the index regardless of which graph-build path ran last).

    Falls back to a full rebuild (`build_runtime_graph`) whenever the delta
    cannot be safely computed: no cache to diff against, the cached
    node_payload was shed (production-scale caches always shed it -- the
    compact centrality survives but the per-node embedding/surface needed to
    patch in a delta does not), or the drift is large enough that streaming
    the delta would touch most of the corpus anyway (no cheaper than a full
    rebuild). When a delta IS applied, the result is topology-equivalent to a
    full rebuild over the same corpus state: every node changed after the
    cache's `saved_at` is refreshed from the live row, every node tombstoned
    since is removed, and the full edge set is re-applied (edges carry their
    own `updated_at` and are cheap relative to the decrypt-heavy record scan).
    """
    from iai_mcp import runtime_graph_cache
    from iai_mcp.graph import MemoryGraph

    with store.count_memo():
        cached = runtime_graph_cache.try_load(store)
        if cached is None:
            return build_runtime_graph(store)

        assignment, rich_club, cached_node_payload, cached_max_degree, _node_degrees = cached
        cache_results = runtime_graph_cache.try_load_cache_results(store)
        if cache_results is None:
            return build_runtime_graph(store)
        cached_centrality, cached_payload_record_count = cache_results

        # The delta path patches the per-node embedding/surface payload; a
        # shed payload (the production-scale norm) leaves nothing to patch, so
        # fall back to the full rebuild rather than serving a delta built on an
        # incomplete node set.
        if not cached_node_payload:
            return build_runtime_graph(store)

        records_count = store.active_records_count()
        if not _within_drift_tolerance(cached_payload_record_count, records_count):
            # Large/mismatched drift: a delta stream would touch most of the
            # corpus anyway, so the full rebuild is no more expensive and stays
            # the single source of truth for a cache this stale.
            return build_runtime_graph(store)

        cache_saved_at: "datetime | None" = None
        try:
            raw_cache_data = runtime_graph_cache._load_and_decrypt_cache(store)
            if raw_cache_data is not None:
                cache_saved_at = _parse_record_timestamp(raw_cache_data.get("saved_at"))
        except Exception:  # noqa: BLE001 -- a saved_at read failure forces a full delta scan
            cache_saved_at = None

        graph = MemoryGraph()
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
        node_payload_for_cache = dict(cached_node_payload)
        cached_ids = set(node_payload_for_cache.keys())

        # ---- Delta 1: records changed (added/updated/tombstoned) since the
        # cache was saved. When `cache_saved_at` cannot be resolved, every row
        # is treated as "changed" -- degrading to the same corpus size as a
        # full stream, but still cheaper than a full rebuild (no community
        # detection / betweenness child is spawned).
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
            "updated_at",
            "tombstoned_at",
        ]
        chunk_rows = _graph_stream_chunk_rows()
        rows_since_yield = 0
        seen_active_ids: set[str] = set()
        decrypt_fail_events = 0
        for row in store.iter_record_columns(stream_cols, batch_size=1024):
            rows_since_yield += 1
            if rows_since_yield >= chunk_rows:
                _yield_to_event_loop()
                rows_since_yield = 0

            rid_s = str(row["id"])
            tombstoned = row.get("tombstoned_at") is not None
            pending = int(row.get("embedding_pending") or 0) != 0

            if tombstoned or pending:
                if rid_s in node_payload_for_cache:
                    graph.remove_node(rid_s)
                    node_payload_for_cache.pop(rid_s, None)
                continue

            seen_active_ids.add(rid_s)

            row_updated_at = _parse_record_timestamp(row.get("updated_at"))
            is_new = rid_s not in cached_ids
            is_changed = (
                is_new
                or cache_saved_at is None
                or row_updated_at is None
                or row_updated_at > cache_saved_at
            )
            if not is_changed:
                continue

            rid = UUID(rid_s)
            _comm_raw = row.get("community_id")
            community_id = UUID(_comm_raw) if _comm_raw else None
            _emb_raw = row.get("embedding")
            embedding = (
                list(_emb_raw) if _emb_raw is not None else [0.0] * EMBED_DIM
            )
            literal_raw = row.get("literal_surface") or ""
            try:
                from iai_mcp.crypto import is_encrypted
                if is_encrypted(literal_raw):
                    literal_raw = store._decrypt_for_record(rid, literal_raw)
            except Exception:  # noqa: BLE001 -- InvalidTag / OSError / ValueError / RuntimeError
                decrypt_fail_events += 1
                continue

            tier = row.get("tier") or "episodic"
            centrality = float(row.get("centrality") or 0.0)
            pinned = bool(row.get("pinned") or False)
            tags_raw = row.get("tags_json") or "[]"
            try:
                import json as _json
                tags_list = (
                    _json.loads(tags_raw)
                    if isinstance(tags_raw, str)
                    else list(tags_raw)
                )
                if not isinstance(tags_list, list):
                    tags_list = []
            except (ValueError, TypeError):
                tags_list = []
            language = str(row.get("language") or "en")

            payload = {
                "embedding": list(embedding),
                "surface": str(literal_raw),
                "centrality": centrality,
                "tier": str(tier),
                "pinned": pinned,
                "tags": list(tags_list),
                "language": language,
            }
            graph.add_node(rid, community_id=community_id, embedding=embedding)
            graph.set_node_payload(rid, payload)
            node_payload_for_cache[rid_s] = payload

        if decrypt_fail_events > 0:
            log.warning(
                "graph_build_incremental_decrypt_failed_summary",
                extra={"total_skip_events": decrypt_fail_events},
            )

        # A cached node absent from this live scan (should not happen under a
        # correct active-records predicate, but the corpus can shrink between
        # the cache-key probe and this stream) is dropped defensively so the
        # graph never carries a phantom node the store no longer has.
        for stale_id in cached_ids - seen_active_ids:
            if stale_id in node_payload_for_cache:
                graph.remove_node(stale_id)
                node_payload_for_cache.pop(stale_id, None)

        # ---- Delta 2: re-apply the full edge set. Edges are cheap relative to
        # the decrypt-heavy record scan (no per-row decrypt), so re-streaming
        # them in full keeps the edge set exactly correct without needing a
        # second timestamp-diff pass; still chunked/yielding so the corpus
        # stream never dominates a single uninterrupted slice.
        _flush_buffered_edges_for_build(store)
        edges_query = (
            store.db.open_table("edges")
            .search()
            .select(["src", "dst", "weight", "edge_type"])
        )
        edge_rows_since_yield = 0
        edge_chunk_rows = _graph_stream_chunk_rows()
        for batch in edges_query.to_batches(batch_size=2048):
            for row in batch.to_pylist():
                edge_rows_since_yield += 1
                if edge_rows_since_yield >= edge_chunk_rows:
                    _yield_to_event_loop()
                    edge_rows_since_yield = 0
                _replay_runtime_edge(graph, row)

        try:
            deg_values = [d for _, d in graph.degrees()]
            max_degree = max(deg_values) if deg_values else 0
        except (ValueError, RuntimeError, AttributeError):
            max_degree = cached_max_degree
        if max_degree == 0 and cached_max_degree > 0:
            max_degree = cached_max_degree
        graph._max_degree = int(max_degree)

        # Centrality: apply the cached map to every surviving node; a delta
        # node absent from the cached map (newly added since the cache was
        # built) gets a neutral 0.0 until the next full rebuild recomputes
        # betweenness -- the same bounded-staleness degrade the warm cache-hit
        # path already tolerates (seeds rank 0.6*cos + 0.4*centrality, so a
        # neutral term degrades to cosine-led ranking, never a crash).
        resolved_centrality: dict = {}
        for nid_s in node_payload_for_cache:
            try:
                nid = UUID(nid_s)
            except (ValueError, TypeError):
                continue
            cval = float(cached_centrality.get(nid, 0.0)) if cached_centrality else 0.0
            graph.set_node_centrality(nid, cval)
            resolved_centrality[nid] = cval
            node_payload_for_cache[nid_s]["centrality"] = cval

        graph._centrality_resolved = True

        runtime_graph_cache.save(
            store, assignment, rich_club,
            node_payload=node_payload_for_cache,
            max_degree=int(getattr(graph, "_max_degree", 0) or 0),
        )

        try:
            store.register_graph_sync_hook(_make_graph_sync_hook(graph))
        except (AttributeError, TypeError, RuntimeError) as exc:
            log.warning("graph_sync_hook registration failed: %s", exc)

        return graph, assignment, rich_club
