from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable
from enum import Enum
from itertools import combinations
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from iai_mcp.aaak import enforce_language_tagged, generate_aaak_index
from iai_mcp.events import write_event
from iai_mcp.guard import BudgetLedger, RateLimitLedger, should_call_llm
from iai_mcp.store import EDGES_TABLE, MemoryStore, _uuid_literal
from iai_mcp.types import MemoryRecord


class SleepMode(str, Enum):

    ACTIVITY = "activity"
    TIME = "time"
    MANUAL = "manual"


@dataclass
class SleepConfig:

    mode: SleepMode = SleepMode.ACTIVITY
    quiet_window: tuple[int, int] = (22, 6)
    require_idle_minutes: int = 30
    max_defer_hours: int = 48
    on_user_resume: str = "defer_remaining"
    light_on_exit: bool = True
    llm_enabled: bool = False
    llm_tier: int = 1


DECAY_EPSILON: float = 0.01
DECAY_GRACE_DAYS: int = 90
DECAY_BASE: float = 0.9
FSRS_STABILITY_BOOST: float = 0.2
CLUSTER_MIN_SIZE: int = 3
HEAVY_LTP_DELTA: float = 0.05
#: Edge kinds that carry real record-to-record connectivity. Consolidation
#: clusters over ALL of them: waking co-activation (hebbian), encoding-time
#: similarity residue (pattern_separation_seed), and sleep replay. Clustering
#: over hebbian alone starves the minter — a store whose hebbian edges are
#: only canonical self-loops yields zero clusters forever.
CLUSTER_EDGE_TYPES: tuple[str, ...] = (
    "hebbian",
    "pattern_separation_seed",
    "hebbian_cluster_replay",
)
CLUSTER_EDGE_MIN_WEIGHT: float = 0.05
#: Per-cycle mint cap: a first run over a mature corpus finds thousands of
#: communities; minting them all in one cycle floods the semantic tier and
#: the cycle budget. Largest communities first; the rest mature next cycles.
MAX_SUMMARIES_PER_CYCLE: int = 50
#: A community already covered >= this fraction by an existing summary is
#: skipped unless it regrew past CLUSTER_REGROWTH_RATIO of the covered size.
CLUSTER_COVERAGE_SKIP: float = 0.6
CLUSTER_REGROWTH_RATIO: float = 1.5
#: Clusters past this size are stamped with a community id but NOT
#: summarized — a five-snippet summary of hundreds of records is noise, and
#: an unpartitionable (flat-fallback) giant must not become a mega-summary
#: that then coverage-blocks finer minting inside it forever.
CLUSTER_MAX_SUMMARY_MEMBERS: int = 200


def should_run_heavy(
    now_utc: datetime,
    last_activity_utc: datetime,
    config: SleepConfig,
    tz: ZoneInfo,
) -> tuple[bool, str]:
    idle_minutes = (now_utc - last_activity_utc).total_seconds() / 60.0

    if idle_minutes >= config.max_defer_hours * 60:
        return True, f"max_defer_hours ({config.max_defer_hours}h) exceeded"

    if config.mode == SleepMode.MANUAL:
        return False, "manual-only mode"

    if config.mode == SleepMode.TIME:
        local = now_utc.astimezone(tz)
        ok = local.hour == 3
        return ok, f"TIME mode, local hour={local.hour}"

    if idle_minutes < config.require_idle_minutes:
        return False, f"idle < {config.require_idle_minutes}min"

    local = now_utc.astimezone(tz)
    start_h, end_h = config.quiet_window
    if start_h > end_h:
        in_window = (local.hour >= start_h) or (local.hour < end_h)
    else:
        in_window = start_h <= local.hour < end_h
    if not in_window:
        return False, (
            f"outside quiet window {config.quiet_window}, "
            f"local hour={local.hour}"
        )
    return True, ""


def _apply_fsrs(record: MemoryRecord, now: datetime) -> MemoryRecord:
    if record.never_decay:
        return record
    record.stability = min(1.0, record.stability + FSRS_STABILITY_BOOST)
    record.last_reviewed = now
    return record


#: Edges applied per transaction. One commit (one write-generation bump, one
#: index publication at most) per chunk instead of per edge; small enough
#: that a chunk lands in ~a second and the step yields to a foreground read
#: within one chunk.
DECAY_CHUNK_EDGES = 400


def _decay_edges(
    store: MemoryStore, epsilon: float = DECAY_EPSILON,
    plasticity_gain: float = 1.0,
    should_defer: "Callable[[int], bool] | None" = None,
) -> dict:
    tbl = store.db.open_table(EDGES_TABLE)
    df = tbl.to_pandas()
    if df.empty:
        return {"decayed": 0, "pruned": 0, "interrupted": False}

    now = datetime.now(timezone.utc)

    decayable_kinds = (
        "hebbian",
        "hebbian_structure",
        "pattern_separation_seed",
        "hebbian_cluster_replay",
    )
    # Plan first (pure compute off every lock), apply in chunked transactions
    # second. Applied edges get updated_at=now, so a deferred run resumes for
    # free: the grace-window check skips them on the next pass — the step is
    # restart-idempotent without a persisted cursor.
    updates: list[tuple[str, str, str, float]] = []
    prunes: list[tuple[str, str, str]] = []
    hebbian = df[df["edge_type"].isin(decayable_kinds)]
    for _, row in hebbian.iterrows():
        try:
            last = row["updated_at"]
            if last is None:
                continue
            try:
                py = last.to_pydatetime() if hasattr(last, "to_pydatetime") else last
            except (TypeError, ValueError, AttributeError):
                py = last
            if isinstance(py, str):
                try:
                    py = datetime.fromisoformat(py.replace("Z", "+00:00"))
                except (TypeError, ValueError):
                    continue
            if not isinstance(py, datetime):
                continue
            if py.tzinfo is None:
                py = py.replace(tzinfo=timezone.utc)

            days = (now - py).total_seconds() / 86400.0
            if days <= DECAY_GRACE_DAYS:
                continue

            new_weight = float(row["weight"]) * (
                DECAY_BASE ** ((days - DECAY_GRACE_DAYS) * plasticity_gain)
            )

            src_lit = _uuid_literal(row["src"])
            dst_lit = _uuid_literal(row["dst"])
            edge_kind = str(row["edge_type"])
            if edge_kind not in decayable_kinds:
                continue
            if new_weight < epsilon:
                prunes.append((src_lit, dst_lit, edge_kind))
            else:
                updates.append((src_lit, dst_lit, edge_kind, float(new_weight)))
        except ValueError:
            continue

    from iai_mcp.hippo import _txn

    decayed = 0
    pruned = 0
    work: list[tuple[str, tuple]] = [("update", u) for u in updates]
    work.extend(("prune", p) for p in prunes)
    for chunk_no, chunk_start in enumerate(range(0, len(work), DECAY_CHUNK_EDGES)):
        if should_defer is not None and should_defer(chunk_no):
            return {"decayed": decayed, "pruned": pruned, "interrupted": True}
        chunk = work[chunk_start:chunk_start + DECAY_CHUNK_EDGES]
        with store.db._conn_lock:
            with _txn(store.db._conn):
                for kind, item in chunk:
                    if kind == "prune":
                        src_lit, dst_lit, edge_kind = item
                        tbl.delete(
                            f"src = '{src_lit}' AND dst = '{dst_lit}' "
                            f"AND edge_type = '{edge_kind}'"
                        )
                        pruned += 1
                    else:
                        src_lit, dst_lit, edge_kind, new_weight = item
                        tbl.update(
                            where=(
                                f"src = '{src_lit}' AND dst = '{dst_lit}' "
                                f"AND edge_type = '{edge_kind}'"
                            ),
                            values={
                                "weight": new_weight,
                                "updated_at": now,
                            },
                        )
                        decayed += 1

    return {"decayed": decayed, "pruned": pruned, "interrupted": False}


def run_light_consolidation(
    store: MemoryStore, session_id: str,
) -> dict:
    now = datetime.now(timezone.utc)
    records = store.all_records()
    fsrs_ticked = 0

    for r in records:
        if r.never_decay:
            continue
        if not r.provenance:
            continue
        last_prov = r.provenance[-1]
        try:
            ts_str = last_prov.get("ts", "")
            prov_ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if prov_ts.tzinfo is None:
                prov_ts = prov_ts.replace(tzinfo=timezone.utc)
            if (now - prov_ts).total_seconds() < 3600:
                _apply_fsrs(r, now)
                store.update_record(r)
                fsrs_ticked += 1
        except (TypeError, ValueError, KeyError, AttributeError):
            continue

    write_event(
        store,
        kind="cls_consolidation_run",
        data={
            "mode": "light",
            "fsrs_ticked": fsrs_ticked,
            "record_count": len(records),
        },
        severity="info",
        session_id=session_id,
    )
    return {
        "mode": "light",
        "fsrs_ticked": fsrs_ticked,
        "cooccurrence_updates": 0,
    }


def _connective_edges(edges_df) -> list[tuple[UUID, UUID, float]]:
    if edges_df.empty:
        return []
    mask = (
        edges_df["edge_type"].isin(CLUSTER_EDGE_TYPES)
        & (edges_df["src"] != edges_df["dst"])
        & (edges_df["weight"] >= CLUSTER_EDGE_MIN_WEIGHT)
    )
    out: list[tuple[UUID, UUID, float]] = []
    for _, row in edges_df[mask].iterrows():
        try:
            out.append((UUID(row["src"]), UUID(row["dst"]), float(row["weight"])))
        except (ValueError, TypeError):
            continue
    return out


#: A connected component larger than this is partitioned with community
#: detection instead of being consolidated whole — one transitive-similarity
#: mega-component must not become a single mega-summary.
CLUSTER_PARTITION_THRESHOLD: int = 200


def _connected_components(
    kept: "list[tuple[UUID, UUID, float, str]]",
) -> "list[list[UUID]]":
    adj: dict[UUID, set[UUID]] = {}
    for src, dst, _w, _t in kept:
        adj.setdefault(src, set()).add(dst)
        adj.setdefault(dst, set()).add(src)
    visited: set[UUID] = set()
    components: list[list[UUID]] = []
    for node in adj:
        if node in visited:
            continue
        stack = [node]
        component: list[UUID] = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            component.append(cur)
            stack.extend(n for n in adj[cur] if n not in visited)
        components.append(component)
    return components


def _partition_giant_component(
    component: "list[UUID]",
    kept: "list[tuple[UUID, UUID, float, str]]",
    records_by_id: "dict[UUID, MemoryRecord]",
) -> "list[list[UUID]]":
    from iai_mcp.community import detect_communities
    from iai_mcp.graph import MemoryGraph

    comp_set = set(component)
    graph = MemoryGraph()
    graph.clear_and_rebuild(
        (
            (rid, None, list(records_by_id[rid].embedding or []), {})
            for rid in component
        ),
        (e for e in kept if e[0] in comp_set and e[1] in comp_set),
    )
    assignment = detect_communities(graph, prior=None, prior_mode="cold")
    members: dict = {}
    for node, community in assignment.node_to_community.items():
        rid = node if isinstance(node, UUID) else UUID(str(node))
        members.setdefault(community, []).append(rid)
    return list(members.values())


def _build_consolidation_clusters(
    store: MemoryStore,
    records_by_id: "dict[UUID, MemoryRecord]",
    edges_df=None,
) -> list[list[UUID]]:
    """Cluster the connective graph: connected components are the clusters;
    a component past CLUSTER_PARTITION_THRESHOLD is split by community
    detection so transitive similarity chains cannot merge everything."""
    if edges_df is None:
        edges_df = store.db.open_table(EDGES_TABLE).to_pandas()
    edges = _connective_edges(edges_df)
    if not edges:
        return []

    episodic = {
        rid for rid, rec in records_by_id.items() if rec.tier == "episodic"
    }
    kept: list[tuple[UUID, UUID, float, str]] = [
        (src, dst, weight, "hebbian")
        for src, dst, weight in edges
        if src in episodic and dst in episodic
    ]
    if not kept:
        return []

    clusters: list[list[UUID]] = []
    for component in _connected_components(kept):
        if len(component) < CLUSTER_MIN_SIZE:
            continue
        if len(component) <= CLUSTER_PARTITION_THRESHOLD:
            clusters.append(component)
            continue
        try:
            parts = _partition_giant_component(component, kept, records_by_id)
        except Exception:  # noqa: BLE001 -- fall back to the whole component
            parts = [component]
        clusters.extend(p for p in parts if len(p) >= CLUSTER_MIN_SIZE)
    clusters.sort(key=lambda m: (-len(m), str(min(m))))
    return clusters


def _live_records_by_id(store: MemoryStore) -> "dict[UUID, MemoryRecord]":
    """Live (non-tombstoned) records only. MemoryRecord does not carry the
    tombstone column, so liveness comes from a narrow id query."""
    live_ids: set[str] = set()
    try:
        with store.db._conn_lock:
            rows = store.db._conn.execute(
                "SELECT id FROM records WHERE tombstoned_at IS NULL"
            ).fetchall()
        live_ids = {str(row["id"]) for row in rows}
    except Exception:  # noqa: BLE001 -- fall back to the unfiltered view
        return {r.id: r for r in store.all_records()}
    return {
        r.id: r for r in store.all_records() if str(r.id) in live_ids
    }


def _build_hebbian_clusters(store: MemoryStore) -> list[list[UUID]]:
    records_by_id = _live_records_by_id(store)
    return _build_consolidation_clusters(store, records_by_id)


def _tier0_schema_surfacing(store: MemoryStore) -> list[dict]:
    tag_counts: dict[str, int] = {}
    record_count = 0
    for row in store.iter_record_columns(
        ["tags_json", "tombstoned_at"], batch_size=1024,
    ):
        if row.get("tombstoned_at") is not None:
            continue
        record_count += 1
        tags_raw = row.get("tags_json") or "[]"
        try:
            tags = json.loads(tags_raw) if tags_raw else []
        except (TypeError, json.JSONDecodeError):
            tags = []
        for t in tags:
            # idem: tags are per-record idempotency hashes and entity:
            # tags are per-record name anchors; a "pattern" over either
            # is noise that only surfaces when duplicates/tombstones
            # inflate its count.
            if t.startswith(("raw:", "domain:", "idem:", "entity:")):
                continue
            tag_counts[t] = tag_counts.get(t, 0) + 1
    if record_count < CLUSTER_MIN_SIZE:
        return []
    candidates: list[dict] = []
    for tag, count in tag_counts.items():
        if count >= 3:
            candidates.append(
                {
                    "pattern": f"tag:{tag}",
                    "confidence": min(1.0, count / 10.0),
                    "evidence_count": count,
                }
            )
    return candidates


def _create_semantic_summary(
    store: MemoryStore,
    cluster: list[MemoryRecord],
    summary_text: str,
    language: str,
) -> "tuple[UUID, bool]":
    from iai_mcp.embed import embedder_for_store

    emb = embedder_for_store(store).embed(summary_text)
    now = datetime.now(timezone.utc)
    summary_id = uuid4()
    summary = MemoryRecord(
        id=summary_id,
        tier="semantic",
        literal_surface=summary_text,
        aaak_index="",
        embedding=emb,
        community_id=None,
        centrality=0.0,
        detail_level=3,
        pinned=False,
        stability=0.5,
        difficulty=0.3,
        last_reviewed=now,
        never_decay=True,
        never_merge=False,
        provenance=[
            {
                "ts": now.isoformat(),
                "cue": "cls_consolidation",
                "session_id": "system",
            }
        ],
        created_at=now,
        updated_at=now,
        tags=["semantic", "cls_summary"],
        language=language,
    )
    enforce_language_tagged(summary)
    summary.aaak_index = generate_aaak_index(summary)
    store.insert(summary)

    # insert may dedup-fold the summary into an existing near-identical
    # record, rewriting summary.id to the survivor — edges must bind to the
    # id that actually lives in the table, never the pre-insert uuid.
    pairs = [(summary.id, source.id) for source in cluster]
    if pairs:
        store.boost_edges(
            pairs,
            edge_type="consolidated_from",
            delta=1.0,
        )
        # Coverage edges are the summary's provenance ledger — they must
        # never sit in the crash-vulnerable edge buffer waiting for a
        # threshold flush.
        from iai_mcp.store import flush_edge_buffer

        flush_edge_buffer(store)
    return summary.id, summary.id != summary_id


def _persist_tier1_schemas(
    store: MemoryStore,
    budget: "BudgetLedger",
    rate: "RateLimitLedger",
    llm_enabled: bool,
) -> "tuple[list, int]":
    persisted = 0
    candidates: list = []
    try:
        from iai_mcp.lilli.cycle.schema import build_pattern_keeper_map
        from iai_mcp.schema import (
            induce_schemas_tier1,
            persist_schema,
        )

        candidates = induce_schemas_tier1(
            store, budget=budget, rate=rate,
            llm_enabled=llm_enabled,
        )
        # One keeper scan for the whole candidate loop — per-candidate corpus
        # rescans starve every other store consumer for the induction run.
        keeper_map = build_pattern_keeper_map(store) if candidates else {}
        for cand in candidates:
            if cand.status == "auto":
                kid = persist_schema(store, cand, keeper_map=keeper_map)
                keeper_map.setdefault(f"pattern:{cand.pattern}", kid)
                persisted += 1
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        write_event(
            store,
            kind="schema_induction_run",
            data={"error": str(exc), "status": "failed"},
            severity="warning",
            session_id="system",
        )
    return candidates, persisted


def _existing_summary_members(
    edges_df, semantic_ids: "set[UUID]",
) -> "dict[UUID, set[UUID]]":
    """summary_id -> source-record ids it consolidated. Edge writes canonicalize
    pair order, so the summary side is identified by tier, not by src position."""
    out: dict[UUID, set[UUID]] = {}
    if edges_df.empty:
        return out
    rows = edges_df[edges_df["edge_type"] == "consolidated_from"]
    for _, row in rows.iterrows():
        try:
            a, b = UUID(row["src"]), UUID(row["dst"])
        except (ValueError, TypeError):
            continue
        if a in semantic_ids and b not in semantic_ids:
            out.setdefault(a, set()).add(b)
        elif b in semantic_ids and a not in semantic_ids:
            out.setdefault(b, set()).add(a)
    return out


def _cluster_already_summarized(
    member_set: "set[UUID]",
    summaries: "dict[UUID, set[UUID]]",
) -> bool:
    """A community mostly covered by an existing summary is re-minted only
    after it regrows past CLUSTER_REGROWTH_RATIO of the covered size —
    otherwise every cycle would re-summarize every stable cluster."""
    if not member_set:
        return True
    for covered in summaries.values():
        overlap = len(member_set & covered)
        if not overlap:
            continue
        if overlap / len(member_set) >= CLUSTER_COVERAGE_SKIP:
            if len(member_set) < CLUSTER_REGROWTH_RATIO * len(covered):
                return True
    return False


def _persist_community_assignment(
    store: MemoryStore,
    clusters: "list[list[UUID]]",
    records_by_id: "dict[UUID, MemoryRecord]",
) -> int:
    """Stamp each clustered record with its community id (id = min member,
    stable across cycles for an unchanged community). Partial-column
    merge_insert touches ONLY community_id + aaak_index — the stored index
    embeds the room, so it must be regenerated in the same write or it
    freezes at R:unknown from mint. Returns a read-back-verified count:
    writes are sample-checked against the table, and a failed sample
    reports zero rather than claiming the attempt as fact."""
    from iai_mcp.aaak import generate_aaak_index

    updates: list[dict] = []
    for member_ids in clusters:
        community_id = str(min(member_ids))
        for rid in member_ids:
            rec = records_by_id.get(rid)
            if rec is None:
                continue
            if str(rec.community_id or "") != community_id:
                rec.community_id = community_id
                updates.append({
                    "id": str(rid),
                    "community_id": community_id,
                    "aaak_index": generate_aaak_index(rec),
                })
    if not updates:
        return 0
    tbl = store.db.open_table("records")
    tbl.merge_insert(["id"]).when_matched_update_all().execute(updates)
    sample = updates[:: max(1, len(updates) // 25)][:25]
    try:
        with store.db._conn_lock:
            for u in sample:
                row = store.db._conn.execute(
                    "SELECT community_id FROM records WHERE id = ?",
                    (u["id"],),
                ).fetchone()
                if row is None or str(row["community_id"] or "") != u["community_id"]:
                    return 0
    except Exception:  # noqa: BLE001 -- verification failure must not crash the pass
        return 0
    return len(updates)


def _process_cluster_summaries(store: MemoryStore) -> int:
    from itertools import islice

    from iai_mcp.lilli.cycle.sleep_pipeline import MAX_PAIRS_PER_CLUSTER
    from iai_mcp.store._buffers import flush_edge_buffer, flush_record_buffer

    # Read-your-writes: the coverage dedup reads consolidated_from edges from
    # the table, so buffered writes from a prior pass must land first.
    flush_record_buffer(store)
    flush_edge_buffer(store)
    edges_df = store.db.open_table(EDGES_TABLE).to_pandas()
    # Tombstoned records must not cluster, must not seed summary text, and a
    # tombstoned summary must not coverage-block re-minting of its cluster.
    records_by_id = _live_records_by_id(store)
    clusters = _build_consolidation_clusters(store, records_by_id, edges_df)
    if not clusters:
        return 0
    semantic_ids = {
        rid for rid, rec in records_by_id.items() if rec.tier == "semantic"
    }
    summaries = _existing_summary_members(edges_df, semantic_ids)
    try:
        stamped = _persist_community_assignment(store, clusters, records_by_id)
    except Exception:  # noqa: BLE001 -- stamping is auxiliary to minting
        stamped = 0
    summaries_created = 0
    summaries_folded = 0
    # Accumulate every cluster's intra-cluster pairs and potentiate them in ONE
    # boost_edges call after the loop. Issuing a separate boost_edges per cluster
    # re-materializes and re-scans the entire edges table once per cluster, which
    # grinds for minutes at scale; a single batched write does it once.
    all_pairs: list[tuple[UUID, UUID]] = []
    skipped_oversize = 0
    for cluster_ids in clusters:
        if summaries_created + summaries_folded >= MAX_SUMMARIES_PER_CYCLE:
            break
        cluster_recs = [records_by_id[i] for i in cluster_ids if i in records_by_id]
        if len(cluster_recs) < CLUSTER_MIN_SIZE:
            continue
        if len(cluster_recs) > CLUSTER_MAX_SUMMARY_MEMBERS:
            skipped_oversize += 1
            continue
        if _cluster_already_summarized({r.id for r in cluster_recs}, summaries):
            continue
        cluster_recs.sort(key=lambda r: r.updated_at or r.created_at, reverse=True)
        langs = [r.language for r in cluster_recs if r.language]
        dom_lang = max(set(langs), key=langs.count) if langs else "en"
        summary_text = (
            f"Cluster summary ({len(cluster_recs)} records, lang={dom_lang}): "
            + "; ".join(r.literal_surface[:80] for r in cluster_recs[:5])
        )
        summary_id, folded = _create_semantic_summary(
            store, cluster_recs, summary_text, dom_lang
        )
        summaries[summary_id] = {r.id for r in cluster_recs}
        # A fold means the knowledge already existed — the survivor gained
        # this cluster's coverage, but no new semantic record was born, and
        # telemetry must not claim one.
        if folded:
            summaries_folded += 1
        else:
            summaries_created += 1

        # Cap the per-cluster pairing. combinations over the nodes of a connected
        # component is O(k^2); a single large community would otherwise expand
        # into millions of pairs and exhaust memory. The cap bounds the work per
        # cluster while still potentiating a representative sample of its edges.
        cluster_pairs = list(
            islice(combinations(cluster_ids, 2), MAX_PAIRS_PER_CLUSTER)
        )
        all_pairs.extend(cluster_pairs)

    if all_pairs:
        store.boost_edges(
            all_pairs,
            delta=HEAVY_LTP_DELTA,
            edge_type="hebbian",
        )
    write_event(
        store,
        kind="cluster_summary_pass",
        data={
            "clusters_found": len(clusters),
            "summaries_created": summaries_created,
            "summaries_folded_into_existing": summaries_folded,
            "community_ids_stamped": stamped,
            "oversize_clusters_skipped": skipped_oversize,
        },
        severity="info",
        session_id="system",
    )
    return summaries_created


def _emit_cls_consolidation_run(
    store: MemoryStore,
    session_id: str,
    *,
    summaries_created: int,
    decay_result: dict,
    schema_candidates: int,
    schemas_induced: int,
    tier: str = "tier0",
    tier_eligible: str = "tier0",
    batch_submitted: bool = False,
) -> None:
    write_event(
        store,
        kind="cls_consolidation_run",
        data={
            "mode": "heavy",
            "tier": tier,
            "tier_eligible": tier_eligible,
            "summaries_created": summaries_created,
            "decay_result": decay_result,
            "schema_candidates": schema_candidates,
            "schemas_induced": schemas_induced,
            "batch_submitted": batch_submitted,
        },
        severity="info",
        session_id=session_id,
    )


def run_heavy_consolidation(
    store: MemoryStore,
    session_id: str,
    config: SleepConfig,
    budget: BudgetLedger,
    rate: RateLimitLedger,
    has_api_key: bool = False,
) -> dict:
    now = datetime.now(timezone.utc)

    decay_result = _decay_edges(store)

    llm_ok, _llm_reason = should_call_llm(
        budget=budget,
        rate=rate,
        llm_enabled=config.llm_enabled,
        has_api_key=has_api_key,
    )
    tier = "tier1" if llm_ok else "tier0"
    effective_tier = "tier0"
    batch_submitted = False

    summaries_created = _process_cluster_summaries(store)

    schemas = _tier0_schema_surfacing(store)

    _schema_candidates, schemas_induced = _persist_tier1_schemas(
        store, budget, rate, config.llm_enabled,
    )

    _emit_cls_consolidation_run(
        store,
        session_id,
        summaries_created=summaries_created,
        decay_result=decay_result,
        schema_candidates=len(schemas),
        schemas_induced=schemas_induced,
        tier=effective_tier,
        tier_eligible=tier,
        batch_submitted=batch_submitted,
    )

    return {
        "mode": "heavy",
        "tier": effective_tier,
        "summaries_created": summaries_created,
        "decay_result": decay_result,
        "schema_candidates": schemas,
        "schemas_induced": schemas_induced,
    }
