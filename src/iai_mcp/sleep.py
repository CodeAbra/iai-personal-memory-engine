from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
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


def _decay_edges(
    store: MemoryStore, epsilon: float = DECAY_EPSILON,
    plasticity_gain: float = 1.0,
) -> dict:
    tbl = store.db.open_table(EDGES_TABLE)
    df = tbl.to_pandas()
    if df.empty:
        return {"decayed": 0, "pruned": 0}

    now = datetime.now(timezone.utc)
    decayed = 0
    pruned = 0

    decayable_kinds = (
        "hebbian",
        "hebbian_structure",
        "pattern_separation_seed",
        "hebbian_cluster_replay",
    )
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

            new_weight = float(row["weight"]) * (DECAY_BASE ** ((days - DECAY_GRACE_DAYS) * plasticity_gain))

            src_lit = _uuid_literal(row["src"])
            dst_lit = _uuid_literal(row["dst"])
            edge_kind = str(row["edge_type"])
            if edge_kind not in decayable_kinds:
                continue
            if new_weight < epsilon:
                tbl.delete(
                    f"src = '{src_lit}' AND dst = '{dst_lit}' "
                    f"AND edge_type = '{edge_kind}'"
                )
                pruned += 1
            else:
                tbl.update(
                    where=(
                        f"src = '{src_lit}' AND dst = '{dst_lit}' "
                        f"AND edge_type = '{edge_kind}'"
                    ),
                    values={
                        "weight": float(new_weight),
                        "updated_at": now,
                    },
                )
                decayed += 1
        except ValueError:
            continue

    return {"decayed": decayed, "pruned": pruned}


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


def _build_hebbian_clusters(store: MemoryStore) -> list[list[UUID]]:
    edges_df = store.db.open_table(EDGES_TABLE).to_pandas()
    if edges_df.empty:
        return []
    hebbian = edges_df[edges_df["edge_type"] == "hebbian"]
    if hebbian.empty:
        return []

    adj: dict[UUID, set[UUID]] = {}
    for _, row in hebbian.iterrows():
        src = UUID(row["src"])
        dst = UUID(row["dst"])
        adj.setdefault(src, set()).add(dst)
        adj.setdefault(dst, set()).add(src)

    visited: set[UUID] = set()
    clusters: list[list[UUID]] = []
    for node in list(adj.keys()):
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
            for neigh in adj.get(cur, set()):
                if neigh not in visited:
                    stack.append(neigh)
        if len(component) >= CLUSTER_MIN_SIZE:
            clusters.append(component)
    return clusters


def _tier0_schema_surfacing(store: MemoryStore) -> list[dict]:
    tag_counts: dict[str, int] = {}
    record_count = 0
    for row in store.iter_record_columns(["tags_json"], batch_size=1024):
        record_count += 1
        tags_raw = row.get("tags_json") or "[]"
        try:
            tags = json.loads(tags_raw) if tags_raw else []
        except (TypeError, json.JSONDecodeError):
            tags = []
        for t in tags:
            if t.startswith("raw:") or t.startswith("domain:"):
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
) -> UUID:
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

    pairs = [(summary_id, source.id) for source in cluster]
    if pairs:
        store.boost_edges(
            pairs,
            edge_type="consolidated_from",
            delta=1.0,
        )
    return summary_id


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


def _process_cluster_summaries(store: MemoryStore) -> int:
    from itertools import islice

    from iai_mcp.lilli.cycle.sleep_pipeline import MAX_PAIRS_PER_CLUSTER

    clusters = _build_hebbian_clusters(store)
    records_by_id = {r.id: r for r in store.all_records()}
    summaries_created = 0
    # Accumulate every cluster's intra-cluster pairs and potentiate them in ONE
    # boost_edges call after the loop. Issuing a separate boost_edges per cluster
    # re-materializes and re-scans the entire edges table once per cluster, which
    # grinds for minutes at scale; a single batched write does it once.
    all_pairs: list[tuple[UUID, UUID]] = []
    for cluster_ids in clusters:
        cluster_recs = [records_by_id[i] for i in cluster_ids if i in records_by_id]
        if len(cluster_recs) < CLUSTER_MIN_SIZE:
            continue
        langs = [r.language for r in cluster_recs if r.language]
        dom_lang = max(set(langs), key=langs.count) if langs else "en"
        summary_text = (
            f"Cluster summary ({len(cluster_recs)} records, lang={dom_lang}): "
            + "; ".join(r.literal_surface[:80] for r in cluster_recs[:5])
        )
        _create_semantic_summary(store, cluster_recs, summary_text, dom_lang)
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
