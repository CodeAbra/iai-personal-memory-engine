from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable
from uuid import UUID, uuid4

from iai_mcp.events import write_event
from iai_mcp.guard import BudgetLedger, RateLimitLedger
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord, SCHEMA_VERSION_CURRENT


AUTO_INDUCT_COOCCURRENCE: int = 5
AUTO_INDUCT_CONFIDENCE: float = 0.85
USER_APPROVAL_COOCCURRENCE: int = 3
USER_APPROVAL_CONFIDENCE: float = 0.65
MAX_EVIDENCE_PER_SCHEMA: int = 50
PROVISIONAL_ENTROPY_MIN: float = 0.8


@dataclass
class SchemaCandidate:

    pattern: str
    confidence: float
    evidence_count: int
    evidence_ids: list[UUID] = field(default_factory=list)
    domain: str | None = None
    exceptions: list[UUID] = field(default_factory=list)
    status: str = "auto"


def _tag_cooccurrence(records: Iterable) -> dict:
    pairs: dict = {}
    for r in records:
        if hasattr(r, "tags"):
            raw_tags = r.tags or []
            rid = r.id
        else:
            tags_raw = r.get("tags_json") or "[]"
            try:
                raw_tags = json.loads(tags_raw) if tags_raw else []
            except (TypeError, json.JSONDecodeError):
                raw_tags = []
            id_raw = r.get("id")
            if id_raw is None:
                continue
            try:
                rid = UUID(id_raw) if isinstance(id_raw, str) else id_raw
            except (ValueError, AttributeError):
                continue

        # idem: tags are per-record idempotency hashes and entity: tags
        # are per-record name anchors — neither generalizes past its
        # record, and up to 10 anchors per record would make this
        # all-pairs dict quadratic in the anchor vocabulary.
        tags = [
            t for t in raw_tags
            if not t.startswith(("raw:", "domain:", "idem:", "entity:"))
        ]
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                key = frozenset([tags[i], tags[j]])
                pairs.setdefault(key, []).append(rid)
    return pairs


def induce_schemas_tier0(store: MemoryStore) -> list[SchemaCandidate]:
    rows = list(store.iter_record_columns(["id", "tags_json"], batch_size=1024))
    if len(rows) < 3:
        return []

    pair_counts = _tag_cooccurrence(rows)
    candidates: list[SchemaCandidate] = []
    for pair, evidence in pair_counts.items():
        count = len(evidence)
        confidence = min(1.0, count / 10.0)
        pattern = f"tags:{'+'.join(sorted(pair))}"
        # Confidence IS count/10 — re-checking both in one band anchors the
        # gate to itself: the approval tier demanded confidence >= 0.65 of
        # counts capped below 5 (max 0.4), so it could never fire, and counts
        # between the bands dropped entirely. Auto keeps the double gate;
        # everything else with enough co-occurrence goes to the human.
        if count >= AUTO_INDUCT_COOCCURRENCE and confidence >= AUTO_INDUCT_CONFIDENCE:
            status = "auto"
        elif count >= USER_APPROVAL_COOCCURRENCE:
            status = "pending_user_approval"
        else:
            continue
        candidates.append(
            SchemaCandidate(
                pattern=pattern,
                confidence=confidence,
                evidence_count=count,
                evidence_ids=list(evidence[:MAX_EVIDENCE_PER_SCHEMA]),
                status=status,
            )
        )
    return candidates


def induce_schemas_tier1(
    store: MemoryStore,
    budget: BudgetLedger,  # noqa: ARG001 -- retained for callers, no longer debited
    rate: RateLimitLedger,  # noqa: ARG001 -- retained for callers, no longer consulted
    llm_enabled: bool = True,  # noqa: ARG001 -- retained for callers, ignored
) -> list[SchemaCandidate]:
    write_event(
        store,
        kind="llm_health",
        data={
            "component": "schema_induction",
            "tier": "tier0_fallback",
            "reason": "subscription_only",
        },
        severity="info",
    )
    return induce_schemas_tier0(store)


def _majority_language(evidence_ids: list[UUID], store: MemoryStore) -> str:
    langs: list[str] = []
    for eid in evidence_ids:
        rec = store.get(eid)
        if rec is None:
            continue
        if rec.language:
            langs.append(rec.language)
    if not langs:
        return "en"
    best = langs[0]
    best_count = langs.count(best)
    seen: set[str] = {best}
    for lang in langs[1:]:
        if lang in seen:
            continue
        seen.add(lang)
        c = langs.count(lang)
        if c > best_count:
            best = lang
            best_count = c
    return best


def build_pattern_keeper_map(store: MemoryStore) -> "dict[str, UUID]":
    """One corpus scan resolving every semantic keeper's `pattern:*` tag.

    A loop persisting MANY candidates must build this once and hand it to
    each persist_schema call: the per-candidate fallback below walks the
    whole corpus, and rescanning it per candidate starves every other store
    consumer for the length of the induction run.
    """
    keeper_map: "dict[str, UUID]" = {}
    try:
        for row in store.iter_record_columns(
            ["id", "tier", "tags_json"], batch_size=1024
        ):
            if row.get("tier") != "semantic":
                continue
            tags_raw = row.get("tags_json") or "[]"
            try:
                tags = json.loads(tags_raw) if tags_raw else []
            except (TypeError, json.JSONDecodeError):
                tags = []
            id_raw = row.get("id")
            if id_raw is None:
                continue
            for tag in tags:
                if (
                    isinstance(tag, str)
                    and tag.startswith("pattern:")
                    and tag not in keeper_map
                ):
                    try:
                        keeper_map[tag] = (
                            UUID(id_raw) if isinstance(id_raw, str) else id_raw
                        )
                    except (ValueError, AttributeError):
                        continue
    except (OSError, RuntimeError, ValueError, KeyError):
        return keeper_map
    return keeper_map


def persist_schema(
    store: MemoryStore,
    candidate: SchemaCandidate,
    keeper_map: "dict[str, UUID] | None" = None,
) -> UUID:
    from iai_mcp.aaak import enforce_language_tagged, generate_aaak_index
    from iai_mcp.embed import embedder_for_store

    summary = (
        f"Schema: {candidate.pattern} (confidence={candidate.confidence:.2f})"
    )

    pattern_tag = f"pattern:{candidate.pattern}"
    existing_keeper_id: UUID | None = None
    if keeper_map is not None:
        existing_keeper_id = keeper_map.get(pattern_tag)
    else:
        try:
            for row in store.iter_record_columns(
                ["id", "tier", "tags_json"], batch_size=1024
            ):
                if row.get("tier") != "semantic":
                    continue
                tags_raw = row.get("tags_json") or "[]"
                try:
                    tags = json.loads(tags_raw) if tags_raw else []
                except (TypeError, json.JSONDecodeError):
                    tags = []
                if pattern_tag in tags:
                    id_raw = row.get("id")
                    if id_raw is None:
                        continue
                    try:
                        existing_keeper_id = (
                            UUID(id_raw) if isinstance(id_raw, str) else id_raw
                        )
                    except (ValueError, AttributeError):
                        continue
                    break
        except (OSError, RuntimeError, ValueError, KeyError):
            existing_keeper_id = None

    if existing_keeper_id is not None:
        from iai_mcp.store import EDGES_TABLE

        delta = max(0.1, candidate.confidence)
        new_pairs = [(ev_id, existing_keeper_id) for ev_id in candidate.evidence_ids]
        if new_pairs:
            store.boost_edges(
                new_pairs,
                edge_type="schema_instance_of",
                delta=delta,
            )

        try:
            # Filtered COUNT, never to_pandas: materializing the whole edges
            # table per persisted candidate starves the store for the run.
            keeper_str = str(existing_keeper_id)
            total_evidence = int(
                store.db.open_table(EDGES_TABLE).count_rows(
                    "edge_type = 'schema_instance_of'"
                    f" AND (dst = '{keeper_str}' OR src = '{keeper_str}')"
                )
            )
        except (OSError, RuntimeError, ValueError, KeyError):
            total_evidence = len(candidate.evidence_ids)

        write_event(
            store,
            kind="schema_reinforced",
            data={
                "schema_id": str(existing_keeper_id),
                "pattern": candidate.pattern,
                "evidence_added": len(candidate.evidence_ids),
                "total_evidence": total_evidence,
            },
            severity="info",
            source_ids=[existing_keeper_id, *candidate.evidence_ids[:5]],
        )
        return existing_keeper_id

    emb = embedder_for_store(store).embed(summary)
    now = datetime.now(timezone.utc)
    schema_id = uuid4()
    derived_language = _majority_language(candidate.evidence_ids, store)
    schema_rec = MemoryRecord(
        id=schema_id,
        tier="semantic",
        literal_surface=summary,
        aaak_index="",
        embedding=emb,
        community_id=None,
        centrality=0.0,
        detail_level=3,
        pinned=False,
        stability=0.7,
        difficulty=0.3,
        last_reviewed=now,
        never_decay=True,
        never_merge=False,
        provenance=[
            {
                "ts": now.isoformat(),
                "cue": "schema_induction",
                "session_id": "system",
            }
        ],
        created_at=now,
        updated_at=now,
        tags=[
            "schema",
            candidate.status,
            f"pattern:{candidate.pattern}",
        ],
        language=derived_language,
        s5_trust_score=0.5,
        profile_modulation_gain={},
        schema_version=SCHEMA_VERSION_CURRENT,
    )
    enforce_language_tagged(schema_rec)
    schema_rec.aaak_index = generate_aaak_index(schema_rec)
    store.insert(schema_rec)

    instance_pairs = [(ev_id, schema_id) for ev_id in candidate.evidence_ids]
    if instance_pairs:
        store.boost_edges(
            instance_pairs,
            edge_type="schema_instance_of",
            delta=max(0.1, candidate.confidence),
        )

    write_event(
        store,
        kind="schema_induction_run",
        data={
            "schema_id": str(schema_id),
            "pattern": candidate.pattern,
            "confidence": candidate.confidence,
            "evidence_count": candidate.evidence_count,
            "status": candidate.status,
        },
        severity="info",
        source_ids=[schema_id, *candidate.evidence_ids[:5]],
    )
    return schema_id


def provisional_schemas_for_recall(
    store: MemoryStore,
    hits: list,
    entropy_bits: float,
    records_cache: "dict | None" = None,
) -> list[dict]:
    if entropy_bits < PROVISIONAL_ENTROPY_MIN or len(hits) < 3:
        return []

    hit_ids = {h.record_id for h in hits}
    if records_cache is not None:
        by_id = {
            rid: rec for rid, rec in records_cache.items() if rid in hit_ids
        }
    else:
        try:
            all_recs = store.all_records()
        except (OSError, RuntimeError, ValueError):
            return []
        by_id = {r.id: r for r in all_recs if r.id in hit_ids}

    tag_count: Counter = Counter()
    for h in hits:
        rec = by_id.get(h.record_id)
        if rec is None:
            continue
        for t in (rec.tags or []):
            if t.startswith("raw:") or t.startswith("domain:"):
                continue
            tag_count[t] += 1

    provisional: list[dict] = []
    for tag, cnt in tag_count.most_common(3):
        if cnt >= 2:
            source_ids: list[str] = []
            for h in hits:
                rec = by_id.get(h.record_id)
                if rec is None:
                    continue
                if tag in (rec.tags or []):
                    source_ids.append(str(h.record_id))
                if len(source_ids) >= 5:
                    break
            provisional.append(
                {
                    "kind": "provisional_schema",
                    "severity": "info",
                    "source_ids": source_ids,
                    "text": f"Potential schema: tag={tag} cnt={cnt}",
                    "provisional": True,
                    "entropy": entropy_bits,
                }
            )
    return provisional
