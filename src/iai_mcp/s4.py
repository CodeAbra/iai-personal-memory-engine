from __future__ import annotations

import logging
from uuid import UUID

import numpy as np

from iai_mcp.events import write_event
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryHit, MemoryRecord

logger = logging.getLogger(__name__)


S4_VIGILANCE_RHO = 0.97

MONOTROPIC_MAX_PAIRWISE = 100

S4_MONOTROPIC_THETA = 0.7


def _cosine(a: list[float], b: list[float]) -> float:
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def on_read_check(
    store: MemoryStore,
    hits: list[MemoryHit],
    session_id: str,
) -> list[dict]:
    if len(hits) < 2:
        return []

    hint_list: list[dict] = []

    records: dict[UUID, MemoryRecord] = {}
    for h in hits:
        rec = store.get(h.record_id)
        if rec is not None:
            records[h.record_id] = rec
    if len(records) < 2:
        return []

    contradict_pairs: set[tuple[str, str]] = set()
    try:
        edges_df = store.db.open_table("edges").to_pandas()
    except (OSError, RuntimeError, ValueError):
        edges_df = None
    if edges_df is not None and not edges_df.empty:
        contradict_df = edges_df[edges_df["edge_type"] == "contradicts"]
        hit_ids = {str(h.record_id) for h in hits}
        for _, row in contradict_df.iterrows():
            src = row["src"]
            dst = row["dst"]
            if src in hit_ids and dst in hit_ids:
                contradict_pairs.add(tuple(sorted([src, dst])))

    hit_records = list(records.values())
    for i in range(len(hit_records)):
        for j in range(i + 1, len(hit_records)):
            a = hit_records[i]
            b = hit_records[j]
            key = tuple(sorted([str(a.id), str(b.id)]))
            sim = _cosine(a.embedding, b.embedding)

            if key in contradict_pairs:
                hint = {
                    "kind": "s4_contradiction",
                    "severity": "warning",
                    "source_ids": [str(a.id), str(b.id)],
                    "text": (
                        f"inconsistency: records have a contradicts edge; "
                        f"review {a.id}, {b.id}"
                    ),
                    "similarity": sim,
                }
                hint_list.append(hint)
                write_event(
                    store,
                    kind="s4_contradiction",
                    data={
                        "source_ids": list(key),
                        "similarity": sim,
                        "mechanism": "contradicts_edge",
                    },
                    severity="warning",
                    session_id=session_id,
                    source_ids=[a.id, b.id],
                )
                continue

            if sim >= S4_VIGILANCE_RHO:
                a_tags = set(a.tags or [])
                b_tags = set(b.tags or [])
                polarity_conflict = (
                    ("positive" in a_tags and "negative" in b_tags)
                    or ("negative" in a_tags and "positive" in b_tags)
                    or ("asserted" in a_tags and "retracted" in b_tags)
                    or ("retracted" in a_tags and "asserted" in b_tags)
                )
                if polarity_conflict:
                    hint = {
                        "kind": "s4_contradiction",
                        "severity": "info",
                        "source_ids": [str(a.id), str(b.id)],
                        "text": (
                            f"inconsistency: near-duplicate ({sim:.3f}) with "
                            f"conflicting polarity tags"
                        ),
                        "similarity": sim,
                    }
                    hint_list.append(hint)
                    write_event(
                        store,
                        kind="s4_contradiction",
                        data={
                            "source_ids": list(key),
                            "similarity": sim,
                            "mechanism": "tag_polarity",
                        },
                        severity="info",
                        session_id=session_id,
                        source_ids=[a.id, b.id],
                    )
    return hint_list


def on_read_check_batch(
    store: MemoryStore,
    hits: list[MemoryHit],
    session_id: str,
    records_cache: "dict[UUID, MemoryRecord] | None" = None,
    contradicts_outgoing: "dict[str, list[str]] | None" = None,
) -> list[dict]:
    if len(hits) < 2:
        return []

    hint_list: list[dict] = []

    records: dict[UUID, MemoryRecord] = {}
    if records_cache is not None:
        for h in hits:
            rec = records_cache.get(h.record_id)
            if rec is not None:
                records[h.record_id] = rec
    else:
        all_recs = store.all_records()
        by_id = {r.id: r for r in all_recs}
        for h in hits:
            rec = by_id.get(h.record_id)
            if rec is not None:
                records[h.record_id] = rec
    if len(records) < 2:
        return []

    contradict_pairs: set[tuple[str, str]] = set()
    hit_ids = {str(h.record_id) for h in hits}
    if contradicts_outgoing is not None:
        for src, dsts in contradicts_outgoing.items():
            if src in hit_ids:
                for dst in dsts:
                    if dst in hit_ids:
                        contradict_pairs.add(tuple(sorted([src, dst])))
    else:
        try:
            edges_df = store.db.open_table("edges").to_pandas()
        except (OSError, RuntimeError, ValueError):
            edges_df = None
        if edges_df is not None and not edges_df.empty:
            contradict_df = edges_df[edges_df["edge_type"] == "contradicts"]
            for _, row in contradict_df.iterrows():
                src = row["src"]
                dst = row["dst"]
                if src in hit_ids and dst in hit_ids:
                    contradict_pairs.add(tuple(sorted([src, dst])))

    hit_records = list(records.values())
    for i in range(len(hit_records)):
        for j in range(i + 1, len(hit_records)):
            a = hit_records[i]
            b = hit_records[j]
            key = tuple(sorted([str(a.id), str(b.id)]))
            sim = _cosine(a.embedding, b.embedding)

            if key in contradict_pairs:
                hint = {
                    "kind": "s4_contradiction",
                    "severity": "warning",
                    "source_ids": [str(a.id), str(b.id)],
                    "text": (
                        f"inconsistency: records have a contradicts edge; "
                        f"review {a.id}, {b.id}"
                    ),
                    "similarity": sim,
                }
                hint_list.append(hint)
                write_event(
                    store,
                    kind="s4_contradiction",
                    data={
                        "source_ids": list(key),
                        "similarity": sim,
                        "mechanism": "contradicts_edge",
                    },
                    severity="warning",
                    session_id=session_id,
                    source_ids=[a.id, b.id],
                )
                continue

            if sim >= S4_VIGILANCE_RHO:
                a_tags = set(a.tags or [])
                b_tags = set(b.tags or [])
                polarity_conflict = (
                    ("positive" in a_tags and "negative" in b_tags)
                    or ("negative" in a_tags and "positive" in b_tags)
                    or ("asserted" in a_tags and "retracted" in b_tags)
                    or ("retracted" in a_tags and "asserted" in b_tags)
                )
                if polarity_conflict:
                    hint = {
                        "kind": "s4_contradiction",
                        "severity": "info",
                        "source_ids": [str(a.id), str(b.id)],
                        "text": (
                            f"inconsistency: near-duplicate ({sim:.3f}) with "
                            f"conflicting polarity tags"
                        ),
                        "similarity": sim,
                    }
                    hint_list.append(hint)
                    write_event(
                        store,
                        kind="s4_contradiction",
                        data={
                            "source_ids": list(key),
                            "similarity": sim,
                            "mechanism": "tag_polarity",
                        },
                        severity="info",
                        session_id=session_id,
                        source_ids=[a.id, b.id],
                    )
    return hint_list


def monotropic_proactive_check(
    store: MemoryStore,
    new_record: MemoryRecord,
    profile_state: dict,
    session_id: str,
) -> list[dict]:
    md = profile_state.get("monotropism_depth", {})
    if not isinstance(md, dict):
        return []

    from iai_mcp.core import get_community_names
    community_names = get_community_names()

    if new_record.community_id is None:
        return []
    name = community_names.get(str(new_record.community_id))
    if name is None:
        return []

    depth = md.get(name, 0.0)
    if depth <= S4_MONOTROPIC_THETA:
        return []

    if new_record.detail_level < 4:
        return []

    same_domain = [
        r for r in store.all_records()
        if r.id != new_record.id
        and r.community_id is not None
        and community_names.get(str(r.community_id)) == name
    ]

    if len(same_domain) > MONOTROPIC_MAX_PAIRWISE:
        write_event(
            store,
            kind="s4_monotropic_skip",
            data={
                "count": len(same_domain),
                "record_id": str(new_record.id),
            },
            severity="warning",
            session_id=session_id,
        )
        return []

    hints: list[dict] = []
    for r in same_domain:
        sim = _cosine(new_record.embedding, r.embedding)
        if sim >= S4_VIGILANCE_RHO:
            hint = {
                "kind": "s4_monotropic_contradiction",
                "severity": "info",
                "source_ids": [str(new_record.id), str(r.id)],
                "text": (
                    f"monotropic near-duplicate in {name}: sim={sim:.3f}"
                ),
                "similarity": sim,
            }
            hints.append(hint)
            write_event(
                store,
                kind="s4_monotropic_contradiction",
                data={
                    "source_ids": [str(new_record.id), str(r.id)],
                    "similarity": sim,
                },
                severity="info",
                session_id=session_id,
                source_ids=[new_record.id, r.id],
            )
    return hints


def run_offline_pass(store: MemoryStore) -> dict:
    from iai_mcp import sigma

    out: dict = {}
    try:
        out["sigma"] = sigma.compute_and_emit(store)
    except Exception as exc:  # noqa: BLE001 -- diagnostic catch-all; must not crash pass
        logger.warning("s4_offline_sigma_failed", extra={"err": str(exc)[:200]})
        try:
            write_event(
                store,
                kind="s4_error",
                data={"step": "sigma", "error": repr(exc)},
                severity="warning",
            )
        except Exception:  # noqa: BLE001 -- event write failure is non-fatal
            pass
        out["sigma"] = {"error": repr(exc)}
    return out


_s4_bg_cursor: int = 0


def s4_background_scan(store: "MemoryStore", batch_size: int = 50) -> dict:
    global _s4_bg_cursor
    from iai_mcp.events import write_event, query_events
    from iai_mcp.store import EDGES_TABLE

    try:
        tbl = store.db.open_table(EDGES_TABLE)
        df = (
            tbl.search()
            .where("edge_type = 'contradicts'")
            .limit(batch_size)
            .to_pandas()
        )
    except (OSError, RuntimeError, ValueError):
        return {"scanned": 0, "flagged": 0}

    if df.empty:
        _s4_bg_cursor = 0
        return {"scanned": 0, "flagged": 0}

    try:
        existing_events = query_events(store, kind="s4_contradiction_flagged", limit=200)
    except (OSError, RuntimeError, ValueError):
        existing_events = []
    flagged_pairs: set[tuple[str, str]] = set()
    for ev in existing_events:
        d = ev.get("data", {})
        s = d.get("src", "")
        t = d.get("dst", "")
        if s and t:
            flagged_pairs.add((s, t))

    flagged = 0
    for _, row in df.iterrows():
        src_id = row.get("src", "")
        dst_id = row.get("dst", "")
        if not src_id or not dst_id:
            continue
        if (src_id, dst_id) in flagged_pairs:
            continue
        try:
            write_event(
                store,
                kind="s4_contradiction_flagged",
                data={
                    "src": src_id,
                    "dst": dst_id,
                    "source": "background_scan",
                    "resolution": "pending_rem",
                },
                severity="info",
            )
            flagged += 1
            flagged_pairs.add((src_id, dst_id))
        except (OSError, RuntimeError, ValueError):
            pass

    _s4_bg_cursor += len(df)
    return {"scanned": len(df), "flagged": flagged}
