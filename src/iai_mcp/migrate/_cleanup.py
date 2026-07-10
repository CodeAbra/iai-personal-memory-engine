from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

from iai_mcp.events import write_event
from iai_mcp.store import (
    MemoryStore,
)
from iai_mcp.types import (
    MemoryRecord,
)


log = logging.getLogger(__name__)


def cleanup_schema_duplicates(
    store: MemoryStore,
    *,
    apply: bool = False,
    store_path: "Path | None" = None,
) -> dict:
    import shutil
    from pathlib import Path
    from datetime import datetime, timezone

    from iai_mcp.store import EDGES_TABLE
    from iai_mcp.types import SEMANTIC_PRUNED_TIER

    groups: dict[str, list[MemoryRecord]] = {}
    try:
        all_records = store.all_records()
    except (OSError, ValueError, RuntimeError) as exc:
        log.error("schema cleanup all_records read failed: %s", exc)
        return {
            "mode": "apply" if apply else "dry-run",
            "groups": 0,
            "keepers": 0,
            "pruned": 0,
            "edges_reinforced": 0,
            "snapshot_dir": None,
        }

    for rec in all_records:
        if rec.tier != "semantic":
            continue
        pattern_tag = next(
            (t for t in (rec.tags or []) if t.startswith("pattern:")),
            None,
        )
        if pattern_tag is None or ":" not in pattern_tag:
            continue
        pattern = pattern_tag.split(":", 1)[1]
        groups.setdefault(pattern, []).append(rec)

    dup_groups = {p: recs for p, recs in groups.items() if len(recs) > 1}

    keepers: list[MemoryRecord] = []
    duplicates: list[MemoryRecord] = []
    for pattern, recs in dup_groups.items():
        recs_sorted = sorted(recs, key=lambda r: r.created_at)
        keepers.append(recs_sorted[0])
        duplicates.extend(recs_sorted[1:])

    edges_to_reinforce = 0
    try:
        edges_df = store.db.open_table(EDGES_TABLE).to_pandas()
        dup_id_strs = {str(d.id) for d in duplicates}
        if dup_id_strs and "edge_type" in edges_df.columns:
            mask = (
                (edges_df["edge_type"] == "schema_instance_of")
                & (
                    edges_df["dst"].isin(dup_id_strs)
                    | edges_df["src"].isin(dup_id_strs)
                )
            )
            edges_to_reinforce = int(mask.sum())
    except (OSError, ValueError, KeyError) as exc:
        log.error("schema cleanup edges scan failed: %s", exc)
        edges_to_reinforce = 0

    snapshot_dir: str | None = None

    if apply and (keepers or duplicates):
        iai_root = Path(store_path) if store_path is not None else Path(store.root)
        src_hippo = iai_root / "hippo"
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snap = iai_root / f"hippo-pre-cleanup-{ts}"
        snapshot_source = src_hippo if src_hippo.exists() else iai_root
        shutil.copytree(snapshot_source, snap)
        snapshot_dir = str(snap)

        keeper_by_pattern: dict[str, MemoryRecord] = {}
        for k in keepers:
            kp = next(
                (t for t in (k.tags or []) if t.startswith("pattern:")),
                None,
            )
            if kp and ":" in kp:
                keeper_by_pattern[kp.split(":", 1)[1]] = k

        try:
            edges_df = store.db.open_table(EDGES_TABLE).to_pandas()
            for dup in duplicates:
                dp = next(
                    (t for t in (dup.tags or []) if t.startswith("pattern:")),
                    None,
                )
                if dp is None or ":" not in dp:
                    continue
                pattern = dp.split(":", 1)[1]
                keeper = keeper_by_pattern.get(pattern)
                if keeper is None or keeper.id == dup.id:
                    continue
                dup_str = str(dup.id)
                incoming_mask = (
                    (edges_df["edge_type"] == "schema_instance_of")
                    & ((edges_df["dst"] == dup_str) | (edges_df["src"] == dup_str))
                )
                incoming = edges_df[incoming_mask]
                if incoming.empty:
                    continue
                pairs: list[tuple[UUID, UUID]] = []
                for _, row in incoming.iterrows():
                    other_str = (
                        row["src"] if row["dst"] == dup_str else row["dst"]
                    )
                    if other_str == dup_str:
                        continue
                    try:
                        other_id = UUID(str(other_str))
                    except (TypeError, ValueError):
                        continue
                    pairs.append((other_id, keeper.id))
                if pairs:
                    store.boost_edges(
                        pairs,
                        edge_type="schema_instance_of",
                        delta=0.1,
                    )
        except (OSError, ValueError, RuntimeError) as exc:
            log.error("schema cleanup edge reinforce failed: %s", exc)

        for dup in duplicates:
            try:
                store.delete(dup.id)
                pruned_rec = MemoryRecord(
                    id=dup.id,
                    tier=SEMANTIC_PRUNED_TIER,
                    literal_surface=dup.literal_surface,
                    aaak_index=dup.aaak_index,
                    embedding=dup.embedding,
                    community_id=dup.community_id,
                    centrality=dup.centrality,
                    detail_level=dup.detail_level,
                    pinned=False,
                    stability=dup.stability,
                    difficulty=dup.difficulty,
                    last_reviewed=dup.last_reviewed,
                    never_decay=False,
                    never_merge=dup.never_merge,
                    provenance=dup.provenance,
                    created_at=dup.created_at,
                    updated_at=datetime.now(timezone.utc),
                    tags=dup.tags,
                    language=dup.language,
                    s5_trust_score=dup.s5_trust_score,
                    profile_modulation_gain=dup.profile_modulation_gain,
                    schema_version=dup.schema_version,
                    structure_hv=dup.structure_hv,
                )
                store.insert(pruned_rec)
            except (OSError, ValueError, RuntimeError):
                continue

    summary: dict = {
        "mode": "apply" if apply else "dry-run",
        "groups": len(dup_groups),
        "keepers": len(keepers),
        "pruned": len(duplicates),
        "edges_reinforced": int(edges_to_reinforce),
        "snapshot_dir": snapshot_dir,
    }
    try:
        write_event(
            store,
            kind="schema_cleanup_run",
            data=summary,
            severity="info",
            source_ids=[k.id for k in keepers[:5]] if keepers else None,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        log.error("schema_cleanup_run event write failed: %s", exc)
    return summary
