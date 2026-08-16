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


def _pattern_of(rec: "MemoryRecord") -> str:
    tag = next(
        (t for t in (rec.tags or []) if t.startswith("pattern:")), None,
    )
    return tag.split(":", 1)[1] if tag and ":" in tag else ""


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

    # Noise schemas: a pattern referencing an idem hash can never generalize
    # past its one record — prune them regardless of duplication.
    noise: list[MemoryRecord] = []
    pruned_ids = {d.id for d in duplicates}
    keeper_ids = {k.id for k in keepers}
    for pattern, recs in groups.items():
        if "idem:" not in pattern:
            continue
        for rec in recs:
            if rec.id in pruned_ids:
                continue
            noise.append(rec)
            pruned_ids.add(rec.id)
    keepers = [k for k in keepers if "idem:" not in _pattern_of(k)]
    duplicates.extend(noise)

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
        "noise_pruned": len(noise),
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


def cleanup_idem_duplicates(
    store: MemoryStore,
    *,
    apply: bool = False,
    store_path: "Path | None" = None,
) -> dict:
    """Collapse live records sharing one idempotency key down to the earliest.

    The idem tag is the exact-key identity of a captured turn; two live
    records with the same key are the same event stored twice (historical
    check-then-insert races and tag-index holes). The keeper is the earliest
    copy; the rest are tombstoned — content is preserved byte-identical in
    the keeper, so the verbatim invariant holds.
    """
    import json as _json
    import shutil
    from datetime import datetime, timezone

    from iai_mcp.store._buffers import flush_record_buffer

    flush_record_buffer(store)
    groups: dict[str, list[tuple[str, str]]] = {}
    with store.db._conn_lock:
        rows = store.db._conn.execute(
            "SELECT id, tags_json, created_at FROM records "
            "WHERE tombstoned_at IS NULL"
        ).fetchall()
        for row in rows:
            try:
                tags = _json.loads(row["tags_json"] or "[]")
            except (TypeError, ValueError):
                continue
            for tag in tags:
                if isinstance(tag, str) and tag.startswith("idem:"):
                    groups.setdefault(tag, []).append(
                        (str(row["id"]), str(row["created_at"] or ""))
                    )
                    break

    dup_groups = {t: rs for t, rs in groups.items() if len(rs) > 1}
    to_tombstone: list[str] = []
    for _tag, members in dup_groups.items():
        members_sorted = sorted(members, key=lambda m: m[1])
        to_tombstone.extend(mid for mid, _ in members_sorted[1:])

    snapshot_dir: str | None = None
    tombstoned = 0
    if apply and to_tombstone:
        iai_root = Path(store_path) if store_path is not None else Path(store.root)
        src_hippo = iai_root / "hippo"
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snap = iai_root / f"hippo-pre-idem-dedup-{ts}"
        shutil.copytree(src_hippo if src_hippo.exists() else iai_root, snap)
        snapshot_dir = str(snap)

        now_iso = datetime.now(timezone.utc).isoformat()
        tbl = store.db.open_table("records")
        tombstoned_ids: list[str] = []
        for rid in to_tombstone:
            try:
                tbl.update(
                    where=f"id = '{rid}'",
                    values={"tombstoned_at": now_iso, "live": 0},
                )
                tombstoned += 1
                tombstoned_ids.append(rid)
            except (OSError, ValueError, RuntimeError) as exc:
                log.warning("idem dedup tombstone failed for %s: %s", rid, exc)
        try:
            store._invalidate_corpus_count("active")
        except Exception:  # noqa: BLE001 -- count cache refresh is best-effort
            pass
        _inv_x = getattr(store, "invalidate_exact_index", None)
        if callable(_inv_x):
            _inv_x()  # no-raise contract; see MemoryStore.invalidate_exact_index
        buf = getattr(store, "_recency_buffer", None)
        if buf is not None:
            for rid in tombstoned_ids:
                try:
                    buf.evict(str(rid))
                except Exception as exc:  # noqa: BLE001 -- eviction is best-effort
                    log.debug("recency buffer evict failed for %s: %s", rid, exc)

    summary = {
        "mode": "apply" if apply else "dry-run",
        "groups": len(dup_groups),
        "extra_copies": len(to_tombstone),
        "tombstoned": tombstoned,
        "snapshot_dir": snapshot_dir,
    }
    try:
        write_event(
            store,
            kind="idem_dedup_run",
            data=summary,
            severity="info",
            session_id="system",
        )
    except (OSError, ValueError, RuntimeError):
        pass
    return summary


CONSOLIDATION_BACKFILL_OVERSIZE = 200
"""Communities past this size are never claimed wholesale — mirrors the
minting-side oversize skip (mega-clusters are stamped, not compressed)."""

_BACKFILL_BATCH = 500


def backfill_consolidated_edges(
    store: MemoryStore,
    *,
    apply: bool = False,
    store_path: "Path | None" = None,
) -> dict:
    """Re-link live semantic summaries to their source moments.

    A summary's consolidated_from edges are its coverage ledger; historical
    edge loss left old summaries anchored to a fraction of their cluster.
    Expansion is structural only: a summary anchored by at least one
    surviving edge into a community claims that community's remaining live
    episodic members. Summaries with zero surviving anchors are counted but
    never guessed at — they re-cover naturally when their cluster re-mints
    and dedup-folds into them.
    """
    import shutil
    from datetime import datetime, timezone

    from iai_mcp.store._buffers import flush_record_buffer

    flush_record_buffer(store)

    sem: set[str] = set()
    epi_community: dict[str, "str | None"] = {}
    with store.db._conn_lock:
        rows = store.db._conn.execute(
            "SELECT id, tier, community_id FROM records WHERE tombstoned_at IS NULL"
        ).fetchall()
        for row in rows:
            rid = str(row["id"])
            if row["tier"] == "semantic":
                sem.add(rid)
            elif row["tier"] == "episodic":
                comm = row["community_id"]
                epi_community[rid] = str(comm) if comm else None
        edge_rows = store.db._conn.execute(
            "SELECT src, dst FROM edges WHERE edge_type = 'consolidated_from'"
        ).fetchall()

    # Edge rows carry no direction (pair order is canonicalized on write) —
    # the summary side is identified by tier, never by column.
    anchors_of: dict[str, set[str]] = {}
    for row in edge_rows:
        a, b = str(row["src"]), str(row["dst"])
        if a in sem and b in epi_community:
            anchors_of.setdefault(a, set()).add(b)
        elif b in sem and a in epi_community:
            anchors_of.setdefault(b, set()).add(a)

    community_members: dict[str, set[str]] = {}
    for rid, comm in epi_community.items():
        if comm:
            community_members.setdefault(comm, set()).add(rid)

    proposed: list[tuple[UUID, UUID]] = []
    oversize_skipped: set[str] = set()
    for sid, anchors in anchors_of.items():
        communities = {epi_community[m] for m in anchors if epi_community.get(m)}
        for comm in communities:
            candidates = community_members.get(comm) or set()
            if len(candidates) > CONSOLIDATION_BACKFILL_OVERSIZE:
                oversize_skipped.add(comm)
                continue
            for rid in sorted(candidates - anchors):
                proposed.append((UUID(sid), UUID(rid)))

    snapshot_dir: "str | None" = None
    written = 0
    if apply and proposed:
        iai_root = Path(store_path) if store_path is not None else Path(store.root)
        src_hippo = iai_root / "hippo"
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snap = iai_root / f"hippo-pre-edge-backfill-{ts}"
        shutil.copytree(src_hippo if src_hippo.exists() else iai_root, snap)
        snapshot_dir = str(snap)

        for start in range(0, len(proposed), _BACKFILL_BATCH):
            batch = proposed[start:start + _BACKFILL_BATCH]
            try:
                store.boost_edges(batch, edge_type="consolidated_from", delta=1.0)
                written += len(batch)
            except (OSError, ValueError, RuntimeError) as exc:
                log.warning("edge backfill batch failed at %d: %s", start, exc)

    summary = {
        "mode": "apply" if apply else "dry-run",
        "summaries_live": len(sem),
        "summaries_anchored": len(anchors_of),
        "summaries_edgeless": len(sem) - len(anchors_of),
        "links_existing": sum(len(v) for v in anchors_of.values()),
        "links_proposed": len(proposed),
        "links_written": written,
        "communities_oversize_skipped": len(oversize_skipped),
        "snapshot_dir": snapshot_dir,
    }
    try:
        write_event(
            store,
            kind="edge_backfill_run",
            data=summary,
            severity="info",
            session_id="system",
        )
    except (OSError, ValueError, RuntimeError):
        pass
    return summary
