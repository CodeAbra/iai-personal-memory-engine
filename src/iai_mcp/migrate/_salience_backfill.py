"""Backfill the value signal (salience + entity/AAAK coverage) across the
entire existing store.

Two metadata-only mutation classes, both monotonic/idempotent, neither ever
touching ``literal_surface``:
- entity/AAAK: reuses ``backfill_entity_anchors`` verbatim (no re-derivation).
- salience: the capture-time composite (``classify_salience``) for ordinary
  rows, and the cluster-size rule (existing ``cls_summary`` rows -> notable)
  for pre-existing summaries that predate that rule, via
  ``raise_salience_level_if_higher`` (raise-only, never lowers).

Dry-run by default; ``--apply`` snapshots the store dir first.
"""
from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from iai_mcp.types import SALIENCE_LEVEL_RANK


def backfill_salience(
    store,
    *,
    apply: bool = False,
    store_path: "Path | None" = None,
) -> dict:
    import shutil
    from datetime import datetime, timezone

    from iai_mcp.migrate._entity_backfill import backfill_entity_anchors
    from iai_mcp.salience_classify import classify_salience
    from iai_mcp.store._buffers import flush_record_buffer

    snapshot_dir: str | None = None
    if apply:
        iai_root = Path(store_path) if store_path is not None else Path(store.root)
        src_hippo = iai_root / "hippo"
        snapshot_source = src_hippo if src_hippo.exists() else iai_root
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        # A rerun within the same second must never collide with the prior
        # snapshot -- try successive suffixes rather than overwriting it.
        for suffix in ("", "-2", "-3", "-4", "-5"):
            snap = iai_root / f"hippo-pre-salience-backfill-{ts}{suffix}"
            try:
                shutil.copytree(snapshot_source, snap)
                break
            except FileExistsError:
                continue
        else:
            raise FileExistsError(
                f"could not allocate a snapshot dir under {iai_root}"
            )
        snapshot_dir = str(snap)

    entity_summary = backfill_entity_anchors(
        store, apply=apply, store_path=store_path,
    )

    flush_record_buffer(store)
    with store.db._conn_lock:
        rows = store.db._conn.execute(
            "SELECT id, tier, tags_json, aaak_index, epistemic_status,"
            " directive, salience_level FROM records"
            " WHERE tombstoned_at IS NULL"
        ).fetchall()
        snapshot_rows = [
            (
                str(row["id"]),
                str(row["tier"]),
                row["tags_json"],
                row["aaak_index"],
                str(row["epistemic_status"] or "unknown"),
                bool(row["directive"]),
                str(row["salience_level"] or "unflagged"),
            )
            for row in rows
        ]

    scanned = 0
    current_dist = {"unflagged": 0, "notable": 0, "critical": 0}
    projected_dist = {"unflagged": 0, "notable": 0, "critical": 0}
    aaak_nonempty = 0
    entity_tagged = 0
    would_promote = 0
    promoted = 0
    errors: list[str] = []

    for rid_s, tier, tags_json, aaak_index, epistemic_status, directive, salience_level in snapshot_rows:
        scanned += 1
        try:
            tags = json.loads(tags_json) if tags_json else []
        except (TypeError, ValueError):
            tags = []
        tags = [str(t) for t in tags]
        entity_tag_list = [t for t in tags if t.startswith("entity:")]

        if str(aaak_index or ""):
            aaak_nonempty += 1
        if entity_tag_list:
            entity_tagged += 1
        current_dist[salience_level] = current_dist.get(salience_level, 0) + 1

        if tier == "semantic" and "cls_summary" in tags:
            target = "notable"
        else:
            target = classify_salience(
                directive=directive,
                entity_tags=entity_tag_list,
                epistemic_status=epistemic_status,
            )

        final_level = salience_level
        if SALIENCE_LEVEL_RANK.get(target, 0) > SALIENCE_LEVEL_RANK.get(salience_level, 0):
            would_promote += 1
            final_level = target
            if apply:
                try:
                    if store.raise_salience_level_if_higher(UUID(rid_s), target):
                        promoted += 1
                    else:
                        final_level = salience_level
                except Exception as exc:  # noqa: BLE001 -- one bad record must not abort the sweep
                    errors.append(f"{rid_s}: apply: {type(exc).__name__}: {exc}")
                    final_level = salience_level
        projected_dist[final_level] = projected_dist.get(final_level, 0) + 1

    cache_invalidated = False
    if apply and promoted:
        # The runtime-graph cache key covers record/edge counts only, and
        # this sweep changes neither — a daemon rebooting onto the cached
        # graph would serve pre-backfill salience marks. Dropping the cache
        # forces a rebuild from the store.
        from iai_mcp.runtime_graph_cache import CACHE_FILENAME

        cache_root = Path(store_path) if store_path is not None else Path(store.root)
        cache_file = cache_root / CACHE_FILENAME
        try:
            if cache_file.exists():
                cache_file.unlink()
                cache_invalidated = True
        except OSError as exc:
            errors.append(f"graph-cache: {type(exc).__name__}: {exc}")

    return {
        "mode": "apply" if apply else "dry-run",
        "records_scanned": scanned,
        "salience_would_promote": would_promote,
        "salience_promoted": promoted,
        "salience_distribution_current": current_dist,
        "salience_distribution_resulting": projected_dist,
        "aaak_nonempty_rate": (aaak_nonempty / scanned) if scanned else 0.0,
        "entity_tag_rate": (entity_tagged / scanned) if scanned else 0.0,
        "snapshot_dir": snapshot_dir,
        "graph_cache_invalidated": cache_invalidated,
        "entity_backfill": entity_summary,
        "errors": errors,
    }
