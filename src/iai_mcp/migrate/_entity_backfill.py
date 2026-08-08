"""Backfill entity anchors onto records captured before anchor extraction.

Every record minted without ``entity:`` tags gets them derived from its
literal surface, and its AAAK index regenerated so the E: field carries
the anchors; a record already tagged but with an empty index gets the
index refreshed alone. ``refresh=True`` recomputes anchors for the whole
corpus instead — removing anchors the current extractor no longer emits
(stopwords, machine tokens) and adding the ones it now does. Journaled:
dry-run reports, ``apply=True`` writes and records every applied change
(added tags, removed tags, old index) to a journal under the store root.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from iai_mcp.aaak import generate_aaak_index
from iai_mcp.entity_anchors import entity_tags


def backfill_entity_anchors(
    store,
    apply: bool = False,
    store_path: Path | None = None,
    refresh: bool = False,
) -> dict:
    from iai_mcp.store._buffers import flush_record_buffer

    flush_record_buffer(store)

    with store.db._conn_lock:
        rows = store.db._conn.execute(
            "SELECT id, tags_json, aaak_index FROM records"
            " WHERE tombstoned_at IS NULL"
        ).fetchall()
        id_rows = [(str(r[0]), r[1], r[2]) for r in rows]

    scanned = 0
    already_anchored = 0
    errors: list[str] = []
    to_anchor: list[tuple[UUID, list[str], list[str]]] = []
    for rid_s, tags_json, aaak_index in id_rows:
        scanned += 1
        try:
            try:
                tags = json.loads(tags_json) if tags_json else []
            except (ValueError, TypeError):
                tags = []
            existing = [str(t) for t in tags if str(t).startswith("entity:")]
            if not refresh:
                if existing and aaak_index:
                    already_anchored += 1
                    continue
                if existing:
                    # Tagged elsewhere but the index never carried the
                    # anchors: refresh the index alone, no new tags.
                    to_anchor.append((UUID(rid_s), [], []))
                    continue
            rec = store.get(UUID(rid_s))
            if rec is None or not rec.literal_surface:
                continue
            fresh = entity_tags(rec.literal_surface)
            add = [t for t in fresh if t not in existing]
            remove = [t for t in existing if t not in fresh] if refresh else []
            if add or remove:
                to_anchor.append((rec.id, add, remove))
            elif refresh and existing:
                already_anchored += 1
        except Exception as exc:  # noqa: BLE001 -- one bad record must not abort the sweep
            errors.append(f"{rid_s}: scan: {type(exc).__name__}: {exc}")

    written = 0
    aaak_refreshed = 0
    journal_path: str | None = None
    if apply and to_anchor:
        root = Path(store_path) if store_path else Path.home() / ".iai-mcp"
        journal_dir = root / "backfills"
        journal_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(journal_dir, 0o700)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        # "x" mode: a second run in the same second must never truncate
        # the first run's journal.
        for suffix in ("", "-2", "-3", "-4", "-5"):
            journal = journal_dir / f"entity-backfill-{ts}{suffix}.jsonl"
            try:
                jf = journal.open("x", encoding="utf-8")
                break
            except FileExistsError:
                continue
        else:
            raise FileExistsError(
                f"could not allocate a journal file under {journal_dir}"
            )
        journal_path = str(journal)
        with jf:
            os.chmod(journal, 0o600)
            for rid, add, remove in to_anchor:
                try:
                    rec = store.get(rid)
                    if rec is None:
                        continue
                    old_index = rec.aaak_index
                    tags_removed = bool(remove) and store.remove_tags(rid, remove)
                    tags_added = bool(add) and store.add_tags(rid, add)
                    if tags_added or tags_removed:
                        written += 1
                        # Mirror the tag mutations locally — a second full
                        # get() would decrypt the record again just to
                        # re-read tags this loop already knows.
                        kept = [
                            t for t in rec.tags
                            if not (tags_removed and t in remove)
                        ]
                        rec.tags = kept + [
                            t for t in (add if tags_added else [])
                            if t not in kept
                        ]
                    index_set = store.set_aaak_index(
                        rid, generate_aaak_index(rec)
                    )
                    if index_set:
                        aaak_refreshed += 1
                    if tags_added or tags_removed or index_set:
                        # Journal only what actually landed: an undo replay
                        # must never strip tags this run did not add.
                        jf.write(json.dumps({
                            "record_id": str(rid),
                            "added_tags": add if tags_added else [],
                            "removed_tags": remove if tags_removed else [],
                            "old_aaak_index": old_index,
                        }, ensure_ascii=False) + "\n")
                        jf.flush()
                except Exception as exc:  # noqa: BLE001 -- one bad record must not abort the sweep
                    errors.append(f"{rid}: apply: {type(exc).__name__}: {exc}")

    cache_invalidated = False
    if apply and (written or aaak_refreshed):
        # The runtime-graph cache key covers record/edge counts only, and
        # this sweep changes neither — a daemon rebooting onto the cached
        # graph would serve pre-backfill payloads without the anchors.
        # Dropping the cache forces a rebuild from the store.
        from iai_mcp.runtime_graph_cache import CACHE_FILENAME

        root = Path(store_path) if store_path else Path.home() / ".iai-mcp"
        cache_file = root / CACHE_FILENAME
        try:
            if cache_file.exists():
                cache_file.unlink()
                cache_invalidated = True
        except OSError as exc:
            errors.append(f"graph-cache: {type(exc).__name__}: {exc}")

    return {
        "mode": "apply" if apply else "dry-run",
        "refresh": bool(refresh),
        "graph_cache_invalidated": cache_invalidated,
        "records_scanned": scanned,
        "already_anchored": already_anchored,
        "records_to_anchor": len(to_anchor),
        "anchors_proposed": sum(len(a) for _, a, _ in to_anchor),
        "anchors_to_remove": sum(len(r) for _, _, r in to_anchor),
        "records_written": written,
        "aaak_refreshed": aaak_refreshed,
        "journal": journal_path,
        "errors": errors,
    }
