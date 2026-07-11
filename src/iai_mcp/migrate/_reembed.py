from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pyarrow as pa

from iai_mcp.events import write_event
from iai_mcp.store import (
    MemoryStore,
    RECORDS_TABLE,
)
from iai_mcp.types import (
    MemoryRecord,
)

from iai_mcp.migrate import STAGING_TABLE, OLD_TABLE_PREFIX, PROGRESS_FILE


log = logging.getLogger(__name__)


def _records_schema_at_dim(db, dim: int) -> pa.Schema:
    # Staging inherits the LIVE records schema (only the embedding dim
    # changes) — a hand-copied column list silently drifts behind
    # _reconcile_columns additions and breaks the staging insert.
    base = db.open_table(RECORDS_TABLE).schema
    idx = base.get_field_index("embedding")
    return base.set(idx, pa.field("embedding", pa.list_(pa.float32(), dim)))


def _progress_path(store: MemoryStore) -> Path:
    return Path(store.root) / PROGRESS_FILE


def _progress_read(store: MemoryStore) -> dict:
    path = _progress_path(store)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _progress_write(store: MemoryStore, state: dict) -> None:
    target = _progress_path(store)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".migration-progress.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, target)
    except (OSError, ValueError) as exc:
        log.error("progress save failed: %s", exc)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _progress_clear(store: MemoryStore) -> None:
    path = _progress_path(store)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _stage_record_to_table(
    store: MemoryStore,
    target_tbl,
    rec: MemoryRecord,
    new_embedding: list[float],
) -> None:
    if not rec.structure_hv:
        from iai_mcp.tem import bind_structure
        rec.structure_hv = bind_structure(rec)
    new_rec = MemoryRecord(
        id=rec.id,
        tier=rec.tier,
        literal_surface=rec.literal_surface,
        aaak_index=rec.aaak_index,
        embedding=new_embedding,
        structure_hv=rec.structure_hv,
        community_id=rec.community_id,
        centrality=rec.centrality,
        detail_level=rec.detail_level,
        pinned=rec.pinned,
        stability=rec.stability,
        difficulty=rec.difficulty,
        last_reviewed=rec.last_reviewed,
        never_decay=rec.never_decay,
        never_merge=rec.never_merge,
        provenance=rec.provenance,
        created_at=rec.created_at,
        updated_at=rec.updated_at,
        tags=rec.tags,
        language=rec.language,
        s5_trust_score=rec.s5_trust_score,
        profile_modulation_gain=rec.profile_modulation_gain,
        schema_version=rec.schema_version,
    )
    target_tbl.add([store._to_row(new_rec)])


def _stage_loop(
    store: MemoryStore,
    target_embedder,
    target_dim: int,
    target_tbl,
    source_iter,
    *,
    total: int,
    started_at_iso: str,
    started_idx: int = 0,
    already_staged_ids: Optional[set[str]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> tuple[int, list[str]]:
    staged_count = 0
    failures: list[str] = []
    staged_ids: list[str] = list(already_staged_ids or [])
    skipped_set: set[str] = set(staged_ids)

    idx = started_idx
    for rec in source_iter:
        rec_id_str = str(rec.id)
        if rec_id_str in skipped_set:
            continue
        if progress is not None:
            try:
                progress(idx, total)
            except (TypeError, ValueError):
                pass
        try:
            new_embedding = target_embedder.embed(rec.literal_surface)
            _stage_record_to_table(store, target_tbl, rec, new_embedding)
        except (KeyboardInterrupt, SystemExit):
            raise
        except (OSError, ValueError, RuntimeError) as exc:
            log.warning(
                "migrate_reembed_per_row_failed",
                extra={
                    "record_id": rec_id_str,
                    "error": str(exc)[:160],
                },
            )
            failures.append(rec_id_str)
            idx += 1
            continue

        staged_count += 1
        staged_ids.append(rec_id_str)
        _progress_write(
            store,
            {
                "started_at": started_at_iso,
                "ts": int(time.time()),
                "row_index": idx,
                "last_rid": rec_id_str,
                "total": total,
                "target_dim": target_dim,
                "target_model_key": getattr(target_embedder, "model_key", "unknown"),
                "staged_ids": staged_ids,
                "failures": failures,
            },
        )
        idx += 1

    return staged_count, failures


def _swap_tables_filesystem(db, *, source: str, dest: str) -> None:
    from iai_mcp.hippo import HippoDB

    if isinstance(db, HippoDB):
        db._conn.execute(  # nosemgrep
            f"ALTER TABLE [{source}] RENAME TO [{dest}]"
        )
        return
    raise RuntimeError(
        f"re-embed table swap requires a Hippo store, got {type(db).__name__}"
    )


def _validate_and_swap(
    store: MemoryStore,
    *,
    source_dim: int,
    target_dim: int,
    target_embedder,
    staged_count: int,
    failures: list[str],
    duration_sec: float,
) -> dict:
    orig = store.db.open_table(RECORDS_TABLE).count_rows()
    staged = store.db.open_table(STAGING_TABLE).count_rows()
    if orig > 0 and staged < orig * 0.99:
        log.error(
            "migrate_reembed_validate_failed",
            extra={
                "orig": orig,
                "staged": staged,
                "ratio": staged / max(orig, 1),
                "failures": len(failures),
            },
        )
        raise RuntimeError(
            f"reembed staging produced {staged}/{orig} rows "
            f"({staged/max(orig,1):.3%}); refusing to swap. Inspect tables "
            f"manually or run `iai-mcp migrate --rollback`."
        )

    try:
        write_event(
            store,
            kind="migration_reembed",
            data={
                "source_dim": source_dim,
                "target_dim": target_dim,
                "updated": staged_count,
                "duration_sec": duration_sec,
                "target_model_key": getattr(target_embedder, "model_key", "unknown"),
                "failures": len(failures),
            },
            severity="info",
        )
    except (OSError, ValueError, RuntimeError) as exc:
        log.error("migration_reembed event write failed: %s", exc)

    ts = int(time.time())
    old_name = f"{OLD_TABLE_PREFIX}{ts}"
    _swap_tables_filesystem(store.db, source=RECORDS_TABLE, dest=old_name)
    _swap_tables_filesystem(store.db, source=STAGING_TABLE, dest=RECORDS_TABLE)

    store._embed_dim = target_dim

    _progress_clear(store)

    return {
        "source_dim": source_dim,
        "target_dim": target_dim,
        "updated": staged_count,
        "skipped": 0,
        "failures": len(failures),
        "duration_sec": duration_sec,
        "old_table": old_name,
    }


def migrate_reembed_to_current_dim(
    store: MemoryStore,
    target_embedder,
    dry_run: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    t0 = time.time()

    source_dim = int(store.embed_dim)
    target_dim = int(target_embedder.DIM)
    started_at_iso = datetime.now(timezone.utc).isoformat()

    if source_dim == target_dim:
        try:
            write_event(
                store,
                kind="migration_reembed",
                data={
                    "source_dim": source_dim,
                    "target_dim": target_dim,
                    "updated": 0,
                    "no_op": True,
                    "duration_sec": time.time() - t0,
                    "target_model_key": getattr(
                        target_embedder, "model_key", "unknown"
                    ),
                },
                severity="info",
            )
        except (OSError, ValueError, RuntimeError) as exc:
            log.error("migration_reembed no-op event write failed: %s", exc)
        return {
            "source_dim": source_dim,
            "target_dim": target_dim,
            "updated": 0,
            "skipped": store.db.open_table(RECORDS_TABLE).count_rows(),
            "no_op": True,
            "duration_sec": time.time() - t0,
        }

    if dry_run:
        return {
            "source_dim": source_dim,
            "target_dim": target_dim,
            "would_update": store.db.open_table(RECORDS_TABLE).count_rows(),
            "duration_sec": time.time() - t0,
        }

    if STAGING_TABLE in set(store.db.table_names()):
        store.db.drop_table(STAGING_TABLE)
    target_tbl = store.db.create_table(
        STAGING_TABLE, schema=_records_schema_at_dim(store.db, target_dim)
    )

    total = store.db.open_table(RECORDS_TABLE).count_rows()
    source_iter = store.iter_records()
    staged_count, failures = _stage_loop(
        store,
        target_embedder,
        target_dim,
        target_tbl,
        source_iter,
        total=total,
        started_at_iso=started_at_iso,
        progress=progress,
    )

    duration_sec = time.time() - t0
    return _validate_and_swap(
        store,
        source_dim=source_dim,
        target_dim=target_dim,
        target_embedder=target_embedder,
        staged_count=staged_count,
        failures=failures,
        duration_sec=duration_sec,
    )


def detect_partial_migration(db) -> dict:
    names = set(db.table_names())
    has_records = RECORDS_TABLE in names
    has_staging = STAGING_TABLE in names
    old_tables = sorted(n for n in names if n.startswith(OLD_TABLE_PREFIX))

    if not has_staging and not old_tables:
        return {"state": "clean"}

    if has_staging and not has_records and not old_tables:
        return {
            "state": "partial_swap_inconsistent",
            "staging": STAGING_TABLE,
            "old_tables": old_tables,
            "reason": (
                "records_v_new present but neither records nor records_old_<ts> "
                "exist; manual recovery required."
            ),
        }

    if has_staging and has_records:
        return {
            "state": "needs_rollback",
            "old_tables": old_tables,
            "reason": (
                "records_v_new present alongside records — staging did not "
                "complete; recover by dropping records_v_new (rollback) or "
                "resuming from migration_progress.json."
            ),
        }

    if not has_staging and has_records and old_tables:
        return {
            "state": "needs_cleanup",
            "old_tables": old_tables,
            "reason": "successful swap from prior boot; drop old tables.",
        }

    if has_staging and old_tables and not has_records:
        return {
            "state": "needs_rollback",
            "old_tables": old_tables,
            "reason": (
                "records_v_new + records_old_<ts> present, records absent — "
                "swap interrupted between renames; rollback from records_old_<ts>."
            ),
        }

    return {
        "state": "unknown",
        "has_records": has_records,
        "has_staging": has_staging,
        "old_tables": old_tables,
    }


def _rollback(db, store: MemoryStore) -> int:
    names = set(db.table_names())
    has_records = RECORDS_TABLE in names
    has_staging = STAGING_TABLE in names
    old_tables = sorted(n for n in names if n.startswith(OLD_TABLE_PREFIX))

    try:
        if has_staging and has_records:
            db.drop_table(STAGING_TABLE)
            _progress_clear(store)
            log.info(
                "migrate_reembed_rollback_drop_staging",
                extra={"records_count": db.open_table(RECORDS_TABLE).count_rows()},
            )
            return 0

        if not has_records and old_tables:
            newest_old = old_tables[-1]
            if has_staging:
                db.drop_table(STAGING_TABLE)
            _swap_tables_filesystem(db, source=newest_old, dest=RECORDS_TABLE)
            try:
                tbl = db.open_table(RECORDS_TABLE)
                emb_field = tbl.schema.field("embedding")
                actual_dim = getattr(emb_field.type, "list_size", None)
                if actual_dim and int(actual_dim) > 0:
                    store._embed_dim = int(actual_dim)
            except (OSError, ValueError, KeyError, AttributeError) as exc:
                log.error("rollback embed_dim refresh failed: %s", exc)
            _progress_clear(store)
            log.info(
                "migrate_reembed_rollback_restore_old",
                extra={
                    "restored_from": newest_old,
                    "records_count": db.open_table(RECORDS_TABLE).count_rows(),
                },
            )
            return 0

        if has_records and old_tables and not has_staging:
            for old in old_tables:
                try:
                    db.drop_table(old)
                except (OSError, RuntimeError) as exc:
                    log.warning(
                        "migrate_reembed_rollback_drop_old_failed",
                        extra={"table": old, "error": str(exc)[:160]},
                    )
            _progress_clear(store)
            return 0

        if has_records and not has_staging and not old_tables:
            _progress_clear(store)
            return 0

        log.error(
            "migrate_reembed_rollback_unrecoverable",
            extra={
                "has_records": has_records,
                "has_staging": has_staging,
                "old_tables": old_tables,
            },
        )
        return 2
    except (OSError, ValueError, RuntimeError) as exc:
        log.error(
            "migrate_reembed_rollback_failed",
            extra={"error": str(exc)[:200]},
        )
        return 1


def _resume(db, store: MemoryStore, target_embedder) -> int:
    progress_state = _progress_read(store)
    if not progress_state:
        log.error(
            "migrate_reembed_resume_no_progress_file",
            extra={"path": str(_progress_path(store))},
        )
        return 1

    target_dim = int(target_embedder.DIM)
    saved_target_dim = int(progress_state.get("target_dim") or 0)
    if saved_target_dim and saved_target_dim != target_dim:
        log.error(
            "migrate_reembed_resume_dim_mismatch",
            extra={
                "saved_target_dim": saved_target_dim,
                "embedder_dim": target_dim,
            },
        )
        return 1

    names = set(db.table_names())
    if RECORDS_TABLE not in names:
        log.error("migrate_reembed_resume_records_missing")
        return 2

    if STAGING_TABLE not in names:
        target_tbl = db.create_table(
            STAGING_TABLE, schema=_records_schema_at_dim(db, target_dim)
        )
        already_staged: set[str] = set()
    else:
        target_tbl = db.open_table(STAGING_TABLE)
        already_staged = set(progress_state.get("staged_ids") or [])

    source_dim = int(store.embed_dim)
    started_at_iso = progress_state.get(
        "started_at", datetime.now(timezone.utc).isoformat()
    )
    total = db.open_table(RECORDS_TABLE).count_rows()
    last_idx = int(progress_state.get("row_index") or 0)

    t0 = time.time()
    try:
        staged_count, failures = _stage_loop(
            store,
            target_embedder,
            target_dim,
            target_tbl,
            store.iter_records(),
            total=total,
            started_at_iso=started_at_iso,
            started_idx=last_idx + 1,
            already_staged_ids=already_staged,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        log.error(
            "migrate_reembed_resume_stage_failed",
            extra={"error": str(exc)[:200]},
        )
        return 2

    total_staged = len(already_staged) + staged_count

    duration_sec = time.time() - t0
    try:
        _validate_and_swap(
            store,
            source_dim=source_dim,
            target_dim=target_dim,
            target_embedder=target_embedder,
            staged_count=total_staged,
            failures=failures,
            duration_sec=duration_sec,
        )
    except RuntimeError as exc:
        log.error(
            "migrate_reembed_resume_validate_failed",
            extra={"error": str(exc)[:200]},
        )
        return 2
    return 0
