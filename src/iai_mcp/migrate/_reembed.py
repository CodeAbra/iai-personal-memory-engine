from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

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


def _create_canonical_staging(db):
    """Create the staging table from the CANONICAL records DDL, not from a
    pyarrow-derived column list. Arrow schemas carry names and types but no
    SQL constraints, so an arrow-derived staging table loses
    ``vec_label INTEGER PRIMARY KEY AUTOINCREMENT`` and ``id UNIQUE`` — after
    the swap, inserts leave ``vec_label`` NULL (the rowid alias is gone) and
    the next label-map load crashes the daemon into a restart loop. The SQL
    DDL is embedding-dim independent (the column is a BLOB), so the canonical
    text is correct for any target dim; live columns added after the DDL was
    written are aligned by the same reconcile pass the boot path uses."""
    from iai_mcp.hippo._table import _DDL_RECORDS, _pa_type_to_sqlite

    ddl = _DDL_RECORDS.replace(
        "CREATE TABLE IF NOT EXISTS records (",
        f"CREATE TABLE IF NOT EXISTS {STAGING_TABLE} (",
        1,
    )
    from iai_mcp.hippo._db import _txn

    with db._conn_lock, _txn(db._conn):
        db._conn.execute(ddl)

    live_schema = db.open_table(RECORDS_TABLE).schema
    drift = [
        (f.name, _pa_type_to_sqlite(f.type)) for f in live_schema
    ]
    db._reconcile_columns(STAGING_TABLE, drift)
    return db.open_table(STAGING_TABLE)


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
        with os.fdopen(fd, "w", encoding="utf-8") as f:
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
    # Copy the authoritative SQL row byte-for-byte and replace only the vector.
    # Rebuilding from MemoryRecord would drop storage-only fields such as
    # vec_label, tombstones and pending state, and would decrypt/re-encrypt text.
    with store.db.ro_conn() as conn:
        source_row = conn.execute(
            "SELECT * FROM records WHERE id = ?", (str(rec.id),)
        ).fetchone()
    if source_row is None:
        raise ValueError(f"source record disappeared during re-embedding: {rec.id}")
    staged_row = dict(source_row)
    staged_row["embedding"] = new_embedding
    target_tbl.add([staged_row])


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
    batch_size: int = 256,
) -> tuple[int, list[str]]:
    staged_count = 0
    failures: list[str] = []
    skipped_set: set[str] = set(already_staged_ids or [])

    idx = started_idx
    pending_batch: list[MemoryRecord] = []

    def flush_batch(batch: list[MemoryRecord]) -> None:
        nonlocal idx, staged_count
        if not batch:
            return
        try:
            batch_method = getattr(target_embedder, "embed_batch", None)
            if callable(batch_method):
                vectors = batch_method([rec.literal_surface for rec in batch])
            else:
                vectors = [target_embedder.embed(rec.literal_surface) for rec in batch]
            if len(vectors) != len(batch):
                raise ValueError(
                    f"embed_batch returned {len(vectors)} vectors for {len(batch)} records"
                )
        except (KeyboardInterrupt, SystemExit):
            raise
        except (OSError, ValueError, RuntimeError) as exc:
            batch_ids = [str(rec.id) for rec in batch]
            log.warning(
                "migrate_reembed_batch_failed",
                extra={"record_ids": batch_ids, "error": str(exc)[:160]},
            )
            failures.extend(batch_ids)
            idx += len(batch)
            return

        last_staged_id: str | None = None
        for rec, new_embedding in zip(batch, vectors, strict=True):
            rec_id_str = str(rec.id)
            if progress is not None:
                try:
                    progress(idx, total)
                except (TypeError, ValueError):
                    pass
            try:
                _stage_record_to_table(store, target_tbl, rec, new_embedding)
            except (KeyboardInterrupt, SystemExit):
                raise
            except (OSError, ValueError, RuntimeError) as exc:
                log.warning(
                    "migrate_reembed_per_row_failed",
                    extra={"record_id": rec_id_str, "error": str(exc)[:160]},
                )
                failures.append(rec_id_str)
                idx += 1
                continue

            staged_count += 1
            last_staged_id = rec_id_str
            idx += 1

        # One fsynced checkpoint per provider batch keeps the migration
        # crash-safe without rewriting a growing JSON list for every row.
        # Resume reconstructs the authoritative staged-id set from the table.
        if last_staged_id is not None:
            _progress_write(
                store,
                {
                    "started_at": started_at_iso,
                    "ts": int(time.time()),
                    "row_index": idx - 1,
                    "last_rid": last_staged_id,
                    "total": total,
                    "target_dim": target_dim,
                    "target_model_key": getattr(
                        target_embedder, "model_key", "unknown"
                    ),
                    "failures": failures,
                },
            )

    effective_batch_size = (
        max(1, int(batch_size))
        if bool(getattr(target_embedder, "supports_batch", False))
        else 1
    )
    for rec in source_iter:
        if str(rec.id) in skipped_set:
            continue
        pending_batch.append(rec)
        if len(pending_batch) >= effective_batch_size:
            flush_batch(pending_batch)
            pending_batch = []
    flush_batch(pending_batch)

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
    if staged != orig or failures:
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
            f"reembed staging produced {staged}/{orig} rows with "
            f"{len(failures)} failures; refusing to swap. Inspect tables "
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

    from iai_mcp.hippo import HippoDB, _txn

    if not isinstance(store.db, HippoDB):
        raise RuntimeError(
            f"re-embed table swap requires a Hippo store, got {type(store.db).__name__}"
        )
    ts = int(time.time())
    old_name = f"{OLD_TABLE_PREFIX}{ts}"
    # The two renames and dimension metadata are one storage transition. A
    # crash can therefore expose either the complete old layout or the
    # complete new layout, never a 1024d table advertised as 384d.
    with store.db._conn_lock:
        with _txn(store.db._conn):
            store.db._conn.execute(
                f"ALTER TABLE [{RECORDS_TABLE}] RENAME TO [{old_name}]"
            )
            store.db._conn.execute(
                f"ALTER TABLE [{STAGING_TABLE}] RENAME TO [{RECORDS_TABLE}]"
            )
            store.db._conn.execute(
                "UPDATE [_hippo_meta] SET value = ? WHERE key = 'embed_dim'",
                (str(target_dim),),
            )
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
        "restart_required": True,
    }


def migrate_reembed_to_current_dim(
    store: MemoryStore,
    target_embedder,
    dry_run: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
    *,
    force: bool = False,
    batch_size: int = 256,
) -> dict:
    t0 = time.time()

    # A caller may migrate in the same process that just captured records.
    # Make the buffered writes authoritative before counting or staging them;
    # otherwise a small corpus can be swapped as an apparently empty table.
    from iai_mcp.store import flush_record_buffer

    flush_record_buffer(store)

    source_dim = int(store.embed_dim)
    target_dim = int(target_embedder.DIM)
    started_at_iso = datetime.now(timezone.utc).isoformat()

    if source_dim == target_dim and not force:
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
    target_tbl = _create_canonical_staging(store.db)

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
        batch_size=batch_size,
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
        target_tbl = _create_canonical_staging(db)
        already_staged: set[str] = set()
    else:
        target_tbl = db.open_table(STAGING_TABLE)
        with db.ro_conn() as conn:
            already_staged = {
                str(row[0])
                for row in conn.execute(f"SELECT id FROM [{STAGING_TABLE}]").fetchall()
            }

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
