from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from iai_mcp.embed import Embedder
from iai_mcp.events import write_event
from iai_mcp.store import (
    RECORDS_TABLE,
    MemoryStore,
    _uuid_literal,
)
from iai_mcp.types import (
    SCHEMA_VERSION_CURRENT,
    SCHEMA_VERSION_LEGACY,
    MemoryRecord,
)


log = logging.getLogger(__name__)


STAGING_TABLE = "records_v_new"
OLD_TABLE_PREFIX = "records_old_"
PROGRESS_FILE = "migration_progress.json"
CRYPTO_RECOVER_STAGING = "records_crypto_recover_stage"
REDACT_UNDECRYPTABLE_MARKER = "<REDACTED: pre-2026-04-30 key rotation>"


def _db_table_names_set(db) -> set[str]:
    res = db.list_tables()
    if hasattr(res, "tables"):
        return set(res.tables)
    return set(res)


def migrate_v1_to_v2(
    store: MemoryStore,
    embedder: Optional[Embedder] = None,
    dry_run: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    t0 = time.time()
    if embedder is not None:
        emb = embedder
    else:
        from iai_mcp.embed import embedder_for_store
        emb = embedder_for_store(store)

    all_records = store.all_records()
    v1_records = [r for r in all_records if r.schema_version == SCHEMA_VERSION_LEGACY]
    total = len(v1_records)
    migrated = 0

    for idx, record in enumerate(v1_records):
        if progress is not None:
            try:
                progress(idx, total)
            except (TypeError, ValueError):
                pass

        new_lang = record.language if (record.language and record.language.strip()) else "en"

        if dry_run:
            migrated += 1
            continue

        new_embedding = emb.embed(record.literal_surface)

        updated = MemoryRecord(
            id=record.id,
            tier=record.tier,
            literal_surface=record.literal_surface,
            aaak_index=record.aaak_index,
            embedding=new_embedding,
            structure_hv=record.structure_hv,
            community_id=record.community_id,
            centrality=record.centrality,
            detail_level=record.detail_level,
            pinned=record.pinned,
            stability=record.stability,
            difficulty=record.difficulty,
            last_reviewed=record.last_reviewed,
            never_decay=record.never_decay,
            never_merge=record.never_merge,
            provenance=record.provenance,
            created_at=record.created_at,
            updated_at=record.updated_at,
            tags=record.tags,
            language=new_lang,
            s5_trust_score=0.5,
            profile_modulation_gain={},
            schema_version=SCHEMA_VERSION_CURRENT,
        )
        tbl = store.db.open_table(RECORDS_TABLE)
        tbl.delete(f"id = '{_uuid_literal(record.id)}'")
        store.insert(updated)
        migrated += 1

    duration_sec = time.time() - t0

    if not dry_run and migrated > 0:
        write_event(
            store,
            kind="migration_v1_to_v2",
            data={
                "record_count": migrated,
                "duration_sec": duration_sec,
            },
            severity="info",
        )

    return {
        "records_migrated": migrated,
        "skipped": max(0, len(all_records) - total),
        "duration_sec": duration_sec,
        "previous_model": "bge-small-en-v1.5",
        "new_model": emb.model_key,
    }


from iai_mcp.migrate._reembed import (  # noqa: E402
    migrate_reembed_to_current_dim,
    detect_partial_migration,
    _rollback,
    _resume,
    _stage_loop,
    _stage_record_to_table,
    _records_schema_at_dim,
    _validate_and_swap,
    _swap_tables_filesystem,
    _progress_path,
    _progress_read,
    _progress_write,
    _progress_clear,
)
from iai_mcp.migrate._crypto_mig import (  # noqa: E402
    migrate_encryption_v2_to_v3,
    migrate_crypto_recover_prior_key,
    migrate_redact_undecryptable_records,
    _encrypt_or_passthrough,
    _decrypt_field_try_keys,
    _memory_record_from_raw_row_multikey,
)
from iai_mcp.migrate._hv_codec import (  # noqa: E402
    migrate_hd_vector_to_structure_hv_v3_to_v4,
    migrate_codec_metadata_v4_to_v5,
    _migrate_add_hv_tier_columns,
)
from iai_mcp.migrate._cleanup import cleanup_schema_duplicates  # noqa: E402
from iai_mcp.migrate._timestamps import (  # noqa: E402
    migrate_rederive_collapsed_timestamps,
    _find_transcript_ts,
)
from iai_mcp.migrate._dedupe import migrate_dedupe_episodic_captures  # noqa: E402
from iai_mcp.migrate._salvage import migrate_salvage_torn_permanent_failed  # noqa: E402
from iai_mcp.migrate._reembed_from_text import migrate_reembed_from_text  # noqa: E402
from iai_mcp.migrate._dead_pid_unlock import (  # noqa: E402
    migrate_unlock_dead_pid_processing_files,
)
from iai_mcp.migrate._to_lilli_verify import (  # noqa: E402
    verify_store_equality,
    VerifyReport,
)
from iai_mcp.migrate._to_lilli import (  # noqa: E402
    migrate_sqlite_to_lilli,
    MigrateReport,
)
from iai_mcp.migrate._role_column import migrate_role_column  # noqa: E402
from iai_mcp.migrate._live_flag_backfill import migrate_live_flag_backfill  # noqa: E402


__all__ = [
    "STAGING_TABLE",
    "OLD_TABLE_PREFIX",
    "PROGRESS_FILE",
    "CRYPTO_RECOVER_STAGING",
    "REDACT_UNDECRYPTABLE_MARKER",
    "migrate_v1_to_v2",
    "migrate_reembed_to_current_dim",
    "detect_partial_migration",
    "migrate_encryption_v2_to_v3",
    "migrate_crypto_recover_prior_key",
    "migrate_redact_undecryptable_records",
    "migrate_hd_vector_to_structure_hv_v3_to_v4",
    "migrate_codec_metadata_v4_to_v5",
    "cleanup_schema_duplicates",
    "migrate_rederive_collapsed_timestamps",
    "migrate_dedupe_episodic_captures",
    "migrate_salvage_torn_permanent_failed",
    "migrate_reembed_from_text",
    "migrate_unlock_dead_pid_processing_files",
    "verify_store_equality",
    "VerifyReport",
    "migrate_sqlite_to_lilli",
    "MigrateReport",
    "migrate_role_column",
    "migrate_live_flag_backfill",
]
