from __future__ import annotations

import logging
from iai_mcp import errors
import time
from typing import Callable, Optional

from iai_mcp.events import write_event
from iai_mcp.store import (
    MemoryStore,
    RECORDS_TABLE,
    _uuid_literal,
)


log = logging.getLogger(__name__)


def migrate_hd_vector_to_structure_hv_v3_to_v4(
    store: MemoryStore,
    dry_run: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    t0 = time.time()
    result: dict = {
        "processed": 0,
        "updated": 0,
        "skipped": 0,
        "duration_ms": 0.0,
        "column_renamed_from": "hd_vector_json",
        "column_renamed_to": "structure_hv",
    }

    all_records = store.all_records()
    total = len(all_records)
    result["processed"] = total

    from iai_mcp.tem import bind_structure
    from iai_mcp.types import (
        SCHEMA_VERSION_V4,
        STRUCTURE_HV_BYTES,
    )

    tbl = store.db.open_table(RECORDS_TABLE)
    for idx, record in enumerate(all_records):
        if progress is not None:
            try:
                progress(idx, total)
            except (TypeError, ValueError):
                pass

        already_v4 = record.schema_version >= SCHEMA_VERSION_V4
        has_full_hv = (
            isinstance(record.structure_hv, (bytes, bytearray))
            and len(record.structure_hv) == STRUCTURE_HV_BYTES
        )
        if already_v4 and has_full_hv:
            result["skipped"] += 1
            continue

        if dry_run:
            result["updated"] += 1
            continue

        if not has_full_hv:
            record.structure_hv = bind_structure(record)
        record.schema_version = SCHEMA_VERSION_V4

        try:
            tbl.delete(f"id = '{_uuid_literal(record.id)}'")
        except (OSError, ValueError, RuntimeError):
            pass
        store.insert(record)
        result["updated"] += 1

    result["duration_ms"] = (time.time() - t0) * 1000.0

    if not dry_run and (result["updated"] > 0 or result["skipped"] > 0):
        write_event(
            store,
            kind="migration_v3_to_v4",
            data={
                "processed": result["processed"],
                "updated": result["updated"],
                "skipped": result["skipped"],
                "duration_ms": result["duration_ms"],
                "column_renamed_from": result["column_renamed_from"],
                "column_renamed_to": result["column_renamed_to"],
            },
            severity="info",
        )

    return result


def _migrate_add_hv_tier_columns(conn) -> dict:
    result = {"hv_tier_added": False, "structure_hv_payload_added": False}

    try:
        conn.execute(
            "ALTER TABLE records ADD COLUMN hv_tier TEXT NOT NULL DEFAULT 'bsc'"
        )
        result["hv_tier_added"] = True
    except errors.OperationalError as exc:
        if "duplicate column name" not in str(exc).lower():
            raise

    try:
        conn.execute(
            "ALTER TABLE records ADD COLUMN structure_hv_payload BLOB NOT NULL DEFAULT x''"
        )
        result["structure_hv_payload_added"] = True
    except errors.OperationalError as exc:
        if "duplicate column name" not in str(exc).lower():
            raise

    return result


def migrate_codec_metadata_v4_to_v5(
    store: "MemoryStore",
    dry_run: bool = False,
) -> dict:
    from iai_mcp.hippo import HippoDB

    db = store.db
    if not isinstance(db, HippoDB):
        return {"dry_run": dry_run, "hv_tier_added": False, "structure_hv_payload_added": False, "note": "non-hippo backend; skipped"}

    existing = {
        row["name"]
        for row in db._conn.execute("PRAGMA table_info(records)").fetchall()
    }
    needs_hv_tier = "hv_tier" not in existing
    needs_payload = "structure_hv_payload" not in existing

    if dry_run:
        return {
            "dry_run": True,
            "hv_tier_added": needs_hv_tier,
            "structure_hv_payload_added": needs_payload,
        }

    result = _migrate_add_hv_tier_columns(db._conn)
    result["dry_run"] = False
    return result
