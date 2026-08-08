from __future__ import annotations

import json
import logging
from iai_mcp import errors
import time
from datetime import datetime, timezone
from typing import Callable, Optional
from uuid import UUID

from iai_mcp.crypto import encrypt_field, is_encrypted
from iai_mcp.events import write_event
from iai_mcp.store import (
    EVENTS_TABLE,
    MemoryStore,
    RECORDS_TABLE,
    _uuid_literal,
)
from iai_mcp.types import (
    SCHEMA_VERSION_CURRENT,
    MemoryRecord,
)

from iai_mcp.migrate import (
    CRYPTO_RECOVER_STAGING,
    OLD_TABLE_PREFIX,
    REDACT_UNDECRYPTABLE_MARKER,
    _db_table_names_set,
    _swap_tables_filesystem,
    detect_partial_migration,
)


log = logging.getLogger(__name__)


def _decrypt_field_try_keys(
    ciphertext: str,
    record_id: UUID,
    keys: list[bytes],
) -> str:
    from cryptography.exceptions import InvalidTag

    from iai_mcp.crypto import decrypt_field

    if not is_encrypted(ciphertext):
        return str(ciphertext or "")
    ad = _uuid_literal(record_id).encode("ascii")
    last_exc: Exception | None = None
    for key in keys:
        if key is None or len(key) != 32:
            continue
        try:
            return decrypt_field(ciphertext, key, associated_data=ad)
        except (InvalidTag, ValueError) as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    raise ValueError("no valid keys supplied for decrypt")


def _memory_record_from_raw_row_multikey(
    store: MemoryStore,
    row: dict,
    keys: list[bytes],
) -> MemoryRecord:
    import pandas as pd

    from uuid import UUID as _UUID

    row_uuid = _UUID(row["id"])
    structure_raw = row.get("structure_hv")
    if structure_raw is None:
        structure_hv = b""
    elif isinstance(structure_raw, (bytes, bytearray)):
        structure_hv = bytes(structure_raw)
    else:
        structure_hv = b""

    community_raw = row.get("community_id") or ""
    community_id = _UUID(community_raw) if community_raw else None

    raw_version = row.get("schema_version")
    try:
        version_int = int(raw_version) if raw_version is not None else SCHEMA_VERSION_CURRENT
    except (TypeError, ValueError):
        version_int = SCHEMA_VERSION_CURRENT
    schema_version = version_int

    lang_raw = row.get("language")
    is_empty_language = lang_raw is None or (isinstance(lang_raw, str) and lang_raw == "")
    if is_empty_language and schema_version == 1:
        language = "__LEGACY_EMPTY__"
    elif is_empty_language:
        language = "en"
    else:
        language = str(lang_raw)

    s5_raw = row.get("s5_trust_score")
    s5_trust_score = float(s5_raw) if s5_raw is not None else 0.5

    gain_raw = row.get("profile_modulation_gain_json") or "{}"
    gain_plain = _decrypt_field_try_keys(str(gain_raw), row_uuid, keys)
    try:
        profile_modulation_gain = json.loads(gain_plain) or {}
    except (TypeError, json.JSONDecodeError):
        profile_modulation_gain = {}

    last_reviewed_raw = row.get("last_reviewed")
    try:
        last_reviewed = None if pd.isna(last_reviewed_raw) else last_reviewed_raw
    except (TypeError, ValueError):
        last_reviewed = last_reviewed_raw

    literal_raw = row.get("literal_surface", "")
    literal_plain = _decrypt_field_try_keys(str(literal_raw), row_uuid, keys)

    provenance_raw = row.get("provenance_json") or "[]"
    provenance_plain = _decrypt_field_try_keys(str(provenance_raw), row_uuid, keys)
    try:
        provenance_list = json.loads(provenance_plain) if provenance_plain else []
    except (TypeError, json.JSONDecodeError):
        provenance_list = []

    rec = MemoryRecord(
        id=row_uuid,
        tier=row.get("tier", "episodic"),
        literal_surface=literal_plain,
        aaak_index=row.get("aaak_index") or "",
        embedding=(
            list(row["embedding"])
            if row.get("embedding") is not None
            else []
        ),
        community_id=community_id,
        centrality=float(row.get("centrality", 0.0) or 0.0),
        detail_level=int(row.get("detail_level", 1)),
        pinned=bool(row.get("pinned", False)),
        stability=float(row.get("stability") or 0.0),
        difficulty=float(row.get("difficulty") or 0.0),
        last_reviewed=last_reviewed,
        never_decay=bool(row.get("never_decay", False)),
        never_merge=bool(row.get("never_merge", False)),
        provenance=provenance_list,
        created_at=row.get("created_at") or datetime.now(timezone.utc),
        updated_at=row.get("updated_at") or datetime.now(timezone.utc),
        tags=json.loads(row.get("tags_json") or "[]"),
        language=language,
        s5_trust_score=s5_trust_score,
        profile_modulation_gain=profile_modulation_gain,
        schema_version=schema_version,
        structure_hv=structure_hv,
    )
    if language == "__LEGACY_EMPTY__":
        rec.language = ""
    return rec


def migrate_crypto_recover_prior_key(
    store: MemoryStore,
    prior_key: bytes,
    *,
    dry_run: bool = False,
) -> dict:
    from cryptography.exceptions import InvalidTag

    from iai_mcp.crypto import KEY_BYTES

    # One key or several retained generations (a `.crypto.key.pre-rotate`
    # sidecar concatenates KEY_BYTES blocks) — every generation is tried.
    if isinstance(prior_key, (list, tuple)):
        prior_generations = list(prior_key)
    else:
        prior_generations = [prior_key]
    if not prior_generations or any(
        len(k) != KEY_BYTES for k in prior_generations
    ):
        raise ValueError(
            f"each prior key must be {KEY_BYTES} raw bytes"
        )

    mig = detect_partial_migration(store.db)
    if mig["state"] not in ("clean", "needs_cleanup"):
        raise RuntimeError(
            "crypto recover requires a non-partial reembed state "
            f"(got {mig['state']!r}); resolve migrate --rollback/--resume first."
        )

    cur_key = store._key()
    key_chain = [cur_key] + [k for k in prior_generations if k != cur_key]

    names = _db_table_names_set(store.db)
    if CRYPTO_RECOVER_STAGING in names:
        try:
            store.db.drop_table(CRYPTO_RECOVER_STAGING)
        except (OSError, ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"drop stale {CRYPTO_RECOVER_STAGING} failed: {exc}"
            ) from exc

    orig_tbl = store.db.open_table(RECORDS_TABLE)
    orig_count = int(orig_tbl.count_rows())
    if orig_count == 0:
        return {"no_op": True, "reason": "empty_store", "records_staged": 0, "dry_run": dry_run}

    df = orig_tbl.to_pandas()
    needs_prior = 0
    for _, r in df.iterrows():
        rid = UUID(str(r["id"]))
        lit = str(r.get("literal_surface") or "")
        if not is_encrypted(lit):
            continue
        try:
            _decrypt_field_try_keys(lit, rid, [cur_key])
        except (InvalidTag, ValueError):
            try:
                _decrypt_field_try_keys(lit, rid, prior_generations)
                needs_prior += 1
            except (InvalidTag, ValueError):
                raise RuntimeError(
                    f"record {rid}: literal_surface not decryptable with the "
                    "current key or any prior generation — run crypto "
                    "redact-undecryptable (rows the current key or a retained "
                    "generation opens are preserved or re-keyed; only rows "
                    "nothing opens are redacted) or restore backup"
                ) from None

    if needs_prior == 0:
        return {
            "no_op": True,
            "reason": "all_rows_decrypt_with_current_key",
            "records_staged": 0,
            "dry_run": dry_run,
        }

    if dry_run:
        return {
            "no_op": False,
            "dry_run": True,
            "would_stage": orig_count,
            "rows_needing_prior_key": needs_prior,
        }

    schema = orig_tbl.schema
    staging_tbl = store.db.create_table(CRYPTO_RECOVER_STAGING, schema=schema)
    staged = 0
    t0 = time.time()
    for _, r in df.iterrows():
        row_dict = r.to_dict()
        rec = _memory_record_from_raw_row_multikey(store, row_dict, key_chain)
        staging_tbl.add([store._to_row(rec)])
        staged += 1

    if staged != orig_count:
        try:
            store.db.drop_table(CRYPTO_RECOVER_STAGING)
        except (OSError, RuntimeError) as exc:
            log.error("failed to drop staging table after mismatch: %s", exc)
        raise RuntimeError(
            f"staging row count mismatch: staged={staged} orig={orig_count}"
        )

    duration_sec = time.time() - t0
    try:
        write_event(
            store,
            kind="migration_crypto_recover",
            data={
                "records_staged": staged,
                "duration_sec": duration_sec,
                "rows_needed_prior_key": needs_prior,
            },
            severity="info",
        )
    except (OSError, ValueError, RuntimeError) as exc:
        log.error("migration_crypto_recover event write failed: %s", exc)

    ts = int(time.time())
    old_name = f"{OLD_TABLE_PREFIX}{ts}"
    _swap_tables_filesystem(store.db, source=RECORDS_TABLE, dest=old_name)
    _swap_tables_filesystem(
        store.db, source=CRYPTO_RECOVER_STAGING, dest=RECORDS_TABLE
    )

    return {
        "no_op": False,
        "records_staged": staged,
        "duration_sec": duration_sec,
        "dry_run": False,
        "old_table": old_name,
        "rows_needed_prior_key": needs_prior,
    }


def _retained_generations(store: MemoryStore) -> list[bytes]:
    """Key generations retained by a partial rotation. Redaction is a
    terminal verb: a row one of these still opens must be RE-KEYED, never
    destroyed while its key sits on disk."""
    from iai_mcp.crypto import KEY_BYTES

    import os

    try:
        sidecar = store._crypto_key_wrapper._key_file_path().with_name(
            ".crypto.key.pre-rotate"
        )
        if not sidecar.exists():
            return []
        if os.name != "nt":
            st = os.stat(sidecar)
            if st.st_mode & 0o077 != 0:
                # Fail LOUD: silently ignoring the sidecar would route
                # recoverable rows into redaction.
                raise RuntimeError(
                    f"retained key file at {sidecar} has insecure mode "
                    f"0o{st.st_mode & 0o777:03o}; run: chmod 600 {sidecar}"
                )
        raw = sidecar.read_bytes()
    except OSError:
        return []
    return [
        raw[i:i + KEY_BYTES]
        for i in range(0, len(raw) - len(raw) % KEY_BYTES, KEY_BYTES)
    ]


def migrate_redact_undecryptable_records(store: MemoryStore) -> dict:
    tbl = store.db.open_table(RECORDS_TABLE)
    if tbl.count_rows() == 0:
        return {
            "redacted": 0, "skipped_ok": 0, "skipped_plain": 0,
            "records_rekeyed": 0,
        }

    retained = _retained_generations(store)
    cur_key = store._key()
    key_chain = [cur_key] + [k for k in retained if k != cur_key]
    df = tbl.to_pandas()
    redacted = 0
    skipped_ok = 0
    skipped_plain = 0
    records_rekeyed = 0
    for _, r in df.iterrows():
        rid = UUID(str(r["id"]))
        lit = str(r.get("literal_surface") or "")
        if not is_encrypted(lit):
            skipped_plain += 1
            continue
        plain = store._decrypt_for_record_or_none(rid, lit)
        if plain is not None:
            skipped_ok += 1
            continue
        prov_raw = str(r.get("provenance_json") or "[]")
        gain_raw = str(r.get("profile_modulation_gain_json") or "{}")
        if retained:
            healed = None
            try:
                healed = _decrypt_field_try_keys(lit, rid, retained)
            except Exception:  # noqa: BLE001 -- opens under no retained generation either
                healed = None
            if healed is not None:
                try:
                    prov_plain = (
                        _decrypt_field_try_keys(prov_raw, rid, key_chain)
                        if is_encrypted(prov_raw) else prov_raw
                    )
                except Exception:  # noqa: BLE001 -- companion column beyond every generation
                    prov_plain = "[]"
                try:
                    gain_plain = (
                        _decrypt_field_try_keys(gain_raw, rid, key_chain)
                        if is_encrypted(gain_raw) else gain_raw
                    )
                except Exception:  # noqa: BLE001 -- companion column beyond every generation
                    gain_plain = "{}"
                tbl.update(
                    where=f"id = '{_uuid_literal(rid)}'",
                    values={
                        "literal_surface": store._encrypt_for_record(
                            rid, healed
                        ),
                        "provenance_json": store._encrypt_for_record(
                            rid, prov_plain
                        ),
                        "profile_modulation_gain_json": (
                            store._encrypt_for_record(rid, gain_plain)
                        ),
                        "updated_at": datetime.now(timezone.utc),
                    },
                )
                records_rekeyed += 1
                continue
        # Companion columns get the same courtesy as the literal: every
        # available generation is tried before content is dropped.
        if is_encrypted(prov_raw):
            try:
                prov_plain = _decrypt_field_try_keys(prov_raw, rid, key_chain)
            except Exception:  # noqa: BLE001 -- opens under no available generation
                prov_plain = "[]"
        else:
            prov_plain = prov_raw
        if is_encrypted(gain_raw):
            try:
                gain_plain = _decrypt_field_try_keys(gain_raw, rid, key_chain)
            except Exception:  # noqa: BLE001 -- opens under no available generation
                gain_plain = "{}"
        else:
            gain_plain = gain_raw
        new_lit = store._encrypt_for_record(rid, REDACT_UNDECRYPTABLE_MARKER)
        new_prov = store._encrypt_for_record(rid, prov_plain)
        new_gain = store._encrypt_for_record(rid, gain_plain)
        tbl.update(
            where=f"id = '{_uuid_literal(rid)}'",
            values={
                "literal_surface": new_lit,
                "provenance_json": new_prov,
                "profile_modulation_gain_json": new_gain,
                "updated_at": datetime.now(timezone.utc),
            },
        )
        redacted += 1
        try:
            write_event(
                store,
                kind="crypto_redaction",
                data={"record_id": str(rid), "reason": "undecryptable_literal"},
                severity="warning",
            )
        except (OSError, ValueError, RuntimeError) as exc:
            log.error("crypto_redaction event write failed: %s", exc)

    return {
        "redacted": redacted,
        "skipped_ok": skipped_ok,
        "skipped_plain": skipped_plain,
        "records_rekeyed": records_rekeyed,
    }


def migrate_redact_undecryptable_events(store: MemoryStore) -> dict:
    """Terminal escape for event rows that decrypt under no available key
    generation — current OR retained. A row a retained generation still
    opens is re-keyed to the current key, never redacted; only then is the
    payload replaced with a marker so the next `crypto rotate` can report
    clean instead of staying `partial` forever."""
    from iai_mcp.crypto import decrypt_field

    tbl = store.db.open_table(EVENTS_TABLE)
    df = tbl.to_pandas()
    events_redacted = 0
    events_ok = 0
    events_plain = 0
    events_rekeyed = 0
    retained = _retained_generations(store)
    cur_key = store._key()
    for _, r in df.iterrows():
        raw = str(r.get("data_json") or "{}")
        if not is_encrypted(raw):
            events_plain += 1
            continue
        eid = str(r["id"])
        ad = eid.encode("ascii")
        try:
            decrypt_field(raw, cur_key, associated_data=ad)
            events_ok += 1
            continue
        except Exception:  # noqa: BLE001 -- undecryptable under the current key
            pass
        healed = None
        for k in retained:
            try:
                healed = decrypt_field(raw, k, associated_data=ad)
                break
            except Exception:  # noqa: BLE001 -- try the next retained generation
                continue
        if healed is not None:
            new_ct = encrypt_field(healed, cur_key, associated_data=ad)
            try:
                tbl.update(where=f"id = '{eid}'", values={"data_json": new_ct})
                events_rekeyed += 1
            except (OSError, ValueError, RuntimeError) as exc:
                log.error("event re-key update failed id=%s: %s", eid, exc)
            continue
        marker = json.dumps({"redacted": REDACT_UNDECRYPTABLE_MARKER})
        new_ct = encrypt_field(marker, cur_key, associated_data=ad)
        try:
            tbl.update(where=f"id = '{eid}'", values={"data_json": new_ct})
            events_redacted += 1
        except (OSError, ValueError, RuntimeError) as exc:
            log.error("event redaction update failed id=%s: %s", eid, exc)
    if events_redacted:
        try:
            write_event(
                store,
                kind="crypto_redaction",
                data={
                    "events_redacted": events_redacted,
                    "reason": "undecryptable_event_payload",
                },
                severity="warning",
            )
        except (OSError, ValueError, RuntimeError) as exc:
            log.error("crypto_redaction event write failed: %s", exc)
    return {
        "events_redacted": events_redacted,
        "events_ok": events_ok,
        "events_plain": events_plain,
        "events_rekeyed": events_rekeyed,
    }


def _encrypt_or_passthrough(
    store: MemoryStore,
    record_id: UUID,
    value: str,
) -> tuple[str, bool]:
    if is_encrypted(value):
        return value, False
    ad = _uuid_literal(record_id).encode("ascii")
    ct = encrypt_field(value or "", store._key(), associated_data=ad)
    return ct, True


def migrate_encryption_v2_to_v3(
    store: MemoryStore,
    dry_run: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    t0 = time.time()
    result = {
        "records_migrated": 0,
        "events_migrated": 0,
        "records_scanned": 0,
        "events_scanned": 0,
        "duration_sec": 0.0,
    }

    records_tbl = store.db.open_table(RECORDS_TABLE)
    records_df = records_tbl.to_pandas()
    result["records_scanned"] = int(len(records_df))

    records_updates: list[dict] = []
    record_total = len(records_df)
    for idx, (_, row) in enumerate(records_df.iterrows()):
        if progress is not None:
            try:
                progress(idx, record_total)
            except (TypeError, ValueError):
                pass
        try:
            rid = UUID(str(row["id"]))
        except (ValueError, TypeError):
            continue

        literal_raw = row.get("literal_surface") or ""
        prov_raw = row.get("provenance_json") or "[]"
        gain_raw = row.get("profile_modulation_gain_json") or "{}"

        any_plaintext = any(
            not is_encrypted(v) for v in (literal_raw, prov_raw, gain_raw)
        )
        if not any_plaintext:
            continue

        if dry_run:
            result["records_migrated"] += 1
            continue

        new_literal, _ = _encrypt_or_passthrough(store, rid, literal_raw)
        new_prov, _ = _encrypt_or_passthrough(store, rid, prov_raw)
        new_gain, _ = _encrypt_or_passthrough(store, rid, gain_raw)
        records_updates.append(
            {
                "id": _uuid_literal(rid),
                "literal_surface": new_literal,
                "provenance_json": new_prov,
                "profile_modulation_gain_json": new_gain,
            }
        )
        result["records_migrated"] += 1

    if not dry_run and records_updates:
        now = datetime.now(timezone.utc)
        from iai_mcp.hippo import _schema
        update_tbl = _schema.table(
            {
                "id": [u["id"] for u in records_updates],
                "literal_surface": [u["literal_surface"] for u in records_updates],
                "provenance_json": [u["provenance_json"] for u in records_updates],
                "profile_modulation_gain_json": [
                    u["profile_modulation_gain_json"] for u in records_updates
                ],
                "updated_at": [now] * len(records_updates),
            }
        )
        try:
            records_tbl.merge_insert("id").when_matched_update_all().execute(update_tbl)
        except (OSError, ValueError, AttributeError, RuntimeError, errors.IntegrityError) as exc:
            log.error("merge_insert fallback triggered: %s", exc)
            for u in records_updates:
                try:
                    records_tbl.update(
                        where=f"id = '{u['id']}'",
                        values={
                            "literal_surface": u["literal_surface"],
                            "provenance_json": u["provenance_json"],
                            "profile_modulation_gain_json": u[
                                "profile_modulation_gain_json"
                            ],
                            "updated_at": now,
                        },
                    )
                except (OSError, ValueError, RuntimeError):
                    continue

    events_tbl = store.db.open_table(EVENTS_TABLE)
    events_df = events_tbl.to_pandas()
    result["events_scanned"] = int(len(events_df))

    events_updates: list[dict] = []
    for _, row in events_df.iterrows():
        data_raw = row.get("data_json") or "{}"
        if is_encrypted(data_raw):
            continue
        event_id = str(row["id"])
        if dry_run:
            result["events_migrated"] += 1
            continue
        ad = event_id.encode("ascii")
        new_data = encrypt_field(data_raw, store._key(), associated_data=ad)
        events_updates.append({"id": event_id, "data_json": new_data})
        result["events_migrated"] += 1

    if not dry_run and events_updates:
        for u in events_updates:
            try:
                events_tbl.update(
                    where=f"id = '{u['id']}'",
                    values={"data_json": u["data_json"]},
                )
            except (OSError, ValueError, RuntimeError):
                continue

    result["duration_sec"] = time.time() - t0

    if not dry_run and (
        result["records_migrated"] > 0 or result["events_migrated"] > 0
    ):
        write_event(
            store,
            kind="migration_v2_to_v3",
            data={
                "record_count": result["records_migrated"],
                "event_count": result["events_migrated"],
                "duration_sec": result["duration_sec"],
                "columns_encrypted": [
                    "records.literal_surface",
                    "records.provenance_json",
                    "records.profile_modulation_gain_json",
                    "events.data_json",
                ],
                "algorithm": "AES-256-GCM",
                "format": "iai:enc:v1:",
            },
            severity="info",
        )

    return result
