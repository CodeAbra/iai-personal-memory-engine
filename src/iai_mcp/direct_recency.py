from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)


def _parse_ts(val: Any) -> datetime | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val if val.tzinfo is not None else val.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _decrypt_if_needed(
    value: str | None,
    record_uuid_bytes: bytes,
    key: bytes | None,
) -> str:
    if not value:
        return ""
    from iai_mcp.crypto import is_encrypted, decrypt_field

    if not is_encrypted(value):
        return value
    if key is None:
        return value
    try:
        return decrypt_field(value, key, associated_data=record_uuid_bytes)
    except Exception:  # noqa: BLE001
        return ""


def read_recent_user_turns_direct(
    store_root: Path | str,
    n: int = 10,
    session_id: str | None = None,
) -> list:
    from iai_mcp.hippo import direct_recency_rows_from_store

    root = Path(store_root)

    rows = direct_recency_rows_from_store(root)
    if not rows:
        return []

    key: bytes | None = None
    try:
        from iai_mcp.crypto import CryptoKey

        ck = CryptoKey(user_id="default", store_root=root)
        key = ck.get_or_create()
    except Exception:  # noqa: BLE001
        pass

    return _build_records(rows, key=key, n=n, session_id=session_id)


def _build_records(
    rows: list[dict],
    *,
    key: bytes | None,
    n: int,
    session_id: str | None,
) -> list:
    from iai_mcp.types import MemoryRecord

    results: list[MemoryRecord] = []
    for row in rows:
        row_dict = dict(row)
        try:
            rec = _row_to_record(row_dict, key=key)
        except Exception:  # noqa: BLE001
            continue

        if rec.tier != "episodic":
            continue
        if "role:user" not in (rec.tags or []) and not rec.embedding_pending:
            continue

        if session_id is not None:
            prov_session = (rec.provenance or [{}])[0].get("session_id")
            if prov_session != session_id:
                continue

        results.append(rec)
        if session_id is None and len(results) >= n:
            break

    results.sort(
        key=lambda r: r.created_at or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return results[:n]


def _row_to_record(row: dict, *, key: bytes | None) -> Any:
    from iai_mcp.types import MemoryRecord, SCHEMA_VERSION_CURRENT, HV_TIER_ENUM

    row_id = row["id"]
    row_uuid = UUID(row_id)
    row_uuid_bytes = row_id.lower().encode("ascii")

    literal_raw = _decrypt_if_needed(row.get("literal_surface") or "", row_uuid_bytes, key)
    provenance_raw = _decrypt_if_needed(row.get("provenance_json") or "[]", row_uuid_bytes, key)
    gain_raw_enc = row.get("profile_modulation_gain_json") or "{}"
    gain_raw = _decrypt_if_needed(gain_raw_enc, row_uuid_bytes, key)

    try:
        provenance_list: list[dict] = json.loads(provenance_raw) if provenance_raw else []
    except (TypeError, json.JSONDecodeError):
        provenance_list = []
    try:
        profile_modulation_gain: dict = json.loads(gain_raw) or {}
    except (TypeError, json.JSONDecodeError):
        profile_modulation_gain = {}
    try:
        tags: list[str] = json.loads(row.get("tags_json") or "[]")
    except (TypeError, json.JSONDecodeError):
        tags = []

    community_raw = row.get("community_id") or ""
    community_id = UUID(community_raw) if community_raw else None

    lang_raw = row.get("language")
    raw_version = row.get("schema_version")
    try:
        schema_version = int(raw_version) if raw_version is not None else SCHEMA_VERSION_CURRENT
    except (TypeError, ValueError):
        schema_version = SCHEMA_VERSION_CURRENT
    is_empty_language = lang_raw is None or (isinstance(lang_raw, str) and lang_raw == "")
    if is_empty_language and schema_version == 1:
        language = "__LEGACY_EMPTY__"
    elif is_empty_language:
        language = "en"
    else:
        language = str(lang_raw)

    hv_tier_raw = row.get("hv_tier")
    structure_hv_payload_raw = row.get("structure_hv_payload")
    if hv_tier_raw is None or hv_tier_raw not in HV_TIER_ENUM:
        hv_tier = "bsc"
        structure_hv_payload = b""
    elif structure_hv_payload_raw is not None and not isinstance(
        structure_hv_payload_raw, (bytes, bytearray)
    ):
        hv_tier = "bsc"
        structure_hv_payload = b""
    else:
        hv_tier = str(hv_tier_raw)
        structure_hv_payload = (
            bytes(structure_hv_payload_raw)
            if isinstance(structure_hv_payload_raw, (bytes, bytearray))
            else b""
        )

    s5_raw = row.get("s5_trust_score")
    s5_trust_score = float(s5_raw) if s5_raw is not None else 0.5

    rec = MemoryRecord(
        id=row_uuid,
        tier=row.get("tier", "episodic"),
        literal_surface=literal_raw,
        aaak_index=row.get("aaak_index") or "",
        embedding=[],
        community_id=community_id,
        centrality=float(row.get("centrality", 0.0) or 0.0),
        detail_level=int(row.get("detail_level") or 1),
        pinned=bool(row.get("pinned") or False),
        stability=float(row.get("stability") or 0.0),
        difficulty=float(row.get("difficulty") or 0.0),
        last_reviewed=_parse_ts(row.get("last_reviewed")),
        never_decay=bool(row.get("never_decay") or False),
        never_merge=bool(row.get("never_merge") or False),
        provenance=provenance_list,
        created_at=_parse_ts(row.get("created_at")) or datetime.now(timezone.utc),
        updated_at=_parse_ts(row.get("updated_at")) or datetime.now(timezone.utc),
        tags=tags,
        language=language,
        s5_trust_score=s5_trust_score,
        profile_modulation_gain=profile_modulation_gain,
        schema_version=schema_version,
        structure_hv=b"",
        hv_tier=hv_tier,
        structure_hv_payload=structure_hv_payload,
        embedding_pending=int(row.get("embedding_pending") or 0),
        salience_level=row.get("salience_level") or "unflagged",
    )
    if language == "__LEGACY_EMPTY__":
        rec.language = ""
    return rec
