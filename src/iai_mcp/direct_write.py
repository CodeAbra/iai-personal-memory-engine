from __future__ import annotations

import json
import logging
import os
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

log = logging.getLogger(__name__)


def _resolve_store_root() -> Path:
    env = os.environ.get("IAI_MCP_STORE")
    return Path(env) if env else Path.home() / ".iai-mcp"


def write_turn_direct(
    store_root: Path | str | None = None,
    *,
    text: str,
    session_id: str = "-",
    role: str = "user",
    ts_iso: str | None = None,
    deferred_embedding: bool = False,
    cue: str | None = None,
    tier: str = "episodic",
    source_uuid: str | None = None,
) -> dict[str, Any]:
    from iai_mcp.capture import _idem_tag, MIN_CAPTURE_LEN, MAX_CAPTURE_LEN, TIER_ENUM
    from iai_mcp.crypto import CryptoKey
    from iai_mcp.hippo import AccessMode, HippoDB

    root = Path(store_root) if store_root is not None else _resolve_store_root()

    if tier not in TIER_ENUM:
        return {"status": "skipped", "record_id": None, "reason": f"invalid tier {tier!r}"}

    text = (text or "").strip()
    if len(text) < MIN_CAPTURE_LEN:
        return {"status": "skipped", "record_id": None, "reason": "too short"}
    if len(text) > MAX_CAPTURE_LEN:
        text = text[:MAX_CAPTURE_LEN]

    now = datetime.fromisoformat(ts_iso) if ts_iso else datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    ts_norm = now.isoformat()

    idem_t = _idem_tag(session_id, role, ts_norm, text, source_uuid=source_uuid)

    # Thread the SAME at-rest key provider the normal write path uses (see
    # MemoryStore.__init__). Only a lazy callable crosses into the engine —
    # never the key bytes — so _encrypt_for_uuid encrypts literal_surface /
    # provenance at rest exactly like the daemon path. Without this, the
    # provider defaults to None and every encrypt call no-ops to plaintext.
    _crypto_key_wrapper = CryptoKey(user_id="default", store_root=root)

    def _key_provider() -> bytes:
        return _crypto_key_wrapper.get_or_create()

    db = HippoDB(
        root,
        crypto_key_provider=_key_provider,
        access_mode=AccessMode.SHARED,
    )
    try:
        existing_id = _find_record_by_tag_direct(db, idem_t)
        if existing_id is not None:
            return {
                "status": "reinforced",
                "record_id": str(existing_id),
                "reason": "exact-key re-drain",
            }

        record_id = str(uuid4())
        tags = ["capture", f"role:{role}", idem_t]
        tags_json = json.dumps(tags)
        provenance = [{"ts": ts_norm, "cue": cue or "(direct-write)", "session_id": session_id, "role": role}]
        provenance_json = json.dumps(provenance)

        embedding: list[float] | None = None
        if not deferred_embedding:
            try:
                embedding = _try_get_embedding_fast(text, cue or text)
            except Exception:
                embedding = None

        if embedding is not None and len(embedding) == db._embed_dim:
            _insert_row_with_embedding(db, record_id, tier, text, tags_json, provenance_json, ts_norm, ts_norm, embedding)
            _write_sidecar(root, record_id, embedding, db)
        else:
            db.insert_pending_row(
                record_id=record_id,
                tier=tier,
                literal_surface=text,
                tags_json=tags_json,
                provenance_json=provenance_json,
                created_at=ts_norm,
                updated_at=ts_norm,
            )

        return {
            "status": "inserted",
            "record_id": record_id,
            "reason": f"tier={tier}",
        }
    finally:
        db.close()


def _find_record_by_tag_direct(db: Any, tag: str) -> str | None:
    tag_json_literal = json.dumps(tag)
    with db._conn_lock:
        rows = db._conn.execute(
            "SELECT id, tags_json FROM records WHERE tombstoned_at IS NULL"
        ).fetchall()
    for row in rows:
        tags_raw = row["tags_json"] or "[]"
        if tag_json_literal not in tags_raw:
            continue
        try:
            tags = json.loads(tags_raw)
        except (ValueError, TypeError):
            continue
        if tag in tags:
            return row["id"]
    return None


def _try_get_embedding_fast(text: str, cue: str) -> list[float] | None:
    socket_path = os.environ.get("IAI_DAEMON_SOCKET_PATH")
    if socket_path:
        try:
            from iai_mcp._ipc import make_sync_ipc_socket
            s, addr = make_sync_ipc_socket()
            s.settimeout(0.1)
            s.connect(addr)
            s.close()
        except (OSError, ConnectionRefusedError, FileNotFoundError):
            return None
    else:
        return None

    return None


def _insert_row_with_embedding(
    db: Any,
    record_id: str,
    tier: str,
    literal_surface: str,
    tags_json: str,
    provenance_json: str,
    created_at: str,
    updated_at: str,
    embedding: list[float],
) -> None:
    blob = struct.pack(f"<{len(embedding)}f", *embedding)
    # Encrypt the confidential columns at rest with the same uuid-derived
    # associated data the deferred-embed path (insert_pending_row) uses, so an
    # eager-embedded direct write is indistinguishable from the daemon path on
    # disk. No-op when no key provider is configured.
    literal_surface = db._encrypt_for_uuid(record_id, literal_surface)
    provenance_json = db._encrypt_for_uuid(record_id, provenance_json)
    with db._conn_lock:
        db._conn.execute(
            "INSERT INTO records"
            " (id, tier, literal_surface, aaak_index, embedding, embedding_pending,"
            "  provenance_json, created_at, updated_at, tags_json,"
            "  community_id, detail_level, centrality, stability, difficulty,"
            "  pinned, never_decay, never_merge, s5_trust_score,"
            "  schema_version, language,"
            "  hv_tier, structure_hv_payload, live)"
            " VALUES (?, ?, ?, '', ?, 0, ?, ?, ?, ?, '', 1, 0.0, 0.0, 0.0,"
            "  0, 0, 0, 0.5, 1, 'en', 'bsc', x'', 1)",
            (
                record_id,
                tier,
                literal_surface,
                blob,
                provenance_json,
                created_at,
                updated_at,
                tags_json,
            ),
        )
        db._conn.commit()


def _write_sidecar(root: Path, record_id: str, embedding: list[float], db: Any) -> None:
    import re
    if not re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', record_id):
        log.warning("direct_write: sidecar skipped — record_id not uuid4: %r", record_id)
        return

    with db._conn_lock:
        row = db._conn.execute(
            "SELECT vec_label FROM records WHERE id = ?", (record_id,)
        ).fetchone()
    if row is None:
        return
    vec_label = int(row["vec_label"])

    sidecar_dir = root / ".pending-embeddings"
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    blob = struct.pack(f"<{len(embedding)}f", *embedding)
    npy_tmp = sidecar_dir / f"{record_id}.npy.tmp"
    json_tmp = sidecar_dir / f"{record_id}.json.tmp"
    npy_final = sidecar_dir / f"{record_id}.npy"
    json_final = sidecar_dir / f"{record_id}.json"

    try:
        npy_tmp.write_bytes(blob)
        json_tmp.write_text(json.dumps({"uuid": record_id, "vec_label": vec_label}))
        os.replace(npy_tmp, npy_final)
        os.replace(json_tmp, json_final)
    except OSError as exc:
        log.warning("direct_write: sidecar write failed for %s: %s", record_id, exc)
        for p in (npy_tmp, json_tmp):
            try:
                p.unlink()
            except OSError:
                pass


