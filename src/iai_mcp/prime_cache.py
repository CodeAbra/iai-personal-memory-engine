"""Priming cache: what usually comes next, read from `_hippo_meta`.

Built nightly from `proc_transitions` (PK src,dst,source) into one encrypted
blob with two sub-dicts -- `seed_to_chunks` (record_id -> list[chunk_id],
LIST because one src can start several transitions) and `chunk_members`
(chunk_id -> [src, dst]). Tombstoned chunks are excluded by construction --
build() filters against the records table's tombstoned_at column, never
priming toward a dead chunk. Cached as a per-store-instance attribute; the
blob changes at most nightly, so per-store-generation lifetime is correct.
`invalidate(store)` forces a re-read after a persist.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from cryptography.exceptions import InvalidTag

from iai_mcp.crypto import CryptoKeyError, decrypt_field, encrypt_field, is_encrypted
from iai_mcp.lilli.profile.persistence import _hippo_db

PRIME_CACHE_META_KEY = "prime_cache_v1"
PRIME_CACHE_AAD = b"prime_cache"
PRIME_CACHE_BLOB_VERSION = 1

_CACHE_ATTR: str = "_prime_cache"


def _dead_chunk_ids(db: Any, chunk_ids: set[str]) -> set[str]:
    """chunk_ids whose records row is tombstoned -- the same liveness
    predicate chunk.py's _live_transition_chunk_id uses (tombstoned_at IS
    NULL means live). Chunked IN-windows, no MIN() in SELECT: both drivers.
    """
    dead: set[str] = set()
    ordered = sorted(chunk_ids)
    for start in range(0, len(ordered), 400):
        window = ordered[start : start + 400]
        placeholders = ", ".join("?" for _ in window)
        for r in db._conn.execute(
            "SELECT id FROM records"  # nosemgrep: sql-injection
            f" WHERE tombstoned_at IS NOT NULL AND id IN ({placeholders})",
            tuple(window),
        ).fetchall():
            dead.add(r["id"])
    return dead


def build(store: Any) -> dict[str, dict]:
    db = _hippo_db(store)
    if db is None:
        return {"seed_to_chunks": {}, "chunk_members": {}}
    with db._conn_lock:
        rows = db._conn.execute(
            "SELECT src, dst, source, chunk_id, count, session_count "
            "FROM proc_transitions",
        ).fetchall()
        chunk_ids = {row["chunk_id"] for row in rows if row["chunk_id"]}
        dead_chunk_ids = _dead_chunk_ids(db, chunk_ids) if chunk_ids else set()
    seed_to_chunks: dict[str, list[str]] = {}
    chunk_members: dict[str, list[str]] = {}
    for row in rows:
        chunk_id = row["chunk_id"]
        if not chunk_id or chunk_id in dead_chunk_ids:
            continue
        src, dst = row["src"], row["dst"]
        bucket = seed_to_chunks.setdefault(src, [])
        if chunk_id not in bucket:
            bucket.append(chunk_id)
        chunk_members[chunk_id] = [src, dst]
    return {"seed_to_chunks": seed_to_chunks, "chunk_members": chunk_members}


def save(store: Any, blob: dict[str, dict]) -> bool:
    db = _hippo_db(store)
    if db is None:
        return False
    payload = {
        "version": PRIME_CACHE_BLOB_VERSION,
        "seed_to_chunks": blob.get("seed_to_chunks", {}),
        "chunk_members": blob.get("chunk_members", {}),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    ct = encrypt_field(
        json.dumps(payload), store._key(), associated_data=PRIME_CACHE_AAD,
    )
    with db._conn_lock:
        db._conn.execute(
            "DELETE FROM _hippo_meta WHERE key = ?", (PRIME_CACHE_META_KEY,),
        )
        db._conn.execute(
            "INSERT OR IGNORE INTO _hippo_meta (key, value) VALUES (?, ?)",
            (PRIME_CACHE_META_KEY, ct),
        )
        db._conn.commit()
    return True


def load(store: Any) -> dict[str, dict]:
    cached = getattr(store, _CACHE_ATTR, None)
    if cached is not None:
        return cached
    result = _load_uncached(store)
    try:
        setattr(store, _CACHE_ATTR, result)
    except (AttributeError, TypeError):
        pass
    return result


def _load_uncached(store: Any) -> dict[str, dict]:
    # Boot-warmup calls this -- like retrieval_weight_cache.load(), it must
    # never raise. The whole read (raw DB fetch included, not just
    # decrypt/parse) is inside the guard.
    try:
        db = _hippo_db(store)
        if db is None:
            return {}
        with db._conn_lock:
            row = db._conn.execute(
                "SELECT value FROM _hippo_meta WHERE key = ?",
                (PRIME_CACHE_META_KEY,),
            ).fetchone()
        if row is None:
            return {}
        value = row["value"]
        if not is_encrypted(value):
            return {}
        try:
            plaintext = decrypt_field(
                value, store._key(), associated_data=PRIME_CACHE_AAD,
            )
        except (InvalidTag, ValueError, CryptoKeyError):
            # Fail-safe: never overwrite ciphertext this process cannot open
            # -- a miss here is silent, not destructive.
            return {}
        try:
            payload = json.loads(plaintext)
        except (ValueError, TypeError):
            return {}
        return {
            "seed_to_chunks": payload.get("seed_to_chunks", {}),
            "chunk_members": payload.get("chunk_members", {}),
        }
    except Exception:  # noqa: BLE001 -- the recall boot path must never raise on this read
        return {}


def invalidate(store: Any) -> None:
    try:
        if hasattr(store, _CACHE_ATTR):
            delattr(store, _CACHE_ATTR)
    except (AttributeError, TypeError):
        pass
