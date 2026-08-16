"""Durable home for the autistic-cognition profile: one AES-256-GCM
encrypted JSON blob in the store's ``_hippo_meta`` table, plus in-place
hydration of the live process mappings from it.

Single-writer: the daemon is the only process that calls
``save_profile_state``. Every reader (the CLI, tests, the brain view) only
loads. A miss therefore never writes a default blob -- it leaves the caller's
in-memory state untouched.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from cryptography.exceptions import InvalidTag

from iai_mcp.crypto import CryptoKeyError, decrypt_field, encrypt_field, is_encrypted
from iai_mcp.lilli.profile.knobs import PROFILE_KNOBS, _validate, default_state

logger = logging.getLogger(__name__)

PROFILE_META_KEY = "profile_state"
PROFILE_META_ORPHAN_KEY = "profile_state.orphan"
PROFILE_BLOB_VERSION = 1
PROFILE_BLOB_AAD = b"profile_state"


def _hippo_db(store: Any) -> "object | None":
    db = getattr(store, "db", None)
    if db is None:
        return None
    try:
        from iai_mcp.hippo import HippoDB
    except ImportError:
        return None
    if not isinstance(db, HippoDB):
        return None
    return db


def save_profile_state(store: Any, *, knobs: dict, posterior: dict, pins: dict) -> bool:
    db = _hippo_db(store)
    if db is None:
        return False
    payload = {
        "version": PROFILE_BLOB_VERSION,
        "knobs": dict(knobs),
        "posterior": dict(posterior),
        "pins": dict(pins),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    ct = encrypt_field(
        json.dumps(payload), store._key(), associated_data=PROFILE_BLOB_AAD
    )
    preserved_orphan = False
    with db._conn_lock:
        existing = db._conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = ?", (PROFILE_META_KEY,)
        ).fetchone()
        if existing is not None:
            existing_value = existing["value"]
            if is_encrypted(existing_value):
                try:
                    decrypt_field(
                        existing_value, store._key(), associated_data=PROFILE_BLOB_AAD
                    )
                except (InvalidTag, ValueError, CryptoKeyError):
                    orphan_row = db._conn.execute(
                        "SELECT value FROM _hippo_meta WHERE key = ?",
                        (PROFILE_META_ORPHAN_KEY,),
                    ).fetchone()
                    if orphan_row is None:
                        db._conn.execute(
                            "INSERT OR IGNORE INTO _hippo_meta (key, value) VALUES (?, ?)",
                            (PROFILE_META_ORPHAN_KEY, existing_value),
                        )
                        db._conn.commit()
                        preserved_orphan = True
        db._conn.execute("DELETE FROM _hippo_meta WHERE key = ?", (PROFILE_META_KEY,))
        db._conn.execute(
            "INSERT OR IGNORE INTO _hippo_meta (key, value) VALUES (?, ?)",
            (PROFILE_META_KEY, ct),
        )
        db._conn.commit()
    if preserved_orphan:
        _warn_unreadable(store, "preserved_before_overwrite")
    return True


def reencrypt_profile_blob(store: Any, old_keys: list[bytes], new_key: bytes) -> str:
    """Re-encrypt the durable profile blob to `new_key`, trying each of
    `old_keys` to open it. Returns 'rotated' | 'absent' | 'stranded'. Never
    overwrites ciphertext it cannot open -- a decrypt failure across every
    supplied key leaves the row untouched."""
    db = _hippo_db(store)
    if db is None:
        return "absent"
    with db._conn_lock:
        row = db._conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = ?", (PROFILE_META_KEY,)
        ).fetchone()
        if row is None:
            return "absent"
        raw = row["value"]
        if not is_encrypted(raw):
            return "stranded"
        plaintext = None
        for k in old_keys:
            try:
                plaintext = decrypt_field(raw, k, associated_data=PROFILE_BLOB_AAD)
                break
            except (InvalidTag, ValueError, CryptoKeyError):
                continue
        if plaintext is None:
            return "stranded"
        ct = encrypt_field(plaintext, new_key, associated_data=PROFILE_BLOB_AAD)
        db._conn.execute("DELETE FROM _hippo_meta WHERE key = ?", (PROFILE_META_KEY,))
        db._conn.execute(
            "INSERT OR IGNORE INTO _hippo_meta (key, value) VALUES (?, ?)",
            (PROFILE_META_KEY, ct),
        )
        db._conn.commit()
        return "rotated"


def _warn_unreadable(store: Any, reason: str) -> None:
    try:
        from iai_mcp.events import write_event

        write_event(
            store,
            "profile_state_unreadable",
            {"reason": reason},
            severity="warning",
        )
    except Exception:  # noqa: BLE001 -- the audit write must not mask the original failure
        logger.debug("profile_state_unreadable event write failed", exc_info=True)


def persist_after_user_set(store: Any, state: dict, knob: str) -> bool:
    """Persist the whole live profile and mark `knob` as user-pinned. Read-modify-write of the
    durable blob, preserving accumulated posterior. Daemon-process writer only."""
    blob = load_profile_state(store) or {}
    pins = dict(blob.get("pins", {}))
    pins[knob] = datetime.now(timezone.utc).isoformat()
    return save_profile_state(
        store, knobs=dict(state), posterior=blob.get("posterior", {}), pins=pins,
    )


def _filter_registry_members(raw: dict, dropped: list) -> dict:
    out: dict = {}
    for name, value in raw.items():
        if name not in PROFILE_KNOBS:
            dropped.append(name)
            continue
        out[name] = value
    return out


def load_profile_state(store: Any) -> "dict | None":
    db = _hippo_db(store)
    if db is None:
        return None
    with db._conn_lock:
        row = db._conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = ?",
            (PROFILE_META_KEY,),
        ).fetchone()
    if row is None:
        return None
    value = row["value"]
    if not is_encrypted(value):
        _warn_unreadable(store, "not_encrypted")
        return None
    try:
        plaintext = decrypt_field(value, store._key(), associated_data=PROFILE_BLOB_AAD)
    except (InvalidTag, ValueError, CryptoKeyError):
        _warn_unreadable(store, "decrypt_failed")
        return None
    try:
        payload = json.loads(plaintext)
    except (ValueError, TypeError):
        _warn_unreadable(store, "invalid_json")
        return None
    version = payload.get("version")
    if not isinstance(version, int) or version > PROFILE_BLOB_VERSION:
        _warn_unreadable(store, "unsupported_version")
        return None

    dropped: list[str] = []
    knobs_out: dict = {}
    for name, value in dict(payload.get("knobs") or {}).items():
        spec = PROFILE_KNOBS.get(name)
        if spec is None:
            dropped.append(name)
            continue
        ok, _reason = _validate(spec.value_schema, value)
        if not ok:
            dropped.append(name)
            continue
        knobs_out[name] = value

    posterior_out = _filter_registry_members(dict(payload.get("posterior") or {}), dropped)
    pins_out = _filter_registry_members(dict(payload.get("pins") or {}), dropped)

    return {
        "knobs": knobs_out,
        "posterior": posterior_out,
        "pins": pins_out,
        "dropped": dropped,
        "updated_at": payload.get("updated_at"),
    }


def hydrate_profile(store: Any, state: dict, posterior: dict) -> dict:
    """Load the stored profile into *state* / *posterior* IN PLACE.

    Read-only w.r.t. the store. Never rebinds either mapping -- callers that
    alias ``state`` (e.g. ``core.LIVE_KNOBS``) must keep seeing the same
    object after this call.
    """
    blob = load_profile_state(store)
    if blob is None:
        return {"hydrated": False, "dropped": []}
    merged = default_state()
    merged.update(blob["knobs"])
    state.clear()
    state.update(merged)
    posterior.clear()
    posterior.update(blob["posterior"])
    return {"hydrated": True, "dropped": blob["dropped"], "pins": blob["pins"]}
