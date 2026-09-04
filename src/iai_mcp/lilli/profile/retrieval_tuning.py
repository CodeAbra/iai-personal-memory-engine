"""Retrieval rank weight tuning: aggregate-window bounded observe/apply plus
its own encrypted `_hippo_meta` persistence, disjoint from the sealed
`profile_state` blob (`persistence.py`).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from cryptography.exceptions import InvalidTag

from iai_mcp.crypto import CryptoKeyError, decrypt_field, encrypt_field, is_encrypted
from iai_mcp.lilli.profile.tuner import RetrievalFeedback, update_retrieval_weights

logger = logging.getLogger(__name__)

RETRIEVAL_MIN_SAMPLES: int = 20
MAX_WEIGHT_DELTA: float = 0.05
PROD_W_COSINE_MIN: float = 0.90
PROD_W_COSINE_MAX: float = 1.10
DEFAULT_W_COSINE: float = 1.0

RETRIEVAL_WEIGHTS_META_KEY: str = "retrieval_weights_state"
RETRIEVAL_WEIGHTS_META_ORPHAN_KEY: str = "retrieval_weights_state.orphan"
RETRIEVAL_WEIGHTS_BLOB_VERSION: int = 1
# Distinct AAD -- never PROFILE_BLOB_AAD -- so a swapped blob cannot misdecrypt
# as the sealed knobs blob.
RETRIEVAL_WEIGHTS_BLOB_AAD: bytes = b"retrieval_weights_state"

_AGGREGATE_SAMPLE_SIZE: int = 1000


def observe_retrieval_weight(window_rows: list[dict]) -> tuple[float, int, dict]:
    """Aggregate a window of (retrieval_reinforced x retrieval_used) session
    joins -- each row `{"hit_ids": [...], "reinforced_ids": [...]}` -- into
    ONE mean use_rate. `n` counts only rows that carry a rate (non-empty
    hit_ids) -- a min_samples gate downstream must not be satisfiable by
    padding a window with signal-free rows.
    """
    rates: list[float] = []
    total_hit_ids = 0
    total_reinforced_ids = 0
    for row in window_rows:
        hit_ids = set(row.get("hit_ids") or [])
        reinforced_ids = set(row.get("reinforced_ids") or [])
        total_hit_ids += len(hit_ids)
        total_reinforced_ids += len(reinforced_ids)
        if hit_ids:
            rates.append(len(hit_ids & reinforced_ids) / len(hit_ids))

    n = len(rates)
    observed_use_rate = sum(rates) / n if rates else 0.0
    return observed_use_rate, n, {
        "total_hit_ids": total_hit_ids,
        "total_reinforced_ids": total_reinforced_ids,
        "rated_rows": n,
    }


def apply_retrieval_weight(current: float, observed_use_rate: float, n: int) -> float:
    """Aggregate the whole window into one bounded, narrow-clamped nightly
    step -- never fold per event (that saturates W_COSINE toward MAX_WEIGHT
    within ~80 events).
    """
    if n < RETRIEVAL_MIN_SAMPLES:
        return max(PROD_W_COSINE_MIN, min(PROD_W_COSINE_MAX, current))

    clamped_rate = max(0.0, min(1.0, observed_use_rate))
    reinforced_count = round(clamped_rate * _AGGREGATE_SAMPLE_SIZE)
    hit_ids = [uuid4() for _ in range(_AGGREGATE_SAMPLE_SIZE)]
    reinforced_ids = hit_ids[:reinforced_count]
    feedback = RetrievalFeedback(
        query_type="_aggregate", hit_ids=hit_ids, used_ids=reinforced_ids,
    )
    proposed = update_retrieval_weights(feedback, {"W_COSINE": current})
    proposed_w_cosine = proposed["W_COSINE"]

    delta = max(-MAX_WEIGHT_DELTA, min(MAX_WEIGHT_DELTA, proposed_w_cosine - current))
    new_value = current + delta
    return max(PROD_W_COSINE_MIN, min(PROD_W_COSINE_MAX, new_value))


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


def save_retrieval_weights_state(store: Any, weights: dict) -> bool:
    db = _hippo_db(store)
    if db is None:
        return False
    payload = {
        "version": RETRIEVAL_WEIGHTS_BLOB_VERSION,
        "weights": dict(weights),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    ct = encrypt_field(
        json.dumps(payload), store._key(), associated_data=RETRIEVAL_WEIGHTS_BLOB_AAD
    )
    preserved_orphan = False
    with db._conn_lock:
        existing = db._conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = ?", (RETRIEVAL_WEIGHTS_META_KEY,)
        ).fetchone()
        if existing is not None:
            existing_value = existing["value"]
            if is_encrypted(existing_value):
                try:
                    decrypt_field(
                        existing_value, store._key(), associated_data=RETRIEVAL_WEIGHTS_BLOB_AAD
                    )
                except (InvalidTag, ValueError, CryptoKeyError):
                    orphan_row = db._conn.execute(
                        "SELECT value FROM _hippo_meta WHERE key = ?",
                        (RETRIEVAL_WEIGHTS_META_ORPHAN_KEY,),
                    ).fetchone()
                    if orphan_row is None:
                        db._conn.execute(
                            "INSERT OR IGNORE INTO _hippo_meta (key, value) VALUES (?, ?)",
                            (RETRIEVAL_WEIGHTS_META_ORPHAN_KEY, existing_value),
                        )
                        db._conn.commit()
                        preserved_orphan = True
        db._conn.execute("DELETE FROM _hippo_meta WHERE key = ?", (RETRIEVAL_WEIGHTS_META_KEY,))
        db._conn.execute(
            "INSERT OR IGNORE INTO _hippo_meta (key, value) VALUES (?, ?)",
            (RETRIEVAL_WEIGHTS_META_KEY, ct),
        )
        db._conn.commit()
    if preserved_orphan:
        logger.warning("retrieval_weights_state_unreadable: preserved_before_overwrite")
    return True


def load_retrieval_weights_state(store: Any) -> dict:
    defaults = {"W_COSINE": DEFAULT_W_COSINE}
    db = _hippo_db(store)
    if db is None:
        return defaults
    with db._conn_lock:
        row = db._conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = ?", (RETRIEVAL_WEIGHTS_META_KEY,)
        ).fetchone()
    if row is None:
        return defaults
    value = row["value"]
    if not is_encrypted(value):
        logger.warning("retrieval_weights_state_unreadable: not_encrypted")
        return defaults
    try:
        plaintext = decrypt_field(value, store._key(), associated_data=RETRIEVAL_WEIGHTS_BLOB_AAD)
    except (InvalidTag, ValueError, CryptoKeyError):
        logger.warning("retrieval_weights_state_unreadable: decrypt_failed")
        return defaults
    try:
        payload = json.loads(plaintext)
    except (ValueError, TypeError):
        logger.warning("retrieval_weights_state_unreadable: invalid_json")
        return defaults
    version = payload.get("version")
    if not isinstance(version, int) or version > RETRIEVAL_WEIGHTS_BLOB_VERSION:
        logger.warning("retrieval_weights_state_unreadable: unsupported_version")
        return defaults
    weights = payload.get("weights")
    if not isinstance(weights, dict) or "W_COSINE" not in weights:
        return defaults
    return weights
