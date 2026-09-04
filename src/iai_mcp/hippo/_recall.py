"""Daemon-down recall helpers: recency SQL + read-only ANN fallback."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from iai_mcp import _sqlite_stdlib

from iai_mcp.hippo import _vecindex
import numpy as np

from iai_mcp.types import EMBED_DIM

logger = logging.getLogger(__name__)


def _aad_for_id(row_id: str) -> bytes:
    """Derive the AES AAD for a stored id, identical to the write side.

    The write path lowercases the canonical UUID before encoding as ASCII
    (store `_uuid_literal(id).encode("ascii")`, engine `uuid_str.lower()
    .encode("ascii")`). The degraded read rail MUST mirror that normalization
    or a non-lowercased id would fail to decrypt.
    """
    return row_id.lower().encode("ascii")


def _decrypt_degraded_surface(
    row_id: str,
    surface: str,
    crypto_key: bytes,
    decrypt_field,
) -> "str | None":
    """Decrypt a degraded-rail literal_surface, fail-loud on failure.

    Returns the plaintext on success, or None when decryption fails — in which
    case the caller MUST skip the row. A decrypt failure NEVER surfaces raw
    ciphertext as user-facing content — decrypt failure is fail-loud, never
    silent ciphertext-as-content.
    """
    try:
        return decrypt_field(surface, crypto_key, _aad_for_id(row_id))
    except Exception as exc:  # noqa: BLE001 — undecryptable: skip, never surface ciphertext
        logger.warning(
            "degraded_recall_decrypt_failed",
            extra={"id": row_id, "err": f"{type(exc).__name__}: {exc}"[:160]},
        )
        return None


def _parse_ts(value: str) -> datetime:
    """Parse a UTC timestamp string (T-form or space-form) to an aware datetime."""
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00").replace(" ", "T"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


_DIRECT_RECENCY_SQL = (
    "SELECT"
    " id, tier, literal_surface, aaak_index,"
    " community_id, centrality, detail_level, pinned,"
    " stability, difficulty, last_reviewed, never_decay, never_merge,"
    " provenance_json, created_at, updated_at, tags_json, language,"
    " s5_trust_score, profile_modulation_gain_json, schema_version,"
    " hv_tier, structure_hv_payload, salience_level,"
    " COALESCE(embedding_pending, 0) AS embedding_pending"
    " FROM records WHERE live = 1 ORDER BY created_at DESC"
)

_DIRECT_RECENCY_SQL_LIMITED = (
    "SELECT"
    " id, tier, literal_surface, aaak_index,"
    " community_id, centrality, detail_level, pinned,"
    " stability, difficulty, last_reviewed, never_decay, never_merge,"
    " provenance_json, created_at, updated_at, tags_json, language,"
    " s5_trust_score, profile_modulation_gain_json, schema_version,"
    " hv_tier, structure_hv_payload, salience_level,"
    " COALESCE(embedding_pending, 0) AS embedding_pending"
    " FROM records WHERE live = 1 ORDER BY created_at DESC"
    " LIMIT ?"
)

_DIRECT_TEMPORAL_SQL = (
    "SELECT"
    " id, tier, literal_surface, aaak_index,"
    " community_id, centrality, detail_level, pinned,"
    " stability, difficulty, last_reviewed, never_decay, never_merge,"
    " provenance_json, created_at, updated_at, tags_json, language,"
    " s5_trust_score, profile_modulation_gain_json, schema_version,"
    " hv_tier, structure_hv_payload,"
    " COALESCE(embedding_pending, 0) AS embedding_pending"
    " FROM records"
    " WHERE (tombstoned_at IS NULL OR datetime(tombstoned_at) > datetime(?))"
    "   AND datetime(created_at) <= datetime(?)"
    " ORDER BY datetime(created_at) DESC"
    " LIMIT ?"
)


def _no_flock_recency_rows_from_store(
    db_path: Path,
    limit: "int | None" = None,
) -> list[dict]:
    from iai_mcp.hippo._raw_open import open_store_conn
    conn = None
    try:
        _eng = open_store_conn(db_path, read_only=True)
        if _eng is not None:
            conn = _eng
        else:
            conn = _sqlite_stdlib.connect(
                str(db_path),
                check_same_thread=False,
                isolation_level=None,
            )
            conn.row_factory = _sqlite_stdlib.Row
        conn.execute("PRAGMA busy_timeout=2000")
        conn.execute("PRAGMA query_only=ON")
        if limit is not None:
            cursor = conn.execute(_DIRECT_RECENCY_SQL_LIMITED, (limit,))
        else:
            cursor = conn.execute(_DIRECT_RECENCY_SQL)
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    except Exception:  # noqa: BLE001
        return []
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass


def reconcile_index_mid_run(hippo: "HippoDB") -> dict:
    return hippo._rebuild_index_from_sqlite()


def direct_recency_rows_from_store(
    store_root: "Path | str",
    limit: "int | None" = None,
) -> list[dict]:
    from iai_mcp.hippo import AccessMode, HippoDB
    root = Path(store_root)
    db_path = root / "hippo" / "brain.sqlite3"
    if not db_path.exists():
        return []

    db: "HippoDB | None" = None
    try:
        db = HippoDB(
            root,
            access_mode=AccessMode.SHARED,
            read_only=True,
            _lock_timeout_override=0.20,
        )
        with db._conn_lock:
            if limit is not None:
                cursor = db._conn.execute(_DIRECT_RECENCY_SQL_LIMITED, (limit,))
            else:
                cursor = db._conn.execute(_DIRECT_RECENCY_SQL)
            rows = cursor.fetchall()
        return [dict(r) for r in rows]
    except Exception:  # noqa: BLE001 — fall through to no-flock fallback
        pass
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:  # noqa: BLE001
                pass

    return _no_flock_recency_rows_from_store(db_path, limit=limit)


def load_hnsw_readonly(store_root: "str | Path", embed_dim: int) -> "_vecindex.Index | None":
    hnsw_path = Path(store_root) / "hippo" / "records.hnsw"
    if not hnsw_path.exists():
        return None
    try:
        idx = _vecindex.Index(space="cosine", dim=embed_dim)
        idx.load_index(str(hnsw_path), max_elements=0)
        idx.set_ef(200)
        idx.set_num_threads(1)
        return idx
    except Exception:  # noqa: BLE001 — corrupt or incompatible index
        return None


def _ann_lookup_client(
    store_root: "str | Path",
    cue_vec: "list[float]",
    *,
    k: int = 10,
    embed_dim: int = EMBED_DIM,
) -> "list[int]":
    idx = load_hnsw_readonly(store_root, embed_dim)
    if idx is None or idx.get_current_count() == 0:
        return []
    try:
        k_actual = min(k, idx.get_current_count())
        cue_np = np.array(cue_vec, dtype=np.float32).reshape(1, -1)
        labels_arr, _ = idx.knn_query(cue_np, k=k_actual)
        return [int(lbl) for lbl in labels_arr[0]]
    except Exception:  # noqa: BLE001 — index incompatible or corrupted
        return []


def degraded_semantic_recall(
    store_root: "str | Path",
    cue: str,
    limit: int = 10,
    *,
    session_id: "str | None" = None,
) -> "list[dict]":
    from iai_mcp.hippo import AccessMode, HippoDB
    root = Path(store_root)

    db: "HippoDB | None" = None
    try:
        db = HippoDB(
            root,
            access_mode=AccessMode.SHARED,
            read_only=True,
            _lock_timeout_override=0.25,
        )
        with db._conn_lock:
            rows = db._conn.execute(_DIRECT_RECENCY_SQL_LIMITED, (limit,)).fetchall()
        row_dicts = [dict(r) for r in rows]
    except Exception:  # noqa: BLE001
        row_dicts = []
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:  # noqa: BLE001
                pass

    if not row_dicts:
        row_dicts = direct_recency_rows_from_store(root, limit=limit)

    _crypto_key: "bytes | None" = None
    try:
        from iai_mcp.crypto import CryptoKey as _CryptoKey
        _crypto_key = _CryptoKey(store_root=root).get_or_create()
    except Exception:  # noqa: BLE001 — no key available: leave ciphertext as-is
        pass

    try:
        from iai_mcp.crypto import decrypt_field as _decrypt_field, is_encrypted as _is_enc
    except Exception:  # noqa: BLE001
        _decrypt_field = None  # type: ignore[assignment]
        _is_enc = None  # type: ignore[assignment]

    seen_ids: set[str] = set()
    results: list[dict] = []
    for row in row_dicts:
        row_id = str(row.get("id") or "")
        if row_id in seen_ids:
            continue
        seen_ids.add(row_id)
        surface = row.get("literal_surface") or ""
        if surface and _crypto_key is not None and _is_enc is not None and _decrypt_field is not None:
            if _is_enc(surface):
                decrypted = _decrypt_degraded_surface(
                    row_id, surface, _crypto_key, _decrypt_field
                )
                if decrypted is None:
                    # Fail-loud: skip the undecryptable row rather than return
                    # raw ciphertext as if it were content.
                    continue
                surface = decrypted
        results.append({
            "literal_surface": surface,
            "score": 0.0,
            "_degraded": True,
            "_source": "direct-store",
        })
        if len(results) >= limit:
            break

    return results


def degraded_temporal_recall(
    store_root: "str | Path",
    cue: str,
    as_of: str,
    limit: int = 10,
    *,
    session_id: "str | None" = None,
) -> "list[dict]":
    """Time-bounded daemon-down recall.

    cue is accepted for API parity; v1 returns time-bounded recency rows without
    cue ranking on the direct rail. as_of MUST be a canonical UTC ISO string
    (caller normalizes upstream). Returns dicts tagged _source: 'direct-store'.
    """
    from iai_mcp.hippo import AccessMode, HippoDB
    root = Path(store_root)

    db: "HippoDB | None" = None
    row_dicts: list[dict] = []
    shared_lock_open_failed = False
    try:
        db = HippoDB(
            root,
            access_mode=AccessMode.SHARED,
            read_only=True,
            _lock_timeout_override=0.25,
        )
        with db._conn_lock:
            rows = db._conn.execute(
                _DIRECT_TEMPORAL_SQL, (as_of, as_of, limit),
            ).fetchall()
        row_dicts = [dict(r) for r in rows]
    except Exception:  # noqa: BLE001
        shared_lock_open_failed = True
        row_dicts = []
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:  # noqa: BLE001
                pass

    if shared_lock_open_failed or not row_dicts:
        fallback_rows = direct_recency_rows_from_store(root, limit=limit)
        _as_of_dt = _parse_ts(as_of)

        def _created_at_lte(row: dict) -> bool:
            val = row.get("created_at")
            if not val:
                return False
            try:
                return _parse_ts(str(val)) <= _as_of_dt
            except (TypeError, ValueError):
                return False

        row_dicts = [r for r in fallback_rows if _created_at_lte(r)]

    _crypto_key: "bytes | None" = None
    try:
        from iai_mcp.crypto import CryptoKey as _CryptoKey
        _crypto_key = _CryptoKey(store_root=root).get_or_create()
    except Exception:  # noqa: BLE001 — no key available: leave ciphertext as-is
        pass

    try:
        from iai_mcp.crypto import decrypt_field as _decrypt_field, is_encrypted as _is_enc
    except Exception:  # noqa: BLE001
        _decrypt_field = None  # type: ignore[assignment]
        _is_enc = None  # type: ignore[assignment]

    seen_ids: set[str] = set()
    results: list[dict] = []
    for row in row_dicts:
        row_id = str(row.get("id") or "")
        if row_id in seen_ids:
            continue
        seen_ids.add(row_id)
        surface = row.get("literal_surface") or ""
        if surface and _crypto_key is not None and _is_enc is not None and _decrypt_field is not None:
            if _is_enc(surface):
                decrypted = _decrypt_degraded_surface(
                    row_id, surface, _crypto_key, _decrypt_field
                )
                if decrypted is None:
                    # Fail-loud: skip the undecryptable row rather than return
                    # raw ciphertext as if it were content.
                    continue
                surface = decrypted
        results.append({
            "literal_surface": surface,
            "score": 0.0,
            "_degraded": True,
            "_source": "direct-store",
        })
        if len(results) >= limit:
            break

    return results
