"""Daemon-down recall helpers: recency SQL + read-only ANN fallback."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import hnswlib
import numpy as np

from iai_mcp.types import EMBED_DIM


_DIRECT_RECENCY_SQL = (
    "SELECT"
    " id, tier, literal_surface, aaak_index,"
    " community_id, centrality, detail_level, pinned,"
    " stability, difficulty, last_reviewed, never_decay, never_merge,"
    " provenance_json, created_at, updated_at, tags_json, language,"
    " s5_trust_score, profile_modulation_gain_json, schema_version,"
    " structure_hv, hv_tier, structure_hv_payload,"
    " COALESCE(embedding_pending, 0) AS embedding_pending"
    " FROM records WHERE tombstoned_at IS NULL ORDER BY created_at DESC"
)

_DIRECT_RECENCY_SQL_LIMITED = (
    "SELECT"
    " id, tier, literal_surface, aaak_index,"
    " community_id, centrality, detail_level, pinned,"
    " stability, difficulty, last_reviewed, never_decay, never_merge,"
    " provenance_json, created_at, updated_at, tags_json, language,"
    " s5_trust_score, profile_modulation_gain_json, schema_version,"
    " structure_hv, hv_tier, structure_hv_payload,"
    " COALESCE(embedding_pending, 0) AS embedding_pending"
    " FROM records WHERE tombstoned_at IS NULL ORDER BY created_at DESC"
    " LIMIT ?"
)


def _no_flock_recency_rows_from_store(
    db_path: Path,
    limit: "int | None" = None,
) -> list[dict]:
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        conn.execute("PRAGMA busy_timeout=2000")
        conn.execute("PRAGMA query_only=ON")
        conn.row_factory = sqlite3.Row
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


def load_hnsw_readonly(store_root: "str | Path", embed_dim: int) -> "hnswlib.Index | None":
    hnsw_path = Path(store_root) / "hippo" / "records.hnsw"
    if not hnsw_path.exists():
        return None
    try:
        idx = hnswlib.Index(space="cosine", dim=embed_dim)
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
        labels_arr, _distances = idx.knn_query(cue_np, k=k_actual)
        # Clamp cosine distance to its mathematical range — the BLAS backend
        # can produce sub-epsilon negatives on Linux. Normalized at the source
        # so any future caller reading distances does not re-encounter the bug.
        _distances = [max(0.0, min(2.0, float(d))) for d in _distances[0]]
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
            try:
                if _is_enc(surface):
                    aad = row_id.encode("utf-8")
                    surface = _decrypt_field(surface, _crypto_key, aad)
            except Exception:  # noqa: BLE001 — leave ciphertext if decrypt fails
                pass
        results.append({
            "literal_surface": surface,
            "score": 0.0,
            "_degraded": True,
            "_source": "direct-store",
        })
        if len(results) >= limit:
            break

    return results
