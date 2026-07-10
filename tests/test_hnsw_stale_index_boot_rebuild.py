"""Boot self-heal of a stale/undersized on-disk ANN index.

A store's on-disk ``records.hnsw`` can fall behind live SQLite: the index file
holds fewer elements than the live record count (the SQLite table is the source
of truth). When such a store is reopened, the boot-integrity check MUST detect
the divergence from the *loaded index element count* and rebuild the index from
SQLite, so recall serves every live record — not just the subset that happened
to be in the stale file.

These tests reproduce that skew hermetically in ``tmp_path`` (never the real
store or the live daemon): seed a healthy store, overwrite its index file with
one built from only the first ``N - k`` rows, reopen, and assert the reopened
index self-heals to the live SQLite count AND serves a previously-missing
record as its own nearest neighbour at distance ~0.

Parametrized over both storage drivers (the default stdlib SQLite backend and
the lilli engine) so the self-heal is proven on both boot paths. The lilli arm
skips only if the native engine is genuinely unavailable in the runtime; it
never masks the stdlib arm.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import hnswlib
import numpy as np
import pytest

from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord

_DIM = 32
_N_RECORDS = 200
_N_MISSING = 50  # rows dropped from the stale on-disk index (mirrors a real skew)


def _lilli_available() -> bool:
    try:
        import iai_mcp_native  # noqa: F401, PLC0415
        from iai_mcp_native import engine  # noqa: F401, PLC0415
        from iai_mcp.lillibrain.connection import (  # noqa: F401, PLC0415
            get_lilli_engine_conn,
            register_lilli_conn,
        )
    except Exception:  # noqa: BLE001
        return False
    return True


_DRIVER_PARAMS = [
    pytest.param("stdlib", id="stdlib"),
    pytest.param(
        "lilli",
        id="lilli",
        marks=pytest.mark.skipif(
            not _lilli_available(),
            reason="lilli native engine not built in this runtime",
        ),
    ),
]


@pytest.fixture(autouse=True)
def _small_embed_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a small embedding dimension so the suite stays fast and never loads
    the real embedder (raw float32 vectors are written directly)."""
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(_DIM))


def _set_driver(monkeypatch: pytest.MonkeyPatch, driver: str) -> None:
    if driver == "stdlib":
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    else:
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", driver)


def _unit_vec(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(_DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def _make_record(vec: np.ndarray) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="fixture record for alice",
        aaak_index="",
        embedding=vec.tolist(),
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )


def _seed_store(tmp_path, n: int, seed: int = 7) -> MemoryStore:
    rng = np.random.default_rng(seed)
    store = MemoryStore(path=tmp_path)
    for _ in range(n):
        store.insert(_make_record(_unit_vec(rng)))
    return store


def _label_vectors(store: MemoryStore) -> tuple[np.ndarray, np.ndarray]:
    """Read (vec_label, embedding) for the active corpus directly from sqlite.

    Returns parallel arrays: labels (int64) and normalized float32 vectors,
    ordered by vec_label (the same order the rebuild path uses).
    """
    db = store.db
    with db._conn_lock:
        rows = db._conn.execute(
            "SELECT vec_label, embedding FROM records"
            " WHERE tombstoned_at IS NULL"
            " AND COALESCE(embedding_pending, 0) = 0"
            " ORDER BY vec_label"
        ).fetchall()
    labels = np.array([int(r["vec_label"]) for r in rows], dtype=np.int64)
    vecs = np.stack(
        [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
    )
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    vecs = vecs / norms
    return labels, vecs


def _live_sqlite_count(store: MemoryStore) -> int:
    db = store.db
    with db._conn_lock:
        row = db._conn.execute(
            "SELECT COUNT(*) FROM records"
            " WHERE tombstoned_at IS NULL"
            " AND COALESCE(embedding_pending, 0) = 0"
        ).fetchone()
    return int(row[0])


def _seed_then_write_stale_index(
    tmp_path, keep: int
) -> tuple[int, np.ndarray, int]:
    """Seed a healthy store, then leave a stale on-disk index in its place.

    Returns (live_count, missing_query_vector, missing_label).

    The store's ``close()`` always persists its in-memory (healthy) index to the
    on-disk path, so the stale file MUST be written AFTER the store is fully
    closed — otherwise close would overwrite it with a complete index and the
    skew would never reach the reopen. The stale index holds only the first
    ``keep`` (vec_label, embedding) rows ordered by vec_label; everything from
    ``keep`` onward stays in SQLite but is absent from the file — the
    stale/undersized skew.
    """
    store = _seed_store(tmp_path, _N_RECORDS)
    try:
        labels, vecs = _label_vectors(store)
        live_count = labels.shape[0]
        assert keep < live_count, "stale index must hold FEWER rows than SQLite"
        hnsw_path = store.db._hnsw_path
        tmp_index_path = store.db._hnsw_tmp_path
        # The harness must never reach outside the pytest tmp_path sandbox.
        assert (
            "tmp" in str(hnsw_path) or str(hnsw_path).startswith("/private")
        ), hnsw_path
    finally:
        store.close()  # persists a HEALTHY index — the stale write comes next.

    cap = max(live_count * 2, keep * 2, 16)
    idx = hnswlib.Index(space="cosine", dim=_DIM)
    idx.init_index(max_elements=cap, allow_replace_deleted=True)
    idx.add_items(vecs[:keep], labels[:keep])
    assert idx.get_current_count() == keep

    # Pick a record that is present in SQLite but ABSENT from the stale index.
    missing_idx = keep  # first dropped row
    missing_label = int(labels[missing_idx])
    missing_vec = vecs[missing_idx].copy()

    # Overwrite the (now healthy) on-disk file with the stale one, and clear any
    # sibling .tmp so the boot loader cannot pick a healthier candidate.
    idx.save_index(str(hnsw_path))
    if tmp_index_path.exists():
        tmp_index_path.unlink()

    return live_count, missing_vec, missing_label


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_boot_rebuilds_stale_index(tmp_path, monkeypatch, driver):
    """A reopened store whose on-disk index holds fewer rows than live SQLite
    must self-heal the loaded index to the live count."""
    _set_driver(monkeypatch, driver)

    live_count, _missing_vec, _missing_label = _seed_then_write_stale_index(
        tmp_path, keep=_N_RECORDS - _N_MISSING
    )

    reopened = MemoryStore(path=tmp_path)
    try:
        loaded_count = reopened.db._hnsw.get_current_count()
        assert loaded_count == live_count, (
            "reopened index did not self-heal to the live SQLite count: "
            f"loaded={loaded_count} live={live_count}"
        )
    finally:
        reopened.close()


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_recall_serves_missing_record(tmp_path, monkeypatch, driver):
    """After the self-heal, a probe equal to a previously-missing record's
    embedding returns that record as its top hit at distance ~0."""
    _set_driver(monkeypatch, driver)

    _live_count, missing_vec, missing_label = _seed_then_write_stale_index(
        tmp_path, keep=_N_RECORDS - _N_MISSING
    )

    reopened = MemoryStore(path=tmp_path)
    try:
        db = reopened.db
        with db._hnsw_lock:
            count = len(db._label_map)
            k = min(10, count)
            found_labels, dists = db._hnsw.knn_query(missing_vec, k=k)
        top_label = int(found_labels[0][0])
        top_dist = float(dists[0][0])
        assert top_label == missing_label, (
            "previously-missing record is not the top hit after self-heal: "
            f"top={top_label} expected={missing_label}"
        )
        assert top_dist <= 1e-4, f"top hit distance not ~0: {top_dist}"
    finally:
        reopened.close()
