from __future__ import annotations

import inspect
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from iai_mcp.hippo import _vecindex as hnswlib
import numpy as np
import pytest

from iai_mcp.daemon import _hippo_health_check_on_boot
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path)


_STALE_DIM = 32
_STALE_N = 200
_STALE_MISSING = 50


@pytest.fixture
def _small_embed_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a small embedding dimension so the stale-index case never loads the
    real embedder (raw float32 vectors are written directly)."""
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(_STALE_DIM))


def _stale_unit_vec(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(_STALE_DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def _stale_record(vec: np.ndarray) -> MemoryRecord:
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


def _seed_then_write_stale_index(tmp_path: Path, keep: int) -> None:
    """Seed a healthy store, then leave a stale on-disk index in its place.

    The store's ``close()`` always persists its in-memory (healthy) index, so the
    stale file is written AFTER the store is closed — the index holds only the
    first ``keep`` rows while SQLite keeps all of them (the undersized skew).
    """
    rng = np.random.default_rng(7)
    store = MemoryStore(path=tmp_path)
    try:
        for _ in range(_STALE_N):
            store.insert(_stale_record(_stale_unit_vec(rng)))
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
        live_count = labels.shape[0]
        assert keep < live_count, "stale index must hold FEWER rows than SQLite"
        hnsw_path = db._hnsw_path
        tmp_index_path = db._hnsw_tmp_path
        assert (
            "tmp" in str(hnsw_path) or str(hnsw_path).startswith("/private")
        ), hnsw_path
    finally:
        store.close()

    cap = max(live_count * 2, keep * 2, 16)
    idx = hnswlib.Index(space="cosine", dim=_STALE_DIM)
    idx.init_index(max_elements=cap, allow_replace_deleted=True)
    idx.add_items(vecs[:keep], labels[:keep])
    idx.save_index(str(hnsw_path))
    if tmp_index_path.exists():
        tmp_index_path.unlink()


def test_hippo_health_check_on_clean_store(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    try:
        result = _hippo_health_check_on_boot(store)
    finally:
        store.db.close()

    assert result["action"] == "ok", f"Expected ok, got: {result}"
    assert result["sqlite_count"] == 0
    assert result["hnsw_active_count"] == 0
    assert "hnsw_raw_count" in result


def test_hippo_health_check_on_corrupted_hnsw(tmp_path: Path) -> None:
    hnsw_path = tmp_path / "hippo" / "records.hnsw"
    hnsw_path.parent.mkdir(parents=True, exist_ok=True)
    hnsw_path.write_bytes(b"not a valid hnswlib index -- corrupted by test")

    store = _make_store(tmp_path)
    try:
        result = _hippo_health_check_on_boot(store)
    finally:
        store.db.close()

    assert isinstance(result, dict)
    assert "action" in result
    assert result["action"] in ("ok", "divergence_at_boot", "sqlite_count_failed")


def test_no_run_bounded_startup_optimize_in_daemon() -> None:
    import iai_mcp.daemon as _daemon  # noqa: PLC0415

    assert not hasattr(
        _daemon, "_run_bounded_startup_optimize"
    ), "_run_bounded_startup_optimize was not deleted from daemon.py"


def test_no_post_bind_full_optimize_in_daemon() -> None:
    import iai_mcp.daemon as _daemon  # noqa: PLC0415

    assert not hasattr(
        _daemon, "_post_bind_full_optimize"
    ), "_post_bind_full_optimize was not deleted from daemon.py"

    src = inspect.getsource(_daemon)
    assert "_post_bind_full_optimize" not in src, (
        "_post_bind_full_optimize string found in daemon source"
    )


def test_no_skip_startup_optimize_env_handling() -> None:
    import iai_mcp.daemon as _daemon  # noqa: PLC0415

    src = inspect.getsource(_daemon)
    assert "IAI_MCP_SKIP_STARTUP_OPTIMIZE" not in src, (
        "IAI_MCP_SKIP_STARTUP_OPTIMIZE string found in daemon source — env handling not removed"
    )


def test_health_check_uses_label_map_for_action_verdict() -> None:
    src = inspect.getsource(_hippo_health_check_on_boot)
    assert "_label_map" in src, "Missing _label_map in health check source"
    assert "hnsw_active_count" in src, "Missing hnsw_active_count in health check"
    assert "hnsw_raw_count" in src, (
        "hnsw_raw_count not consulted in the action verdict"
    )
    assert "sqlite_count == active_label_count == hnsw_raw_count" in src, (
        "Three-operand parity check not found in health check verdict"
    )


def test_health_check_flags_stale_hnsw(tmp_path: Path, _small_embed_dim) -> None:
    """After a store whose on-disk index was stale is opened, the boot-integrity
    check rebuilds the index before the health check runs.  The health check
    therefore sees a consistent, fully-rebuilt store and reports 'ok', with all
    three counts (sqlite, label_map, hnsw) equal.  The hnsw_raw_count field is
    always present in the result so callers can compare it against sqlite_count."""
    _seed_then_write_stale_index(tmp_path, keep=_STALE_N - _STALE_MISSING)

    store = _make_store(tmp_path)
    try:
        result = _hippo_health_check_on_boot(store)
    finally:
        store.db.close()

    assert "hnsw_raw_count" in result, (
        "hnsw_raw_count missing from health check result"
    )
    assert result["hnsw_raw_count"] == result["sqlite_count"], (
        "boot self-heal should have rebuilt the index to the live count before "
        f"the health check ran: {result}"
    )
    assert result["action"] == "ok", (
        "health check should return 'ok' after the boot self-heal: " f"{result}"
    )
