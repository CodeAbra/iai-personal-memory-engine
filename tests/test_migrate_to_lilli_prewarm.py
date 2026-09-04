"""Regression: a freshly-migrated lilli store carries a warm runtime graph cache.

After migrate_sqlite_to_lilli completes, the migrated store must have a persisted
runtime graph cache so the first daemon boot takes the warm cache-hit path instead
of a cold full rebuild.

Hermetic: tmp HOME + tmp store + the test passphrase, single process, no xdist.
Fixtures use ``alice``.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from iai_mcp.migrate import (
    MigrateReport,
    migrate_sqlite_to_lilli,
    verify_store_equality,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(i: int, vec: list[float], *, created_at: datetime):
    from iai_mcp.types import MemoryRecord

    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=f"alice said thing number {i} about memory and graphs",
        aaak_index="",
        embedding=vec,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"session_id": "sess", "role": "user"}],
        created_at=created_at,
        updated_at=created_at,
        tags=["role:user"],
        language="en",
    )


def _build_source_store(
    root: Path,
    *,
    monkeypatch: pytest.MonkeyPatch,
    n_records: int,
) -> str:
    """Build a synthetic stdlib source store with connected edges; return its db path."""
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "stdlib")
    monkeypatch.setenv("IAI_MCP_STORE", str(root))

    from iai_mcp.store import MemoryStore, flush_edge_buffer, flush_record_buffer

    store = MemoryStore(root)
    rng = np.random.RandomState(42)
    ids: list = []
    base = datetime(2026, 5, 1, 9, 0, 0, tzinfo=timezone.utc)
    for i in range(n_records):
        vec = rng.randn(384).tolist()
        rec = _make_record(i, vec, created_at=base + timedelta(seconds=i))
        ids.append(rec.id)
        store.insert(rec)
    flush_record_buffer(store)

    # Create a connected graph so centrality has non-zero values on at least
    # some nodes.  A chain plus a few cross-links produces multiple communities
    # and non-zero betweenness on the interior nodes.
    edge_pairs = [(ids[i], ids[i + 1]) for i in range(min(n_records - 1, 30))]
    if n_records >= 10:
        edge_pairs += [
            (ids[0], ids[5]),
            (ids[3], ids[8]),
            (ids[1], ids[6]),
        ]
    store.boost_edges(edge_pairs)
    flush_edge_buffer(store)

    src_db = str(root / "hippo" / "brain.sqlite3")
    store.db.close()
    return src_db


def _crypto_key(root: Path) -> bytes:
    from iai_mcp.crypto import CryptoKey

    return CryptoKey(store_root=root).get_or_create()


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def migrated_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[tuple[str, Path, Path, bytes, MigrateReport], None, None]:
    """Migrate a non-trivial synthetic store; yield (src_db, src_root, dst_root, key, report)."""
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    src_root.mkdir()

    src_db = _build_source_store(
        src_root,
        monkeypatch=monkeypatch,
        n_records=40,
    )
    key = _crypto_key(src_root)

    report = migrate_sqlite_to_lilli(src_db, dst_root, batch=20)
    yield src_db, src_root, dst_root, key, report


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_prewarm_report_done(migrated_store):
    """The migrator reports that the pre-warm completed."""
    _src_db, _src_root, _dst_root, _key, report = migrated_store
    assert report.prewarm_done is True, (
        "pre-warm did not complete (prewarm_done is False); "
        "first daemon boot will pay a cold rebuild"
    )
    assert report.prewarm_elapsed_sec > 0.0


def test_cache_file_exists(migrated_store):
    """The runtime_graph_cache file is present on disk under the migrated store root."""
    from iai_mcp import runtime_graph_cache

    _src_db, _src_root, dst_root, _key, _report = migrated_store
    cache_path = dst_root / runtime_graph_cache.CACHE_FILENAME
    assert cache_path.exists(), (
        f"cache file missing at {cache_path}; pre-warm did not write the cache"
    )


def test_cache_results_non_empty(migrated_store, monkeypatch: pytest.MonkeyPatch):
    """try_load_cache_results returns a non-None result with non-all-zero centrality."""
    from iai_mcp import runtime_graph_cache
    from iai_mcp.store import MemoryStore

    _src_db, _src_root, dst_root, _key, _report = migrated_store
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    monkeypatch.setenv("IAI_MCP_STORE", str(dst_root))

    store = MemoryStore(dst_root)
    try:
        results = runtime_graph_cache.try_load_cache_results(store)
        assert results is not None, (
            "try_load_cache_results returned None; cache is absent or fails the key gate"
        )
        centrality_map, payload_record_count = results
        assert len(centrality_map) > 0, "cached centrality map is empty"
        assert any(v != 0.0 for v in centrality_map.values()), (
            "all centrality values are zero; pre-warm built a degenerate cache"
        )
        assert payload_record_count > 0, "payload_record_count is 0 in the cache"
    finally:
        store.db.close()


def test_rebuild_not_needed(migrated_store, monkeypatch: pytest.MonkeyPatch):
    """On a freshly-migrated store _runtime_graph_rebuild_needed returns False."""
    from iai_mcp.retrieve import _runtime_graph_rebuild_needed
    from iai_mcp.store import MemoryStore

    _src_db, _src_root, dst_root, _key, _report = migrated_store
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    monkeypatch.setenv("IAI_MCP_STORE", str(dst_root))

    store = MemoryStore(dst_root)
    try:
        rebuild_needed = _runtime_graph_rebuild_needed(store)
        assert rebuild_needed is False, (
            "_runtime_graph_rebuild_needed returned True on a freshly-migrated store; "
            "the first daemon boot will pay a cold rebuild"
        )
    finally:
        store.db.close()


def test_payload_record_count_within_drift(migrated_store, monkeypatch: pytest.MonkeyPatch):
    """Cached payload_record_count is within drift tolerance of the active record count."""
    from iai_mcp import runtime_graph_cache
    from iai_mcp.retrieve import _within_drift_tolerance
    from iai_mcp.store import MemoryStore

    _src_db, _src_root, dst_root, _key, _report = migrated_store
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    monkeypatch.setenv("IAI_MCP_STORE", str(dst_root))

    store = MemoryStore(dst_root)
    try:
        results = runtime_graph_cache.try_load_cache_results(store)
        assert results is not None, "cache absent; cannot check payload_record_count"
        _centrality_map, payload_record_count = results
        active = store.active_records_count()
        # No writes happened between migration and this check, so the counts
        # must be exactly equal (no drift at all).
        assert payload_record_count == active, (
            f"payload_record_count ({payload_record_count}) != active records "
            f"({active}); drift gate would force a rebuild"
        )
        assert _within_drift_tolerance(payload_record_count, active), (
            "payload_record_count is outside drift tolerance; rebuild would fire anyway"
        )
    finally:
        store.db.close()


def test_six_dimension_verify_still_green(
    migrated_store, monkeypatch: pytest.MonkeyPatch
):
    """The 6-dimension migrator verify stays GREEN after the pre-warm.

    The pre-warm is read+cache-write only; it must not alter migrated
    records, edges, events, or any other stored data.
    """
    src_db, src_root, dst_root, key, _report = migrated_store
    monkeypatch.setenv("IAI_MCP_STORE", str(src_root))
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)

    rep = verify_store_equality(src_db, dst_root, key, src_root=src_root)
    failures = [
        (dim, info.reason)
        for dim, info in rep.dimensions.items()
        if not info.ok
    ]
    assert not failures, (
        "6-dimension verify failed after pre-warm — pre-warm may have "
        f"mutated migrated data: {failures}"
    )
