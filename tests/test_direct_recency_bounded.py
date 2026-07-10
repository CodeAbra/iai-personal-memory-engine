from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord
from iai_mcp.hippo import (
    direct_recency_rows_from_store,
    _DIRECT_RECENCY_SQL_LIMITED,
)


def _random_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _make_rec(
    tier: str = "episodic",
    text: str = "user turn",
    seed: int = 0,
) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=_random_vec(seed),
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


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    s = MemoryStore(str(tmp_path / "store"))
    yield s


def test_limited_sql_constant_ends_with_limit_q():
    assert _DIRECT_RECENCY_SQL_LIMITED.rstrip().endswith("LIMIT ?"), (
        f"_DIRECT_RECENCY_SQL_LIMITED must end with 'LIMIT ?': {_DIRECT_RECENCY_SQL_LIMITED!r}"
    )


def test_direct_recency_rows_bounded_limit(store):
    n = 20
    limit = 5
    for i in range(n):
        r = _make_rec(seed=i)
        store.insert(r)

    root = store.db._store_root
    rows = direct_recency_rows_from_store(root, limit=limit)
    assert len(rows) <= limit, (
        f"Expected at most {limit} rows with limit={limit}, got {len(rows)}"
    )


def test_direct_recency_rows_bounded_returns_rows(store):
    for i in range(8):
        r = _make_rec(seed=100 + i)
        store.insert(r)

    root = store.db._store_root
    rows = direct_recency_rows_from_store(root, limit=3)
    assert len(rows) > 0, "Bounded direct_recency_rows_from_store must return rows"
    assert len(rows) <= 3, f"Must not exceed limit=3, got {len(rows)}"


def test_direct_recency_rows_unbounded_consumer_unchanged(store):
    n = 15
    for i in range(n):
        r = _make_rec(seed=200 + i)
        store.insert(r)

    root = store.db._store_root
    rows_all = direct_recency_rows_from_store(root)
    rows_limited = direct_recency_rows_from_store(root, limit=3)

    assert len(rows_all) >= n, (
        f"Unbounded path must return all {n} rows, got {len(rows_all)}"
    )
    assert len(rows_limited) <= 3, "Bounded path must cap at limit=3"
    assert len(rows_all) > len(rows_limited), (
        "Unbounded must return more rows than bounded when store > limit"
    )


def test_degraded_semantic_recall_bounded(store):
    from iai_mcp.hippo import degraded_semantic_recall

    n = 20
    for i in range(n):
        r = _make_rec(seed=300 + i, text=f"user turn {i}")
        store.insert(r)

    root = store.db._store_root
    limit = 5
    results = degraded_semantic_recall(root, "user turn", limit=limit)
    assert len(results) <= limit, (
        f"degraded_semantic_recall with limit={limit} must return at most {limit} results, "
        f"got {len(results)}"
    )


def test_degraded_semantic_recall_returns_degraded_flag(store):
    from iai_mcp.hippo import degraded_semantic_recall

    for i in range(5):
        r = _make_rec(seed=400 + i)
        store.insert(r)

    root = store.db._store_root
    results = degraded_semantic_recall(root, "user turn", limit=5)
    if results:
        for res in results:
            assert res.get("_degraded") is True, (
                f"All results must have _degraded=True: {res}"
            )
