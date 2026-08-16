"""Regression for the cold-buffer lazy-warm wipe: a pushed-but-unflushed
recency marker (``source_rowid == -1``) must survive a lazy warm instead of
being dropped by a destructive ``replace_all``.

Covers the ``RecencyBuffer.merge_warm`` unit contract directly, that
``warm_recency_buffer`` routes through it, and the realistic-scale regression
(seeded past ``maxlen`` so capacity eviction pressure is real). The async
daemon write-queue path is explicitly out of scope and untouched — this file
never calls ``enable_async_writes``.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp.store import MemoryStore
from iai_mcp.store._recency_buffer import RecencyBuffer, RecencyMarker
from iai_mcp.types import EMBED_DIM, MemoryRecord


# ---------------------------------------------------------------------------
# Unit-level: RecencyBuffer.merge_warm contract, no store needed.
# ---------------------------------------------------------------------------


def _marker(
    id_: str,
    *,
    created_at: datetime | None,
    source_rowid: int,
    role: str | None = "user",
    tier: str = "episodic",
    embedding_pending: int = 0,
) -> RecencyMarker:
    return RecencyMarker(
        id=id_,
        literal_surface=f"marker {id_}",
        created_at=created_at,
        session_id=None,
        embedding_pending=embedding_pending,
        role=role,
        source_rowid=source_rowid,
        tier=tier,
    )


def test_merge_warm_preserves_unflushed_sentinel_not_in_sql_set():
    buf = RecencyBuffer(maxlen=5)
    now = datetime.now(timezone.utc)
    sentinel = _marker("s1", created_at=now, source_rowid=-1)
    buf.push(sentinel)

    sql_set = [
        _marker(f"sql{i}", created_at=now - timedelta(minutes=i), source_rowid=i)
        for i in range(3)
    ]
    buf.merge_warm(sql_set)

    assert buf.is_warm
    ids = {e.id for e in buf}
    assert "s1" in ids, "unflushed sentinel was wiped by merge_warm"
    assert {f"sql{i}" for i in range(3)} <= ids


def test_merge_warm_sql_row_supersedes_matching_sentinel():
    """A same-id SQL row is the flushed version and wins over its sentinel."""
    buf = RecencyBuffer(maxlen=5)
    now = datetime.now(timezone.utc)
    buf.push(_marker("dup", created_at=now, source_rowid=-1))

    flushed = _marker("dup", created_at=now, source_rowid=42)
    buf.merge_warm([flushed])

    entries = {e.id: e for e in buf}
    assert entries["dup"].source_rowid == 42, (
        "flushed SQL row did not supersede its own sentinel"
    )


def test_merge_warm_sentinel_exempt_from_eviction_even_with_oldest_created_at():
    """A sentinel that would sort as the oldest entry must never be evicted."""
    buf = RecencyBuffer(maxlen=2)
    now = datetime.now(timezone.utc)
    # Oldest possible created_at — _evict_oldest's natural first victim.
    buf.push(_marker("s1", created_at=now - timedelta(days=10), source_rowid=-1))

    sql_set = [
        _marker(f"sql{i}", created_at=now - timedelta(minutes=i), source_rowid=i)
        for i in range(2)
    ]
    buf.merge_warm(sql_set)

    ids = {e.id for e in buf}
    assert "s1" in ids, "sentinel was evicted despite being exempt"
    assert len(ids) == 3, (
        f"expected 2 SQL rows (maxlen) + 1 exempt sentinel = 3 entries, got {len(ids)}"
    )


def test_merge_warm_null_created_at_sentinel_exempt_from_eviction():
    """_evict_oldest treats a null created_at as datetime.min — the FIRST
    victim under naive eviction. A sentinel with a null created_at (an
    unresolved rowid feed that also failed to parse a timestamp) must still
    survive: sentinels are exempt from eviction by construction, not by
    timestamp luck."""
    buf = RecencyBuffer(maxlen=2)
    now = datetime.now(timezone.utc)
    buf.push(_marker("s1", created_at=None, source_rowid=-1))

    sql_set = [
        _marker(f"sql{i}", created_at=now - timedelta(minutes=i), source_rowid=i)
        for i in range(2)
    ]
    buf.merge_warm(sql_set)

    ids = {e.id for e in buf}
    assert "s1" in ids, "null-created_at sentinel was evicted first despite the eviction exemption"


def test_merge_warm_sentinel_survives_eviction_firing_on_the_sql_set_itself():
    """The SQL-warmed set alone can exceed maxlen (a caller is not required to
    pre-bound it to maxlen), which fires eviction on the SQL entries WHILE a
    sentinel is present — the combination the exemption logic must get right:
    eviction trims the SQL rows down to maxlen, and the sentinel (evaluated
    separately, after the SQL set is built) is never a candidate."""
    buf = RecencyBuffer(maxlen=2)
    now = datetime.now(timezone.utc)
    buf.push(_marker("s1", created_at=now, source_rowid=-1))

    sql_set = [
        _marker(f"sql{i}", created_at=now - timedelta(minutes=i), source_rowid=i)
        for i in range(3)  # exceeds maxlen=2 on its own
    ]
    buf.merge_warm(sql_set)

    ids = {e.id for e in buf}
    assert "s1" in ids, "sentinel was evicted while eviction fired on the SQL set"
    non_sentinel_ids = ids - {"s1"}
    assert len(non_sentinel_ids) == 2, (
        f"expected the SQL set trimmed to maxlen=2 non-sentinel entries, got {len(non_sentinel_ids)}"
    )


def test_merge_warm_matches_replace_all_when_no_sentinels_present():
    """Safe drop-in: with no unflushed sentinels, merge_warm and replace_all
    produce the same result."""
    now = datetime.now(timezone.utc)
    sql_set = [
        _marker(f"sql{i}", created_at=now - timedelta(minutes=i), source_rowid=i)
        for i in range(5)
    ]

    buf_replace = RecencyBuffer(maxlen=3)
    buf_replace.replace_all(sql_set)

    buf_merge = RecencyBuffer(maxlen=3)
    buf_merge.merge_warm(sql_set)

    assert {e.id for e in buf_replace} == {e.id for e in buf_merge}
    assert buf_replace.is_warm and buf_merge.is_warm


# ---------------------------------------------------------------------------
# Store-level: warm_recency_buffer routes through merge_warm.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _close_stores(monkeypatch):
    created: list[MemoryStore] = []
    orig_init = MemoryStore.__init__

    def _tracking_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(MemoryStore, "__init__", _tracking_init)
    try:
        yield
    finally:
        for store in created:
            try:
                store.close()
            except Exception:  # noqa: BLE001 -- teardown must never raise
                pass


def _random_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _monkeypatch_env(monkeypatch, tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")


def _make_role_user(text: str, created_at: datetime, seed: int) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
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
        created_at=created_at,
        updated_at=created_at,
        tags=["capture", "role:user"],
        language="en",
    )


def _sql_row_exists(store: MemoryStore, record_id) -> bool:
    with store.db.ro_conn() as conn:
        row = conn.execute(
            "SELECT id FROM records WHERE id = ?", (str(record_id),)
        ).fetchone()
    return row is not None


def test_warm_recency_buffer_routes_through_merge_warm(tmp_path, monkeypatch):
    _monkeypatch_env(monkeypatch, tmp_path)
    store_path = tmp_path / "route-check-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    merge_calls: list[int] = [0]
    replace_calls: list[int] = [0]
    orig_merge = store._recency_buffer.merge_warm
    orig_replace = store._recency_buffer.replace_all

    def _spy_merge(entries):
        merge_calls[0] += 1
        return orig_merge(entries)

    def _spy_replace(entries):
        replace_calls[0] += 1
        return orig_replace(entries)

    monkeypatch.setattr(store._recency_buffer, "merge_warm", _spy_merge)
    monkeypatch.setattr(store._recency_buffer, "replace_all", _spy_replace)

    store.warm_recency_buffer()

    assert merge_calls[0] == 1, "warm_recency_buffer did not call merge_warm"
    assert replace_calls[0] == 0, (
        "warm_recency_buffer still calls the destructive replace_all"
    )


# ---------------------------------------------------------------------------
# Regression: unflushed distinctive marker survives lazy warm at >maxlen
# scale (eviction pressure is real, mirroring the reproduced cold-buffer wipe).
# ---------------------------------------------------------------------------


def _seed_flushed_corpus(store: MemoryStore, n: int, base_ts: datetime) -> None:
    """Insert n flushed role:user episodic records via the normal sync path.

    The autouse conftest autoflush fixture is active by default in this
    function's caller (IAI_MCP_TEST_NO_AUTOFLUSH unset), so each insert lands
    in SQL immediately.
    """
    for i in range(n):
        rec = _make_role_user(
            f"seed corpus turn {i}", created_at=base_ts + timedelta(minutes=i), seed=1000 + i
        )
        store.insert(rec)


def test_unflushed_marker_survives_lazy_warm_at_over_maxlen_scale(tmp_path, monkeypatch):
    _monkeypatch_env(monkeypatch, tmp_path)
    store_path = tmp_path / "over-maxlen-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    maxlen = store._recency_buffer._maxlen
    assert maxlen == 200, f"expected the default maxlen=200, got {maxlen}"
    n_seed = maxlen + 60  # > maxlen, so the SQL warm set is FULL and eviction fires

    base_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _seed_flushed_corpus(store, n_seed, base_ts)

    # Reset to the natural cold state at fresh open — SQL already holds the
    # full corpus, but the in-RAM buffer has not been warmed from it yet.
    store._recency_buffer.clear()
    assert not store._recency_buffer.is_warm

    # Push ONE distinctive marker that is NOT yet flushed to SQL.
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")
    distinctive_ts = base_ts + timedelta(minutes=n_seed + 5)
    distinctive = _make_role_user(
        "distinctive unflushed post-seed marker", created_at=distinctive_ts, seed=999
    )
    store.insert(distinctive)

    assert not _sql_row_exists(store, distinctive.id), (
        "setup invariant broken: the distinctive marker must still be "
        "unflushed for this regression to mean anything"
    )

    # Trigger the lazy warm on a cold buffer with a FULL (maxlen) SQL set.
    markers = store.recent_pending_markers(n=maxlen + 10)
    marker_ids = {str(m.id) for m in markers}
    assert str(distinctive.id) in marker_ids, (
        "distinctive unflushed marker was dropped by the lazy warm at "
        ">maxlen scale — the cold-buffer wipe regressed"
    )

    # The async daemon write-queue path is out of scope and untouched by this
    # fix; this test never calls enable_async_writes.


def test_replace_all_path_drops_unflushed_marker_negative_control(
    tmp_path, monkeypatch
):
    """Negative control: replaying the identical scenario with merge_warm
    replaced by the destructive replace_all reproduces the drop — proving
    the regression above is falsifiable, not vacuously green."""
    _monkeypatch_env(monkeypatch, tmp_path)
    store_path = tmp_path / "negative-control-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    # Substitute the destructive replace_all for merge_warm on this instance.
    monkeypatch.setattr(
        store._recency_buffer, "merge_warm", store._recency_buffer.replace_all
    )

    maxlen = store._recency_buffer._maxlen
    n_seed = maxlen + 60
    base_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _seed_flushed_corpus(store, n_seed, base_ts)

    store._recency_buffer.clear()
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")
    distinctive_ts = base_ts + timedelta(minutes=n_seed + 5)
    distinctive = _make_role_user(
        "distinctive unflushed post-seed marker (negative control)",
        created_at=distinctive_ts,
        seed=998,
    )
    store.insert(distinctive)
    assert not _sql_row_exists(store, distinctive.id)

    markers = store.recent_pending_markers(n=maxlen + 10)
    marker_ids = {str(m.id) for m in markers}
    assert str(distinctive.id) not in marker_ids, (
        "negative control did not reproduce the drop — the substitution "
        "did not actually exercise the destructive replace_all path"
    )
