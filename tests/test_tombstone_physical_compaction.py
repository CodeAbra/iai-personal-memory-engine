"""Physical tombstone compaction: sleep-step deletion, grace-window live-scan
filtering, and batched deletes.

The sleep step ``step_compact_hippo`` (``iai_mcp.lilli.cycle
.sleep_pipeline._optimize``) is the sole site that physically DELETEs
tombstoned rows, once they age past ``tombstone_ttl_sec`` (default 7 days).
Migration never compacts (see point 3 below) -- physical removal always
happens on the next scheduled maintenance cycle.

1. **Grace-window rows stay physically present, but are not scanned.** A
   record tombstoned but not yet past ``tombstone_ttl_sec`` stays PHYSICALLY
   present in the table (correct, per the TTL contract -- compaction must
   never drop a not-yet-aged row). The production recency/count query filters
   on the indexed ``live`` column (``WHERE live = 1``), so grace-window
   tombstones are skipped without violating the TTL contract (nothing is
   dropped early -- the rows are simply no longer walked).
   ``test_grace_window_tombstones_bloat_live_scan_surface`` pins this: the
   recency query visits zero cells (fully index-served) rather than scaling
   with the full physical row count.

2. **Batched delete.** ``HippoTable.delete`` (``iai_mcp/hippo/_table.py``)
   issues a single ``DELETE FROM records WHERE <predicate>`` per call. On a
   large tombstoned set, one unbounded delete over the entire aged set can
   trigger a B-tree pager rebalance storm (many pages touched in one
   transaction), so ``step_compact_hippo`` must batch the aged-tombstone drop
   into multiple bounded delete calls instead of one unbounded call -- see
   ``test_aged_tombstone_drop_is_batched``.

3. **Migrate finalize is verbatim, no compaction.** ``migrate_sqlite_to_lilli``
   (``iai_mcp/migrate/_to_lilli.py``) copies EVERY row verbatim, including
   rows already past their TTL in the source -- a fresh migrate never
   compacts. Aged tombstones present immediately after migrate are cleaned
   up by the next nightly ``step_compact_hippo`` cycle instead -- see
   ``test_migrate_finalize_is_verbatim_no_compaction``.

Correctness pin: live records survive compaction untouched; only
tombstoned-and-aged rows are removed -- see
``test_compaction_removes_only_aged_tombstoned_rows``.

Both drivers where meaningful: the compaction step operates on the lilli
storage driver in production (the sleep pipeline runs against the daemon's
store, which is lilli post-cutover); the stdlib driver is exercised as a
correctness no-op control since ``step_compact_hippo`` is driver-agnostic
(it only calls ``HippoTable``/``MemoryStore`` methods).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp.store import MemoryStore, RECORDS_TABLE
from iai_mcp.store._buffers import flush_record_buffer
from iai_mcp.types import MemoryRecord

_DIM = 32


@pytest.fixture(autouse=True)
def _small_embed_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(_DIM))


def _unit_vec(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(_DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def _make_record(vec: np.ndarray, *, created_at: datetime) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="fixture record for alice",
        aaak_index="",
        embedding=vec.tolist(),
        community_id=None,
        centrality=0.1,
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
        tags=[],
        language="en",
    )


class _PipelineShim:
    """Minimal driver exposing only what step_compact_hippo reads.

    ``_fixed_now`` lets the test push "now" past the tombstone TTL without
    depending on real wall-clock sleep.
    """

    def __init__(self, store: MemoryStore, fixed_now: datetime) -> None:
        self._store = store
        self._fixed_now = fixed_now

    def _check_interrupt(self, *_args, **_kwargs) -> bool:
        return False

    def _now(self) -> datetime:
        return self._fixed_now


def _seed_store(
    store: MemoryStore,
    *,
    n_live: int,
    n_tombstoned_aged: int,
    n_tombstoned_grace: int,
    now: datetime,
    aged_past_ttl_sec: int,
    ttl_sec: int,
) -> dict[str, list]:
    """Seed a store with three record populations; return their ids by class.

    ``n_tombstoned_aged`` records are tombstoned at a timestamp older than
    ``now - ttl_sec - aged_past_ttl_sec`` seconds (well past TTL -- eligible
    for the drop). ``n_tombstoned_grace`` records are tombstoned at
    ``now`` (well within the grace window -- NOT eligible for the drop).
    """
    rng = np.random.default_rng(7)
    ids: dict[str, list] = {"live": [], "aged": [], "grace": []}

    tbl = store.db.open_table(RECORDS_TABLE)

    for _ in range(n_live):
        rec = _make_record(_unit_vec(rng), created_at=now)
        store.insert(rec)
        ids["live"].append(rec.id)

    for _ in range(n_tombstoned_aged):
        rec = _make_record(_unit_vec(rng), created_at=now)
        store.insert(rec)
        ids["aged"].append(rec.id)

    for _ in range(n_tombstoned_grace):
        rec = _make_record(_unit_vec(rng), created_at=now)
        store.insert(rec)
        ids["grace"].append(rec.id)

    flush_record_buffer(store)

    aged_ts = now - timedelta(seconds=ttl_sec + aged_past_ttl_sec)
    aged_str = aged_ts.strftime("%Y-%m-%d %H:%M:%S")
    for rid in ids["aged"]:
        tbl.update(where=f"id = '{rid}'", values={"tombstoned_at": aged_str})

    grace_str = now.strftime("%Y-%m-%d %H:%M:%S")
    for rid in ids["grace"]:
        tbl.update(where=f"id = '{rid}'", values={"tombstoned_at": grace_str})

    return ids


# ---------------------------------------------------------------------------
# Gap 1 -- grace-window bloat: physical rows are NOT dropped within the grace
# window (correct, per TTL contract), but the live-scan surface no longer
# walks them -- the B fix's indexed `live` column serves the production
# recency query directly, skipping grace-window tombstones without touching
# the TTL contract. This is a POST-B-fix pin, not a C RED marker: physical
# compaction (this file's C fix) never runs early on grace-window rows, so it
# cannot be what closed this gap -- B did.
# ---------------------------------------------------------------------------


def _native_available() -> bool:
    try:
        from iai_mcp_native import engine  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _native_available(),
    reason="iai_mcp_native.engine submodule not installed (build the native wheel)",
)
def test_grace_window_tombstones_bloat_live_scan_surface(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grace-window-tombstoned rows remain physically present (correct, per
    the TTL contract) but the live-scan surface (the production recency
    query) no longer pays for any of them -- the B fix's indexed ``live``
    column serves the query directly. THE PIN: the recency query's
    ``cells_visited_count()`` is now ZERO (fully index-served), independent
    of the full physical row count (live + grace-tombstoned) -- proving the
    live-scan surface is no longer bloated by grace-window tombstones
    (measured here with LIMIT set to the full live count, isolating this
    from any compaction-driven row-count change: even though physical
    compaction never touches grace-window rows, B's index still serves the
    query without walking them).
    """
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    monkeypatch.setenv("IAI_MCP_ERASURE_TOMBSTONE_TTL_SEC", "604800")  # 7 days, default
    now = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    store = MemoryStore(path=tmp_path)
    try:
        n_live, n_grace = 200, 200
        _seed_store(
            store,
            n_live=n_live,
            n_tombstoned_aged=0,
            n_tombstoned_grace=n_grace,
            now=now,
            aged_past_ttl_sec=0,
            ttl_sec=604800,
        )

        tbl = store.db.open_table(RECORDS_TABLE)
        physical_before = tbl.count_rows()
        assert physical_before == n_live + n_grace

        from iai_mcp.lilli.cycle.sleep_pipeline._optimize import step_compact_hippo

        done, info = step_compact_hippo(_PipelineShim(store, now), None)
        assert done is True

        physical_after = tbl.count_rows()
        assert physical_after == physical_before, (
            "grace-window-tombstoned rows must NOT be dropped early (the TTL "
            "contract) -- if this assertion fails, the fix violated the "
            "grace-window guarantee"
        )

        conn = store.db._conn
        if not hasattr(conn, "cells_visited_count"):
            pytest.skip("cells_visited_count() not available on this connection")

        from iai_mcp.hippo._recall import _DIRECT_RECENCY_SQL_LIMITED

        # Ask for exactly the live-row count -- isolating the effect from any
        # compaction-driven row-count change: even though LIMIT == the number
        # of genuinely live rows and physical compaction never touches
        # grace-window rows (they are correctly still present per the TTL
        # contract, asserted above), the walk no longer has to step over the
        # grace-window-tombstoned rows interleaved above/among them in
        # created_at order -- the query is served entirely by the `live`
        # index.
        conn.execute(_DIRECT_RECENCY_SQL_LIMITED, (n_live,)).fetchall()
        conn.reset_cells_visited_count()
        conn.execute(_DIRECT_RECENCY_SQL_LIMITED, (n_live,)).fetchall()
        visited = conn.cells_visited_count()

        assert visited == 0, (
            f"expected the recency query (LIMIT={n_live}, exactly the live "
            f"row count) to be fully index-served (0 cells visited) now that "
            f"the B fix's indexed `live` column serves it directly; got "
            f"{visited} cells visited out of {physical_after} physical rows "
            f"-- if this is nonzero, the query is falling back to a "
            f"sequential scan instead of the live index."
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Gap 2 -- unbatched delete: a large aged-tombstone set is dropped via ONE
# unbounded DELETE, not bounded batches.
# ---------------------------------------------------------------------------


def test_aged_tombstone_drop_is_batched(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A large aged-tombstoned set must be dropped via multiple bounded
    DELETE batches, not one unbounded DELETE over the whole set.

    RED on HEAD: ``step_compact_hippo`` calls ``tbl.delete(drop_where)``
    exactly once for the entire aged set (``_optimize.py`` line ~99) --
    instrumented here by monkeypatching ``HippoTable.delete`` to record every
    invocation and how many rows each one's WHERE clause matches.
    """
    monkeypatch.setenv("IAI_MCP_ERASURE_TOMBSTONE_TTL_SEC", "60")
    now = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    store = MemoryStore(path=tmp_path)
    try:
        n_aged = 250  # large enough that one unbounded delete is a real risk
        _seed_store(
            store,
            n_live=10,
            n_tombstoned_aged=n_aged,
            n_tombstoned_grace=0,
            now=now,
            aged_past_ttl_sec=120,
            ttl_sec=60,
        )

        tbl = store.db.open_table(RECORDS_TABLE)
        physical_before = tbl.count_rows()
        assert physical_before == 10 + n_aged

        from iai_mcp.hippo._table import HippoTable

        delete_calls: list[str] = []
        original_delete = HippoTable.delete

        def _recording_delete(self, where: str) -> None:
            delete_calls.append(where)
            return original_delete(self, where)

        monkeypatch.setattr(HippoTable, "delete", _recording_delete)

        from iai_mcp.lilli.cycle.sleep_pipeline._optimize import step_compact_hippo

        done, info = step_compact_hippo(_PipelineShim(store, now), None)
        assert done is True

        physical_after = tbl.count_rows()
        assert physical_after == 10, (
            "all aged-tombstoned rows must be physically dropped after "
            "compaction regardless of batching strategy"
        )

        drop_calls = [c for c in delete_calls if "tombstoned_at <" in c]
        assert len(drop_calls) > 1, (
            f"expected the aged-tombstone drop to issue MULTIPLE bounded "
            f"DELETE batches for a {n_aged}-row aged set; got "
            f"{len(drop_calls)} delete call(s) matching the drop predicate "
            f"({drop_calls}) -- one unbounded tbl.delete(drop_where) "
            f"covering the entire aged set risks a B-tree pager rebalance "
            f"storm on a large tombstoned set; the aged-tombstone drop must "
            f"be batched into bounded chunks to avoid it."
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Migrate finalize is verbatim: a fresh migrate imports the full tombstone
# bloat as-is, including rows already past TTL. Compaction is the nightly
# sleep step's job, not migrate's.
# ---------------------------------------------------------------------------


def test_migrate_finalize_is_verbatim_no_compaction(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh migrate copies every row verbatim, including tombstoned rows
    already past their TTL -- no compaction pass runs at finalize. Aged
    tombstones are cleaned up by the next nightly compaction cycle instead.
    """
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "stdlib")
    monkeypatch.setenv("IAI_MCP_ERASURE_TOMBSTONE_TTL_SEC", "60")
    now = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    src_root.mkdir()

    monkeypatch.setenv("IAI_MCP_STORE", str(src_root))
    src_store = MemoryStore(path=src_root)
    n_live, n_aged, n_grace = 15, 40, 10
    _seed_store(
        src_store,
        n_live=n_live,
        n_tombstoned_aged=n_aged,
        n_tombstoned_grace=n_grace,
        now=now,
        aged_past_ttl_sec=3600,
        ttl_sec=60,
    )
    src_tbl = src_store.db.open_table(RECORDS_TABLE)
    src_physical = src_tbl.count_rows()
    assert src_physical == n_live + n_aged + n_grace
    src_db_path = str(src_root / "hippo" / "brain.sqlite3")
    src_store.db.close()

    from iai_mcp.migrate import migrate_sqlite_to_lilli

    report = migrate_sqlite_to_lilli(src_db_path, dst_root, batch=20)
    assert report.rows_copied.get("records") == src_physical, (
        "verbatim-copy contract: every row (live + tombstoned, any age) is "
        "copied as-is"
    )

    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    monkeypatch.setenv("IAI_MCP_STORE", str(dst_root))
    from iai_mcp.hippo import HippoDB

    dst = HippoDB(dst_root)
    try:
        with dst._conn_lock:
            dst_physical = int(
                dst._conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
            )
    finally:
        dst.close()

    # No compaction runs at migrate finalize: the migrated destination's
    # physical row count equals the source's full row count (live +
    # aged-tombstoned + grace), aged rows included.
    assert dst_physical == src_physical, (
        f"expected a verbatim copy (no finalize compaction), so "
        f"dst_physical == src_physical; got src_physical={src_physical} "
        f"({n_live} live + {n_aged} aged-past-TTL + {n_grace} grace-window), "
        f"dst_physical={dst_physical}"
    )


# ---------------------------------------------------------------------------
# Regression guard -- a source store with aged tombstones must pass verify
# GREEN after migrate, and a pinned+tombstoned-aged row must survive the
# next nightly compaction while the unpinned aged rows are dropped.
# ---------------------------------------------------------------------------


def test_migrate_verify_green_with_aged_tombstones(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """verify_store_equality is GREEN after migrating a source store that
    holds tombstones older than the retention window; the migrated
    destination retains those tombstones verbatim until the next nightly
    compaction, which drops them while a pinned+tombstoned row survives.
    """
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "stdlib")
    monkeypatch.setenv("IAI_MCP_ERASURE_TOMBSTONE_TTL_SEC", "60")
    now = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    src_root.mkdir()

    monkeypatch.setenv("IAI_MCP_STORE", str(src_root))
    src_store = MemoryStore(path=src_root)
    n_live, n_aged, n_grace = 10, 5, 3
    ttl_sec = 60
    aged_past_ttl_sec = 3600
    ids = _seed_store(
        src_store,
        n_live=n_live,
        n_tombstoned_aged=n_aged,
        n_tombstoned_grace=n_grace,
        now=now,
        aged_past_ttl_sec=aged_past_ttl_sec,
        ttl_sec=ttl_sec,
    )

    rng = np.random.default_rng(13)
    pinned_rec = _make_record(_unit_vec(rng), created_at=now)
    src_store.insert(pinned_rec)
    flush_record_buffer(src_store)
    aged_ts = now - timedelta(seconds=ttl_sec + aged_past_ttl_sec)
    aged_str = aged_ts.strftime("%Y-%m-%d %H:%M:%S")
    src_tbl = src_store.db.open_table(RECORDS_TABLE)
    src_tbl.update(
        where=f"id = '{pinned_rec.id}'",
        values={"tombstoned_at": aged_str, "pinned": True},
    )

    src_physical = src_tbl.count_rows()
    assert src_physical == n_live + n_aged + n_grace + 1

    from iai_mcp.crypto import CryptoKey

    key = CryptoKey(store_root=src_root).get_or_create()
    src_db_path = str(src_root / "hippo" / "brain.sqlite3")
    src_store.db.close()

    from iai_mcp.migrate import migrate_sqlite_to_lilli, verify_store_equality

    migrate_sqlite_to_lilli(src_db_path, dst_root, batch=20)

    verify_report = verify_store_equality(src_db_path, dst_root, key, src_root=src_root)
    assert verify_report.ok is True, {
        name: dim.reason for name, dim in verify_report.dimensions.items()
    }

    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    monkeypatch.setenv("IAI_MCP_STORE", str(dst_root))
    from iai_mcp.hippo import HippoDB

    dst = HippoDB(dst_root)
    try:
        with dst._conn_lock:
            dst_physical = int(
                dst._conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
            )
    finally:
        dst.close()
    assert dst_physical == src_physical, (
        "aged tombstones (and the pinned+tombstoned row) must be present "
        "verbatim immediately after migrate"
    )

    dst_store = MemoryStore(path=dst_root)
    try:
        from iai_mcp.lilli.cycle.sleep_pipeline._optimize import step_compact_hippo

        done, _info = step_compact_hippo(_PipelineShim(dst_store, now), None)
        assert done is True

        tbl = dst_store.db.open_table(RECORDS_TABLE)
        for rid in ids["aged"]:
            row = tbl.count_rows(filter=f"id = '{rid}'")
            assert row == 0, (
                f"aged-past-TTL tombstoned record {rid} must be dropped by "
                f"the next nightly compaction after migrate"
            )
        pinned_row = tbl.count_rows(
            filter=f"id = '{pinned_rec.id}' AND tombstoned_at IS NULL",
        )
        assert pinned_row == 1, (
            "the pinned+tombstoned-aged record must survive compaction "
            "(restored to live, never physically removed)"
        )
    finally:
        dst_store.close()


# ---------------------------------------------------------------------------
# Correctness pin -- must hold before AND after the C fix.
# ---------------------------------------------------------------------------


def test_compaction_removes_only_aged_tombstoned_rows(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live rows and grace-window-tombstoned rows must survive compaction
    untouched; only aged-past-TTL tombstoned rows are physically removed.
    """
    monkeypatch.setenv("IAI_MCP_ERASURE_TOMBSTONE_TTL_SEC", "60")
    now = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    store = MemoryStore(path=tmp_path)
    try:
        n_live, n_aged, n_grace = 12, 18, 9
        ids = _seed_store(
            store,
            n_live=n_live,
            n_tombstoned_aged=n_aged,
            n_tombstoned_grace=n_grace,
            now=now,
            aged_past_ttl_sec=3600,
            ttl_sec=60,
        )

        from iai_mcp.lilli.cycle.sleep_pipeline._optimize import step_compact_hippo

        done, info = step_compact_hippo(_PipelineShim(store, now), None)
        assert done is True

        tbl = store.db.open_table(RECORDS_TABLE)

        for rid in ids["live"]:
            row = tbl.count_rows(filter=f"id = '{rid}'")
            assert row == 1, f"live record {rid} must survive compaction"
        for rid in ids["grace"]:
            row = tbl.count_rows(filter=f"id = '{rid}'")
            assert row == 1, (
                f"grace-window-tombstoned record {rid} must survive "
                f"compaction (not yet past TTL)"
            )
        for rid in ids["aged"]:
            row = tbl.count_rows(filter=f"id = '{rid}'")
            assert row == 0, (
                f"aged-past-TTL tombstoned record {rid} must be physically "
                f"removed by compaction"
            )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# stdlib no-op arm -- driver-agnostic correctness control.
# ---------------------------------------------------------------------------


def test_compaction_stdlib_noop_arm(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """step_compact_hippo is driver-agnostic (HippoTable/MemoryStore only) --
    the same aged-drop correctness must hold on the stdlib sqlite3 driver.
    """
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "stdlib")
    monkeypatch.setenv("IAI_MCP_ERASURE_TOMBSTONE_TTL_SEC", "60")
    now = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    store = MemoryStore(path=tmp_path)
    try:
        n_live, n_aged = 8, 12
        ids = _seed_store(
            store,
            n_live=n_live,
            n_tombstoned_aged=n_aged,
            n_tombstoned_grace=0,
            now=now,
            aged_past_ttl_sec=3600,
            ttl_sec=60,
        )

        from iai_mcp.lilli.cycle.sleep_pipeline._optimize import step_compact_hippo

        done, info = step_compact_hippo(_PipelineShim(store, now), None)
        assert done is True

        tbl = store.db.open_table(RECORDS_TABLE)
        assert tbl.count_rows() == n_live
        for rid in ids["live"]:
            assert tbl.count_rows(filter=f"id = '{rid}'") == 1
        for rid in ids["aged"]:
            assert tbl.count_rows(filter=f"id = '{rid}'") == 0
    finally:
        store.close()
