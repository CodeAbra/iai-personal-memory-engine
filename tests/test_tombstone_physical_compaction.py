"""Physical tombstone compaction in sleep AND at migrate finalize (C), plus the
B-fix's closure of the grace-window live-scan-surface gap.

The existing sleep step ``step_compact_hippo`` (``iai_mcp.lilli.cycle
.sleep_pipeline._optimize``) already physically DELETEs tombstoned rows once
they age past ``tombstone_ttl_sec`` (default 7 days) -- three gaps were
tracked here (the finalize-compaction fix closes two of them; the third was
already closed by the deletion-sweep fix):

1. **Grace-window bloat -- CLOSED BY B, not C.** A record tombstoned but not
   yet past ``tombstone_ttl_sec`` stays PHYSICALLY present in the table
   (correct, per the TTL contract -- C's compaction must never drop a
   not-yet-aged row). Before the B fix landed, the recency/count queries
   still walked every grace-window-tombstoned row for the full grace window
   (default 7 days) -- physical compaction (C) cannot shrink this cost on its
   own, since the rows are not yet eligible for removal. The B fix (the
   indexed ``live`` column, filtering the production recency query to
   ``WHERE live = 1``) closes this gap independently of C: the live-scan
   surface now skips grace-window tombstones without violating the TTL
   contract (nothing is dropped early -- the rows are simply no longer
   walked). ``test_grace_window_tombstones_bloat_live_scan_surface`` now
   pins the POST-B-fix behavior: the recency query visits zero cells (fully
   index-served) rather than scaling with the full physical row count.

2. **Unbatched delete.** ``HippoTable.delete`` (``iai_mcp/hippo/_table.py``)
   issues ONE ``DELETE FROM records WHERE <predicate>`` for the entire aged
   set. On a large tombstoned set this single delete can trigger a B-tree
   pager rebalance storm (many pages touched in one transaction). The fix
   must batch the delete into bounded chunks. RED: this module asserts (by
   monkeypatching ``HippoTable.delete``) that ``step_compact_hippo`` issues
   MULTIPLE bounded delete calls for a large aged-tombstone set, not one
   unbounded call -- see ``test_aged_tombstone_drop_is_batched``.

3. **No migrate-finalize compaction.** ``migrate_sqlite_to_lilli``
   (``iai_mcp/migrate/_to_lilli.py``) copies EVERY row verbatim, including
   rows already past their TTL in the source -- there is no compaction pass
   at finalize, so a fresh migrate imports the full tombstone bloat rather
   than starting clean. RED: this module migrates a source store where a
   fraction of rows are tombstoned well past TTL and asserts the migrated
   destination's physical row count is STRICTLY LESS than the source's (a
   compaction pass ran, dropping the aged rows) -- this assertion FAILS on
   HEAD (dst equals src; no compaction runs) -- see
   ``test_migrate_finalize_imports_full_tombstone_bloat``.

Correctness pin (must hold before AND after the C fix): live records survive
compaction untouched; only tombstoned-and-aged rows are removed -- see
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
            f"({drop_calls}) -- HEAD issues exactly one unbounded "
            f"tbl.delete(drop_where) covering the entire aged set, risking a "
            f"B-tree pager rebalance storm on a large tombstoned set. "
            f"Expected RED state on HEAD; the C fix (179-03/04) must batch "
            f"this delete into bounded chunks."
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Gap 3 -- migrate finalize does not compact: a fresh migrate imports the
# full tombstone bloat verbatim, including rows already past TTL.
# ---------------------------------------------------------------------------


def test_migrate_finalize_imports_full_tombstone_bloat(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh migrate must not import tombstoned rows already past their TTL
    verbatim -- a compaction pass at finalize should drop them so the fresh
    store starts clean rather than importing stale bloat.

    RED on HEAD: ``migrate_sqlite_to_lilli`` copies every row verbatim with
    no compaction step; the migrated destination's physical row count still
    equals the source's full row count (live + aged-tombstoned + grace).
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
        "the migrator's own verbatim-copy contract: every row (live + "
        "tombstoned, any age) is copied as-is -- this is not the bug, it is "
        "the CURRENT design that the compaction-at-finalize fix must run "
        "AFTER"
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

    # THE RED MARKER: a compaction pass must run at migrate finalize so the
    # aged-past-TTL tombstoned rows are dropped rather than imported
    # verbatim -- dst_physical must be strictly LESS than src_physical (the
    # aged rows gone). migrate_sqlite_to_lilli runs a compaction pass at
    # finalize; without it the full tombstone bloat, aged rows included,
    # would be imported verbatim and dst_physical == src_physical.
    assert dst_physical < src_physical, (
        f"expected a compaction pass to run at migrate finalize, dropping "
        f"the {n_aged} aged-past-TTL tombstoned rows so dst_physical < "
        f"src_physical; got src_physical={src_physical} "
        f"({n_live} live + {n_aged} aged-past-TTL + {n_grace} grace-window), "
        f"dst_physical={dst_physical} (equal -- the full tombstone bloat "
        f"was imported verbatim, no finalize compaction ran). Expected RED "
        f"state on HEAD; the C fix (179-03/04) must add a compaction pass "
        f"at migrate finalize."
    )
    assert dst_physical >= n_live + n_grace, (
        "sanity: the migrated store must at minimum retain every live and "
        "grace-window row (these must never be silently dropped by any "
        "future finalize-compaction change)"
    )


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
