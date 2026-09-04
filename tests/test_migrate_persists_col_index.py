"""Regression: migrate persists the recall-index sidecar (records.colindex) at
the VERY END of finalize, AFTER the runtime-graph-cache pre-warm.

The engine already warm-LOADS a fingerprint-matching sidecar on every fresh
``open`` / ``open_read_only`` (fully generation-gated). The only historical gap
was that ``migrate_sqlite_to_lilli`` never persisted one, so every RO-pool slot
cold-built by full O(N) scan on its first ``id = ?`` / ``vec_label IN`` read.

The ORDERING is the load-bearing detail under test: ``_prewarm_graph_cache``
opens the migrated store as a full ``MemoryStore``, which runs the
stamp-gated boot migrations (``migrate_role_column``,
``migrate_record_tags_backfill``). On a migrate SOURCE that lacks the
``role_backfill_done`` stamp, ``migrate_role_column``'s bulk ``UPDATE`` fires
during prewarm and advances the engine's column-generation counter. A persist
placed BEFORE prewarm would be silently invalidated by the generation check at
the next open -- this file's ordering-bug arm is the one that would catch that
regression.

Hermetic: tmp HOME + tmp store, single process, no xdist. Fixtures use
``alice``.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from iai_mcp.hippo._db import DEFAULT_STORAGE_DRIVER
from iai_mcp.migrate import MigrateReport, migrate_sqlite_to_lilli

_LILLI = os.environ.get("LILLI_STORAGE_DRIVER", DEFAULT_STORAGE_DRIVER).lower() == "lilli"

_lilli_only = pytest.mark.skipif(
    not _LILLI,
    reason="records.colindex persistence is lilli-only; skipped under stdlib "
    "(test_stdlib_driver_no_persist_no_error runs unconditionally instead)",
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
    stamp_role_backfill: bool,
) -> str:
    """Build a synthetic stdlib source store; return its db path.

    When ``stamp_role_backfill`` is False the ``_hippo_meta`` completion stamp
    is never written, so a fresh open of the migrated destination re-runs
    ``migrate_role_column``'s bulk UPDATE during the migrator's own prewarm --
    the exact ordering-bug repro condition.
    """
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "stdlib")
    monkeypatch.setenv("IAI_MCP_STORE", str(root))

    from iai_mcp.store import MemoryStore, flush_record_buffer

    store = MemoryStore(root)
    rng = np.random.RandomState(7)
    ids: list = []
    base = datetime(2026, 5, 1, 9, 0, 0, tzinfo=timezone.utc)
    for i in range(n_records):
        vec = rng.randn(384).tolist()
        rec = _make_record(i, vec, created_at=base + timedelta(seconds=i))
        ids.append(rec.id)
        store.insert(rec)
    flush_record_buffer(store)

    if stamp_role_backfill:
        with store.db._conn_lock:
            store.db._conn.execute(
                "INSERT OR REPLACE INTO _hippo_meta (key, value) VALUES (?, ?)",
                ("role_backfill_done", "1"),
            )
            store.db._conn.commit()
    else:
        # MemoryStore's own construction runs _run_boot_migrations ->
        # migrate_role_column, which (a) stamps role_backfill_done on ANY
        # open and (b) already back-filled `role` from `tags_json` for every
        # row inserted above (insert() derives role at write time too, so
        # there is nothing left pending). To reproduce a genuine
        # no-stamp-with-pending-work source (the pre-role-column /
        # synthetic-store condition this arm targets -- the one that makes
        # prewarm's migrate_role_column actually ISSUE a bulk UPDATE and
        # thereby advance col_generation), null out `role` on every row
        # while leaving `tags_json` intact, then delete the stamp. The next
        # open (prewarm's MemoryStore, inside migrate finalize) finds
        # `role IS NULL AND tags_json IS NOT NULL` rows and re-derives them.
        with store.db._conn_lock:
            store.db._conn.execute("UPDATE records SET role = NULL")
            store.db._conn.execute(
                "DELETE FROM _hippo_meta WHERE key = ?", ("role_backfill_done",)
            )
            store.db._conn.commit()
            row = store.db._conn.execute(
                "SELECT value FROM _hippo_meta WHERE key = ?",
                ("role_backfill_done",),
            ).fetchone()
            pending = store.db._conn.execute(
                "SELECT COUNT(*) FROM records "
                "WHERE role IS NULL AND tags_json IS NOT NULL"
            ).fetchone()
        assert row is None, (
            "source store still carries the role_backfill_done stamp after "
            "explicit deletion -- the ordering-bug repro condition is not met"
        )
        assert pending[0] > 0, (
            "source store has no role IS NULL / tags_json IS NOT NULL rows -- "
            "migrate_role_column would have no pending work during prewarm, "
            "so col_generation would never advance and this arm would not "
            "actually exercise the ordering bug"
        )

    src_db = str(root / "hippo" / "brain.sqlite3")
    store.db.close()
    return src_db


def _reopen_lilli(dst_root: Path, monkeypatch: pytest.MonkeyPatch):
    """Fresh-open the migrated destination as an explicit lilli store.

    The env is forced to lilli only for the duration of migrate_sqlite_to_lilli
    itself (restored in its own finally), so every re-open of the destination
    afterward must re-force it here.
    """
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    monkeypatch.setenv("IAI_MCP_STORE", str(dst_root))

    from iai_mcp.store import MemoryStore

    return MemoryStore(dst_root)


def _lilli_conn(dst_root: Path, monkeypatch: pytest.MonkeyPatch):
    """Open the migrated store read-only; return (store, conn) with conn
    exposing the native persist/scan-count surface."""
    store = _reopen_lilli(dst_root, monkeypatch)
    conn = getattr(store.db, "_conn", None)
    if conn is None or not hasattr(conn, "full_scan_count"):
        store.db.close()
        pytest.fail(
            "store.db._conn does not expose full_scan_count -- the native "
            "engine surface is missing or stale."
        )
    return store, conn


def _known_ids_and_vec_labels(
    dst_root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], list[int]]:
    store = _reopen_lilli(dst_root, monkeypatch)
    try:
        with store.db._conn_lock:
            rows = store.db._conn.execute(
                "SELECT id, vec_label FROM records "
                "WHERE vec_label IS NOT NULL ORDER BY vec_label LIMIT 5"
            ).fetchall()
        ids = [r[0] for r in rows]
        labels = [int(r[1]) for r in rows]
        return ids, labels
    finally:
        store.db.close()


def _sidecar_path(dst_root: Path) -> Path:
    return Path(dst_root) / "hippo" / "records.colindex"


# ---------------------------------------------------------------------------
# Base arm: source WITH the role_backfill_done stamp.
# ---------------------------------------------------------------------------


@_lilli_only
def test_sidecar_persisted_and_warm_loaded_base_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    src_root.mkdir()

    src_db = _build_source_store(
        src_root, monkeypatch=monkeypatch, n_records=40, stamp_role_backfill=True
    )

    report = migrate_sqlite_to_lilli(src_db, dst_root, batch=20)

    assert report.col_index_persisted is True, (
        f"col_index_persisted must be True on a lilli dest; report={report}"
    )
    assert not report.col_index_persist_error

    sidecar = _sidecar_path(dst_root)
    assert sidecar.exists(), f"records.colindex sidecar missing at {sidecar}"

    ids, labels = _known_ids_and_vec_labels(dst_root, monkeypatch)
    assert ids and labels, "migrated store has no vec_label-bearing rows to probe"

    store, conn = _lilli_conn(dst_root, monkeypatch)
    try:
        conn.reset_full_scan_count()

        with store.db._conn_lock:
            row = store.db._conn.execute(
                "SELECT id FROM records WHERE id = ?", (ids[0],)
            ).fetchone()
        assert row is not None

        placeholders = ", ".join("?" for _ in labels)
        with store.db._conn_lock:
            rows = store.db._conn.execute(
                f"SELECT vec_label FROM records WHERE vec_label IN ({placeholders})",
                labels,
            ).fetchall()
        assert len(rows) == len(labels)

        assert conn.full_scan_count() == 0, (
            "a freshly-migrated store with a persisted sidecar must serve "
            "both id=? and vec_label IN with ZERO full scans (warm-LOADED, "
            "not cold-built)"
        )
    finally:
        store.db.close()


# ---------------------------------------------------------------------------
# ORDERING-BUG arm: source WITHOUT the stamp -- prewarm's migrate_role_column
# fires its bulk UPDATE and advances col_generation. This arm FAILS if the
# persist is placed before prewarm.
# ---------------------------------------------------------------------------


@_lilli_only
def test_sidecar_survives_no_stamp_ordering_bug_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    src_root.mkdir()

    src_db = _build_source_store(
        src_root, monkeypatch=monkeypatch, n_records=40, stamp_role_backfill=False
    )

    report = migrate_sqlite_to_lilli(src_db, dst_root, batch=20)

    assert report.col_index_persisted is True, (
        f"col_index_persisted must be True even on a no-stamp source; "
        f"report={report}"
    )
    assert not report.col_index_persist_error

    sidecar = _sidecar_path(dst_root)
    assert sidecar.exists(), f"records.colindex sidecar missing at {sidecar}"

    ids, labels = _known_ids_and_vec_labels(dst_root, monkeypatch)
    assert ids and labels

    store, conn = _lilli_conn(dst_root, monkeypatch)
    try:
        conn.reset_full_scan_count()

        with store.db._conn_lock:
            row = store.db._conn.execute(
                "SELECT id FROM records WHERE id = ?", (ids[0],)
            ).fetchone()
        assert row is not None

        placeholders = ", ".join("?" for _ in labels)
        with store.db._conn_lock:
            rows = store.db._conn.execute(
                f"SELECT vec_label FROM records WHERE vec_label IN ({placeholders})",
                labels,
            ).fetchall()
        assert len(rows) == len(labels)

        assert conn.full_scan_count() == 0, (
            "ORDERING-BUG ARM: a no-stamp source causes migrate_role_column's "
            "bulk UPDATE to fire during prewarm and advance col_generation. "
            "If the persist ran BEFORE prewarm, this would leave a stale "
            "sidecar that load_col_indexes rejects on the next open -- this "
            "assertion (full_scan_count()==0) would then FAIL. The persist "
            "must run AFTER prewarm to survive this."
        )
    finally:
        store.db.close()


# ---------------------------------------------------------------------------
# stdlib no-op arm.
# ---------------------------------------------------------------------------


def test_stdlib_driver_no_persist_no_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The persist helper is a no-op, no-error skip on a stdlib destination.

    ``migrate_sqlite_to_lilli`` always forces ``LILLI_STORAGE_DRIVER=lilli``
    for its FRESH destination (``_to_lilli.py``), so a stdlib DESTINATION is
    never reachable through the full pipeline -- there is no such arm to
    drive end-to-end. This test instead exercises ``_persist_col_index``
    (the exact function the finalize block calls) directly against a raw
    stdlib ``HippoDB``, proving the lilli-gated no-op contract at the unit
    level: the persist call is skipped (the stdlib connection has no
    ``persist_col_indexes`` attribute), no sidecar is written, and the report
    fields stay at their False/empty defaults.
    """
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "stdlib")
    dst_root = tmp_path / "dst_stdlib"
    dst_root.mkdir()

    from iai_mcp.hippo import HippoDB
    from iai_mcp.migrate._to_lilli import MigrateReport, _persist_col_index

    dst = HippoDB(dst_root)
    try:
        assert dst._storage_driver == "stdlib", (
            "test setup invariant: this arm must open a genuine stdlib "
            f"HippoDB, got driver={dst._storage_driver!r}"
        )
        report = MigrateReport()
        _persist_col_index(dst, report)

        assert report.col_index_persisted is False, (
            "col_index_persisted must be False when the destination lacks "
            "the native persist_col_indexes surface"
        )
        assert not report.col_index_persist_error
        assert not (Path(dst_root) / "hippo" / "records.colindex").exists(), (
            "no sidecar file should be written on a stdlib destination"
        )
    finally:
        dst.close()


# ---------------------------------------------------------------------------
# Persist-failure fail-safety arm.
# ---------------------------------------------------------------------------


@_lilli_only
def test_persist_failure_never_fails_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    src_root.mkdir()

    src_db = _build_source_store(
        src_root, monkeypatch=monkeypatch, n_records=15, stamp_role_backfill=True
    )

    from iai_mcp_native import engine

    def _boom(self) -> None:
        raise RuntimeError("simulated persist_col_indexes failure")

    monkeypatch.setattr(engine.Connection, "persist_col_indexes", _boom, raising=True)

    report = migrate_sqlite_to_lilli(src_db, dst_root, batch=20)

    assert report.col_index_persisted is False
    assert report.col_index_persist_error, (
        f"a persist failure must be recorded on the report; report={report}"
    )
    assert isinstance(report, MigrateReport)
    assert sum(report.rows_copied.values()) > 0, (
        "the migrated DATA must survive a persist failure -- rows_copied "
        "must be non-empty"
    )

    # The migrated store must still be usable (data intact, just a cold
    # rebuild ahead on first indexed read).
    store = _reopen_lilli(dst_root, monkeypatch)
    try:
        with store.db._conn_lock:
            row = store.db._conn.execute("SELECT COUNT(*) FROM records").fetchone()
        assert int(row[0]) == 15
    finally:
        store.db.close()
