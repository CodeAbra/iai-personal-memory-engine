"""Reader-pool ColIndex activation diagnostic.

Answers whether the single-column hash ColIndex OR-union fast path
(``crates/lilliengine/src/exec.rs`` ``indexed_in_columns``/``collect_indexed_in``)
actually reaches the pooled read-only connections ``HippoDB.ro_conn()`` hands to
the recall read path, or whether those connections silently fall back to a
column scan.

Two independent adoption mechanisms are exercised, matching the two ways a live
daemon's reader pool can end up with a built index:

1. Boot warm-up (``daemon/_boot_warmup.py:run_boot_warmup``, scheduled at
   startup by ``daemon/__init__.py``): the writer calls
   ``Connection.publish_read_models()`` once, then every reader-pool slot
   opened afterward adopts it — either via the writer-snapshot export/adopt
   handoff (``lillibrain/connection.py:1738-1749``) or via the process-wide
   built-index cache lookup a read-only ``Connection::new`` performs on open
   (``crates/lilliengine/src/conn.rs:483-485,501-560``).
2. Steady-state demand/publish/adopt (no explicit boot-warmup call): a reader
   opens first (registering demand, ``conn.rs:558``), a write commits
   afterward (``conn.rs:1822`` calls ``publish_indexes_on_demand``, gated by
   active demand and commit spacing, ``conn.rs:644-679``), and the next borrow
   refreshes the stale pool slot (``hippo/_ro_pool.py:411-440``) to adopt it.

Every store here lives under ``tmp_path`` — never a real or prod store. No
file under ``src/`` or ``rust/`` is touched by this diagnostic.
"""

from __future__ import annotations

import os
import struct
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from iai_mcp.hippo import _txn
from iai_mcp.hippo._db import DEFAULT_STORAGE_DRIVER
from iai_mcp.store import MemoryStore

_LILLI = os.environ.get("LILLI_STORAGE_DRIVER", DEFAULT_STORAGE_DRIVER).lower() == "lilli"
_EMBED_DIM = 384
_WARM_REPEATS = 15


def _require_native_reader_pool_surface() -> None:
    """Hard-fail (not skip) if the native module is present but stale."""
    try:
        from iai_mcp_native import engine
    except ImportError:
        return  # native module entirely absent — handled by driver skip below
    for attr in ("col_index_ready", "cells_visited_count", "publish_read_models"):
        assert hasattr(engine.Connection, attr), (
            f"iai_mcp_native.engine.Connection is missing {attr} — the native "
            "extension is stale. Run `cd rust/iai_mcp_native && maturin "
            "develop --release` before running this suite."
        )


if _LILLI:
    _require_native_reader_pool_surface()


_lilli_only = pytest.mark.skipif(
    not _LILLI,
    reason="col_index_ready is a lilli-engine-only PyO3 binding; the stdlib "
    "driver has no ColIndex/reader-pool mechanism at all (see "
    "test_col_index_ready_absent_on_stdlib below, which runs unconditionally).",
)


def _norm_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(_EMBED_DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def _embedding_blob(seed: int) -> bytes:
    vec = _norm_vec(seed)
    return struct.pack(f"<{_EMBED_DIM}f", *vec)


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path)


def _insert_records(
    store: MemoryStore,
    n: int,
    *,
    seed_offset: int,
    salience: str | None = None,
    with_labels: bool = False,
) -> list[int]:
    """Insert *n* plain records directly via SQL.

    Returns the assigned ``vec_label``s only when ``with_labels`` is True — a
    large padding batch skips the round-trip lookup entirely so a single
    fixture can seed a table sized well past the SQL placeholder count of the
    id lookup itself.
    """
    db = store.db
    now_iso = datetime.now(timezone.utc).isoformat()
    cols = ["id", "tier", "literal_surface", "embedding", "tombstoned_at",
            "embedding_pending", "created_at", "updated_at"]
    if salience is not None:
        cols.append("salience_level")
    rows = []
    ids = []
    for i in range(n):
        rid = str(uuid.uuid4())
        ids.append(rid)
        row = [rid, "episodic", None, _embedding_blob(seed_offset + i), None,
               0, now_iso, now_iso]
        if salience is not None:
            row.append(salience)
        rows.append(tuple(row))
    placeholders = ", ".join("?" for _ in cols)
    sql = f"INSERT INTO records ({', '.join(cols)}) VALUES ({placeholders})"
    with db._conn_lock:
        with _txn(db._conn):
            db._conn.executemany(sql, rows)
    if not with_labels:
        return []
    with db._conn_lock:
        ph = ", ".join("?" for _ in ids)
        cur = db._conn.execute(
            f"SELECT vec_label FROM records WHERE id IN ({ph}) ORDER BY vec_label",
            ids,
        )
        return [int(r[0]) for r in cur.fetchall()]


def _insert_edges(store: MemoryStore, rows: list[tuple]) -> None:
    db = store.db
    sql = "INSERT OR IGNORE INTO edges (src, dst, edge_type, weight) VALUES (?, ?, ?, ?)"
    with db._conn_lock:
        with _txn(db._conn):
            db._conn.executemany(sql, rows)


def _median_ms(fn, n: int = _WARM_REPEATS) -> float:
    fn()  # cold call, discarded — never counted as evidence
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def _assert_borrowed_from_pool(store: MemoryStore, fallback_before: int) -> None:
    pool = store.db._ro_pool
    assert pool is not None, "lilli driver must construct a reader pool"
    assert pool.writer_fallback_count == fallback_before, (
        "ro_conn() fell back to the shared writer connection — this run "
        "measured the writer path, not the pooled reader the recall path "
        "actually borrows from"
    )


# ---------------------------------------------------------------------------
# Scenario 1: boot-warmup publish (daemon/__init__.py's real startup sequence)
# ---------------------------------------------------------------------------


@_lilli_only
def test_reader_pool_adopts_after_boot_warmup_publish(tmp_path: Path) -> None:
    store = _make_store(tmp_path)

    hub = str(uuid.uuid4())
    bystanders = [str(uuid.uuid4()) for _ in range(5)]

    padding_edges = [
        (str(uuid.uuid4()), str(uuid.uuid4()),
         "hebbian" if i % 2 == 0 else "contradicts",
         round(0.10 + (i % 80) * 0.01, 3))
        for i in range(1000)
    ]
    control_target_edges = [
        (str(uuid.uuid4()), str(uuid.uuid4()),
         "hebbian" if i % 2 == 0 else "contradicts", 0.999)
        for i in range(60)
    ]
    hub_edges = [
        (hub, str(uuid.uuid4()), "hebbian" if i % 2 == 0 else "contradicts",
         round(0.5 + (i % 10) * 0.02, 3))
        for i in range(30)
    ] + [
        (str(uuid.uuid4()), hub, "hebbian" if i % 2 == 0 else "contradicts",
         round(0.5 + (i % 10) * 0.02, 3))
        for i in range(20)
    ]
    bystander_edges = [
        (b, str(uuid.uuid4()), "hebbian", 0.7) for b in bystanders for _ in range(2)
    ]
    _insert_edges(store, padding_edges + control_target_edges + hub_edges + bystander_edges)

    _insert_records(store, 800, seed_offset=0)
    control_labels = _insert_records(
        store, 150, seed_offset=800, salience="diagnostic-control", with_labels=True,
    )

    # Writer-side publish, mirroring daemon/_boot_warmup.py:177-189 exactly.
    store.db._conn.publish_read_models()

    # A SELECT * shape (ann_inlist's exact shape) routes through the engine's
    # whole-row "materializing" scan branch on an index miss, which increments
    # full_scan_count(), NOT cells_visited_count() (crates/lillibrain/src/pager.rs
    # note_cells_visited is only called by the column-selective streaming scan;
    # a narrower projection like ge_incident's SELECT src,dst,edge_type,weight
    # uses that path instead). full_scan_count()/reset_full_scan_count() are
    # exposed on the writer's full Connection class (py.rs:858-869) but NOT on
    # the reader-pool's RawConn wrapper (py.rs:412-517) — this measurement runs
    # on the writer, at the SAME committed generation and the SAME
    # col_index_ready state just asserted True on the reader below, so the
    # result characterizes the identical code path the reader borrows into,
    # even though it is not literally observable through the reader object.
    writer = store.db._conn
    ann_writer_sql = f"SELECT * FROM records WHERE vec_label IN ({', '.join('?' for _ in control_labels)})"
    records_writer_control_sql = "SELECT * FROM records WHERE salience_level IN (?)"
    with store.db._conn_lock:
        writer.reset_full_scan_count()
        writer.execute(ann_writer_sql, control_labels).fetchall()
        ann_writer_full_scans = writer.full_scan_count()

        writer.reset_full_scan_count()
        writer.execute(records_writer_control_sql, ["diagnostic-control"]).fetchall()
        records_control_writer_full_scans = writer.full_scan_count()
    print(
        f"[records, writer-side full_scan_count()] ann_inlist-shaped="
        f"{ann_writer_full_scans} forced-scan control (non-indexed "
        f"salience_level column)={records_control_writer_full_scans}"
    )
    assert ann_writer_full_scans == 0, (
        f"the ann_inlist-shaped SELECT * ... vec_label IN(...) query triggered "
        f"{ann_writer_full_scans} whole-tree materializing scan(s) on the "
        f"writer even though col_index_ready('records') is True — the fast "
        f"path is not being taken for this exact query shape"
    )
    assert records_control_writer_full_scans >= 1, (
        "the forced-scan control over a genuinely non-indexed column "
        f"(salience_level) triggered {records_control_writer_full_scans} "
        "whole-tree scans — expected >= 1; if this is 0 the control query is "
        "not actually forcing a scan and the comparison above is meaningless"
    )

    fallback_before = store.db._ro_pool.writer_fallback_count
    with store.db.ro_conn() as conn:
        _assert_borrowed_from_pool(store, fallback_before)

        ready_edges = conn.col_index_ready("edges")
        ready_records = conn.col_index_ready("records")
        print(
            f"[boot-warmup] col_index_ready(edges)={ready_edges} "
            f"col_index_ready(records)={ready_records}"
        )
        assert ready_edges is True, (
            "col_index_ready('edges') is False on the reader-pool connection "
            "after publish_read_models() + a fresh borrow — the writer-publish "
            "-> reader-adopt wiring is not reaching the pool (see "
            "crates/lilliengine/src/conn.rs:483-485, 501-560 and "
            "src/iai_mcp/lillibrain/connection.py:1738-1749)"
        )
        assert ready_records is True, (
            "col_index_ready('records') is False on the reader-pool connection "
            "after publish_read_models() + a fresh borrow — same wiring gap as "
            "above, for the records table"
        )

        ids = [hub, *bystanders]
        ph = ", ".join("?" for _ in ids)
        ge_incident_sql = (
            f"SELECT src, dst, edge_type, weight FROM edges"
            f" WHERE (src IN ({ph}) OR dst IN ({ph})) AND edge_type IN (?, ?)"
        )
        ge_incident_params = ids + ids + ["hebbian", "contradicts"]
        edges_control_sql = "SELECT src, dst, edge_type, weight FROM edges WHERE weight > ?"
        edges_control_params = [0.99]

        cv0 = conn.cells_visited_count()
        matched = conn.execute(ge_incident_sql, ge_incident_params).fetchall()
        cv1 = conn.cells_visited_count()
        index_hit_cells = cv1 - cv0

        cv2 = conn.cells_visited_count()
        control_matched = conn.execute(edges_control_sql, edges_control_params).fetchall()
        cv3 = conn.cells_visited_count()
        control_scan_cells = cv3 - cv2

        print(
            f"[edges] ge_incident-shaped: matched={len(matched)} "
            f"cells_visited_delta={index_hit_cells} | forced-scan control: "
            f"matched={len(control_matched)} cells_visited_delta={control_scan_cells} "
            f"| table_total_rows={len(padding_edges) + len(control_target_edges) + len(hub_edges) + len(bystander_edges)}"
        )

        edges_ms_index = _median_ms(lambda: conn.execute(ge_incident_sql, ge_incident_params).fetchall())
        edges_ms_control = _median_ms(lambda: conn.execute(edges_control_sql, edges_control_params).fetchall())
        print(
            f"[edges] warm median ms: ge_incident-shaped={edges_ms_index:.3f} "
            f"forced-scan control={edges_ms_control:.3f}"
        )

        assert index_hit_cells < control_scan_cells, (
            f"the ge_incident-shaped OR/IN query visited {index_hit_cells} "
            f"cells, no fewer than the forced-full/column-scan control's "
            f"{control_scan_cells} — with col_index_ready True this should be "
            f"a probe of the matched rowset, not a scan of the padded table"
        )

        ann_sql = f"SELECT * FROM records WHERE vec_label IN ({', '.join('?' for _ in control_labels)})"
        records_control_sql = "SELECT * FROM records WHERE salience_level IN (?)"
        records_control_params = ["diagnostic-control"]

        cv4 = conn.cells_visited_count()
        ann_matched = conn.execute(ann_sql, control_labels).fetchall()
        cv5 = conn.cells_visited_count()
        ann_index_hit_cells = cv5 - cv4

        cv6 = conn.cells_visited_count()
        records_control_matched = conn.execute(records_control_sql, records_control_params).fetchall()
        cv7 = conn.cells_visited_count()
        records_control_cells = cv7 - cv6

        print(
            f"[records] ann_inlist-shaped: matched={len(ann_matched)} "
            f"cells_visited_delta={ann_index_hit_cells} | forced-scan control "
            f"(non-indexed salience_level column): matched={len(records_control_matched)} "
            f"cells_visited_delta={records_control_cells} | table_total_rows=950"
        )

        ann_ms_index = _median_ms(lambda: conn.execute(ann_sql, control_labels).fetchall())
        ann_ms_control = _median_ms(lambda: conn.execute(records_control_sql, records_control_params).fetchall())
        print(
            f"[records] warm median ms: ann_inlist-shaped={ann_ms_index:.3f} "
            f"forced-scan control={ann_ms_control:.3f}"
        )

        # cells_visited_count() is NOT a valid index-hit/full-scan discriminator
        # for a SELECT * shape on the READER: both queries above show delta 0
        # (empirically confirmed) because SELECT * routes through the
        # full_scan_count()-tracked materializing scan, not the
        # cells_visited_count()-tracked column-selective streaming scan — and
        # full_scan_count() is not exposed on this reader-pool connection type
        # at all (see the writer-side assertions above, which ARE a valid
        # discriminator for this exact shape). Printed for completeness, not
        # asserted on, so this test never encodes a false expectation about a
        # counter that structurally cannot distinguish these two queries here.
        print(
            f"[records] reader-side cells_visited_count() is not a valid "
            f"discriminator for SELECT * shapes: index-hit delta="
            f"{ann_index_hit_cells}, control delta={records_control_cells} "
            f"(both expected near 0 regardless of col_index_ready state)"
        )

    store.close()


# ---------------------------------------------------------------------------
# Scenario 2: steady-state demand/publish/adopt, no explicit boot-warmup call
# ---------------------------------------------------------------------------


@_lilli_only
def test_reader_pool_adopts_via_demand_publish_without_boot_warmup(tmp_path: Path) -> None:
    store = _make_store(tmp_path)

    # Prime demand BEFORE any write: a read-only open registers demand for
    # every col-indexed table unconditionally (conn.rs:501-558), independent
    # of what it queries.
    with store.db.ro_conn() as primer:
        primer.execute("SELECT 1 FROM edges LIMIT 1")

    hub = str(uuid.uuid4())
    edges = [
        (hub, str(uuid.uuid4()), "hebbian", 0.6) for _ in range(10)
    ] + [
        (str(uuid.uuid4()), str(uuid.uuid4()), "contradicts", 0.4) for _ in range(90)
    ]
    now_iso = datetime.now(timezone.utc).isoformat()
    records = [
        (str(uuid.uuid4()), "episodic", None, _embedding_blob(2000 + i), None, 0, now_iso, now_iso)
        for i in range(100)
    ]
    # Both tables written in ONE transaction (one commit): the writer's commit
    # spacing gate (conn.rs:2551-2566) is keyed per (path, table) but the
    # PREVIOUS-attempt timestamp it compares against is updated on every
    # publish CHECK for every demanded table, not only tables whose rows
    # actually changed in that commit (conn.rs:644-679's `demanded` loop walks
    # every col-indexed table with active demand). Writing edges and records
    # as two SEPARATE back-to-back commits made the second commit's publish
    # check for the table untouched by the first commit land inside the first
    # commit's own spacing window and get throttle-suppressed — a real,
    # reproducible, SELF-HEALING race (heals on the writer's next statement
    # once the interval passes; verified directly, not asserted here — see
    # the DIAGNOSIS). One transaction sidesteps it entirely and is also the
    # more representative shape (a single logical write touching both
    # tables), consistent with the mutation-signalling proxy's own "signal on
    # every mutating statement AND on commit" contract (hippo/_db.py:64-81).
    with store.db._conn_lock:
        with _txn(store.db._conn):
            store.db._conn.executemany(
                "INSERT OR IGNORE INTO edges (src, dst, edge_type, weight) VALUES (?, ?, ?, ?)",
                edges,
            )
            store.db._conn.executemany(
                "INSERT INTO records (id, tier, literal_surface, embedding, tombstoned_at, "
                "embedding_pending, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                records,
            )

    # The writer's commit above (routed through _MutationSignallingConn) both
    # published the demanded indexes (conn.rs:644-679, spacing allows on the
    # first commit for a fresh table key) and bumped every registered pool's
    # generation counter (hippo/_db.py:64-158). The next borrow refreshes the
    # stale slot and adopts (hippo/_ro_pool.py:411-440).
    fallback_before = store.db._ro_pool.writer_fallback_count
    with store.db.ro_conn() as conn:
        _assert_borrowed_from_pool(store, fallback_before)
        ready_edges = conn.col_index_ready("edges")
        ready_records = conn.col_index_ready("records")
        print(
            f"[demand-publish, no boot-warmup] col_index_ready(edges)={ready_edges} "
            f"col_index_ready(records)={ready_records}"
        )
        assert ready_edges is True, (
            "col_index_ready('edges') is False via the demand/publish/adopt "
            "path (no publish_read_models() call) — the mutation-signalling "
            "proxy or the commit-time publish (conn.rs:644-679) is not "
            "reaching the reader pool"
        )
        assert ready_records is True, (
            "col_index_ready('records') is False via the demand/publish/adopt "
            "path (no publish_read_models() call) — same wiring gap as above, "
            "for the records table"
        )

    store.close()


# ---------------------------------------------------------------------------
# Scenario 3: cross-table commit-spacing throttle can transiently suppress a
# publish; the writer's next statement past the interval self-heals it. This
# is NOT reachable in production, where run_boot_warmup's publish_read_models()
# call (daemon/_boot_warmup.py:177-189, scheduled at daemon/__init__.py:2253-
# 2261) publishes both tables unconditionally, ignoring spacing entirely
# (conn.rs:711-716), before any reader or post-boot write exists.
# ---------------------------------------------------------------------------


@_lilli_only
def test_cross_table_commit_spacing_can_transiently_suppress_and_self_heals(tmp_path: Path) -> None:
    store = _make_store(tmp_path)

    with store.db.ro_conn() as primer:
        primer.execute("SELECT 1 FROM edges LIMIT 1")

    # Two SEPARATE back-to-back commits, one per table. The commit-spacing
    # registry (conn.rs:2551-2566, RO_TABLE_LAST_COMMIT) is keyed per
    # (path, table) but publish_indexes_on_demand's `demanded` loop
    # (conn.rs:644-679) checks spacing for EVERY col-indexed table with
    # active demand on EVERY commit, not only the table that changed — so the
    # edges-only commit's demand-check for "records" (an empty table at that
    # point) consumes the records spacing window, and the records commit
    # milliseconds later lands inside it and gets throttle-suppressed
    # (record_publish_pending, conn.rs:676).
    hub = str(uuid.uuid4())
    _insert_edges(store, [(hub, str(uuid.uuid4()), "hebbian", 0.6) for _ in range(10)])
    _insert_records(store, 20, seed_offset=3000)

    with store.db.ro_conn() as conn:
        ready_immediately = conn.col_index_ready("records")
    print(f"[cross-table spacing race] col_index_ready(records) immediately after={ready_immediately}")

    # Past the spacing interval (LILLI_INDEX_PUBLISH_MIN_INTERVAL_MS, default
    # 250ms), ANY further writer statement drains the suppressed publish
    # (flush_suppressed_publishes, conn.rs:744-762, called from execute() at
    # conn.rs:1148).
    time.sleep(0.3)
    _insert_edges(store, [(str(uuid.uuid4()), str(uuid.uuid4()), "hebbian", 0.5)])

    with store.db.ro_conn() as conn:
        ready_after_heal = conn.col_index_ready("records")
    print(f"[cross-table spacing race] col_index_ready(records) after self-heal={ready_after_heal}")

    assert ready_after_heal is True, (
        "col_index_ready('records') did not recover after the spacing "
        "interval passed and a further writer statement ran — this would be "
        "a genuine (not self-healing) wiring gap, unlike the transient race "
        "this test otherwise documents"
    )

    store.close()


# ---------------------------------------------------------------------------
# stdlib driver: the mechanism under test does not exist on this driver
# ---------------------------------------------------------------------------


def test_col_index_ready_absent_on_stdlib(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """N/A-by-construction, not a failure: the stdlib driver has no ColIndex,
    no reader pool, and no ``col_index_ready`` binding at all — this test
    documents that absence rather than asserting a mechanism that cannot
    exist under this driver (hippo/_db.py:665-669: ``_ro_pool`` is only
    constructed when ``_storage_driver == "lilli"``; a plain
    ``sqlite3.Connection`` has no such method)."""
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "stdlib")
    store = MemoryStore(path=tmp_path)
    assert getattr(store.db, "_storage_driver", "stdlib") == "stdlib"
    assert store.db._ro_pool is None

    with store.db.ro_conn() as conn:
        assert not hasattr(conn, "col_index_ready"), (
            "a stdlib ro_conn() connection unexpectedly exposes "
            "col_index_ready — the driver-selection assumption above is wrong "
            "and this test's N/A framing no longer applies"
        )
    store.close()
