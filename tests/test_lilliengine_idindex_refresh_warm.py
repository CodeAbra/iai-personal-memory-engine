"""Committed regression fence: the records table's warm `IdIndex` and
`OrderedColIndex` must survive a write-generation-move refresh instead of
being dropped and lazily full-rebuilt on the next indexed read.

Seeds a deterministic synthetic native store through the PRODUCTION `records`
DDL (mirroring src/iai_mcp/hippo/_table.py, including every declared index),
so `catalog_col_index_columns` and `catalog_ordered_index_column` land in the
same state production hits. `created_at` is the ordered column real recall
traffic queries (`ORDER BY created_at DESC LIMIT k`, confirmed against
src/iai_mcp/hippo/_recall.py) and is backed by the non-partial
`idx_records_created_at` declared index.

Cost is measured with `cells_visited_count()` — NEVER `full_scan_count()`
(the streaming scan path is deliberately uncounted there) and NEVER
`id_ready`/an is-built flag (sampled after `execute()`, so a costly lazy
rebuild reports identically to a real adoption). Every cost assertion is
paired with a completeness twin: a row committed immediately before the
refresh must be returned by the post-refresh read, for both a single-row
write and a top-level executemany-bulk write.
"""
from __future__ import annotations

import os
import struct
import threading
import time

import pytest

os.environ.setdefault("LILLI_FSYNC_MODE", "fast")


def _native_available() -> bool:
    try:
        from iai_mcp_native import engine  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _native_available(),
    reason="iai_mcp_native.engine submodule not installed (build the native wheel)",
)

_N_ROWS = 10_500
_LIMIT = 20
_ORDERED_COLUMN = "created_at"


def _records_ddl() -> tuple[str, list[str]]:
    from iai_mcp.hippo import _table

    return _table._DDL_RECORDS, list(_table._DDL_RECORDS_INDEXES)


def _row(i: int) -> list:
    return [
        f"id-{i:06d}",
        "episodic",
        f"surface {i}",
        None,
        struct.pack("<4f", 0.1, 0.2, 0.3, 0.4),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        f"2026-01-{(i % 28) + 1:02d}T00:00:{i % 60:02d}",
        None,
        None,
        None,
        None,
        "bsc",
        b"",
        1,
        "unknown",
        "unflagged",
        None,
        None,
    ]


_COLS = (
    "id, tier, literal_surface, aaak_index, embedding, structure_hv, "
    "community_id, centrality, detail_level, pinned, stability, difficulty, "
    "last_reviewed, never_decay, never_merge, tombstoned_at, created_at, "
    "updated_at, tags_json, language, s5_trust_score, hv_tier, "
    "structure_hv_payload, embedding_pending, epistemic_status, "
    "salience_level, live, directive"
)
_INSERT_SQL = f"INSERT INTO records ({_COLS}) VALUES ({', '.join(['?'] * 28)})"


def _seed(conn, n: int) -> None:
    ddl, indexes = _records_ddl()
    conn.execute(ddl, [])
    for stmt in indexes:
        conn.execute(stmt, [])
    conn.execute("BEGIN", [])
    for i in range(n):
        conn.execute(_INSERT_SQL, _row(i))
    conn.execute("COMMIT", [])


def _engine_conn(path):
    from iai_mcp_native import engine

    conn = engine.Connection.open(str(path), 384)
    assert isinstance(conn, engine.Connection), (
        "the connection under test is not the Rust engine.Connection — "
        "refusing to run a false-green proof against another engine"
    )
    return conn


_ID_SQL = "SELECT id FROM records WHERE id IN (?, ?, ?)"
_ORDERED_SQL = f"SELECT id FROM records ORDER BY {_ORDERED_COLUMN} DESC LIMIT {_LIMIT}"


def _full_scan_expected_ids(conn, ids: list[str]) -> set[str]:
    rows = conn.execute("SELECT id FROM records", []).fetchall()
    present = {r[0] for r in rows}
    return {i for i in ids if i in present}


@pytest.fixture()
def store(tmp_path):
    path = tmp_path / "refresh_warm.db"
    rw = _engine_conn(path)
    _seed(rw, _N_ROWS)
    return path, rw


def test_id_and_ordered_stay_warm_across_single_row_write_and_refresh(store) -> None:
    """Single-row INSERT + COMMIT: id and ordered reads stay O(LIMIT)/O(1)
    post-refresh, with the just-committed row present in both (completeness).
    """
    from iai_mcp_native import engine

    path, rw = store
    ro = engine.Connection.open_read_only(str(path), 384)
    assert isinstance(ro, engine.Connection)

    probe_ids = ["id-000005", "id-005000", f"id-{_N_ROWS - 1:06d}"]
    ro.execute(_ID_SQL, probe_ids).fetchall()
    ro.execute(_ORDERED_SQL, []).fetchall()
    ro.reset_cells_visited_count()
    ro.execute(_ID_SQL, probe_ids).fetchall()
    assert ro.cells_visited_count() == 0, "id read did not start warm — the fence proves nothing"
    ro.reset_cells_visited_count()
    ro.execute(_ORDERED_SQL, []).fetchall()
    assert ro.cells_visited_count() == 0, "ordered read did not start warm — the fence proves nothing"

    new_id = "new-single-row"
    rw.execute("BEGIN", [])
    rw.execute(_INSERT_SQL, [new_id] + _row(_N_ROWS)[1:])
    rw.execute("COMMIT", [])

    advanced = ro.raw_conn(read_only=True).refresh()
    assert advanced, "refresh_read_view reported no snapshot advance after a committed write"

    probe_with_new = probe_ids + [new_id]
    ro.reset_cells_visited_count()
    id_rows = ro.execute(
        "SELECT id FROM records WHERE id IN (?, ?, ?, ?)", probe_with_new
    ).fetchall()
    id_cells = ro.cells_visited_count()
    assert id_cells <= 8, f"post-refresh id read cost {id_cells} cells — expected O(LIMIT)/O(1)"
    got_ids = {r[0] for r in id_rows}
    assert got_ids == set(probe_with_new), (
        f"completeness twin failed for id read: expected {set(probe_with_new)}, got {got_ids}"
    )

    ro.reset_cells_visited_count()
    ordered_rows = ro.execute(_ORDERED_SQL, []).fetchall()
    ordered_cells = ro.cells_visited_count()
    assert ordered_cells <= _LIMIT + 4, (
        f"post-refresh ordered read cost {ordered_cells} cells — expected O(LIMIT), "
        f"not O(corpus)={_N_ROWS}"
    )
    assert len(ordered_rows) == _LIMIT
    assert new_id not in {r[0] for r in ordered_rows}, (
        "new-single-row's created_at is not among the top rows by construction — "
        "a completeness bug would show it displacing a real top row"
    )


def test_id_and_ordered_stay_warm_across_executemany_bulk_write_and_refresh(store) -> None:
    """Top-level executemany bulk INSERT: id and ordered reads stay
    O(LIMIT)/O(1) post-refresh, with every bulk-inserted row visible
    (the reproduced P3 gap this plan closes — a bulk INSERT with no open
    transaction must still advance col_generation)."""
    from iai_mcp_native import engine

    path, rw = store
    ro = engine.Connection.open_read_only(str(path), 384)

    probe_ids = ["id-000010", "id-009999"]
    ro.execute("SELECT id FROM records WHERE id IN (?, ?)", probe_ids).fetchall()
    ro.execute(_ORDERED_SQL, []).fetchall()

    bulk_n = 25
    bulk_ids = [f"bulk-{i:05d}" for i in range(bulk_n)]
    bulk_rows = []
    for i, bid in enumerate(bulk_ids):
        row = _row(_N_ROWS + i)
        row[0] = bid
        bulk_rows.append(row)
    rw.executemany(_INSERT_SQL, bulk_rows)
    rw.commit()

    advanced = ro.raw_conn(read_only=True).refresh()
    assert advanced, "refresh_read_view reported no snapshot advance after the bulk commit"

    check_ids = probe_ids + bulk_ids[:3]
    placeholders = ", ".join(["?"] * len(check_ids))
    ro.reset_cells_visited_count()
    id_rows = ro.execute(f"SELECT id FROM records WHERE id IN ({placeholders})", check_ids).fetchall()
    id_cells = ro.cells_visited_count()
    assert id_cells <= 8, f"post-refresh id read cost {id_cells} cells after a bulk write — expected O(LIMIT)"
    got_ids = {r[0] for r in id_rows}
    assert got_ids == set(check_ids), (
        f"completeness twin failed for the bulk-executemany route: expected {set(check_ids)}, "
        f"got {got_ids} — a bulk INSERT with no open transaction must still advance "
        "col_generation or a pre-existing reader cache silently omits the new rows"
    )

    ro.reset_cells_visited_count()
    ordered_rows = ro.execute(_ORDERED_SQL, []).fetchall()
    ordered_cells = ro.cells_visited_count()
    assert ordered_cells <= _LIMIT + 4, (
        f"post-refresh ordered read cost {ordered_cells} cells after a bulk write — "
        f"expected O(LIMIT), not O(corpus)={_N_ROWS + bulk_n}"
    )
    assert len(ordered_rows) == _LIMIT


def test_writer_side_publish_and_maintenance_stay_bounded(store) -> None:
    """Bounded WRITER-side cost: publishing/maintaining the ordered component
    after a single-row commit on a warmed store does not scan O(corpus)."""
    path, rw = store
    from iai_mcp_native import engine

    ro = engine.Connection.open_read_only(str(path), 384)
    ro.execute(_ORDERED_SQL, []).fetchall()  # registers reader demand

    rw.execute(_ID_SQL, ["id-000001", "id-000002", "id-000003"]).fetchall()
    rw.execute(_ORDERED_SQL, []).fetchall()

    rw.reset_full_scan_count()
    rw.execute("BEGIN", [])
    rw.execute(_INSERT_SQL, ["bounded-writer-1"] + _row(_N_ROWS + 100)[1:])
    rw.execute("COMMIT", [])
    assert rw.full_scan_count() == 0, (
        "a single-row commit on a warmed writer must not pay a whole-tree rescan "
        "while publishing/maintaining the ordered component"
    )


def test_concurrent_reader_never_observes_a_torn_view_across_refresh(store) -> None:
    """A reader issuing id/ordered SELECTs while a generation-move + refresh
    runs on another thread must never observe a torn or non-monotonic view."""
    path, rw = store
    from iai_mcp_native import engine

    ro = engine.Connection.open_read_only(str(path), 384)
    ro.execute(_ID_SQL, ["id-000001", "id-000002", "id-000003"]).fetchall()
    ro.execute(_ORDERED_SQL, []).fetchall()

    stop = threading.Event()
    errors: list[Exception] = []
    seen_counts: list[int] = []

    def reader_loop() -> None:
        try:
            while not stop.is_set():
                rows = ro.execute(_ORDERED_SQL, []).fetchall()
                seen_counts.append(len(rows))
                ro.execute(_ID_SQL, ["id-000001", "id-000002", "id-000003"]).fetchall()
        except Exception as exc:  # noqa: BLE001 -- surfaced via `errors`
            errors.append(exc)

    def writer_loop() -> None:
        for i in range(15):
            rw.execute("BEGIN", [])
            rw.execute(_INSERT_SQL, [f"concurrent-{i}"] + _row(_N_ROWS + 200 + i)[1:])
            rw.execute("COMMIT", [])
            ro.raw_conn(read_only=True).refresh()
            time.sleep(0.001)

    t_reader = threading.Thread(target=reader_loop)
    t_reader.start()
    writer_loop()
    stop.set()
    t_reader.join(timeout=5)

    assert not errors, f"concurrent reader raised: {errors!r}"
    assert all(c == _LIMIT for c in seen_counts), (
        f"a torn/non-monotonic ordered read returned a row count other than {_LIMIT}: {seen_counts}"
    )
