"""Boot-time store-read regression tests for the lilli storage driver.

Asserts that COUNT(*) queries do not materialise row payloads (they must use a
key-only walk), and that the projected record-column stream completes within the
measured stdlib aggregate budget.

The key oracle for the COUNT tests is the engine's full-scan counter:
``Connection.full_scan_count()`` increments each time the engine walks the
whole B-tree and copies every row payload into a heap buffer.  A correct bare
or filtered COUNT should drive this counter to zero by using a key-only page
walk instead.  On current HEAD every COUNT calls ``full_scan()``, so the
counter is > 0, making these tests RED for the right reason.

Thresholds for the aggregate timing test derive from the measured stdlib
baselines at 30 000 records:
  - Store-read stages (filtered COUNT + record stream + edge read): ~3.8 s.
  - Budget: 6 s (approx 1.6x to account for driver overhead + CI noise).

These tests run ONLY when LILLI_STORAGE_DRIVER=lilli is set.
"""

from __future__ import annotations

import os
import struct
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Driver guard: entire module is lilli-only.
# ---------------------------------------------------------------------------

_LILLI = os.environ.get("LILLI_STORAGE_DRIVER", "").lower() == "lilli"

pytestmark = pytest.mark.skipif(
    not _LILLI,
    reason="boot-perf regressions run only under LILLI_STORAGE_DRIVER=lilli",
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_RECORDS = 30_000  # Fixed corpus: large enough to surface the full-scan cost.
_N_EDGES = 45_000    # Representative graph density (~1.6 edges/record).

_EMBED_DIM = 384
_EMBED_BYTES = _EMBED_DIM * 4  # float32, little-endian; 1536 bytes per row.

# Aggregate store-read budget (seconds): filtered COUNT + record stream + edge read.
# Stdlib baseline: ~3.8 s at 30k records.
# Budget: 6 s (approx 1.6x with CI noise margin).
_STORE_READ_BUDGET_S = 6.0

# A single filtered COUNT(*) must stay well under a second even when the corpus
# exceeds the page-cache budget. Before the scan-resistant read path, a corpus
# larger than the cache made every full-scan COUNT thrash the LRU (an eviction
# pass per page), so the filtered COUNT took ~7-8 s; the scan-resistant read
# keeps an uncached scan page out of the cache, so the walk is O(corpus) with no
# evictions and the count returns in well under a second. Budget 1.5 s leaves
# generous CI-noise headroom over the measured ~0.2 s.
_FILTERED_COUNT_BUDGET_S = 1.5

# A warm runtime-graph build on a prewarmed store must be a cache HIT in a few
# seconds, never a cold full rebuild. Before the cache-freshness alignment, the
# saved payload count (which counted tombstoned rows) never matched the active
# count, so every boot rejected the cache and ran a ~60-80 s cold rebuild. With
# the alignment + the count memo + the scan-resistant reads, the warm build is a
# few seconds. Budget 12 s is well under the cold-rebuild floor and above the
# measured ~4 s warm build.
_WARM_BUILD_BUDGET_S = 12.0


# ---------------------------------------------------------------------------
# Module-scoped fixture: build the synthetic 30k-record lilli store once.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lilli_boot_store(tmp_path_factory):
    """A 30 000-record + 45 000-edge lilli MemoryStore on a tmp path."""
    store_root = tmp_path_factory.mktemp("lilli-boot-perf") / ".iai-mcp"
    store_root.mkdir(parents=True, exist_ok=True)

    # Snapshot env vars; use direct os.environ writes — monkeypatch is
    # function-scoped and incompatible with module scope.
    _saved: dict[str, str | None] = {}
    _set_vars = {
        "HOME": str(store_root.parent),
        "IAI_MCP_STORE": str(store_root),
        "IAI_DAEMON_SOCKET_PATH": str(store_root / "no-such.sock"),
        "IAI_MCP_CRYPTO_PASSPHRASE": "test-boot-perf-passphrase-lilli",
        "LILLI_STORAGE_DRIVER": "lilli",
        "LILLI_FSYNC_MODE": "fast",
    }
    for k, v in _set_vars.items():
        _saved[k] = os.environ.get(k)
        os.environ[k] = v

    try:
        from iai_mcp.store import MemoryStore

        store = MemoryStore(path=store_root)
        db = store.db

        _populate_records(db, _N_RECORDS)
        _populate_edges(db, _N_EDGES)

        yield store

        store.close()
    finally:
        for k, prior in _saved.items():
            if prior is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prior


def _make_embedding_blob(seed_byte: int) -> bytes:
    """A deterministic 384-float blob (little-endian float32)."""
    val = float(seed_byte % 128)
    return struct.pack(f"<{_EMBED_DIM}f", *([val] * _EMBED_DIM))


def _populate_records(db, n: int) -> None:
    """Insert n synthetic records via the underlying SQLite connection.

    Two-thirds of rows are active (tombstoned_at IS NULL, embedding_pending=0).
    One-third are tombstoned so the filtered COUNT returns a non-trivial subset.
    Embedding BLOBs are full-width (1536 bytes) to match production row size,
    making the per-row payload cost identical to the real boot path.
    """
    from iai_mcp.hippo import _txn

    rows = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for i in range(1, n + 1):
        rid = str(uuid.UUID(int=i))
        embedding = _make_embedding_blob(i % 256)
        tombstoned_at = None if i % 3 != 0 else "2024-01-01 00:00:00"
        rows.append((
            rid,            # id
            "episodic",     # tier
            None,           # literal_surface
            embedding,      # embedding — full 1536-byte blob
            tombstoned_at,
            0,              # embedding_pending
            now_iso,        # created_at
            now_iso,        # updated_at
        ))

    sql = (
        "INSERT INTO records "
        "(id, tier, literal_surface, embedding, tombstoned_at, "
        " embedding_pending, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    )

    with db._conn_lock:
        with _txn(db._conn):
            db._conn.executemany(sql, rows)


def _populate_edges(db, n: int) -> None:
    """Insert n synthetic edges (src, dst, edge_type, weight)."""
    from iai_mcp.hippo import _txn

    rows = []
    for i in range(n):
        src_idx = (i % _N_RECORDS) + 1
        dst_idx = ((i * 7 + 3) % _N_RECORDS) + 1
        src = str(uuid.UUID(int=src_idx))
        dst = str(uuid.UUID(int=dst_idx))
        edge_type = "hebbian" if i % 2 == 0 else "contradicts"
        weight = round(0.5 + (i % 10) * 0.05, 3)
        rows.append((src, dst, edge_type, weight))

    sql = (
        "INSERT OR IGNORE INTO edges (src, dst, edge_type, weight) "
        "VALUES (?, ?, ?, ?)"
    )

    with db._conn_lock:
        with _txn(db._conn):
            db._conn.executemany(sql, rows)


def _get_lilli_conn(db):
    """Return the underlying lilli engine Connection, or None if not available.

    The lilli driver stores its connection as db._conn (an
    iai_mcp_native.engine.Connection).  This function verifies the connection
    has the scan-counter interface before returning it.
    """
    conn = getattr(db, "_conn", None)
    if conn is None:
        return None
    if not hasattr(conn, "full_scan_count"):
        return None
    return conn


# ---------------------------------------------------------------------------
# Test 1 — RED on HEAD
#
# A bare COUNT(*) FROM records must complete without triggering a full scan.
# The engine's full_scan_count() counter increments each time the B-tree walk
# materialises every row payload into a heap buffer.  A correct key-only COUNT
# increments this counter zero times.
#
# On current HEAD, count_rows_no_decode (exec.rs) calls store.tree(root).full_scan()
# then .len(), so the counter increments once for every COUNT(*) query.
# ---------------------------------------------------------------------------

def test_count_no_payload_copy(lilli_boot_store) -> None:
    """Bare COUNT(*) FROM records must not trigger a payload-copying full scan.

    The oracle is the engine's full-scan counter: a key-only COUNT increments
    it zero times; the current implementation increments it once (it copies
    every 1536-byte embedding blob into memory just to call .len() on the
    result).  A correct fix routes COUNT(*) through a key-only page-header walk.
    """
    store = lilli_boot_store
    db = store.db
    lconn = _get_lilli_conn(db)

    count = store.active_records_count()
    assert count > 0, "active_records_count() must return a positive number"
    assert count < _N_RECORDS, "active_records_count() must be less than total rows"

    if lconn is None:
        pytest.skip("full_scan_count not available on this connection type")

    lconn.reset_full_scan_count()
    store.active_records_count()
    scans = lconn.full_scan_count()

    assert scans == 0, (
        f"bare COUNT(*) triggered {scans} full scan(s) — expected 0. "
        f"A key-only walk reads page headers without copying row payload bytes. "
        f"Current implementation calls full_scan() which materialises every "
        f"1536-byte embedding blob ({_N_RECORDS} rows × {_EMBED_BYTES} bytes) "
        f"before returning the count. Fix: use a key-only B-tree header walk."
    )


# ---------------------------------------------------------------------------
# Test 2 — RED on HEAD
#
# A filtered COUNT(*) WHERE tombstoned_at IS NULL AND COALESCE(embedding_pending, 0) = 0
# is the exact query active_records_count() issues during daemon boot.
# On current HEAD this also calls full_scan(), decoding the full row to evaluate
# a 2-column predicate, so full_scan_count() is 1 after the query.
# ---------------------------------------------------------------------------

def test_projected_count_skips_embedding(lilli_boot_store) -> None:
    """Filtered COUNT(*) must not decode the embedding BLOB for the predicate.

    The WHERE predicate references only tombstoned_at and embedding_pending.
    A correct column-selective scan reads only those two columns, skipping the
    1536-byte embedding payload.  On current HEAD the engine decodes every full
    row (including the embedding) to evaluate the predicate.

    Oracle: full_scan_count() must be zero after the query.
    """
    store = lilli_boot_store
    db = store.db
    lconn = _get_lilli_conn(db)

    if lconn is None:
        pytest.skip("full_scan_count not available on this connection type")

    lconn.reset_full_scan_count()
    with db._conn_lock:
        row = db._conn.execute(
            "SELECT COUNT(*) FROM records"
            " WHERE tombstoned_at IS NULL"
            " AND COALESCE(embedding_pending, 0) = 0"
        ).fetchone()
    scans = lconn.full_scan_count()

    count = int(row[0]) if row else 0
    assert count > 0, "filtered COUNT must return a positive number"

    assert scans == 0, (
        f"filtered COUNT(*) triggered {scans} full scan(s) — expected 0. "
        f"The WHERE clause references only tombstoned_at and embedding_pending; "
        f"the engine should decode only those two columns to evaluate the predicate, "
        f"skipping the 1536-byte embedding BLOB. Fix: column-selective decode that "
        f"does not materialise columns absent from the WHERE clause and SELECT list."
    )


# ---------------------------------------------------------------------------
# Test 3 — Aggregate store-read timing
#
# Time the boot graph-build store-read stages: filtered COUNT + column stream
# + edge read.  Budget is set from the measured stdlib baseline (3.8 s) plus
# a 1.6x CI noise margin.  On current HEAD the lilli path is slower than
# stdlib at this corpus size; this test guards against the gap widening.
# ---------------------------------------------------------------------------

def test_boot_graph_build_within_budget(lilli_boot_store) -> None:
    """Boot graph-build store-read stages must not use a full scan for the COUNT.

    The boot graph build calls active_records_count() (a filtered COUNT) then
    streams the record column set.  The COUNT must not trigger a payload-copying
    full scan (oracle: full_scan_count() == 0 after the COUNT).  The stream
    correctly reads all rows including the embedding (the embedding is a
    requested column), so the full-scan counter for the stream stage is not
    asserted here.

    A secondary timing guard ensures the aggregate does not regress beyond the
    measured stdlib baseline plus CI margin.  Stdlib baseline: ~3.8 s at 30 000
    records; budget: 6 s (~1.6x with CI noise margin).
    """
    store = lilli_boot_store
    db = store.db
    lconn = _get_lilli_conn(db)

    # Stage 1: filtered COUNT — must not trigger a full payload scan.
    if lconn is not None:
        lconn.reset_full_scan_count()
    t0 = time.monotonic()
    _count = store.active_records_count()
    stage1_elapsed = time.monotonic() - t0
    if lconn is not None:
        count_scans = lconn.full_scan_count()
    else:
        count_scans = None

    # Stage 2: column stream — embedding IS a requested column, so a page walk
    # is expected.  We measure timing but do not assert the scan counter here.
    stream_cols = [
        "id",
        "embedding",
        "community_id",
        "embedding_pending",
        "literal_surface",
        "tier",
        "centrality",
        "pinned",
        "tags_json",
        "language",
    ]
    t1 = time.monotonic()
    n_streamed = sum(1 for _ in store.iter_record_columns(stream_cols, batch_size=1024))
    stage2_elapsed = time.monotonic() - t1

    # Stage 3: edge read.
    t2 = time.monotonic()
    edges_df = store.db.open_table("edges").to_pandas()
    stage3_elapsed = time.monotonic() - t2

    total_elapsed = stage1_elapsed + stage2_elapsed + stage3_elapsed

    assert n_streamed > 0, "iter_record_columns must yield at least one row"
    assert len(edges_df) > 0, "edges table must contain rows"

    # Primary RED gate: the COUNT stage must not trigger a full payload scan.
    if count_scans is not None:
        assert count_scans == 0, (
            f"active_records_count() triggered {count_scans} full scan(s) during "
            f"the boot graph-build COUNT stage — expected 0. "
            f"Stage breakdown: COUNT={stage1_elapsed:.3f}s, "
            f"stream={stage2_elapsed:.3f}s, edges={stage3_elapsed:.3f}s. "
            f"Fix: route the filtered COUNT through a column-selective scan that "
            f"does not materialise the 1536-byte embedding payload."
        )

    # Secondary timing guard: aggregate must not regress beyond the stdlib baseline.
    assert total_elapsed < _STORE_READ_BUDGET_S, (
        f"store-read stages took {total_elapsed:.3f}s — exceeds the "
        f"{_STORE_READ_BUDGET_S}s budget. "
        f"Stage breakdown: COUNT={stage1_elapsed:.3f}s, "
        f"stream={stage2_elapsed:.3f}s, edges={stage3_elapsed:.3f}s. "
        f"Stdlib baseline: ~3.8 s total at {_N_RECORDS} records."
    )


# ---------------------------------------------------------------------------
# Test 4 — filtered COUNT(*) sub-second budget (scan-resistant read path)
#
# The records corpus exceeds the default page-cache budget, so a full-scan
# COUNT must NOT thrash the cache. The scan-resistant read keeps uncached scan
# pages out of the LRU, so the filtered COUNT is O(corpus) with zero evictions
# and returns in well under a second. RED before the fix (~7-8 s of cache
# thrash), GREEN after (~0.2 s).
# ---------------------------------------------------------------------------

def test_filtered_count_sub_second(lilli_boot_store) -> None:
    """A single filtered active-records COUNT must return well under a second.

    This is the exact COUNT the daemon boot's drift gate issues. With the
    scan-resistant read path, an uncached scan page is not inserted into the LRU,
    so a full-scan COUNT never thrashes the cache and the walk is O(corpus). The
    drift gate calls this on every boot, so it must be cheap.
    """
    store = lilli_boot_store

    # Warm any one-time open cost out of the measurement.
    store.active_records_count()

    t0 = time.monotonic()
    count = store.active_records_count()
    elapsed = time.monotonic() - t0

    assert count > 0, "filtered COUNT must return a positive number"
    assert elapsed < _FILTERED_COUNT_BUDGET_S, (
        f"filtered active_records_count took {elapsed:.3f}s — exceeds the "
        f"{_FILTERED_COUNT_BUDGET_S}s budget. The drift gate issues this COUNT "
        f"on every boot; a scan-resistant read keeps it O(corpus) with no cache "
        f"thrash so it stays sub-second."
    )


# ---------------------------------------------------------------------------
# Test 5 — warm runtime-graph build cache-HIT budget
#
# A prewarmed runtime-graph cache must be recognised as fresh and rebuilt
# cheaply (cache HIT) on the next build — never a cold full rebuild. RED before
# the cache-freshness alignment (the saved payload count counted tombstoned
# rows, so the drift gate rejected the cache and ran a ~60-80 s cold rebuild
# on every boot); GREEN after (a few-second warm build).
# ---------------------------------------------------------------------------

def test_warm_build_runtime_graph_is_cache_hit(lilli_boot_store) -> None:
    """Second build_runtime_graph on a prewarmed store is a fast cache HIT.

    The first build populates and saves the cache. The second must recognise it
    as fresh (payload_record_count == active_records_count, both excluding
    tombstoned rows) and reconstruct cheaply, well under the cold-rebuild floor.
    """
    from iai_mcp.retrieve import build_runtime_graph, _runtime_graph_rebuild_needed

    store = lilli_boot_store

    # Prewarm: the first build computes communities + centrality and saves the
    # cache. (Cost not asserted — it is the one-time migration prewarm, off the
    # daemon boot path.)
    build_runtime_graph(store)

    # The freshly-saved cache must be recognised as not-needing-rebuild.
    assert _runtime_graph_rebuild_needed(store) is False, (
        "a freshly-prewarmed cache must be recognised as fresh — the saved "
        "payload_record_count must equal active_records_count (both excluding "
        "tombstoned records) so the drift gate accepts the cache"
    )

    # Warm build: must be a cache HIT, a few seconds, not a cold rebuild.
    t0 = time.monotonic()
    graph, _assignment, _rich_club = build_runtime_graph(store)
    elapsed = time.monotonic() - t0

    assert sum(1 for _ in graph.iter_nodes()) > 0, "warm build must produce a non-empty graph"
    assert elapsed < _WARM_BUILD_BUDGET_S, (
        f"warm build_runtime_graph took {elapsed:.3f}s — exceeds the "
        f"{_WARM_BUILD_BUDGET_S}s budget. A prewarmed cache must HIT (cheap "
        f"reconstruction), not run a cold full rebuild (~60-80 s). Check the "
        f"cache-freshness drift gate: the graph stream must exclude tombstoned "
        f"records so payload_record_count matches active_records_count."
    )
