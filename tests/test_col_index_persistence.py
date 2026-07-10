"""Host-level contracts for the persisted lilliengine column-index sidecar.

The Rust engine (``crates/lilliengine``) can build, persist, and load the
``col IN (...)`` secondary index (``ColIndex``) to an on-disk sidecar
(``records.colindex``) gated by a per-table write-generation fingerprint.
These tests prove the HOST side of that contract: a fresh ``HippoDB`` open
loads the sidecar instead of paying the cold full-scan build, a write after
persist correctly invalidates the sidecar, and the nightly sleep-cycle step
re-emits the sidecar at the end of every consolidation pass.

Lilli-only except the explicit stdlib no-op case: the stdlib sqlite3 driver
has persistent B-tree secondary indexes and no cold-ColIndex class at all.

Every store here lives under ``tmp_path`` — never a real or prod store.
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from iai_mcp.hippo import _txn
from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
from iai_mcp.store import MemoryStore

_LILLI = os.environ.get("LILLI_STORAGE_DRIVER", "").lower() == "lilli"

_EMBED_DIM = 384


def _require_native_persist_surface() -> None:
    """Hard-fail (not skip) if the native module is present but stale.

    The engine surface (``Connection.persist_col_indexes``) shipped in the
    prior wave; if the installed extension lacks it, the maturin release
    build has not been refreshed and every test below would be measuring the
    wrong binary.
    """
    try:
        from iai_mcp_native import engine
    except ImportError:
        return  # native module entirely absent — handled by driver skip below
    assert hasattr(engine.Connection, "persist_col_indexes"), (
        "iai_mcp_native.engine.Connection is missing persist_col_indexes — "
        "the native extension is stale. Run "
        "`cd rust/iai_mcp_native && maturin develop --release` before "
        "running this suite."
    )


if _LILLI:
    _require_native_persist_surface()


_lilli_only = pytest.mark.skipif(
    not _LILLI,
    reason="column-index persistence is lilli-only; skipped under stdlib "
    "(test_stdlib_driver_untouched runs unconditionally instead)",
)


def _norm_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(_EMBED_DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def _embedding_blob(seed: int) -> bytes:
    import struct
    vec = _norm_vec(seed)
    return struct.pack(f"<{_EMBED_DIM}f", *vec)


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path)


def _seed_records(store: MemoryStore, n: int) -> list[int]:
    """Insert n plain records directly via SQL; return their vec_labels.

    Bypasses the embedder entirely (raw float32 blobs), mirroring the
    fixture style already proven in tests/test_daemon_hippo_boot.py and
    tests/test_lilli_boot_perf.py — no embedder needed, deterministic.
    """
    db = store.db
    now_iso = datetime.now(timezone.utc).isoformat()
    rows = []
    for i in range(n):
        rid = str(uuid.uuid4())
        rows.append((
            rid,
            "episodic",
            None,
            _embedding_blob(i),
            None,   # tombstoned_at
            0,      # embedding_pending
            now_iso,
            now_iso,
        ))
    sql = (
        "INSERT INTO records "
        "(id, tier, literal_surface, embedding, tombstoned_at, "
        " embedding_pending, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    )
    ids = [r[0] for r in rows]
    with db._conn_lock:
        with _txn(db._conn):
            db._conn.executemany(sql, rows)

    with db._conn_lock:
        placeholders = ", ".join("?" for _ in ids)
        cur = db._conn.execute(
            f"SELECT vec_label FROM records WHERE id IN ({placeholders}) "
            "ORDER BY vec_label",
            ids,
        )
        return [int(row[0]) for row in cur.fetchall()]


def _seed_edges(store: MemoryStore, n: int) -> None:
    db = store.db
    rows = []
    for i in range(n):
        src = str(uuid.uuid4())
        dst = str(uuid.uuid4())
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


def _vec_label_in_read(store: MemoryStore, labels: list[int]) -> list[tuple]:
    """Issue the exact col-IN-shaped read production uses in query_similar
    (hippo/_table.py: SELECT * FROM records WHERE vec_label IN (...))."""
    db = store.db
    placeholders = ", ".join("?" for _ in labels)
    sql = f"SELECT vec_label FROM records WHERE vec_label IN ({placeholders})"
    with db._conn_lock:
        cur = db._conn.execute(sql, labels)
        return cur.fetchall()


def _sidecar_path(store: MemoryStore) -> Path:
    return Path(store.root) / "hippo" / "records.colindex"


def _lilli_conn(store: MemoryStore):
    conn = getattr(store.db, "_conn", None)
    if conn is None or not hasattr(conn, "persist_col_indexes"):
        pytest.fail(
            "store.db._conn does not expose persist_col_indexes — the "
            "native engine surface is missing or stale."
        )
    return conn


# ---------------------------------------------------------------------------
# Contract 1: a fresh open with a persisted, matching sidecar pays zero scans
# — even on a HELD-OUT vec_label never queried before the persist.
# ---------------------------------------------------------------------------

@_lilli_only
def test_fresh_open_with_sidecar_pays_zero_scans(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    labels = _seed_records(store, 200)
    _seed_edges(store, 60)

    probed = labels[:5]
    held_out = labels[-1:]

    conn = _lilli_conn(store)
    # Build lazily via one representative read, then persist.
    _vec_label_in_read(store, probed)
    conn.persist_col_indexes()
    store.close()

    assert _sidecar_path(store).exists(), "persist must create records.colindex"

    reopened = _make_store(tmp_path)
    rconn = _lilli_conn(reopened)
    rconn.reset_full_scan_count()

    rows = _vec_label_in_read(reopened, held_out)

    assert rconn.full_scan_count() == 0, (
        "a fresh open with a matching persisted sidecar must serve a "
        "held-out vec_label with zero full scans — a loaded sidecar is "
        "complete-by-construction (every row was filed at persist time), "
        "not just the probed subset."
    )
    assert [r[0] for r in rows] == list(held_out)
    reopened.close()


# ---------------------------------------------------------------------------
# Contract 2: a write after persist advances the write-generation; the next
# fresh open discards the stale sidecar and lazily rebuilds (correct, slower).
# ---------------------------------------------------------------------------

@_lilli_only
def test_stale_sidecar_lazy_rebuilds(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    labels = _seed_records(store, 50)

    conn = _lilli_conn(store)
    _vec_label_in_read(store, labels[:3])
    conn.persist_col_indexes()

    # A committed write AFTER the persist advances the table's generation.
    new_labels = _seed_records(store, 5)
    store.close()

    reopened = _make_store(tmp_path)
    rconn = _lilli_conn(reopened)
    rconn.reset_full_scan_count()
    rconn.reset_cells_visited_count()

    all_labels = labels + new_labels
    rows = _vec_label_in_read(reopened, all_labels)

    assert sorted(r[0] for r in rows) == sorted(all_labels), (
        "the post-persist rows must be visible — a diverged sidecar must "
        "never be trusted"
    )
    # The probe query (`WHERE vec_label IN (...)`) is a selective projection,
    # so a rebuilt-by-scan fallback runs through the column-selective streaming
    # scan, not the whole-tree materializing walk — `full_scan_count()` never
    # ticks for that shape by design. `cells_visited_count()` is the counter
    # that observes it.
    assert rconn.cells_visited_count() >= len(all_labels), (
        "a stale (generation-mismatched) sidecar must be discarded and the "
        "lazy rebuild must scan the corpus (expected cells_visited_count() "
        f">= {len(all_labels)}, got {rconn.cells_visited_count()} — this "
        "would mean a stale sidecar was silently trusted)"
    )
    reopened.close()


# ---------------------------------------------------------------------------
# Contract 3: the RECALL_INDEX_REBUILD sleep step emits a fresh sidecar as
# the FINAL act of every cycle (end-of-cycle re-emission, not a one-time
# artifact).
# ---------------------------------------------------------------------------

@_lilli_only
def test_recall_index_rebuild_step_emits_sidecar(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _seed_records(store, 30)

    pipeline = SleepPipeline(store)
    done, payload = pipeline._step_recall_index_rebuild(None)

    assert done is True, "the step must return done=True"
    assert "colindex_persist_elapsed_sec" in payload, (
        f"the step result must carry the persist timing key; got {payload}"
    )
    sidecar = _sidecar_path(store)
    assert sidecar.exists(), (
        "RECALL_INDEX_REBUILD must persist records.colindex as its final act"
    )
    first_mtime = sidecar.stat().st_mtime_ns

    # A write, then a second cycle: the sidecar must be RE-EMITTED with a
    # fresh stamp (end-of-cycle cadence, not a persist-once artifact).
    time.sleep(0.01)
    _seed_records(store, 5)
    done2, payload2 = pipeline._step_recall_index_rebuild(None)

    assert done2 is True
    assert "colindex_persist_elapsed_sec" in payload2
    second_mtime = sidecar.stat().st_mtime_ns
    assert second_mtime > first_mtime, (
        "the second sleep cycle must re-emit the sidecar with a newer mtime"
    )
    store.close()


# ---------------------------------------------------------------------------
# Contract 4: a persist failure inside the step can NEVER fail the step or
# discard the graph-cache result already computed in the same step body.
# ---------------------------------------------------------------------------

@_lilli_only
def test_persist_failure_never_fails_the_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store(tmp_path)
    _seed_records(store, 10)

    # Confirm the connection actually exposes the surface before breaking it.
    _lilli_conn(store)

    from iai_mcp_native import engine

    def _boom(self) -> None:
        raise RuntimeError("simulated persist failure")

    # The PyO3 Connection class does not allow per-instance attribute
    # overrides (read-only slot descriptor) — patch the class method instead.
    monkeypatch.setattr(engine.Connection, "persist_col_indexes", _boom, raising=True)

    pipeline = SleepPipeline(store)
    done, payload = pipeline._step_recall_index_rebuild(None)

    assert done is True, (
        "a persist failure must never fail the step — the step must "
        "degrade gracefully, not crash the sleep pipeline"
    )
    assert "error" not in payload or payload.get("rebuilt") is not False, (
        f"a colindex-persist failure must NOT be converted into the "
        f"step's own {{'error': ..., 'rebuilt': False}} shape — the "
        f"graph-cache outcome computed earlier in the same step body must "
        f"survive; got {payload}"
    )
    assert payload.get("colindex_persist_error"), (
        f"the persist failure must be reported via colindex_persist_error; "
        f"got {payload}"
    )
    assert not _sidecar_path(store).exists() or _sidecar_path(store).stat().st_size >= 0
    store.close()


# ---------------------------------------------------------------------------
# Contract 5: under the stdlib driver, the step is unaffected — no sidecar,
# no error, no behavior change. Runs regardless of LILLI_STORAGE_DRIVER.
# ---------------------------------------------------------------------------

def test_stdlib_driver_untouched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    store = MemoryStore(path=tmp_path)
    assert getattr(store.db, "_storage_driver", "stdlib") == "stdlib"

    _seed_records(store, 10)

    pipeline = SleepPipeline(store)
    done, payload = pipeline._step_recall_index_rebuild(None)

    assert done is True
    assert "colindex_persist_elapsed_sec" not in payload, (
        "the stdlib driver must never emit the colindex persist timing key"
    )
    assert "colindex_persist_error" not in payload
    assert not _sidecar_path(store).exists(), (
        "the stdlib driver must never create records.colindex"
    )
    store.close()
