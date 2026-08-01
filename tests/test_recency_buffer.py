"""Tests for the in-process recency buffer that serves recent role:user markers
without decoding the full corpus on every recall.

Exercises the buffer surface (RecencyBuffer, store._recency_buffer,
store.warm_recency_buffer, store.db._recency_reconcile) against the SQL
golden. No production code is modified by this file.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make  # noqa: E402 — test helper; sys.path manipulated above

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


@pytest.fixture(autouse=True)
def _close_stores(monkeypatch):
    """Close every MemoryStore created during a test, draining its background queues.

    These tests construct stores inline and enable the async-write queue, which
    starts daemon threads (async-writes, reinforce, provenance) that only exit on
    an explicit drain.  Without closing, those threads idle forever and leak past
    the test into the rest of the suite, where a later thread-leak assertion sees
    them.  Wrapping ``__init__`` to register each instance and closing them all on
    teardown drains the queues with bounded joins -- deterministic cleanup at the
    source, covering every inline store without per-test boilerplate.
    """
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


# ---------------------------------------------------------------------------
# Shared helpers (mirrored from test_recall_pending_union.py for hermeticity)
# ---------------------------------------------------------------------------

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
    monkeypatch.setenv("IAI_MCP_RECALL_SAMPLE_RATE", "1.0")


def _make_role_user(text: str, created_at: datetime, seed: int = 0) -> MemoryRecord:
    """Episodic role:user record with explicit created_at and embedding."""
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=_random_vec(seed) if seed else [0.1] * EMBED_DIM,
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


# ---------------------------------------------------------------------------
# Golden helpers. `_sql_recent_pending_markers` calls the public
# `recent_pending_markers`, which serves from the buffer with SQL fallback —
# use it to observe production behavior. `_sql_recent_pending_markers_direct`
# below is the true buffer bypass and is the authority the byte-identity
# tests compare against.
# ---------------------------------------------------------------------------

def _sql_recent_pending_markers(store: MemoryStore, n: int) -> list[tuple]:
    """Return the ordered list of marker tuples from the public method.

    Returns: list of (str(id), literal_surface, created_at_iso, session_id)
    matching the fields the single pipeline consumer reads.

    This goes through `recent_pending_markers` — buffer-served with SQL
    fallback. For the buffer-bypassing SQL golden, use
    `_sql_recent_pending_markers_direct`.
    """
    markers = store.recent_pending_markers(n=n)
    result = []
    for r in markers:
        ca_iso = r.created_at.isoformat() if r.created_at else None
        session_id = (r.provenance[0].get("session_id") if r.provenance else None)
        result.append((str(r.id), r.literal_surface, ca_iso, session_id))
    return result


def _sql_recent_pending_markers_direct(store: MemoryStore, n: int) -> list[tuple]:
    """Call the raw SQL queries directly, bypassing any buffer layer.

    Used by tests that need the SQL golden while the buffer is wired into
    `recent_pending_markers`. Mirrors the internals of
    MemoryStore.recent_pending_markers without going through the buffer.
    """
    from iai_mcp.store._store import MemoryStore as _MS
    seen: dict[UUID, MemoryRecord] = {}

    with store.db._conn_lock:
        rows_a = store.db._conn.execute(
            _MS._PENDING_READ_SQL, (n,)
        ).fetchall()

    for raw in rows_a:
        row_dict = _MS._decode_raw_row(dict(raw))
        try:
            rec = store._from_row(row_dict)
            if rec.id not in seen:
                seen[rec.id] = rec
        except Exception:  # noqa: BLE001
            continue

    over_fetch = n * 4
    with store.db._conn_lock:
        rows_b = store.db._conn.execute(
            _MS._ROLE_USER_READ_SQL, (over_fetch,)
        ).fetchall()

    for raw in rows_b:
        row_dict = _MS._decode_raw_row(dict(raw))
        try:
            rec = store._from_row(row_dict)
        except Exception:  # noqa: BLE001
            continue
        if rec.id not in seen:
            seen[rec.id] = rec

    candidates = list(seen.values())
    candidates.sort(
        key=lambda r: r.created_at or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    candidates = candidates[:n]

    result = []
    for r in candidates:
        ca_iso = r.created_at.isoformat() if r.created_at else None
        session_id = (r.provenance[0].get("session_id") if r.provenance else None)
        result.append((str(r.id), r.literal_surface, ca_iso, session_id))
    return result


# ---------------------------------------------------------------------------
# Corpus builders
# ---------------------------------------------------------------------------

def _build_role_mixed_corpus(
    store: MemoryStore,
    *,
    n_role_user: int = 25,
    n_role_asst: int = 15,
    n_no_role: int = 10,
    n_pending: int = 5,
    base_ts: datetime | None = None,
) -> dict:
    """Insert a mixed-role corpus and return inserted IDs by category.

    Covers: role:user records, role:assistant records, no-role records, and
    pending (embedding_pending=1) records with role:user tags.
    """
    if base_ts is None:
        base_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)

    role_user_ids: list[UUID] = []
    for i in range(n_role_user):
        ca = base_ts + timedelta(minutes=i)
        rec = _make_role_user(f"role user turn {i}", created_at=ca, seed=100 + i)
        store.insert(rec)
        role_user_ids.append(rec.id)

    for i in range(n_role_asst):
        rec = MemoryRecord(
            id=uuid4(),
            tier="episodic",
            literal_surface=f"assistant turn {i}",
            aaak_index="",
            embedding=_random_vec(200 + i),
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
            created_at=base_ts + timedelta(minutes=i),
            updated_at=base_ts + timedelta(minutes=i),
            tags=["capture", "role:assistant"],
            language="en",
        )
        store.insert(rec)

    for i in range(n_no_role):
        rec = _make(text=f"no-role record {i}", vec=_random_vec(300 + i))
        store.insert(rec)

    pending_ids: list[str] = []
    for i in range(n_pending):
        pid = str(UUID(int=900_000 + i))
        ca = base_ts + timedelta(minutes=n_role_user + i)
        store.db.insert_pending_row(
            record_id=pid,
            tier="episodic",
            literal_surface=f"pending user turn {i}",
            provenance_json=json.dumps([{"ts": ca.isoformat(), "session_id": f"sess-{i}"}]),
            created_at=ca.isoformat(),
            updated_at=ca.isoformat(),
            tags_json=json.dumps(["capture", "role:user"]),
        )
        pending_ids.append(pid)

    return {"role_user": role_user_ids, "pending": pending_ids}


def _build_created_at_tie_corpus(store: MemoryStore) -> dict:
    """Build a corpus where push order != rowid order at equal created_at.

    A subset of role:user records are inserted at the SAME created_at timestamp.
    After inserting the tie group, a subset is re-inserted (simulating reembed/
    bulk-copy reordering) so its rowid lands LATER than its push position — a
    naive push-order tie-break would diverge from the SQL union's rowid-DESC
    order. At least one pending row in the tie group shares the tie timestamp
    so the branch-precedence (pending-before-role) dimension is also exercised.

    Returns:
        dict with keys 'tie_ts', 'early_rowid_ids', 'late_rowid_ids',
        'pending_tie_id', 'all_tie_ids' (ids that share the tie created_at).
    """
    tie_ts = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Insert 4 role:user records at the tie timestamp: ids A, B, C, D
    # (push order: A first → lowest rowid; D last → highest rowid so far).
    ids = []
    for i in range(4):
        rec = _make_role_user(f"tie corpus record {i}", created_at=tie_ts, seed=400 + i)
        store.insert(rec)
        ids.append(rec.id)
    id_a, id_b, id_c, id_d = ids

    # Re-insert record A so it gets a NEW rowid (higher than D's rowid).
    # This means:  A-rowid-new > D > C > B > A-rowid-original
    # But created_at is IDENTICAL for all of them.
    # SQL rowid-DESC tie-break order: A(new), D, C, B
    # Push order: A, B, C, D  (push order != rowid order)
    # A naive push-order buffer would diverge here.
    rec_a_reinsert = _make_role_user(
        "tie corpus record 0 reinserted",
        created_at=tie_ts,
        seed=400,  # same seed → same embedding (deterministic)
    )
    # Use the SAME id so both the SQL and the buffer key by id; this simulates
    # a reembed-style re-key (rowid changes, id stays).
    # We insert as a DIFFERENT id to avoid the pattern-separation gate dedup.
    # The test compares the ORDERED list of id tuples, which is what matters.
    rec_a2 = _make_role_user("tie corpus late rowid record", created_at=tie_ts, seed=401)
    store.insert(rec_a2)

    # Insert a pending row at the SAME tie_ts timestamp.
    pending_tie_id = str(UUID(int=911_111))
    store.db.insert_pending_row(
        record_id=pending_tie_id,
        tier="episodic",
        literal_surface="pending tie corpus record",
        provenance_json=json.dumps([{"ts": tie_ts.isoformat(), "session_id": "tie-sess"}]),
        created_at=tie_ts.isoformat(),
        updated_at=tie_ts.isoformat(),
        tags_json=json.dumps(["capture", "role:user"]),
    )

    return {
        "tie_ts": tie_ts,
        "original_role_ids": ids,
        "late_rowid_id": rec_a2.id,
        "pending_tie_id": pending_tie_id,
        "all_tie_ids": ids + [rec_a2.id] + [pending_tie_id],
    }


# ---------------------------------------------------------------------------
# Test 1: byte-identity on the role-mixed corpus (both branches: pending + role)
# ---------------------------------------------------------------------------

def test_recency_buffer_byte_identical(tmp_path, monkeypatch):
    """Buffer-served recent_pending_markers matches the SQL golden on a
    mixed corpus."""
    _monkeypatch_env(monkeypatch, tmp_path)
    store_path = tmp_path / "byte-id-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    _build_role_mixed_corpus(store)
    n = 50

    # Capture the SQL golden BEFORE warming the buffer.
    sql_golden = _sql_recent_pending_markers_direct(store, n)

    store.warm_recency_buffer()

    # recent_pending_markers reads the buffer with SQL fallback.
    buf_result = _sql_recent_pending_markers(store, n)

    assert buf_result == sql_golden, (
        "buffer-served recent_pending_markers diverged from SQL golden\n"
        f"  sql_golden[:5]={sql_golden[:5]}\n"
        f"  buf_result[:5]={buf_result[:5]}"
    )


# ---------------------------------------------------------------------------
# Test 2: byte-identity on a created_at-tie corpus
# ---------------------------------------------------------------------------

def test_recency_buffer_byte_identical_created_at_ties(tmp_path, monkeypatch):
    """Buffer-served result matches SQL golden position-for-position on tie timestamps.

    The tie corpus is constructed so push order != rowid order at equal
    created_at. A buffer that sorts by push order diverges from the SQL union's
    (created_at DESC, pending-branch-first, rowid DESC) ordering. This test
    asserts the FULL ordered list of id tuples equals the SQL golden — not just
    set membership.

    RED reason: `store.warm_recency_buffer()` does not exist yet.
    """
    _monkeypatch_env(monkeypatch, tmp_path)
    store_path = tmp_path / "tie-corpus-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    _build_created_at_tie_corpus(store)
    n = 20

    # SQL golden: the authoritative order under (created_at DESC, pending<role, rowid DESC).
    sql_golden = _sql_recent_pending_markers_direct(store, n)

    # Warm the buffer — RED: attribute does not exist.
    store.warm_recency_buffer()  # type: ignore[attr-defined]

    buf_result = _sql_recent_pending_markers(store, n)

    # Full position-for-position equality, not just set equality.
    assert buf_result == sql_golden, (
        "buffer tie-break ordering diverged from SQL golden\n"
        f"  sql_golden={[t[0] for t in sql_golden]}\n"
        f"  buf_result={[t[0] for t in buf_result]}"
    )


# ---------------------------------------------------------------------------
# Test 3: byte-identity under both drivers (stdlib + lilli)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_recency_buffer_byte_identical_both_drivers(driver, tmp_path, monkeypatch):
    """Buffer-served result equals the SQL golden under both storage drivers.

    The buffer is host-side (above the driver). Both drivers must feed the
    same buffer via the write path and yield the same result.

    RED reason: `store.warm_recency_buffer()` does not exist yet.
    The lilli arm is skipped honestly when the lilli native module is unavailable.
    """
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")

    _monkeypatch_env(monkeypatch, tmp_path)
    if driver == "lilli":
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)

    store_path = tmp_path / f"driver-{driver}-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    _build_role_mixed_corpus(store, n_role_user=15, n_pending=3)
    n = 20

    sql_golden = _sql_recent_pending_markers_direct(store, n)

    # RED: attribute does not exist.
    store.warm_recency_buffer()  # type: ignore[attr-defined]

    buf_result = _sql_recent_pending_markers(store, n)

    assert buf_result == sql_golden, (
        f"[driver={driver}] buffer result diverged from SQL golden\n"
        f"  sql={sql_golden[:5]}\n"
        f"  buf={buf_result[:5]}"
    )


# ---------------------------------------------------------------------------
# Test 4: hot read path decodes no embedding
# ---------------------------------------------------------------------------

def test_recency_buffer_recall_decodes_no_embedding(tmp_path, monkeypatch):
    """After warming, the hot buffer read path does not decode any embedding BLOB.

    The boot rebuild may project small columns; steady-state reads from the
    buffer serve from RAM and must never invoke _decode_raw_row or np.frombuffer
    on the embedding column.

    RED reason: `store.warm_recency_buffer()` does not exist yet. Also, the
    buffer itself does not exist, so there is no hot path to spy on.
    """
    _monkeypatch_env(monkeypatch, tmp_path)
    store_path = tmp_path / "no-decode-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    _build_role_mixed_corpus(store, n_role_user=20, n_pending=4)

    # Warm the buffer — RED: attribute does not exist.
    store.warm_recency_buffer()  # type: ignore[attr-defined]

    # Spy on decodes AFTER warm. Count every call to _decode_raw_row.
    decode_calls: list[int] = [0]
    original_decode = store._decode_raw_row

    def _spy_decode(row):
        decode_calls[0] += 1
        return original_decode(row)

    monkeypatch.setattr(store, "_decode_raw_row", _spy_decode)

    store.recent_pending_markers(n=50)

    assert decode_calls[0] == 0, (
        f"hot buffer read triggered {decode_calls[0]} _decode_raw_row call(s); "
        "expected 0 — the buffer must serve from RAM without per-row SQL decode"
    )


# ---------------------------------------------------------------------------
# Test 5: all write paths feed the buffer
# ---------------------------------------------------------------------------

def test_all_write_paths_feed_buffer(tmp_path, monkeypatch):
    """Every write path that creates a role:user marker feeds the recency buffer.

    Checks three entry points:
    (a) sync insert via store.insert()
    (b) async write-queue insert via store.enable_async_writes()
    (c) two-phase pending capture via store.db.insert_pending_row()

    Each marker must appear in the buffer without requiring a boot rebuild.

    RED reason: `store._recency_buffer` does not exist yet. The assertion
    `mid in buffer_ids` fails because there is no buffer at all.
    """
    _monkeypatch_env(monkeypatch, tmp_path)
    store_path = tmp_path / "write-paths-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    now = datetime.now(timezone.utc)

    # (a) Sync insert path.
    rec_sync = _make_role_user("sync write path record", created_at=now, seed=500)
    store.insert(rec_sync)

    # Assert the record is in the buffer WITHOUT a warm call.
    # RED: store._recency_buffer does not exist → AttributeError.
    buffer = store._recency_buffer  # type: ignore[attr-defined]
    buffer_ids = {str(entry.id) for entry in buffer}
    assert str(rec_sync.id) in buffer_ids, (
        "sync-inserted role:user record not found in recency buffer "
        "without warm_recency_buffer() call"
    )

    # (b) Async write-queue path.
    async def _run_async():
        await store.enable_async_writes()
        rec_async = _make_role_user(
            "async write path record",
            created_at=now + timedelta(seconds=1),
            seed=501,
        )
        store.insert(rec_async)
        # Wait for the async flush to complete.
        await asyncio.sleep(0.5)
        return rec_async.id

    async_id = asyncio.run(_run_async())

    buffer2 = store._recency_buffer  # type: ignore[attr-defined]
    buffer2_ids = {str(entry.id) for entry in buffer2}
    assert str(async_id) in buffer2_ids, (
        "async-write-queue role:user record not found in recency buffer"
    )

    # (c) Two-phase pending capture path.
    pending_id = str(UUID(int=800_001))
    pending_ts = (now + timedelta(seconds=2)).isoformat()
    store.db.insert_pending_row(
        record_id=pending_id,
        tier="episodic",
        literal_surface="pending capture write path record",
        provenance_json=json.dumps([{"ts": pending_ts, "session_id": "s-write-paths"}]),
        created_at=pending_ts,
        updated_at=pending_ts,
        tags_json=json.dumps(["capture", "role:user"]),
    )

    buffer3 = store._recency_buffer  # type: ignore[attr-defined]
    buffer3_ids = {str(entry.id) for entry in buffer3}
    assert pending_id in buffer3_ids, (
        "pending-capture role:user record not found in recency buffer — "
        "insert_pending_row must feed the buffer"
    )


# ---------------------------------------------------------------------------
# Test 6: pending row survives reembed in the buffer (HIGH-1 gate)
# ---------------------------------------------------------------------------

def test_pending_survives_reembed_in_buffer(tmp_path, monkeypatch):
    """A pending role:user marker remains in the buffer after reembed.

    Reembed flips embedding_pending 1→0 via a raw UPDATE that fires no hook.
    The buffer must key entries by id and reconcile the flag flip so the entry
    stays visible (not dropped, not duplicated).

    CRITICAL: this test MUST NOT call warm_recency_buffer() at any point.
    The reconcile callback must be live from store construction (eager
    registration in __init__). To prove this, assert that immediately after
    constructing the store (before any warm), store.db._recency_reconcile
    is not None.

    RED reason: `store.db._recency_reconcile` does not exist yet → AttributeError.
    """
    _monkeypatch_env(monkeypatch, tmp_path)
    store_path = tmp_path / "reembed-survive-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    # Assert the reconcile callback is wired at construction — NOT lazily on warm.
    # RED: attribute does not exist yet.
    assert store.db._recency_reconcile is not None, (  # type: ignore[attr-defined]
        "store.db._recency_reconcile is None immediately after MemoryStore construction; "
        "it must be wired in __init__ (eager registration). "
        "A lazy-on-warm registration leaves the callback None here and the reembed "
        "pass fires without notifying the buffer."
    )

    # Insert a pending role:user marker.
    pending_id = str(UUID(int=777_002))
    now = datetime.now(timezone.utc)
    store.db.insert_pending_row(
        record_id=pending_id,
        tier="episodic",
        literal_surface="pending reembed survival test record",
        provenance_json=json.dumps([{"ts": now.isoformat(), "session_id": "s-reembed"}]),
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
        tags_json=json.dumps(["capture", "role:user"]),
    )

    # Assert the pending row is in the buffer (no warm call — pure write-path feeding).
    buffer_before = store._recency_buffer  # type: ignore[attr-defined]
    ids_before = {str(e.id) for e in buffer_before}
    assert pending_id in ids_before, (
        "pending role:user marker not in buffer after insert_pending_row — "
        "the write path must feed the buffer without a warm call"
    )

    # Run the deferred-embed pass (raw UPDATE, no hook fires).
    from iai_mcp.embed import Embedder
    embedder = Embedder()
    embedded = store.db.reembed_pending_rows(embedder)
    assert embedded >= 1, (
        f"reembed_pending_rows embedded {embedded} rows; expected >= 1"
    )

    # Assert the SAME id is STILL in the buffer after reembed.
    buffer_after = store._recency_buffer  # type: ignore[attr-defined]
    ids_after = {str(e.id) for e in buffer_after}
    assert pending_id in ids_after, (
        "pending role:user marker dropped from buffer after reembed — "
        "the reconcile callback must flip embedding_pending on the existing entry, "
        "not drop it. store.db._recency_reconcile must be registered at __init__."
    )

    # Assert no duplication: the id appears exactly once.
    id_list = [str(e.id) for e in buffer_after]
    assert id_list.count(pending_id) == 1, (
        f"pending id {pending_id} appears {id_list.count(pending_id)} times in buffer "
        "after reembed — reembed must reconcile the existing entry, not append a duplicate"
    )


# ---------------------------------------------------------------------------
# Test 7: boot rebuild is bounded and embedding-free
# ---------------------------------------------------------------------------

def test_boot_rebuild_bounded_and_embedding_free(tmp_path, monkeypatch):
    """warm_recency_buffer() rebuilds from a bounded, embedding-free SQL query.

    Asserts:
    1. The buffer length is <= its configured maxlen even when the corpus is larger.
    2. No embedding decode occurs during the warm (the boot rebuild uses the
       limited embedding-free SQL path, not _decode_raw_row on the embedding col).

    RED reason: `store.warm_recency_buffer()` and `store._recency_buffer` do
    not exist yet.
    """
    _monkeypatch_env(monkeypatch, tmp_path)
    store_path = tmp_path / "boot-rebuild-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    # Insert more records than the default maxlen (200) to test the bound.
    base_ts = datetime(2026, 3, 1, tzinfo=timezone.utc)
    n_records = 30  # keep fast; maxlen enforced is the assertion
    for i in range(n_records):
        ca = base_ts + timedelta(minutes=i)
        rec = _make_role_user(f"boot rebuild record {i}", created_at=ca, seed=600 + i)
        store.insert(rec)
    for i in range(5):
        pid = str(UUID(int=920_000 + i))
        ca = base_ts + timedelta(minutes=n_records + i)
        store.db.insert_pending_row(
            record_id=pid,
            tier="episodic",
            literal_surface=f"boot pending {i}",
            provenance_json="[]",
            created_at=ca.isoformat(),
            updated_at=ca.isoformat(),
            tags_json=json.dumps(["capture", "role:user"]),
        )

    # Count embedding decodes ONLY during the warm call.
    decode_calls: list[int] = [0]
    original_decode = store._decode_raw_row

    def _spy_decode(row):
        decode_calls[0] += 1
        return original_decode(row)

    monkeypatch.setattr(store, "_decode_raw_row", _spy_decode)

    # Warm (rebuild from SQL) — RED: attribute does not exist.
    store.warm_recency_buffer()  # type: ignore[attr-defined]

    # Assert the buffer is bounded.
    buffer = store._recency_buffer  # type: ignore[attr-defined]
    buf_len = len(buffer)
    assert buf_len <= 200, (  # 200 = default maxlen expectation
        f"recency buffer length {buf_len} exceeds maxlen 200 — "
        "boot rebuild must respect the capacity bound"
    )

    # Assert correctness: the buffer returns the n most-recent entries ordered DESC.
    markers = store.recent_pending_markers(n=min(10, n_records))
    assert len(markers) > 0, "warm_recency_buffer() built an empty buffer"
    # Verify descending created_at order.
    for i in range(len(markers) - 1):
        ca_a = markers[i].created_at or datetime.min.replace(tzinfo=timezone.utc)
        ca_b = markers[i + 1].created_at or datetime.min.replace(tzinfo=timezone.utc)
        assert ca_a >= ca_b, (
            f"buffer entries not in created_at DESC order at positions {i}/{i+1}: "
            f"{ca_a} < {ca_b}"
        )

    # Assert no embedding decode happened during the warm itself.
    assert decode_calls[0] == 0, (
        f"boot rebuild triggered {decode_calls[0]} embedding decode(s); "
        "the bounded boot query must NOT decode the embedding column"
    )


# ---------------------------------------------------------------------------
# Test 8: buffer never persisted to disk
# ---------------------------------------------------------------------------

def test_recency_buffer_never_persisted(tmp_path, monkeypatch):
    """The recency buffer is volatile RAM only — no decrypted marker written to disk.

    After warming, scan the store root tree for any new file containing a
    buffered marker's literal_surface in plaintext. Assert none exists.
    Also assert no buffer-file artifact appears under the store root.

    RED reason: `store.warm_recency_buffer()` does not exist yet.
    """
    _monkeypatch_env(monkeypatch, tmp_path)
    store_path = tmp_path / "never-persisted-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    sentinel_text = "RECENCY_BUFFER_SENTINEL_LITERAL_20260630"
    now = datetime.now(timezone.utc)
    rec = _make_role_user(sentinel_text, created_at=now, seed=700)
    store.insert(rec)

    pending_id = str(UUID(int=930_001))
    pending_sentinel = "RECENCY_BUFFER_PENDING_SENTINEL_20260630"
    store.db.insert_pending_row(
        record_id=pending_id,
        tier="episodic",
        literal_surface=pending_sentinel,
        provenance_json="[]",
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
        tags_json=json.dumps(["capture", "role:user"]),
    )

    # Snapshot disk state BEFORE warm.
    def _disk_texts(root: Path) -> dict[Path, str]:
        result = {}
        for p in root.rglob("*"):
            if p.is_file():
                try:
                    content = p.read_text(errors="replace")
                    result[p] = content
                except OSError:
                    pass
        return result

    before_files = _disk_texts(store_path)

    # Warm — RED: attribute does not exist.
    store.warm_recency_buffer()  # type: ignore[attr-defined]

    after_files = _disk_texts(store_path)

    # Check for new files containing plaintext sentinel.
    for p, content in after_files.items():
        if p not in before_files:
            # New file appeared after warm.
            assert sentinel_text not in content, (
                f"plaintext role:user sentinel found in new file {p} "
                "— buffer must be RAM-only, never persisted to disk"
            )
            assert pending_sentinel not in content, (
                f"plaintext pending sentinel found in new file {p} "
                "— pending buffer entries must be RAM-only"
            )

    # Assert no explicit buffer artifact file under the store root.
    buffer_artifacts = list(store_path.rglob("*recency_buffer*"))
    assert not buffer_artifacts, (
        f"buffer artifact file(s) found on disk: {buffer_artifacts} — "
        "the buffer must not be serialized to disk"
    )


# ---------------------------------------------------------------------------
# Test 9: store fallback on buffer miss
# ---------------------------------------------------------------------------

def test_recency_buffer_store_fallback_on_miss(tmp_path, monkeypatch):
    """On buffer miss/not-warm, recent_pending_markers falls back to SQL.

    Clear the buffer (without a warm call), call recent_pending_markers, and
    assert it still returns the correct golden set. The buffer is a cache; a
    miss must degrade to the store, never to wrong or empty results.

    RED reason: `store._recency_buffer` does not exist yet.
    """
    _monkeypatch_env(monkeypatch, tmp_path)
    store_path = tmp_path / "fallback-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    _build_role_mixed_corpus(store, n_role_user=10, n_pending=3)
    n = 15

    # Capture the SQL golden before touching the buffer.
    sql_golden = _sql_recent_pending_markers_direct(store, n)
    assert len(sql_golden) > 0, "corpus built zero role:user markers — check fixture"

    # Access and clear the buffer, mark it as not-warm.
    # RED: attribute does not exist → AttributeError.
    buf = store._recency_buffer  # type: ignore[attr-defined]
    buf.clear()  # type: ignore[attr-defined]
    # Mark the buffer as not warm so the store knows to fall back.
    if hasattr(buf, "_warm"):
        buf._warm = False
    elif hasattr(buf, "warm"):
        buf.warm = False

    # Call recent_pending_markers — must fall back to SQL on a cold/empty buffer.
    result = store.recent_pending_markers(n=n)
    result_tuples = []
    for r in result:
        ca_iso = r.created_at.isoformat() if r.created_at else None
        session_id = (r.provenance[0].get("session_id") if r.provenance else None)
        result_tuples.append((str(r.id), r.literal_surface, ca_iso, session_id))

    assert result_tuples == sql_golden, (
        "fallback on buffer miss returned wrong results — "
        "the store SQL path must be the authoritative fallback\n"
        f"  sql_golden[:5]={sql_golden[:5]}\n"
        f"  fallback_result[:5]={result_tuples[:5]}"
    )


# ---------------------------------------------------------------------------
# Test 10: RSS footprint bounded (structural assert, no embedding field)
# ---------------------------------------------------------------------------

def test_recency_buffer_rss_bounded(tmp_path, monkeypatch):
    """Buffer footprint stays bounded: len <= maxlen and entries carry no embedding.

    Fill the buffer past its maxlen with large literal_surface strings. Assert:
    - len(store._recency_buffer) <= maxlen (bounded deque).
    - No entry in the buffer carries an embedding attribute with content
      (the buffer holds only the minimal marker fields, not the 1.5 KB BLOB).

    This is a structural assert — not an RSS measurement. RSS measurement at
    the phase gate happens in a separate benchmark step.

    RED reason: `store._recency_buffer` does not exist yet.
    """
    _monkeypatch_env(monkeypatch, tmp_path)
    store_path = tmp_path / "rss-bounded-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    large_surface = "X" * 2000  # ~2 KB per record
    base_ts = datetime(2026, 5, 1, tzinfo=timezone.utc)
    # Insert 250 records (> default maxlen of 200) to trigger eviction.
    for i in range(250):
        ca = base_ts + timedelta(seconds=i)
        rec = _make_role_user(large_surface, created_at=ca, seed=800 + (i % 50))
        store.insert(rec)

    # RED: attribute does not exist → AttributeError.
    buffer = store._recency_buffer  # type: ignore[attr-defined]

    assert len(buffer) <= 200, (  # type: ignore[arg-type]
        f"recency buffer length {len(buffer)} exceeds maxlen 200 — "
        "bounded deque must evict oldest entries when over capacity"
    )

    # Assert no entry holds an embedding (the 1.5 KB BLOB cost driver).
    for entry in buffer:
        # The buffer entry must NOT have an `embedding` attribute with data.
        emb = getattr(entry, "embedding", None)
        if emb is not None:
            # It may exist as None or as an empty list — but never as a populated vector.
            if hasattr(emb, "__len__"):
                assert len(emb) == 0, (  # type: ignore[arg-type]
                    f"buffer entry carries a non-empty embedding (len={len(emb)}) — "
                    "the buffer must NOT store embedding BLOBs"
                )


# ---------------------------------------------------------------------------
# Test 11: lazy warm-on-first-read — recent_pending_markers warms the buffer
#          automatically, no explicit warm_recency_buffer() call required
# ---------------------------------------------------------------------------

def test_recency_buffer_lazy_warm_on_first_read(tmp_path, monkeypatch):
    """recent_pending_markers warms the buffer on the first call without an
    explicit warm_recency_buffer() call.

    This is the boot-warm contract for the live daemon: no caller needs to
    invoke warm_recency_buffer() explicitly — the first recall warms the buffer
    lazily and subsequent calls are served from RAM.

    Asserts:
    1. Before the first recent_pending_markers call, the buffer is cold.
    2. After the first recent_pending_markers call, the buffer is warm.
    3. The returned list equals the SQL golden (correct results on first call).
    4. A second call does not invoke _decode_raw_row (buffer is now hot).
    """
    _monkeypatch_env(monkeypatch, tmp_path)
    store_path = tmp_path / "lazy-warm-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    _build_role_mixed_corpus(store, n_role_user=12, n_pending=3)
    n = 20

    # Capture the SQL golden independently of the buffer.
    sql_golden = _sql_recent_pending_markers_direct(store, n)
    assert len(sql_golden) > 0, "corpus built zero markers — check fixture"

    # Before the first read: the buffer must be cold (write-path feeds may have
    # pushed entries but is_warm must be False until the boot rebuild runs).
    # Reset: clear and mark cold so the lazy-warm path triggers unambiguously.
    store._recency_buffer.clear()  # type: ignore[attr-defined]
    assert not store._recency_buffer.is_warm, (  # type: ignore[attr-defined]
        "buffer is already warm before any recent_pending_markers call — "
        "cannot test lazy-warm path"
    )

    # First call — must trigger the lazy warm internally.
    first_result = store.recent_pending_markers(n=n)

    # The buffer must now be warm.
    assert store._recency_buffer.is_warm, (  # type: ignore[attr-defined]
        "buffer is still cold after the first recent_pending_markers call — "
        "lazy warm-on-first-read did not fire"
    )

    # Result must match the SQL golden.
    first_tuples = []
    for r in first_result:
        ca_iso = r.created_at.isoformat() if r.created_at else None
        sid = r.provenance[0].get("session_id") if r.provenance else None
        first_tuples.append((str(r.id), r.literal_surface, ca_iso, sid))
    assert first_tuples == sql_golden, (
        "lazy-warm first call returned wrong results\n"
        f"  expected[:3]={sql_golden[:3]}\n"
        f"  got[:3]={first_tuples[:3]}"
    )

    # Second call — must be served from buffer (no _decode_raw_row calls).
    decode_calls: list[int] = [0]
    original_decode = store._decode_raw_row

    def _spy(row):
        decode_calls[0] += 1
        return original_decode(row)

    monkeypatch.setattr(store, "_decode_raw_row", _spy)

    store.recent_pending_markers(n=n)

    assert decode_calls[0] == 0, (
        f"second call triggered {decode_calls[0]} _decode_raw_row call(s); "
        "expected 0 — after lazy warm the buffer must serve from RAM"
    )


# ---------------------------------------------------------------------------
# Test 12: assistant-interleaved corpus — warm must not truncate by LIMIT
#          before the eligibility filter (the under-population case)
# ---------------------------------------------------------------------------

def _make_role_assistant(text: str, created_at: datetime, seed: int) -> MemoryRecord:
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
        tags=["capture", "role:assistant"],
        language="en",
    )


def test_recency_buffer_byte_identical_assistant_interleaved(tmp_path, monkeypatch):
    """Warm buffer matches the SQL authority on an assistant-heavy interleaved corpus.

    Build a corpus where the most-recent rows are role:assistant (and no-role)
    so they would consume a naive `ORDER BY created_at DESC LIMIT maxlen` budget
    BEFORE eligibility filtering — crowding older role:user turns out of the
    served set. With a small maxlen, the warm set must STILL equal the SQL
    authority's eligible ordered list (the warm SQL must push the eligibility
    predicate before the LIMIT).
    """
    _monkeypatch_env(monkeypatch, tmp_path)
    store_path = tmp_path / "asst-interleaved-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    # Shrink the buffer so the LIMIT actually bites against the corpus.
    store._recency_buffer._maxlen = 30

    base_ts = datetime(2026, 2, 1, tzinfo=timezone.utc)
    # Interleave: older user turns first, then a dense block of NEWER assistant
    # and no-role rows. A pre-filter LIMIT would fill with the newest assistant
    # rows and drop the older user turns; the authority over-fetches the role
    # branch and surfaces all user turns up to n.
    for i in range(40):
        store.insert(_make_role_user(f"user turn {i}", base_ts + timedelta(minutes=i), seed=100 + i))
    for i in range(60):
        ca = base_ts + timedelta(minutes=100 + i)  # strictly newer than all user turns
        store.insert(_make_role_assistant(f"assistant turn {i}", ca, seed=300 + i))
    for i in range(20):
        ca = base_ts + timedelta(minutes=200 + i)  # newest of all, no role
        store.insert(_make(text=f"no-role record {i}", vec=_random_vec(500 + i)))

    n = 25
    sql_golden = _sql_recent_pending_markers_direct(store, n)
    assert len(sql_golden) == n, (
        f"authority returned {len(sql_golden)} eligible markers; expected {n} "
        "(40 user turns exist, so n=25 must be fully served by the authority)"
    )

    store.warm_recency_buffer()
    buf_result = _sql_recent_pending_markers(store, n)

    assert buf_result == sql_golden, (
        "warm buffer under-populated vs SQL authority on an assistant-interleaved "
        "corpus — the warm LIMIT must apply AFTER the eligibility predicate\n"
        f"  sql_golden ids[:5]={[t[0] for t in sql_golden[:5]]}\n"
        f"  buf_result ids[:5]={[t[0] for t in buf_result[:5]]}\n"
        f"  len(sql)={len(sql_golden)} len(buf)={len(buf_result)}"
    )


# ---------------------------------------------------------------------------
# Test 13: concurrency — two cold reads never observe an empty/partial buffer
# ---------------------------------------------------------------------------

def test_recency_buffer_concurrent_cold_reads_no_empty(tmp_path, monkeypatch):
    """Concurrent cold recalls never return a truncated/empty buffer result.

    Many threads hit a freshly-cleared (cold) store at once. The single-flight
    warm lock plus atomic replace_all must guarantee every thread sees the full
    expected marker set — never an empty or half-filled buffer from a racing
    clear-then-fill.
    """
    _monkeypatch_env(monkeypatch, tmp_path)
    store_path = tmp_path / "concurrent-cold-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    _build_role_mixed_corpus(store, n_role_user=60, n_pending=10)
    n = 50
    expected_len = len(_sql_recent_pending_markers_direct(store, n))
    assert expected_len > 0

    # Force cold so every thread races the lazy warm.
    store._recency_buffer.clear()

    results: list[int] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(16)

    def _worker():
        try:
            barrier.wait()
            out = store.recent_pending_markers(n=n)
            results.append(len(out))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent cold reads raised: {errors[:3]}"
    assert results, "no worker produced a result"
    # Every thread must have seen the full set — never an empty/partial buffer.
    assert all(r == expected_len for r in results), (
        "a concurrent cold read observed a truncated/empty buffer "
        f"(expected every result == {expected_len}, got distinct lengths "
        f"{sorted(set(results))}) — the lazy-warm TOCTOU is not closed"
    )


# ---------------------------------------------------------------------------
# Test 14: tombstoned rows — buffer and SQL authority agree (both exclude them)
# ---------------------------------------------------------------------------

def test_recency_buffer_tombstone_consistent_with_sql(tmp_path, monkeypatch):
    """A soft-tombstoned role:user row is excluded by BOTH the buffer and SQL.

    Insert duplicate idem-tagged role:user records, run the dedupe migration
    (non-dry) to soft-tombstone the duplicate, then assert the buffer-served
    result equals the SQL authority — both must drop the tombstoned row.
    """
    from iai_mcp.migrate._dedupe import migrate_dedupe_episodic_captures

    _monkeypatch_env(monkeypatch, tmp_path)
    store_path = tmp_path / "tombstone-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    base_ts = datetime(2026, 4, 1, tzinfo=timezone.utc)
    # Some plain role:user turns.
    for i in range(5):
        store.insert(_make_role_user(f"plain user {i}", base_ts + timedelta(minutes=i), seed=100 + i))

    # Two idem-duplicate role:user records (same idem tag) → dedupe tombstones one.
    idem = "idem:dup-recency-tombstone"
    for i in range(2):
        rec = _make_role_user("duplicate user turn", base_ts + timedelta(minutes=10 + i), seed=700 + i)
        rec.tags = ["capture", "role:user", idem]
        store.insert(rec)

    # Warm the buffer so the duplicates are in it BEFORE tombstoning.
    store.warm_recency_buffer()

    res = migrate_dedupe_episodic_captures(store, dry_run=False)
    assert res["tombstoned"] >= 1, f"dedupe tombstoned {res['tombstoned']}; expected >= 1"

    n = 20
    sql_golden = _sql_recent_pending_markers_direct(store, n)
    buf_result = _sql_recent_pending_markers(store, n)

    assert buf_result == sql_golden, (
        "buffer diverged from SQL authority after a dedupe soft-tombstone — "
        "both paths must exclude tombstoned rows\n"
        f"  sql ids={[t[0] for t in sql_golden]}\n"
        f"  buf ids={[t[0] for t in buf_result]}"
    )


# ---------------------------------------------------------------------------
# Test 15: non-episodic role:user row — buffer and SQL authority agree (exclude)
# ---------------------------------------------------------------------------

def test_recency_buffer_non_episodic_role_user_excluded(tmp_path, monkeypatch):
    """A semantic (non-episodic) role:user row is excluded by BOTH paths.

    The SQL authority's role branch requires tier='episodic'. The buffer must
    apply the same tier predicate so a tier='semantic' role:user record does not
    appear in the buffer while being absent from SQL.
    """
    _monkeypatch_env(monkeypatch, tmp_path)
    store_path = tmp_path / "non-episodic-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))

    base_ts = datetime(2026, 5, 1, tzinfo=timezone.utc)
    for i in range(4):
        store.insert(_make_role_user(f"episodic user {i}", base_ts + timedelta(minutes=i), seed=100 + i))

    # A semantic role:user record — newest, so it would lead the buffer if the
    # tier predicate were missing.
    sem = MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface="semantic role user row",
        aaak_index="",
        embedding=_random_vec(900),
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
        created_at=base_ts + timedelta(minutes=100),
        updated_at=base_ts + timedelta(minutes=100),
        tags=["capture", "role:user"],
        language="en",
    )
    store.insert(sem)

    store.warm_recency_buffer()

    n = 20
    sql_golden = _sql_recent_pending_markers_direct(store, n)
    buf_result = _sql_recent_pending_markers(store, n)

    sql_ids = {t[0] for t in sql_golden}
    assert str(sem.id) not in sql_ids, "authority unexpectedly included a semantic role:user row"
    assert buf_result == sql_golden, (
        "buffer included a non-episodic role:user row the SQL authority excludes\n"
        f"  sql ids={[t[0] for t in sql_golden]}\n"
        f"  buf ids={[t[0] for t in buf_result]}"
    )


# ---------------------------------------------------------------------------
# Test 16: naive-datetime ordering normalizes to UTC (tz-independent)
# ---------------------------------------------------------------------------

def test_recency_buffer_naive_datetime_orders_as_utc(tmp_path, monkeypatch):
    """A naive created_at fed to the buffer orders as UTC, not host-local time.

    Two markers at the same instant — one naive, one tz-aware UTC — must sort as
    equal on the created_at dimension (the naive one is treated as UTC). Under a
    non-UTC host TZ a naive .timestamp() would offset them.
    """
    from iai_mcp.store._recency_buffer import RecencyBuffer, RecencyMarker

    monkeypatch.setenv("TZ", "America/New_York")
    try:
        import time as _time
        _time.tzset()
    except (AttributeError, OSError):
        pass

    buf = RecencyBuffer(maxlen=10)
    instant_naive = datetime(2026, 6, 1, 12, 0, 0)  # naive
    instant_aware = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    buf.push(RecencyMarker(
        id="naive", literal_surface="n", created_at=instant_naive,
        session_id=None, embedding_pending=0, role="user", source_rowid=2,
        tier="episodic",
    ))
    buf.push(RecencyMarker(
        id="aware", literal_surface="a", created_at=instant_aware,
        session_id=None, embedding_pending=0, role="user", source_rowid=1,
        tier="episodic",
    ))

    out = buf.markers(10)
    # Same instant (naive treated as UTC) → created_at epoch equal → tie broken
    # by rowid DESC: rowid 2 ('naive') before rowid 1 ('aware').
    assert [m.id for m in out] == ["naive", "aware"], (
        "naive created_at did not normalize to UTC for ordering — it was likely "
        f"interpreted in host-local time. got order={[m.id for m in out]}"
    )


# ---------------------------------------------------------------------------
# Test 17: replace_all is atomic — a reader interleaving mid-swap never sees
#          an empty/partial buffer (the structural guarantee behind WR-02)
# ---------------------------------------------------------------------------

def test_recency_buffer_replace_all_atomic_no_torn_read(tmp_path, monkeypatch):
    """A concurrent reader never observes an empty buffer during replace_all.

    The pre-fix warm did clear()-then-fill across SEPARATE lock acquisitions, so
    a reader could land in the gap and see an empty buffer. replace_all swaps the
    whole set under ONE lock hold. This test hammers a reader against a thread
    repeatedly calling replace_all and asserts the reader never sees an empty
    snapshot while the buffer is supposed to be full.
    """
    from iai_mcp.store._recency_buffer import RecencyBuffer, RecencyMarker

    buf = RecencyBuffer(maxlen=200)
    full_set = [
        RecencyMarker(
            id=f"id-{i}", literal_surface=f"s{i}",
            created_at=datetime(2026, 6, 1, tzinfo=timezone.utc) + timedelta(seconds=i),
            session_id=None, embedding_pending=0, role="user", source_rowid=i,
            tier="episodic",
        )
        for i in range(120)
    ]
    buf.replace_all(full_set)

    stop = threading.Event()
    saw_empty: list[int] = []

    def _swapper():
        while not stop.is_set():
            buf.replace_all(full_set)

    def _reader():
        for _ in range(20000):
            got = buf.markers(200)
            if len(got) == 0:
                saw_empty.append(1)
                break

    sw = threading.Thread(target=_swapper)
    rd = threading.Thread(target=_reader)
    sw.start()
    rd.start()
    rd.join()
    stop.set()
    sw.join()

    assert not saw_empty, (
        "reader observed an EMPTY buffer during a concurrent replace_all — "
        "the swap is not atomic (clear-then-fill window is open)"
    )
