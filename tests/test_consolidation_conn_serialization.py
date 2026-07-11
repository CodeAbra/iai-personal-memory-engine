from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from iai_mcp.hippo import AccessMode, HippoIntegrityError, _txn, _txn_owners, _txn_owners_lock
from iai_mcp.maintenance import optimize_hippo_storage
from iai_mcp.sleep import _decay_edges
from iai_mcp.store import EDGES_TABLE, MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord

_FORBIDDEN_SUBSTRINGS = (
    "cannot start a transaction within a transaction",
    "cannot commit - no transaction is active",
    "bad parameter or other api misuse",
    "database table is locked",
    "cannot vacuum",
    "sql statements in progress",
)


def _make_record(seed: int) -> MemoryRecord:
    rng = np.random.RandomState(seed)
    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=f"record seed {seed}",
        aaak_index="",
        embedding=rng.randn(EMBED_DIM).tolist(),
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        language="en",
    )


def _assert_no_forbidden(messages: list[str]) -> None:
    for msg in messages:
        low = msg.lower()
        for sig in _FORBIDDEN_SUBSTRINGS:
            assert sig not in low, (
                f"shared-connection consolidation race regressed: "
                f"forbidden signature {sig!r} in {msg!r}"
            )


def test_edge_writer_concurrent_with_vacuum_no_connection_corruption(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path, user_id="test")
    try:
        ids: list[uuid.UUID] = []
        for i in range(40):
            rec = _make_record(1000 + i)
            store.insert(rec)
            ids.append(rec.id)
        flush_record_buffer(store)

        errors: list[str] = []
        stop = threading.Event()

        def writer() -> None:
            n = 0
            while not stop.is_set():
                try:
                    a = ids[n % len(ids)]
                    b = ids[(n + 1) % len(ids)]
                    store.boost_edges([(a, b)], delta=0.1, edge_type="hebbian")
                    n += 1
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"writer: {type(exc).__name__}: {exc}")

        def vacuumer() -> None:
            for _ in range(15):
                if stop.is_set():
                    break
                report = optimize_hippo_storage(store)
                for table in ("records", "edges", "events"):
                    err = report.get(table, {}).get("error")
                    if err:
                        errors.append(f"vacuum[{table}]: {err}")
                time.sleep(0.005)

        ta = threading.Thread(target=writer, daemon=True)
        tb = threading.Thread(target=vacuumer, daemon=True)
        ta.start()
        tb.start()
        tb.join(timeout=30)
        stop.set()
        ta.join(timeout=5)

        _assert_no_forbidden(errors)
        assert not errors, f"unexpected errors under concurrent writer+VACUUM: {errors}"
    finally:
        store.close()


def test_escalate_sets_intent_flag_then_vacuum_clean(tmp_path: Path) -> None:
    intent_path = tmp_path / "hippo" / ".consolidation-pending"

    store = MemoryStore(tmp_path, user_id="test", access_mode=AccessMode.SHARED)
    try:
        for i in range(20):
            store.insert(_make_record(2000 + i))
        flush_record_buffer(store)

        assert not intent_path.exists(), "intent flag should be clear before escalate"

        store.db.escalate_to_exclusive()
        assert intent_path.exists(), (
            "escalate_to_exclusive must SET the consolidation-intent flag "
            "before the VACUUM window so SHARED clients back off"
        )
        assert store.db._access_mode is AccessMode.EXCLUSIVE

        report = optimize_hippo_storage(store)
        for table in ("records", "edges", "events"):
            assert report.get(table, {}).get("error") is None, (
                f"VACUUM under escalation must not fail for {table}: "
                f"{report.get(table, {}).get('error')}"
            )

        store.db.downgrade_to_shared()
        assert not intent_path.exists(), (
            "downgrade_to_shared must CLEAR the intent flag so clients proceed"
        )
        assert store.db._access_mode is AccessMode.SHARED
    finally:
        store.close()


def test_escalate_sets_intent_when_already_exclusive(tmp_path: Path) -> None:
    intent_path = tmp_path / "hippo" / ".consolidation-pending"

    store = MemoryStore(tmp_path, user_id="test", access_mode=AccessMode.EXCLUSIVE)
    try:
        for i in range(10):
            store.insert(_make_record(3000 + i))
        flush_record_buffer(store)

        assert store.db._access_mode is AccessMode.EXCLUSIVE
        assert not intent_path.exists()

        store.db.escalate_to_exclusive()
        assert intent_path.exists(), (
            "escalate_to_exclusive must set the consolidation-intent flag even "
            "when already EXCLUSIVE (cross-process signal independent of flock)"
        )

        report = optimize_hippo_storage(store)
        for table in ("records", "edges", "events"):
            assert report.get(table, {}).get("error") is None, (
                f"VACUUM under already-EXCLUSIVE escalation must not fail for "
                f"{table}: {report.get(table, {}).get('error')}"
            )

        store.db.downgrade_to_shared()
        assert not intent_path.exists()
        assert store.db._access_mode is AccessMode.SHARED
    finally:
        store.close()


def test_legacy_edge_decay_endpoint_concurrent_with_vacuum_no_corruption(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path, user_id="test")
    try:
        ids: list[uuid.UUID] = []
        for i in range(40):
            rec = _make_record(4000 + i)
            store.insert(rec)
            ids.append(rec.id)
        flush_record_buffer(store)
        for i in range(len(ids) - 1):
            store.boost_edges([(ids[i], ids[i + 1])], delta=0.3, edge_type="hebbian")

        errors: list[str] = []
        stop = threading.Event()

        def legacy_writer() -> None:
            tbl = store.db.open_table(EDGES_TABLE)
            n = 0
            while not stop.is_set():
                try:
                    a = str(ids[n % len(ids)])
                    b = str(ids[(n + 1) % len(ids)])
                    now = datetime.now(timezone.utc)
                    tbl.update(
                        where=f"src = '{a}' AND dst = '{b}' AND edge_type = 'hebbian'",
                        values={"weight": 0.25, "updated_at": now},
                    )
                    tbl.delete(
                        f"src = '{a}' AND dst = '{b}' AND edge_type = 'hebbian'"
                    )
                    store.boost_edges(
                        [(ids[n % len(ids)], ids[(n + 1) % len(ids)])],
                        delta=0.3,
                        edge_type="hebbian",
                    )
                    n += 1
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"legacy_writer: {type(exc).__name__}: {exc}")

        def canonical_thread() -> None:
            n = 0
            while not stop.is_set():
                try:
                    report = optimize_hippo_storage(store)
                    for table in ("records", "edges", "events"):
                        err = report.get(table, {}).get("error")
                        if err:
                            errors.append(f"vacuum[{table}]: {err}")
                    a = ids[n % len(ids)]
                    b = ids[(n + 1) % len(ids)]
                    store.boost_edges([(a, b)], delta=0.1, edge_type="hebbian")
                    n += 1
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"canonical_thread: {type(exc).__name__}: {exc}")
                time.sleep(0.005)

        ta = threading.Thread(target=legacy_writer, daemon=True)
        tb = threading.Thread(target=canonical_thread, daemon=True)
        ta.start()
        tb.start()
        tb.join(timeout=15)
        stop.set()
        ta.join(timeout=5)
        tb.join(timeout=5)

        _assert_no_forbidden(errors)
        assert not errors, (
            f"unexpected errors under concurrent M1/M2+VACUUM: {errors}"
        )

        df = store.db.open_table(EDGES_TABLE).to_pandas()
        assert df["weight"].notna().all(), "corrupt NULL weight after race"
        assert (df["weight"] >= 0).all(), "negative weight after race"
    finally:
        store.close()


def _insert_stale_edge(
    store: MemoryStore,
    src: uuid.UUID,
    dst: uuid.UUID,
    weight: float,
    days_old: int = 120,
    edge_type: str = "hebbian",
) -> None:
    old_ts = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    tbl = store.db.open_table(EDGES_TABLE)
    import pyarrow as pa
    rows = pa.Table.from_pylist(
        [{"src": str(src), "dst": str(dst), "edge_type": edge_type,
          "weight": weight, "updated_at": old_ts}],
        schema=pa.schema([
            ("src", pa.string()), ("dst", pa.string()),
            ("edge_type", pa.string()), ("weight", pa.float32()),
            ("updated_at", pa.string()),
        ]),
    )
    (
        tbl.merge_insert(["src", "dst", "edge_type"])
        .when_matched_update_all()
        .execute(rows)
    )


def test_real_decay_edges_concurrent_with_vacuum_no_corruption(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path, user_id="test")
    try:
        ids: list[uuid.UUID] = []
        for i in range(40):
            rec = _make_record(5000 + i)
            store.insert(rec)
            ids.append(rec.id)
        flush_record_buffer(store)

        for i in range(0, len(ids) - 1, 2):
            _insert_stale_edge(store, ids[i], ids[i + 1], weight=0.5, days_old=120)
        for i in range(1, len(ids) - 1, 2):
            _insert_stale_edge(store, ids[i], ids[i + 1], weight=0.02, days_old=120)

        errors: list[str] = []
        stop = threading.Event()
        decay_results: list[dict] = []

        def legacy_decay_thread() -> None:
            n = 0
            while not stop.is_set():
                try:
                    result = _decay_edges(store)
                    decay_results.append(result)
                    pair_i = n % (len(ids) - 1)
                    _insert_stale_edge(
                        store, ids[pair_i], ids[pair_i + 1],
                        weight=0.5, days_old=120,
                    )
                    _insert_stale_edge(
                        store, ids[(pair_i + 1) % len(ids)],
                        ids[(pair_i + 2) % len(ids)],
                        weight=0.02, days_old=120,
                    )
                    n += 1
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"decay_thread: {type(exc).__name__}: {exc}")

        def canonical_thread() -> None:
            n = 0
            while not stop.is_set():
                try:
                    report = optimize_hippo_storage(store)
                    for table in ("records", "edges", "events"):
                        err = report.get(table, {}).get("error")
                        if err:
                            errors.append(f"vacuum[{table}]: {err}")
                    a = ids[n % len(ids)]
                    b = ids[(n + 1) % len(ids)]
                    store.boost_edges([(a, b)], delta=0.1, edge_type="hebbian")
                    n += 1
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"canonical_thread: {type(exc).__name__}: {exc}")
                time.sleep(0.01)

        ta = threading.Thread(target=legacy_decay_thread, daemon=True)
        tb = threading.Thread(target=canonical_thread, daemon=True)
        ta.start()
        tb.start()
        tb.join(timeout=15)
        stop.set()
        ta.join(timeout=5)
        tb.join(timeout=5)

        _assert_no_forbidden(errors)
        assert not errors, (
            f"unexpected errors under concurrent _decay_edges+VACUUM: {errors}"
        )

        total_decayed = sum(r.get("decayed", 0) for r in decay_results)
        total_pruned = sum(r.get("pruned", 0) for r in decay_results)
        assert total_decayed > 0, (
            "_decay_edges never exercised M1 (update) — "
            "edges may not have been stale enough or weight class misconfigured"
        )
        assert total_pruned > 0, (
            "_decay_edges never exercised M2 (delete/prune) — "
            "edges may not have fallen below DECAY_EPSILON"
        )

        df = store.db.open_table(EDGES_TABLE).to_pandas()
        if not df.empty:
            assert df["weight"].notna().all(), "corrupt NULL weight after race"
            assert (df["weight"] >= 0).all(), "negative weight after race"
    finally:
        store.close()


def test_txn_same_thread_reentry_does_not_raise(tmp_path: Path) -> None:
    import sqlite3 as _sqlite3
    db_path = tmp_path / "test_reentry.db"
    conn = _sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
    try:
        conn.execute("CREATE TABLE t (x INTEGER)")
        with _txn(conn):
            with _txn(conn):
                conn.execute("INSERT INTO t VALUES (1)")
        row = conn.execute("SELECT x FROM t").fetchone()
        assert row is not None and row[0] == 1, "nested _txn did not commit"
    finally:
        conn.close()


def test_txn_no_owner_caller_managed_outer_txn_does_not_raise(tmp_path: Path) -> None:
    import sqlite3 as _sqlite3
    db_path = tmp_path / "test_no_owner.db"
    conn = _sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
    try:
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.execute("BEGIN")
        with _txn(conn):
            conn.execute("INSERT INTO t VALUES (99)")
        conn.execute("COMMIT")
        row = conn.execute("SELECT x FROM t").fetchone()
        assert row is not None and row[0] == 99
        with _txn_owners_lock:
            assert id(conn) not in _txn_owners, "stale owner entry after no-owner txn"
    finally:
        conn.close()


def test_txn_foreign_owner_raises_hippo_integrity_error(tmp_path: Path) -> None:
    import sqlite3 as _sqlite3
    db_path = tmp_path / "test_foreign_owner.db"
    conn = _sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
    import threading as _threading

    barrier_txn_open = _threading.Event()
    barrier_release_a = _threading.Event()
    raised: list[Exception] = []
    a_errors: list[str] = []

    def thread_a() -> None:
        import contextlib
        try:
            with contextlib.ExitStack() as stack:
                stack.enter_context(_txn(conn))
                barrier_txn_open.set()
                barrier_release_a.wait(timeout=5)
        except Exception as exc:  # noqa: BLE001
            a_errors.append(f"thread_a unexpected error: {exc}")

    def thread_b() -> None:
        barrier_txn_open.wait(timeout=5)
        try:
            with _txn(conn):
                pass
            raised.append(None)
        except HippoIntegrityError as exc:
            raised.append(exc)
        except Exception as exc:  # noqa: BLE001
            raised.append(exc)
        finally:
            barrier_release_a.set()

    ta = _threading.Thread(target=thread_a, daemon=True)
    tb = _threading.Thread(target=thread_b, daemon=True)
    ta.start()
    tb.start()
    ta.join(timeout=10)
    tb.join(timeout=10)

    assert not a_errors, f"thread_a failed unexpectedly: {a_errors}"
    assert len(raised) == 1, "thread_b did not complete"
    assert isinstance(raised[0], HippoIntegrityError), (
        f"expected HippoIntegrityError from foreign-owner thread, "
        f"got: {type(raised[0]).__name__}: {raised[0]}"
    )
    with _txn_owners_lock:
        assert id(conn) not in _txn_owners, (
            "owner-map entry NOT cleared after A's _txn exit — "
            "stale entry would false-positive the next legitimate caller"
        )
    conn.close()


# ---------------------------------------------------------------------------
# MR-01 guard: _ensure_tables must route through _txn/_conn_lock uniformly
# ---------------------------------------------------------------------------


def test_ensure_tables_does_not_leave_stale_txn_owner(tmp_path: Path) -> None:
    """HippoDB boot must not leave a stale _txn_owners entry after _ensure_tables.

    Before the MR-01 fix _ensure_tables hand-rolled BEGIN/ROLLBACK/COMMIT outside
    _txn(), so it never registered a _txn_owners entry. With the fix it uses _txn()
    which registers and then removes an entry. After boot the map must be clean —
    a stale entry would cause every subsequent _txn() caller on the same conn to
    see 'txn in progress' and silently yield into a foreign transaction.
    """
    store = MemoryStore(tmp_path, user_id="test")
    try:
        conn = store.db._conn
        with _txn_owners_lock:
            assert id(conn) not in _txn_owners, (
                "_txn_owners still has an entry for the boot connection after "
                "HippoDB.__init__ — _ensure_tables leaked a stale owner (MR-01)"
            )
    finally:
        store.close()


def test_ensure_tables_boot_conn_accepts_txn_after_init(tmp_path: Path) -> None:
    """After boot the shared connection accepts a fresh _txn with no leftover state.

    Proves the _txn() path is taken (owner registered then cleared), not the old
    hand-rolled path (which never touched _txn_owners). A fresh _txn on the boot
    conn must complete without error — no stale in_transaction or journal from
    _ensure_tables.
    """
    store = MemoryStore(tmp_path, user_id="test")
    try:
        conn = store.db._conn
        # A no-op _txn under _conn_lock must complete cleanly after boot.
        completed = []
        with store.db._conn_lock, _txn(conn):
            completed.append(True)
        assert completed, "_txn on the boot connection failed after _ensure_tables"
        with _txn_owners_lock:
            assert id(conn) not in _txn_owners, (
                "owner entry not cleared after post-boot _txn exit"
            )
    finally:
        store.close()
