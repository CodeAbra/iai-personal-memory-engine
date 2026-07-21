"""Concurrency and SQLite-parity tests for the vendored storage engine.

The engine's ``LilliBrainConnection`` advertises ``check_same_thread=False``
semantics, which under SQLite means the connection serializes all cross-thread
statement execution itself (SERIALIZED mode). These tests drive the raw engine
directly (no HippoDB, no encryption, no ANN) and assert:

- concurrent writers + readers on ONE shared connection never deadlock;
- a nested ``BEGIN`` raises the same ``sqlite3.OperationalError`` family as
  stdlib ``sqlite3`` (transaction-within-a-transaction parity);
- the serialization mutex is re-entrant (a held mutex + ``cursor.execute`` ->
  ``connection.execute`` on the same thread does not self-deadlock);
- ``executemany`` restores the pager transaction methods even when a row mid
  batch raises, leaving the connection usable.
"""
from __future__ import annotations

import sqlite3

from iai_mcp import errors as iai_errors
import threading
from pathlib import Path

import pytest

from iai_mcp.lillibrain.connection import (
    LilliBrainConnection,
    _noop,
    _open_lilli_db,
)


def _open_engine(tmp_path: Path) -> LilliBrainConnection:
    """Open a raw engine connection with a small standalone events-shaped table."""
    pager, catalog, root_map = _open_lilli_db(tmp_path / "engine.db")
    conn = LilliBrainConnection(pager, catalog, root_map)
    conn.execute("CREATE TABLE events (id TEXT, n INTEGER)")
    return conn


@pytest.mark.timeout(30)
def test_concurrent_writers_readers_no_deadlock(tmp_path: Path) -> None:
    """3 writers + 3 readers on ONE shared connection complete without deadlock.

    Each writer issues BEGIN/INSERT/COMMIT in a loop; a nested-BEGIN raise from
    an overlapping writer is caught and the writer retries (mirrors the live
    bypass pattern). Every thread must join within the timeout — a pre-fix
    deadlock in the pager-method swap fails this fast via the timeout marker.
    """
    conn = _open_engine(tmp_path)

    stop = threading.Event()
    errors: list[str] = []
    write_count = threading.Semaphore(0)

    def writer(worker: int) -> None:
        i = 0
        while not stop.is_set():
            try:
                conn.execute("BEGIN")
                conn.execute(
                    "INSERT INTO events (id, n) VALUES (?, ?)",
                    (f"w{worker}-{i}", i),
                )
                conn.execute("COMMIT")
                write_count.release()
                i += 1
            except iai_errors.OperationalError:
                # Overlapping writer hit a nested BEGIN — retry on the next loop.
                continue
            except Exception as exc:  # noqa: BLE001
                errors.append(f"writer {worker}: {type(exc).__name__}: {exc}")
                return

    def reader() -> None:
        for _ in range(200):
            if stop.is_set():
                break
            try:
                cur = conn.execute("SELECT COUNT(*) FROM events")
                row = cur.fetchone()
                if row is None:
                    errors.append("reader: COUNT(*) returned no row")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"reader: {type(exc).__name__}: {exc}")
                return

    writers = [threading.Thread(target=writer, args=(w,)) for w in range(3)]
    readers = [threading.Thread(target=reader) for _ in range(3)]
    for t in writers + readers:
        t.start()

    for t in readers:
        t.join(timeout=20)
    stop.set()
    for t in writers:
        t.join(timeout=10)

    alive = [t for t in writers + readers if t.is_alive()]
    assert not alive, f"{len(alive)} thread(s) did not join — deadlock"
    assert not errors, f"unexpected thread errors: {errors[:5]}"

    final = conn.execute("SELECT COUNT(*) FROM events").fetchone()
    assert final is not None
    assert int(final[0]) >= 0
    conn.close()


@pytest.mark.timeout(30)
def test_nested_begin_raises(tmp_path: Path) -> None:
    """A second BEGIN while in a transaction raises the SQLite OperationalError.

    Differential arm: stdlib ``sqlite3`` (manual-BEGIN / isolation_level=None)
    raises the same ``sqlite3.OperationalError`` with the identical text, so the
    engine matches the drop-in contract byte-for-byte.
    """
    conn = _open_engine(tmp_path)
    conn.execute("BEGIN")
    with pytest.raises(iai_errors.OperationalError) as engine_exc:
        conn.execute("BEGIN")
    assert "cannot start a transaction within a transaction" in str(engine_exc.value)
    conn.execute("ROLLBACK")
    conn.close()

    # Differential arm: the stdlib driver (through the translating wrapper)
    # raises the same family + message.
    from iai_mcp import _sqlite_stdlib
    std = _sqlite_stdlib.connect(":memory:", isolation_level=None)
    std.execute("BEGIN")
    with pytest.raises(iai_errors.OperationalError) as std_exc:
        std.execute("BEGIN")
    assert "cannot start a transaction within a transaction" in str(std_exc.value)
    assert type(engine_exc.value) is type(std_exc.value)
    std.close()


@pytest.mark.timeout(10)
def test_cursor_execute_reentrant_under_mutex(tmp_path: Path) -> None:
    """Holding the serialization mutex and driving a cursor does not self-deadlock.

    ``cursor.execute`` re-enters ``connection.execute`` on the same thread; the
    mutex must be re-entrant (RLock) so the held-mutex path returns cleanly.
    """
    conn = _open_engine(tmp_path)
    conn.execute("INSERT INTO events (id, n) VALUES (?, ?)", ("seed", 1))

    with conn._exec_lock:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM events")
        row = cursor.fetchone()
    assert row is not None
    assert int(row[0]) == 1
    conn.close()


@pytest.mark.timeout(10)
def test_executemany_restores_pager_on_error(tmp_path: Path) -> None:
    """A mid-batch executemany raise leaves the pager transaction methods restored.

    With no outer transaction the no-txn path swaps the pager's begin/commit/
    rollback methods to no-ops and back; a row whose param the encoder rejects
    must restore the real bound methods on the error path so the connection
    stays usable for a subsequent clean transaction.
    """
    conn = _open_engine(tmp_path)

    bad_rows = [("a", 1), ("b", object()), ("c", 3)]
    with pytest.raises(Exception):  # noqa: B017 — encoder rejects the bad param mid-batch
        conn.executemany("INSERT INTO events (id, n) VALUES (?, ?)", bad_rows)

    # The 4x pager-method swap must be restored — not left at _noop.
    assert conn._pager.begin_write is not _noop
    assert conn._pager.commit is not _noop
    assert conn._pager.rollback is not _noop
    assert conn._in_transaction is False

    # The connection is still usable: a clean transaction commits.
    conn.execute("BEGIN")
    conn.execute("INSERT INTO events (id, n) VALUES (?, ?)", ("ok", 42))
    conn.execute("COMMIT")
    row = conn.execute("SELECT COUNT(*) FROM events").fetchone()
    assert int(row[0]) == 1
    conn.close()


# ---------------------------------------------------------------------------
# HR-01 guard: autocommit DML failure must NOT brick the write path
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
def test_autocommit_dml_failure_does_not_brick_write_path(tmp_path: Path) -> None:
    """A mid-write executor failure on the autocommit path leaves the connection usable.

    Fault injection: patch pager.commit to raise after begin_write() has opened
    the journal, simulating an OS write error (EIO / disk-full at commit time).
    Without the HR-01 fix the journal is left open, _in_transaction stays False,
    and every subsequent write raises RuntimeError("A write transaction is already
    in progress").  With the fix the autocommit branch rolls the pager back before
    re-raising, so the next INSERT succeeds.
    """
    conn = _open_engine(tmp_path)
    pager = conn._pager

    # Save the real commit so we can restore it after the fault.
    _real_commit = pager.__class__.commit

    raised: list[Exception] = []

    def _failing_commit(self) -> None:
        raise OSError("simulated EIO at commit — disk full")

    try:
        # Inject the fault: commit will raise after begin_write opened the journal.
        pager.__class__.commit = _failing_commit  # type: ignore[method-assign]
        try:
            conn.execute("INSERT INTO events (id, n) VALUES (?, ?)", ("fault", 1))
        except OSError as exc:
            raised.append(exc)
        finally:
            pager.__class__.commit = _real_commit  # type: ignore[method-assign]

        # The original error must have propagated.
        assert len(raised) == 1, "expected the injected OSError to propagate"
        assert "simulated EIO" in str(raised[0]), f"wrong exception: {raised[0]}"

        # After the fix: pager journal is rolled back, write path is not bricked.
        # This INSERT must succeed (no RuntimeError about "transaction in progress").
        conn.execute("INSERT INTO events (id, n) VALUES (?, ?)", ("recovery", 99))

        # Confirm the recovery row is visible (the fault row must NOT be present).
        rows = conn.execute("SELECT id FROM events ORDER BY id").fetchall()
        ids = [r[0] for r in rows]
        assert "recovery" in ids, "recovery INSERT did not persist"
        assert "fault" not in ids, "faulted INSERT must not have persisted"
    finally:
        conn.close()


@pytest.mark.timeout(10)
def test_autocommit_dml_failure_pager_journal_closed(tmp_path: Path) -> None:
    """After an autocommit DML failure the pager journal is None (not left open).

    Directly inspects pager._journal and pager._wal to confirm the rollback path
    cleared the write state, complementing the behavioural test above.
    """
    conn = _open_engine(tmp_path)
    pager = conn._pager
    _real_commit = pager.__class__.commit

    def _failing_commit(self) -> None:
        raise RuntimeError("simulated pager commit barrier failure")

    try:
        pager.__class__.commit = _failing_commit  # type: ignore[method-assign]
        try:
            conn.execute("INSERT INTO events (id, n) VALUES (?, ?)", ("x", 0))
        except RuntimeError:
            pass
        finally:
            pager.__class__.commit = _real_commit  # type: ignore[method-assign]

        # HR-01 fix: journal must be None (rolled back), _in_transaction must be False.
        assert pager._journal is None, (
            "pager journal left open after autocommit DML failure — "
            "HR-01 rollback did not fire"
        )
        assert conn._in_transaction is False, (
            "_in_transaction inconsistently True after autocommit failure"
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# MR-01 guard: __exit__ must not close the pager (sqlite3 parity)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
def test_context_manager_leaves_connection_open(tmp_path: Path) -> None:
    """with conn: ends the transaction but leaves the connection usable.

    stdlib sqlite3.Connection.__exit__ commits (or rolls back on error) without
    closing the connection, so subsequent execute() calls work. This test
    verifies that LilliBrainConnection.__exit__ does NOT close the pager —
    the primary contract violation fixed by the MR-01 patch.
    """
    conn = _open_engine(tmp_path)
    try:
        # Successful with-block: commit path.
        with conn:
            conn.execute("INSERT INTO events (id, n) VALUES (?, ?)", ("a", 1))

        # Connection must still be alive — execute should not raise after a
        # successful with-block exits.
        row = conn.execute("SELECT COUNT(*) FROM events").fetchone()
        assert int(row[0]) == 1, "row inserted inside with-block not visible after exit"

        # Error path: exception exits the with-block; pager must still be open.
        try:
            with conn:
                raise RuntimeError("deliberate error inside with-block")
        except RuntimeError:
            pass

        # The pager must still be accessible after the error-path __exit__.
        row = conn.execute("SELECT COUNT(*) FROM events").fetchone()
        assert int(row[0]) == 1, "connection unusable after error-path __exit__"

        # One more clean insert and with-block to confirm full reuse.
        with conn:
            conn.execute("INSERT INTO events (id, n) VALUES (?, ?)", ("c", 3))
        row = conn.execute("SELECT COUNT(*) FROM events").fetchone()
        assert int(row[0]) == 2, "third insert via with-block not visible"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# LR-01 guard: close() must rollback (not commit) an open transaction
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
def test_close_with_open_txn_discards_uncommitted_rows(tmp_path: Path) -> None:
    """close() with an open transaction must discard (rollback) uncommitted writes.

    stdlib sqlite3.Connection.close() rolls back any uncommitted transaction on
    close — it is NOT a commit point. LilliBrainConnection must match this:
    closing a connection with an open BEGIN must discard the partial write, not
    persist it.

    Protocol:
    1. Open a connection, insert a committed baseline row.
    2. BEGIN a transaction and insert a second (uncommitted) row.
    3. Call close() without COMMIT.
    4. Reopen the same database file.
    5. Assert the committed baseline is visible; the uncommitted row is NOT.
    """
    db_file = tmp_path / "lr01.db"
    pager, catalog, root_map = _open_lilli_db(db_file)
    conn = LilliBrainConnection(pager, catalog, root_map)
    conn.execute("CREATE TABLE items (id TEXT, v INTEGER)")

    # Committed baseline.
    conn.execute("INSERT INTO items (id, v) VALUES (?, ?)", ("committed", 1))

    # Open a transaction, write a second row, then close WITHOUT committing.
    conn.execute("BEGIN")
    conn.execute("INSERT INTO items (id, v) VALUES (?, ?)", ("uncommitted", 2))
    assert conn._in_transaction is True, "sanity: BEGIN should set _in_transaction"
    conn.close()  # must rollback, not commit

    # Reopen the same file and check only the committed row survived.
    pager2, catalog2, root_map2 = _open_lilli_db(db_file)
    conn2 = LilliBrainConnection(pager2, catalog2, root_map2)
    try:
        rows = conn2.execute("SELECT id FROM items ORDER BY id").fetchall()
        ids = [r[0] for r in rows]
        assert "committed" in ids, "committed row must survive close"
        assert "uncommitted" not in ids, (
            "uncommitted row persisted after close() — close() committed instead of "
            "rolling back, breaking sqlite3 parity (LR-01)"
        )
    finally:
        conn2.close()
