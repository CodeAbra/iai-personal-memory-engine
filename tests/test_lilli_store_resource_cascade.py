"""Regression: the shared engine connection borrowed at a second open of the
same store path must outlive the owner's close().

A second HippoDB open at a path already in the connection registry borrows the
owner's engine connection. The advisory writer lock and the main-db / WAL file
handles live once, behind that shared connection. Tearing them out the moment
the owner closes breaks every live borrower, cursor, and raw adapter.

Each test forces its reference window explicitly (it holds the borrower / leaked
raw open / file-less connection itself) so the assertion is deterministic and
does not depend on garbage-collection timing.
"""

from __future__ import annotations

import gc

import pytest


def _require_lilli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-pass-cascade")
    monkeypatch.setenv("LILLI_FSYNC_MODE", "fast")


def _import_hippo():
    """Import HippoDB after the driver env is patched; skip if engine absent."""
    try:
        from iai_mcp.hippo import HippoDB  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"native lilli engine unavailable: {exc!r}")
    return HippoDB


def test_borrower_can_write_after_owner_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A borrower that shares the owner's connection can still write after the
    owner closes — the shared pager must outlive the owner."""
    _require_lilli(monkeypatch)
    HippoDB = _import_hippo()

    d = str(tmp_path)
    owner = HippoDB(d)
    owner._conn.execute("CREATE TABLE IF NOT EXISTS t (k INTEGER PRIMARY KEY)")
    owner._conn.commit()

    borrower = HippoDB(d)  # registry-hit borrow of the shared connection
    owner.close()          # must not tear the shared file out from the borrower
    try:
        borrower._conn.execute("INSERT INTO t (k) VALUES (1)")
        borrower._conn.commit()
    finally:
        try:
            borrower.close()
        except Exception:  # noqa: BLE001
            pass


def test_closed_writer_raw_open_releases_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A leaked writer-mode raw open must not keep the exclusive lock from a
    later store open at the same path — reopening must succeed."""
    _require_lilli(monkeypatch)
    HippoDB = _import_hippo()
    from iai_mcp.lillibrain.connection import get_lilli_raw_conn  # noqa: PLC0415

    d = str(tmp_path)
    db = HippoDB(d)
    path = db._db_path
    db.close()  # registry now clear; an on-disk lilli store remains

    # Writer-mode raw open takes the exclusive advisory lock; leak it (never
    # close) to mimic a fixture or exception path that skipped the release.
    raw = get_lilli_raw_conn(path, read_only=False, allow_writer=True)
    try:
        db2 = HippoDB(d)  # must not collide with the leaked writer lock
        db2.close()
    finally:
        try:
            raw.close()
        except Exception:  # noqa: BLE001
            pass


def test_closed_registered_conn_not_handed_out_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A registered connection whose engine handle was already closed must not
    be borrowed stale by a later open — the open must succeed."""
    _require_lilli(monkeypatch)
    HippoDB = _import_hippo()
    from iai_mcp.lillibrain.connection import _LILLI_CONN_REGISTRY  # noqa: PLC0415

    d = str(tmp_path)
    owner = HippoDB(d)
    path = owner._db_path
    # db._conn is the mutation-signalling seam; the registry stores the RAW
    # engine connection it drives.
    engine_conn = getattr(owner._conn, "_wrapped", owner._conn)

    # Close the engine handle directly while leaving the registry entry in
    # place (no owner.close() deregister), forcing the file-less-borrow window.
    engine_conn.close()
    assert _LILLI_CONN_REGISTRY.get(path) is engine_conn

    db2 = HippoDB(d)  # must not borrow the file-less connection
    db2.close()


def test_borrowed_writer_raw_open_release_is_last_holder_aware(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A writer raw open that is borrowed by a later open keeps the backing store
    (file/lock) alive on the borrow's own hold after the original closes — the
    file/lock lifecycle rides the engine refcount, not the registry slot.

    Sequence: register a writer raw open (registry miss), borrow it via a second
    raw open at the same path (a distinct shared handle, still live), then close
    the original raw open. The original's close surrenders its own hold; the live
    borrow keeps the store open, so the borrow can still write. Once the borrow
    closes too — it is the last holder — the file/lock is released deterministically
    and a fresh open at the same path takes the exclusive lock without a collision.
    """
    _require_lilli(monkeypatch)
    HippoDB = _import_hippo()
    from iai_mcp.lillibrain.connection import (  # noqa: PLC0415
        _LILLI_CONN_REGISTRY,
        get_lilli_raw_conn,
    )

    d = str(tmp_path)
    db = HippoDB(d)
    path = db._db_path
    db.close()  # registry clear; an on-disk lilli store remains

    raw = get_lilli_raw_conn(path, read_only=False, allow_writer=True)  # registry miss → writer + lock
    borrow = get_lilli_raw_conn(path, read_only=False, allow_writer=True)  # registry hit → shared handle
    raw.close()  # owner closes; the live borrow keeps the store/lock alive

    # The closed owner is unborrowable, so its registry slot is dropped. The
    # backing store is kept alive by the live borrow's own engine hold, not by
    # the registry — the file/lock lifecycle rides the engine refcount.
    assert _LILLI_CONN_REGISTRY.get(path) is None
    # The borrow can still operate while it holds the store.
    borrow.execute("CREATE TABLE IF NOT EXISTS t (k INTEGER PRIMARY KEY)")
    borrow.commit()
    borrow.close()  # last live holder → file/lock released deterministically here

    # The lock must be free immediately at the last holder's close — no GC needed.
    db2 = HippoDB(d)
    db2.close()

    # Dropping the leaked adapter objects is a no-op safety net (already released).
    del raw, borrow
    gc.collect()


def test_last_reference_release_frees_lock_and_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Positive last-reference invariant: after the owner and a borrower both
    close, the advisory lock and the registry slot are actually released — a fresh
    engine open at the path succeeds (it takes the lock) and the registry is empty.
    """
    _require_lilli(monkeypatch)
    HippoDB = _import_hippo()
    from iai_mcp.lillibrain.connection import _LILLI_CONN_REGISTRY  # noqa: PLC0415

    d = str(tmp_path)
    owner = HippoDB(d)
    path = owner._db_path
    borrower = HippoDB(d)  # registry-hit borrow (distinct shared handle)

    owner.close()
    borrower.close()

    assert _LILLI_CONN_REGISTRY.get(path) is None, (
        "registry slot must be empty after the last holder closes"
    )

    # Drop the HippoDB objects so the lingering engine Connection handles release
    # via the Drop backstop, then assert the advisory lock is actually free.
    del owner, borrower
    gc.collect()

    # The advisory lock must be free: a fresh engine open takes it without a
    # try_lock collision (this is a registry miss → fresh exclusive-lock open).
    try:
        from iai_mcp_native import engine as _engine  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"native engine unavailable: {exc!r}")

    fresh = _engine.Connection.open(path, 0)
    fresh.close()


def _require_engine():
    try:
        from iai_mcp_native import engine as _engine  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"native engine unavailable: {exc!r}")
    return _engine


def test_owner_close_with_live_cursor_releases_lock_at_last_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The owner's close() must surrender its own hold on the backing store so
    the file/advisory lock is released DETERMINISTICALLY at the true last close
    — never deferred to garbage collection.

    A cursor kept live across the owner's close() keeps the shared backing store
    above one holder, so the owner's close() correctly defers the file/lock
    release. But once that last live holder (the cursor) closes, the lock MUST be
    free immediately: a fresh open at the same path succeeds without deleting or
    GC-collecting the owner Python object. Holding the owner reference is the
    point — if close() does not drop the owner's own arc, the owner pins the
    store past its own close() and the fresh open collides with the still-held
    lock.
    """
    _require_lilli(monkeypatch)
    _engine = _require_engine()

    path = str(tmp_path / "store")
    owner = _engine.Connection.open(path, 0)
    owner.execute("CREATE TABLE IF NOT EXISTS t (k INTEGER PRIMARY KEY)")
    owner.commit()

    # A live cursor is a distinct holder of the shared backing store.
    cur = owner.cursor()
    owner.close()  # cursor still live → file/lock release correctly deferred

    # Closing the last live holder must release the lock now — without dropping
    # or GC-collecting `owner`, which is still referenced here.
    cur.close()

    assert owner is not None  # keep the owner reference live (no GC release)
    fresh = _engine.Connection.open(path, 0)  # must take the lock cleanly
    fresh.close()


def test_owner_close_with_live_borrow_releases_lock_at_last_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Same invariant for a share()/borrow holder: the owner's close() drops its
    own hold, so once the borrow closes the lock is free without GC of the owner.
    """
    _require_lilli(monkeypatch)
    _engine = _require_engine()

    path = str(tmp_path / "store")
    owner = _engine.Connection.open(path, 0)
    owner.execute("CREATE TABLE IF NOT EXISTS t (k INTEGER PRIMARY KEY)")
    owner.commit()

    borrow = owner.share()  # a distinct shared holder of the backing store
    owner.close()           # borrow live → deferred release

    borrow.execute("INSERT INTO t (k) VALUES (1)")
    borrow.commit()
    borrow.close()          # last live holder → release now

    assert owner is not None  # owner kept alive; release must not depend on GC
    fresh = _engine.Connection.open(path, 0)
    fresh.close()


def test_engine_close_is_idempotent_and_raises_after_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A second close() on the engine Connection is a no-op, and any op after
    close raises the closed-connection error (stdlib sqlite3 closed-conn shape).
    """
    _require_lilli(monkeypatch)
    _engine = _require_engine()

    path = str(tmp_path / "store")
    conn = _engine.Connection.open(path, 0)
    conn.execute("CREATE TABLE IF NOT EXISTS t (k INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    assert conn.is_closed is True
    conn.close()  # idempotent: second close is a no-op, no raise

    with pytest.raises(Exception):
        conn.execute("SELECT 1")
