"""On-disk page corruption must surface as a distinct, non-swallowable error.

The record store raises a bare ``sqlite3.DatabaseError`` for a checksum
mismatch, never the ``sqlite3.OperationalError`` that a routine
``except OperationalError`` around a query would swallow. Corrupting a stored
payload byte breaks the leaf page's CRC, and reading it back must fail loud with
the corruption-class exception so a caller can quarantine rather than silently
degrade.
"""

from __future__ import annotations

import errno
import gc
import logging
import os
import sqlite3

from iai_mcp import errors

import pytest

native = pytest.importorskip(
    "iai_mcp_native", reason="native extension not built"
)


def test_page_corruption_surfaces_as_databaseerror_not_operationalerror(tmp_path):
    store_cls = native.store.Store
    path = str(tmp_path / "brain.lilli")
    marker = b"hello_world_unique_payload_marker_xyz"

    store = store_cls.open(path)
    root = store.create_tree()
    store.insert(root, 1, marker)
    del store
    gc.collect()

    # The record layer stores the value verbatim, so the marker is findable on
    # disk. Flipping a byte inside it breaks the leaf page's checksum.
    with open(path, "rb") as fh:
        data = bytearray(fh.read())
    idx = data.find(marker)
    assert idx != -1, "payload must be stored verbatim at the record layer"
    data[idx + 5] ^= 0xFF
    with open(path, "wb") as fh:
        fh.write(data)

    reopened = store_cls.open(path)
    with pytest.raises(errors.DatabaseError) as exc_info:
        reopened.get(root, 1)

    # The load-bearing distinction: OperationalError is a DatabaseError subclass
    # that recall handlers routinely catch; corruption must NOT be one, so it
    # escapes that handler and reaches a corruption-aware path.
    assert not isinstance(exc_info.value, sqlite3.OperationalError), (
        "storage corruption must surface as a bare sqlite3.DatabaseError, not "
        "the swallowable sqlite3.OperationalError"
    )


# ---------------------------------------------------------------------------
# id-index build fails loud on an undecodable row (both drivers, parity)
# ---------------------------------------------------------------------------

_IDX_DDL = (
    "CREATE TABLE records ("
    "vec_label INTEGER PRIMARY KEY, id TEXT NOT NULL, tier TEXT, payload TEXT)"
)
_IDX_INSERT = (
    "INSERT INTO records (vec_label, id, tier, payload) VALUES (?, ?, ?, ?)"
)
# header_size=2 varint, then serial type 10 (reserved) — decode_record raises
# ValueError("unknown serial_type 10") on this payload.
_UNDECODABLE_PAYLOAD = b"\x02\x0a"


def _seed_python_records(db_path):
    from iai_mcp.lillibrain.connection import LilliBrainConnection, _open_lilli_db

    pager, catalog, root_map = _open_lilli_db(db_path)
    conn = LilliBrainConnection(pager, catalog, root_map)
    conn.execute(_IDX_DDL)
    for i in range(5):
        conn.execute(_IDX_INSERT, [i, f"rec-{i:03d}", "episodic", f"payload-{i}"])
    return conn


def _plant_undecodable_cell(conn):
    from iai_mcp.lillibrain.btree.cursor import BtreeCursor

    root_page = conn._root_map.get("records")
    conn._pager.begin_write()
    BtreeCursor(conn._pager, root_page).insert(999999, _UNDECODABLE_PAYLOAD)
    conn._pager.commit()
    # Force the next point-lookup to rebuild the id-index from the corrupt tree.
    conn._id_row_cache.pop("records", None)


def test_python_driver_id_index_fail_loud_on_undecodable_row(tmp_path):
    """The pure-Python reference driver no longer silently drops an undecodable
    id-index row: one bad row fails the whole build, so a point-lookup UPDATE on
    a valid id raises instead of silently returning an empty result."""
    # Control: a healthy id-index build lets the point-lookup UPDATE succeed.
    ctrl = _seed_python_records(tmp_path / "control.db")
    ctrl._id_row_cache.pop("records", None)
    ctrl.execute("UPDATE records SET tier = ? WHERE id = ?", ["x", "rec-002"])

    conn = _seed_python_records(tmp_path / "corrupt.db")
    _plant_undecodable_cell(conn)
    with pytest.raises(ValueError, match="serial_type"):
        conn.execute("UPDATE records SET tier = ? WHERE id = ?", ["x", "rec-002"])


def test_native_and_python_id_index_fail_loud_parity(tmp_path):
    """Both drivers fail loud on a corrupt id-index row: the Python driver on a
    planted undecodable payload, the native engine on byte-flipped bytes — so a
    differential test can never mistake a silent Python drop for engine parity."""
    from iai_mcp_native import engine

    # Pure-Python reference driver: planted undecodable id-index row raises.
    conn = _seed_python_records(tmp_path / "py.db")
    _plant_undecodable_cell(conn)
    with pytest.raises(Exception):
        conn.execute("UPDATE records SET tier = ? WHERE id = ?", ["x", "rec-002"])

    # Native engine: a byte-corrupted id-index row raises the same fail-loud way.
    path = str(tmp_path / "native.db")
    ndb = engine.Connection.open(path, 384)
    ndb.execute(_IDX_DDL, [])
    ndb.execute("BEGIN", [])
    marker = "UNIQUE_PAYLOAD_MARKER_ZZZ"
    for i in range(5):
        payload = marker if i == 3 else f"payload-{i}"
        ndb.execute(_IDX_INSERT, [i, f"rec-{i:03d}", "episodic", payload])
    ndb.execute("COMMIT", [])
    del ndb
    gc.collect()

    with open(path, "rb") as fh:
        data = bytearray(fh.read())
    off = data.find(marker.encode())
    assert off != -1, "payload marker must be stored verbatim on disk"
    data[off + 5] ^= 0xFF
    with open(path, "wb") as fh:
        fh.write(data)

    reopened = engine.Connection.open(path, 384)
    with pytest.raises(errors.DatabaseError):
        reopened.execute("UPDATE records SET tier = ? WHERE id = ?", ["x", "rec-002"])


# ---------------------------------------------------------------------------
# WAL recovery open-failure becomes observable
# ---------------------------------------------------------------------------


def test_wal_recovery_open_fail_is_logged(tmp_path, caplog, monkeypatch):
    """When an existing WAL sidecar cannot be opened during recovery the call
    returns without crashing, but the un-replayed committed frames are logged."""
    from iai_mcp.lillibrain import wal as wal_mod

    db_path = tmp_path / "main.db"
    db_path.write_bytes(b"")
    wal_path = tmp_path / "main.db-wal"
    wal_path.write_bytes(b"\x00" * 64)  # sidecar exists so os.open is reached

    real_open = os.open

    def _selective_open(path, *args, **kwargs):
        if str(wal_path) in str(path):
            raise OSError(errno.EACCES, "forced sidecar open failure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(wal_mod.os, "open", _selective_open)
    caplog.set_level(logging.WARNING, logger="iai_mcp.lillibrain.wal")
    wal_mod.recover_wal_on_open(str(db_path), str(wal_path), 4096)  # must not raise

    hits = [
        r
        for r in caplog.records
        if r.name == "iai_mcp.lillibrain.wal"
        and r.levelno == logging.WARNING
        and "WAL recovery could not open sidecar" in r.getMessage()
        and str(wal_path) in r.getMessage()
    ]
    assert hits, "a failed WAL sidecar open must be logged"


# ---------------------------------------------------------------------------
# Unreadable rollback journal — observable, with an ENOENT split
# ---------------------------------------------------------------------------


def test_rollback_journal_unreadable_is_logged(tmp_path, caplog, monkeypatch):
    """A present-but-unreadable rollback journal is logged (the interrupted
    transaction is NOT rolled back) and left in place; a spurious ENOENT takes
    the silent treat-as-absent branch."""
    import pathlib

    from iai_mcp.lillibrain import pager as pager_mod
    from iai_mcp.lillibrain.connection import _open_lilli_db

    pgr, _catalog, _root_map = _open_lilli_db(tmp_path / "engine.db")
    journal_path = pgr._journal_path()
    journal_path.write_bytes(b"\x00" * 64)  # present journal

    # Force the rollback-journal branch: report a non-WAL write-version byte.
    real_pread = os.pread

    def _pread(fd, size, offset):
        if size == 1 and offset == pager_mod.HDR_WRITE_VERSION_OFFSET:
            return b"\x01"  # rollback-journal mode
        return real_pread(fd, size, offset)

    monkeypatch.setattr(pager_mod.os, "pread", _pread)

    real_read_bytes = pathlib.Path.read_bytes

    def _read_bytes_eacces(self):
        if str(self) == str(journal_path):
            raise OSError(errno.EACCES, "forced unreadable journal")
        return real_read_bytes(self)

    monkeypatch.setattr(pathlib.Path, "read_bytes", _read_bytes_eacces)
    caplog.set_level(logging.WARNING, logger="iai_mcp.lillibrain.pager")
    pgr._recover_if_needed()  # must not raise

    hits = [
        r
        for r in caplog.records
        if r.name == "iai_mcp.lillibrain.pager"
        and r.levelno == logging.WARNING
        and "rollback journal unreadable" in r.getMessage()
        and str(journal_path) in r.getMessage()
    ]
    assert hits, "a present-but-unreadable rollback journal must be logged"
    assert journal_path.exists(), "the journal must not be unlinked on a read error"

    # A spurious ENOENT (vanished between exists() and read) is silent.
    caplog.clear()

    def _read_bytes_enoent(self):
        if str(self) == str(journal_path):
            raise OSError(errno.ENOENT, "vanished between exists() and read")
        return real_read_bytes(self)

    monkeypatch.setattr(pathlib.Path, "read_bytes", _read_bytes_enoent)
    pgr._recover_if_needed()

    enoent_warnings = [
        r
        for r in caplog.records
        if r.name == "iai_mcp.lillibrain.pager"
        and r.levelno == logging.WARNING
        and "rollback journal unreadable" in r.getMessage()
    ]
    assert not enoent_warnings, "ENOENT must take the silent treat-as-absent branch"
