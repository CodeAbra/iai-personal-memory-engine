"""On-disk page corruption must surface as a distinct, non-swallowable error.

The record store raises a bare ``sqlite3.DatabaseError`` for a checksum
mismatch, never the ``sqlite3.OperationalError`` that a routine
``except OperationalError`` around a query would swallow. Corrupting a stored
payload byte breaks the leaf page's CRC, and reading it back must fail loud with
the corruption-class exception so a caller can quarantine rather than silently
degrade.
"""

from __future__ import annotations

import gc
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
