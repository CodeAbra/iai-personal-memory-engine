"""Regression for the O(tree-height) max-key descent used by INSERT key derivation.

``BtreeCursor.max_key()`` must return the largest key in the tree (or ``None``
when empty) by descending the rightmost spine, never by materialising every row.
The INSERT path derives the next AUTOINCREMENT/rowid key from it, so it must
agree exactly with the old ``max(k for k, _ in full_scan())`` across an empty
tree, a single-leaf tree, a multi-level tree (post leaf/interior split), and
after deletes (keys are reusable, matching plain-rowid semantics).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from iai_mcp.lillibrain.btree.cursor import BtreeCursor
from iai_mcp.lillibrain.pager import Pager


@pytest.fixture()
def lilli_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")


def _fresh_cursor(db_path: Path) -> tuple[Pager, BtreeCursor]:
    pager = Pager(db_path)
    return pager, BtreeCursor(pager, pager.root_page_no)


def _payload(key: int) -> bytes:
    return f"row-{key}".encode().ljust(64, b"\x00")


def _scan_max(cursor: BtreeCursor) -> int | None:
    pairs = cursor.full_scan()
    return max((k for k, _ in pairs), default=None)


def test_max_key_empty_tree_is_none(lilli_driver: None, tmp_path: Path) -> None:
    pager, cursor = _fresh_cursor(tmp_path / "empty.db")
    try:
        assert cursor.max_key() is None
        assert _scan_max(cursor) is None
    finally:
        pager.close()


def test_max_key_single_leaf(lilli_driver: None, tmp_path: Path) -> None:
    pager, cursor = _fresh_cursor(tmp_path / "single.db")
    try:
        pager.begin_write()
        for k in (7, 2, 11, 5):
            cursor.insert(k, _payload(k))
        pager.commit()
        assert cursor.max_key() == 11
        assert cursor.max_key() == _scan_max(cursor)
    finally:
        pager.close()


def test_max_key_multilevel_tree_after_splits(
    lilli_driver: None, tmp_path: Path
) -> None:
    # Large payloads force leaf splits and eventually an interior split, so the
    # rightmost-child descent is exercised across more than one tree level.
    pager, cursor = _fresh_cursor(tmp_path / "multi.db")
    try:
        big = b"x" * 3000
        pager.begin_write()
        for k in range(1, 300):
            cursor.insert(k, big)
        pager.commit()
        assert cursor.max_key() == 299
        assert cursor.max_key() == _scan_max(cursor)
        # Inserting out of order must not change the maximum.
        pager.begin_write()
        cursor.insert(150, big)
        pager.commit()
        assert cursor.max_key() == 299
    finally:
        pager.close()


def test_max_key_after_delete_of_current_max(
    lilli_driver: None, tmp_path: Path
) -> None:
    pager, cursor = _fresh_cursor(tmp_path / "del.db")
    try:
        pager.begin_write()
        for k in range(1, 50):
            cursor.insert(k, _payload(k))
        pager.commit()
        assert cursor.max_key() == 49
        pager.begin_write()
        cursor.delete(49)
        cursor.delete(48)
        pager.commit()
        # The maximum drops to the new rightmost cell — keys are reusable, which
        # is exactly the plain-rowid next-key behaviour the INSERT path relies on.
        assert cursor.max_key() == 47
        assert cursor.max_key() == _scan_max(cursor)
    finally:
        pager.close()
