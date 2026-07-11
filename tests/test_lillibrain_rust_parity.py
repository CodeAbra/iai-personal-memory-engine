"""Storage parity: the native record store vs the in-tree Python reference.

Drives identical operation sequences through the native `iai_mcp_native.store`
surface and the Python `lillibrain` reference engine, asserting byte-and-order
identical results for CRUD, ordered scans, the verbatim blob corpus, and
crash-and-reopen recovery.

The Python reference is the parity oracle and stays untouched. Runs hermetic in
a temp dir with the fast fsync escape so the property loops do not block on
per-commit drive flushes.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

os.environ.setdefault("LILLI_FSYNC_MODE", "fast")


def _native_available() -> bool:
    try:
        from iai_mcp_native import store  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _native_available(), reason="iai_mcp_native store submodule not installed"
)


# --- reference oracle driver ------------------------------------------------
#
# The Python reference uses one tree per file rooted at a fixed page. Both
# engines expose the same key->value record semantics; only the root page
# number differs (a structural detail), so the differential compares the
# (key, value) content, never the page geometry.


class _Reference:
    """A single-tree key->value store backed by the Python `lillibrain` engine."""

    def __init__(self, path: str) -> None:
        from iai_mcp.lillibrain.btree.cursor import BtreeCursor
        from iai_mcp.lillibrain.pager import Pager

        self._Pager = Pager
        self._Cursor = BtreeCursor
        self._path = path
        self._open()

    def _open(self) -> None:
        self._pager = self._Pager(self._path)
        self._cursor = self._Cursor(self._pager, self._pager.root_page_no)

    def insert(self, key: int, value: bytes) -> None:
        self._pager.begin_write()
        self._cursor.insert(key, value)
        self._pager.commit()

    def delete(self, key: int) -> None:
        self._pager.begin_write()
        self._cursor.delete(key)
        self._pager.commit()

    def get(self, key: int) -> bytes | None:
        for k, v in self._cursor.full_scan():
            if k == key:
                return v
        return None

    def max_key(self) -> int | None:
        return self._cursor.max_key()

    def full_scan(self) -> list[tuple[int, bytes]]:
        return self._cursor.full_scan()

    def range_scan(self, lo: int, hi: int) -> list[tuple[int, bytes]]:
        return [(k, v) for k, v in self._cursor.full_scan() if lo <= k <= hi]

    def close(self) -> None:
        self._pager.close()

    def reopen(self) -> None:
        self._pager.close()
        self._open()


class _Native:
    """The same key->value store, backed by the native `iai_mcp_native.store`."""

    def __init__(self, path: str) -> None:
        from iai_mcp_native import store

        self._store_mod = store
        self._path = path
        self._open()
        self._root = self._store.create_tree()

    def _open(self) -> None:
        self._store = self._store_mod.Store.open(self._path)

    def insert(self, key: int, value: bytes) -> None:
        self._store.begin_write()
        self._store.insert(self._root, key, value)
        self._store.commit()

    def delete(self, key: int) -> None:
        self._store.begin_write()
        self._store.delete(self._root, key)
        self._store.commit()

    def get(self, key: int) -> bytes | None:
        return self._store.get(self._root, key)

    def max_key(self) -> int | None:
        return self._store.max_key(self._root)

    def full_scan(self) -> list[tuple[int, bytes]]:
        return [(k, v) for k, v in self._store.full_scan(self._root)]

    def range_scan(self, lo: int, hi: int) -> list[tuple[int, bytes]]:
        return [(k, v) for k, v in self._store.range_scan(self._root, lo, hi)]

    def integrity(self) -> list[str]:
        return list(self._store.check_integrity(self._root))

    def reopen(self) -> None:
        # Drop the handle so its single-writer lock and file are released, then
        # reopen against the same path. The root page number is stable.
        root = self._root
        del self._store
        self._open()
        self._root = root


def _assert_state_equal(ref: _Reference, nat: _Native) -> None:
    assert ref.full_scan() == nat.full_scan()
    assert ref.max_key() == nat.max_key()


# --- tests ------------------------------------------------------------------


def test_crud_parity(tmp_path: Path) -> None:
    """Identical insert/get/delete/re-insert sequences agree on scans + max."""
    ref = _Reference(str(tmp_path / "ref.lilli"))
    nat = _Native(str(tmp_path / "nat.lilli"))

    rng = random.Random(20260626)
    keys = list(range(1, 201))
    rng.shuffle(keys)

    # Step 1: insert in shuffled key order.
    for k in keys:
        payload = f"value-for-{k}".encode()
        ref.insert(k, payload)
        nat.insert(k, payload)
    _assert_state_equal(ref, nat)

    # Point lookups agree, present and absent.
    for k in [1, 50, 137, 200]:
        assert ref.get(k) == nat.get(k)
    for absent in [0, 201, 999]:
        assert ref.get(absent) == nat.get(absent) is None

    # Range scans agree across several windows.
    for lo, hi in [(10, 20), (1, 200), (150, 160), (300, 400)]:
        assert ref.range_scan(lo, hi) == nat.range_scan(lo, hi)

    # Step 2: delete a third of the keys.
    for k in keys[::3]:
        ref.delete(k)
        nat.delete(k)
    _assert_state_equal(ref, nat)

    # Step 3: re-insert the deleted keys with new payloads (insert-or-replace).
    for k in keys[::3]:
        payload = f"reinserted-{k}".encode()
        ref.insert(k, payload)
        nat.insert(k, payload)
    _assert_state_equal(ref, nat)

    # Replace-in-place an existing key; both engines see the new value.
    ref.insert(50, b"replaced")
    nat.insert(50, b"replaced")
    assert ref.get(50) == nat.get(50) == b"replaced"
    _assert_state_equal(ref, nat)

    assert nat.integrity() == []


def test_verbatim_corpus(tmp_path: Path) -> None:
    """Adversarial blobs round-trip byte-identical through the native store."""
    corpus = [
        b"",  # empty
        b"\x00",  # lone NUL
        b"a\x00b\x00c",  # embedded NULs
        "emoji \U0001f9e0\U0001f600 mix".encode(),  # 4-byte emoji
        "你好世界".encode(),  # CJK
        "שלום عالم".encode(),  # RTL Hebrew/Arabic
        b"trailing whitespace   \t\n   ",  # significant trailing whitespace
        b"   leading whitespace",
        bytes(range(256)),  # every byte value
        b"\xff" * 100,  # high bytes
        b"normal ascii payload",
    ]

    ref = _Reference(str(tmp_path / "ref.lilli"))
    nat = _Native(str(tmp_path / "nat.lilli"))

    for i, blob in enumerate(corpus):
        key = i + 1
        ref.insert(key, blob)
        nat.insert(key, blob)

    for i, blob in enumerate(corpus):
        key = i + 1
        ref_val = ref.get(key)
        nat_val = nat.get(key)
        assert nat_val == blob, f"native mutated blob at key {key}"
        assert ref_val == nat_val, f"native vs reference diverged at key {key}"

    assert ref.full_scan() == nat.full_scan()
    assert nat.integrity() == []


def test_crash_recovery_reopen(tmp_path: Path) -> None:
    """Committed writes survive a reopen; integrity stays clean."""
    ref = _Reference(str(tmp_path / "ref.lilli"))
    nat = _Native(str(tmp_path / "nat.lilli"))

    payloads = {1: b"alpha", 2: b"beta\x00bin", 3: b"gamma", 4: b"delta"}
    for k, v in payloads.items():
        ref.insert(k, v)
        nat.insert(k, v)

    before_ref = ref.full_scan()
    before_nat = nat.full_scan()
    assert before_ref == before_nat

    # Reopen both stores on the same path.
    ref.reopen()
    nat.reopen()

    assert ref.full_scan() == before_ref, "reference lost committed data on reopen"
    assert nat.full_scan() == before_nat, "native lost committed data on reopen"
    assert ref.full_scan() == nat.full_scan()
    assert nat.integrity() == []

    # The reopened native store keeps serving point lookups verbatim.
    for k, v in payloads.items():
        assert nat.get(k) == v


def test_uncommitted_write_absent_on_reopen(tmp_path: Path) -> None:
    """A write transaction left uncommitted does not survive a reopen."""
    from iai_mcp_native import store

    path = str(tmp_path / "nat.lilli")
    st = store.Store.open(path)
    root = st.create_tree()

    st.begin_write()
    st.insert(root, 1, b"committed")
    st.commit()

    # Open a transaction and write, but drop the handle without committing.
    st.begin_write()
    st.insert(root, 2, b"pending")
    del st

    reopened = store.Store.open(path)
    survivors = [(k, v) for k, v in reopened.full_scan(root)]
    assert survivors == [(1, b"committed")]
    assert list(reopened.check_integrity(root)) == []
