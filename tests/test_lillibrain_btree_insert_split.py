"""Engine-level regression for the insert-path leaf-split page overflow.

Drives the pager + BtreeCursor + check_integrity directly with opaque byte
payloads (no embedder, no crypto, no daemon) so the cases are deterministic
and fast. Three complementary cases reproduce the same defect class — a leaf
split that lands cells so a resulting page exceeds one page:

  1. Count-midpoint co-location: a leaf carrying many small cells plus two
     large cells whose count-midpoint split co-locates both large cells on one
     half, overflowing it.
  2. The unsplittable case: a legal leaf [5198B, 2067B] plus a 7828B insert
     yields the sorted set [5198, 7828, 2067] for which NO two-way split keeps
     both halves within one page — three pages are required.
  3. Parent cascade: enough large inserts that leaf splits cascade and the
     parent interior node itself overflows, forcing an interior split / new
     root.

Each asserts the durable post-fix contract: no error, ascending keys across
ALL leaves, correct total count, byte-exact payload reassembly (the verbatim /
literal-surface invariant), and a clean structural integrity check. They are
NOT wrapped in ``pytest.raises`` — the split-path overflow makes them RED today
(struct.error / RuntimeError on the overflowing split) and they flip GREEN once
the split distributes cells into as many pages as needed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from iai_mcp.lillibrain import constants as constants_module
from iai_mcp.lillibrain.btree import cursor as cursor_module
from iai_mcp.lillibrain.btree.cursor import BtreeCursor
from iai_mcp.lillibrain.btree.delete import (
    _balanced_nway_leaf_split,
    _merged_leaf_footprint,
)
from iai_mcp.lillibrain.check_integrity import check_integrity
from iai_mcp.lillibrain.constants import (
    PAGE_INTERIOR_TABLE,
    PAGE_LEAF_TABLE,
    PAGE_SIZE,
)
from iai_mcp.lillibrain.pager import Pager


@pytest.fixture()
def lilli_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select the in-tree pager/B+tree/WAL engine for the duration of a test."""
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")


def _fresh_tree(db_path: Path) -> tuple[Pager, int]:
    """Open a hermetic single-tree pager whose root leaf is the integrity root.

    A fresh Pager initialises page 1 (DB header) and page 2 (an empty B+tree
    root leaf) automatically; ``check_integrity`` walks the tree from
    ``pager.root_page_no`` (page 2), so the cursor must drive that same root for
    the structural oracle to cover the cells under test. Returns (pager, root).
    """
    pager = Pager(db_path)
    return pager, pager.root_page_no


def _payload(n_bytes: int, fill: int) -> bytes:
    """Deterministic opaque cell payload of exactly n_bytes (no embedder/RNG)."""
    return bytes([fill & 0xFF]) * n_bytes


def test_nway_partition_unsplittable_set_makes_three_groups() -> None:
    """The two-way-unsplittable set [5198, 7828, 2067] partitions into 3 groups."""
    cells = [(10, _payload(5198, 0xAA)), (15, _payload(7828, 0xCC)), (20, _payload(2067, 0xBB))]
    groups = _balanced_nway_leaf_split(cells)

    assert len(groups) == 3, f"expected 3 groups, got {len(groups)}"
    assert [k for grp in groups for k, _ in grp] == [10, 15, 20], "key order not preserved"
    for grp in groups:
        assert grp, "no group may be empty"
        assert _merged_leaf_footprint(grp) <= PAGE_SIZE, "group overflows one page"


def test_nway_partition_uniform_large_each_group_fits() -> None:
    """A run of uniform large cells partitions into page-fitting non-empty groups."""
    cells = [(k, _payload(1656, k)) for k in range(1, 41)]  # 40 cells, ~4 per page
    groups = _balanced_nway_leaf_split(cells)

    assert len(groups) >= 2, "a 40-cell large set must span multiple groups"
    flat = [k for grp in groups for k, _ in grp]
    assert flat == list(range(1, 41)), "key order not preserved across groups"
    for grp in groups:
        assert grp, "no group may be empty"
        assert _merged_leaf_footprint(grp) <= PAGE_SIZE, "group overflows one page"


def test_nway_partition_preserves_every_cell_once() -> None:
    """Every input cell appears exactly once, in order, across all groups."""
    cells = [(k, _payload(3000 + (k * 311) % 4000, k)) for k in range(1, 31)]
    groups = _balanced_nway_leaf_split(cells)

    rebuilt = [c for grp in groups for c in grp]
    assert rebuilt == cells, "cells must be conserved in order with no loss/dup"
    for grp in groups:
        assert grp and _merged_leaf_footprint(grp) <= PAGE_SIZE


def test_count_midpoint_colocation_splits_without_overflow(
    lilli_driver: None, tmp_path: Path
) -> None:
    """A split whose count-midpoint co-locates two large cells overflows a half.

    Seed a leaf with many small (200-byte) cells interleaved with two large
    (4000-byte) cells, then insert one more cell that triggers a split. A
    count-midpoint split that places both 4000-byte cells on the same half
    overflows that page (two 4000-byte cells plus pointers exceed 8192). The
    durable fix must instead distribute the cells by bytes so every resulting
    page fits one page. Assert no error, ascending keys across all leaves,
    correct total count, byte-exact payloads, and clean integrity.
    """
    pager, root_page_no = _fresh_tree(tmp_path / "colocation.db")
    cursor = BtreeCursor(pager, root_page_no)

    # Two large cells sit at the high end of the key space (keys 1000, 1001) so
    # the count-midpoint of the merged set lands between the small cells and the
    # large pair, co-locating both 4000-byte cells on the right half.
    small_keys = list(range(16, 30))  # fourteen 200-byte cells
    large_keys = [1000, 1001]         # two 4000-byte cells
    inserted: dict[int, bytes] = {}

    pager.begin_write()
    for k in small_keys:
        v = _payload(200, k)
        cursor.insert(k, v)
        inserted[k] = v
    for k in large_keys:
        v = _payload(4000, k & 0xFF)
        cursor.insert(k, v)
        inserted[k] = v
    # One more small cell tips the leaf over its byte budget and triggers the
    # split whose count-midpoint co-locates the two large cells.
    extra_key = 30
    extra_val = _payload(200, extra_key)
    cursor.insert(extra_key, extra_val)
    inserted[extra_key] = extra_val
    pager.commit()

    survivors = cursor.full_scan()
    keys = [k for k, _ in survivors]
    expected = sorted(inserted)

    assert keys == expected, f"keys not ascending/complete: {keys} != {expected}"
    assert len(survivors) == len(expected), (
        f"row count {len(survivors)} != expected {len(expected)}"
    )
    for k, payload in survivors:
        assert payload == inserted[k], f"payload corrupted for key {k}"
    check_integrity(pager, raise_on_error=True)


def test_no_two_way_split_unsplittable_case(
    lilli_driver: None, tmp_path: Path
) -> None:
    """A legal leaf plus one insert yields a set no two-way split can place.

    A leaf holding a 5198-byte cell and a 2067-byte cell is legal (both fit one
    page together: 12 + 4 + 5203 + 2071 = 7290 <= 8192). Inserting a 7828-byte
    cell at key 15 produces the sorted set [5198, 7828, 2067] for which NO
    two-way cut keeps both halves within one page:
      [5198] | [7828, 2067]  -> right = 12 + 4 + 7831 + 2070 = 9917 > 8192
      [5198, 7828] | [2067]  -> left  = 12 + 4 + 5201 + 7831 = 13048 > 8192
    The durable fix must produce three leaves. This is the case that proves an
    N-way split (not a write-ceiling change) is required.
    """
    pager, root_page_no = _fresh_tree(tmp_path / "unsplittable.db")
    cursor = BtreeCursor(pager, root_page_no)

    cells = {10: _payload(5198, 0xAA), 20: _payload(2067, 0xBB), 15: _payload(7828, 0xCC)}

    pager.begin_write()
    cursor.insert(10, cells[10])
    cursor.insert(20, cells[20])
    cursor.insert(15, cells[15])  # third insert: no two-way split fits
    pager.commit()

    survivors = cursor.full_scan()
    keys = [k for k, _ in survivors]

    assert keys == [10, 15, 20], f"keys not ascending/complete: {keys} != [10, 15, 20]"
    assert len(survivors) == 3, f"row count {len(survivors)} != 3"
    for k, payload in survivors:
        assert payload == cells[k], f"payload corrupted for key {k}"
    check_integrity(pager, raise_on_error=True)


def test_parent_cascade_interior_split_new_root(
    lilli_driver: None, tmp_path: Path
) -> None:
    """Many large inserts cascade leaf splits until the parent interior overflows.

    Inserting a long run of large (4000-7000 byte) cells forces repeated leaf
    splits; each split copies a separator up to the parent interior node. With
    enough cells the parent interior node overflows its own byte budget and must
    split, eventually allocating a new root level. The fix must run that cascade
    without error and keep every invariant: ascending keys across all leaves,
    correct count, byte-exact payloads, clean integrity (which also verifies all
    leaves are at equal depth and the sibling chain matches the tree walk).
    """
    pager, root_page_no = _fresh_tree(tmp_path / "cascade.db")
    cursor = BtreeCursor(pager, root_page_no)

    n_rows = 400
    inserted: dict[int, bytes] = {}

    pager.begin_write()
    for k in range(1, n_rows + 1):
        # Vary the payload size across 4000-7000 bytes so leaves pack unevenly,
        # exercising the byte-balanced split under realistic mixed geometry.
        size = 4000 + (k * 137) % 3001
        v = _payload(size, k)
        cursor.insert(k, v)
        inserted[k] = v
    pager.commit()

    survivors = cursor.full_scan()
    keys = [k for k, _ in survivors]
    expected = list(range(1, n_rows + 1))

    assert keys == expected, "keys not ascending/complete across the cascade"
    assert len(survivors) == n_rows, f"row count {len(survivors)} != {n_rows}"
    for k, payload in survivors:
        assert payload == inserted[k], f"payload corrupted for key {k}"
    check_integrity(pager, raise_on_error=True)


def _root_page_type(pager: Pager) -> int:
    """Return the page-type byte (leaf vs interior) of the fixed root page."""
    return pager.read_page(pager.root_page_no)[0]


def test_count_capped_split_on_byte_fitting_leaf_no_degenerate_root(
    lilli_driver: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A small cell-count cap must not turn a byte-fitting leaf into a bad root.

    The split dispatch is gated by a secondary cell-count ceiling, but the
    splitter partitions by bytes. When the count ceiling is injected small, a
    leaf that still fits one page by bytes is dispatched to the split path, and
    the byte partitioner returns a SINGLE group with zero separators. The split
    must treat a one-group partition as "no split needed" — rewrite the leaf in
    place — rather than build a degenerate single-child, zero-key root.

    Before the guard this produced a root interior node with ``keys=[]`` and one
    child (an integrity-clean but structurally degenerate node that chains an
    unbounded stack of single-child levels on every further capped insert). The
    assertions below fail against that degenerate root and pass once a one-group
    partition is a no-op.
    """
    # Inject a small count ceiling so the dispatch gate fires on byte-fitting
    # leaves. The leaf gate reads the cursor-module binding; the splitter reads
    # the constants-module binding — patch both so gate and splitter agree.
    monkeypatch.setattr(cursor_module, "LEAF_MAX_CELLS", 4)
    monkeypatch.setattr(constants_module, "LEAF_MAX_CELLS", 4)

    pager, root_page_no = _fresh_tree(tmp_path / "count_cap_byte_fit.db")
    cursor = BtreeCursor(pager, root_page_no)

    # Five tiny cells: all five fit one page by bytes (the byte partition yields a
    # SINGLE group), but the fifth insert trips the count ceiling (5 >= 4) and is
    # dispatched to the split path — the exact gate-vs-splitter mismatch trigger
    # (one group, zero separators).
    inserted: dict[int, bytes] = {}
    pager.begin_write()
    for k in range(1, 6):
        v = _payload(8, k)
        cursor.insert(k, v)
        inserted[k] = v
    pager.commit()

    # No degenerate single-child root: the root must stay a leaf (the no-op
    # rewrite keeps the single leaf as root), never a one-child zero-key interior.
    root_type = _root_page_type(pager)
    if root_type == PAGE_INTERIOR_TABLE:
        from iai_mcp.lillibrain.btree.page import read_interior_node

        keys, children = read_interior_node(memoryview(pager.read_page(root_page_no)))
        assert len(children) >= 2, (
            f"degenerate root: interior root with {len(children)} child(ren) "
            f"and keys={keys} (a one-group split must not build a root)"
        )
    else:
        assert root_type == PAGE_LEAF_TABLE, (
            f"unexpected root page type {root_type:#x}"
        )

    survivors = cursor.full_scan()
    keys = [k for k, _ in survivors]
    expected = sorted(inserted)
    assert keys == expected, f"keys not ascending/complete: {keys} != {expected}"
    assert len(survivors) == len(expected), (
        f"row count {len(survivors)} != expected {len(expected)}"
    )
    for k, payload in survivors:
        assert payload == inserted[k], f"payload corrupted for key {k}"
    check_integrity(pager, raise_on_error=True)


def test_count_cap_genuine_overflow_still_splits_nway(
    lilli_driver: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuinely byte-overflowing leaf still splits N>=2 under a small cap.

    The single-group no-op guard must not suppress a real split. With the same
    small count ceiling, insert large cells that genuinely overflow one page —
    the byte partition must still yield N>=2 groups, real separators, and a
    well-formed multi-child root, exactly as in production.
    """
    monkeypatch.setattr(cursor_module, "LEAF_MAX_CELLS", 4)
    monkeypatch.setattr(constants_module, "LEAF_MAX_CELLS", 4)

    pager, root_page_no = _fresh_tree(tmp_path / "count_cap_real_overflow.db")
    cursor = BtreeCursor(pager, root_page_no)

    # Five 3000-byte cells cannot share one page (three already exceed 8192), so
    # the byte partition yields multiple groups — a genuine split, not a no-op.
    inserted: dict[int, bytes] = {}
    pager.begin_write()
    for k in range(1, 6):
        v = _payload(3000, k)
        cursor.insert(k, v)
        inserted[k] = v
    pager.commit()

    root_type = _root_page_type(pager)
    assert root_type == PAGE_INTERIOR_TABLE, (
        "a genuinely overflowing leaf must promote to a multi-child interior root"
    )
    from iai_mcp.lillibrain.btree.page import read_interior_node

    root_keys, root_children = read_interior_node(
        memoryview(pager.read_page(root_page_no))
    )
    assert len(root_children) >= 2 and len(root_keys) >= 1, (
        f"real overflow must yield >=2 children / >=1 separator, "
        f"got children={len(root_children)} keys={root_keys}"
    )

    survivors = cursor.full_scan()
    keys = [k for k, _ in survivors]
    expected = sorted(inserted)
    assert keys == expected, f"keys not ascending/complete: {keys} != {expected}"
    assert len(survivors) == len(expected)
    for k, payload in survivors:
        assert payload == inserted[k], f"payload corrupted for key {k}"
    check_integrity(pager, raise_on_error=True)
