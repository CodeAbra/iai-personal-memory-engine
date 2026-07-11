"""Engine-level regression for the delete-path leaf-merge page overflow.

Drives the pager + BtreeCursor + check_integrity directly with opaque byte
payloads (no embedder, no crypto, no daemon) so the cases are deterministic
and fast. Two complementary cases reproduce the same defect:

  1. A small-cap deterministic case: a tiny injected cell cap forces leaves to
     hold only a few cells so deletes force a merge of two multi-cell siblings
     whose combined cells exceed one page.
  2. A large-payload production-geometry case: 1656-byte inline cells (the
     records-table shape: a 1536-byte embedding-sized BLOB plus small columns)
     so a leaf holds ~4 cells and a forced merge of two ~half-full siblings
     overflows a single 8192-byte page.

Both assert the durable post-fix contract: no error, surviving keys in ascending
order, correct remaining count, and a clean structural integrity check. They are
NOT wrapped in ``pytest.raises`` — a missing capacity guard makes them RED today
(struct.error on the overflowing merge) and they flip GREEN once the merge path
redistributes instead of blind-concatenating.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from iai_mcp.lillibrain.btree.cursor import BtreeCursor
from iai_mcp.lillibrain.btree.delete import (
    _byte_midpoint_split,
    _merged_leaf_footprint,
)
from iai_mcp.lillibrain.check_integrity import check_integrity
from iai_mcp.lillibrain.constants import PAGE_SIZE
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


def test_small_payload_merge_redistributes_without_overflow(
    lilli_driver: None, tmp_path: Path
) -> None:
    """A deterministic merge of two multi-cell siblings overflows one page on a
    tree seeded with small (~700-byte) cells; the delete must redistribute.

    This case proves the defect reproduces WITHOUT 1.5 KB payloads. Insert splits
    by bytes, so ~700-byte cells pack ~11 per leaf; a multi-leaf tree of 20 such
    cells, deleted down by a contiguous run, drives ``_rebalance_leaf`` to merge
    two byte-full siblings whose combined cells exceed PAGE_SIZE — the same
    blind-concat overflow as the production-geometry case, but small and fast.
    """
    pager, root_page_no = _fresh_tree(tmp_path / "small_payload.db")
    cursor = BtreeCursor(pager, root_page_no)

    cell_size = 700
    seeded = list(range(1, 21))  # 20 keys spanning several leaves
    pager.begin_write()
    for k in seeded:
        cursor.insert(k, _payload(cell_size, k))
    pager.commit()

    deleted = [2, 4, 6, 8, 10, 12, 14]
    pager.begin_write()
    for k in deleted:
        cursor.delete(k)
    pager.commit()

    survivors = cursor.full_scan()
    keys = [k for k, _ in survivors]
    expected = [k for k in seeded if k not in deleted]

    assert keys == expected, f"keys not ascending/correct: {keys} != {expected}"
    assert len(survivors) == len(expected), (
        f"row count {len(survivors)} != expected {len(expected)}"
    )
    for k, payload in survivors:
        assert payload == _payload(cell_size, k), f"payload corrupted for key {k}"
    check_integrity(pager, raise_on_error=True)


def test_large_payload_merge_redistributes_without_overflow(
    lilli_driver: None, tmp_path: Path
) -> None:
    """Production records geometry: 1656-byte inline cells, seed 10, delete 5.

    Each cell stays INLINE (1656 < OVERFLOW_THRESHOLD 8157), so a leaf holds
    ~4 cells (12 + 4*2 + 4*1656 = 6644 <= 8192; a 5th would overflow). The
    default cap (9999) is in effect, so this exercises the byte-driven path:
    a forced merge of two ~half-full siblings exceeds one page (7 cells = 11592
    bytes > 8192) and must redistribute rather than blind-concatenate.
    """
    pager, root_page_no = _fresh_tree(tmp_path / "records_geometry.db")
    cursor = BtreeCursor(pager, root_page_no)

    cell_size = 1656
    seeded = list(range(1, 11))  # 10 keys
    pager.begin_write()
    for k in seeded:
        cursor.insert(k, _payload(cell_size, k))
    pager.commit()

    deleted = [3, 5, 7, 9, 1]
    pager.begin_write()
    for k in deleted:
        cursor.delete(k)
    pager.commit()

    survivors = cursor.full_scan()
    keys = [k for k, _ in survivors]
    expected = [k for k in seeded if k not in deleted]

    assert keys == expected, f"keys not ascending/correct: {keys} != {expected}"
    assert len(survivors) == 5, f"remaining count {len(survivors)} != 5"
    # Payload bytes survive verbatim through any redistribute/re-spill.
    for k, payload in survivors:
        assert payload == _payload(cell_size, k), f"payload corrupted for key {k}"
    check_integrity(pager, raise_on_error=True)


def test_unequal_cell_split_keeps_both_halves_within_page() -> None:
    """The exact geometry a greedy first-crossing split overflows on.

    A merged set of one large cell and two medium cells
    (``[7000, 5000, 3000]``) has no balanced cut at the first boundary past
    half-total: a greedy split would put ``[7000, 5000]`` on the left (~12 KB,
    over one 8192-byte page), which then raises ``struct.error`` at write time.
    The balanced split must instead cut at ``[7000] | [5000, 3000]`` so BOTH
    halves stay within one page. Ascending key order is preserved across the
    boundary (left holds the lower keys).
    """
    cells = [
        (1, _payload(7000, 1)),
        (2, _payload(5000, 2)),
        (3, _payload(3000, 3)),
    ]
    left, right = _byte_midpoint_split(cells)

    assert [k for k, _ in left] + [k for k, _ in right] == [1, 2, 3], (
        "split must preserve ascending key order across the boundary"
    )
    assert left and right, "both halves must be non-empty"
    assert _merged_leaf_footprint(left) <= PAGE_SIZE, (
        f"left half overflows page: {_merged_leaf_footprint(left)} > {PAGE_SIZE}"
    )
    assert _merged_leaf_footprint(right) <= PAGE_SIZE, (
        f"right half overflows page: {_merged_leaf_footprint(right)} > {PAGE_SIZE}"
    )


def test_byte_midpoint_split_rejects_under_two_cells() -> None:
    """A 0/1-cell split is meaningless and must fail loud, not silently."""
    with pytest.raises(ValueError, match="at least 2 cells"):
        _byte_midpoint_split([])
    with pytest.raises(ValueError, match="at least 2 cells"):
        _byte_midpoint_split([(1, _payload(100, 1))])


def test_byte_midpoint_split_both_halves_fit_over_unequal_sizes() -> None:
    """Property sweep: across many unequal two-page cell sets the balanced split
    always keeps both halves within one page (or fails loud — never overflows).

    Each generated set fits two pages (every individual cell is inline-bounded
    and the set is built so a valid 1/1..k cut exists), mirroring what a merge of
    two real leaves produces. A greedy first-crossing split overflows a half on
    many of these; the balanced split must not.
    """
    sizes_pool = [7000, 6500, 6000, 5000, 4000, 3000, 2000, 1500, 800, 400, 200]
    seen_redistribute = False
    for i, big in enumerate(sizes_pool):
        for j, mid in enumerate(sizes_pool):
            for small in sizes_pool:
                cells = [
                    (1, _payload(big, 1)),
                    (2, _payload(mid, 2)),
                    (3, _payload(small, 3)),
                ]
                # Only meaningful when the merged set actually overflows one page
                # (otherwise no redistribute would fire).
                if _merged_leaf_footprint(cells) <= PAGE_SIZE:
                    continue
                # A merge only happens between two siblings that each already fit
                # one page, so the merged set must admit at least one cut into two
                # page-fitting groups. Skip synthetic sets with no valid split
                # (those cannot arise from merging two single-page leaves).
                if not any(
                    _merged_leaf_footprint(cells[:c]) <= PAGE_SIZE
                    and _merged_leaf_footprint(cells[c:]) <= PAGE_SIZE
                    for c in range(1, len(cells))
                ):
                    continue
                left, right = _byte_midpoint_split(cells)
                seen_redistribute = True
                assert [k for k, _ in left] + [k for k, _ in right] == [1, 2, 3]
                assert left and right
                assert _merged_leaf_footprint(left) <= PAGE_SIZE
                assert _merged_leaf_footprint(right) <= PAGE_SIZE
    assert seen_redistribute, "sweep never exercised an overflowing merged set"
