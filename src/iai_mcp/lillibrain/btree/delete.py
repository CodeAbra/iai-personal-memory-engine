"""B+tree deletion: leaf remove, borrow/merge rebalance, interior underflow, root-shrink.

Three entry points consumed by BtreeCursor.delete:

  remove_cell_from_leaf   — remove one cell from a leaf page; return whether anything changed.
  _rebalance_leaf         — fix a leaf that fell below the minimum occupancy:
                            borrow from right, borrow from left, or merge with a sibling.
  _check_interior_underflow — fix an interior node that fell below minimum:
                              borrow via rotation through parent separator, or merge
                              (pull separator down) and propagate upward.
  _shrink_root_if_needed  — when the interior root has zero keys, promote its sole child
                            into the fixed root page (tree height decreases by one;
                            root page number never changes — mirror of _root_split discipline).

Minimum occupancy rule (consistent with _insert_into_interior in split.py):
  LEAF_MIN_CELLS     = LEAF_MAX_CELLS // 2
  INTERIOR_MIN_KEYS  = LEAF_MAX_CELLS // 2

A non-root node with fewer than MIN cells/keys is underflowing. The root is exempt:
it may hold any number of keys including zero (the empty-tree case for leaf root,
or the root-shrink trigger for interior root).

Freed pages (from merge operations) go to the freelist via free_page (trunk
journaled before mutation so aborted transactions fully restore the freelist).
"""
from __future__ import annotations

import bisect
import logging
import struct
from typing import TYPE_CHECKING

from iai_mcp.lillibrain.btree.page import (
    get_next_leaf,
    page_type,
    read_all_leaf_cells,
    read_interior_node,
    set_next_leaf,
    write_interior_node,
    write_leaf_cells,
)
from iai_mcp.lillibrain.constants import (
    LEAF_MAX_CELLS,
    OVERFLOW_THRESHOLD,
    PAGE_INTERIOR_TABLE,
    PAGE_LEAF_TABLE,
    PAGE_SIZE,
)
from iai_mcp.lillibrain.freelist import free_page

if TYPE_CHECKING:
    from iai_mcp.lillibrain.pager import Journal, Pager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimum occupancy thresholds
# ---------------------------------------------------------------------------

# Non-root leaf with fewer than LEAF_MIN_CELLS cells is underflowing.
LEAF_MIN_CELLS: int = LEAF_MAX_CELLS // 2

# Non-root interior node with fewer than INTERIOR_MIN_KEYS keys is underflowing.
# Uses the same constant as leaves so interior-underflow paths are exercised
# whenever LEAF_MAX_CELLS is small.
INTERIOR_MIN_KEYS: int = LEAF_MAX_CELLS // 2


# ---------------------------------------------------------------------------
# Public entry point 1: remove one cell from a leaf
# ---------------------------------------------------------------------------


def remove_cell_from_leaf(pager: Pager, leaf_page_no: int, key: int) -> bool:
    """Remove the cell for key from the leaf page at leaf_page_no.

    Returns True if a cell was found and removed; False if the key was absent
    (idempotent — calling with an absent key is safe and makes no change).

    When the cell being removed is spilled (its declared payload length exceeds
    the spill threshold), the overflow chain is freed through the journaled
    freelist helper before the cell pointer is dropped.  The caller holds an
    open write transaction, so the chain free and the pointer removal commit
    or roll back together — no partial state survives a crash.

    Remaining cells (including any other spilled cells) are preserved exactly —
    their overflow chain pointers are copied verbatim into the rewritten page;
    no chain is freed or reallocated for remaining cells.

    The caller is responsible for holding an open write transaction so that
    the modified page enters the dirty set and is committed or rolled back.
    """
    from iai_mcp.lillibrain.btree.overflow import free_overflow_chain  # noqa: PLC0415
    from iai_mcp.lillibrain.record import decode_varint  # noqa: PLC0415

    page_data = pager.read_page(leaf_page_no)
    mv = memoryview(page_data)

    # Read raw cells (no pager) for key discovery — keys are always inline.
    cells = read_all_leaf_cells(mv)

    idx = bisect.bisect_left([c[0] for c in cells], key)
    if idx >= len(cells) or cells[idx][0] != key:
        # Key not present — no-op.
        return False

    # Detect spill and free only the removed cell's overflow chain.
    from iai_mcp.lillibrain.btree.page import _LEAF_PTR_ARRAY_START  # noqa: PLC0415
    (cell_ptr,) = struct.unpack_from(">H", mv, _LEAF_PTR_ARRAY_START + idx * 2)
    _key_val, kn = decode_varint(mv, cell_ptr)
    payload_len, vn = decode_varint(mv, cell_ptr + kn)

    if payload_len > OVERFLOW_THRESHOLD:
        ovf_ptr_offset = cell_ptr + kn + vn
        (first_ovf,) = struct.unpack_from(">I", mv, ovf_ptr_offset)
        journal = pager._journal  # None in WAL mode
        free_overflow_chain(pager, journal, first_ovf)

    # Rewrite the page preserving remaining cells verbatim.
    # Remaining cells are rebuilt from scratch by reading raw inline payloads
    # (no pager — spilled cells return the raw 4-byte overflow pointer as their
    # "payload") and writing without pager so those raw pointers are written as
    # inline bytes.  However, that would embed the 4-byte pointer as inline data
    # and lose the spill format.  Instead, rebuild the page directly at the
    # binary level: copy each remaining cell's raw bytes from the old page to
    # the new page and rebuild the pointer array from scratch.
    num_cells, _ccs, _fb, _frag = _read_leaf_header_raw(mv)
    sibling = struct.unpack_from(">I", page_data, 8)[0]

    # Collect raw cell bytes for each remaining cell in key order.
    raw_cells: list[bytes] = []
    for i in range(num_cells):
        if i == idx:
            continue
        (ptr,) = struct.unpack_from(">H", mv, _LEAF_PTR_ARRAY_START + i * 2)
        # Determine cell byte length by finding what the next lower offset is.
        # Cells are packed from the end downward; cell_content_start is the
        # lowest occupied offset on the page.  Walk ptr pointers to find size.
        _k, kn_i = decode_varint(mv, ptr)
        pl_i, vn_i = decode_varint(mv, ptr + kn_i)
        if pl_i > OVERFLOW_THRESHOLD:
            cell_len = kn_i + vn_i + 4  # key + len varint + 4-byte ovf ptr
        else:
            cell_len = kn_i + vn_i + pl_i
        raw_cells.append(bytes(mv[ptr : ptr + cell_len]))

    # Write remaining cells into a fresh page buffer.
    new_page = bytearray(PAGE_SIZE)
    write_pos = PAGE_SIZE
    ptrs: list[int] = []
    for raw in raw_cells:
        write_pos -= len(raw)
        new_page[write_pos : write_pos + len(raw)] = raw
        ptrs.append(write_pos)

    from iai_mcp.lillibrain.btree.page import write_leaf_header as _wlh  # noqa: PLC0415
    for i, ptr in enumerate(ptrs):
        struct.pack_into(">H", new_page, _LEAF_PTR_ARRAY_START + i * 2, ptr)
    cell_content_start = write_pos if ptrs else PAGE_SIZE
    _wlh(new_page, num_cells=len(ptrs), cell_content_start=cell_content_start)
    struct.pack_into(">I", new_page, 8, sibling)
    pager.write_page(leaf_page_no, new_page)

    logger.debug(
        "remove_cell_from_leaf: leaf=%d key=%d remaining_cells=%d",
        leaf_page_no, key, len(ptrs),
    )
    return True


def _read_leaf_header_raw(mv: memoryview) -> tuple[int, int, int, int]:
    """Decode the 8-byte leaf page header.  Returns (num_cells, cell_content_start, first_freeblock, fragmented_bytes)."""
    import struct as _s  # noqa: PLC0415
    _pt, fb, nc, ccs, frag = _s.unpack_from(">B HHH B", mv, 0)
    return nc, ccs, fb, frag


# ---------------------------------------------------------------------------
# Private helpers: overflow chain management for leaf rewrites
# ---------------------------------------------------------------------------


def _free_spilled_cell_chains(
    pager: Pager,
    journal: object,
    page_mv: memoryview,
    cell_indices: list[int],
) -> None:
    """Free overflow chains for the cells at cell_indices on the given page.

    Reads declared payload lengths directly from the raw page binary to detect
    spilled cells (payload_len > OVERFLOW_THRESHOLD), then frees each chain
    through the journaled freelist helper before the cell is dropped.

    Called by borrow/merge helpers before rewriting a leaf page so that moving
    a spilled cell to another page does not leave its old overflow chain orphaned.
    """
    from iai_mcp.lillibrain.btree.overflow import free_overflow_chain  # noqa: PLC0415
    from iai_mcp.lillibrain.btree.page import _LEAF_PTR_ARRAY_START  # noqa: PLC0415
    from iai_mcp.lillibrain.record import decode_varint  # noqa: PLC0415

    for idx in cell_indices:
        (cell_ptr,) = struct.unpack_from(">H", page_mv, _LEAF_PTR_ARRAY_START + idx * 2)
        _key_val, kn = decode_varint(page_mv, cell_ptr)
        payload_len, vn = decode_varint(page_mv, cell_ptr + kn)
        if payload_len > OVERFLOW_THRESHOLD:
            ovf_ptr_offset = cell_ptr + kn + vn
            (first_ovf,) = struct.unpack_from(">I", page_mv, ovf_ptr_offset)
            free_overflow_chain(pager, journal, first_ovf)  # type: ignore[arg-type]


def _write_leaf_page_with_overflow(
    pager: Pager,
    journal: object,
    page_no: int,
    cells: list[tuple[int, bytes]],
    sibling_link: int,
) -> None:
    """Write cells to page_no, spilling oversized payloads to overflow chains.

    Convenience wrapper that passes pager/journal to write_leaf_cells and sets
    the sibling link in the written page.  Used by borrow/merge helpers so that
    cells carrying full reassembled payloads are correctly re-spilled when they
    exceed the threshold.
    """
    new_page = bytearray(PAGE_SIZE)
    write_leaf_cells(new_page, cells, pager=pager, journal=journal)  # type: ignore[arg-type]
    struct.pack_into(">I", new_page, 8, sibling_link)
    pager.write_page(page_no, new_page)


# ---------------------------------------------------------------------------
# Private helpers: leaf rebalance (borrow / merge)
# ---------------------------------------------------------------------------


def _borrow_from_right_leaf(
    pager: Pager,
    path_stack: list[tuple[int, int]],
    leaf_page_no: int,
    right_sibling_no: int,
    parent_page_no: int,
    child_idx: int,
) -> None:
    """Move the first cell of right_sibling_no to leaf_page_no.

    Updates the parent separator (the key at position child_idx-1 in the parent)
    to the new first key of the right sibling after the borrow.
    """
    journal = pager._journal  # None in WAL mode

    # Read both leaves with full payload reassembly so oversized cells are
    # re-spilled correctly when written to their new page.
    left_page = pager.read_page(leaf_page_no)
    left_mv = memoryview(left_page)
    left_cells = read_all_leaf_cells(left_mv, pager=pager)
    left_sibling_link = struct.unpack_from(">I", left_page, 8)[0]

    right_page = pager.read_page(right_sibling_no)
    right_mv = memoryview(right_page)
    right_cells = read_all_leaf_cells(right_mv, pager=pager)
    right_sibling_link = struct.unpack_from(">I", right_page, 8)[0]

    # The borrowed cell (right_cells[0]) moves to the left page, and the right page
    # is completely rewritten by _write_leaf_page_with_overflow which re-spills every
    # oversized payload into fresh overflow chains.  Free ALL old chains on the right
    # page before writing so none are orphaned.  This includes the borrowed cell (index
    # 0, moving to the left) and all remaining cells (indices 1+, staying on the right
    # but getting new chains via write_leaf_cells).
    n_right = len(right_cells)
    if n_right:
        _free_spilled_cell_chains(pager, journal, right_mv, list(range(n_right)))

    # Move first cell of right to end of left.
    borrowed = right_cells[0]
    new_left_cells = left_cells + [borrowed]
    new_right_cells = right_cells[1:]

    # New separator in parent = new first key of right sibling.
    new_separator = new_right_cells[0][0]

    # Write left leaf (borrowed cell re-spills if oversized).
    _write_leaf_page_with_overflow(pager, journal, leaf_page_no, new_left_cells, left_sibling_link)

    # Write right leaf (remaining cells get fresh chains via re-spill).
    _write_leaf_page_with_overflow(pager, journal, right_sibling_no, new_right_cells, right_sibling_link)

    # The separator between leaf_page_no (child_idx) and right_sibling_no (child_idx+1)
    # is at parent key index child_idx.
    _update_parent_separator(pager, parent_page_no, child_idx, new_separator)

    logger.debug(
        "_borrow_from_right_leaf: leaf=%d right=%d borrowed_key=%d new_sep=%d",
        leaf_page_no, right_sibling_no, borrowed[0], new_separator,
    )


def _borrow_from_left_leaf(
    pager: Pager,
    leaf_page_no: int,
    left_sibling_no: int,
    parent_page_no: int,
    child_idx: int,
) -> None:
    """Move the last cell of left_sibling_no to leaf_page_no.

    Updates the parent separator (key at position child_idx-1) to the key
    of the borrowed cell (which is now the new first key of leaf_page_no).
    """
    journal = pager._journal  # None in WAL mode

    left_page = pager.read_page(left_sibling_no)
    left_mv = memoryview(left_page)
    left_cells = read_all_leaf_cells(left_mv, pager=pager)
    left_sibling_link = struct.unpack_from(">I", left_page, 8)[0]

    right_page = pager.read_page(leaf_page_no)
    right_mv = memoryview(right_page)
    right_cells = read_all_leaf_cells(right_mv, pager=pager)
    right_sibling_link = struct.unpack_from(">I", right_page, 8)[0]

    # Both pages are completely rewritten by _write_leaf_page_with_overflow which
    # re-spills every oversized payload into fresh overflow chains.  Free ALL old
    # chains on BOTH pages before writing so none are orphaned.
    n_left = len(left_cells)
    if n_left:
        _free_spilled_cell_chains(pager, journal, left_mv, list(range(n_left)))
    n_right = len(right_cells)
    if n_right:
        _free_spilled_cell_chains(pager, journal, right_mv, list(range(n_right)))

    # Move last cell of left to front of right.
    borrowed = left_cells[-1]
    new_left_cells = left_cells[:-1]
    new_right_cells = [borrowed] + right_cells

    # New separator = key of the borrowed cell (now first in right leaf).
    new_separator = borrowed[0]

    # Write left leaf (remaining cells get fresh chains via re-spill).
    _write_leaf_page_with_overflow(pager, journal, left_sibling_no, new_left_cells, left_sibling_link)

    # Write right leaf (borrowed cell and all existing cells re-spill if oversized).
    _write_leaf_page_with_overflow(pager, journal, leaf_page_no, new_right_cells, right_sibling_link)

    # Update the parent separator at position child_idx - 1.
    # The separator between left_sibling and leaf_page_no is at parent key index child_idx-1.
    _update_parent_separator(pager, parent_page_no, child_idx - 1, new_separator)

    logger.debug(
        "_borrow_from_left_leaf: leaf=%d left=%d borrowed_key=%d new_sep=%d",
        leaf_page_no, left_sibling_no, borrowed[0], new_separator,
    )


def _merged_leaf_footprint(cells: list[tuple[int, bytes]]) -> int:
    """Return the on-page byte footprint of cells laid out on one leaf page.

    Uses the same byte arithmetic the insert path owns (_on_page_cell_size and
    _LEAF_PTR_ARRAY_START from cursor) so the merge capacity decision is computed
    from a single source of truth. Each cell's declared length is the full
    reassembled payload length; _on_page_cell_size charges a 4-byte on-page
    pointer for cells whose declared length spills past OVERFLOW_THRESHOLD.
    """
    from iai_mcp.lillibrain.btree.cursor import (  # noqa: PLC0415
        _LEAF_PTR_ARRAY_START,
        _on_page_cell_size,
    )

    ptr_area = _LEAF_PTR_ARRAY_START + len(cells) * 2
    cell_bytes = sum(_on_page_cell_size(k, len(payload)) for k, payload in cells)
    return ptr_area + cell_bytes


def _byte_midpoint_split(
    cells: list[tuple[int, bytes]],
) -> tuple[list[tuple[int, bytes]], list[tuple[int, bytes]]]:
    """Split cells into (left, right) so BOTH halves fit one page by bytes.

    Scans every interior boundary (1..n-1) and keeps the split where both
    halves' on-page footprint stays within one page, choosing the boundary that
    minimizes the larger half so neither side ends up near-empty. A greedy
    first-crossing split is wrong for unequal cell sizes: a single large cell can
    push the left half above PAGE_SIZE even though a valid balanced boundary
    exists elsewhere.

    Cells must already be globally sorted (left_cells + right_cells), so each
    half stays sorted and ordered across the page boundary; left keeps the lower
    keys, right the higher.

    Raises ValueError on fewer than two cells (a 0/1-cell split is meaningless)
    and RuntimeError when no interior boundary keeps both halves within one page
    (unreachable in practice: an inline cell is bounded because inserts spill any
    payload past OVERFLOW_THRESHOLD to an overflow chain, leaving only a 4-byte
    on-page pointer — so the no-split case fails loud instead of corrupting).
    """
    n = len(cells)
    if n < 2:
        raise ValueError(
            f"_byte_midpoint_split needs at least 2 cells to split, got {n}"
        )

    # Scan every interior boundary; keep the one minimizing the larger half among
    # the boundaries where BOTH halves fit a page. Footprints are computed from
    # the insert path's byte arithmetic (single source of truth), which charges a
    # 4-byte on-page pointer for spilled cells, so the costs match what
    # write_leaf_cells will lay down.
    best_idx: int | None = None
    best_max: int | None = None
    for split_idx in range(1, n):
        left = cells[:split_idx]
        right = cells[split_idx:]
        footprint_left = _merged_leaf_footprint(left)
        footprint_right = _merged_leaf_footprint(right)
        if footprint_left > PAGE_SIZE or footprint_right > PAGE_SIZE:
            continue
        larger_half = max(footprint_left, footprint_right)
        if best_max is None or larger_half < best_max:
            best_max = larger_half
            best_idx = split_idx

    if best_idx is None:
        # Fail loud rather than write an overflowing half (would raise
        # struct.error or silently corrupt the page). This must survive python -O,
        # so it is a raise, not an assert.
        raise RuntimeError(
            "redistribute: no two-page split keeps both halves within PAGE_SIZE "
            f"({PAGE_SIZE} bytes); cell set must spill payloads before this path"
        )

    return cells[:best_idx], cells[best_idx:]


def _balanced_nway_leaf_split(
    cells: list[tuple[int, bytes]],
) -> list[list[tuple[int, bytes]]]:
    """Partition globally-sorted cells into N>=2 key-ordered groups each fitting one page.

    Greedy byte-pack left -> right in key order: open a group, keep adding cells
    while the running on-page footprint stays within one page, and close-and-open
    a new group with the next cell when adding it would exceed PAGE_SIZE. Every
    closed group holds at least one cell (a byte-partition guard, never a count
    guard), so no group is ever empty, and groups stay globally key-ordered
    (group i's keys all precede group i+1's) because cells enter already sorted
    and the packer never reorders.

    Footprints are computed via _merged_leaf_footprint (the single source of
    truth shared with the insert byte arithmetic), which charges a spilled cell
    only a 4-byte on-page pointer, so the partition matches what write_leaf_cells
    will actually lay down. The unsplittable two-way set [5198, 7828, 2067]
    packs to [[5198], [7828], [2067]] — each cell alone fits a page.

    Raises RuntimeError (not assert — survives python -O) if any single cell's
    own footprint exceeds one page. Unreachable for current data: an inline cell
    is bounded by OVERFLOW_THRESHOLD = PAGE_SIZE - 35 and a spilled cell costs
    only a 4-byte on-page pointer, so every singleton group fits.
    """
    # Fail-loud backstop: a single cell that cannot fit one page alone. This is
    # the named guard the partition contract requires; it is unreachable today
    # because the insert path spills any payload past OVERFLOW_THRESHOLD.
    for cell in cells:
        footprint = _merged_leaf_footprint([cell])
        if footprint > PAGE_SIZE:
            raise RuntimeError(
                f"leaf split: a single cell footprint {footprint} exceeds one "
                f"page ({PAGE_SIZE} bytes); payload must spill before this path"
            )

    groups: list[list[tuple[int, bytes]]] = []
    current: list[tuple[int, bytes]] = []
    for cell in cells:
        if current and _merged_leaf_footprint(current + [cell]) > PAGE_SIZE:
            groups.append(current)
            current = [cell]
        else:
            current.append(cell)
    if current:
        groups.append(current)

    return groups


def _borrow_fits_receiver(
    receiver_cells: list[tuple[int, bytes]],
    borrowed: tuple[int, bytes],
) -> bool:
    """Return True when the receiving leaf has room for one more borrowed cell.

    Computes the receiving page's footprint with the borrowed cell added using the
    same byte arithmetic the insert path owns. Guards against borrowing into a
    byte-full leaf, which would overflow it; on False the caller falls through to
    merge/redistribute instead.
    """
    return _merged_leaf_footprint(receiver_cells + [borrowed]) <= PAGE_SIZE


def _merge_leaves(
    pager: Pager,
    journal: Journal,
    path_stack: list[tuple[int, int]],
    left_page_no: int,
    right_page_no: int,
    parent_page_no: int,
    separator_idx: int,
) -> None:
    """Rebalance two adjacent leaves: true-merge when they fit one page, else redistribute.

    Reads both leaves' cells and decides by byte footprint:

      - Combined cells fit one page: true merge. All cells move to the left page,
        the right page is freed, and the separator + right child pointer are
        removed from the parent (_remove_from_parent). If the parent underflows,
        _check_interior_underflow runs recursively.

      - Combined cells do NOT fit one page: redistribute. The combined (sorted)
        cells are split by byte-midpoint across both pages, both pages are
        rewritten, the sibling chain stays left -> right -> old_right_sibling, and
        the parent separator is UPDATED to the new boundary key (no page freed,
        no parent key removed). A wrong/removed separator would silently misroute
        lookups, so the separator is always corrected to the new split boundary.

    Both branches free the old overflow chains on every rewritten page before the
    rewrite so no chain is orphaned, and every page write joins the caller's open
    write transaction (crash-safe).
    """
    # Read both leaves with full payload reassembly.
    left_page = pager.read_page(left_page_no)
    left_mv = memoryview(left_page)
    left_cells = read_all_leaf_cells(left_mv, pager=pager)

    right_page = pager.read_page(right_page_no)
    right_mv = memoryview(right_page)
    right_cells = read_all_leaf_cells(right_mv, pager=pager)

    # Get the right leaf's sibling link before either page is freed or rewritten.
    right_sibling_link = struct.unpack_from(">I", right_page, 8)[0]

    # Both pages are completely rewritten (true-merge rewrites the left page;
    # redistribute rewrites both), re-spilling every oversized payload into fresh
    # overflow chains. Free ALL old chains on BOTH pages before writing so none
    # are orphaned.
    n_left = len(left_cells)
    if n_left:
        _free_spilled_cell_chains(pager, journal, left_mv, list(range(n_left)))
    n_right = len(right_cells)
    if n_right:
        _free_spilled_cell_chains(pager, journal, right_mv, list(range(n_right)))

    merged_cells = left_cells + right_cells

    if _merged_leaf_footprint(merged_cells) <= PAGE_SIZE:
        # True merge: combined cells fit one page.
        _write_leaf_page_with_overflow(
            pager, journal, left_page_no, merged_cells, right_sibling_link
        )
        free_page(pager, right_page_no, journal=journal)

        logger.debug(
            "_merge_leaves (merge): left=%d right=%d (freed) merged_count=%d",
            left_page_no, right_page_no, len(merged_cells),
        )

        # Remove the separator and right child pointer from the parent.
        _remove_from_parent(
            pager, journal, path_stack, parent_page_no, separator_idx,
            right_child_idx=separator_idx + 1,
        )
        return

    # Redistribute: combined cells overflow one page. Split by byte-midpoint and
    # rewrite both pages, keeping both children and updating the parent separator.
    new_left, new_right = _byte_midpoint_split(merged_cells)

    # Sibling chain: left -> right -> old_right_sibling. Set the left page's link
    # to the right page, and the right page's link to the old right sibling.
    _write_leaf_page_with_overflow(pager, journal, left_page_no, new_left, right_page_no)
    _write_leaf_page_with_overflow(pager, journal, right_page_no, new_right, right_sibling_link)

    # Update the separator at separator_idx to the new boundary key (the first key
    # of the right page). Do NOT remove it — both children remain.
    _update_parent_separator(pager, parent_page_no, separator_idx, new_right[0][0])

    logger.debug(
        "_merge_leaves (redistribute): left=%d right=%d left_count=%d right_count=%d new_sep=%d",
        left_page_no, right_page_no, len(new_left), len(new_right), new_right[0][0],
    )


def _update_parent_separator(
    pager: Pager,
    parent_page_no: int,
    key_idx: int,
    new_key: int,
) -> None:
    """Replace the key at position key_idx in the parent interior node."""
    page_data = pager.read_page(parent_page_no)
    mv = memoryview(page_data)
    keys, children = read_interior_node(mv)

    keys[key_idx] = new_key

    new_page = bytearray(PAGE_SIZE)
    write_interior_node(new_page, keys, children)
    pager.write_page(parent_page_no, new_page)


def _remove_from_parent(
    pager: Pager,
    journal: Journal,
    path_stack: list[tuple[int, int]],
    parent_page_no: int,
    key_idx: int,
    right_child_idx: int,
) -> None:
    """Remove key at key_idx and child at right_child_idx from an interior node.

    After removing the separator the parent may underflow. If the parent is not
    the root, _check_interior_underflow is called on it.
    """
    page_data = pager.read_page(parent_page_no)
    mv = memoryview(page_data)
    keys, children = read_interior_node(mv)

    new_keys = keys[:key_idx] + keys[key_idx + 1 :]
    new_children = children[:right_child_idx] + children[right_child_idx + 1 :]

    new_page = bytearray(PAGE_SIZE)
    write_interior_node(new_page, new_keys, new_children)
    pager.write_page(parent_page_no, new_page)

    logger.debug(
        "_remove_from_parent: parent=%d removed_key_idx=%d remaining_keys=%d",
        parent_page_no, key_idx, len(new_keys),
    )

    # If the parent is the root, do not rebalance — root may have 0 keys
    # (single-child root triggers root-shrink in _shrink_root_if_needed).
    if not path_stack:
        # parent_page_no is the root — nothing to rebalance here.
        return

    # Check if the parent underflows now.
    if len(new_keys) < INTERIOR_MIN_KEYS:
        _check_interior_underflow(pager, journal, path_stack, parent_page_no)


# ---------------------------------------------------------------------------
# Public entry point 2: rebalance an underflowing leaf
# ---------------------------------------------------------------------------


def _rebalance_leaf(
    pager: Pager,
    journal: Journal,
    path_stack: list[tuple[int, int]],
    leaf_page_no: int,
) -> None:
    """Fix an underflowing non-root leaf via borrow or merge.

    Tries in order:
      1. borrow from right sibling (if right exists and has spare cells)
      2. borrow from left sibling  (if left exists and has spare cells)
      3. merge with right sibling  (if right exists)
      4. merge with left sibling   (only option left)

    The path_stack is (root → parent of leaf), so the last entry is the
    direct parent of leaf_page_no.
    """
    if not path_stack:
        # Leaf is the root — no rebalancing needed; root may be empty.
        return

    parent_page_no, child_idx = path_stack[-1]
    parent_stack = path_stack[:-1]  # stack above the parent

    page_data = pager.read_page(parent_page_no)
    mv = memoryview(page_data)
    parent_keys, parent_children = read_interior_node(mv)

    # Find our position in the parent's children list.
    # child_idx is the index stored when we descended through this parent.
    # Verify it matches (it should be the index we stored during move_to).
    assert parent_children[child_idx] == leaf_page_no, (
        f"_rebalance_leaf: parent={parent_page_no} child_idx={child_idx} "
        f"expected leaf_page_no={leaf_page_no}, got {parent_children[child_idx]}"
    )

    right_sibling_no = parent_children[child_idx + 1] if child_idx + 1 < len(parent_children) else None
    left_sibling_no = parent_children[child_idx - 1] if child_idx - 1 >= 0 else None

    # The receiving leaf gains one cell on a borrow; its used bytes are read once
    # so the borrow can be rejected when the moved cell would overflow it.
    recv_page = pager.read_page(leaf_page_no)
    recv_mv = memoryview(recv_page)
    recv_cells_raw = read_all_leaf_cells(recv_mv, pager=pager)

    # Check right sibling capacity.
    if right_sibling_no is not None:
        right_page = pager.read_page(right_sibling_no)
        right_mv = memoryview(right_page)
        right_cells = read_all_leaf_cells(right_mv)
        if len(right_cells) > LEAF_MIN_CELLS:
            # The borrowed cell is the right sibling's first cell. Only borrow if
            # it fits the receiving leaf by bytes; otherwise fall through to
            # merge/redistribute (a byte-full receiver must not overflow).
            borrowed = read_all_leaf_cells(right_mv, pager=pager)[0]
            if _borrow_fits_receiver(recv_cells_raw, borrowed):
                _borrow_from_right_leaf(
                    pager, path_stack, leaf_page_no, right_sibling_no,
                    parent_page_no, child_idx
                )
                return

    # Check left sibling capacity.
    if left_sibling_no is not None:
        left_page = pager.read_page(left_sibling_no)
        left_mv = memoryview(left_page)
        left_cells = read_all_leaf_cells(left_mv)
        if len(left_cells) > LEAF_MIN_CELLS:
            # The borrowed cell is the left sibling's last cell.
            borrowed = read_all_leaf_cells(left_mv, pager=pager)[-1]
            if _borrow_fits_receiver(recv_cells_raw, borrowed):
                _borrow_from_left_leaf(
                    pager, leaf_page_no, left_sibling_no,
                    parent_page_no, child_idx
                )
                return

    # Must merge. Prefer merging with right sibling; fall back to left.
    if right_sibling_no is not None:
        # separator_idx = the key at position child_idx in the parent separates
        # child_idx from child_idx+1 (right sibling).
        separator_idx = child_idx
        _merge_leaves(
            pager, journal, parent_stack,
            leaf_page_no, right_sibling_no,
            parent_page_no, separator_idx,
        )
    elif left_sibling_no is not None:
        # Merge into the left sibling; separator_idx = child_idx - 1.
        separator_idx = child_idx - 1
        _merge_leaves(
            pager, journal, parent_stack,
            left_sibling_no, leaf_page_no,
            parent_page_no, separator_idx,
        )
    else:
        # Only child in the parent — should not happen for a non-root leaf
        # in a valid tree (parent always has at least 2 children).
        logger.warning(
            "_rebalance_leaf: leaf=%d has no siblings (parent=%d children=%s); "
            "skipping rebalance",
            leaf_page_no, parent_page_no, parent_children,
        )


# ---------------------------------------------------------------------------
# Private helpers: interior rebalance (borrow / merge)
# ---------------------------------------------------------------------------


def _borrow_from_right_interior(
    pager: Pager,
    interior_page_no: int,
    right_sibling_no: int,
    parent_page_no: int,
    child_idx: int,
) -> None:
    """Rotate one (key, child) from right_sibling_no through the parent into interior_page_no.

    Interior borrow (right):
      - Pull the parent separator (key at position child_idx) down into the
        underflowing node as its new rightmost key.
      - Push the right sibling's first key up to become the new parent separator.
      - Move the right sibling's first child (left-most child) to become the new
        rightmost child of the underflowing node.
    """
    left_page = pager.read_page(interior_page_no)
    left_keys, left_children = read_interior_node(memoryview(left_page))

    right_page = pager.read_page(right_sibling_no)
    right_keys, right_children = read_interior_node(memoryview(right_page))

    parent_page = pager.read_page(parent_page_no)
    parent_keys, parent_children = read_interior_node(memoryview(parent_page))

    # The separator between interior_page_no and right_sibling_no is parent_keys[child_idx].
    pulled_separator = parent_keys[child_idx]
    pushed_up = right_keys[0]
    adopted_child = right_children[0]

    # Update underflowing node: append pulled_separator as new rightmost key;
    # adopt the first child of right sibling.
    new_left_keys = left_keys + [pulled_separator]
    new_left_children = left_children + [adopted_child]

    # Update right sibling: remove first key and first child.
    new_right_keys = right_keys[1:]
    new_right_children = right_children[1:]

    # Update parent separator.
    new_parent_keys = parent_keys[:child_idx] + [pushed_up] + parent_keys[child_idx + 1 :]

    new_left_page = bytearray(PAGE_SIZE)
    write_interior_node(new_left_page, new_left_keys, new_left_children)
    pager.write_page(interior_page_no, new_left_page)

    new_right_page = bytearray(PAGE_SIZE)
    write_interior_node(new_right_page, new_right_keys, new_right_children)
    pager.write_page(right_sibling_no, new_right_page)

    new_parent_page = bytearray(PAGE_SIZE)
    write_interior_node(new_parent_page, new_parent_keys, parent_children)
    pager.write_page(parent_page_no, new_parent_page)

    logger.debug(
        "_borrow_from_right_interior: node=%d right=%d pulled=%d pushed_up=%d",
        interior_page_no, right_sibling_no, pulled_separator, pushed_up,
    )


def _borrow_from_left_interior(
    pager: Pager,
    interior_page_no: int,
    left_sibling_no: int,
    parent_page_no: int,
    child_idx: int,
) -> None:
    """Rotate one (key, child) from left_sibling_no through the parent into interior_page_no.

    Interior borrow (left):
      - Pull the parent separator (key at position child_idx-1) down into the
        underflowing node as its new first key (at the front).
      - Push the left sibling's last key up to become the new parent separator.
      - Move the left sibling's last child to become the new first child of the
        underflowing node.
    """
    right_page = pager.read_page(interior_page_no)
    right_keys, right_children = read_interior_node(memoryview(right_page))

    left_page = pager.read_page(left_sibling_no)
    left_keys, left_children = read_interior_node(memoryview(left_page))

    parent_page = pager.read_page(parent_page_no)
    parent_keys, parent_children = read_interior_node(memoryview(parent_page))

    # The separator between left_sibling and interior_page_no is parent_keys[child_idx-1].
    pulled_separator = parent_keys[child_idx - 1]
    pushed_up = left_keys[-1]
    donated_child = left_children[-1]

    # Update underflowing node: prepend pulled_separator as new first key;
    # prepend donated child as new first child.
    new_right_keys = [pulled_separator] + right_keys
    new_right_children = [donated_child] + right_children

    # Update left sibling: remove last key and last child.
    new_left_keys = left_keys[:-1]
    new_left_children = left_children[:-1]

    # Update parent separator.
    new_parent_keys = (
        parent_keys[: child_idx - 1] + [pushed_up] + parent_keys[child_idx:]
    )

    new_right_page = bytearray(PAGE_SIZE)
    write_interior_node(new_right_page, new_right_keys, new_right_children)
    pager.write_page(interior_page_no, new_right_page)

    new_left_page = bytearray(PAGE_SIZE)
    write_interior_node(new_left_page, new_left_keys, new_left_children)
    pager.write_page(left_sibling_no, new_left_page)

    new_parent_page = bytearray(PAGE_SIZE)
    write_interior_node(new_parent_page, new_parent_keys, parent_children)
    pager.write_page(parent_page_no, new_parent_page)

    logger.debug(
        "_borrow_from_left_interior: node=%d left=%d pulled=%d pushed_up=%d",
        interior_page_no, left_sibling_no, pulled_separator, pushed_up,
    )


def _merge_interior_nodes(
    pager: Pager,
    journal: Journal,
    path_stack: list[tuple[int, int]],
    left_page_no: int,
    right_page_no: int,
    parent_page_no: int,
    separator_idx: int,
) -> None:
    """Merge right_page_no into left_page_no, pulling the parent separator down.

    The merged interior node = left_keys + [parent_separator] + right_keys,
    children = left_children + right_children.
    The parent separator at separator_idx and the right child pointer are removed
    from the parent. The right page is freed.
    """
    left_page = pager.read_page(left_page_no)
    left_keys, left_children = read_interior_node(memoryview(left_page))

    right_page = pager.read_page(right_page_no)
    right_keys, right_children = read_interior_node(memoryview(right_page))

    parent_page = pager.read_page(parent_page_no)
    parent_keys, parent_children = read_interior_node(memoryview(parent_page))

    # Pull the separator down between left and right.
    pulled_separator = parent_keys[separator_idx]

    merged_keys = left_keys + [pulled_separator] + right_keys
    merged_children = left_children + right_children

    new_left_page = bytearray(PAGE_SIZE)
    write_interior_node(new_left_page, merged_keys, merged_children)
    pager.write_page(left_page_no, new_left_page)

    # Free the right interior page.
    free_page(pager, right_page_no, journal=journal)

    logger.debug(
        "_merge_interior_nodes: left=%d right=%d (freed) separator=%d merged_keys=%d",
        left_page_no, right_page_no, pulled_separator, len(merged_keys),
    )

    # Remove the separator and right child pointer from the parent.
    new_parent_keys = parent_keys[:separator_idx] + parent_keys[separator_idx + 1 :]
    new_parent_children = (
        parent_children[: separator_idx + 1] + parent_children[separator_idx + 2 :]
    )

    new_parent_page = bytearray(PAGE_SIZE)
    write_interior_node(new_parent_page, new_parent_keys, new_parent_children)
    pager.write_page(parent_page_no, new_parent_page)

    # If the parent is the root, it may now have zero keys (triggers root-shrink
    # in _shrink_root_if_needed). Do not rebalance the root here.
    if not path_stack:
        return

    # Otherwise the parent may underflow.
    if len(new_parent_keys) < INTERIOR_MIN_KEYS:
        _check_interior_underflow(pager, journal, path_stack, parent_page_no)


# ---------------------------------------------------------------------------
# Public entry point 3: rebalance an underflowing interior node
# ---------------------------------------------------------------------------


def _check_interior_underflow(
    pager: Pager,
    journal: Journal,
    path_stack: list[tuple[int, int]],
    interior_page_no: int,
) -> None:
    """Fix an underflowing interior node via borrow or merge, then recurse upward.

    path_stack is the stack ABOVE interior_page_no (root → parent of interior_page_no).
    If path_stack is empty, interior_page_no is the root — exempt from underflow.
    """
    if not path_stack:
        # Root is exempt — root-shrink is handled separately in _shrink_root_if_needed.
        return

    parent_page_no, child_idx = path_stack[-1]
    parent_stack = path_stack[:-1]

    parent_page = pager.read_page(parent_page_no)
    parent_keys, parent_children = read_interior_node(memoryview(parent_page))

    assert parent_children[child_idx] == interior_page_no, (
        f"_check_interior_underflow: parent={parent_page_no} child_idx={child_idx} "
        f"expected={interior_page_no}, got={parent_children[child_idx]}"
    )

    right_sibling_no = (
        parent_children[child_idx + 1] if child_idx + 1 < len(parent_children) else None
    )
    left_sibling_no = parent_children[child_idx - 1] if child_idx - 1 >= 0 else None

    # Try borrowing from right sibling.
    if right_sibling_no is not None:
        right_page = pager.read_page(right_sibling_no)
        right_keys, _ = read_interior_node(memoryview(right_page))
        if len(right_keys) > INTERIOR_MIN_KEYS:
            _borrow_from_right_interior(
                pager, interior_page_no, right_sibling_no, parent_page_no, child_idx
            )
            return

    # Try borrowing from left sibling.
    if left_sibling_no is not None:
        left_page = pager.read_page(left_sibling_no)
        left_keys, _ = read_interior_node(memoryview(left_page))
        if len(left_keys) > INTERIOR_MIN_KEYS:
            _borrow_from_left_interior(
                pager, interior_page_no, left_sibling_no, parent_page_no, child_idx
            )
            return

    # Must merge. Prefer merging with right sibling.
    if right_sibling_no is not None:
        separator_idx = child_idx
        _merge_interior_nodes(
            pager, journal, parent_stack,
            interior_page_no, right_sibling_no,
            parent_page_no, separator_idx,
        )
    elif left_sibling_no is not None:
        separator_idx = child_idx - 1
        _merge_interior_nodes(
            pager, journal, parent_stack,
            left_sibling_no, interior_page_no,
            parent_page_no, separator_idx,
        )
    else:
        logger.warning(
            "_check_interior_underflow: node=%d has no siblings "
            "(parent=%d children=%s); skipping rebalance",
            interior_page_no, parent_page_no, parent_children,
        )


# ---------------------------------------------------------------------------
# Public entry point 4: root-shrink
# ---------------------------------------------------------------------------


def _shrink_root_if_needed(pager: Pager, journal: Journal) -> None:
    """Shrink the root if it is an interior node with zero keys (one child).

    When the interior root has exactly one child and zero keys, that child
    becomes the new root content. The child's content is copied into the fixed
    root page and the old child page is freed. The root page number never changes
    (mirror of _root_split fixed-root discipline from split.py).

    Repeat until the root is either a leaf or has at least one key.
    """
    while True:
        root_data = pager.read_page(pager.root_page_no)
        mv = memoryview(root_data)
        ptype = page_type(mv)

        if ptype == PAGE_LEAF_TABLE:
            # Root is already a leaf — nothing to shrink.
            return

        # Root is an interior node.
        root_keys, root_children = read_interior_node(mv)

        if len(root_keys) > 0:
            # Root has keys — tree is not degenerate; stop.
            return

        # Root has zero keys → exactly one child remains.
        assert len(root_children) == 1, (
            f"_shrink_root_if_needed: root has 0 keys but "
            f"{len(root_children)} children (expected 1)"
        )

        only_child_no = root_children[0]

        # Copy child content into the fixed root page.
        child_data = pager.read_page(only_child_no)
        new_root_page = bytearray(child_data)
        pager.write_page(pager.root_page_no, new_root_page)

        # Free the old child page.
        free_page(pager, only_child_no, journal=journal)

        logger.debug(
            "_shrink_root_if_needed: root=%d promoted child=%d (freed); "
            "height decreased by one",
            pager.root_page_no, only_child_no,
        )

        # Loop to handle multi-level collapse (e.g. interior root → interior child
        # also has zero keys after the copy).
