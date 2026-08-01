"""B+tree cursor: tree descent, insertion with split dispatch, deletion, and sequential scan.

BtreeCursor wraps a Pager and provides:
  - move_to(key): descent from the root to the correct leaf, building a path
    stack for split-code ascent; returns (leaf_page_no, insert_index).
  - insert(key, value): insert-or-replace; dispatches to leaf_split_and_insert
    when the target leaf is full.
  - delete(key): remove a key; idempotent if absent; rebalances the leaf via
    borrow/merge on underflow and shrinks the root when the interior root collapses.
  - next() / full_scan(): sequential scan via the leaf sibling-link chain.
  - path_stack(): the (page_no, child_index) stack built by the last move_to,
    root → leaf order, consumed by split ascent and delete rebalance.

The cursor calls pager.read_page and pager.write_page exclusively — it never
accesses file descriptors or journal objects directly. The caller must hold an
open write transaction (pager.begin_write()) for any insert that may split.
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
    read_interior_header,
    read_interior_node,
    read_leaf_header,
    write_leaf_cells,
    write_leaf_header,
)
from iai_mcp.lillibrain.constants import (
    INTERIOR_HEADER_SIZE,
    LEAF_HEADER_SIZE,
    LEAF_MAX_CELLS,
    OVERFLOW_THRESHOLD,
    PAGE_INTERIOR_TABLE,
    PAGE_LEAF_TABLE,
    PAGE_SIZE,
)
from iai_mcp.lillibrain.record import decode_record_views, decode_varint, encode_varint

if TYPE_CHECKING:
    import collections.abc

    from iai_mcp.lillibrain.pager import Pager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NeedsSplit signal
# ---------------------------------------------------------------------------


# Byte offset where the cell pointer array starts on a leaf page.
# LEAF_HEADER_SIZE (8) + sibling-link reservation (4) = 12.
_LEAF_PTR_ARRAY_START: int = LEAF_HEADER_SIZE + 4

# The default leaf cell ceiling. At this value byte capacity governs splits and
# the count-based underflow floor is suppressed (it would otherwise flag every
# real leaf as underflowing). A test that injects a smaller LILLI_LEAF_MAX_CELLS
# re-enables the count floor so merge/borrow paths fire on a few small cells.
_SMALL_CAP_SENTINEL: int = 9999


def _on_page_cell_size(k: int, declared_len: int) -> int:
    """Return the on-page byte cost of one leaf cell given its key and declared payload length.

    For spilled cells (declared_len > OVERFLOW_THRESHOLD) the on-page payload is
    a 4-byte overflow-page pointer; for inline cells it is declared_len bytes.
    The declared_len varint is always sized from the actual declared length, not
    from the on-page payload bytes — this is what distinguishes correct sizing from
    the old approach of using len(returned_payload).
    """
    payload_bytes = 4 if declared_len > OVERFLOW_THRESHOLD else declared_len
    return len(encode_varint(k)) + len(encode_varint(declared_len)) + payload_bytes


def _leaf_has_room_for(
    mv: memoryview,
    num_cells: int,
    exclude_idx: int | None,
    new_key: int,
    new_value: bytes,
) -> bool:
    """Return True if the new cell fits alongside the existing cells on this leaf page.

    Reads the declared payload length for each existing cell directly from the
    raw page binary via decode_varint so that spilled cells (which store only a
    4-byte overflow pointer on-page) are sized using their true declared length,
    not the 4 bytes returned by read_all_leaf_cells when called without a pager.

    exclude_idx: if not None, the existing cell at that index is skipped (used
    when a replace removes the old cell before checking whether the new value fits).

    Computes the exact byte footprint of all retained existing cells plus the new
    cell, using the same layout as write_leaf_cells (cells packed downward from the
    end of the page; pointer array growing upward from _LEAF_PTR_ARRAY_START).

    Returns True when all cells fit within PAGE_SIZE; False triggers a leaf split.
    """
    # Sum on-page costs of existing cells, reading declared lengths from the binary.
    existing_bytes = 0
    kept = 0
    for i in range(num_cells):
        if i == exclude_idx:
            continue
        (ptr,) = struct.unpack_from(">H", mv, _LEAF_PTR_ARRAY_START + i * 2)
        k, kn = decode_varint(mv, ptr)
        declared_len, _vn = decode_varint(mv, ptr + kn)
        existing_bytes += _on_page_cell_size(k, declared_len)
        kept += 1

    # New cell: len(new_value) is the authoritative declared length (caller supplies
    # the real value, not a raw page payload).
    new_cell_bytes = _on_page_cell_size(new_key, len(new_value))

    n_total = kept + 1  # existing kept + the new cell
    ptr_area = _LEAF_PTR_ARRAY_START + n_total * 2
    return ptr_area + existing_bytes + new_cell_bytes <= PAGE_SIZE


def _leaf_used_bytes(mv: memoryview, num_cells: int) -> int:
    """Return the on-page byte footprint currently used by a leaf page.

    Reads each cell's declared payload length from the raw page binary (so spilled
    cells are charged a 4-byte on-page pointer, not the reassembled length) and
    adds the pointer-array area. Mirrors the arithmetic of _leaf_has_room_for.
    """
    used = _LEAF_PTR_ARRAY_START + num_cells * 2
    for i in range(num_cells):
        (ptr,) = struct.unpack_from(">H", mv, _LEAF_PTR_ARRAY_START + i * 2)
        k, kn = decode_varint(mv, ptr)
        declared_len, _vn = decode_varint(mv, ptr + kn)
        used += _on_page_cell_size(k, declared_len)
    return used


def _leaf_underflows(mv: memoryview, num_cells: int) -> bool:
    """Return True when a leaf is below the half-page fill threshold (or empty).

    A leaf is underfull by bytes when its used on-page footprint is less than
    half the page, or when it holds no cells. This is the byte-driven analogue of
    a count floor: a byte-full leaf (e.g. four 1656-byte cells) is NOT underfull,
    so deletes do not force a rebalance on every byte-full leaf.
    """
    if num_cells == 0:
        return True
    return _leaf_used_bytes(mv, num_cells) < PAGE_SIZE // 2


def _interior_has_room_for(keys: list[int], new_key: int, page_size: int) -> bool:
    """Return True if adding new_key fits within the interior page byte budget.

    Interior page layout:
      INTERIOR_HEADER_SIZE bytes of header
      + n * 2 bytes of pointer array (one slot per cell)
      + n * (4 + varint_len(key)) bytes of cell data (4-byte left-child + key)

    The rightmost child pointer lives in the header, not in a cell, so it is
    excluded from the cell count.  Returns False when the combined footprint
    would exceed page_size bytes.
    """
    from iai_mcp.lillibrain.record import encode_varint  # noqa: PLC0415

    all_keys = list(keys) + [new_key]
    n = len(all_keys)
    ptr_area = INTERIOR_HEADER_SIZE + n * 2
    cell_bytes = sum(4 + len(encode_varint(k)) for k in all_keys)
    return ptr_area + cell_bytes <= page_size


# ---------------------------------------------------------------------------
# Raw leaf-page rewrite helpers
# ---------------------------------------------------------------------------
#
# These helpers rebuild a leaf page by binary-copying existing cells verbatim
# (preserving any overflow chain pointers without re-allocating chains) and
# either inserting or replacing exactly one cell that may spill if oversized.
# They avoid the "read full payload → re-spill all cells" approach which would
# double-allocate overflow chains for cells that did not change.


def _raw_cell_bytes(mv: memoryview, cell_idx: int, num_cells: int) -> bytes:
    """Return the raw on-page bytes for the cell at cell_idx on a leaf page.

    Reads the declared payload length from the binary to determine whether the
    cell is inline or spilled, then extracts exactly the on-page bytes.
    """
    (ptr,) = struct.unpack_from(">H", mv, _LEAF_PTR_ARRAY_START + cell_idx * 2)
    _k, kn = decode_varint(mv, ptr)
    pl, vn = decode_varint(mv, ptr + kn)
    cell_len = kn + vn + (4 if pl > OVERFLOW_THRESHOLD else pl)
    return bytes(mv[ptr : ptr + cell_len])


def _raw_copy_leaf_excluding(
    mv: memoryview, num_cells: int, exclude_idx: int, sibling_link: int
) -> bytearray:
    """Return a new leaf page with all cells except exclude_idx copied verbatim."""
    from iai_mcp.lillibrain.btree.page import write_leaf_header as _wlh  # noqa: PLC0415
    from iai_mcp.lillibrain.constants import OVERFLOW_THRESHOLD as _OT  # noqa: PLC0415

    new_page = bytearray(PAGE_SIZE)
    write_pos = PAGE_SIZE
    ptrs: list[int] = []
    for i in range(num_cells):
        if i == exclude_idx:
            continue
        raw = _raw_cell_bytes(mv, i, num_cells)
        write_pos -= len(raw)
        new_page[write_pos : write_pos + len(raw)] = raw
        ptrs.append(write_pos)
    for j, p in enumerate(ptrs):
        struct.pack_into(">H", new_page, _LEAF_PTR_ARRAY_START + j * 2, p)
    ccs = write_pos if ptrs else PAGE_SIZE
    _wlh(new_page, num_cells=len(ptrs), cell_content_start=ccs)
    struct.pack_into(">I", new_page, 8, sibling_link)
    return new_page


def _write_leaf_replacing_one(
    pager: object,
    journal: object,
    page_no: int,
    mv: memoryview,
    num_cells: int,
    replace_idx: int,
    new_key: int,
    new_value: bytes,
    sibling_link: int,
) -> None:
    """Write a leaf page replacing the cell at replace_idx with (new_key, new_value).

    All other cells are binary-copied verbatim (overflow chain pointers preserved).
    The replacement cell is written fresh and spills if len(new_value) > OVERFLOW_THRESHOLD.
    """
    from iai_mcp.lillibrain.btree.overflow import alloc_overflow_chain as _aoc  # noqa: PLC0415
    from iai_mcp.lillibrain.btree.page import write_leaf_header as _wlh  # noqa: PLC0415

    # Build the replacement cell binary.
    key_enc = encode_varint(new_key)
    if len(new_value) > OVERFLOW_THRESHOLD:
        first_ovf = _aoc(pager, journal, new_value)  # type: ignore[arg-type]
        new_cell = key_enc + encode_varint(len(new_value)) + struct.pack(">I", first_ovf)
    else:
        new_cell = key_enc + encode_varint(len(new_value)) + new_value

    # Collect all cells in order, substituting the new cell at replace_idx.
    raw_cells: list[bytes] = []
    for i in range(num_cells):
        if i == replace_idx:
            raw_cells.append(new_cell)
        else:
            raw_cells.append(_raw_cell_bytes(mv, i, num_cells))

    new_page = bytearray(PAGE_SIZE)
    write_pos = PAGE_SIZE
    ptrs: list[int] = []
    for raw in raw_cells:
        write_pos -= len(raw)
        new_page[write_pos : write_pos + len(raw)] = raw
        ptrs.append(write_pos)
    for j, p in enumerate(ptrs):
        struct.pack_into(">H", new_page, _LEAF_PTR_ARRAY_START + j * 2, p)
    ccs = write_pos if ptrs else PAGE_SIZE
    _wlh(new_page, num_cells=len(ptrs), cell_content_start=ccs)
    struct.pack_into(">I", new_page, 8, sibling_link)
    pager.write_page(page_no, new_page)  # type: ignore[attr-defined]


def _write_leaf_inserting_one(
    pager: object,
    journal: object,
    page_no: int,
    mv: memoryview,
    num_cells: int,
    insert_idx: int,
    new_key: int,
    new_value: bytes,
    sibling_link: int,
) -> None:
    """Write a leaf page inserting (new_key, new_value) at insert_idx.

    All existing cells are binary-copied verbatim (overflow chain pointers
    preserved).  The new cell is written fresh and spills if oversized.
    """
    from iai_mcp.lillibrain.btree.overflow import alloc_overflow_chain as _aoc  # noqa: PLC0415
    from iai_mcp.lillibrain.btree.page import write_leaf_header as _wlh  # noqa: PLC0415

    key_enc = encode_varint(new_key)
    if len(new_value) > OVERFLOW_THRESHOLD:
        first_ovf = _aoc(pager, journal, new_value)  # type: ignore[arg-type]
        new_cell = key_enc + encode_varint(len(new_value)) + struct.pack(">I", first_ovf)
    else:
        new_cell = key_enc + encode_varint(len(new_value)) + new_value

    raw_cells: list[bytes] = []
    for i in range(num_cells):
        raw_cells.append(_raw_cell_bytes(mv, i, num_cells))
    raw_cells.insert(insert_idx, new_cell)

    new_page = bytearray(PAGE_SIZE)
    write_pos = PAGE_SIZE
    ptrs: list[int] = []
    for raw in raw_cells:
        write_pos -= len(raw)
        new_page[write_pos : write_pos + len(raw)] = raw
        ptrs.append(write_pos)
    for j, p in enumerate(ptrs):
        struct.pack_into(">H", new_page, _LEAF_PTR_ARRAY_START + j * 2, p)
    ccs = write_pos if ptrs else PAGE_SIZE
    _wlh(new_page, num_cells=len(ptrs), cell_content_start=ccs)
    struct.pack_into(">I", new_page, 8, sibling_link)
    pager.write_page(page_no, new_page)  # type: ignore[attr-defined]


class NeedsSplit(Exception):
    """Raised by BtreeCursor.insert when the target leaf is full.

    Carries the leaf page number and the path stack so that the split code
    can ascend and restructure the tree without redescending.

    Attributes:
        leaf_page_no: The page number of the full leaf.
        path_stack: Copy of the cursor's path stack (root → parent of leaf).
    """

    def __init__(self, leaf_page_no: int, path_stack: list[tuple[int, int]]) -> None:
        self.leaf_page_no = leaf_page_no
        self.path_stack = path_stack
        super().__init__(
            f"leaf page {leaf_page_no} is full (NeedsSplit); split required before insert"
        )


# ---------------------------------------------------------------------------
# BtreeCursor
# ---------------------------------------------------------------------------


class BtreeCursor:
    """Cursor for navigating and mutating a LilliBrain B+tree.

    The cursor holds a Pager reference and the fixed root page number. It does
    not open transactions — callers must call pager.begin_write() before any
    insert and pager.commit() or pager.rollback() afterward.

    The cursor is stateful: move_to() updates the internal path stack and the
    current leaf iterator position used by next(). Callers should call move_to()
    before any next() traversal to reset to a known position.
    """

    def __init__(self, pager: Pager, root_page_no: int) -> None:
        self._pager = pager
        self._root_page_no = root_page_no

        # Path stack: list of (page_no, child_index) from root → parent of leaf.
        # Built by move_to(); consumed by NeedsSplit / split code.
        self._path_stack: list[tuple[int, int]] = []

        # Current position for next() iteration.
        # _scan_page_no: the leaf page currently being iterated.
        # _scan_cell_idx: the next cell index to return.
        self._scan_page_no: int | None = None
        self._scan_cell_idx: int = 0
        self._scan_cells: list[tuple[int, bytes]] = []

    # ---------------------------------------------------------------------------
    # Tree descent
    # ---------------------------------------------------------------------------

    def move_to(self, key: int) -> tuple[int, int]:
        """Descend from the root to the leaf that should contain key.

        Builds self._path_stack as [(page_no, child_index), ...] from the root
        down to (but not including) the leaf. Each tuple records the page number
        of an interior node and the index of the child followed.

        Returns (leaf_page_no, insert_index) where insert_index is the position
        in the leaf's cell array at which key should be inserted (0 = before all
        cells; num_cells = after all cells; an existing key's own index = replace).
        """
        self._path_stack = []
        page_no = self._root_page_no

        while True:
            page_data = self._pager.read_page(page_no)
            mv = memoryview(page_data)
            ptype = page_type(mv)

            if ptype == PAGE_LEAF_TABLE:
                # At the leaf — find insert index.
                cells = read_all_leaf_cells(mv)
                keys = [c[0] for c in cells]
                # bisect_left: returns the leftmost position for key.
                # If keys[idx] == key, this is the replace position.
                idx = bisect.bisect_left(keys, key)
                return page_no, idx

            # Interior node — descend.
            keys, children = read_interior_node(mv)
            # child_index = number of keys less than or equal to key that we
            # pass to the right → use bisect_right to find the child slot.
            child_idx = bisect.bisect_right(keys, key)
            self._path_stack.append((page_no, child_idx))
            page_no = children[child_idx]

    def _leftmost_leaf(self) -> int:
        """Descend from the root always taking child[0] to reach the leftmost leaf."""
        page_no = self._root_page_no
        while True:
            page_data = self._pager.read_page(page_no)
            mv = memoryview(page_data)
            ptype = page_type(mv)
            if ptype == PAGE_LEAF_TABLE:
                return page_no
            _keys, children = read_interior_node(mv)
            page_no = children[0]

    def max_key(self) -> int | None:
        """Return the largest key in the tree, or None when the tree is empty.

        Descends from the root always following the rightmost child to reach the
        rightmost leaf, then returns that leaf's last cell key. Because the tree
        is strictly key-ordered (cells ascend within a leaf and leaves ascend
        along the sibling chain), the rightmost leaf's last cell holds the global
        maximum. This is O(tree height), not O(rows): each page contributes one
        header read plus, at the leaf, one cell-pointer + one key varint decode —
        no leaf-cell materialisation and no sibling-chain walk.
        """
        page_no = self._root_page_no
        while True:
            page_data = self._pager.read_page(page_no)
            mv = memoryview(page_data)
            if page_type(mv) == PAGE_LEAF_TABLE:
                num_cells, _ccs, _fb, _frag = read_leaf_header(mv)
                if num_cells == 0:
                    return None
                (ptr,) = struct.unpack_from(
                    ">H", mv, _LEAF_PTR_ARRAY_START + (num_cells - 1) * 2
                )
                key, _n = decode_varint(mv, ptr)
                return key
            # Interior node — follow the rightmost child from the header.
            _num_cells, _ccs, rightmost = read_interior_header(mv)
            page_no = rightmost

    def path_stack(self) -> list[tuple[int, int]]:
        """Return the path stack from the last move_to call.

        Each entry is (page_no, child_index), ordered from root to the parent
        of the leaf. The split code pops from this list to ascend.
        """
        return list(self._path_stack)

    # ---------------------------------------------------------------------------
    # Insert (with split dispatch)
    # ---------------------------------------------------------------------------

    def insert(self, key: int, value: bytes) -> None:
        """Insert or replace a (key, value) cell into the correct leaf.

        If the leaf has byte capacity for the new cell (or the key already exists
        and is being replaced in-place), the cell is written in key order.
        Byte capacity is the governing gate: a leaf splits when the encoded cell
        bytes would overflow the page, not when a fixed cell count is reached.
        LEAF_MAX_CELLS provides a secondary ceiling for test injection.

        If the leaf is full and the key is new, leaf_split_and_insert is called
        to split the leaf and propagate the separator upward. The caller must
        hold an open write transaction (pager.begin_write()) before any insert
        that may trigger a split — the journal is required for freelist alloc.

        Raises RuntimeError if insert is called without an open write transaction
        when a split is needed.
        """
        leaf_page_no, insert_idx = self.move_to(key)
        page_data = self._pager.read_page(leaf_page_no)
        mv = memoryview(page_data)
        # Read without pager: keys are always inline; payload bytes returned
        # for spilled cells are the raw 4-byte overflow pointer (not the full
        # payload).  We use these only for key comparisons and raw-copy rewrites.
        cells = read_all_leaf_cells(mv)

        # Check whether this is a replace (key already present at insert_idx).
        if insert_idx < len(cells) and cells[insert_idx][0] == key:
            # Read the old declared payload length directly from the page binary
            # to decide whether the existing cell is spilled.
            (old_cell_ptr,) = struct.unpack_from(">H", mv, _LEAF_PTR_ARRAY_START + insert_idx * 2)
            _old_key, _okn = decode_varint(mv, old_cell_ptr)
            old_payload_len, _ovn = decode_varint(mv, old_cell_ptr + _okn)

            if old_payload_len > OVERFLOW_THRESHOLD:
                # Free the old overflow chain before rewriting the cell.
                _ovf_offset = old_cell_ptr + _okn + _ovn
                (old_first_ovf,) = struct.unpack_from(">I", mv, _ovf_offset)
                from iai_mcp.lillibrain.btree.overflow import free_overflow_chain as _foc  # noqa: PLC0415
                _foc(self._pager, self._pager._journal, old_first_ovf)

            # Check room using declared lengths from the page binary (exclude_idx
            # skips the replaced cell so only the remaining cells are measured).
            if _leaf_has_room_for(mv, len(cells), insert_idx, key, value):
                # In-place replace: rebuild page raw-copying all unchanged cells
                # and writing only the replaced cell fresh (re-spilling if needed).
                sibling = struct.unpack_from(">I", page_data, 8)[0]
                _write_leaf_replacing_one(
                    self._pager, self._pager._journal, leaf_page_no,
                    mv, len(cells), insert_idx, key, value, sibling
                )
                logger.debug(
                    "BtreeCursor.insert (replace): key=%d leaf=%d", key, leaf_page_no
                )
                return
            # Replacement value is too large for the page: slim the page by
            # removing the old cell (chain already freed above), then fall
            # through to the split path.
            if self._pager._journal is None and not self._pager._wal_mode:
                raise RuntimeError(
                    "BtreeCursor.insert (replace-grow): no write transaction open"
                )
            # Rebuild page without the replaced cell (raw-copy remaining cells).
            sibling_link = struct.unpack_from(">I", page_data, 8)[0]
            slimmed_page = _raw_copy_leaf_excluding(mv, len(cells), insert_idx, sibling_link)
            self._pager.write_page(leaf_page_no, slimmed_page)
            page_data = slimmed_page
            mv = memoryview(page_data)
            cells = read_all_leaf_cells(mv)
            # fall through to split dispatch below

        if _leaf_has_room_for(mv, len(cells), None, key, value) and len(cells) < LEAF_MAX_CELLS:
            # Byte capacity is the governing gate; LEAF_MAX_CELLS is a secondary ceiling
            # that allows tests to inject an artificially small cap so split/merge paths
            # fire without requiring large payloads.
            #
            # New insertion: append the new cell using write_leaf_cells with pager
            # so oversized payloads spill correctly.  Existing cells are raw-copied
            # via the combined write approach.
            sibling = struct.unpack_from(">I", page_data, 8)[0]
            _write_leaf_inserting_one(
                self._pager, self._pager._journal, leaf_page_no,
                mv, len(cells), insert_idx, key, value, sibling
            )
            logger.debug(
                "BtreeCursor.insert: key=%d leaf=%d cells=%d", key, leaf_page_no, len(cells) + 1
            )
            return

        # Leaf is full — dispatch to the split code.
        if self._pager._journal is None and not self._pager._wal_mode:
            raise RuntimeError(
                "BtreeCursor.insert: leaf is full but no write transaction is open; "
                "call pager.begin_write() before inserting when splits may occur"
            )
        # In WAL mode there is no journal; pass None so freelist helpers skip
        # before-image recording (WAL rollback uses checksum-chain rewind instead).
        journal = self._pager._journal  # None in WAL mode

        from iai_mcp.lillibrain.btree.split import leaf_split_and_insert
        leaf_split_and_insert(
            self._pager,
            journal,
            list(self._path_stack),
            leaf_page_no,
            key,
            value,
        )
        logger.debug(
            "BtreeCursor.insert (split): key=%d leaf=%d", key, leaf_page_no
        )

    # ---------------------------------------------------------------------------
    # Sequential scan
    # ---------------------------------------------------------------------------

    def next(self) -> tuple[int, bytes] | None:
        """Return the next (key, value) cell in key-ascending order.

        On the first call, starts from the leftmost leaf and follows the sibling
        chain. Returns None when all cells have been visited.

        Reinitialise by calling move_to() or full_scan().
        """
        # Initialise the scan position lazily on first call.
        if self._scan_page_no is None:
            self._scan_page_no = self._leftmost_leaf()
            self._scan_cells = read_all_leaf_cells(
                memoryview(self._pager.read_page(self._scan_page_no)),
                pager=self._pager,
            )
            self._scan_cell_idx = 0

        while True:
            if self._scan_cell_idx < len(self._scan_cells):
                cell = self._scan_cells[self._scan_cell_idx]
                self._scan_cell_idx += 1
                return cell

            # Exhausted current leaf — follow sibling link.
            sibling = get_next_leaf(self._pager, self._scan_page_no)
            if sibling == 0:
                # End of chain.
                self._scan_page_no = None
                return None

            self._scan_page_no = sibling
            self._scan_cells = read_all_leaf_cells(
                memoryview(self._pager.read_page(sibling)),
                pager=self._pager,
            )
            self._scan_cell_idx = 0

    def delete(self, key: int) -> None:
        """Remove the cell for key from the B+tree.

        If key is not present, this is a no-op (idempotent). The caller must
        hold an open write transaction (pager.begin_write()) before calling
        delete, because freed pages are journaled via the freelist.

        After removing the cell from the leaf, if the leaf falls below the
        minimum occupancy (LEAF_MIN_CELLS) and is not the root, _rebalance_leaf
        is called to borrow from or merge with a sibling. After any rebalance,
        _shrink_root_if_needed collapses an interior root with zero keys.

        Raises RuntimeError if called without an open write transaction.
        """
        from iai_mcp.lillibrain.btree.delete import (
            LEAF_MIN_CELLS,
            _rebalance_leaf,
            _shrink_root_if_needed,
            remove_cell_from_leaf,
        )

        if self._pager._journal is None and not self._pager._wal_mode:
            raise RuntimeError(
                "BtreeCursor.delete: no write transaction is open; "
                "call pager.begin_write() before delete"
            )
        journal = self._pager._journal  # None in WAL mode

        leaf_page_no, _insert_idx = self.move_to(key)
        removed = remove_cell_from_leaf(self._pager, leaf_page_no, key)

        if not removed:
            # Key was absent — nothing to rebalance.
            logger.debug("BtreeCursor.delete: key=%d not found (no-op)", key)
            return

        logger.debug("BtreeCursor.delete: key=%d removed from leaf=%d", key, leaf_page_no)

        # Check whether the leaf underflows. Root leaf is exempt.
        is_root_leaf = (leaf_page_no == self._root_page_no)
        if not is_root_leaf:
            page_data = self._pager.read_page(leaf_page_no)
            page_mv = memoryview(page_data)
            from iai_mcp.lillibrain.btree.page import read_all_leaf_cells
            cells = read_all_leaf_cells(page_mv)

            # Byte-driven underflow governs the default cap: a leaf below the
            # half-page fill threshold (or empty) rebalances. The count floor
            # fires only under a small injected cap (LEAF_MAX_CELLS below the
            # default ceiling) so small-cap merge/borrow tests still exercise
            # those paths; at the default cap the count floor would flag every
            # real leaf, so it is gated off there.
            underflow = _leaf_underflows(page_mv, len(cells))
            if not underflow and LEAF_MAX_CELLS < _SMALL_CAP_SENTINEL:
                underflow = len(cells) < LEAF_MIN_CELLS

            if underflow:
                _rebalance_leaf(
                    self._pager,
                    journal,
                    list(self._path_stack),
                    leaf_page_no,
                )

        # Collapse an interior root with zero keys (tree height shrinks).
        _shrink_root_if_needed(self._pager, journal)

    def full_scan(self) -> list[tuple[int, bytes]]:
        """Return all (key, value) pairs in key-ascending order.

        Traverses the entire leaf chain from the leftmost leaf. Resets the
        scan state for subsequent next() calls.
        """
        # Reset scan state.
        self._scan_page_no = None
        self._scan_cells = []
        self._scan_cell_idx = 0

        result: list[tuple[int, bytes]] = []
        item = self.next()
        while item is not None:
            result.append(item)
            item = self.next()

        # Reset scan state again so a subsequent next() restarts cleanly.
        self._scan_page_no = None
        self._scan_cells = []
        self._scan_cell_idx = 0
        return result

    def scan_column_views(
        self, col_index: int
    ) -> "collections.abc.Generator[tuple[int, memoryview | bytes], None, None]":
        """Yield (key, column_slice) for every row, traversing the leaf chain.

        For each leaf page the page buffer is obtained from the pager (a mapped
        memoryview when the mapped read path is ON).  Each row is decoded with
        decode_record_views so BLOB columns at *col_index* are returned as a
        zero-copy memoryview slice into that page buffer for inline cells.  For
        cells whose payload spilled to an overflow chain the reassembled bytes
        object is returned instead (a copy, but correct).

        The consumer must materialise each yielded view before calling next()
        on this generator, because the view aliases the page buffer directly —
        it is invalid once the page mapping is refreshed (next commit, checkpoint,
        or file-grow).

        TEXT, integer, and float columns at *col_index* are returned as their
        decoded Python values (str, int, float) regardless of zero-copy status.

        This scan does not alter the cursor's next() position: the scan-state
        fields (_scan_page_no, _scan_cells, _scan_cell_idx) are restored to their
        pre-call values after the generator is exhausted or closed.
        """
        import collections.abc  # noqa: PLC0415

        from iai_mcp.lillibrain.btree.overflow import read_overflow_chain  # noqa: PLC0415

        saved = None
        try:
            # Save and reset scan state inside the try so the finally block can
            # restore it unconditionally via the `if saved is not None` guard,
            # preventing UnboundLocalError if an exception occurs before the
            # assignment completes.
            saved = (self._scan_page_no, self._scan_cells[:], self._scan_cell_idx)
            self._scan_page_no = None
            self._scan_cells = []
            self._scan_cell_idx = 0

            page_no = self._leftmost_leaf()
            while page_no != 0:
                page_buf = self._pager.read_page(page_no)
                # Wrap as memoryview for decode_record_views regardless of whether
                # the pager returned a bytearray or a mapped slice — both support
                # the memoryview() constructor.
                page_mv = memoryview(page_buf)

                # Walk the cell pointer array directly so we can pass the page
                # memoryview to decode_record_views for each cell.
                from iai_mcp.lillibrain.btree.page import read_leaf_header  # noqa: PLC0415

                num_cells, _ccs, _fb, _frag = read_leaf_header(page_mv)
                for i in range(num_cells):
                    (ptr,) = struct.unpack_from(">H", page_mv, _LEAF_PTR_ARRAY_START + i * 2)
                    key, kn = decode_varint(page_mv, ptr)
                    payload_len, vn = decode_varint(page_mv, ptr + kn)
                    payload_start = ptr + kn + vn

                    if payload_len > OVERFLOW_THRESHOLD:
                        # Spilled cell: reassemble from overflow chain — cannot be
                        # a single contiguous page slice, so return a bytes copy.
                        (first_ovf,) = struct.unpack_from(">I", page_mv, payload_start)
                        payload_bytes = read_overflow_chain(
                            self._pager, first_ovf, payload_len
                        )
                        row_vals, _ = decode_record_views(payload_bytes, 0)
                    else:
                        # Inline cell: decode the payload slice as a memoryview so
                        # BLOB columns are returned as zero-copy page slices.
                        payload_mv = page_mv[payload_start : payload_start + payload_len]
                        row_vals, _ = decode_record_views(payload_mv, 0)

                    col_val = row_vals[col_index] if col_index < len(row_vals) else None
                    yield key, col_val  # type: ignore[misc]

                # Follow the sibling link.
                from iai_mcp.lillibrain.btree.page import get_next_leaf  # noqa: PLC0415

                page_no = get_next_leaf(self._pager, page_no)
        finally:
            # Restore scan state if the save completed; guards against
            # UnboundLocalError when an exception fires before `saved` is set.
            if saved is not None:
                self._scan_page_no, self._scan_cells, self._scan_cell_idx = saved
