"""B+tree page layout: header encode/decode, cell pointer array, and cell I/O.

Implements the binary page format used by the LilliBrain B+tree. All mutations
operate on a mutable bytearray via struct.pack_into; all reads accept a memoryview
to avoid copying page buffers on the hot path.

Layout summary
--------------
Leaf page (8-byte header):
  [0]     page_type byte      = PAGE_LEAF_TABLE (0x0D)
  [1:3]   first_freeblock     uint16 BE
  [3:5]   num_cells           uint16 BE
  [5:7]   cell_content_start  uint16 BE
  [7]     fragmented_bytes    uint8
  [8:]    cell pointer array  (num_cells * 2 bytes, each uint16 BE offset)
  [8 + LEAF_SIBLING_OFFSET : +4]  next_leaf sibling link (uint32 BE)

  Cell format: encode_varint(rowid) + encode_varint(payload_len) + payload
  Cells are written from the end of the page downward; cell_content_start is
  the lowest byte offset used by cell content.

Interior page (12-byte header):
  [0]     page_type byte      = PAGE_INTERIOR_TABLE (0x05)
  [1:3]   first_freeblock     uint16 BE  (unused; always 0 written here)
  [3:5]   num_cells           uint16 BE
  [5:7]   cell_content_start  uint16 BE
  [7]     fragmented_bytes    uint8       (unused; always 0)
  [8:12]  rightmost_child     uint32 BE
  [12:]   cell pointer array

  Interior cell format: uint32 BE (left_child) + encode_varint(key)
  Rightmost child lives in the header (index len(keys), i.e. one more than
  the number of interior cells).

Sibling link
------------
The leaf sibling pointer is stored at a fixed offset in the header area — the
8 bytes after LEAF_HEADER_SIZE in the first cell-pointer slot that is reserved
for this purpose. Specifically, 4 bytes are reserved at offset
LEAF_HEADER_SIZE (bytes 8–11) of every leaf page for the next-sibling page
number (uint32 BE, 0 = no sibling). The cell pointer array starts at offset
LEAF_HEADER_SIZE + 4 = 12. The fixed 4-byte reservation does not constrain
page capacity because it occupies space that would otherwise be unused header
padding.
"""
from __future__ import annotations

import struct

import struct as _struct

from iai_mcp.lillibrain.constants import (
    INTERIOR_HEADER_SIZE,
    LEAF_HEADER_SIZE,
    OVERFLOW_DATA_OFFSET,
    OVERFLOW_THRESHOLD,
    PAGE_INTERIOR_TABLE,
    PAGE_LEAF_TABLE,
    PAGE_SIZE,
)
from iai_mcp.lillibrain.record import decode_varint, encode_varint

# ---------------------------------------------------------------------------
# Internal layout constants
# ---------------------------------------------------------------------------

# Leaf sibling link: 4 bytes at LEAF_HEADER_SIZE (bytes 8–11).
# The cell pointer array starts at byte 12 (LEAF_HEADER_SIZE + 4).
_LEAF_SIBLING_OFFSET = LEAF_HEADER_SIZE          # uint32 BE at bytes [8:12]
_LEAF_PTR_ARRAY_START = LEAF_HEADER_SIZE + 4     # 12

# Interior cell pointer array starts right after the 12-byte header.
_INTERIOR_PTR_ARRAY_START = INTERIOR_HEADER_SIZE  # 12

# Sizes of cell format components
_INTERIOR_CHILD_SIZE = 4   # uint32 BE left-child pointer


# ---------------------------------------------------------------------------
# Leaf page
# ---------------------------------------------------------------------------


def write_leaf_header(
    page: bytearray,
    num_cells: int,
    cell_content_start: int,
    first_freeblock: int = 0,
    fragmented_bytes: int = 0,
) -> None:
    """Write the 8-byte leaf page header into page at offset 0.

    Fields written (all big-endian):
      [0]   page_type = PAGE_LEAF_TABLE
      [1:3] first_freeblock
      [3:5] num_cells
      [5:7] cell_content_start
      [7]   fragmented_bytes
    """
    struct.pack_into(
        ">B HHH B",
        page,
        0,
        PAGE_LEAF_TABLE,
        first_freeblock,
        num_cells,
        cell_content_start,
        fragmented_bytes,
    )


def read_leaf_header(page: memoryview) -> tuple[int, int, int, int]:
    """Decode the 8-byte leaf page header from page.

    Returns (num_cells, cell_content_start, first_freeblock, fragmented_bytes).
    """
    _page_type, first_fb, num_cells, ccs, frag = struct.unpack_from(">B HHH B", page, 0)
    return num_cells, ccs, first_fb, frag


# ---------------------------------------------------------------------------
# Interior page
# ---------------------------------------------------------------------------


def write_interior_header(
    page: bytearray,
    num_cells: int,
    cell_content_start: int,
    rightmost_child: int,
) -> None:
    """Write the 12-byte interior page header into page at offset 0.

    Fields written (all big-endian):
      [0]    page_type = PAGE_INTERIOR_TABLE
      [1:3]  first_freeblock = 0
      [3:5]  num_cells
      [5:7]  cell_content_start
      [7]    fragmented_bytes = 0
      [8:12] rightmost_child
    """
    struct.pack_into(
        ">B HHH B I",
        page,
        0,
        PAGE_INTERIOR_TABLE,
        0,            # first_freeblock (unused)
        num_cells,
        cell_content_start,
        0,            # fragmented_bytes (unused)
        rightmost_child,
    )


def read_interior_header(page: memoryview) -> tuple[int, int, int]:
    """Decode the 12-byte interior page header from page.

    Returns (num_cells, cell_content_start, rightmost_child).
    """
    _page_type, _first_fb, num_cells, ccs, _frag, rightmost = struct.unpack_from(
        ">B HHH B I", page, 0
    )
    return num_cells, ccs, rightmost


# ---------------------------------------------------------------------------
# Cell pointer array
# ---------------------------------------------------------------------------


def insert_cell_ptr(
    page: bytearray,
    header_size: int,
    key_order_index: int,
    num_cells_before: int,
    cell_offset: int,
) -> None:
    """Insert a 2-byte big-endian cell pointer at key_order_index.

    The pointer array is located at page[header_size : header_size + num_cells_before*2].
    Existing pointers at and after key_order_index are shifted right by 2 bytes
    to make room, then cell_offset is written at the new slot.

    This function writes directly into the mutable bytearray; callers are
    responsible for updating num_cells in the page header afterward.
    """
    # Where the new pointer slot starts
    ptr_array_start = header_size
    insert_at = ptr_array_start + key_order_index * 2
    tail_len = (num_cells_before - key_order_index) * 2

    if tail_len > 0:
        # Shift the tail right by 2 bytes in place (non-overlapping with src when shifting right).
        page[insert_at + 2 : insert_at + 2 + tail_len] = page[insert_at : insert_at + tail_len]

    struct.pack_into(">H", page, insert_at, cell_offset)


# ---------------------------------------------------------------------------
# Leaf cell I/O
# ---------------------------------------------------------------------------


def write_leaf_cells(
    page: bytearray,
    cells: list[tuple[int, bytes]],
    *,
    pager: object = None,
    journal: object = None,
) -> None:
    """Write a sorted list of (key, payload) cells into a leaf page.

    Cells are laid out from the end of the page downward. The pointer array
    starts at _LEAF_PTR_ARRAY_START (= LEAF_HEADER_SIZE + 4), with pointers
    in key-ascending order. The header is updated with num_cells and
    cell_content_start. The sibling-link slot (bytes 8–11) is preserved.

    Overflow spilling (optional):
      When pager and journal are supplied, any cell whose payload length exceeds
      OVERFLOW_THRESHOLD is spilled entirely to a freelist-allocated overflow
      chain.  The on-page cell then contains only the key varint, the
      total-length varint (full payload size), and a four-byte big-endian
      first-overflow-page pointer with zero payload bytes on the leaf.

      When pager is None (the default), every cell is written inline exactly
      as before.  All existing call sites that omit pager/journal continue to
      work unchanged.
    """
    from iai_mcp.lillibrain.btree.overflow import alloc_overflow_chain  # noqa: PLC0415

    write_pos = PAGE_SIZE
    ptrs: list[int] = []

    for key, payload in cells:
        key_enc = encode_varint(key)

        if pager is not None and len(payload) > OVERFLOW_THRESHOLD:
            # Spill: allocate chain, store pointer in the cell (no on-page payload).
            first_ovf = alloc_overflow_chain(pager, journal, payload)  # type: ignore[arg-type]
            payload_len_enc = encode_varint(len(payload))
            cell = key_enc + payload_len_enc + _struct.pack(">I", first_ovf)
        else:
            # Inline: key varint + payload-length varint + payload bytes.
            payload_enc = encode_varint(len(payload))
            cell = key_enc + payload_enc + payload

        write_pos -= len(cell)
        page[write_pos : write_pos + len(cell)] = cell
        ptrs.append(write_pos)

    # Write pointer array starting at _LEAF_PTR_ARRAY_START
    for i, ptr in enumerate(ptrs):
        struct.pack_into(">H", page, _LEAF_PTR_ARRAY_START + i * 2, ptr)

    cell_content_start = write_pos if cells else PAGE_SIZE
    write_leaf_header(page, num_cells=len(cells), cell_content_start=cell_content_start)


def read_all_leaf_cells(
    page: memoryview,
    pager: object = None,
) -> list[tuple[int, bytes]]:
    """Read all (key, payload) cells from a leaf page in key order.

    Follows the cell pointer array from _LEAF_PTR_ARRAY_START.

    When pager is supplied, spilled cells (payload_len > OVERFLOW_THRESHOLD)
    are transparently reassembled from their overflow chain so callers always
    receive the complete (key, payload) pair regardless of storage strategy.
    When pager is None, spilled cells return only the on-page bytes (four-byte
    overflow pointer), which is sufficient for structural operations that do not
    need the payload content (e.g. check_integrity key-order scan, split key
    extraction).
    """
    from iai_mcp.lillibrain.btree.overflow import read_overflow_chain  # noqa: PLC0415

    num_cells, _ccs, _fb, _frag = read_leaf_header(page)
    result: list[tuple[int, bytes]] = []

    for i in range(num_cells):
        (ptr,) = struct.unpack_from(">H", page, _LEAF_PTR_ARRAY_START + i * 2)
        key, n = decode_varint(page, ptr)
        payload_len, m = decode_varint(page, ptr + n)
        payload_start = ptr + n + m

        if payload_len > OVERFLOW_THRESHOLD:
            # Spilled cell: four-byte first-overflow-page pointer follows the varints.
            (first_ovf,) = struct.unpack_from(">I", page, payload_start)
            if pager is not None:
                payload = read_overflow_chain(pager, first_ovf, payload_len)  # type: ignore[arg-type]
            else:
                # No pager: return the raw four-byte pointer as the payload.
                # Structural callers (key extraction, integrity key-order checks)
                # only need the key; they do not dereference the payload.
                payload = bytes(page[payload_start : payload_start + 4])
        else:
            payload = bytes(page[payload_start : payload_start + payload_len])

        result.append((key, payload))

    return result


# ---------------------------------------------------------------------------
# Interior node I/O
# ---------------------------------------------------------------------------


def write_interior_node(
    page: bytearray,
    keys: list[int],
    children: list[int],
) -> None:
    """Write an interior node's keys and children pointers into a page.

    Requires len(children) == len(keys) + 1. The rightmost child is stored
    in the page header; each remaining child-separator pair is written as an
    interior cell: uint32 BE (left_child) + varint(key).

    Cells are laid out from the end of the page downward; the pointer array
    follows the 12-byte header.
    """
    if len(children) != len(keys) + 1:
        raise ValueError(
            f"write_interior_node: len(children)={len(children)} must equal "
            f"len(keys)+1={len(keys)+1}"
        )

    # Rightmost child goes into the header (last entry in children list).
    rightmost = children[-1]

    write_pos = PAGE_SIZE
    ptrs: list[int] = []

    # Each interior cell: uint32 left_child + varint key.
    # children[i] is the left child of keys[i]; children[-1] is rightmost.
    for i, key in enumerate(keys):
        left_child = children[i]
        key_enc = encode_varint(key)
        cell = struct.pack(">I", left_child) + key_enc
        write_pos -= len(cell)
        page[write_pos : write_pos + len(cell)] = cell
        ptrs.append(write_pos)

    # Write pointer array starting at _INTERIOR_PTR_ARRAY_START
    for i, ptr in enumerate(ptrs):
        struct.pack_into(">H", page, _INTERIOR_PTR_ARRAY_START + i * 2, ptr)

    cell_content_start = write_pos if keys else PAGE_SIZE
    write_interior_header(
        page,
        num_cells=len(keys),
        cell_content_start=cell_content_start,
        rightmost_child=rightmost,
    )


def read_interior_node(page: memoryview) -> tuple[list[int], list[int]]:
    """Read the (keys, children) from an interior page.

    Returns (keys, children) where len(children) == len(keys) + 1.
    The rightmost child comes from the page header; others from the cell data.
    """
    num_cells, _ccs, rightmost = read_interior_header(page)
    keys: list[int] = []
    children: list[int] = []

    for i in range(num_cells):
        (ptr,) = struct.unpack_from(">H", page, _INTERIOR_PTR_ARRAY_START + i * 2)
        (left_child,) = struct.unpack_from(">I", page, ptr)
        key, _n = decode_varint(page, ptr + _INTERIOR_CHILD_SIZE)
        children.append(left_child)
        keys.append(key)

    children.append(rightmost)
    return keys, children


# ---------------------------------------------------------------------------
# Leaf sibling link
# ---------------------------------------------------------------------------


def get_next_leaf(pager: object, page_no: int) -> int:
    """Return the sibling page number stored in the leaf at page_no.

    Returns 0 if there is no sibling (end of the leaf chain).
    The sibling link is stored as a uint32 BE at bytes [8:12] of the page
    (immediately after the 8-byte leaf header).
    """
    page = pager.read_page(page_no)  # type: ignore[attr-defined]
    (sibling,) = struct.unpack_from(">I", page, _LEAF_SIBLING_OFFSET)
    return sibling


def set_next_leaf(pager: object, page_no: int, sibling_no: int) -> None:
    """Write sibling_no as the next-leaf pointer of the leaf at page_no.

    The update is written into the page via pager.write_page so it enters
    the dirty set and is committed with the surrounding transaction.
    """
    page = bytearray(pager.read_page(page_no))  # type: ignore[attr-defined]
    struct.pack_into(">I", page, _LEAF_SIBLING_OFFSET, sibling_no)
    pager.write_page(page_no, page)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Page type query
# ---------------------------------------------------------------------------


def page_type(page: memoryview) -> int:
    """Return the page-type byte from the first byte of page.

    Returns PAGE_LEAF_TABLE or PAGE_INTERIOR_TABLE.
    Raises ValueError for any other value.
    """
    (ptype,) = struct.unpack_from(">B", page, 0)
    if ptype not in (PAGE_LEAF_TABLE, PAGE_INTERIOR_TABLE):
        raise ValueError(
            f"unknown page type byte {ptype:#04x}; "
            f"expected {PAGE_LEAF_TABLE:#04x} (leaf) or {PAGE_INTERIOR_TABLE:#04x} (interior)"
        )
    return ptype
