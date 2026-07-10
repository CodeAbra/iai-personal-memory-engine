"""Overflow page chain: allocate, reassemble, and free multi-page payload chains.

When a payload exceeds the spill threshold (PAGE_SIZE - 35), its entire content
is moved to a chain of overflow pages.  The leaf cell stores only the key varint,
the total-length varint, and a four-byte big-endian first-overflow-page pointer.

Overflow page binary layout:
  Offset  Size  Field
  0       4     next_overflow_page  uint32 BE; 0 = last page in chain
  4       P-4   payload_bytes       raw payload bytes (P = PAGE_SIZE)

All freelist mutations go through the existing journaled alloc_page / free_page
helpers, so overflow page allocations and frees are crash-safe with no additional
before-image logic needed beyond what the freelist already provides.
"""
from __future__ import annotations

import math
import struct
from typing import TYPE_CHECKING

from iai_mcp.lillibrain.constants import (
    OVERFLOW_DATA_OFFSET,
    OVERFLOW_NEXT_OFFSET,
    OVERFLOW_THRESHOLD,
    PAGE_SIZE,
)
from iai_mcp.lillibrain.freelist import alloc_page, free_page

if TYPE_CHECKING:
    from iai_mcp.lillibrain.pager import Journal, Pager

# Usable payload bytes per overflow page (the rest is the next-pointer).
_OVERFLOW_USABLE: int = PAGE_SIZE - OVERFLOW_DATA_OFFSET


# ---------------------------------------------------------------------------
# Error classes
# ---------------------------------------------------------------------------


class OverflowChainError(RuntimeError):
    """Raised when an overflow chain is structurally invalid during reading."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def alloc_overflow_chain(
    pager: Pager,
    journal: Journal | None,
    payload: bytes,
) -> int:
    """Allocate overflow pages for payload and return the first page number.

    Allocates ceil(len(payload) / (PAGE_SIZE - 4)) pages from the freelist.
    Each page contains a uint32 BE next-page pointer (0 on the last page)
    followed by up to PAGE_SIZE - 4 payload bytes.

    All freelist mutations are journaled via the supplied journal (None in
    WAL mode, where WAL rollback handles recovery instead).

    Returns the first overflow page number, which is stored in the leaf cell.
    Raises ValueError if payload is empty.
    """
    if not payload:
        raise ValueError("alloc_overflow_chain: payload must be non-empty")

    page_count = math.ceil(len(payload) / _OVERFLOW_USABLE)

    # First pass: allocate all pages so we know their numbers before writing.
    page_nos: list[int] = []
    for _ in range(page_count):
        pno = alloc_page(pager, journal=journal)
        page_nos.append(pno)

    first_page_no = page_nos[0]
    assert first_page_no != 0, "alloc_overflow_chain: freelist returned page 0"

    # Second pass: write each page with its next-pointer and payload slice.
    for i, pno in enumerate(page_nos):
        next_pno = page_nos[i + 1] if i + 1 < page_count else 0
        chunk_start = i * _OVERFLOW_USABLE
        chunk = payload[chunk_start : chunk_start + _OVERFLOW_USABLE]

        buf = bytearray(PAGE_SIZE)
        struct.pack_into(">I", buf, OVERFLOW_NEXT_OFFSET, next_pno)
        buf[OVERFLOW_DATA_OFFSET : OVERFLOW_DATA_OFFSET + len(chunk)] = chunk
        pager.write_page(pno, buf)

    return first_page_no


def read_overflow_chain(
    pager: Pager,
    first_overflow_page: int,
    total_len: int,
) -> bytes:
    """Reassemble the payload from an overflow chain.

    Follows next-page pointers starting at first_overflow_page until the
    pointer is zero, collecting payload bytes from each page.  The final
    page is sliced so that exactly total_len bytes are returned.

    Raises OverflowChainError if:
      - a page number in the chain is out of range (< 1 or > db_size)
      - the chain ends before total_len bytes are collected
      - a page number repeats (cycle in the chain)
    """
    db_size = pager.read_db_header_field("db_size")
    chunks: list[bytes] = []
    collected = 0
    current = first_overflow_page
    seen: set[int] = set()

    while current != 0:
        if current in seen:
            raise OverflowChainError(
                f"overflow chain cycle detected at page {current}"
            )
        if current < 1 or current > db_size:
            raise OverflowChainError(
                f"overflow chain: page number {current} out of range "
                f"(db_size={db_size})"
            )
        seen.add(current)

        page_data = pager.read_page(current)
        (next_pg,) = struct.unpack_from(">I", page_data, OVERFLOW_NEXT_OFFSET)

        chunk = bytes(page_data[OVERFLOW_DATA_OFFSET:])
        remaining = total_len - collected
        if next_pg == 0:
            # Last page — slice to exact remaining byte count.
            chunk = chunk[:remaining]
        chunks.append(chunk)
        collected += len(chunk)
        current = next_pg

    if collected < total_len:
        raise OverflowChainError(
            f"overflow chain starting at page {first_overflow_page}: "
            f"collected {collected} bytes but declared total_len={total_len}"
        )

    result = b"".join(chunks)
    assert len(result) == total_len, (
        f"overflow reassembly: got {len(result)} bytes, expected {total_len}"
    )
    return result


def free_overflow_chain(
    pager: Pager,
    journal: Journal | None,
    first_overflow_page: int,
) -> None:
    """Free all overflow pages in the chain rooted at first_overflow_page.

    Reads each page's next-pointer, frees the page through the journaled
    freelist helper, then advances.  Terminates when the next-pointer is zero.

    Raises OverflowChainError if a page number repeats in the chain (cycle),
    guarding against unbounded loops on corrupt storage.

    Must be called inside the same write transaction as the leaf cell removal
    so the chain free and the leaf pointer update commit or roll back together.
    """
    current = first_overflow_page
    seen: set[int] = set()
    while current != 0:
        if current in seen:
            raise OverflowChainError(
                f"overflow chain cycle during free at page {current}"
            )
        seen.add(current)
        page_data = pager.read_page(current)
        (next_pg,) = struct.unpack_from(">I", page_data, OVERFLOW_NEXT_OFFSET)
        free_page(pager, current, journal=journal)
        current = next_pg
