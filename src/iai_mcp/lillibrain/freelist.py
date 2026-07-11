"""Freelist trunk page, page allocation, and page freeing for the LilliBrain engine.

Freelist trunk pages are journaled as before-images before any allocation or
free operation mutates them, ensuring that an aborted transaction fully restores
the freelist state (no page leaks, no double-frees).

Invariants enforced here:
  Trunk page before-image recorded before any alloc mutates the trunk.
  Trunk page before-image recorded before free_page mutates the trunk.
  DB header page (page 1) before-image recorded alongside trunk on the alloc path.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iai_mcp.lillibrain.pager import Journal, Pager

# ---------------------------------------------------------------------------
# Trunk page binary layout (big-endian, one page)
# ---------------------------------------------------------------------------
#
#  Offset  Size  Description
#  ------  ----  -----------
#   0       4    next_trunk  (0 = end of chain)
#   4       4    leaf_count  (number of free leaf page numbers stored here)
#   8+      4*N  leaf_page_no[0..N-1]
#
# Max leaf entries per trunk page:
#   (page_size - 8) // 4
#
_TRUNK_FIXED_SIZE = 8       # next_trunk(4) + leaf_count(4)
_TRUNK_ENTRY_SIZE = 4       # each leaf page number


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class FreelistTrunkPage:
    """Decoded content of a freelist trunk page.

    next_trunk: page number of the next trunk (0 = end of chain).
    leaf_pages: list of free leaf page numbers stored in this trunk.
    """

    next_trunk: int
    leaf_pages: list[int] = field(default_factory=list)

    @classmethod
    def decode(cls, data: bytes | bytearray | memoryview) -> FreelistTrunkPage:
        """Decode a raw page buffer into a FreelistTrunkPage.

        Reads next_trunk and leaf_count from the fixed header, then reads
        leaf_count page numbers from the variable-length body.
        """
        (next_trunk,) = struct.unpack_from(">I", data, 0)
        (leaf_count,) = struct.unpack_from(">I", data, 4)
        leaf_pages: list[int] = []
        for i in range(leaf_count):
            offset = _TRUNK_FIXED_SIZE + i * _TRUNK_ENTRY_SIZE
            (pno,) = struct.unpack_from(">I", data, offset)
            leaf_pages.append(pno)
        return cls(next_trunk=next_trunk, leaf_pages=leaf_pages)

    def encode(self, page_size: int) -> bytes:
        """Encode this trunk page into a byte string of exactly page_size bytes.

        Raises ValueError if the leaf_pages list exceeds the maximum capacity
        for the given page size.
        """
        max_leaves = (page_size - _TRUNK_FIXED_SIZE) // _TRUNK_ENTRY_SIZE
        if len(self.leaf_pages) > max_leaves:
            raise ValueError(
                f"FreelistTrunkPage: {len(self.leaf_pages)} leaf entries exceed "
                f"max {max_leaves} for page_size={page_size}"
            )
        buf = bytearray(page_size)
        struct.pack_into(">I", buf, 0, self.next_trunk)
        struct.pack_into(">I", buf, 4, len(self.leaf_pages))
        for i, pno in enumerate(self.leaf_pages):
            struct.pack_into(">I", buf, _TRUNK_FIXED_SIZE + i * _TRUNK_ENTRY_SIZE, pno)
        return bytes(buf)


# ---------------------------------------------------------------------------
# Corruption error
# ---------------------------------------------------------------------------


class FreelistCorruptionError(RuntimeError):
    """Freelist trunk chain is internally inconsistent (cycle or invalid page number)."""


# ---------------------------------------------------------------------------
# Freelist alloc/free — trunk-journaling invariants
# ---------------------------------------------------------------------------


def alloc_page(pager: Pager, *, journal: Journal | None) -> int:
    """Allocate one page number, preferring the freelist over file extension.

    In rollback-journal mode, the freelist trunk page and DB header page
    (page 1) are recorded as before-images before any in-memory modification,
    so that an aborted transaction can fully restore the freelist state.
    In write-ahead mode, journal is None and before-image recording is skipped
    (WAL rollback rewinds via the checksum-chain snapshot, not before-images).

    Returns the allocated page number (1-indexed).
    """
    first_trunk = pager.read_db_header_field("first_freelist_trunk")

    if first_trunk == 0:
        # No freelist — extend the file (no trunk to journal)
        return pager.extend_file()

    # Journal the trunk page before-image before any mutation (rollback mode only).
    trunk_data = pager.read_page(first_trunk)
    if journal is not None:
        journal.record_before_image(first_trunk, trunk_data)

    # Journal the DB header page (page 1 holds freelist_count and
    # first_freelist_trunk) before any freelist mutation (rollback mode only).
    if journal is not None:
        journal.record_before_image(1, pager.read_page(1))

    trunk = FreelistTrunkPage.decode(trunk_data)

    if trunk.leaf_pages:
        # Pop a leaf page from the trunk
        allocated = trunk.leaf_pages.pop()
        pager.write_page(first_trunk, trunk.encode(pager._page_size))
        freelist_count = pager.read_db_header_field("freelist_count")
        pager.update_db_header("freelist_count", max(0, freelist_count - 1))
        return allocated
    else:
        # Trunk leaf list is empty — pop the trunk page itself, advance chain
        next_trunk = trunk.next_trunk
        pager.update_db_header("first_freelist_trunk", next_trunk)
        freelist_count = pager.read_db_header_field("freelist_count")
        pager.update_db_header("freelist_count", max(0, freelist_count - 1))
        return first_trunk


def free_page(pager: Pager, page_no: int, *, journal: Journal | None) -> None:
    """Push page_no onto the freelist trunk and increment freelist_count.

    In rollback-journal mode, the current trunk page is recorded as a
    before-image before any mutation so that an aborted transaction can restore
    the trunk state.  In write-ahead mode, journal is None and before-image
    recording is skipped (WAL rollback rewinds via the checksum-chain snapshot).

    If the current trunk page is full or no trunk exists, a new trunk page
    is allocated to hold the freed page.
    """
    first_trunk = pager.read_db_header_field("first_freelist_trunk")
    page_size = pager._page_size
    max_leaves = (page_size - _TRUNK_FIXED_SIZE) // _TRUNK_ENTRY_SIZE

    # Journal the DB header page (page 1) before any mutation (rollback mode only).
    header_data = pager.read_page(1)
    if journal is not None:
        journal.record_before_image(1, header_data)

    if first_trunk == 0:
        # No current trunk — page_no itself becomes the new trunk (with no leaves)
        new_trunk = FreelistTrunkPage(next_trunk=0, leaf_pages=[])
        pager.write_page(page_no, new_trunk.encode(page_size))
        pager.update_db_header("first_freelist_trunk", page_no)
        freelist_count = pager.read_db_header_field("freelist_count")
        pager.update_db_header("freelist_count", freelist_count + 1)
        return

    # Load the current trunk and check capacity
    trunk_data = pager.read_page(first_trunk)
    trunk = FreelistTrunkPage.decode(trunk_data)

    if len(trunk.leaf_pages) < max_leaves:
        # Journal the trunk before-image before mutation (rollback mode only).
        if journal is not None:
            journal.record_before_image(first_trunk, trunk_data)
        trunk.leaf_pages.append(page_no)
        pager.write_page(first_trunk, trunk.encode(page_size))
        freelist_count = pager.read_db_header_field("freelist_count")
        pager.update_db_header("freelist_count", freelist_count + 1)
    else:
        # Current trunk is full — page_no becomes the new trunk, points to old trunk
        new_trunk = FreelistTrunkPage(next_trunk=first_trunk, leaf_pages=[])
        pager.write_page(page_no, new_trunk.encode(page_size))
        pager.update_db_header("first_freelist_trunk", page_no)
        freelist_count = pager.read_db_header_field("freelist_count")
        pager.update_db_header("freelist_count", freelist_count + 1)


def freelist_pages(pager: Pager) -> set[int]:
    """Return the complete set of page numbers currently on the freelist chain.

    Walks the trunk chain from first_freelist_trunk, collecting every trunk
    page number and every leaf page number. Guards against cycles by bounding
    the walk by db_size — raises FreelistCorruptionError if a page repeats.
    """
    result: set[int] = set()
    db_size = pager.read_db_header_field("db_size")
    trunk_no = pager.read_db_header_field("first_freelist_trunk")
    seen: set[int] = set()

    while trunk_no != 0:
        if trunk_no in seen:
            raise FreelistCorruptionError(
                f"Freelist chain cycle detected: page {trunk_no} repeated "
                f"(seen={sorted(seen)})"
            )
        if trunk_no > db_size:
            raise FreelistCorruptionError(
                f"Freelist trunk page {trunk_no} exceeds db_size={db_size}"
            )
        seen.add(trunk_no)
        result.add(trunk_no)

        trunk_data = pager.read_page(trunk_no)
        trunk = FreelistTrunkPage.decode(trunk_data)

        for leaf_no in trunk.leaf_pages:
            if leaf_no in seen:
                raise FreelistCorruptionError(
                    f"Freelist chain cycle detected: leaf page {leaf_no} repeated "
                    f"(seen={sorted(seen)})"
                )
            seen.add(leaf_no)
            result.add(leaf_no)

        trunk_no = trunk.next_trunk

    return result
