"""Page-level constants for the LilliBrain storage engine."""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Magic and page geometry
# ---------------------------------------------------------------------------

DB_MAGIC = b"LILLI storage 1\x00"  # 16 bytes — own magic, not SQLite's
PAGE_SIZE = 8192                    # 8 KiB per page; records whose payload exceeds the usable
                                    # threshold (PAGE_SIZE - 35) spill into a chain of overflow
                                    # pages that the freelist allocates and the codec reassembles.
HEADER_SIZE = 100                   # bytes reserved for DB header on page 1

# ---------------------------------------------------------------------------
# Page type bytes (SQLite-compatible values)
# ---------------------------------------------------------------------------

PAGE_LEAF_TABLE = 0x0D
PAGE_INTERIOR_TABLE = 0x05

# ---------------------------------------------------------------------------
# Page header sizes
# ---------------------------------------------------------------------------

LEAF_HEADER_SIZE = 8       # bytes
INTERIOR_HEADER_SIZE = 12  # bytes (includes 4-byte rightmost child pointer)

# ---------------------------------------------------------------------------
# Leaf cell cap — env-injectable ceiling; byte capacity governs splits in production
# ---------------------------------------------------------------------------

# The byte-capacity check (_leaf_has_room_for) is the governing split trigger
# in production: a 32 KiB page holds far more than this ceiling before it fills
# by bytes, so the cap is effectively unreachable at the default.  Test suites
# may inject a small value (e.g. LILLI_LEAF_MAX_CELLS=4) so that split, merge,
# and borrow paths fire without requiring large payloads.
LEAF_MAX_CELLS: int = int(os.getenv("LILLI_LEAF_MAX_CELLS", "9999"))

# ---------------------------------------------------------------------------
# DB header field byte offsets (big-endian throughout)
# ---------------------------------------------------------------------------

HDR_MAGIC_OFFSET = 0              # 16 bytes
HDR_PAGE_SIZE_OFFSET = 16         # 2 bytes
HDR_WRITE_VERSION_OFFSET = 18     # 1 byte (1=rollback journal)
HDR_READ_VERSION_OFFSET = 19      # 1 byte
HDR_DB_SIZE_OFFSET = 28           # 4 bytes (total pages)
HDR_FIRST_FREELIST_OFFSET = 32    # 4 bytes (first trunk page, 0=empty)
HDR_FREELIST_COUNT_OFFSET = 36    # 4 bytes (total free pages)
HDR_SCHEMA_COOKIE_OFFSET = 40     # 4 bytes

# ---------------------------------------------------------------------------
# Write-ahead log format
# ---------------------------------------------------------------------------

WAL_MAGIC_BE = 0x377F0683  # big-endian checksum variant; first 4 bytes of a WAL file
WAL_FORMAT_VERSION = 3_007_000  # format version stored in the WAL header (bytes 4-7)
WAL_HEADER_SIZE = 32             # bytes: fixed-size WAL file header
WAL_FRAME_HEADER_SIZE = 24       # bytes: per-frame header (page-no + commit-size + checksum pair)
WAL_WRITE_VERSION = 2            # value of HDR_WRITE_VERSION_OFFSET when write-ahead mode is active
                                 # (1 = rollback journal, 2 = write-ahead log)
WAL_FRAME_SIZE = WAL_FRAME_HEADER_SIZE + PAGE_SIZE  # total bytes per frame: header + one page

# ---------------------------------------------------------------------------
# Write-ahead log auto-checkpoint
# ---------------------------------------------------------------------------

WAL_AUTOCHECKPOINT_FRAMES = 1_000  # merge the sidecar into the main file when committed
                                    # frame count reaches this threshold (mirrors SQLite default)

# ---------------------------------------------------------------------------
# Overflow page constants
# ---------------------------------------------------------------------------

# Spill threshold: a payload whose total length exceeds this value is moved
# entirely to a chain of overflow pages; zero payload bytes remain on the leaf.
OVERFLOW_THRESHOLD: int = PAGE_SIZE - 35

# Byte offset within an overflow page of the uint32 BE next-page pointer.
# A zero next-pointer terminates the chain.
OVERFLOW_NEXT_OFFSET: int = 0

# Byte offset within an overflow page where payload bytes begin.
OVERFLOW_DATA_OFFSET: int = 4

# Internal page-type marker for overflow pages.  Never written to a page
# (overflow pages carry no distinguishable type byte; the first four bytes
# are the next-page pointer).  Used only by check_integrity when labelling
# pages discovered by following cell overflow pointers.
PAGE_TYPE_OVERFLOW: int = 0xFF
