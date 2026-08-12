"""Write-ahead log writer for the LilliBrain storage engine.

Appends page data to a sidecar file in frame-sized chunks so the main
database file is not overwritten until a checkpoint copies committed frames
back.  The in-process page-to-frame index eliminates the need for a shared
memory file; the single-writer model is enforced by the caller.

Frame layout (24-byte header + PAGE_SIZE bytes of page data):

  Offset  Size  Field
       0     4  Page number (1-indexed)
       4     4  db_size_after: 0 for non-commit frames; DB page count for commit frames
       8     4  Salt-1 (copied from the WAL header)
      12     4  Salt-2 (copied from the WAL header)
      16     4  Checksum-1: cumulative Fibonacci two-sum through this frame
      20     4  Checksum-2: second half of cumulative checksum

The checksum algorithm is the big-endian Fibonacci two-sum variant:

  for each big-endian uint32 pair (word0, word1) in the input:
      s0 = (s0 + word0 + s1) & 0xFFFF_FFFF
      s1 = (s1 + word1 + s0) & 0xFFFF_FFFF

The chain runs from the WAL header through every frame in order.  A torn
or forged frame breaks the chain; recovery stops at the first mismatch.
"""
from __future__ import annotations

import logging
import os
import random
import struct
from pathlib import Path
from typing import TYPE_CHECKING

from iai_mcp.lillibrain.constants import (
    PAGE_SIZE,
    WAL_FORMAT_VERSION,
    WAL_FRAME_HEADER_SIZE,
    WAL_FRAME_SIZE,
    WAL_HEADER_SIZE,
    WAL_MAGIC_BE,
)

# Module-level binding so tests can patch wal_mod.durable_fsync directly
# without having to also patch io_mod.durable_fsync (the two-site patch rule).
from iai_mcp.lillibrain.io import durable_fsync  # noqa: E402

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Checksum
# ---------------------------------------------------------------------------

_MASK: int = 0xFFFF_FFFF


def _wal_checksum(data: bytes, s0: int, s1: int) -> tuple[int, int]:
    """Compute the cumulative Fibonacci two-sum WAL checksum over data.

    Input length must be a multiple of 8 bytes.  Chain across frames by
    passing the (s0, s1) pair from the previous frame.  Big-endian
    interpretation (magic == 0x377F0683).

    Raises AssertionError when len(data) % 8 != 0.
    """
    assert len(data) % 8 == 0, "WAL checksum input must be a multiple of 8 bytes"
    offset = 0
    end = len(data)
    while offset < end:
        word0 = int.from_bytes(data[offset : offset + 4], "big")
        word1 = int.from_bytes(data[offset + 4 : offset + 8], "big")
        s0 = (s0 + word0 + s1) & _MASK
        s1 = (s1 + word1 + s0) & _MASK
        offset += 8
    return s0, s1


# ---------------------------------------------------------------------------
# WALWriter
# ---------------------------------------------------------------------------


class WALWriter:
    """Owns the write-ahead sidecar file descriptor and the in-process page index.

    The sidecar is named by appending '-wal' to the database file path
    (e.g. 'brain.lilli' → 'brain.lilli-wal').  It is created with mode 0o600.

    The caller must hold the pager's write lock before calling commit_transaction
    or rollback.  WALWriter itself is not thread-safe.
    """

    def __init__(self, db_path: str | Path, *, page_size: int = PAGE_SIZE) -> None:
        self._db_path = Path(db_path)
        self._page_size = page_size

        # Sidecar path: hyphen suffix to mirror the .journal convention.
        self._path: Path = self._db_path.parent / (self._db_path.name + "-wal")

        self._wal_fd: int = os.open(
            str(self._path), os.O_CREAT | os.O_RDWR, 0o600
        )
        os.chmod(str(self._path), 0o600)
        # Truncate to zero before writing the header.  Recovery unlinks the
        # sidecar before constructing WALWriter, so this is always a fresh-file
        # path.  The explicit truncate removes any stale frame bytes that could
        # survive if that guarantee ever breaks, so correctness does not rely
        # solely on salt-pair uniqueness.
        os.ftruncate(self._wal_fd, 0)

        # Random 32-bit salts seeded fresh on each WAL file creation.
        self._salt1: int = random.randint(0, _MASK)
        self._salt2: int = random.randint(0, _MASK)
        self._checkpoint_seq: int = 0

        # Running checksum pair — updated by every _append_frame call.
        self._running_s0: int = 0
        self._running_s1: int = 0

        # Write-head: byte offset of the next frame to be written.
        self._next_frame_offset: int = WAL_HEADER_SIZE

        # In-process page-to-frame index.  Maps page_no → 0-based frame index
        # within the current WAL cycle.  Only updated on successful commit.
        self._wal_index: dict[int, int] = {}

        # Frame counter: total committed frames in the current WAL cycle.
        self._committed_frame_count: int = 0

        # Committed snapshot — the durable state after each successful commit.
        # rollback() restores the running state to this snapshot so the next
        # commit overwrites the rolled-back slots without a checksum-chain gap.
        self._committed_s0: int = 0
        self._committed_s1: int = 0
        self._committed_next_frame_offset: int = WAL_HEADER_SIZE

        # Write the 32-byte WAL header and seed both the running and committed
        # checksum state from the header checksum.
        self._write_wal_header()

    # ---------------------------------------------------------------------------
    # Header
    # ---------------------------------------------------------------------------

    def _write_wal_header(self) -> None:
        """Write the 32-byte WAL header at offset 0 of the sidecar.

        Seeds the running checksum pair (and the committed snapshot) from the
        header checksum so the frame chain starts from a known state.
        """
        body = struct.pack(
            ">IIIIII",
            WAL_MAGIC_BE,
            WAL_FORMAT_VERSION,
            self._page_size,
            self._checkpoint_seq,
            self._salt1,
            self._salt2,
        )
        s0, s1 = _wal_checksum(body, 0, 0)
        header = body + struct.pack(">II", s0, s1)
        os.pwrite(self._wal_fd, header, 0)

        # Seed running state from the header checksum.
        self._running_s0 = s0
        self._running_s1 = s1

        # The header is durable by the time the first commit's fsync fires;
        # seed the committed snapshot identically so rollback() on an
        # un-committed writer is a clean no-op.
        self._committed_s0 = s0
        self._committed_s1 = s1
        self._committed_next_frame_offset = WAL_HEADER_SIZE

    # ---------------------------------------------------------------------------
    # Frame append
    # ---------------------------------------------------------------------------

    def _append_frame(
        self, page_no: int, page_data: bytes, db_size_after: int
    ) -> None:
        """Append one frame at the current write-head and advance it.

        Checksum input is the first 8 bytes of the frame header concatenated
        with the full page data (8 + PAGE_SIZE bytes, both multiples of 8).
        Updates _running_s0/_running_s1 and _next_frame_offset in place.
        """
        if len(page_data) != self._page_size:
            raise ValueError(
                f"page_data length {len(page_data)} != page_size {self._page_size}"
            )
        raw_header = struct.pack(
            ">IIII", page_no, db_size_after, self._salt1, self._salt2
        )
        # Checksum covers the first 8 bytes of the frame header + page data.
        new_s0, new_s1 = _wal_checksum(
            raw_header[:8] + page_data, self._running_s0, self._running_s1
        )
        full_header = raw_header + struct.pack(">II", new_s0, new_s1)
        os.pwrite(self._wal_fd, full_header + page_data, self._next_frame_offset)
        self._next_frame_offset += WAL_FRAME_SIZE
        self._running_s0 = new_s0
        self._running_s1 = new_s1

    # ---------------------------------------------------------------------------
    # Transaction interface
    # ---------------------------------------------------------------------------

    def commit_transaction(
        self, dirty_pages: dict[int, bytes], db_size_after: int
    ) -> None:
        """Append all dirty pages as WAL frames and issue exactly one durable sync.

        The last frame carries the commit marker (db_size_after > 0); all
        preceding frames in this transaction have db_size_after == 0.

        On success:
        - _wal_index is updated for every committed page (newest frame wins).
        - _committed_frame_count is advanced by the number of frames written.
        - The committed snapshot (_committed_s0, _committed_s1,
          _committed_next_frame_offset) is updated to the post-commit state so
          a later rollback() can restore the running state exactly.
        """
        items = list(dirty_pages.items())
        if not items:
            return

        start_frame_idx = self._committed_frame_count
        for i, (page_no, page_data) in enumerate(items):
            is_last = i == len(items) - 1
            commit_size = db_size_after if is_last else 0
            self._append_frame(page_no, page_data, db_size_after=commit_size)

        # One fsync for the entire transaction — the only sync on the hot write path.
        durable_fsync(self._wal_fd)

        # Update the in-process index: newest frame index wins for each page.
        for idx, (page_no, _) in enumerate(items):
            self._wal_index[page_no] = start_frame_idx + idx

        self._committed_frame_count += len(items)

        # Snapshot the committed state so rollback() can rewind exactly here.
        self._committed_s0 = self._running_s0
        self._committed_s1 = self._running_s1
        self._committed_next_frame_offset = self._next_frame_offset

    def rollback(self) -> None:
        """Discard in-progress (un-fsynced) frames by rewinding to the committed snapshot.

        Restores _running_s0, _running_s1, and _next_frame_offset to the values
        recorded at the last successful commit.  The rolled-back frame slots are
        harmlessly overwritten by the next commit; recovery ignores them because
        they carry no commit marker in the chain.

        Does NOT touch _wal_index (only commit_transaction writes the index) and
        does NOT truncate the sidecar file.
        """
        self._running_s0 = self._committed_s0
        self._running_s1 = self._committed_s1
        self._next_frame_offset = self._committed_next_frame_offset

    # ---------------------------------------------------------------------------
    # Read path
    # ---------------------------------------------------------------------------

    def read_page(self, page_no: int, db_fd: int) -> bytes:
        """Return the page bytes for page_no from the newest committed frame or the main file.

        When the in-process index has a committed frame for page_no, reads that
        frame from the sidecar and strips the 24-byte frame header, returning
        exactly page_size bytes.  When no frame exists for page_no, falls back to
        reading the page at its canonical offset in the main database file.
        """
        frame_idx = self._wal_index.get(page_no)
        if frame_idx is not None:
            frame_offset = WAL_HEADER_SIZE + frame_idx * WAL_FRAME_SIZE
            raw = os.pread(self._wal_fd, WAL_FRAME_SIZE, frame_offset)
            return raw[WAL_FRAME_HEADER_SIZE:]
        # No committed frame for this page — read from the main file.
        return os.pread(db_fd, self._page_size, (page_no - 1) * self._page_size)

    # ---------------------------------------------------------------------------
    # Checkpoint
    # ---------------------------------------------------------------------------

    def checkpoint(self, db_fd: int) -> None:
        """Merge every committed frame into the main file and reset the sidecar.

        The strict two-sync ordering enforces the durability barrier:

          1. durable_fsync(sidecar) — every committed frame is durable before
             any main-file page is overwritten.
          2. Copy the newest frame of every indexed page into the main file via
             os.pwrite at the page's canonical offset.
          3. durable_fsync(main file) — all copies are durable.
          4. Reset the sidecar in place: increment the checkpoint sequence,
             draw a fresh salt2, truncate the sidecar to header size, rewrite
             the header with the new salts, clear the in-process index and
             counters.  The sidecar fd remains open throughout (never close or
             reopen — reopening resets the kernel error sequence and can hide a
             prior failed writeback).
        """
        if not self._wal_index:
            # Nothing to checkpoint — sidecar is already at header-only state.
            return

        # Step 1: ensure the sidecar is durable before touching the main file.
        durable_fsync(self._wal_fd)

        # Step 2: copy the newest frame of each indexed page into the main file.
        for page_no, frame_idx in self._wal_index.items():
            frame_offset = WAL_HEADER_SIZE + frame_idx * WAL_FRAME_SIZE
            raw = os.pread(self._wal_fd, WAL_FRAME_SIZE, frame_offset)
            page_data = raw[WAL_FRAME_HEADER_SIZE:]
            os.pwrite(db_fd, page_data, (page_no - 1) * self._page_size)

        # Step 3: make all copied pages durable in the main file.
        durable_fsync(db_fd)

        # Step 4: reset the sidecar in place — keep the fd open.
        self._checkpoint_seq += 1
        self._salt2 = random.randint(0, _MASK)

        os.ftruncate(self._wal_fd, WAL_HEADER_SIZE)

        # Rewrite the header with the new checkpoint sequence and new salt2.
        # _write_wal_header reseeds _running_s0/_running_s1 and the committed
        # snapshot from the new header checksum.
        self._write_wal_header()

        # Clear the in-process index and reset frame counters.
        self._wal_index.clear()
        self._committed_frame_count = 0
        self._next_frame_offset = WAL_HEADER_SIZE

    # ---------------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------------

    def close(self) -> None:
        """Close the sidecar file descriptor."""
        try:
            os.close(self._wal_fd)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------


def recover_wal_on_open(
    db_path: Path | str,
    wal_path: Path | str,
    page_size: int,
) -> None:
    """Scan the write-ahead sidecar and replay committed frames into the main file.

    Called from the pager at open time, before any reads are served.  Opens the
    sidecar read-only; does not require a live WALWriter.

    Scan algorithm:
      1. Read the 32-byte header; verify its magic and checksum pair.  On any
         failure, discard the sidecar and return.
      2. Unpack the header salts and seed the running checksum from the header
         checksum pair.
      3. Walk frames sequentially:
         a. Read the 24-byte frame header and page_size bytes of page data.
            Break on a short read (torn frame).
         b. If the frame's salt pair does not match the header salts, break
            (stale frame from a prior checkpoint cycle).  This check precedes
            the checksum check so a frame with a valid checksum under old salts
            is still rejected.
         c. Recompute the chained checksum over frame_header[:8] + page_data.
            Break on mismatch (corrupt or torn frame).
         d. Advance the running pair; add the page to the pending accumulator.
         e. When the frame's db_size_after field is non-zero (commit marker),
            merge pending into committed and reset pending.
      4. If committed is non-empty, open the main file r+b, write each committed
         page at its canonical offset, flush, and durable_fsync.
      5. Unlink the sidecar (idempotent: if the sidecar is absent the next
         open is a no-op; re-running over the same bytes writes identical bytes
         to the same offsets, leaving the main file unchanged).

    Idempotency: re-running recovery over the same sidecar writes the same bytes
    to the same offsets in the main file, then unlinks.  A crash between step 4
    and step 5 leaves the next open in an identical starting state.
    """
    db_path = Path(db_path)
    wal_path = Path(wal_path)

    if not wal_path.exists():
        return

    try:
        wal_fd = os.open(str(wal_path), os.O_RDONLY)
    except OSError as exc:
        logger.warning(
            "WAL recovery could not open sidecar %s; committed frames not "
            "replayed this open: %s",
            wal_path,
            exc,
        )
        return

    try:
        # Step 1: read and verify the 32-byte WAL header.
        raw_header = os.pread(wal_fd, WAL_HEADER_SIZE, 0)
        if len(raw_header) < WAL_HEADER_SIZE:
            # Too short to be a valid WAL — discard.
            wal_path.unlink(missing_ok=True)
            return

        magic, _ver, hdr_pgsz, _chk_seq, hdr_salt1, hdr_salt2, hdr_s0, hdr_s1 = (
            struct.unpack(">IIIIIIII", raw_header)
        )
        if magic != WAL_MAGIC_BE:
            wal_path.unlink(missing_ok=True)
            return
        # Cross-check page geometry: a sidecar written with a different page size
        # would cause frame boundaries to be parsed at the wrong stride.  Discard
        # rather than silently replaying wrong bytes.
        if hdr_pgsz != page_size:
            wal_path.unlink(missing_ok=True)
            return

        # Verify the header checksum (covers the first 24 bytes of the header).
        computed_s0, computed_s1 = _wal_checksum(raw_header[:24], 0, 0)
        if (computed_s0, computed_s1) != (hdr_s0, hdr_s1):
            wal_path.unlink(missing_ok=True)
            return

        # Step 2: seed the running checksum from the header checksum pair.
        running_s0, running_s1 = hdr_s0, hdr_s1

        # Accumulators: pending holds frames in the current (uncommitted) transaction;
        # committed holds all frames up to and including the last commit marker.
        pending: dict[int, bytes] = {}
        committed: dict[int, bytes] = {}

        # Step 3: walk frames sequentially.
        read_offset = WAL_HEADER_SIZE
        while True:
            # 3a: read the 24-byte frame header.
            fh_raw = os.pread(wal_fd, WAL_FRAME_HEADER_SIZE, read_offset)
            if len(fh_raw) < WAL_FRAME_HEADER_SIZE:
                break  # end of file or torn header — stop

            # Read the page data.
            page_data = os.pread(wal_fd, page_size, read_offset + WAL_FRAME_HEADER_SIZE)
            if len(page_data) < page_size:
                break  # torn page data — stop

            page_no, db_size_after, f_salt1, f_salt2, f_s0, f_s1 = (
                struct.unpack(">IIIIII", fh_raw)
            )

            # 3b: salt check before checksum check.  A stale frame from a prior
            # checkpoint cycle must be rejected even if its checksum is internally
            # valid under the old salt pair.
            if (f_salt1, f_salt2) != (hdr_salt1, hdr_salt2):
                break  # stale frame — stop replay

            # 3c: verify the chained checksum over frame_header[:8] + page_data.
            new_s0, new_s1 = _wal_checksum(fh_raw[:8] + page_data, running_s0, running_s1)
            if (new_s0, new_s1) != (f_s0, f_s1):
                break  # checksum mismatch — corrupt or torn frame — stop

            # 3d: advance running pair and accumulate into pending.
            running_s0, running_s1 = new_s0, new_s1
            pending[page_no] = page_data
            read_offset += WAL_FRAME_SIZE

            # 3e: commit marker — merge pending into committed.
            if db_size_after > 0:
                committed.update(pending)
                pending = {}

    finally:
        try:
            os.close(wal_fd)
        except OSError:
            pass

    # Step 4: write committed pages to the main file.
    if committed:
        db_fd = os.open(str(db_path), os.O_RDWR)
        try:
            for page_no, page_bytes in committed.items():
                os.pwrite(db_fd, page_bytes, (page_no - 1) * page_size)
            durable_fsync(db_fd)
        finally:
            try:
                os.close(db_fd)
            except OSError:
                pass

    # Step 5: remove the sidecar so the next open starts clean.
    wal_path.unlink(missing_ok=True)
