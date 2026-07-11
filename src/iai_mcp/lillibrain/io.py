"""Durable file-sync abstractions for the LilliBrain storage engine.

Wraps OS-level fsync primitives so callers never call os.fsync() directly.
F_FULLFSYNC on macOS forces the drive controller's DRAM cache to NAND.
"""
from __future__ import annotations

import fcntl
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Platform sentinel
# ---------------------------------------------------------------------------

_DARWIN: bool = sys.platform == "darwin"


def _full_flush_disabled() -> bool:
    """True when the caller opted out of the macOS full drive-cache flush.

    Set ``LILLI_FSYNC_MODE=fast`` to substitute a plain ``os.fsync`` for
    ``F_FULLFSYNC``. Intended for test and CI runs, where logical correctness —
    not power-loss durability — is under test: the per-commit ``F_FULLFSYNC``
    latency otherwise makes large property-test runs take hours. Leave unset in
    production so committed pages reach NAND.
    """
    return os.environ.get("LILLI_FSYNC_MODE") == "fast"


# ---------------------------------------------------------------------------
# Durable sync helpers
# ---------------------------------------------------------------------------


def durable_fsync(fd: int) -> None:
    """Flush fd to stable storage.

    On macOS, F_FULLFSYNC forces the drive controller's volatile DRAM cache
    to NAND. On Linux, os.fsync issues FLUSH to the storage device. Falls
    back to os.fsync if F_FULLFSYNC is unavailable (network FS, FUSE, VMs).
    Raises OSError on failure — callers must not silence this.
    """
    if _DARWIN and not _full_flush_disabled():
        try:
            ret = fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
            if ret != 0:
                os.fsync(fd)
        except (AttributeError, OSError):
            os.fsync(fd)
    else:
        os.fsync(fd)


def durable_fdatasync(fd: int) -> None:
    """Flush fd data to stable storage, skipping metadata if supported.

    On macOS, F_FULLFSYNC is used (no fdatasync equivalent). On Linux,
    os.fdatasync skips non-critical metadata updates for performance.
    Falls back to os.fsync when fdatasync is unavailable (macOS, FUSE).
    Raises OSError on failure — callers must not silence this.
    """
    if _DARWIN and not _full_flush_disabled():
        try:
            ret = fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
            if ret != 0:
                os.fsync(fd)
        except (AttributeError, OSError):
            os.fsync(fd)
    else:
        _fdatasync = getattr(os, "fdatasync", None)
        if _fdatasync is not None:
            _fdatasync(fd)
        else:
            os.fsync(fd)


def fsync_directory(path: str | Path) -> None:
    """Flush the directory entry for ``path`` to stable storage.

    Ensures that newly created files in the directory are durable after a
    rename/unlink sequence. The directory fd is opened read-only, fsync'd,
    then closed. Some platforms (macOS) silently succeed on directory fsync
    without flushing; callers must also fsync the file itself when durability
    of the file content is required.

    Raises OSError if the directory cannot be opened or flushed.
    """
    dir_path = str(path)
    dir_fd = os.open(dir_path, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
