"""flock() compatibility seam.

POSIX re-exports ``fcntl.flock`` and its constants verbatim. Windows maps the
same surface onto an ``msvcrt`` 1-byte region lock: every holder locks the
first byte, which is all flock's whole-file-mutex semantics need. Windows has
no shared locks — ``LOCK_SH`` degrades to exclusive, and ``SHARED_IS_EXCLUSIVE``
tells lock-mode transitions (shared→exclusive and back) to relabel their
bookkeeping instead of re-locking a region this process already owns (a
same-process re-lock raises on Windows). A non-blocking failure is re-raised
with ``EAGAIN`` so POSIX-shaped retry loops work unchanged.
"""
from __future__ import annotations

try:
    from fcntl import LOCK_EX, LOCK_NB, LOCK_SH, LOCK_UN, flock  # noqa: F401

    SHARED_IS_EXCLUSIVE = False
except ImportError:  # Windows
    import errno as _errno
    import msvcrt as _msvcrt
    import os as _os

    LOCK_SH = 0x1
    LOCK_EX = 0x2
    LOCK_NB = 0x4
    LOCK_UN = 0x8
    SHARED_IS_EXCLUSIVE = True

    def flock(fd: int, operation: int) -> None:
        _os.lseek(fd, 0, _os.SEEK_SET)
        if operation & LOCK_UN:
            try:
                _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)
            except OSError:
                # flock(LOCK_UN) on an unheld lock is a no-op; mirror that.
                pass
            return
        mode = _msvcrt.LK_NBLCK if operation & LOCK_NB else _msvcrt.LK_LOCK
        try:
            _msvcrt.locking(fd, mode, 1)
        except OSError as exc:
            raise OSError(_errno.EAGAIN, "lock held (msvcrt)") from exc
