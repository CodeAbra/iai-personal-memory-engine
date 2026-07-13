"""Platform-agnostic file locking shim.

On POSIX: thin wrapper around fcntl.flock.
On Windows: msvcrt.locking with errno normalisation so callers checking
errno.EWOULDBLOCK / errno.EAGAIN on non-blocking failures work unchanged.
The file offset is saved/restored around each call (msvcrt locks relative to
the file position; fcntl.flock does not move it) and the blocking path polls
so it waits indefinitely like POSIX rather than giving up after msvcrt's ~10 s.

Known divergence — shared locks are not truly shared on Windows.
``msvcrt.locking`` only offers exclusive byte-range locks, so LOCK_SH is
serviced as an exclusive lock: a second concurrent reader blocks where POSIX
would let both in. This is a throughput limitation, not a correctness one, and
it is deliberately NOT fixed with Win32 ``LockFileEx`` (which does support
shared locks) because callers in ``hippo/_db.py`` rely on fcntl.flock's atomic
lock *conversion* — downgrading EXCLUSIVE->SHARED and escalating SHARED->
EXCLUSIVE in place on the same fd. ``LockFileEx`` has no atomic conversion
(you must Unlock then re-Lock, racing other waiters), so swapping it in would
trade a throughput limit for a correctness hazard on the conversion paths.
A faithful port would need those call sites reworked to a conversion-free
protocol first.
"""
from __future__ import annotations

import os
import platform

if platform.system() == "Windows":
    import errno as _errno
    import msvcrt as _msvcrt
    import time as _time

    LOCK_SH = 1
    LOCK_EX = 2
    LOCK_NB = 4
    LOCK_UN = 8

    _LOCK_BYTES = 2**30
    # Poll interval when emulating POSIX's block-until-acquired behaviour.
    _BLOCK_POLL_SECONDS = 0.05

    def flock(fd: int, operation: int) -> None:
        if not isinstance(fd, int):
            fd = fd.fileno()
        # msvcrt.locking locks bytes starting from the current file position, so
        # we must seek to 0 to lock a consistent byte range across callers.
        # fcntl.flock leaves the file offset untouched, however, so save the
        # caller's offset and restore it afterwards to match POSIX semantics.
        try:
            saved_offset: int | None = os.lseek(fd, 0, os.SEEK_CUR)
        except OSError:
            saved_offset = None
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            if operation & LOCK_UN:
                try:
                    _msvcrt.locking(fd, _msvcrt.LK_UNLCK, _LOCK_BYTES)
                except OSError:
                    pass
            elif operation & (LOCK_EX | LOCK_SH):
                if operation & LOCK_NB:
                    try:
                        _msvcrt.locking(fd, _msvcrt.LK_NBLCK, _LOCK_BYTES)
                    except OSError:
                        raise OSError(
                            _errno.EWOULDBLOCK, "resource temporarily unavailable"
                        )
                else:
                    # POSIX flock blocks until the lock is acquired, but msvcrt
                    # has no infinite-block mode (LK_LOCK gives up after ~10 s
                    # and raises). Poll LK_NBLCK so a blocking acquire matches
                    # POSIX semantics instead of spuriously failing under long
                    # contention (e.g. while the consolidator holds the lock).
                    while True:
                        try:
                            _msvcrt.locking(fd, _msvcrt.LK_NBLCK, _LOCK_BYTES)
                            break
                        except OSError:
                            os.lseek(fd, 0, os.SEEK_SET)
                            _time.sleep(_BLOCK_POLL_SECONDS)
        finally:
            if saved_offset is not None:
                try:
                    os.lseek(fd, saved_offset, os.SEEK_SET)
                except OSError:
                    pass

else:
    import fcntl as _fcntl

    LOCK_SH = _fcntl.LOCK_SH
    LOCK_EX = _fcntl.LOCK_EX
    LOCK_NB = _fcntl.LOCK_NB
    LOCK_UN = _fcntl.LOCK_UN

    def flock(fd: int, operation: int) -> None:
        _fcntl.flock(fd, operation)


__all__ = ["flock", "LOCK_EX", "LOCK_NB", "LOCK_SH", "LOCK_UN"]
