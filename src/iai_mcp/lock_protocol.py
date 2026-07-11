from __future__ import annotations

import errno
import fcntl
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _intent_path(lock_path: Path) -> Path:
    return lock_path.parent / ".consolidation-pending"


def set_consolidation_intent(lock_path: Path) -> None:
    flag = _intent_path(lock_path)
    flag.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(flag), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
    except FileExistsError:
        pass


def clear_consolidation_intent(lock_path: Path) -> None:
    flag = _intent_path(lock_path)
    try:
        flag.unlink()
    except FileNotFoundError:
        pass


def check_consolidation_intent(lock_path: Path) -> bool:
    return _intent_path(lock_path).exists()


is_consolidation_pending = check_consolidation_intent


def cleanup_stale_consolidation_intent(lock_path: Path) -> None:
    flag = _intent_path(lock_path)
    try:
        flag.unlink()
        logger.warning(
            "cleanup_stale_intent: removed stale %s — previous daemon may have crashed",
            flag,
        )
    except FileNotFoundError:
        pass


def acquire_client_shared_nb(fd: int, lock_path: Path) -> bool:
    if check_consolidation_intent(lock_path):
        return False

    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
            return False
        raise


__all__ = [
    "set_consolidation_intent",
    "clear_consolidation_intent",
    "check_consolidation_intent",
    "is_consolidation_pending",
    "cleanup_stale_consolidation_intent",
    "acquire_client_shared_nb",
]
