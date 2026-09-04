from __future__ import annotations

import logging
import threading
from typing import Callable

logger = logging.getLogger(__name__)

# id(store) is reused by CPython once the object is freed; every family that
# keys a module-level dict/cache on id(store) MUST register its purge here so
# construction-time purge_store() clears it before a fresh store can inherit
# a dead store's entry.
_REGISTRY_LOCK = threading.RLock()
_PURGE_CALLBACKS: list[Callable[[int], None]] = []

# A persistently-failing family purge leaves a ghost that would otherwise
# surface only in error logs; counted here so it is observable without a
# store (this module stays a leaf -- no events.write_event / TELEMETRY_*).
_PURGE_FAILURE_COUNTS: dict[str, int] = {}


def register_store_purge(fn: Callable[[int], None]) -> None:
    with _REGISTRY_LOCK:
        if fn not in _PURGE_CALLBACKS:
            _PURGE_CALLBACKS.append(fn)


def purge_failure_counts() -> dict[str, int]:
    with _REGISTRY_LOCK:
        return dict(_PURGE_FAILURE_COUNTS)


def purge_store(store_id: int) -> None:
    with _REGISTRY_LOCK:
        callbacks = list(_PURGE_CALLBACKS)
    for fn in callbacks:
        try:
            fn(store_id)
        except Exception as exc:  # noqa: BLE001 -- one bad family must not block the rest
            name = getattr(fn, "__name__", repr(fn))
            with _REGISTRY_LOCK:
                _PURGE_FAILURE_COUNTS[name] = _PURGE_FAILURE_COUNTS.get(name, 0) + 1
            logger.error(
                "purge_store_callback_failed",
                extra={"callback": name, "err": str(exc)[:200]},
            )
            continue
