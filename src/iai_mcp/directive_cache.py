"""Synchronous global cache for the directive segment injected at session start.

Refreshed from inside capture_turn/contradict themselves (see capture.py,
retrieve.py) so every call site -- RPC handlers, the deferred_drain_worker
child process, and any future caller -- stays current by construction,
independent of the session-start precache throttle.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

log = logging.getLogger(__name__)

DIRECTIVES_CACHE_PATH = Path.home() / ".iai-mcp" / ".directives.cached.md"


def write_directives_cache(store, *, cache_path: Path | None = None) -> None:
    # Resolved at call time, not bound as a default -- a caller that
    # monkeypatches the module-level DIRECTIVES_CACHE_PATH constant (tests)
    # must actually redirect the write; a bound default would freeze the
    # value present at import time and ignore the patch.
    if cache_path is None:
        cache_path = DIRECTIVES_CACHE_PATH
    try:
        from iai_mcp.session import render_directive_segment
        from iai_mcp.store import flush_record_buffer

        # render_directive_segment reads records via a raw SQL scan; a
        # just-inserted row can still be sitting in the unflushed record
        # buffer, invisible to that scan, without this flush.
        try:
            flush_record_buffer(store)
        except Exception as exc:  # noqa: BLE001 -- best-effort visibility flush
            log.debug("directive_cache_flush_failed: %s", exc)

        rendered = render_directive_segment(store)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Writer-unique tmp: concurrent captures/contradicts must never
        # interleave their writes through a shared tmp name.
        tmp_path = cache_path.with_suffix(
            f"{cache_path.suffix}.tmp{os.getpid()}.{threading.get_ident()}"
        )
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(rendered)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, cache_path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
    except Exception as exc:  # noqa: BLE001 -- cache write must never break the caller
        log.warning("directive cache write failed: %s", exc, exc_info=True)


__all__ = ["DIRECTIVES_CACHE_PATH", "write_directives_cache"]
