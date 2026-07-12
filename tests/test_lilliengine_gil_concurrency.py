"""Concurrency regression: a long engine scan must not hold the GIL.

The probe below disables normal periodic GIL switching, starts an observer
thread, then runs full-table scans. The observer can make progress before the
scan loop returns only if the native engine explicitly releases the GIL.
"""
from __future__ import annotations

import os
import sys
import threading

import pytest

os.environ.setdefault("LILLI_FSYNC_MODE", "fast")


# ---------------------------------------------------------------------------
# Skip guard — native module must be installed
# ---------------------------------------------------------------------------


def _native_available() -> bool:
    try:
        from iai_mcp_native import engine  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _native_available(),
    reason="iai_mcp_native.engine submodule not installed (build the native wheel)",
)

# ---------------------------------------------------------------------------
# Schema / SQL constants
# ---------------------------------------------------------------------------

_DDL = (
    "CREATE TABLE items ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "key TEXT NOT NULL, "
    "value TEXT NOT NULL)"
)
_INSERT = "INSERT INTO items (key, value) VALUES (?, ?)"
_SELECT_ALL = "SELECT id, key, value FROM items"

# Row count: enough for a single `run_execute` scan to take 50-150 ms.
# The counter thread's gap during that window is the GIL-hold signal.
_POPULATE_ROWS = 80_000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_engine(path: str):
    from iai_mcp_native import engine

    return engine.Connection.open(path, 384)


def _make_store(path: str, n: int = _POPULATE_ROWS) -> None:
    """Create and populate a lilli store, then close it."""
    conn = _open_engine(path)
    conn.execute(_DDL)
    batch = [(f"key_{i}", f"value_{i}") for i in range(n)]
    conn.executemany(_INSERT, batch)
    conn.close()


# ---------------------------------------------------------------------------
# Synchronized GIL-release probe
# ---------------------------------------------------------------------------


def _thread_progresses_during_scan(path: str) -> bool:
    """Return whether another Python thread ran inside the native scan span."""
    ready = threading.Event()
    start = threading.Event()
    progressed = threading.Event()

    def observer() -> None:
        ready.set()
        start.wait()
        progressed.set()

    thread = threading.Thread(target=observer, daemon=True)
    thread.start()
    assert ready.wait(timeout=5), "observer thread did not become ready"

    conn = _open_engine(path)
    old_switch_interval = sys.getswitchinterval()
    made_progress = False
    try:
        # Prevent the Python evaluator from handing the GIL to the observer
        # between scans. Progress must happen inside py.allow_threads(...).
        sys.setswitchinterval(10.0)
        start.set()
        for _ in range(4):
            conn.execute(_SELECT_ALL)
            if progressed.is_set():
                made_progress = True
                break
    finally:
        # Capture the result above, before restoring normal switching or
        # joining: either cleanup step would let a GIL-blocked observer run.
        sys.setswitchinterval(old_switch_interval)
        start.set()
        thread.join(timeout=2.0)
        conn.close()

    return made_progress


# ---------------------------------------------------------------------------
# The regression test
# ---------------------------------------------------------------------------


def test_engine_scan_releases_gil(tmp_path):
    """The engine releases the GIL while executing a full-table scan."""
    path = str(tmp_path / "scan.lilli")
    _make_store(path)

    # Warm up the store so the first scan uses the page cache.
    warmup_conn = _open_engine(path)
    warmup_conn.execute(_SELECT_ALL).fetchall()
    warmup_conn.close()

    assert _thread_progresses_during_scan(path), (
        "observer thread made no progress during four full-table scans; "
        "the native engine appears to hold the GIL during store I/O"
    )
