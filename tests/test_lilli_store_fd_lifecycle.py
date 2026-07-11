"""Regression: HippoDB.close() under the lilli storage driver releases all OS
file handles (main db fd + WAL sidecar fd) deterministically at close() time,
without requiring a GC cycle."""

from __future__ import annotations

import os
import tempfile

import pytest


def _fd_count() -> int:
    return len(os.listdir("/dev/fd"))


def _require_lilli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")


@pytest.mark.parametrize("n", [1, 10, 40])
def test_lilli_store_releases_all_fds_on_close(monkeypatch: pytest.MonkeyPatch, n: int) -> None:
    """Open N stores, close them, assert fd count returns to baseline.

    The lilli engine opens two OS handles per store: the main db file and the
    WAL sidecar.  Both must be closed by HippoDB.close() without relying on
    Python GC to drop the Rust connection object.
    """
    _require_lilli(monkeypatch)

    # Import after env patch so the driver is visible at import time.
    from iai_mcp.hippo import HippoDB  # noqa: PLC0415

    base = _fd_count()
    dirs = [tempfile.mkdtemp() for _ in range(n)]
    dbs = [HippoDB(d) for d in dirs]

    assert _fd_count() > base, "expected fds to increase while stores are open"

    for db in dbs:
        db.close()

    leaked = _fd_count() - base
    assert leaked == 0, (
        f"HippoDB.close() leaked {leaked} fd(s) across {n} store(s); "
        "the lilli engine must close the db and WAL file handles at close() time"
    )


def test_lilli_store_releases_fds_with_live_cursor_at_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Open N stores each with a live cursor across the owner close, then close
    the cursor (the last holder) and assert fd count returns to baseline.

    A cursor held live across the owner's close keeps the shared backing store
    above one holder, so the owner's close defers the file/WAL release. The fix
    surrenders the owner's own hold at close() so the LAST holder (the cursor)
    releases the db and WAL fds deterministically at its close — not at GC. Before
    the fix the owner pinned the store, so the fds were retained until GC and this
    open-N/close-N cycle leaked handles (and, at scale, advisory locks).
    """
    _require_lilli(monkeypatch)

    from iai_mcp.hippo import HippoDB  # noqa: PLC0415

    n = 40
    base = _fd_count()
    dirs = [tempfile.mkdtemp() for _ in range(n)]
    dbs = [HippoDB(d) for d in dirs]

    # A live cursor per store keeps each backing store above one holder.
    cursors = [db._conn.cursor() for db in dbs]
    for db in dbs:
        db.close()  # cursor still live → file/WAL release deferred to last holder

    # Close every last holder; the fds must be released here, not at GC.
    for cur in cursors:
        cur.close()

    leaked = _fd_count() - base
    assert leaked == 0, (
        f"close with a live cursor leaked {leaked} fd(s) across {n} store(s); "
        "the last holder must release the db and WAL handles at close() time"
    )
