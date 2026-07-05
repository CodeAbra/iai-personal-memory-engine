"""Regression guard: the daemon's hippo lock cycle must never block/wedge.

Before the fix, ``HippoDB.downgrade_to_shared()`` called a *blocking*
``flock(base_fd, LOCK_SH)`` -- the only lock site in ``hippo/_db.py`` without
``LOCK_NB``. On Windows the ``_filelock`` shim services ``LOCK_SH`` as an
exclusive msvcrt byte-range lock with no atomic EX->SH conversion, so a blocking
``LOCK_SH`` on a fd that already holds ``LOCK_EX`` polled ``LK_NBLCK`` forever
against the daemon's own range -- wedging every thread under
``_PROCESS_LOCKS_GUARD`` until the liveness watchdog force-killed the daemon
(the recurring ``pre_kill_forensic_dump`` at ``downgrade_to_shared``).
``escalate_to_exclusive()`` had the symmetric self-conflict, bounded by its
deadline so it *degraded* into ``HippoLockHeldError`` rather than wedging.

These tests run the exact daemon sequence (EXCLUSIVE -> downgrade -> escalate ->
downgrade). Crucially they run on **all** platforms -- including Windows, where
the wedge lived and where ``test_hippo_concurrency.py`` is skipped, which is why
the bug went uncaught for weeks.

Every lock call -- including ``close()`` teardown -- runs in a watchdog thread
with a join timeout. A reintroduced blocking flock therefore makes an assertion
FAIL within the timeout instead of hanging the suite (the wedged thread holds
``_PROCESS_LOCKS_GUARD``, so an un-bounded ``close()`` would otherwise deadlock).
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from iai_mcp.hippo import AccessMode, HippoLockHeldError
from iai_mcp.hippo._db import HippoDB

# A healthy conversion is sub-millisecond; escalate's own intent budget is 4s.
# 15s is far below the wedge's "forever" yet generous enough never to flake on a
# slow CI box.
_TIMEOUT_S = 15.0


def _run_bounded(fn):
    """Run ``fn()`` in a daemon thread joined with a timeout.

    Returns ``(completed, elapsed_s, error)``. ``completed is False`` means the
    call did not return within the timeout -- i.e. it wedged. This mirrors how
    the daemon invokes these conversions (via ``asyncio.to_thread`` on a pool
    thread), and guarantees a regression fails the test rather than hanging it.
    """
    box: dict = {}

    def _target() -> None:
        t0 = time.monotonic()
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 -- surfaced to the assertion
            box["error"] = exc
        finally:
            box["elapsed"] = time.monotonic() - t0

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(_TIMEOUT_S)
    return (not t.is_alive()), box.get("elapsed"), box.get("error")


def _make_exclusive_db(tmp_path: Path) -> HippoDB:
    return HippoDB(
        path=tmp_path,
        crypto_key_provider=lambda: b"0" * 32,
        access_mode=AccessMode.EXCLUSIVE,
    )


def test_downgrade_to_shared_does_not_wedge_holding_exclusive(tmp_path: Path) -> None:
    db = _make_exclusive_db(tmp_path)
    try:
        assert db._access_mode is AccessMode.EXCLUSIVE

        completed, elapsed, error = _run_bounded(db.downgrade_to_shared)

        assert completed, (
            f"downgrade_to_shared() did not return within {_TIMEOUT_S}s -- "
            "the blocking-flock wedge is back"
        )
        assert error is None, f"downgrade_to_shared() raised: {error!r}"
        assert db._access_mode is AccessMode.SHARED
        assert elapsed is not None and elapsed < 2.0, (
            f"downgrade unexpectedly slow ({elapsed:.3f}s): a bounded but "
            "contended flock retry loop may have crept in"
        )
    finally:
        _run_bounded(db.close)  # bounded so a wedge cannot deadlock teardown


def test_full_lock_cycle_no_wedge(tmp_path: Path) -> None:
    """EXCLUSIVE -> downgrade(SHARED) -> escalate(EXCLUSIVE) -> downgrade(SHARED)."""
    db = _make_exclusive_db(tmp_path)
    try:
        steps = (
            ("downgrade", db.downgrade_to_shared, AccessMode.SHARED),
            ("escalate", db.escalate_to_exclusive, AccessMode.EXCLUSIVE),
            ("downgrade", db.downgrade_to_shared, AccessMode.SHARED),
        )
        for name, fn, expected in steps:
            completed, elapsed, error = _run_bounded(fn)
            assert completed, (
                f"{name}() did not return within {_TIMEOUT_S}s -- lock cycle wedged"
            )
            assert error is None, f"{name}() raised: {error!r}"
            assert db._access_mode is expected, (
                f"after {name}(): expected {expected}, got {db._access_mode}"
            )
            assert elapsed is not None and elapsed < 5.0, (
                f"{name}() unexpectedly slow: {elapsed:.3f}s"
            )
    finally:
        _run_bounded(db.close)


def test_escalate_relabels_when_already_holding_lock(tmp_path: Path) -> None:
    """escalate_to_exclusive() must reclaim EXCLUSIVE from a lock this fd already
    holds without timing out into HippoLockHeldError -- the Windows self-conflict
    that used to make sleep consolidation silently run without the EXCLUSIVE lock.
    """
    db = _make_exclusive_db(tmp_path)
    try:
        # Reach SHARED first (the daemon's state before a sleep cycle). Bounded so
        # a broken downgrade fails here rather than hanging the suite.
        completed, _, _ = _run_bounded(db.downgrade_to_shared)
        assert completed and db._access_mode is AccessMode.SHARED, "setup downgrade wedged"

        completed, elapsed, error = _run_bounded(db.escalate_to_exclusive)

        assert completed, f"escalate_to_exclusive() wedged (> {_TIMEOUT_S}s)"
        assert not isinstance(error, HippoLockHeldError), (
            "escalate_to_exclusive() spuriously raised HippoLockHeldError on a "
            "lock this fd already holds (Windows self-conflict regression)"
        )
        assert error is None, f"escalate_to_exclusive() raised: {error!r}"
        assert db._access_mode is AccessMode.EXCLUSIVE
        assert elapsed is not None and elapsed < 5.0, (
            f"escalate unexpectedly slow ({elapsed:.3f}s): the intent-budget "
            "deadline may be firing on a self-conflict"
        )
    finally:
        _run_bounded(db.close)
