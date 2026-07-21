from __future__ import annotations

from iai_mcp import _flock as fcntl
import os
import threading
import time
from pathlib import Path



def test_consolidator_acquires_lock_ex_under_continuous_readers(
    hermetic_store: Path,
) -> None:
    from iai_mcp.lock_protocol import (  # type: ignore[import]
        set_consolidation_intent,
        clear_consolidation_intent,
        acquire_client_shared_nb,
    )

    lock_path = hermetic_store / "hippo" / ".lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch()

    stop_event = threading.Event()
    reader_errors: list[Exception] = []

    def _reader_loop() -> None:
        fd = os.open(str(lock_path), os.O_RDONLY)
        try:
            while not stop_event.is_set():
                acquired = acquire_client_shared_nb(fd, lock_path)
                if acquired:
                    time.sleep(0.001)
                    fcntl.flock(fd, fcntl.LOCK_UN)
                else:
                    time.sleep(0.001)
        except Exception as exc:
            reader_errors.append(exc)
        finally:
            os.close(fd)

    threads = [threading.Thread(target=_reader_loop, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()

    time.sleep(0.05)

    set_consolidation_intent(lock_path)
    try:
        fd_ex = os.open(str(lock_path), os.O_RDWR)
        acquired = False
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            try:
                fcntl.flock(fd_ex, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                time.sleep(0.01)
        if acquired:
            fcntl.flock(fd_ex, fcntl.LOCK_UN)
        os.close(fd_ex)
    finally:
        clear_consolidation_intent(lock_path)
        stop_event.set()
        for t in threads:
            t.join(timeout=2.0)

    assert not reader_errors, f"reader loop errors: {reader_errors}"
    assert acquired, (
        "consolidator did not acquire LOCK_EX within 4 s under continuous readers "
        "(REQ-4 yield protocol / F3 starvation)"
    )


def test_recency_read_during_busy_meets_slo(hermetic_store: Path) -> None:
    from iai_mcp.direct_recency import read_recent_user_turns_direct  # type: ignore[import]

    lock_path = hermetic_store / "hippo" / ".lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch()

    from iai_mcp.store import MemoryStore, flush_record_buffer
    from iai_mcp.types import EMBED_DIM, MemoryRecord
    import uuid
    from datetime import datetime, timezone

    store = MemoryStore(hermetic_store)
    try:
        rec = MemoryRecord(
            id=uuid.uuid4(),
            tier="episodic",
            literal_surface="busy-vacuum probe text",
            aaak_index="",
            embedding=[0.0] * EMBED_DIM,
            community_id=None,
            centrality=0.0,
            detail_level=1,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[{"session_id": "test-session", "role": "user"}],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tags=["role:user"],
            language="en",
        )
        store.insert(rec)
        flush_record_buffer(store)
    finally:
        store.close()

    ready = threading.Event()
    done = threading.Event()

    def _hold_exclusive() -> None:
        fd = os.open(str(lock_path), os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            ready.set()
            done.wait(timeout=3.0)
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    t = threading.Thread(target=_hold_exclusive, daemon=True)
    t.start()
    ready.wait(timeout=2.0)

    try:
        t0 = time.monotonic()
        turns = read_recent_user_turns_direct(hermetic_store, n=5)
        elapsed = time.monotonic() - t0
    finally:
        done.set()
        t.join(timeout=2.0)

    assert elapsed <= 1.5, (
        f"recency read during busy exclusive op took {elapsed:.3f} s (SLO ≤1.5 s)"
    )
    surfaces = [turn.literal_surface for turn in turns]
    assert any("busy-vacuum probe text" in s for s in surfaces), (
        "stored turn not found during exclusive-lock contention"
    )


def test_check_then_lock_toctou_consolidator_not_starved(
    hermetic_store: Path,
) -> None:
    from iai_mcp.lock_protocol import (  # type: ignore[import]
        set_consolidation_intent,
        clear_consolidation_intent,
        acquire_client_shared_nb,
        check_consolidation_intent,
    )

    lock_path = hermetic_store / "hippo" / ".lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch()

    stop_event = threading.Event()
    churn_errors: list[Exception] = []
    post_acquire_recheck_count = 0

    _prober_ready = threading.Event()
    _intent_set = threading.Event()

    def _churn_client() -> None:
        nonlocal post_acquire_recheck_count
        fd = os.open(str(lock_path), os.O_RDONLY)
        try:
            while not stop_event.is_set():
                acquired = acquire_client_shared_nb(fd, lock_path)
                if acquired:
                    if check_consolidation_intent(lock_path):
                        fcntl.flock(fd, fcntl.LOCK_UN)
                        post_acquire_recheck_count += 1
                    else:
                        time.sleep(0.0005)
                        fcntl.flock(fd, fcntl.LOCK_UN)
                else:
                    time.sleep(0.001)
        except Exception as exc:
            churn_errors.append(exc)
        finally:
            os.close(fd)

    def _prober_client() -> None:
        nonlocal post_acquire_recheck_count
        fd = os.open(str(lock_path), os.O_RDONLY)
        try:
            if check_consolidation_intent(lock_path):
                return

            _prober_ready.set()

            _intent_set.wait(timeout=2.0)

            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except OSError:
                return

            if check_consolidation_intent(lock_path):
                fcntl.flock(fd, fcntl.LOCK_UN)
                post_acquire_recheck_count += 1
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    threads = [threading.Thread(target=_churn_client, daemon=True) for _ in range(6)]
    prober = threading.Thread(target=_prober_client, daemon=True)
    for t in threads:
        t.start()
    prober.start()

    time.sleep(0.05)

    _prober_ready.wait(timeout=1.0)

    set_consolidation_intent(lock_path)
    _intent_set.set()

    try:
        fd_ex = os.open(str(lock_path), os.O_RDWR)
        acquired = False
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            try:
                fcntl.flock(fd_ex, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                time.sleep(0.005)
        if acquired:
            fcntl.flock(fd_ex, fcntl.LOCK_UN)
        os.close(fd_ex)
    finally:
        clear_consolidation_intent(lock_path)
        stop_event.set()
        for t in threads:
            t.join(timeout=2.0)
        prober.join(timeout=2.0)

    assert not churn_errors, f"churn client errors: {churn_errors}"
    assert acquired, (
        "H1 TOCTOU: consolidator did not acquire LOCK_EX within 4 s under "
        "sustained client churn with post-acquire recheck (REQ-4 / H1)"
    )
    assert post_acquire_recheck_count > 0, (
        "H1 TOCTOU: post-acquire intent recheck never fired — "
        "the recheck-and-release path was not exercised"
    )


def test_client_lock_wait_bounded_below_slo(hermetic_store: Path) -> None:
    from iai_mcp.lock_protocol import acquire_client_shared_nb  # type: ignore[import]

    lock_path = hermetic_store / "hippo" / ".lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch()

    ready = threading.Event()
    done = threading.Event()

    def _hold_ex() -> None:
        fd = os.open(str(lock_path), os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            ready.set()
            done.wait(timeout=0.6)
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    t = threading.Thread(target=_hold_ex, daemon=True)
    t.start()
    ready.wait(timeout=1.0)

    fd_sh = os.open(str(lock_path), os.O_RDONLY)
    try:
        t0 = time.monotonic()
        done.set()
        acquired = False
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            result = acquire_client_shared_nb(fd_sh, lock_path)
            if result:
                acquired = True
                break
            time.sleep(0.01)
        elapsed = time.monotonic() - t0
        if acquired:
            fcntl.flock(fd_sh, fcntl.LOCK_UN)
    finally:
        os.close(fd_sh)
        t.join(timeout=2.0)

    assert elapsed < 1.5, (
        f"client lock wait took {elapsed:.3f} s — must be strictly < 1.5 s "
        "(not the 2000 ms busy_timeout ceiling)"
    )
    assert acquired, "client shared lock was never acquired within the 1.5 s SLO"
