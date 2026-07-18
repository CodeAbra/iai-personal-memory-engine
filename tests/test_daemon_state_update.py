"""Contracts for the atomic daemon-state write path.

The daemon, its socket handlers, the watchdog, and the operator CLI all
write the same runtime-state file from different processes and threads.
Every writer must go through ``update_state`` (flock -> fresh load ->
targeted mutation -> save): a whole-dict ``save_state`` of a long-held
dict silently erases every key another writer persisted since that dict
was loaded — observed live as a consumed force-rem flag resurrecting
within seconds and re-triggering consolidation on every tick for hours.
"""

from __future__ import annotations

import json
import multiprocessing
import threading
from pathlib import Path

import pytest

from iai_mcp.daemon_state import (
    DAEMON_STATE_LOCK_FILENAME,
    load_state,
    save_state,
    update_state,
)


@pytest.fixture()
def state_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    return tmp_path


def test_update_state_applies_mutation_and_returns_saved_dict(state_root: Path) -> None:
    save_state({"a": 1})

    result = update_state(lambda d: d.__setitem__("b", 2))

    assert result == {"a": 1, "b": 2}
    on_disk = json.loads((state_root / ".daemon-state.json").read_text())
    assert on_disk == {"a": 1, "b": 2}


def test_update_state_reads_fresh_state_not_caller_view(state_root: Path) -> None:
    save_state({"force_rem_request": {"pending": True}})
    stale_view = load_state()

    update_state(
        lambda d: d.__setitem__("force_rem_request", {"pending": False}),
    )

    # A later writer holding the stale view persists only ITS key — the
    # cleared flag must survive.
    def _persist_own_key(d: dict) -> None:
        d["last_session_open"] = {"session_id": "s1"}

    update_state(_persist_own_key)

    final = load_state()
    assert final["force_rem_request"] == {"pending": False}
    assert final["last_session_open"] == {"session_id": "s1"}
    assert stale_view["force_rem_request"] == {"pending": True}  # stale stays stale


def test_update_state_releases_lock_on_mutator_exception(state_root: Path) -> None:
    save_state({"a": 1})

    with pytest.raises(RuntimeError):
        update_state(lambda d: (_ for _ in ()).throw(RuntimeError("boom")))

    # Lock must be free: a follow-up update succeeds without blocking.
    result = update_state(lambda d: d.__setitem__("after", True))
    assert result["after"] is True
    assert (state_root / DAEMON_STATE_LOCK_FILENAME).exists()


def test_concurrent_threaded_updates_lose_nothing(state_root: Path) -> None:
    save_state({})
    n_writers, n_rounds = 8, 25

    def _writer(idx: int) -> None:
        for r in range(n_rounds):
            update_state(lambda d: d.__setitem__(f"w{idx}", r))

    threads = [threading.Thread(target=_writer, args=(i,)) for i in range(n_writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = load_state()
    for i in range(n_writers):
        assert final.get(f"w{i}") == n_rounds - 1


def _process_writer(store_root: str, key: str, rounds: int) -> None:
    import os

    os.environ["IAI_MCP_STORE"] = store_root
    from iai_mcp.daemon_state import update_state as _update

    for r in range(rounds):
        _update(lambda d: d.__setitem__(key, r))


def test_concurrent_cross_process_updates_lose_nothing(state_root: Path) -> None:
    save_state({"anchor": True})
    ctx = multiprocessing.get_context("spawn")
    rounds = 10
    procs = [
        ctx.Process(target=_process_writer, args=(str(state_root), f"p{i}", rounds))
        for i in range(3)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0

    final = load_state()
    assert final.get("anchor") is True
    for i in range(3):
        assert final.get(f"p{i}") == rounds - 1


def test_consumed_force_rem_survives_unrelated_handler_writes(state_root: Path) -> None:
    # The live failure mode end-to-end: an honored force-rem clear followed
    # by unrelated socket-handler writes driven from a dict loaded BEFORE
    # the clear. The pending flag must stay consumed.
    save_state(
        {
            "force_rem_request": {"ts": "t0", "pending": True},
            "fsm_state": "SLEEP",
        }
    )
    shared_boot_dict = load_state()  # what the daemon holds in memory

    def _honor(d: dict) -> None:
        req = dict(d.get("force_rem_request") or {})
        req["pending"] = False
        req["honored_at"] = "t1"
        d["force_rem_request"] = req

    update_state(_honor)

    # Handler-style targeted writes sourced from the stale shared dict.
    shared_boot_dict["last_session_open"] = {"session_id": "s2", "ts": "t2"}
    update_state(
        lambda d: d.__setitem__(
            "last_session_open", shared_boot_dict["last_session_open"],
        ),
    )
    shared_boot_dict["scheduler_paused"] = False
    update_state(
        lambda d: d.__setitem__(
            "scheduler_paused", shared_boot_dict["scheduler_paused"],
        ),
    )

    final = load_state()
    assert final["force_rem_request"]["pending"] is False
    assert final["force_rem_request"]["honored_at"] == "t1"
    assert final["last_session_open"]["session_id"] == "s2"


def test_consume_first_turn_drops_only_own_session(state_root: Path) -> None:
    from iai_mcp.daemon_state import consume_first_turn

    save_state({"first_turn_pending": {"s1": "2026-07-16T00:00:00+00:00"}})
    shared = load_state()

    # Another writer registers a second session after `shared` was loaded.
    update_state(
        lambda d: d["first_turn_pending"].__setitem__(
            "s2", "2026-07-16T00:00:01+00:00",
        ),
    )

    assert consume_first_turn(shared, "s1") is True

    final = load_state()
    assert "s1" not in final["first_turn_pending"]
    assert "s2" in final["first_turn_pending"]  # the concurrent entry survives
