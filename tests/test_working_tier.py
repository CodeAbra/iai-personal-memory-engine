"""RED test scaffold for the active-task structured tier (working tier).

Targets ``iai_mcp.working_tier`` (not yet built). These tests establish the
falsifiable acceptance contract before the module exists: structured task
state survives across turns, populates from the sensory buffer and from
recall, consolidates durable results on close, stays off the awake recall
critical path, and holds content byte-identical (verbatim) end to end.

The task boundary is an idle gap on the per-turn feed, never a per-turn
hook: a gap larger than the configured idle-close threshold closes and
reopens the task; a smaller gap keeps the same task; session end also
closes and consolidates.

No production source file is modified by this test file.
"""

from __future__ import annotations

import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _helpers import make_tmp_store  # noqa: E402 — test helper; sys.path manipulated above

from iai_mcp import core
from iai_mcp.sensory import sensory_append, sensory_pending
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _monkeypatch_env(monkeypatch, tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _make_record(text: str, *, created_at: datetime | None = None, role: str = "user") -> MemoryRecord:
    now = created_at if created_at is not None else datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.5,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[f"role:{role}"],
        language="en",
    )


def _seed_one_record(store: MemoryStore, text: str = "reference content for probe") -> None:
    store.insert(_make_record(text))


@pytest.fixture(autouse=True)
def _reset_working_tier_singleton():
    """Reset the process-global singleton before and after every test.

    The working tier is process-global by design; leaked state from one
    test must never bleed into the next.
    """
    try:
        import iai_mcp.working_tier as wt
        wt._reset()
    except Exception:
        pass
    yield
    try:
        import iai_mcp.working_tier as wt
        wt._reset()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# SC-a: structured task state survives across turns, never reset
# ---------------------------------------------------------------------------

def test_task_state_survives_across_turns():
    """A task holds goal/sub-goals/hypotheses/focus accumulated across turns.

    Pure in-RAM logic — no storage engine involved, so no driver param.
    """
    from iai_mcp import working_tier as wt

    wt.open_task("investigate the recall latency regression")
    wt.update_task(sub_goal="reproduce the slow path locally", hypothesis="ANN cap is too tight", focus="pipeline.py K_CANDIDATES")
    wt.update_task(sub_goal="measure warm-cache p95", hypothesis="community gate over-filters", focus="community gate scoring bonus")

    entry = wt.read_task()
    assert entry is not None, "expected an open task after open_task()"
    assert entry.goal == "investigate the recall latency regression"
    assert "reproduce the slow path locally" in entry.open_subgoals
    assert "measure warm-cache p95" in entry.open_subgoals
    assert "ANN cap is too tight" in entry.hypotheses
    assert "community gate over-filters" in entry.hypotheses
    assert entry.focus == "community gate scoring bonus"


# ---------------------------------------------------------------------------
# SC-b: populated from sensory AND from recall, observable mid-task
# ---------------------------------------------------------------------------

def test_populate_from_sensory_observable(tmp_path, monkeypatch):
    """Sensory content and update_from_record are both reflected in read_task().

    Pure in-RAM + a sensory write; no storage-engine assertion, so no
    driver param.
    """
    _monkeypatch_env(monkeypatch, tmp_path)
    from iai_mcp import working_tier as wt

    wt.open_task("triage a flaky test")

    session_id = "sess-populate-probe"
    sensory_append(session_id, "user", "the flaky test fails only under load")

    wt.populate_from_sensory(session_id)
    entry = wt.read_task()
    assert entry is not None
    joined_raw = " ".join(entry.raw_sensory)
    assert "the flaky test fails only under load" in joined_raw

    record = _make_record("recall surfaced a similar flaky test from last week")
    wt.update_from_record(record)

    entry_after = wt.read_task()
    assert entry_after is not None
    joined_after = " ".join(entry_after.raw_sensory)
    assert "recall surfaced a similar flaky test from last week" in joined_after


def test_read_task_reflects_sensory_tail_without_promotion():
    """read_task() reflects raw sensory turns via the live sensory_pending()
    tail fold, WITHOUT any store.insert / promotion ever firing.

    Proves per-turn continuity does not depend on the promotion cadence:
    freshness comes from read_task()'s live sensory_pending() fold, not
    from the promoted-insert feed seam. Pure in-RAM — no driver param, and
    store.insert is never called in this test.
    """
    from iai_mcp import working_tier as wt

    wt.open_task("chase down a memory leak")

    session_id = "sess-freshness-probe"
    sensory_append(session_id, "user", "RSS climbed steadily over the last hour")
    sensory_append(session_id, "assistant", "checking the hnswlib rebuild path for leaks")

    wt.populate_from_sensory(session_id)
    entry = wt.read_task()
    assert entry is not None
    joined = " ".join(entry.raw_sensory)
    assert "RSS climbed steadily over the last hour" in joined
    assert "checking the hnswlib rebuild path for leaks" in joined


# ---------------------------------------------------------------------------
# SC-c: task-scoped — created per task, cleared on completion, distinct
# from the episodic store, consolidates durable results
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_close_task_consolidates_and_clears(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    _monkeypatch_env(monkeypatch, tmp_path)
    from iai_mcp import working_tier as wt

    store = make_tmp_store(tmp_path / f"store-{driver}")
    try:
        wt.open_task("resolve the daemon boot-perf regression")
        wt.update_task(result="durable finding Y: boot rebuild was blocking on the awake lock")

        result = wt.close_task(store)
        assert result is not None

        found = [
            r for r in store.iter_records(where="tier = 'episodic'")
            if "durable finding Y" in (r.literal_surface or "")
        ]
        assert len(found) >= 1, (
            f"[driver={driver}] expected a Hippo record carrying the "
            f"consolidated result; found {len(found)}"
        )

        assert wt.read_task() is None, "expected task cleared after close_task()"
        assert getattr(store, "_working_set", None) is None, (
            "working tier must not be reachable through any store.* attribute"
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# SC-d: bounded + never on the awake recall critical path, both drivers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_recall_path_no_working_tier_calls(driver, tmp_path, monkeypatch):
    """memory_recall dispatch never touches the working tier, on either driver.

    Also asserts the ~4±1-slot bound: WORKING_TIER_MAX_SLOTS is enforced on
    the dataclass lists.
    """
    _select_driver(driver, monkeypatch)
    _monkeypatch_env(monkeypatch, tmp_path)
    from iai_mcp import working_tier as wt

    def _boom(*args, **kwargs):
        raise AssertionError(
            "working-tier function called during memory_recall dispatch"
        )

    monkeypatch.setattr(wt, "read_task", _boom)
    monkeypatch.setattr(wt, "update_from_record", _boom)
    monkeypatch.setattr(wt, "populate_from_sensory", _boom)

    store_path = tmp_path / f"driver-{driver}-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_path)
    try:
        _seed_one_record(store, "reference content for working-tier recall no-op probe")

        resp = core.dispatch(store, "memory_recall", {
            "cue": "reference content",
            "session_id": f"sess-noop-{driver}",
            "cue_embedding": [0.1] * EMBED_DIM,
        })
        assert "error" not in resp, (
            f"[driver={driver}] recall dispatch errored: {resp.get('error')}"
        )
    finally:
        store.close()

    # WORKING_TIER_MAX_SLOTS bound: opening a fresh, un-monkeypatched task
    # and overflowing the results list must raise or bounded-evict, never
    # grow past the cap.
    monkeypatch.undo()
    from iai_mcp import working_tier as wt2
    wt2._reset()
    wt2.open_task("bound probe task")
    overflowed = False
    try:
        for i in range(wt2.WORKING_TIER_MAX_SLOTS + 3):
            wt2.update_task(result=f"result number {i} for bound probe overflow check")
    except ValueError:
        overflowed = True
    entry = wt2.read_task()
    if not overflowed:
        assert entry is not None
        assert len(entry.results) <= wt2.WORKING_TIER_MAX_SLOTS, (
            "WORKING_TIER_MAX_SLOTS must bound the results list "
            "(silent-bounded-eviction or a raised ValueError)"
        )


# ---------------------------------------------------------------------------
# SC-e: verbatim/lossless holds across population/consult/consolidation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_consolidation_preserves_literal_surface(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    _monkeypatch_env(monkeypatch, tmp_path)
    from iai_mcp import working_tier as wt

    store = make_tmp_store(tmp_path / f"store-verbatim-{driver}")
    try:
        irregular = (
            f"Odd   Spacing,MixedCase!!  and\ttabs -- {driver} working-tier verbatim probe??"
        )
        wt.open_task("verify verbatim consolidation")
        wt.update_task(result=irregular)

        wt.close_task(store)

        episodic = list(store.iter_records(where="tier = 'episodic'"))
        matches = [r for r in episodic if r.literal_surface == irregular]
        candidates = [r.literal_surface for r in episodic if driver in (r.literal_surface or "")]
        assert len(matches) == 1, (
            f"[driver={driver}] expected byte-identical literal_surface "
            f"(modulo strip/truncate) for the consolidated result; "
            f"candidates found: {candidates!r}"
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Idle-gap task boundary: gap > threshold closes+reopens; gap <
# threshold keeps the same task; session-end close consolidates.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_idle_gap_closes_and_reopens_task(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    _monkeypatch_env(monkeypatch, tmp_path)
    from iai_mcp import working_tier as wt

    store = make_tmp_store(tmp_path / f"store-idle-{driver}")
    try:
        # (a) gap > threshold: close+reopen, consolidate-then-clear-then-open.
        wt.open_task("task one: the first investigation")
        wt.update_task(result="finding from task one")
        entry_before = wt.read_task()
        assert entry_before is not None
        last_turn = entry_before.last_turn_ts

        far_future = datetime.fromtimestamp(
            last_turn + wt.WORKING_TIER_IDLE_CLOSE_SEC + 1, tz=timezone.utc
        )
        record_far = _make_record("a brand new turn arriving after a long idle gap", created_at=far_future)
        wt.update_from_record(record_far, store=store)

        found = [
            r for r in store.iter_records(where="tier = 'episodic'")
            if "finding from task one" in (r.literal_surface or "")
        ]
        assert len(found) >= 1, (
            f"[driver={driver}] expected the stale task's result consolidated "
            f"into Hippo on the idle-gap close; found {len(found)}"
        )

        entry_after_gap = wt.read_task()
        assert entry_after_gap is not None, "expected a fresh task opened after the idle-gap close"
        assert entry_after_gap.goal != "task one: the first investigation", (
            "expected a FRESH task after the idle gap, not the prior goal"
        )
        assert "finding from task one" not in entry_after_gap.results, (
            "the prior task's result must not be appended to the fresh task"
        )

        # (b) gap < threshold: same task, folded in, nothing consolidated.
        wt._reset()
        wt.open_task("task two: same-task continuity check")
        entry_b_before = wt.read_task()
        assert entry_b_before is not None
        same_goal = entry_b_before.goal
        last_turn_b = entry_b_before.last_turn_ts

        near_future = datetime.fromtimestamp(
            last_turn_b + wt.WORKING_TIER_IDLE_CLOSE_SEC - 1, tz=timezone.utc
        )
        record_near = _make_record("a turn arriving well within the idle window", created_at=near_future)
        wt.update_from_record(record_near, store=store)

        entry_b_after = wt.read_task()
        assert entry_b_after is not None
        assert entry_b_after.goal == same_goal, (
            "a gap below the idle threshold must keep the SAME task"
        )

        # (c) session-end close consolidates a SHORT (< MIN_CAPTURE_LEN)
        # durable result BYTE-IDENTICAL — no padding, no internal label.
        wt.update_task(result="final")
        close_result = wt.close_task(store)
        assert close_result is not None
        episodic = list(store.iter_records(where="tier = 'episodic'"))
        exact = [r for r in episodic if r.literal_surface == "final"]
        assert len(exact) >= 1, (
            f"[driver={driver}] a short durable result must be stored "
            f"byte-identical; surfaces: "
            f"{[r.literal_surface for r in episodic]!r}"
        )
        # No internal consolidation label may ever reach a stored surface.
        assert not any(
            "working-tier consolidation" in (r.literal_surface or "")
            for r in episodic
        ), (
            f"[driver={driver}] internal consolidation label leaked into a "
            f"stored literal_surface"
        )
        assert wt.read_task() is None
    finally:
        store.close()


# ---------------------------------------------------------------------------
# CR-01 regression: a short durable result is stored verbatim, no label leak
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_short_durable_result_stored_byte_identical(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    _monkeypatch_env(monkeypatch, tmp_path)
    from iai_mcp import working_tier as wt

    store = make_tmp_store(tmp_path / f"store-short-{driver}")
    try:
        wt.open_task("verify a short durable result survives verbatim")
        wt.update_task(result="ok")  # 2 chars — well below MIN_CAPTURE_LEN
        wt.close_task(store)

        episodic = list(store.iter_records(where="tier = 'episodic'"))
        assert any(r.literal_surface == "ok" for r in episodic), (
            f"[driver={driver}] short result 'ok' must be stored byte-identical; "
            f"surfaces: {[r.literal_surface for r in episodic]!r}"
        )
        assert not any(
            "consolidation" in (r.literal_surface or "") for r in episodic
        ), f"[driver={driver}] internal label leaked into stored content"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# WR-01 regression: an out-of-order (older) record does not rewind the clock
# ---------------------------------------------------------------------------

def test_out_of_order_record_does_not_rewind_idle_clock():
    """A late-flushed older record must not rewind last_turn_ts nor spuriously
    close the task. Pure in-RAM — no driver param, no store.insert."""
    from iai_mcp import working_tier as wt

    now = datetime.now(timezone.utc)
    wt.update_from_record(_make_record("open the task with the current turn", created_at=now))
    entry = wt.read_task()
    assert entry is not None
    ts_after_first = entry.last_turn_ts

    # An older record arrives out of order (e.g. async write-queue reorder).
    older = now - timedelta(seconds=wt.WORKING_TIER_IDLE_CLOSE_SEC + 10)
    wt.update_from_record(_make_record("a stale turn delivered late", created_at=older))

    entry_after = wt.read_task()
    assert entry_after is not None
    assert entry_after.last_turn_ts == ts_after_first, (
        "an out-of-order older record must not rewind the idle clock"
    )
    # The older record must NOT have spuriously closed and reopened the task.
    assert entry_after.goal == entry.goal, (
        "an out-of-order older record must not close/reopen the task"
    )


# ---------------------------------------------------------------------------
# WR-02 regression: a mid-loop store error surfaces undelivered results
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_partial_consolidation_failure_is_observable(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    _monkeypatch_env(monkeypatch, tmp_path)
    from iai_mcp import capture as cap
    from iai_mcp import working_tier as wt

    store = make_tmp_store(tmp_path / f"store-partial-{driver}")
    try:
        wt.open_task("verify a partial consolidation failure is observable")
        wt.update_task(result="first durable result that should persist fine")
        wt.update_task(result="second durable result that hits the injected error")

        calls = {"n": 0}
        real_capture_turn = cap.capture_turn

        def _flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("injected mid-loop store error")
            return real_capture_turn(*args, **kwargs)

        monkeypatch.setattr(cap, "capture_turn", _flaky)

        result = wt.close_task(store)
        assert result["status"] == "partial", (
            f"[driver={driver}] a mid-loop failure must yield a partial status, "
            f"got {result!r}"
        )
        assert result["undelivered"], (
            f"[driver={driver}] the undelivered results must be surfaced, "
            f"never silently dropped"
        )
        assert any(
            "second durable result" in u for u in result["undelivered"]
        ), f"[driver={driver}] the failed result must appear in undelivered"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# WR-03 regression: the first folded turn seeds a verbatim goal
# ---------------------------------------------------------------------------

def test_first_turn_seeds_verbatim_goal():
    """The wired write-seam open path derives goal from the first turn's
    verbatim content, not the empty string. Pure in-RAM — no driver param."""
    from iai_mcp import working_tier as wt

    first = "investigate why warm recall p95 regressed after the boot rebuild"
    wt.update_from_record(_make_record(first))

    entry = wt.read_task()
    assert entry is not None
    assert entry.goal == first[:wt.WORKING_TIER_MAX_GOAL_CHARS], (
        "the fresh task's goal must be the verbatim (bounded) first-turn content"
    )
    assert entry.goal != "", "a write-seam-opened task must not be goal-less"
