"""The nightly re-exposure probe scheduler: opens at most once per interval
on an unpinned blank_recall knob, resets the posterior durably, and clears
its own marker after the run that consumed (or failed to consume) it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from iai_mcp import core
from iai_mcp.events import query_events, write_event
from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
from iai_mcp.lilli.cycle.sleep_pipeline._knob_tune_specs import (
    MIN_PROBE_FOLLOW_SESSIONS,
    PROBE_INTERVAL_NIGHTS,
    seed_incumbent_posterior,
    _apply_task_support_tuning,
)
from iai_mcp.lilli.profile.knobs import default_state, profile_set
from iai_mcp.lilli.profile.persistence import load_profile_state, save_profile_state
from iai_mcp.store import MemoryStore

_DRIVER_PARAMS = [
    pytest.param("stdlib", id="stdlib"),
    pytest.param("lilli", id="lilli"),
]


def _set_driver(monkeypatch: pytest.MonkeyPatch, driver: str) -> None:
    if driver == "stdlib":
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    else:
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", driver)


@pytest.fixture(autouse=True)
def _restore_live_profile():
    saved_state = dict(core._profile_state)
    saved_posterior = dict(core._posterior_state)
    saved_hydrated = set(core._profile_hydrated_stores)
    saved_probe_until = core._task_support_probe_active_until
    core._profile_hydrated_stores.clear()
    core.set_task_support_probe_active_until(None)
    yield
    core._profile_state.clear()
    core._profile_state.update(saved_state)
    core._posterior_state.clear()
    core._posterior_state.update(saved_posterior)
    core._profile_hydrated_stores.clear()
    core._profile_hydrated_stores.update(saved_hydrated)
    core.set_task_support_probe_active_until(saved_probe_until)


def _seed_blank_recall_unpinned(store: MemoryStore) -> None:
    knobs = default_state()
    knobs["task_support"] = "blank_recall"
    save_profile_state(store, knobs=knobs, posterior={}, pins={})


def _write_retrieval_row(
    store: MemoryStore, *, session_id: str, ts: datetime,
    hit_ids: list[str], suggestion_ids: list[str], probe: bool,
) -> None:
    write_event(
        store,
        kind="retrieval_used",
        data={
            "hit_ids": hit_ids,
            "query": "cue",
            "used": True,
            "budget_used": 10,
            "path": "recall_for_response",
            "session_id": session_id,
            "timestamp": ts.isoformat(),
            "suggestion_ids": suggestion_ids,
            "suggestions_visible": True,
            "probe": probe,
        },
        severity="info",
        session_id=session_id,
        buffered=False,
    )


def _write_probe_session(
    store: MemoryStore, *, prefix: str, i: int, base: datetime, followed: bool,
) -> None:
    sid = f"{prefix}-{i}"
    t0 = base + timedelta(minutes=10 * i)
    shown = f"sugg-{prefix}-{i}"
    _write_retrieval_row(
        store, session_id=sid, ts=t0, hit_ids=[f"hit-{prefix}-{i}-a"],
        suggestion_ids=[shown], probe=True,
    )
    second_hit = shown if followed else f"hit-{prefix}-{i}-unrelated"
    _write_retrieval_row(
        store, session_id=sid, ts=t0 + timedelta(minutes=1),
        hit_ids=[second_hit], suggestion_ids=[], probe=True,
    )


# ---------------------------------------------------------------------------
# 1. DURABILITY: probe-open reset survives fresh-process hydration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_probe_open_reset_survives_fresh_process_hydration(
    tmp_path, monkeypatch, driver,
) -> None:
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path)
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: now)
    _seed_blank_recall_unpinned(store)

    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")
    done, payload = pipe._step_knob_tune(None)
    assert done is True
    assert payload["knobs_skipped"].get("task_support") == "skipped_no_signal"

    blob = load_profile_state(store)
    assert blob is not None
    ts_posterior = blob["posterior"].get("task_support", {})
    assert "alphas" not in ts_posterior, "the probe-open reset must clear accumulated mass"
    assert ts_posterior.get("probe_active_until")
    assert query_events(store, kind="task_support_probe", limit=5), (
        "the probe-open event must be durable"
    )
    assert core.task_support_probe_active(), (
        "the SAME process must see the probe without waiting for a restart"
    )

    store.close()
    core._profile_state.clear()
    core._profile_state.update(default_state())
    core._posterior_state.clear()
    core.set_task_support_probe_active_until(None)

    store_b = MemoryStore(path=tmp_path)
    result = core.ensure_profile_hydrated(store_b)
    assert result["hydrated"] is True
    assert "alphas" not in core._posterior_state.get("task_support", {})
    assert core._posterior_state["task_support"].get("probe_active_until")
    assert core.task_support_probe_active(), (
        "a fresh process must reload the probe expiry from the durable event"
    )


# ---------------------------------------------------------------------------
# 2. PIN: a pinned task_support opens no probe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_pinned_task_support_opens_no_probe(tmp_path, monkeypatch, driver) -> None:
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path)
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: now)
    local_state = default_state()
    profile_set("task_support", "blank_recall", local_state, store=store)

    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")
    done, payload = pipe._step_knob_tune(None)

    assert done is True
    assert payload["knobs_skipped"]["task_support"] == "skipped_pinned_by_user"
    assert not query_events(store, kind="task_support_probe", limit=5)


# ---------------------------------------------------------------------------
# 3. CADENCE: cadence not elapsed => no new probe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_cadence_not_elapsed_opens_no_new_probe(tmp_path, monkeypatch, driver) -> None:
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path)
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: now)
    _seed_blank_recall_unpinned(store)

    prior_opened = now - timedelta(days=1)
    write_event(
        store, kind="task_support_probe",
        data={
            "active_until": (prior_opened + timedelta(hours=48)).isoformat(),
            "opened_night": prior_opened.isoformat(),
        },
        severity="info", buffered=False,
    )
    assert len(query_events(store, kind="task_support_probe", limit=5)) == 1

    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")
    done, _payload = pipe._step_knob_tune(None)

    assert done is True
    events = query_events(store, kind="task_support_probe", limit=5)
    assert len(events) == 1, "cadence not elapsed must not open a second probe"


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_cadence_elapsed_after_interval_opens_a_new_probe(tmp_path, monkeypatch, driver) -> None:
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path)
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: now)
    _seed_blank_recall_unpinned(store)

    prior_opened = now - timedelta(days=PROBE_INTERVAL_NIGHTS, hours=1)
    write_event(
        store, kind="task_support_probe",
        data={
            "active_until": (prior_opened + timedelta(hours=48)).isoformat(),
            "opened_night": prior_opened.isoformat(),
        },
        severity="info", buffered=False,
    )

    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")
    done, _payload = pipe._step_knob_tune(None)

    assert done is True
    events = query_events(store, kind="task_support_probe", limit=5)
    assert len(events) == 2, "an elapsed cadence must open a fresh probe"


# ---------------------------------------------------------------------------
# 4. STEP-LEVEL RECOVERY: a closed probe with real follow-through moves the
#    knob through the real step, and the marker is gone afterward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_step_level_recovery_moves_knob_and_clears_marker(tmp_path, monkeypatch, driver) -> None:
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path)
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: now)

    knobs = default_state()
    knobs["task_support"] = "blank_recall"
    opened_at = now - timedelta(days=3)
    active_until = opened_at + timedelta(hours=48)  # already expired
    posterior = {"task_support": {"probe_active_until": active_until.isoformat()}}
    save_profile_state(store, knobs=knobs, posterior=posterior, pins={})
    write_event(
        store, kind="task_support_probe",
        data={"active_until": active_until.isoformat(), "opened_night": opened_at.isoformat()},
        severity="info", buffered=False,
    )
    # The step reads the pending marker off core._posterior_state (the live
    # global), not off the durable blob directly -- hydrate it in, the way
    # a real process would have from the night the probe opened.
    assert core.ensure_profile_hydrated(store)["hydrated"] is True

    base = now - timedelta(hours=40)
    for i in range(MIN_PROBE_FOLLOW_SESSIONS):
        _write_probe_session(store, prefix="recov", i=i, base=base, followed=True)

    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")
    done, payload = pipe._step_knob_tune(None)

    assert done is True
    events = query_events(store, kind="profile_tuned", limit=5)
    rows = {r["knob"]: r for r in events[0]["data"]["knobs"]}
    assert rows["task_support"]["reason"] == "moved"
    assert rows["task_support"]["to"] == "cued_recognition"
    assert "task_support" in payload["knobs_moved"]

    blob = load_profile_state(store)
    assert blob["knobs"]["task_support"] == "cued_recognition"
    assert "probe_active_until" not in blob["posterior"].get("task_support", {})


# ---------------------------------------------------------------------------
# 5. RE-ARM after an EMPTY probe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_empty_probe_does_not_move_and_re_arms_incumbent_defense(tmp_path, monkeypatch, driver) -> None:
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path)
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: now)

    knobs = default_state()
    knobs["task_support"] = "blank_recall"
    opened_at = now - timedelta(days=3)
    active_until = opened_at + timedelta(hours=48)
    posterior = {"task_support": {"probe_active_until": active_until.isoformat()}}
    save_profile_state(store, knobs=knobs, posterior=posterior, pins={})
    write_event(
        store, kind="task_support_probe",
        data={"active_until": active_until.isoformat(), "opened_night": opened_at.isoformat()},
        severity="info", buffered=False,
    )
    assert core.ensure_profile_hydrated(store)["hydrated"] is True

    # Only one probe session -- below MIN_PROBE_FOLLOW_SESSIONS.
    base = now - timedelta(hours=40)
    _write_probe_session(store, prefix="empty", i=0, base=base, followed=True)

    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")
    done, payload = pipe._step_knob_tune(None)

    assert done is True
    assert payload["knobs_skipped"]["task_support"] == "skipped_no_signal"
    assert "task_support" not in payload["knobs_moved"]

    blob = load_profile_state(store)
    assert blob["knobs"]["task_support"] == "blank_recall"
    ts_posterior = blob["posterior"].get("task_support", {})
    assert "probe_active_until" not in ts_posterior, (
        "the expired marker must be cleared post-loop even on an empty probe"
    )

    # Re-arm: the marker's absence means seed_incumbent_posterior no longer
    # early-returns, so the incumbent gets defended again.
    reseed_target = dict(ts_posterior)
    seeded = seed_incumbent_posterior("task_support", "blank_recall", reseed_target)
    assert seeded.get("alphas", {}).get("blank_recall") is not None, (
        "an empty posterior after the marker clear must re-arm the incumbent seed"
    )

    # The stale-marker leak is closed: without the marker, apply structurally
    # cannot up-set the knob on a thin/false cued_recognition observation.
    thin_verdict = _apply_task_support_tuning("blank_recall", "cued_recognition", ts_posterior)
    assert thin_verdict == "blank_recall"
