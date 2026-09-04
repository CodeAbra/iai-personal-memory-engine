"""step_knob_tune's parallel retrieval-weight observe/apply: proven
non-interfering with the sealed 10-knob registry, observable on skip,
own-key-persisting on apply, and deterministic on the pinned nearest-
preceding pairing rule.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from iai_mcp import core
from iai_mcp.core._query_dispatch import EVENTS_QUERY_WHITELIST
from iai_mcp.events import query_events, write_event
from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline, _knob_tune
from iai_mcp.lilli.cycle.sleep_pipeline._knob_tune import _pair_retrieval_feedback
from iai_mcp.lilli.cycle.sleep_pipeline._knob_tune_specs import MIN_SESSIONS_FOR_WAKE_DEPTH
from iai_mcp.lilli.profile.knobs import default_state
from iai_mcp.lilli.profile.persistence import load_profile_state
from iai_mcp.lilli.profile.retrieval_tuning import (
    RETRIEVAL_MIN_SAMPLES,
    load_retrieval_weights_state,
    observe_retrieval_weight,
)
from iai_mcp.store import MemoryStore

_DRIVER_PARAMS = [
    pytest.param("stdlib", id="stdlib"),
    pytest.param("lilli", id="lilli"),
]


def _set_driver(monkeypatch: pytest.MonkeyPatch, driver: str) -> None:
    if driver == "stdlib":
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    else:
        pytest.importorskip("iai_mcp_native")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", driver)


@pytest.fixture(autouse=True)
def _restore_live_profile():
    """Idiom from test_profile_self_tuning.py -- in-place restore keeps
    core.LIVE_KNOBS aliased to core._profile_state the same way production
    hydration must."""
    saved_state = dict(core._profile_state)
    saved_posterior = dict(core._posterior_state)
    saved_hydrated = set(core._profile_hydrated_stores)
    core._profile_hydrated_stores.clear()
    yield
    core._profile_state.clear()
    core._profile_state.update(saved_state)
    core._posterior_state.clear()
    core._posterior_state.update(saved_posterior)
    core._profile_hydrated_stores.clear()
    core._profile_hydrated_stores.update(saved_hydrated)


def _reset_live_profile_to_defaults() -> None:
    core._profile_state.clear()
    core._profile_state.update(default_state())
    core._posterior_state.clear()


def _seed_sealed_knob_window(store: MemoryStore, now: datetime) -> None:
    """Enough session_started + first_turn_recall traffic to clear
    wake_depth's min_samples gate -- gives the per-knob loop real signal to
    evaluate, so the byte-identical comparison below is not vacuous."""
    base = now - timedelta(minutes=30)
    for i in range(MIN_SESSIONS_FOR_WAKE_DEPTH):
        sid = f"alice-{i}"
        ts = base + timedelta(seconds=60 * i)
        write_event(
            store, kind="session_started",
            data={"session_id": sid, "total_cached_tokens": 100, "timestamp": ts.isoformat()},
            severity="info", buffered=False,
        )
        write_event(
            store, kind="first_turn_recall",
            data={"session_id": sid, "cue_len": 5}, severity="info", buffered=False,
        )


def _emit_used(store: MemoryStore, sid: str, hit_ids: list[str]) -> None:
    write_event(
        store, kind="retrieval_used",
        data={"hit_ids": hit_ids, "query": "q", "used": True, "budget_used": 1, "path": "baseline_recall"},
        severity="info", session_id=sid, buffered=False,
    )


def _emit_reinforced(store: MemoryStore, sid: str, reinforced_ids: list[str]) -> None:
    write_event(
        store, kind="retrieval_reinforced",
        data={
            "session_id": sid, "reinforced_ids": reinforced_ids,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        severity="info", session_id=sid, buffered=False,
    )


def _seed_paired_retrieval_window(store: MemoryStore, n: int, use_rate: float = 1.0) -> None:
    """n paired (retrieval_used -> retrieval_reinforced) sessions, each with
    10 hits and a use_rate fraction of them reinforced -- clears
    RETRIEVAL_MIN_SAMPLES when n >= it."""
    reinforced_count = round(use_rate * 10)
    for i in range(n):
        sid = f"bob-{i}"
        hit_ids = [str(uuid4()) for _ in range(10)]
        _emit_used(store, sid, hit_ids)
        time.sleep(0.001)
        _emit_reinforced(store, sid, hit_ids[:reinforced_count])
        time.sleep(0.001)


# ---------------------------------------------------------------------------
# Non-interference: byte-identical sealed-registry output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_sealed_knob_output_byte_identical_with_retrieval_logic_active(
    tmp_path, monkeypatch, driver,
) -> None:
    _set_driver(monkeypatch, driver)
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: now)

    _reset_live_profile_to_defaults()
    store_off = MemoryStore(path=tmp_path / "off")
    _seed_sealed_knob_window(store_off, now)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_knob_tune, "tune_retrieval_weight", lambda *a, **kw: {"skipped": True})
        pipe_off = SleepPipeline(store_off, lifecycle_state_path=tmp_path / "off-lifecycle.json")
        done_off, payload_off = pipe_off._step_knob_tune(None)
    assert done_off is True
    blob_off = load_profile_state(store_off)
    events_off = query_events(store_off, kind="profile_tuned", limit=10)

    _reset_live_profile_to_defaults()
    store_on = MemoryStore(path=tmp_path / "on")
    _seed_sealed_knob_window(store_on, now)
    _seed_paired_retrieval_window(store_on, RETRIEVAL_MIN_SAMPLES, use_rate=1.0)
    pipe_on = SleepPipeline(store_on, lifecycle_state_path=tmp_path / "on-lifecycle.json")
    done_on, payload_on = pipe_on._step_knob_tune(None)
    assert done_on is True
    blob_on = load_profile_state(store_on)
    events_on = query_events(store_on, kind="profile_tuned", limit=10)

    assert blob_on["knobs"] == blob_off["knobs"]
    assert blob_on["posterior"] == blob_off["posterior"]
    assert events_on[0]["data"]["knobs"] == events_off[0]["data"]["knobs"]
    assert events_on[0]["data"]["moved_count"] == events_off[0]["data"]["moved_count"]
    assert payload_on["knobs_moved"] == payload_off["knobs_moved"]
    assert payload_on["knobs_skipped"] == payload_off["knobs_skipped"]
    assert payload_on["persisted"] == payload_off["persisted"]

    # And prove the retrieval-weight apply path actually ran in the "on"
    # run -- the byte-identical comparison above must not pass vacuously
    # because the logic never fired.
    weights_on = load_retrieval_weights_state(store_on)
    assert weights_on["W_COSINE"] != 1.0


# ---------------------------------------------------------------------------
# The skip/apply event must be reachable through the operator-facing
# events_query dispatch, not just the internal query_events API -- an event
# absent from the whitelist is invisible to the surface an operator actually
# uses, defeating the "queryable" claim.
# ---------------------------------------------------------------------------


def test_retrieval_weight_tuned_is_whitelisted_for_events_query() -> None:
    assert "retrieval_weight_tuned" in EVENTS_QUERY_WHITELIST


def test_events_query_dispatch_accepts_retrieval_weight_tuned(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)

    result = core.dispatch(store, "events_query", {"kind": "retrieval_weight_tuned"})

    assert "error" not in result


# ---------------------------------------------------------------------------
# Skip path: observable below the floor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_skip_path_emits_observable_event_below_floor(tmp_path, monkeypatch, driver) -> None:
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path)
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: now)
    _seed_paired_retrieval_window(store, RETRIEVAL_MIN_SAMPLES - 1, use_rate=1.0)

    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")
    done, _payload = pipe._step_knob_tune(None)
    assert done is True

    events = query_events(store, kind="retrieval_weight_tuned", limit=10)
    assert events
    data = events[0]["data"]
    assert data["skipped"] is True
    assert data["n"] == RETRIEVAL_MIN_SAMPLES - 1
    assert data["min_samples"] == RETRIEVAL_MIN_SAMPLES
    assert data["w_cosine"] == 1.0

    weights = load_retrieval_weights_state(store)
    assert weights["W_COSINE"] == 1.0


# ---------------------------------------------------------------------------
# Apply path: own-key persist, absent from sealed profile knobs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_apply_path_persists_under_dedicated_key_not_sealed_knobs(
    tmp_path, monkeypatch, driver,
) -> None:
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path)
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: now)
    _seed_sealed_knob_window(store, now)
    _seed_paired_retrieval_window(store, RETRIEVAL_MIN_SAMPLES, use_rate=1.0)

    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")
    done, _payload = pipe._step_knob_tune(None)
    assert done is True

    events = query_events(store, kind="retrieval_weight_tuned", limit=10)
    assert events
    data = events[0]["data"]
    assert data["skipped"] is False
    assert data["n"] == RETRIEVAL_MIN_SAMPLES
    assert data["w_cosine"] > 1.0

    weights = load_retrieval_weights_state(store)
    assert weights["W_COSINE"] == data["w_cosine"]

    blob = load_profile_state(store)
    assert blob is not None
    assert "W_COSINE" not in blob["knobs"], (
        "the tuned retrieval weight must never appear in the sealed profile knobs dict"
    )


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_nightly_tune_invalidates_cache_without_a_manual_call(
    tmp_path, monkeypatch, driver,
) -> None:
    """The nightly tune's own invalidate() call must make the very NEXT
    recall observe the new weight with no caller invalidating the cache
    manually -- the whole point of a live feedback loop is that it takes
    effect without a process restart."""
    from iai_mcp import retrieval_weight_cache
    from iai_mcp.types import EMBED_DIM, MemoryRecord
    from tests._helpers import stub_embedder_for_store

    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path)
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: now)
    _seed_sealed_knob_window(store, now)
    _seed_paired_retrieval_window(store, RETRIEVAL_MIN_SAMPLES, use_rate=1.0)

    vec = [0.0] * EMBED_DIM
    vec[0] = 1.0

    class _FixedEmbedder:
        def embed(self, text: str) -> list[float]:
            return list(vec)

    stub_embedder_for_store(monkeypatch, _FixedEmbedder())
    rec = MemoryRecord(
        id=uuid4(), tier="episodic", literal_surface="cache invalidation probe", aaak_index="",
        embedding=vec, community_id=None, centrality=0.0, detail_level=2, pinned=False,
        stability=0.0, difficulty=0.0, last_reviewed=None, never_decay=False, never_merge=False,
        provenance=[], created_at=now, updated_at=now, tags=[], language="en",
    )
    store.insert(rec)

    # Warm the cache with the pre-tune weight -- the state a long-lived
    # daemon process is already in before its own nightly sleep cycle runs.
    pre_weights = retrieval_weight_cache.load(store)
    assert pre_weights["W_COSINE"] == 1.0

    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")
    done, _payload = pipe._step_knob_tune(None)
    assert done is True

    persisted = load_retrieval_weights_state(store)
    assert persisted["W_COSINE"] != 1.0, (
        "the tune must have actually moved the weight for this control to be meaningful"
    )

    monkeypatch.setenv("IAI_MCP_RECALL_RUST_SCORER_OFF", "1")
    monkeypatch.setenv("IAI_MCP_EXACT_AUTHORITY_OFF", "1")
    resp = core.dispatch(store, "memory_recall", {
        "cue": "cache invalidation probe", "session_id": "cr-01-probe",
        "budget_tokens": 2000, "cue_embedding": vec,
    })
    assert "error" not in resp
    assert resp["hits"], "expected a hit for the cosine=1.0 probe record"
    reason = resp["hits"][0]["reason"]
    expected_coef = f"*{persisted['W_COSINE']:g}"
    assert expected_coef in reason, (
        f"nightly tune persisted W_COSINE={persisted['W_COSINE']} but the very next recall "
        f"(no manual invalidate) still served the stale cached weight: {reason!r}"
    )


# ---------------------------------------------------------------------------
# Pinned pairing rule -- deterministic, expected-value assertions
# ---------------------------------------------------------------------------


def _row(ts: datetime, session_id: str, data: dict) -> dict:
    return {"ts": ts, "session_id": session_id, "data": data}


def test_pairing_selects_nearest_preceding_recall_and_intersects_used_ids() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    hit_ids_1 = [f"h1-{i}" for i in range(3)]
    hit_ids_2 = [f"h2-{i}" for i in range(3)]
    stray_id = "stray-not-in-either-recall"

    used_rows = [
        _row(t0, "s1", {"hit_ids": hit_ids_1}),
        _row(t0 + timedelta(seconds=10), "s1", {"hit_ids": hit_ids_2}),
    ]
    # reinforced_ids intersect ONLY the later (nearest-preceding) recall's
    # hits, plus one stray id present in neither recall's hit set.
    reinforced_rows = [
        _row(
            t0 + timedelta(seconds=20), "s1",
            {"reinforced_ids": [hit_ids_2[0], hit_ids_2[1], stray_id]},
        ),
    ]

    window_rows = _pair_retrieval_feedback(reinforced_rows, used_rows)

    assert len(window_rows) == 1
    row = window_rows[0]
    assert set(row["hit_ids"]) == set(hit_ids_2), (
        "must pair to the LAST recall, not the earlier one"
    )
    assert set(row["reinforced_ids"]) == {hit_ids_2[0], hit_ids_2[1]}, (
        "used_ids must be the intersection with the paired recall's hits -- "
        "the stray id excluded so use_rate cannot exceed 1.0"
    )

    observed, n, _signal = observe_retrieval_weight(window_rows)
    assert n == 1
    assert observed == pytest.approx(2 / 3)


def test_pairing_drops_reinforce_with_no_preceding_recall_in_session() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    used_rows: list[dict] = []  # no recall at all in this session
    reinforced_rows = [_row(t0, "lonely-session", {"reinforced_ids": ["x"]})]

    window_rows = _pair_retrieval_feedback(reinforced_rows, used_rows)

    assert window_rows == []
    observed, n, _signal = observe_retrieval_weight(window_rows)
    assert n == 0
    assert observed == 0.0


def test_pairing_drops_dash_sentinel_session() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    used_rows = [_row(t0, "-", {"hit_ids": ["h1", "h2"]})]
    reinforced_rows = [_row(t0 + timedelta(seconds=5), "-", {"reinforced_ids": ["h1"]})]

    window_rows = _pair_retrieval_feedback(reinforced_rows, used_rows)

    assert window_rows == []


def test_pairing_many_reinforces_can_pair_to_the_same_recall() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    hit_ids = [f"h-{i}" for i in range(4)]
    used_rows = [_row(t0, "s2", {"hit_ids": hit_ids})]
    reinforced_rows = [
        _row(t0 + timedelta(seconds=5), "s2", {"reinforced_ids": hit_ids[:2]}),
        _row(t0 + timedelta(seconds=6), "s2", {"reinforced_ids": hit_ids[2:]}),
    ]

    window_rows = _pair_retrieval_feedback(reinforced_rows, used_rows)

    assert len(window_rows) == 2
    observed, n, _signal = observe_retrieval_weight(window_rows)
    assert n == 2
    assert observed == pytest.approx(0.5)
