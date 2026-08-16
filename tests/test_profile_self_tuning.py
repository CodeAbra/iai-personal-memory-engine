"""Persist-on-set durability, pin-at-set-time, and the extensible tuning
spec registry for the autistic-cognition profile.

Assertions on the durable blob read and decrypt the raw ``_hippo_meta`` row
directly -- never through ``profile_get``, which returns the registry
default for a missing key and would make "the write ran" indistinguishable
from "the write never happened".
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from iai_mcp import core
from iai_mcp.core._query_dispatch import EVENTS_QUERY_WHITELIST
from iai_mcp.crypto import decrypt_field, encrypt_field, is_encrypted
from iai_mcp.events import query_events, write_event
from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
from iai_mcp.lilli.profile.knobs import PROFILE_KNOBS, bayesian_update, default_state, profile_set
from iai_mcp.lilli.profile.persistence import (
    PROFILE_BLOB_AAD,
    PROFILE_META_KEY,
    PROFILE_META_ORPHAN_KEY,
    load_profile_state,
    save_profile_state,
)
from iai_mcp.store import MemoryStore

from iai_mcp.lilli.cycle.sleep_pipeline._knob_tune_specs import (
    BOOL_HYSTERESIS_BAND,
    INCUMBENT_SEED_MASS,
    INERTIA_GAP_THRESHOLD_S,
    MAX_EVENTS,
    MIN_SESSIONS_FOR_INERTIA,
    MIN_SESSIONS_FOR_WAKE_DEPTH,
    TUNING_SPECS,
    WAKE_DEPTH_AUTO_CEILING,
    WAKE_DEPTH_LADDER,
    WINDOW_DAYS,
    TuningSpec,
    bool_flip_allowed,
    clamp_enum_step,
    seed_incumbent_posterior,
)


@pytest.fixture(autouse=True)
def _restore_live_profile():
    """Idiom from tests/test_knobs_applied_telemetry.py:250-251 -- in-place
    restore keeps core.LIVE_KNOBS aliased to core._profile_state the same
    way production hydration must."""
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


def _read_decrypted_blob(store: MemoryStore) -> dict:
    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = ?", (PROFILE_META_KEY,)
        ).fetchone()
    assert row is not None, "expected a profile_state row to exist"
    value = row["value"]
    assert is_encrypted(value)
    plaintext = decrypt_field(value, store._key(), associated_data=PROFILE_BLOB_AAD)
    return json.loads(plaintext)


def _write_raw_meta(store: MemoryStore, key: str, value: str) -> None:
    with store.db._conn_lock:
        store.db._conn.execute("DELETE FROM _hippo_meta WHERE key = ?", (key,))
        store.db._conn.execute(
            "INSERT OR IGNORE INTO _hippo_meta (key, value) VALUES (?, ?)", (key, value)
        )
        store.db._conn.commit()


def _read_raw_meta(store: MemoryStore, key: str) -> "str | None":
    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = ?", (key,)
        ).fetchone()
    return row["value"] if row is not None else None


# ---------------------------------------------------------------------------
# Task 1: persist-on-set, pin-at-set-time, source discriminator, whitelist
# ---------------------------------------------------------------------------


def test_user_set_persists_blob_immediately(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    state = default_state()

    result = profile_set("dunn_quadrant", "seeking", state, store=store)

    assert result["status"] == "ok"
    blob = _read_decrypted_blob(store)
    assert blob["knobs"]["dunn_quadrant"] == "seeking"


def test_user_set_pins_knob_at_set_time(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    state = default_state()

    profile_set("dunn_quadrant", "seeking", state, store=store)

    blob = _read_decrypted_blob(store)
    assert "dunn_quadrant" in blob["pins"]
    # Must parse as an ISO-8601 timestamp -- raises ValueError otherwise.
    datetime.fromisoformat(blob["pins"]["dunn_quadrant"])


def test_user_set_survives_fresh_process_hydration(tmp_path) -> None:
    store_a = MemoryStore(path=tmp_path)
    state = default_state()
    profile_set("dunn_quadrant", "seeking", state, store=store_a)
    store_a.close()

    core._profile_state.clear()
    core._profile_state.update(default_state())
    assert core._profile_state["dunn_quadrant"] == "neutral"

    store_b = MemoryStore(path=tmp_path)
    result = core.ensure_profile_hydrated(store_b)

    assert result["hydrated"] is True
    assert core._profile_state["dunn_quadrant"] == "seeking"
    assert core.LIVE_KNOBS is core._profile_state


def test_profile_updated_event_source_defaults_to_user(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    state = default_state()

    profile_set("dunn_quadrant", "seeking", state, store=store)

    events = query_events(store, kind="profile_updated", limit=10)
    assert events
    assert events[0]["data"]["source"] == "user"


def test_profile_updated_event_source_tuner_is_not_pinned(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    state = default_state()

    profile_set("dunn_quadrant", "seeking", state, store=store, source="tuner")

    events = query_events(store, kind="profile_updated", limit=10)
    assert events
    assert events[0]["data"]["source"] == "tuner"
    # A non-user set never writes the durable blob -- no pin, no persist.
    assert load_profile_state(store) is None


def test_pins_accumulate_across_successive_user_sets(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    state = default_state()

    profile_set("dunn_quadrant", "seeking", state, store=store)
    profile_set("masking_off", False, state, store=store)

    blob = _read_decrypted_blob(store)
    assert set(blob["pins"]) == {"dunn_quadrant", "masking_off"}, (
        "a later user set must not collapse an earlier pin"
    )
    assert blob["knobs"]["dunn_quadrant"] == "seeking"
    assert blob["knobs"]["masking_off"] is False


def test_user_set_to_incumbent_value_still_pins(tmp_path) -> None:
    """The pin means 'the user chose this', not 'the value moved' -- a
    user who re-asserts the value the knob already holds is locking it
    against auto-tuning just as much as one who changes it."""
    store = MemoryStore(path=tmp_path)
    state = default_state()
    assert state["dunn_quadrant"] == "neutral"

    result = profile_set("dunn_quadrant", "neutral", state, store=store)

    assert result["status"] == "ok"
    blob = _read_decrypted_blob(store)
    assert "dunn_quadrant" in blob["pins"], (
        "a no-op-valued user set must still record a pin"
    )


def test_user_set_with_failed_persist_does_not_report_bare_ok(
    tmp_path, monkeypatch,
) -> None:
    store = MemoryStore(path=tmp_path)
    state = default_state()

    def _boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(
        "iai_mcp.lilli.profile.persistence.persist_after_user_set", _boom
    )

    result = profile_set("dunn_quadrant", "seeking", state, store=store)

    assert result["status"] != "ok", (
        "a failed durable persist must not be disguised as full success"
    )
    assert result["persisted"] is False
    # The live in-process value still updates this session.
    assert state["dunn_quadrant"] == "seeking"
    events = query_events(store, kind="profile_state_unreadable", limit=10)
    assert events, "a failed persist must surface a warning event"
    assert events[-1]["data"]["reason"] == "persist_after_user_set_failed"
    assert events[-1]["severity"] == "warning"


def test_user_set_with_persist_returning_false_does_not_report_bare_ok(
    tmp_path, monkeypatch,
) -> None:
    """The non-exceptional failure path: save_profile_state returns False
    (e.g. store.db is not a real HippoDB) rather than raising."""
    store = MemoryStore(path=tmp_path)
    state = default_state()

    monkeypatch.setattr(
        "iai_mcp.lilli.profile.persistence.persist_after_user_set",
        lambda *_args, **_kwargs: False,
    )

    result = profile_set("dunn_quadrant", "seeking", state, store=store)

    assert result["status"] != "ok"
    assert result["persisted"] is False
    events = query_events(store, kind="profile_state_unreadable", limit=10)
    assert events, "a False return from persist_after_user_set must also warn"
    assert events[-1]["data"]["reason"] == "persist_after_user_set_failed"


def test_set_preserves_existing_posterior(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    save_profile_state(
        store,
        knobs=default_state(),
        posterior={"literal_preservation": {"alphas": {"strong": 2.0}}},
        pins={},
    )
    state = default_state()

    profile_set("dunn_quadrant", "seeking", state, store=store)

    blob = _read_decrypted_blob(store)
    assert blob["posterior"] == {"literal_preservation": {"alphas": {"strong": 2.0}}}
    assert blob["knobs"]["dunn_quadrant"] == "seeking"


def test_set_without_store_mutates_state_writes_no_blob() -> None:
    state = default_state()

    result = profile_set("dunn_quadrant", "seeking", state)

    assert result["status"] == "ok"
    assert state["dunn_quadrant"] == "seeking"


def test_events_whitelist_accepts_tuning_kinds() -> None:
    assert "profile_tuned" in EVENTS_QUERY_WHITELIST
    assert "first_turn_recall" in EVENTS_QUERY_WHITELIST


def test_events_query_dispatch_accepts_tuning_kinds(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)

    result = core.dispatch(store, "events_query", {"kind": "profile_tuned"})
    assert "error" not in result
    result = core.dispatch(store, "events_query", {"kind": "first_turn_recall"})
    assert "error" not in result


def test_save_preserves_undecryptable_existing_blob_as_orphan(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    wrong_ct = encrypt_field(
        json.dumps(
            {"version": 1, "knobs": {}, "posterior": {}, "pins": {}, "updated_at": "x"}
        ),
        store._key(),
        associated_data=b"embed_identity",
    )
    _write_raw_meta(store, PROFILE_META_KEY, wrong_ct)

    assert (
        save_profile_state(store, knobs={"masking_off": True}, posterior={}, pins={})
        is True
    )

    orphan = _read_raw_meta(store, PROFILE_META_ORPHAN_KEY)
    assert orphan == wrong_ct

    events = query_events(store, kind="profile_state_unreadable", limit=10)
    assert events
    assert events[0]["severity"] == "warning"
    assert events[0]["data"]["reason"] == "preserved_before_overwrite"

    # The new blob is written and readable.
    blob = load_profile_state(store)
    assert blob is not None
    assert blob["knobs"]["masking_off"] is True


def test_save_never_overwrites_an_earlier_orphan(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    wrong_ct = encrypt_field(
        json.dumps(
            {"version": 1, "knobs": {}, "posterior": {}, "pins": {}, "updated_at": "x"}
        ),
        store._key(),
        associated_data=b"embed_identity",
    )
    _write_raw_meta(store, PROFILE_META_KEY, wrong_ct)
    save_profile_state(store, knobs={"masking_off": True}, posterior={}, pins={})
    first_orphan = _read_raw_meta(store, PROFILE_META_ORPHAN_KEY)
    assert first_orphan == wrong_ct

    # A second undecryptable row appears (e.g. a restore under the wrong key
    # again) -- the FIRST orphan must never be clobbered.
    other_wrong_ct = encrypt_field(
        json.dumps({"version": 1, "knobs": {}, "posterior": {}, "pins": {}}),
        store._key(),
        associated_data=b"some_other_field",
    )
    _write_raw_meta(store, PROFILE_META_KEY, other_wrong_ct)
    save_profile_state(store, knobs={"masking_off": False}, posterior={}, pins={})

    second_orphan = _read_raw_meta(store, PROFILE_META_ORPHAN_KEY)
    assert second_orphan == first_orphan, "the write-once orphan slot must not be replaced"


def test_save_on_readable_existing_blob_does_not_write_orphan(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    save_profile_state(store, knobs={"masking_off": True}, posterior={}, pins={})

    save_profile_state(store, knobs={"masking_off": False}, posterior={}, pins={})

    assert _read_raw_meta(store, PROFILE_META_ORPHAN_KEY) is None


# ---------------------------------------------------------------------------
# Task 2: the extensible TuningSpec registry
# ---------------------------------------------------------------------------


def test_registry_populated_specs_are_signal_backed() -> None:
    assert "wake_depth" in TUNING_SPECS
    assert "inertia_awareness" in TUNING_SPECS
    for name, spec in TUNING_SPECS.items():
        assert spec.knob == name
        assert spec.kinds
        assert spec.min_samples >= 1
        assert callable(spec.observe) and callable(spec.apply)


# User-set-only: none of these may ever gain a TUNING_SPECS entry.
NOT_TUNABLE_KNOBS = (
    "interest_boost",
    "dunn_quadrant",
    "demand_avoidance_tolerance",
    "masking_off",
    "scene_construction_scaffold",
    "literal_preservation",
)


def test_signal_less_knobs_absent_from_tuning_specs() -> None:
    for knob in NOT_TUNABLE_KNOBS:
        assert knob not in TUNING_SPECS, (
            f"{knob} is user-set-only and must never gain a TUNING_SPECS entry"
        )


def test_signal_less_knobs_report_skipped_not_tunable(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")

    done, _payload = pipe._step_knob_tune(None)

    assert done is True
    events = query_events(store, kind="profile_tuned", limit=10)
    assert events
    rows = {row["knob"]: row["reason"] for row in events[0]["data"]["knobs"]}
    for knob in NOT_TUNABLE_KNOBS:
        assert rows[knob] == "skipped_not_tunable"


def test_tuning_spec_min_samples_match_named_constants() -> None:
    assert TUNING_SPECS["wake_depth"].min_samples == MIN_SESSIONS_FOR_WAKE_DEPTH
    assert TUNING_SPECS["inertia_awareness"].min_samples == MIN_SESSIONS_FOR_INERTIA


def test_wake_depth_observe_empty_returns_no_observation() -> None:
    spec = TUNING_SPECS["wake_depth"]

    observed, n, signal = spec.observe({}, current="minimal")

    assert observed is None
    assert n == 0
    assert signal == "implicit"


def test_wake_depth_observe_dedups_by_distinct_session_id() -> None:
    spec = TUNING_SPECS["wake_depth"]
    started = [
        {"session_id": "alice-1", "total_cached_tokens": 100} for _ in range(10)
    ]
    recalled = [{"session_id": "alice-1", "cue_len": 5}]

    observed, n, signal = spec.observe(
        {"session_started": started, "first_turn_recall": recalled}, current="minimal"
    )

    assert n == 1, "ten rows sharing one session_id must count as ONE session of evidence"
    assert observed == "standard"
    assert signal == "implicit"


def test_wake_depth_observe_deeper_on_first_turn_recall() -> None:
    spec = TUNING_SPECS["wake_depth"]
    started = [{"session_id": "alice-1", "total_cached_tokens": 100}]
    recalled = [{"session_id": "alice-1", "cue_len": 5}]

    observed, _n, _signal = spec.observe(
        {"session_started": started, "first_turn_recall": recalled}, current="minimal"
    )

    assert observed == "standard"


def test_wake_depth_observe_never_proposes_deep() -> None:
    spec = TUNING_SPECS["wake_depth"]
    started = [{"session_id": f"alice-{i}", "total_cached_tokens": 0} for i in range(5)]
    recalled = [{"session_id": f"alice-{i}"} for i in range(5)]

    observed, _n, _signal = spec.observe(
        {"session_started": started, "first_turn_recall": recalled}, current="standard"
    )

    assert observed in ("minimal", "standard")
    assert observed != "deep"


def test_inertia_observe_empty_returns_no_observation() -> None:
    spec = TUNING_SPECS["inertia_awareness"]

    observed, n, signal = spec.observe({}, current=False)

    assert observed is None
    assert n == 0
    assert signal == "implicit"


def test_inertia_observe_true_above_gap_threshold() -> None:
    spec = TUNING_SPECS["inertia_awareness"]
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    started = [
        {"session_id": "alice-1", "timestamp": base.isoformat()},
        {
            "session_id": "alice-2",
            "timestamp": (
                base + timedelta(seconds=INERTIA_GAP_THRESHOLD_S * 2)
            ).isoformat(),
        },
    ]

    observed, n, _signal = spec.observe({"session_started": started}, current=False)

    assert observed is True
    assert n == 2


def test_inertia_observe_false_below_gap_threshold() -> None:
    spec = TUNING_SPECS["inertia_awareness"]
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    started = [
        {"session_id": "alice-1", "timestamp": base.isoformat()},
        {
            "session_id": "alice-2",
            "timestamp": (
                base + timedelta(seconds=INERTIA_GAP_THRESHOLD_S / 4)
            ).isoformat(),
        },
    ]

    observed, n, _signal = spec.observe({"session_started": started}, current=False)

    assert observed is False
    assert n == 2


def test_observe_never_raises_on_partial_or_malformed_rows() -> None:
    """A missing session_id/timestamp, a naive timestamp mixed with aware
    ones, and an unparsable timestamp must all degrade to skipped rows, never
    an uncaught exception out of the nightly step."""
    rows = [
        {},
        {"session_id": None, "timestamp": "2026-01-01T00:00:00+00:00"},
        {"session_id": "alice-1"},
        {"session_id": "alice-2", "timestamp": "not-a-date"},
        {"session_id": "alice-3", "timestamp": "2026-01-01T00:00:00"},
        {"session_id": "alice-4", "timestamp": "2026-01-05T00:00:00+00:00"},
    ]
    for name, current in (("wake_depth", "minimal"), ("inertia_awareness", False)):
        spec = TUNING_SPECS[name]
        spec.observe(
            {"session_started": rows, "first_turn_recall": [{}]}, current=current
        )


def test_clamp_enum_step_moves_at_most_one_rung() -> None:
    assert clamp_enum_step("minimal", "deep", WAKE_DEPTH_LADDER) == "standard"
    assert clamp_enum_step("standard", "minimal", WAKE_DEPTH_LADDER) == "minimal"
    assert clamp_enum_step("minimal", "minimal", WAKE_DEPTH_LADDER) == "minimal"


def test_wake_depth_apply_moves_one_rung_and_caps_at_standard() -> None:
    spec = TUNING_SPECS["wake_depth"]

    assert spec.apply("minimal", "deep", {}) == "standard"
    assert spec.apply("standard", "deep", {}) == WAKE_DEPTH_AUTO_CEILING
    assert spec.apply("minimal", "standard", {}) == "standard"


def test_bool_flip_allowed_respects_hysteresis_band() -> None:
    assert bool_flip_allowed(1.0, 1.0) is False
    margin_below = BOOL_HYSTERESIS_BAND - 0.1
    margin_at = BOOL_HYSTERESIS_BAND
    assert bool_flip_allowed(1.0 + margin_below, 1.0) is False
    assert bool_flip_allowed(1.0 + margin_at, 1.0) is True


def test_inertia_apply_flips_only_past_hysteresis_band() -> None:
    spec = TUNING_SPECS["inertia_awareness"]
    weak = {"inertia_awareness": {"alpha": 1.0 + BOOL_HYSTERESIS_BAND - 0.1, "beta": 1.0}}
    strong = {"inertia_awareness": {"alpha": 1.0 + BOOL_HYSTERESIS_BAND, "beta": 1.0}}

    assert spec.apply(False, True, weak) is False
    assert spec.apply(False, True, strong) is True


def test_seed_incumbent_posterior_is_idempotent() -> None:
    posterior: dict = {}

    first = seed_incumbent_posterior("wake_depth", "minimal", posterior)
    assert first == {"alphas": {"minimal": INCUMBENT_SEED_MASS}}
    assert posterior["wake_depth"]["alphas"]["minimal"] == INCUMBENT_SEED_MASS

    posterior["wake_depth"]["alphas"]["standard"] = 1.3
    second = seed_incumbent_posterior("wake_depth", "minimal", posterior)

    assert second is posterior["wake_depth"]
    assert posterior["wake_depth"]["alphas"]["minimal"] == INCUMBENT_SEED_MASS, (
        "re-seeding on an already-evaluated knob must not re-arm the defense"
    )


def test_seed_incumbent_posterior_is_a_noop_for_bool_knobs() -> None:
    posterior: dict = {}

    result = seed_incumbent_posterior("inertia_awareness", False, posterior)

    assert result == {}
    assert "inertia_awareness" not in posterior


def test_incumbent_defended_once_then_overtaken_by_sustained_signal() -> None:
    """A single night of implicit signal cannot beat the seeded incumbent;
    a real signal sustained across ~7 qualifying nights does -- bounded,
    not never."""
    posterior: dict = {}
    state = {"wake_depth": "minimal"}
    seed_incumbent_posterior("wake_depth", "minimal", posterior)

    new_value, posterior = bayesian_update(
        "wake_depth", "implicit", "standard", state, posterior
    )
    assert new_value == "minimal", "a single qualifying night must not move the knob"

    for _ in range(6):
        seed_incumbent_posterior("wake_depth", "minimal", posterior)
        new_value, posterior = bayesian_update(
            "wake_depth", "implicit", "standard", state, posterior
        )

    assert new_value == "standard", (
        "a signal sustained across the expected number of qualifying nights "
        "must overtake the incumbent"
    )


def test_tuning_spec_dataclass_fields_are_declarative() -> None:
    spec = TUNING_SPECS["wake_depth"]
    assert isinstance(spec, TuningSpec)
    assert spec.knob == "wake_depth"
    assert spec.kinds == ("session_started", "first_turn_recall")


def test_named_window_and_event_bound_constants_exist() -> None:
    assert WINDOW_DAYS == 7
    assert MAX_EVENTS > 0


# ---------------------------------------------------------------------------
# The registry-driven KNOB_TUNE step -- composed, real behavior
# ---------------------------------------------------------------------------


def _seed_sessions(
    store: MemoryStore,
    n: int,
    *,
    base: datetime,
    recall_fraction: float = 1.0,
    large_cache: bool = False,
    gap_seconds: float = 60.0,
    session_prefix: str = "alice",
) -> list[str]:
    """Write N distinct session_started events (+ a matching first_turn_recall
    for the leading `recall_fraction` of them), unbuffered so `query_events`
    sees them -- a buffered write sits in the in-memory event buffer and is
    invisible to the tuner, passing a negative control for the wrong reason.
    """
    session_ids: list[str] = []
    recall_count = int(round(n * recall_fraction))
    for i in range(n):
        sid = f"{session_prefix}-{i}"
        session_ids.append(sid)
        ts = base + timedelta(seconds=gap_seconds * i)
        write_event(
            store,
            kind="session_started",
            data={
                "session_id": sid,
                "total_cached_tokens": 3000 if large_cache else 100,
                "timestamp": ts.isoformat(),
            },
            severity="info",
            buffered=False,
        )
        if i < recall_count:
            write_event(
                store,
                kind="first_turn_recall",
                data={"session_id": sid, "cue_len": 5},
                severity="info",
                buffered=False,
            )
    return session_ids


def test_composed_wake_depth_moves_after_sustained_signal_and_survives_reopen(
    tmp_path, monkeypatch,
) -> None:
    store = MemoryStore(path=tmp_path)
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: now)
    base = now - timedelta(minutes=30)
    _seed_sessions(
        store, MIN_SESSIONS_FOR_WAKE_DEPTH,
        base=base, recall_fraction=1.0, gap_seconds=60.0,
    )

    cutoff = now - timedelta(days=WINDOW_DAYS)
    assert len(
        query_events(store, kind="session_started", since=cutoff, limit=1000)
    ) >= MIN_SESSIONS_FOR_WAKE_DEPTH, "anti-vacuity: the seeded signal must be visible before the step runs"

    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")

    # Night one: ample signal, but the incumbent defense holds -- durability
    # lands immediately even though the value has not moved yet.
    done, _payload = pipe._step_knob_tune(None)
    assert done is True
    blob = _read_decrypted_blob(store)
    assert blob["knobs"]["wake_depth"] == "minimal"
    assert "wake_depth" in blob["posterior"]

    moved = False
    for _ in range(11):
        done, _payload = pipe._step_knob_tune(None)
        assert done is True
        blob = _read_decrypted_blob(store)
        if blob["knobs"]["wake_depth"] == "standard":
            moved = True
            break
    assert moved, (
        "sustained qualifying signal must overtake the incumbent within a "
        "bounded number of nights"
    )

    core._profile_state.clear()
    core._profile_state.update(default_state())
    assert core._profile_state["wake_depth"] == "minimal"
    store.close()

    store_b = MemoryStore(path=tmp_path)
    result = core.ensure_profile_hydrated(store_b)
    assert result["hydrated"] is True
    assert core._profile_state["wake_depth"] == "standard"
    assert core.LIVE_KNOBS is core._profile_state


def test_negative_control_no_events_reports_skipped_no_signal(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")

    done, _payload = pipe._step_knob_tune(None)

    assert done is True
    assert load_profile_state(store) is None, "a quiet night must write no profile_state blob"
    assert core._profile_state == default_state()
    events = query_events(store, kind="profile_tuned", limit=10)
    assert events
    rows = {row["knob"]: row["reason"] for row in events[0]["data"]["knobs"]}
    assert rows["wake_depth"] == "skipped_no_signal"
    assert rows["inertia_awareness"] == "skipped_no_signal"


def test_below_threshold_control_reports_skipped_insufficient_samples(
    tmp_path, monkeypatch,
) -> None:
    store = MemoryStore(path=tmp_path)
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: now)
    _seed_sessions(
        store, MIN_SESSIONS_FOR_WAKE_DEPTH - 1,
        base=now - timedelta(minutes=10), recall_fraction=1.0, gap_seconds=30.0,
    )
    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")

    done, payload = pipe._step_knob_tune(None)

    assert done is True
    assert payload["knobs_skipped"]["wake_depth"] == "skipped_insufficient_samples"
    assert "wake_depth" not in payload["knobs_moved"]
    blob = load_profile_state(store)
    if blob is not None:
        assert blob["knobs"].get("wake_depth", "minimal") == "minimal"


def test_single_qualifying_night_does_not_flip_the_incumbent(
    tmp_path, monkeypatch,
) -> None:
    """One night with sufficient sample count, all voting the same direction,
    is a single 0.3-weight bayesian_update call -- far short of the 3.0-mass
    incumbent seed. Step-level regression for the one-weak-sample flip guarded
    at the unit level by seed_incumbent_posterior."""
    store = MemoryStore(path=tmp_path)
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: now)
    _seed_sessions(
        store, MIN_SESSIONS_FOR_WAKE_DEPTH,
        base=now - timedelta(minutes=15), recall_fraction=1.0, gap_seconds=45.0,
    )
    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")

    done, payload = pipe._step_knob_tune(None)

    assert done is True
    assert "wake_depth" not in payload["knobs_moved"]
    blob = _read_decrypted_blob(store)
    assert blob["knobs"]["wake_depth"] == "minimal"


def test_pin_control_prevents_move_and_preserves_pinned_value(
    tmp_path, monkeypatch,
) -> None:
    store = MemoryStore(path=tmp_path)
    local_state = default_state()
    # Deliberately divergent from core._profile_state's live "minimal" --
    # proves the tuner reconciles from the durable pin, not the live global.
    profile_set("wake_depth", "deep", local_state, store=store)

    now = datetime.now(timezone.utc)
    monkeypatch.setattr("iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: now)
    _seed_sessions(
        store, MIN_SESSIONS_FOR_WAKE_DEPTH * 2,
        base=now - timedelta(minutes=20), recall_fraction=1.0, gap_seconds=30.0,
    )
    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")

    done, payload = pipe._step_knob_tune(None)

    assert done is True
    assert payload["knobs_skipped"]["wake_depth"] == "skipped_pinned_by_user"
    assert "wake_depth" not in payload["knobs_moved"]
    blob = _read_decrypted_blob(store)
    assert blob["knobs"]["wake_depth"] == "deep", (
        "the tuner's write must never clobber a pinned value with a "
        "divergent live-process default"
    )
    assert "wake_depth" in blob["pins"]


def test_user_set_to_incumbent_value_pins_and_tuner_keeps_skipping(
    tmp_path, monkeypatch,
) -> None:
    """An explicit user set to the value a knob already holds (e.g. locking
    wake_depth at its 'minimal' seed default) must pin it just as firmly as
    a value-changing set -- sibling test
    test_composed_wake_depth_moves_after_sustained_signal_and_survives_reopen
    proves the SAME seeded signal moves an unpinned wake_depth to 'standard'
    within 11 nights, so a pin that fails to hold here is observable."""
    store = MemoryStore(path=tmp_path)
    local_state = default_state()
    assert local_state["wake_depth"] == "minimal"
    profile_set("wake_depth", "minimal", local_state, store=store)
    blob = _read_decrypted_blob(store)
    assert "wake_depth" in blob["pins"], "a no-op-valued user set must still pin"

    now = datetime.now(timezone.utc)
    monkeypatch.setattr("iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: now)
    base = now - timedelta(minutes=30)
    _seed_sessions(
        store, MIN_SESSIONS_FOR_WAKE_DEPTH,
        base=base, recall_fraction=1.0, gap_seconds=60.0,
    )
    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")

    for _ in range(11):
        done, payload = pipe._step_knob_tune(None)
        assert done is True
        assert payload["knobs_skipped"]["wake_depth"] == "skipped_pinned_by_user"
        assert "wake_depth" not in payload["knobs_moved"]
        blob = _read_decrypted_blob(store)
        assert blob["knobs"]["wake_depth"] == "minimal", (
            "sustained signal must never overtake an incumbent-value pin"
        )


def test_tuner_writes_do_not_self_pin_second_night_still_eligible(
    tmp_path, monkeypatch,
) -> None:
    store = MemoryStore(path=tmp_path)
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: now)
    _seed_sessions(
        store, MIN_SESSIONS_FOR_WAKE_DEPTH,
        base=now - timedelta(minutes=20), recall_fraction=1.0, gap_seconds=60.0,
    )
    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")

    pipe._step_knob_tune(None)
    blob_1 = _read_decrypted_blob(store)
    assert "wake_depth" not in blob_1["pins"], "the tuner's own write must never add a pin"
    mass_1 = blob_1["posterior"]["wake_depth"]["alphas"]["standard"]
    updated_at_1 = blob_1["updated_at"]

    pipe._step_knob_tune(None)
    blob_2 = _read_decrypted_blob(store)
    assert "wake_depth" not in blob_2["pins"]
    mass_2 = blob_2["posterior"]["wake_depth"]["alphas"]["standard"]
    assert mass_2 > mass_1, (
        "a second night with signal must still be eligible -- the "
        "DELETE-then-INSERT in save_profile_state must actually replace the "
        "row, not INSERT OR IGNORE no-op over a self-pin"
    )
    assert blob_2["updated_at"] >= updated_at_1


def test_persist_failure_surfaces_as_false_and_warning_event(
    tmp_path, monkeypatch,
) -> None:
    store = MemoryStore(path=tmp_path)
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: now)
    _seed_sessions(
        store, MIN_SESSIONS_FOR_WAKE_DEPTH,
        base=now - timedelta(minutes=10), recall_fraction=1.0, gap_seconds=45.0,
    )
    monkeypatch.setattr(
        "iai_mcp.lilli.profile.persistence.save_profile_state",
        lambda *a, **kw: False,
    )
    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")

    done, payload = pipe._step_knob_tune(None)

    assert done is True
    assert payload["persisted"] is False
    events = query_events(store, kind="profile_tuned", severity="warning", limit=10)
    assert events


def test_interrupt_mid_loop_leaves_globals_pristine(tmp_path, monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: now)

    assert sorted(PROFILE_KNOBS)[2] == "inertia_awareness"
    assert sorted(PROFILE_KNOBS)[-1] == "wake_depth"

    def _seed_both_knobs(target_store: MemoryStore) -> None:
        # Long, evenly-spaced gaps vote inertia_awareness True; every session
        # recalls, voting wake_depth deeper -- both specs clear min_samples.
        _seed_sessions(
            target_store, max(MIN_SESSIONS_FOR_WAKE_DEPTH, MIN_SESSIONS_FOR_INERTIA),
            base=now - timedelta(days=1),
            recall_fraction=1.0,
            gap_seconds=INERTIA_GAP_THRESHOLD_S * 2,
        )

    control_store = MemoryStore(path=tmp_path / "control")
    _seed_both_knobs(control_store)
    control_pipe = SleepPipeline(
        control_store, lifecycle_state_path=tmp_path / "control-lifecycle.json",
    )
    done, _payload = control_pipe._step_knob_tune(None)
    assert done is True
    assert core._posterior_state.get("inertia_awareness", {}).get("alpha") is not None, (
        "non-vacuity: the seeded window must actually reach bayesian_update's "
        "bool branch for inertia_awareness on an uninterrupted pass -- alpha "
        "only appears there, never from seed_incumbent_posterior's bool "
        "no-op -- or the pristine assertion below would pass for the wrong "
        "reason"
    )

    core._profile_state.clear()
    core._profile_state.update(default_state())
    core._posterior_state.clear()
    before_state = dict(core._profile_state)
    before_post = dict(core._posterior_state)

    store = MemoryStore(path=tmp_path / "main")
    _seed_both_knobs(store)
    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")

    calls = {"n": 0}

    def interrupt_check() -> bool:
        calls["n"] += 1
        # Fires on the 4th call (idx=3, interest_boost) -- after
        # inertia_awareness (idx=2) was already proposed, before wake_depth
        # (idx=9, last).
        return calls["n"] >= 4

    done2, payload2 = pipe._step_knob_tune(interrupt_check)

    assert done2 is False
    assert payload2 == {}
    assert load_profile_state(store) is None, "an interrupt must write no profile_state blob"
    assert core._profile_state == before_state, (
        "an interrupt must leave core._profile_state pristine -- the loop "
        "must work on a copy and commit only after finishing uninterrupted"
    )
    assert core._posterior_state == before_post


def test_report_covers_every_registry_knob_with_closed_vocabulary(
    tmp_path, monkeypatch,
) -> None:
    store = MemoryStore(path=tmp_path)
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: now)
    _seed_sessions(
        store, MIN_SESSIONS_FOR_WAKE_DEPTH,
        base=now - timedelta(minutes=10), recall_fraction=1.0, gap_seconds=45.0,
    )
    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")

    done, _payload = pipe._step_knob_tune(None)
    assert done is True

    events = query_events(store, kind="profile_tuned", limit=10)
    assert events
    rows = events[0]["data"]["knobs"]
    seen = {row["knob"] for row in rows}
    assert seen == set(PROFILE_KNOBS), "every registry knob must appear exactly once"
    assert len(rows) == len(PROFILE_KNOBS), "no knob may appear more than once"

    allowed_reasons = {
        "moved", "evaluated", "skipped_no_signal", "skipped_insufficient_samples",
        "skipped_not_tunable", "skipped_pinned_by_user",
    }
    for row in rows:
        assert row["reason"] in allowed_reasons

    not_tunable = {r["knob"] for r in rows if r["reason"] == "skipped_not_tunable"}
    assert not_tunable == set(PROFILE_KNOBS) - {
        "wake_depth", "inertia_awareness", "task_support", "monotropism_depth",
    }


def test_wake_depth_never_persists_past_the_standard_ceiling(
    tmp_path, monkeypatch,
) -> None:
    store = MemoryStore(path=tmp_path)
    now = datetime.now(timezone.utc)
    monkeypatch.setattr("iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: now)
    _seed_sessions(
        store, MIN_SESSIONS_FOR_WAKE_DEPTH,
        base=now - timedelta(minutes=10), recall_fraction=1.0, gap_seconds=45.0,
    )
    pipe = SleepPipeline(store, lifecycle_state_path=tmp_path / "lifecycle.json")

    seen_values = set()
    for _ in range(20):
        done, _payload = pipe._step_knob_tune(None)
        assert done is True
        blob = _read_decrypted_blob(store)
        seen_values.add(blob["knobs"]["wake_depth"])

    assert seen_values <= {"minimal", "standard"}
    assert "deep" not in seen_values
    assert WAKE_DEPTH_AUTO_CEILING == "standard"
