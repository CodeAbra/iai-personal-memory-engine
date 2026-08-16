"""Durability of the autistic-cognition profile: an encrypted blob in
``_hippo_meta`` plus in-place hydration of the live process mappings.

Assertions read the decrypted blob or the live-dict identity directly --
never through ``profile_get``, which returns a registry default for a
missing key and would make "hydration ran and produced the default"
indistinguishable from "hydration never ran".
"""

from __future__ import annotations

import json

import pytest

from iai_mcp import core
from iai_mcp.crypto import encrypt_field, is_encrypted
from iai_mcp.lilli.profile.knobs import default_state
from iai_mcp.lilli.profile.persistence import (
    PROFILE_BLOB_AAD,
    PROFILE_META_KEY,
    hydrate_profile,
    load_profile_state,
    save_profile_state,
)
from iai_mcp.events import query_events
from iai_mcp.store import MemoryStore


@pytest.fixture(autouse=True)
def _restore_live_profile():
    """Idiom from tests/test_knobs_applied_telemetry.py:250-251 -- in-place
    restore keeps core.LIVE_KNOBS aliased to core._profile_state the same
    way production hydration must. Also resets the process-global
    once-per-store cache so every test starts as if no store root had ever
    been hydrated in this process."""
    saved_state = dict(core._profile_state)
    saved_posterior = dict(core._posterior_state)
    saved_hydrated = set(core._profile_hydrated_stores)
    saved_community_names = dict(core._community_names_cache)
    core._profile_hydrated_stores.clear()
    core._community_names_cache.clear()
    yield
    core._profile_state.clear()
    core._profile_state.update(saved_state)
    core._posterior_state.clear()
    core._posterior_state.update(saved_posterior)
    core._profile_hydrated_stores.clear()
    core._profile_hydrated_stores.update(saved_hydrated)
    core._community_names_cache.clear()
    core._community_names_cache.update(saved_community_names)


def _write_raw_meta(store: MemoryStore, value: str) -> None:
    with store.db._conn_lock:
        store.db._conn.execute(
            "DELETE FROM _hippo_meta WHERE key = ?", (PROFILE_META_KEY,)
        )
        store.db._conn.execute(
            "INSERT OR IGNORE INTO _hippo_meta (key, value) VALUES (?, ?)",
            (PROFILE_META_KEY, value),
        )
        store.db._conn.commit()


def _read_raw_meta(store: MemoryStore) -> "str | None":
    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = ?", (PROFILE_META_KEY,)
        ).fetchone()
    return row["value"] if row is not None else None


class _NonHippoStore:
    """Duck-typed store whose ``db`` is not a HippoDB -- the legacy-driver
    guard must no-op, never raise."""

    db = object()

    def _key(self) -> bytes:
        return b"0" * 32


# ---------------------------------------------------------------------------
# Task 1: encrypted blob read/write
# ---------------------------------------------------------------------------


def test_round_trip_preserves_knobs_posterior_pins(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    knobs = {"literal_preservation": "loose", "monotropism_depth": {"alice": 0.7}}
    posterior = {"literal_preservation": {"alpha": 2.0, "beta": 1.0}}
    pins = {"literal_preservation": "2026-08-01T10:00:00+00:00"}

    assert save_profile_state(store, knobs=knobs, posterior=posterior, pins=pins) is True

    blob = load_profile_state(store)
    assert blob is not None
    assert blob["knobs"] == knobs
    assert blob["posterior"] == posterior
    assert blob["pins"] == pins
    assert blob["dropped"] == []


def test_stored_blob_is_encrypted_never_plaintext(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    save_profile_state(store, knobs={"masking_off": True}, posterior={}, pins={})

    raw = _read_raw_meta(store)
    assert raw is not None
    assert is_encrypted(raw)
    assert "masking_off" not in raw


def test_aad_mismatch_fails_open_and_reports_unreadable(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    payload = json.dumps(
        {"version": 1, "knobs": {}, "posterior": {}, "pins": {}, "updated_at": "x"}
    )
    # Encrypted under a DIFFERENT associated_data -- as if the ciphertext had
    # been copied from another _hippo_meta key.
    wrong_aad_ct = encrypt_field(payload, store._key(), associated_data=b"embed_identity")
    _write_raw_meta(store, wrong_aad_ct)

    assert load_profile_state(store) is None
    events = query_events(store, kind="profile_state_unreadable")
    assert events, "expected a profile_state_unreadable warning event"
    assert events[0]["severity"] == "warning"
    assert events[0]["data"]["reason"] == "decrypt_failed"


def test_unknown_knob_name_dropped_not_raised(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    save_profile_state(
        store,
        knobs={"literal_preservation": "loose", "a_knob_that_was_removed": "x"},
        posterior={},
        pins={},
    )

    blob = load_profile_state(store)
    assert blob is not None
    assert "a_knob_that_was_removed" not in blob["knobs"]
    assert blob["knobs"]["literal_preservation"] == "loose"
    assert "a_knob_that_was_removed" in blob["dropped"]


def test_out_of_schema_value_dropped_keeps_rest(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    save_profile_state(
        store,
        knobs={
            "literal_preservation": "not-in-the-enum",
            "masking_off": True,
        },
        posterior={},
        pins={},
    )

    blob = load_profile_state(store)
    assert blob is not None
    assert "literal_preservation" not in blob["knobs"]
    assert blob["knobs"]["masking_off"] is True
    assert "literal_preservation" in blob["dropped"]


def test_non_hippo_store_never_raises() -> None:
    store = _NonHippoStore()
    assert save_profile_state(store, knobs={}, posterior={}, pins={}) is False
    assert load_profile_state(store) is None


def test_plaintext_value_treated_as_corrupt(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    _write_raw_meta(store, json.dumps({"version": 1, "knobs": {"masking_off": True}}))

    assert load_profile_state(store) is None
    events = query_events(store, kind="profile_state_unreadable")
    assert events, "expected a profile_state_unreadable warning event"
    assert events[0]["severity"] == "warning"
    assert events[0]["data"]["reason"] == "not_encrypted"


# ---------------------------------------------------------------------------
# Task 2: in-place hydration and its call sites
# ---------------------------------------------------------------------------


def test_hydration_never_rebinds_live_knobs_alias(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    save_profile_state(
        store, knobs={"masking_off": False}, posterior={}, pins={}
    )

    assert core.LIVE_KNOBS is core._profile_state
    hydrate_profile(store, core._profile_state, core._posterior_state)
    assert core.LIVE_KNOBS is core._profile_state, (
        "hydration rebound _profile_state -- LIVE_KNOBS now aliases a stale dict"
    )
    assert core._profile_state["masking_off"] is False


def test_reopen_after_process_reset_recovers_saved_value(tmp_path) -> None:
    store_a = MemoryStore(path=tmp_path)
    save_profile_state(
        store_a,
        knobs={"dunn_quadrant": "seeking"},
        posterior={"dunn_quadrant": {"seeking": 1.0}},
        pins={},
    )
    store_a.close()

    # Simulate the process losing its in-memory state (a restart) without
    # rebinding -- the aliasing hazard this plan exists to prevent.
    core._profile_state.clear()
    core._profile_state.update(default_state())
    assert core._profile_state["dunn_quadrant"] == "neutral"

    store_b = MemoryStore(path=tmp_path)
    result = hydrate_profile(store_b, core._profile_state, core._posterior_state)

    assert result["hydrated"] is True
    assert core._profile_state["dunn_quadrant"] == "seeking"
    assert core._posterior_state["dunn_quadrant"] == {"seeking": 1.0}


def test_hydration_on_store_with_no_blob_leaves_defaults_and_writes_nothing(
    tmp_path,
) -> None:
    store = MemoryStore(path=tmp_path)
    core._profile_state.clear()
    core._profile_state.update(default_state())
    before = dict(core._profile_state)

    result = hydrate_profile(store, core._profile_state, core._posterior_state)

    assert result == {"hydrated": False, "dropped": []}
    assert core._profile_state == before
    assert _read_raw_meta(store) is None


def test_ensure_profile_hydrated_reads_at_most_once_per_store(
    tmp_path, monkeypatch
) -> None:
    store = MemoryStore(path=tmp_path)
    save_profile_state(
        store, knobs={"masking_off": False}, posterior={}, pins={}
    )

    calls = {"n": 0}
    real_hydrate = hydrate_profile

    def _counting_hydrate(store_arg, state_arg, posterior_arg):
        calls["n"] += 1
        return real_hydrate(store_arg, state_arg, posterior_arg)

    monkeypatch.setattr(
        "iai_mcp.lilli.profile.persistence.hydrate_profile", _counting_hydrate
    )

    core.dispatch(store, "profile_get", {})
    core.dispatch(store, "profile_get", {})

    assert calls["n"] == 1, "hydration re-read the store on a second dispatch"
    assert core._profile_state["masking_off"] is False


def test_hydration_never_raises_out_of_dispatch(tmp_path, monkeypatch) -> None:
    store = MemoryStore(path=tmp_path)
    save_profile_state(store, knobs={"masking_off": False}, posterior={}, pins={})

    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated hydration blowup")

    monkeypatch.setattr(
        "iai_mcp.lilli.profile.persistence.hydrate_profile", _boom
    )

    result = core.dispatch(store, "profile_get", {})
    assert result["live"]["masking_off"] in (True, False)


def test_ensure_profile_hydrated_is_the_daemon_boot_entry_point(tmp_path) -> None:
    """The daemon calls exactly ``core.ensure_profile_hydrated(store)`` once,
    after the store opens and before the socket binds -- no prior dispatch on
    that store root. Exercised here as a cold call: no store-specific setup
    beyond what save_profile_state + a fresh process would have."""
    store = MemoryStore(path=tmp_path)
    save_profile_state(
        store,
        knobs={"dunn_quadrant": "seeking", "interest_boost": 0.4},
        posterior={},
        pins={},
    )
    assert core._topology_store_key(store) not in core._profile_hydrated_stores

    result = core.ensure_profile_hydrated(store)

    assert result["hydrated"] is True
    assert core._profile_state["dunn_quadrant"] == "seeking"
    assert core._profile_state["interest_boost"] == 0.4
    assert core.LIVE_KNOBS is core._profile_state


def test_boot_hydration_loads_persisted_community_names_without_a_sleep_cycle(
    tmp_path,
) -> None:
    """A restart must not depend on an in-process nightly cycle to resolve
    community names: the map is persisted, so the boot seam has to load it
    the same way it loads the profile blob. Fails against the pre-fix code,
    where `ensure_profile_hydrated` never touches `community_names`."""
    from uuid import uuid4

    from iai_mcp.lilli.profile.community_names import save_community_names
    from iai_mcp.lilli.profile.knobs import profile_modulation_for_record
    from iai_mcp.types import EMBED_DIM, MemoryRecord
    from datetime import datetime, timezone

    store = MemoryStore(path=tmp_path)
    community_id = uuid4()
    assert save_community_names(
        store,
        reverse_index={str(community_id): "jazz"},
        provenance={},
    ) is True

    # No naming cycle has run in this process -- simulate a cold restart.
    assert core._topology_store_key(store) not in core._profile_hydrated_stores
    assert core.get_community_names() == {}

    core.ensure_profile_hydrated(store)

    assert core.get_community_names() == {str(community_id): "jazz"}

    now = datetime.now(timezone.utc)
    record = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="x",
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=community_id,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )
    gains = profile_modulation_for_record(
        record, {"monotropism_depth": {"jazz": 0.4}},
    )
    assert gains["monotropism_depth"] == pytest.approx(1.4)
