"""Freshness-verdict synthesis over a claim_check response.

Pure tier coverage over hand-built recall-response dicts, plus a
store-backed acceptance proving the co-presence heuristic fires on the
real dispatch/recall path.
"""

from __future__ import annotations

import pytest


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401, PLC0415
        except ImportError:
            pytest.skip("iai_mcp_native not built")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def test_contradicted_via_anti_hit():
    from datetime import datetime, timezone

    from iai_mcp.claim_check import CONTRADICTED, synthesize_verdict

    response = {
        "hits": [
            {"score": 0.9, "valid_to": None, "captured_at": "2026-01-01T00:00:00+00:00", "community_id": None},
        ],
        "anti_hits": [
            {"score": 0.5, "valid_to": None, "captured_at": "2026-01-01T00:00:00+00:00", "community_id": None},
        ],
    }
    verdict = synthesize_verdict(response, now=datetime(2026, 8, 1, tzinfo=timezone.utc))
    assert verdict["tier"] == CONTRADICTED, verdict


def test_contradicted_via_valid_to():
    from datetime import datetime, timezone

    from iai_mcp.claim_check import CONTRADICTED, synthesize_verdict

    response = {
        "hits": [
            {
                "score": 0.9,
                "valid_to": "2026-02-01T00:00:00+00:00",
                "captured_at": "2026-01-01T00:00:00+00:00",
                "community_id": None,
            },
        ],
        "anti_hits": [],
    }
    verdict = synthesize_verdict(response, now=datetime(2026, 8, 1, tzinfo=timezone.utc))
    assert verdict["tier"] == CONTRADICTED, verdict


def test_anchored_staleness_yields_unconfirmed():
    from datetime import datetime, timezone

    from iai_mcp.claim_check import CURRENT, NO_EVIDENCE, UNCONFIRMED, synthesize_verdict

    response = {
        "hits": [
            {"score": 0.95, "valid_to": None, "captured_at": "2026-01-01T00:00:00+00:00", "community_id": None},
            {"score": 0.4, "valid_to": None, "captured_at": "2026-02-05T00:00:00+00:00", "community_id": None},
        ],
        "anti_hits": [],
    }
    verdict = synthesize_verdict(response, now=datetime(2026, 8, 1, tzinfo=timezone.utc))
    assert verdict["tier"] == UNCONFIRMED, verdict
    assert verdict["tier"] != CURRENT
    assert verdict["tier"] != NO_EVIDENCE


def test_recent_anchor_yields_current():
    """A negative-precision case: an older record is present but the
    top-scored (anchor) hit is recent and nothing postdates it -- stays
    CURRENT even with old records around."""
    from datetime import datetime, timezone

    from iai_mcp.claim_check import CURRENT, UNCONFIRMED, synthesize_verdict

    response = {
        "hits": [
            {"score": 0.3, "valid_to": None, "captured_at": "2026-01-01T00:00:00+00:00", "community_id": None},
            {"score": 0.95, "valid_to": None, "captured_at": "2026-07-28T00:00:00+00:00", "community_id": None},
        ],
        "anti_hits": [],
    }
    verdict = synthesize_verdict(response, now=datetime(2026, 8, 1, tzinfo=timezone.utc))
    assert verdict["tier"] == CURRENT, verdict
    assert verdict["tier"] != UNCONFIRMED


def test_different_community_pair_not_flagged():
    from datetime import datetime, timezone

    from iai_mcp.claim_check import CURRENT, UNCONFIRMED, synthesize_verdict

    response = {
        "hits": [
            {
                "score": 0.95, "valid_to": None,
                "captured_at": "2026-01-01T00:00:00+00:00", "community_id": "community-a",
            },
            {
                "score": 0.4, "valid_to": None,
                "captured_at": "2026-02-05T00:00:00+00:00", "community_id": "community-b",
            },
        ],
        "anti_hits": [],
    }
    verdict = synthesize_verdict(response, now=datetime(2026, 8, 1, tzinfo=timezone.utc))
    assert verdict["tier"] == CURRENT, verdict
    assert verdict["tier"] != UNCONFIRMED


def test_none_community_still_pairs():
    from datetime import datetime, timezone

    from iai_mcp.claim_check import UNCONFIRMED, synthesize_verdict

    response = {
        "hits": [
            {
                "score": 0.95, "valid_to": None,
                "captured_at": "2026-01-01T00:00:00+00:00", "community_id": "community-a",
            },
            {"score": 0.4, "valid_to": None, "captured_at": "2026-02-05T00:00:00+00:00", "community_id": None},
        ],
        "anti_hits": [],
    }
    verdict = synthesize_verdict(response, now=datetime(2026, 8, 1, tzinfo=timezone.utc))
    assert verdict["tier"] == UNCONFIRMED, verdict


def test_no_evidence_empty_hits():
    from datetime import datetime, timezone

    from iai_mcp.claim_check import NO_EVIDENCE, synthesize_verdict

    response = {"hits": [], "anti_hits": []}
    verdict = synthesize_verdict(response, now=datetime(2026, 8, 1, tzinfo=timezone.utc))
    assert verdict["tier"] == NO_EVIDENCE, verdict


def test_malformed_timestamps_never_raise():
    """Malformed/missing captured_at, valid_to, and score are no-signal,
    never a raise."""
    from datetime import datetime, timezone

    from iai_mcp.claim_check import CURRENT, synthesize_verdict

    response = {
        "hits": [
            {"valid_to": "garbage", "captured_at": "not-a-date", "community_id": None},
        ],
        "anti_hits": [],
    }
    verdict = synthesize_verdict(response, now=datetime(2026, 8, 1, tzinfo=timezone.utc))
    assert verdict["tier"] == CURRENT, verdict


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_dispatch_claim_check_roundtrip(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    from iai_mcp.capture import capture_turn
    from iai_mcp.core import dispatch
    from iai_mcp.store import MemoryStore, flush_record_buffer

    store = MemoryStore(path=tmp_path)
    seed = capture_turn(
        store, cue="c", text="alice's release ships next Tuesday",
        session_id="s1", role="user", live_turn=True,
    )
    assert seed["status"] == "inserted", seed
    flush_record_buffer(store)

    response = dispatch(
        store, "claim_check",
        {"cue": "alice's release ships next Tuesday", "session_id": "s1"},
    )

    assert "hits" in response, response
    assert "anti_hits" in response, response
    assert "verdict" in response, response
    assert response.get("_source") == "claim_check", response


def test_claim_check_does_not_inflate_recall_telemetry(tmp_path, monkeypatch):
    from iai_mcp import core
    from iai_mcp.capture import capture_turn
    from iai_mcp.store import MemoryStore, flush_record_buffer

    store = MemoryStore(path=tmp_path)
    seed = capture_turn(
        store, cue="c", text="alice's release ships next Tuesday",
        session_id="s1", role="user", live_turn=True,
    )
    assert seed["status"] == "inserted", seed
    flush_record_buffer(store)

    events_seen: list[str] = []
    import iai_mcp.events as events_mod

    def _spy_write_event(store_arg, kind, data, **kwargs):
        events_seen.append(kind)

    monkeypatch.setattr(events_mod, "write_event", _spy_write_event)

    # core._arousal_state is a process-global mutated in place -- an
    # absolute counter would depend on whatever earlier test in the same
    # session already moved it. Compare deltas around each call instead.
    def _snapshot():
        state = core._arousal_state
        if state is None:
            return None
        return (state.success_count, state.error_count, state.level)

    before_claim_check = _snapshot()
    core.dispatch(
        store, "claim_check",
        {"cue": "alice's release ships next Tuesday", "session_id": "s1"},
    )
    assert "recall_dispatched" not in events_seen, (
        f"claim_check must not inflate recall telemetry: {events_seen}"
    )
    after_claim_check = _snapshot()
    assert after_claim_check == before_claim_check or before_claim_check is None, (
        f"claim_check must not move arousal: {before_claim_check} -> {after_claim_check}"
    )

    events_seen.clear()
    before_direct_recall = _snapshot()
    core.dispatch(
        store, "memory_recall",
        {"cue": "alice's release ships next Tuesday", "session_id": "s1"},
    )
    assert "recall_dispatched" in events_seen, (
        "a direct memory_recall must still be counted"
    )
    after_direct_recall = _snapshot()
    assert after_direct_recall != before_direct_recall, (
        "a direct memory_recall must still move arousal"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_claim_check_recall_is_free_of_side_effects(driver, tmp_path, monkeypatch):
    """claim_check's internal probe recall must stay genuinely read-only,
    matching its readOnlyHint: true MCP contract -- it must not reinforce
    its probe hits, wire coactivation, set the trajectory-coupling
    injection globals, consume a session's first-turn warm-cascade slot,
    emit the pask_teachback_pass telemetry event, or consume a pending
    overnight digest. A direct memory_recall over the same cue must still
    do all six -- proves the spies are wired correctly, not silently
    no-op."""
    _select_driver(driver, monkeypatch)
    from iai_mcp import core
    from iai_mcp.capture import capture_turn
    from iai_mcp.store import MemoryStore, flush_record_buffer

    store = MemoryStore(path=tmp_path)
    seed = capture_turn(
        store, cue="c", text="alice's release ships next Tuesday",
        session_id="s1", role="user", live_turn=True,
    )
    assert seed["status"] == "inserted", seed
    flush_record_buffer(store)

    reinforce_calls: list = []
    monkeypatch.setattr(
        store, "queue_reinforce",
        lambda ids: reinforce_calls.append(list(ids)),
    )

    import iai_mcp.retrieve as retrieve_mod
    coactivation_calls: list = []
    real_potentiate = retrieve_mod.potentiate_coactivation

    def _spy_potentiate(store_arg, wire_ids):
        coactivation_calls.append(list(wire_ids))
        return real_potentiate(store_arg, wire_ids)

    monkeypatch.setattr(retrieve_mod, "potentiate_coactivation", _spy_potentiate)

    import iai_mcp.daemon_state as daemon_state_mod
    consume_calls: list = []
    real_consume = daemon_state_mod.consume_first_turn

    def _spy_consume(state, session_id):
        consume_calls.append(session_id)
        return real_consume(state, session_id)

    monkeypatch.setattr(daemon_state_mod, "consume_first_turn", _spy_consume)

    digest_calls: list = []
    real_get_pending_digest = daemon_state_mod.get_pending_digest

    def _spy_get_pending_digest(state, now):
        digest_calls.append(now)
        return real_get_pending_digest(state, now)

    monkeypatch.setattr(daemon_state_mod, "get_pending_digest", _spy_get_pending_digest)

    import iai_mcp.events as events_mod
    pask_event_calls: list = []
    real_write_event = events_mod.write_event

    def _spy_write_event(store_arg, kind, data, **kwargs):
        if kind == "pask_teachback_pass":
            pask_event_calls.append(kind)
        return real_write_event(store_arg, kind, data, **kwargs)

    monkeypatch.setattr(events_mod, "write_event", _spy_write_event)

    core._last_injection_embedding = None
    core._last_injection_ids = []

    claim_resp = core.dispatch(
        store, "claim_check",
        {"cue": "alice's release ships next Tuesday", "session_id": "s1-claim"},
    )
    assert "verdict" in claim_resp, claim_resp

    assert reinforce_calls == [], (
        f"claim_check must not reinforce its probe hits: {reinforce_calls}"
    )
    assert coactivation_calls == [], (
        f"claim_check must not wire coactivation from its probe hits: {coactivation_calls}"
    )
    assert consume_calls == [], (
        f"claim_check must not consume the first-turn warm-cascade slot: {consume_calls}"
    )
    assert digest_calls == [], (
        f"claim_check must not consume the pending overnight digest: {digest_calls}"
    )
    assert pask_event_calls == [], (
        f"claim_check must not emit pask_teachback_pass telemetry: {pask_event_calls}"
    )
    assert core._last_injection_embedding is None, (
        "claim_check must not set the trajectory-coupling injection embedding"
    )
    assert core._last_injection_ids == [], (
        "claim_check must not set the trajectory-coupling injection ids"
    )

    reinforce_calls.clear()
    coactivation_calls.clear()
    consume_calls.clear()
    digest_calls.clear()
    pask_event_calls.clear()
    core._last_injection_embedding = None
    core._last_injection_ids = []

    recall_resp = core.dispatch(
        store, "memory_recall",
        {"cue": "alice's release ships next Tuesday", "session_id": "s1-direct"},
    )
    assert "error" not in recall_resp, recall_resp

    assert reinforce_calls, "a direct memory_recall must still reinforce its hits"
    assert coactivation_calls, "a direct memory_recall must still run coactivation wiring"
    assert consume_calls, "a direct memory_recall must still consume the first-turn slot"
    assert digest_calls, "a direct memory_recall must still check the pending overnight digest"
    assert pask_event_calls, (
        "a direct memory_recall must still emit pask_teachback_pass telemetry"
    )
    assert core._last_injection_embedding is not None, (
        "a direct memory_recall must still set the trajectory-coupling injection embedding"
    )
    assert core._last_injection_ids, (
        "a direct memory_recall must still set the trajectory-coupling injection ids"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_stale_uncorrected_claim_flagged(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    from iai_mcp.capture import capture_turn
    from iai_mcp.core import dispatch
    from iai_mcp.store import MemoryStore, flush_record_buffer

    store = MemoryStore(path=tmp_path)

    stale_text = "alice's onboarding project is queued and has not started yet"
    stale = capture_turn(
        store, cue="c", text=stale_text, session_id="s1", role="user",
        live_turn=True, ts="2026-01-01T00:00:00+00:00",
    )
    assert stale["status"] == "inserted", stale

    newer_text = "alice's onboarding project shipped last night, rollout is complete"
    newer = capture_turn(
        store, cue="c", text=newer_text, session_id="s1", role="user",
        live_turn=True, ts="2026-03-01T00:00:00+00:00",
    )
    assert newer["status"] == "inserted", newer
    flush_record_buffer(store)

    response = dispatch(store, "claim_check", {"cue": stale_text, "session_id": "s1"})

    hit_ids = {h["record_id"] for h in response.get("hits") or []}
    assert stale["record_id"] in hit_ids, response
    assert newer["record_id"] in hit_ids, response

    verdict = response.get("verdict") or {}
    assert verdict.get("tier") == "UNCONFIRMED", response
    assert verdict.get("tier") != "CURRENT"
    assert verdict.get("tier") != "NO_EVIDENCE"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_current_claim_not_flagged(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    from iai_mcp.capture import capture_turn
    from iai_mcp.core import dispatch
    from iai_mcp.store import MemoryStore, flush_record_buffer

    store = MemoryStore(path=tmp_path)

    old_text = "alice's onboarding project is queued and has not started yet"
    old = capture_turn(
        store, cue="c", text=old_text, session_id="s1", role="user",
        live_turn=True, ts="2026-01-01T00:00:00+00:00",
    )
    assert old["status"] == "inserted", old

    current_text = "alice's billing migration finished this morning, fully deployed"
    current = capture_turn(
        store, cue="c", text=current_text, session_id="s1", role="user",
        live_turn=True, ts="2026-08-20T00:00:00+00:00",
    )
    assert current["status"] == "inserted", current
    flush_record_buffer(store)

    response = dispatch(store, "claim_check", {"cue": current_text, "session_id": "s1"})

    hit_ids = {h["record_id"] for h in response.get("hits") or []}
    assert current["record_id"] in hit_ids, response

    verdict = response.get("verdict") or {}
    assert verdict.get("tier") == "CURRENT", response
    assert verdict.get("tier") != "UNCONFIRMED"
