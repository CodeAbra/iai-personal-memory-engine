from __future__ import annotations

from pathlib import Path

import pytest

from iai_mcp.lilli.errors import BundleCapacityError
from iai_mcp.lilli.tiers.bsc import (
    BSC_MAX_BUNDLE_PAIRS,
    _TELEMETRY_ROLE_SATURATION_KIND,
    _max_bundle_pairs,
    bundle,
    filler_hv,
)

def _open_store(tmpdir: str, monkeypatch: pytest.MonkeyPatch):
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv("IAI_MCP_KEYRING_BYPASS", "true")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-saturation-pp")
    return MemoryStore(path=Path(tmpdir) / "store")

def _close_store(store) -> None:
    try:
        store.close()
    except Exception:
        pass

def test_max_bundle_pairs_default_D_4096() -> None:
    assert BSC_MAX_BUNDLE_PAIRS == 10
    assert _max_bundle_pairs(4096) == 10

def test_max_bundle_pairs_D_10000() -> None:
    assert _max_bundle_pairs(10000) == 25

def test_max_bundle_pairs_D_2048() -> None:
    assert _max_bundle_pairs(2048) == 5

def test_max_bundle_pairs_tiny_D_at_least_one() -> None:
    assert _max_bundle_pairs(8) >= 1

def test_bundle_at_capacity_succeeds_D_4096() -> None:
    roles = [
        "WHEN", "WHERE", "ROLE", "PROJECT", "COMMUNITY_ID",
        "TEMPORAL_POSITION", "ACTOR", "OBJECT", "INTENT", "MODALITY",
    ]
    assert len(roles) == 10 == BSC_MAX_BUNDLE_PAIRS
    pairs = [(r, filler_hv(f"v{i}")) for i, r in enumerate(roles)]
    result = bundle(pairs)
    assert isinstance(result, bytes)
    assert len(result) == 512

def test_bundle_raises_at_capacity_plus_one_D_4096() -> None:
    roles = [
        "WHEN", "WHERE", "ROLE", "PROJECT", "COMMUNITY_ID",
        "TEMPORAL_POSITION", "ACTOR", "OBJECT", "INTENT", "MODALITY", "LANG",
    ]
    assert len(roles) == 11
    pairs = [(r, filler_hv(f"v{i}")) for i, r in enumerate(roles)]
    with pytest.raises(BundleCapacityError) as exc_info:
        bundle(pairs)
    err_str = str(exc_info.value)
    assert "D=4096" in err_str
    assert "10" in err_str

def test_bundle_capacity_error_is_value_error() -> None:
    assert issubclass(BundleCapacityError, ValueError)

def test_bundle_at_capacity_D_10000_succeeds_with_25_pairs() -> None:
    assert _max_bundle_pairs(10000) == 25
    pairs_25 = [(r if i < 18 else f"EXTRA_{i}", filler_hv(f"v{i}", D=10000))
                for i, r in enumerate(
                    list("WHEN WHERE ROLE PROJECT COMMUNITY_ID TEMPORAL_POSITION "
                         "ACTOR OBJECT INTENT MODALITY LANG SESSION_ID "
                         "TIER VALENCE CERTAINTY SOURCE TOPIC PARENT_ID".split())
                    + [f"EX{j}" for j in range(7)]
                )]
    assert len(pairs_25) == 25
    result = bundle(pairs_25, D=10000)
    assert len(result) == 1250

    pairs_26 = pairs_25 + [("EXTRA_26", filler_hv("v26", D=10000))]
    with pytest.raises(BundleCapacityError):
        bundle(pairs_26, D=10000)

def test_bundle_below_warn_threshold_no_telemetry(tmp_path, monkeypatch) -> None:
    from iai_mcp.events import query_events

    store = _open_store(str(tmp_path), monkeypatch)
    try:
        roles = ["WHEN", "WHERE", "ROLE", "PROJECT", "COMMUNITY_ID", "TEMPORAL_POSITION", "ACTOR"]
        assert len(roles) == 7
        pairs = [(r, filler_hv(f"v{i}")) for i, r in enumerate(roles)]
        bundle(pairs, store=store)
        events = query_events(store, kind="role_saturation_warning")
        assert len(events) == 0, f"Expected no saturation events at 7 pairs, got {len(events)}"
    finally:
        _close_store(store)

def test_bundle_above_warn_threshold_emits_telemetry(tmp_path, monkeypatch) -> None:
    from iai_mcp.events import query_events

    store = _open_store(str(tmp_path), monkeypatch)
    try:
        roles = [
            "WHEN", "WHERE", "ROLE", "PROJECT", "COMMUNITY_ID",
            "TEMPORAL_POSITION", "ACTOR", "OBJECT", "INTENT",
        ]
        assert len(roles) == 9
        pairs = [(r, filler_hv(f"v{i}")) for i, r in enumerate(roles)]
        result = bundle(pairs, store=store)
        assert isinstance(result, bytes), "bundle should succeed at 9 pairs"

        events = query_events(store, kind="role_saturation_warning")
        assert len(events) >= 1, f"Expected >= 1 saturation event at 9 pairs, got {len(events)}"

        payload = events[0]["data"]
        assert payload.get("D") == 4096
        assert payload.get("n_pairs") == 9
        assert payload.get("max_pairs") == 10
    finally:
        _close_store(store)

def test_bundle_above_warn_threshold_no_store_no_emit() -> None:
    roles = [
        "WHEN", "WHERE", "ROLE", "PROJECT", "COMMUNITY_ID",
        "TEMPORAL_POSITION", "ACTOR", "OBJECT", "INTENT",
    ]
    pairs = [(r, filler_hv(f"v{i}")) for i, r in enumerate(roles)]
    result = bundle(pairs)
    assert isinstance(result, bytes)

def test_telemetry_kind_string_matches_events_module() -> None:
    from iai_mcp import events

    if not hasattr(events, "TELEMETRY_ROLE_SATURATION"):
        pytest.skip("events.TELEMETRY_ROLE_SATURATION not yet defined (46-10 has not shipped)")

    assert _TELEMETRY_ROLE_SATURATION_KIND == events.TELEMETRY_ROLE_SATURATION, (
        f"bsc._TELEMETRY_ROLE_SATURATION_KIND={_TELEMETRY_ROLE_SATURATION_KIND!r} "
        f"!= events.TELEMETRY_ROLE_SATURATION={events.TELEMETRY_ROLE_SATURATION!r}"
    )

def test_bundle_over_cap_emits_then_raises(tmp_path, monkeypatch) -> None:
    from iai_mcp.events import query_events

    store = _open_store(str(tmp_path), monkeypatch)
    try:
        roles = [
            "WHEN", "WHERE", "ROLE", "PROJECT", "COMMUNITY_ID",
            "TEMPORAL_POSITION", "ACTOR", "OBJECT", "INTENT", "MODALITY", "LANG",
        ]
        assert len(roles) == 11
        pairs = [(r, filler_hv(f"v{i}")) for i, r in enumerate(roles)]

        with pytest.raises(BundleCapacityError):
            bundle(pairs, store=store)

        events = query_events(store, kind="role_saturation_warning")
        assert len(events) >= 1, (
            "EMIT-THEN-RAISE contract violated: telemetry event not found in store "
            "after catching BundleCapacityError. The emit must happen before the raise."
        )
    finally:
        _close_store(store)


def test_bundle_over_cap_no_store_logs_then_raises(caplog) -> None:
    import logging

    roles = [
        "WHEN", "WHERE", "ROLE", "PROJECT", "COMMUNITY_ID",
        "TEMPORAL_POSITION", "ACTOR", "OBJECT", "INTENT", "MODALITY", "LANG",
    ]
    assert len(roles) == 11
    pairs = [(r, filler_hv(f"v{i}")) for i, r in enumerate(roles)]

    with caplog.at_level(logging.ERROR, logger="iai_mcp.lilli.tiers.bsc"):
        with pytest.raises(BundleCapacityError):
            bundle(pairs)

    # EMIT-THEN-RAISE contract: with store=None and n > max_pairs, the saturation
    # signal must still be recorded (degraded to a log-only emit) before the raise.
    saturation_logs = [
        r for r in caplog.records
        if r.levelno >= logging.ERROR
        and r.name == "iai_mcp.lilli.tiers.bsc"
        and "saturation" in r.getMessage().lower()
    ]
    assert len(saturation_logs) >= 1, (
        "EMIT-THEN-RAISE contract violated: with store=None and n > max_pairs, no "
        "saturation log was recorded before BundleCapacityError. The saturation "
        "signal is lost exactly in the failure case it exists to record."
    )
    msg = saturation_logs[0].getMessage()
    assert "11" in msg and "10" in msg, msg
