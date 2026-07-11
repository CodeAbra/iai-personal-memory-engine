from __future__ import annotations

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pytest

@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    return tmp_path

def test_detect_tz_returns_iana_string():
    from iai_mcp.tz import detect_tz

    key = detect_tz()
    assert isinstance(key, str)
    assert len(key) > 0
    ZoneInfo(key)  # noqa: B018 -- constructing is the check

def test_detect_tz_matches_system_or_utc_fallback():
    from iai_mcp.tz import detect_tz

    key = detect_tz()
    assert key == "UTC" or "/" in key

def test_load_user_tz_reads_config(isolated_store):
    from iai_mcp.tz import load_user_tz

    cfg = isolated_store / "config.json"
    cfg.write_text(json.dumps({"user": {"timezone": "Asia/Tokyo"}}))
    tz = load_user_tz()
    assert tz.key == "Asia/Tokyo"

def test_load_user_tz_defaults_on_missing_config(isolated_store):
    from iai_mcp.tz import detect_tz, load_user_tz

    assert not (isolated_store / "config.json").exists()
    tz = load_user_tz()
    assert isinstance(tz, ZoneInfo)
    assert tz.key == detect_tz() or tz.key == "UTC"

def test_load_user_tz_rejects_invalid_iana(isolated_store):
    from iai_mcp.tz import load_user_tz

    cfg = isolated_store / "config.json"
    cfg.write_text(json.dumps({"user": {"timezone": "Garbage/Not-Real"}}))
    with pytest.raises(ZoneInfoNotFoundError):
        load_user_tz()

def test_load_user_tz_handles_malformed_json(isolated_store):
    from iai_mcp.tz import load_user_tz

    cfg = isolated_store / "config.json"
    cfg.write_text("not-valid-json{")
    tz = load_user_tz()
    assert isinstance(tz, ZoneInfo)

def test_config_auto_seeds_timezone_on_first_run(isolated_store):
    from iai_mcp.tz import load_user_tz

    assert not (isolated_store / "config.json").exists()
    load_user_tz()

    cfg_path = isolated_store / "config.json"
    assert cfg_path.exists()

    cfg = json.loads(cfg_path.read_text())
    assert "user" in cfg
    assert "timezone" in cfg["user"]
    ZoneInfo(cfg["user"]["timezone"])  # noqa: B018

def test_config_autoseeded_value_stable_across_loads(isolated_store):
    from iai_mcp.tz import load_user_tz

    tz1 = load_user_tz()
    tz2 = load_user_tz()
    assert tz1.key == tz2.key

def test_config_load_respects_user_override(isolated_store):
    from iai_mcp.tz import load_user_tz

    load_user_tz()

    cfg_path = isolated_store / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["user"]["timezone"] = "Europe/Moscow"
    cfg_path.write_text(json.dumps(cfg))

    tz = load_user_tz()
    assert tz.key == "Europe/Moscow"

def test_to_local_converts_utc():
    from iai_mcp.tz import to_local

    utc_dt = datetime(2026, 4, 17, 12, 0, tzinfo=timezone.utc)
    local = to_local(utc_dt, ZoneInfo("America/Los_Angeles"))
    assert local.hour == 5
    assert local.tzinfo.key == "America/Los_Angeles"

def test_to_local_handles_naive_datetime():
    from iai_mcp.tz import to_local

    naive = datetime(2026, 4, 17, 12, 0)
    local = to_local(naive, ZoneInfo("UTC"))
    assert local.tzinfo is not None
    assert local.hour == 12
