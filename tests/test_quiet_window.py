from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


def _fresh_store(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")
    from iai_mcp.store import MemoryStore
    return MemoryStore()

def _seed_sessions(
    store,
    *,
    local_tz: ZoneInfo,
    day_start_local: datetime,
    hours: list[float],
    days: int = 7,
    sessions_per_hour: int = 3,
) -> None:
    for d in range(days):
        for h in hours:
            for s in range(sessions_per_hour):
                local_dt = day_start_local + timedelta(days=d, hours=h, minutes=5 * s)
                if local_dt.tzinfo is None:
                    local_dt = local_dt.replace(tzinfo=local_tz)
                utc_dt = local_dt.astimezone(timezone.utc)
                _insert_event_direct(store, kind="session_started", ts=utc_dt, data={"n": s})

def _insert_event_direct(store, *, kind: str, ts: datetime, data: dict) -> None:
    import json
    from uuid import uuid4

    from iai_mcp.crypto import encrypt_field
    from iai_mcp.store import EVENTS_TABLE

    event_id = str(uuid4())
    data_plain = json.dumps(data)
    ad = event_id.encode("ascii")
    data_ct = encrypt_field(data_plain, store._key(), associated_data=ad)
    row = {
        "id": event_id,
        "kind": kind,
        "severity": "",
        "domain": "",
        "ts": ts,
        "data_json": data_ct,
        "session_id": "-",
        "source_ids_json": json.dumps([]),
    }
    store.db.open_table(EVENTS_TABLE).add([row])

def test_western_9_to_5_user(tmp_path, monkeypatch):
    from iai_mcp.quiet_window import (
        BUCKET_MINUTES,
        learn_quiet_window,
    )

    tz = ZoneInfo("America/New_York")
    store = _fresh_store(tmp_path, monkeypatch)

    day_start = datetime(2026, 4, 1, 0, 0).replace(tzinfo=tz)
    _seed_sessions(
        store,
        local_tz=tz,
        day_start_local=day_start,
        hours=[9, 10, 11, 12, 13, 14, 15, 16, 17],
        days=7,
        sessions_per_hour=3,
    )

    now = (day_start + timedelta(days=7, hours=8)).astimezone(timezone.utc)
    result = learn_quiet_window(store, now, tz)
    assert result is not None, "should detect quiet window for 9-5 user"
    start_bucket, duration = result

    start_hour = (start_bucket * BUCKET_MINUTES) // 60
    start_minute = (start_bucket * BUCKET_MINUTES) % 60
    in_evening = start_hour >= 17 and (start_hour > 17 or start_minute >= 30)
    in_early_morning = start_hour <= 2
    assert in_evening or in_early_morning, (
        f"expected quiet start in 17:30-02:00 evening/night band, "
        f"got {start_hour}:{start_minute:02d} (bucket={start_bucket})"
    )
    assert 6 <= duration <= 16, f"duration out of range: {duration}"

def test_nocturnal_autistic_user(tmp_path, monkeypatch):
    from iai_mcp.quiet_window import BUCKET_MINUTES, learn_quiet_window

    tz = ZoneInfo("Europe/Moscow")
    store = _fresh_store(tmp_path, monkeypatch)

    day_start = datetime(2026, 4, 1, 0, 0).replace(tzinfo=tz)
    _seed_sessions(
        store,
        local_tz=tz,
        day_start_local=day_start,
        hours=[22, 23, 24, 25, 26, 27, 28],
        days=7,
        sessions_per_hour=3,
    )

    now = (day_start + timedelta(days=7, hours=12)).astimezone(timezone.utc)
    result = learn_quiet_window(store, now, tz)
    assert result is not None, "should detect quiet window for nocturnal user"
    start_bucket, duration = result
    start_hour = (start_bucket * BUCKET_MINUTES) // 60
    assert 4 <= start_hour <= 21, (
        f"nocturnal: expected quiet start in 04-21 band, got {start_hour}:00"
    )
    end_bucket = (start_bucket + duration) % 48
    end_hour = (end_bucket * BUCKET_MINUTES) // 60
    assert end_hour <= 22, (
        f"nocturnal window ends at {end_hour}:00, overlaps active 22-04 band"
    )
    assert 6 <= duration <= 16

def test_shift_worker_alternating(tmp_path, monkeypatch):
    from iai_mcp.quiet_window import learn_quiet_window

    tz = ZoneInfo("UTC")
    store = _fresh_store(tmp_path, monkeypatch)

    day_start = datetime(2026, 4, 1, 0, 0).replace(tzinfo=tz)
    for d in range(7):
        if d in (0, 1, 4, 5):
            hours = [8, 9, 10, 11, 12, 13, 14, 15]
        else:
            hours = [20, 21, 22, 23, 24, 25, 26, 27]
        _seed_sessions(
            store,
            local_tz=tz,
            day_start_local=day_start + timedelta(days=d),
            hours=hours,
            days=1,
            sessions_per_hour=2,
        )

    now = (day_start + timedelta(days=7)).astimezone(timezone.utc)
    result = learn_quiet_window(store, now, tz)
    if result is not None:
        start_bucket, duration = result
        assert 0 <= start_bucket < 48
        assert 6 <= duration <= 16

def test_new_user_insufficient_days(tmp_path, monkeypatch):
    from iai_mcp.quiet_window import learn_quiet_window

    tz = ZoneInfo("UTC")
    store = _fresh_store(tmp_path, monkeypatch)

    day_start = datetime(2026, 4, 1, 0, 0).replace(tzinfo=tz)
    _seed_sessions(
        store,
        local_tz=tz,
        day_start_local=day_start,
        hours=[9, 10, 11, 12, 13],
        days=2,
        sessions_per_hour=3,
    )

    now = (day_start + timedelta(days=2, hours=14)).astimezone(timezone.utc)
    result = learn_quiet_window(store, now, tz)
    assert result is None, "should return None when <7d data"

def test_24_7_user_no_quiet_span(tmp_path, monkeypatch):
    from iai_mcp.quiet_window import learn_quiet_window

    tz = ZoneInfo("UTC")
    store = _fresh_store(tmp_path, monkeypatch)

    day_start = datetime(2026, 4, 1, 0, 0).replace(tzinfo=tz)
    _seed_sessions(
        store,
        local_tz=tz,
        day_start_local=day_start,
        hours=list(range(24)),
        days=7,
        sessions_per_hour=3,
    )

    now = (day_start + timedelta(days=7)).astimezone(timezone.utc)
    result = learn_quiet_window(store, now, tz)
    assert result is None, "24/7 uniform user should return None"

def test_dst_spring_forward_no_crash(tmp_path, monkeypatch):
    from iai_mcp.quiet_window import learn_quiet_window

    tz = ZoneInfo("America/New_York")
    store = _fresh_store(tmp_path, monkeypatch)

    day_start = datetime(2026, 3, 5, 0, 0).replace(tzinfo=tz)
    _seed_sessions(
        store,
        local_tz=tz,
        day_start_local=day_start,
        hours=[9, 10, 12, 14, 17, 20],
        days=7,
        sessions_per_hour=2,
    )

    now = (day_start + timedelta(days=7)).astimezone(timezone.utc)
    result = learn_quiet_window(store, now, tz)
    if result is not None:
        start_bucket, duration = result
        assert 0 <= start_bucket < 48
        assert 6 <= duration <= 16

def test_should_relearn_24h_cadence():
    from iai_mcp.quiet_window import should_relearn

    now = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    assert should_relearn(None, now) is True
    assert should_relearn(now - timedelta(hours=25), now) is True
    assert should_relearn(now - timedelta(hours=24), now) is True
    assert should_relearn(now - timedelta(hours=12), now) is False

