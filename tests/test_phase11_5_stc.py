from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import pytest

from iai_mcp.daemon import _load_stc_config
from iai_mcp.events import query_events, write_event
from iai_mcp.peri_event_buffer import (
    PeriEventBuffer,
    PeriEventEntry,
    set_buffer,
)
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord

@pytest.fixture(autouse=True)
def _isolate_iai_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai-mcp-store"))
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    for var in (
        "IAI_MCP_PERI_EVENT_BUFFER_SIZE",
        "IAI_MCP_PERI_EVENT_WINDOW_SEC",
        "IAI_MCP_STC_STRONG_EVENT_TYPES",
        "IAI_MCP_STC_DRY_RUN",
    ):
        monkeypatch.delenv(var, raising=False)

@pytest.fixture
def stc_buffer() -> Iterator[PeriEventBuffer]:
    buf = PeriEventBuffer(maxlen=20)
    set_buffer(buf)
    try:
        yield buf
    finally:
        set_buffer(None)

def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(
        path=str(tmp_path / "iai-mcp-store"),
        user_id="alice",
        read_consistency_interval=timedelta(seconds=0),
    )

def _make_record(
    *,
    embed_dim: int,
    tier: str = "semantic",
    literal: str = "alice prefers tea over coffee",
    created_at: datetime | None = None,
) -> MemoryRecord:
    now = created_at if created_at is not None else datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid.uuid4(),
        tier=tier,
        literal_surface=literal,
        aaak_index="",
        embedding=[0.01] * embed_dim,
        community_id=None,
        centrality=0.0,
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
        language="en",
        tags=["t"],
    )

def test_buffer_maxlen_respected() -> None:
    buf = PeriEventBuffer(maxlen=3)
    now = datetime.now(timezone.utc)
    for i in range(5):
        buf.add(uuid.uuid4(), now, "semantic")
    assert len(buf) == 3
    entries = buf.flush_within_window(now, 3600)
    assert len(entries) == 3
    assert all(isinstance(e, PeriEventEntry) for e in entries)
    assert all(e.original_tier == "semantic" for e in entries)

def test_flush_within_window_filters_by_time() -> None:
    buf = PeriEventBuffer(maxlen=10)
    now = datetime.now(timezone.utc)
    buf.add(uuid.uuid4(), now - timedelta(seconds=10), "semantic")
    buf.add(uuid.uuid4(), now - timedelta(seconds=100), "semantic")
    buf.add(uuid.uuid4(), now - timedelta(seconds=5000), "semantic")

    in_window = buf.flush_within_window(now, 1800)
    assert len(in_window) == 2

    all_entries = buf.flush_within_window(now, 10_000)
    assert len(all_entries) == 3
    assert len(buf) == 3

def test_strong_event_triggers_upgrade_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stc_buffer: PeriEventBuffer,
) -> None:
    monkeypatch.setenv("IAI_MCP_STC_DRY_RUN", "false")

    store = _make_store(tmp_path)
    embed_dim = store._embed_dim

    in_window_records: list[MemoryRecord] = []
    for i in range(5):
        rec = _make_record(
            embed_dim=embed_dim,
            tier="semantic",
            literal=f"alice said something {i}",
        )
        store.insert(rec)
        stc_buffer.add(rec.id, rec.created_at, "semantic")
        in_window_records.append(rec)

    old_ts = datetime.now(timezone.utc) - timedelta(seconds=4000)
    rec_old = _make_record(
        embed_dim=embed_dim,
        tier="semantic",
        literal="bob said something old",
        created_at=old_ts,
    )
    store.insert(rec_old)
    stc_buffer.add(rec_old.id, old_ts, "semantic")

    write_event(store, "memory_capture", {"text": "alice noted a thing"})

    for rec in in_window_records:
        fetched = store.get(rec.id)
        assert fetched is not None
        assert fetched.tier == "episodic", (
            f"in-window record {rec.id} must be upgraded to episodic, "
            f"got {fetched.tier!r}"
        )

    fetched_old = store.get(rec_old.id)
    assert fetched_old is not None
    assert fetched_old.tier == "semantic", (
        f"out-of-window record must NOT be upgraded, got {fetched_old.tier!r}"
    )

    events = query_events(store, kind="stc_upgrade_pass", limit=20)
    assert len(events) == 5, (
        f"expected 5 stc_upgrade_pass events, got {len(events)}"
    )
    expected_keys = {
        "record_id", "from_tier", "to_tier",
        "trigger_event_type", "dry_run_mode",
    }
    in_window_ids = {str(r.id) for r in in_window_records}
    seen_ids: set[str] = set()
    for ev in events:
        body = ev["data"]
        assert set(body.keys()) == expected_keys, (
            f"event body keys must be {expected_keys}, got {set(body.keys())}"
        )
        assert body["from_tier"] == "semantic"
        assert body["to_tier"] == "episodic"
        assert body["trigger_event_type"] == "memory_capture"
        assert body["dry_run_mode"] is False
        assert body["record_id"] in in_window_ids
        seen_ids.add(body["record_id"])
    assert seen_ids == in_window_ids, (
        f"every in-window record must have one event; "
        f"missing={in_window_ids - seen_ids}, extra={seen_ids - in_window_ids}"
    )

def test_upgrade_tier_downgrade_raises_and_event_body_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_STC_DRY_RUN", "false")
    store = _make_store(tmp_path)
    embed_dim = store._embed_dim

    rec_ep = _make_record(
        embed_dim=embed_dim, tier="episodic", literal="alice already episodic",
    )
    store.insert(rec_ep)
    with pytest.raises(ValueError, match="refusing non-upgrade"):
        store.upgrade_tier(
            rec_ep.id, "semantic",
            trigger_event_type="test_downgrade",
            dry_run=False,
        )
    refetched = store.get(rec_ep.id)
    assert refetched is not None
    assert refetched.tier == "episodic"

    with pytest.raises(ValueError, match="refusing non-upgrade"):
        store.upgrade_tier(
            rec_ep.id, "episodic",
            trigger_event_type="test_sametier",
            dry_run=False,
        )

    with pytest.raises(ValueError, match="invalid new_tier"):
        store.upgrade_tier(
            rec_ep.id, "not_a_tier",
            trigger_event_type="test_invalid",
            dry_run=False,
        )

    rec_sem = _make_record(
        embed_dim=embed_dim, tier="semantic", literal="bob said hello",
    )
    store.insert(rec_sem)
    ok = store.upgrade_tier(
        rec_sem.id, "episodic",
        trigger_event_type="test_event",
        dry_run=False,
    )
    assert ok is True
    refetched = store.get(rec_sem.id)
    assert refetched is not None
    assert refetched.tier == "episodic"

    events = query_events(store, kind="stc_upgrade_pass", limit=20)
    matching = [
        ev for ev in events
        if ev["data"].get("record_id") == str(rec_sem.id)
    ]
    assert len(matching) == 1
    body = matching[0]["data"]
    assert set(body.keys()) == {
        "record_id", "from_tier", "to_tier",
        "trigger_event_type", "dry_run_mode",
    }
    assert body["from_tier"] == "semantic"
    assert body["to_tier"] == "episodic"
    assert body["trigger_event_type"] == "test_event"
    assert body["dry_run_mode"] is False

def test_dry_run_preserves_no_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stc_buffer: PeriEventBuffer,
) -> None:
    monkeypatch.setenv("IAI_MCP_STC_DRY_RUN", "true")

    store = _make_store(tmp_path)
    embed_dim = store._embed_dim

    records: list[MemoryRecord] = []
    for i in range(3):
        rec = _make_record(
            embed_dim=embed_dim, tier="semantic",
            literal=f"alice noted {i}",
        )
        store.insert(rec)
        stc_buffer.add(rec.id, rec.created_at, "semantic")
        records.append(rec)

    pre_buffer_len = len(stc_buffer)
    assert pre_buffer_len == 3

    write_event(store, "memory_capture", {"text": "alice triggered a strong event"})

    for rec in records:
        fetched = store.get(rec.id)
        assert fetched is not None
        assert fetched.tier == "semantic", (
            f"dry-run must NOT mutate row tier; got {fetched.tier!r} for {rec.id}"
        )

    events = query_events(store, kind="stc_upgrade_pass", limit=20)
    assert len(events) == 3, (
        f"expected 3 stc_upgrade_pass events in dry-run, got {len(events)}"
    )
    for ev in events:
        body = ev["data"]
        assert body["dry_run_mode"] is True
        assert body["from_tier"] == "semantic"
        assert body["to_tier"] == "episodic"
        assert body["trigger_event_type"] == "memory_capture"

    assert len(stc_buffer) == 3, (
        f"dry-run must preserve buffer entries, got len={len(stc_buffer)}"
    )

@pytest.mark.parametrize(
    "env_var, bad_value",
    [
        ("IAI_MCP_PERI_EVENT_BUFFER_SIZE", "0"),
        ("IAI_MCP_PERI_EVENT_BUFFER_SIZE", "not-an-int"),
        ("IAI_MCP_PERI_EVENT_WINDOW_SEC", "-1"),
        ("IAI_MCP_PERI_EVENT_WINDOW_SEC", "0"),
        ("IAI_MCP_STC_STRONG_EVENT_TYPES", ""),
        ("IAI_MCP_STC_DRY_RUN", "banana"),
    ],
)
def test_env_var_fail_loud(
    monkeypatch: pytest.MonkeyPatch, env_var: str, bad_value: str,
) -> None:
    monkeypatch.setenv(env_var, bad_value)
    with pytest.raises(ValueError, match=env_var):
        _load_stc_config()

def test_stc_upgrade_pass_NOT_in_default_strong_events_no_recursion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stc_buffer: PeriEventBuffer,
) -> None:
    monkeypatch.setenv("IAI_MCP_STC_DRY_RUN", "false")
    cfg = _load_stc_config()
    assert "stc_upgrade_pass" not in cfg.strong_event_types

    store = _make_store(tmp_path)
    embed_dim = store._embed_dim
    rec = _make_record(
        embed_dim=embed_dim, tier="semantic",
        literal="alice peri-event candidate",
    )
    store.insert(rec)
    stc_buffer.add(rec.id, rec.created_at, "semantic")

    write_event(
        store,
        "stc_upgrade_pass",
        {
            "record_id": str(uuid.uuid4()),
            "from_tier": "semantic",
            "to_tier": "episodic",
            "trigger_event_type": "fake_test_emit",
            "dry_run_mode": False,
        },
    )

    fetched = store.get(rec.id)
    assert fetched is not None
    assert fetched.tier == "semantic", (
        f"defense-in-depth: stc_upgrade_pass must NOT fan out by default; "
        f"got {fetched.tier!r}"
    )

    events = query_events(store, kind="stc_upgrade_pass", limit=20)
    assert len(events) == 1, (
        f"expected exactly one stc_upgrade_pass (the fake emit, no cascade); "
        f"got {len(events)}"
    )
    assert events[0]["data"]["trigger_event_type"] == "fake_test_emit"

if __name__ == "__main__":  # pragma: no cover -- direct-run convenience
    raise SystemExit(pytest.main([__file__, "-v"]))
