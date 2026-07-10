from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4
from zoneinfo import ZoneInfo


from iai_mcp.types import EMBED_DIM, MemoryRecord

def _record(
    *,
    text: str = "hi",
    vec: list[float] | None = None,
    tags: list[str] | None = None,
    tier: str = "episodic",
    detail_level: int = 2,
    language: str = "en",
    never_decay: bool = False,
) -> MemoryRecord:
    if vec is None:
        vec = [1.0] + [0.0] * (EMBED_DIM - 1)
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=vec,
        community_id=None,
        centrality=0.0,
        detail_level=detail_level,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=never_decay,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=list(tags or []),
        language=language,
    )

def test_sleep_mode_enum_has_three_values():
    from iai_mcp.sleep import SleepMode

    assert SleepMode.ACTIVITY.value == "activity"
    assert SleepMode.TIME.value == "time"
    assert SleepMode.MANUAL.value == "manual"

def test_sleep_config_defaults():
    from iai_mcp.sleep import SleepConfig, SleepMode

    cfg = SleepConfig()
    assert cfg.mode == SleepMode.ACTIVITY
    assert cfg.quiet_window == (22, 6)
    assert cfg.require_idle_minutes == 30
    assert cfg.max_defer_hours == 48
    assert cfg.llm_enabled is False
    assert cfg.light_on_exit is True

def test_should_run_heavy_activity_mode_inside_window():
    from iai_mcp.sleep import SleepConfig, SleepMode, should_run_heavy

    cfg = SleepConfig(mode=SleepMode.ACTIVITY)
    tz = ZoneInfo("UTC")
    now = datetime(2026, 1, 1, 23, 30, tzinfo=timezone.utc)
    last = now - timedelta(minutes=40)
    ok, reason = should_run_heavy(now, last, cfg, tz)
    assert ok is True
    assert reason == ""

def test_should_run_heavy_activity_mode_outside_window():
    from iai_mcp.sleep import SleepConfig, SleepMode, should_run_heavy

    cfg = SleepConfig(mode=SleepMode.ACTIVITY)
    tz = ZoneInfo("UTC")
    now = datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc)
    last = now - timedelta(minutes=40)
    ok, reason = should_run_heavy(now, last, cfg, tz)
    assert ok is False
    assert "quiet window" in reason.lower() or "outside" in reason.lower()

def test_should_run_heavy_activity_mode_too_recent():
    from iai_mcp.sleep import SleepConfig, SleepMode, should_run_heavy

    cfg = SleepConfig(mode=SleepMode.ACTIVITY)
    tz = ZoneInfo("UTC")
    now = datetime(2026, 1, 1, 23, 30, tzinfo=timezone.utc)
    last = now - timedelta(minutes=5)
    ok, reason = should_run_heavy(now, last, cfg, tz)
    assert ok is False
    assert "idle" in reason.lower()

def test_should_run_heavy_time_mode_only_at_3am():
    from iai_mcp.sleep import SleepConfig, SleepMode, should_run_heavy

    cfg = SleepConfig(mode=SleepMode.TIME)
    tz = ZoneInfo("UTC")
    now_2am = datetime(2026, 1, 1, 2, 30, tzinfo=timezone.utc)
    ok_2, _ = should_run_heavy(now_2am, now_2am - timedelta(hours=1), cfg, tz)
    assert ok_2 is False

    now_3am = datetime(2026, 1, 1, 3, 30, tzinfo=timezone.utc)
    ok_3, _ = should_run_heavy(now_3am, now_3am - timedelta(hours=1), cfg, tz)
    assert ok_3 is True

def test_should_run_heavy_manual_mode_never_auto():
    from iai_mcp.sleep import SleepConfig, SleepMode, should_run_heavy

    cfg = SleepConfig(mode=SleepMode.MANUAL)
    tz = ZoneInfo("UTC")
    now = datetime(2026, 1, 1, 23, 30, tzinfo=timezone.utc)
    last = now - timedelta(minutes=40)
    ok, reason = should_run_heavy(now, last, cfg, tz)
    assert ok is False
    assert "manual" in reason.lower()

def test_should_run_heavy_48h_force():
    from iai_mcp.sleep import SleepConfig, SleepMode, should_run_heavy

    cfg = SleepConfig(mode=SleepMode.ACTIVITY)
    tz = ZoneInfo("UTC")
    now = datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc)
    last = now - timedelta(hours=50)
    ok, reason = should_run_heavy(now, last, cfg, tz)
    assert ok is True
    assert "defer" in reason.lower() or "48" in reason

def test_should_run_heavy_respects_user_tz_tokyo():
    from iai_mcp.sleep import SleepConfig, SleepMode, should_run_heavy

    cfg = SleepConfig(mode=SleepMode.ACTIVITY)
    tz = ZoneInfo("Asia/Tokyo")
    now = datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc)
    last = now - timedelta(minutes=40)
    ok, reason = should_run_heavy(now, last, cfg, tz)
    assert ok is True

def test_should_run_heavy_respects_user_tz_utc():
    from iai_mcp.sleep import SleepConfig, SleepMode, should_run_heavy

    cfg = SleepConfig(mode=SleepMode.ACTIVITY)
    tz = ZoneInfo("UTC")
    now = datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc)
    last = now - timedelta(minutes=40)
    ok, reason = should_run_heavy(now, last, cfg, tz)
    assert ok is False

def test_run_light_consolidation_returns_expected_shape(tmp_path):
    from iai_mcp.sleep import run_light_consolidation
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    result = run_light_consolidation(store, session_id="s-light")
    assert isinstance(result, dict)
    assert "fsrs_ticked" in result
    assert "cooccurrence_updates" in result
    assert result["mode"] == "light"

def test_run_light_consolidation_no_llm_call(tmp_path, monkeypatch):
    from iai_mcp import sleep as sleep_mod
    from iai_mcp.sleep import run_light_consolidation
    from iai_mcp.store import MemoryStore

    call_count = {"n": 0}
    original_should = sleep_mod.should_call_llm

    def _counting(*args, **kwargs):
        call_count["n"] += 1
        return original_should(*args, **kwargs)

    monkeypatch.setattr(sleep_mod, "should_call_llm", _counting)

    store = MemoryStore(path=tmp_path)
    store.insert(_record())

    run_light_consolidation(store, session_id="s-light")
    assert call_count["n"] == 0

def test_run_light_consolidation_emits_event(tmp_path):
    from iai_mcp.events import query_events
    from iai_mcp.sleep import run_light_consolidation
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    run_light_consolidation(store, session_id="s-x")
    events = query_events(store, kind="cls_consolidation_run")
    assert len(events) >= 1
    ev = events[0]
    assert ev["data"]["mode"] == "light"
    assert ev["session_id"] == "s-x"

def test_run_heavy_consolidation_uses_d_guard(tmp_path, monkeypatch):
    from iai_mcp.events import query_events
    from iai_mcp.guard import BudgetLedger, RateLimitLedger
    from iai_mcp.sleep import SleepConfig, run_heavy_consolidation
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    recs = [_record(text=f"rec {i}") for i in range(3)]
    for r in recs:
        store.insert(r)
    store.boost_edges([(recs[0].id, recs[1].id)], edge_type="hebbian", delta=0.5)
    store.boost_edges([(recs[1].id, recs[2].id)], edge_type="hebbian", delta=0.5)
    store.boost_edges([(recs[0].id, recs[2].id)], edge_type="hebbian", delta=0.5)

    cfg = SleepConfig(llm_enabled=False)
    budget = BudgetLedger(store)
    rate = RateLimitLedger(store)

    result = run_heavy_consolidation(
        store, session_id="s-heavy", config=cfg, budget=budget, rate=rate,
        has_api_key=False,
    )
    assert result["mode"] == "heavy"
    assert result["tier"] == "tier0"

    events = query_events(store, kind="cls_consolidation_run")
    heavy_events = [e for e in events if e["data"].get("mode") == "heavy"]
    assert len(heavy_events) >= 1
    assert heavy_events[0]["data"]["tier"] == "tier0"

def test_run_heavy_consolidation_creates_consolidated_from_edges(tmp_path):
    from iai_mcp.guard import BudgetLedger, RateLimitLedger
    from iai_mcp.sleep import SleepConfig, run_heavy_consolidation
    from iai_mcp.store import EDGES_TABLE, MemoryStore

    store = MemoryStore(path=tmp_path)
    recs = [_record(text=f"fact {i}") for i in range(3)]
    for r in recs:
        store.insert(r)
    store.boost_edges([(recs[0].id, recs[1].id)], edge_type="hebbian", delta=0.5)
    store.boost_edges([(recs[1].id, recs[2].id)], edge_type="hebbian", delta=0.5)
    store.boost_edges([(recs[0].id, recs[2].id)], edge_type="hebbian", delta=0.5)

    cfg = SleepConfig(llm_enabled=False)
    budget = BudgetLedger(store)
    rate = RateLimitLedger(store)
    result = run_heavy_consolidation(
        store, session_id="s-cons", config=cfg, budget=budget, rate=rate,
        has_api_key=False,
    )
    assert result["summaries_created"] >= 1

    edges_df = store.db.open_table(EDGES_TABLE).to_pandas()
    cf = edges_df[edges_df["edge_type"] == "consolidated_from"]
    assert len(cf) >= 3

def test_run_heavy_consolidation_mem01_preserves_sources(tmp_path):
    from iai_mcp.guard import BudgetLedger, RateLimitLedger
    from iai_mcp.sleep import SleepConfig, run_heavy_consolidation
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    literals = ["fact alpha", "fact beta", "fact gamma"]
    recs = [_record(text=t) for t in literals]
    for r in recs:
        store.insert(r)
    store.boost_edges(
        [(recs[0].id, recs[1].id), (recs[1].id, recs[2].id), (recs[0].id, recs[2].id)],
        edge_type="hebbian", delta=0.5,
    )

    run_heavy_consolidation(
        store, session_id="s", config=SleepConfig(llm_enabled=False),
        budget=BudgetLedger(store), rate=RateLimitLedger(store),
        has_api_key=False,
    )

    for rec, expected in zip(recs, literals):
        reloaded = store.get(rec.id)
        assert reloaded is not None
        assert reloaded.literal_surface == expected

def test_run_heavy_consolidation_empty_store(tmp_path):
    from iai_mcp.guard import BudgetLedger, RateLimitLedger
    from iai_mcp.sleep import SleepConfig, run_heavy_consolidation
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    result = run_heavy_consolidation(
        store, session_id="s", config=SleepConfig(llm_enabled=False),
        budget=BudgetLedger(store), rate=RateLimitLedger(store),
        has_api_key=False,
    )
    assert result["summaries_created"] == 0

def test_run_heavy_consolidation_no_cluster_below_threshold(tmp_path):
    from iai_mcp.guard import BudgetLedger, RateLimitLedger
    from iai_mcp.sleep import SleepConfig, run_heavy_consolidation
    from iai_mcp.store import EDGES_TABLE, MemoryStore

    store = MemoryStore(path=tmp_path)
    r1, r2 = _record(text="a"), _record(text="b")
    store.insert(r1)
    store.insert(r2)
    store.boost_edges([(r1.id, r2.id)], edge_type="hebbian", delta=0.5)

    run_heavy_consolidation(
        store, session_id="s", config=SleepConfig(llm_enabled=False),
        budget=BudgetLedger(store), rate=RateLimitLedger(store),
        has_api_key=False,
    )

    edges_df = store.db.open_table(EDGES_TABLE).to_pandas()
    cf = edges_df[edges_df["edge_type"] == "consolidated_from"]
    assert len(cf) == 0
