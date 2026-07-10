from __future__ import annotations




def test_budget_ledger_daily_cap_enforced(tmp_path):
    from iai_mcp.guard import BudgetLedger
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    bl = BudgetLedger(store, daily_usd_cap=0.10, monthly_usd_cap=3.00)

    ok, _ = bl.can_spend(0.05)
    assert ok is True

    bl.record_spend(0.08)
    ok, _ = bl.can_spend(0.03)
    ok2, reason = bl.can_spend(0.03)
    assert ok2 is False
    assert "daily" in reason.lower()


def test_budget_ledger_daily_allows_under_cap(tmp_path):
    from iai_mcp.guard import BudgetLedger
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    bl = BudgetLedger(store, daily_usd_cap=0.10)
    bl.record_spend(0.05)
    ok, _ = bl.can_spend(0.04)
    assert ok is True


def test_budget_ledger_monthly_cap_enforced(tmp_path):
    from iai_mcp.guard import BudgetLedger
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    bl = BudgetLedger(store, daily_usd_cap=10.0, monthly_usd_cap=0.20)
    bl.record_spend(0.15)
    ok, reason = bl.can_spend(0.10)
    assert ok is False
    assert "monthly" in reason.lower()


def test_budget_ledger_daily_used(tmp_path):
    from iai_mcp.guard import BudgetLedger
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    bl = BudgetLedger(store)
    assert bl.daily_used() == 0.0
    bl.record_spend(0.01)
    bl.record_spend(0.02)
    assert abs(bl.daily_used() - 0.03) < 1e-5


def test_budget_ledger_monthly_used(tmp_path):
    from iai_mcp.guard import BudgetLedger
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    bl = BudgetLedger(store)
    bl.record_spend(0.05)
    bl.record_spend(0.03)
    assert abs(bl.monthly_used() - 0.08) < 1e-5


def test_budget_ledger_persists_across_reopen(tmp_path):
    from iai_mcp.guard import BudgetLedger
    from iai_mcp.store import MemoryStore

    store1 = MemoryStore(path=tmp_path)
    BudgetLedger(store1).record_spend(0.05)
    del store1

    store2 = MemoryStore(path=tmp_path)
    bl = BudgetLedger(store2)
    assert abs(bl.daily_used() - 0.05) < 1e-5


def test_ratelimit_ledger_no_history_not_in_cooldown(tmp_path):
    from iai_mcp.guard import RateLimitLedger
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    rl = RateLimitLedger(store)
    assert rl.in_cooldown() is False


def test_ratelimit_ledger_record_429_enters_cooldown(tmp_path):
    from iai_mcp.guard import RateLimitLedger
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    rl = RateLimitLedger(store)
    rl.record_429()
    assert rl.in_cooldown() is True


def test_ratelimit_ledger_persists_across_reopen(tmp_path):
    from iai_mcp.guard import RateLimitLedger
    from iai_mcp.store import MemoryStore

    store1 = MemoryStore(path=tmp_path)
    RateLimitLedger(store1).record_429()
    del store1

    store2 = MemoryStore(path=tmp_path)
    assert RateLimitLedger(store2).in_cooldown() is True


def test_should_call_llm_tier_0_fallback_llm_disabled(tmp_path):
    from iai_mcp.guard import BudgetLedger, RateLimitLedger, should_call_llm
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    bl = BudgetLedger(store)
    rl = RateLimitLedger(store)
    ok, reason = should_call_llm(bl, rl, llm_enabled=False, has_api_key=True)
    assert ok is False
    assert "llm_enabled" in reason


def test_should_call_llm_no_api_key(tmp_path):
    from iai_mcp.guard import BudgetLedger, RateLimitLedger, should_call_llm
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    bl = BudgetLedger(store)
    rl = RateLimitLedger(store)
    ok, reason = should_call_llm(bl, rl, llm_enabled=True, has_api_key=False)
    assert ok is False
    assert "api key" in reason.lower()


def test_should_call_llm_daily_cap_hit(tmp_path):
    from iai_mcp.guard import BudgetLedger, RateLimitLedger, should_call_llm
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    bl = BudgetLedger(store, daily_usd_cap=0.01, monthly_usd_cap=3.0)
    bl.record_spend(0.009)
    rl = RateLimitLedger(store)
    ok, reason = should_call_llm(
        bl, rl, llm_enabled=True, has_api_key=True, estimated_usd=0.005
    )
    assert ok is False
    assert "daily" in reason.lower()


def test_should_call_llm_monthly_cap_hit(tmp_path):
    from iai_mcp.guard import BudgetLedger, RateLimitLedger, should_call_llm
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    bl = BudgetLedger(store, daily_usd_cap=10.0, monthly_usd_cap=0.02)
    bl.record_spend(0.015)
    rl = RateLimitLedger(store)
    ok, reason = should_call_llm(
        bl, rl, llm_enabled=True, has_api_key=True, estimated_usd=0.01
    )
    assert ok is False
    assert "monthly" in reason.lower()


def test_should_call_llm_in_cooldown(tmp_path):
    from iai_mcp.guard import BudgetLedger, RateLimitLedger, should_call_llm
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    bl = BudgetLedger(store)
    rl = RateLimitLedger(store)
    rl.record_429()
    ok, reason = should_call_llm(bl, rl, llm_enabled=True, has_api_key=True)
    assert ok is False
    assert "cooldown" in reason.lower()


def test_should_call_llm_all_green(tmp_path):
    from iai_mcp.guard import BudgetLedger, RateLimitLedger, should_call_llm
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    bl = BudgetLedger(store)
    rl = RateLimitLedger(store)
    ok, reason = should_call_llm(bl, rl, llm_enabled=True, has_api_key=True)
    assert ok is True
    assert reason == "ok"


def test_should_call_llm_ordering_llm_enabled_first(tmp_path):
    from iai_mcp.guard import BudgetLedger, RateLimitLedger, should_call_llm
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    bl = BudgetLedger(store, daily_usd_cap=0.01)
    bl.record_spend(0.02)
    rl = RateLimitLedger(store)
    rl.record_429()

    ok, reason = should_call_llm(bl, rl, llm_enabled=False, has_api_key=False)
    assert ok is False
    assert "llm_enabled" in reason


def test_should_call_llm_ordering_cap_before_cooldown(tmp_path):
    from iai_mcp.guard import BudgetLedger, RateLimitLedger, should_call_llm
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    bl = BudgetLedger(store, daily_usd_cap=0.01)
    bl.record_spend(0.02)
    rl = RateLimitLedger(store)
    rl.record_429()

    ok, reason = should_call_llm(
        bl, rl, llm_enabled=True, has_api_key=True, estimated_usd=0.001
    )
    assert ok is False
    assert "daily" in reason.lower()
