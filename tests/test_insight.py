from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make


def _fresh_store(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai"))
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "384")
    from iai_mcp.store import MemoryStore
    return MemoryStore()


def _seed_evidence(store, monkeypatch, n_patterns: int = 3):
    """Insert real records and route them through tier0 as pattern evidence.

    The insight mint requires verified consolidated_from sources; every test
    that expects a mint seeds them here.
    """
    from iai_mcp.schema import SchemaCandidate

    recs = []
    for i in range(max(1, n_patterns)):
        rec = _make(text=f"evidence record {i}")
        store.insert(rec)
        recs.append(rec)
    candidates = [
        SchemaCandidate(
            pattern=f"pattern-{i}",
            confidence=0.1 * (i + 1),
            evidence_count=3 + i,
            evidence_ids=[recs[min(i, len(recs) - 1)].id],
        )
        for i in range(n_patterns)
    ]
    monkeypatch.setattr(
        "iai_mcp.insight.induce_schemas_tier0",
        lambda _store: candidates,
    )
    return recs


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    from iai_mcp import daemon_state
    state_path = tmp_path / ".daemon-state.json"
    monkeypatch.setattr(daemon_state, "STATE_PATH", state_path)
    return state_path


@pytest.fixture
def creds_ok(monkeypatch):
    monkeypatch.setattr(
        "iai_mcp.insight.verify_credentials_subscription",
        lambda: {"ok": True, "billing_type": "stripe_subscription"},
    )


@pytest.fixture
def mock_claude_ok(monkeypatch, creds_ok, isolated_state):
    calls: list[dict] = []

    async def fake_invoke(prompt: str, *, model: str = "haiku", **_kw):
        calls.append({"prompt": prompt, "model": model})
        return {
            "ok": True,
            "data": {"result": "unifying insight text"},
            "tokens_in": 200,
            "tokens_out": 40,
            "cost_usd": 0.0,
        }

    monkeypatch.setattr("iai_mcp.insight.invoke_reflection_once", fake_invoke)
    return calls


def test_one_call_per_night(tmp_path, monkeypatch, mock_claude_ok):
    from iai_mcp.insight import generate_overnight_insight
    store = _fresh_store(tmp_path, monkeypatch)
    _seed_evidence(store, monkeypatch)

    result = asyncio.run(generate_overnight_insight(store, "sess-A"))
    assert result["ok"] is True
    assert len(mock_claude_ok) == 1

    asyncio.run(generate_overnight_insight(store, "sess-B"))
    assert len(mock_claude_ok) == 2


def test_prompt_template(tmp_path, monkeypatch, mock_claude_ok):
    from iai_mcp.insight import generate_overnight_insight
    store = _fresh_store(tmp_path, monkeypatch)
    _seed_evidence(store, monkeypatch)
    asyncio.run(generate_overnight_insight(store, "sess-A"))

    prompt = mock_claude_ok[0]["prompt"]
    assert "3 locally-found patterns" in prompt
    assert "1 surprising episode" in prompt
    assert "unifying insight" in prompt
    assert "1-2 sentences" in prompt
    assert mock_claude_ok[0]["model"] == "haiku"


def test_patterns_from_schemas(tmp_path, monkeypatch, mock_claude_ok):
    from iai_mcp.insight import generate_overnight_insight

    store = _fresh_store(tmp_path, monkeypatch)
    _seed_evidence(store, monkeypatch, n_patterns=5)

    asyncio.run(generate_overnight_insight(store, "sess-A"))
    prompt = mock_claude_ok[0]["prompt"]

    assert "pattern-4" in prompt
    assert "pattern-3" in prompt
    assert "pattern-2" in prompt


def test_surprise_from_events(tmp_path, monkeypatch, mock_claude_ok):
    from iai_mcp.insight import generate_overnight_insight

    store = _fresh_store(tmp_path, monkeypatch)
    rec = _make(text="surprise source record")
    store.insert(rec)

    monkeypatch.setattr("iai_mcp.insight.induce_schemas_tier0", lambda _s: [])
    fake_events = [
        {"kind": "art_gate_high_novelty",
         "data": {"summary": "UNEXPECTED-MARKER-ALPHA"},
         "source_ids": [str(rec.id)], "ts": "x"},
        {"kind": "routine_event", "data": {"summary": "boring"}, "ts": "y"},
    ]
    monkeypatch.setattr(
        "iai_mcp.insight.query_events",
        lambda _store, *, since=None, limit=1000: fake_events,
    )

    asyncio.run(generate_overnight_insight(store, "sess-A"))
    prompt = mock_claude_ok[0]["prompt"]
    assert "UNEXPECTED-MARKER-ALPHA" in prompt


def test_record_tag(tmp_path, monkeypatch, mock_claude_ok):
    from iai_mcp.insight import generate_overnight_insight

    store = _fresh_store(tmp_path, monkeypatch)
    _seed_evidence(store, monkeypatch)
    inserted: list = []

    real_insert = store.insert

    def spy_insert(rec):
        inserted.append(rec)
        return real_insert(rec)

    monkeypatch.setattr(store, "insert", spy_insert)

    result = asyncio.run(generate_overnight_insight(store, "sess-A"))
    assert result["ok"] is True
    minted = [
        r for r in inserted
        if "overnight_insight" in (getattr(r, "tags", []) or [])
    ]
    assert len(minted) == 1
    rec = minted[0]
    assert rec.tier == "semantic"
    assert rec.literal_surface == "unifying insight text"


def test_insight_mint_carries_evidence_edges(tmp_path, monkeypatch, mock_claude_ok):
    from iai_mcp.insight import generate_overnight_insight

    store = _fresh_store(tmp_path, monkeypatch)
    sources = _seed_evidence(store, monkeypatch)
    inserted: list = []
    real_insert = store.insert
    monkeypatch.setattr(
        store, "insert", lambda r: inserted.append(r) or real_insert(r),
    )

    result = asyncio.run(generate_overnight_insight(store, "sess-A"))
    assert result["ok"] is True
    minted = [
        r for r in inserted
        if "overnight_insight" in (getattr(r, "tags", []) or [])
    ]
    insight_id = minted[0].id

    edges = store.incident_edges(
        [insight_id], edge_types=["consolidated_from"], top_k=None,
    )
    neighbors = {str(t[0]) for lst in edges.values() for t in lst}
    source_strs = {str(s.id) for s in sources}
    assert neighbors & source_strs, (
        f"insight minted without consolidated_from edges to its sources; "
        f"edges={edges}"
    )


def test_no_evidence_sources_skips_mint(tmp_path, monkeypatch, mock_claude_ok):
    from iai_mcp.insight import generate_overnight_insight

    store = _fresh_store(tmp_path, monkeypatch)
    monkeypatch.setattr("iai_mcp.insight.induce_schemas_tier0", lambda _s: [])
    monkeypatch.setattr(
        "iai_mcp.insight.query_events",
        lambda _s, *, since=None, limit=1000: [],
    )

    result = asyncio.run(generate_overnight_insight(store, "sess-A"))
    assert result["ok"] is False
    assert result["reason"] == "no_evidence_sources"
    assert len(mock_claude_ok) == 0, "grounding gate must fire before the LLM call"


def test_nonexistent_sources_do_not_count(tmp_path, monkeypatch, mock_claude_ok):
    from uuid import uuid4 as _uuid4

    from iai_mcp.insight import generate_overnight_insight
    from iai_mcp.schema import SchemaCandidate

    store = _fresh_store(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "iai_mcp.insight.induce_schemas_tier0",
        lambda _s: [SchemaCandidate(
            pattern="ghost", confidence=0.9, evidence_count=5,
            evidence_ids=[_uuid4(), _uuid4()],
        )],
    )
    monkeypatch.setattr(
        "iai_mcp.insight.query_events",
        lambda _s, *, since=None, limit=1000: [],
    )

    result = asyncio.run(generate_overnight_insight(store, "sess-A"))
    assert result["ok"] is False
    assert result["reason"] == "no_evidence_sources"
    assert len(mock_claude_ok) == 0


def test_budget_gate_blocks(tmp_path, monkeypatch, creds_ok, isolated_state):
    from iai_mcp.claude_cli import BUDGET_STATE_KEY, DAILY_QUOTA_BUDGET_PCT, ESTIMATED_DAILY_TOKEN_CEILING
    from iai_mcp.daemon_state import save_state
    from iai_mcp.insight import generate_overnight_insight

    store = _fresh_store(tmp_path, monkeypatch)

    calls: list = []

    async def fake_invoke(prompt, *, model="haiku", **_kw):
        calls.append(1)
        return {"ok": True, "data": {"result": "x"}, "tokens_in": 1, "tokens_out": 1, "cost_usd": 0.0}

    monkeypatch.setattr("iai_mcp.insight.invoke_reflection_once", fake_invoke)

    daily_cap = int(DAILY_QUOTA_BUDGET_PCT * ESTIMATED_DAILY_TOKEN_CEILING)
    save_state({BUDGET_STATE_KEY: {
        "daily_used_tokens": daily_cap,
        "weekly_buffer_used_tokens": 10_000_000,
        "last_reset_date": datetime.now(timezone.utc).date().isoformat(),
        "claude_disabled": False,
        "claude_disabled_reason": None,
    }})

    result = asyncio.run(generate_overnight_insight(store, "sess-A"))
    assert result["ok"] is False
    assert result["reason"] == "budget_exceeded"
    assert calls == []


def test_claude_disabled_blocks(tmp_path, monkeypatch, creds_ok, isolated_state):
    from iai_mcp.claude_cli import BUDGET_STATE_KEY
    from iai_mcp.daemon_state import save_state
    from iai_mcp.insight import generate_overnight_insight

    store = _fresh_store(tmp_path, monkeypatch)

    calls: list = []

    async def fake_invoke(prompt, *, model="haiku", **_kw):
        calls.append(1)
        return {"ok": True, "data": {"result": "x"}, "tokens_in": 1, "tokens_out": 1, "cost_usd": 0.0}

    monkeypatch.setattr("iai_mcp.insight.invoke_reflection_once", fake_invoke)
    save_state({BUDGET_STATE_KEY: {
        "daily_used_tokens": 0,
        "weekly_buffer_used_tokens": 0,
        "last_reset_date": datetime.now(timezone.utc).date().isoformat(),
        "claude_disabled": True,
        "claude_disabled_reason": "api_billing_detected",
    }})

    result = asyncio.run(generate_overnight_insight(store, "sess-A"))
    assert result["ok"] is False
    assert result["reason"] == "claude_disabled_c3"
    assert calls == []


def test_credentials_gate_blocks(tmp_path, monkeypatch, isolated_state):
    from iai_mcp.insight import generate_overnight_insight

    store = _fresh_store(tmp_path, monkeypatch)
    calls: list = []

    async def fake_invoke(prompt, *, model="haiku", **_kw):
        calls.append(1)
        return {"ok": True}

    monkeypatch.setattr("iai_mcp.insight.invoke_reflection_once", fake_invoke)
    monkeypatch.setattr(
        "iai_mcp.insight.verify_credentials_subscription",
        lambda: {"ok": False, "reason": "not_subscription"},
    )

    result = asyncio.run(generate_overnight_insight(store, "sess-A"))
    assert result["ok"] is False
    assert result["reason"] == "credentials_check_failed"
    assert calls == []


def test_budget_recorded(tmp_path, monkeypatch, mock_claude_ok, isolated_state):
    from iai_mcp.claude_cli import BUDGET_STATE_KEY
    from iai_mcp.daemon_state import load_state
    from iai_mcp.insight import generate_overnight_insight

    store = _fresh_store(tmp_path, monkeypatch)
    _seed_evidence(store, monkeypatch)
    asyncio.run(generate_overnight_insight(store, "sess-A"))

    state = load_state()
    assert state[BUDGET_STATE_KEY]["daily_used_tokens"] == 240


def test_api_billing_detected_no_store(tmp_path, monkeypatch, creds_ok, isolated_state):
    from iai_mcp.insight import generate_overnight_insight

    store = _fresh_store(tmp_path, monkeypatch)
    _seed_evidence(store, monkeypatch)
    inserted: list = []
    real_insert = store.insert
    monkeypatch.setattr(store, "insert", lambda r: inserted.append(r) or real_insert(r))

    async def fake_invoke(prompt, *, model="haiku", **_kw):
        return {
            "ok": False,
            "reason": "api_billing_detected",
            "cost_usd": 0.05,
            "data": {"result": "hostile"},
            "tokens_in": 100,
            "tokens_out": 20,
        }

    monkeypatch.setattr("iai_mcp.insight.invoke_reflection_once", fake_invoke)

    result = asyncio.run(generate_overnight_insight(store, "sess-A"))
    assert result["ok"] is False
    assert result["reason"] == "api_billing_detected"
    assert all(
        "overnight_insight" not in (getattr(r, "tags", []) or [])
        and getattr(r, "tag", None) != "overnight_insight"
        for r in inserted
    )


def test_event_emitted(tmp_path, monkeypatch, mock_claude_ok):
    from iai_mcp.events import query_events
    from iai_mcp.insight import generate_overnight_insight

    store = _fresh_store(tmp_path, monkeypatch)
    _seed_evidence(store, monkeypatch)
    asyncio.run(generate_overnight_insight(store, "sess-A"))

    events = query_events(store, kind="overnight_insight_generated", limit=10)
    assert len(events) >= 1
    assert events[0]["data"].get("session_id") == "sess-A"
    assert events[0]["data"].get("sources", 0) >= 1
