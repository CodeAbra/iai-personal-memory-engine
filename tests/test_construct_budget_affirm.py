from __future__ import annotations



def test_default_construct_budget_ms_is_1000():
    from iai_mcp.semantic_recall import _DEFAULT_CONSTRUCT_BUDGET_MS
    assert _DEFAULT_CONSTRUCT_BUDGET_MS == 1000


def test_env_override_construct_budget_ms_parses(monkeypatch):
    from iai_mcp.semantic_recall import _construct_budget_ms

    monkeypatch.setenv("IAI_MCP_EMBED_CONSTRUCT_BUDGET_MS", "500")
    assert _construct_budget_ms() == 500

    monkeypatch.setenv("IAI_MCP_EMBED_CONSTRUCT_BUDGET_MS", "not-a-number")
    assert _construct_budget_ms() == 1000

    monkeypatch.delenv("IAI_MCP_EMBED_CONSTRUCT_BUDGET_MS", raising=False)
    assert _construct_budget_ms() == 1000
