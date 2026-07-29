from __future__ import annotations



def test_total_session_cost_reports_per_turn():
    from bench.total_session_cost import run_total_session_cost

    out = run_total_session_cost(wake_depth="minimal")

    assert "per_turn" in out
    assert isinstance(out["per_turn"], list)
    assert len(out["per_turn"]) == 10, (
        f"session-cost script has 10 turns; got {len(out['per_turn'])}"
    )
    assert out["total_tokens"] == sum(out["per_turn"])
    assert out["adapter"] == "iai-mcp"
    assert out["wake_depth"] == "minimal"


def test_total_session_cost_minimal_le_standard():
    from bench.total_session_cost import run_total_session_cost

    minimal = run_total_session_cost(wake_depth="minimal")
    standard = run_total_session_cost(wake_depth="standard")

    assert minimal["total_tokens"] <= standard["total_tokens"], (
        f"minimal {minimal['total_tokens']} > standard {standard['total_tokens']}"
        " — TOK-11 regression"
    )


def test_total_session_cost_counter_mode_disclosed():
    from bench.total_session_cost import run_total_session_cost

    out = run_total_session_cost(wake_depth="minimal")
    assert out["mode"] in (
        "anthropic-count-tokens",
        "tiktoken-cl100k-proxy",
        "heuristic-char4",
        "injected",
    )


def test_total_session_cost_fails_when_above_ref():
    from bench.total_session_cost import run_total_session_cost

    out = run_total_session_cost(wake_depth="standard", refs={"external": 1})
    assert out["passed"] is False
    assert out["refs"]["external"] == 1


def test_total_session_cost_passes_without_refs():
    from bench.total_session_cost import run_total_session_cost

    out = run_total_session_cost(wake_depth="minimal")
    assert out["passed"] is True
    assert out["refs"] == {}


def test_total_session_cost_main_exits_int():
    from bench import total_session_cost

    code = total_session_cost.main(argv=["--wake-depth", "minimal"])
    assert code in (0, 1)


def test_total_session_cost_injected_counter():
    from bench.total_session_cost import run_total_session_cost

    def _fixed(text: str) -> int:
        return max(1, len(text))

    out = run_total_session_cost(
        wake_depth="minimal", count_tokens_fn=_fixed,
    )
    assert out["mode"] == "injected"
    assert out["total_tokens"] >= 10
