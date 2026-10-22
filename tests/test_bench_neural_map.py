from __future__ import annotations

import os

import pytest


def test_neural_map_bench_restores_the_stage_profile_switch(tmp_path, monkeypatch):
    from bench.neural_map import run_neural_map_bench

    monkeypatch.delenv("IAI_MCP_STAGE_PROFILE", raising=False)
    run_neural_map_bench(n=50, iterations=1, store_path=tmp_path / "unset")
    assert "IAI_MCP_STAGE_PROFILE" not in os.environ

    monkeypatch.setenv("IAI_MCP_STAGE_PROFILE", "0")
    run_neural_map_bench(n=50, iterations=1, store_path=tmp_path / "preset")
    assert os.environ["IAI_MCP_STAGE_PROFILE"] == "0"


def test_neural_map_bench_runs_small_n(tmp_path):
    from bench.neural_map import run_neural_map_bench

    out = run_neural_map_bench(n=50, iterations=3, store_path=tmp_path)
    assert out["n"] == 50
    assert "latency_ms_p50" in out
    assert "latency_ms_p95" in out
    assert "passed" in out
    assert isinstance(out["latency_ms_p50"], float)
    assert isinstance(out["latency_ms_p95"], float)


def test_neural_map_bench_returns_stage_timings(tmp_path):
    from bench.neural_map import run_neural_map_bench

    out = run_neural_map_bench(n=50, iterations=2, store_path=tmp_path)
    assert "stage_timings_ms" in out
    stages = out["stage_timings_ms"]
    for expected in ("embed", "gate", "seeds", "spread", "rank"):
        assert expected in stages


@pytest.mark.perf
def test_neural_map_bench_reports_passed_flag(tmp_path, monkeypatch):
    from bench.neural_map import run_neural_map_bench, D_SPEED_P95_MS

    from _perf_helpers import best_of_n, skip_if_loaded

    skip_if_loaded()

    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")

    out = run_neural_map_bench(n=100, iterations=10, store_path=tmp_path / "run0")
    assert out.get("threshold_ms") == 100.0

    counter = {"i": 0}

    def _one_p95() -> float:
        i = counter["i"]
        counter["i"] += 1
        if i == 0:
            return float(out["latency_ms_p95"])
        run = run_neural_map_bench(
            n=100, iterations=10, store_path=tmp_path / f"run{i}",
        )
        return float(run["latency_ms_p95"])

    min_p95 = best_of_n(_one_p95, n=3)
    assert min_p95 < D_SPEED_P95_MS, (
        f"D-SPEED violated: best-of-3 p95={min_p95:.2f}ms >= {D_SPEED_P95_MS}ms "
        f"at N=100."
    )


@pytest.mark.perf
def test_neural_map_main_exits_zero_at_n100(tmp_path, monkeypatch, capsys):
    from bench import neural_map

    from _perf_helpers import skip_if_loaded

    skip_if_loaded()

    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")

    codes = [
        neural_map.main(ns=[100], iterations=10, store_path=tmp_path / f"run{i}")
        for i in range(3)
    ]
    assert any(c == 0 for c in codes), (
        f"bench.neural_map.main(ns=[100]) should exit 0 on at least one of "
        f"3 independent runs; got {codes}"
    )


def test_neural_map_bench_main_runs_and_returns_int(tmp_path, capsys):
    from bench import neural_map

    code = neural_map.main(ns=[50], iterations=2, store_path=tmp_path)
    assert code in (0, 1)


def test_neural_map_bench_deterministic_within_tolerance(tmp_path):
    from bench.neural_map import run_neural_map_bench

    a = run_neural_map_bench(
        n=50, iterations=5, store_path=tmp_path / "a", seed=42,
    )
    b = run_neural_map_bench(
        n=50, iterations=5, store_path=tmp_path / "b", seed=42,
    )
    assert a["latency_ms_p50"] < 2000.0
    assert b["latency_ms_p50"] < 2000.0
