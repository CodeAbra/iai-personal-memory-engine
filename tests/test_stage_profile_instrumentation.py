from __future__ import annotations

import sys
import time as _time_mod
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make

import iai_mcp.pipeline as _pipeline_mod
from iai_mcp.embed import Embedder
from iai_mcp.pipeline import recall_for_response
from iai_mcp.retrieve import build_runtime_graph
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM

N_RECORDS = 30
EXPECTED_STAGES = {
    "embed", "pool", "pool_collection", "gate", "centrality", "seeds",
    "spread", "degree", "reachable_count", "rank", "hit_assembly",
    "scored_count",
}
# Zero-incremental-perf_counter() stages: pool_collection and centrality
# each reuse a pre-existing UNCONDITIONAL timer (already runs regardless of
# IAI_MCP_STAGE_PROFILE for the recall_timing telemetry sample), so turning
# the flag on adds only a dict write, no new perf_counter() call.
# reachable_count and scored_count record an already-computed size/count,
# never a timer at all.
_STAGE_TIMER_CALL_PAIRS = len(
    EXPECTED_STAGES
    - {"pool_collection", "centrality", "reachable_count", "scored_count"}
) * 2

CUE = "what did we discuss about stage profiling"


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _random_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _build_store(tmp_path: Path, n: int) -> MemoryStore:
    store = MemoryStore(str(tmp_path / "store"))
    for i in range(n):
        rec = _make(text=f"Stage profile fixture record {i}", vec=_random_vec(4_000 + i))
        store.insert(rec)
    return store


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_stage_profile_on_reports_real_non_fabricated_timings(tmp_path, monkeypatch, driver):
    _select_driver(driver, monkeypatch)
    monkeypatch.setenv("IAI_MCP_STAGE_PROFILE", "1")

    store = _build_store(tmp_path, N_RECORDS)
    graph, assignment, rich_club = build_runtime_graph(store)
    embedder = Embedder()

    _pipeline_mod._last_stage_timings_ms.clear()
    recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=CUE, session_id="stage-profile-test",
        budget_tokens=1500, mode="concept",
    )

    timings = _pipeline_mod._last_stage_timings_ms
    assert set(timings.keys()) == EXPECTED_STAGES, timings
    for stage in EXPECTED_STAGES:
        assert timings[stage] >= 0.0, (stage, timings)

    total = timings["seeds"] + timings["spread"] + timings["rank"]
    if total > 0.0:
        ratios = (
            timings["seeds"] / total,
            timings["spread"] / total,
            timings["rank"] / total,
        )
        # The bug this guards against: bench/neural_map.py used to fabricate
        # seeds/spread/rank as an exact 0.2/0.3/0.5 split of a leftover
        # remainder rather than measuring them. Real per-stage timings must
        # not reproduce that exact ratio.
        assert ratios != pytest.approx((0.2, 0.3, 0.5), abs=1e-6)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_stage_profile_off_adds_zero_timer_calls_and_zero_dict_writes(tmp_path, monkeypatch, driver):
    _select_driver(driver, monkeypatch)
    # Deterministic call-count comparison: the recall_timing telemetry sample
    # (default 10%) advances global random state and would otherwise make
    # the off-vs-on perf_counter() call delta non-reproducible across the
    # two calls below.
    monkeypatch.setenv("IAI_MCP_RECALL_SAMPLE_RATE", "0")

    store = _build_store(tmp_path, N_RECORDS)
    graph, assignment, rich_club = build_runtime_graph(store)
    embedder = Embedder()
    real_perf_counter = _time_mod.perf_counter

    def _run(stage_profile_on: bool) -> int:
        if stage_profile_on:
            monkeypatch.setenv("IAI_MCP_STAGE_PROFILE", "1")
        else:
            monkeypatch.delenv("IAI_MCP_STAGE_PROFILE", raising=False)
        _pipeline_mod._last_stage_timings_ms.clear()
        call_count = {"n": 0}

        def _counting_perf_counter():
            call_count["n"] += 1
            return real_perf_counter()

        monkeypatch.setattr(_pipeline_mod.time, "perf_counter", _counting_perf_counter)
        try:
            recall_for_response(
                store=store, graph=graph, assignment=assignment, rich_club=rich_club,
                embedder=embedder, cue=CUE, session_id="stage-profile-test",
                budget_tokens=1500, mode="concept",
            )
        finally:
            monkeypatch.setattr(_pipeline_mod.time, "perf_counter", real_perf_counter)
        return call_count["n"]

    # Warm-up call (untimed, flag off): recall_for_response has its own
    # one-shot caching (centrality resolution, pool normalization) that
    # would otherwise change the perf_counter() call count between the
    # first and second call regardless of the stage-profile flag. Warming
    # the cache first makes the two measured calls below comparable.
    _run(stage_profile_on=False)

    off_count = _run(stage_profile_on=False)
    assert _pipeline_mod._last_stage_timings_ms == {}

    on_count = _run(stage_profile_on=True)
    assert set(_pipeline_mod._last_stage_timings_ms.keys()) == EXPECTED_STAGES

    # Flag off must add exactly zero of the stage-timer perf_counter() calls
    # (each of the six stages calls perf_counter() twice, start and end);
    # every perf_counter() call the off-path performs belongs to pre-existing,
    # unconditional instrumentation elsewhere in the recall path.
    assert on_count - off_count == _STAGE_TIMER_CALL_PAIRS, (off_count, on_count)
