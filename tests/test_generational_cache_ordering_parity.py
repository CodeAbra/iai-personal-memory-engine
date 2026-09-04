"""Cache-ON vs cache-OFF must serve byte-identical ordering and scores.

A dedicated file, never folded into a perf test: a reordering regression
must never be able to hide behind a passing latency assertion. Both storage
drivers are exercised (a differential missing that plumbing silently
validates only stdlib).

The cache-ON arm reuses ONE graph across two calls and asserts a genuine
cache HIT on the second call (zero graph.get_payload reads, the only
retained Python-resident rank cache) before diffing -- otherwise both arms
would be MISSes and the comparison would prove nothing. graph.degrees()
runs fresh on every call by design (retired memoization, see
test_rank_cache_retirement.py) and is not part of the HIT contract this
file checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_store import _make  # noqa: E402

import iai_mcp.pipeline as _pipeline_mod  # noqa: E402
from iai_mcp.pipeline import recall_for_response  # noqa: E402
from iai_mcp.retrieve import build_runtime_graph  # noqa: E402
from iai_mcp.store import MemoryStore, flush_edge_buffer, flush_record_buffer  # noqa: E402
from iai_mcp.types import EMBED_DIM  # noqa: E402


def _select_driver(driver: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _unit_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _populate(
    store: MemoryStore, n: int, seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Batch-flush: per-insert commit is O(rows) index churn, not needed until
    # the corpus is queried. Opts out of conftest's autoflush-per-insert via
    # its documented env switch, then does the one flush the query below needs.
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")
    rng = np.random.default_rng(seed)
    for i in range(n):
        v = rng.standard_normal(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        store.insert(_make(text=f"ordering parity fixture record {i}", vec=v.tolist()))
    monkeypatch.delenv("IAI_MCP_TEST_NO_AUTOFLUSH", raising=False)
    try:
        flush_record_buffer(store)
        flush_edge_buffer(store)
    except Exception:
        pass


def _run(store, graph, assignment, rich_club, cue_vec):
    return recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=None,
        cue="ordering parity probe cue",
        session_id="ordering-parity-test",
        cue_embedding=list(cue_vec),
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
@pytest.mark.parametrize("n", [1000, 10000])
def test_cache_on_off_hits_anti_hits_scores_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, driver: str, n: int,
) -> None:
    _select_driver(driver, monkeypatch)
    # _age_penalty(created_at) reads datetime.now(timezone.utc) fresh on every
    # call -- two temporally-separated calls (however close) see a different
    # "now" and every candidate's recency term shifts by that same tiny wall-
    # clock delta, producing a uniform ~1e-9 score drift unrelated to the
    # cache. Frozen here so the differential isolates the cache mechanism,
    # not the pre-existing, cache-independent recency-decay tick.
    monkeypatch.setattr(_pipeline_mod, "_age_penalty", lambda created_at: 0.0)

    store = MemoryStore(path=str(tmp_path / f"store-{driver}-{n}"))
    _populate(store, n=n, seed=20260825, monkeypatch=monkeypatch)
    graph, assignment, rich_club = build_runtime_graph(store)
    cue_vec = _unit_vec(999)

    # Warm-up call on the cache-ON (default) path: populates the generational
    # cache. Its own result is discarded -- it is a guaranteed MISS.
    _run(store, graph, assignment, rich_club, cue_vec)

    original_get_payload = graph.get_payload
    payload_calls: list = []

    def _counting_get_payload(rid):
        payload_calls.append(rid)
        return original_get_payload(rid)

    graph.get_payload = _counting_get_payload
    try:
        resp_on = _run(store, graph, assignment, rich_club, cue_vec)
    finally:
        graph.get_payload = original_get_payload

    assert not payload_calls, (
        "cache-ON second call must HIT the records_cache (zero graph reads) -- "
        f"{len(payload_calls)} payload reads happened; without a genuine HIT "
        "this differential proves nothing"
    )

    # Cache-OFF arm: same graph, same store, no mutation in between -- the
    # kill-switch forces the exact pre-cache unconditional rebuild.
    monkeypatch.setenv("IAI_MCP_GENERATIONAL_CACHE_OFF", "1")
    resp_off = _run(store, graph, assignment, rich_club, cue_vec)

    on_hit_ids = [h.record_id for h in resp_on.hits]
    off_hit_ids = [h.record_id for h in resp_off.hits]
    assert on_hit_ids == off_hit_ids, (
        f"hit ordering diverges cache-ON vs cache-OFF at n={n} driver={driver}: "
        f"{on_hit_ids} != {off_hit_ids}"
    )

    on_anti_ids = [h.record_id for h in resp_on.anti_hits]
    off_anti_ids = [h.record_id for h in resp_off.anti_hits]
    assert on_anti_ids == off_anti_ids, (
        f"anti_hit ordering diverges cache-ON vs cache-OFF at n={n} driver={driver}: "
        f"{on_anti_ids} != {off_anti_ids}"
    )

    on_scores = [h.score for h in resp_on.hits]
    off_scores = [h.score for h in resp_off.hits]
    assert on_scores == off_scores, (
        f"hit scores are not byte-identical cache-ON vs cache-OFF at n={n} "
        f"driver={driver}: {on_scores} != {off_scores}"
    )

    on_anti_scores = [h.score for h in resp_on.anti_hits]
    off_anti_scores = [h.score for h in resp_off.anti_hits]
    assert on_anti_scores == off_anti_scores, (
        f"anti_hit scores are not byte-identical cache-ON vs cache-OFF at n={n} "
        f"driver={driver}: {on_anti_scores} != {off_anti_scores}"
    )
