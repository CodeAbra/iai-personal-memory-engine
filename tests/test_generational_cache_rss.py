"""Steady-state memory plateau with the records_cache warm, at corpus scale.

records_cache caches ``SimpleRecordView`` per candidate node (holding the
decrypted ``literal_surface``) resident on the graph for its lifetime. This
measures the marginal memory cost of that residency rather than assuming it:
``str()`` on an already-materialized str returns the same object in CPython,
so the marginal cost is the SimpleRecordView + dict-container overhead, not a
plaintext duplication.

Today the production recall path (``core/__init__.py``'s "memory_recall"
handler) hand-builds a fresh candidate ``MemoryGraph()`` every call, so this
residency window is bench/graph-scoped: the cache lives and dies within one
call-local graph and never accumulates across real requests. It becomes a
persistent-graph concern once a future milestone reuses the graph across
calls in production, where this bound must be re-measured against real
request volume and revisited.

Opt-in (``@pytest.mark.slow``, ``--runslow``): builds a 50k-record synthetic
corpus and drives two warm recall loops (cache-off baseline, then cache-on)
to a memory plateau -- several minutes total. macOS ``phys_footprint`` only
(the same charged-memory metric ``test_consolidation_no_leak.py`` uses, not
resident-set size, which counts reusable pages the allocator has freed but
not returned).
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_store import _make  # noqa: E402

from tests.rss_rca.phys_footprint_harness import _phys_footprint_bytes  # noqa: E402

import iai_mcp.pipeline as _pipeline_mod  # noqa: E402
from iai_mcp.pipeline import recall_for_response  # noqa: E402
from iai_mcp.retrieve import build_runtime_graph  # noqa: E402
from iai_mcp.store import MemoryStore, flush_edge_buffer, flush_record_buffer  # noqa: E402
from iai_mcp.types import EMBED_DIM  # noqa: E402

N_RECORDS = 50_000
_SEED = 20260825
_PLATEAU_WINDOW = 5
_PLATEAU_TOLERANCE_BYTES = 6 * 1024 * 1024  # 6 MiB band across the trailing window = settled
_MAX_SAMPLES = 40

# Calibration (derived from one clean run of this exact configuration on
# macOS, 50k synthetic records, --runslow; see the module docstring for the
# measurement shape -- never a hardcoded guess):
#   cache-off steady-state phys_footprint: 1,959,168,640 bytes (~1868.4 MiB)
#   cache-on  steady-state phys_footprint: 1,961,560,704 bytes (~1870.7 MiB)
#   observed delta (cache-on - cache-off):     2,392,064 bytes (~2.28 MiB)
# The delta is small: str() on an already-materialized str returns the same
# object in CPython, so the cache's marginal cost is the SimpleRecordView +
# dict-container overhead per candidate node, not plaintext duplication, as
# the module docstring predicts. _RSS_HEADROOM_BUDGET_BYTES is set to
# roughly 2x the observed delta, comfortably above per-run noise while
# still catching a genuine multi-MiB residency regression (same protocol as
# test_consolidation_no_leak.py's _LEAK_BUDGET_BYTES).
_RSS_HEADROOM_BUDGET_BYTES = 5 * 1024 * 1024  # ~5 MiB, ~2x the observed 2.28 MiB delta


def _unit_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _populate(
    store: MemoryStore, n: int, seed: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # tests/conftest.py's autouse insert wrapper flushes the record/edge/event
    # buffers after EVERY insert -- fine at unit-test scale, prohibitively
    # slow at 50k sequential inserts (each flush pays a per-record ANN/edge
    # commit). Opt out for the bulk-load loop, then flush once at the end,
    # matching bench/neural_map.py's own insert-loop shape.
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")
    rng = np.random.default_rng(seed)
    for i in range(n):
        v = rng.standard_normal(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        store.insert(_make(text=f"rss plateau fixture record {i}", vec=v.tolist()))
    try:
        flush_record_buffer(store)
        flush_edge_buffer(store)
    except Exception:
        pass


def _plateau(sample_fn) -> int:
    """Call ``sample_fn`` repeatedly until the trailing window of samples
    lands within ``_PLATEAU_TOLERANCE_BYTES``; return the last sample."""
    samples: list[int] = []
    for _ in range(_MAX_SAMPLES):
        gc.collect()
        samples.append(sample_fn())
        if len(samples) >= _PLATEAU_WINDOW:
            window = samples[-_PLATEAU_WINDOW:]
            if max(window) - min(window) <= _PLATEAU_TOLERANCE_BYTES:
                return window[-1]
    return samples[-1]


@pytest.mark.slow
@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="phys_footprint is macOS-only (proc_pid_rusage)",
)
@pytest.mark.timeout(2400)
def test_records_cache_steady_state_rss_plateau_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryStore(path=str(tmp_path / "store"))
    _populate(store, n=N_RECORDS, seed=_SEED, monkeypatch=monkeypatch)
    graph, assignment, rich_club = build_runtime_graph(store)

    # _age_penalty reads wall-clock time; irrelevant to a memory measurement
    # but frozen anyway for run-to-run reproducibility of which candidates
    # rank into the served set.
    monkeypatch.setattr(_pipeline_mod, "_age_penalty", lambda created_at: 0.0)

    cue_vecs = [_unit_vec(4000 + i) for i in range(5)]

    def _run_once(cue_vec: list[float]) -> None:
        recall_for_response(
            store=store,
            graph=graph,
            assignment=assignment,
            rich_club=rich_club,
            embedder=None,
            cue="rss plateau probe",
            session_id="rss-plateau-test",
            cue_embedding=list(cue_vec),
        )

    # Cache-OFF baseline plateau FIRST: the cache must never have been
    # populated before this measurement, or a leftover cached view would
    # inflate the "baseline" and understate the cache's real marginal cost
    # (the kill-switch bypasses the cache read/write path entirely -- it does
    # not clear an already-populated cache).
    monkeypatch.setenv("IAI_MCP_GENERATIONAL_CACHE_OFF", "1")

    def _sample_off() -> int:
        for cv in cue_vecs:
            _run_once(cv)
        v = _phys_footprint_bytes()
        assert v is not None, "phys_footprint unavailable on this host"
        return v

    off_plateau = _plateau(_sample_off)

    # Cache-ON plateau SECOND, on the SAME graph (no mutation happened in
    # between) so the generational cache actually engages and stays warm.
    monkeypatch.delenv("IAI_MCP_GENERATIONAL_CACHE_OFF", raising=False)

    def _sample_on() -> int:
        for cv in cue_vecs:
            _run_once(cv)
        v = _phys_footprint_bytes()
        assert v is not None, "phys_footprint unavailable on this host"
        return v

    on_plateau = _plateau(_sample_on)

    bound = off_plateau + _RSS_HEADROOM_BUDGET_BYTES
    assert on_plateau <= bound, (
        f"cache-warm steady-state phys_footprint {on_plateau} exceeds the "
        f"bound {bound} (= cache-off baseline {off_plateau} + committed "
        f"headroom {_RSS_HEADROOM_BUDGET_BYTES}) at N={N_RECORDS}"
    )
