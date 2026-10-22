"""Daemon boot-sequence RSS bound.

`test_boot_sequence_delta_flat_across_scale` in this module: the
consolidated boot sequence -- `runtime_graph_cache._rebuild_and_save_rgc`
(DB->child streaming) then `retrieve.build_runtime_graph` (in-thread
reconstruct), exactly what `daemon._boot_preload_body` runs post-fix -- must
NOT reproduce the pre-fix monotonic RSS climb a historical local
reproduction measured at production scale: peak 785.9/1155.0/1654.1 MB at
N=3000/8000/15000, i.e. a boot-sequence delta over the after-warm-embedder
baseline of +274 MB @3k -> +425 MB @8k (slope ~30.9 KB/record). This is a
BOUNDED proxy at two small Ns (each measured in its own fresh capped
subprocess, see below) so it completes in well under the full N=3k/8k/15k
sweep's wall time; the full sweep is a separate, documented, manual-only run.

The metric is PHASE-SCOPED: peak_total_rss during the harness's
"during-boot-sequence" phase window minus the after-warm-embedder-equivalent
baseline, matching `tests/rss_rca/phys_footprint_harness.py`'s
`run_boot_sequence_arm` + `_PeakSampler.peak_for_phase`. The pre-fix numbers
above were themselves each run's GLOBAL peak, but in each of those historical
runs the global peak occurred DURING the graph-build phase (confirmed by
that reproduction's own per-phase measurement) -- i.e. they were already
effectively phase-scoped by observation. Using an explicit phase-scoped peak
here makes the comparison robust even if a LATER phase (sleep-cycle, drain --
both skipped by this bounded proxy, see below) would otherwise dominate.

Each N is measured in its OWN capped subprocess, via
`phys_footprint_harness._run_capped_worker(..., boot_sequence_only=True)` --
the SAME subprocess path the full N=3k/8k/15k sweep uses. This is not merely
literal-compliance with the harness's own hermetic-subprocess design: an
EARLIER in-process version of this test (running two `_rebuild_and_save_rgc`
lifecycles back-to-back in the SAME pytest process) showed 2-3x run-to-run
variance in the measured delta (allocator/fragmentation state carrying over
between lifecycles in one process) -- exactly the noise a fresh-process
measurement, matched to the real sweep's own methodology, eliminates.
`--boot-sequence-only` skips the sleep-cycle and drain (irrelevant to this
phase-scoped metric, see `run_boot_sequence_arm`'s own docstring) so each
subprocess still completes in seconds, not minutes, at these small Ns.

The assertion is on the PER-RECORD SLOPE between the two Ns, not on raw
inter-N growth -- and the tolerance is derived from the FIX'S OWN MECHANISM,
not fitted to an observed number. The pre-fix path materialized the corpus
graph payload up to THREE times (in-parent MemoryGraph, a separate uncompacted
`node_payload_for_cache` at ~15 KB/node, and the community-detection child's
own full copy), yielding a measured pre-fix slope of ~30.9 KB/record
((425-274) MB / 5000 records, 3k->8k). A prior pair of fixes removed the
extra child-spawn-on-reconstruct materialization and the redundant
`node_payload_for_cache` copy, leaving ONE irreducible copy: the DB->child
worker's own graph working set, which community detection + betweenness
centrality fundamentally require. This module's subprocess-isolated
measurement lands at roughly half the pre-fix slope, consistent with
"meaningfully reduced, not eliminated." The assertion below encodes a 2/3
cutoff: generous enough to absorb run-to-run noise, but still tight enough
that a regression reinstating EITHER removed copy (which would roughly
double or triple the residual) fails it. This is NOT a claim of literal
flatness/zero-slope -- the durable measurement record documents the honest
non-zero-slope finding.

`test_first_contact_bounded`: the first post-boot RPC's own graph fetch
(`retrieve.build_runtime_graph`, which every recall caller must invoke to
obtain the `graph`/`assignment`/`rich_club` triple `recall_for_response`
requires) must hit `store._warm_graph_bundle` -- populated during boot -- and
must NOT spawn a detection/centrality child NOR re-materialize the corpus.
This is the direct guard against a cost that only appears to move off boot
by relocating to the first live client rather than being bounded: a test
that measured only the boot window would pass even if the cost moved to
first contact.
"""
from __future__ import annotations

import multiprocessing
from pathlib import Path

import numpy as np
import pytest

from iai_mcp import retrieve, runtime_graph_cache
from iai_mcp.pipeline import recall_for_response
from iai_mcp.store import MemoryStore, flush_edge_buffer, flush_record_buffer
from tests.rss_rca.corpus import build_corpus
from tests.rss_rca.phys_footprint_harness import (
    _BYTES_PER_MIB,
    _PeakSampler,
    _build_synthetic_edges,
    _run_capped_worker,
    run_boot_sequence_arm,
    sample_aggregate,
)

# Pre-fix numbers, verbatim from a historical local reproduction of the
# boot-time OOM (macOS, hermetic tmp store, edge_factor=1.3): "N=3000 ->
# baseline (after-warm-embedder) 511.8 MB total RSS, PEAK during
# build_runtime_graph 785.9 MB ... growth of +274 MB"; "N=8000 -> baseline
# 729.6 MB, PEAK during build_runtime_graph 1155.0 MB ... growth of +425 MB".
_PREFIX_DELTA_MB_AT_3K = 274.0
_PREFIX_DELTA_MB_AT_8K = 425.0
_PREFIX_SLOPE_KB_PER_RECORD = (
    (_PREFIX_DELTA_MB_AT_8K - _PREFIX_DELTA_MB_AT_3K) * 1024.0 / (8000 - 3000)
)
# Mechanism-derived tolerance, NOT fitted to this run's own numbers: the
# pre-fix slope came from up to THREE full-corpus-payload copies (in-parent
# graph, the redundant node_payload_for_cache, the detection child's own
# copy). A prior pair of fixes removes two of those three, leaving ONE
# irreducible copy (the DB->child worker's own graph working set for
# community detection + betweenness centrality). A regression that
# reinstated EITHER removed copy would roughly double or triple the
# residual slope back toward the pre-fix multi-copy figure -- comfortably
# caught even at this generous 2/3 cutoff, while a genuinely single-copy
# design should land well under it (this module's own bounded-proxy
# measurement: ~50-55% of the pre-fix slope, comfortable margin below 2/3).
_TOLERANCE_FRACTION = 2.0 / 3.0

_EDGE_FACTOR = 1.3


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p)
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None)
    )
    yield fake


class _FixedVecEmbedder:
    """Returns the SAME cue vector for every cue text -- no real model load."""

    def __init__(self, vec: list[float]) -> None:
        self.DIM = len(vec)
        self._vec = list(vec)

    def embed(self, text: str) -> list[float]:
        return list(self._vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vec) for _ in texts]


def _install_child_spawn_guards(monkeypatch: "pytest.MonkeyPatch") -> dict:
    """Count the three child-spawn axes the boot-sequence contract governs.

    ``rgc_worker_spawn`` counts EVERY `multiprocessing.Process` construction
    reachable through `_rebuild_and_save_rgc`'s own spawn-context worker (it
    must fire exactly once per `_rebuild_and_save_rgc` call); ``detect`` and
    ``centrality`` count the SEPARATE community-detection / betweenness-
    centrality children that the FOLLOWING reconstruct
    (`retrieve.build_runtime_graph`) must NOT spawn (a cache-hit reconstruct
    is child-free by construction: no detection/centrality child spawn on
    that path).
    """
    counts = {"detect": 0, "centrality": 0, "rgc_worker_spawn": 0}

    real_detect = retrieve._detect_communities_isolated
    real_centrality_child = runtime_graph_cache.compute_centrality_in_child
    real_get_context = multiprocessing.get_context

    def _counting_detect(store_arg, graph, **kw):
        counts["detect"] += 1
        return real_detect(store_arg, graph, **kw)

    def _counting_centrality_child(graph, **kw):
        counts["centrality"] += 1
        return real_centrality_child(graph, **kw)

    class _CountingCtx:
        """Wraps a real multiprocessing context, counting `.Process(...)`.

        Delegates everything else unchanged -- `_rebuild_and_save_rgc` only
        ever calls `.Pipe()` and `.Process()` on the context it gets back
        from `multiprocessing.get_context("spawn")`.
        """

        def __init__(self, inner: object) -> None:
            self._inner = inner

        def Process(self, *args: object, **kwargs: object) -> object:
            counts["rgc_worker_spawn"] += 1
            return self._inner.Process(*args, **kwargs)  # type: ignore[attr-defined]

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

    def _counting_get_context(method: str | None = None) -> object:
        return _CountingCtx(real_get_context(method))

    monkeypatch.setattr(retrieve, "_detect_communities_isolated", _counting_detect)
    monkeypatch.setattr(
        runtime_graph_cache, "compute_centrality_in_child", _counting_centrality_child
    )
    monkeypatch.setattr(multiprocessing, "get_context", _counting_get_context)
    return counts


def _seed_corpus_with_edges(store: MemoryStore, n: int, seed: int) -> int:
    """Insert ``n`` synthetic records (no HD payload -- irrelevant to graph
    memory scaling and slow to build) plus ~edge_factor*n bounded-cluster
    edges, via the SAME corpus/edge builders the phys_footprint harness uses
    at production scale, mirrored here at bounded size."""
    inserted = build_corpus(
        store, n=n, seed=seed, dim=store.embed_dim,
        varied_tags=True, with_hv_payload=False,
    )
    flush_record_buffer(store)
    _build_synthetic_edges(store, n, _EDGE_FACTOR, seed)
    flush_edge_buffer(store)
    return inserted


def _measure_boot_sequence_delta(
    store: MemoryStore, *, guard_counts: dict | None = None,
) -> dict:
    """Run the boot-sequence arm on ``store`` and return the phase-scoped
    delta plus timing. If ``guard_counts`` is given (from
    `_install_child_spawn_guards`), it must already be installed via
    monkeypatch before calling this."""
    baseline = sample_aggregate("baseline-equivalent")
    sampler = _PeakSampler(interval=0.05)
    sampler.start()
    try:
        rgc_result = run_boot_sequence_arm(store, sampler)
    finally:
        sampler.stop()
    phase_peak = sampler.peak_for_phase("during-boot-sequence")
    assert phase_peak is not None, (
        "the peak sampler recorded no sample during the boot-sequence phase"
        " -- the phase window was too short for the sampler's interval, or"
        " set_phase was never called"
    )
    delta_bytes = phase_peak["total_rss"] - baseline["total_rss"]
    return {
        "rgc_result": rgc_result,
        "baseline_total_rss": baseline["total_rss"],
        "peak_total_rss": phase_peak["total_rss"],
        "delta_bytes": delta_bytes,
    }


def _store_at(root: Path, name: str) -> MemoryStore:
    s = MemoryStore(path=root / name / "store")
    s.root = root / name
    return s


@pytest.mark.slow
def test_boot_sequence_child_spawn_contract(tmp_path: Path):
    """The boot-sequence arm spawns the DB->child rebuild worker EXACTLY
    ONCE, and the following reconstruct spawns NO second detection/
    centrality child. Small-N, in-process, monkeypatch-counted -- a pure
    call-count contract, not a numeric RSS comparison, so it needs none of
    the subprocess isolation the flatness proxy below requires."""
    store_ = _store_at(tmp_path, "spawn-contract")
    _seed_corpus_with_edges(store_, 20, seed=90001)
    with pytest.MonkeyPatch.context() as m:
        guard = _install_child_spawn_guards(m)
        result = _measure_boot_sequence_delta(store_, guard_counts=guard)
    assert result["rgc_result"].get("rebuilt") is True, (
        f"boot rebuild did not complete: {result['rgc_result']}"
    )
    assert guard["rgc_worker_spawn"] == 1, (
        "_rebuild_and_save_rgc must spawn its DB->child worker exactly"
        f" once, spawned {guard['rgc_worker_spawn']} times"
    )
    assert guard["detect"] == 0 and guard["centrality"] == 0, (
        "the reconstruct after _rebuild_and_save_rgc spawned a second"
        f" detection/centrality child: {guard}"
    )


def _measure_boot_sequence_delta_subprocess(
    n: int, seed: int, out_dir: Path,
) -> dict:
    """Run one N's boot-sequence-only measurement in a FRESH capped
    subprocess (the same path `_run_capped_worker` gives the real N=3k/8k/
    15k sweep) -- isolates this N's allocator state from any other
    measurement in the same test session. Returns the worker's JSON result."""
    result = _run_capped_worker(
        n=n, seed=seed, edge_factor=_EDGE_FACTOR, deferred_events=0,
        cap_gib=2.5, sample_interval=0.5, sleep_budget_sec=1.0,
        out_dir=out_dir, boot_sequence_only=True,
    )
    assert result.get("ok"), f"N={n} boot-sequence-only worker failed: {result}"
    assert result.get("graph_err") is None, (
        f"N={n} boot-sequence arm raised: {result['graph_err']}"
    )
    assert result.get("boot_sequence_delta_bytes") is not None, (
        f"N={n} worker result missing boot_sequence_delta_bytes: {result}"
    )
    return result


def _mean_delta_mb(n: int, seed_base: int, out_dir: Path, repeats: int) -> float:
    """Mean boot-sequence delta (MiB) over ``repeats`` independent
    subprocess measurements at ``n``. A single point estimate at these small
    Ns proved too noisy run-to-run (subprocess/embedder/numba-JIT startup
    jitter is a sizeable fraction of the N-proportional signal at this
    scale) -- averaging over repeats is a real variance reduction, not a
    tolerance fit."""
    deltas_mb = []
    for i in range(repeats):
        result = _measure_boot_sequence_delta_subprocess(
            n, seed_base + i, out_dir,
        )
        deltas_mb.append(result["boot_sequence_delta_bytes"] / _BYTES_PER_MIB)
    return sum(deltas_mb) / len(deltas_mb)


@pytest.mark.slow
def test_boot_sequence_delta_flat_across_scale(tmp_path: Path):
    """The boot-sequence delta must not scale with corpus size the way the
    pre-fix path did. Bounded proxy at N=1000/5000 (a wider ΔN than an
    earlier N=500/2000 attempt -- see module docstring: a narrower ΔN made
    the computed slope too sensitive to per-run startup jitter), each N
    averaged over 2 independent fresh capped subprocesses. The full
    N=3k/8k/15k sweep is a separate documented manual-only measurement."""
    n1, n2 = 1000, 5000
    out_dir = tmp_path / "harness-out"
    out_dir.mkdir()

    delta1_mb = _mean_delta_mb(n1, 91001, out_dir, repeats=2)
    delta2_mb = _mean_delta_mb(n2, 92001, out_dir, repeats=2)

    # The discriminating quantity is the PER-RECORD slope between the two
    # Ns, not the raw inter-N delta growth (see module docstring for why).
    observed_slope_kb_per_record = (delta2_mb - delta1_mb) * 1024.0 / (n2 - n1)
    tolerance_kb_per_record = _PREFIX_SLOPE_KB_PER_RECORD * _TOLERANCE_FRACTION

    diagnostic = (
        f"N={n1} mean_delta={delta1_mb:.1f} MB; N={n2} mean_delta={delta2_mb:.1f} MB"
        f" (each mean of 2 independent subprocess runs); "
        f"observed_slope={observed_slope_kb_per_record:.2f} KB/record vs"
        f" tolerance={tolerance_kb_per_record:.2f} KB/record"
        f" ({_TOLERANCE_FRACTION:.2f}x the pre-fix slope of"
        f" {_PREFIX_SLOPE_KB_PER_RECORD:.2f} KB/record)."
        f" NOTE: this asserts the slope dropped to roughly the one-copy"
        f" mechanism's prediction, NOT literal zero-slope flatness -- the"
        f" durable measurement record documents the honest non-zero-slope"
        f" finding."
    )
    assert observed_slope_kb_per_record <= tolerance_kb_per_record, (
        f"boot-sequence delta still scales with corpus size close to the"
        f" pre-fix multi-copy slope: {diagnostic}"
    )
    print(f"\n[boot-sequence RSS slope proxy] {diagnostic}")


@pytest.mark.slow
def test_first_contact_bounded(tmp_path: Path):
    """The first post-boot RPC must hit the warm bundle, spawn no child,
    and not re-materialize the corpus. Spawns a real subprocess and builds
    a 300-record corpus -- heavier than a plain unit assertion, so it is
    gated behind --runslow like this file's other subprocess-spawning
    test."""
    n = 300
    store_ = _store_at(tmp_path, "first-contact")
    _seed_corpus_with_edges(store_, n, seed=92001)

    boot_delta = _measure_boot_sequence_delta(store_)
    assert boot_delta["rgc_result"].get("rebuilt") is True, (
        f"boot rebuild did not complete: {boot_delta['rgc_result']}"
    )
    assert getattr(store_, "_warm_graph_bundle", None) is not None, (
        "store._warm_graph_bundle must be populated after the boot sequence"
        " completes, before preload_ready would fire in the real daemon"
    )

    baseline = sample_aggregate("post-boot-baseline")
    with pytest.MonkeyPatch.context() as m:
        guard = _install_child_spawn_guards(m)

        # First post-boot RPC: every recall caller must first fetch
        # (graph, assignment, rich_club) via build_runtime_graph -- the exact
        # call that consults store._warm_graph_bundle internally -- before
        # calling recall_for_response with the result. This is the real
        # production call sequence, not a bypass of the warm-bundle lookup.
        graph, assignment, rich_club = retrieve.build_runtime_graph(store_)
        embedder = _FixedVecEmbedder(
            np.random.default_rng(1).normal(size=store_.embed_dim).tolist()
        )
        response = recall_for_response(
            store=store_, graph=graph, assignment=assignment, rich_club=rich_club,
            embedder=embedder, cue="first post-boot recall probe",
            session_id="first-contact", budget_tokens=1500,
        )
        assert response is not None

        after = sample_aggregate("post-first-contact")

    assert guard["rgc_worker_spawn"] == 0, (
        "the first post-boot RPC spawned the DB->child rebuild worker --"
        f" it should have hit the warm bundle instead: {guard}"
    )
    assert guard["detect"] == 0 and guard["centrality"] == 0, (
        f"the first post-boot RPC spawned a detection/centrality child: {guard}"
    )

    first_contact_delta_mb = (after["total_rss"] - baseline["total_rss"]) / _BYTES_PER_MIB
    boot_delta_mb = boot_delta["delta_bytes"] / _BYTES_PER_MIB
    # A second full-corpus materialization would cost roughly the SAME as
    # the original boot-sequence build (empirically ~150-220 MB at this N,
    # dominated by the spawn+numba-JIT cold start the boot-sequence itself
    # pays once); bounding first-contact to a third of that (with an
    # absolute floor so a near-zero/negative boot delta never produces a
    # pathological near-zero bound) leaves headroom over ordinary run-to-run
    # allocator noise while still clearly failing a true relocation, which
    # would land within a small factor of boot_delta_mb itself, not a third
    # of it.
    bound_mb = max(15.0, 0.35 * max(boot_delta_mb, 0.0))
    diagnostic = (
        f"first_contact_delta={first_contact_delta_mb:.1f} MB vs bound={bound_mb:.1f} MB"
        f" (boot_delta={boot_delta_mb:.1f} MB)"
    )
    assert first_contact_delta_mb <= bound_mb, (
        f"first post-boot RPC re-materialized a corpus-scale amount of RSS --"
        f" the cost relocated to first contact instead of being bounded: {diagnostic}"
    )
    print(f"\n[first-contact RSS bound] {diagnostic}")
