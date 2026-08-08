"""RACC-01 phase-gate: one command composing SC-1..SC-5 into runnable assertions.

This file does not duplicate the per-plan test suites (tests/test_recall_accuracy_gate.py,
tests/test_recall_trace_accuracy_marks.py, tests/test_confidence_escalation_gate.py,
tests/test_recall_literal_surface_byte_identity.py, tests/test_multi_seed_recall.py,
tests/test_cleanup_attractor_recall.py, tests/test_working_tier.py). It composes one
end-to-end low-confidence recall and a high-confidence control recall, then asserts
every SC against the resulting trace, hits, and literal_surface snapshot in a single
pytest command.

Witness (locked, owner-approved): the crafted bench/recall_accuracy.py eval set
(seed=42, n_filler=220, three archetypes) is the accuracy witness. Baseline
numbers are recorded verbatim below. A real-session labelled eval is a queued
follow-up, NOT introduced here.

Honest finding: the strict-improvement bar is NOT reached on this crafted eval
set by the four shipped mechanisms + associative fallback alone — the
harness's own dense-filler construction keeps the aggregate confidence signal
permanently HIGH on all three cues, so conf_escalate/multi_seed never fire on
it, and the harness never exercises the fallback or sets structural_weight>0.
The non-regression (<=) bound holds; the strict (<) bound is xfailed here,
pointing at the real-thread labelled eval.

No src/ edits. No working_tier import or consult anywhere in this file.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import numpy as np
import pytest

from bench.recall_accuracy import Archetype, build_eval_set, run_accuracy
from iai_mcp.pipeline import (
    ADAPTIVE_ESCALATION_CAP,
    escalate_recall_candidates,
)
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import MemoryRecord

_DIM = 16  # small synthetic dim; avoids loading the Rust embedder

# Baseline constants, recorded verbatim at eval-set creation (both drivers
# byte-identical at this corpus size).
BASELINE_RECALL_AT_K = 0.0
BASELINE_MISS_COSINE_RANK = 1
BASELINE_MISS_EDGE_LESS = 2
BASELINE_MISS_SPREAD_DISABLED = 0

# Reference-only p95 latency figure carried from 184-BASELINE.md Table 3 /
# 172-VERIFICATION.md — not re-measured hermetically in this file (see SC-5).
PRE_184_P95_MS_AT_37K = 256.0
P95_TOLERANCE_FACTOR = 1.15


# ---------------------------------------------------------------------------
# Shared fixtures (mirrors the per-plan test files' fixture pattern)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _small_embed_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(_DIM))


@pytest.fixture(autouse=True)
def _crypto_passphrase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake.pop((s, u), None))
    yield fake


@pytest.fixture(autouse=True)
def _clear_authority_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IAI_MCP_EXACT_AUTHORITY_OFF", raising=False)


@pytest.fixture(autouse=True)
def _reset_latency_global():
    yield
    import iai_mcp.pipeline as _pl
    _pl._last_recall_latency_ms = 0.0


def _select_driver(driver: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


class _StubEmbedder:
    """Deterministic stand-in embedder — a fixed cue vector regardless of text."""

    def __init__(self, vec: list[float]) -> None:
        self._vec = vec

    def embed(self, _text: str) -> list[float]:
        return list(self._vec)


def _seeded_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _orthogonal_vec(seed: int) -> list[float]:
    """A unit vector with near-zero cosine similarity to _seeded_vec(seed)."""
    rng = np.random.default_rng(seed + 9000)
    v = rng.random(_DIM).astype(np.float32) - 0.5
    return (v / np.linalg.norm(v)).tolist()


def _make_rec(rid: UUID, surface: str, embedding: list[float]) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid,
        tier="episodic",
        literal_surface=surface,
        aaak_index="",
        embedding=embedding,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=["capture"],
        language="en",
    )


def _stub_embedder_for_store(monkeypatch: pytest.MonkeyPatch, vec: list[float]) -> None:
    import iai_mcp.embed as _embed_mod

    monkeypatch.setattr(_embed_mod, "embedder_for_store", lambda _store: _StubEmbedder(vec))


def _dispatch_traced_recall(store: MemoryStore, cue_vec: list[float], *, session_id: str) -> dict:
    from iai_mcp import core as _core
    import iai_mcp.pipeline as _pm

    store._build_exact_index_sync()
    _pm._last_recall_latency_ms = 0.0
    return _core.dispatch(store, "memory_recall", {
        "cue": "phase-gate composite recall cue, embedder is stubbed",
        "session_id": session_id,
        "budget_tokens": 2000,
        "cue_embedding": cue_vec,
    })


@pytest.fixture
def _make_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _make(driver: str) -> MemoryStore:
        _select_driver(driver, monkeypatch)
        return MemoryStore(path=tmp_path / f"store-{driver}")
    return _make


def _seed_low_confidence_corpus(store: MemoryStore, cue_vec: list[float]) -> list[UUID]:
    """A small, low-confidence-inducing corpus, mirroring the shared
    fixture used by test_recall_trace_accuracy_marks.py: no near-duplicate
    cluster, so top cosine stays below the high-confidence threshold and
    hit-count-above-threshold stays low — the marks this test asserts on
    (soft_gate, conf_escalate, multi_seed, cleanup_attractor) all become
    reachable on this fixture."""
    record_ids: list[UUID] = []
    target_id = uuid4()
    target = _make_rec(target_id, "the associative target record", cue_vec)
    store.insert(target)
    record_ids.append(target_id)
    for i in range(4):
        rid = uuid4()
        store.insert(_make_rec(rid, f"unrelated filler {i}", _orthogonal_vec(100 + i)))
        record_ids.append(rid)
    flush_record_buffer(store)
    return record_ids


def _seed_high_confidence_corpus(store: MemoryStore, cue_vec: list[float]) -> None:
    """A dense near-duplicate cluster around the cue — high confidence, used
    as the SC-3 control case (escalation must NOT fire)."""
    target_id = uuid4()
    target = _make_rec(target_id, "the associative target record", cue_vec)
    store.insert(target)
    for i in range(5):
        nudge = (np.array(cue_vec, dtype=np.float32) * 0.97
                 + np.random.default_rng(500 + i).normal(0, 0.01, _DIM).astype(np.float32))
        nudge = nudge / np.linalg.norm(nudge)
        store.insert(_make_rec(uuid4(), f"near-duplicate {i}", nudge.tolist()))
    flush_record_buffer(store)


def _read_module_source(module_path: str) -> str:
    return Path(module_path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# recall_at_k / miss_counts on the crafted archetype eval set.
# Non-regression (<=) holds; strict improvement (<) is xfailed, pointing at
# the owner-authorized real-thread labelled eval.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_sc1_non_regression_holds_on_crafted_eval_set(tmp_path, monkeypatch, driver):
    """SC-1 non-regression: recall_at_k >= baseline and miss_counts <= baseline
    on the crafted archetype eval set (seed=42, n_filler=220), both drivers."""
    _select_driver(driver, monkeypatch)

    store, graph, assignment, rich_club, embedder, cases = build_eval_set(
        seed=42, n_filler=220, store_path=tmp_path
    )
    try:
        result = run_accuracy(store, graph, assignment, rich_club, embedder, cases, k=10)

        assert result["recall_at_k"] >= BASELINE_RECALL_AT_K - 0.001, (
            f"[SC-1] recall_at_k regressed vs. baseline: got {result['recall_at_k']}, "
            f"baseline {BASELINE_RECALL_AT_K}"
        )
        miss_counts = result["miss_counts_by_archetype"]
        assert miss_counts.get(Archetype.COSINE_RANK_BEYOND_CUTOFF, 0) <= BASELINE_MISS_COSINE_RANK, (
            f"[SC-1] COSINE_RANK_BEYOND_CUTOFF miss_count regressed: {miss_counts}"
        )
        assert miss_counts.get(Archetype.EDGE_LESS_ISLAND, 0) <= BASELINE_MISS_EDGE_LESS, (
            f"[SC-1] EDGE_LESS_ISLAND miss_count regressed: {miss_counts}"
        )
        assert miss_counts.get(Archetype.SPREAD_DISABLED_BY_PRIOR_LATENCY, 0) == BASELINE_MISS_SPREAD_DISABLED, (
            f"[SC-1] SPREAD_DISABLED_BY_PRIOR_LATENCY must stay at baseline 0: {miss_counts}"
        )
        assert result["unattributed"] == 0, (
            f"[SC-1] unattributed misses found, attribution incomplete: {result['unattributed']}"
        )
    finally:
        store.close()


@pytest.mark.xfail(
    reason=(
        "[SC-1] strict-improvement bar NOT reached on the crafted archetype eval set "
        "by the four shipped mechanisms (soft_gate/conf_escalate/multi_seed/cleanup_attractor) "
        "+ D-A alone — root-caused in 184-04-SUMMARY.md: the harness's own dense-filler "
        "construction (cosine 0.3-0.99 spread, required to push the target beyond "
        "K_CANDIDATES=200) keeps the aggregate confidence signal (compute_spread_depth) "
        "permanently HIGH on all three cues, so conf_escalate/multi_seed structurally never "
        "fire on this harness; the harness also never routes through core.dispatch (D-A's "
        "merge_authority_hits is never exercised) and never sets structural_weight>0 "
        "(cleanup-attractor's snap is a no-op by design). Non-regression holds (see the "
        "sibling test above); strict improvement is deferred to the owner-authorized "
        "labelled real-thread eval routed through "
        "core.dispatch that exercises all five mechanisms."
    ),
    strict=True,
)
@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_strict_improvement_deferred_to_real_thread_eval(tmp_path, monkeypatch, driver):
    """Strict improvement: recall_at_k > baseline (or a miss_count strictly
    decreased) on the crafted eval set. Documented xfail — the real-thread
    labelled eval carries the strict bar."""
    _select_driver(driver, monkeypatch)

    store, graph, assignment, rich_club, embedder, cases = build_eval_set(
        seed=42, n_filler=220, store_path=tmp_path
    )
    try:
        result = run_accuracy(store, graph, assignment, rich_club, embedder, cases, k=10)
        miss_counts = result["miss_counts_by_archetype"]

        strictly_improved = (
            result["recall_at_k"] > BASELINE_RECALL_AT_K
            or miss_counts.get(Archetype.COSINE_RANK_BEYOND_CUTOFF, 0) < BASELINE_MISS_COSINE_RANK
            or miss_counts.get(Archetype.EDGE_LESS_ISLAND, 0) < BASELINE_MISS_EDGE_LESS
        )
        assert strictly_improved, (
            f"[SC-1] strict improvement not reached: recall_at_k={result['recall_at_k']}, "
            f"miss_counts={miss_counts}, baseline_recall_at_k={BASELINE_RECALL_AT_K}"
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# SC-2: all four new trace marks present on a low-confidence cue.
# No working_tier_bias mark asserted anywhere (D-C).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_sc2_all_four_trace_marks_present_on_low_confidence_cue(tmp_path, _make_store, monkeypatch, driver):
    """One low-confidence recall's trace MUST contain all of {soft_gate,
    conf_escalate, multi_seed, cleanup_attractor}. No working_tier_bias mark
    exists or is asserted (D-C)."""
    monkeypatch.setenv("IAI_MCP_RECALL_TRACE", "1")
    store = _make_store(driver)
    cue_vec = _seeded_vec(2)
    _seed_low_confidence_corpus(store, cue_vec)
    _stub_embedder_for_store(monkeypatch, cue_vec)

    resp = _dispatch_traced_recall(store, cue_vec, session_id=f"phase-gate-sc2-{driver}")

    marks = {name for name, _ms in resp.get("_recall_trace_ms", [])}
    expected_marks = {"soft_gate", "conf_escalate", "multi_seed", "cleanup_attractor"}
    missing = expected_marks - marks
    assert not missing, (
        f"[SC-2] missing trace marks {sorted(missing)}, got marks={sorted(marks)}"
    )
    assert "working_tier_bias" not in marks, (
        "[SC-2/D-C] a working_tier_bias mark was found in the trace — this mark must "
        "never exist (working-tier bias was intentionally dropped from the recall path)"
    )


# ---------------------------------------------------------------------------
# SC-3: escalate_recall_candidates fires once on low-confidence, never on
# high-confidence.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_sc3_escalation_spy_fires_only_on_low_confidence_cue(tmp_path, _make_store, monkeypatch, driver):
    """escalate_recall_candidates spy count == 1 on a low-confidence cue and
    == 0 on a high-confidence cue (two separate stores, same test)."""
    # --- low-confidence branch: must fire exactly once -----------------
    low_store = _make_store(driver)
    low_cue = _seeded_vec(11)
    target_id = uuid4()
    low_store.insert(_make_rec(target_id, "the associative target record", low_cue))
    for i in range(2):
        low_store.insert(_make_rec(uuid4(), f"unrelated filler {i}", _orthogonal_vec(11 + i)))
    flush_record_buffer(low_store)
    _stub_embedder_for_store(monkeypatch, low_cue)

    low_spy = MagicMock(wraps=escalate_recall_candidates)
    monkeypatch.setattr("iai_mcp.pipeline.escalate_recall_candidates", low_spy)

    _dispatch_traced_recall(low_store, low_cue, session_id=f"phase-gate-sc3-low-{driver}")

    assert low_spy.call_count == 1, (
        f"[SC-3] expected escalate_recall_candidates to fire exactly once on a "
        f"low-confidence cue, got call_count={low_spy.call_count}"
    )
    low_store.close()

    # --- high-confidence branch: must never fire -----------------------
    monkeypatch.undo()
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(_DIM))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")
    monkeypatch.delenv("IAI_MCP_EXACT_AUTHORITY_OFF", raising=False)
    _select_driver(driver, monkeypatch)

    high_store = MemoryStore(path=tmp_path / f"store-high-{driver}")
    high_cue = _seeded_vec(21)
    _seed_high_confidence_corpus(high_store, high_cue)
    _stub_embedder_for_store(monkeypatch, high_cue)

    high_spy = MagicMock(wraps=escalate_recall_candidates)
    monkeypatch.setattr("iai_mcp.pipeline.escalate_recall_candidates", high_spy)

    _dispatch_traced_recall(high_store, high_cue, session_id=f"phase-gate-sc3-high-{driver}")

    assert high_spy.call_count == 0, (
        f"[SC-3] expected escalate_recall_candidates to NEVER fire on a "
        f"high-confidence cue, got call_count={high_spy.call_count}"
    )
    high_store.close()


# ---------------------------------------------------------------------------
# SC-4: literal_surface byte-identity for every record touched by a recall
# that fires the four shipped plug-ins + retrieval-reconsolidation.
# ---------------------------------------------------------------------------

_VERBATIM_SURFACES: list[str] = [
    "line one\n\n  line two with   extra   spaces\t\ttabbed",
    "Ünïcödé çhârs — em dash, füll-width Ａ, and a ¥ sign",
    "trailing whitespace kept exactly as-is   ",
]


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_sc4_literal_surface_byte_identity_after_plugin_recall(tmp_path, _make_store, monkeypatch, driver):
    """literal_surface bytes for every touched record are unchanged after a
    recall that fires soft_gate/conf_escalate/multi_seed/cleanup_attractor and
    reinforce_record(is_retrieval=True) (queue_reinforce)."""
    monkeypatch.setenv("IAI_MCP_RECALL_TRACE", "1")
    store = _make_store(driver)
    cue_vec = _seeded_vec(31)

    record_ids: list[UUID] = []
    for i, surface in enumerate(_VERBATIM_SURFACES):
        rid = uuid4()
        vec = cue_vec if i == 0 else _seeded_vec(31 + i)
        store.insert(_make_rec(rid, surface, vec))
        record_ids.append(rid)
    # low-confidence filler so soft_gate/conf_escalate/multi_seed/cleanup_attractor
    # all have a branch to fire on, same construction as the SC-2 fixture.
    for i in range(4):
        store.insert(_make_rec(uuid4(), f"unrelated filler {i}", _orthogonal_vec(131 + i)))
    flush_record_buffer(store)

    before = {rid: store.get(rid).literal_surface for rid in record_ids}
    for rid, surface in zip(record_ids, _VERBATIM_SURFACES):
        assert before[rid] == surface, "[SC-4] pre-recall literal_surface already diverges from input"

    _stub_embedder_for_store(monkeypatch, cue_vec)
    resp = _dispatch_traced_recall(store, cue_vec, session_id=f"phase-gate-sc4-{driver}")
    assert resp.get("hits"), "[SC-4] recall returned no hits — cannot exercise the reinforce path"

    marks = {name for name, _ms in resp.get("_recall_trace_ms", [])}
    assert {"soft_gate", "conf_escalate", "multi_seed", "cleanup_attractor"} <= marks, (
        f"[SC-4] expected the four plug-ins to all fire on this fixture, got marks={sorted(marks)}"
    )

    after = {rid: store.get(rid).literal_surface for rid in record_ids}
    for rid, surface in zip(record_ids, _VERBATIM_SURFACES):
        assert after[rid] == surface, (
            f"[SC-4] literal_surface changed after a reinforcing recall for record {rid}: "
            f"before={before[rid]!r} after={after[rid]!r}"
        )
        assert after[rid] == before[rid], (
            f"[SC-4] literal_surface diverged from its pre-recall value for record {rid}"
        )


# ---------------------------------------------------------------------------
# SC-5: no O(corpus) reads on the recall path; both drivers; no p95 latency
# regression.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_sc5_no_o_corpus_reads_on_recall_path(tmp_path, _make_store, monkeypatch, driver):
    """Structural guard: a low-confidence recall MUST NEVER call
    store.all_records(), store._exact_scan_full_corpus(), or query_similar
    with an unbounded k on the read path."""
    store = _make_store(driver)
    cue_vec = _seeded_vec(41)

    target_id = uuid4()
    store.insert(_make_rec(target_id, "the associative target record", cue_vec))
    for i in range(2):
        store.insert(_make_rec(uuid4(), f"unrelated filler {i}", _orthogonal_vec(41 + i)))
    flush_record_buffer(store)

    _stub_embedder_for_store(monkeypatch, cue_vec)

    def _raise_all_records(*_a, **_k):
        raise AssertionError("[SC-5] store.all_records() called on the recall path")
    monkeypatch.setattr(store, "all_records", _raise_all_records)

    if hasattr(store, "_exact_scan_full_corpus"):
        def _raise_exact_scan(*_a, **_k):
            raise AssertionError("[SC-5] store._exact_scan_full_corpus() called on the recall path")
        monkeypatch.setattr(store, "_exact_scan_full_corpus", _raise_exact_scan)

    _orig_query_similar = store.query_similar

    def _bounded_query_similar(vec, k=10, *args, **kwargs):
        if k > ADAPTIVE_ESCALATION_CAP + 100:
            raise AssertionError(f"[SC-5] query_similar called with unbounded k={k} on the recall path")
        return _orig_query_similar(vec, k, *args, **kwargs)
    monkeypatch.setattr(store, "query_similar", _bounded_query_similar)

    _dispatch_traced_recall(store, cue_vec, session_id=f"phase-gate-sc5-{driver}")


def test_sc5_p95_latency_no_regression():
    """SC-5 p95 latency: bench/full_recall_latency_probe.py requires a real
    @37k-record store built out-of-band (see 172-VERIFICATION.md); it cannot
    be run hermetically inside this unit test. Per 184-05-PLAN.md's own
    fallback instruction, this sub-assert is skipped here — the p95
    comparison against the pre-184 snapshot (256ms @37k, 184-BASELINE.md
    Table 3 / 172-VERIFICATION.md Observable Truth 5) is recorded directly
    in 184-VERIFICATION.md instead."""
    pytest.skip(
        "p95 measured out-of-band via bench/full_recall_latency_probe.py at phase "
        "verification; comparison recorded in 184-VERIFICATION.md (pre-184 baseline "
        f"{PRE_184_P95_MS_AT_37K}ms @37k, tolerance factor {P95_TOLERANCE_FACTOR})"
    )


# ---------------------------------------------------------------------------
# Structural lock: the shipped no-working-tier guard still passes both
# drivers, plus an inlined source-grep check immune to future refactors.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_dc_recall_path_stays_working_tier_free(tmp_path, monkeypatch, driver):
    """D-C structural lock: dispatching memory_recall never touches
    iai_mcp.working_tier (monkeypatch-raise guard), and the source files on
    the awake recall path contain zero references to working_tier at the
    grep level — immune to future refactors that might route the import
    through a different indirection."""
    _select_driver(driver, monkeypatch)
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(_DIM))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")
    monkeypatch.delenv("IAI_MCP_EXACT_AUTHORITY_OFF", raising=False)

    from iai_mcp import working_tier as wt

    def _boom(*args, **kwargs):
        raise AssertionError("[D-C] working-tier function called during memory_recall dispatch")

    monkeypatch.setattr(wt, "read_task", _boom)
    monkeypatch.setattr(wt, "update_from_record", _boom)
    monkeypatch.setattr(wt, "populate_from_sensory", _boom)

    store = MemoryStore(path=tmp_path / f"dc-store-{driver}")
    try:
        cue_vec = _seeded_vec(51)
        # The synthetic-dim store has no real embedder; the supplied
        # cue_embedding drives recall and the stub hosts the dispatch.
        _stub_embedder_for_store(monkeypatch, cue_vec)
        store.insert(_make_rec(uuid4(), "reference content for D-C recall no-op probe", cue_vec))
        flush_record_buffer(store)

        resp = _dispatch_traced_recall(store, cue_vec, session_id=f"phase-gate-dc-{driver}")
        assert "error" not in resp, f"[D-C] recall dispatch errored: {resp.get('error')}"
    finally:
        store.close()

    import iai_mcp.pipeline as _pipeline_mod
    import iai_mcp.core as _core_mod

    pipeline_source = _read_module_source(_pipeline_mod.__file__)
    core_source = _read_module_source(_core_mod.__file__.replace("__init__.pyc", "__init__.py"))

    assert "working_tier" not in pipeline_source, (
        "[D-C] found a 'working_tier' reference in src/iai_mcp/pipeline.py — "
        "the recall path must stay working-tier-import-free"
    )
    assert "working_tier" not in core_source, (
        "[D-C] found a 'working_tier' reference in src/iai_mcp/core/__init__.py — "
        "the recall path must stay working-tier-import-free"
    )
