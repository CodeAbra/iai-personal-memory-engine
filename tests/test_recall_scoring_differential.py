"""Recall-ranking differential harness.

Compares two ranked-result producers end-to-end on production-shaped,
3-band, real-embedded cues and asserts byte-identical (tie-tolerant) top-k
on both storage drivers. The producer is a callable parameter --
`(cue, mode, cue_intent, k) -> list[tuple[record_id, score]]` -- so a future
Rust candidate producer registers here with zero edits to the comparison
loop itself.

Today's Python recall path is registered as BOTH the reference and the
candidate producer, proving the harness is deterministic and driver-agnostic
before any Rust candidate exists. Non-vacuity is proven by two committed
positive controls (a perturbed-score producer and a perturbed-membership
producer), both routed through the exact same comparison loop the green
path-vs-itself test uses.
"""
from __future__ import annotations

import builtins
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from uuid import UUID, uuid4

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_recall_stage_profile import _monkeypatch_env  # noqa: E402

from tests._synthetic_cue_corpus import (  # noqa: E402
    CueSpec,
    apply_term_discrimination_edges,
    build_bucket_b_evidence_fixture,
    build_corpus_records,
    build_cue_set,
    build_synthetic_corpus_and_cues,
    build_term_discrimination_fixture,
    flatten_cues,
    insert_corpus,
    warm_lexical_index_for_fixture,
)
from tests.test_exact_authority_index import (  # noqa: E402
    _TOP_K_TIE_TOL,
    _assert_top_k_tie_tolerant,
)

import iai_mcp.pipeline as _pm  # noqa: E402
from iai_mcp.cue_router import _classify_cue  # noqa: E402
from iai_mcp.embed import Embedder  # noqa: E402
from iai_mcp.pipeline import recall_for_response  # noqa: E402
from iai_mcp.store import MemoryStore  # noqa: E402
from iai_mcp.types import MemoryRecord  # noqa: E402

_TOP_K = 10
_SEED = 0

Producer = Callable[[str, str, "str | None", int], "list[tuple[str, float]]"]


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake.pop((s, u), None))
    yield fake


def _select_driver(driver: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built -- lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _freeze_age_penalty(monkeypatch: pytest.MonkeyPatch, at: "datetime | None" = None) -> datetime:
    """Enumerated recall-path clock reads that can affect a SCORE: only
    `_age_penalty` (T4) reads `datetime.now()` into a score; `temporal_cue`'s
    `parse_date_mentions`/`matches_mentions` (T16) never call `now()` at all;
    the NULL-created_at clock fallbacks (`_from_row`, `_payload_created_at`)
    are structurally unreachable because every corpus record carries an
    explicit, non-null `created_at`. Freezing this one function is therefore
    a complete freeze of the recall path's clock surface, not a partial one."""
    frozen_now = at or datetime.now(timezone.utc)

    def _frozen(created_at: datetime) -> float:
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        days = (frozen_now - created_at).total_seconds() / 86400.0
        if days < 0:
            return 0.0
        return min(1.0, days / _pm.AGE_HALF_LIFE_DAYS)

    monkeypatch.setattr(_pm, "_age_penalty", _frozen)
    return frozen_now


def _build_driver_store(
    driver: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> "tuple[MemoryStore, object, object, list, Embedder]":
    _select_driver(driver, monkeypatch)
    _monkeypatch_env(monkeypatch, tmp_path)
    store_root = tmp_path / f"differential-{driver}"
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    embedder = Embedder()
    records = build_corpus_records(seed=_SEED, embedder=embedder)
    store = MemoryStore(path=store_root)
    insert_corpus(store, records)

    from iai_mcp.retrieve import build_runtime_graph
    graph, assignment, rich_club = build_runtime_graph(store)
    return store, graph, assignment, rich_club, embedder


def _make_python_producer(
    store, graph, assignment, rich_club, embedder: Embedder, *, profile_state: "dict | None" = None,
) -> Producer:
    def _producer(cue: str, mode: str, cue_intent: "str | None", k: int) -> "list[tuple[str, float]]":
        del cue_intent  # recall_for_response derives it internally from cue text
        _pm._last_recall_latency_ms = 0.0
        response = recall_for_response(
            store=store, graph=graph, assignment=assignment, rich_club=rich_club,
            embedder=embedder, cue=cue, session_id="recall-scoring-differential",
            budget_tokens=1500, mode=mode, profile_state=profile_state,
        )
        return [(str(h.record_id), h.score) for h in response.hits[:k]]

    return _producer


def _make_rust_producer(
    store, graph, assignment, rich_club, embedder: Embedder, *, profile_state: "dict | None" = None,
) -> Producer:
    """The candidate producer for the cutover gate: identical call shape to
    `_make_python_producer`, differing only in `use_rust_scorer=True` -- the
    seam `_resolve_use_rust_scorer` reads. A caller that forgot this one
    keyword would silently compare Python against itself; every test that
    uses this producer also asserts the Rust scorer actually fired (see
    `_wrap_rust_entry_counter`), so a resolution failure is caught, not
    silently green."""
    def _producer(cue: str, mode: str, cue_intent: "str | None", k: int) -> "list[tuple[str, float]]":
        del cue_intent
        _pm._last_recall_latency_ms = 0.0
        response = recall_for_response(
            store=store, graph=graph, assignment=assignment, rich_club=rich_club,
            embedder=embedder, cue=cue, session_id="recall-scoring-differential-rust",
            budget_tokens=1500, mode=mode, profile_state=profile_state,
            use_rust_scorer=True,
        )
        return [(str(h.record_id), h.score) for h in response.hits[:k]]

    return _producer


def _wrap_call_counter(monkeypatch: pytest.MonkeyPatch, obj, attr: str) -> list:
    """Installs a counting wrapper (`wraps=` semantics, without pulling in
    `MagicMock` call-count bookkeeping for a hot path) around `obj.attr`,
    returning the list its length grows against. Used to prove control flow
    actually entered a code path at runtime -- "the resolution returned
    True" and "the branch actually ran to completion" are different claims,
    and only the counter proves the second one."""
    original = getattr(obj, attr)
    calls: list = []

    def _wrapped(*a, **k):
        calls.append(1)
        return original(*a, **k)

    monkeypatch.setattr(obj, attr, _wrapped)
    return calls


def _run_differential(cues: "list[CueSpec]", reference: Producer, candidate: Producer) -> None:
    """The one comparison loop every test in this file drives -- the green
    path-vs-itself test AND both RED positive controls call this exact
    function, so a control that fails to route through it would not be
    exercising the harness body at all."""
    for idx, cue in enumerate(cues):
        _, cue_intent, _ = _classify_cue(cue.text)
        expected = reference(cue.text, cue.mode, cue_intent, _TOP_K)
        got = candidate(cue.text, cue.mode, cue_intent, _TOP_K)
        _assert_top_k_tie_tolerant(expected, got, k=_TOP_K, cue_seed=idx)


def _make_score_offset_producer(reference: Producer) -> Producer:
    """Perturbation option 1 (research-blessed): offset exactly one score by
    10x the tie tolerance. Caught by the comparator's per-position score
    check."""
    def _producer(cue: str, mode: str, cue_intent: "str | None", k: int) -> "list[tuple[str, float]]":
        results = reference(cue, mode, cue_intent, k)
        if not results:
            return results
        rid, score = results[0]
        return [(rid, score + 10 * _TOP_K_TIE_TOL)] + list(results[1:])

    return _producer


def _make_membership_drop_producer(reference: Producer) -> Producer:
    """Perturbation option 2: drop one fused term's effect on MEMBERSHIP
    rather than score -- the last slot keeps its expected score (so the
    per-position score check alone would pass) but its id is replaced by one
    absent from the reference's own top-k, caught only by the top-k id-set
    assertion. Proves the harness catches a membership regression, not only
    a score regression."""
    def _producer(cue: str, mode: str, cue_intent: "str | None", k: int) -> "list[tuple[str, float]]":
        results = reference(cue, mode, cue_intent, k)
        if len(results) < 2:
            return results
        last_id, last_score = results[-1]
        fake_id = f"perturbed-nonexistent-{last_id}"
        return list(results[:-1]) + [(fake_id, last_score)]

    return _producer


# ---------------------------------------------------------------------------
# Green: today's Python path vs itself, both drivers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_python_path_vs_itself_byte_identical_top_k(tmp_path, monkeypatch, driver):
    _freeze_age_penalty(monkeypatch)
    store, graph, assignment, rich_club, embedder = _build_driver_store(driver, tmp_path, monkeypatch)
    cues_by_band = build_cue_set(seed=_SEED)
    cues = flatten_cues(cues_by_band)

    reference_producer = _make_python_producer(store, graph, assignment, rich_club, embedder)
    candidate_producer = reference_producer  # identical seam: a future candidate swaps only this line

    _run_differential(cues, reference_producer, candidate_producer)


# ---------------------------------------------------------------------------
# Cutover gate: the Rust-scored (default) path vs the kill-switch-selected
# Python Bucket-A reference.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_rust_path_vs_python_byte_identical_top_k(tmp_path, monkeypatch, driver):
    _freeze_age_penalty(monkeypatch)
    store, graph, assignment, rich_club, embedder = _build_driver_store(driver, tmp_path, monkeypatch)
    cues_by_band = build_cue_set(seed=_SEED)
    cues = flatten_cues(cues_by_band)

    reference_producer = _make_python_producer(store, graph, assignment, rich_club, embedder)
    # Reference must be CALLABLE -- if the kill-switch's Python Bucket-A path
    # was deleted upstream, this gate is unreachable and must say so loudly,
    # not report a false green from an all-Rust comparison.
    try:
        reference_producer(cues[0].text, cues[0].mode, None, _TOP_K)
    except Exception as exc:  # noqa: BLE001 -- surfaced as a hard failure below
        pytest.fail(
            f"reference producer (kill-switch Python Bucket-A path) is not "
            f"callable -- the differential gate is unreachable: {exc}"
        )

    candidate_producer = _make_rust_producer(store, graph, assignment, rich_club, embedder)
    rust_entries = _wrap_call_counter(monkeypatch, _pm, "_t11_t12_flags")

    _run_differential(cues, reference_producer, candidate_producer)

    assert len(rust_entries) == len(cues), (
        f"Rust scorer entry count {len(rust_entries)} != cue count {len(cues)} -- "
        "the candidate producer silently fell back to the Python reference for "
        "at least one cue, which would make this a Python-vs-Python comparison"
    )


# ---------------------------------------------------------------------------
# Non-vacuity: a genuine Rust scoring regression must turn this gate RED, not
# only a Python one. `IAI_MCP_RANK_PERTURB_W_COSINE` (rust/iai_mcp_rank_core/
# src/lib.rs) overrides the cosine weight for the CANDIDATE call only -- the
# reference call runs with the env var absent, so this is a real Rust-vs-
# Python divergence through the exact same comparator every other test in
# this file uses, not a hand-rolled score-offset producer.
# ---------------------------------------------------------------------------

def test_rust_perturbation_goes_red(tmp_path, monkeypatch):
    _freeze_age_penalty(monkeypatch)
    store, graph, assignment, rich_club, embedder = _build_driver_store("stdlib", tmp_path, monkeypatch)
    cues = flatten_cues(build_cue_set(seed=_SEED))

    reference_producer = _make_python_producer(store, graph, assignment, rich_club, embedder)
    rust_producer = _make_rust_producer(store, graph, assignment, rich_club, embedder)

    def _perturbed_candidate(cue: str, mode: str, cue_intent: "str | None", k: int) -> "list[tuple[str, float]]":
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("IAI_MCP_RANK_PERTURB_W_COSINE", "0.1")
            return rust_producer(cue, mode, cue_intent, k)

    with pytest.raises(AssertionError):
        _run_differential(cues, reference_producer, _perturbed_candidate)

    # Negative arm: with the env var unset, the same candidate must go back
    # to GREEN in the same process -- proving the override (not some other
    # side effect of the rebuild) is what flips the gate.
    assert "IAI_MCP_RANK_PERTURB_W_COSINE" not in os.environ
    _run_differential(cues, reference_producer, rust_producer)


# ---------------------------------------------------------------------------
# Margin-truncation: a Bucket-B adjustment promoting a candidate from just
# outside a bare top-k into the served top-k must not be dropped by the
# k+k_margin window the Rust candidate scores over
# (RUST_SCORER_K_MARGIN=32). 263-TERM-FREEZE.md names this obligation as
# outstanding for a future candidate producer -- this closes it.
# ---------------------------------------------------------------------------

def test_bucket_b_margin_promotion_not_dropped(tmp_path, monkeypatch):
    """Shrinks `k` itself (`_POST_RANK_MAX_HITS`) rather than growing the
    pool: the margin obligation is a statement about the k+k_margin window,
    not about corpus composition. The unmodified 10-record bucket-b fixture
    already has a committed rank flip (`test_bucket_b_terms_measurably_
    change_score_and_rank`'s T14 assertion) -- SE ranks BELOW EP unboosted,
    ABOVE EP boosted -- so a k=1 window turns that exact flip into an
    outside-bare-k-into-served-top-k promotion with zero synthetic fixture
    engineering. `RUST_SCORER_K_MARGIN` stays untouched (its real production
    value) -- only `k` shrinks, so the promoted row must genuinely survive
    the Rust scan window, not just a relaxed test-only margin."""
    _freeze_age_penalty(monkeypatch)
    store, graph, assignment, rich_club, embedder, fixture = _build_bucket_b_store(
        tmp_path, monkeypatch,
    )
    cue = fixture.probe_cue["T14_tier_boost"]
    se = str(fixture.ids["t14_semantic"])
    monkeypatch.setattr(_pm, "_POST_RANK_MAX_HITS", 1)
    _k = 1

    def recall(*, use_rust: bool) -> "list[tuple[str, float]]":
        _pm._last_recall_latency_ms = 0.0
        response = recall_for_response(
            store=store, graph=graph, assignment=assignment, rich_club=rich_club,
            embedder=embedder, cue=cue, session_id="margin-truncation",
            budget_tokens=100_000, mode="concept", use_rust_scorer=use_rust,
        )
        return [(str(h.record_id), h.score) for h in response.hits]

    # Precondition: with the tier boost OFF, SE must rank OUTSIDE bare k=1
    # on both drivers -- otherwise the k+k_margin window is never exercised.
    monkeypatch.setenv("IAI_MCP_TIER_BOOST", "1.0")
    baseline_python = {rid for rid, _s in recall(use_rust=False)}
    baseline_rust = {rid for rid, _s in recall(use_rust=True)}
    assert se not in baseline_python, (
        f"precondition failed: {se} already ranks inside bare k={_k} "
        f"({baseline_python}) without the tier boost -- the margin window "
        "this test targets is never exercised"
    )
    assert se not in baseline_rust, (
        f"precondition failed: {se} already ranks inside bare k={_k} "
        f"({baseline_rust}) without the tier boost"
    )

    # With the (default) tier boost active, SE must be the SOLE served hit
    # on BOTH sides -- a Rust candidate that silently truncated the
    # promoted row before the boost ever applied would show up here as a
    # membership divergence, not a score one (Bucket-B applies to winners
    # only, after the Rust scoring call returns).
    monkeypatch.delenv("IAI_MCP_TIER_BOOST", raising=False)
    boosted_python = [rid for rid, _s in recall(use_rust=False)[:_k]]
    boosted_rust = [rid for rid, _s in recall(use_rust=True)[:_k]]
    assert se in boosted_python, (
        f"precondition failed: tier boost did not promote {se} into the "
        f"python k={_k} at all -- {boosted_python}"
    )
    assert se in boosted_rust, (
        f"Bucket-B margin promotion dropped: {se} is in the python k={_k} "
        f"({boosted_python}) but absent from the rust k={_k} "
        f"({boosted_rust}) -- the k+k_margin window silently "
        "truncated a promotable candidate"
    )


# ---------------------------------------------------------------------------
# Non-vacuity: committed RED positive controls, routed through the same loop
# ---------------------------------------------------------------------------

def test_perturbed_score_producer_makes_comparator_red(tmp_path, monkeypatch):
    _freeze_age_penalty(monkeypatch)
    store, graph, assignment, rich_club, embedder = _build_driver_store("stdlib", tmp_path, monkeypatch)
    cues_by_band = build_cue_set(seed=_SEED)
    cues = flatten_cues(cues_by_band)

    reference_producer = _make_python_producer(store, graph, assignment, rich_club, embedder)
    perturbed_producer = _make_score_offset_producer(reference_producer)

    with pytest.raises(AssertionError):
        _run_differential(cues, reference_producer, perturbed_producer)


def test_perturbed_membership_producer_makes_comparator_red(tmp_path, monkeypatch):
    _freeze_age_penalty(monkeypatch)
    store, graph, assignment, rich_club, embedder = _build_driver_store("stdlib", tmp_path, monkeypatch)
    cues_by_band = build_cue_set(seed=_SEED)
    cues = flatten_cues(cues_by_band)

    reference_producer = _make_python_producer(store, graph, assignment, rich_club, embedder)
    perturbed_producer = _make_membership_drop_producer(reference_producer)

    with pytest.raises(AssertionError):
        _run_differential(cues, reference_producer, perturbed_producer)


# ---------------------------------------------------------------------------
# Bucket-A per-term discrimination: every term the Rust port reimplements
# must be individually able to fail this harness. "Every term appears" is
# NOT the bar -- a term with zero variance across the corpus is invisible to
# the comparator even when correctly wired (a flat additive term shifts
# every candidate's score by the same constant, never reordering anything).
# Each of the 12 Bucket-A terms in the frozen fused-score term table gets
# its own probe pair on a small, purpose-built fixture
# (build_term_discrimination_fixture) and its own targeted corruption, then
# is driven through _run_differential exactly like every other control in
# this file.
#
# T5 (spread_contrib) and T7 (structural_similarity) are OFF by default in
# production (W_SPREAD_ACT=0.0; structural_weight defaults to 0.0 via
# profile_state) -- their reference producer explicitly enables the real,
# existing kill-switch/knob for the probe call. This is not a corpus-
# flatness fix: no amount of corpus structure makes either term's
# contribution nonzero while its knob is off. The Rust port still must
# implement the formula correctly for when the knob is on, so the control
# stays meaningful -- but it is a materially weaker guarantee than a term
# that fires under default settings, and is labelled as such below.
#
# T10 (valence) is wired now: MemoryRecord, RankCandidateView and
# SimpleRecordView all carry a `valence` field, decoded and clamped from the
# stored column and carried into the graph-pool payload. This fixture's
# records never write a non-default value, so `getattr(rec, "valence",
# None)` still resolves to the default 0.0 for every probe pair here -- the
# control's getattr shadow forces a real value (0.6) the fixture does not
# otherwise write, proving the FORMULA (`s *= 1 + valence`) is rank-changing
# once a non-default value reaches it.
# ---------------------------------------------------------------------------

def _build_term_discrimination_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _select_driver("stdlib", monkeypatch)
    _monkeypatch_env(monkeypatch, tmp_path)
    store_root = tmp_path / "term-discrimination"
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    embedder = Embedder()
    fixture = build_term_discrimination_fixture(seed=_SEED, embedder=embedder)
    store = MemoryStore(path=store_root)
    insert_corpus(store, fixture.records)
    apply_term_discrimination_edges(store, fixture)

    from iai_mcp.retrieve import build_runtime_graph
    graph, assignment, rich_club = build_runtime_graph(store)
    # build_runtime_graph bumps the corpus-count-cache generation (its own
    # cache materialization is itself a corpus-changing write from the
    # count cache's point of view) -- warming the lexical index BEFORE this
    # call stamps a generation that is stale by the time a query reads it.
    warm_lexical_index_for_fixture(store, fixture)
    return store, graph, assignment, rich_club, embedder, fixture


def _shadowed_getattr_forcing(name: str, forced_value, only_for_id: "object | None" = None):
    """Module-level `getattr` shadow (installed as `iai_mcp.pipeline.getattr`
    -- Python resolves a bare `getattr(...)` call inside pipeline.py against
    that module's own globals before falling through to builtins, so this
    intercepts ONLY bare getattr calls textually inside pipeline.py, never
    other modules). Forces `name` to `forced_value` -- for every record if
    `only_for_id` is None, otherwise only when the record's `.id` matches --
    and passes every other read through to the real builtin."""
    def _getattr(obj, attr, default=None):
        if attr == name and (only_for_id is None or getattr(obj, "id", None) == only_for_id):
            return forced_value
        return builtins.getattr(obj, attr, default)

    return _getattr


def _run_single_term_control(
    term: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    reference_kwargs_fn, candidate_patches_fn,
) -> None:
    """Shared shape for every Bucket-A per-term control: build the
    discrimination store, build a reference producer (optionally under a
    knob `reference_kwargs_fn` installs on `monkeypatch` -- the OUTER,
    test-scoped monkeypatch, active for the whole test), then a candidate
    producer whose corruption is installed via `pytest.MonkeyPatch.context()`
    scoped to exactly one call (`candidate_patches_fn`) -- so the reference
    call is NEVER perturbed. Asserts the comparator goes RED for this term's
    probe cue."""
    _freeze_age_penalty(monkeypatch)
    store, graph, assignment, rich_club, embedder, fixture = _build_term_discrimination_store(
        tmp_path, monkeypatch,
    )
    reference_kwargs_fn(monkeypatch, store, graph, assignment, rich_club, fixture)
    reference_producer = _make_python_producer(
        store, graph, assignment, rich_club, embedder,
        profile_state=getattr(reference_kwargs_fn, "profile_state", None),
    )

    def _candidate(cue: str, mode: str, cue_intent: "str | None", k: int) -> "list[tuple[str, float]]":
        with pytest.MonkeyPatch.context() as mp:
            candidate_patches_fn(mp, store, graph, assignment, rich_club, fixture)
            del cue_intent
            _pm._last_recall_latency_ms = 0.0
            response = recall_for_response(
                store=store, graph=graph, assignment=assignment, rich_club=rich_club,
                embedder=embedder, cue=cue, session_id="term-discrimination-candidate",
                budget_tokens=1500, mode=mode,
                profile_state=getattr(reference_kwargs_fn, "profile_state", None),
            )
            return [(str(h.record_id), h.score) for h in response.hits[:k]]

    probe_cue = fixture.probe_cue[term]
    with pytest.raises(AssertionError):
        _run_differential(
            [CueSpec(text=probe_cue, band="probe", mode="concept")],
            reference_producer, _candidate,
        )
    # Sanity: the reference call itself must be UNPERTURBED -- if it were
    # leaking the candidate's monkeypatch context, both sides would cancel
    # and pytest.raises above would never have fired in the first place, but
    # this also guards against a leak that happens to still change the
    # OUTPUT in a way that still raises for the wrong reason.
    _, cue_intent, _ = _classify_cue(probe_cue)
    reference_again = reference_producer(probe_cue, "concept", cue_intent, _TOP_K)
    reference_once_more = reference_producer(probe_cue, "concept", cue_intent, _TOP_K)
    assert reference_again == reference_once_more, (
        f"{term}: reference producer is non-deterministic across repeated "
        "calls after a perturbed candidate call ran -- the candidate's "
        "monkeypatch context leaked past its own call"
    )


def _no_extra_reference_setup(monkeypatch, store, graph, assignment, rich_club, fixture) -> None:
    pass


def _run_single_term_control_rust(
    term: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    reference_kwargs_fn, candidate_patches_fn,
) -> None:
    """Rust-routed sibling of `_run_single_term_control`: the CANDIDATE call
    goes through the Rust scorer (`use_rust_scorer=True`) instead of the
    pre-Rust Python-vs-itself methodology, proving the perturbation reaches
    the shipped FFI path the cutover gate actually compares."""
    _freeze_age_penalty(monkeypatch)
    store, graph, assignment, rich_club, embedder, fixture = _build_term_discrimination_store(
        tmp_path, monkeypatch,
    )
    reference_kwargs_fn(monkeypatch, store, graph, assignment, rich_club, fixture)
    reference_producer = _make_python_producer(
        store, graph, assignment, rich_club, embedder,
        profile_state=getattr(reference_kwargs_fn, "profile_state", None),
    )

    def _candidate(cue: str, mode: str, cue_intent: "str | None", k: int) -> "list[tuple[str, float]]":
        with pytest.MonkeyPatch.context() as mp:
            candidate_patches_fn(mp, store, graph, assignment, rich_club, fixture)
            del cue_intent
            _pm._last_recall_latency_ms = 0.0
            response = recall_for_response(
                store=store, graph=graph, assignment=assignment, rich_club=rich_club,
                embedder=embedder, cue=cue, session_id="term-discrimination-rust-candidate",
                budget_tokens=1500, mode=mode,
                profile_state=getattr(reference_kwargs_fn, "profile_state", None),
                use_rust_scorer=True,
            )
            return [(str(h.record_id), h.score) for h in response.hits[:k]]

    probe_cue = fixture.probe_cue[term]
    with pytest.raises(AssertionError):
        _run_differential(
            [CueSpec(text=probe_cue, band="probe", mode="concept")],
            reference_producer, _candidate,
        )


def test_wrong_degree_term_goes_red(tmp_path, monkeypatch):
    """Both DEGREE failure modes must independently redden the Rust-routed
    cutover gate: an all-zero degree weight, and separately an unfiltered
    exclusion set that lets an excluded-edge-type hub (entity_shared-only
    edges) gain rank it must not. Each arm builds its own store under a
    distinct subdirectory -- the shared term-discrimination builder uses a
    fixed store path, and reusing it across two builds in one test would
    silently accrete a second fixture's records into the first's store."""
    arm1 = tmp_path / "degree-zero-weight"
    arm1.mkdir()

    def _corrupt_zero_weight(mp, store, graph, assignment, rich_club, fixture):
        mp.setenv("IAI_MCP_W_DEGREE", "0")

    _run_single_term_control_rust(
        "T3_degree", arm1, monkeypatch, _no_extra_reference_setup, _corrupt_zero_weight,
    )

    arm2 = tmp_path / "degree-unfiltered-exclusion"
    arm2.mkdir()

    def _corrupt_unfiltered_exclusion(mp, store, graph, assignment, rich_club, fixture):
        mp.setattr(graph, "RANKING_DEGREE_EXCLUDED", frozenset())

    _run_single_term_control_rust(
        "T3_degree_unfiltered_exclusion", arm2, monkeypatch,
        _no_extra_reference_setup, _corrupt_unfiltered_exclusion,
    )


def test_wrong_lexical_term_goes_red(tmp_path, monkeypatch):
    """Perturbs the FLAG path: `_t11_t12_flags` batches T11 through the
    Rust `trigram_t11_flags` helper for the Rust FFI boundary -- corrupting
    that helper here corrupts the flag Rust actually reads, never a
    resident-surface read (removed from the Rust index; the flag path is
    the only source T11/T12 have)."""
    def _corrupt_trigram_flag(mp, store, graph, assignment, rich_club, fixture):
        from iai_mcp_native import rank as _rank_native

        mp.setattr(
            _rank_native, "trigram_t11_flags",
            lambda cue_lower, surfaces_lower: [False] * len(surfaces_lower),
        )

    _run_single_term_control_rust(
        "T11_trigram", tmp_path, monkeypatch, _no_extra_reference_setup, _corrupt_trigram_flag,
    )


def test_t1_cosine_perturbation_red(tmp_path, monkeypatch):
    def _corrupt(mp, store, graph, assignment, rich_club, fixture):
        mp.setattr(_pm, "W_COSINE", 0.0)

    _run_single_term_control("T1_cosine", tmp_path, monkeypatch, _no_extra_reference_setup, _corrupt)


def test_t2_aaak_perturbation_red(tmp_path, monkeypatch):
    def _corrupt(mp, store, graph, assignment, rich_club, fixture):
        mp.setattr(_pm, "W_AAAK", 0.0)

    _run_single_term_control("T2_aaak", tmp_path, monkeypatch, _no_extra_reference_setup, _corrupt)


def test_t3_degree_perturbation_red(tmp_path, monkeypatch):
    def _corrupt(mp, store, graph, assignment, rich_club, fixture):
        mp.setenv("IAI_MCP_W_DEGREE", "0")

    _run_single_term_control("T3_degree", tmp_path, monkeypatch, _no_extra_reference_setup, _corrupt)


def test_t4_age_perturbation_red(tmp_path, monkeypatch):
    def _corrupt(mp, store, graph, assignment, rich_club, fixture):
        mp.setattr(_pm, "W_AGE", 0.0)

    _run_single_term_control("T4_age", tmp_path, monkeypatch, _no_extra_reference_setup, _corrupt)


def test_t5_spread_contrib_perturbation_red_requires_nondefault_knob(tmp_path, monkeypatch):
    """T5 is OFF by default in production (W_SPREAD_ACT=0.0, per the
    module's own "MUST stay 0.0 in prod" docstring). This control enables
    IAI_MCP_W_SPREAD_ACT on BOTH the reference and candidate call -- the
    reference is not the plain default path here -- and corrupts by making
    the candidate's graph traversal report no spread provenance at all."""
    def _enable_knob(monkeypatch, store, graph, assignment, rich_club, fixture):
        monkeypatch.setenv("IAI_MCP_W_SPREAD_ACT", "0.5")

    def _corrupt(mp, store, graph, assignment, rich_club, fixture):
        mp.setenv("IAI_MCP_W_SPREAD_ACT", "0.5")
        mp.setattr(graph, "two_hop_neighborhood_with_provenance", lambda *a, **k: {})

    _run_single_term_control("T5_spread_contrib", tmp_path, monkeypatch, _enable_knob, _corrupt)


def test_t6_community_contrib_perturbation_red(tmp_path, monkeypatch):
    def _corrupt(mp, store, graph, assignment, rich_club, fixture):
        mp.setenv("IAI_MCP_COMMUNITY_BIAS", "0")

    _run_single_term_control("T6_community_contrib", tmp_path, monkeypatch, _no_extra_reference_setup, _corrupt)


def test_t7_structural_similarity_perturbation_red_requires_nondefault_knob(tmp_path, monkeypatch):
    """T7 is unreachable for TWO independent reasons, not one: OFF by
    default in production (structural_weight defaults to 0.0 via
    profile_state, which recall_for_response is normally called without),
    AND `structure_hv` never reaches the graph-sourced candidate pool at
    all -- `build_runtime_graph`'s node payloads never carry a
    "structure_hv" key, and the SimpleRecordView construction that reads
    graph payloads never asks for one, so `rec.structure_hv` is `b""`
    (falsy) for every graph-pool candidate regardless of what is actually
    stored, which keeps the gate `... and rec.structure_hv` permanently
    closed on the hot path. This control passes
    profile_state={"structural_weight": 0.6} AND injects a real
    structure_hv onto ONE candidate via a SimpleRecordView.__init__ patch
    (both installed on the OUTER, test-scoped monkeypatch -- shared by
    reference and candidate, restoring what correct hydration would
    provide), then corrupts by forcing
    hebbian_structure.structural_similarity to always return 0.0."""
    def _enable_knob_and_inject_structure_hv(monkeypatch, store, graph, assignment, rich_club, fixture):
        from iai_mcp import tem

        hit_id = fixture.ids["structural_hit"]
        hit_rec = store.get(hit_id)
        injected_hv = tem.bind_structure(hit_rec)
        original_init = _pm.SimpleRecordView.__init__

        def _patched_init(self, *a, **kw):
            original_init(self, *a, **kw)
            if kw.get("id") == hit_id:
                self.structure_hv = injected_hv

        monkeypatch.setattr(_pm.SimpleRecordView, "__init__", _patched_init)
        monkeypatch.setattr(graph, "_records_view_cache", None, raising=False)

    _enable_knob_and_inject_structure_hv.profile_state = {"structural_weight": 0.6}

    def _corrupt(mp, store, graph, assignment, rich_club, fixture):
        mp.setattr("iai_mcp.hebbian_structure.structural_similarity", lambda *a, **k: 0.0)

    _run_single_term_control(
        "T7_structural_similarity", tmp_path, monkeypatch,
        _enable_knob_and_inject_structure_hv, _corrupt,
    )


def test_t9_stability_perturbation_red(tmp_path, monkeypatch):
    def _corrupt(mp, store, graph, assignment, rich_club, fixture):
        mp.setattr(_pm, "getattr", _shadowed_getattr_forcing("stability", 0.5), raising=False)

    _run_single_term_control("T9_stability", tmp_path, monkeypatch, _no_extra_reference_setup, _corrupt)


def test_t10_valence_perturbation_red_requires_dead_field_injection(tmp_path, monkeypatch):
    """valence is wired now (see module docstring above), but this fixture's
    records never write a non-default value, so a real perturbation still
    needs the field forced -- the reference producer injects a valence value
    via a getattr shadow installed on the OUTER (test-scoped) monkeypatch --
    a test-only mechanism, not a production knob. The candidate call runs
    inside its own nested `pytest.MonkeyPatch.context()`, which does NOT
    automatically undo the outer shadow (they are independent monkeypatch
    instances) -- the candidate must EXPLICITLY restore the real builtin
    getattr for the duration of its own call, or it silently inherits the
    same injection and the control goes vacuously green."""
    def _inject(monkeypatch, store, graph, assignment, rich_club, fixture):
        monkeypatch.setattr(
            _pm, "getattr",
            _shadowed_getattr_forcing("valence", 0.6, only_for_id=fixture.ids["valence_a"]),
            raising=False,
        )

    def _corrupt(mp, store, graph, assignment, rich_club, fixture):
        mp.setattr(_pm, "getattr", builtins.getattr, raising=False)

    _run_single_term_control("T10_valence", tmp_path, monkeypatch, _inject, _corrupt)


def test_t11_trigram_perturbation_red(tmp_path, monkeypatch):
    def _corrupt(mp, store, graph, assignment, rich_club, fixture):
        mp.setattr(_pm, "_trigram_jaccard", lambda a, b: 0.0)

    _run_single_term_control("T11_trigram", tmp_path, monkeypatch, _no_extra_reference_setup, _corrupt)


def test_t12_fts_hits_perturbation_red(tmp_path, monkeypatch):
    """Corrupts by editing ONLY the hydrated `literal_surface` for the
    fts-probe record's graph payload -- never the store -- so cosine/
    trigram stay untouched; _trigram_jaccard is neutralized on both sides so
    T11 cannot mask or explain the T12-only delta."""
    def _neutralize_trigram(monkeypatch, store, graph, assignment, rich_club, fixture):
        monkeypatch.setattr(_pm, "_trigram_jaccard", lambda a, b: 0.0)

    def _corrupt(mp, store, graph, assignment, rich_club, fixture):
        mp.setattr(_pm, "_trigram_jaccard", lambda a, b: 0.0)
        target_id = fixture.ids["fts_hit"]
        original_get_payload = graph.get_payload

        def _patched(rid):
            node = original_get_payload(rid)
            if rid == target_id and node is not None:
                node = dict(node)
                node["surface"] = str(node.get("surface", "")).replace("quokka77bridge", "redacted")
            return node

        mp.setattr(graph, "get_payload", _patched)
        # Force a rebuild of the cached records view -- otherwise the
        # reference call's already-cached hydration would be served as-is
        # and this patch would never run.
        mp.setattr(graph, "_records_view_cache", None, raising=False)

    _run_single_term_control("T12_fts_hits", tmp_path, monkeypatch, _neutralize_trigram, _corrupt)


def test_t13_lex_rank_perturbation_red(tmp_path, monkeypatch):
    """The fixture's ~36-record corpus is too small for the probe token's
    IDF to clear the shipped LEX_FUSION_MIN_IDF=4.0 floor by default
    (measured max_idf ~3.2 here) -- IAI_MCP_LEX_MIN_IDF is a real, existing
    env override (not a monkeypatch hack) lowered for this probe on BOTH
    the reference and candidate call, a corpus-size artifact rather than a
    term-flatness one."""
    def _lower_min_idf(monkeypatch, store, graph, assignment, rich_club, fixture):
        monkeypatch.setenv("IAI_MCP_LEX_MIN_IDF", "2.0")

    def _corrupt(mp, store, graph, assignment, rich_club, fixture):
        mp.setenv("IAI_MCP_LEX_MIN_IDF", "2.0")
        mp.setattr(store, "lexical_query_warm", lambda *a, **k: [])

    _run_single_term_control("T13_lex_rank", tmp_path, monkeypatch, _lower_min_idf, _corrupt)


# ---------------------------------------------------------------------------
# now frozen: verified by construction, both a positive and a negative arm
# ---------------------------------------------------------------------------

def test_frozen_now_byte_identical_across_clock_tick_straddle(tmp_path, monkeypatch):
    store, graph, assignment, rich_club, embedder = _build_driver_store("stdlib", tmp_path, monkeypatch)
    producer = _make_python_producer(store, graph, assignment, rich_club, embedder)
    cue = build_cue_set(seed=_SEED)["specific"][0]

    # Negative arm first: prove the freeze is load-bearing on this corpus --
    # a real 90-day jump between two calls must NOT be byte-identical. A
    # straddle test that passes whether or not the freeze holds is vacuous.
    t0 = _freeze_age_penalty(monkeypatch, at=datetime.now(timezone.utc))
    result_t0 = producer(cue.text, cue.mode, None, _TOP_K)
    _freeze_age_penalty(monkeypatch, at=t0 + timedelta(days=90))
    result_t0_plus_90d = producer(cue.text, cue.mode, None, _TOP_K)
    assert result_t0 != result_t0_plus_90d, (
        "age penalty had zero effect on this cue's scored output across a "
        "90-day jump -- the freeze requirement would be untestable (inert) "
        "on this corpus/cue combination"
    )

    # Positive arm: freeze held at the SAME instant across a real wall-clock
    # sleep straddling a full second boundary -- not a claim, a real check.
    frozen_now = _freeze_age_penalty(monkeypatch, at=datetime.now(timezone.utc))
    result_a = producer(cue.text, cue.mode, None, _TOP_K)
    time.sleep(1.3)
    _freeze_age_penalty(monkeypatch, at=frozen_now)
    result_b = producer(cue.text, cue.mode, None, _TOP_K)
    assert result_a == result_b, (
        f"recall output diverged across a frozen-now clock-tick straddle: "
        f"{result_a} vs {result_b}"
    )


# ---------------------------------------------------------------------------
# Cue-set health re-assertion (the generator's shape guards)
# ---------------------------------------------------------------------------


def test_cue_set_health_guards():
    report = build_synthetic_corpus_and_cues(seed=_SEED)

    print(f"\n  cue-set health: total_cues={report.total_cues}")

    assert report.total_cues >= 50, report.total_cues
    for band in ("specific", "vague", "novel"):
        assert len(report.cues_by_band[band]) > 0, f"{band} band is empty"


# ---------------------------------------------------------------------------
# Bucket-B evidence: T8/T14/T15/T16/T17 are Python-applied to only the
# winner rows after the resident-scoring call returns -- BOTH sides of the
# top-k differential apply them identically regardless of which side did
# the Bucket-A scoring, so they are not part of the RED per-term controls
# above. This records real, measured evidence that each term (a) moves a
# candidate's final score and (b) where the term's own mechanics allow a
# uniform-vs-per-candidate distinction, flips relative rank order against a
# Bucket-A-only baseline. The obligation this hands to a future candidate
# producer (asserting a Bucket-B promotion is never lost past a margin-
# bounded top-(k+margin) window) is documented in the frozen fused-score
# term table, not asserted here.
# ---------------------------------------------------------------------------

def _build_bucket_b_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *,
    with_t17_edge: bool = True, fixture=None, embedder: "Embedder | None" = None,
):
    _select_driver("stdlib", monkeypatch)
    _monkeypatch_env(monkeypatch, tmp_path)
    suffix = "with-edge" if with_t17_edge else "no-edge"
    store_root = tmp_path / f"bucket-b-evidence-{suffix}"
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    embedder = embedder or Embedder()
    # A caller comparing two store variants (e.g. T17's with-edge/no-edge
    # arms) MUST pass the SAME fixture object to both calls: rebuilding via
    # build_bucket_b_evidence_fixture twice reuses the same seed for
    # deterministic content/embeddings but mints fresh uuid4() record ids
    # each time, which would silently break any id-based cross-store lookup.
    fixture = fixture or build_bucket_b_evidence_fixture(seed=_SEED, embedder=embedder)
    store = MemoryStore(path=store_root)
    insert_corpus(store, fixture.records)
    if with_t17_edge:
        apply_term_discrimination_edges(store, fixture)

    from iai_mcp.retrieve import build_runtime_graph
    graph, assignment, rich_club = build_runtime_graph(store)
    return store, graph, assignment, rich_club, embedder, fixture


def test_bucket_b_terms_measurably_change_score_and_rank(tmp_path, monkeypatch):
    _freeze_age_penalty(monkeypatch)
    store, graph, assignment, rich_club, embedder, fixture = _build_bucket_b_store(tmp_path, monkeypatch)

    def recall(cue: str, *, profile_state=None) -> dict:
        _pm._last_recall_latency_ms = 0.0
        response = recall_for_response(
            store=store, graph=graph, assignment=assignment, rich_club=rich_club,
            embedder=embedder, cue=cue, session_id="bucket-b-evidence",
            budget_tokens=100_000, mode="concept", profile_state=profile_state,
        )
        return {str(h.record_id): h.score for h in response.hits}

    evidence: dict[str, str] = {}

    # T8 -- interest_boost/dunn_quadrant apply the SAME gain to every
    # candidate: a uniform multiplier cannot flip relative rank order by
    # construction (order-preserving). Score-only evidence, documented.
    cue = fixture.probe_cue["T8_profile_modulation"]
    tid = str(fixture.ids["t8_target"])
    s_off = recall(cue).get(tid)
    s_on = recall(cue, profile_state={"interest_boost": 0.8}).get(tid)
    assert s_on != s_off, "T8 interest_boost had no measurable effect on score"
    evidence["T8_profile_modulation"] = (
        f"score {s_off:.6f} -> {s_on:.6f} (uniform multiplier -- rank-order-preserving by construction)"
    )

    # T14 tier boost -- near-tied pair, rank flip vs the IAI_MCP_TIER_BOOST=1.0 baseline.
    cue = fixture.probe_cue["T14_tier_boost"]
    ep, se = str(fixture.ids["t14_episodic"]), str(fixture.ids["t14_semantic"])
    monkeypatch.setenv("IAI_MCP_TIER_BOOST", "1.0")
    base = recall(cue)
    monkeypatch.delenv("IAI_MCP_TIER_BOOST", raising=False)
    boosted = recall(cue)
    assert base[ep] > base[se], "T14 baseline precondition failed: episodic should lead untboosted"
    assert boosted[se] > boosted[ep], (
        f"T14 tier boost did not flip rank order: baseline episodic={base[ep]:.6f} "
        f"semantic={base[se]:.6f}; boosted episodic={boosted[ep]:.6f} semantic={boosted[se]:.6f}"
    )
    evidence["T14_tier_boost"] = (
        f"baseline episodic={base[ep]:.6f} > semantic={base[se]:.6f}; "
        f"boosted episodic={boosted[ep]:.6f} < semantic={boosted[se]:.6f} -- rank flipped"
    )

    # T15 salience -- `salience_level` is not a SimpleRecordView field at
    # all, so it is unreachable on the graph-hydrated hot path exactly like
    # T10; the injection here is the same test-only getattr shadow.
    cue = fixture.probe_cue["T15_salience"]
    uf, cr = str(fixture.ids["t15_unflagged"]), str(fixture.ids["t15_critical"])
    base = recall(cue)
    shadow = _shadowed_getattr_forcing("salience_level", "critical", only_for_id=fixture.ids["t15_critical"])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_pm, "getattr", shadow, raising=False)
        mp.setattr(graph, "_records_view_cache", None, raising=False)
        injected = recall(cue)
    assert base[uf] > base[cr], "T15 baseline precondition failed: unflagged should lead uninjected"
    assert injected[cr] > injected[uf], (
        f"T15 salience injection did not flip rank order: baseline unflagged={base[uf]:.6f} "
        f"critical={base[cr]:.6f}; injected unflagged={injected[uf]:.6f} critical={injected[cr]:.6f}"
    )
    evidence["T15_salience"] = (
        f"baseline unflagged={base[uf]:.6f} > critical={base[cr]:.6f}; "
        f"injected unflagged={injected[uf]:.6f} < critical={injected[cr]:.6f} -- rank flipped "
        "(salience_level unreachable via SimpleRecordView -- test-only injection, like T10)"
    )

    # T16 temporal match -- twin pair (identical text -> perfect Bucket-A
    # tie), rank flip vs the IAI_MCP_TEMPORAL_BOOST=1.0 baseline.
    cue = fixture.probe_cue["T16_temporal_match"]
    nm, mt = str(fixture.ids["t16_nomatch"]), str(fixture.ids["t16_match"])
    monkeypatch.setenv("IAI_MCP_TEMPORAL_BOOST", "1.0")
    base = recall(cue)
    monkeypatch.delenv("IAI_MCP_TEMPORAL_BOOST", raising=False)
    boosted = recall(cue)
    assert base[nm] >= base[mt], "T16 baseline precondition failed: twin tie should not already favor match"
    assert boosted[mt] > boosted[nm], (
        f"T16 temporal boost did not flip rank order: baseline nomatch={base[nm]:.6f} "
        f"match={base[mt]:.6f}; boosted nomatch={boosted[nm]:.6f} match={boosted[mt]:.6f}"
    )
    evidence["T16_temporal_match"] = (
        f"baseline (twin tie) nomatch={base[nm]:.6f} match={base[mt]:.6f}; "
        f"boosted nomatch={boosted[nm]:.6f} < match={boosted[mt]:.6f} -- rank flipped"
    )

    print("\n  Bucket-B evidence:")
    for term, line in evidence.items():
        print(f"    {term}: {line}")


def test_t17_historical_verbatim_anchor_rewrite_changes_score(tmp_path, monkeypatch):
    """`store.boost_edges` canonicalizes (src, dst) by lexicographic sort of
    the id STRINGS (documented store invariant) -- it does not preserve
    caller-intended direction. For a directional edge type like
    "contradicts", which side of the pair lands as `src` (the
    "contradicts_outgoing" / historical-anchor role `build_temporal_
    validity_maps` reads) is therefore NOT determined by insertion order,
    and is resolved here at runtime from the built map rather than assumed
    from the fixture's own anchor/correction naming."""
    _freeze_age_penalty(monkeypatch)
    embedder = Embedder()
    fixture = build_bucket_b_evidence_fixture(seed=_SEED, embedder=embedder)
    store, graph, assignment, rich_club, embedder, fixture = _build_bucket_b_store(
        tmp_path, monkeypatch, with_t17_edge=True, fixture=fixture, embedder=embedder,
    )
    store_no_edge, graph_no_edge, assignment_no_edge, rich_club_no_edge, embedder_no_edge, _ = (
        _build_bucket_b_store(
            tmp_path, monkeypatch, with_t17_edge=False, fixture=fixture, embedder=embedder,
        )
    )

    from iai_mcp.retrieve import build_temporal_validity_maps

    outgoing, _ts = build_temporal_validity_maps(store)
    anchor_id = str(fixture.ids["t17_anchor"])
    correction_id = str(fixture.ids["t17_correction"])
    if anchor_id in outgoing and correction_id in outgoing[anchor_id]:
        resolved_anchor = anchor_id
    elif correction_id in outgoing and anchor_id in outgoing[correction_id]:
        resolved_anchor = correction_id
    else:
        pytest.fail(f"contradicts edge not found in either direction: outgoing={outgoing}")

    cue = fixture.probe_cue["T17_historical_verbatim_rewrite"]

    def recall(s, g, a, rc, e) -> dict:
        _pm._last_recall_latency_ms = 0.0
        response = recall_for_response(
            store=s, graph=g, assignment=a, rich_club=rc, embedder=e,
            cue=cue, session_id="bucket-b-t17", budget_tokens=100_000, mode="concept",
        )
        return {str(h.record_id): h.score for h in response.hits}

    no_edge_scores = recall(store_no_edge, graph_no_edge, assignment_no_edge, rich_club_no_edge, embedder_no_edge)
    with_edge_scores = recall(store, graph, assignment, rich_club, embedder)

    assert with_edge_scores[resolved_anchor] != no_edge_scores[resolved_anchor], (
        f"T17 historical_verbatim anchor rewrite had no measurable effect: "
        f"no-edge score={no_edge_scores[resolved_anchor]:.6f} "
        f"with-edge score={with_edge_scores[resolved_anchor]:.6f}"
    )
    print(
        f"\n  T17 evidence: resolved_anchor={resolved_anchor} "
        f"no-edge={no_edge_scores[resolved_anchor]:.6f} "
        f"with-edge={with_edge_scores[resolved_anchor]:.6f}"
    )


# ---------------------------------------------------------------------------
# Fixture-graph membership verdict: does the harness's `build_runtime_graph`
# pool match the live dispatch path's own candidate pool? RECORDED, not left
# silent. VERDICT (a), BOUNDED: the live dispatch's Layer-1 candidate
# collection (`core/__init__.py`, `store.query_similar(cue_vec,
# k=K_CANDIDATES, decode="rank")`, before its own exact-authority/hop/
# rich-club widening -- each of which can only ADD ids, never remove any)
# returns literally every active record whenever the corpus does not exceed
# K_CANDIDATES=200, coinciding BY CONSTRUCTION with `build_runtime_graph`'s
# whole-corpus pool (`graph.iter_nodes()`, what `_collect_graph_pool` reads
# into `pool_ids`). Asserted below over real ids on real cues, not narrated.
# It does NOT generalize past K_CANDIDATES-sized corpora -- a corpus larger
# than K_CANDIDATES (a store with hundreds of records, well past this
# harness's synthetic corpora) would need a genuinely live-path-shaped
# fixture; this harness's corpora (<=60 records) never require one, so none
# is added here.
# ---------------------------------------------------------------------------

def test_fixture_membership_matches_live_path_pool_bounded_by_k_candidates(tmp_path, monkeypatch):
    from iai_mcp.pipeline import K_CANDIDATES

    _freeze_age_penalty(monkeypatch)
    store, graph, assignment, rich_club, embedder = _build_driver_store("stdlib", tmp_path, monkeypatch)
    cues = flatten_cues(build_cue_set(seed=_SEED))
    node_ids = set(graph.iter_nodes())
    assert 0 < len(node_ids) <= K_CANDIDATES, (
        f"corpus size {len(node_ids)} is outside (0, K_CANDIDATES={K_CANDIDATES}] -- "
        "the bounded-equivalence precondition this verdict relies on no longer holds"
    )
    for cue in cues[:8]:
        cue_vec = np.asarray(embedder.embed(cue.text), dtype=np.float32)
        cue_norm = float(np.linalg.norm(cue_vec))
        if cue_norm > 0:
            cue_vec = cue_vec / cue_norm
        live_path_ids = {r.id for r, _s in store.query_similar(cue_vec, k=K_CANDIDATES, decode="rank")}
        assert live_path_ids == node_ids, (
            f"live-path ANN pool ({len(live_path_ids)} ids) diverges from the harness's "
            f"whole-corpus pool ({len(node_ids)} ids) for cue {cue.text!r} -- verdict (a) "
            "does not hold for this corpus/cue combination"
        )


# ---------------------------------------------------------------------------
# Post-write-delta: a cue whose candidate scope includes a record upserted
# AFTER the resident Rust index's last full rebuild must be read through the
# overlay-aware path (committed CSR + bounded delta), never silently
# invisible to only one side -- guarding against a false-green differential
# where every OTHER cue in this file is measured against a pre-write-warmed
# index.
# ---------------------------------------------------------------------------

def test_post_write_delta_overlay_differential(tmp_path, monkeypatch):
    from iai_mcp.store import flush_record_buffer
    from iai_mcp.store._rank_index import rank_index_for

    _freeze_age_penalty(monkeypatch)
    store, graph, assignment, rich_club, embedder = _build_driver_store("stdlib", tmp_path, monkeypatch)

    reference_producer = _make_python_producer(store, graph, assignment, rich_club, embedder)
    candidate_producer = _make_rust_producer(store, graph, assignment, rich_club, embedder)

    # Warm the resident Rust index BEFORE the write -- `score()` only builds
    # on `self._index is None`; every later call must drain the write
    # through the double-buffer overlay instead of re-scanning the store.
    warm_cue = build_cue_set(seed=_SEED)["specific"][0]
    candidate_producer(warm_cue.text, warm_cue.mode, None, _TOP_K)

    post_write_text = (
        "Alice's post-write delta probe: a new espresso grinder arrived for "
        "the kitchen counter this afternoon."
    )
    vec = np.asarray(embedder.embed(post_write_text), dtype=np.float32)
    vec_norm = float(np.linalg.norm(vec))
    if vec_norm > 0:
        vec = vec / vec_norm
    new_rec = MemoryRecord(
        id=uuid4(), tier="episodic", literal_surface=post_write_text, aaak_index="",
        embedding=vec.tolist(), community_id=None, centrality=0.0, detail_level=2,
        pinned=False, stability=0.5, difficulty=0.0, last_reviewed=None,
        never_decay=False, never_merge=False, provenance=[],
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
        tags=[], language="en",
    )
    store.insert(new_rec)
    flush_record_buffer(store)
    store._build_exact_index_sync()
    graph.add_node(new_rec.id, community_id=None, embedding=list(new_rec.embedding))
    graph.set_node_payload(new_rec.id, {
        "embedding": list(new_rec.embedding), "surface": new_rec.literal_surface,
        "tier": new_rec.tier, "tags": [], "language": "en", "aaak_index": "",
        "created_at": new_rec.created_at.isoformat(), "stability": 0.5, "centrality": 0.0,
    })
    _rank_handle = rank_index_for(store, graph)
    _rank_handle.feed("upsert", new_rec)
    # Draining a warm index for a post-write delta is the documented caller
    # responsibility (`_rank_index.py::score`'s own docstring) -- `feed()`
    # alone only queues the op; `snapshot()` is what applies it into the
    # generation-tagged buffer `score()` reads.
    _rank_handle.snapshot(graph)

    post_write_cue = CueSpec(text=post_write_text, band="post-write-delta", mode="concept")
    expected = reference_producer(post_write_cue.text, post_write_cue.mode, None, _TOP_K)
    got = candidate_producer(post_write_cue.text, post_write_cue.mode, None, _TOP_K)

    expected_ids = {rid for rid, _s in expected}
    assert str(new_rec.id) in expected_ids, (
        "post-write record absent even from the Python reference -- this "
        "fixture does not actually exercise a post-write candidate"
    )
    _assert_top_k_tie_tolerant(expected, got, k=_TOP_K, cue_seed=99999)


# ---------------------------------------------------------------------------
# An unknown-created_at candidate must not perturb real-timestamp ranking,
# and must itself score without crashing (both drivers).
# ---------------------------------------------------------------------------

def _freeze_age_penalty_allow_unknown(
    monkeypatch: pytest.MonkeyPatch, at: "datetime | None" = None,
) -> datetime:
    """Same clock freeze as `_freeze_age_penalty`, extended with a None-guard
    so an unknown-`created_at` candidate does not crash the frozen scorer --
    `_freeze_age_penalty`'s own replacement still assumes a real `datetime`
    and would raise on `None`."""
    frozen_now = at or datetime.now(timezone.utc)

    def _frozen(created_at: "datetime | None") -> float:
        if created_at is None:
            return 0.0
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        days = (frozen_now - created_at).total_seconds() / 86400.0
        if days < 0:
            return 0.0
        return min(1.0, days / _pm.AGE_HALF_LIFE_DAYS)

    monkeypatch.setattr(_pm, "_age_penalty", _frozen)
    return frozen_now


def _add_unknown_created_at_node(graph, embedder: Embedder, text: str) -> UUID:
    """Adds a graph-only node (no store record) whose payload carries
    complete `embedding`/`surface` keys but OMITS `created_at` entirely --
    the exact shape the records_cache build resolves into a
    `SimpleRecordView` with `created_at is None`, never falling through to
    a store fallback (there is no store record for this id at all)."""
    vec = np.asarray(embedder.embed(text), dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    node_id = uuid4()
    graph.add_node(node_id, community_id=None, embedding=list(vec))
    graph.set_node_payload(node_id, {
        "embedding": list(vec), "surface": text,
        "tier": "episodic", "tags": [], "language": "en", "aaak_index": "",
        "stability": 0.5, "centrality": 0.0,
    })
    return node_id


_SOURDOUGH_SPECIFIC_CUE = (
    "How often does Alice need to feed her sourdough starter during the "
    "summer heat wave?"
)
_UNRELATED_NOVEL_TEXT = "Notes on antique typewriter restoration techniques."


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_unknown_created_at_candidate_leaves_real_ranking_unchanged(tmp_path, monkeypatch, driver):
    """Invariance leg: adding one unknown-created_at graph-only candidate
    to the pool must not perturb the top-k order or fused scores of the
    corpus's real-timestamp records, for a cue that does not retrieve it."""
    _freeze_age_penalty_allow_unknown(monkeypatch)
    store, graph, assignment, rich_club, embedder = _build_driver_store(driver, tmp_path, monkeypatch)
    reference_producer = _make_python_producer(store, graph, assignment, rich_club, embedder)

    cue = CueSpec(text=_SOURDOUGH_SPECIFIC_CUE, band="specific", mode="concept")
    _, cue_intent, _ = _classify_cue(cue.text)
    before = reference_producer(cue.text, cue.mode, cue_intent, _TOP_K)
    assert before, "fixture precondition: the specific cue must retrieve real corpus records"

    unknown_id = _add_unknown_created_at_node(graph, embedder, _UNRELATED_NOVEL_TEXT)

    after = reference_producer(cue.text, cue.mode, cue_intent, _TOP_K)
    after_ids = {rid for rid, _s in after}
    assert str(unknown_id) not in after_ids, (
        "fixture precondition: the topically-unrelated unknown candidate "
        "must not rank into this cue's top-k, or this leg does not isolate "
        "ranking invariance from the unknown-handling leg"
    )
    _assert_top_k_tie_tolerant(before, after, k=_TOP_K, cue_seed=1)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_unknown_created_at_candidate_scored_without_crash(tmp_path, monkeypatch, driver):
    """Unknown-handling leg: a graph-only candidate whose payload omits
    `created_at` must resolve to a `SimpleRecordView` with `created_at is
    None` (representation vacuity guard -- never the age/score NUMBER, which
    is numerically indistinguishable from a fresh real timestamp when an
    unknown age scores neutral) and must surface in ranked output without
    recall raising."""
    from iai_mcp.pipeline import SimpleRecordView

    _freeze_age_penalty_allow_unknown(monkeypatch)
    store, graph, assignment, rich_club, embedder = _build_driver_store(driver, tmp_path, monkeypatch)
    reference_producer = _make_python_producer(store, graph, assignment, rich_club, embedder)

    candidate_text = (
        "Alice's brand-new espresso grinder review, freshly written and not "
        "part of any existing corpus record."
    )
    unknown_id = _add_unknown_created_at_node(graph, embedder, candidate_text)

    cue = CueSpec(text=candidate_text, band="unknown-created-at-probe", mode="concept")
    _, cue_intent, _ = _classify_cue(cue.text)
    results = reference_producer(cue.text, cue.mode, cue_intent, _TOP_K)

    result_ids = {rid for rid, _s in results}
    assert str(unknown_id) in result_ids, (
        f"unknown-created_at candidate did not surface in ranked output: {results}"
    )

    _cached = getattr(graph, "_records_view_cache", None)
    assert _cached is not None, (
        "records_cache generational cache was not populated by the recall "
        "call -- cannot inspect the actual records_cache entry"
    )
    view = _cached[1].get(unknown_id)
    assert view is not None, "unknown candidate not resolved into the records_cache view"
    assert isinstance(view, SimpleRecordView), (
        f"unknown candidate resolved to {type(view)!r}, not the graph-view "
        "SimpleRecordView -- a store-fallback resolution would make this leg vacuous"
    )
    assert view.created_at is None, (
        "representation vacuity guard: an unknown created_at must resolve to "
        "None, not a fabricated now()"
    )
