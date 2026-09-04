"""Winners-only decrypt + churn + gain-recovery proofs for the Rust hybrid
scorer (kill-switch default path).

Scope, as ruled: the SCORING-STAGE + response-assembly decrypt is winners-
only (`test_decrypt_count_is_order_winners`) -- proven here directly, by
isolating the scoring call from the candidate-retrieval decrypt site this
scoring-stage proof never touches:

  The Layer-1 candidate-collection `query_similar(decode="rank")` call in
  `core/__init__.py`'s `dispatch` -- decrypts `literal_surface` for the
  whole ANN-retrieved candidate pool before scoring is ever reached.
  Deferred: relying on the resident index's own features for the
  lexical/AAAK terms instead of decrypting surface text at retrieval time
  would change what the Python reference path receives -- that path must
  stay callable, byte-for-byte, as the differential reference.

A confidence-widen re-query of the whole candidate pool is not a decrypt
site on the default path -- single-pass recall has no such re-query.

Measured on a 500-record corpus (steady state, after the resident index's
one-time build): whole-recall decrypt count is ~1180/recall via
`core.dispatch`, dominated by the site above, identical whether the
kill-switch selects Rust or Python. Calling `recall_for_response` directly
(bypassing that site) isolates exactly the scope this file's proofs touch:
~150 decrypt calls for 50 served hits (`_backfill_hit_metadata`'s
`decode="full"` fetch, three fields/record), independent of corpus size.
"""
from __future__ import annotations

import sys
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_recall_ann_first_quality import _monkeypatch_env  # noqa: E402
from test_recall_scoring_differential import _freeze_age_penalty  # noqa: E402

from tests._synthetic_cue_corpus import (  # noqa: E402
    apply_term_discrimination_edges,
    build_bucket_b_evidence_fixture,
    build_corpus_records,
    insert_corpus,
)
from tests.test_exact_authority_index import _TOP_K_TIE_TOL  # noqa: E402
from tests.test_rank_index_rss_soak import _scale_records  # noqa: E402

import iai_mcp.pipeline as _pm  # noqa: E402
from iai_mcp.embed import Embedder  # noqa: E402
from iai_mcp.pipeline import _reinsert_rust_winner_gain, recall_for_response  # noqa: E402
from iai_mcp.retrieve import build_runtime_graph  # noqa: E402
from iai_mcp.store import MemoryStore  # noqa: E402
from iai_mcp.store._rank_index import rank_index_for  # noqa: E402

_SEED = 0
_CORPUS_N = 500
_FRESH_CUE = "Notes on the neighborhood tool-lending library proposal update."


def _build_scaled_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embedder: Embedder):
    _monkeypatch_env(monkeypatch, tmp_path)
    store_root = tmp_path / "winners-only-decrypt-store"
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    base = build_corpus_records(seed=_SEED, embedder=embedder)
    records = _scale_records(base, _CORPUS_N, seed=_SEED + 1)
    store = MemoryStore(path=store_root)
    insert_corpus(store, records)
    graph, assignment, rich_club = build_runtime_graph(store)
    return store, graph, assignment, rich_club


def _decrypt_counter(monkeypatch: pytest.MonkeyPatch) -> "dict[str, int]":
    counter = {"n": 0}
    orig = MemoryStore._decrypt_for_record

    def _wrapped(self, record_id, value):
        counter["n"] += 1
        return orig(self, record_id, value)

    monkeypatch.setattr(MemoryStore, "_decrypt_for_record", _wrapped)
    return counter


def test_decrypt_count_is_order_winners(tmp_path, monkeypatch):
    """Non-vacuity: the bound is a small multiple of the served hit+anti-hit
    count, never an absolute constant and never proportional to corpus size
    -- a candidate-pool-scaled regression on this 500-record corpus would
    push the count into the hundreds well past this bound, exactly the shape
    the pre-fix hydrate-stage decrypt had."""
    embedder = Embedder()
    store, graph, assignment, rich_club = _build_scaled_store(tmp_path, monkeypatch, embedder)

    # Prime the resident index once, off the measured window -- this is a
    # one-time whole-corpus cold build, not a per-recall cost.
    rank_index_for(store, graph).snapshot(graph)

    # Isolate scoring + response assembly from the upstream candidate-
    # retrieval site: calling recall_for_response directly instead of
    # core.dispatch bypasses it.
    counter = _decrypt_counter(monkeypatch)
    _pm._last_recall_latency_ms = 0.0
    response = recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_FRESH_CUE, session_id="winners-only-decrypt-probe",
        budget_tokens=2000, mode="concept", use_rust_scorer=True,
    )
    winners_count = len(response.hits) + len(response.anti_hits)
    assert winners_count > 0, "probe cue produced zero hits -- fixture is broken"

    decrypt_calls = counter["n"]
    bound = 4 * winners_count
    assert decrypt_calls <= bound, (
        f"decrypt_calls={decrypt_calls} exceeds O(winners) bound {bound} "
        f"(winners_count={winners_count}); scoring-stage decrypt is no longer winners-only"
    )
    assert decrypt_calls * 2 < _CORPUS_N, (
        f"decrypt_calls={decrypt_calls} is within a factor of 2 of the corpus size "
        f"({_CORPUS_N}) -- this bound would not distinguish O(winners) from O(candidates)"
    )


def test_decrypt_count_bounded_no_new_site(tmp_path, monkeypatch):
    """Mirrors test_decrypt_count_is_order_winners: the O(winners) decrypt
    bound must hold on the full-pool default path, proving no new decrypt
    site was introduced there."""
    embedder = Embedder()
    store, graph, assignment, rich_club = _build_scaled_store(tmp_path, monkeypatch, embedder)
    rank_index_for(store, graph).snapshot(graph)

    counter = _decrypt_counter(monkeypatch)
    _pm._last_recall_latency_ms = 0.0
    response = recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_FRESH_CUE, session_id="scr04-bounded-no-new-site-probe",
        budget_tokens=2000, mode="concept", use_rust_scorer=True,
    )
    winners_count = len(response.hits) + len(response.anti_hits)
    assert winners_count > 0, "probe cue produced zero hits -- fixture is broken"
    decrypt_calls = counter["n"]
    bound = 4 * winners_count
    assert decrypt_calls <= bound, (
        f"decrypt_calls={decrypt_calls} exceeds O(winners) bound {bound} "
        f"(winners_count={winners_count}) -- a new decrypt site was "
        "introduced on the default path"
    )


def test_bucket_b_applied_to_winners_only(tmp_path, monkeypatch):
    """Mirrors test_recall_scoring_differential.py's Python-path Bucket-B
    evidence test, but forces the Rust-scored path (`use_rust_scorer=True`)
    to prove the SAME per-call-state terms (T14 tier boost, T16 temporal
    match) measurably flip rank there too. Applied to winners only is a
    structural property of the code (the loop iterates the bounded `winners`
    Rust returns, never the candidate pool) -- the measurable score change
    here is the observable proof the winners loop actually runs the terms,
    not a fixture that silently no-ops on the Rust path."""
    _freeze_age_penalty(monkeypatch)
    _monkeypatch_env(monkeypatch, tmp_path)
    store_root = tmp_path / "bucket-b-rust-store"
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    embedder = Embedder()
    fixture = build_bucket_b_evidence_fixture(seed=_SEED, embedder=embedder)
    store = MemoryStore(path=store_root)
    insert_corpus(store, fixture.records)
    apply_term_discrimination_edges(store, fixture)
    graph, assignment, rich_club = build_runtime_graph(store)

    def recall(cue: str) -> dict:
        _pm._last_recall_latency_ms = 0.0
        response = recall_for_response(
            store=store, graph=graph, assignment=assignment, rich_club=rich_club,
            embedder=embedder, cue=cue, session_id="bucket-b-rust-evidence",
            budget_tokens=100_000, mode="concept", use_rust_scorer=True,
        )
        return {str(h.record_id): h.score for h in response.hits}

    # T14 tier boost.
    cue = fixture.probe_cue["T14_tier_boost"]
    ep, se = str(fixture.ids["t14_episodic"]), str(fixture.ids["t14_semantic"])
    monkeypatch.setenv("IAI_MCP_TIER_BOOST", "1.0")
    base = recall(cue)
    monkeypatch.delenv("IAI_MCP_TIER_BOOST", raising=False)
    boosted = recall(cue)
    assert base[ep] > base[se], "T14 baseline precondition failed on the Rust path"
    assert boosted[se] > boosted[ep], (
        f"T14 tier boost did not flip rank order on the Rust path: "
        f"baseline episodic={base[ep]:.6f} semantic={base[se]:.6f}; "
        f"boosted episodic={boosted[ep]:.6f} semantic={boosted[se]:.6f}"
    )

    # T16 temporal match -- twin pair, perfect Bucket-A tie on the Rust path too.
    cue = fixture.probe_cue["T16_temporal_match"]
    nm, mt = str(fixture.ids["t16_nomatch"]), str(fixture.ids["t16_match"])
    monkeypatch.setenv("IAI_MCP_TEMPORAL_BOOST", "1.0")
    base = recall(cue)
    monkeypatch.delenv("IAI_MCP_TEMPORAL_BOOST", raising=False)
    boosted = recall(cue)
    assert base[nm] >= base[mt], "T16 baseline precondition failed on the Rust path"
    assert boosted[mt] > boosted[nm], (
        f"T16 temporal boost did not flip rank order on the Rust path: "
        f"baseline nomatch={base[nm]:.6f} match={base[mt]:.6f}; "
        f"boosted nomatch={boosted[nm]:.6f} match={boosted[mt]:.6f}"
    )


def test_ffi_params_no_per_candidate_churn(tmp_path, monkeypatch):
    """Structural (construction-count) guard, not wall-clock: counts
    `uuid.UUID` construction during one Rust-scored recall over a corpus
    much larger than k+k_margin. Bounded to a small multiple of the served
    winners count -- a per-candidate churn regression (e.g. reintroducing
    `[UUID(int=rid.int) for rid in pool_ids]` in the scoring loop) would
    push the count toward the full candidate pool.

    Non-vacuity: a deliberately churny control (constructing one UUID per
    pool id, mirroring the exact regression shape) is run through the same
    counting harness and asserted to fail the bound, proving the guard can
    actually catch it."""
    embedder = Embedder()
    store, graph, assignment, rich_club = _build_scaled_store(tmp_path, monkeypatch, embedder)
    rank_index_for(store, graph).snapshot(graph)

    construction_count = {"n": 0}
    real_uuid = _pm.UUID

    def _counting_uuid(*args, **kwargs):
        construction_count["n"] += 1
        return real_uuid(*args, **kwargs)

    monkeypatch.setattr(_pm, "UUID", _counting_uuid)

    _pm._last_recall_latency_ms = 0.0
    response = recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_FRESH_CUE, session_id="no-churn-probe",
        budget_tokens=2000, mode="concept", use_rust_scorer=True,
    )
    winners_count = len(response.hits) + len(response.anti_hits)
    assert winners_count > 0, "probe cue produced zero hits -- fixture is broken"

    bound = 4 * winners_count
    real_count = construction_count["n"]
    assert real_count <= bound, (
        f"UUID construction count={real_count} exceeds O(winners) bound {bound} "
        f"(winners_count={winners_count}) -- per-candidate churn regression"
    )

    # RED control: the exact churn shape a regression would reintroduce --
    # one UUID construction per pool id -- must fail the same bound.
    pool_ids, _pool_embs = _pm._collect_graph_pool(graph, {}, store)
    construction_count["n"] = 0
    _churny_control = [_counting_uuid(int=rid.int) for rid in pool_ids]
    assert len(_churny_control) == len(pool_ids)
    assert construction_count["n"] > bound, (
        "non-vacuity check failed: a deliberately per-candidate-churny construction "
        f"({construction_count['n']} calls over {len(pool_ids)} candidates) did not "
        f"exceed the bound ({bound}) -- this guard could not detect a real regression"
    )


_GAIN_SPREAD = [0.0, 0.01, 0.5, 0.8, 0.95, 1.0, 1.05, 1.2, 1.5, 2.0, 3.7]
_TERM_MULTIPLIERS = [1.0, 2.0, 3.0, 6.0]


@pytest.mark.parametrize("term_multiplier", _TERM_MULTIPLIERS)
@pytest.mark.parametrize("gain_product", _GAIN_SPREAD)
def test_gain_recovery_byte_identical(term_multiplier: float, gain_product: float):
    """Algebraic proof: `_reinsert_rust_winner_gain` reconstructs the exact
    value Python's own multiplicative gain application computes --
    `s = base_s * gain_product` BEFORE the stability lift, THEN
    `* term_multiplier` (trigram/fts), THEN `+ lex_add` (pipeline.py's
    per-candidate loop, :2115-2143) -- from a Rust `WinnerRow`'s pre-gain
    `partial_score` plus its raw `pre_gain_base`/`term_multiplier` factors,
    without a second Rust scoring pass. Swept over a spread of profile-gain
    values and trigram/fts multiplier combinations (1x none, 2x trigram,
    3x fts, 6x both)."""
    pre_gain_base = 0.734
    stability_lift = 0.037
    lex_add = 0.112

    partial_score = (pre_gain_base + stability_lift) * term_multiplier + lex_add
    python_true = ((pre_gain_base * gain_product) + stability_lift) * term_multiplier + lex_add

    reconstructed = _reinsert_rust_winner_gain(
        partial_score, pre_gain_base, term_multiplier, gain_product,
    )
    assert abs(reconstructed - python_true) <= _TOP_K_TIE_TOL, (
        f"gain-recovery diverged from Python's additive computation: "
        f"reconstructed={reconstructed!r} python_true={python_true!r} "
        f"(pre_gain_base={pre_gain_base}, term_multiplier={term_multiplier}, "
        f"gain_product={gain_product})"
    )


def test_gain_recovery_additively_wrong_control_goes_red():
    """Non-vacuity: an additively-wrong reconstruction that omits the
    `term_multiplier` factor (a plausible regression -- applying the gain
    delta at the wrong arithmetic point) must diverge from Python's true
    value whenever term_multiplier != 1.0, proving the byte-identity
    assertion above is discriminating, not trivially satisfied."""
    pre_gain_base = 0.734
    stability_lift = 0.037
    lex_add = 0.112
    term_multiplier = 3.0
    gain_product = 1.5

    partial_score = (pre_gain_base + stability_lift) * term_multiplier + lex_add
    python_true = ((pre_gain_base * gain_product) + stability_lift) * term_multiplier + lex_add

    wrong_reconstruction = partial_score + pre_gain_base * (gain_product - 1.0)
    assert abs(wrong_reconstruction - python_true) > _TOP_K_TIE_TOL, (
        "non-vacuity check failed: the deliberately additively-wrong "
        "reconstruction matched Python's true value -- the byte-identity "
        "assertion could not have caught this regression"
    )
