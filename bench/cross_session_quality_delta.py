"""Rank-aware chain-integrity quality comparator for cross-session recall.

The shipped drop-out-only comparator
(``tests/test_retrieval_weight_recall_differential.py::_comparator``) treats
a reshuffle within the returned set as benign by design -- it can see
eviction but not reordering. This module measures RANK POSITION over the
ordered dispatch output instead, which is strictly more sensitive: a gold id
that stays in the returned set but moves to a better or worse index still
registers a non-zero delta.

The measured before/after quantity is recall quality with the memory chain
INTACT versus a genuinely INCOMPLETE (partial-capture) chain -- no
retrieval-weight lever is armed anywhere in this module. A degraded twin
corpus is built by OMITTING a chosen subset of gold-bearing records at
insertion time (never a hard-DELETE of an already-written row); every cue
is dispatched once against its own fresh store, so no within-store
co-retrieval self-loop warm-up can skew one leg relative to another. An A/A
floor from two INTACT replays (identical gold set, nuisance-only filler
difference) bounds a non-vacuous acceptance margin.

Hermetic: builds its own isolated synthetic corpus via
``bench.recall_accuracy.build_eval_set`` -- never opens a copy of the
operator's live store. Diagnostics carry record ids and counts only, never
cue text or stored content.

Non-vacuity is proven at two layers, never conflated: a real degradation
run (the decisive rung) proves the end-to-end pipeline resolves a genuine
quality difference; a comparator-input plant sized at floor scale (the
near-floor rung, built only in the test suite, mirroring the shape of
``tests/test_proc_corpus_census.py``'s planted-signal-vs-noise pair) proves
the statistical verdict machinery has resolution at that scale. A
no-displacement control (``partition_cues_by_involvement`` /
``no_displacement_value_metric``) checks that a degradation never worsens
cues whose own gold was untouched. Every replay resolves to exactly one of
passed / failed / excluded-unavailable via
``iai_mcp.availability.classify_session_outcome`` -- an unmeasurable
replay is excluded, never scored as a pass.
"""
from __future__ import annotations

import contextlib
import dataclasses
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable

_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)

import numpy as np

from bench.proc_corpus_census import two_sided_aa_floor
from bench.recall_accuracy import (
    _flush,
    _sample_active_record_embeddings,
    _scan_active_corpus,
    _unit_vec,
    build_eval_set,
)
from bench.recall_accuracy_real import (
    _BASELINE_STRUCTURAL_WEIGHT,
    _dispatch_real_cue,
    warm_eval_copy_store,
)
from iai_mcp import runtime_graph_cache
from iai_mcp.availability import (
    classify_session_outcome,
    format_measurable_summary,
    summarize_measurable,
)
from iai_mcp.lilli.profile.retrieval_tuning import RETRIEVAL_MIN_SAMPLES
from iai_mcp.retrieve import build_runtime_graph

# Larger than any bounded dispatch hit-list length -- can never collide with
# a genuine rank. Unlike class_cosine_rank's same-named parameter (excluded
# by its caller from that function's distribution), this sentinel IS folded
# directly into value_metric's mean: a dropped-and-found-elsewhere gold
# contributes a full ~NOT_FOUND_RANK-scale jump, not a small continuous
# delta.
NOT_FOUND_RANK: int = 9999

_DEFAULT_CUE_WORD_COUNT = 6

# Arbitrary, fixed seed for the A/A floor's nuisance-redraw twin -- distinct
# from the corpus's own build seed so the redraw is never the same RNG
# stream as the corpus construction it perturbs.
_DEFAULT_NUISANCE_SEED = 1


def rank_of(ordered_ids: "list[str]", gold_id: str) -> int:
    """0-based rank of gold_id in ordered_ids, or NOT_FOUND_RANK when absent."""
    try:
        return ordered_ids.index(gold_id)
    except ValueError:
        return NOT_FOUND_RANK


def signed_rank_delta(before_ids: "list[str]", after_ids: "list[str]", gold_id: str) -> int:
    """rank_before(gold) - rank_after(gold): positive means gold moved to a
    lower (better) index after -- an improvement; negative, a regression.
    Both ranks come from the same ORDERED dispatch output, so a reshuffle
    within an unchanged returned set still produces a non-zero delta -- the
    eviction-only shipped comparator cannot see that case at all.
    """
    return rank_of(before_ids, gold_id) - rank_of(after_ids, gold_id)


def per_cue_rank_deltas(
    cues: "list[dict]",
    before_ids_by_cue: "dict[str, list[str]]",
    after_ids_by_cue: "dict[str, list[str]]",
) -> "dict[str, int]":
    """Signed rank-delta for every cue, keyed by cue_id."""
    return {
        cue["cue_id"]: signed_rank_delta(
            before_ids_by_cue[cue["cue_id"]],
            after_ids_by_cue[cue["cue_id"]],
            cue["gold_id"],
        )
        for cue in cues
    }


def value_metric(deltas: "Iterable[int]") -> float:
    """Mean signed rank-delta over the measured cues; 0.0 for an empty vector."""
    values = list(deltas)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _self_labelled_cues(
    store,
    *,
    n_cues: int,
    seed: int,
    cue_word_count: int = _DEFAULT_CUE_WORD_COUNT,
) -> "list[dict]":
    """Derive up to n_cues labelled cues straight from the corpus's own
    active records: each cue's text is the first words of a record's own
    literal_surface, and that record's own id is the cue's gold target --
    the same self-labelling run_baseline already applies with n_samples, no
    external fixture needed.
    """
    from iai_mcp.session import _clean_surface

    rng = np.random.default_rng(seed)
    id_to_vec, class_size = _scan_active_corpus(store)
    samples, _excluded = _sample_active_record_embeddings(
        store, n_cues, rng, id_to_vec=id_to_vec, class_size=class_size
    )
    ids = [uid for uid, _vec in samples]
    batch = store.get_batch(ids)
    cues: "list[dict]" = []
    for i, uid in enumerate(ids):
        rec = batch.get(uid)
        if rec is None:
            continue
        words = _clean_surface(rec.literal_surface or "").split()
        if not words:
            continue
        cue_text = " ".join(words[:cue_word_count])
        if not cue_text:
            continue
        cues.append({"cue_id": f"cue-{i}", "gold_id": str(uid), "cue_text": cue_text})
    return cues


def _dispatch_all(store, cues: "list[dict]", structural_weight: float) -> "dict[str, list[str]]":
    return {
        cue["cue_id"]: _dispatch_real_cue(store, cue["cue_text"], structural_weight)
        for cue in cues
    }


def _rebuild_graph(store) -> None:
    runtime_graph_cache.invalidate(store)
    build_runtime_graph(store)


def _resolved_cue_ids(cues: "list[dict]", ids_by_cue: "dict[str, list[str]]") -> "list[str]":
    """cue_ids whose own gold is present (a real rank) in ids_by_cue -- the
    only cues eligible to serve as a genuine positive-control drop (dropping
    a gold that was already unreachable proves nothing)."""
    return [
        cue["cue_id"]
        for cue in cues
        if rank_of(ids_by_cue[cue["cue_id"]], cue["gold_id"]) != NOT_FOUND_RANK
    ]


def _assert_commensurable(real_deltas: "dict[str, int]", null_deltas: "dict[str, int]") -> None:
    """The null and real delta vectors must pair cue-for-cue -- same length,
    identical cue keys -- or the floor they feed is meaningless."""
    if len(real_deltas) != len(null_deltas) or set(real_deltas) != set(null_deltas):
        raise ValueError(
            "real and null delta vectors are not commensurable: cue keys or length differ"
        )


def _build_corpus_twin(
    batch: "dict",
    keep_ids: "list",
    gold_ids: "set[str]",
    *,
    nuisance_seed: "int | None",
    dim: int,
    store_path: Path,
):
    """Insert a subset of an already-fetched record batch into a fresh store
    at store_path, in the given deterministic id order.

    Any id excluded from keep_ids is genuinely never written into this
    store -- omission at insertion time, never a hard-DELETE of an
    already-persisted row. When nuisance_seed is given, every kept record
    whose id is NOT in gold_ids gets a freshly redrawn unit embedding (the
    sole nuisance axis for the A/A floor); gold-bearing records always
    reinsert byte-identical to the source batch.
    """
    from iai_mcp.store import MemoryStore

    rng = np.random.default_rng(nuisance_seed) if nuisance_seed is not None else None
    twin = MemoryStore(path=store_path)
    try:
        for uid in keep_ids:
            rec = batch.get(uid)
            if rec is None:
                continue
            if rng is not None and str(uid) not in gold_ids:
                rec = dataclasses.replace(rec, embedding=_unit_vec(rng, dim).tolist())
            twin.insert(rec)
        _flush(twin)
        _rebuild_graph(twin)
        warm_eval_copy_store(twin)
    except BaseException:
        # A failure anywhere above leaves this store handle open with no
        # reference outside this function -- close it before propagating.
        twin.close()
        raise
    return twin


class DegenerateFloorError(RuntimeError):
    """Raised when the measured A/A null floor's upper bound is not
    strictly positive -- a collapsed floor would make the gate cosmetic
    (any positive delta would clear it)."""


def chain_integrity_verdict(
    *, value_metric: float, aa_floor: "tuple[float, float]", established_cues: int
) -> "dict":
    """Local go/PARK verdict mirroring proc_corpus_census.park_verdict's
    shape: proceeds only if value_metric strictly clears the measured A/A
    floor's upper bound AND established_cues clears the imported
    statistical-power floor (RETRIEVAL_MIN_SAMPLES); otherwise PARK.
    established_cues == 0 flags underpowered, distinct from a genuinely
    measured flat/negative value_metric -- both can read value_metric ==
    0.0, and only established_cues tells them apart.

    Raises DegenerateFloorError before computing any verdict when the
    floor's own upper bound is not strictly positive -- a collapsed null
    means the nuisance axis vanished and any positive delta would clear it
    vacuously.
    """
    if aa_floor[1] <= 0.0:
        raise DegenerateFloorError(
            f"measured A/A floor upper bound is not strictly positive ({aa_floor[1]!r}); "
            "the nuisance axis collapsed to a degenerate null -- no verdict is safe to emit"
        )
    value_clears_floor = value_metric > aa_floor[1]
    established_clears_floor = established_cues >= RETRIEVAL_MIN_SAMPLES
    verdict = "proceed" if (value_clears_floor and established_clears_floor) else "PARK"
    return {
        "verdict": verdict,
        "value_clears_floor": value_clears_floor,
        "established_clears_floor": established_clears_floor,
        "established_cues": established_cues,
        "underpowered": established_cues == 0,
        "power_floor": RETRIEVAL_MIN_SAMPLES,
        "aa_floor": aa_floor,
        "value_metric": value_metric,
    }


@contextlib.contextmanager
def _storage_driver_env(driver: "str | None"):
    """Overrides LILLI_STORAGE_DRIVER for the with-block, restoring the
    prior value (or its prior absence) on exit -- never leaves the process
    environment mutated after the block ends, even on exception."""
    had_prior = "LILLI_STORAGE_DRIVER" in os.environ
    prior = os.environ.get("LILLI_STORAGE_DRIVER")
    if driver == "lilli":
        os.environ["LILLI_STORAGE_DRIVER"] = "lilli"
    elif driver == "stdlib":
        os.environ.pop("LILLI_STORAGE_DRIVER", None)
    try:
        yield
    finally:
        if had_prior:
            os.environ["LILLI_STORAGE_DRIVER"] = prior
        else:
            os.environ.pop("LILLI_STORAGE_DRIVER", None)


def run_chain_integrity_slice(
    *,
    seed: int = 0,
    n_filler: int = 220,
    n_cues: int = 10,
    nuisance_seed: int = _DEFAULT_NUISANCE_SEED,
    driver: "str | None" = None,
) -> dict:
    """One hermetic chain-integrity measurement: memory chain INTACT versus
    a genuinely INCOMPLETE (partial-capture) twin, measured end-to-end
    through the real dispatch path -- no retrieval-weight lever anywhere.

    Builds one synthetic labelled source corpus, derives self-labelled cues
    from it, then builds three twins by re-inserting the source corpus's
    own already-generated records into fresh stores through the SAME
    reinsertion path (never a second copy of the operator's live store,
    never a hard-DELETE):

      - intact_a: every active record, byte-identical to the source.
      - degraded: every active record EXCEPT a chosen subset of
        gold-bearing records that actually resolve in intact_a, genuinely
        omitted at insertion time -- the synthetic analog of turns the
        capture pipeline never durably wrote.
      - intact_b: every active record, with every NON-gold record's
        embedding freshly redrawn (the nuisance axis for the A/A floor) --
        the gold set itself stays byte-identical to intact_a.

    Every cue is dispatched exactly once against each of the three twins
    (never twice against the same store), eliminating cross-leg skew from
    same-cue-repeated co-retrieval self-loop warm-up.

    Direction convention: signed_rank_delta(before, after, gold) is
    rank_before - rank_after, positive meaning gold moved to a better
    (lower) index. The real vector treats degraded as "before" and
    intact_a as "after", so intact-better-than-degraded -- a genuine
    quality drop under degradation -- reads positive. The null vector
    contrasts the two INTACT replays (intact_a as "before", intact_b as
    "after"): a should-be-near-zero, two-sided quantity, never
    repeat-vs-first, never the degradation itself.

    Returns the real and null per-cue delta vectors, the real vector's
    scalar value_metric, the measured A/A floor, and enough bookkeeping
    (dropped ids, per-cue ranks) for the positive-control checks. Does NOT
    compute a go/PARK verdict -- call chain_integrity_verdict on this
    result's value_metric/aa_floor/resolved_cues when a verdict is wanted
    (resolved_cues, not n_cues: only cues whose own gold resolves in
    intact_a carry any signal, so the power gate must be fed that
    subset's count, not the total cue count). A verdict is a separate,
    deliberate step, not an automatic side effect of the measurement.
    """
    with _storage_driver_env(driver):
        with tempfile.TemporaryDirectory(prefix="iai-mcp-quality-delta-src-") as td_src:
            source_store, _graph, _assignment, _rich_club, _embedder, _cases = build_eval_set(
                seed=seed, n_filler=n_filler, store_path=Path(td_src)
            )
            try:
                cues = _self_labelled_cues(source_store, n_cues=n_cues, seed=seed)
                if not cues:
                    raise RuntimeError("no self-labelled cues derivable from the synthetic corpus")

                id_to_vec, _class_size = _scan_active_corpus(source_store)
                all_ids = sorted(id_to_vec.keys(), key=lambda u: u.hex)
                dim = len(next(iter(id_to_vec.values())))
                gold_ids = {cue["gold_id"] for cue in cues}
                batch = source_store.get_batch(all_ids)
            finally:
                source_store.close()

        with tempfile.TemporaryDirectory(prefix="iai-mcp-quality-delta-a-") as td_a:
            intact_a = _build_corpus_twin(
                batch, all_ids, gold_ids, nuisance_seed=None, dim=dim, store_path=Path(td_a)
            )
            try:
                intact_a_ids_by_cue = _dispatch_all(intact_a, cues, _BASELINE_STRUCTURAL_WEIGHT)
            finally:
                intact_a.close()

        resolved = _resolved_cue_ids(cues, intact_a_ids_by_cue)
        if not resolved:
            raise RuntimeError(
                "no self-labelled cue resolved its own gold in the intact leg -- "
                "cannot construct a positive-control drop"
            )
        n_drop = max(1, len(resolved) // 2)
        drop_cue_ids = set(resolved[:n_drop])
        drop_ids = {cue["gold_id"] for cue in cues if cue["cue_id"] in drop_cue_ids}

        degraded_keep_ids = [uid for uid in all_ids if str(uid) not in drop_ids]
        with tempfile.TemporaryDirectory(prefix="iai-mcp-quality-delta-deg-") as td_deg:
            degraded = _build_corpus_twin(
                batch, degraded_keep_ids, gold_ids, nuisance_seed=None, dim=dim, store_path=Path(td_deg)
            )
            try:
                degraded_ids_by_cue = _dispatch_all(degraded, cues, _BASELINE_STRUCTURAL_WEIGHT)
            finally:
                degraded.close()

        with tempfile.TemporaryDirectory(prefix="iai-mcp-quality-delta-b-") as td_b:
            intact_b = _build_corpus_twin(
                batch, all_ids, gold_ids, nuisance_seed=nuisance_seed, dim=dim, store_path=Path(td_b)
            )
            try:
                intact_b_ids_by_cue = _dispatch_all(intact_b, cues, _BASELINE_STRUCTURAL_WEIGHT)
            finally:
                intact_b.close()

        real_deltas = per_cue_rank_deltas(cues, degraded_ids_by_cue, intact_a_ids_by_cue)
        null_deltas = per_cue_rank_deltas(cues, intact_a_ids_by_cue, intact_b_ids_by_cue)
        _assert_commensurable(real_deltas, null_deltas)

        intact_a_ranks = {
            cue["cue_id"]: rank_of(intact_a_ids_by_cue[cue["cue_id"]], cue["gold_id"])
            for cue in cues
        }
        degraded_ranks = {
            cue["cue_id"]: rank_of(degraded_ids_by_cue[cue["cue_id"]], cue["gold_id"])
            for cue in cues
        }
        # Uninvolved cues (own gold not among the drop targets) whose rank
        # crossed the NOT_FOUND_RANK boundary between intact_a and
        # degraded, split by direction: "gained" is a gold that had no own
        # rank in intact_a and surfaced once competing records were
        # removed (an incidental corpus-shrinkage effect, not a
        # displacement of an established rank); "lost" is a gold that DID
        # have an own rank in intact_a and dropped out of the degraded
        # hit-list -- the genuine unfavorable-displacement signal a
        # no-displacement control exists to catch.
        uninvolved_gained = sum(
            1
            for cue in cues
            if cue["cue_id"] not in drop_cue_ids
            and intact_a_ranks[cue["cue_id"]] == NOT_FOUND_RANK
            and degraded_ranks[cue["cue_id"]] != NOT_FOUND_RANK
        )
        uninvolved_lost = sum(
            1
            for cue in cues
            if cue["cue_id"] not in drop_cue_ids
            and intact_a_ranks[cue["cue_id"]] != NOT_FOUND_RANK
            and degraded_ranks[cue["cue_id"]] == NOT_FOUND_RANK
        )

        return {
            "value_metric": value_metric(real_deltas.values()),
            "real_deltas": real_deltas,
            "null_deltas": null_deltas,
            "aa_floor": two_sided_aa_floor(list(null_deltas.values())),
            "n_cues": len(cues),
            "resolved_cues": len(resolved),
            "drop_ids": sorted(drop_ids),
            "drop_cue_ids": sorted(drop_cue_ids),
            "intact_a_ranks": intact_a_ranks,
            "degraded_ranks": degraded_ranks,
            "uninvolved_gained": uninvolved_gained,
            "uninvolved_lost": uninvolved_lost,
            "driver": driver or os.environ.get("LILLI_STORAGE_DRIVER", "stdlib"),
        }


def partition_cues_by_involvement(
    cue_ids: "Iterable[str]", drop_cue_ids: "Iterable[str]"
) -> "tuple[list[str], list[str]]":
    """Partition cue_ids into (affected, uninvolved) by whether the cue's
    own gold was among the ones a degradation chose to drop. affected is
    exactly the cue_ids present in drop_cue_ids (in cue_ids order);
    uninvolved is every other cue_id -- the partition a no-displacement
    control measures over.
    """
    drop_set = set(drop_cue_ids)
    affected = [cid for cid in cue_ids if cid in drop_set]
    uninvolved = [cid for cid in cue_ids if cid not in drop_set]
    return affected, uninvolved


def no_displacement_value_metric(
    real_deltas: "dict[str, int]", uninvolved_cue_ids: "Iterable[str]"
) -> float:
    """value_metric restricted to the uninvolved partition -- the
    no-displacement control's own metric arm. Reuses the exact same
    real_deltas vector and value_metric reducer the decisive rung uses;
    this is a restriction to a sub-population, never a second metric.
    An untouched gold's own rank is expected to stay within measurement
    noise of the A/A floor -- a statistical claim verified against that
    floor, not a structural guarantee: the twin-build's graph rebuild
    recomputes community/centrality assignment globally on the smaller
    corpus, so it can in principle move an unrelated cue's rank in either
    direction, not just remove a local competitor. Callers should restrict
    uninvolved_cue_ids to cues that already resolve their own gold in the
    intact_a baseline -- a cue with no established rank there has no "own
    rank" for this control's claim to be about, and folding it in mixes an
    unrelated effect (a previously-unfindable gold surfacing once
    competing records shrink the corpus) into what is meant to be a
    displacement measurement.
    """
    return value_metric(real_deltas[cid] for cid in uninvolved_cue_ids)


# Functional (code-free) statement of the two measured self-tuning-lever
# findings -- the honest invariant reading on this system's own
# self-tuning mechanisms: neither one produces an observable cross-session
# recall improvement on a controlled corpus.
SELF_TUNING_FINDINGS: str = (
    "Both measured self-tuning levers were evaluated on a controlled corpus and neither "
    "produces above-floor cross-session rank improvement: the retrieval-weight lever is "
    "dominated by exact-cosine authority merge plus term overlap, so a single realistic "
    "tuning step is indistinguishable from its own measurement floor; the co-retrieval "
    "reinforcement lever is a self-loop that only entrenches whatever a dispatch already "
    "returned, so a gold outside the returned set is never promoted toward a better rank."
)


def replay_outcome_from_slice(slice_result: "dict | None") -> dict:
    """Classify one replay's tri-state outcome from an already-computed
    chain-integrity slice result, or None when the replay produced no
    result at all (a store/corpus build failure). available=True only when
    a verdict can be computed without hitting a degenerate floor; passed is
    then (verdict == "proceed"). A degenerate/collapsed floor (an
    intentionally unmeasurable corpus) also resolves to unavailable, never
    a pass or fail. classify_session_outcome is called exactly once,
    unconditionally on the resolved (available, passed) pair -- never
    wrapped in a broad except that could swallow a replay into neither
    pass, fail, nor excluded.
    """
    verdict: "dict | None" = None
    if slice_result is None:
        available, passed = False, None
    else:
        try:
            verdict = chain_integrity_verdict(
                value_metric=slice_result["value_metric"],
                aa_floor=slice_result["aa_floor"],
                established_cues=slice_result["resolved_cues"],
            )
            available, passed = True, verdict["verdict"] == "proceed"
        except DegenerateFloorError:
            available, passed = False, None
    outcome = classify_session_outcome(available=available, passed=passed)
    return {"available": available, "passed": passed, "verdict": verdict, "outcome": outcome}


def build_harness_report(slice_results: "list[dict | None]") -> dict:
    """Classify every replay's slice_result through
    replay_outcome_from_slice/classify_session_outcome, then assemble the
    measurable summary, its rendered string, and the named self-tuning
    negative-findings field. Every replay resolves through this path -- no
    replay may vanish from the denominator silently.
    """
    replays = [replay_outcome_from_slice(sr) for sr in slice_results]
    summary = summarize_measurable(r["outcome"] for r in replays)
    return {
        "replays": replays,
        "measurable_summary": summary,
        "measurable_summary_rendered": format_measurable_summary(summary),
        "self_tuning_findings": SELF_TUNING_FINDINGS,
    }
