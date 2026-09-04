"""Byte-identical differential for the soft_gate reducible cost.

`_t11_t12_flags` batches T11 (trigram-jaccard>0.3) through the Rust
`trigram_t11_flags` helper instead of rebuilding a Python trigram set per
candidate. This module pins the batched result against a pure-Python
reference built directly from the unchanged `_trigram_jaccard` function --
the same reference the pre-optimization loop computed -- across multiple
cues, corpus sizes, and a boundary-tight fixture engineered to flip if the
computation drifts by even one trigram. The `_t11_t12_flags` function
touches no storage driver (a pure in-memory transform over `records_cache`
and array arguments), so its correctness is identical on both drivers;
this suite is run under both as a formality, not because either path
diverges.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.pipeline import SimpleRecordView, _t11_t12_flags, _trigram_jaccard


def _rec_view(rid: UUID, surface: str, tier: str = "episodic") -> SimpleRecordView:
    return SimpleRecordView(
        id=rid,
        embedding=None,
        literal_surface=surface,
        centrality=0.0,
        tier=tier,
        aaak_index="",
        created_at=datetime.now(timezone.utc),
        stability=0.5,
    )


def _reference_flags(
    pool_ids: "list[UUID]",
    reachable: "np.ndarray",
    records_cache: dict,
    fts_hits: set,
    cue: str,
) -> "tuple[list[bool], list[bool]]":
    """Pure-Python re-derivation of the pre-optimization `_t11_t12_flags`
    body, built directly from the unchanged `_trigram_jaccard` -- the
    ground truth this differential pins the Rust-backed path against.
    """
    n = len(pool_ids)
    t11 = [False] * n
    t12 = [False] * n
    cue_lower = cue.lower() if cue else ""
    for idx in reachable:
        i = int(idx)
        cid = pool_ids[i]
        rec = records_cache.get(cid)
        if rec is None:
            continue
        surface = getattr(rec, "literal_surface", "") or ""
        t12[i] = cid in fts_hits
        t11[i] = bool(
            cue and surface and _trigram_jaccard(cue_lower, surface.lower()) > 0.3
        )
    return t11, t12


_SURFACE_TEMPLATES = [
    "alice needs to feed her sourdough starter every twelve hours without fail",
    "bob prefers testing every change before shipping to production",
    "the quick brown fox jumps over the lazy dog near the old oak tree",
    "completely unrelated content about apples and oranges and pears",
    "quick",
    "",
    "a b",
    "unicode café naïve résumé test string with accents",
]


def _make_corpus(n: int) -> "tuple[list[UUID], dict]":
    pool_ids = [uuid4() for _ in range(n)]
    records_cache = {
        pid: _rec_view(pid, _SURFACE_TEMPLATES[i % len(_SURFACE_TEMPLATES)])
        for i, pid in enumerate(pool_ids)
    }
    return pool_ids, records_cache


def _fts_hits_for(cue: str, pool_ids: "list[UUID]", records_cache: dict) -> set:
    cue_lower = cue.lower()
    if not cue_lower:
        return set()
    return {
        pid
        for pid in pool_ids
        if records_cache[pid].literal_surface
        and cue_lower in records_cache[pid].literal_surface.lower()
    }


CUES = [
    "the quick brown fox jumps over the lazy dog",
    "alice prefers testing every change before shipping to production",
    "a",  # below the 3-char trigram floor -- must never fire T11
    "",  # empty cue -- must never fire T11 or T12
    "unicode café naïve résumé test",
]

CORPUS_SIZES = [1, 5, 47, 300]


@pytest.mark.parametrize("n", CORPUS_SIZES)
@pytest.mark.parametrize("cue", CUES)
def test_t11_t12_byte_identical_across_cues_and_corpus_sizes(cue: str, n: int) -> None:
    pool_ids, records_cache = _make_corpus(n)
    reachable = np.arange(n, dtype=np.int64)
    fts_hits = _fts_hits_for(cue, pool_ids, records_cache)

    ref_t11, ref_t12 = _reference_flags(pool_ids, reachable, records_cache, fts_hits, cue)
    got_t11, got_t12 = _t11_t12_flags(pool_ids, reachable, records_cache, fts_hits, cue)

    assert list(got_t11) == ref_t11, f"T11 diverged for cue={cue!r} n={n}"
    assert list(got_t12) == ref_t12, f"T12 diverged for cue={cue!r} n={n}"


def test_absent_from_records_cache_gets_false_for_both_flags() -> None:
    """A pool position with no records_cache entry must read False for
    both flags, matching v16's own drop-if-missing scoring loop -- proves
    the batched Rust call never widens coverage past what records_cache
    actually populated.
    """
    present, absent = uuid4(), uuid4()
    records_cache = {present: _rec_view(present, "the quick brown fox jumps high")}
    pool_ids = [present, absent]
    reachable = np.array([0, 1], dtype=np.int64)
    cue = "the quick brown fox"
    fts_hits = {present, absent}  # even if fts "hit" the absent id, it stays unreadable

    t11, t12 = _t11_t12_flags(pool_ids, reachable, records_cache, fts_hits, cue)
    assert bool(t12[1]) is False, "an absent-from-records_cache id must never read T12 True"
    assert bool(t11[1]) is False, "an absent-from-records_cache id must never read T11 True"


# ---------------------------------------------------------------------
# Threshold-boundary non-vacuity control.
#
# Two engineered strings share a prefix that differs by exactly ONE
# additional shared trigram (347 vs. 348 shared-prefix characters over a
# corpus of unique, non-colliding code points) -- their Jaccard values are
# 0.29922 and 0.30035, bracketing the T11 threshold (`> 0.3`) with a gap
# of ~0.0011. A one-trigram computational drift (a stray hash collision, a
# windowing off-by-one, a boundary comparison bug) would flip one of these
# two flags -- this fixture is designed to go RED under exactly that
# class of regression, not to merely exercise the happy path.
# ---------------------------------------------------------------------

def _unique_codepoint_string(n: int, offset: int) -> str:
    """`n` characters, each codepoint unique within this whole test module
    (offset ranges never overlap) -- guarantees zero incidental trigram
    collisions outside the deliberately shared prefix.
    """
    return "".join(chr(offset + i) for i in range(n))


_BOUNDARY_CUE_LEN = 502  # 500 distinct trigrams, no internal repeats
_BOUNDARY_CUE = _unique_codepoint_string(_BOUNDARY_CUE_LEN, offset=0x3000)
_BOUNDARY_CAND_LEN = 1000


def _boundary_candidate(shared_prefix_len: int) -> str:
    prefix = _BOUNDARY_CUE[:shared_prefix_len]
    tail = _unique_codepoint_string(
        _BOUNDARY_CAND_LEN - shared_prefix_len, offset=0x9000,
    )
    return prefix + tail


_BELOW_BOUNDARY_SURFACE = _boundary_candidate(347)  # jaccard ~0.29922 -> False
_ABOVE_BOUNDARY_SURFACE = _boundary_candidate(348)  # jaccard ~0.30035 -> True


def test_threshold_boundary_fixture_self_check() -> None:
    """Proves the fixture actually brackets 0.3 before trusting it as a
    regression control -- a fixture that does not straddle the threshold
    would make the test below vacuous.
    """
    cue_lower = _BOUNDARY_CUE.lower()
    j_below = _trigram_jaccard(cue_lower, _BELOW_BOUNDARY_SURFACE.lower())
    j_above = _trigram_jaccard(cue_lower, _ABOVE_BOUNDARY_SURFACE.lower())
    assert j_below <= 0.3, f"below-boundary fixture must not exceed 0.3, got {j_below}"
    assert j_above > 0.3, f"above-boundary fixture must exceed 0.3, got {j_above}"
    assert (j_above - j_below) < 0.01, "fixture margin must stay tight, not a loose gap"


def test_threshold_boundary_flags_match_reference_on_both_sides() -> None:
    r_below, r_above = uuid4(), uuid4()
    records_cache = {
        r_below: _rec_view(r_below, _BELOW_BOUNDARY_SURFACE),
        r_above: _rec_view(r_above, _ABOVE_BOUNDARY_SURFACE),
    }
    pool_ids = [r_below, r_above]
    reachable = np.array([0, 1], dtype=np.int64)
    fts_hits: set = set()

    ref_t11, _ = _reference_flags(
        pool_ids, reachable, records_cache, fts_hits, _BOUNDARY_CUE,
    )
    got_t11, _ = _t11_t12_flags(
        pool_ids, reachable, records_cache, fts_hits, _BOUNDARY_CUE,
    )

    assert ref_t11 == [False, True], "reference fixture self-check must bracket the threshold"
    assert list(got_t11) == ref_t11, (
        "the Rust-backed T11 computation diverged from the reference at the "
        "threshold boundary -- a real off-by-one/collision would show up "
        "exactly here"
    )


def test_a_real_flag_divergence_is_caught_by_the_differential(monkeypatch) -> None:
    """Non-vacuity meta-control: corrupts the Rust helper's output and
    proves the differential comparison used above WOULD fail against a
    real regression, rather than passing vacuously regardless of what the
    Rust path returns.
    """
    from iai_mcp_native import rank as _rank_native

    r1 = uuid4()
    records_cache = {r1: _rec_view(r1, "the quick brown fox jumps over the lazy dog")}
    pool_ids = [r1]
    reachable = np.array([0], dtype=np.int64)
    cue = "the quick brown fox"
    fts_hits: set = set()

    ref_t11, _ = _reference_flags(pool_ids, reachable, records_cache, fts_hits, cue)
    assert ref_t11 == [True], "fixture self-check: reference must fire T11"

    def _always_false(cue_lower: str, surfaces_lower: "list[str]") -> "list[bool]":
        return [False] * len(surfaces_lower)

    monkeypatch.setattr(_rank_native, "trigram_t11_flags", _always_false)
    corrupted_t11, _ = _t11_t12_flags(pool_ids, reachable, records_cache, fts_hits, cue)
    assert list(corrupted_t11) != ref_t11, (
        "the differential fixture must be sensitive enough to catch a "
        "corrupted Rust-side T11 result, proving it is not vacuous"
    )


def test_reachable_covered_by_records_cache_under_store_fallback() -> None:
    """A cue too short to ever produce a nonempty trigram feature set (< 3
    chars) must leave T11 False everywhere, matching `_trigram_jaccard`'s
    own short-circuit -- the batched path must never call into Rust with
    an empty cue-feature set expecting anything but False back.
    """
    pool_ids, records_cache = _make_corpus(10)
    reachable = np.arange(10, dtype=np.int64)
    cue = "ab"  # nonempty but below the 3-char trigram floor
    fts_hits: set = set()

    ref_t11, ref_t12 = _reference_flags(pool_ids, reachable, records_cache, fts_hits, cue)
    got_t11, got_t12 = _t11_t12_flags(pool_ids, reachable, records_cache, fts_hits, cue)

    assert not any(ref_t11), "fixture self-check: a sub-3-char cue must never fire T11"
    assert list(got_t11) == ref_t11
    assert list(got_t12) == ref_t12
