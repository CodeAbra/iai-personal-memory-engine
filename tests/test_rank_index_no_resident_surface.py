"""Guards the zero-new-plaintext lock on the resident Rust rank index.

The resident index holds no whole-corpus raw literal-surface text: the
surface arena/span and its reader are gone (`aaak_index` -- the owner-
accepted keyword class -- stays). T11 (trigram-jaccard>0.3, x2.0) is
decided over `records_cache` surfaces -- the same source v16 reads -- via
a call-scoped Rust helper (`trigram_t11_flags`) that never retains a
trigram representation past one call; T12 (whole-cue substring, x3.0)
stays a plain `fts_hits` membership test. Both reach the Rust scorer as
per-pool boolean flag arrays. Postings/BM25 are re-sourced from a per-slot
token representation, never a resident surface. The scalar-upsert dedup
uses a >=128-bit content hash.
"""
from __future__ import annotations

import inspect
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp import pipeline
from iai_mcp.pipeline import SimpleRecordView, _t11_t12_flags, _trigram_jaccard
from iai_mcp.types import EMBED_DIM
from tests.test_recall_core_unit import _FakeEmbedder, _build_store_and_graph, _flat_assignment

_DIM = 4


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


def _build_index(surfaces: "list[str]", aaak: "list[str] | None" = None):
    """A minimal `RankIndex` straight from parallel columns -- mirrors
    `_rank_index.py::_build()`'s constructor call, never a store/graph
    round trip. Returns `(index, ids)`."""
    from iai_mcp_native import rank as _rank_native

    n = len(surfaces)
    aaak = aaak or [""] * n
    ids = list(range(1, n + 1))
    matrix = np.zeros((n, _DIM), dtype=np.float32)
    idx = _rank_native.RankIndex(
        _DIM, 1, ids, matrix, [], surfaces, aaak,
        ["2024-01-01T00:00:00+00:00"] * n, [0.5] * n, ["episodic"] * n,
        [[] for _ in range(n)], [0] * n, [0.0] * n, [False] * n,
    )
    return idx, ids


# ---------------------------------------------------------------------
# Resident-no-surface guard: mechanically proven, not narrated.
# ---------------------------------------------------------------------

def test_no_resident_raw_surface_column():
    surfaces = [
        "a distinct non-empty literal surface sentence about sourdough starters",
        "another distinct surface sentence entirely about something else",
    ]
    aaak = ["E:alice/T:capture", "E:bob/T:doc:notes.md"]
    idx, _ids = _build_index(surfaces, aaak)

    # No PyO3 surface reader exists at all on the object's API -- the whole
    # class of "read resident raw surface" access is absent, not merely
    # unused.
    assert not hasattr(idx, "surface_text")
    assert not hasattr(idx, "surface_span")

    expected_aaak_bytes = sum(len(s.encode("utf-8")) for s in aaak)
    surface_bytes_would_have_been = sum(len(s.encode("utf-8")) for s in surfaces)
    assert expected_aaak_bytes != surface_bytes_would_have_been, (
        "fixture self-check: aaak and surface byte totals must differ, or "
        "this test cannot distinguish 'arena holds aaak only' from "
        "'arena holds aaak+surface'"
    )
    assert idx.resident_text_arena_len() == expected_aaak_bytes, (
        "the resident text arena must account for aaak bytes ONLY -- a "
        f"total of {idx.resident_text_arena_len()} does not match the "
        f"aaak-only expectation of {expected_aaak_bytes} (surface-only "
        f"total would have been {surface_bytes_would_have_been})"
    )


def test_no_resident_raw_surface_column_with_empty_aaak_is_zero():
    # A corpus with real, non-empty surfaces but EMPTY aaak_index must
    # leave the arena at zero bytes -- the strongest possible negative
    # proof that surface content never reaches it.
    surfaces = ["surface one has real content", "surface two also has real content"]
    idx, _ids = _build_index(surfaces, aaak=["", ""])
    assert idx.resident_text_arena_len() == 0


# ---------------------------------------------------------------------
# Token-hashing residency guard: the resident per-slot
# token-frequency map must be keyed by an irreversible hashed feature id,
# never a readable word string. An accessor-absence check alone passes
# trivially against a still-plaintext structure (no accessor ever read one
# back even before this change) -- the footprint check below is the actual
# non-vacuous control: it fails for a String-keyed representation and only
# passes once the resident structure is genuinely bounded per entry
# regardless of word length.
# ---------------------------------------------------------------------

def test_no_resident_token_strings_footprint_bounded_not_word_length_scaled():
    long_rare_tokens = [
        "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxrarewordone",
        "yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyrarewordtwo",
        "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzrarewordthree",
    ]
    surfaces = [" ".join(long_rare_tokens)]
    idx, _ids = _build_index(surfaces)

    # No accessor reads back a resident raw token string -- the whole class
    # of "read resident token text" access is absent, not merely unused.
    assert not hasattr(idx, "token_text")
    assert not hasattr(idx, "token_span")
    assert not hasattr(idx, "resident_token_strings")

    would_have_been_string_bytes = sum(len(t.encode("utf-8")) for t in long_rare_tokens)
    assert would_have_been_string_bytes > 100, (
        "fixture self-check: the engineered tokens must be long enough that "
        "a String-keyed resident representation's byte total clearly "
        "exceeds a fixed-width hashed-feature footprint, or this test "
        "cannot distinguish the two representations"
    )

    footprint = idx.resident_token_footprint_bytes()
    assert footprint < would_have_been_string_bytes, (
        "resident token footprint must be bounded by a fixed per-entry "
        f"width, not the raw word bytes -- {footprint} is not below the "
        f"would-have-been String total of {would_have_been_string_bytes}; "
        "a String-keyed token_freqs would exceed this bound for these "
        "engineered long rare tokens"
    )


def test_build_decrypt_fires_at_most_once_per_store_lifetime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from iai_mcp.store._rank_index import rank_index_for

    store, graph, _recs = _build_store_and_graph(tmp_path, n=6)
    handle = rank_index_for(store, graph)

    calls = {"n": 0}
    orig_build = handle._build

    def _spy_build(g):
        calls["n"] += 1
        orig_build(g)

    monkeypatch.setattr(handle, "_build", _spy_build)

    for _ in range(4):
        handle.snapshot(graph, [])

    assert calls["n"] == 1, (
        "the whole-corpus decrypt build must fire at most once per store "
        f"lifetime for an unchanged graph identity -- got {calls['n']} calls; "
        "a per-recall re-trigger would re-pay the whole-corpus decrypt cost "
        "on every recall instead of once"
    )


def test_lexical_quality_preserved_under_hashing_ordering_not_byte_identity():
    """The bar for token hashing is lexical-recall QUALITY non-regression,
    not a byte-identical BM25 float -- feature-hashing collisions are
    permitted at standard rates. This asserts ordering survives
    hashing rather than pinning an exact score."""
    idx, ids = _build_index(
        ["a marimba concert last night", "completely different subject entirely"],
    )
    cosine = np.array([0.5, 0.5], dtype=np.float32)
    t11 = np.array([False, False])
    t12 = np.array([False, False])
    winners, _coverage, _damp = idx.score(
        ids, cosine, np.array([0, 1], dtype=np.int64),
        np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64),
        t11, t12, False, "xylophone_quartz_marimba", 0, 0.0, 1.0,
        set(), {}, 0.0, 0.0, {}, {}, 0.0, 0.0, 0.02, 0.0, None, True, 0.0, 1.0, 100, 0,
    )
    by_id = {w[0]: w[1] for w in winners}
    assert by_id[ids[0]] > by_id[ids[1]], (
        "the record sharing a lexical token with the cue must still outrank "
        "the record with no shared token after hashing -- collisions are "
        "permitted, but the ordering they exist to preserve must not break"
    )


# ---------------------------------------------------------------------
# Reachable set stays a subset of records_cache coverage.
# ---------------------------------------------------------------------

def _spy_t11_t12_flags(monkeypatch: pytest.MonkeyPatch, captured: dict):
    orig = pipeline._t11_t12_flags

    def spy(pool_ids, reachable_indices, records_cache, fts_hits, cue):
        captured["pool_ids"] = list(pool_ids)
        captured["reachable_indices"] = [int(i) for i in reachable_indices]
        captured["records_cache"] = records_cache
        captured["records_cache_keys"] = set(records_cache.keys())
        captured["fts_hits"] = set(fts_hits)
        captured["cue"] = cue
        return orig(pool_ids, reachable_indices, records_cache, fts_hits, cue)

    monkeypatch.setattr(pipeline, "_t11_t12_flags", spy)


def test_reachable_covered_by_records_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store, graph, recs = _build_store_and_graph(tmp_path, n=12)
    embedder = _FakeEmbedder()
    captured: dict = {}
    _spy_t11_t12_flags(monkeypatch, captured)

    result = pipeline._recall_core(
        store=store, graph=graph, assignment=_flat_assignment(recs),
        rich_club=[], embedder=embedder,
        cue="rec0", session_id="s-cov", use_rust_scorer=True,
    )
    assert result is not None
    assert "reachable_indices" in captured, "the Rust scorer path must have run"

    reachable_ids = {captured["pool_ids"][i] for i in captured["reachable_indices"]}
    assert reachable_ids.issubset(captured["records_cache_keys"]), (
        "on a representative recall, every reachable pool_id must have a "
        "records_cache entry -- a scored pool_id with no entry gets a "
        "silently-false T11/T12 flag"
    )

    # Full-pool coverage, proven structurally: after pool_ids is collected,
    # a single gap-backfill pass scans pool_ids for any id records_cache
    # still lacks and writes records_cache[rid] = rec for every one of them
    # in the SAME statement group -- source-pinned so a future edit that
    # drops the write reopens a silent T11/T12-flag gap. This scan covers
    # the whole pool unconditionally (the multi-seed widen block this pin
    # previously anchored to was deleted with the escalation mechanism;
    # there is no separate widen path left to distinguish).
    src = inspect.getsource(pipeline._recall_core)
    gap_scan_anchor = "rid for rid in pool_ids if rid not in records_cache"
    assert gap_scan_anchor in src, "pool-gap coverage scan not found -- scan pattern went stale"
    write_anchor = "records_cache[rid] = rec"
    assert write_anchor in src
    # The write must be inside the SAME backfill block as the gap scan.
    idx_gap_scan = src.find(gap_scan_anchor)
    # Search AFTER the gap scan: `write_anchor` also appears earlier, inside
    # a documentation comment describing this exact invariant -- the code
    # occurrence, not the comment, is what this pin must anchor on.
    idx_write = src.find(write_anchor, idx_gap_scan)
    assert idx_gap_scan != -1 and idx_write != -1
    assert 0 < idx_write - idx_gap_scan < 800, (
        "the records_cache[rid] = rec write must sit close to (in the same "
        "backfill block as) the pool-gap coverage scan -- if these drift "
        "apart, the scan could enumerate gaps without actually closing "
        "them, reopening the reachable-vs-records_cache coverage gap"
    )


def test_reachable_covered_under_store_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """STOP-gate finding (escalated, not silently closed): when the
    graph-sourced `records_cache` build fails, the code falls back to an
    UNRELATED bounded store slice (pipeline.py's `if not _records_base:`
    branch) while `pool_ids` still comes from `graph.iter_nodes()` via a
    SEPARATE, unaffected `_collect_graph_pool` call. `reachable` can then
    contain pool_ids absent from `records_cache`. This is real and
    reproducible, not hypothetical -- but it is BYTE-IDENTICAL to v16, not
    a new gap: v16's own scoring loop drops any candidate absent from
    records_cache entirely (`rec = records_cache.get(cid); if rec is None:
    continue`, pipeline.py), so a False T11/T12 flag for such an id changes
    nothing observable -- the candidate is unservable either way. See
    HANDOFF in the plan's SUMMARY for the escalation note."""
    store, graph, recs = _build_store_and_graph(tmp_path, n=12)
    embedder = _FakeEmbedder()

    # An extra graph node with an embedding but NO store-backed record and
    # NO "surface" payload key: it reaches `pool_ids` (`_collect_graph_pool`
    # admits any node with a resolvable embedding) but can never reach
    # `records_cache` under the store-fallback branch (built purely from
    # `store.iter_records`) OR the normal branch (requires "surface" in the
    # payload) -- guaranteeing a real, reproducible divergence regardless
    # of corpus scale, not one that happens to coincide away on a small
    # fixture.
    extra_id = uuid4()
    from iai_mcp.types import EMBED_DIM
    extra_vec = [0.0] * EMBED_DIM
    extra_vec[1] = 1.0
    graph.add_node(extra_id, community_id=None, embedding=extra_vec)

    captured: dict = {}
    _spy_t11_t12_flags(monkeypatch, captured)

    def _raise(*_a, **_kw):
        raise ValueError("forced payload-build failure")

    monkeypatch.setattr(graph, "get_payload", _raise)

    result = pipeline._recall_core(
        store=store, graph=graph, assignment=_flat_assignment(recs),
        rich_club=[], embedder=embedder,
        cue="rec0", session_id="s-fallback", use_rust_scorer=True,
    )
    assert result is not None
    assert "reachable_indices" in captured

    assert extra_id in captured["pool_ids"], (
        "fixture self-check: the extra store-absent node must reach "
        "pool_ids via _collect_graph_pool's embedding-only admission"
    )
    assert extra_id not in captured["records_cache_keys"], (
        "fixture self-check: the extra store-absent node must NEVER reach "
        "records_cache under the forced store-fallback branch -- if this "
        "assertion fails, the fixture is not exercising the intended "
        "divergence and the STOP-gate finding below is unverified"
    )

    reachable_ids = {captured["pool_ids"][i] for i in captured["reachable_indices"]}
    if extra_id in reachable_ids:
        # The uncovered id must resolve to False for both flags -- never
        # raise, never silently score against stale/mismatched data. This
        # IS byte-identical to v16: v16's own scoring loop
        # (`rec = records_cache.get(cid); if rec is None: continue`) drops
        # any candidate absent from records_cache entirely, so a False
        # flag here changes nothing observable -- the candidate is
        # unservable either way.
        t11, t12 = _t11_t12_flags(
            captured["pool_ids"],
            np.array(captured["reachable_indices"], dtype=np.int64),
            captured["records_cache"],
            captured["fts_hits"], captured["cue"],
        )
        extra_pos = captured["pool_ids"].index(extra_id)
        assert not bool(t11[extra_pos])
        assert not bool(t12[extra_pos])


# ---------------------------------------------------------------------
# Verbatim-mode tier-source divergence: a pool_id PRESENT in records_cache
# whose tier disagrees with Rust's own resident tier column must still get
# its T11/T12 flag populated. `_t11_t12_flags` must be called with the
# PRE-verbatim-filter reachable union, not the narrower one Python's own
# `episodic_ids` filter produces -- Rust re-derives and re-filters its own
# `reachable` independently, so a flag scoped to the narrower Python-side
# set can silently read as False for a position Rust still scores.
# ---------------------------------------------------------------------

def test_verbatim_flag_survives_prefilter_union_not_postfilter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    store, graph, recs = _build_store_and_graph(tmp_path, n=6, semantic_indices=[5])
    target = recs[5]
    assert target.tier == "semantic", "fixture self-check: target must be non-episodic"

    # A cue vector pinned exactly to record 5's one-hot embedding puts it at
    # the top of the cosine frontier (and thus in the pre-filter reachable
    # union) regardless of what text the cue string carries.
    target_vec = [0.0] * EMBED_DIM
    target_vec[5 % EMBED_DIM] = 1.0
    embedder = _FakeEmbedder(vec=target_vec)
    cue = "rec5"  # exact literal-surface match for record 5 only (>=4 chars)

    debug: dict = {}
    monkeypatch.setattr(pipeline, "_VERBATIM_FILTER_DEBUG", debug)
    captured: dict = {}
    _spy_t11_t12_flags(monkeypatch, captured)

    result = pipeline._recall_core(
        store=store, graph=graph, assignment=_flat_assignment(recs),
        rich_club=[], embedder=embedder,
        cue=cue, session_id="s-verbatim-divergence", use_rust_scorer=True,
        mode="verbatim",
    )
    assert result is not None
    assert "reachable_indices" in captured, "the Rust scorer path must have run"

    # Fixture self-check, independent of the fix: record 5 must land in the
    # PRE-filter union (proving it is a candidate Rust's own independently-
    # tiered reachable set can retain) while the verbatim/episodic narrowing
    # excludes it from the POST-filter union (proving a real divergence, not
    # a vacuous fixture).
    assert target.id in debug.get("pre_filter_reachable_ids", []), (
        "fixture self-check: record 5 must reach the pre-filter reachable "
        "union via the cosine frontier"
    )
    assert target.id not in debug.get("post_filter_reachable_ids", []), (
        "fixture self-check: a non-episodic-in-records_cache record must be "
        "dropped by Python's own verbatim/episodic narrowing, or this "
        "fixture exercises no divergence at all"
    )

    target_pos = captured["pool_ids"].index(target.id)
    t11, t12 = _t11_t12_flags(
        captured["pool_ids"],
        np.array(captured["reachable_indices"], dtype=np.int64),
        captured["records_cache"],
        captured["fts_hits"], captured["cue"],
    )
    assert bool(t11[target_pos]) or bool(t12[target_pos]), (
        "T11/T12 must fire for a candidate that survives the PRE-filter "
        "reachable union even though Python's own verbatim/episodic "
        "narrowing would have dropped it -- a False here means "
        "`_t11_t12_flags` was scoped to the narrower post-verbatim-filter "
        "set, silently dropping the x2.0/x3.0 boost (MEDIUM-01)"
    )


# ---------------------------------------------------------------------
# T11/T12 flags match the unchanged functions, byte-identical.
# ---------------------------------------------------------------------

def test_t11_t12_flags_match_unchanged_functions():
    r1, r2, r3 = uuid4(), uuid4(), uuid4()
    records_cache = {
        r1: _rec_view(r1, "the quick brown fox jumps over the lazy dog"),
        r2: _rec_view(r2, "completely unrelated content about apples and oranges"),
        r3: _rec_view(r3, "quick"),
    }
    pool_ids = [r1, r2, r3]
    reachable = np.array([0, 1, 2], dtype=np.int64)
    cue = "the quick brown fox"
    cue_lower = cue.lower()
    fts_hits = {
        rid for rid, rec in records_cache.items()
        if rec.literal_surface and cue_lower in rec.literal_surface.lower()
    }

    t11, t12 = _t11_t12_flags(pool_ids, reachable, records_cache, fts_hits, cue)

    for i, rid in enumerate(pool_ids):
        rec = records_cache[rid]
        expected_t11 = bool(
            cue and rec.literal_surface
            and _trigram_jaccard(cue_lower, rec.literal_surface.lower()) > 0.3
        )
        expected_t12 = rid in fts_hits
        assert bool(t11[i]) == expected_t11, f"T11 mismatch for {rid}"
        assert bool(t12[i]) == expected_t12, f"T12 mismatch for {rid}"
    # Non-vacuous: at least one true and one false per flag in this fixture.
    assert any(t11) and not all(t11)
    assert any(t12) and not all(t12)


def test_empty_surface_flag_parity():
    r1 = uuid4()
    records_cache = {r1: _rec_view(r1, "")}
    pool_ids = [r1]
    reachable = np.array([0], dtype=np.int64)
    cue = "any cue text at all"
    fts_hits: set = set()

    t11, t12 = _t11_t12_flags(pool_ids, reachable, records_cache, fts_hits, cue)
    assert not bool(t11[0]), "an empty surface must never fire T11"
    assert not bool(t12[0]), "an empty surface must never fire T12"


def test_verbatim_and_midword_fragment_flags_match():
    full = "Alice needs to feed her sourdough starter every twelve hours without fail"
    r_full, r_frag = uuid4(), uuid4()
    frag = full[: len(full) // 2]  # a partial-quote fragment, likely mid-word
    records_cache = {
        r_full: _rec_view(r_full, full),
        r_frag: _rec_view(r_frag, "an unrelated record the fragment cue must not match"),
    }
    pool_ids = [r_full, r_frag]
    reachable = np.array([0, 1], dtype=np.int64)

    for cue in (full, frag):
        cue_lower = cue.lower()
        fts_hits = {
            rid for rid, rec in records_cache.items()
            if rec.literal_surface and cue_lower in rec.literal_surface.lower()
        }
        t11, t12 = _t11_t12_flags(pool_ids, reachable, records_cache, fts_hits, cue)
        expected_t12_full = bool(cue and len(cue) >= 4 and cue_lower in full.lower())
        expected_t11_full = bool(
            cue and full and _trigram_jaccard(cue_lower, full.lower()) > 0.3
        )
        assert bool(t12[0]) == expected_t12_full, f"T12 mismatch on cue={cue!r}"
        assert bool(t11[0]) == expected_t11_full, f"T11 mismatch on cue={cue!r}"
        # Both a full verbatim quote-back and a half-length fragment quote
        # must fire T12 (exact substring) against their own source record.
        assert expected_t12_full is True, f"fixture must be a real substring for cue={cue!r}"


def test_t11_only_gate_flag_parity():
    # Trigram-jaccard fires (high character-ngram overlap) but the cue is
    # NOT a literal substring of the surface -- an adjacent transposition,
    # matching the Rust-side fixture shape. T11 and T12 are independent:
    # this must never reuse fts_hits for T11 (a conflated pair would make
    # this fixture's T11 silently equal T12's False).
    r1 = uuid4()
    records_cache = {r1: _rec_view(r1, "abcdfeghij")}
    pool_ids = [r1]
    reachable = np.array([0], dtype=np.int64)
    cue = "abcdefghij"
    cue_lower = cue.lower()
    fts_hits = {
        rid for rid, rec in records_cache.items()
        if rec.literal_surface and cue_lower in rec.literal_surface.lower()
    }
    assert not fts_hits, "fixture self-check: the cue must NOT be a literal substring"

    t11, t12 = _t11_t12_flags(pool_ids, reachable, records_cache, fts_hits, cue)
    assert bool(t11[0]) is True, "trigram similarity must fire T11 despite no substring match"
    assert bool(t12[0]) is False, "T12 must stay False when the cue is not a literal substring"


# ---------------------------------------------------------------------
# Postings/BM25 unchanged after surface removal.
# ---------------------------------------------------------------------

def test_postings_bm25_unchanged_after_surface_removal():
    idx, ids = _build_index(
        ["a marimba concert last night", "completely different subject entirely"],
    )
    cosine = np.array([0.5, 0.5], dtype=np.float32)
    t11 = np.array([False, False])
    t12 = np.array([False, False])
    winners, _coverage, _damp = idx.score(
        ids, cosine, np.array([0, 1], dtype=np.int64),
        np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64),
        t11, t12, False, "xylophone_quartz_marimba", 0, 0.0, 1.0,
        set(), {}, 0.0, 0.0, {}, {}, 0.0, 0.0, 0.02, 0.0, None, True, 0.0, 1.0, 100, 0,
    )
    by_id = {w[0]: w[1] for w in winners}
    # rank 0 (only shared-token match, "marimba") -> stability lift (0.05)
    # + lex_fusion_w / (1 + 0) == 1.0, on top of base cosine 0.5.
    assert abs(by_id[ids[0]] - 1.55) < 1e-9, by_id[ids[0]]
    assert abs(by_id[ids[1]] - 0.55) < 1e-9, by_id[ids[1]]


def test_postings_bm25_reflects_upserted_surface_not_stale_bulk_build():
    idx, ids = _build_index(["initial content with no shared token"])
    idx.feed(
        "upsert", ids[0], vector=np.zeros(_DIM, dtype=np.float32),
        surface="now mentions marimba explicitly",
    )
    idx.snapshot(2, [])  # drain the queued feed into the published buffer
    cosine = np.array([0.5], dtype=np.float32)
    t11 = np.array([False])
    t12 = np.array([False])
    winners, _coverage, _damp = idx.score(
        ids, cosine, np.array([0], dtype=np.int64),
        np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64),
        t11, t12, False, "xylophone_quartz_marimba", 0, 0.0, 1.0,
        set(), {}, 0.0, 0.0, {}, {}, 0.0, 0.0, 0.02, 0.0, None, True, 0.0, 1.0, 100, 0,
    )
    assert abs(winners[0][1] - 1.55) < 1e-9, (
        "postings re-sourced from the token representation must reflect "
        "the upserted surface, not the stale bulk-build one"
    )


# ---------------------------------------------------------------------
# Content-hash dedup, behaviorally proven via the public FFI.
# ---------------------------------------------------------------------

def test_dedup_content_hash_no_collision():
    idx, ids = _build_index(["original content alpha unique token alphaword"])
    _gen = [1]

    def _postings(token: str) -> dict:
        _gen[0] += 1
        return idx.snapshot(_gen[0], [token])[4].get(token, {})

    assert _postings("alphaword") == {ids[0]: 1}
    assert _postings("betaword") == {}

    # Genuinely different content: the dedup check must NOT wrongly treat
    # this as unchanged (a hash collision would leave "alphaword" stale and
    # miss "betaword" entirely).
    idx.feed(
        "upsert", ids[0], vector=np.zeros(_DIM, dtype=np.float32),
        surface="completely different content beta unique token betaword",
    )
    assert _postings("alphaword") == {}, (
        "a stale posting for the OLD surface's token means the dedup check "
        "wrongly skipped the recompute -- the staleness bug a hash "
        "collision would cause"
    )
    assert _postings("betaword") == {ids[0]: 1}

    # Byte-identical re-feed: must not corrupt or drop the current
    # (already-correct) token representation.
    idx.feed(
        "upsert", ids[0], vector=np.zeros(_DIM, dtype=np.float32),
        surface="completely different content beta unique token betaword",
    )
    assert _postings("betaword") == {ids[0]: 1}
    assert _postings("alphaword") == {}

    # The production hash itself: a Rust-level collision-freedom proof
    # (two engineered distinct surfaces do not collide under the >=128-bit
    # composite hash) plus a deliberately weak/truncated control that DOES
    # collide on an engineered pair (proving the methodology would catch a
    # real collision) live in `cargo test -p iai_mcp_rank_core` --
    # `content_hash128_does_not_collide_on_two_engineered_distinct_surfaces`
    # and `weak_truncated_hash_collides_but_the_production_hash_does_not_
    # on_the_same_pair` -- because only Rust has direct access to the
    # private hash function; this test proves the same guarantee's
    # OBSERVABLE consequence through the shipped public API.
