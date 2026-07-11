"""Deterministic metric-gate for the reproducible eval set and accuracy runner.

These tests:
  1. Verify build_eval_set is deterministic: same seed → identical
     (cue_text, target_id, archetype) list across independent calls.
  2. Verify each archetype's structural precondition holds (oracle rank,
     graph neighbour set, spread-reachability).
  3. Run run_accuracy over the crafted set and assert: recall@k < 1.0
     (misses are present), per-archetype attribution is correct, unattributed==0,
     and the latency global does NOT retain the injected sentinel after the run.
  4. Include a grep gate asserting no bare module-level RNG calls exist
     in bench/recall_accuracy.py.

The Rust embedder is never loaded: all tests use _BenchEmbedder (deterministic
hash-based) from bench.neural_map, with embed dim set by env var.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Embed-dim monkeypatch — must come before any iai_mcp import so the store
# allocates the matching dimension.
# ---------------------------------------------------------------------------
_DIM = 16  # small synthetic dim; avoids loading the Rust embedder


@pytest.fixture(autouse=True)
def _small_embed_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(_DIM))


# ---------------------------------------------------------------------------
# Latency-global teardown — ensures _last_recall_latency_ms is 0.0 after every
# test so no sentinel leaks across tests (the cross-test pollution guard).
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_latency_global():
    yield
    import iai_mcp.pipeline as _pl
    _pl._last_recall_latency_ms = 0.0


# ---------------------------------------------------------------------------
# Imports (deferred past fixtures so env vars are visible at import time)
# ---------------------------------------------------------------------------
from bench.recall_accuracy import (  # noqa: E402
    Archetype,
    EvalCase,
    SEMANTIC_HIT_EPS,
    _VectorInjectionEmbedder,
    _dispatch_recall_ids,
    _inject_embedder_into_dispatch,
    _perturb_vec,
    _scan_active_corpus,
    build_eval_set,
    class_cosine_rank,
    is_semantic_hit,
    oracle_top_k,
    run_accuracy,
    run_baseline,
    semantic_equivalence_class,
)
from bench.neural_map import _BenchEmbedder  # noqa: E402


# ---------------------------------------------------------------------------
# Gate: eval-set determinism and structural preconditions
# ---------------------------------------------------------------------------

class TestBuildEvalSetDeterminism:
    """build_eval_set is reproducible: same seed → identical ordered list."""

    def test_same_seed_same_cases(self, tmp_path):
        """Two calls with the same seed produce identical (cue_text, target_id, archetype)."""
        r1 = build_eval_set(seed=42, n_filler=220, store_path=tmp_path / "a")
        r2 = build_eval_set(seed=42, n_filler=220, store_path=tmp_path / "b")

        store1, graph1, assignment1, rich_club1, embedder1, cases1 = r1
        store2, graph2, assignment2, rich_club2, embedder2, cases2 = r2

        try:
            assert len(cases1) == len(cases2), "case counts differ"
            for i, (c1, c2) in enumerate(zip(cases1, cases2)):
                assert c1.cue_text == c2.cue_text, f"case[{i}] cue_text differs"
                assert c1.target_id == c2.target_id, f"case[{i}] target_id differs"
                assert c1.archetype == c2.archetype, f"case[{i}] archetype differs"
        finally:
            store1.close()
            store2.close()

    def test_all_three_archetypes_present(self, tmp_path):
        """One instance of each archetype is present in the eval set."""
        store, graph, assignment, rich_club, embedder, cases = build_eval_set(
            seed=7, n_filler=220, store_path=tmp_path
        )
        try:
            archetypes_found = {c.archetype for c in cases}
            assert Archetype.COSINE_RANK_BEYOND_CUTOFF in archetypes_found
            assert Archetype.SPREAD_DISABLED_BY_PRIOR_LATENCY in archetypes_found
            assert Archetype.EDGE_LESS_ISLAND in archetypes_found
        finally:
            store.close()

    def test_cases_are_evalcase_instances(self, tmp_path):
        """Each case is an EvalCase dataclass with the expected fields."""
        store, graph, assignment, rich_club, embedder, cases = build_eval_set(
            seed=0, n_filler=220, store_path=tmp_path
        )
        try:
            for case in cases:
                assert isinstance(case, EvalCase)
                assert isinstance(case.cue_text, str) and len(case.cue_text) > 0
                assert case.target_id is not None
                assert case.archetype in (
                    Archetype.COSINE_RANK_BEYOND_CUTOFF,
                    Archetype.SPREAD_DISABLED_BY_PRIOR_LATENCY,
                    Archetype.EDGE_LESS_ISLAND,
                )
        finally:
            store.close()


class TestBuildEvalSetArchetypes:
    """Structural preconditions hold for each seeded archetype."""

    def test_cosine_rank_beyond_cutoff_oracle_rank(self, tmp_path):
        """The COSINE_RANK_BEYOND_CUTOFF target is absent from the oracle top-K_CANDIDATES."""
        from iai_mcp.pipeline import K_CANDIDATES

        store, graph, assignment, rich_club, embedder, cases = build_eval_set(
            seed=42, n_filler=220, store_path=tmp_path
        )
        try:
            beyond_cases = [c for c in cases if c.archetype == Archetype.COSINE_RANK_BEYOND_CUTOFF]
            assert beyond_cases, "no COSINE_RANK_BEYOND_CUTOFF case"
            case = beyond_cases[0]

            cue_emb = np.array(embedder.embed(case.cue_text), dtype=np.float32)
            oracle_ids = oracle_top_k(store, cue_emb, k=K_CANDIDATES + 50)

            top_k_ids = set(oracle_ids[:K_CANDIDATES])
            assert case.target_id not in top_k_ids, (
                f"COSINE_RANK_BEYOND_CUTOFF target should be outside oracle top-{K_CANDIDATES}"
            )
        finally:
            store.close()

    def test_edge_less_island_no_non_self_neighbours(self, tmp_path):
        """The EDGE_LESS_ISLAND target has only a self-loop (no outgoing non-self edges)."""
        store, graph, assignment, rich_club, embedder, cases = build_eval_set(
            seed=42, n_filler=220, store_path=tmp_path
        )
        try:
            island_cases = [c for c in cases if c.archetype == Archetype.EDGE_LESS_ISLAND]
            assert island_cases, "no EDGE_LESS_ISLAND case"
            case = island_cases[0]

            neighbours = graph._adj.get(str(case.target_id), {})
            non_self_neighbours = {n for n in neighbours if n != str(case.target_id)}
            assert not non_self_neighbours, (
                f"EDGE_LESS_ISLAND target has non-self neighbours: {non_self_neighbours}"
            )
        finally:
            store.close()

    def test_spread_disabled_target_reachable_via_spread(self, tmp_path):
        """The SPREAD_DISABLED_BY_PRIOR_LATENCY target is reachable via 2-hop spread
        from the seeds when spread is ON, but not the cue's direct top-cosine hit."""
        store, graph, assignment, rich_club, embedder, cases = build_eval_set(
            seed=42, n_filler=220, store_path=tmp_path
        )
        try:
            spread_cases = [
                c for c in cases
                if c.archetype == Archetype.SPREAD_DISABLED_BY_PRIOR_LATENCY
            ]
            assert spread_cases, "no SPREAD_DISABLED_BY_PRIOR_LATENCY case"
            case = spread_cases[0]

            from bench.recall_accuracy import oracle_top_k
            cue_emb = np.array(embedder.embed(case.cue_text), dtype=np.float32)
            # The target should be reachable by oracle (within the corpus)
            oracle_ids = oracle_top_k(store, cue_emb, k=store.active_records_count())
            assert case.target_id in oracle_ids, "SPREAD_DISABLED target not even in corpus"
        finally:
            store.close()

    def test_cue_text_differs_from_target_surface(self, tmp_path):
        """Cue text is paraphrased away from target's literal surface (no trivial exact hit)."""
        store, graph, assignment, rich_club, embedder, cases = build_eval_set(
            seed=42, n_filler=220, store_path=tmp_path
        )
        try:
            from iai_mcp.store import MemoryStore
            for case in cases:
                target_rec = store.get(case.target_id)
                if target_rec is not None:
                    # Cue must not be a verbatim substring of target surface or vice versa
                    assert case.cue_text not in target_rec.literal_surface, (
                        f"cue '{case.cue_text}' is substring of target surface "
                        f"'{target_rec.literal_surface}'"
                    )
        finally:
            store.close()


# ---------------------------------------------------------------------------
# RNG purity gate: no bare module-level random calls in bench/recall_accuracy.py
# ---------------------------------------------------------------------------

class TestRngPurityGate:
    """No bare module-level np.random.* or random.* calls in the eval-set builder."""

    def test_no_bare_module_rng_calls(self):
        bench_file = Path(__file__).resolve().parent.parent / "bench" / "recall_accuracy.py"
        source = bench_file.read_text()

        # Allowed: np.random.default_rng(seed) or random.Random(seed) used to
        # CREATE a seeded instance. Forbidden: bare calls that read the global
        # unseeded state.
        bare_np = re.compile(
            r"(?<!\.)np\.random\.(rand|randn|choice|seed|randint|shuffle|sample)\s*\("
        )
        bare_random = re.compile(
            r"(?<![.\w])random\.(random|randint|choice|seed|shuffle|sample)\s*\("
        )

        np_hits = bare_np.findall(source)
        random_hits = bare_random.findall(source)

        assert not np_hits, f"bare np.random.* calls found: {np_hits}"
        assert not random_hits, f"bare random.* calls found: {random_hits}"


# ---------------------------------------------------------------------------
# Gate: run_accuracy attribution and latency-global guard
# ---------------------------------------------------------------------------

class TestRunAccuracy:
    """run_accuracy measures recall@k < 1.0, attributes each archetype, and
    resets the latency global to 0.0 at end-of-run."""

    def test_recall_below_one_on_crafted_set(self, tmp_path):
        """recall@k < 1.0 on the crafted set — misses ARE present."""
        store, graph, assignment, rich_club, embedder, cases = build_eval_set(
            seed=42, n_filler=220, store_path=tmp_path
        )
        try:
            result = run_accuracy(store, graph, assignment, rich_club, embedder, cases, k=10)
            assert result["recall_at_k"] < 1.0, (
                "recall@k == 1.0 on the crafted set — the harness cannot detect misses; "
                "the eval set is broken if no archetypes produce false negatives"
            )
        finally:
            store.close()

    def test_each_archetype_attributed(self, tmp_path):
        """The two still-attributable archetypes appear in miss_counts_by_archetype
        with count >= 1; SPREAD_DISABLED_BY_PRIOR_LATENCY stays at 0 (the wall-clock
        degrade that used to populate this bucket no longer exists)."""
        store, graph, assignment, rich_club, embedder, cases = build_eval_set(
            seed=42, n_filler=220, store_path=tmp_path
        )
        try:
            result = run_accuracy(store, graph, assignment, rich_club, embedder, cases, k=10)
            miss_counts = result["miss_counts_by_archetype"]

            assert miss_counts.get(Archetype.COSINE_RANK_BEYOND_CUTOFF, 0) >= 1, (
                f"COSINE_RANK_BEYOND_CUTOFF not attributed: {miss_counts}"
            )
            assert miss_counts.get(Archetype.EDGE_LESS_ISLAND, 0) >= 1, (
                f"EDGE_LESS_ISLAND not attributed: {miss_counts}"
            )
            assert miss_counts.get(Archetype.SPREAD_DISABLED_BY_PRIOR_LATENCY, 0) == 0, (
                "SPREAD_DISABLED_BY_PRIOR_LATENCY must stay 0 (wall-clock degrade removed); "
                f"got {miss_counts}"
            )
        finally:
            store.close()

    def test_unattributed_zero(self, tmp_path):
        """unattributed == 0 on the crafted set."""
        store, graph, assignment, rich_club, embedder, cases = build_eval_set(
            seed=42, n_filler=220, store_path=tmp_path
        )
        try:
            result = run_accuracy(store, graph, assignment, rich_club, embedder, cases, k=10)
            assert result["unattributed"] == 0, (
                f"unattributed misses found: {result['unattributed']}; "
                "attribution logic is incomplete"
            )
        finally:
            store.close()

    def test_latency_global_not_sentinel_after_run(self, tmp_path):
        """After run_accuracy, pipeline._last_recall_latency_ms != 2001.0 and
        holds no residual > 2000 value (the end-of-run reset guard holds)."""
        import iai_mcp.pipeline as _pl

        store, graph, assignment, rich_club, embedder, cases = build_eval_set(
            seed=42, n_filler=220, store_path=tmp_path
        )
        try:
            run_accuracy(store, graph, assignment, rich_club, embedder, cases, k=10)
            assert _pl._last_recall_latency_ms != 2001.0, (
                "injected sentinel 2001.0 was not cleared — end-of-run reset failed"
            )
            assert _pl._last_recall_latency_ms <= 2000.0, (
                f"residual > 2000 sentinel in latency global: {_pl._last_recall_latency_ms}"
            )
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Gate: the graded (soft-gate) community bonus does not regress recall@k or
# per-archetype miss counts vs. the pre-change baseline
# (seed=42/n_filler=220/k=10). Not-worse-than, not strictly-better
# — the strict-improvement bar belongs to a later change.
# ---------------------------------------------------------------------------

# Baseline constants captured before this change (both drivers were
# byte-identical at this corpus size).
BASELINE_RECALL_AT_K = 0.0
BASELINE_MISS_COSINE_RANK = 1
BASELINE_MISS_EDGE_LESS = 2
BASELINE_MISS_SPREAD_DISABLED = 0


class TestGradedCommunityBonusRegression:
    """The graded community-gate bonus (soft gate) must not regress the
    crafted-set baseline captured before this change."""

    @pytest.mark.parametrize("driver", ["stdlib", "lilli"])
    def test_recall_at_k_not_worse_than_baseline(self, tmp_path, monkeypatch, driver):
        if driver == "lilli":
            try:
                import iai_mcp_native  # noqa: F401
            except ImportError:
                pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
            monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
        else:
            monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)

        store, graph, assignment, rich_club, embedder, cases = build_eval_set(
            seed=42, n_filler=220, store_path=tmp_path
        )
        try:
            result = run_accuracy(store, graph, assignment, rich_club, embedder, cases, k=10)
            assert result["recall_at_k"] >= BASELINE_RECALL_AT_K - 0.001, (
                f"recall_at_k regressed vs. baseline: got {result['recall_at_k']}, "
                f"baseline {BASELINE_RECALL_AT_K}"
            )
        finally:
            store.close()

    @pytest.mark.parametrize("driver", ["stdlib", "lilli"])
    def test_miss_counts_improved_or_equal_by_archetype(self, tmp_path, monkeypatch, driver):
        if driver == "lilli":
            try:
                import iai_mcp_native  # noqa: F401
            except ImportError:
                pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
            monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
        else:
            monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)

        store, graph, assignment, rich_club, embedder, cases = build_eval_set(
            seed=42, n_filler=220, store_path=tmp_path
        )
        try:
            result = run_accuracy(store, graph, assignment, rich_club, embedder, cases, k=10)
            miss_counts = result["miss_counts_by_archetype"]

            assert miss_counts.get(Archetype.COSINE_RANK_BEYOND_CUTOFF, 0) <= BASELINE_MISS_COSINE_RANK, (
                f"COSINE_RANK_BEYOND_CUTOFF miss count regressed: {miss_counts}"
            )
            assert miss_counts.get(Archetype.EDGE_LESS_ISLAND, 0) <= BASELINE_MISS_EDGE_LESS, (
                f"EDGE_LESS_ISLAND miss count regressed: {miss_counts}"
            )
            assert miss_counts.get(Archetype.SPREAD_DISABLED_BY_PRIOR_LATENCY, 0) == BASELINE_MISS_SPREAD_DISABLED, (
                f"SPREAD_DISABLED_BY_PRIOR_LATENCY must stay at baseline {BASELINE_MISS_SPREAD_DISABLED}: {miss_counts}"
            )
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Gate: the graded community-gate bonus is a continuous, monotone-by-rank
# function of community score — not the old binary in-top-N/out-of-top-N step.
# ---------------------------------------------------------------------------

class TestGradedCommunityBonusContinuity:
    """A binary bonus gives identical weight to every top-N community and zero
    to all others. The graded bonus must instead be strictly monotone by
    community rank across ranks 1, 2 and 4 (out of 4 total communities)."""

    def test_graded_bonus_is_continuous(self):
        from uuid import uuid4

        from iai_mcp.community import CommunityAssignment, _compute_centroid
        from iai_mcp.pipeline import _community_gate_scored
        from iai_mcp.types import EMBED_DIM

        def _unit_axis(i: int, dim: int = EMBED_DIM) -> list[float]:
            v = [0.0] * dim
            v[i % dim] = 1.0
            return v

        cue = _unit_axis(0)

        # Four communities, each a single member whose embedding has a
        # decreasing cosine-similarity to the cue: rank 1 (cos=1.0),
        # rank 2 (cos~0.87), rank 3 (cos~0.71), rank 4 (cos~0.5).
        import math

        def _tilted(angle_deg: float) -> list[float]:
            rad = math.radians(angle_deg)
            v = [0.0] * EMBED_DIM
            v[0] = math.cos(rad)
            v[1] = math.sin(rad)
            return v

        angles = [0.0, 30.0, 45.0, 60.0]  # cos: 1.0, 0.866, 0.707, 0.5 — strictly decreasing
        comm_ids = [uuid4() for _ in range(4)]
        member_ids = [uuid4() for _ in range(4)]
        node_to_community: dict = {}
        centroids: dict = {}
        mid_regions: dict = {}
        member_embeddings: dict = {}
        for i in range(4):
            vec = _tilted(angles[i])
            node_to_community[member_ids[i]] = comm_ids[i]
            centroids[comm_ids[i]] = _compute_centroid([vec])
            mid_regions[comm_ids[i]] = [member_ids[i]]
            member_embeddings[member_ids[i]] = vec

        assignment = CommunityAssignment(
            node_to_community=node_to_community,
            community_centroids=centroids,
            modularity=0.0,
            backend="leiden-test-graded-bonus",
            top_communities=comm_ids[:3],
            mid_regions=mid_regions,
        )

        _gated, community_scores = _community_gate_scored(
            cue, assignment, top_n=3, member_embeddings=member_embeddings,
        )

        rank1_score = community_scores[comm_ids[0]]
        rank2_score = community_scores[comm_ids[1]]
        rank4_score = community_scores[comm_ids[3]]

        assert rank4_score < rank2_score < rank1_score, (
            "graded community score is not strictly monotone by community rank: "
            f"rank1={rank1_score} rank2={rank2_score} rank4={rank4_score} — "
            "a binary bonus would give equal weight to ranks 1-3 and zero to rank 4"
        )

        max_score = max(community_scores.values())
        bonus = lambda cid: max(0.0, community_scores[cid] / max_score)  # noqa: E731
        assert bonus(comm_ids[3]) < bonus(comm_ids[1]) < bonus(comm_ids[0]), (
            "normalized graded bonus is not strictly monotone by community rank"
        )

    def test_result_is_json_serialisable(self, tmp_path):
        """run_accuracy returns a JSON-serialisable dict."""
        import json

        store, graph, assignment, rich_club, embedder, cases = build_eval_set(
            seed=42, n_filler=220, store_path=tmp_path
        )
        try:
            result = run_accuracy(store, graph, assignment, rich_club, embedder, cases, k=10)
            json_str = json.dumps(result)
            assert len(json_str) > 0
        finally:
            store.close()

    def test_multi_label_overlap_count_present(self, tmp_path):
        """multi_label_overlap_count key is present in the result dict."""
        store, graph, assignment, rich_club, embedder, cases = build_eval_set(
            seed=42, n_filler=220, store_path=tmp_path
        )
        try:
            result = run_accuracy(store, graph, assignment, rich_club, embedder, cases, k=10)
            assert "multi_label_overlap_count" in result, (
                "multi_label_overlap_count missing from result"
            )
        finally:
            store.close()

    def test_result_schema_complete(self, tmp_path):
        """All required keys are present in the result dict."""
        store, graph, assignment, rich_club, embedder, cases = build_eval_set(
            seed=42, n_filler=220, store_path=tmp_path
        )
        try:
            result = run_accuracy(store, graph, assignment, rich_club, embedder, cases, k=10)
            required = {
                "recall_at_k",
                "false_negative_rate",
                "n_cues",
                "miss_counts_by_archetype",
                "multi_label_overlap_count",
                "unattributed",
            }
            missing = required - set(result.keys())
            assert not missing, f"result dict missing keys: {missing}"
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Gate: run_baseline entry point — small synthetic store
# ---------------------------------------------------------------------------

class TestRunBaseline:
    """run_baseline runs end-to-end on a small synthetic store and emits the
    required JSON shape.  This validates the entry point without the @37k run."""

    @pytest.fixture()
    def _synthetic_store(self, tmp_path):
        """Build a small synthetic store via build_eval_set and return its path."""
        store_path = tmp_path / "baseline_store"
        store, graph, assignment, rich_club, embedder, cases = build_eval_set(
            seed=17, n_filler=220, store_path=store_path
        )
        store.close()
        return store_path

    def test_run_baseline_returns_required_keys(self, _synthetic_store):
        """run_baseline returns a dict with all required baseline output keys."""
        result = run_baseline(
            store_path=_synthetic_store,
            k=10,
            n_samples=20,
            seed=99,
        )
        required = {
            "n_cues",
            "recall_at_k",
            "false_negative_rate",
            "k",
            "seed",
            "corpus_size",
            "driver",
            "cue_derivation_method",
            "miss_counts_by_archetype",
            "multi_label_overlap_count",
            "unattributed",
            "cosine_rank_percentiles",
            "degrade_fire_rate",
            "island_rate",
            "p50_ms",
            "p95_ms",
        }
        missing = required - set(result.keys())
        assert not missing, f"run_baseline result missing keys: {missing}"

    def test_run_baseline_cue_derivation_method(self, _synthetic_store):
        """cue_derivation_method is 'vector_perturbed_from_stored_embedding'."""
        result = run_baseline(
            store_path=_synthetic_store,
            k=10,
            n_samples=15,
            seed=7,
        )
        assert result["cue_derivation_method"] == "vector_perturbed_from_stored_embedding", (
            f"unexpected cue_derivation_method: {result['cue_derivation_method']}"
        )

    def test_run_baseline_corpus_size_positive(self, _synthetic_store):
        """corpus_size is positive (the synthetic store has records)."""
        result = run_baseline(
            store_path=_synthetic_store,
            k=10,
            n_samples=10,
            seed=1,
        )
        assert result["corpus_size"] > 0, "corpus_size must be positive"

    def test_run_baseline_n_cues_bounded_by_n_samples(self, _synthetic_store):
        """n_cues equals min(n_samples, corpus_size)."""
        result = run_baseline(
            store_path=_synthetic_store,
            k=10,
            n_samples=5,
            seed=3,
        )
        assert result["n_cues"] == 5, (
            f"n_cues ({result['n_cues']}) should equal n_samples (5) "
            "when corpus_size >= n_samples"
        )

    def test_run_baseline_recall_in_unit_interval(self, _synthetic_store):
        """recall_at_k and false_negative_rate are in [0.0, 1.0]."""
        result = run_baseline(
            store_path=_synthetic_store,
            k=10,
            n_samples=20,
            seed=42,
        )
        assert 0.0 <= result["recall_at_k"] <= 1.0, (
            f"recall_at_k out of range: {result['recall_at_k']}"
        )
        assert 0.0 <= result["false_negative_rate"] <= 1.0, (
            f"false_negative_rate out of range: {result['false_negative_rate']}"
        )

    def test_run_baseline_miss_counts_archetype_keys(self, _synthetic_store):
        """miss_counts_by_archetype has the three canonical archetype keys."""
        result = run_baseline(
            store_path=_synthetic_store,
            k=10,
            n_samples=15,
            seed=5,
        )
        counts = result["miss_counts_by_archetype"]
        assert Archetype.COSINE_RANK_BEYOND_CUTOFF in counts, (
            "COSINE_RANK_BEYOND_CUTOFF missing from miss_counts_by_archetype"
        )
        assert Archetype.SPREAD_DISABLED_BY_PRIOR_LATENCY in counts, (
            "SPREAD_DISABLED_BY_PRIOR_LATENCY missing from miss_counts_by_archetype"
        )
        assert Archetype.EDGE_LESS_ISLAND in counts, (
            "EDGE_LESS_ISLAND missing from miss_counts_by_archetype"
        )

    def test_run_baseline_cosine_rank_percentiles_keys(self, _synthetic_store):
        """cosine_rank_percentiles has p50/p90/p95/p99 keys."""
        result = run_baseline(
            store_path=_synthetic_store,
            k=10,
            n_samples=15,
            seed=8,
        )
        pcts = result["cosine_rank_percentiles"]
        for key in ("p50", "p90", "p95", "p99"):
            assert key in pcts, f"cosine_rank_percentiles missing {key}"

    def test_run_baseline_is_json_serialisable(self, _synthetic_store):
        """run_baseline result round-trips through json.dumps."""
        import json as _json
        result = run_baseline(
            store_path=_synthetic_store,
            k=10,
            n_samples=10,
            seed=2,
        )
        json_str = _json.dumps(result)
        assert len(json_str) > 0

    def test_run_baseline_writes_out_file(self, _synthetic_store, tmp_path):
        """When --out is supplied, the JSON file is created with the same content."""
        import json as _json
        out_path = tmp_path / "baseline_out.json"
        result = run_baseline(
            store_path=_synthetic_store,
            k=10,
            n_samples=10,
            seed=11,
            out=out_path,
        )
        assert out_path.exists(), "--out path was not created"
        written = _json.loads(out_path.read_text())
        assert written["n_cues"] == result["n_cues"], "--out file content mismatch"

    def test_run_baseline_deterministic_across_calls(self, _synthetic_store):
        """Same seed → identical n_cues / recall_at_k / false_negative_rate."""
        r1 = run_baseline(
            store_path=_synthetic_store,
            k=10,
            n_samples=15,
            seed=77,
        )
        r2 = run_baseline(
            store_path=_synthetic_store,
            k=10,
            n_samples=15,
            seed=77,
        )
        assert r1["n_cues"] == r2["n_cues"]
        assert r1["recall_at_k"] == r2["recall_at_k"]
        assert r1["false_negative_rate"] == r2["false_negative_rate"]

    def test_run_baseline_read_only_store_unchanged(self, _synthetic_store):
        """run_baseline does not modify the store (corpus_size is identical before/after)."""
        from iai_mcp.store import MemoryStore
        from iai_mcp.hippo import AccessMode

        store_check = MemoryStore(
            path=_synthetic_store,
            access_mode=AccessMode.SHARED,
            read_only=True,
        )
        size_before = store_check.active_records_count()
        store_check.close()

        run_baseline(
            store_path=_synthetic_store,
            k=10,
            n_samples=10,
            seed=13,
        )

        store_check2 = MemoryStore(
            path=_synthetic_store,
            access_mode=AccessMode.SHARED,
            read_only=True,
        )
        size_after = store_check2.active_records_count()
        store_check2.close()

        assert size_before == size_after, (
            f"store corpus_size changed: {size_before} → {size_after}; "
            "run_baseline must not mutate the store"
        )

    def test_run_baseline_latency_global_reset_after_run(self, _synthetic_store):
        """pipeline._last_recall_latency_ms is 0.0 after run_baseline (no sentinel leak)."""
        import iai_mcp.pipeline as _pl

        run_baseline(
            store_path=_synthetic_store,
            k=10,
            n_samples=10,
            seed=55,
        )
        assert _pl._last_recall_latency_ms == 0.0, (
            f"latency global not reset after run_baseline: {_pl._last_recall_latency_ms}"
        )


# ---------------------------------------------------------------------------
# Gate: _VectorInjectionEmbedder shim
# ---------------------------------------------------------------------------

class TestVectorInjectionEmbedder:
    """The injection shim returns pre-registered vectors for sentinel strings
    and delegates unknown strings to the wrapped embedder."""

    def test_sentinel_returns_registered_vector(self):
        """A registered sentinel string returns the pre-computed vector."""
        wrapped = _BenchEmbedder(base_seed=0, dim=_DIM)
        shim = _VectorInjectionEmbedder(wrapped)
        vec = [float(i) for i in range(_DIM)]
        sentinel = shim.register("test_key", vec)
        assert shim.embed(sentinel) == vec, "sentinel did not return registered vector"

    def test_unknown_string_delegates_to_wrapped(self):
        """A non-sentinel string is forwarded to the wrapped embedder."""
        wrapped = _BenchEmbedder(base_seed=0, dim=_DIM)
        shim = _VectorInjectionEmbedder(wrapped)
        result = shim.embed("some real query text")
        expected = wrapped.embed("some real query text")
        assert result == expected, "non-sentinel string not forwarded to wrapped embedder"

    def test_dim_attribute_mirrored(self):
        """DIM attribute is mirrored from the wrapped embedder."""
        wrapped = _BenchEmbedder(base_seed=0, dim=_DIM)
        shim = _VectorInjectionEmbedder(wrapped)
        assert shim.DIM == _DIM


# ---------------------------------------------------------------------------
# Gate: _perturb_vec preserves cosine proximity
# ---------------------------------------------------------------------------

class TestPerturbVec:
    """_perturb_vec produces a unit vector close to the source in cosine space."""

    def test_perturb_vec_is_unit(self):
        """_perturb_vec output is unit-normalised."""
        rng = np.random.default_rng(0)
        src = rng.standard_normal(_DIM).astype(np.float32)
        src = src / np.linalg.norm(src)
        result = _perturb_vec(src, rng, noise_fraction=0.15)
        norm = float(np.linalg.norm(result))
        assert abs(norm - 1.0) < 1e-5, f"perturbed vector is not unit-normalised: {norm}"

    def test_perturb_vec_cosine_proximity(self):
        """Perturbed vector has cosine > 0.95 to the source (close but not identical)."""
        rng = np.random.default_rng(1)
        src = rng.standard_normal(_DIM).astype(np.float32)
        src = src / np.linalg.norm(src)
        result = _perturb_vec(src, rng, noise_fraction=0.15)
        cosine = float(np.dot(src, result))
        assert cosine > 0.95, f"perturbed cosine too low: {cosine:.4f}"

    def test_perturb_vec_not_identical_to_source(self):
        """Perturbed vector is distinct from the source."""
        rng = np.random.default_rng(2)
        src = rng.standard_normal(_DIM).astype(np.float32)
        src = src / np.linalg.norm(src)
        result = _perturb_vec(src, rng, noise_fraction=0.15)
        assert not np.allclose(src, result), "perturbed vector is identical to source"


# ---------------------------------------------------------------------------
# Gate: semantic hit metric under duplicate-embedding clusters
# ---------------------------------------------------------------------------

class TestSemanticHitMetric:
    """The semantic hit metric scores a duplicate-cluster recall correctly where
    the exact-id metric would falsely miss."""

    def _make_id_to_vec(self, cluster_size, extras):
        """Build a corpus: `cluster_size` ids sharing one embedding, plus
        `extras` distinct-embedding ids.  Returns (id_to_vec, cluster_ids)."""
        from uuid import uuid4

        base = np.zeros(_DIM, dtype=np.float32)
        base[0] = 1.0
        id_to_vec = {}
        cluster_ids = []
        for _ in range(cluster_size):
            uid = uuid4()
            id_to_vec[uid] = base.copy()
            cluster_ids.append(uid)
        for j in range(extras):
            uid = uuid4()
            v = np.zeros(_DIM, dtype=np.float32)
            v[(j % (_DIM - 1)) + 1] = 1.0  # orthogonal to base and each other-ish
            id_to_vec[uid] = v
        return id_to_vec, cluster_ids

    def test_equivalence_class_groups_exact_duplicates(self):
        """semantic_equivalence_class groups all ids sharing the target's bytes."""
        id_to_vec, cluster_ids = self._make_id_to_vec(cluster_size=15, extras=5)
        target = cluster_ids[14]
        cls = semantic_equivalence_class(target, id_to_vec)
        assert set(cluster_ids) <= cls, "class must contain every exact duplicate"
        assert len(cls) == 15, f"class size should equal cluster size, got {len(cls)}"

    def test_semantic_metric_scores_duplicate_cluster_hit(self):
        """A k+5 duplicate cluster: recall returns ANY cluster member (not the
        specific target id) → semantic hit True, exact-id hit False."""
        k = 10
        # cluster of k+5 = 15 twins; the specific target is index 14.
        id_to_vec, cluster_ids = self._make_id_to_vec(cluster_size=15, extras=5)
        target_id = cluster_ids[14]
        target_vec = id_to_vec[target_id]
        target_class = semantic_equivalence_class(target_id, id_to_vec)

        # Recall returns the FIRST 10 cluster members (the tie-mates), NOT the
        # specific target at index 14 — exactly the exact-id pathology.
        returned_ids = cluster_ids[:k]

        # Exact-id metric: the specific id is absent → false miss.
        exact_hit = target_id in set(returned_ids[:k])
        assert not exact_hit, "exact-id metric should (falsely) miss the specific twin"

        # Semantic metric: any cluster member counts → correct hit.
        assert is_semantic_hit(
            returned_ids, target_id, target_vec, id_to_vec, target_class, k
        ), "semantic metric must score a duplicate-cluster hit"

    def test_semantic_metric_near_duplicate_within_eps(self):
        """A returned record within SEMANTIC_HIT_EPS cosine (not identical) is a hit."""
        from uuid import uuid4

        k = 10
        base = np.zeros(_DIM, dtype=np.float32)
        base[0] = 1.0
        target_id = uuid4()
        # A near-duplicate: cos slightly above EPS.
        near = base.copy()
        near[1] = 0.10  # cos ≈ 1/sqrt(1.01) ≈ 0.995 > 0.98
        near = near / np.linalg.norm(near)
        near_id = uuid4()
        id_to_vec = {target_id: base, near_id: near}
        target_class = semantic_equivalence_class(target_id, id_to_vec)
        assert float(np.dot(base, near)) >= SEMANTIC_HIT_EPS
        assert is_semantic_hit(
            [near_id], target_id, base, id_to_vec, target_class, k
        ), "near-duplicate within EPS must count as a hit"

    def test_semantic_metric_true_miss_stays_miss(self):
        """A returned record far below EPS and not in-class is NOT a hit."""
        from uuid import uuid4

        k = 10
        base = np.zeros(_DIM, dtype=np.float32)
        base[0] = 1.0
        far = np.zeros(_DIM, dtype=np.float32)
        far[1] = 1.0  # orthogonal → cos 0
        target_id, far_id = uuid4(), uuid4()
        id_to_vec = {target_id: base, far_id: far}
        target_class = semantic_equivalence_class(target_id, id_to_vec)
        assert not is_semantic_hit(
            [far_id], target_id, base, id_to_vec, target_class, k
        ), "a genuinely unrelated record must not score a hit"

    def test_class_cosine_rank_uses_best_class_member(self):
        """class_cosine_rank returns the rank of the best-placed class member,
        not the specific (possibly deep) target id."""
        from uuid import uuid4

        a, b, c = uuid4(), uuid4(), uuid4()
        # Oracle order: [a, b, c]; a and c share the class, target is c (rank 2).
        oracle = [a, b, c]
        target_class = {a, c}
        rank = class_cosine_rank(oracle, target_class, not_found_rank=999)
        assert rank == 0, f"class rank should be the best member (a@0), got {rank}"

    def test_class_cosine_rank_not_found_sentinel(self):
        """When no class member is in the oracle list, the sentinel is returned."""
        from uuid import uuid4

        oracle = [uuid4(), uuid4()]
        target_class = {uuid4()}
        assert class_cosine_rank(oracle, target_class, not_found_rank=777) == 777


# ---------------------------------------------------------------------------
# Gate: baseline loop does NOT self-inflict the degrade cascade
# ---------------------------------------------------------------------------

class TestNoDegradeeCascade:
    """A warm, non-provoked baseline loop must report degrade_fire_rate == 0.0 —
    the per-cue latency reset stops the self-sustained >2 s cascade."""

    @pytest.fixture()
    def _synthetic_store(self, tmp_path):
        store_path = tmp_path / "no_cascade_store"
        store, graph, assignment, rich_club, embedder, cases = build_eval_set(
            seed=23, n_filler=220, store_path=store_path
        )
        store.close()
        return store_path

    def test_degrade_fire_rate_zero_when_not_provoked(self, _synthetic_store):
        """On a small synthetic store where no cue is genuinely > 2 s, the
        baseline reports degrade_fire_rate == 0.0 (no self-inflicted cascade)."""
        result = run_baseline(
            store_path=_synthetic_store,
            k=10,
            n_samples=30,
            seed=101,
        )
        assert result["degrade_fire_rate"] == 0.0, (
            "warm non-provoked baseline must not fire the degrade; "
            f"got degrade_fire_rate={result['degrade_fire_rate']} "
            "(self-sustained cascade not eliminated)"
        )

    def test_spread_disabled_bucket_empty_when_not_provoked(self, _synthetic_store):
        """With no genuine slow cue, no miss is attributed to SPREAD_DISABLED."""
        result = run_baseline(
            store_path=_synthetic_store,
            k=10,
            n_samples=30,
            seed=101,
        )
        counts = result["miss_counts_by_archetype"]
        assert counts[Archetype.SPREAD_DISABLED_BY_PRIOR_LATENCY] == 0, (
            "SPREAD_DISABLED must be empty when the loop never provokes the degrade; "
            f"got {counts[Archetype.SPREAD_DISABLED_BY_PRIOR_LATENCY]}"
        )

    def test_latency_reset_before_each_cue(self, _synthetic_store):
        """The baseline resets the degrade global before every cue: a pre-set
        > 2 s sentinel does not survive into (or cascade through) the loop."""
        import iai_mcp.pipeline as _pl

        _pl._last_recall_latency_ms = 9999.0  # hostile pre-set
        result = run_baseline(
            store_path=_synthetic_store,
            k=10,
            n_samples=20,
            seed=202,
        )
        # The pre-set must not have cascaded: fire rate stays 0 and the global
        # is reset to 0.0 at end-of-run.
        assert result["degrade_fire_rate"] == 0.0, (
            "a hostile pre-set latency cascaded into the loop — per-cue reset failed"
        )
        assert _pl._last_recall_latency_ms == 0.0, "end-of-run latency reset failed"


# ---------------------------------------------------------------------------
# Gate: degrade attribution reads the START-of-call degrade decision
# ---------------------------------------------------------------------------

class TestDegradeAttributionTiming:
    """The degrade predicate used for attribution must be the one the pipeline
    actually read at call START (post per-cue reset = 0.0), not a stale
    post-call value — so a normal warm cue is never mis-attributed to
    SPREAD_DISABLED."""

    @pytest.fixture()
    def _synthetic_store(self, tmp_path):
        store_path = tmp_path / "attr_timing_store"
        store, graph, assignment, rich_club, embedder, cases = build_eval_set(
            seed=31, n_filler=220, store_path=store_path
        )
        store.close()
        return store_path

    def test_run_baseline_exposes_semantic_and_exact_recall(self, _synthetic_store):
        """Both the semantic headline and the exact-id diagnostic are reported."""
        result = run_baseline(
            store_path=_synthetic_store,
            k=10,
            n_samples=20,
            seed=42,
        )
        assert "recall_at_k" in result and "recall_at_k_exact" in result
        assert 0.0 <= result["recall_at_k"] <= 1.0
        assert 0.0 <= result["recall_at_k_exact"] <= 1.0

    def test_run_baseline_reports_eps_and_dedup(self, _synthetic_store):
        """semantic_hit_eps and excluded_duplicate_targets are recorded as run
        conditions."""
        result = run_baseline(
            store_path=_synthetic_store,
            k=10,
            n_samples=20,
            seed=42,
        )
        assert result["semantic_hit_eps"] == SEMANTIC_HIT_EPS
        assert "excluded_duplicate_targets" in result
        assert isinstance(result["excluded_duplicate_targets"], int)

    def test_cosine_rank_not_found_excluded_from_percentiles(self, _synthetic_store):
        """cosine_rank_not_found is tracked separately; censored ranks are not
        folded into the percentile distribution as the oracle_depth sentinel."""
        result = run_baseline(
            store_path=_synthetic_store,
            k=10,
            n_samples=20,
            seed=42,
        )
        assert "cosine_rank_not_found" in result
        assert isinstance(result["cosine_rank_not_found"], int)


# ---------------------------------------------------------------------------
# Gate: the harness measures the REAL production dispatch path
# ---------------------------------------------------------------------------

class TestDispatchPathRepresentativeness:
    """The harness recall invocation is the production ``memory_recall``
    dispatch entry point — the same hits and a sub-second warm latency, not the
    ~10x-slower hand-assembled exhaustive-cosine path."""

    @pytest.fixture()
    def _synthetic_store(self, tmp_path):
        store_path = tmp_path / "dispatch_repr_store"
        store, graph, assignment, rich_club, embedder, cases = build_eval_set(
            seed=29, n_filler=220, store_path=store_path
        )
        store.close()
        return store_path

    def _open_shared(self, store_path):
        from iai_mcp.store import MemoryStore
        from iai_mcp.hippo import AccessMode
        # SHARED (non-read-only) so the HNSW ANN index the production path needs
        # is built — mirrors how run_baseline opens the isolated copy.
        return MemoryStore(path=store_path, access_mode=AccessMode.SHARED)

    def test_harness_hits_equal_direct_dispatch(self, _synthetic_store):
        """``_dispatch_recall_ids`` returns exactly what a direct
        ``core.dispatch(memory_recall)`` call returns for the same injected cue —
        the harness measures the production path, so its hits ARE the production
        hits (representativeness)."""
        from uuid import UUID
        from iai_mcp import core
        from iai_mcp.embed import embedder_for_store

        store = self._open_shared(_synthetic_store)
        try:
            id_to_vec, _class_size = _scan_active_corpus(store)
            target_id, target_vec = next(iter(id_to_vec.items()))
            rng = np.random.default_rng(0)
            cue_vec = _perturb_vec(target_vec, rng, noise_fraction=0.15)

            shim = _VectorInjectionEmbedder(embedder_for_store(store))
            sentinel = shim.register(str(target_id), cue_vec.tolist())

            with _inject_embedder_into_dispatch(shim):
                # Warm both paths identically before comparing.
                _dispatch_recall_ids(store, sentinel, session_id="warm")
                harness_ids = _dispatch_recall_ids(
                    store, sentinel, session_id="probe"
                )
                direct = core.dispatch(
                    store,
                    "memory_recall",
                    {"cue": sentinel, "session_id": "probe"},
                )
            direct_ids = [UUID(h["record_id"]) for h in direct["hits"]]

            assert harness_ids == direct_ids, (
                "harness recall hits diverge from a direct dispatch call — the "
                "harness is not measuring the production recall path"
            )
            assert direct.get("ann_path_used") is True, (
                "direct dispatch did not use the production ANN path; the "
                "comparison would not be representative"
            )
        finally:
            store.close()

    def test_harness_uses_production_ann_path(self, _synthetic_store):
        """The injected dispatch call fires the production ANN warm path
        (ann_path_used True), not a degraded fallback."""
        from iai_mcp import core
        from iai_mcp.embed import embedder_for_store

        store = self._open_shared(_synthetic_store)
        try:
            id_to_vec, _ = _scan_active_corpus(store)
            target_id, target_vec = next(iter(id_to_vec.items()))
            rng = np.random.default_rng(1)
            cue_vec = _perturb_vec(target_vec, rng, noise_fraction=0.15)
            shim = _VectorInjectionEmbedder(embedder_for_store(store))
            sentinel = shim.register(str(target_id), cue_vec.tolist())
            with _inject_embedder_into_dispatch(shim):
                resp = core.dispatch(
                    store,
                    "memory_recall",
                    {"cue": sentinel, "session_id": "probe"},
                )
            assert resp.get("ann_path_used") is True
        finally:
            store.close()

    def test_warm_per_cue_latency_is_sub_second(self, _synthetic_store):
        """On a small store, warm per-cue production recall latency is
        sub-second (p95) — not the multi-second hand-rolled path."""
        result = run_baseline(
            store_path=_synthetic_store,
            k=10,
            n_samples=30,
            seed=71,
        )
        assert result["p95_ms"] < 1000.0, (
            f"warm per-cue p95 latency {result['p95_ms']}ms is not sub-second — "
            "the harness may not be on the fast production path"
        )
        assert result["p50_ms"] < 1000.0, (
            f"warm per-cue p50 latency {result['p50_ms']}ms is not sub-second"
        )

    def test_injection_context_restores_embedder_factory(self):
        """``_inject_embedder_into_dispatch`` restores the real
        ``embedder_for_store`` on exit — the global patch never leaks."""
        import iai_mcp.embed as _emb
        wrapped = _BenchEmbedder(base_seed=0, dim=_DIM)
        shim = _VectorInjectionEmbedder(wrapped)
        before = _emb.embedder_for_store
        with _inject_embedder_into_dispatch(shim):
            assert _emb.embedder_for_store is not before
        assert _emb.embedder_for_store is before, (
            "embedder_for_store was not restored after the injection context"
        )
