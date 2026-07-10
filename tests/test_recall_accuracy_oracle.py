"""Deterministic gate for the exhaustive-cosine oracle and recall metrics.

These tests pin the oracle and metric functions on a fixed synthetic corpus
using a small embedding dimension (via environment-level monkeypatching) so
the real Rust embedder is never loaded and the suite stays fast.

The oracle is the offline ground-truth scan: an exhaustive cosine scan over
every active record in the corpus.  It is the correct-but-slow answer that the
graph thread-following path is judged against.  It must:

- return the most-similar record first (cosine top-k, descending);
- be deterministic (two calls on the same store return identical ordered ids);
- be side-effect-free (the active-record count is unchanged after a call);
- handle edge cases: empty corpus → empty list; k > N → all N ids returned.

The metric functions (recall_at_k, is_false_negative, mean_recall_at_k,
false_negative_rate) are pure host-side functions that score the graph path
against the oracle without touching any recall internals.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from uuid import UUID, uuid4

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Embed-dim monkeypatch — must come before any iai_mcp import so the Rust
# embedder init sees the env var and the store uses a matching dimension.
# ---------------------------------------------------------------------------
_DIM = 16  # small synthetic dim; avoids loading real embedder


@pytest.fixture(autouse=True)
def _small_embed_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(_DIM))


# ---------------------------------------------------------------------------
# Imports (deferred past the fixture so env var is visible at module level)
# ---------------------------------------------------------------------------
from iai_mcp.store import MemoryStore  # noqa: E402
from iai_mcp.types import MemoryRecord  # noqa: E402

# oracle + metric module under test
from bench.recall_accuracy import (  # noqa: E402
    false_negative_rate,
    is_false_negative,
    mean_recall_at_k,
    oracle_top_k,
    recall_at_k,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vec(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(_DIM).astype(np.float32)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v


def _make_record(vec: np.ndarray) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="fixture record for alice",
        aaak_index="",
        embedding=vec.tolist(),
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
        tags=[],
        language="en",
    )


def _seed_store(
    tmp_path, n: int, seed: int = 42
) -> tuple[MemoryStore, list[MemoryRecord], np.random.Generator]:
    rng = np.random.default_rng(seed)
    store = MemoryStore(path=tmp_path)
    records: list[MemoryRecord] = []
    for _ in range(n):
        rec = _make_record(_unit_vec(rng))
        store.insert(rec)
        records.append(rec)
    return store, records, rng


# ---------------------------------------------------------------------------
# Oracle tests
# ---------------------------------------------------------------------------


class TestOracleTopK:
    """Exhaustive-cosine oracle correctness and safety properties."""

    def test_top1_is_exact_match(self, tmp_path):
        """When cue equals a stored vector the oracle returns that record first."""
        store, records, rng = _seed_store(tmp_path, 40)
        try:
            # Use the first record's embedding as cue.
            first_rec = records[0]
            cue = np.array(first_rec.embedding, dtype=np.float32)

            result = oracle_top_k(store, cue, k=1)
            assert len(result) == 1
            assert result[0] == first_rec.id
        finally:
            store.close()

    def test_deterministic_across_calls(self, tmp_path):
        """Two calls on the same store return the same ordered id list."""
        store, records, rng = _seed_store(tmp_path, 40)
        try:
            cue = _unit_vec(rng)
            result_a = oracle_top_k(store, cue, k=10)
            result_b = oracle_top_k(store, cue, k=10)
            assert result_a == result_b
        finally:
            store.close()

    def test_side_effect_free(self, tmp_path):
        """Active record count is unchanged before and after an oracle call."""
        store, records, rng = _seed_store(tmp_path, 40)
        try:
            count_before = store.active_records_count()
            cue = _unit_vec(rng)
            oracle_top_k(store, cue, k=10)
            count_after = store.active_records_count()
            assert count_before == count_after
        finally:
            store.close()

    def test_empty_corpus_returns_empty(self, tmp_path):
        """An empty store → empty oracle result, no crash."""
        store = MemoryStore(path=tmp_path)
        try:
            cue = _unit_vec(np.random.default_rng(0))
            result = oracle_top_k(store, cue, k=5)
            assert result == []
        finally:
            store.close()

    def test_k_larger_than_n_returns_all(self, tmp_path):
        """k > corpus size → the oracle returns all N records, no crash."""
        store, records, rng = _seed_store(tmp_path, 10)
        try:
            cue = _unit_vec(rng)
            result = oracle_top_k(store, cue, k=1000)
            assert len(result) == 10
            # All returned ids must be from the inserted records.
            inserted_ids = {r.id for r in records}
            assert set(result) == inserted_ids
        finally:
            store.close()

    def test_results_keyed_by_uuid(self, tmp_path):
        """Oracle returns UUID objects matching MemoryRecord.id, not integers."""
        store, records, rng = _seed_store(tmp_path, 20)
        try:
            cue = _unit_vec(rng)
            result = oracle_top_k(store, cue, k=5)
            for rid in result:
                assert isinstance(rid, UUID), f"expected UUID, got {type(rid)}"
        finally:
            store.close()

    def test_exact_cosine_rank_order(self, tmp_path):
        """The oracle rank matches an independent exact-cosine calculation."""
        n = 40
        seed = 99
        rng = np.random.default_rng(seed)
        store = MemoryStore(path=tmp_path)
        records: list[MemoryRecord] = []
        vecs: list[np.ndarray] = []
        try:
            for _ in range(n):
                v = _unit_vec(rng)
                rec = _make_record(v)
                store.insert(rec)
                records.append(rec)
                vecs.append(v)

            cue = _unit_vec(rng)
            k = 10

            # Independent exact-cosine reference.
            mat = np.stack(vecs)  # already unit-norm
            sims = mat @ cue
            ref_order = np.argsort(-sims)[:k]
            ref_ids = [records[i].id for i in ref_order]

            oracle_result = oracle_top_k(store, cue, k=k)
            assert oracle_result == ref_ids, (
                "Oracle order does not match independent exact-cosine reference"
            )
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Metric tests
# ---------------------------------------------------------------------------


class TestRecallAtK:
    """Pure host-side recall@k metric correctness."""

    def test_perfect_match_scores_one(self):
        ids = [uuid4() for _ in range(5)]
        assert recall_at_k(ids, ids, k=5) == pytest.approx(1.0)

    def test_zero_overlap_scores_zero(self):
        oracle = [uuid4() for _ in range(5)]
        returned = [uuid4() for _ in range(5)]
        assert recall_at_k(returned, oracle, k=5) == pytest.approx(0.0)

    def test_partial_overlap(self):
        ids = [uuid4() for _ in range(6)]
        oracle = ids[:4]
        returned = [ids[0], ids[1], uuid4(), uuid4()]
        # 2 overlap out of min(4, 4)=4 → 0.5
        assert recall_at_k(returned, oracle, k=4) == pytest.approx(0.5)

    def test_empty_oracle_returns_one(self):
        """Empty oracle → denominator 0 → 1.0 by convention."""
        assert recall_at_k([uuid4()], [], k=5) == pytest.approx(1.0)

    def test_k_truncates_returned(self):
        """Only the first k elements of returned_ids are considered."""
        ids = [uuid4() for _ in range(6)]
        oracle = [ids[4]]  # the 5th element
        returned = [uuid4(), uuid4(), uuid4(), uuid4(), ids[4], ids[4]]
        # The match is at position 4 (0-indexed), which is within k=5.
        assert recall_at_k(returned, oracle, k=5) == pytest.approx(1.0)
        # But if k=4 we miss it.
        assert recall_at_k(returned, oracle, k=4) == pytest.approx(0.0)


class TestIsFalseNegative:
    """False-negative detection: the most-relevant memory is absent."""

    def test_top1_present_not_fn(self):
        oracle = [uuid4() for _ in range(3)]
        returned = [oracle[0], uuid4()]
        assert is_false_negative(returned, oracle, k=2) is False

    def test_top1_absent_is_fn(self):
        top1 = uuid4()
        oracle = [top1, uuid4(), uuid4()]
        returned = [oracle[1], oracle[2]]
        assert is_false_negative(returned, oracle, k=2) is True

    def test_empty_oracle_not_fn(self):
        """An empty oracle means there is no most-relevant memory to miss."""
        assert is_false_negative([uuid4()], [], k=5) is False

    def test_top1_present_at_boundary(self):
        """The oracle top-1 present as the k-th element is not a false negative."""
        ids = [uuid4() for _ in range(5)]
        oracle = ids  # ids[0] is the top-1
        returned = [ids[4], ids[3], ids[2], ids[1], ids[0]]  # top-1 at position 4
        assert is_false_negative(returned, oracle, k=5) is False


class TestAggregateMetrics:
    """mean_recall_at_k and false_negative_rate over a list of pairs."""

    def _pairs(self):
        # Two perfect pairs + one full miss.
        a = [uuid4() for _ in range(3)]
        b = [uuid4() for _ in range(3)]
        miss_oracle = [uuid4() for _ in range(3)]
        miss_returned = [uuid4() for _ in range(3)]  # no overlap
        return [
            (a, a),  # perfect
            (b, b),  # perfect
            (miss_returned, miss_oracle),  # full miss
        ]

    def test_mean_recall_at_k(self):
        pairs = self._pairs()
        # Two perfect (1.0) + one zero (0.0) → 2/3
        val = mean_recall_at_k(pairs, k=3)
        assert val == pytest.approx(2.0 / 3.0, abs=1e-9)

    def test_false_negative_rate(self):
        pairs = self._pairs()
        # Only the last pair is a false negative → 1/3
        val = false_negative_rate(pairs, k=3)
        assert val == pytest.approx(1.0 / 3.0, abs=1e-9)

    def test_all_perfect_zero_fn_rate(self):
        ids = [uuid4() for _ in range(5)]
        pairs = [(ids, ids)] * 4
        assert false_negative_rate(pairs, k=5) == pytest.approx(0.0)

    def test_empty_pairs_returns_one_and_zero(self):
        """Empty list of pairs: mean recall defaults to 1.0; FN rate to 0.0."""
        assert mean_recall_at_k([], k=5) == pytest.approx(1.0)
        assert false_negative_rate([], k=5) == pytest.approx(0.0)
