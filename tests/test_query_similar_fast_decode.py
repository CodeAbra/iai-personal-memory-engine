"""Both-driver byte-identity differential for the ``query_similar`` ANN seed decode.

``query_similar`` (the recall ANN candidate fetch) offers two decode arms for
the same knn_query + ``vec_label IN`` row fetch: a direct-decode arm
(``HippoQuery.to_row_dicts``) that feeds rows straight into ``_from_row``, and
a DataFrame arm (``HippoQuery.to_pandas`` -> ``_ann_to_pandas``) that
materializes a pandas frame first via an ``iterrows()``/``to_dict()`` walk.
The direct-decode arm is the hot path; the DataFrame arm is reachable via a
kill-switch and as the per-call fail-safe fallback.

This file pins the equivalence: both arms must return byte-identical
``(record, score)`` lists (same records, same field values including a
fully-decoded ``literal_surface``, same scores, same order — including tied
distances), on both storage drivers, with a live in-session differential
(never a frozen snapshot). It also pins the mechanism (zero DataFrame
construction on the hot path when the direct-decode arm is active), the
kill-switch that restores the DataFrame arm, the fail-safe fallback when the
direct-decode arm raises, the AES full-decode guarantee (no surface
deferral), and the existing over-fetch/tombstone-post-filter edge semantics.
"""
from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM

_DRIVER_PARAMS = [
    pytest.param("stdlib", id="stdlib"),
    pytest.param("lilli", id="lilli"),
]


@pytest.fixture(autouse=True)
def _crypto_passphrase(monkeypatch):
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")
    yield


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch):
    import keyring as _keyring

    fake: dict = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake.pop((s, u), None))
    yield fake


def _set_driver(monkeypatch: pytest.MonkeyPatch, driver: str) -> None:
    if driver == "stdlib":
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    else:
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", driver)


def _unit_vec(seed: int, dim: int = EMBED_DIM) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _nudge(base: list[float], seed: int, strength: float = 0.05) -> list[float]:
    """Return a unit vector near *base* (cosine close to 1, but distinct)."""
    rng = np.random.default_rng(seed)
    b = np.array(base, dtype=np.float32)
    noise = rng.standard_normal(len(base)).astype(np.float32)
    noise -= float(np.dot(noise, b)) * b
    noise_norm = float(np.linalg.norm(noise))
    if noise_norm > 1e-8:
        noise /= noise_norm
    v = (1 - strength) * b + strength * noise
    v /= float(np.linalg.norm(v))
    return v.tolist()


def _record_fields_equal(a, b) -> bool:
    """Full field equality between two MemoryRecord instances."""
    for f in fields(a):
        va = getattr(a, f.name)
        vb = getattr(b, f.name)
        if f.name == "embedding":
            if len(va) != len(vb):
                return False
            if not np.allclose(np.array(va, dtype=np.float32), np.array(vb, dtype=np.float32)):
                return False
            continue
        if va != vb:
            return False
    return True


def _build_corpus(store: MemoryStore, n: int = 24, seed_base: int = 100) -> dict:
    """Build a corpus covering: distinct cosine distances, a tombstoned record
    inside the raw ANN top-k, embedding_pending rows, tier variety, and enough
    records that k < corpus (over-fetch trim exercised).

    Returns a dict with the cue vector and named record ids.
    """
    base_vec = _unit_vec(seed_base)
    live_ids: list[UUID] = []
    for i in range(n):
        vec = _nudge(base_vec, seed_base + i + 1, strength=0.02 + 0.01 * i)
        rec = _make(text=f"live record {i}", vec=vec)
        store.insert(rec)
        live_ids.append(rec.id)

    # A record placed VERY close to the cue (near-guaranteed top-k membership)
    # that will be soft-tombstoned after insert — proves the where post-filter
    # drops it from query_similar's decoded output on both paths.
    tomb_vec = _nudge(base_vec, seed_base + 999, strength=0.01)
    tomb_rec = _make(text="soon to be tombstoned", vec=tomb_vec)
    store.insert(tomb_rec)

    # A tier-filtered probe record.
    tier_vec = _nudge(base_vec, seed_base + 500, strength=0.03)
    tier_rec = _make(tier="semantic", text="tier probe", vec=tier_vec)
    store.insert(tier_rec)

    now_iso_tombstone(store, tomb_rec.id)

    return {
        "cue": base_vec,
        "live_ids": live_ids,
        "tombstoned_id": tomb_rec.id,
        "tier_id": tier_rec.id,
        "n_live": n,
    }


def now_iso_tombstone(store: MemoryStore, record_id: UUID) -> None:
    """Soft-tombstone a record via direct SQL UPDATE — mirrors the erasure-
    agent code path (tests/test_exact_authority_lifecycle.py precedent): the
    row stays in SQL + the hnsw index (until the next sleep-cycle rebuild),
    only ``tombstoned_at`` flips, so it still surfaces in the raw ANN top-k
    and must be caught by query_similar's where post-filter.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    with store.db._conn_lock:
        store.db._conn.execute(
            "UPDATE records SET tombstoned_at = ? WHERE id = ?",
            (now, str(record_id)),
        )
        store.db._conn.commit()
    store.invalidate_exact_index()


# ---------------------------------------------------------------------------
# Test 1 — byte-identity differential (fast path vs pandas path, live)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_fast_decode_byte_identical_to_pandas_path(tmp_path, monkeypatch, driver):
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path / "store")
    try:
        corpus = _build_corpus(store)
        cue = corpus["cue"]

        monkeypatch.delenv("IAI_MCP_ANN_FAST_DECODE_OFF", raising=False)
        fast_results = store.query_similar(cue, k=5)

        monkeypatch.setenv("IAI_MCP_ANN_FAST_DECODE_OFF", "1")
        pandas_results = store.query_similar(cue, k=5)
        monkeypatch.delenv("IAI_MCP_ANN_FAST_DECODE_OFF", raising=False)

        assert len(fast_results) == len(pandas_results)
        assert len(fast_results) > 0

        for (rec_f, score_f), (rec_p, score_p) in zip(fast_results, pandas_results):
            assert rec_f.id == rec_p.id
            assert score_f == pytest.approx(score_p, abs=1e-9)
            assert _record_fields_equal(rec_f, rec_p), (
                f"record field mismatch for id={rec_f.id}"
            )

        # tier-filtered call must also match
        monkeypatch.delenv("IAI_MCP_ANN_FAST_DECODE_OFF", raising=False)
        fast_tier = store.query_similar(cue, k=3, tier="semantic")
        monkeypatch.setenv("IAI_MCP_ANN_FAST_DECODE_OFF", "1")
        pandas_tier = store.query_similar(cue, k=3, tier="semantic")
        monkeypatch.delenv("IAI_MCP_ANN_FAST_DECODE_OFF", raising=False)

        assert len(fast_tier) == len(pandas_tier)
        for (rec_f, score_f), (rec_p, score_p) in zip(fast_tier, pandas_tier):
            assert rec_f.id == rec_p.id
            assert score_f == pytest.approx(score_p, abs=1e-9)
            assert _record_fields_equal(rec_f, rec_p)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 2 — zero DataFrame materialization on the hot path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_fast_decode_builds_zero_dataframes(tmp_path, monkeypatch, driver):
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path / "store")
    try:
        corpus = _build_corpus(store, n=10)
        cue = corpus["cue"]

        from iai_mcp.hippo import _table as _table_mod

        call_count = {"n": 0}
        original_ann_to_pandas = _table_mod.HippoQuery._ann_to_pandas

        def _spy(self):
            call_count["n"] += 1
            return original_ann_to_pandas(self)

        monkeypatch.setattr(_table_mod.HippoQuery, "_ann_to_pandas", _spy)
        monkeypatch.delenv("IAI_MCP_ANN_FAST_DECODE_OFF", raising=False)

        store.query_similar(cue, k=5)

        assert call_count["n"] == 0, (
            "query_similar invoked the pandas DataFrame materialization path "
            "while the fast decode branch was active"
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 3 — kill-switch: the pandas path is taken and correct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_fast_decode_kill_switch_restores_pandas_path(tmp_path, monkeypatch, driver):
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path / "store")
    try:
        corpus = _build_corpus(store, n=10)
        cue = corpus["cue"]

        from iai_mcp.hippo import _table as _table_mod

        call_count = {"n": 0}
        original_ann_to_pandas = _table_mod.HippoQuery._ann_to_pandas

        def _spy(self):
            call_count["n"] += 1
            return original_ann_to_pandas(self)

        monkeypatch.setattr(_table_mod.HippoQuery, "_ann_to_pandas", _spy)
        monkeypatch.setenv("IAI_MCP_ANN_FAST_DECODE_OFF", "1")

        results = store.query_similar(cue, k=5)

        assert call_count["n"] == 1, (
            "kill-switch must route query_similar through the pandas "
            "materialization path (_ann_to_pandas) exactly once"
        )
        assert len(results) > 0
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 4 — AES-fence: full decode, real float embedding, no deferral
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_fast_decode_full_aes_decode_and_embedding_shape(tmp_path, monkeypatch, driver):
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path / "store")
    try:
        corpus = _build_corpus(store, n=8)
        cue = corpus["cue"]

        monkeypatch.delenv("IAI_MCP_ANN_FAST_DECODE_OFF", raising=False)
        results = store.query_similar(cue, k=5)

        assert len(results) > 0
        for rec, _score in results:
            assert rec.literal_surface, (
                f"record {rec.id} has empty literal_surface — "
                "surface deferral leaked into the fast decode branch"
            )
            assert len(rec.embedding) == EMBED_DIM, (
                f"record {rec.id} embedding has {len(rec.embedding)} elements, "
                f"expected {EMBED_DIM} floats (BLOB decode missing — likely a "
                "raw byte-int list from an undecoded embedding column)"
            )
            assert all(isinstance(x, float) for x in rec.embedding)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 5 — edge semantics parity: empty store, over-fetch/clamp, all-tombstoned
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_fast_decode_empty_store_returns_empty_list(tmp_path, monkeypatch, driver):
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path / "store")
    try:
        cue = _unit_vec(1)
        monkeypatch.delenv("IAI_MCP_ANN_FAST_DECODE_OFF", raising=False)
        assert store.query_similar(cue, k=5) == []

        monkeypatch.setenv("IAI_MCP_ANN_FAST_DECODE_OFF", "1")
        assert store.query_similar(cue, k=5) == []
        monkeypatch.delenv("IAI_MCP_ANN_FAST_DECODE_OFF", raising=False)
    finally:
        store.close()


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_fast_decode_k_larger_than_corpus_returns_all_live_no_error(
    tmp_path, monkeypatch, driver
):
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path / "store")
    try:
        base_vec = _unit_vec(7)
        for i in range(4):
            vec = _nudge(base_vec, 7000 + i, strength=0.05)
            store.insert(_make(text=f"small corpus {i}", vec=vec))

        monkeypatch.delenv("IAI_MCP_ANN_FAST_DECODE_OFF", raising=False)
        fast = store.query_similar(base_vec, k=100)

        monkeypatch.setenv("IAI_MCP_ANN_FAST_DECODE_OFF", "1")
        legacy = store.query_similar(base_vec, k=100)
        monkeypatch.delenv("IAI_MCP_ANN_FAST_DECODE_OFF", raising=False)

        assert len(fast) == len(legacy)
        assert len(fast) == 4
        for (rec_f, score_f), (rec_p, score_p) in zip(fast, legacy):
            assert rec_f.id == rec_p.id
            assert score_f == pytest.approx(score_p, abs=1e-9)
    finally:
        store.close()


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_fast_decode_all_tombstoned_top_k_underfills_identically(
    tmp_path, monkeypatch, driver
):
    """When the raw ANN top-k is entirely tombstoned, both arms must return
    the same (possibly empty, possibly under-filled) result — the over-fetch
    discipline is a best-effort mitigation, not a correctness guarantee, and
    both code paths must degrade identically."""
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path / "store")
    try:
        base_vec = _unit_vec(55)
        tomb_ids = []
        for i in range(3):
            vec = _nudge(base_vec, 5500 + i, strength=0.01)
            rec = _make(text=f"will be tombstoned {i}", vec=vec)
            store.insert(rec)
            tomb_ids.append(rec.id)

        # a distant filler so the corpus isn't literally empty post-tombstone
        far_vec = _unit_vec(999)
        store.insert(_make(text="distant filler", vec=far_vec))

        for rid in tomb_ids:
            now_iso_tombstone(store, rid)

        monkeypatch.delenv("IAI_MCP_ANN_FAST_DECODE_OFF", raising=False)
        fast = store.query_similar(base_vec, k=3)

        monkeypatch.setenv("IAI_MCP_ANN_FAST_DECODE_OFF", "1")
        legacy = store.query_similar(base_vec, k=3)
        monkeypatch.delenv("IAI_MCP_ANN_FAST_DECODE_OFF", raising=False)

        assert len(fast) == len(legacy)
        fast_ids = {r.id for r, _s in fast}
        legacy_ids = {r.id for r, _s in legacy}
        assert fast_ids == legacy_ids
        assert fast_ids.isdisjoint(set(tomb_ids))
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 6 — tie-distance ordering: both arms must be stable and order-identical
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_fast_decode_tie_distance_order_matches_pandas_path(tmp_path, monkeypatch, driver):
    """A tie block of bit-identical embeddings straddling the k boundary must
    sort identically (same order, not merely the same set) across the
    direct-decode arm, the DataFrame arm (kill-switch on), and the per-call
    fail-safe fallback. Both arms sort with a stable algorithm, so ties break
    on fetch order — never an arbitrary quicksort permutation."""
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path / "store")
    try:
        base_vec = _unit_vec(77)

        # 12 bit-identical embeddings tied at distance 0.0 to the cue,
        # straddling the k=5 cutoff inside the 3x over-fetch. never_merge=True
        # bypasses the pattern-separation dedupe gate that would otherwise
        # collapse these duplicate-embedding inserts.
        tie_ids: list[UUID] = []
        for i in range(12):
            rec = _make(
                text=f"tied record {i}", vec=list(base_vec), never_merge=True
            )
            store.insert(rec)
            tie_ids.append(rec.id)

        # a handful of distinct-distance fillers so the tie block isn't the
        # entire corpus.
        for i in range(6):
            vec = _nudge(base_vec, 8800 + i, strength=0.2 + 0.05 * i)
            store.insert(_make(text=f"distinct filler {i}", vec=vec))

        monkeypatch.delenv("IAI_MCP_ANN_FAST_DECODE_OFF", raising=False)
        fast_results = store.query_similar(base_vec, k=5)

        monkeypatch.setenv("IAI_MCP_ANN_FAST_DECODE_OFF", "1")
        pandas_results = store.query_similar(base_vec, k=5)
        monkeypatch.delenv("IAI_MCP_ANN_FAST_DECODE_OFF", raising=False)

        assert len(fast_results) == 5
        assert len(pandas_results) == 5

        fast_ids = [r.id for r, _s in fast_results]
        pandas_ids = [r.id for r, _s in pandas_results]
        assert fast_ids == pandas_ids, (
            "fast-decode and pandas arms must return the SAME tie-block "
            f"records in the SAME order; got fast={fast_ids} "
            f"pandas={pandas_ids}"
        )
        # All 5 returned records must come from the 12-wide tie block —
        # the tie block is closer to the cue than every distinct filler.
        assert set(fast_ids) <= set(tie_ids)

        # Now force the per-call fail-safe (fast arm raises) and confirm it
        # falls back to the identical pandas-arm order, not a re-permuted one.
        from iai_mcp.hippo import _table as _table_mod

        def _raise(self):
            raise RuntimeError("synthetic fast-decode failure")

        monkeypatch.setattr(_table_mod.HippoQuery, "to_row_dicts", _raise)
        monkeypatch.delenv("IAI_MCP_ANN_FAST_DECODE_OFF", raising=False)
        failsafe_results = store.query_similar(base_vec, k=5)
        failsafe_ids = [r.id for r, _s in failsafe_results]
        assert failsafe_ids == pandas_ids, (
            "fail-safe fallback must match the pandas arm's tie order exactly"
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 7 — fail-safe fallback: fast arm raises, recall never breaks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_fast_decode_failure_falls_back_to_pandas_path(tmp_path, monkeypatch, driver):
    """A fast-decode arm exception must never escape into recall — the
    per-call fail-safe must transparently return the pandas-arm result for
    the same cue."""
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path / "store")
    try:
        corpus = _build_corpus(store, n=10)
        cue = corpus["cue"]

        monkeypatch.delenv("IAI_MCP_ANN_FAST_DECODE_OFF", raising=False)
        expected = store.query_similar(cue, k=5)

        from iai_mcp.hippo import _table as _table_mod

        def _raise(self):
            raise RuntimeError("synthetic fast-decode failure")

        monkeypatch.setattr(_table_mod.HippoQuery, "to_row_dicts", _raise)
        monkeypatch.delenv("IAI_MCP_ANN_FAST_DECODE_OFF", raising=False)

        results = store.query_similar(cue, k=5)

        assert len(results) == len(expected)
        assert len(results) > 0
        result_ids = [r.id for r, _s in results]
        expected_ids = [r.id for r, _s in expected]
        assert result_ids == expected_ids
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 8 — NaN-distance ordering: NaN rows sort last, stable among themselves
# ---------------------------------------------------------------------------


def test_to_row_dicts_nan_distance_sorts_last_and_stable(monkeypatch):
    """A row whose ``vec_label`` is missing from the distance map (defensive
    branch — unreachable via the SQL ``IN`` fetch today, but must still sort
    correctly if it ever occurs) must land after every real-distance row,
    preserving fetch order among the NaN rows themselves — matching pandas'
    ``na_position='last'`` default."""
    from iai_mcp.hippo._table import HippoQuery

    class _FakeCursor:
        description = [("vec_label",), ("literal_surface",)]

    query = HippoQuery.__new__(HippoQuery)
    fetched = [
        (1, "has-distance-a"),
        (99, "nan-row-1"),  # 99 absent from dist_map -> NaN
        (2, "has-distance-b"),
        (98, "nan-row-2"),  # 98 absent from dist_map -> NaN
    ]
    dist_map = {1: 0.5, 2: 0.1}

    monkeypatch_core = (_FakeCursor(), fetched, dist_map)

    def _fake_core(self):
        return monkeypatch_core

    query._ann_vector = object()
    query._ann_db = object()
    monkeypatch.setattr(HippoQuery, "_ann_knn_fetch_core", _fake_core)
    rows = query.to_row_dicts()

    surfaces = [r["literal_surface"] for r in rows]
    # Real-distance rows sort ascending by distance (0.1 then 0.5), NaN rows
    # land last in their original relative (fetch) order.
    assert surfaces == [
        "has-distance-b",
        "has-distance-a",
        "nan-row-1",
        "nan-row-2",
    ]


# ---------------------------------------------------------------------------
# Test 9 — to_row_dicts on a non-ANN query degrades to [] instead of raising
# ---------------------------------------------------------------------------


def test_to_row_dicts_non_ann_query_returns_empty_list(tmp_path):
    """``to_row_dicts`` must mirror ``to_pandas``'s ANN-branch guard: a query
    built without an ANN vector has no hnsw index to knn_query against, so it
    must degrade to an empty list rather than raising ``AttributeError`` on
    ``None._hnsw_lock``."""
    store = MemoryStore(path=tmp_path / "store")
    try:
        store.insert(_make(text="unrelated record"))
        tbl = store.db.open_table("records")
        q = tbl.search()  # no vector -> ann_vector/ann_db unset
        assert q.to_row_dicts() == []
    finally:
        store.close()
