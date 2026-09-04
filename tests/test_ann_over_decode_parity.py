"""Byte-identical served-set parity for the ``query_similar`` trim-before-decode
optimization.

The fast-decode branch used to decode every fetched row (up to
``k * over_fetch_factor``) before trimming to ``k`` — decoding roughly 3x more
rows than are ever served. This file pins the replacement: decode stops the
instant ``k`` valid rows are collected (trim-before-decode), with an underfill
top-up that keeps scanning into the remaining fetched rows whenever a locally
filtered row (tombstoned/pending) would otherwise leave the served set short
of ``k``. The served candidate set (ids, order, scores) must stay
byte-identical to the pre-change eager-decode behavior.
"""
from __future__ import annotations

import sys
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


def _build_corpus(store: MemoryStore, n: int, seed_base: int, tombstone_n: int = 0) -> dict:
    """Corpus with n live records plus tombstone_n near-cue records that get
    soft-tombstoned after insert -- exercises the real SQL post-filter
    alongside the trim-before-decode change."""
    base_vec = _unit_vec(seed_base)
    for i in range(n):
        vec = _nudge(base_vec, seed_base + i + 1, strength=0.01 + 0.01 * i)
        store.insert(_make(text=f"live record {i}", vec=vec))

    tomb_ids: list[UUID] = []
    for i in range(tombstone_n):
        vec = _nudge(base_vec, seed_base + 900 + i, strength=0.005)
        rec = _make(text=f"soon tombstoned {i}", vec=vec)
        store.insert(rec)
        tomb_ids.append(rec.id)
    for rid in tomb_ids:
        _tombstone(store, rid)

    return {"cue": base_vec, "tombstoned_ids": tomb_ids}


def _tombstone(store: MemoryStore, record_id: UUID) -> None:
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
# Test 1 -- byte-identical served set: trim-before-decode fast path vs the
# untouched eager DataFrame path, both drivers, both decode tiers, multiple
# corpus sizes (with and without real tombstones mixed into the raw top-k).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
@pytest.mark.parametrize("decode_mode", ["full", "rank"])
@pytest.mark.parametrize("n,tombstone_n,k", [(30, 0, 5), (40, 4, 6)])
def test_trim_before_decode_byte_identical_to_eager_pandas_path(
    tmp_path, monkeypatch, driver, decode_mode, n, tombstone_n, k
):
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path / "store")
    try:
        corpus = _build_corpus(store, n=n, seed_base=200, tombstone_n=tombstone_n)
        cue = corpus["cue"]

        monkeypatch.delenv("IAI_MCP_ANN_FAST_DECODE_OFF", raising=False)
        trimmed = store.query_similar(cue, k=k, decode=decode_mode)

        monkeypatch.setenv("IAI_MCP_ANN_FAST_DECODE_OFF", "1")
        eager = store.query_similar(cue, k=k, decode=decode_mode)
        monkeypatch.delenv("IAI_MCP_ANN_FAST_DECODE_OFF", raising=False)

        assert len(trimmed) == len(eager)
        assert len(trimmed) > 0
        trimmed_ids = [r.id for r, _s in trimmed]
        eager_ids = [r.id for r, _s in eager]
        assert trimmed_ids == eager_ids, "served ids/order must be byte-identical"
        for (_r_t, s_t), (_r_e, s_e) in zip(trimmed, eager):
            assert s_t == pytest.approx(s_e, abs=1e-9)
        assert set(trimmed_ids).isdisjoint(set(corpus["tombstoned_ids"]))
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 2 -- high-local-tombstone-density fixture: synthetic row_dicts where
# the raw first-k window is entirely locally-invalid, proving the underfill
# top-up scans deeper and still refills to k valid served rows.
# ---------------------------------------------------------------------------


def _synthetic_row_dicts(k: int, n_tombstoned_at_front: int, n_valid_after: int) -> list[dict]:
    """row_dicts already sorted ascending by distance (matching the real
    to_row_dicts contract): the first n_tombstoned_at_front rows are locally
    tombstoned, followed by n_valid_after genuinely valid rows."""
    rows: list[dict] = []
    idx = 0
    for _ in range(n_tombstoned_at_front):
        rows.append(
            {
                "id": str(uuid4()),
                "literal_surface": "tombstoned candidate",
                "embedding": [0.1] * EMBED_DIM,
                "_distance": 0.01 * idx,
                "tombstoned_at": "2020-01-01T00:00:00+00:00",
                "embedding_pending": 0,
            }
        )
        idx += 1
    for _ in range(n_valid_after):
        rows.append(
            {
                "id": str(uuid4()),
                "literal_surface": "valid candidate",
                "embedding": [0.2] * EMBED_DIM,
                "_distance": 0.01 * idx,
                "tombstoned_at": None,
                "embedding_pending": 0,
            }
        )
        idx += 1
    return rows


def _patch_to_row_dicts(monkeypatch, rows: list[dict]) -> None:
    from iai_mcp.hippo import _table as _table_mod

    def _fake(self, substage_timings=None):
        return rows

    monkeypatch.setattr(_table_mod.HippoQuery, "to_row_dicts", _fake)


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_underfill_topup_refills_to_k_under_high_tombstone_density(
    tmp_path, monkeypatch, driver
):
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path / "store")
    try:
        store.insert(_make(text="filler so corpus is non-empty"))
        k = 5
        rows = _synthetic_row_dicts(k, n_tombstoned_at_front=5, n_valid_after=5)
        valid_ids = {UUID(r["id"]) for r in rows if r["tombstoned_at"] is None}
        _patch_to_row_dicts(monkeypatch, rows)

        monkeypatch.delenv("IAI_MCP_ANN_FAST_DECODE_OFF", raising=False)
        served = store.query_similar(_unit_vec(1), k=k, decode="rank")

        served_ids = {r.id for r, _s in served}
        assert len(served) == k, "top-up must refill to k valid rows"
        assert served_ids == valid_ids, "served set must be exactly the valid rows"
        assert served_ids.isdisjoint(
            {UUID(r["id"]) for r in rows if r["tombstoned_at"] is not None}
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 3 -- non-vacuity control: a naive decode-first-k-then-filter (no
# top-up) implementation applied to the SAME fixture must underfill/differ,
# proving the fixture actually exercises the top-up path.
# ---------------------------------------------------------------------------


def _naive_trim_without_topup(rows: list[dict], k: int) -> set[UUID]:
    """Mirrors the pre-fix behavior on this row shape: decode only the first
    k rows, filter for validity within that window, never scan further."""
    out: set[UUID] = set()
    for row in rows[:k]:
        if row["tombstoned_at"] is not None:
            continue
        out.add(UUID(row["id"]))
    return out


def test_underfill_topup_fixture_is_non_vacuous(tmp_path, monkeypatch):
    _set_driver(monkeypatch, "lilli")
    store = MemoryStore(path=tmp_path / "store")
    try:
        store.insert(_make(text="filler so corpus is non-empty"))
        k = 5
        rows = _synthetic_row_dicts(k, n_tombstoned_at_front=5, n_valid_after=5)
        valid_ids = {UUID(r["id"]) for r in rows if r["tombstoned_at"] is None}
        _patch_to_row_dicts(monkeypatch, rows)

        naive_ids = _naive_trim_without_topup(rows, k)
        assert len(naive_ids) < k, (
            "fixture must be constructed so a naive decode-first-k-without-"
            "topup implementation underfills -- otherwise it proves nothing"
        )
        assert naive_ids != valid_ids

        monkeypatch.delenv("IAI_MCP_ANN_FAST_DECODE_OFF", raising=False)
        served = store.query_similar(_unit_vec(1), k=k, decode="rank")
        served_ids = {r.id for r, _s in served}

        assert served_ids == valid_ids
        assert served_ids != naive_ids
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 4 -- type pin: the trim loop's tombstoned_at/embedding_pending recheck
# assumes specific Python types for a live (non-tombstoned, non-pending) row
# on both drivers. If a driver ever returns a truthy-but-not-caught type
# (e.g. the string "0", or "" instead of None), the recheck would silently
# drop a valid row -- pin the real types so that regression is caught here,
# not as a served-set mismatch.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_live_row_dict_tombstone_pending_field_types(tmp_path, monkeypatch, driver):
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path / "store")
    try:
        rec = _make(text="type pin probe")
        store.insert(rec)

        tbl = store.db.open_table("records")
        q = (
            tbl.search(list(rec.embedding))
            .distance_type("cosine")
            .where("tombstoned_at IS NULL AND COALESCE(embedding_pending, 0) = 0")
            .limit(5)
        )
        rows = q.to_row_dicts()
        assert len(rows) == 1

        row = rows[0]
        assert row.get("tombstoned_at") is None, (
            f"live row tombstoned_at must be None, got {row.get('tombstoned_at')!r} "
            f"({type(row.get('tombstoned_at'))}) -- the trim loop's "
            "`is not None` check assumes this"
        )
        pending = row.get("embedding_pending")
        assert pending in (None, 0, False) and not isinstance(pending, str), (
            f"live row embedding_pending must be None/0/False, got {pending!r} "
            f"({type(pending)}) -- the trim loop's `not in (None, 0, False)` "
            "check assumes this and a string '0' would defeat it"
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 5 -- decode-call-count: the fast-decode branch must call the row
# decoder exactly len(served) times, strictly less than rows_fetched when
# the corpus exceeds k -- the direct proof that decode volume, not just the
# served set, shrank.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_decode_call_count_matches_served_not_fetched(tmp_path, monkeypatch, driver):
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path / "store")
    try:
        corpus = _build_corpus(store, n=40, seed_base=300)
        cue = corpus["cue"]
        k = 5

        call_count = {"n": 0}
        original = store._from_row_rank_view

        def _spy(row):
            call_count["n"] += 1
            return original(row)

        monkeypatch.setattr(store, "_from_row_rank_view", _spy)

        substage_timings: dict = {}
        monkeypatch.delenv("IAI_MCP_ANN_FAST_DECODE_OFF", raising=False)
        served = store.query_similar(
            cue, k=k, decode="rank", substage_timings=substage_timings
        )

        rows_fetched = int(substage_timings["rows_fetched"])
        assert rows_fetched > k, (
            "fixture must over-fetch beyond k for this test to prove anything"
        )
        assert call_count["n"] == len(served)
        assert call_count["n"] <= k
        assert call_count["n"] < rows_fetched, (
            "decode-call count must be strictly less than the fetched pool "
            "size once the corpus exceeds k -- otherwise the over-fetched "
            "pool is still being decoded in full"
        )
    finally:
        store.close()
