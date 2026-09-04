from __future__ import annotations

import random
import time
from uuid import UUID, uuid4

import pytest

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM
from tests.test_store import _make

def _seed(
    store: MemoryStore, n: int, *, seed: int = 0, compact: bool = False
) -> list[UUID]:
    from iai_mcp.store import RECORDS_TABLE

    rnd = random.Random(seed)
    ids: list[UUID] = []
    for i in range(n):
        vec = [rnd.random() for _ in range(EMBED_DIM)]
        r = _make(text=f"fact {i} :: verbatim payload {rnd.random():.6f}", vec=vec)
        store.insert(r)
        ids.append(r.id)
    if compact:
        try:
            tbl = store.db.open_table(RECORDS_TABLE)
            tbl.optimize()
        except Exception:
            pass
    return ids

def test_get_unknown_id_returns_none(tmp_path):
    store = MemoryStore(path=tmp_path)
    _seed(store, n=5)
    phantom = uuid4()
    assert store.get(phantom) is None

def test_get_known_id_roundtrip_with_decrypt(tmp_path):
    store = MemoryStore(path=tmp_path)
    # non-English fixture data: multilingual behavior under test — keep
    verbatim = "пусть каждое слово сохранится точно — G2 fidelity"
    r = _make(text=verbatim)
    store.insert(r)
    got = store.get(r.id)
    assert got is not None
    assert got.id == r.id
    assert got.literal_surface == verbatim

def test_get_does_not_call_unfiltered_to_pandas(tmp_path, monkeypatch):
    store = MemoryStore(path=tmp_path)
    _seed(store, n=20)
    target = _seed(store, n=1)[0]

    from iai_mcp.hippo import HippoTable
    from iai_mcp.store import RECORDS_TABLE

    target_cls = HippoTable
    base_to_pandas = target_cls.to_pandas
    unfiltered_calls: list[dict] = []

    def traced(self, *args, **kwargs):
        if "filter" not in kwargs:
            unfiltered_calls.append({"args": args, "kwargs": dict(kwargs)})
        return base_to_pandas(self, *args, **kwargs)

    monkeypatch.setattr(target_cls, "to_pandas", traced)

    got = store.get(target)
    assert got is not None
    assert got.id == target
    assert not unfiltered_calls, (
        "store.get called HippoTable.to_pandas() without a filter — "
        "full-scan path still in use. Expected filter-pushdown via "
        "tbl.search(...).where(...)."
    )

    # Positive control: prove the patch fires on the class store.get queries
    # through. A deliberate full-table to_pandas MUST be recorded, so the
    # primary assertion above cannot pass as a silent no-op against a class the
    # query path never touches.
    store.db.open_table(RECORDS_TABLE).to_pandas()
    assert unfiltered_calls, (
        "perf-fence mechanism check failed: a direct HippoTable.to_pandas() was "
        "not intercepted — the monkeypatch is not wired to the class store.get "
        "queries through"
    )

@pytest.mark.perf
def test_get_perf_fence_n1k(tmp_path):
    from _perf_helpers import best_of_n, skip_if_loaded

    skip_if_loaded()

    store = MemoryStore(path=tmp_path)
    ids = _seed(store, n=1000, compact=True)
    rnd = random.Random(42)
    picks = [rnd.choice(ids) for _ in range(100)]

    store.get(picks[0])

    def _measure() -> tuple[float, float, float]:
        samples_ms: list[float] = []
        for rid in picks:
            t0 = time.perf_counter()
            rec = store.get(rid)
            samples_ms.append((time.perf_counter() - t0) * 1000.0)
            assert rec is not None and rec.id == rid
        total = sum(samples_ms)
        mean = total / len(samples_ms)
        samples_ms.sort()
        p95 = samples_ms[int(0.95 * len(samples_ms)) - 1]
        return (p95, total, mean)

    p95, total, mean = best_of_n(_measure, n=3)

    assert total <= 500.0, f"N=1k 100x store.get total {total:.1f} ms > 500 ms budget"
    assert mean <= 5.0, f"N=1k store.get mean {mean:.2f} ms > 5 ms/call"
    assert p95 <= 10.0, f"N=1k store.get p95 {p95:.2f} ms > 10 ms/call"

def test_get_matches_full_scan_baseline(tmp_path):
    store = MemoryStore(path=tmp_path)
    ids = _seed(store, n=1000)
    rnd = random.Random(7)
    picks = [rnd.choice(ids) for _ in range(50)]

    tbl = store.db.open_table("records")
    df = tbl.to_pandas()

    for rid in picks:
        got = store.get(rid)
        assert got is not None
        baseline_row = df[df["id"] == str(rid)].iloc[0].to_dict()
        baseline = store._from_row(baseline_row)

        assert got.id == baseline.id
        assert got.literal_surface == baseline.literal_surface
        assert list(got.embedding) == list(baseline.embedding)
        assert got.tags == baseline.tags
        assert got.provenance == baseline.provenance
        assert got.language == baseline.language
        assert got.community_id == baseline.community_id
        assert got.centrality == baseline.centrality
        assert got.stability == baseline.stability
        assert got.difficulty == baseline.difficulty
        assert got.last_reviewed == baseline.last_reviewed
        assert got.updated_at == baseline.updated_at
