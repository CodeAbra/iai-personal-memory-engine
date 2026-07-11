from __future__ import annotations


import numpy as np

from bench.verbatim import _make_noise, _make_pinned
from iai_mcp.retrieve import recall
from iai_mcp.store import EMBED_DIM, MemoryStore

NOISE_SEED = 20260419

def _seed_store(tmp_path, n_pinned: int, n_noise: int, dim: int = EMBED_DIM):
    store = MemoryStore(path=tmp_path)
    pinned_texts = [
        f"Alice pinned verbatim day {i}: phrase-{i}-{'x' * 10}"
        for i in range(n_pinned)
    ]
    pinned_records = [_make_pinned(t, dim=dim) for t in pinned_texts]
    for r in pinned_records:
        store.insert(r)

    rng = np.random.default_rng(NOISE_SEED)
    for j in range(n_noise):
        store.insert(_make_noise(j, rng, dim=dim))
    return store, pinned_records, pinned_texts

def test_topk_rank_identical_across_sequential_queries(tmp_path):
    store, pinned, pinned_texts = _seed_store(tmp_path, n_pinned=30, n_noise=100)
    dim = store.embed_dim
    cue = [1.0] * dim

    resp0 = recall(
        store=store,
        cue_embedding=cue,
        cue_text="probe",
        session_id="t0",
        budget_tokens=5000,
        k_hits=30,
        k_anti=3,
        mode="concept",
    )
    baseline_ids = tuple((h.record_id, h.literal_surface) for h in resp0.hits)
    assert len(baseline_ids) >= 1, "recall returned zero hits; harness broken"

    for i in range(1, 20):
        resp = recall(
            store=store,
            cue_embedding=cue,
            cue_text="probe",
            session_id=f"t{i}",
            budget_tokens=5000,
            k_hits=30,
            k_anti=3,
            mode="concept",
        )
        current = tuple((h.record_id, h.literal_surface) for h in resp.hits)
        assert current == baseline_ids, (
            f"rank drift at iteration {i}: top-k set changed between sequential "
            f"recalls with identical cue. Baseline={baseline_ids}, current={current}. "
            f"This indicates effect (c) provenance-write amplification is perturbing "
            f"the LanceDB vector index."
        )

    for rec in pinned:
        updated = store.get(rec.id)
        assert updated is not None, f"pinned record {rec.id} vanished"
        assert len(updated.provenance) >= 20, (
            f"pinned {rec.id} has "
            f"{len(updated.provenance)} provenance entries after 20 recalls "
            f"(expected >= 20)."
        )

def test_topk_contains_all_pinned_at_runbook_profile(tmp_path):
    store, pinned, _ = _seed_store(tmp_path, n_pinned=50, n_noise=200)
    dim = store.embed_dim
    cue = [1.0] * dim

    resp = recall(
        store=store,
        cue_embedding=cue,
        cue_text="probe",
        session_id="runbook",
        budget_tokens=50_000,
        k_hits=60,
        k_anti=3,
        mode="concept",
    )
    hit_ids = {h.record_id for h in resp.hits}
    pinned_ids = {r.id for r in pinned}
    missing = pinned_ids - hit_ids
    assert not missing, (
        f"runbook profile: "
        f"{len(missing)}/{len(pinned_ids)} pinned records missing from top-60. "
        f"Missing surface (first 3): "
        f"{sorted(str(m)[:8] for m in list(missing)[:3])}"
    )

def test_no_literal_surface_mutation(tmp_path):
    store, pinned, _ = _seed_store(tmp_path, n_pinned=10, n_noise=40)
    dim = store.embed_dim
    cue = [1.0] * dim

    pre = {r.id: store.get(r.id).literal_surface for r in pinned}

    for i in range(20):
        recall(
            store=store,
            cue_embedding=cue,
            cue_text=f"probe-{i}",
            session_id=f"s{i}",
            budget_tokens=5000,
            k_hits=15,
            k_anti=3,
            mode="concept",
        )

    post = {r.id: store.get(r.id).literal_surface for r in pinned}
    assert pre.keys() == post.keys()
    for rid in pre:
        assert pre[rid] == post[rid], (
            f"literal_surface of record {rid} mutated "
            f"by recall path. Before={pre[rid]!r}, after={post[rid]!r}."
        )
