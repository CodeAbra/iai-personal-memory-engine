"""Proves that a deep-verbatim planted target -- calibrated to the four real
deep-hit ranks 260-MEASURE.md measured (1055/1243/1546/1615) -- is served by
the full-pool recall path, with its rescue riding the T12 substring
multiplier, and that every embedded graph node (including one whose
embedding only resolves through the store fallback) reaches the scored
frontier on the >200-record live-path-shaped fixture.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_recall_scoring_differential import _freeze_age_penalty  # noqa: E402
from test_recall_stage_profile import _monkeypatch_env  # noqa: E402

from _synthetic_cue_corpus import (  # noqa: E402
    build_live_path_shaped_corpus, build_production_shaped_cue_set, insert_live_path_corpus,
)

import iai_mcp.pipeline as _pm
from iai_mcp import core
from iai_mcp.community import CommunityAssignment
from iai_mcp.daemon import _boot_warmup
from iai_mcp.embed import Embedder
from iai_mcp.graph import MemoryGraph
from iai_mcp.pipeline import K_CANDIDATES, recall_for_response
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake.pop((s, u), None))
    yield fake


@pytest.fixture(autouse=True)
def _clear_lex_fusion_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IAI_MCP_LEX_FUSION_OFF", raising=False)


def _select_driver(driver: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built -- lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


# ---------------------------------------------------------------------------
# Planted deep-verbatim controls, calibrated to the real ranks
# 260-MEASURE.md measured (1055/1243/1546/1615)
# ---------------------------------------------------------------------------

_DEEP_DIM = 16
_DEEP_N_BELOW = 402
_DEEP_ABOVE_COSINE_HIGH = 0.45
_DEEP_TARGET_COSINE = 0.10
_DEEP_BELOW_COSINE_LOW = 0.001
_DEEP_VERBATIM_PHRASE = "the widen threshold percentile derivation for wave two"

REAL_DEEP_HIT_RANKS = [1055, 1243, 1546, 1615]


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _orthonormal_pair(seed: int, dim: int) -> "tuple[np.ndarray, np.ndarray]":
    rng = np.random.default_rng(seed)
    cue = _unit(rng.random(dim).astype(np.float32))
    raw = rng.random(dim).astype(np.float32)
    raw -= np.dot(raw, cue) * cue
    return cue, _unit(raw)


def _vec_at_cosine(cue: np.ndarray, orth: np.ndarray, cosine: float) -> "list[float]":
    mag = float(np.sqrt(max(0.0, 1.0 - cosine * cosine)))
    return _unit(cosine * cue + mag * orth).tolist()


def _make_rec(vec: "list[float]", text: str, rec_id: "UUID | None" = None) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rec_id or uuid4(), tier="episodic", literal_surface=text, aaak_index="",
        embedding=vec, community_id=None, centrality=0.0, detail_level=2,
        pinned=False, stability=0.0, difficulty=0.0, last_reviewed=None,
        never_decay=False, never_merge=False, provenance=[],
        created_at=now, updated_at=now, tags=[], language="en",
    )


def _build_deep_verbatim_store(
    driver: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target_rank: int,
) -> "tuple[MemoryStore, np.ndarray, UUID]":
    """A corpus with a VERBATIM-match target (the cue text as a contiguous
    substring of literal_surface) placed at an exact cosine rank --
    calibrated to one of the real deep-hit ranks 260-MEASURE.md measured."""
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built -- lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(_DEEP_DIM))
    store = MemoryStore(path=tmp_path / f"deep-verbatim-{driver}-{target_rank}")

    cue, orth = _orthonormal_pair(seed=1000 + target_rank, dim=_DEEP_DIM)

    n_above = target_rank
    above_cosines = np.linspace(_DEEP_ABOVE_COSINE_HIGH, _DEEP_TARGET_COSINE + 0.001, n_above)
    for i, c in enumerate(above_cosines):
        vec = _vec_at_cosine(cue, orth, float(c))
        store.insert(_make_rec(vec, f"unrelated filler record number {i}"))

    target_id = uuid4()
    target_vec = _vec_at_cosine(cue, orth, _DEEP_TARGET_COSINE)
    # literal_surface CONTAINS the cue text verbatim (T12: `cue_lower in
    # literal_surface.lower()`, pipeline.py:730) -- a short suffix keeps the
    # trigram Jaccard against the cue high too (T11), so the multiplier
    # this control exercises is the full x2.0*x3.0=x6.0 combined tier.
    target_text = f"{_DEEP_VERBATIM_PHRASE} is discussed in this note."
    store.insert(_make_rec(target_vec, target_text, rec_id=target_id))

    below_cosines = np.linspace(_DEEP_TARGET_COSINE - 0.001, _DEEP_BELOW_COSINE_LOW, _DEEP_N_BELOW)
    for i, c in enumerate(below_cosines):
        vec = _vec_at_cosine(cue, orth, float(c))
        store.insert(_make_rec(vec, f"another unrelated filler record {i}"))

    flush_record_buffer(store)
    store._build_exact_index_sync()
    return store, cue, target_id


# ---------------------------------------------------------------------------
# Full-embedded-pool inclusion: the embedding_pending / _new_ids case is
# covered by construction, and the full-pool path scores past K_CANDIDATES.
# ---------------------------------------------------------------------------


def _spy_t11_t12_flags(monkeypatch: pytest.MonkeyPatch, captured: dict) -> None:
    orig = _pm._t11_t12_flags

    def spy(pool_ids, reachable_indices, records_cache, fts_hits, cue):
        captured["pool_ids"] = list(pool_ids)
        captured["records_cache_keys"] = set(records_cache.keys())
        return orig(pool_ids, reachable_indices, records_cache, fts_hits, cue)

    monkeypatch.setattr(_pm, "_t11_t12_flags", spy)


def test_embedding_pending_node_is_scored_not_dropped(tmp_path, monkeypatch):
    """A graph node that is embedding_pending at pool-build time -- present
    on the graph, absent from both the graph-resident embedding and the
    graph-resident "surface" payload key, but a real store-backed record
    with a genuine embedding + surface -- must reach the scored pool BY
    CONSTRUCTION: pool_ids (via _collect_graph_pool's store fallback),
    records_cache (via the backfill), and the served hits."""
    from test_recall_core_unit import _FakeEmbedder, _build_store_and_graph, _flat_assignment, _make

    store, graph, recs = _build_store_and_graph(tmp_path, n=12)

    pending_vec = [0.0] * EMBED_DIM
    pending_vec[0] = 1.0
    pending_rec = _make(pending_vec, text="the pending node target phrase")
    store.insert(pending_rec)
    # Graph-resident but embedding_pending: no usable graph embedding, no
    # "surface" payload key.
    graph.add_node(pending_rec.id, community_id=None, embedding=[])

    assignment = _flat_assignment(recs + [pending_rec])
    embedder = _FakeEmbedder(vec=pending_vec)

    captured: dict = {}
    _spy_t11_t12_flags(monkeypatch, captured)

    result = _pm._recall_core(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=embedder,
        cue="the pending node target phrase", session_id="s-pending",
        use_rust_scorer=True,
    )
    assert result is not None
    assert pending_rec.id in captured.get("pool_ids", []), (
        "fixture precondition failed: the embedding_pending node must "
        "reach pool_ids via _collect_graph_pool's store fallback"
    )
    assert pending_rec.id in captured.get("records_cache_keys", set()), (
        "the embedding_pending node reached pool_ids but never got a "
        "records_cache entry -- it would be silently dropped at the "
        "winners loop's `rec is None: continue` check"
    )
    served_ids = [h.record_id for h in result.scored_hits]
    assert pending_rec.id in served_ids, (
        "the embedding_pending node was scored but never surfaced in the "
        "served hits -- it is being dropped downstream of the "
        "records_cache backfill"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_full_pool_scores_all_embedded_graph_nodes(tmp_path, monkeypatch, driver):
    """On the >200-record live-path fixture, the default path's scored
    frontier covers every pool position -- the [:K_CANDIDATES] truncation
    is inert."""
    from iai_mcp.store._rank_index import _RankIndexHandle

    _select_driver(driver, monkeypatch)
    _monkeypatch_env(monkeypatch, tmp_path)

    embedder = Embedder()
    records, edges = build_live_path_shaped_corpus(embedder=embedder)
    assert len(records) > K_CANDIDATES, (
        f"fixture precondition failed: corpus ({len(records)}) does not "
        f"exceed K_CANDIDATES ({K_CANDIDATES})"
    )

    store_root = tmp_path / f"full-pool-{driver}"
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    store = MemoryStore(path=store_root)
    insert_live_path_corpus(store, records, edges)
    _boot_warmup.run_boot_warmup(store)

    cues = build_production_shaped_cue_set()
    probe_cue = cues["vague"][0]

    captured: dict = {}
    _orig_score = _RankIndexHandle.score

    def _spy_score(self, graph, pool_ids, cosine, cosine_top_indices, *rest, **kwargs):
        captured["pool_ids"] = list(pool_ids)
        captured["cosine_top_indices"] = np.array(cosine_top_indices)
        return _orig_score(self, graph, pool_ids, cosine, cosine_top_indices, *rest, **kwargs)

    monkeypatch.setattr(_RankIndexHandle, "score", _spy_score)

    _pm._last_recall_latency_ms = 0.0
    core.dispatch(store, "memory_recall", {
        "cue": probe_cue.text, "session_id": "full-pool-probe", "budget_tokens": 2000,
    })

    pool_ids = captured.get("pool_ids") or []
    cosine_top_indices = captured.get("cosine_top_indices")
    assert cosine_top_indices is not None, "the Rust scorer path must have run"
    assert len(pool_ids) > K_CANDIDATES, (
        f"pool_ids ({len(pool_ids)}) does not exceed K_CANDIDATES "
        f"({K_CANDIDATES}) on {driver} -- the truncation-is-inert proof "
        "needs a pool that actually exceeds the old cutoff"
    )
    assert len(cosine_top_indices) == len(pool_ids), (
        f"cosine_top_indices ({len(cosine_top_indices)}) does not cover the "
        f"full pool ({len(pool_ids)}) on {driver} -- the [:K_CANDIDATES] "
        "truncation is still active on the default path"
    )
    assert set(int(i) for i in cosine_top_indices) == set(range(len(pool_ids))), (
        f"cosine_top_indices on {driver} does not span every pool position "
        "-- some embedded graph node was excluded from the scored frontier"
    )


class _NullEmbedder:
    """Never called: cue_embedding is supplied directly, so text->vector
    embedding is bypassed entirely."""

    def embed(self, text: str) -> "list[float]":
        raise AssertionError("embedder.embed() called -- cue_embedding should have bypassed it")


def _flat_assignment(recs: "list[MemoryRecord]", dim: int) -> CommunityAssignment:
    cid = uuid4()
    centroid = [1.0] + [0.0] * (dim - 1)
    return CommunityAssignment(
        node_to_community={r.id: cid for r in recs},
        community_centroids={cid: centroid},
        modularity=0.0, backend="flat", top_communities=[cid],
        mid_regions={cid: [r.id for r in recs]},
    )


@pytest.mark.parametrize("target_rank", REAL_DEEP_HIT_RANKS)
@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_planted_target_rescue_depends_on_t12(driver, target_rank, tmp_path, monkeypatch):
    """The target's rescue into the served top-k rides the T12 substring
    multiplier (pipeline.py:1891-1895), not incidental cosine: with the
    corpus wired as a whole-corpus graph (so the target is a scoreable node
    the full-pool path always scores), forcing T11/T12 False leaves cosine
    rank alone (deep, at target_rank out of ~2200) far below the served
    top-k -- proven both directions."""
    store, cue, target_id = _build_deep_verbatim_store(driver, tmp_path, monkeypatch, target_rank)

    all_recs = list(store.all_records())
    graph = MemoryGraph()
    for rec in all_recs:
        graph.add_node(rec.id, community_id=None, embedding=list(rec.embedding))
        graph.set_node_payload(rec.id, {
            "embedding": list(rec.embedding), "surface": rec.literal_surface,
            "centrality": 0.0, "tier": rec.tier, "tags": rec.tags, "language": "en",
            "created_at": str(getattr(rec, "created_at", "") or ""),
        })
    assignment = _flat_assignment(all_recs, _DEEP_DIM)

    # The cue text itself, verbatim -- must literally appear inside the
    # target's literal_surface for T12 to fire (pipeline.py:730).
    cue_text = _DEEP_VERBATIM_PHRASE

    def _recall() -> object:
        # use_rust_scorer=True: recall_for_response's own function-level
        # default (when called directly, not via core.dispatch) is the
        # Python reference path, which computes T11/T12 inline rather than
        # through _t11_t12_flags -- the monkeypatch below only intercepts
        # the Rust-scorer wiring's call site (pipeline.py:1814).
        _pm._last_recall_latency_ms = 0.0
        return recall_for_response(
            store=store, graph=graph, assignment=assignment, rich_club=[],
            embedder=_NullEmbedder(), cue=cue_text, session_id="deep-verbatim-t12-probe",
            budget_tokens=3000, mode="concept", cue_embedding=cue.tolist(),
            use_rust_scorer=True,
        )

    resp_t12_on = _recall()
    on_ids = [h.record_id for h in resp_t12_on.hits]

    _orig_flags = _pm._t11_t12_flags

    def _forced_false(pool_ids, reachable_indices, records_cache, fts_hits, cue_):
        n = len(pool_ids)
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)

    monkeypatch.setattr(_pm, "_t11_t12_flags", _forced_false)
    resp_t12_off = _recall()
    monkeypatch.setattr(_pm, "_t11_t12_flags", _orig_flags)
    off_ids = [h.record_id for h in resp_t12_off.hits]

    assert target_id in on_ids, (
        f"target (rank {target_rank}, {driver}) absent from served hits "
        "with T12 active -- fixture precondition failed (the rescue this "
        "control is meant to test never happened)"
    )
    assert target_id not in off_ids, (
        f"target (rank {target_rank}, {driver}) still served with T11/T12 "
        "forced False -- its rescue does not actually depend on the T12 "
        "substring multiplier, the control has no teeth"
    )


